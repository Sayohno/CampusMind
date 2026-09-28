"""CampusMind Response Guard + Safe Fallback。

PolicyEngine 在生成前规定“系统允许做什么”；ResponseGuard 在生成后检查
“这个候选回答能不能发”。高风险硬规则不交给 LLM-as-Judge。
"""
from __future__ import annotations

import re
from typing import Any

from core.schemas import AgentRequest, AgentResult, GuardResult, IntentType, RiskLevel


class ResponseGuard:
    """轻量确定性 Guard。

    当前版本重点保护两类边界：
    1) 高风险路径不能伪造已完成的外部转介/报警/联系，也不能漏掉现实支持引导；
    2) 资源事实必须经过 verified retrieval，证据不足时不能补具体电话/时间/地址/预约渠道。
    """

    _fabricated_action_patterns = [
        re.compile(
            r"(?:我|系统|CampusMind).{0,10}(?:已经|已)(?:帮你|为你)?"
            r"(?:联系|通知|报警|转介|预约)(?:了|好|完成)?",
            re.IGNORECASE,
        ),
        re.compile(r"(?:已经|已)(?:替你|为你).{0,8}(?:联系|通知|报警|转介|预约)", re.IGNORECASE),
    ]
    _diagnosis_patterns = [
        re.compile(r"你.{0,3}(?:患有|得了|就是|属于)(?:抑郁症|焦虑症|双相|精神疾病)"),
        re.compile(r"可以诊断为(?:抑郁症|焦虑症|双相|精神疾病)"),
    ]
    _real_world_support_cues = (
        "可信赖",
        "身边的人",
        "家人",
        "朋友",
        "学校正式支持",
        "学校官方",
        "当地紧急服务",
        "现实中",
        "现实里的",
        "紧急服务",
    )
    _phone_pattern = re.compile(r"(?<!\d)(?:\+?\d[\d\-\s]{6,}\d)(?!\d)")
    _clock_pattern = re.compile(r"(?<!\d)(?:[01]?\d|2[0-3]):[0-5]\d(?!\d)")
    _hour_pattern = re.compile(r"(?<!\d)\d{1,2}\s*(?:点|时)(?!\d)")
    _address_pattern = re.compile(r"(?:路|街|大道|巷)\s*\d+\s*号|(?:校区|学院).{0,10}\d+\s*号楼")
    _appointment_channel_pattern = re.compile(
        r"(?:通过|使用|打开|关注).{0,10}(?:公众号|小程序|APP|App|电话).{0,8}(?:预约|联系)"
    )

    def evaluate(self, request: AgentRequest, result: AgentResult) -> GuardResult:
        text = (result.response or "").strip()
        violations: list[str] = []

        if not text:
            violations.append("empty_response")

        is_high_risk = (
            request.risk_assessment.risk_level == RiskLevel.HIGH
            or request.policy.force_escalation
        )
        if is_high_risk:
            if self._matches_any(text, self._fabricated_action_patterns):
                violations.append("fabricated_external_action")
            if self._matches_any(text, self._diagnosis_patterns):
                violations.append("medical_diagnosis_claim")
            if not any(cue in text for cue in self._real_world_support_cues):
                violations.append("missing_real_world_support_guidance")

        rag_execution = self._latest_executed_tool(result, "search_verified_resources")
        needs_grounding = request.intent_result.intent == IntentType.RESOURCE_QUERY
        if needs_grounding and rag_execution is None:
            violations.append("resource_answer_without_verified_retrieval")
        elif rag_execution is not None:
            rag_result = rag_execution.get("result")
            if isinstance(rag_result, dict):
                self._check_resource_grounding(text, rag_result, violations)

        return GuardResult(passed=not violations, violations=violations)

    def safe_fallback(self, request: AgentRequest, guard: GuardResult) -> str:
        is_high_risk = (
            request.risk_assessment.risk_level == RiskLevel.HIGH
            or request.policy.force_escalation
        )
        if is_high_risk:
            return (
                "我现在不能把刚才那段候选回复直接发给你，因为它没有通过 CampusMind 的高风险安全检查。"
                "当前系统也没有真实人工转介、报警或热线联动能力，因此不会声称已经替你联系任何机构。"
                "如果你此刻存在现实中的紧急危险，请优先联系身边可信赖的人、学校正式支持渠道或当地紧急服务。"
                "如果不是立即危险，也可以先告诉我你现在最需要解决的事情，我会在安全边界内继续陪你梳理。"
            )

        if request.intent_result.intent == IntentType.RESOURCE_QUERY:
            return (
                "当前候选回复没有通过资源事实校验。CampusMind 目前没有足够的已验证证据确认具体地址、电话、"
                "开放时间或预约方式，请以所在学校官网、学生工作部门或学校正式发布渠道为准。"
            )

        return (
            "当前候选回复没有通过系统的输出校验，因此没有直接发送。"
            "你可以换一种说法再问，我会在当前安全和事实边界内重新处理。"
        )

    @staticmethod
    def _matches_any(text: str, patterns: list[re.Pattern[str]]) -> bool:
        return any(pattern.search(text) for pattern in patterns)

    @staticmethod
    def _latest_executed_tool(result: AgentResult, name: str) -> dict[str, Any] | None:
        for item in reversed(result.tool_executions):
            if item.name == name and item.status == "executed":
                return item.model_dump(mode="python")
        return None

    def _check_resource_grounding(
        self,
        text: str,
        rag_result: dict[str, Any],
        violations: list[str],
    ) -> None:
        evidence = rag_result.get("evidence") or []
        evidence_text = "\n".join(
            str(item.get("content", ""))
            for item in evidence
            if isinstance(item, dict)
        )

        # 没有证据时，如果回答出现可核验的具体资源事实，直接拦截。
        if not rag_result.get("found"):
            if self._contains_specific_resource_fact(text):
                violations.append("ungrounded_specific_resource_claim")
            return

        # 即使检索有结果，也不能凭空补 evidence 里不存在的电话/时间/地址/预约渠道。
        for label, pattern in (
            ("phone", self._phone_pattern),
            ("clock", self._clock_pattern),
            ("hour", self._hour_pattern),
            ("address", self._address_pattern),
            ("appointment_channel", self._appointment_channel_pattern),
        ):
            for match in pattern.finditer(text):
                fact = match.group(0)
                if fact and fact not in evidence_text:
                    violations.append(f"ungrounded_{label}_claim")
                    break

    def _contains_specific_resource_fact(self, text: str) -> bool:
        return any(
            pattern.search(text)
            for pattern in (
                self._phone_pattern,
                self._clock_pattern,
                self._hour_pattern,
                self._address_pattern,
                self._appointment_channel_pattern,
            )
        )
