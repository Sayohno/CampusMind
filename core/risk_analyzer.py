"""CampusMind RiskAnalyzer：风险信号、语境与功能影响的确定性风险识别。

这里只进行系统内部工程风险分级，不进行医疗诊断。

RiskAnalyzer 负责 deterministic fast path：
- 明确风险表达直接识别；
- 区分用户本人表达与论文、小说、程序、活动等讨论语境；
- 识别部分功能受损信号；
- 对存在歧义的情况交给上层 SemanticAnalyzer 做语义消歧。

最终 Agent 路由、Tool 权限和高风险升级策略仍由
RiskPolicyEngine / Runtime 决定。
"""
from __future__ import annotations

import re

from core.schemas import RiskAssessment, RiskLevel


class RiskAnalyzer:
    # 明确高风险表达。
    _high_signals = (
        "撑不下去",
        "撑不住",
        "不想活",
        "想死",
        "自杀",
        "伤害自己",
        "结束生命",
        "活不下去",
    )

    # 隐式高风险表达。
    #
    # 这些词不能单独机械判断。
    # 例如“文件消失了”显然不是风险表达。
    _indirect_high_signals = (
        "从这个世界消失",
        "彻底消失",
        "结束这一切",
        "再也不醒来",
        "不用再醒来",
    )

    # 中风险困扰信号。
    _medium_signals = (
        "焦虑",
        "睡不着",
        "失眠",
        "很难受",
        "特别难受",
        "压力很大",
        "崩溃",
        "绝望",
        "连续几天睡不好",
        "整晚睡不好",
        "睡不好",
        "压抑",
        "胸口堵",
    )

    # 功能受损相关信号。
    _functional_signals = (
        "状态很差",
        "白天也没精神",
        "没精神",
        "提不起劲",
        "影响到上课",
        "影响上课",
        "影响学习",
        "影响工作",
        "影响睡眠",
        "不太想一个人扛",
        "一个人扛着",
    )

    # 说明风险词可能只是“被讨论对象”，
    # 而不是用户本人正在表达风险。
    _meta_context_markers = (
        "论文",
        "论文题目",
        "研究",
        "研究框架",
        "讲座",
        "宣传周",
        "宣传活动",
        "活动报名",
        "程序",
        "代码",
        "关键词",
        "检测",
        "翻译",
        "小说",
        "课程作业",
        "职业",
        "文章",
        "润色",
        "讨论",
        "课程",
        "题目",
    )

    # 明显的个人困扰语境。
    #
    # 这里不能只判断有没有“我”，因为：
    #
    # “我在做一个检测‘伤害自己’关键词的程序”
    #
    # 虽然有“我”，但并不是用户本人风险表达。
    _personal_distress_markers = (
        "我最近",
        "我这几天",
        "我一直",
        "我现在",
        "我开始觉得",
        "我真的",
        "我已经",
        "我觉得自己",
        "最近我",
        "这几天我",
        "让我很",
        "已经影响到",
        "已经影响",
        "我不想活",
        "我想死",
        "我撑不住",
        "我撑不下去",
        "我活不下去",
        "自己不想活",
        "自己想死",
        "自己活不下去",
        "自己撑不住",
    )

    # 隐式 high-risk 中常见的“解脱结果期待”。
    _relief_markers = (
        "更轻松",
        "能解脱",
        "解脱",
        "就好了",
        "会更好",
    )

    def analyze(
        self,
        message: str,
    ) -> RiskAssessment:
        text = message.strip().lower()

        # 引号中的内容可能只是：
        #
        # 小说台词
        # 翻译内容
        # 文章引用
        # 被检测的关键词
        #
        # deterministic fast path 不把这些内容
        # 直接当成用户本人风险表达。
        #
        # 原始 text 仍然保留，
        # 后面的 needs_semantic_fallback() 会用完整文本做消歧判断。
        risk_text = self._remove_quoted_text(text)

        # 1) 判断是否属于论文 / 小说 / 程序 / 活动等 meta context。
        meta_context = self._is_meta_context(text)

        # personal distress 只在去除引用后的正文上判断。
        personal_distress = self._has_personal_distress(
            risk_text
        )

        # 如果明显属于讨论 / 引用 / 研究语境，
        # 并且正文里没有个人困扰信号，
        # deterministic fast path 暂时按 LOW。
        #
        # 例如：
        #
        # “自杀预防宣传周在哪里报名？”
        #
        # “小说人物说‘我不想活了’，怎么翻译？”
        #
        # “检测‘伤害自己’关键词的程序怎么写？”
        if (
            meta_context
            and not personal_distress
        ):
            return RiskAssessment(
                risk_level=RiskLevel.LOW,
                signals=[
                    "contextual_non_personal_reference"
                ],
                confidence=0.88,
            )

        # 2) 明确 high-risk。
        #
        # 注意这里扫描的是 risk_text，
        # 即已经去掉引用内容后的用户正文。
        high_hits = [
            signal
            for signal in self._high_signals
            if signal in risk_text
        ]

        if high_hits:
            return RiskAssessment(
                risk_level=RiskLevel.HIGH,
                signals=high_hits,
                confidence=min(
                    0.90
                    + 0.02 * (len(high_hits) - 1),
                    0.98,
                ),
            )

        # 3) 隐式 high-risk。
        indirect_high_hits = [
            signal
            for signal in self._indirect_high_signals
            if signal in risk_text
        ]

        if (
            indirect_high_hits
            and personal_distress
        ):
            relief_hits = [
                marker
                for marker in self._relief_markers
                if marker in risk_text
            ]

            signals = (
                indirect_high_hits
                + relief_hits
            )

            return RiskAssessment(
                risk_level=RiskLevel.HIGH,
                signals=signals,
                confidence=(
                    0.92
                    if relief_hits
                    else 0.88
                ),
            )

        # 4) 明确 medium-risk。
        medium_hits = [
            signal
            for signal in self._medium_signals
            if signal in risk_text
        ]

        if medium_hits:
            return RiskAssessment(
                risk_level=RiskLevel.MEDIUM,
                signals=medium_hits,
                confidence=min(
                    0.78
                    + 0.03 * (len(medium_hits) - 1),
                    0.92,
                ),
            )

        # 5) 功能影响。
        functional_hits = [
            signal
            for signal in self._functional_signals
            if signal in risk_text
        ]

        has_explicit_impairment = any(
            marker in risk_text
            for marker in (
                "影响到上课",
                "影响上课",
                "影响学习",
                "影响工作",
                "影响睡眠",
            )
        )

        # 两个及以上弱功能信号，
        # 或存在明确“已经影响...”表达，
        # 判为 medium。
        if (
            len(functional_hits) >= 2
            or has_explicit_impairment
        ):
            return RiskAssessment(
                risk_level=RiskLevel.MEDIUM,
                signals=functional_hits,
                confidence=min(
                    0.72
                    + 0.04 * len(functional_hits),
                    0.88,
                ),
            )

        # 6) 没有足够风险证据。
        return RiskAssessment(
            risk_level=RiskLevel.LOW,
            signals=[],
            confidence=0.75,
        )

    def needs_semantic_fallback(
        self,
        message: str,
        result: RiskAssessment,
    ) -> bool:
        """判断当前规则结果是否值得进一步做 LLM 语义消歧。

        True 不代表规则一定错。

        它表示：
        当前消息存在引用、meta context、隐式风险或功能影响等
        deterministic rules 难以可靠判断的情况。
        """
        text = message.strip().lower()

        # 注意：
        # 这里故意使用原始全文 text，
        # 不删除引号。
        #
        # 因为：
        #
        # “小说人物说‘我不想活了’，怎么翻译？”
        #
        # deterministic analyze() 应该先给 LOW，
        # 但这里仍然应该发现：
        #
        # high-risk phrase + 小说/翻译语境
        #
        # 从而进入 SemanticAnalyzer 做最终语义消歧。
        high_hits = [
            signal
            for signal in self._high_signals
            if signal in text
        ]

        indirect_high_hits = [
            signal
            for signal in self._indirect_high_signals
            if signal in text
        ]

        medium_hits = [
            signal
            for signal in self._medium_signals
            if signal in text
        ]

        # 风险词和论文 / 小说 / 程序 / 活动等
        # meta context 同时出现。
        if (
            self._is_meta_context(text)
            and (
                high_hits
                or indirect_high_hits
                or medium_hits
            )
        ):
            return True

        # deterministic 结果为 LOW，
        # 但全文仍存在潜在 high-risk phrase。
        if (
            result.risk_level == RiskLevel.LOW
            and (
                high_hits
                or indirect_high_hits
            )
        ):
            return True

        functional_hits = [
            signal
            for signal in self._functional_signals
            if signal in text
        ]

        temporal_or_personal = any(
            marker in text
            for marker in (
                "最近",
                "这几天",
                "我觉得",
                "我总",
                "我一直",
                "我现在",
                "第二天",
                "越来越",
            )
        )

        # 例如：
        #
        # “最近晚上脑子停不下来，
        # 第二天上课也没精神。”
        #
        # 固定 medium phrase 可能没有命中，
        # 但存在持续性 + 功能影响，
        # 适合交给 SemanticAnalyzer。
        if (
            result.risk_level == RiskLevel.LOW
            and functional_hits
            and temporal_or_personal
        ):
            return True

        return False

    def has_potential_high_signal(
        self,
        message: str,
    ) -> bool:
        """判断原始消息中是否存在潜在 high-risk signal。

        这个方法不是最终风险分类器。

        它只用于一个场景：

        Semantic fallback 本来应该做语义消歧，
        但模型/API/JSON 解析失败。

        此时如果原始消息中仍包含潜在 high-risk signal，
        Orchestrator 可以选择安全侧 fail-closed。
        """
        text = message.strip().lower()

        return any(
            signal in text
            for signal in (
                *self._high_signals,
                *self._indirect_high_signals,
            )
        )

    def _is_meta_context(
        self,
        text: str,
    ) -> bool:
        return any(
            marker in text
            for marker in self._meta_context_markers
        )

    def _has_personal_distress(
        self,
        text: str,
    ) -> bool:
        return any(
            marker in text
            for marker in self._personal_distress_markers
        )

    @staticmethod
    def _remove_quoted_text(
        text: str,
    ) -> str:
        """移除常见引号内部的引用文本。

        例如：

        小说人物说“我不想活了”，这句话怎么翻译？

        会得到近似：

        小说人物说 ，这句话怎么翻译？

        这样 deterministic RiskAnalyzer 不会把小说台词
        直接当成用户本人风险表达。

        原始全文仍然保留给 needs_semantic_fallback()，
        所以引用中的风险内容并不会被系统彻底忽略。
        """
        patterns = (
            r"“[^”]*”",
            r"‘[^’]*’",
            r'"[^"]*"',
            r"'[^']*'",
        )

        result = text

        for pattern in patterns:
            result = re.sub(
                pattern,
                " ",
                result,
            )

        return result