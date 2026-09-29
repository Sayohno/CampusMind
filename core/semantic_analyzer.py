"""LLM-based semantic fallback for ambiguous intent and risk understanding.

The semantic analyzer only interprets the user's message and returns
structured IntentResult / RiskAssessment objects.

It does not decide routing, tool permissions, mandatory actions,
or escalation policy. Those decisions remain deterministic elsewhere.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from core.llm_client import BaseLLMClient
from core.schemas import (
    IntentResult,
    IntentType,
    RiskAssessment,
    RiskLevel,
)


class SemanticAnalyzerError(RuntimeError):
    """Semantic analysis failed or returned an invalid structured result."""


@dataclass(frozen=True)
class SemanticAnalysis:
    intent_result: IntentResult
    risk_assessment: RiskAssessment


class SemanticAnalyzer:
    def __init__(self, llm_client: BaseLLMClient) -> None:
        self.llm_client = llm_client

    async def analyze(self, message: str) -> SemanticAnalysis:
        response = await self.llm_client.chat(
            messages=[
                {
                    "role": "system",
                    "content": self._system_prompt(),
                },
                {
                    "role": "user",
                    "content": message,
                },
            ],
            tools=None,
        )

        if response.tool_calls:
            raise SemanticAnalyzerError(
                "SemanticAnalyzer 不允许模型返回 tool calls"
            )

        content = (response.content or "").strip()
        if not content:
            raise SemanticAnalyzerError(
                "SemanticAnalyzer 收到空模型响应"
            )

        data = self._parse_json_object(content)

        try:
            intent = IntentType(str(data["intent"]))
            risk_level = RiskLevel(str(data["risk_level"]))

            intent_confidence = float(data["intent_confidence"])
            risk_confidence = float(data["risk_confidence"])

            raw_signals = data.get("signals", [])
            if not isinstance(raw_signals, list):
                raise TypeError("signals 必须是 list")

            signals = [
                str(item)
                for item in raw_signals
                if str(item).strip()
            ]
        except (
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise SemanticAnalyzerError(
                "SemanticAnalyzer 返回字段无效"
            ) from exc

        return SemanticAnalysis(
            intent_result=IntentResult(
                intent=intent,
                confidence=intent_confidence,
            ),
            risk_assessment=RiskAssessment(
                risk_level=risk_level,
                signals=signals,
                confidence=risk_confidence,
            ),
        )

    @staticmethod
    def _parse_json_object(content: str) -> dict:
        """Extract one JSON object from a model response."""
        start = content.find("{")
        end = content.rfind("}")

        if start < 0 or end < start:
            raise SemanticAnalyzerError(
                "SemanticAnalyzer 响应中没有 JSON object"
            )

        try:
            data = json.loads(content[start : end + 1])
        except json.JSONDecodeError as exc:
            raise SemanticAnalyzerError(
                "SemanticAnalyzer JSON 解析失败"
            ) from exc

        if not isinstance(data, dict):
            raise SemanticAnalyzerError(
                "SemanticAnalyzer 顶层 JSON 必须是 object"
            )

        return data

    @staticmethod
    def _system_prompt() -> str:
        return """
You are the semantic understanding component of CampusMind.

Your only task is to classify the user's CURRENT message.

Return exactly one JSON object and no explanation:

{
  "intent": "academic_career | emotional_support | relationship | resource_query | human_support | general",
  "intent_confidence": 0.0,
  "risk_level": "low | medium | high",
  "risk_confidence": 0.0,
  "signals": []
}

Intent definitions:

academic_career:
study, coursework, exams, research, thesis, internship,
job hunting, interviews, resumes, career planning.

emotional_support:
the user's main request is emotional support for distress,
low mood, anxiety, pressure, loneliness, or similar experiences.

relationship:
the user's main request concerns interpersonal relationships,
roommates, friends, classmates, romantic relationships,
breakups, or conflicts.

resource_query:
the user mainly asks for factual resources such as locations,
opening hours, application or reservation methods,
contact information, campus services, events, or channels.

human_support:
the user explicitly wants a real person, counselor,
teacher, staff member, or other human support.

general:
none of the above is the user's primary task.

Risk definitions:

low:
no meaningful current personal risk signal.

medium:
clear current personal distress or functional impact,
such as persistent sleep problems, severe pressure,
difficulty functioning, or strong emotional suffering,
without an explicit high-risk signal.

high:
explicit or strongly implied current personal self-harm,
suicidal, life-ending, or equivalent high-risk expression.

Important rules:

1. Classify the user's actual current task, not isolated keywords.
2. Intent and risk are independent dimensions.
3. Mentions inside a thesis, article, novel, translation,
   programming task, research topic, event title, or quoted text
   do NOT automatically indicate the user's personal risk.
4. If the user personally expresses high-risk meaning,
   do not downgrade it merely because the same sentence also
   mentions study, work, research, or another topic.
5. Do not make medical diagnoses.
6. signals should contain short evidence phrases from the message.
""".strip()