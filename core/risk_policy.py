"""CampusMind 确定性 RiskPolicyEngine。"""
from __future__ import annotations

from core.schemas import (
    AgentName,
    IntentResult,
    IntentType,
    PolicyDecision,
    RiskAssessment,
    RiskLevel,
)


class RiskPolicyEngine:
    _normal_route: dict[IntentType, AgentName] = {
        IntentType.ACADEMIC_CAREER: AgentName.GUIDANCE,
        IntentType.EMOTIONAL_SUPPORT: AgentName.SUPPORT,
        IntentType.RELATIONSHIP: AgentName.SUPPORT,
        IntentType.RESOURCE_QUERY: AgentName.RESOURCE,
        IntentType.HUMAN_SUPPORT: AgentName.SUPPORT,
        IntentType.GENERAL: AgentName.GENERAL,
    }

    def decide(
        self,
        intent_result: IntentResult,
        risk_assessment: RiskAssessment,
    ) -> PolicyDecision:
        risk = risk_assessment.risk_level

        if risk == RiskLevel.HIGH:
            return PolicyDecision(
                autonomy_level="deterministic",
                allowed_agent=AgentName.ESCALATION,
                mandatory_tools=[
                    "search_verified_resources",
                    "upsert_support_case",
                ],
                allow_general_advice=False,
                force_escalation=True,
            )

        if risk == RiskLevel.MEDIUM:
            if intent_result.intent in {
                IntentType.EMOTIONAL_SUPPORT,
                IntentType.RELATIONSHIP,
                IntentType.HUMAN_SUPPORT,
            }:
                agent = AgentName.SUPPORT
            else:
                agent = self._normal_route[intent_result.intent]

            return PolicyDecision(
                autonomy_level="controlled",
                allowed_agent=agent,
                mandatory_tools=[],
                allow_general_advice=True,
                force_escalation=False,
            )

        return PolicyDecision(
            autonomy_level="flexible",
            allowed_agent=self._normal_route[intent_result.intent],
            mandatory_tools=[],
            allow_general_advice=True,
            force_escalation=False,
        )
