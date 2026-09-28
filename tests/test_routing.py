import unittest

from main import build_orchestrator
from core.schemas import AgentName, ChatRequest, IntentType, RiskLevel


class RoutingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.orchestrator = build_orchestrator()

    async def run_case(self, message: str):
        return await self.orchestrator.handle(
            ChatRequest(
                user_id="test_user",
                conversation_id="test_conv",
                message=message,
            )
        )

    async def test_low_academic_goes_to_guidance(self):
        result = await self.run_case("秋招怎么准备？")
        self.assertEqual(result.intent_result.intent, IntentType.ACADEMIC_CAREER)
        self.assertEqual(result.risk_assessment.risk_level, RiskLevel.LOW)
        self.assertEqual(result.selected_agent, AgentName.GUIDANCE)

    async def test_resource_query_goes_to_resource_agent(self):
        result = await self.run_case("学校心理咨询中心怎么预约？")
        self.assertEqual(result.intent_result.intent, IntentType.RESOURCE_QUERY)
        self.assertEqual(result.risk_assessment.risk_level, RiskLevel.LOW)
        self.assertEqual(result.selected_agent, AgentName.RESOURCE)

    async def test_medium_emotional_support_goes_to_support(self):
        result = await self.run_case("最近失恋了，晚上睡不好，心里很难受，想找个人聊聊。")
        self.assertEqual(result.risk_assessment.risk_level, RiskLevel.MEDIUM)
        self.assertEqual(result.selected_agent, AgentName.SUPPORT)

    async def test_high_risk_overrides_business_intent(self):
        result = await self.run_case("秋招一直失败，我真的快撑不下去了。")
        self.assertEqual(result.intent_result.intent, IntentType.ACADEMIC_CAREER)
        self.assertEqual(result.risk_assessment.risk_level, RiskLevel.HIGH)
        self.assertEqual(result.selected_agent, AgentName.ESCALATION)
        self.assertTrue(result.policy_decision.force_escalation)
        self.assertIn("search_verified_resources", result.policy_decision.mandatory_tools)

    async def test_high_human_support_goes_to_escalation(self):
        result = await self.run_case("我现在撑不住，想立刻找个人。")
        self.assertEqual(result.risk_assessment.risk_level, RiskLevel.HIGH)
        self.assertEqual(result.selected_agent, AgentName.ESCALATION)
        self.assertTrue(result.agent_result.handoff_requested)
