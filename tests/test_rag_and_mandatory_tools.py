import unittest

from agents.base_agent import EscalationAgent
from agents.factory import build_agents
from agents.tools import build_default_tool_registry
from core.llm_client import ScriptedLLMClient, StaticLLMClient
from core.schemas import (
    AgentName,
    AgentRequest,
    CaseStatus,
    IntentResult,
    IntentType,
    LLMResponse,
    PolicyDecision,
    RiskAssessment,
    RiskLevel,
)
from rag.resource_store import VerifiedResourceStore
from state.case_store import InMemoryCaseStore


class RagRetrievalTests(unittest.IsolatedAsyncioTestCase):
    async def test_resource_store_returns_relevant_verified_evidence(self):
        store = VerifiedResourceStore.demo()
        hits = store.search("学校心理咨询中心怎么预约？", top_k=3)

        self.assertGreaterEqual(len(hits), 1)
        self.assertEqual(hits[0].resource_id, "demo_counseling_access")
        self.assertTrue(hits[0].source_ref.startswith("demo://"))
        self.assertIn("不包含任何真实学校", hits[0].content)

    async def test_unrelated_query_returns_no_evidence(self):
        store = VerifiedResourceStore.demo()
        hits = store.search("今天天气怎么样？", top_k=3)
        self.assertEqual(hits, [])

    async def test_search_verified_resources_tool_returns_grounding_contract(self):
        cases = InMemoryCaseStore()
        registry = build_default_tool_registry(case_store=cases)
        state = cases.ensure(
            user_id="u1",
            conversation_id="c1",
            topic=IntentType.RESOURCE_QUERY.value,
            risk_level=RiskLevel.LOW,
        )
        request = AgentRequest(
            user_id="u1",
            conversation_id="c1",
            message="学校心理咨询中心周日几点开门？",
            intent_result=IntentResult(intent=IntentType.RESOURCE_QUERY, confidence=0.9),
            risk_assessment=RiskAssessment(risk_level=RiskLevel.LOW, confidence=0.9),
            policy=PolicyDecision(
                autonomy_level="flexible",
                allowed_agent=AgentName.RESOURCE,
            ),
            case_state=state,
        )

        result = await registry.execute(
            tool_name="search_verified_resources",
            arguments={"query": request.message, "top_k": 3},
            request=request,
            allowed_tools={"search_verified_resources"},
        )

        self.assertTrue(result["found"])
        self.assertGreaterEqual(result["count"], 1)
        self.assertIn("必须明确无法确认", result["grounding_rule"])
        self.assertTrue(result["evidence"][0]["source_ref"].startswith("demo://"))

    async def test_resource_agent_can_call_rag_and_continue_generation(self):
        cases = InMemoryCaseStore()
        registry = build_default_tool_registry(case_store=cases)
        state = cases.ensure(
            user_id="u_resource",
            conversation_id="c_resource",
            topic=IntentType.RESOURCE_QUERY.value,
            risk_level=RiskLevel.LOW,
        )
        request = AgentRequest(
            user_id="u_resource",
            conversation_id="c_resource",
            message="学校心理咨询中心怎么预约？",
            intent_result=IntentResult(intent=IntentType.RESOURCE_QUERY, confidence=0.95),
            risk_assessment=RiskAssessment(risk_level=RiskLevel.LOW, confidence=0.95),
            policy=PolicyDecision(
                autonomy_level="flexible",
                allowed_agent=AgentName.RESOURCE,
            ),
            case_state=state,
        )
        from core.schemas import ToolCall
        llm = ScriptedLLMClient([
            LLMResponse(tool_calls=[
                ToolCall(
                    id="rag_1",
                    name="search_verified_resources",
                    arguments={"query": request.message, "top_k": 3},
                )
            ]),
            LLMResponse(content="当前演示知识库无法确认真实学校的具体预约规则，请以学校正式渠道为准。"),
        ])
        agent = build_agents(llm_client=llm, tool_registry=registry)[AgentName.RESOURCE]

        result = await agent.run(request)

        self.assertIn("search_verified_resources", result.tools_used)
        self.assertEqual(result.runtime_rounds, 2)
        self.assertIn("无法确认", result.response)
        tool_messages = [m for m in llm.seen_messages[1] if m.get("role") == "tool"]
        self.assertEqual(len(tool_messages), 1)
        self.assertIn("demo_counseling_access", tool_messages[0]["content"])

    async def test_resource_agent_has_rag_tool_scope(self):
        cases = InMemoryCaseStore()
        registry = build_default_tool_registry(case_store=cases)
        agents = build_agents(
            llm_client=StaticLLMClient("ok"),
            tool_registry=registry,
        )
        self.assertIn(
            "search_verified_resources",
            agents[AgentName.RESOURCE].allowed_tools,
        )


class MandatoryToolTests(unittest.IsolatedAsyncioTestCase):
    def make_high_risk_request(self, cases: InMemoryCaseStore) -> AgentRequest:
        state = cases.ensure(
            user_id="u2",
            conversation_id="c2",
            topic=IntentType.ACADEMIC_CAREER.value,
            risk_level=RiskLevel.HIGH,
        )
        return AgentRequest(
            user_id="u2",
            conversation_id="c2",
            message="秋招一直失败，我真的快撑不下去了。",
            intent_result=IntentResult(
                intent=IntentType.ACADEMIC_CAREER,
                confidence=0.9,
            ),
            risk_assessment=RiskAssessment(
                risk_level=RiskLevel.HIGH,
                signals=["撑不下去"],
                confidence=0.95,
            ),
            policy=PolicyDecision(
                autonomy_level="deterministic",
                allowed_agent=AgentName.ESCALATION,
                mandatory_tools=[
                    "search_verified_resources",
                    "upsert_support_case",
                ],
                allow_general_advice=False,
                force_escalation=True,
            ),
            case_state=state,
        )

    async def test_high_risk_mandatory_tools_execute_before_llm(self):
        cases = InMemoryCaseStore()
        registry = build_default_tool_registry(case_store=cases)
        llm = ScriptedLLMClient([LLMResponse(content="受控高风险回复")])
        agent = build_agents(llm_client=llm, tool_registry=registry)[AgentName.ESCALATION]

        result = await agent.run(self.make_high_risk_request(cases))

        self.assertEqual(result.response, "受控高风险回复")
        self.assertEqual(
            result.tools_used[:2],
            ["search_verified_resources", "upsert_support_case"],
        )
        self.assertEqual(llm.call_count, 1)
        state = cases.get(user_id="u2", conversation_id="c2")
        self.assertEqual(state.status, CaseStatus.ESCALATED)
        self.assertIn("high_risk_policy_path_activated", state.known_context)

        first_messages = llm.seen_messages[0]
        mandatory_contexts = [
            message.get("content", "")
            for message in first_messages
            if message.get("role") == "system"
            and message.get("content", "").startswith("CampusMind mandatory tool results")
        ]
        self.assertEqual(len(mandatory_contexts), 1)
        self.assertIn("search_verified_resources", mandatory_contexts[0])
        self.assertIn("demo_high_risk_boundary", mandatory_contexts[0])

    async def test_mandatory_tool_permission_failure_fails_closed(self):
        cases = InMemoryCaseStore()
        registry = build_default_tool_registry(case_store=cases)
        llm = ScriptedLLMClient([LLMResponse(content="不应该被调用")])
        agent = EscalationAgent(
            name=AgentName.ESCALATION,
            system_prompt="test",
            llm_client=llm,
            tool_registry=registry,
            allowed_tools={"upsert_support_case"},
        )

        result = await agent.run(self.make_high_risk_request(cases))

        self.assertEqual(llm.call_count, 0)
        self.assertIn("必要能力暂时不可用", result.response)
        self.assertTrue(result.handoff_requested)
        self.assertEqual(result.runtime_rounds, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
