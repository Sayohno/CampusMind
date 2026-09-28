import unittest

from agents.base_agent import GuidanceAgent
from agents.factory import build_agents
from agents.tools import build_default_tool_registry
from core.llm_client import ScriptedLLMClient
from core.schemas import (
    AgentName,
    AgentRequest,
    ChatRequest,
    IntentResult,
    IntentType,
    LLMResponse,
    PolicyDecision,
    RiskAssessment,
    RiskLevel,
    ToolCall,
)
from main import build_orchestrator
from state.case_store import InMemoryCaseStore
from state.memory_store import InMemoryConversationMemory


class ToolDedupTests(unittest.IsolatedAsyncioTestCase):
    def make_low_request(self) -> AgentRequest:
        return AgentRequest(
            user_id="u_dedup",
            conversation_id="c_dedup",
            message="帮我分析当前请求。",
            intent_result=IntentResult(intent=IntentType.ACADEMIC_CAREER, confidence=0.9),
            risk_assessment=RiskAssessment(risk_level=RiskLevel.LOW, confidence=0.9),
            policy=PolicyDecision(
                autonomy_level="flexible",
                allowed_agent=AgentName.GUIDANCE,
            ),
        )

    async def test_exact_duplicate_llm_tool_call_is_skipped(self):
        llm = ScriptedLLMClient(
            [
                LLMResponse(tool_calls=[
                    ToolCall(id="a1", name="inspect_request_context", arguments={"focus": "risk"})
                ]),
                LLMResponse(tool_calls=[
                    ToolCall(id="a2", name="inspect_request_context", arguments={"focus": "risk"})
                ]),
                LLMResponse(content="完成。"),
            ]
        )
        agent = GuidanceAgent(
            name=AgentName.GUIDANCE,
            system_prompt="test",
            llm_client=llm,
            tool_registry=build_default_tool_registry(case_store=InMemoryCaseStore()),
            allowed_tools={"inspect_request_context"},
            max_rounds=3,
        )

        result = await agent.run(self.make_low_request())

        self.assertEqual(result.tools_used, ["inspect_request_context"])
        self.assertEqual(
            [x.status for x in result.tool_executions],
            ["executed", "skipped_duplicate"],
        )
        self.assertEqual(result.runtime_rounds, 3)

    async def test_same_tool_with_different_arguments_is_allowed(self):
        llm = ScriptedLLMClient(
            [
                LLMResponse(tool_calls=[
                    ToolCall(id="b1", name="inspect_request_context", arguments={"focus": "risk"})
                ]),
                LLMResponse(tool_calls=[
                    ToolCall(id="b2", name="inspect_request_context", arguments={"focus": "intent"})
                ]),
                LLMResponse(content="完成。"),
            ]
        )
        agent = GuidanceAgent(
            name=AgentName.GUIDANCE,
            system_prompt="test",
            llm_client=llm,
            tool_registry=build_default_tool_registry(case_store=InMemoryCaseStore()),
            allowed_tools={"inspect_request_context"},
            max_rounds=3,
        )

        result = await agent.run(self.make_low_request())

        self.assertEqual(
            result.tools_used,
            ["inspect_request_context", "inspect_request_context"],
        )
        self.assertEqual(
            [x.status for x in result.tool_executions],
            ["executed", "executed"],
        )

    async def test_llm_exact_repeat_of_mandatory_tool_is_skipped(self):
        cases = InMemoryCaseStore()
        state = cases.ensure(
            user_id="u_high",
            conversation_id="c_high",
            topic=IntentType.ACADEMIC_CAREER.value,
            risk_level=RiskLevel.HIGH,
        )
        request = AgentRequest(
            user_id="u_high",
            conversation_id="c_high",
            message="秋招一直失败，我真的快撑不下去了。",
            intent_result=IntentResult(intent=IntentType.ACADEMIC_CAREER, confidence=0.9),
            risk_assessment=RiskAssessment(
                risk_level=RiskLevel.HIGH,
                signals=["撑不下去"],
                confidence=0.95,
            ),
            policy=PolicyDecision(
                autonomy_level="deterministic",
                allowed_agent=AgentName.ESCALATION,
                mandatory_tools=["search_verified_resources", "upsert_support_case"],
                allow_general_advice=False,
                force_escalation=True,
            ),
            case_state=state,
        )
        llm = ScriptedLLMClient(
            [
                LLMResponse(tool_calls=[
                    ToolCall(
                        id="same_search",
                        name="search_verified_resources",
                        arguments={"query": request.message, "top_k": 3},
                    )
                ]),
                LLMResponse(content="请联系身边可信赖的人或学校正式支持渠道。"),
            ]
        )
        registry = build_default_tool_registry(case_store=cases)
        agent = build_agents(llm_client=llm, tool_registry=registry)[AgentName.ESCALATION]

        result = await agent.run(request)

        self.assertEqual(
            result.tools_used,
            ["search_verified_resources", "upsert_support_case"],
        )
        search_records = [x for x in result.tool_executions if x.name == "search_verified_resources"]
        self.assertEqual([x.source for x in search_records], ["mandatory", "llm"])
        self.assertEqual([x.status for x in search_records], ["executed", "skipped_duplicate"])


class ResponseGuardTests(unittest.IsolatedAsyncioTestCase):
    async def test_high_risk_fabricated_external_action_uses_safe_fallback(self):
        memory = InMemoryConversationMemory()
        llm = ScriptedLLMClient([
            LLMResponse(content="我已经帮你联系学校心理中心并完成转介，你等电话就好。")
        ])
        orchestrator = build_orchestrator(llm_client=llm, memory_store=memory)

        result = await orchestrator.handle(
            ChatRequest(
                user_id="u_guard1",
                conversation_id="c_guard1",
                message="秋招一直失败，我真的快撑不下去了。",
            )
        )

        self.assertTrue(result.agent_result.safe_fallback_used)
        self.assertFalse(result.agent_result.guard_result.passed)
        self.assertIn("fabricated_external_action", result.agent_result.guard_result.violations)
        self.assertIn("不能把刚才那段候选回复直接发给你", result.agent_result.response)
        self.assertIn("我已经帮你联系学校心理中心", result.agent_result.candidate_response)
        history = memory.get_recent(user_id="u_guard1", conversation_id="c_guard1", limit=10)
        self.assertEqual(history[-1].content, result.agent_result.response)
        self.assertNotIn("已经帮你联系学校心理中心", history[-1].content)

    async def test_high_risk_safe_response_passes_guard(self):
        llm = ScriptedLLMClient([
            LLMResponse(
                content=(
                    "我不能替你联系任何机构。如果你现在有现实中的紧急危险，"
                    "请优先联系身边可信赖的人、学校正式支持渠道或当地紧急服务。"
                )
            )
        ])
        orchestrator = build_orchestrator(llm_client=llm)

        result = await orchestrator.handle(
            ChatRequest(
                user_id="u_guard2",
                conversation_id="c_guard2",
                message="秋招一直失败，我真的快撑不下去了。",
            )
        )

        self.assertFalse(result.agent_result.safe_fallback_used)
        self.assertTrue(result.agent_result.guard_result.passed)
        self.assertEqual(result.agent_result.guard_result.violations, [])

    async def test_resource_answer_without_rag_is_blocked(self):
        llm = ScriptedLLMClient([
            LLMResponse(content="学校心理咨询中心在东门旁边，工作日开放。")
        ])
        orchestrator = build_orchestrator(llm_client=llm)

        result = await orchestrator.handle(
            ChatRequest(
                user_id="u_guard3",
                conversation_id="c_guard3",
                message="学校心理咨询中心怎么预约？",
            )
        )

        self.assertTrue(result.agent_result.safe_fallback_used)
        self.assertIn(
            "resource_answer_without_verified_retrieval",
            result.agent_result.guard_result.violations,
        )
        self.assertIn("没有足够的已验证证据", result.agent_result.response)

    async def test_rag_then_ungrounded_phone_is_blocked(self):
        question = "学校心理咨询中心怎么预约？"
        llm = ScriptedLLMClient(
            [
                LLMResponse(tool_calls=[
                    ToolCall(
                        id="rag_guard",
                        name="search_verified_resources",
                        arguments={"query": question, "top_k": 3},
                    )
                ]),
                LLMResponse(content="可以拨打 12345678 预约。"),
            ]
        )
        orchestrator = build_orchestrator(llm_client=llm)

        result = await orchestrator.handle(
            ChatRequest(
                user_id="u_guard4",
                conversation_id="c_guard4",
                message=question,
            )
        )

        self.assertTrue(result.agent_result.safe_fallback_used)
        self.assertIn("ungrounded_phone_claim", result.agent_result.guard_result.violations)
        self.assertNotIn("12345678", result.agent_result.response)

    async def test_rag_grounded_abstention_passes(self):
        question = "学校心理咨询中心怎么预约？"
        llm = ScriptedLLMClient(
            [
                LLMResponse(tool_calls=[
                    ToolCall(
                        id="rag_ok",
                        name="search_verified_resources",
                        arguments={"query": question, "top_k": 3},
                    )
                ]),
                LLMResponse(
                    content=(
                        "当前已验证资料无法确认你所在学校的具体预约方式、电话或开放时间，"
                        "请以学校官网、学生工作部门或学校正式发布渠道为准。"
                    )
                ),
            ]
        )
        orchestrator = build_orchestrator(llm_client=llm)

        result = await orchestrator.handle(
            ChatRequest(
                user_id="u_guard5",
                conversation_id="c_guard5",
                message=question,
            )
        )

        self.assertFalse(result.agent_result.safe_fallback_used)
        self.assertTrue(result.agent_result.guard_result.passed)
        self.assertIn("search_verified_resources", result.agent_result.tools_used)


if __name__ == "__main__":
    unittest.main(verbosity=2)
