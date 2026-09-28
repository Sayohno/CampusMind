import unittest

from agents.agent_orchestrator import AgentOrchestrator, Router
from agents.factory import build_agents
from agents.tools import ToolArgumentError, build_default_tool_registry
from core.intent_analyzer import IntentAnalyzer
from core.llm_client import ScriptedLLMClient, StaticLLMClient
from core.risk_analyzer import RiskAnalyzer
from core.risk_policy import RiskPolicyEngine
from core.schemas import (
    AgentName,
    AgentRequest,
    CaseStatus,
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


class StateLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_handle_writes_memory_and_case_state(self):
        memory = InMemoryConversationMemory()
        cases = InMemoryCaseStore()
        orchestrator = build_orchestrator(
            llm_client=StaticLLMClient("测试回复"),
            memory_store=memory,
            case_store=cases,
        )

        result = await orchestrator.handle(
            ChatRequest(
                user_id="u1",
                conversation_id="c1",
                message="秋招怎么准备？",
            )
        )

        history = memory.get_recent(user_id="u1", conversation_id="c1", limit=10)
        state = cases.get(user_id="u1", conversation_id="c1")

        self.assertEqual([m.role for m in history], ["user", "assistant"])
        self.assertEqual(history[0].content, "秋招怎么准备？")
        self.assertEqual(history[1].content, "测试回复")
        self.assertEqual(result.memory_size, 2)
        self.assertIsNotNone(state)
        self.assertEqual(state.topic, IntentType.ACADEMIC_CAREER.value)
        self.assertEqual(state.current_risk_level, RiskLevel.LOW)
        self.assertEqual(state.status, CaseStatus.ACTIVE)
        self.assertTrue(result.agent_result.case_updated)

    async def test_second_turn_receives_previous_conversation_memory(self):
        memory = InMemoryConversationMemory()
        cases = InMemoryCaseStore()
        llm = ScriptedLLMClient(
            [
                LLMResponse(content="第一轮回复"),
                LLMResponse(content="第二轮回复"),
            ]
        )
        registry = build_default_tool_registry(case_store=cases)
        agents = build_agents(llm_client=llm, tool_registry=registry)
        orchestrator = AgentOrchestrator(
            intent_analyzer=IntentAnalyzer(),
            risk_analyzer=RiskAnalyzer(),
            policy_engine=RiskPolicyEngine(),
            router=Router(agents),
            memory_store=memory,
            case_store=cases,
        )

        await orchestrator.handle(
            ChatRequest(user_id="u1", conversation_id="c1", message="秋招怎么准备？")
        )
        await orchestrator.handle(
            ChatRequest(user_id="u1", conversation_id="c1", message="简历已经改完了，下一步呢？")
        )

        second_messages = llm.seen_messages[1]
        contents = [msg.get("content") for msg in second_messages]
        self.assertIn("秋招怎么准备？", contents)
        self.assertIn("第一轮回复", contents)
        self.assertEqual(second_messages[-1]["content"], "简历已经改完了，下一步呢？")
        self.assertEqual(memory.count(user_id="u1", conversation_id="c1"), 4)

    async def test_high_risk_marks_case_escalated(self):
        cases = InMemoryCaseStore()
        orchestrator = build_orchestrator(
            llm_client=StaticLLMClient("受控回复"),
            case_store=cases,
        )

        result = await orchestrator.handle(
            ChatRequest(
                user_id="u2",
                conversation_id="c2",
                message="秋招一直失败，我真的快撑不下去了。",
            )
        )

        self.assertEqual(result.case_state.current_risk_level, RiskLevel.HIGH)
        self.assertEqual(result.case_state.status, CaseStatus.ESCALATED)
        self.assertEqual(result.selected_agent, AgentName.ESCALATION)


class CaseToolTests(unittest.IsolatedAsyncioTestCase):
    def make_request(self, cases: InMemoryCaseStore) -> AgentRequest:
        state = cases.ensure(
            user_id="u3",
            conversation_id="c3",
            topic=IntentType.ACADEMIC_CAREER.value,
            risk_level=RiskLevel.LOW,
        )
        return AgentRequest(
            user_id="u3",
            conversation_id="c3",
            message="我已经把简历改完了。",
            intent_result=IntentResult(intent=IntentType.ACADEMIC_CAREER, confidence=0.9),
            risk_assessment=RiskAssessment(risk_level=RiskLevel.LOW, confidence=0.9),
            policy=PolicyDecision(
                autonomy_level="flexible",
                allowed_agent=AgentName.GUIDANCE,
            ),
            case_state=state,
        )

    async def test_upsert_support_case_updates_structured_state(self):
        cases = InMemoryCaseStore()
        request = self.make_request(cases)
        registry = build_default_tool_registry(case_store=cases)

        result = await registry.execute(
            tool_name="upsert_support_case",
            arguments={
                "goal": "完成秋招投递",
                "add_actions_done": ["完成第一版简历"],
                "set_next_actions": ["开始岗位投递"],
            },
            request=request,
            allowed_tools={"upsert_support_case"},
        )

        state = cases.get(user_id="u3", conversation_id="c3")
        self.assertTrue(result["updated"])
        self.assertEqual(state.goal, "完成秋招投递")
        self.assertEqual(state.actions_done, ["完成第一版简历"])
        self.assertEqual(state.next_actions, ["开始岗位投递"])
        self.assertEqual(state.current_risk_level, RiskLevel.LOW)

    async def test_llm_cannot_override_case_risk_level(self):
        cases = InMemoryCaseStore()
        request = self.make_request(cases)
        registry = build_default_tool_registry(case_store=cases)

        with self.assertRaisesRegex(ToolArgumentError, "未声明参数"):
            await registry.execute(
                tool_name="upsert_support_case",
                arguments={"risk_level": "high"},
                request=request,
                allowed_tools={"upsert_support_case"},
            )

    async def test_runtime_can_call_case_tool_and_continue_llm(self):
        cases = InMemoryCaseStore()
        request = self.make_request(cases)
        llm = ScriptedLLMClient(
            [
                LLMResponse(
                    tool_calls=[
                        ToolCall(
                            id="case_1",
                            name="upsert_support_case",
                            arguments={
                                "add_actions_done": ["完成第一版简历"],
                                "set_next_actions": ["开始岗位投递"],
                            },
                        )
                    ]
                ),
                LLMResponse(content="已记录你的当前进度。"),
            ]
        )
        registry = build_default_tool_registry(case_store=cases)
        agent = build_agents(llm_client=llm, tool_registry=registry)[AgentName.GUIDANCE]

        result = await agent.run(request)
        state = cases.get(user_id="u3", conversation_id="c3")

        self.assertEqual(result.response, "已记录你的当前进度。")
        self.assertIn("upsert_support_case", result.tools_used)
        self.assertTrue(result.case_updated)
        self.assertEqual(state.actions_done, ["完成第一版简历"])
        self.assertEqual(state.next_actions, ["开始岗位投递"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
