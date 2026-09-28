import unittest

from agents.base_agent import GuidanceAgent
from agents.tools import build_default_tool_registry
from core.llm_client import ScriptedLLMClient
from state.case_store import InMemoryCaseStore
from core.schemas import (
    AgentName,
    AgentRequest,
    IntentResult,
    IntentType,
    LLMResponse,
    PolicyDecision,
    RiskAssessment,
    RiskLevel,
    ToolCall,
)


def make_request() -> AgentRequest:
    return AgentRequest(
        user_id="u1",
        conversation_id="c1",
        message="帮我分析一下当前请求。",
        intent_result=IntentResult(intent=IntentType.ACADEMIC_CAREER, confidence=0.9),
        risk_assessment=RiskAssessment(risk_level=RiskLevel.LOW, confidence=0.8),
        policy=PolicyDecision(
            autonomy_level="flexible",
            allowed_agent=AgentName.GUIDANCE,
        ),
    )


class AgentRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_allowed_tool_is_executed_and_result_returns_to_llm(self):
        llm = ScriptedLLMClient(
            [
                LLMResponse(
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            name="inspect_request_context",
                            arguments={"focus": "risk"},
                        )
                    ]
                ),
                LLMResponse(content="已根据工具结果完成回答。"),
            ]
        )
        agent = GuidanceAgent(
            name=AgentName.GUIDANCE,
            system_prompt="test",
            llm_client=llm,
            tool_registry=build_default_tool_registry(case_store=InMemoryCaseStore()),
            allowed_tools={"inspect_request_context"},
        )

        result = await agent.run(make_request())

        self.assertEqual(result.response, "已根据工具结果完成回答。")
        self.assertEqual(result.tools_used, ["inspect_request_context"])
        self.assertEqual(result.runtime_rounds, 2)
        self.assertEqual(llm.call_count, 2)
        second_call_messages = llm.seen_messages[1]
        self.assertTrue(any(msg.get("role") == "tool" for msg in second_call_messages))

    async def test_disallowed_tool_is_rejected_by_runtime(self):
        llm = ScriptedLLMClient(
            [
                LLMResponse(
                    tool_calls=[
                        ToolCall(
                            id="call_forbidden",
                            name="inspect_request_context",
                            arguments={},
                        )
                    ]
                ),
                LLMResponse(content="已看到权限错误，不再调用工具。"),
            ]
        )
        agent = GuidanceAgent(
            name=AgentName.GUIDANCE,
            system_prompt="test",
            llm_client=llm,
            tool_registry=build_default_tool_registry(case_store=InMemoryCaseStore()),
            allowed_tools=set(),
        )

        result = await agent.run(make_request())

        self.assertEqual(result.tools_used, [])
        self.assertEqual(result.response, "已看到权限错误，不再调用工具。")
        tool_messages = [m for m in llm.seen_messages[1] if m.get("role") == "tool"]
        self.assertEqual(len(tool_messages), 1)
        self.assertIn("ToolPermissionError", tool_messages[0]["content"])

    async def test_max_rounds_stops_infinite_tool_loop(self):
        llm = ScriptedLLMClient(
            [
                LLMResponse(
                    tool_calls=[
                        ToolCall(id="c1", name="inspect_request_context", arguments={})
                    ]
                ),
                LLMResponse(
                    tool_calls=[
                        ToolCall(id="c2", name="inspect_request_context", arguments={})
                    ]
                ),
            ]
        )
        agent = GuidanceAgent(
            name=AgentName.GUIDANCE,
            system_prompt="test",
            llm_client=llm,
            tool_registry=build_default_tool_registry(case_store=InMemoryCaseStore()),
            allowed_tools={"inspect_request_context"},
            max_rounds=2,
        )

        result = await agent.run(make_request())

        self.assertIn("最大工具调用轮数", result.response)
        self.assertEqual(result.runtime_rounds, 2)
        self.assertEqual(result.tools_used, ["inspect_request_context"])
        self.assertEqual(
            [item.status for item in result.tool_executions],
            ["executed", "skipped_duplicate"],
        )
