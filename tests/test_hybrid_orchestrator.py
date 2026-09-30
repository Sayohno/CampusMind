import unittest

from core.config import Settings
from core.llm_client import (
    ScriptedLLMClient,
    StaticLLMClient,
)
from core.schemas import (
    AgentName,
    ChatRequest,
    IntentType,
    LLMResponse,
    RiskLevel,
)
from core.semantic_analyzer import SemanticAnalyzer
from main import build_orchestrator


class HybridOrchestratorTests(
    unittest.IsolatedAsyncioTestCase
):
    def _build_hybrid(
        self,
        semantic_responses: list[LLMResponse],
    ):
        """
        Agent 使用 StaticLLMClient，
        SemanticAnalyzer 单独使用 ScriptedLLMClient。

        这样测试只验证：
        deterministic rule
            -> semantic fallback
            -> policy
            -> router

        不依赖真实 API。
        """
        agent_llm = StaticLLMClient(
            "这是用于集成测试的 Agent 回复。"
        )

        semantic_llm = ScriptedLLMClient(
            semantic_responses
        )

        orchestrator = build_orchestrator(
            Settings(),
            llm_client=agent_llm,
        )

        orchestrator.semantic_analyzer = (
            SemanticAnalyzer(semantic_llm)
        )

        return orchestrator, semantic_llm

    async def test_clear_request_skips_semantic_fallback(
        self,
    ):
        """
        明确请求应该走 deterministic fast path，
        不浪费一次 Semantic LLM 调用。
        """
        orchestrator, semantic_llm = (
            self._build_hybrid([])
        )

        result = await orchestrator.handle(
            ChatRequest(
                user_id="hybrid_user_1",
                conversation_id="hybrid_conv_1",
                message="秋招怎么准备？",
            )
        )

        self.assertEqual(
            result.intent_result.intent,
            IntentType.ACADEMIC_CAREER,
        )

        self.assertEqual(
            result.selected_agent,
            AgentName.GUIDANCE,
        )

        self.assertEqual(
            semantic_llm.call_count,
            0,
        )

    async def test_implicit_intent_uses_semantic_fallback(
        self,
    ):
        """
        固定关键词难以识别的隐式求职表达，
        应进入 SemanticAnalyzer。
        """
        orchestrator, semantic_llm = (
            self._build_hybrid(
                [
                    LLMResponse(
                        content="""
                        {
                          "intent": "academic_career",
                          "intent_confidence": 0.94,
                          "risk_level": "low",
                          "risk_confidence": 0.91,
                          "signals": [
                            "投出去很多份都没有回音"
                          ]
                        }
                        """
                    )
                ]
            )
        )

        result = await orchestrator.handle(
            ChatRequest(
                user_id="hybrid_user_2",
                conversation_id="hybrid_conv_2",
                message=(
                    "投出去很多份都没有回音，"
                    "我不知道下一步该怎么调整。"
                ),
            )
        )

        self.assertEqual(
            semantic_llm.call_count,
            1,
        )

        self.assertEqual(
            result.intent_result.intent,
            IntentType.ACADEMIC_CAREER,
        )

        self.assertEqual(
            result.risk_assessment.risk_level,
            RiskLevel.LOW,
        )

        self.assertEqual(
            result.selected_agent,
            AgentName.GUIDANCE,
        )

    async def test_quoted_high_risk_text_can_be_disambiguated(
        self,
    ):
        """
        风险词只是小说引用时，
        SemanticAnalyzer 可以确认它不是用户本人风险。
        """
        orchestrator, semantic_llm = (
            self._build_hybrid(
                [
                    LLMResponse(
                        content="""
                        {
                          "intent": "general",
                          "intent_confidence": 0.97,
                          "risk_level": "low",
                          "risk_confidence": 0.98,
                          "signals": [
                            "小说人物",
                            "翻译"
                          ]
                        }
                        """
                    )
                ]
            )
        )

        result = await orchestrator.handle(
            ChatRequest(
                user_id="hybrid_user_3",
                conversation_id="hybrid_conv_3",
                message=(
                    "小说人物说“我不想活了”，"
                    "这句话怎么翻译得自然？"
                ),
            )
        )

        self.assertEqual(
            semantic_llm.call_count,
            1,
        )

        self.assertEqual(
            result.intent_result.intent,
            IntentType.GENERAL,
        )

        self.assertEqual(
            result.risk_assessment.risk_level,
            RiskLevel.LOW,
        )

        self.assertEqual(
            result.selected_agent,
            AgentName.GENERAL,
        )

        self.assertFalse(
            result.policy_decision.force_escalation
        )

    async def test_semantic_failure_with_high_signal_fails_closed(
        self,
    ):
        """
        如果本来需要进行 high-risk 语义消歧，
        但 SemanticAnalyzer 自己失败，
        系统不能因为模型/API 故障而漏掉潜在高风险。

        应 fail-closed -> HIGH -> EscalationAgent。
        """
        orchestrator, semantic_llm = (
            self._build_hybrid(
                [
                    LLMResponse(
                        content="这不是合法 JSON"
                    )
                ]
            )
        )

        result = await orchestrator.handle(
            ChatRequest(
                user_id="hybrid_user_4",
                conversation_id="hybrid_conv_4",
                message=(
                    "小说人物说“我不想活了”，"
                    "这句话怎么翻译得自然？"
                ),
            )
        )

        self.assertEqual(
            semantic_llm.call_count,
            1,
        )

        self.assertEqual(
            result.risk_assessment.risk_level,
            RiskLevel.HIGH,
        )

        self.assertTrue(
            result.policy_decision.force_escalation
        )

        self.assertEqual(
            result.selected_agent,
            AgentName.ESCALATION,
        )

    async def test_real_mode_builds_semantic_analyzer(
        self,
    ):
        """
        验证 main.py 的装配关系：

        static -> 不启用 SemanticAnalyzer
        real   -> 启用 SemanticAnalyzer

        这里仍然不会访问网络。
        """
        fake_real_llm = ScriptedLLMClient([])

        orchestrator = build_orchestrator(
            Settings(
                llm_mode="real",
            ),
            llm_client=fake_real_llm,
        )

        self.assertIsNotNone(
            orchestrator.semantic_analyzer
        )

        self.assertIsInstance(
            orchestrator.semantic_analyzer,
            SemanticAnalyzer,
        )


if __name__ == "__main__":
    unittest.main()