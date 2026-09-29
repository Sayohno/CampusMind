import unittest

from core.llm_client import ScriptedLLMClient
from core.schemas import (
    IntentType,
    LLMResponse,
    RiskLevel,
)
from core.semantic_analyzer import (
    SemanticAnalyzer,
    SemanticAnalyzerError,
)


class SemanticAnalyzerTests(unittest.IsolatedAsyncioTestCase):
    async def test_semantic_analyzer_parses_structured_result(self):
        llm = ScriptedLLMClient(
            [
                LLMResponse(
                    content="""
                    {
                      "intent": "academic_career",
                      "intent_confidence": 0.93,
                      "risk_level": "low",
                      "risk_confidence": 0.88,
                      "signals": ["投出去很多份都没有回音"]
                    }
                    """
                )
            ]
        )

        analyzer = SemanticAnalyzer(llm)

        result = await analyzer.analyze(
            "投出去很多份都没有回音，我不知道下一步怎么调整。"
        )

        self.assertEqual(
            result.intent_result.intent,
            IntentType.ACADEMIC_CAREER,
        )
        self.assertEqual(
            result.risk_assessment.risk_level,
            RiskLevel.LOW,
        )
        self.assertEqual(llm.call_count, 1)

    async def test_semantic_analyzer_handles_contextual_hard_negative(self):
        llm = ScriptedLLMClient(
            [
                LLMResponse(
                    content="""
                    {
                      "intent": "general",
                      "intent_confidence": 0.96,
                      "risk_level": "low",
                      "risk_confidence": 0.98,
                      "signals": ["小说人物", "翻译"]
                    }
                    """
                )
            ]
        )

        analyzer = SemanticAnalyzer(llm)

        result = await analyzer.analyze(
            "小说人物说“我不想活了”，这句话怎么翻译得自然？"
        )

        self.assertEqual(
            result.intent_result.intent,
            IntentType.GENERAL,
        )
        self.assertEqual(
            result.risk_assessment.risk_level,
            RiskLevel.LOW,
        )

    async def test_semantic_analyzer_rejects_invalid_json(self):
        llm = ScriptedLLMClient(
            [
                LLMResponse(
                    content="我认为这是 academic_career。"
                )
            ]
        )

        analyzer = SemanticAnalyzer(llm)

        with self.assertRaises(SemanticAnalyzerError):
            await analyzer.analyze("测试")

    async def test_semantic_analyzer_sends_no_tools(self):
        llm = ScriptedLLMClient(
            [
                LLMResponse(
                    content="""
                    {
                      "intent": "general",
                      "intent_confidence": 0.8,
                      "risk_level": "low",
                      "risk_confidence": 0.9,
                      "signals": []
                    }
                    """
                )
            ]
        )

        analyzer = SemanticAnalyzer(llm)

        await analyzer.analyze("你好")

        self.assertEqual(llm.seen_tools[0], [])