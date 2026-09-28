import json
import unittest
from pathlib import Path

from agents.agent_orchestrator import Router
from agents.factory import build_agents
from agents.tools import build_default_tool_registry
from core.intent_analyzer import IntentAnalyzer
from core.llm_client import StaticLLMClient
from core.risk_analyzer import RiskAnalyzer
from core.risk_policy import RiskPolicyEngine
from core.schemas import AgentName
from rag.resource_store import VerifiedResourceStore
from state.case_store import InMemoryCaseStore


class GoldenCaseRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).with_name("golden_cases.json")
        cls.cases = json.loads(path.read_text(encoding="utf-8"))
        cls.intent = IntentAnalyzer()
        cls.risk = RiskAnalyzer()
        cls.policy = RiskPolicyEngine()

    def test_golden_cases_route_and_policy_are_stable(self):
        for case in self.cases:
            with self.subTest(case=case["name"]):
                intent_result = self.intent.analyze(case["message"])
                risk_result = self.risk.analyze(case["message"])
                policy = self.policy.decide(intent_result, risk_result)
                self.assertEqual(intent_result.intent.value, case["intent"])
                self.assertEqual(risk_result.risk_level.value, case["risk"])
                self.assertEqual(policy.allowed_agent.value, case["agent"])
                self.assertEqual(policy.mandatory_tools, case["mandatory_tools"])


class ToolScopeRegressionTests(unittest.TestCase):
    def setUp(self):
        case_store = InMemoryCaseStore()
        registry = build_default_tool_registry(
            case_store=case_store,
            resource_store=VerifiedResourceStore.demo(),
        )
        self.agents = build_agents(
            llm_client=StaticLLMClient("ok"),
            tool_registry=registry,
        )

    def test_guidance_does_not_expose_rag_for_general_advice(self):
        guidance = self.agents[AgentName.GUIDANCE]
        self.assertNotIn("search_verified_resources", guidance.allowed_tools)
        self.assertNotIn("get_support_case", guidance.allowed_tools)
        self.assertEqual(guidance.allowed_tools, {"upsert_support_case"})

    def test_resource_agent_has_only_verified_retrieval(self):
        resource = self.agents[AgentName.RESOURCE]
        self.assertEqual(resource.allowed_tools, {"search_verified_resources"})

    def test_escalation_has_exact_mandatory_tool_permissions(self):
        escalation = self.agents[AgentName.ESCALATION]
        self.assertIn("search_verified_resources", escalation.allowed_tools)
        self.assertIn("upsert_support_case", escalation.allowed_tools)

    def test_general_agent_exposes_no_tools(self):
        self.assertEqual(self.agents[AgentName.GENERAL].allowed_tools, set())


if __name__ == "__main__":
    unittest.main(verbosity=2)
