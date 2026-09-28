import unittest

from fastapi.testclient import TestClient

from core.config import Settings
from core.llm_client import ScriptedLLMClient, StaticLLMClient
from core.schemas import LLMResponse, ToolCall
from main import build_orchestrator, create_app
from observability.trace_store import InMemoryTraceStore


class ApiHealthAndTraceTests(unittest.TestCase):
    def setUp(self):
        settings = Settings(llm_mode="static")
        orchestrator = build_orchestrator(
            settings,
            llm_client=StaticLLMClient("这是一个普通求职建议。"),
        )
        self.client = TestClient(
            create_app(
                settings=settings,
                orchestrator=orchestrator,
                trace_store=InMemoryTraceStore(),
            )
        )

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["service"], "CampusMind")
        self.assertEqual(body["llm_mode"], "static")

    def test_json_response_declares_utf8(self):
        response = self.client.get("/health")
        self.assertIn("application/json", response.headers["content-type"])
        self.assertIn("charset=utf-8", response.headers["content-type"].lower())

    def test_low_risk_chat_returns_request_id_and_trace(self):
        response = self.client.post(
            "/chat",
            json={
                "user_id": "api_u1",
                "conversation_id": "api_c1",
                "message": "秋招怎么准备？",
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["intent"], "academic_career")
        self.assertEqual(body["risk"], "low")
        self.assertEqual(body["selected_agent"], "GuidanceAgent")
        self.assertTrue(body["request_id"].startswith("req_"))

        trace_response = self.client.get(f"/traces/{body['request_id']}")
        self.assertEqual(trace_response.status_code, 200)
        trace = trace_response.json()
        self.assertEqual(trace["status"], "success")
        self.assertEqual(trace["selected_agent"], "GuidanceAgent")
        self.assertGreaterEqual(trace["latency_ms"], 0)

    def test_trace_does_not_store_raw_user_message(self):
        response = self.client.post(
            "/chat",
            json={
                "user_id": "private_user",
                "conversation_id": "private_conv",
                "message": "这是不应该出现在 Trace 里的原始文本",
            },
        )
        request_id = response.json()["request_id"]
        trace_text = self.client.get(f"/traces/{request_id}").text
        self.assertNotIn("这是不应该出现在 Trace 里的原始文本", trace_text)
        self.assertNotIn("private_user", trace_text)


class ApiGoldenCaseTests(unittest.TestCase):
    def test_resource_query_calls_rag_and_returns_grounded_response(self):
        llm = ScriptedLLMClient(
            [
                LLMResponse(
                    tool_calls=[
                        ToolCall(
                            id="rag_http_1",
                            name="search_verified_resources",
                            arguments={"query": "学校心理咨询中心怎么预约？", "top_k": 3},
                        )
                    ]
                ),
                LLMResponse(
                    content=(
                        "当前演示知识库无法确认真实学校的具体预约方式、电话或开放时间，"
                        "请以学校正式渠道为准。"
                    )
                ),
            ]
        )
        settings = Settings(llm_mode="static")
        app = create_app(
            settings=settings,
            orchestrator=build_orchestrator(settings, llm_client=llm),
            trace_store=InMemoryTraceStore(),
        )
        client = TestClient(app)

        response = client.post(
            "/chat",
            json={
                "user_id": "api_resource",
                "conversation_id": "api_resource_conv",
                "message": "学校心理咨询中心怎么预约？",
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["selected_agent"], "ResourceAgent")
        self.assertIn("search_verified_resources", body["tools_used"])
        self.assertTrue(body["guard_passed"])
        self.assertFalse(body["safe_fallback_used"])
        self.assertIn("无法确认", body["response"])

    def test_high_risk_http_case_forces_escalation_and_mandatory_tools(self):
        llm = ScriptedLLMClient(
            [
                LLMResponse(
                    content=(
                        "请先联系身边可信赖的人或学校正式支持渠道。"
                        "当前演示资料无法确认具体电话、地址或开放时间；"
                        "如存在现实中的紧急危险，请联系当地紧急服务。"
                    )
                )
            ]
        )
        settings = Settings(llm_mode="static")
        app = create_app(
            settings=settings,
            orchestrator=build_orchestrator(settings, llm_client=llm),
            trace_store=InMemoryTraceStore(),
        )
        client = TestClient(app)

        response = client.post(
            "/chat",
            json={
                "user_id": "api_high",
                "conversation_id": "api_high_conv",
                "message": "秋招一直失败，我真的快撑不下去了。",
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["risk"], "high")
        self.assertEqual(body["selected_agent"], "EscalationAgent")
        self.assertIn("search_verified_resources", body["tools_used"])
        self.assertIn("upsert_support_case", body["tools_used"])
        self.assertTrue(body["guard_passed"])
        self.assertEqual(body["case_state"]["status"], "escalated")

        trace = client.get(f"/traces/{body['request_id']}").json()
        self.assertEqual(trace["risk"], "high")
        self.assertEqual(trace["selected_agent"], "EscalationAgent")
        traced_tools = {item["name"] for item in trace["tool_executions"]}
        self.assertIn("search_verified_resources", traced_tools)
        self.assertIn("upsert_support_case", traced_tools)
        for item in trace["tool_executions"]:
            self.assertIn("argument_keys", item)
            self.assertNotIn("arguments", item)


if __name__ == "__main__":
    unittest.main(verbosity=2)
