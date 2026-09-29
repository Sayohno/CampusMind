import unittest

from fastapi.testclient import TestClient

from core.config import Settings
from core.llm_client import StaticLLMClient
from main import build_orchestrator, create_app
from observability.trace_store import InMemoryTraceStore


class SseAndVueDemoTests(unittest.TestCase):
    def setUp(self):
        settings = Settings(llm_mode="static")
        orchestrator = build_orchestrator(
            settings,
            llm_client=StaticLLMClient("这是 SSE Demo 的最终回答。"),
        )
        self.client = TestClient(
            create_app(
                settings=settings,
                orchestrator=orchestrator,
                trace_store=InMemoryTraceStore(),
            )
        )

    def test_demo_page_is_served(self):
        response = self.client.get("/demo")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("CampusMind Agent Demo", response.text)
        self.assertIn("Vue", response.text)
        self.assertIn("/chat/stream", response.text)

    def test_stream_endpoint_emits_sse_contract_and_trace(self):
        response = self.client.post(
            "/chat/stream",
            json={
                "user_id": "sse_user",
                "conversation_id": "sse_conv",
                "message": "秋招怎么准备？",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        body = response.text
        self.assertIn("event: accepted", body)
        self.assertIn("event: stage", body)
        self.assertIn("event: meta", body)
        self.assertIn("event: chunk", body)
        self.assertIn("event: done", body)
        self.assertIn('"intent":"academic_career"', body)
        self.assertIn('"selected_agent":"GuidanceAgent"', body)
        self.assertIn("这是 SSE Demo 的最终回答。", body)

        # 从 accepted 事件里取 request_id，并确认 Trace 已经落库。
        request_id = None
        for block in body.split("\n\n"):
            if "event: accepted" not in block:
                continue
            for line in block.splitlines():
                if line.startswith("data:"):
                    import json

                    request_id = json.loads(line[5:].strip())["request_id"]
        self.assertIsNotNone(request_id)
        trace = self.client.get(f"/traces/{request_id}")
        self.assertEqual(trace.status_code, 200)
        self.assertEqual(trace.json()["selected_agent"], "GuidanceAgent")


if __name__ == "__main__":
    unittest.main(verbosity=2)
