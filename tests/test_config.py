import os
import unittest
from unittest.mock import patch

from core.config import Settings, build_llm_client
from core.llm_client import OpenAICompatibleLLMClient, StaticLLMClient


class SettingsTests(unittest.TestCase):
    def test_static_mode_needs_no_api_key(self):
        settings = Settings(llm_mode="static")
        client = build_llm_client(settings)
        self.assertIsInstance(client, StaticLLMClient)

    def test_real_mode_builds_openai_compatible_client(self):
        settings = Settings(
            llm_mode="real",
            llm_api_key="test-key",
            llm_base_url="https://example.test/v1",
            llm_model="test-model",
        )
        client = build_llm_client(settings)
        self.assertIsInstance(client, OpenAICompatibleLLMClient)
        self.assertEqual(client.base_url, "https://example.test/v1")
        self.assertEqual(client.model, "test-model")

    def test_real_mode_missing_config_fails_fast(self):
        with self.assertRaisesRegex(ValueError, "LLM_API_KEY"):
            build_llm_client(Settings(llm_mode="real"))

    def test_settings_reads_environment(self):
        env = {
            "LLM_MODE": "real",
            "LLM_API_KEY": "abc",
            "LLM_BASE_URL": "https://example.test/v1",
            "LLM_MODEL": "model-x",
            "CAMPUSMIND_DEMO_MESSAGE": "测试消息",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.llm_mode, "real")
        self.assertEqual(settings.llm_api_key, "abc")
        self.assertEqual(settings.demo_message, "测试消息")


if __name__ == "__main__":
    unittest.main(verbosity=2)
