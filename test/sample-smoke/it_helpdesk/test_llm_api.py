"""LLM_API selects the model client; the Anthropic path must stay valid for current Claude models."""
import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SAMPLE = Path(__file__).resolve().parents[3] / "samples" / "it-helpdesk-agent"
HAVE_DEPS = all(importlib.util.find_spec(m) for m in ("langchain_anthropic", "langchain_openai", "langgraph"))


@unittest.skipUnless(HAVE_DEPS, "needs the sample's requirements installed")
class LlmApiSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(SAMPLE))

    def build(self, **env):
        from agent import build_llm
        from config import Config
        with patch.dict(os.environ, env, clear=True):
            return build_llm(Config.from_env())

    def test_default_is_openai_unchanged(self):
        llm = self.build(OPENAI_API_KEY="k")
        self.assertEqual(type(llm).__name__, "ChatOpenAI")
        self.assertEqual(llm.model_name, "gpt-4o-mini")

    def test_anthropic_through_agent_manager_provider(self):
        llm = self.build(LLM_API="anthropic", USE_LLM_PROVIDER="true", LLM_EFFORT="low",
                         LLM_PROVIDER_URL="http://gw.example/claude", LLM_PROVIDER_KEY="platform-key")
        self.assertEqual(type(llm).__name__, "ChatAnthropic")
        self.assertEqual(llm.model, "claude-haiku-5-5")
        self.assertEqual(llm.anthropic_api_url, "http://gw.example/claude")
        self.assertEqual(llm.anthropic_api_key.get_secret_value(), "platform-key")
        self.assertEqual(llm.default_headers, {"API-Key": "platform-key"})
        self.assertEqual(llm.output_config, {"effort": "low"})
        # current Claude models reject non-default sampling parameters
        self.assertIsNone(llm.temperature)
        self.assertNotIn("temperature", llm._get_request_payload([("user", "hi")]))

    def test_anthropic_direct_without_effort(self):
        llm = self.build(LLM_API="anthropic", ANTHROPIC_API_KEY="k", LLM_MODEL="claude-sonnet-5-5")
        self.assertEqual(llm.model, "claude-sonnet-5-5")
        self.assertIsNone(llm.output_config)

    def test_unknown_api_is_rejected(self):
        from config import Config
        with patch.dict(os.environ, {"LLM_API": "gemini"}, clear=True):
            with self.assertRaises(RuntimeError):
                Config.from_env()


if __name__ == "__main__":
    unittest.main()
