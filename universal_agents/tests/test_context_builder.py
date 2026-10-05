"""Тесты сборки API-сообщений с картинками (context_builder)."""

from __future__ import annotations

import unittest

from universal_agents.constants import ENVIRONMENT_PREFIX
from universal_agents.context_builder import prepare_messages_for_api
from universal_agents.models import AssistantMessage, ToolCall, ToolResult, UserMessage

from tests.conftest import make_agent

B64 = "aGVsbG8taW1hZ2U="


class TestContextBuilderImages(unittest.TestCase):
    """prepare_messages_for_api: шапка в текстовой части, картинки после неё."""

    def test_user_without_images_keeps_str_content(self):
        agent = make_agent()
        agent.history.add(UserMessage("обычное сообщение"))
        msgs = prepare_messages_for_api(agent)
        users = [m for m in msgs if m["role"] == "user"]
        self.assertIsInstance(users[-1]["content"], str)
        self.assertIn("обычное сообщение", users[-1]["content"])

    def test_user_with_images_header_inside_text_part(self):
        agent = make_agent()
        agent.history.add(UserMessage("текст задачи", images=[B64]))
        msgs = prepare_messages_for_api(agent)
        user = [m for m in msgs if m["role"] == "user"][-1]
        self.assertIsInstance(user["content"], list)
        text_part = user["content"][0]
        self.assertEqual(text_part["type"], "text")
        # Шапка (KV-заголовок с таймстемпом) и текст — в одной текстовой части.
        self.assertIn(ENVIRONMENT_PREFIX, text_part["text"])
        self.assertIn("текст задачи", text_part["text"])
        self.assertEqual(user["content"][1]["type"], "image_url")

    def test_tool_result_with_images_passes_through_builder(self):
        agent = make_agent()
        agent.history.add(UserMessage("задача"))
        agent.history.add(AssistantMessage(
            content="делаю скриншот",
            tool_calls=[ToolCall(id="t1", name="screenshot", arguments="{}")],
        ))
        agent.history.add(ToolResult("t1", "screenshot", "готово", images=[B64]))
        msgs = prepare_messages_for_api(agent)
        tool = [m for m in msgs if m["role"] == "tool"][-1]
        self.assertIsInstance(tool["content"], list)
        self.assertEqual(tool["content"][0]["type"], "text")


if __name__ == "__main__":
    unittest.main()
