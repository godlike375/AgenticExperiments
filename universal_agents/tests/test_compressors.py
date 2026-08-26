import unittest
from unittest import mock
from types import SimpleNamespace
from universal_agents.compressors import (
    is_summary_message,
    _find_existing_summary,
)
from universal_agents.history import ChatHistory
from universal_agents.models import UserMessage, AssistantMessage, ToolResult
from universal_agents.config import Config
from universal_agents.llm_client import LLMClient


def _fake_msg(content="... summary ..."):
    return SimpleNamespace(content=content, tool_calls=None)


def _history():
    h = ChatHistory("sys")
    h.add(UserMessage("task: create readme"))
    h.add(AssistantMessage(content="a1"))
    h.add(ToolResult.success("t1", "read", "some file content"))
    h.add(AssistantMessage(content="a2"))
    h.add(UserMessage("please summarize"))
    return h


def _history_with_summary():
    h = ChatHistory("sys")
    h.add(UserMessage("old facts\nKEY FACTS: src/main.py\nold decision X", is_summary=True))
    h.add(UserMessage("new: fix bug in main.py"))
    h.add(AssistantMessage(content="fixed"))
    return h


class TestSummaryHelpers(unittest.TestCase):
    def test_is_summary_message(self):
        # Детекция только по метаданным объекта, без текстовых маркеров
        self.assertFalse(is_summary_message(UserMessage("xyz")))
        self.assertFalse(is_summary_message(UserMessage("[SUMMARY of messages 1-5]: xyz")))
        self.assertFalse(is_summary_message(UserMessage("It's an auto-generated text. Your past dialog summary")))
        self.assertTrue(is_summary_message(UserMessage("flagged", is_summary=True)))
        self.assertFalse(is_summary_message(UserMessage("normal user message")))
        self.assertFalse(is_summary_message(UserMessage("")))

    def test_find_existing_summary(self):
        msgs = _history().get_all()
        self.assertIsNone(_find_existing_summary(msgs, len(msgs) - 1))

        h = _history()
        h.replace_range(2, 2, [UserMessage("earlier facts\npath /etc/x", is_summary=True)])
        msgs = h.get_all()
        found = _find_existing_summary(msgs, len(msgs) - 1)
        self.assertIsNotNone(found)
        self.assertEqual(found["index"], 2)
        # body — это текст после первой строки-заголовка
        self.assertIn("path /etc/x", found["body"])
        self.assertIn("earlier facts", found["full_content"])


if __name__ == "__main__":
    unittest.main()