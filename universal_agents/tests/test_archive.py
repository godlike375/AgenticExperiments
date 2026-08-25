import unittest

from universal_agents.archive import HistoryArchive
from universal_agents.models import (
    AssistantMessage,
    ToolCall,
    ToolResult,
    UserMessage,
)
from universal_agents.history import ChatHistory


def _dialog() -> list:
    h = ChatHistory("sys")
    h.add(UserMessage("please fix the login bug in src/auth.py"))
    h.add(AssistantMessage(
        content="Investigating the auth flow.",
        tool_calls=[ToolCall(id="t1", name="read", arguments='{"path": "src/auth.py"}')],
    ))
    h.add(ToolResult.success("t1", "read", "def login(): raise ValueError('bad token')"))
    h.add(AssistantMessage(content="Found it: token expiry check inverted. Fixed and added test."))
    h.add(UserMessage("why did you touch the middleware?"))
    return h.get_all()[1:]


class TestArchiveBasics(unittest.TestCase):
    def test_append_and_len(self):
        a = HistoryArchive()
        msgs = _dialog()
        self.assertEqual(a.append_messages(msgs), len(msgs))
        self.assertEqual(len(a), 5)

    def test_search_finds_content(self):
        a = HistoryArchive()
        a.append_messages(_dialog())
        out = a.search("login bug")
        self.assertIn("seq=", out)
        self.assertIn("auth.py", out)

    def test_search_by_tool_call_arguments(self):
        a = HistoryArchive()
        a.append_messages(_dialog())
        out = a.search("src/auth.py")
        self.assertIn("ASSISTANT", out)  # вызов инструмента виден в контенте ассистента

    def test_role_filter(self):
        a = HistoryArchive()
        a.append_messages(_dialog())
        only_user = a.search("middleware", role="user")
        self.assertIn("USER", only_user)
        none_assistant = a.search("middleware", role="assistant")
        self.assertIn("No matches", none_assistant)

    def test_no_matches_message(self):
        a = HistoryArchive()
        a.append_messages(_dialog())
        self.assertIn("No matches", a.search("quantum entanglement"))

    def test_empty_query(self):
        self.assertIn("Empty", HistoryArchive().search(""))


class TestReadSpan(unittest.TestCase):
    def setUp(self):
        self.archive = HistoryArchive()
        self.msgs = _dialog()
        self.archive.append_messages(self.msgs)

    def test_read_range_inclusive(self):
        out = self.archive.read_span(2, 4)
        self.assertIn("#2", out)
        self.assertIn("#3", out)
        self.assertIn("#4", out)
        self.assertNotIn("#5 USER", out.split("#5")[1] if "#5" in out else "")

    def test_read_missing_range_hint(self):
        out = self.archive.read_span(100, 200)
        self.assertIn("Archived range covers seq 1..5", out.replace("\n", " "))

    def test_max_chars_truncation(self):
        out = self.archive.read_span(2, 6, max_chars=120)
        self.assertLessEqual(len(out), 400)


if __name__ == "__main__":
    unittest.main()
