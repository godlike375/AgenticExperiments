import unittest

from universal_agents.history import ChatHistory
from universal_agents.models import (
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolCall,
    ToolResult,
)


def history_with_dialog():
    h = ChatHistory("sys")
    h.add(UserMessage("u1"))
    h.add(AssistantMessage(content="a1", tool_calls=[ToolCall(id="t1", name="read", arguments="{}")]))
    h.add(ToolResult.success("t1", "read", "data"))
    h.add(AssistantMessage(content="a2"))
    h.add(UserMessage("u2"))
    h.add(AssistantMessage(content="a3"))
    return h


class TestChatHistory(unittest.TestCase):
    def test_initial_state(self):
        h = ChatHistory("sys")
        self.assertEqual(len(h), 1)
        self.assertIsInstance(h[0], SystemMessage)
        self.assertIsNone(h.get_last_message())

    def test_get_all_api(self):
        h = history_with_dialog()
        api = h.get_all_api()
        self.assertEqual(api[0], {"role": "system", "content": "sys"})
        self.assertEqual(api[1]["role"], "user")
        self.assertEqual(api[3]["role"], "tool")

    def test_remove_at(self):
        h = history_with_dialog()
        h.remove_at({3})  # tool result
        roles = [m.to_api_dict()["role"] for m in h]
        self.assertNotIn("tool", roles)
        self.assertEqual(len(h), 6)

    def test_replace_range(self):
        h = history_with_dialog()
        h.replace_range(1, 2, [UserMessage("summary")])
        self.assertEqual(len(h), 6)
        self.assertEqual(h[1].content, "summary")

    def test_normalize_keeps_sequence(self):
        h = history_with_dialog()
        h.normalize()
        roles = [m.to_api_dict()["role"] for m in h]
        self.assertEqual(roles, ["system", "user", "assistant", "tool", "assistant", "user", "assistant"])

    def test_normalize_merges_consecutive_same_type(self):
        h = ChatHistory("sys")
        h.add(UserMessage("u1"))
        h.add(UserMessage("u2"))
        h.add(AssistantMessage(content="a1"))
        h.add(AssistantMessage(content="a2"))
        h.normalize()
        roles = [m.to_api_dict()["role"] for m in h]
        self.assertEqual(roles, ["system", "user", "assistant"])
        self.assertEqual(h[1].content, "u1\n\nu2")

    def test_normalize_drops_orphan_tool_result(self):
        h = ChatHistory("sys")
        h.add(UserMessage("u1"))
        h.add(ToolResult.success("t1", "read", "data"))
        h.normalize()
        roles = [m.to_api_dict()["role"] for m in h]
        self.assertEqual(roles, ["system", "user"])

    def test_delete_range_invalidates_header_cache(self):
        h = history_with_dialog()
        h[5]._cached_header = "cached"
        h.delete_range(1, 3)
        # после удаления индексы сдвинулись: 0 sys, 1 a2, 2 u2, 3 a3
        self.assertIsNone(h[2]._cached_header)

    def test_compress_old_messages(self):
        h = history_with_dialog()
        h.compress_old_messages("long summary", preserve_last=2)
        self.assertEqual(h[0].to_api_dict()["role"], "system")
        self.assertTrue("summary" in h[1].content)
        # preserve_last = 2 последних сообщения сохраняются
        self.assertEqual(h[-1].content, "a3")

    def test_save_load_roundtrip(self, tmpdir=None):
        import tempfile, os
        h = history_with_dialog()
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "hist.json")
            h.save(path, loaded_tools=["read", "edit_file"])
            h2 = ChatHistory("sys")
            tools = h2.load(path)
            self.assertEqual(tools, ["read", "edit_file"])
            self.assertEqual(len(h2), len(h))


if __name__ == "__main__":
    unittest.main()
