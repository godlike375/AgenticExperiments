import unittest

from universal_agents.models import (
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolCall,
    ToolResult,
)
from universal_agents.config import Config


class TestMessages(unittest.TestCase):
    def test_tool_call_api_dict(self):
        tc = ToolCall(id="t1", name="read", arguments='{"path": "a.py"}')
        self.assertEqual(tc.to_api_dict(), {
            "id": "t1",
            "type": "function",
            "function": {"name": "read", "arguments": '{"path": "a.py"}'},
        })

    def test_assistant_message_api_dict(self):
        tc = ToolCall(id="t1", name="read", arguments="{}")
        msg = AssistantMessage(content="hi", tool_calls=[tc], reasoning_content="thinking")
        d = msg.to_api_dict()
        self.assertEqual(d["role"], "assistant")
        self.assertEqual(d["content"], "hi")
        # По умолчанию reasoning_content не попадает в контекст
        self.assertNotIn("reasoning_content", d)
        self.assertEqual(len(d["tool_calls"]), 1)
        self.assertTrue(msg.has_tool_calls())

    def test_assistant_message_reasoning_toggle(self):
        tc = ToolCall(id="t1", name="read", arguments="{}")
        msg = AssistantMessage(content="hi", tool_calls=[tc], reasoning_content="thinking")
        original = Config.KEEP_REASONING_CONTENT_IN_HISTORY
        try:
            Config.KEEP_REASONING_CONTENT_IN_HISTORY = True
            self.assertEqual(msg.to_api_dict()["reasoning_content"], "thinking")
            Config.KEEP_REASONING_CONTENT_IN_HISTORY = False
            self.assertNotIn("reasoning_content", msg.to_api_dict())
        finally:
            Config.KEEP_REASONING_CONTENT_IN_HISTORY = original

    def test_assistant_message_has_streamed_field(self):
        msg = AssistantMessage(content="x", streamed=True)
        self.assertTrue(msg.streamed)
        msg2 = AssistantMessage(content="x")
        self.assertFalse(msg2.streamed)

    def test_tool_result_factories(self):
        ok = ToolResult.success("t1", "read", "content")
        self.assertFalse(ok.is_error)
        self.assertFalse(ok.is_user_denied)
        self.assertEqual(ok.content, "content")

        err = ToolResult.error("t1", "read", "boom")
        self.assertTrue(err.is_error)
        self.assertTrue(err.content.startswith("Error:"))

        denied = ToolResult.user_denied("t1", "read")
        self.assertTrue(denied.is_user_denied)
        self.assertFalse(denied.is_error)

    def test_tool_result_api_dict(self):
        tr = ToolResult.success("t1", "read", "data")
        self.assertEqual(tr.to_api_dict(), {
            "role": "tool",
            "tool_call_id": "t1",
            "name": "read",
            "content": "data",
        })

    def test_system_and_user_roundtrip(self):
        self.assertEqual(SystemMessage("s").to_api_dict(), {"role": "system", "content": "s"})
        self.assertEqual(UserMessage("u").to_api_dict(), {"role": "user", "content": "u"})


if __name__ == "__main__":
    unittest.main()
