import unittest

from universal_agents.models import UserMessage, AssistantMessage, ToolCall, ToolResult
from universal_agents.rendering import render_message
from universal_agents.constants import ENVIRONMENT_PREFIX


class TestRendering(unittest.TestCase):
    def test_user_message(self):
        self.assertEqual(render_message(UserMessage("hello")), "👤 User: hello")

    def test_assistant_message(self):
        msg = AssistantMessage(content="answer", reasoning_content="thinking")
        rendered = render_message(msg)
        self.assertIn("answer", rendered)
        self.assertIn("thinking", rendered)

    def test_assistant_streamed_hides_replayed_text(self):
        msg = AssistantMessage(content="answer", reasoning_content="thinking", streamed=True)
        rendered = render_message(msg)
        self.assertNotIn("answer", rendered)
        self.assertNotIn("thinking", rendered)

    def test_assistant_with_tool_call(self):
        msg = AssistantMessage(content="", tool_calls=[ToolCall(id="t1", name="read", arguments="{}")])
        rendered = render_message(msg)
        self.assertIn("read", rendered)

    def test_tool_result(self):
        ok = render_message(ToolResult.success("t1", "read", "data"))
        self.assertIn("✅", ok)
        err = render_message(ToolResult.error("t1", "read", "boom"))
        self.assertIn("❌", err)

    def test_system_message_empty(self):
        from universal_agents.models import SystemMessage
        self.assertEqual(render_message(SystemMessage("sys")), "")


if __name__ == "__main__":
    unittest.main()
