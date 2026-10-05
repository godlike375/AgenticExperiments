import unittest

from universal_agents.models import UserMessage, AssistantMessage, ToolCall, ToolResult
from universal_agents.rendering import render_message
from universal_agents.constants import ENVIRONMENT_PREFIX

B64 = "aGVsbG8taW1hZ2U="


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


class TestRenderingImages(unittest.TestCase):
    def test_render_shows_image_marker(self):
        tr = ToolResult("t1", "screenshot", "ок", images=[B64, B64])
        self.assertIn("[+2 image(s)]", render_message(tr))
        um = UserMessage("текст", images=[B64])
        self.assertIn("[+1 image(s)]", render_message(um))
        um_plain = UserMessage("текст")
        self.assertNotIn("image", render_message(um_plain))


if __name__ == "__main__":
    unittest.main()
