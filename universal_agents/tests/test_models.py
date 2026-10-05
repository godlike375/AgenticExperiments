import json
import unittest
from unittest import mock

from universal_agents.history import ChatHistory
from universal_agents.models import (
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolCall,
    ToolResult,
)
from universal_agents.config import Config

B64 = "aGVsbG8taW1hZ2U="


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
        original = Config.KEEP_REASONING_CONTENT_IN_HISTORY
        try:
            Config.KEEP_REASONING_CONTENT_IN_HISTORY = False
            d = msg.to_api_dict()
            self.assertEqual(d["role"], "assistant")
            self.assertEqual(d["content"], "hi")
            # При KEEP_REASONING_CONTENT_IN_HISTORY=False reasoning_content не попадает в контекст.
            self.assertNotIn("reasoning_content", d)
            self.assertEqual(len(d["tool_calls"]), 1)
            self.assertTrue(msg.has_tool_calls())
        finally:
            Config.KEEP_REASONING_CONTENT_IN_HISTORY = original

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


class TestImageSerialization(unittest.TestCase):
    """to_api_dict: строка без картинок, список частей с картинками."""

    def test_tool_result_without_images_is_plain_str(self):
        tr = ToolResult.success("t1", "read", "ok")
        d = tr.to_api_dict()
        self.assertIsInstance(d["content"], str)
        self.assertEqual(d["content"], "ok")

    def test_user_message_without_images_is_plain_str(self):
        d = UserMessage("просто текст").to_api_dict()
        self.assertIsInstance(d["content"], str)

    def test_tool_result_with_images_is_parts(self):
        tr = ToolResult("t1", "screenshot", "Скриншот 1920x1080", images=[B64])
        d = tr.to_api_dict()
        self.assertIsInstance(d["content"], list)
        self.assertEqual(d["content"][0], {"type": "text", "text": "Скриншот 1920x1080"})
        self.assertEqual(d["content"][1]["type"], "image_url")
        self.assertTrue(
            d["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        )

    def test_user_message_with_images_is_parts(self):
        d = UserMessage("текст", images=[B64]).to_api_dict()
        self.assertIsInstance(d["content"], list)
        self.assertEqual(d["content"][0]["type"], "text")
        self.assertEqual(d["content"][1]["type"], "image_url")

    def test_serialization_is_stable_between_calls(self):
        """KV-кэш: повторная сериализация одного объекта байт-идентична."""
        tr = ToolResult("t1", "screenshot", "ok", images=[B64])
        um = UserMessage("текст", images=[B64])
        for msg in (tr, um):
            a = json.dumps(msg.to_api_dict(), ensure_ascii=False, sort_keys=True)
            b = json.dumps(msg.to_api_dict(), ensure_ascii=False, sort_keys=True)
            self.assertEqual(a, b)


class TestImagePersist(unittest.TestCase):
    """to_persist_dict / load_from_payload: картинки в JSON только под флагом."""

    def test_persist_strips_images_by_default(self):
        tr = ToolResult("t1", "screenshot", "ок", images=[B64])
        um = UserMessage("текст", images=[B64])
        with mock.patch.object(Config, "SAVE_IMAGES", False):
            self.assertNotIn("_images", tr.to_persist_dict())
            self.assertNotIn("_images", um.to_persist_dict())
            self.assertIsInstance(tr.to_persist_dict()["content"], str)
            self.assertIsInstance(um.to_persist_dict()["content"], str)

    def test_persist_keeps_images_with_flag(self):
        tr = ToolResult("t1", "screenshot", "ок", images=[B64])
        with mock.patch.object(Config, "SAVE_IMAGES", True):
            self.assertEqual(tr.to_persist_dict()["_images"], [B64])

    def test_save_load_roundtrip_with_images(self):
        h = ChatHistory("sys")
        h.add(UserMessage("привет", images=[B64]))
        h.add(ToolResult("t1", "screenshot", "ок", images=[B64]))
        with mock.patch.object(Config, "SAVE_IMAGES", True):
            payload = {"messages": [m.to_persist_dict() for m in h.get_all()]}

        h2 = ChatHistory("sys")
        h2.load_from_payload(payload)
        users = [m for m in h2.get_all() if isinstance(m, UserMessage)]
        tools = [m for m in h2.get_all() if isinstance(m, ToolResult)]
        self.assertEqual(users[0].images, [B64])
        self.assertEqual(tools[0].images, [B64])
        # content восстанавливается строкой.
        self.assertEqual(users[0].content, "привет")
        self.assertEqual(tools[0].content, "ок")

    def test_load_without_images_field(self):
        h = ChatHistory("sys")
        payload = {"messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "tool", "tool_call_id": "t1", "name": "screenshot", "content": "ok"},
        ]}
        h.load_from_payload(payload)
        users = [m for m in h.get_all() if isinstance(m, UserMessage)]
        tools = [m for m in h.get_all() if isinstance(m, ToolResult)]
        self.assertEqual(users[0].images, [])
        self.assertEqual(tools[0].images, [])


if __name__ == "__main__":
    unittest.main()
