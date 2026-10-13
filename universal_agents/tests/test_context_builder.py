
from __future__ import annotations

import unittest

from universal_agents.constants import ENVIRONMENT_PREFIX
from universal_agents.context_builder import prepare_messages_for_api
from universal_agents.models import AssistantMessage, ToolCall, ToolResult, UserMessage
from universal_agents.tools.fs import read as _read_tool

from tests.conftest import make_agent

B64 = "aGVsbG8taW1hZ2U="


class TestContextBuilderImages(unittest.TestCase):

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


class TestHeaderCachePolicy(unittest.TestCase):

    def _agent_with_dialog(self):
        agent = make_agent()
        agent.history.add(UserMessage("задача"))
        agent.history.add(AssistantMessage(
            content="смотрю", tool_calls=[ToolCall(id="t1", name=_read_tool.__name__, arguments="{}")]))
        agent.history.add(ToolResult.success("t1", _read_tool.__name__, "данные"))
        agent.history.add(UserMessage("уточнение"))
        agent.history.add(AssistantMessage(content="вижу"))
        return agent

    def test_removing_tail_refreshes_surviving_question_budget(self):
        agent = self._agent_with_dialog()
        prepare_messages_for_api(agent)
        agent.token_tracker.update_from_usage({"prompt_tokens": 50_000, "total_tokens": 50_000})
        before = prepare_messages_for_api(agent)
        self.assertNotIn("Memory:", before[1]["content"])

        agent.history.remove_at({4, 5})
        after = prepare_messages_for_api(agent)

        self.assertEqual(after[0], before[0])
        self.assertIn("Memory:", after[1]["content"])

    def test_budget_line_is_a_snapshot_of_first_render(self):
        agent = self._agent_with_dialog()
        first = prepare_messages_for_api(agent)[1]["content"]
        agent.token_tracker.update_from_usage({"prompt_tokens": 90_000, "total_tokens": 90_000})
        self.assertEqual(prepare_messages_for_api(agent)[1]["content"], first)


    def test_deleting_before_a_message_keeps_its_header(self):
        agent = self._agent_with_dialog()
        prepare_messages_for_api(agent)
        target = agent.history[4]  # «уточнение» — последний user
        header = target._cached_header
        self.assertIsNotNone(header)

        agent.history.delete_range(1, 3)
        self.assertIs(agent.history[1], target)

        msgs = prepare_messages_for_api(agent, debug_hash_check=True)
        self.assertEqual(target._cached_header, header)
        self.assertTrue(msgs[1]["content"].startswith(header))

    def test_position_shift_without_byte_change_is_not_reported(self):
        agent = self._agent_with_dialog()
        warnings: list[str] = []
        agent.on_system_msg = warnings.append
        prepare_messages_for_api(agent, debug_hash_check=True)

        agent.history.delete_range(1, 3)
        prepare_messages_for_api(agent, debug_hash_check=True)

        self.assertEqual(warnings, [])

    def test_mutation_before_last_user_reports_prefix_change(self):
        agent = self._agent_with_dialog()
        warnings: list[str] = []
        agent.on_system_msg = warnings.append
        prepare_messages_for_api(agent, debug_hash_check=True)
        agent.token_tracker.update_from_usage({"prompt_tokens": 70_000, "total_tokens": 70_000})

        agent.history.delete_range(1, 3)
        prepare_messages_for_api(agent, debug_hash_check=True)

        self.assertTrue(any("[PREFIX-HASH]" in w for w in warnings), warnings)


if __name__ == "__main__":
    unittest.main()
