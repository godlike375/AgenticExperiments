import unittest

from universal_agents.llm_client import LoopDetector
from universal_agents.models import UserMessage, AssistantMessage, ToolCall


class TestLoopDetector(unittest.TestCase):
    def setUp(self):
        self.detector = LoopDetector()

    def test_normalize_args_ignores_whitespace_and_key_order(self):
        self.assertEqual(
            LoopDetector.normalize_args('{ "b": 2, "a": 1 }'),
            '{"a":1,"b":2}',
        )
        self.assertEqual(LoopDetector.normalize_args("{}"), "")
        self.assertEqual(LoopDetector.normalize_args(""), "")
        self.assertEqual(LoopDetector.normalize_args("not json"), "not json")

    def test_detects_duplicate_in_turn(self):
        history = [
            UserMessage("hi"),
            AssistantMessage(content="", tool_calls=[ToolCall(id="t1", name="read", arguments="{}")]),
        ]
        self.assertTrue(self.detector.check_duplicate_in_turn("read", "{}", history))

    def test_ignores_calls_before_user_message(self):
        history = [
            AssistantMessage(content="", tool_calls=[ToolCall(id="t1", name="read", arguments="{}")]),
            UserMessage("hi"),
            AssistantMessage(content="", tool_calls=[ToolCall(id="t2", name="search", arguments="{}")]),
        ]
        # read вызывался до начала хода — не считается дублем
        self.assertFalse(self.detector.check_duplicate_in_turn("read", "{}", history))
        # search уже вызван в текущем ходу с теми же аргументами — дубль
        self.assertTrue(self.detector.check_duplicate_in_turn("search", "{}", history))
        # другие аргументы — не дубль
        self.assertFalse(self.detector.check_duplicate_in_turn("search", '{"x": 1}', history))

    def test_detects_semantically_duplicate_after_user(self):
        history = [
            UserMessage("hi"),
            AssistantMessage(content="", tool_calls=[ToolCall(id="t1", name="read", arguments='{"a": 1}')]),
        ]
        self.assertTrue(self.detector.check_duplicate_in_turn("read", '{ "a" : 1 }', history))
        self.assertFalse(self.detector.check_duplicate_in_turn("read", '{"a": 2}', history))

    def test_repeated_create_plan_same_args_is_loop(self):
        history = [
            UserMessage("do the task"),
            AssistantMessage(content="", tool_calls=[
                ToolCall(id="c1", name="create_plan", arguments='{"plan":[{"id":"t2","title":"X"}]}')
            ]),
        ]
        self.assertTrue(self.detector.check_duplicate_in_turn(
            "create_plan", '{"plan":[{"id":"t2","title":"X"}]}', history))

    def test_create_plan_revision_with_different_args_is_allowed(self):
        history = [
            UserMessage("do the task"),
            AssistantMessage(content="", tool_calls=[
                ToolCall(id="c1", name="create_plan", arguments='{"plan":[{"id":"t1","title":"X"}]}')
            ]),
        ]
        self.assertFalse(self.detector.check_duplicate_in_turn(
            "create_plan", '{"plan":[{"id":"t2","title":"Y"}]}', history))

    def test_create_plan_resets_duplicate_scan_for_other_tools(self):
        history = [
            UserMessage("do the task"),
            AssistantMessage(content="", tool_calls=[ToolCall(id="r", name="read", arguments="{}")]),
            AssistantMessage(content="", tool_calls=[
                ToolCall(id="c1", name="create_plan", arguments='{"plan":[{"id":"t1","title":"X"}]}')
            ]),
        ]
        # Повторный read ПОСЛЕ create_plan не считается дублем (ревизия = граница контекста)
        self.assertFalse(self.detector.check_duplicate_in_turn("read", "{}", history))


if __name__ == "__main__":
    unittest.main()
