import unittest

from universal_agents.llm_client import TokenUsageTracker
from universal_agents.models import UserMessage


class TestTokenUsageTracker(unittest.TestCase):
    def test_estimate_tokens(self):
        self.assertEqual(TokenUsageTracker.estimate_tokens("x" * 46), 20)
        self.assertEqual(TokenUsageTracker.estimate_tokens(""), 0)

    def test_remaining(self):
        tracker = TokenUsageTracker("system prompt", max_context_tokens=1000)
        tracker.update_from_usage({"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})
        self.assertEqual(tracker.last_usage["prompt_tokens"], 100)
        self.assertLessEqual(tracker.get_remaining(), 900)

    def test_format_user_token_info(self):
        tracker = TokenUsageTracker("sys", max_context_tokens=1000)
        self.assertEqual(tracker.format_user_token_info(), "")
        tracker.update_from_usage({"prompt_tokens": 300, "completion_tokens": 50, "total_tokens": 350})
        info = tracker.format_user_token_info()
        # «Tokens spent» берётся из поля 'total_tokens' последнего вызова,
        # «Remaining» — окно контекста (max - prompt_tokens последнего вызова).
        self.assertIn("350", info)
        self.assertIn("700", info)


if __name__ == "__main__":
    unittest.main()
