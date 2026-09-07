import collections
import threading
import unittest

from universal_agents.ui import CLI
from universal_agents.agent import LLMAgent


class TestCLIPollInput(unittest.TestCase):
    def _make_cli(self):
        cli = CLI(LLMAgent(system_prompt="sys"))
        cli._line_queue = collections.deque()
        cli._line_lock = threading.Lock()
        return cli

    def test_returns_user_text(self):
        cli = self._make_cli()
        cli._line_queue.append("hello world\n")
        self.assertEqual(cli._poll_input(), "hello world")
        self.assertEqual(list(cli._line_queue), [])

    def test_ignores_empty_injected_line(self):
        """Пустая строка ('\n', инжектированная _request_stop для разблокировки воркера)
        не должна трактоваться как пользовательский ввод и вызывать повторное прерывание."""
        cli = self._make_cli()
        cli._line_queue.append("\n")
        self.assertIsNone(cli._poll_input())
        # строка остаётся в очереди — её может прочитать ожидающий воркер (_get_line)
        self.assertEqual(list(cli._line_queue), ["\n"])

    def test_ignores_command(self):
        cli = self._make_cli()
        cli._line_queue.append("/regen 1\n")
        self.assertIsNone(cli._poll_input())
        self.assertEqual(list(cli._line_queue), ["/regen 1\n"])

    def test_returns_stop_command(self):
        cli = self._make_cli()
        cli._line_queue.append("q\n")
        self.assertEqual(cli._poll_input(), "q")


if __name__ == "__main__":
    unittest.main()