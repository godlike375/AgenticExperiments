import collections
import threading
import time
import unittest
from unittest import mock

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


class TestCompactHistory(unittest.TestCase):
    def test_command_registered(self):
        cli = CLI(LLMAgent(system_prompt="sys"))
        self.assertIn("/compact_history", cli.commands)
        self.assertEqual(cli.commands["/compact_history"].__func__, cli.cmd_compact_history.__func__)

    def test_cmd_compact_history_compacted(self):
        cli = CLI(LLMAgent(system_prompt="sys"))
        seen = []
        with mock.patch("universal_agents.ui.ConsoleUI.system_msg", side_effect=seen.append):
            with mock.patch.object(cli.agent, "_auto_summarize_dialogue", return_value=True) as comp:
                with mock.patch.object(cli.agent, "_get_context_usage_percent", side_effect=[50.0, 30.0]):
                    cli.cmd_compact_history([])
        comp.assert_called_once_with(force=True)
        self.assertTrue(any("History compacted" in msg and "50% -> 30%" in msg for msg in seen))

    def test_cmd_compact_history_left_unchanged(self):
        cli = CLI(LLMAgent(system_prompt="sys"))
        seen = []
        with mock.patch("universal_agents.ui.ConsoleUI.system_msg", side_effect=seen.append):
            with mock.patch.object(cli.agent, "_auto_summarize_dialogue", return_value=False):
                with mock.patch.object(cli.agent, "_get_context_usage_percent", side_effect=[70.0, 70.0]):
                    cli.cmd_compact_history([])
        self.assertTrue(any("History left unchanged" in msg for msg in seen))

    def _make_monitored_cli(self, slow_compact):
        """CLI с полноценным stdin-монитором и компакцией, блокирующейся до stop_event."""
        agent = mock.Mock()
        agent.stop_event = threading.Event()
        agent.request_stop = agent.stop_event.set
        agent._get_context_usage_percent.return_value = 50.0
        agent._auto_summarize_dialogue.side_effect = slow_compact
        cli = CLI(agent)
        cli._line_queue = collections.deque()
        cli._line_lock = threading.Lock()
        cli._line_cond = threading.Condition(cli._line_lock)
        return cli

    def test_cmd_compact_history_stops_on_q(self):
        started = threading.Event()
        done = threading.Event()

        def slow_compact(force=False):
            started.set()
            while not cli.agent.stop_event.is_set():
                time.sleep(0.02)
            return False

        cli = self._make_monitored_cli(slow_compact)

        def run_cmd():
            cli.cmd_compact_history([])
            done.set()

        with mock.patch("universal_agents.ui.ConsoleUI.system_msg"):
            t = threading.Thread(target=run_cmd, daemon=True)
            t.start()
            self.assertTrue(started.wait(5))
            cli._inject_line("q\n")
            t.join(10)
        self.assertTrue(done.is_set(), "compaction didn't stop after q")
        self.assertTrue(cli.agent.stop_event.is_set())

    def test_cmd_compact_history_stops_on_plain_text_and_requeues(self):
        started = threading.Event()
        done = threading.Event()

        def slow_compact(force=False):
            started.set()
            while not cli.agent.stop_event.is_set():
                time.sleep(0.02)
            return False

        cli = self._make_monitored_cli(slow_compact)

        def run_cmd():
            cli.cmd_compact_history([])
            done.set()

        with mock.patch("universal_agents.ui.ConsoleUI.system_msg"):
            t = threading.Thread(target=run_cmd, daemon=True)
            t.start()
            self.assertTrue(started.wait(5))
            cli._inject_line("продолжай\n")
            t.join(10)
            # Текст пользователя возвращён в очередь (после инжектированного '\n'), чтобы
            # стать обычным сообщением для следующего хода.
            self.assertEqual(list(cli._line_queue), ["\n", "продолжай"])
        self.assertTrue(done.is_set(), "compaction didn't stop on plain text")
        self.assertTrue(cli.agent.stop_event.is_set())


if __name__ == "__main__":
    unittest.main()