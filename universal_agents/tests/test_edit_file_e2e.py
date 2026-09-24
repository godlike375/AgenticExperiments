import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from universal_agents.models import AssistantMessage, ToolCall
from universal_agents.tools.fs import line_range_edit, match_replace_edit
from universal_agents.tools.builtin import answer_to_system

from tests.conftest import make_agent as make_test_agent


class TestEditFileEndToEnd(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)

    def _path(self, name, content=None):
        p = os.path.join(self._tmp, name)
        if content is not None:
            with open(p, "w", encoding="utf-8") as f:
                f.write(content)
        return p

    def test_full_flow_dry_run_then_answer_yes(self):
        f = self._path("flow.py", "import os\n\ndef foo():\n    return 1\n")
        preview, resolve, _ask = line_range_edit(
            f, "import sys\n\ndef bar():\n    return 2\n", start_line=1, end_line=4, dry_run="true"
        )
        self.assertIn("-import os", preview)
        self.assertIn("+import sys", preview)

        class FakeAgent:
            def pop_pending_operation(self):
                return {"resolve": resolve}

        agent = FakeAgent()
        result = answer_to_system(agent, "yes")
        self.assertIn("Replaced L1-4", result)
        self.assertEqual(
            open(f, encoding="utf-8").read().rstrip("\n"),
            "import sys\n\ndef bar():\n    return 2",
        )

    def test_full_flow_answer_no_keeps_file(self):
        f = self._path("flow_no.py", "a\nb\nc\n")
        preview, resolve, _ask = line_range_edit(f, "X\n", start_line=2, end_line=2, dry_run="true")
        self.assertIn("-b", preview)
        self.assertIn("+X", preview)

        class FakeAgent:
            def pop_pending_operation(self):
                return {"resolve": resolve}

        agent = FakeAgent()
        result = answer_to_system(agent, "no")
        self.assertEqual(open(f, encoding="utf-8").read(), "a\nb\nc\n")

    def test_full_flow_answer_with_text_comment(self):
        f = self._path("flow_comment.py", "l1\nl2\n")
        preview, resolve, _ask = line_range_edit(f, "NEW\n", start_line=2, end_line=2, dry_run="true")

        class FakeAgent:
            def pop_pending_operation(self):
                return {"resolve": resolve}

        agent = FakeAgent()
        result = answer_to_system(agent, "yes, please proceed")
        self.assertIn("Replaced L2-2", result)
        self.assertIn("NEW", open(f, encoding="utf-8").read())

    def test_full_flow_create_new_file_via_edit(self):
        f = os.path.join(self._tmp, "brand_new.py")
        preview, resolve, _ask = line_range_edit(f, "hello\n", start_line=1, dry_run="true")
        self.assertIn("+hello", preview)

        class FakeAgent:
            def pop_pending_operation(self):
                return {"resolve": resolve}

        agent = FakeAgent()
        result = answer_to_system(agent, "yes")
        self.assertIn("Replaced", result)
        self.assertEqual(open(f, encoding="utf-8").read(), "hello")

    # --- Строковый результат dry_run — обычный ответ, а не конфиг-ошибка. ---

    def test_dry_run_error_string_is_not_config_error(self):
        """Ненайденная подстрока в dry_run: модель видит 'substring not found', а не конфиг-ошибку."""
        path = os.path.join(self._tmp, "nomatch.xml")
        with open(path, "w", encoding="utf-8") as f:
            f.write("<root>\n    <x/>\n</root>\n")

        responses_holder = []

        def _run():
            call = AssistantMessage(
                content="Поменяю тег.",
                tool_calls=[ToolCall(id="c1", name=match_replace_edit.__name__, arguments=json.dumps({
                    "path": path, "old": "<nope>", "new_text": "<r>", "mode": "one"}))],
            )
            final = AssistantMessage(content="Узор не найден, менять нечего.")
            responses_holder[:] = [(call, None, None), (final, None, None)]

            agent = make_test_agent(
                system_prompt="You edit files. Answer only with final text.",
                external_plugins={match_replace_edit.__name__: match_replace_edit},
            )
            agent.trust_dir(self._tmp)
            seen = []
            def fake_call(messages, prefill=None, **kwargs):
                seen.append([m.get("content", "") for m in messages if m.get("role") == "tool"])
                return responses_holder.pop(0)

            with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake_call):
                result = agent.chat("Измени файл", max_iter=10)
            return agent, result, seen

        agent, result, seen = _run()
        self.assertIn("Узор не найден", result)
        self.assertIsNone(agent._pending_operation)
        flat = [c for turn in seen for c in turn]
        self.assertTrue(any("substring not found" in c for c in flat))
        self.assertFalse(any("Tool configuration error" in c for c in flat))

    def test_dry_run_nothing_changed_is_not_config_error(self):
        """No-op правка в dry_run: модель видит 'Nothing changed', а не конфиг-ошибку."""
        path = os.path.join(self._tmp, "same.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("a = 1\nb = 2\n")

        responses_holder = []

        def _run():
            call = AssistantMessage(
                content="Отредактирую строку.",
                tool_calls=[ToolCall(id="c1", name=line_range_edit.__name__, arguments=json.dumps({
                    "path": path, "new_text": "b = 2\n", "start_line": 2, "end_line": 2}))],
            )
            final = AssistantMessage(content="Файл уже имеет нужное содержимое.")
            responses_holder[:] = [(call, None, None), (final, None, None)]

            agent = make_test_agent(
                system_prompt="You edit files. Answer only with final text.",
                external_plugins={line_range_edit.__name__: line_range_edit},
            )
            agent.trust_dir(self._tmp)
            seen = []
            def fake_call(messages, prefill=None, **kwargs):
                seen.append([m.get("content", "") for m in messages if m.get("role") == "tool"])
                return responses_holder.pop(0)

            with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake_call):
                result = agent.chat("Измени файл", max_iter=10)
            return agent, result, seen

        agent, result, seen = _run()
        self.assertIn("Файл уже имеет", result)
        self.assertIsNone(agent._pending_operation)
        flat = [c for turn in seen for c in turn]
        self.assertTrue(any("Nothing changed" in c for c in flat))
        self.assertFalse(any("Tool configuration error" in c for c in flat))


if __name__ == "__main__":
    unittest.main()
