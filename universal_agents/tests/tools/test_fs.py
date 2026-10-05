import json
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from universal_agents.config import Config
from universal_agents.models import AssistantMessage, ToolCall
from universal_agents.tools import fs as fs_module
from universal_agents.tools.builtin import respond_to_system
from universal_agents.tools.fs import line_range_edit, match_replace_edit, _make_diff_preview
from universal_agents.tools.fs import read as _read_tool

from tests.conftest import make_agent as make_test_agent


class TestMakeDiffPreview(unittest.TestCase):
    def test_replacement_has_context_marks(self):
        old = "head1\nctx_a\nold_x\nold_y\nctx_b\ntail1\n"
        new = "head1\nctx_a\nnew_x\nctx_b\ntail1\n"
        out = _make_diff_preview(old, new, "f.py", replaced=2, added=1)
        self.assertIn("-old_x", out)
        self.assertIn("-old_y", out)
        self.assertIn("+new_x", out)
        self.assertIn(" ctx_a", out)
        self.assertIn(" ctx_b", out)
        self.assertNotIn("-ctx_a", out)
        self.assertNotIn("-head1", out)

    def test_multiple_hunks_not_collapsed(self):
        old = "h1\nh2\nold_a\nctx\nold_b\nh5\nh6\n"
        new = "h1\nh2\nnew_a\nctx\nnew_b\nh5\nh6\n"
        out = _make_diff_preview(old, new, "f.py")
        self.assertEqual(out.count("-old_a"), 1)
        self.assertEqual(out.count("+new_a"), 1)
        self.assertEqual(out.count("-old_b"), 1)
        self.assertEqual(out.count("+new_b"), 1)

    def test_no_context_at_file_edges(self):
        old = "10\n20\n30\n"
        new = "10\n20\n30\n40\n"
        out = _make_diff_preview(old, new, "f.py")
        self.assertIn("+40", out)
        self.assertIn(" 30", out)
        self.assertNotIn("-10", out)


class TestEditFile(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)

    def _path(self, name, content=None):
        p = os.path.join(self._tmp, name)
        if content is not None:
            with open(p, "w", encoding="utf-8") as f:
                f.write(content)
        return p

    def test_replace_mid_file_dry_run_preview_and_confirm(self):
        f = self._path("a.py", "import os\n\n\ndef foo(x):\n    a = 1\n    b = 2\n    return a\n")
        preview, resolve, ask = line_range_edit(
            f, "def foo(x):\n    b = 99\n    return b\n", start_line=4, end_line=6, dry_run="true"
        )
        self.assertIn("--- ", preview)
        self.assertIn("-    a = 1", preview)
        self.assertIn("+    b = 99", preview)
        self.assertIn(" def foo(x):", preview)
        self.assertIn("     return a", preview)
        self.assertIn("-3+3", preview)
        self.assertIn("answer", ask)
        self.assertFalse(resolve(None, "no"))
        self.assertEqual(resolve(None, "always_other"), None)
        self.assertIn("Replaced L4-6", resolve(None, "yes"))
        after = open(f, encoding="utf-8").read()
        self.assertIn("    b = 99", after)
        self.assertIn("    return a", after)
        self.assertNotIn("    a = 1", after)

    def test_create_new_file(self):
        f = self._path("new.txt")
        preview, resolve, _ask = line_range_edit(f, "hello\nworld\n", start_line=1, dry_run="true")
        self.assertIn("+hello", preview)
        self.assertIn("+world", preview)
        resolve(None, "yes")
        self.assertEqual(open(f, encoding="utf-8").read(), "hello\nworld")

    def test_nothing_changed(self):
        f = self._path("same.txt", "abc\n")
        out = line_range_edit(f, "abc\n", start_line=1, end_line=1, dry_run="true")
        self.assertIsInstance(out, str)
        self.assertIn("Nothing changed", out)

    def test_insert_middle(self):
        f = self._path("mid.txt", "l1\nl2\nl3\nl4\n")
        preview, _resolve, _ask = line_range_edit(f, "X\n", start_line=2, end_line=2, dry_run="true")
        self.assertIn("-l2", preview)
        self.assertIn("+X", preview)
        self.assertIn(" l1", preview)
        self.assertIn(" l3", preview)

    def test_append_at_end(self):
        f = self._path("end.txt", "a\nb\nc\n")
        preview, _resolve, _ask = line_range_edit(f, "d\n", start_line=4, end_line=4, dry_run="true")
        self.assertIn("+d", preview)
        self.assertIn(" c", preview)

    def test_prepend_at_start(self):
        f = self._path("start.txt", "a\nb\nc\n")
        preview, _resolve, _ask = line_range_edit(f, "0\n", start_line=1, end_line=1, dry_run="true")
        self.assertIn("-a", preview)
        self.assertIn("+0", preview)

    def test_delete_range(self):
        f = self._path("del.txt", "a\nb\nc\nd\n")
        preview, _resolve, _ask = line_range_edit(f, "", start_line=2, end_line=3, dry_run="true")
        self.assertIn("-b", preview)
        self.assertIn("-c", preview)
        self.assertIn(" a", preview)
        self.assertIn(" d", preview)

    def test_explicit_context_params(self):
        out = _make_diff_preview(
            "old_a\nold_b\n", "new_a\nnew_b\n", "f.py",
            replaced=2, added=2, context_before="ctx_before", context_after="ctx_after",
        )
        self.assertIn(" ctx_before", out)
        self.assertIn(" ctx_after", out)


class TestMatchReplaceEdit(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)

    def _path(self, name, content=None):
        p = os.path.join(self._tmp, name)
        if content is not None:
            with open(p, "w", encoding="utf-8") as f:
                f.write(content)
        return p

    def test_single_match_replace_flow(self):
        f = self._path("m.py", "a = 1\n\nb = 2\nc = 3\n")
        preview, resolve, _ask = match_replace_edit(f, "b = 2", "b = 99", dry_run="true")
        self.assertIn("-b = 2", preview)
        self.assertIn("+b = 99", preview)
        self.assertEqual(resolve(None, "yes"), "Replaced 1 occurrence at L3 in m.py")
        self.assertEqual(
            open(f, encoding="utf-8").read(),
            "a = 1\n\nb = 99\nc = 3\n",
        )

    def test_multiline_match(self):
        f = self._path("ml.py", "l1\nl2\nl3\nl4\nl5\n")
        preview, resolve, _ask = match_replace_edit(f, "l2\nl3", "X\nY", dry_run="true")
        self.assertIn("-l2", preview)
        self.assertIn("-l3", preview)
        self.assertIn("+X", preview)
        self.assertIn("+Y", preview)
        resolve(None, "yes")
        self.assertEqual(open(f, encoding="utf-8").read(), "l1\nX\nY\nl4\nl5\n")

    def test_one_mode_multiple_matches_error(self):
        f = self._path("mm.py", "x\nx\n")
        out = match_replace_edit(f, "x", "y", mode="one", dry_run="true")
        self.assertIn("Error", out)
        self.assertIn("2 matches", out)

    def test_all_mode_replaces_every_occurrence(self):
        f = self._path("all.py", "x\nkeep\nx\n")
        preview, resolve, _ask = match_replace_edit(f, "x", "y", mode="all", dry_run="true")
        self.assertIn("-x", preview)
        resolve(None, "yes")
        self.assertEqual(open(f, encoding="utf-8").read(), "y\nkeep\ny\n")

    def test_all_mode_preview_isolates_changed_lines(self):
        """Превью mode='all' не должно выглядеть как удаление всего файла."""
        lines = ["line %d" % i for i in range(1, 30)]
        lines[9] = "// old comment"
        f = self._path("iso.py", "\n".join(lines) + "\n")
        preview, resolve, _ask = match_replace_edit(f, "// old comment", "// new comment", mode="all", dry_run="true")
        self.assertIn("-1+1", preview)
        self.assertIn("-// old comment", preview)
        self.assertIn("+// new comment", preview)
        self.assertNotIn("-line 1", preview)
        self.assertNotIn("-line 29", preview)
        resolve(None, "yes")
        content = open(f, encoding="utf-8").read()
        self.assertIn("// new comment", content)
        self.assertIn("line 1", content)
        self.assertIn("line 29", content)

    def test_all_mode_multiple_occurrences_scope_header(self):
        """Заголовок mode='all' — объём вхождений; нетронутые строки не удаляются в превью."""
        f = self._path("iso2.py", "x\nkeep_a\nx\nmiddle\nx\nkeep_b\n")
        preview, _resolve, _ask = match_replace_edit(f, "x", "y", mode="all", dry_run="true")
        self.assertIn("-3+3", preview)
        self.assertNotIn("-keep_a", preview)
        self.assertNotIn("-middle", preview)
        self.assertNotIn("-keep_b", preview)
        self.assertEqual(preview.count("-x"), 3)
        self.assertEqual(preview.count("+y"), 3)

    def test_no_match_error(self):
        f = self._path("nomatch.py", "abc\n")
        out = match_replace_edit(f, "zzz", "yyy", dry_run="true")
        self.assertIn("Error", out)
        self.assertIn("not found", out)

    def test_missing_file_with_old_error(self):
        f = os.path.join(self._tmp, "missing.py")
        out = match_replace_edit(f, "x", "y", dry_run="true")
        self.assertIn("Error", out)

    def test_empty_old_creates_new_file(self):
        f = os.path.join(self._tmp, "new_whole.py")
        preview, resolve, _ask = match_replace_edit(f, "", "hello\nworld\n", dry_run="true")
        self.assertIn("+hello", preview)
        resolve(None, "yes")
        self.assertEqual(open(f, encoding="utf-8").read(), "hello\nworld")

    def test_empty_old_overwrites_whole_file(self):
        f = self._path("ow.py", "old content\n")
        preview, resolve, _ask = match_replace_edit(f, "", "brand new\n", dry_run="true")
        self.assertIn("+brand new", preview)
        resolve(None, "yes")
        self.assertEqual(open(f, encoding="utf-8").read(), "brand new")


def _make_fake_agent():
    """Лёгкий фейк агента: read обращается только к file_states/_read_registrations."""
    agent = SimpleNamespace(
        file_states=SimpleNamespace(
            _history=[],
            should_skip=lambda path, disk_hash: False,
            record=lambda path, disk_hash, content_hash: None,
        ),
        _read_registrations=[],
        on_system_msg=lambda *a, **k: None,
    )
    return agent


class TestReadBatchMode(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)
        self._old_skeleton = Config.BIG_FILE_SKELETON
        self.addCleanup(setattr, Config, "BIG_FILE_SKELETON", self._old_skeleton)
        self._old_lines = Config.MAX_READ_LINES_PER_CALL
        self._old_chars = Config.MAX_READ_CHARS_PER_CALL
        self.addCleanup(setattr, Config, "MAX_READ_LINES_PER_CALL", self._old_lines)
        self.addCleanup(setattr, Config, "MAX_READ_CHARS_PER_CALL", self._old_chars)

    def _write(self, name, total_lines, line_len=40):
        p = os.path.join(self._tmp, name)
        with open(p, "w", encoding="utf-8") as f:
            for i in range(1, total_lines + 1):
                f.write(f"line {i} " + "x" * line_len + "\n")
        return p

    def test_no_range_returns_first_batch_plus_periphery(self):
        Config.BIG_FILE_SKELETON = False
        path = self._write("big.py", total_lines=120)
        agent = _make_fake_agent()
        out = _read_tool(agent, path)

        self.assertIn("Full focus lines 1-80/120", out)
        self.assertIn("\n1 line 1 ", out)
        self.assertIn("\n2 line 2 ", out)
        self.assertIn("\n80 line 80 ", out)
        # Периферийные строки помечены '~' и есть за пределами фокуса.
        self.assertIn("\n~", out)
        peri_idx = [int(l.split(" ")[0][1:]) for l in out.splitlines() if l.startswith("~")]
        self.assertTrue(all(i > 80 for i in peri_idx))
        # Подсказка продолжать диапазоном — только когда файл реально обрезан.
        self.assertIn(f"Use start_line=81", out)
        # Чтение регистрируется как «в контексте».
        self.assertEqual(agent._read_registrations, [path])

    def test_batch_respects_char_limit(self):
        Config.BIG_FILE_SKELETON = False
        path = self._write("chars.py", total_lines=120, line_len=200)
        agent = _make_fake_agent()
        out = _read_tool(agent, path)

        self.assertIn("Full focus lines 1-", out)
        # Только фокус-строки (без '~' и служебных) имеют счётные номера ≤ 80.
        nums = {int(l.split(" ")[0]) for l in out.splitlines() if l[:1].isdigit()}
        self.assertTrue(all(n <= Config.MAX_READ_LINES_PER_CALL for n in nums))

    def test_reread_same_hash_blocked(self):
        Config.BIG_FILE_SKELETON = False
        path = self._write("once.py", total_lines=120)

        seen = {}
        agent = SimpleNamespace(
            file_states=SimpleNamespace(
                _seen=seen,
                should_skip=lambda path, disk_hash: path in seen,
                record=lambda path, disk_hash, content_hash: seen.update({path: disk_hash}),
            ),
            _read_registrations=[],
            on_system_msg=lambda *a, **k: None,
        )
        first = _read_tool(agent, path)
        self.assertIn("Full focus lines 1-80/120", first)
        second = _read_tool(agent, path)
        self.assertNotIn("Full focus lines", second)
        self.assertIn("re-reading", second)

    def test_range_read_unaffected(self):
        # Порционное чтение диапазоном не зависит от флага структуризации.
        Config.BIG_FILE_SKELETON = False
        path = self._write("range.py", total_lines=120)
        agent = _make_fake_agent()
        out = _read_tool(agent, path, start_line=20, end_line=25)

        self.assertIn("Full focus lines 20-25/120", out)
        self.assertIn("\n20 line 20 ", out)
        self.assertIn("\n25 line 25 ", out)
        self.assertIn("\n~", out)
        self.assertNotIn("Use start_line=", out)  # диапазон в пределах лимита

    def test_skeleton_mode_unchanged(self):
        # Скелет-режим по-прежнему использует структуру, а не пачку строк.
        Config.BIG_FILE_SKELETON = True
        path = self._write("skeleton.py", total_lines=120)
        agent = _make_fake_agent()
        with mock.patch.object(fs_module, "_summarize_file", return_value="L1-3 class Foo"):
            out = _read_tool(agent, path)
        self.assertIn("Total lines: 120", out)
        self.assertIn("Content structure", out)
        self.assertIn("L1-3 class Foo", out)
        self.assertNotIn("Full focus lines", out)
        self.assertNotIn("\n1 line 1 ", out)


class _PendingAgent:
    """Минимальный агент для respond_to_system: отдаёт resolve ожидающей операции."""
    def __init__(self, resolve):
        self._resolve = resolve
    def pop_pending_operation(self):
        return {"resolve": self._resolve}


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

        agent = _PendingAgent(resolve)
        result = respond_to_system(agent, "yes")
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

        agent = _PendingAgent(resolve)
        result = respond_to_system(agent, "no")
        self.assertEqual(open(f, encoding="utf-8").read(), "a\nb\nc\n")

    def test_full_flow_answer_with_text_comment(self):
        f = self._path("flow_comment.py", "l1\nl2\n")
        preview, resolve, _ask = line_range_edit(f, "NEW\n", start_line=2, end_line=2, dry_run="true")

        agent = _PendingAgent(resolve)
        result = respond_to_system(agent, "yes, please proceed")
        self.assertIn("Replaced L2-2", result)
        self.assertIn("NEW", open(f, encoding="utf-8").read())

    def test_full_flow_create_new_file_via_edit(self):
        f = os.path.join(self._tmp, "brand_new.py")
        preview, resolve, _ask = line_range_edit(f, "hello\n", start_line=1, dry_run="true")
        self.assertIn("+hello", preview)

        agent = _PendingAgent(resolve)
        result = respond_to_system(agent, "yes")
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