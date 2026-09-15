"""Тесты режима порционного чтения больших файлов при отключённой структуризации.

BIG_FILE_SKELETON=False: read() без диапазона должен вернуть первую «пачку» строк
(до лимита по строкам/символам) + периферию, а не пустую заглушку. Скелет-режим
(BIG_FILE_SKELETON=True) при этом работает как раньше."""

import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from universal_agents.config import Config
from universal_agents.tools import fs as fs_module
from universal_agents.tools.fs import read as _read_tool


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

        calls = {"n": 0}
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


if __name__ == "__main__":
    unittest.main()