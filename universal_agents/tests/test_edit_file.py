import os
import shutil
import tempfile
import unittest

from universal_agents.tools.fs import line_range_edit, match_replace_edit, _make_diff_preview


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


if __name__ == "__main__":
    unittest.main()