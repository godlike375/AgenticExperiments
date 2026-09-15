import os
import shutil
import tempfile
import unittest

from universal_agents.tools.fs import line_range_edit
from universal_agents.tools.builtin import answer_to_system


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


if __name__ == "__main__":
    unittest.main()