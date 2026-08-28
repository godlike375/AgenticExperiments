import unittest

from universal_agents.main import build_allowed_tools


class TestBuildAllowedTools(unittest.TestCase):
    def test_preloaded_added_without_needing_loadable(self):
        allowed = build_allowed_tools(
            loadable=["run_bash_host"],
            preloaded=["read", "edit_file", "load_tool"],
        )
        self.assertEqual(allowed, ["run_bash_host", "read", "edit_file", "load_tool"])

    def test_no_duplicates(self):
        allowed = build_allowed_tools(
            loadable=["read", "run_bash_host"],
            preloaded=["read", "load_tool"],
        )
        self.assertEqual(allowed, ["read", "run_bash_host", "load_tool"])

    def test_preloaded_not_required_in_loadable(self):
        # Ключевой сценарий: предзагруженный инструмент отсутствует в LOADABLE_TOOLS,
        # но всё равно попадает в allow-список (иначе ToolManager его отфильтрует).
        allowed = build_allowed_tools(
            loadable=["run_bash_host"],
            preloaded=["make_plan", "have_done"],
        )
        self.assertIn("make_plan", allowed)
        self.assertIn("have_done", allowed)
        self.assertNotIn("make_plan", ["run_bash_host"])


if __name__ == "__main__":
    unittest.main()
