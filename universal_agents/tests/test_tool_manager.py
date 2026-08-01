import os
import tempfile
import unittest
from unittest import mock

from universal_agents.tool import tool
from universal_agents.tool_manager import ToolManager


@tool(description="alpha tool")
def alpha(x: int) -> str:
    return str(x)


@tool(description="beta tool")
def beta(y: str = "b") -> str:
    return y


@tool(description="unload a tool")
def unload_tool(agent, name: str) -> str:
    return "ok"


class TestToolManagerFilter(unittest.TestCase):
    def test_none_config_loads_all(self):
        tm = ToolManager(None, external_plugins={"alpha": alpha, "beta": beta})
        self.assertEqual(set(tm.tools_map.keys()), {"alpha", "beta"})

    def test_list_config_filters(self):
        tm = ToolManager(["alpha"], external_plugins={"alpha": alpha, "beta": beta})
        self.assertEqual(set(tm.tools_map.keys()), {"alpha"})

    def test_exclude_config_filters(self):
        tm = ToolManager({"exclude": ["beta"]}, external_plugins={"alpha": alpha, "beta": beta})
        self.assertEqual(set(tm.tools_map.keys()), {"alpha"})

    def test_invalid_config_raises(self):
        with self.assertRaises(ValueError):
            ToolManager({"unknown": "x"}, external_plugins={"alpha": alpha})

    def test_schemas(self):
        tm = ToolManager(None, external_plugins={"alpha": alpha})
        self.assertEqual(len(tm.schemas), 1)
        self.assertEqual(tm.schemas[0]["function"]["name"], "alpha")


class TestToolManagerAllowed(unittest.TestCase):
    def test_allowed(self):
        tm = ToolManager(["alpha"], external_plugins={"alpha": alpha, "beta": beta})
        self.assertTrue(tm.is_tool_allowed("alpha"))
        self.assertFalse(tm.is_tool_allowed("beta"))

    def test_allowed_none_config(self):
        tm = ToolManager(None, external_plugins={"alpha": alpha})
        self.assertTrue(tm.is_tool_allowed("anything"))


class TestToolManagerLoadUnload(unittest.TestCase):
    def test_unload_core_tool_protected(self):
        tm = ToolManager(None, external_plugins={"alpha": alpha, "unload_tool": unload_tool})
        self.assertIn("unload_tool", tm.tools_map)
        res = tm.unload("unload_tool")
        self.assertIn("Cannot disable", res)
        self.assertIn("unload_tool", tm.tools_map)

    def test_unload_non_core_removes_unload_tool(self):
        tm = ToolManager(
            ["alpha", "unload_tool"],
            external_plugins={"alpha": alpha, "beta": beta, "unload_tool": unload_tool},
        )
        self.assertIn("unload_tool", tm.tools_map)
        res = tm.unload("alpha")
        self.assertIn("disabled", res)
        self.assertNotIn("alpha", tm.tools_map)
        self.assertNotIn("unload_tool", tm.tools_map)

    def test_unload_unknown(self):
        tm = ToolManager(["alpha"], external_plugins={"alpha": alpha})
        self.assertIn("not loaded", tm.unload("beta"))

    def test_load_disallowed(self):
        tm = ToolManager(["alpha"], external_plugins={"alpha": alpha, "beta": beta})
        res = tm.load("beta")
        self.assertIn("not allowed", res)
        self.assertNotIn("beta", tm.tools_map)

    def test_load_unknown(self):
        tm = ToolManager(None, external_plugins={"alpha": alpha})
        with mock.patch("universal_agents.tool_manager.load_external_plugins", return_value={}):
            res = tm.load("nope")
        self.assertIn("not found", res)

    def test_load_success_adds_unload_tool(self):
        tm = ToolManager(None, external_plugins={"alpha": alpha})
        with mock.patch(
            "universal_agents.tool_manager.load_external_plugins",
            return_value={"alpha": alpha, "beta": beta, "unload_tool": unload_tool},
        ):
            res = tm.load("beta")
        self.assertIn("loaded", res)
        self.assertIn("beta", tm.tools_map)
        self.assertIn("unload_tool", tm.tools_map)


class TestToolManagerTrust(unittest.TestCase):
    def test_trust_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            tm = ToolManager()
            self.assertIn("not a directory", tm.trust_dir(os.path.join(d, "missing")))
            self.assertFalse(tm.is_path_trusted(os.path.join(d, "missing")))

            sub = os.path.join(d, "sub")
            os.makedirs(sub)
            self.assertIn("Trusted", tm.trust_dir(sub))
            self.assertTrue(tm.is_path_trusted(os.path.join(sub, "file.txt")))
            self.assertTrue(tm.is_path_trusted(sub))
            self.assertFalse(tm.is_path_trusted(d))

            self.assertIn("Untrusted", tm.untrust_dir(sub))
            self.assertFalse(tm.is_path_trusted(sub))
            self.assertIn("was not trusted", tm.untrust_dir(sub))


if __name__ == "__main__":
    unittest.main()
