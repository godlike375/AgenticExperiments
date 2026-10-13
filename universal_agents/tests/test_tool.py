import unittest

from universal_agents.main import build_allowed_tools
from universal_agents.tool import tool, ENVIRONMENT_PREFIX
from universal_agents.constants import ENVIRONMENT_PREFIX as CONST_PREFIX
from universal_agents.tool_registry import build_tool_dict
from universal_agents.tools.fs import line_range_edit as _line_range_edit_tool
from universal_agents.tools.fs import read as _read_tool
from universal_agents.tools.builtin import have_done as _have_done_tool
from universal_agents.tools.builtin import load_tool as _load_tool_tool
from universal_agents.tools.builtin import make_plan as _make_plan_tool
from universal_agents.tools.host_shell import run_bash_host as _run_bash_host_tool


@tool(description="greet a person", short_description="greet", name=("str", "Name"))
def greet(name: str) -> str:
    return f"Hello, {name}!"


@tool(description="agent-aware")
def with_agent(agent, path: str) -> str:
    return "ok"


class TestToolDecorator(unittest.TestCase):
    def test_schema(self):
        schema = greet._tool_schema
        self.assertEqual(schema["type"], "function")
        self.assertEqual(schema["function"]["name"], "greet")
        self.assertEqual(schema["function"]["description"], "greet a person")
        self.assertEqual(schema["function"]["parameters"]["required"], ["name"])
        self.assertEqual(schema["function"]["parameters"]["properties"]["name"]["type"], "string")

    def test_has_agent_param(self):
        self.assertFalse(greet._has_agent_param)
        self.assertTrue(with_agent._has_agent_param)

    def test_requires_confirmation_default(self):
        self.assertFalse(greet._requires_confirmation)

    def test_environment_prefix(self):
        self.assertEqual(ENVIRONMENT_PREFIX, CONST_PREFIX)

    def test_build_tool_dict(self):
        info = build_tool_dict(greet, is_instance_method=False)
        self.assertEqual(info["schema"], greet._tool_schema)
        self.assertEqual(info["handler"], greet)
        self.assertFalse(info["is_instance_method"])
        self.assertFalse(info["has_agent_param"])
        self.assertFalse(info["requires_confirmation"])


class TestBuildAllowedTools(unittest.TestCase):
    def test_preloaded_added_without_needing_loadable(self):
        allowed = build_allowed_tools(
            loadable=[_run_bash_host_tool.__name__],
            preloaded=[_read_tool.__name__, _line_range_edit_tool.__name__, _load_tool_tool.__name__],
        )
        self.assertEqual(allowed, [_run_bash_host_tool.__name__, _read_tool.__name__, _line_range_edit_tool.__name__, _load_tool_tool.__name__])

    def test_no_duplicates(self):
        allowed = build_allowed_tools(
            loadable=[_read_tool.__name__, _run_bash_host_tool.__name__],
            preloaded=[_read_tool.__name__, _load_tool_tool.__name__],
        )
        self.assertEqual(allowed, [_read_tool.__name__, _run_bash_host_tool.__name__, _load_tool_tool.__name__])

    def test_preloaded_not_required_in_loadable(self):
        allowed = build_allowed_tools(
            loadable=[_run_bash_host_tool.__name__],
            preloaded=[_make_plan_tool.__name__, _have_done_tool.__name__],
        )
        self.assertIn(_make_plan_tool.__name__, allowed)
        self.assertIn(_have_done_tool.__name__, allowed)


if __name__ == "__main__":
    unittest.main()
