import unittest

from universal_agents.main import build_allowed_tools
from universal_agents.tool import tool, ENVIRONMENT_PREFIX
from universal_agents.constants import ENVIRONMENT_PREFIX as CONST_PREFIX
from universal_agents.tool_registry import build_tool_dict


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


if __name__ == "__main__":
    unittest.main()
