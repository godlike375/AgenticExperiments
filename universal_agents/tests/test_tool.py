import unittest

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


if __name__ == "__main__":
    unittest.main()
