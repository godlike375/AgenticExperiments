import unittest

from universal_agents.agent import LLMAgent
from universal_agents.models import ToolCall
from universal_agents.tool import tool


@tool(description="alpha tool")
def alpha(x: int) -> str:
    return f"alpha:{x}"


@tool(description="beta tool", requires_confirmation=True)
def beta(y: str = "b") -> str:
    return y


def make_parent():
    return LLMAgent(
        system_prompt="sys",
        tools_config=["alpha", "beta"],
        external_plugins={"alpha": alpha, "beta": beta},
    )


class TestSubAgentInheritsSchemas(unittest.TestCase):
    def test_schemas_identical_and_ordered(self):
        parent = make_parent()
        sub = parent.make_sub_agent(denied_tools="*", max_iter=1)
        self.assertEqual(parent.tools, sub._agent.tools)

    def test_denied_star_keeps_schemas_but_blocks_calls(self):
        parent = make_parent()
        sub = parent.make_sub_agent(denied_tools="*", max_iter=1)
        for name in ("alpha", "beta"):
            self.assertTrue(sub._agent.is_tool_denied(name))
        results = sub._agent._execute_tools([ToolCall(id="t1", name="alpha", arguments='{"x": 1}')])
        self.assertTrue(results[0].is_error)
        self.assertIn("forbidden", results[0].content)

    def test_selective_deny_blocks_only_listed(self):
        parent = make_parent()
        sub = parent.make_sub_agent(denied_tools={"alpha"}, max_iter=1)
        self.assertTrue(sub._agent.is_tool_denied("alpha"))
        self.assertFalse(sub._agent.is_tool_denied("beta"))

        denied = sub._agent._execute_tools([ToolCall(id="t1", name="alpha", arguments='{"x": 1}')])[0]
        allowed = sub._agent._execute_tools([ToolCall(id="t2", name="beta", arguments='{"y": "z"}')])[0]
        self.assertTrue(denied.is_error)
        self.assertFalse(allowed.is_error)

    def test_safe_only_denies_confirmation_tools_but_keeps_schemas(self):
        parent = make_parent()
        sub = parent.make_sub_agent(safe_only=True, max_iter=1)
        self.assertEqual(parent.tools, sub._agent.tools)
        self.assertTrue(sub._agent.is_tool_denied("beta"))
        self.assertFalse(sub._agent.is_tool_denied("alpha"))

    def test_no_deny_config_allows_everything(self):
        parent = make_parent()
        sub = parent.make_sub_agent(safe_only=False, max_iter=1)
        self.assertEqual(sub._agent.tools_manager.denied, set())


if __name__ == "__main__":
    unittest.main()
