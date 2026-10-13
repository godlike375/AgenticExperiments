
import unittest
from unittest import mock

from universal_agents.agent import LLMAgent
from universal_agents.models import AssistantMessage, ToolCall

from tests.conftest import double_me


class TestThinkingConsistency(unittest.TestCase):
    def test_thinking_once_consistent_across_whole_turn(self):
        agent = LLMAgent(
            system_prompt="sys",
            tools_config=["double_me"],
            external_plugins={"double_me": double_me},
            autosave_enabled=False,
        )
        bare_call = AssistantMessage(
            content="",
            tool_calls=[ToolCall(id="t1", name="double_me", arguments='{"value": 21}')],
        )
        final_reply = AssistantMessage(content="Готово.")

        seen = []
        responses = [(bare_call, None, None), (final_reply, None, None)]

        def fake_call(messages, reasoning_effort=None, **kwargs):
            seen.append(reasoning_effort)
            return responses.pop(0)

        with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake_call):
            agent._thinking_once = True
            result = agent.chat("compute")

        self.assertEqual(result, "Готово.")
        self.assertTrue(seen, "должен быть хотя бы один API-вызов")
        self.assertTrue(all(e == "low" for e in seen), seen)
        roles = [m.to_api_dict()["role"] for m in agent.history]
        self.assertIn("tool", roles)
        tr = [m for m in agent.history.get_all() if m.to_api_dict()["role"] == "tool"][-1]
        self.assertEqual(tr.content, "42")


if __name__ == "__main__":
    unittest.main()
