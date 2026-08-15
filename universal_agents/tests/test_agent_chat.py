import unittest
from unittest import mock
from types import SimpleNamespace

from universal_agents.agent import LLMAgent
from universal_agents.models import AssistantMessage, ToolCall
from universal_agents.tool import tool


@tool(description="double a value")
def double_me(agent, value: int) -> str:
    return str(value * 2)


def _chunk(delta=None, usage=None, choices=None):
    if choices is None:
        choices = [SimpleNamespace(delta=delta)] if delta is not None else []
    return SimpleNamespace(choices=choices, usage=usage)


def _delta(content=None, tool_calls=None, reasoning_content=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls, reasoning_content=reasoning_content)


def _tc_delta(index, id=None, name=None, arguments=None):
    return SimpleNamespace(
        index=index,
        id=id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class TestAgentChat(unittest.TestCase):
    def test_chat_returns_plain_answer(self):
        agent = LLMAgent(system_prompt="sys")
        fake = AssistantMessage(content="hello back")
        with mock.patch("universal_agents.agent.LLMClient.call", return_value=(fake, None, None)):
            result = agent.chat("hello")
        self.assertIn("hello back", result)
        roles = [m.to_api_dict()["role"] for m in agent.history]
        self.assertEqual(roles, ["system", "user", "assistant"])

    def test_chat_executes_tool_and_finishes(self):
        agent = LLMAgent(
            system_prompt="sys",
            tools_config=["double_me"],
            external_plugins={"double_me": double_me},
        )
        first = AssistantMessage(content="", tool_calls=[ToolCall(id="t1", name="double_me", arguments='{"value": 21}')])
        second = AssistantMessage(content="final answer 42")
        with mock.patch(
            "universal_agents.agent.LLMClient.call",
            side_effect=[(first, None, None), (second, None, None)],
        ):
            result = agent.chat("compute", max_iter=5)
        self.assertIn("final answer 42", result)
        roles = [m.to_api_dict()["role"] for m in agent.history]
        self.assertIn("tool", roles)
        self.assertIn("assistant", roles)

    def test_chat_streaming_executes_tool(self):
        agent = LLMAgent(
            system_prompt="sys",
            tools_config=["double_me"],
            external_plugins={"double_me": double_me},
            streaming_enabled=True,
            on_stream_chunk=lambda _: None,
        )

        def stream1(*args, **kwargs):
            yield _chunk(_delta(tool_calls=[_tc_delta(0, id="t1", name="double_me", arguments='{"value": ')],
                                content=""))
            yield _chunk(_delta(tool_calls=[_tc_delta(0, arguments='21}')]))
            yield _chunk(_delta(tool_calls=[_tc_delta(0, id="t1")]))
            yield _chunk(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15))

        def stream2(*args, **kwargs):
            yield _chunk(_delta(content="final "))
            yield _chunk(_delta(content="answer 42"))
            yield _chunk(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15))

        with mock.patch(
            "universal_agents.agent.LLMClient.stream",
            side_effect=[stream1(), stream2()],
        ):
            result = agent.chat("compute", max_iter=5)
        self.assertIn("final answer 42", result)
        roles = [m.to_api_dict()["role"] for m in agent.history]
        self.assertIn("tool", roles)


    def test_chat_streaming_applies_prefill_and_emits_it(self):
        seen = []

        def on_chunk(chunk):
            seen.append(chunk)

        agent = LLMAgent(
            system_prompt="sys",
            streaming_enabled=True,
            on_stream_chunk=on_chunk,
        )

        def stream(*args, **kwargs):
            yield _chunk(_delta(content="hello back"))

        with mock.patch("universal_agents.agent.LLMClient.stream", return_value=stream()):
            result = agent.chat("hello", prefill="<start>")

        self.assertTrue(result.startswith("<start>"), f"result should start with prefill: {result!r}")
        self.assertEqual(seen[0], "<start>", "prefill should be the first streamed chunk")
        self.assertEqual("".join(seen), "<start>hello back", f"unexpected stream chunks: {seen!r}")

    def test_chat_streaming_emits_prefill_after_reasoning(self):
        seen = []

        def on_chunk(chunk):
            seen.append(chunk)

        agent = LLMAgent(
            system_prompt="sys",
            streaming_enabled=True,
            on_stream_chunk=on_chunk,
        )

        def stream(*args, **kwargs):
            yield _chunk(_delta(reasoning_content="think..."))
            yield _chunk(_delta(content="hello back"))

        with mock.patch("universal_agents.agent.LLMClient.stream", return_value=stream()):
            result = agent.chat("hello", prefill="<start>")

        self.assertTrue(result.startswith("<start>"))
        self.assertEqual(seen, ["<start>", "hello back"], f"unexpected stream chunks: {seen!r}")

    def test_chat_streaming_prefill_with_empty_content(self):
        agent = LLMAgent(
            system_prompt="sys",
            streaming_enabled=True,
            on_stream_chunk=lambda _: None,
        )

        def stream(*args, **kwargs):
            yield _chunk(_delta(content=""))
            yield _chunk(usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2))

        with mock.patch("universal_agents.agent.LLMClient.stream", return_value=stream()):
            result = agent.chat("hello", prefill="X")

        self.assertEqual(result, "X")


if __name__ == "__main__":
    unittest.main()
