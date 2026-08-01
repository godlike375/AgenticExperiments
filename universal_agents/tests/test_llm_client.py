import unittest
from types import SimpleNamespace
from unittest import mock

from universal_agents.generation import GenerationParams
from universal_agents.llm_client import LLMClient


class TestResponsesParsing(unittest.TestCase):
    def test_parse_responses_output(self):
        output = [
            SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text", text="Hello ")]),
            SimpleNamespace(type="function_call", call_id="c1", name="read", arguments="{}"),
        ]
        response = SimpleNamespace(output=output, id="resp_1")
        msg = LLMClient._parse_responses_output(response)
        self.assertEqual(msg.content, "Hello ")
        self.assertEqual(msg._response_id, "resp_1")
        self.assertIsNotNone(msg.tool_calls)
        tc = msg.tool_calls[0]
        self.assertEqual(tc.name, "read")
        self.assertEqual(tc.function.name, "read")
        self.assertEqual(tc.arguments, "{}")

    def test_parse_responses_output_string_content(self):
        response = SimpleNamespace(
            output=[SimpleNamespace(type="message", content="plain")],
            id="resp_2",
        )
        msg = LLMClient._parse_responses_output(response)
        self.assertEqual(msg.content, "plain")
        self.assertIsNone(msg.tool_calls)

    def test_extract_responses_usage(self):
        usage = SimpleNamespace(input_tokens=10, output_tokens=5)
        result = LLMClient._extract_responses_usage(SimpleNamespace(usage=usage))
        self.assertEqual(result, {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})

    def test_extract_responses_usage_none(self):
        self.assertIsNone(LLMClient._extract_responses_usage(SimpleNamespace(usage=None)))


class TestStream(unittest.TestCase):
    def test_stream_returns_error_generator(self):
        fake_client = mock.Mock()
        fake_client.chat.completions.create.side_effect = RuntimeError("boom")
        with mock.patch("universal_agents.llm_client.LLMClient.get_client", return_value=fake_client):
            gen = LLMClient.stream([{"role": "user", "content": "hi"}])
        self.assertEqual(next(gen), {"error": "boom"})

    def test_stream_passes_params_and_prefill(self):
        expected = [SimpleNamespace(choices=[], usage=None)]
        fake_client = mock.Mock()
        fake_client.chat.completions.create.return_value = iter(expected)
        with mock.patch("universal_agents.llm_client.LLMClient.get_client", return_value=fake_client):
            gen = LLMClient.stream(
                [{"role": "user", "content": "hi"}],
                prefill="You:",
                params=GenerationParams(temp=0.3, max_tokens=100),
            )
            self.assertEqual(list(gen), expected)
        kwargs = fake_client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["stream"], True)
        self.assertEqual(kwargs["temperature"], 0.3)
        self.assertEqual(kwargs["max_tokens"], 100)
        self.assertEqual(kwargs["messages"][-1], {"role": "assistant", "content": "You:"})


class TestCall(unittest.TestCase):
    def _fake_response(self, content="world"):
        msg = SimpleNamespace(content=content, tool_calls=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=msg)],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        )

    def test_call_chat_completions_prefill_and_usage(self):
        fake_client = mock.Mock()
        fake_client.chat.completions.create.return_value = self._fake_response()
        with mock.patch("universal_agents.llm_client.LLMClient.get_client", return_value=fake_client):
            result, err, usage = LLMClient.call(
                [{"role": "user", "content": "hi"}],
                prefill="start ",
            )
        self.assertIsNone(err)
        self.assertEqual(result.content, "start world")
        self.assertEqual(usage["total_tokens"], 8)
        last_msg = fake_client.chat.completions.create.call_args.kwargs["messages"][-1]
        self.assertEqual(last_msg["role"], "assistant")

    def test_call_resolves_params(self):
        fake_client = mock.Mock()
        fake_client.chat.completions.create.return_value = self._fake_response()
        with mock.patch("universal_agents.llm_client.LLMClient.get_client", return_value=fake_client):
            LLMClient.call([{"role": "user", "content": "hi"}], params=GenerationParams(temp=0.9))
        kwargs = fake_client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["temperature"], 0.9)

    def test_call_returns_error_tuple(self):
        fake_client = mock.Mock()
        fake_client.chat.completions.create.side_effect = RuntimeError("down")
        with mock.patch("universal_agents.llm_client.LLMClient.get_client", return_value=fake_client):
            result, err, usage = LLMClient.call([{"role": "user", "content": "hi"}])
        self.assertIsNone(result)
        self.assertEqual(err, "down")
        self.assertIsNone(usage)


if __name__ == "__main__":
    unittest.main()
