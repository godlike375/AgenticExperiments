import unittest
from types import SimpleNamespace
from unittest import mock

from universal_agents.agent import LLMAgent
from universal_agents.agent_mixins.streaming_mixin import StreamingMixin
from universal_agents.exceptions import GenerationInterrupted

from tests.conftest import double_me, stream_chunk, stream_delta


class TestWatchDiverged(unittest.TestCase):
    """§Streaming: `_watch_diverged` отличает продолжение прежнего ответа от расходящегося."""

    def test_matches_prefix(self):
        self.assertFalse(StreamingMixin._watch_diverged("The answer is 4", "", "The answer is"))

    def test_detects_divergence(self):
        self.assertTrue(StreamingMixin._watch_diverged("The answer is 4", "", "The answer is 5"))

    def test_empty_watch_prefix_never_diverges(self):
        self.assertFalse(StreamingMixin._watch_diverged("", "", "anything"))

    def test_prefill_counts_toward_prefix(self):
        self.assertFalse(StreamingMixin._watch_diverged("The answer is 4", "The answer is", " 4"))


class TestStreamInterrupt(unittest.TestCase):
    """§Streaming: прерывание стрима через stop_check и проброс GenerationInterrupted."""

    def make_streaming_agent(self):
        return LLMAgent(
            system_prompt="sys",
            streaming_enabled=True,
            on_stream_chunk=lambda _: None,
        )

    def test_stop_check_shortcuts_stream_returns_partial(self):
        agent = self.make_streaming_agent()
        seen = []
        agent.on_stream_chunk = seen.append

        def stream(*args, **kwargs):
            yield stream_chunk(stream_delta(content="hello"))
            yield stream_chunk(stream_delta(content=" world"))
            yield stream_chunk(stream_delta(content=" never seen"))

        def stop_check():
            # останавливаем после накопления "hello world" — третий чанк не должен
            # попасть в ответ. Критерий по содержимому, а не по числу вызовов: watcher
            # (закрывающий стрим) и основной цикл зовут stop_check на разных фазах.
            return "hello world" in "".join(seen)

        with mock.patch("universal_agents.agent.LLMClient.stream", side_effect=stream), \
             mock.patch("universal_agents.llm_client.LLMClient.close_stream"):
            msg, err, usage, _stopped = agent._call_with_streaming([], stop_check=stop_check)

        self.assertIsNone(err)
        self.assertEqual(msg.content, "hello world")
        self.assertNotIn("never seen", msg.content)
        self.assertTrue(msg.streamed)

    def test_exception_with_stop_raises_generation_interrupted(self):
        """Если стрим прерывается ошибкой (connection drop) и пользователь запросил
        остановку, вызывается GenerationInterrupted. close_stream патчится в no-op,
        чтобы watcher не подавлял异常 mock-генератора (реальный HTTP close вызывает
        ConnectionError на next(), здесь эмулируем через RuntimeError)."""
        agent = self.make_streaming_agent()

        def stream(*args, **kwargs):
            yield stream_chunk(stream_delta(content="partial"))
            raise RuntimeError("connection dropped")

        with self.assertRaises(GenerationInterrupted):
            with mock.patch("universal_agents.agent.LLMClient.stream", side_effect=stream), \
                 mock.patch("universal_agents.llm_client.LLMClient.close_stream"):
                agent._call_with_streaming([], stop_check=lambda: True)

    def test_chat_swallows_generation_interrupted(self):
        agent = LLMAgent(
            system_prompt="sys",
            streaming_enabled=True,
            on_stream_chunk=lambda _: None,
            on_system_msg=lambda *a, **k: None,
        )

        def stream(*args, **kwargs):
            yield stream_chunk(stream_delta(content="k"))
            raise GenerationInterrupted()

        with mock.patch("universal_agents.agent.LLMClient.stream", side_effect=stream):
            result = agent.chat("hello", max_iter=3)

        self.assertEqual(result, "")
        self.assertFalse(agent.stop_event.is_set())
        # прерванный ход не оставляет битой последовательности ролей
        roles = [m.to_api_dict()["role"] for m in agent.history]
        self.assertEqual(roles, ["system", "user"])


class TestStreamDivergence(unittest.TestCase):
    """§Streaming: позднее расхождение с прежним ответом → достройка спокойной
    генерацией; расхождение с первого символа — свежий ответ целиком, принимается
    как есть без достройки (иначе холодный рестарт: лишний вызов + риск дубля)."""

    def test_late_divergence_continues_on_calm_temp(self):
        agent = LLMAgent(system_prompt="sys", streaming_enabled=True, on_stream_chunk=lambda _: None)

        def stream(*args, **kwargs):
            yield stream_chunk(stream_delta(content="previous answer "))
            yield stream_chunk(stream_delta(content="teXt!"))

        followup = SimpleNamespace(content=" calm continuation", reasoning_content="")
        with mock.patch("universal_agents.agent.LLMClient.stream", side_effect=stream) as s_stream, \
             mock.patch("universal_agents.agent.LLMClient.call") as s_call:
            s_call.return_value = (followup, None, None)
            msg, err, usage, _stopped = agent._call_with_streaming(
                [], watch_prefix="previous answer text",
                watch_continue_temp=0.1,
            )

        self.assertIsNone(err)
        self.assertIn("previous answer teXt!", msg.content)
        self.assertIn("calm continuation", msg.content)
        self.assertTrue(msg.streamed)
        # расхождение зафиксировано после частичного совпадения — достройка идёт
        # одним вызовом LLMClient.call
        self.assertEqual(s_stream.call_count, 1)
        self.assertEqual(s_call.call_count, 1)
        # написанное передаётся prefill'ом — KV-cache продолжение, а не новая генерация
        self.assertEqual(s_call.call_args.kwargs["prefill"], "previous answer teXt!")

    def test_immediate_divergence_accepted_without_followup(self):
        agent = LLMAgent(system_prompt="sys", streaming_enabled=True, on_stream_chunk=lambda _: None)

        def stream(*args, **kwargs):
            yield stream_chunk(stream_delta(content="completely different text"))

        with mock.patch("universal_agents.agent.LLMClient.stream", side_effect=stream) as s_stream, \
             mock.patch("universal_agents.agent.LLMClient.call") as s_call:
            msg, err, usage, _stopped = agent._call_with_streaming(
                [], watch_prefix="previous answer text",
                watch_continue_temp=0.1,
            )

        self.assertIsNone(err)
        self.assertEqual(msg.content, "completely different text")
        self.assertEqual(s_stream.call_count, 1)
        s_call.assert_not_called()

    def test_no_divergence_returns_clean_stream(self):
        agent = LLMAgent(system_prompt="sys", streaming_enabled=True, on_stream_chunk=lambda _: None)

        def stream(*args, **kwargs):
            yield stream_chunk(stream_delta(content="The expected"))

        with mock.patch("universal_agents.agent.LLMClient.stream", side_effect=stream), \
             mock.patch("universal_agents.agent.LLMClient.call") as called:
            msg, err, usage, _stopped = agent._call_with_streaming(
                [], watch_prefix="The expected answer", watch_continue_temp=0.1,
            )

        self.assertIsNone(err)
        self.assertEqual(msg.content, "The expected")
        called.assert_not_called()


def _mk_toolcall(index, id=None, name=None, arguments=None):
    return SimpleNamespace(
        index=index,
        id=id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class TestAgentStreamingTurns(unittest.TestCase):
    def test_chat_streaming_executes_tool(self):
        agent = LLMAgent(
            system_prompt="sys",
            tools_config=["double_me"],
            external_plugins={"double_me": double_me},
            streaming_enabled=True,
            on_stream_chunk=lambda _: None,
        )

        def stream1(*args, **kwargs):
            yield stream_chunk(stream_delta(content="Let me compute "))
            yield stream_chunk(stream_delta(content="21 * 2 "))
            yield stream_chunk(stream_delta(tool_calls=[_mk_toolcall(0, id="t1", name="double_me", arguments='{"value": ')],
                                content=""))
            yield stream_chunk(stream_delta(tool_calls=[_mk_toolcall(0, arguments='21}')]))
            yield stream_chunk(stream_delta(tool_calls=[_mk_toolcall(0, id="t1")]))
            yield stream_chunk(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15))

        def stream2(*args, **kwargs):
            yield stream_chunk(stream_delta(content="final "))
            yield stream_chunk(stream_delta(content="answer 42"))
            yield stream_chunk(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15))

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
            yield stream_chunk(stream_delta(content="hello back"))

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
            yield stream_chunk(stream_delta(reasoning_content="think..."))
            yield stream_chunk(stream_delta(content="hello back"))

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
            yield stream_chunk(stream_delta(content=""))
            yield stream_chunk(usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2))

        with mock.patch("universal_agents.agent.LLMClient.stream", return_value=stream()):
            result = agent.chat("hello", prefill="X")

        self.assertEqual(result, "X")


if __name__ == "__main__":
    unittest.main()