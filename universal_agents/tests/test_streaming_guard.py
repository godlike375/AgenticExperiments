import unittest
from types import SimpleNamespace
from unittest import mock

from universal_agents.agent import LLMAgent
from universal_agents.agent_mixins.streaming_mixin import StreamingMixin
from universal_agents.exceptions import GenerationInterrupted


def _chunk(delta=None, usage=None, choices=None):
    if choices is None:
        choices = [SimpleNamespace(delta=delta)] if delta is not None else []
    return SimpleNamespace(choices=choices, usage=usage)


def _delta(content=None, tool_calls=None, reasoning_content=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls, reasoning_content=reasoning_content)


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
            yield _chunk(_delta(content="hello"))
            yield _chunk(_delta(content=" world"))
            yield _chunk(_delta(content=" never seen"))

        def stop_check():
            # останавливаем после накопления "hello world" — третий чанк не должен
            # попасть в ответ. Критерий по содержимому, а не по числу вызовов: watcher
            # (закрывающий стрим) и основной цикл зовут stop_check на разных фазах.
            return "hello world" in "".join(seen)

        with mock.patch("universal_agents.agent.LLMClient.stream", side_effect=stream), \
             mock.patch("universal_agents.llm_client.LLMClient.close_stream"):
            msg, err, usage = agent._call_with_streaming([], stop_check=stop_check)

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
            yield _chunk(_delta(content="partial"))
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
            yield _chunk(_delta(content="k"))
            raise GenerationInterrupted()

        with mock.patch("universal_agents.agent.LLMClient.stream", side_effect=stream):
            result = agent.chat("hello", max_iter=3)

        self.assertEqual(result, "")
        self.assertFalse(agent.stop_event.is_set())
        # прерванный ход не оставляет битой последовательности ролей
        roles = [m.to_api_dict()["role"] for m in agent.history]
        self.assertEqual(roles, ["system", "user"])


class TestStreamDivergence(unittest.TestCase):
    """§Streaming: расхождение с прежним ответом → достройка спокойной генерацией."""

    def test_divergence_continues_on_calm_temp(self):
        agent = LLMAgent(system_prompt="sys", streaming_enabled=True, on_stream_chunk=lambda _: None)

        def stream(*args, **kwargs):
            yield _chunk(_delta(content="completely different text"))

        followup = SimpleNamespace(content=" calm continuation", reasoning_content="")
        with mock.patch("universal_agents.agent.LLMClient.stream", side_effect=stream) as s_stream, \
             mock.patch("universal_agents.agent.LLMClient.call") as s_call:
            s_call.return_value = (followup, None, None)
            msg, err, usage = agent._call_with_streaming(
                [], watch_prefix="previous answer text",
                watch_continue_temp=0.1,
            )

        self.assertIsNone(err)
        self.assertIn("completely different text", msg.content)
        self.assertIn("calm continuation", msg.content)
        self.assertTrue(msg.streamed)
        # расхождение зафиксировано на первом чанке — достройка идёт одним вызовом LLMClient.call
        self.assertEqual(s_stream.call_count, 1)
        self.assertEqual(s_call.call_count, 1)
        # прежний ответ передаётся prefill'ом — KV-cache продолжение, а не новая генерация
        self.assertEqual(s_call.call_args.kwargs["prefill"], "completely different text")

    def test_no_divergence_returns_clean_stream(self):
        agent = LLMAgent(system_prompt="sys", streaming_enabled=True, on_stream_chunk=lambda _: None)

        def stream(*args, **kwargs):
            yield _chunk(_delta(content="The expected"))

        with mock.patch("universal_agents.agent.LLMClient.stream", side_effect=stream), \
             mock.patch("universal_agents.agent.LLMClient.call") as called:
            msg, err, usage = agent._call_with_streaming(
                [], watch_prefix="The expected answer", watch_continue_temp=0.1,
            )

        self.assertIsNone(err)
        self.assertEqual(msg.content, "The expected")
        called.assert_not_called()


if __name__ == "__main__":
    unittest.main()