import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from universal_agents.config import Config
from universal_agents.generation import GenerationParams
from universal_agents.llm_client import LLMClient, StreamSession


def _chunk(delta=None, usage=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta)],
        usage=usage,
    )


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

    def test_stream_failure_no_blocking_fallback_when_stopped(self):
        """Если стрим не создался и пользователь запросил остановку, call() не должен
        скатываться в блокирующий _call_chat_completions (его прервать нельзя)."""
        fake_client = mock.Mock()
        fake_client.chat.completions.create.side_effect = RuntimeError("boom")
        with mock.patch("universal_agents.llm_client.LLMClient.get_client", return_value=fake_client), \
                mock.patch.object(Config, "STREAM_ENABLED", True), \
                mock.patch("universal_agents.llm_client.LLMClient._call_chat_completions") as blocking:
            result, err, usage = LLMClient.call(
                [{"role": "user", "content": "hi"}],
                callbacks={"on_stream_chunk": lambda c: None},
                stop_check=lambda: True,
            )
        blocking.assert_not_called()
        self.assertIsNone(result)
        self.assertIn("stopped", err)

    def test_stream_creation_failure_keeps_error_no_blocking_fallback(self):
        """Недоступный стрим возвращает кортеж ошибки (не None), чтобы call() пошёл по
        пути ошибки, а не в блокирующий обычный вызов."""
        fake_client = mock.Mock()
        fake_client.chat.completions.create.side_effect = RuntimeError("boom")
        with mock.patch("universal_agents.llm_client.LLMClient.get_client", return_value=fake_client), \
                mock.patch.object(Config, "STREAM_ENABLED", True), \
                mock.patch("universal_agents.llm_client.LLMClient._call_chat_completions") as blocking:
            result, err, usage = LLMClient.call(
                [{"role": "user", "content": "hi"}],
                callbacks={"on_stream_chunk": lambda c: None},
            )
        blocking.assert_not_called()
        self.assertIsNone(result)
        self.assertIn("stream creation failed", err)


class TestStreamSession(unittest.TestCase):
    """StreamSession: единый потребитель стрима (watchdog + чанковый цикл)."""

    def _chunk(self, delta=None, usage=None):
        return SimpleNamespace(
            choices=[SimpleNamespace(delta=delta)],
            usage=usage,
        )

    def test_consume_returns_content(self):
        fake_client = mock.Mock()
        fake_client.chat.completions.create.return_value = iter([
            self._chunk(SimpleNamespace(content="hello", tool_calls=None, reasoning_content=None)),
            self._chunk(SimpleNamespace(content=" world", tool_calls=None, reasoning_content=None)),
        ])
        with mock.patch("universal_agents.llm_client.LLMClient.get_client", return_value=fake_client):
            session = StreamSession(
                [{"role": "user", "content": "hi"}],
                on_stream_chunk=lambda _: None,
            )
            error, stopped = session.consume()
        self.assertFalse(error)
        self.assertFalse(stopped)
        self.assertEqual(session.acc.content, "hello world")

    def test_consume_stop_check_breaks_loop_and_closes(self):
        """stop_check в цикле прерывает потребление: стрим закрывается, последующие
        чанки не обрабатываются."""
        processed = [0]

        def generator():
            for c in ["a", "b", "c"]:
                processed[0] += 1
                yield self._chunk(SimpleNamespace(content=c, tool_calls=None, reasoning_content=None))

        close_calls = []
        original_close = LLMClient.close_stream

        def track_close(s):
            close_calls.append(1)
            original_close(s)

        def stop():
            return processed[0] >= 2

        with mock.patch("universal_agents.llm_client.LLMClient.stream", return_value=generator()), \
             mock.patch("universal_agents.llm_client.LLMClient.close_stream", side_effect=track_close):
            session = StreamSession(
                [{"role": "user", "content": "hi"}],
                on_stream_chunk=lambda _: None,
            )
            error, stopped = session.consume(stop_check=stop)

        self.assertFalse(error)
        self.assertTrue(stopped)
        self.assertEqual(session.acc.content, "ab")
        self.assertTrue(close_calls)

    def test_consume_watchdog_covers_first_chunk_wait(self):
        """Stop во время ожидания ПЕРВОГО чанка закрывает стрим и завершает consume.
        До починки watchdog стартовал после первого чанка, и 'q' не работал, пока сервер
        долго считал префилл (ввод выглядел замороженным)."""
        released = threading.Event()

        class BlockingStream:
            def __next__(self):
                while not released.wait(0.02):
                    pass
                raise StopIteration

            def close(self):
                released.set()

        stop_evt = threading.Event()
        with mock.patch("universal_agents.llm_client.LLMClient.stream", return_value=BlockingStream()):
            session = StreamSession(
                [{"role": "user", "content": "hi"}],
                on_stream_chunk=lambda _: None,
            )
        result = {}

        def consumer():
            result["res"] = session.consume(stop_check=stop_evt.is_set)

        t = threading.Thread(target=consumer, daemon=True)
        t.start()
        time.sleep(0.2)
        stop_evt.set()
        t.join(3)
        self.assertFalse(t.is_alive(), "consume завис на ожидании первого чанка после stop")
        self.assertTrue(released.is_set(), "watchdog не закрыл стрим на первой фазе")
        self.assertIn("res", result)

    def test_consume_stop_already_set_returns_stopped(self):
        """Если stop запрошен ещё до начала потребления — блокирующий цикл не начинается."""
        with mock.patch("universal_agents.llm_client.LLMClient.stream",
                        return_value=iter([self._chunk(SimpleNamespace(content="never", tool_calls=None, reasoning_content=None))])):
            session = StreamSession(
                [{"role": "user", "content": "hi"}],
                on_stream_chunk=lambda _: None,
            )
            error, stopped = session.consume(stop_check=lambda: True)
        self.assertTrue(stopped)
        self.assertEqual(error, "stopped")
