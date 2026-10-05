import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from universal_agents.config import Config
from universal_agents.generation import GenerationParams
from universal_agents.llm_client import (
    LLMClient,
    LoopDetector,
    StreamAccumulator,
    StreamSession,
    TokenUsageTracker,
    jaccard_similarity,
    text_hash,
)
from universal_agents.agent_mixins.response_mixin import _NO_COMMENT_PREFILL
from universal_agents.models import AssistantMessage, ToolCall, ToolResult, UserMessage
from universal_agents.task_tracker import PLAN_TOOL


from tests.conftest import stream_chunk, stream_delta


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

    def test_min_p_sent_via_extra_body(self):
        """min_p нет в сигнатуре OpenAI-клиента — уходит сырым полем тела запроса."""
        fake_client = mock.Mock()
        fake_client.chat.completions.create.return_value = self._fake_response()
        with mock.patch("universal_agents.llm_client.LLMClient.get_client", return_value=fake_client):
            LLMClient.call(
                [{"role": "user", "content": "hi"}],
                params=GenerationParams(min_p=0.1),
            )
        kwargs = fake_client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs.get("extra_body"), {"min_p": 0.1})

    def test_min_p_omitted_by_default(self):
        """Без настройки min_p поле вообще не отправляется (сервер решит сам)."""
        fake_client = mock.Mock()
        fake_client.chat.completions.create.return_value = self._fake_response()
        with mock.patch("universal_agents.llm_client.LLMClient.get_client", return_value=fake_client), \
                mock.patch.object(Config, "MIN_P", None):
            LLMClient.call([{"role": "user", "content": "hi"}])
        kwargs = fake_client.chat.completions.create.call_args.kwargs
        self.assertNotIn("extra_body", kwargs)
        self.assertNotIn("min_p", kwargs)


class TestStreamSession(unittest.TestCase):
    """StreamSession: единый потребитель стрима (watchdog + чанковый цикл)."""

    def test_consume_returns_content(self):
        fake_client = mock.Mock()
        fake_client.chat.completions.create.return_value = iter([
            stream_chunk(SimpleNamespace(content="hello", tool_calls=None, reasoning_content=None)),
            stream_chunk(SimpleNamespace(content=" world", tool_calls=None, reasoning_content=None)),
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
                yield stream_chunk(SimpleNamespace(content=c, tool_calls=None, reasoning_content=None))

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
                        return_value=iter([stream_chunk(SimpleNamespace(content="never", tool_calls=None, reasoning_content=None))])):
            session = StreamSession(
                [{"role": "user", "content": "hi"}],
                on_stream_chunk=lambda _: None,
            )
            error, stopped = session.consume(stop_check=lambda: True)
        self.assertTrue(stopped)
        self.assertEqual(error, "stopped")


class TestStreamAccumulatorPrefillEcho(unittest.TestCase):
    """Шлюзы по-разному возвращают prefill: LM Studio вырезает его, сырой llama.cpp повторяет."""

    def _text_chunk(self, content):
        return stream_chunk(stream_delta(content=content))

    def test_echoed_prefill_split_across_chunks_is_shown_once(self):
        chunks = []
        acc = StreamAccumulator(prefill="<short_think>", on_stream_chunk=chunks.append)
        acc.process(self._text_chunk("<short"))
        acc.process(self._text_chunk("_think>мысли"))
        acc.flush_prefill()

        self.assertEqual(chunks, ["<short_think>", "мысли"])
        self.assertEqual(acc.content, "<short_think>мысли")
        self.assertEqual(acc.build_message("<short_think>").content, "<short_think>мысли")

    def test_omitted_prefill_is_shown_once(self):
        chunks = []
        acc = StreamAccumulator(prefill="<short_think>", on_stream_chunk=chunks.append)
        acc.process(self._text_chunk("мысли"))
        acc.flush_prefill()

        self.assertEqual(chunks, ["<short_think>", "мысли"])
        self.assertEqual(acc.content, "мысли")
        self.assertEqual(acc.build_message("<short_think>").content, "<short_think>мысли")

    def test_continuation_echo_shows_only_unseen_suffix(self):
        prefill = "<short_think>мысли</short_think>" + _NO_COMMENT_PREFILL
        shown = len("<short_think>мысли</short_think>")
        chunks = []
        acc = StreamAccumulator(prefill=prefill, on_stream_chunk=chunks.append, prefill_shown=shown)
        acc.process(self._text_chunk(prefill))
        acc.process(self._text_chunk("ответ"))
        acc.flush_prefill()

        self.assertEqual(chunks, [_NO_COMMENT_PREFILL, "ответ"])
        self.assertEqual(acc.content, prefill + "ответ")

    def test_continuation_without_echo_shows_only_unseen_suffix(self):
        prefill = "<short_think>мысли</short_think>" + _NO_COMMENT_PREFILL
        shown = len("<short_think>мысли</short_think>")
        chunks = []
        acc = StreamAccumulator(prefill=prefill, on_stream_chunk=chunks.append, prefill_shown=shown)
        acc.process(self._text_chunk("ответ"))
        acc.flush_prefill()

        self.assertEqual(chunks, [_NO_COMMENT_PREFILL, "ответ"])
        self.assertEqual(acc.content, "ответ")


class TestTextHash(unittest.TestCase):
    def test_strips_whitespace(self):
        self.assertEqual(text_hash("  hello  "), text_hash("hello"))

    def test_different_texts_different_hashes(self):
        self.assertNotEqual(text_hash("abc"), text_hash("def"))


class TestBigramJaccard(unittest.TestCase):
    """Биграммный Jaccard: учитывает порядок слов, в отличие от мешка слов."""

    def test_identical_texts(self):
        self.assertEqual(jaccard_similarity("read the file", "read the file"), 1.0)

    def test_completely_different(self):
        self.assertEqual(jaccard_similarity("alpha beta gamma", "delta epsilon zeta"), 0.0)

    def test_word_order_matters(self):
        """Перестановка слов даёт низкую схожесть — ключевое отличие от мешка слов."""
        sim = jaccard_similarity("read A then edit B", "edit B then read A")
        self.assertLess(sim, 0.5)
        self.assertLess(sim, Config.DUPLICATE_SIMILARITY_THRESHOLD)

    def test_partial_overlap(self):
        sim = jaccard_similarity("I will read the file", "I will read the file now")
        self.assertGreater(sim, 0.6)
        self.assertLess(sim, 1.0)

    def test_empty_inputs(self):
        self.assertEqual(jaccard_similarity("", ""), 1.0)
        self.assertEqual(jaccard_similarity("hello", ""), 0.0)

    def test_single_word(self):
        self.assertEqual(jaccard_similarity("hello", "hello"), 1.0)
        self.assertEqual(jaccard_similarity("hello", "world"), 0.0)


class TestLoopDetector(unittest.TestCase):
    def setUp(self):
        self.detector = LoopDetector()

    def test_normalize_args_ignores_whitespace_and_key_order(self):
        self.assertEqual(
            LoopDetector.normalize_args('{ "b": 2, "a": 1 }'),
            '{"a":1,"b":2}',
        )
        self.assertEqual(LoopDetector.normalize_args("{}"), "")
        self.assertEqual(LoopDetector.normalize_args(""), "")
        self.assertEqual(LoopDetector.normalize_args("not json"), "not json")

    def test_detects_duplicate_in_turn(self):
        history = [
            UserMessage("hi"),
            AssistantMessage(content="", tool_calls=[ToolCall(id="t1", name="read", arguments="{}")]),
        ]
        self.assertTrue(self.detector.check_duplicate_in_turn("read", "{}", history))

    def test_ignores_calls_before_user_message(self):
        history = [
            AssistantMessage(content="", tool_calls=[ToolCall(id="t1", name="read", arguments="{}")]),
            UserMessage("hi"),
            AssistantMessage(content="", tool_calls=[ToolCall(id="t2", name="search", arguments="{}")]),
        ]
        # read вызывался до начала хода — не считается дублем
        self.assertFalse(self.detector.check_duplicate_in_turn("read", "{}", history))
        # search уже вызван в текущем ходу с теми же аргументами — дубль
        self.assertTrue(self.detector.check_duplicate_in_turn("search", "{}", history))
        # другие аргументы — не дубль
        self.assertFalse(self.detector.check_duplicate_in_turn("search", '{"x": 1}', history))

    def test_detects_semantically_duplicate_after_user(self):
        history = [
            UserMessage("hi"),
            AssistantMessage(content="", tool_calls=[ToolCall(id="t1", name="read", arguments='{"a": 1}')]),
        ]
        self.assertTrue(self.detector.check_duplicate_in_turn("read", '{ "a" : 1 }', history))
        self.assertFalse(self.detector.check_duplicate_in_turn("read", '{"a": 2}', history))

    def test_repeated_make_plan_same_args_is_loop(self):
        history = [
            UserMessage("do the task"),
            AssistantMessage(content="", tool_calls=[
                ToolCall(id="c1", name=PLAN_TOOL, arguments='{"plan":[{"id":"t2","title":"X"}]}')
            ]),
        ]
        self.assertTrue(self.detector.check_duplicate_in_turn(
            PLAN_TOOL, '{"plan":[{"id":"t2","title":"X"}]}', history))

    def test_make_plan_revision_with_different_args_is_allowed(self):
        history = [
            UserMessage("do the task"),
            AssistantMessage(content="", tool_calls=[
                ToolCall(id="c1", name=PLAN_TOOL, arguments='{"plan":[{"id":"t1","title":"X"}]}')
            ]),
        ]
        self.assertFalse(self.detector.check_duplicate_in_turn(
            PLAN_TOOL, '{"plan":[{"id":"t2","title":"Y"}]}', history))

    def test_make_plan_resets_duplicate_scan_for_other_tools(self):
        history = [
            UserMessage("do the task"),
            AssistantMessage(content="", tool_calls=[ToolCall(id="r", name="read", arguments="{}")]),
            AssistantMessage(content="", tool_calls=[
                ToolCall(id="c1", name=PLAN_TOOL, arguments='{"plan":[{"id":"t1","title":"X"}]}')
            ]),
        ]
        # Повторный read ПОСЛЕ make_plan не считается дублем (ревизия = граница контекста)
        self.assertFalse(self.detector.check_duplicate_in_turn("read", "{}", history))

    def test_failed_call_is_not_duplicate(self):
        # have_done отклонён (NO-WORK-DONE) → повторный вызов после работы не дубль
        history = [
            UserMessage("do the task"),
            AssistantMessage(content="", tool_calls=[ToolCall(id="t1", name="have_done", arguments='{"id":"a1"}')]),
            ToolResult.error("t1", "have_done", "NO-WORK-DONE: you marked done but did nothing"),
            AssistantMessage(content="", tool_calls=[ToolCall(id="r", name="run_bash_host", arguments="{}")]),
            ToolResult.success("r", "run_bash_host", "done"),
        ]
        self.assertFalse(self.detector.check_duplicate_in_turn("have_done", '{"id":"a1"}', history))

    def test_succeeded_call_still_counts_as_duplicate(self):
        # успешный вызов с теми же аргументами — дубль (реального прогресса не было)
        history = [
            UserMessage("do the task"),
            AssistantMessage(content="", tool_calls=[ToolCall(id="t1", name="read", arguments="{}")]),
            ToolResult.success("t1", "read", "ok"),
        ]
        self.assertTrue(self.detector.check_duplicate_in_turn("read", "{}", history))

    def test_duplicate_after_other_tool_is_not_loop(self):
        # Если между двумя одинаковыми вызовами был другой инструмент — это не зацикливание.
        history = [
            UserMessage("do the task"),
            AssistantMessage(content="", tool_calls=[ToolCall(id="t1", name="edit_file", arguments='{"path":"x.txt"}')]),
            ToolResult.success("t1", "edit_file", "ok"),
            AssistantMessage(content="", tool_calls=[ToolCall(id="t2", name="read", arguments='{"path":"x.txt"}')]),
            ToolResult.success("t2", "read", "content"),
        ]
        self.assertFalse(self.detector.check_duplicate_in_turn("edit_file", '{"path":"x.txt"}', history))

    def test_consecutive_same_call_is_still_duplicate(self):
        # Два одинаковых вызова подряд (без другого инструмента между ними) — дубль
        history = [
            UserMessage("do the task"),
            AssistantMessage(content="", tool_calls=[ToolCall(id="t1", name="edit_file", arguments='{"path":"x.txt"}')]),
            ToolResult.success("t1", "edit_file", "ok"),
            AssistantMessage(content="", tool_calls=[ToolCall(id="t2", name="edit_file", arguments='{"path":"x.txt"}')]),
        ]
        self.assertTrue(self.detector.check_duplicate_in_turn("edit_file", '{"path":"x.txt"}', history))


class TestTokenUsageTracker(unittest.TestCase):
    def test_estimate_tokens(self):
        # int(len / CHARS_PER_TOKEN): 46/2.35 = 19.57 -> 19, 47/2.35 = 20.0 -> 20
        self.assertEqual(TokenUsageTracker.estimate_tokens("x" * 46), 19)
        self.assertEqual(TokenUsageTracker.estimate_tokens("x" * 47), 20)
        self.assertEqual(TokenUsageTracker.estimate_tokens(""), 0)

    def test_remaining(self):
        tracker = TokenUsageTracker("system prompt", max_context_tokens=1000)
        tracker.update_from_usage({"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})
        self.assertEqual(tracker.last_usage["prompt_tokens"], 100)
        self.assertLessEqual(tracker.get_remaining(), 900)

    def test_format_user_token_info(self):
        tracker = TokenUsageTracker("sys", max_context_tokens=1000)
        self.assertEqual(tracker.format_user_token_info(), "")
        tracker.update_from_usage({"prompt_tokens": 300, "completion_tokens": 50, "total_tokens": 350})
        info = tracker.format_user_token_info()
        # «Remaining» — окно контекста (max - prompt_tokens последнего вызова).
        self.assertIn("350", info)
        self.assertIn("700", info)


class TestStreamSessionStopMarkers(unittest.TestCase):
    def test_marker_truncates_content_and_closes(self):
        """Стоп-маркер обрезает контент, закрывает стрим и ставит флаг."""
        def stream():
            yield stream_chunk(SimpleNamespace(content="L1-5 foo\n", tool_calls=None, reasoning_content=None))
            yield stream_chunk(SimpleNamespace(content="</structure> more stuff", tool_calls=None, reasoning_content=None))
            yield stream_chunk(SimpleNamespace(content=" never seen", tool_calls=None, reasoning_content=None))

        with mock.patch("universal_agents.llm_client.LLMClient.stream", return_value=stream()), \
             mock.patch("universal_agents.llm_client.LLMClient.close_stream"):
            session = StreamSession(
                [{"role": "user", "content": "hi"}],
                on_stream_chunk=lambda _: None,
            )
            error, stopped = session.consume(stop_markers=("</structure>",))

        self.assertFalse(error)
        self.assertTrue(stopped)
        self.assertTrue(session.stopped_at_marker)
        self.assertEqual(session.acc.content, "L1-5 foo\n</structure>")

    def test_no_marker_leaves_content_untouched(self):
        def stream():
            yield stream_chunk(SimpleNamespace(content="no marker", tool_calls=None, reasoning_content=None))

        with mock.patch("universal_agents.llm_client.LLMClient.stream", return_value=stream()):
            session = StreamSession(
                [{"role": "user", "content": "hi"}],
                on_stream_chunk=lambda _: None,
            )
            error, stopped = session.consume(stop_markers=("</end>",))

        self.assertFalse(stopped)
        self.assertFalse(session.stopped_at_marker)
        self.assertEqual(session.acc.content, "no marker")

    def test_marker_in_first_chunk(self):
        def stream():
            yield stream_chunk(SimpleNamespace(content="<tag>\n</tag>", tool_calls=None, reasoning_content=None))

        with mock.patch("universal_agents.llm_client.LLMClient.stream", return_value=stream()), \
             mock.patch("universal_agents.llm_client.LLMClient.close_stream"):
            session = StreamSession(
                [{"role": "user", "content": "hi"}],
                on_stream_chunk=lambda _: None,
            )
            error, stopped = session.consume(stop_markers=("</tag>",))

        self.assertTrue(stopped)
        self.assertTrue(session.stopped_at_marker)
        self.assertEqual(session.acc.content, "<tag>\n</tag>")

    def test_stop_check_takes_priority_over_marker(self):
        """Если stop_check сработал раньше маркера — это stop_check, а не маркер."""
        processed = [0]

        def stream():
            for c in ["a", "b", "c"]:
                processed[0] += 1
                yield stream_chunk(SimpleNamespace(content=c, tool_calls=None, reasoning_content=None))

        with mock.patch("universal_agents.llm_client.LLMClient.stream", return_value=stream()), \
             mock.patch("universal_agents.llm_client.LLMClient.close_stream"):
            session = StreamSession(
                [{"role": "user", "content": "hi"}],
                on_stream_chunk=lambda _: None,
            )
            error, stopped = session.consume(
                stop_check=lambda: processed[0] >= 2,
                stop_markers=("b",),
            )

        self.assertTrue(stopped)
        self.assertFalse(session.stopped_at_marker)


class TestCallWithStopMarkers(unittest.TestCase):
    def _fake_response(self, content):
        msg = SimpleNamespace(content=content, tool_calls=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=msg)],
            usage=None,
        )

    def test_nonstream_truncates_at_marker(self):
        fake_client = mock.Mock()
        fake_client.chat.completions.create.return_value = self._fake_response(
            "<tag>\ndata\n</tag>\nanalysis"
        )
        with mock.patch("universal_agents.llm_client.LLMClient.get_client", return_value=fake_client):
            result, err, usage = LLMClient.call(
                [{"role": "user", "content": "hi"}],
                stop_markers=("</tag>",),
            )
        self.assertIsNone(err)
        self.assertEqual(result.content, "<tag>\ndata\n</tag>")

    def test_nonstream_no_marker_keeps_content(self):
        fake_client = mock.Mock()
        fake_client.chat.completions.create.return_value = self._fake_response("no tags")
        with mock.patch("universal_agents.llm_client.LLMClient.get_client", return_value=fake_client):
            result, err, usage = LLMClient.call(
                [{"role": "user", "content": "hi"}],
                stop_markers=("</end>",),
            )
        self.assertIsNone(err)
        self.assertEqual(result.content, "no tags")
