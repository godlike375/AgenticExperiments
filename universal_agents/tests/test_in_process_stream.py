
from __future__ import annotations

import json
import threading
import time
import unittest
from unittest import mock

import httpx

from universal_agents.config import Config
from universal_agents.generation import GenerationParams
from universal_agents.llm_client import (
    InProcessStream,
    LLMClient,
    StreamAccumulator,
)


def _sse_response(chunks: list[dict], delay: float = 0) -> httpx.Response:
    lines = []
    for chunk in chunks:
        lines.append(f"data: {json.dumps(chunk)}")
    lines.append("data: [DONE]")
    content = "\n".join(lines).encode()

    request = httpx.Request("POST", "http://test/chat/completions")
    response = httpx.Response(200, content=content, request=request)
    return response


def _text_chunk(content: str) -> dict:
    return {
        "choices": [{"delta": {"content": content}, "index": 0}],
        "object": "chat.completion.chunk",
    }


def _usage_chunk() -> dict:
    return {
        "choices": [],
        "object": "chat.completion.chunk",
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }


def _tool_call_chunk(index: int, id: str, name: str, arguments: str) -> dict:
    return {
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": index,
                    "id": id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }]
            },
            "index": 0,
        }],
        "object": "chat.completion.chunk",
    }


def _reasoning_chunk(content: str) -> dict:
    return {
        "choices": [{"delta": {"reasoning_content": content}, "index": 0}],
        "object": "chat.completion.chunk",
    }


def _error_chunk(message: str) -> dict:
    return {"error": {"message": message}}


def _collect(stream) -> StreamAccumulator:
    acc = StreamAccumulator()
    try:
        for chunk in stream:
            acc.process(chunk)
    finally:
        stream.close()
    return acc


class TestInProcessStreamParsing(unittest.TestCase):
    def test_streams_text_and_usage(self):
        response = _sse_response([
            _text_chunk("hello"),
            _text_chunk(" world"),
            _usage_chunk(),
        ])
        with mock.patch("httpx.Client.send", return_value=response):
            acc = _collect(InProcessStream([{"role": "user", "content": "hi"}]))
        self.assertEqual(acc.content, "hello world")
        self.assertEqual(acc.usage["total_tokens"], 8)

    def test_streams_tool_calls(self):
        response = _sse_response([
            _text_chunk("вызываю "),
            _tool_call_chunk(0, "c1", "read", '{"path":'),
            _tool_call_chunk(0, None, None, '"x.txt"}'),
        ])
        with mock.patch("httpx.Client.send", return_value=response):
            acc = _collect(InProcessStream([{"role": "user", "content": "hi"}]))
        self.assertEqual(acc.content, "вызываю ")
        message = acc.build_message()
        self.assertIsNotNone(message.tool_calls)
        self.assertEqual(message.tool_calls[0].name, "read")
        self.assertEqual(message.tool_calls[0].arguments, '{"path":"x.txt"}')

    def test_reasoning_content_survives(self):
        response = _sse_response([
            _reasoning_chunk("дума"),
            _reasoning_chunk("ю"),
            _text_chunk("ответ"),
        ])
        with mock.patch("httpx.Client.send", return_value=response):
            acc = _collect(InProcessStream([{"role": "user", "content": "hi"}]))
        self.assertEqual(acc.reasoning, "думаю")
        self.assertEqual(acc.content, "ответ")

    def test_create_error_surfaces_as_error_dict(self):
        response = _sse_response([_error_chunk("boom")])
        with mock.patch("httpx.Client.send", return_value=response):
            stream = InProcessStream([{"role": "user", "content": "hi"}])
            try:
                first = next(stream)
            finally:
                stream.close()
        self.assertEqual(first, {"error": {"message": "boom"}})

    def test_send_exception_returns_error_generator(self):
        with mock.patch("httpx.Client.send", side_effect=RuntimeError("connection failed")):
            stream = InProcessStream([{"role": "user", "content": "hi"}])
            try:
                first = next(stream)
            finally:
                stream.close()
        self.assertIn("error", first)
        self.assertIn("connection failed", first["error"])

    def test_malformed_json_lines_skipped(self):
        lines = [
            b"data: {invalid json}",
            b"data: " + json.dumps(_text_chunk("ok")).encode(),
            b"data: [DONE]",
        ]
        content = b"\n".join(lines)
        request = httpx.Request("POST", "http://test/chat/completions")
        response = httpx.Response(200, content=content, request=request)
        with mock.patch("httpx.Client.send", return_value=response):
            acc = _collect(InProcessStream([{"role": "user", "content": "hi"}]))
        self.assertEqual(acc.content, "ok")


class TestInProcessStreamClose(unittest.TestCase):
    def test_close_unblocks_iter_lines(self):
        released = threading.Event()

        class FakeSocket:
            def shutdown(self, how):
                released.set()

        class FakeNetworkStream:
            def get_extra_info(self, key):
                if key == "socket":
                    return FakeSocket()
                return None

        class FakeResponse:
            extensions = {"network_stream": FakeNetworkStream()}

            def iter_lines(self):
                while not released.wait(0.02):
                    pass
                return iter([])

            def close(self):
                pass

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            def build_request(self, method, url, json=None):
                return httpx.Request(method, url)

            def send(self, request, stream=False):
                return FakeResponse()

            def close(self):
                pass

        with mock.patch("httpx.Client", FakeClient):
            stream = InProcessStream([{"role": "user", "content": "hi"}])
            result = {}

            def consume():
                try:
                    next(stream)
                except StopIteration:
                    result["stopped"] = True

            t = threading.Thread(target=consume, daemon=True)
            t.start()
            time.sleep(0.1)
            started = time.monotonic()
            stream.close()
            t.join(2)
            elapsed = time.monotonic() - started

            self.assertFalse(t.is_alive(), "next() не разблокировался после close()")
            self.assertTrue(released.is_set(), "shutdown не был вызван на сокете")
            self.assertLess(elapsed, 1.0, "остановка заняла слишком долго")

    def test_close_without_socket_still_works(self):
        class FakeResponse:
            extensions = {}

            def iter_lines(self):
                return iter([])

            def close(self):
                pass

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            def build_request(self, method, url, json=None):
                return httpx.Request(method, url)

            def send(self, request, stream=False):
                return FakeResponse()

            def close(self):
                pass

        with mock.patch("httpx.Client", FakeClient):
            stream = InProcessStream([{"role": "user", "content": "hi"}])
            stream.close()
            self.assertTrue(stream._closed)


class TestOpenStreamRequest(unittest.TestCase):
    def test_open_stream_sends_request(self):
        with mock.patch("universal_agents.llm_client.httpx.Client") as mock_client_cls:
            mock_client = mock.Mock()
            mock_client_cls.return_value = mock_client
            mock_response = mock.Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.extensions = {}
            mock_response.iter_lines.return_value = iter([])
            mock_client.send.return_value = mock_response
            mock_client.build_request.return_value = mock.Mock()

            gen = LLMClient.open_stream([{"role": "user", "content": "hi"}])
            list(gen)

            mock_client.build_request.assert_called_once()
            args, kwargs = mock_client.build_request.call_args
            self.assertEqual(args[0], "POST")
            self.assertTrue(args[1].startswith("http"))
            self.assertTrue(args[1].endswith("/chat/completions"))
            body = kwargs["json"]
            self.assertTrue(body["stream"])
            self.assertEqual(body["model"], Config.MODEL_NAME)
            self.assertEqual(body["stream_options"], {"include_usage": True})

    def test_min_p_sent_at_top_level(self):
        with mock.patch("universal_agents.llm_client.httpx.Client") as mock_client_cls:
            mock_client = mock.Mock()
            mock_client_cls.return_value = mock_client
            mock_response = mock.Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.extensions = {}
            mock_response.iter_lines.return_value = iter([])
            mock_client.send.return_value = mock_response
            mock_client.build_request.return_value = mock.Mock()

            gen = LLMClient.open_stream(
                [{"role": "user", "content": "hi"}],
                params=GenerationParams(min_p=0.1),
            )
            list(gen)

            body = mock_client.build_request.call_args.kwargs["json"]
            self.assertEqual(body.get("min_p"), 0.1)
            self.assertNotIn("extra_body", body)

    def test_min_p_omitted_by_default(self):
        with mock.patch("universal_agents.llm_client.httpx.Client") as mock_client_cls, \
                mock.patch.object(Config, "MIN_P", None):
            mock_client = mock.Mock()
            mock_client_cls.return_value = mock_client
            mock_response = mock.Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.extensions = {}
            mock_response.iter_lines.return_value = iter([])
            mock_client.send.return_value = mock_response
            mock_client.build_request.return_value = mock.Mock()

            gen = LLMClient.open_stream([{"role": "user", "content": "hi"}])
            list(gen)

            body = mock_client.build_request.call_args.kwargs["json"]
            self.assertNotIn("min_p", body)


class TestOpenStreamSelection(unittest.TestCase):
    def test_open_stream_returns_in_process_stream(self):
        fake = mock.Mock()
        with mock.patch("universal_agents.llm_client.InProcessStream", return_value=fake) as mocked:
            result = LLMClient.open_stream([{"role": "user", "content": "hi"}])
        self.assertIs(result, fake)
        mocked.assert_called_once()


class TestCallEndToEndViaInProcessStream(unittest.TestCase):
    def _use_fake_response(self, response: httpx.Response):
        self._patcher = mock.patch("httpx.Client.send", return_value=response)
        self._patcher.start()
        self.addCleanup(self._patcher.stop)

    def test_call_end_to_end(self):
        self._use_fake_response(_sse_response([
            _text_chunk("hello"),
            _text_chunk(" world"),
            _usage_chunk(),
        ]))
        result, err, usage = LLMClient.call(
            [{"role": "user", "content": "hi"}],
            prefill="start ",
        )
        self.assertIsNone(err)
        self.assertEqual(result.content, "start hello world")
        self.assertEqual(usage["total_tokens"], 8)

    def test_call_collects_live_chunks_via_callbacks(self):
        self._use_fake_response(_sse_response([
            _text_chunk("hello"),
            _text_chunk(" world"),
        ]))
        seen = []
        result, err, _ = LLMClient.call(
            [{"role": "user", "content": "hi"}],
            callbacks={"on_stream_chunk": seen.append},
        )
        self.assertIsNone(err)
        self.assertEqual("".join(seen), "hello world")

    def test_stop_closes_stream_via_watchdog(self):
        released = threading.Event()

        class FakeSocket:
            def shutdown(self, how):
                released.set()

        class FakeNetworkStream:
            def get_extra_info(self, key):
                if key == "socket":
                    return FakeSocket()
                return None

        class FakeResponse:
            extensions = {"network_stream": FakeNetworkStream()}

            def iter_lines(self):
                while not released.wait(0.02):
                    pass
                raise StopIteration

            def raise_for_status(self):
                pass

            def close(self):
                pass

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            def build_request(self, method, url, json=None):
                return httpx.Request(method, url)

            def send(self, request, stream=False):
                return FakeResponse()

            def close(self):
                pass

        stop = threading.Event()
        timer = threading.Timer(0.2, stop.set)
        timer.start()
        started = time.monotonic()
        try:
            with mock.patch("httpx.Client", FakeClient):
                result, err, _ = LLMClient.call(
                    [{"role": "user", "content": "hi"}],
                    stop_check=stop.is_set,
                )
        finally:
            timer.cancel()
        elapsed = time.monotonic() - started
        self.assertIsNone(result)
        self.assertIn("stopped", err)
        self.assertLess(elapsed, 2.0, "остановка не должна ждать таймаута")
        self.assertTrue(released.is_set(), "watchdog не закрыл стрим")
