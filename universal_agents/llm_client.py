import hashlib
import json
import socket
import threading
from typing import Optional, Callable
from types import SimpleNamespace
import httpx
from universal_agents.config import Config
from universal_agents.generation import GenerationParams, apply_stop_markers
from universal_agents.tool_parsing import normalize_args, build_tool_calls
from universal_agents.tools.builtin import make_plan as _make_plan_tool
from universal_agents.exceptions import GenerationInterrupted


def jaccard_similarity(a: str, b: str) -> float:
    words_a = a.split()
    words_b = b.split()
    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0
    if len(words_a) < 2 or len(words_b) < 2:
        return 1.0 if words_a == words_b else 0.0
    bigrams_a = set(zip(words_a, words_a[1:]))
    bigrams_b = set(zip(words_b, words_b[1:]))
    return len(bigrams_a & bigrams_b) / len(bigrams_a | bigrams_b)


def text_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def build_usage_dict(prompt_tokens: int, completion_tokens: int, total_tokens: Optional[int] = None) -> dict:
    if total_tokens is None:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def apply_prefill(content: Optional[str], prefill: Optional[str]) -> str:
    if prefill is None:
        return content or ""
    content = content or ""
    if content.startswith(prefill):
        return content
    return prefill + content


class TokenUsageTracker:
    def __init__(self, system_prompt: str, max_context_tokens: int = 8192):
        self.max_context_tokens = max_context_tokens
        self.last_usage = None
        self.system_prompt = system_prompt

    def update_from_usage(self, usage: dict):
        self.last_usage = usage

    @staticmethod
    def estimate_tokens(text: str) -> int:
        return int(len(text) / Config.CHARS_PER_TOKEN)

    def get_total_context_tokens(self, first_system_message: str = "", last_user_content: str = "") -> int:
        known = self.estimate_tokens(first_system_message)
        if self.last_usage:
            known = self.last_usage.get("prompt_tokens", 0)
        if last_user_content:
            known += self.estimate_tokens(last_user_content)
        return known

    def get_remaining(self, last_user_content: str = ""):
        total = self.get_total_context_tokens(self.system_prompt, last_user_content)
        remaining = self.max_context_tokens - total
        return remaining

    def format_user_token_info(self) -> str:
        if not self.last_usage:
            return ""
        spent = self.last_usage.get("prompt_tokens", 0) or 0
        remaining = self.max_context_tokens - spent
        return f"Context size: {spent} (Remaining: {remaining})"

class LoopDetector:
    def __init__(self):
        self.threshold = 1

    @staticmethod
    def normalize_args(args_str: str) -> str:
        return normalize_args(args_str)

    def check_duplicate_in_turn(self, tool_name: str, arguments: str, messages: list) -> bool:
        from universal_agents.models import UserMessage, AssistantMessage, ToolResult
        norm_args = self.normalize_args(arguments)
        failed_call_ids = set()

        for msg in reversed(messages):
            if isinstance(msg, UserMessage):
                break

            if isinstance(msg, ToolResult):
                if msg.is_error or msg.is_user_denied:
                    failed_call_ids.add(msg.tool_call_id)
                continue

            if isinstance(msg, AssistantMessage):
                found_plan = False
                for tc in msg.tool_calls:
                    if getattr(tc, "name", "") == _make_plan_tool.__name__:
                        found_plan = True
                        if self.normalize_args(tc.arguments) == norm_args:
                            return True
                if found_plan:
                    break
                for tc in msg.tool_calls:
                    if tc.name == tool_name:
                        if tc.id in failed_call_ids:
                            continue
                        if self.normalize_args(tc.arguments) == norm_args:
                            return True
                break
        return False

class StreamAccumulator:

    def __init__(self, prefill=None, on_stream_chunk=None,
                 on_reasoning_start=None, on_reasoning_chunk=None,
                 prefill_shown=0):
        self.content = ""
        self.reasoning = ""
        self.tool_calls_data: dict = {}
        self.usage = None
        self.reasoning_started = False
        self._prefill_pending = prefill
        self._prefill_shown = max(0, prefill_shown or 0)
        self._unresolved_head = ""
        self.on_stream_chunk = on_stream_chunk
        self.on_reasoning_start = on_reasoning_start
        self.on_reasoning_chunk = on_reasoning_chunk

    def _emit_text(self, text: str) -> None:
        if text and self.on_stream_chunk:
            self.on_stream_chunk(text)

    def _prefill_tail(self, pending: str) -> str:
        start = min(max(0, self._prefill_shown), len(pending))
        return pending[start:]

    def _resolve_prefill_echo(self, text: str) -> str:
        pending = self._prefill_pending or ""
        if not pending:
            self._prefill_pending = None
            self._unresolved_head = ""
            return text
        combined = self._unresolved_head + text
        if combined.startswith(pending):
            self._prefill_pending = None
            self._unresolved_head = ""
            self._emit_text(self._prefill_tail(pending))
            return combined[len(pending):]
        if pending.startswith(combined):
            self._unresolved_head = combined
            return ""
        self._prefill_pending = None
        self._unresolved_head = ""
        self._emit_text(self._prefill_tail(pending))
        return combined

    def flush_prefill(self) -> None:
        if self._prefill_pending is None and not self._unresolved_head:
            return
        pending = self._prefill_pending or ""
        head, self._unresolved_head = self._unresolved_head, ""
        self._prefill_pending = None
        self._emit_text(self._prefill_tail(pending))
        self._emit_text(head)

    def process(self, chunk) -> str:
        usage = getattr(chunk, 'usage', None)
        if usage:
            self.usage = build_usage_dict(
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens,
            )
        if not getattr(chunk, 'choices', None):
            return ""
        delta = chunk.choices[0].delta

        rc = getattr(delta, 'reasoning_content', None)
        if rc:
            self.reasoning += rc
            if not self.reasoning_started:
                self.reasoning_started = True
                if self.on_reasoning_start:
                    self.on_reasoning_start()
            if self.on_reasoning_chunk:
                self.on_reasoning_chunk(rc)

        added = ""
        if delta.content:
            if self._prefill_pending is not None:
                visible = self._resolve_prefill_echo(delta.content)
            else:
                visible = delta.content
            added = delta.content
            self.content += added
            self._emit_text(visible)

        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in self.tool_calls_data:
                    self.tool_calls_data[idx] = {
                        "id": tc.id or "",
                        "type": "function",
                        "function": {
                            "name": tc.function.name if tc.function and tc.function.name else "",
                            "arguments": tc.function.arguments if tc.function and tc.function.arguments else ""
                        }
                    }
                else:
                    if tc.id:
                        self.tool_calls_data[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            self.tool_calls_data[idx]["function"]["name"] += tc.function.name
                        if tc.function.arguments:
                            self.tool_calls_data[idx]["function"]["arguments"] += tc.function.arguments
        return added

    def build_message(self, prefill=None):
        return SimpleNamespace(
            content=apply_prefill(self.content, prefill),
            tool_calls=build_tool_calls(self.tool_calls_data) or None,
            reasoning_content=self.reasoning or None,
        )


class StreamSession:

    def __init__(self, messages, temp=None, timeout=None, tools=None, prefill=None,
                 top_p=None, frequency_penalty=None, presence_penalty=None,
                 max_tokens=None, min_p=None, params=None, reasoning_effort="none",
                 on_stream_chunk=None, on_reasoning_start=None, on_reasoning_chunk=None,
                 prefill_shown=0):
        self.acc = StreamAccumulator(
            prefill=prefill,
            on_stream_chunk=on_stream_chunk,
            on_reasoning_start=on_reasoning_start,
            on_reasoning_chunk=on_reasoning_chunk,
            prefill_shown=prefill_shown,
        )
        self._raw = LLMClient.open_stream(
            messages,
            temp=temp, timeout=timeout, tools=tools, prefill=prefill,
            top_p=top_p, frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty, max_tokens=max_tokens, min_p=min_p,
            params=params,
            reasoning_effort=reasoning_effort,
        )
        self.stopped_at_marker = False

    def consume(self, stop_check=None, stop_on_chunk=None, on_stream_start=None,
                stop_markers: tuple = ()):
        error = ""
        stopped = False
        _watch_done = threading.Event()
        try:
            if stop_check and stop_check():
                return "stopped", True
            if on_stream_start:
                on_stream_start()
            if stop_check is not None:
                def _watcher():
                    while not _watch_done.is_set():
                        if stop_check():
                            LLMClient.close_stream(self._raw)
                            break
                        _watch_done.wait(0.05)
                threading.Thread(target=_watcher, daemon=True).start()
            first = next(self._raw, None)
            if first is None:
                if stop_check and stop_check():
                    return "stopped", True
                return "empty stream", False
            if isinstance(first, dict) and "error" in first:
                return f"stream creation failed: {first['error']}", False
            self.acc.process(first)
            if stop_markers:
                content, marker_hit = apply_stop_markers(self.acc.content, stop_markers)
                if marker_hit:
                    self.acc.content = content
                    LLMClient.close_stream(self._raw)
                    self.stopped_at_marker = True
                    stopped = True
            if not stopped:
                for chunk in self._raw:
                    self.acc.process(chunk)
                    if stop_check and stop_check():
                        LLMClient.close_stream(self._raw)
                        stopped = True
                        break
                    if stop_on_chunk and stop_on_chunk():
                        stopped = True
                        break
                    if stop_markers:
                        content, marker_hit = apply_stop_markers(self.acc.content, stop_markers)
                        if marker_hit:
                            self.acc.content = content
                            LLMClient.close_stream(self._raw)
                            self.stopped_at_marker = True
                            stopped = True
                            break
        except Exception as e:
            if isinstance(e, GenerationInterrupted):
                raise
            error = str(e)
        finally:
            _watch_done.set()
            LLMClient.close_stream(self._raw)
        if not error:
            self.acc.flush_prefill()
        return error, stopped


def _get_underlying_socket(response):
    network_stream = response.extensions.get("network_stream")
    if network_stream is None:
        return None
    try:
        return network_stream.get_extra_info("socket")
    except Exception:
        return None


class InProcessStream:

    def __init__(
        self,
        messages: list[dict],
        temp: float = None,
        timeout: int = None,
        tools: list[dict] = None,
        prefill: str = None,
        top_p: float = None,
        frequency_penalty: float = None,
        presence_penalty: float = None,
        max_tokens: int = None,
        min_p: float = None,
        params: GenerationParams = None,
        reasoning_effort: str = "none",
    ):
        self._closed = False
        self._response = None
        self._client = None
        self._iter = None

        temp, timeout, top_p, frequency_penalty, presence_penalty, max_tokens, min_p = LLMClient._resolve_params(
            params, temp, timeout, top_p, frequency_penalty, presence_penalty, max_tokens, min_p
        )

        messages_to_send = LLMClient._prepare_messages(messages, prefill)

        body = {
            "model": Config.MODEL_NAME,
            "messages": messages_to_send,
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": temp if temp is not None else Config.TEMP,
            "max_tokens": max_tokens if max_tokens is not None else Config.MAX_OUTPUT_TOKENS,
            "parallel_tool_calls": False,
            "reasoning_effort": reasoning_effort,
            "frequency_penalty": frequency_penalty if frequency_penalty is not None else Config.FREQUENCY_PENALTY,
            "presence_penalty": presence_penalty if presence_penalty is not None else Config.PRESENCE_PENALTY,
            "top_p": top_p if top_p is not None else Config.TOP_P,
        }
        if tools:
            body["tools"] = tools
        if min_p is not None:
            body["min_p"] = min_p

        http_timeout = httpx.Timeout(
            connect=10,
            read=timeout if timeout is not None else Config.TIMEOUT,
            write=30,
            pool=10,
        )

        url = Config.API_URL.rstrip("/") + "/chat/completions"

        try:
            self._client = httpx.Client(timeout=http_timeout)
            self._response = self._client.send(
                self._client.build_request("POST", url, json=body),
                stream=True,
            )
            self._response.raise_for_status()
            self._iter = self._parse_sse()
        except Exception as e:
            if self._client is not None:
                try:
                    self._client.close()
                except Exception:
                    pass

            def error_generator(err=str(e)):
                yield {"error": err}
            self._iter = error_generator()

    def _parse_sse(self):
        assert self._response is not None
        for line in self._response.iter_lines():
            if self._closed:
                return
            line = line.strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                return
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if "error" in obj:
                yield {"error": obj["error"]}
                return
            yield self._convert_chunk(obj)

    @staticmethod
    def _convert_chunk(obj: dict):
        usage = None
        if obj.get("usage"):
            u = obj["usage"]
            usage = SimpleNamespace(
                prompt_tokens=u.get("prompt_tokens", 0),
                completion_tokens=u.get("completion_tokens", 0),
                total_tokens=u.get("total_tokens"),
            )
        choices = []
        for choice_data in obj.get("choices") or []:
            delta_data = choice_data.get("delta") or {}
            tool_calls = []
            for tc in delta_data.get("tool_calls") or []:
                fn = tc.get("function") or {}
                tool_calls.append(SimpleNamespace(
                    index=tc.get("index", 0),
                    id=tc.get("id"),
                    function=SimpleNamespace(
                        name=fn.get("name"),
                        arguments=fn.get("arguments"),
                    ),
                ))
            delta = SimpleNamespace(
                content=delta_data.get("content"),
                reasoning_content=delta_data.get("reasoning_content"),
                tool_calls=tool_calls or None,
            )
            choices.append(SimpleNamespace(delta=delta))
        return SimpleNamespace(choices=choices, usage=usage)

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration
        try:
            return next(self._iter)
        except StopIteration:
            raise
        except Exception as e:
            if self._closed:
                raise StopIteration
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._response is not None:
            sock = _get_underlying_socket(self._response)
            if sock:
                sock.settimeout(1.0)
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            try:
                self._response.close()
            except Exception:
                pass
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

class LLMClient:
    @classmethod
    def close_stream(cls, stream) -> None:
        if stream is None:
            return
        try:
            if hasattr(stream, "close"):
                stream.close()
            elif hasattr(stream, "_response") and hasattr(stream._response, "close"):
                stream._response.close()
            elif hasattr(stream, "response") and hasattr(stream.response, "close"):
                stream.response.close()
        except Exception:
            pass

    @staticmethod
    def call(
        messages: list[dict],
        temp: float = None,
        timeout: int = None,
        tools: list[dict] = None,
        prefill: str = None,
        top_p: float = None,
        frequency_penalty: float = None,
        presence_penalty: float = None,
        max_tokens: int = None,
        params: GenerationParams = None,
        callbacks: Optional[dict] = None,
        stop_check: Optional[Callable[[], bool]] = None,
        reasoning_effort: str = "none",
        stop_markers: tuple = (),
    ):
        if stop_check and stop_check():
            return None, "stopped", None
        temp, timeout, top_p, frequency_penalty, presence_penalty, max_tokens, min_p = LLMClient._resolve_params(
            params, temp, timeout, top_p, frequency_penalty, presence_penalty, max_tokens
        )

        cb = callbacks or {}
        session = StreamSession(
            messages, temp=temp, timeout=timeout, tools=tools,
            prefill=prefill, top_p=top_p, frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty, max_tokens=max_tokens, min_p=min_p,
            reasoning_effort=reasoning_effort,
            on_stream_chunk=cb.get("on_stream_chunk"),
            on_reasoning_start=cb.get("on_reasoning_start"),
            on_reasoning_chunk=cb.get("on_reasoning_chunk"),
        )
        error, _stopped = session.consume(
            stop_check=stop_check,
            on_stream_start=cb.get("on_stream_start"),
            stop_markers=stop_markers,
        )
        end_cb = cb.get("on_stream_end")
        reasoning_end_cb = cb.get("on_reasoning_end")
        if error:
            if end_cb:
                end_cb()
            if session.acc.reasoning_started and reasoning_end_cb:
                reasoning_end_cb()
            if stop_check and stop_check():
                return None, "stopped: stream unavailable and stop requested", None
            return None, error, None
        if end_cb:
            end_cb()
        if session.acc.reasoning_started and reasoning_end_cb:
            reasoning_end_cb()
        message = session.acc.build_message(prefill)
        message.stopped_at_marker = session.stopped_at_marker
        if Config.DEBUG_LLM_CALLS:
            LLMClient._debug_log(LLMClient._prepare_messages(messages, prefill),
                                 (message, None, session.acc.usage))
        return message, None, session.acc.usage

    @staticmethod
    def open_stream(
        messages: list[dict],
        temp: float = None,
        timeout: int = None,
        tools: list[dict] = None,
        prefill: str = None,
        top_p: float = None,
        frequency_penalty: float = None,
        presence_penalty: float = None,
        max_tokens: int = None,
        min_p: float = None,
        params: GenerationParams = None,
        reasoning_effort: str = "none",
    ):
        return InProcessStream(
            messages, temp=temp, timeout=timeout, tools=tools, prefill=prefill,
            top_p=top_p, frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty, max_tokens=max_tokens, min_p=min_p,
            params=params, reasoning_effort=reasoning_effort,
        )

    @staticmethod
    def _prepare_messages(messages: list[dict], prefill: Optional[str]) -> list[dict]:
        messages_to_send = list(messages)
        if prefill:
            messages_to_send.append({"role": "assistant", "content": prefill})
        return messages_to_send

    @staticmethod
    def _debug_log(messages_to_send, result):
        try:
            print("\n" + "=" * 30)
            print("📤 LLM DEBUG CALL INPUT:")

            print(messages_to_send[-1])

            print("📤 LLM DEBUG CALL OUTPUT:")

            msg, err, usage = result
            if err:
                print(f"  ⚠️ ERROR: {err}")
            else:
                if msg is not None and getattr(msg, "content", None):
                    print(msg.content)
                elif msg is not None and getattr(msg, "tool_calls", None):
                    for tc in msg.tool_calls:
                        print(f"  🔨 {getattr(tc, 'name', '?')}({getattr(tc, 'arguments', '')})")
                else:
                    print("  (no content / empty)")
                if usage:
                    print(f"  ⏱ usage: {usage}")
            print("=" * 30)
        except Exception:
            pass

    @staticmethod
    def _resolve_params(params, temp, timeout, top_p, frequency_penalty, presence_penalty, max_tokens,
                        min_p=None):
        if params is not None:
            p = params.resolved()
            temp = p.temp if temp is None else temp
            timeout = p.timeout if timeout is None else timeout
            top_p = p.top_p if top_p is None else top_p
            frequency_penalty = p.frequency_penalty if frequency_penalty is None else frequency_penalty
            presence_penalty = p.presence_penalty if presence_penalty is None else presence_penalty
            max_tokens = p.max_tokens if max_tokens is None else max_tokens
            min_p = p.min_p if min_p is None else min_p
        return temp, timeout, top_p, frequency_penalty, presence_penalty, max_tokens, min_p
