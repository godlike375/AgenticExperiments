
from __future__ import annotations

import time
from typing import Callable

from universal_agents.config import Config
from universal_agents.generation import GenerationParams
from universal_agents.llm_client import LLMClient, StreamAccumulator, StreamSession
from universal_agents.tool_parsing import build_tool_calls
from universal_agents.exceptions import GenerationInterrupted


class StreamingMixin:

    @staticmethod
    def _watch_diverged(watch_prefix: str, prefill: str, full_content: str) -> bool:
        return bool(watch_prefix) and not watch_prefix.startswith((prefill or "") + full_content)

    @staticmethod
    def _watch_match_len(watch_prefix: str, prefill: str, full_content: str) -> int:
        text = (prefill or "") + (full_content or "")
        target = watch_prefix or ""
        matched = 0
        for left, right in zip(text, target):
            if left != right:
                break
            matched += 1
        return matched

    def _should_continue_after_divergence(
        self,
        watch_prefix: str,
        prefill: str,
        content: str,
        stopped: bool,
        stop_check: Callable[[], bool] = None,
    ) -> bool:
        """Достройка осмысленна только при частичном совпадении с таргетом (иначе — свежий ответ)."""
        if not stopped or (stop_check and stop_check()):
            return False
        if not self._watch_diverged(watch_prefix, prefill, content):
            return False
        matched = self._watch_match_len(watch_prefix, prefill, content)
        return 0 < matched < len(watch_prefix or "")

    def _stream_callbacks(self) -> dict:
        return {
            "on_stream_chunk": self.on_stream_chunk,
            "on_reasoning_start": self.on_reasoning_start,
            "on_reasoning_chunk": self.on_reasoning_chunk,
            "on_reasoning_end": self.on_reasoning_end,
        }

    def _call_with_streaming(
        self,
        messages: list[dict],
        prefill: str = None,
        tools: list[dict] = None,
        params: GenerationParams = None,
        watch_prefix: str = None,
        watch_continue_temp: float = None,
        stop_check: Callable[[], bool] = None,
        reasoning_effort: str = "none",
        stop_markers: tuple = (),
        prefill_shown: int = 0,
        defer_stream_end: bool = False,
    ) -> tuple:
        """Вызов со стримингом: prefill_shown — уже показанное; defer_stream_end — закрывает вызывающий."""
        error = ""
        _stopped = False
        retry_delays = (0, 1, 2, 5, 10)
        for attempt, delay in enumerate(retry_delays, start=1):
            if delay:
                if stop_check and stop_check():
                    raise GenerationInterrupted()
                self.on_system_msg(
                    f"[API Retry] stream creation failed, retry {attempt - 1}/{len(retry_delays) - 1} in {delay}s..."
                )
                time.sleep(delay)
            try:
                session = StreamSession(
                    messages,
                    tools=tools,
                    prefill=prefill,
                    params=params,
                    reasoning_effort=reasoning_effort,
                    on_stream_chunk=self.on_stream_chunk,
                    on_reasoning_start=self.on_reasoning_start,
                    on_reasoning_chunk=self.on_reasoning_chunk,
                    prefill_shown=prefill_shown,
                )
                error, _stopped = session.consume(
                    stop_check=stop_check,
                    stop_on_chunk=(
                        (lambda: self._watch_diverged(watch_prefix, prefill, session.acc.content))
                        if watch_prefix else None
                    ),
                    on_stream_start=None if prefill_shown else self.on_stream_start,
                    stop_markers=stop_markers,
                )
            except Exception as e:
                if isinstance(e, GenerationInterrupted):
                    raise
                if stop_check and stop_check():
                    raise GenerationInterrupted()
                return None, str(e), None, False
            if not (error and error.startswith("stream creation failed")):
                break

        if error:
            if stop_check and stop_check():
                raise GenerationInterrupted()
            return None, error, None, False

        if self.on_stream_end and not prefill_shown and not defer_stream_end:
            self.on_stream_end()
        if session.acc.reasoning_started and self.on_reasoning_end:
            self.on_reasoning_end()

        if self._should_continue_after_divergence(
            watch_prefix, prefill, session.acc.content, _stopped, stop_check,
        ):
            LLMClient.close_stream(session._raw)
            return self._continue_stream_after_divergence(
                messages, tools, prefill, session.acc, watch_continue_temp,
                reasoning_effort=reasoning_effort, stop_check=stop_check,
            )

        message_obj = self._assemble_assistant_message(
            session.acc.content,
            build_tool_calls(session.acc.tool_calls_data),
            session.acc.reasoning,
            prefill=prefill,
            streamed=True,
        )
        return message_obj, None, session.acc.usage, session.stopped_at_marker

    def _continue_stream_after_divergence(
        self,
        messages: list[dict],
        tools: list[dict],
        prefill: str,
        acc: StreamAccumulator,
        watch_continue_temp: float,
        reasoning_effort: str = "none",
        stop_check: Callable[[], bool] = None,
    ) -> tuple:
        partial_text = (prefill or "") + acc.content
        calm_temp = watch_continue_temp if watch_continue_temp is not None else Config.DUPLICATE_CONTINUATION_TEMP
        calm_params = self._gen_params.with_temp(calm_temp)
        followup, ferr, fusage = LLMClient.call(
            messages,
            tools=tools,
            prefill=partial_text,
            params=calm_params,
            callbacks=self._stream_callbacks(),
            reasoning_effort=reasoning_effort,
            stop_check=stop_check,
        )
        tool_calls = build_tool_calls(acc.tool_calls_data)

        if ferr or not followup:
            if self.on_system_msg:
                self.on_system_msg(
                    f"[llm-service] Divergence follow-up failed"
                    f"{f' ({ferr})' if ferr else ' (empty response)'}; keeping partial response."
                )
            msg_obj = self._assemble_assistant_message(
                partial_text,
                tool_calls,
                acc.reasoning,
                prefill=prefill,
                streamed=True,
            )
            return msg_obj, None, acc.usage, False

        followup_reasoning = getattr(followup, 'reasoning_content', None) or ""
        message_obj = self._assemble_assistant_message(
            partial_text + (followup.content or ""),
            tool_calls,
            acc.reasoning + followup_reasoning,
            prefill=prefill,
            streamed=True,
        )
        return message_obj, None, fusage or acc.usage, False
