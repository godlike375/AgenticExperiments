"""Mixin потоковой генерации LLMAgent: сбор чанков, отслеживание расхождения, достройка."""

from __future__ import annotations

import threading

from universal_agents.config import Config
from universal_agents.generation import GenerationParams
from universal_agents.llm_client import LLMClient, StreamAccumulator
from universal_agents.tool_parsing import build_tool_calls
from universal_agents.exceptions import GenerationInterrupted


class StreamingMixin:
    """Реализует streaming-вызов LLM с watch-механизмом против галлюцинаций на высокой температуре."""

    @staticmethod
    def _watch_diverged(watch_prefix: str, prefill: str, full_content: str) -> bool:
        """True, если накопленный текст перестал совпадать с началом прежнего ответа."""
        return bool(watch_prefix) and not watch_prefix.startswith((prefill or "") + full_content)

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
        previous_response_id: str = None,
        params: GenerationParams = None,
        watch_prefix: str = None,
        watch_continue_temp: float = None,
        stop_check: Callable[[], bool] = None,
    ) -> tuple:
        """Вызов LLM со streaming (возвращает (message_obj, error, usage)). Если задан watch_prefix, при расхождении с прежним ответом генерация на горячей температуре прерывается и достраивается спокойной температурой (watch_continue_temp) — буст не успевает вызвать галлюцинации. stop_check — вызывается после каждого чанка; True прерывает стрим."""
        try:
            stream = LLMClient.stream(
                messages,
                tools=tools,
                prefill=prefill,
                previous_response_id=previous_response_id,
                params=params,
            )

            # Watchdog: закрывает соединение при остановке пользователя, в т.ч. во время
            # префилла (до первого чанка), когда stop_check в цикле ниже ещё не сработал.
            LLMClient._active_stream = stream
            _watch_done = threading.Event()
            if stop_check is not None:
                def _watcher():
                    while not _watch_done.is_set():
                        if stop_check():
                            LLMClient.cancel_active()
                            break
                        _watch_done.wait(0.05)
                threading.Thread(target=_watcher, daemon=True).start()

            try:
                acc = StreamAccumulator(
                    prefill=prefill,
                    on_stream_chunk=self.on_stream_chunk,
                    on_reasoning_start=self.on_reasoning_start,
                    on_reasoning_chunk=self.on_reasoning_chunk,
                )

                first_chunk = next(stream)
                if isinstance(first_chunk, dict) and "error" in first_chunk:
                    return None, first_chunk["error"], None

                if self.on_stream_start:
                    self.on_stream_start()

                acc.process(first_chunk)
                diverged = self._watch_diverged(watch_prefix, prefill, acc.content)

                if not diverged:
                    for chunk in stream:
                        if acc.process(chunk) and self._watch_diverged(watch_prefix, prefill, acc.content):
                            diverged = True
                            break
                        if stop_check and stop_check():
                            break

                if self.on_stream_end:
                    self.on_stream_end()
                if acc.reasoning_started and self.on_reasoning_end:
                    self.on_reasoning_end()

                if diverged:
                    return self._continue_stream_after_divergence(
                        messages, tools, prefill, acc, watch_continue_temp
                    )

                message_obj = self._assemble_assistant_message(
                    acc.content,
                    build_tool_calls(acc.tool_calls_data),
                    acc.reasoning,
                    prefill=prefill,
                    streamed=True,
                )

                return message_obj, None, acc.usage

            except Exception as e:
                if isinstance(e, GenerationInterrupted):
                    raise
                if stop_check and stop_check():
                    raise GenerationInterrupted()
                return None, str(e), None
            finally:
                _watch_done.set()
                if LLMClient._active_stream is stream:
                    LLMClient._active_stream = None

        except Exception as e:
            if isinstance(e, GenerationInterrupted):
                raise
            return None, str(e), None

    def _continue_stream_after_divergence(
        self,
        messages: list[dict],
        tools: list[dict],
        prefill: str,
        acc: StreamAccumulator,
        watch_continue_temp: float,
    ) -> tuple:
        """Достраивает прерванный на расхождении ответ спокойной генерацией (тоже со стримингом)."""
        partial_text = (prefill or "") + acc.content
        calm_temp = watch_continue_temp if watch_continue_temp is not None else Config.DUPLICATE_CONTINUATION_TEMP
        calm_params = self._gen_params.with_temp(calm_temp)
        followup, ferr, fusage = LLMClient.call(
            messages,
            tools=tools,
            prefill=partial_text,
            params=calm_params,
            callbacks=self._stream_callbacks(),
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
            return msg_obj, None, acc.usage

        followup_reasoning = getattr(followup, 'reasoning_content', None) or ""
        message_obj = self._assemble_assistant_message(
            partial_text + (followup.content or ""),
            tool_calls,
            acc.reasoning + followup_reasoning,
            prefill=prefill,
            streamed=True,
        )
        return message_obj, None, fusage or acc.usage
