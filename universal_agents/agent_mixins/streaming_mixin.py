"""Mixin потоковой генерации LLMAgent: сбор чанков, отслеживание расхождения, достройка."""

from __future__ import annotations

from universal_agents.config import Config
from universal_agents.generation import GenerationParams
from universal_agents.llm_client import LLMClient, StreamAccumulator
from universal_agents.tool_parsing import build_tool_calls


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
    ) -> tuple:
        """
        Вызов LLM с streaming для текста.
        Возвращает (message_obj, error, usage) как обычный LLMClient.call().

        watch_prefix: если задан, накопленный текст сверяется с ним по префиксу.
        Как только модель перестаёт повторять прежний ответ (расхождение), генерация
        на горячей температуре прерывается, и ответ достраивается отдельным запросом
        на спокойной температуре (watch_continue_temp) начиная с точки расхождения —
        так высокий буст не успевает спровоцировать галлюцинации.
        """
        try:
            stream = LLMClient.stream(
                messages,
                tools=tools,
                prefill=prefill,
                previous_response_id=previous_response_id,
                params=params,
            )

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
