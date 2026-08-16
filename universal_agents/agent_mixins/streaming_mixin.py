"""Mixin потоковой генерации LLMAgent: сбор чанков, отслеживание расхождения, достройка."""

from __future__ import annotations

from universal_agents.config import Config
from universal_agents.generation import GenerationParams
from universal_agents.llm_client import LLMClient, apply_prefill, build_usage_dict
from universal_agents.models import AssistantMessage
from universal_agents.tool_parsing import build_tool_calls


class StreamingMixin:
    """Реализует streaming-вызов LLM с watch-механизмом против галлюцинаций на высокой температуре."""

    @staticmethod
    def _watch_diverged(watch_prefix: str, prefill: str, full_content: str) -> bool:
        """True, если накопленный текст перестал совпадать с началом прежнего ответа."""
        return bool(watch_prefix) and not watch_prefix.startswith((prefill or "") + full_content)

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

            tool_calls_data: dict = {}

            stream_state = {
                "full_content": "",
                "full_reasoning": "",
                "usage": None,
                "reasoning_started": False,
                "prefill_pending": prefill,
            }

            first_chunk = next(stream)
            if isinstance(first_chunk, dict) and "error" in first_chunk:
                return None, first_chunk["error"], None

            if self.on_stream_start:
                self.on_stream_start()

            diverged = False

            self._process_stream_chunk(first_chunk, tool_calls_data)
            self._apply_stream_chunk(first_chunk, stream_state)

            if self._watch_diverged(watch_prefix, prefill, stream_state["full_content"]):
                diverged = True

            if not diverged:
                for chunk in stream:
                    self._process_stream_chunk(chunk, tool_calls_data)
                    added = self._apply_stream_chunk(chunk, stream_state)
                    if added and self._watch_diverged(watch_prefix, prefill, stream_state["full_content"]):
                        diverged = True
                        break

            if self.on_stream_end:
                self.on_stream_end()
            if stream_state["reasoning_started"] and self.on_reasoning_end:
                self.on_reasoning_end()

            if diverged:
                return self._continue_stream_after_divergence(
                    messages, tools, prefill, stream_state, tool_calls_data, watch_continue_temp
                )

            tool_calls = build_tool_calls(tool_calls_data)

            message_obj = AssistantMessage(
                content=stream_state["full_content"],
                tool_calls=tool_calls,
                reasoning_content=stream_state["full_reasoning"],
                streamed=True,
            )

            message_obj.content = apply_prefill(message_obj.content, prefill)

            return message_obj, None, stream_state["usage"]

        except Exception as e:
            return None, str(e), None

    def _continue_stream_after_divergence(
        self,
        messages: list[dict],
        tools: list[dict],
        prefill: str,
        stream_state: dict,
        tool_calls_data: dict,
        watch_continue_temp: float,
    ) -> tuple:
        """Достраивает прерванный на расхождении ответ спокойной генерацией."""
        partial_text = (prefill or "") + stream_state["full_content"]
        calm_temp = watch_continue_temp if watch_continue_temp is not None else Config.DUPLICATE_CONTINUATION_TEMP
        calm_params = self._gen_params.with_temp(calm_temp)
        followup, ferr, fusage = LLMClient.call(
            messages,
            tools=tools,
            prefill=partial_text,
            params=calm_params,
        )
        tool_calls = build_tool_calls(tool_calls_data)

        if ferr or not followup:
            msg_obj = AssistantMessage(
                content=partial_text,
                tool_calls=tool_calls,
                reasoning_content=stream_state["full_reasoning"],
                streamed=True,
            )
            msg_obj.content = apply_prefill(msg_obj.content, prefill)
            return msg_obj, None, stream_state["usage"]

        followup_reasoning = getattr(followup, 'reasoning_content', None) or ""
        message_obj = AssistantMessage(
            content=partial_text + (followup.content or ""),
            tool_calls=tool_calls,
            reasoning_content=stream_state["full_reasoning"] + followup_reasoning,
            streamed=True,
        )
        message_obj.content = apply_prefill(message_obj.content, prefill)
        return message_obj, None, fusage or stream_state["usage"]

    def _apply_stream_chunk(self, chunk, state: dict) -> str:
        """Применяет чанк стрима: usage, reasoning и текстовый контент.
        Возвращает добавленный текстовый delta."""
        usage = getattr(chunk, 'usage', None)
        if usage:
            state["usage"] = build_usage_dict(
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens,
            )
        if not getattr(chunk, 'choices', None):
            return ""
        delta = chunk.choices[0].delta
        rc = getattr(delta, 'reasoning_content', None)
        if rc:
            state["full_reasoning"] += rc
            if not state["reasoning_started"]:
                state["reasoning_started"] = True
                if self.on_reasoning_start:
                    self.on_reasoning_start()
            if self.on_reasoning_chunk:
                self.on_reasoning_chunk(rc)
        added = ""
        if delta.content:
            if state.get("prefill_pending"):
                if self.on_stream_chunk:
                    self.on_stream_chunk(state["prefill_pending"])
                state["prefill_pending"] = None
            added = delta.content
            state["full_content"] += added
            if self.on_stream_chunk:
                self.on_stream_chunk(added)
        return added

    def _process_stream_chunk(self, chunk, tool_calls_data: dict):
        """Обработка чанка для сбора tool calls"""
        if not chunk.choices:
            return

        delta = chunk.choices[0].delta
        if not delta.tool_calls:
            return

        for tc in delta.tool_calls:
            idx = tc.index
            if idx not in tool_calls_data:
                tool_calls_data[idx] = {
                    "id": tc.id or "",
                    "type": "function",
                    "function": {
                        "name": tc.function.name if tc.function and tc.function.name else "",
                        "arguments": tc.function.arguments if tc.function and tc.function.arguments else ""
                    }
                }
            else:
                if tc.id:
                    tool_calls_data[idx]["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        tool_calls_data[idx]["function"]["name"] += tc.function.name
                    if tc.function.arguments:
                        tool_calls_data[idx]["function"]["arguments"] += tc.function.arguments
