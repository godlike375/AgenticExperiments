"""Mixin обработки ответа LLM: построение AssistantMessage, добавление в историю, рендер."""

from __future__ import annotations

from universal_agents.llm_client import apply_prefill
from universal_agents.models import AssistantMessage, ToolCall, ToolResult
from universal_agents.tool_parsing import tc_name, tc_args, detect_broken_call, args_are_valid


class ResponseMixin:
    """Преобразует сырой ответ LLM в сообщение истории и управляет его добавлением/рендером."""

    def _assemble_assistant_message(
        self,
        content: str,
        tool_calls: list[ToolCall] = None,
        reasoning_content: str = "",
        prefill: str = None,
        streamed: bool = False,
    ) -> AssistantMessage:
        """Единая точка сборки AssistantMessage (инвариант §2): строит сообщение и применяет prefill. Раньше дублировалась в 4 местах."""
        message_obj = AssistantMessage(
            content=content or "",
            tool_calls=list(tool_calls or []),
            reasoning_content=reasoning_content or "",
            streamed=streamed,
        )
        message_obj.content = apply_prefill(message_obj.content, prefill)
        return message_obj

    def _build_assistant_msg(self, msg_obj, clean_content: str) -> AssistantMessage:
        tool_calls = []
        if msg_obj.tool_calls:
            for tc in msg_obj.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc_name(tc),
                    arguments=tc_args(tc),
                ))
        return self._assemble_assistant_message(
            clean_content,
            tool_calls,
            getattr(msg_obj, 'reasoning_content', ''),
            streamed=bool(getattr(msg_obj, 'streamed', False) or getattr(msg_obj, '_streamed', False)),
        )

    def _emit_token_info(self):
        parts = []
        if self.token_tracker.last_usage:
            parts.append(self.token_tracker.format_user_token_info())
        if self._tool_usage:
            parts.append(self._format_tool_stats())
        if parts:
            self.on_system_msg(" | ".join(parts))

    def _format_tool_stats(self) -> str:
        total = sum(self._tool_usage.values())
        items = " · ".join(f"{name} ×{count}" for name, count in sorted(self._tool_usage.items(), key=lambda x: -x[1]))
        return f"Tools: {items} ({total} total)"

    def _append_assistant(self, msg: AssistantMessage) -> None:
        """Добавляет сообщение ассистента в историю и рендерит его."""
        self.history.add(msg)
        if self._per_msg_enabled:
            self._summarize_assistant_message(msg)
        self.on_render(msg)
        self._emit_token_info()

    def _append_tool_results(self, results: list[ToolResult]) -> None:
        """Добавляет результаты инструментов в историю и рендерит их."""
        for tr in results:
            self.history.add(tr)
            if self._per_msg_enabled:
                self._maybe_summarize_tool_result(tr)
            self.on_render(tr)
            self._emit_token_info()

    def _process_llm_response(self, message_obj) -> tuple[str, bool, bool]:
        if not message_obj:
            return "Empty response", True, False

        content = message_obj.content or ""
        clean_content = content.strip()
        assistant_msg = self._build_assistant_msg(message_obj, clean_content)

        if assistant_msg.has_tool_calls():
            valid_tc = None
            fallback_tc = assistant_msg.tool_calls[0]

            for tc in assistant_msg.tool_calls:
                if tc.name in self._all_tools and args_are_valid(tc.arguments):
                    valid_tc = tc
                    break

            chosen_tc = valid_tc if valid_tc else fallback_tc

            if len(assistant_msg.tool_calls) > 1:
                self.on_system_msg(f"[MULTIPLE TOOLS DETECTED] Kept only '{chosen_tc.name}', removed others.")
                assistant_msg.tool_calls = [chosen_tc]
                if message_obj.tool_calls:
                    message_obj.tool_calls = [tc for tc in message_obj.tool_calls if tc.id == chosen_tc.id]

        if not clean_content and not assistant_msg.has_tool_calls():
            self.on_system_msg("[EMPTY RESPONSE] Model returned no content. Discarding and retrying...")
            return clean_content, True, False

        self._append_assistant(assistant_msg)

        if not assistant_msg.has_tool_calls():
            if detect_broken_call(clean_content, self._known_tool_names()):
                self.on_system_msg("[BROKEN CALL] Response looks like an unparsed tool call (prose or XML).")
                return clean_content, False, True
            return clean_content, False, False

        tool_results = self._execute_tools(assistant_msg.tool_calls)
        self._append_tool_results(tool_results)

        removed = self.history.remove_failed_call_chains()
        if removed:
            self.on_system_msg(
                f"[CLEANUP] Removed {removed} messages "
                f"({removed // 2} failed calls)"
            )
            self.history.normalize()
            self._on_history_changed()

        tool_error_occurred = any(tr.is_error and not tr.is_user_denied for tr in tool_results)
        return clean_content, tool_error_occurred, False
