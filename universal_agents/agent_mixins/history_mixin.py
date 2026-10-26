
from __future__ import annotations

from typing import Optional

from universal_agents.models import AssistantMessage, ToolResult, UserMessage
from universal_agents.llm_client import text_hash
from universal_agents.tool_parsing import tc_name, tc_args


class HistoryMixin:
    """Операции по удалению/восстановлению сообщений в истории."""

    def _last_failed_tool_signature(self) -> Optional[str]:
        """Сигнатура последней серии ошибок: инструмент+аргументы+хэш ошибки (§1.10)."""
        msgs = self.history.get_all()
        errors: list[ToolResult] = []
        assistant = None
        for i in range(len(msgs) - 1, -1, -1):
            m = msgs[i]
            if isinstance(m, ToolResult) and m.is_error and not m.is_user_denied:
                errors.append(m)
                continue
            if isinstance(m, AssistantMessage) and m.has_tool_calls():
                assistant = m
                break
            break
        if assistant is None:
            return None
        error_by_id = {tr.tool_call_id: tr.content for tr in errors}
        parts = []
        for tc in assistant.tool_calls:
            norm_args = self.loop_detector.normalize_args(tc_args(tc))
            err_content = error_by_id.get(tc.id, "")
            parts.append(f"{tc_name(tc)}|{norm_args}|{text_hash(err_content)}")
        return ";".join(parts) if parts else None

    def _erase_last_assistant(self) -> None:
        msgs = self.history.get_all()
        if msgs and isinstance(msgs[-1], AssistantMessage):
            self.history.remove_at({len(msgs) - 1})
            self.history.normalize()
            self._on_history_changed()

    def _erase_last_failed_tool_call(self) -> int:
        msgs = self.history.get_all()
        removed: set[int] = set()
        for i in range(len(msgs) - 1, -1, -1):
            m = msgs[i]
            if isinstance(m, ToolResult) and m.is_error and not m.is_user_denied:
                removed.add(i)
            elif isinstance(m, AssistantMessage) and m.has_tool_calls():
                removed.add(i)
                break
            else:
                break
        if removed:
            self.history.remove_at(removed)
            self.history.normalize()
            self._on_history_changed()
        return len(removed)

    def _get_last_answer_text(self) -> Optional[str]:
        for msg in reversed(self.history.get_all()):
            if isinstance(msg, AssistantMessage) and (msg.content or "").strip():
                return msg.content.strip()
        return None
