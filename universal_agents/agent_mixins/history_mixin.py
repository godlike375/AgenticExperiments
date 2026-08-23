"""Mixin восстановления истории LLMAgent: стирание сломанных/неудачных сообщений."""

from __future__ import annotations

from typing import Optional

from universal_agents.models import AssistantMessage, ToolResult


class HistoryMixin:
    """Операции по удалению/восстановлению сообщений в истории."""

    def _erase_last_assistant(self) -> None:
        """Удаляет последнее сообщение ассистента из истории (сломанный вызов без tool_calls)."""
        msgs = self.history.get_all()
        if msgs and isinstance(msgs[-1], AssistantMessage):
            self.history.remove_at({len(msgs) - 1})
            self.history.normalize()
            self._on_history_changed()

    def _erase_last_failed_tool_call(self) -> int:
        """Удаляет из истории последний неудачный вызов инструмента (assistant + его error result)."""
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
        """Текст последнего текстового ответа ассистента из истории."""
        for msg in reversed(self.history.get_all()):
            if isinstance(msg, AssistantMessage) and (msg.content or "").strip():
                return msg.content.strip()
        return None
