"""Mixin восстановления истории LLMAgent: стирание сломанных/неудачных сообщений."""

from __future__ import annotations

from typing import Optional

from universal_agents.models import AssistantMessage, ToolResult, UserMessage
from universal_agents.llm_client import text_hash
from universal_agents.tool_parsing import tc_name, tc_args


class HistoryMixin:
    """Операции по удалению/восстановлению сообщений в истории."""

    def _last_failed_tool_signature(self) -> Optional[str]:
        """Сигнатура последней серии ошибок инструментов из истории: инструмент + нормализованные
        аргументы + хэш текста ошибки для каждого упавшего вызова. Используется детектором
        зацикливания (§1): только при ПОЛНОМ совпадении серий (инструменты, аргументы и ошибки)
        ошибки считаются «одинаковыми» — модель реально застряла, а не диагностирует.

        Возвращает None, если в конце истории нет неудачных вызовов (например, пустой ответ)."""
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

    def _mark_confirmation_junk(self, obj) -> None:
        """Помечает сообщение «мусором» подтверждения: неверные попытки до успешного
        answer_to_system, вычищаются скрабом (§1.11). id→obj вместо WeakSet: Message
        неhashable, а сильная ссылка не даёт переиспользовать id — identity не сработает ложно."""
        self._confirmation_junk[id(obj)] = obj

    def _scrub_confirmation_trail(self) -> int:
        """После успешного answer_to_system убирает наги guard'а и помеченный мусор —
        в контексте остаётся только правильный путь подтверждения. Возвращает число удалённых."""
        removed: set[int] = set()
        for i, m in enumerate(self.history.get_all()):
            if id(m) in self._confirmation_junk and self._confirmation_junk[id(m)] is m:
                removed.add(i)
            elif isinstance(m, UserMessage) and m._is_guard_nag:
                removed.add(i)
        self._confirmation_junk.clear()
        if not removed:
            return 0
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
