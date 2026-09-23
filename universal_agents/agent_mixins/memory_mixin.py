"""Mixin рабочей памяти и компакции LLMAgent: session summary (перезапись заметок по схеме 7 разделов), живой хвост истории и архив оригиналов (recall_search/read). Поддерживает per-message плотные саммари в рабочую память."""

from __future__ import annotations

from universal_agents.config import Config
from universal_agents.compressors import SummaryService
from universal_agents.models import UserMessage, AssistantMessage, ToolResult
from universal_agents.task_tracker import compact_completed_tasks


class MemoryMixin:
    """Управляет рабочей памятью и сжатием диалога в session summary: вытесняемый сегмент архивируется и заменяется одним UserMessage-саммари; повторные компакции правят его точечно."""

    def _summarize_assistant_message(self, msg: AssistantMessage) -> None:
        """Плотное саммари последнего сообщения ассистента в рабочую память (вне контекста); только для длиннее порога."""
        SummaryService.summarize_if_long(self, msg)

    def _maybe_summarize_user_message(self, msg: UserMessage) -> None:
        """Плотное саммари длинного сообщения пользователя в рабочую память (вне контекста); только для длиннее порога."""
        SummaryService.summarize_if_long(self, msg)

    def _maybe_summarize_tool_result(self, tr: ToolResult) -> None:
        """Длинные выводы инструментов сразу суммаризируются в рабочую память; при сжатии саммари встанет на их место."""
        SummaryService.summarize_if_long(self, tr)

    def _prune_per_msg_summaries(self) -> None:
        """Убирает из рабочей памяти саммари сообщений, которых больше нет в истории."""
        self.history.prune_per_msg_summaries()

    def _get_context_usage_percent(self) -> float:
        """Процент заполнения контекста по фактическому расходу из API (как заголовок "Context size / Remaining")."""
        total = self.token_tracker.get_total_context_tokens()
        return (total / self.token_tracker.max_context_tokens) * 100

    def _compact_completed_tasks(self) -> int:
        """Структурная компактизация: сжимает завершённые подзадачи (через have_done); нет завершённых — ничего не делает. Применяется перед суммаризацией по порогу токенов. Возвращает число сжатых групп."""
        return compact_completed_tasks(self)

    # ------------------------------------------------------------------
    # Порог срабатывания
    # ------------------------------------------------------------------

    def _current_summary_threshold(self) -> float:
        """Порог авто-компакции в процентах заполнения контекста."""
        return float(Config.AUTO_SUMMARY_THRESHOLD)

    # ------------------------------------------------------------------
    # Авто-компакция диалога
    # ------------------------------------------------------------------

    def _auto_summarize_dialogue(self, force: bool = False) -> bool:
        """Компакция: сегмент уходит в архив, вместо него — session summary (UserMessage после system prompt).

        force=True — принудительная компакция (команда /compact_history): сжимает даже на ассистентской
        границе и сообщает о причинах пропуска. Возвращает True, если история реально сжата."""
        return SummaryService.compact_segment(self, force=force)
