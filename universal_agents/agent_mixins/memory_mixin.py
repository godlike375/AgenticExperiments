
from __future__ import annotations

from universal_agents.config import Config
from universal_agents.compressors import SummaryService
from universal_agents.models import UserMessage, AssistantMessage, ToolResult
from universal_agents.task_tracker import compact_completed_tasks


class MemoryMixin:
    """Рабочая память и компакция: session summary, живой хвост, архив оригиналов."""

    def _summarize_assistant_message(self, msg: AssistantMessage) -> None:
        SummaryService.summarize_if_long(self, msg)

    def _maybe_summarize_user_message(self, msg: UserMessage) -> None:
        SummaryService.summarize_if_long(self, msg)

    def _maybe_summarize_tool_result(self, tr: ToolResult) -> None:
        SummaryService.summarize_if_long(self, tr)

    def _get_context_usage_percent(self) -> float:
        total = self.token_tracker.get_total_context_tokens()
        return (total / self.token_tracker.max_context_tokens) * 100

    def _compact_completed_tasks(self) -> int:
        return compact_completed_tasks(self)


    def _current_summary_threshold(self) -> float:
        return float(Config.AUTO_SUMMARY_THRESHOLD)


    def _auto_summarize_dialogue(self, force: bool = False) -> bool:
        """Компакция: сегмент → архив, вместо него session summary; force — на любой границе."""
        return SummaryService.compact_segment(self, force=force)
