"""Mixin рабочей памяти и компакции LLMAgent.

Слои памяти:
   1) session summary — обычное user-сообщение сразу после system prompt.
      Каждая компакция ПЕРЕПИСЫВАЕТ заметки с нуля по схеме 7 разделов
      (compressors.summarize_history_plain + SECTION_GUIDE); при повторной
      компакции старые заметки подаются модели как контекст для слияния;
   2) живой хвост истории;
  3) архив полных оригиналов вне контекста (agent.archive), доступный модели
     через recall_search / recall_read.

Спарковано (код сохранён, не удаляется):
  - XML-схема STATE-блока (memory_blocks.py + regenerate_state_block);
  - per-message плотные саммари в рабочую память [PER_MSG_SUMMARIES_ENABLED].
"""

from __future__ import annotations

from universal_agents.config import Config, CHARS_PER_TOKEN, MIN_TOKENS_TO_SUMMARIZE
from universal_agents.constants import (
    SUMMARY_PREFIX_USER,
    SUMMARY_PREFIX_AI,
    SUMMARY_PREFIX_TOOL_NAMED,
)
from universal_agents.compressors import (
    _dense_summarize_message,
    summarize_history_plain,
)
from universal_agents.models import UserMessage, AssistantMessage, ToolResult
from universal_agents.task_tracker import compact_completed_tasks


class MemoryMixin:
    """Управляет рабочей памятью и сжатием диалога в session summary.

    Компакция: вытесняемый сегмент архивируется и заменяется одним
    UserMessage-саммари сразу после system prompt. Повторные компакции не
    переписывают саммари с нуля, а правят его точечными заменами текста."""

    def _summarize_assistant_message(self, msg: AssistantMessage) -> None:
        """Плотное саммари последнего сообщения ассистента в рабочую память
        (не попадает в контекст). Только для сообщений длиннее порога."""
        if not Config.PER_MSG_SUMMARIES_ENABLED:
            return
        content = (msg.content or "").strip()
        MIN_CHARS = int(MIN_TOKENS_TO_SUMMARIZE * CHARS_PER_TOKEN)
        if len(content) < MIN_CHARS:
            return
        summary = _dense_summarize_message(self, content)
        stored = f"{SUMMARY_PREFIX_AI} {summary}" if summary else None
        if stored and len(stored) < len(content):
            self.history.set_per_msg_summary(msg, stored)
            self.on_system_msg(
                f"[PER-MSG SUMMARY] Assistant message ({len(content)} chars) "
                f"summarized into working memory ({len(summary)} chars)."
            )

    def _maybe_summarize_user_message(self, msg: UserMessage) -> None:
        """Плотное саммари длинного сообщения пользователя в рабочую память
        (не попадает в контекст). Только для сообщений длиннее порога."""
        if not Config.PER_MSG_SUMMARIES_ENABLED:
            return
        content = (msg.content or "").strip()
        MIN_CHARS = int(MIN_TOKENS_TO_SUMMARIZE * CHARS_PER_TOKEN)
        if len(content) < MIN_CHARS:
            return
        summary = _dense_summarize_message(self, content)
        stored = f"{SUMMARY_PREFIX_USER} {summary}" if summary else None
        if stored and len(stored) < len(content):
            self.history.set_per_msg_summary(msg, stored)
            self.on_system_msg(
                f"[PER-MSG SUMMARY] User message ({len(content)} chars) "
                f"summarized into working memory ({len(summary)} chars)."
            )

    def _maybe_summarize_tool_result(self, tr: ToolResult) -> None:
        """Длинные выводы инструментов (> MIN_TOKENS_TO_SUMMARIZE) сразу
        суммаризируются в рабочую память; при сжатии саммари встанет на их место."""
        if not Config.PER_MSG_SUMMARIES_ENABLED:
            return
        if tr.is_error or tr.is_user_denied:
            return
        # Уже сжатый результат (например DIGEST из выемки больших файлов) не
        # суммаризируем повторно — не делаем суммаризацию суммаризации.
        if getattr(tr, "skip_summarize", False):
            return
        content = (tr.content or "").strip()
        MIN_CHARS = int(MIN_TOKENS_TO_SUMMARIZE * CHARS_PER_TOKEN)
        if len(content) < MIN_CHARS:
            return
        summary = _dense_summarize_message(self, content)
        stored = SUMMARY_PREFIX_TOOL_NAMED.format(name=tr.name) + f" {summary}" if summary else None
        if stored and len(stored) < len(content):
            # Для read заменяем оригинальный контент на суммаризацию прямо в
            # результате инструмента: в контексте и истории остаётся суммаризация,
            # а не исходный файл (экономия памяти).
            if tr.name == "read":
                tr.content = summary
                self.on_system_msg(
                    f"[READ SUMMARIZED] read result replaced with summary "
                    f"({len(content)} -> {len(summary)} chars)."
                )
                return
            self.history.set_per_msg_summary(tr, stored)
            self.on_system_msg(
                f"[PER-MSG SUMMARY] Tool '{tr.name}' output ({len(content)} chars) "
                f"summarized into working memory ({len(summary)} chars)."
            )

    def _prune_per_msg_summaries(self) -> None:
        """Убирает из рабочей памяти саммари сообщений, которых больше нет в истории."""
        self.history.prune_per_msg_summaries()

    def _get_context_usage_percent(self) -> float:
        """Процент заполнения контекста по фактическому расходу из API (тот же
        источник, что и заголовок "Tokens spent / Remaining")."""
        total = self.token_tracker.get_total_context_tokens()
        return (total / self.token_tracker.max_context_tokens) * 100

    def _compact_completed_tasks(self) -> None:
        """Структурная компактизация истории: сжимает завершённые подзадачи
        (размеченных через инструмент have_done). Ничего не делает, если завершённых
        задач нет. Применяется перед грубой суммаризацией по порогу токенов."""
        compact_completed_tasks(self)

    # ------------------------------------------------------------------
    # Порог срабатывания
    # ------------------------------------------------------------------

    def _current_summary_threshold(self) -> float:
        """Порог авто-компакции в процентах заполнения контекста."""
        return float(Config.AUTO_SUMMARY_THRESHOLD)

    # ------------------------------------------------------------------
    # Авто-компакция диалога
    # ------------------------------------------------------------------

    def _auto_summarize_dialogue(self) -> None:
        """Компакция: архивируемый сегмент уходит в архив, а вместо него
        остаётся session summary — UserMessage сразу после system prompt.

        Первая компакция ПИШЕТ заметки с нуля (промпт 7 разделов); повторные
        РЕДАКТИРУЮТ их по точному совпадению текста: блоки SEARCH/REPLACE от
        модели применяются детерминированным парсером. Если служебный вызов
        не удался, история НЕ трогается."""
        preserve_last = Config.AUTO_SUMMARY_PRESERVE_LAST

        # --- Точка срабатывания: только на «безопасных» границах ---
        popped_calls = self.history.pop_pending_tool_calls()

        last = self.history._messages[-1] if self.history._messages else None
        if not (isinstance(last, ToolResult) or isinstance(last, UserMessage)):
            return

        if popped_calls:
            self.on_system_msg(
                f"[AUTO-SUMMARY] Removed {popped_calls} pending tool call(s) before "
                f"summarizing; assistant will re-invoke after compression."
            )

        messages = self.history.get_all()
        start_id = Config.AFTER_SYSTEM_PROMPT
        end_id = len(messages) - 1 - preserve_last
        if start_id > end_id:
            return

        original_len = self.history.content_len(start_id, end_id)

        # --- 1. Архивация оригиналов ДО удаления (recall остаётся возможен) ---
        if Config.MEMORY_ARCHIVE_ENABLED and hasattr(self, "archive"):
            added = self.archive.append_messages(messages[start_id:end_id + 1])
            self.on_system_msg(f"[ARCHIVE] Stored {added} original message(s) for recall.")

        # --- 2. Один вызов LLM: переписываем заметки с нуля.
        # Подаём ПОЛНУЮ историю как структурированные сообщения API (включая
        # system prompt первым сообщением) — префикс совпадает с реальным
        # диалогом, поэтому KV-cache переиспользуется. Инструкция суммаризации
        # (SECTION_GUIDE) добавляется последним сообщением в compressors.
        history_msgs = [m.to_api_dict() for m in messages]
        summary_text = summarize_history_plain(
            self, history_msgs,
        )
        if not summary_text:
            self.on_system_msg(
                "[AUTO-SUMMARY] Compression call failed; history left unchanged."
            )
            return

        # --- 3. Удаляем сегмент из истории, оставляя текст как контекст ---
        self.history.compress_old_messages(summary_text, preserve_last)

        self._on_history_changed()
        self._prune_per_msg_summaries()

        new_len = self.history.content_len(1, len(self.history) - 1)
        final_reduction = 1.0 - (new_len / max(original_len, 1))
        self.on_system_msg(
            f"[AUTO-SUMMARY] Session summary written "
            f"(-{final_reduction:.0%}); originals archived for recall."
        )
