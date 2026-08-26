"""Mixin рабочей памяти и компакции LLMAgent: session summary (перезапись заметок по схеме 7 разделов), живой хвост истории и архив оригиналов (recall_search/read). Поддерживает per-message плотные саммари в рабочую память."""

from __future__ import annotations

from universal_agents.config import Config, CHARS_PER_TOKEN, MIN_TOKENS_TO_SUMMARIZE
from universal_agents.constants import (
    SUMMARY_PREFIX_USER,
    SUMMARY_PREFIX_AI,
    SUMMARY_PREFIX_TOOL_NAMED, ENVIRONMENT_PREFIX_END, ENVIRONMENT_PREFIX, SUMMARY_MARKER,
)
from universal_agents.compressors import (
    _dense_summarize_message,
    summarize_history_plain,
)
from universal_agents.models import UserMessage, AssistantMessage, ToolResult
from universal_agents.task_tracker import compact_completed_tasks


def _raw_summary(content: str) -> str:
    """Извлекает чистый текст саммари из содержимого summary-сообщения (убирая обёртку [[SYSTEM]] и служебный префикс), чтобы сравнивать старую и новую саммари на тождественность."""
    s = content or ""
    marker = f": [{SUMMARY_MARKER}]:"
    if marker in s:
        s = s.split(marker, 1)[1]
    s = s.replace(ENVIRONMENT_PREFIX, "").replace(ENVIRONMENT_PREFIX_END, "")
    return s.strip()


class MemoryMixin:
    """Управляет рабочей памятью и сжатием диалога в session summary: вытесняемый сегмент архивируется и заменяется одним UserMessage-саммари; повторные компакции правят его точечно."""

    def _summarize_assistant_message(self, msg: AssistantMessage) -> None:
        """Плотное саммари последнего сообщения ассистента в рабочую память (вне контекста); только для длиннее порога."""
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
        """Плотное саммари длинного сообщения пользователя в рабочую память (вне контекста); только для длиннее порога."""
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
        """Длинные выводы инструментов сразу суммаризируются в рабочую память; при сжатии саммари встанет на их место."""
        if not Config.PER_MSG_SUMMARIES_ENABLED:
            return
        if tr.is_error or tr.is_user_denied:
            return
        # Уже сжатый результат (DIGEST и т.п.) не суммаризируем повторно.
        if getattr(tr, "skip_summarize", False):
            return
        content = (tr.content or "").strip()
        MIN_CHARS = int(MIN_TOKENS_TO_SUMMARIZE * CHARS_PER_TOKEN)
        if len(content) < MIN_CHARS:
            return
        summary = _dense_summarize_message(self, content)
        stored = SUMMARY_PREFIX_TOOL_NAMED.format(name=tr.name) + f" {summary}" if summary else None
        if stored and len(stored) < len(content):
            # Для read заменяем оригинальный контент на суммаризацию прямо в результате (экономия памяти).
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
        """Процент заполнения контекста по фактическому расходу из API (как заголовок "Context size / Remaining")."""
        total = self.token_tracker.get_total_context_tokens()
        return (total / self.token_tracker.max_context_tokens) * 100

    def _compact_completed_tasks(self) -> None:
        """Структурная компактизация: сжимает завершённые подзадачи (через have_done); нет завершённых — ничего не делает. Применяется перед суммаризацией по порогу токенов."""
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
        """Компакция: сегмент уходит в архив, вместо него — session summary (UserMessage после system prompt). Первая компакция пишет заметки с нуля, повторные правят по SEARCH/REPLACE; при неудаче история не трогается."""
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

        # Один вызов LLM: переписываем заметки с нуля. Подаём полную историю как structured API-сообщения (system prompt первым) в ТОМ ЖЕ формате, что и диалог, иначе префикс разойдётся и KV-кэш сбросится.
        from universal_agents.context_builder import prepare_messages_for_api
        history_msgs = prepare_messages_for_api(self, normalize=False)
        # Несколько синхронных попыток перегенерации саммари, пока не получим непустой результат; история не трогается, пока сжатие не удалось.
        FORBID_TOOLS_MSG = (
            f"{ENVIRONMENT_PREFIX} Возможно, ты попытался вызвать инструмент, но здесь "
            f"инструменты ЗАПРЕЩЕНЫ — сгенерируй саммари ТОЛЬКО текстом, без вызовов "
            f"инструментов. Попробуй ещё раз.{ENVIRONMENT_PREFIX_END}"
        )
        # Старая саммари (для детекта повтора: новая не должна быть ей тождественна).
        prev_summary = self.history.get_session_summary()

        summary_text = None
        for attempt in range(1, Config.AUTO_SUMMARY_MAX_RETRIES + 1):
            # Лёгкий рост температуры между попытками, чтобы не повторять ту же ошибку.
            if prev_summary is not None and summary_text is not None and _raw_summary(summary_text) == _raw_summary(prev_summary):
                temp = Config.SUMMARY_DUPLICATE_TEMP
            else:
                temp = Config.TEMP + 0.2 * (attempt - 1)
            extra = FORBID_TOOLS_MSG if attempt > 1 else None
            summary_text = summarize_history_plain(
                self, history_msgs, temp=temp, extra_instruction=extra,
                include_new_to_previous=prev_summary is not None,
            )
            if summary_text:
                # Повтор саммари: модель проигнорировала новый контент и вернула старую —
                # заставляем перегенерировать (как и при других повторах), с бустом температуры.
                if prev_summary is not None and _raw_summary(summary_text) == _raw_summary(prev_summary):
                    self.on_system_msg(
                        f"[AUTO-SUMMARY] New summary is identical to the existing one "
                        f"(attempt {attempt}/{Config.AUTO_SUMMARY_MAX_RETRIES}); "
                        f"regenerating with temperature boost ({Config.SUMMARY_DUPLICATE_TEMP})."
                    )
                    continue
                break
            self.on_system_msg(
                f"[AUTO-SUMMARY] Compression call failed (attempt {attempt}/"
                f"{Config.AUTO_SUMMARY_MAX_RETRIES}); retrying..."
            )
        if not summary_text:
            self.on_system_msg(
                "[AUTO-SUMMARY] Compression call failed after "
                f"{Config.AUTO_SUMMARY_MAX_RETRIES} attempts; history left unchanged."
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
