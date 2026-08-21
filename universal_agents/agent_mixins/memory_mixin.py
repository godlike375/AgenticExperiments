"""Mixin рабочей памяти и суммаризации LLMAgent (per-message summaries + авто-суммаризация диалога)."""

from __future__ import annotations

from universal_agents.config import Config, CHARS_PER_TOKEN, MIN_TOKENS_TO_SUMMARIZE
from universal_agents.compressors import summarize_dialogue, _dense_summarize_message
from universal_agents.models import UserMessage, AssistantMessage, ToolResult
from universal_agents.task_tracker import compact_completed_tasks


class MemoryMixin:
    """Управляет рабочей памятью (плотные саммари сообщений) и сжатием диалога."""

    def _summarize_assistant_message(self, msg: AssistantMessage) -> None:
        """Плотное саммари последнего сообщения ассистента в рабочую память
        (не попадает в контекст). Только для сообщений длиннее порога."""
        content = (msg.content or "").strip()
        MIN_CHARS = int(MIN_TOKENS_TO_SUMMARIZE * CHARS_PER_TOKEN)
        if len(content) < MIN_CHARS:
            return
        summary = _dense_summarize_message(self, content)
        stored = f"AI: {summary}" if summary else None
        if stored and len(stored) < len(content):
            self._per_msg_summaries[id(msg)] = stored
            self.on_system_msg(
                f"[PER-MSG SUMMARY] Assistant message ({len(content)} chars) "
                f"summarized into working memory ({len(summary)} chars)."
            )

    def _maybe_summarize_user_message(self, msg: UserMessage) -> None:
        """Плотное саммари длинного сообщения пользователя в рабочую память
        (не попадает в контекст). Только для сообщений длиннее порога."""
        content = (msg.content or "").strip()
        MIN_CHARS = int(MIN_TOKENS_TO_SUMMARIZE * CHARS_PER_TOKEN)
        if len(content) < MIN_CHARS:
            return
        summary = _dense_summarize_message(self, content)
        stored = f"USER: {summary}" if summary else None
        if stored and len(stored) < len(content):
            self._per_msg_summaries[id(msg)] = stored
            self.on_system_msg(
                f"[PER-MSG SUMMARY] User message ({len(content)} chars) "
                f"summarized into working memory ({len(summary)} chars)."
            )

    def _maybe_summarize_tool_result(self, tr: ToolResult) -> None:
        """Длинные выводы инструментов (> MIN_TOKENS_TO_SUMMARIZE) сразу
        суммаризируются в рабочую память; при сжатии саммари встанет на их место."""
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
        stored = f"TOOL({tr.name}): {summary}" if summary else None
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
            self._per_msg_summaries[id(tr)] = stored
            self.on_system_msg(
                f"[PER-MSG SUMMARY] Tool '{tr.name}' output ({len(content)} chars) "
                f"summarized into working memory ({len(summary)} chars)."
            )

    def _prune_per_msg_summaries(self) -> None:
        """Убирает из рабочей памяти саммари сообщений, которых больше нет в истории."""
        alive = {id(m) for m in self.history.get_all()}
        self._per_msg_summaries = {k: v for k, v in self._per_msg_summaries.items() if k in alive}

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

    def _auto_summarize_dialogue(self) -> None:
        """Автоматическая суммаризация диалога при превышении порога контекста."""
        preserve_last = Config.AUTO_SUMMARY_PRESERVE_LAST

        # --- Точка срабатывания: только на «безопасных» границах ---
        # Суммаризация допустима сразу после ToolResult либо после сообщения
        # пользователя. Если последнее сообщение — ассистент с вызовом инструмента
        # (незавершённый/висячий вызов), удаляем его из истории и суммаризируем
        # всё ТОЛЬКО ДО него: ассистент перевызовет инструмент уже после
        # суммаризации (так чище, чем тащить полу-вызов через сжатие).
        msgs = self.history._messages
        popped_calls = 0
        while (
            msgs
            and isinstance(msgs[-1], AssistantMessage)
            and msgs[-1].has_tool_calls()
        ):
            msgs.pop()
            popped_calls += 1

        last = msgs[-1] if msgs else None
        if not (isinstance(last, ToolResult) or isinstance(last, UserMessage)):
            # Небезопасная граница (ассистент дал текст, история пуста и т.п.) —
            # откладываем суммаризацию до следующей безопасной точки.
            return

        if popped_calls:
            self.on_system_msg(
                f"[AUTO-SUMMARY] Removed {popped_calls} pending tool call(s) before "
                f"summarizing; assistant will re-invoke after compression."
            )

        total = len(self.history)

        end_id = total - 1 - preserve_last
        start_id = Config.AFTER_SYSTEM_PROMPT

        if start_id > end_id:
            return

        original_len = self.history.content_len(start_id, end_id)

        summary = summarize_dialogue(
            self, start_id=start_id, end_id=end_id,
        )

        if not summary or len(summary) >= original_len:
            return

        reduction = 1.0 - (len(summary) / original_len)
        weak = reduction < Config.AUTO_SUMMARY_MIN_REDUCTION_RATIO

        # Слабое сжатие — переформируем саммари заново, усекая выводы инструментов
        # (блоки RESULT) прямо при сборке текста, без лишних вызовов LLM.
        # В per-message режиме это просто дешёвая пересборка без вызова LLM.
        truncated = False
        if weak and not Config.DISABLE_PER_MESSAGE_SUMMARIZATION:
            rerendered = summarize_dialogue(
                self, start_id=start_id, end_id=end_id,
                truncate_result_ratio=Config.TRUNCATE_TOOL_RESULT_KEEP_RATIO,
                truncate_result_min_chars=Config.TRUNCATE_TOOL_RESULT_CHARS,
            )
            if rerendered:
                summary = rerendered
                truncated = True

        # Итоговая степень сжатия — уже по финальному (возможно усечённому) саммари.
        final_reduction = 1.0 - (len(summary) / original_len)

        self.history.compress_old_messages(summary, preserve_last=preserve_last)
        self.file_states.prune()
        self._prune_per_msg_summaries()


        if truncated:
            self.on_system_msg(
                f"[AUTO-SUMMARY] Weak pre-truncation compression ({reduction:.0%} < "
                f"{Config.AUTO_SUMMARY_MIN_REDUCTION_RATIO:.0%}); truncated RESULT payloads "
                f"(kept {Config.TRUNCATE_TOOL_RESULT_KEEP_RATIO:.0%}, min {Config.TRUNCATE_TOOL_RESULT_CHARS} chars) "
                f"(final -{final_reduction:.0%})"
            )


        self.on_system_msg(
            f"[AUTO-SUMMARY] Context compressed ({int(original_len / Config.CHARS_PER_TOKEN)} -> {int(len(summary) / Config.CHARS_PER_TOKEN)} tokens, -{final_reduction:.0%})"
        )
