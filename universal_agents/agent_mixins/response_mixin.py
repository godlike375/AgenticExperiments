"""Mixin обработки ответа LLM: построение AssistantMessage, добавление в историю, рендер."""

from __future__ import annotations

from typing import Optional

from universal_agents.llm_client import apply_prefill
from universal_agents.models import AssistantMessage, ToolCall, ToolResult
from universal_agents.tool_parsing import tc_name, tc_args, detect_broken_call, args_are_valid

# Prefill для перегенерации голого вызова инструмента без пояснения.
# 'Assistant:' — стартовая приставка, после которой модель должна написать текст.
_NO_COMMENT_PREFILL = 'Assistant: "'


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

    def _process_llm_response(self, message_obj, no_comment_retry_left: int = 1,
                              reasoning_effort: Optional[str] = None,
                              skip_broken_detection: bool = False,
                              allow_tools: bool = True) -> tuple[str, bool, bool, Optional[str]]:
        """Обрабатывает сырой ответ LLM. Возвращает (text, tool_error, broken_call, rerun_prefill);
        rerun_prefill непуст, когда следующий ход надо перегенерировать с указанным prefill (_NO_COMMENT_PREFILL).
        no_comment_retry_left — сколько раз ещё можно перегенерировать голый вызов инструмента
        без пояснения; агенту достаётся из TurnState (Config.NO_COMMENT_RETRIES).
        reasoning_effort — ожидаемое значение хода из API (отличает «модель уже прокомментировала
        ход в reasoning_content»); по умолчанию берётся текущее состояние агента.
        skip_broken_detection — пропустить детекцию сломанного вызова (True для фаз
        структурированного вывода: закрытый маркером XML не должен выглядеть сломанным
        вызовом даже при содержании имён инструментов внутри)."""
        if reasoning_effort is None:
            reasoning_effort = self._reasoning_effort
        if not message_obj:
            return "Empty response", True, False, None

        content = message_obj.content or ""
        clean_content = content.strip()
        # Впрыснутый prefill сам по себе объяснением не считается:
        # модель должна написать текст сама. Обычно prefill попадает в content как
        # отдельное сообщение (тогда substantive == clean_content); если же шлюз
        # приклеил его к ответу — срезаем маркер перед проверкой.
        substantive = clean_content
        if substantive.startswith(_NO_COMMENT_PREFILL):
            substantive = substantive[len(_NO_COMMENT_PREFILL):].strip()
        assistant_msg = self._build_assistant_msg(message_obj, clean_content)

        # ── Служебный режим: инструменты не исполняются ──
        if not allow_tools and assistant_msg.has_tool_calls():
            tool_names = [tc.name for tc in assistant_msg.tool_calls]
            if substantive:
                # Есть текст и tool_calls: инструменты отбрасываем, оставляем только текст.
                assistant_msg.tool_calls = []
                if message_obj.tool_calls:
                    message_obj.tool_calls = []
                self.on_system_msg(
                    f"[SERVICE TURN] Tool call(s) {tool_names} discarded "
                    f"(tools not allowed); keeping text only."
                )
            else:
                # Нет текста — только tool_calls: перегенерируем (NO COMMENT / ошибка).
                assistant_msg.tool_calls = []
                if message_obj.tool_calls:
                    message_obj.tool_calls = []
                if no_comment_retry_left > 0 and reasoning_effort == "none":
                    self.on_system_msg(
                        f"[SERVICE TURN] Bare tool call {tool_names} without text discarded; "
                        f"rerunning with '{_NO_COMMENT_PREFILL}' prefill "
                        f"({no_comment_retry_left} retr{'y' if no_comment_retry_left == 1 else 'ies'} left)."
                    )
                    return clean_content, False, False, _NO_COMMENT_PREFILL
                self.on_system_msg(
                    f"[SERVICE TURN] Bare tool call {tool_names} without text after retries exhausted; "
                    f"requesting regeneration."
                )
                return clean_content, True, False, None

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

        if (
            assistant_msg.has_tool_calls()
            and not substantive
            and reasoning_effort == "none"
        ):
            tool_names = [tc.name for tc in assistant_msg.tool_calls]
            # Голый вызов инструмента без пояснения (reasoning выключен): стираем
            # (не добавляем) пустой ответ ассистента и перегенерируем следующий ход
            # с prefill, чтобы модель начала с текстового комментария перед вызовом.
            # При активном reasoning это не нужно — модель уже прокомментировала ход
            # в reasoning_content. Лимит ретраев у агента; пока они есть (попытка
            # считается только когда prefill реально запрошен) — см. NO_COMMENT_RETRIES.
            if no_comment_retry_left > 0:
                assistant_msg.tool_calls = []
                if message_obj.tool_calls:
                    message_obj.tool_calls = []
                self.on_system_msg(
                    f"[NO COMMENT] Tool call `{tool_names[0]}` with no explanation before were rejected: "
                    f"rerunning with '{_NO_COMMENT_PREFILL}' prefill "
                    f"({no_comment_retry_left} retr{'y' if no_comment_retry_left == 1 else 'ies'} left)."
                )
                return clean_content, False, False, _NO_COMMENT_PREFILL
            # Ретраи исчерпаны (это касается и answer_to_system без pending-задачи):
            # принимаем голый вызов как есть — инструмент сам вернёт явную ошибку или
            # выполнится без комментария, и цикл не зациклится.
            self.on_system_msg(
                f"[NO COMMENT] No retries left for bare tool call `{tool_names[0]}`, executing as-is."
            )

        if not clean_content and not assistant_msg.has_tool_calls():
            self.on_system_msg("[EMPTY RESPONSE] Model returned no content. Discarding and retrying...")
            return clean_content, True, False, None

        self._append_assistant(assistant_msg)

        if not assistant_msg.has_tool_calls():
            if not skip_broken_detection and detect_broken_call(clean_content, self._known_tool_names()):
                self.on_system_msg("[BROKEN CALL] Response looks like an unparsed tool call (prose or XML).")
                return clean_content, False, True, None
            return clean_content, False, False, None

        tool_results = self._execute_tools(assistant_msg.tool_calls)
        self._append_tool_results(tool_results)

        # removed = self.history.remove_failed_call_chains()
        # if removed:
        #     self.on_system_msg(
        #         f"[CLEANUP] Removed {removed} messages "
        #         f"({removed // 2} failed calls)"
        #     )
        #     self.history.normalize()
        #     self._on_history_changed()

        tool_error_occurred = any(tr.is_error and not tr.is_user_denied for tr in tool_results)
        return clean_content, tool_error_occurred, False, None
