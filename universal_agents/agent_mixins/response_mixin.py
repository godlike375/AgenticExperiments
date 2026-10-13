"""Mixin обработки ответа LLM: построение AssistantMessage, добавление в историю, рендер."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

from universal_agents.config import Config
from universal_agents.generation import GenerationParams
from universal_agents.llm_client import apply_prefill
from universal_agents.models import AssistantMessage, ToolCall, ToolResult
from universal_agents.tool_parsing import tc_name, tc_args, detect_broken_call, args_are_valid
from universal_agents.tools.builtin import respond_to_system

_NO_COMMENT_PREFILL = 'LLM Agent: "'


def sim_schema_text() -> str:
    """Схема формата симуляции текстом: '<short_think/>' (для логов и заметок UI)."""
    return "".join(f"<{tag}/>" for tag in Config.SIMULATED_REASONING_TAGS)


def tail_after_tag(content: str, tag: str) -> str:
    """Всё, что ПОСЛЕ закрывающего </tag> — свободный ответ ("" , если закрытия нет)."""
    if not content:
        return ""
    closing = f"</{tag}>"
    idx = content.rfind(closing)
    return content[idx + len(closing):] if idx >= 0 else ""


class SimReasoningGroup:
    """Секция рассуждения одного assistant-сообщения (§2.8).

    Формат ответа: '<short_think>…</short_think>' + свободный текст ответа.
    Форс минимален и сделан ОДНИМ вызовом LLM:
    - prefill открывает секцию ('<short_think>'), стоп-маркеров нет — если модель
      закрыла секцию сама, мы не режём её хвост и даём ей дописать ответ в том же
      вызове (иначе пришлось бы выбрасывать уже написанный ответ);
    - если модель секцию не закрыла, закрывающий тег дописываем сами (иначе рассуждение
      растянулось бы на весь ответ и закрытия не было бы вовсе);
    - всё после закрытия — ответ БЕЗ тегов: второй тег не нужен, пустой ответ не
      нарушение формата (модель вправе сразу уйти в инструмент).

    Единственный дополнительный вызов — когда ответ пуст и вызова инструмента нет:
    тогда к концу секции подставляется _NO_COMMENT_PREFILL (общий механизм NO COMMENT),
    бюджет ограничен retries, чтобы цикл не зациклился.

    Двухфазный режим — когда задана хотя бы одна SIMULATED_REASONING_*-настройка
    сэмплирования: секция генерируется отдельным вызовом (стоп-маркер
    '</short_think>', параметры размышлений), затем ответ — вторым вызовом
    (обычные параметры). Если модель дописала ответ или вызвала инструмент уже
    в фазе размышлений — принимается как есть, лишний вызов не нужен."""


    def __init__(self, retries: int):
        self.reset(retries)

    def reset(self, retries: int) -> None:
        """Новое сообщение агента (после инструмента/ошибки): всё с нуля."""
        self.tag = Config.SIMULATED_REASONING_TAGS[0]
        self.content = ""      # накопленный ответ: '<short_think>…' + ответ
        self.appended = ""     # дописанный закрывающий тег (эхо в живой вывод)
        self.tool_calls: list[ToolCall] = []
        self.reasoning = ""
        self.retries_left = retries
        self._continuation = False
        self.phase = "think" if self._reasoning_overrides() else "single"
        self._shown = 0
        self._injected: list[int] = []


    @staticmethod
    def _reasoning_overrides() -> dict:
        """Заданные SIMULATED_REASONING_*-настройки сэмплирования (имя поля → значение)."""
        names = ("temp", "top_p", "min_p", "frequency_penalty", "presence_penalty")
        return {
            name: getattr(Config, f"SIMULATED_REASONING_{name.upper()}")
            for name in names
            if getattr(Config, f"SIMULATED_REASONING_{name.upper()}") is not None
        }

    def two_phase(self) -> bool:
        """Нужны ли два вызова: задана хотя бы одна настройка размышлений."""
        return self.phase in ("think", "answer")

    def is_think_phase(self) -> bool:
        """Текущий вызов — фаза размышлений (свои параметры + стоп-маркер закрытия)."""
        return self.phase == "think"

    def advance_to_answer(self) -> None:
        """Перейти от фазы размышлений к фазе ответа."""
        if self.phase == "think":
            self.phase = "answer"

    def think_params(self, base: GenerationParams) -> Optional[GenerationParams]:
        """Параметры фазы размышлений: базовые + заданные оверрайды. None — не заданы."""
        overrides = self._reasoning_overrides()
        if not overrides:
            return None
        return replace(base, **overrides)

    def prefill(self) -> str:
        """Prefill вызова: открывающий тег секции, либо накопленный ответ при продолжении."""
        if self._continuation or self.phase == "answer":
            return self.content
        return f"<{self.tag}>"

    def prefill_shown(self) -> int:
        """Длина уже показанной части prefill — печатать в живой вывод только хвост."""
        return self._shown


    def absorb(self, message_obj, prefill: str) -> None:
        """Забирает ответ вызова (текст, tool_calls, reasoning) в накопленный контент."""
        raw = apply_prefill(message_obj.content or "", prefill)
        self.content = raw
        self.tool_calls.extend(message_obj.tool_calls or [])
        self.reasoning += getattr(message_obj, "reasoning_content", "") or ""
        self._shown = len(self.content)

    def close_section(self) -> None:
        """Дописывает '</short_think>', если модель не закрыла секцию сама.

        После закрытия всегда идёт перевод строки (для восприятия: ответ
        начинается с новой строки). Если модель его уже поставила — не дублируем.
        Эхо в живой вывод — только когда перевод вставлен в конец (его ещё можно
        показать); если ответ уже выведен дальше тега, перевод остаётся только
        в истории — прошедший стрим не исправить.
        """
        closing = f"</{self.tag}>"
        if self.content and closing not in self.content:
            self.content += closing
            self.appended = closing
        if self.content:
            pos = self.content.rfind(closing) + len(closing)
            if self.content[pos:pos + 1] != "\n":
                self.content = self.content[:pos] + "\n" + self.content[pos:]
                if pos == len(self.content) - 1:
                    self.appended += "\n"
        self._shown = len(self.content)

    def take_appended(self) -> str:
        """Забрать дописанный хвост для эха в живой вывод. Одноразовый: повторный
        вызов (например, verify фазы answer после think) ничего не вернёт."""
        tail, self.appended = self.appended, ""
        return tail

    def answer_text(self) -> str:
        """Свободный ответ модели — всё после закрывающего тега, без наших вставок."""
        tail = tail_after_tag(self.content, self.tag)
        head_len = len(self.content) - len(tail)
        return self._strip_injected(tail, head_len)

    def final_content(self) -> str:
        """Контент для истории: ответ модели без вклиненных приставок."""
        return self._strip_injected(self.content, 0)

    def _strip_injected(self, text: str, offset: int) -> str:
        """Вырезает куски, вклиненные нами (offset — позиция text в self.content)."""
        if not self._injected:
            return text
        skip: set[int] = set()
        for start in self._injected:
            for pos in range(start, min(start + len(_NO_COMMENT_PREFILL), len(self.content))):
                skip.add(pos - offset)
        return "".join(ch for i, ch in enumerate(text) if i not in skip)

    def start_continuation(self) -> None:
        """Продолжение пустого ответа: приставка после секции, вынуждающая написать текст."""
        self._shown = len(self.content)
        self._injected.append(len(self.content))
        self.content += _NO_COMMENT_PREFILL
        self.appended = ""
        self.tool_calls = []
        self.reasoning = ""
        self._continuation = True


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


    def _sim_verify_phase(self, group: SimReasoningGroup) -> Optional[AssistantMessage]:
        """Доводит ответ до готового сообщения или просит дописать пустой ответ.

        None → нужен ещё один вызов LLM (prefill = group.prefill())."""
        group.close_section()
        appended = group.take_appended()
        if appended and getattr(self, "on_stream_chunk", None):
            self.on_stream_chunk(appended)
        answer = group.answer_text()
        if answer.strip():
            return self._sim_message(group)

        if not group.tool_calls and group.retries_left > 0:
            group.retries_left -= 1
            group.start_continuation()
            self.on_system_msg(
                f"[SIM REASONING] No answer text after </{group.tag}>; continuing with "
                f"'{_NO_COMMENT_PREFILL}' prefill (retries left: {group.retries_left})."
            )
            return None
        if group.tool_calls:
            self.on_system_msg(
                f"[SIM REASONING] No answer text after </{group.tag}>; "
                f"tool call accepted as-is."
            )
        else:
            self.on_system_msg(
                "[SIM REASONING] Empty answer; continuation budget exhausted; "
                "accepting response as-is."
            )
        return self._sim_message(group)

    def _sim_message(self, group: SimReasoningGroup) -> AssistantMessage:
        """Собирает накопленный ответ в одно AssistantMessage (контент уже в стриме показан)."""
        return self._assemble_assistant_message(
            group.final_content(), group.tool_calls, group.reasoning, streamed=True
        )

    def _process_llm_response(self, message_obj, no_comment_retry_left: int = 1,
                              reasoning_effort: Optional[str] = None,
                              skip_broken_detection: bool = False,
                              allow_tools: bool = True,
                              broken_scan_text: Optional[str] = None) -> tuple[str, bool, bool, Optional[str]]:
        """Обрабатывает сырой ответ LLM. Возвращает (text, tool_error, broken_call, rerun_prefill);
        rerun_prefill непуст, когда следующий ход надо перегенерировать с указанным prefill (_NO_COMMENT_PREFILL).
        no_comment_retry_left — сколько раз ещё можно перегенерировать голый вызов инструмента
        без пояснения; агенту достаётся из TurnState (Config.NO_COMMENT_RETRIES).
        reasoning_effort — ожидаемое значение хода из API (отличает «модель уже прокомментировала
        ход в reasoning_content»); по умолчанию берётся текущее состояние агента.
        skip_broken_detection — пропустить детекцию сломанного вызова (True для фаз
        структурированного вывода: закрытый маркером XML не должен выглядеть сломанным
        вызовом даже при содержании имён инструментов внутри).
        broken_scan_text — что именно сканировать на сломанный вызов (по умолчанию весь
        контент). При симуляции reasoning сюда передаётся свободный ответ (всё после
        </short_think>): в секции рассуждения всегда есть теги, и без сужения гейт
        «есть XML-тег» срабатывал бы на каждом ответе, а проза вида «read() выполнен»
        или «Вызываю read() для просмотра…» давала бы ложные срабатывания."""
        if reasoning_effort is None:
            reasoning_effort = self._reasoning_effort
        if not message_obj:
            return "Empty response", True, False, None

        content = message_obj.content or ""
        clean_content = content.strip()
        substantive = clean_content
        if substantive.startswith(_NO_COMMENT_PREFILL):
            substantive = substantive[len(_NO_COMMENT_PREFILL):].strip()
        assistant_msg = self._build_assistant_msg(message_obj, clean_content)

        if not allow_tools and assistant_msg.has_tool_calls():
            tool_names = [tc.name for tc in assistant_msg.tool_calls]
            if substantive:
                assistant_msg.tool_calls = []
                if message_obj.tool_calls:
                    message_obj.tool_calls = []
                self.on_system_msg(
                    f"[SERVICE TURN] Tool call(s) {tool_names} discarded "
                    f"(tools not allowed); keeping text only."
                )
            else:
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
            self.on_system_msg(
                f"[NO COMMENT] No retries left for bare tool call `{tool_names[0]}`, executing as-is."
            )

        if not clean_content and not assistant_msg.has_tool_calls():
            self.on_system_msg("[EMPTY RESPONSE] Model returned no content. Discarding and retrying...")
            return clean_content, True, False, None

        pending_before = self._pending_operation is not None
        self._append_assistant(assistant_msg)
        if pending_before and respond_to_system.__name__ not in [tc.name for tc in assistant_msg.tool_calls]:
            self._mark_confirmation_junk(assistant_msg)

        if not assistant_msg.has_tool_calls():
            scan_text = broken_scan_text if broken_scan_text is not None else clean_content
            if not skip_broken_detection and detect_broken_call(
                scan_text,
                self._known_tool_names(),
                require_tag=broken_scan_text is None,
            ):
                self.on_system_msg("[BROKEN CALL] Response looks like an unparsed tool call (prose or XML).")
                return clean_content, False, True, None
            return clean_content, False, False, None

        tool_results = self._execute_tools(assistant_msg.tool_calls)
        name_by_id = {tc.id: tc.name for tc in assistant_msg.tool_calls}
        for tr in tool_results:
            tname = name_by_id.get(tr.tool_call_id, "")
            if tname == respond_to_system.__name__:
                if pending_before and tr.is_error and not tr.is_user_denied:
                    self._mark_confirmation_junk(assistant_msg)
                    self._mark_confirmation_junk(tr)
            elif pending_before:
                self._mark_confirmation_junk(tr)
        self._append_tool_results(tool_results)
        if any(
                tr.name == respond_to_system.__name__ and not tr.is_error and not tr.is_user_denied
            for tr in tool_results
        ):
            self._scrub_confirmation_trail()


        tool_error_occurred = any(tr.is_error and not tr.is_user_denied for tr in tool_results)
        return clean_content, tool_error_occurred, False, None
