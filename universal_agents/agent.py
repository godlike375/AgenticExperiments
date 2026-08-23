from __future__ import annotations

from typing import Iterable, Union, Callable, Optional

from universal_agents.constants import ENVIRONMENT_PREFIX, ENVIRONMENT_PREFIX_END
from universal_agents.config import Config
from universal_agents.models import UserMessage
from universal_agents.llm_client import LLMClient, TokenUsageTracker, LoopDetector, jaccard_similarity
from universal_agents.history import ChatHistory
from universal_agents.generation import GenerationParams
from universal_agents.tool_manager import ToolManager
from universal_agents.sub_agent import SubAgent
from universal_agents.context_builder import prepare_messages_for_api, get_effective_prefill
from universal_agents.file_states import FileStateTracker
from universal_agents.tool_parsing import tc_name, tc_args

from universal_agents.agent_mixins import (
    ToolsMixin,
    MemoryMixin,
    HistoryMixin,
    StreamingMixin,
    ResponseMixin,
    ExecuteMixin,
    ConsistencyMixin,
)

# Предел последовательных ошибок инструментов за один chat() до сдачи (§1)
MAX_CONSECUTIVE_ERRORS = 5


class LLMAgent(
    ToolsMixin,
    MemoryMixin,
    HistoryMixin,
    StreamingMixin,
    ResponseMixin,
    ExecuteMixin,
    ConsistencyMixin,
):
    def __init__(
        self,
        system_prompt: str = "You are a helpful assistant",
        temp: float = None,
        timeout: int = None,
        tools_config: Union[list[str], dict, None] = None,
        on_render: Callable = lambda x: None,
        on_confirm: Callable[[str, dict], bool] = lambda n, a: True,
        on_system_msg: Callable[[str], None] = lambda x: None,
        external_plugins: dict[str, Callable] = None,
        denied_tools: Union[str, Iterable[str], None] = None,
        max_context_tokens: int = None,
        token_tracker: Optional[TokenUsageTracker] = None,
        top_p: float = None,
        frequency_penalty: float = None,
        presence_penalty: float = None,
        max_tokens: int = None,
        on_stream_chunk: Callable[[str], None] = None,
        on_stream_start: Callable[[], None] = None,
        on_stream_end: Callable[[], None] = None,
        on_reasoning_chunk: Callable[[str], None] = None,
        on_reasoning_start: Callable[[], None] = None,
        on_reasoning_end: Callable[[], None] = None,
        on_service_stream_start: Callable[[], None] = None,
        on_service_stream_chunk: Callable[[str], None] = None,
        on_service_stream_end: Callable[[], None] = None,
        streaming_enabled: bool = None,
        max_generation_attempts: int = None,
        disable_per_msg_summarization: bool = False,
    ):
        self.history = ChatHistory(system_prompt)
        self.file_states = FileStateTracker(self.history)
        self._read_registrations: list[str] = []
        self._gen_params = GenerationParams.from_overrides(
            temp=temp,
            timeout=timeout,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            max_tokens=max_tokens,
        )
        self.temp = self._gen_params.temp
        self.timeout = self._gen_params.timeout
        self.top_p = self._gen_params.top_p
        self.frequency_penalty = self._gen_params.frequency_penalty
        self.presence_penalty = self._gen_params.presence_penalty
        self.max_tokens = self._gen_params.max_tokens
        self.on_render = on_render
        self.on_confirm = on_confirm
        self.on_system_msg = on_system_msg
        self.self_consistency_mode = False
        self.sc_samples = 3
        # Внешний трекер (общий учёт с суб-агентом) или собственный.
        if token_tracker is not None:
            self.token_tracker = token_tracker
        else:
            self.token_tracker = TokenUsageTracker(system_prompt, max_context_tokens if max_context_tokens is not None else Config.MAX_CONTEXT_TOKENS)
        self.tools_manager = ToolManager(tools_config, external_plugins)
        if denied_tools:
            self.tools_manager.deny(denied_tools)
        self._auto_trust_git_root()
        self.loop_detector = LoopDetector()
        self._temp_override: Optional[float] = None
        self._original_temp = temp
        self.on_stream_chunk = on_stream_chunk
        self.on_stream_start = on_stream_start
        self.on_stream_end = on_stream_end
        self.on_reasoning_chunk = on_reasoning_chunk
        self.on_reasoning_start = on_reasoning_start
        self.on_reasoning_end = on_reasoning_end
        self.on_service_stream_start = on_service_stream_start
        self.on_service_stream_chunk = on_service_stream_chunk
        self.on_service_stream_end = on_service_stream_end
        self.streaming_enabled = streaming_enabled if streaming_enabled is not None else Config.STREAM_ENABLED
        self._tool_usage: dict[str, int] = {}
        self._max_generation_attempts = max_generation_attempts
        self._last_response_id: Optional[str] = None
        self._last_sent_msg_count: int = 0
        self._depth: int = 0
        self._disable_per_msg_summarization = disable_per_msg_summarization
        self._compacted_task_ids: set[str] = set()
        self.task_plan: list[str] = []
        self.task_plan_map: dict = {}

    @property
    def _per_msg_enabled(self) -> bool:
        """Гейт per-message суммаризации: выключена глобально (Config) или
        локально (disable_per_msg_summarization)."""
        return not (
            Config.DISABLE_PER_MESSAGE_SUMMARIZATION
            or self._disable_per_msg_summarization
        )

    def _build_duplicate_warning(self, dup_name: str, dup_args: str) -> str:
        """ENVIRONMENT_PREFIX-обёртка предупреждения о повторном дубликате вызова."""
        return (
            f"\n\n{ENVIRONMENT_PREFIX} Your previous attempt to call tool '{dup_name}' "
            f"with args '{dup_args}' was blocked as a duplicate of a recently already made call."
            f" Do NOT call '{dup_name}' again with same params. "
            f"Use other params or a different tool or write a text answer."
            f"{ENVIRONMENT_PREFIX_END}"
        )

    def _build_broken_call_fix(self) -> str:
        """ENVIRONMENT_PREFIX-обёртка фикс-промпта при нераспарсенном вызове инструмента."""
        return (
            f"{ENVIRONMENT_PREFIX} Your previous message looked like a tool call "
            f"but was not successfully parsed by the API. Try to write it properly now."
            f"{ENVIRONMENT_PREFIX_END}"
        )

    def make_sub_agent(
        self,
        *,
        safe_only: bool = False,
        max_iter: int = None,
        temp: float = None,
        on_log: Callable = None,
        depth: int = 0,
        disable_per_msg_summarization: bool = True,
        denied_tools: Union[str, Iterable[str], None] = None,
    ) -> SubAgent:
        """Создаёт SubAgent с ТОЧНО тем же набором схем инструментов, что у родителя.

        Обработчики передаются как есть (в том же порядке) — рендер префикса
        запроса суб-агента совпадает с родительским и KV-кэш переиспользуется.
        Ограничения задаются запретительным конфигом denied_tools (имена или '*'
        — запретить всё): запрещённые инструменты ОСТАЮТСЯ в схемах, но их вызов
        возвращает ошибку «forbidden for this sub-agent».
        safe_only=True дополнительно запрещает требующие подтверждения и
        небезопасные по путям инструменты (схемы тоже остаются в префиксе).
        """
        parent_system_prompt = self.history[0].content if len(self.history) else ""
        parent_history = self.history.get_all()[1:]
        parent_tools = {name: info["handler"] for name, info in self._all_tools.items()}
        effective_denied = self._resolve_denied_tools(denied_tools)
        if safe_only:
            effective_denied |= {
                name for name, info in self._all_tools.items()
                if info.get('requires_confirmation', False) or info.get('path_safety', False)
            }
        return SubAgent(
            parent_system_prompt=parent_system_prompt,
            parent_history=parent_history,
            parent_tools=parent_tools,
            denied_tools=effective_denied,
            max_context_tokens=self.token_tracker.max_context_tokens,
            max_iter=max_iter,
            temp=temp,
            on_log=on_log if on_log is not None else (lambda x: None),
            depth=depth,
            disable_per_msg_summarization=disable_per_msg_summarization,
        )

    def _resolve_denied_tools(self, denied_tools) -> set[str]:
        """Нормализует запретительный конфиг: None | '*'/'all' | имя | коллекция имён."""
        if denied_tools is None:
            return set()
        if isinstance(denied_tools, str):
            if denied_tools in ("*", "all"):
                return set(self._all_tools.keys())
            return {denied_tools}
        return set(denied_tools)

    def _on_history_changed(self) -> None:
        """Очистка после модификации истории: сброс кэша чтений (§2-5) + flush отложенных выгрузок."""
        self.file_states.prune()
        self.tools_manager.flush_pending_unloads()

    def record_usage(self, usage) -> None:
        """Единая точка учёта токенов из ответа LLM."""
        if usage:
            self.token_tracker.update_from_usage(usage)

    def _call_llm(
        self,
        messages: list[dict],
        prefill: Optional[str] = None,
        previous_response_id: Optional[str] = None,
        params: GenerationParams = None,
        watch_prefix: Optional[str] = None,
        watch_continue_temp: Optional[float] = None,
    ) -> tuple:
        """Единая точка транспорта LLM (§2): сама выбирает стриминг или обычный
        вызов. Возвращает (message_obj, error, usage), как LLMClient.call()."""
        tools = self.tools if self.tools else None
        if self.streaming_enabled and self.on_stream_chunk:
            return self._call_with_streaming(
                messages,
                prefill=prefill,
                tools=tools,
                previous_response_id=previous_response_id,
                params=params,
                watch_prefix=watch_prefix,
                watch_continue_temp=watch_continue_temp,
            )
        return LLMClient.call(
            messages,
            tools=tools,
            prefill=prefill,
            previous_response_id=previous_response_id,
            params=params,
        )

    def service_llm_call(
        self,
        msgs: list[dict],
        temp: Optional[float] = None,
        timeout: Optional[int] = None,
        tools=True,
        prefill: Optional[str] = None,
        params: GenerationParams = None,
    ) -> tuple:
        """Служебный вызов LLM (саммаризация, компактизация, consistency и т.п.).

        Идёт через тот же транспорт; если заданы колбэки служебного канала
        (on_service_stream_*) — стримится в отдельный визуальный канал со своей
        меткой, а не как ответ агента в диалоге.
        tools=True означает текущий набор инструментов агента.
        """
        callbacks = None
        if self.streaming_enabled and self.on_service_stream_chunk:
            callbacks = {
                "on_stream_start": self.on_service_stream_start,
                "on_stream_chunk": self.on_service_stream_chunk,
                "on_stream_end": self.on_service_stream_end,
            }
        return LLMClient.call(
            msgs,
            temp=temp,
            timeout=timeout,
            tools=(self.tools if tools else tools),
            prefill=prefill,
            params=params,
            callbacks=callbacks,
        )

    # --------------------------------------------------------
    # Главный цикл
    # --------------------------------------------------------
    def _detect_duplicate(self, message_obj) -> Optional[tuple]:
        """Дефолтный детектор дубликатов (§1): повторный вызов инструмента или
        повтор последнего текстового ответа. Возвращает None либо кортеж
        ('tool_call', (name, args)) / ('answer', prev_answer)."""
        if message_obj.tool_calls:
            current_history = self.history.get_all()
            for tc in message_obj.tool_calls:
                if self.loop_detector.check_duplicate_in_turn(tc_name(tc), tc_args(tc), current_history):
                    return 'tool_call', (tc_name(tc), tc_args(tc))
            return None

        answer_text = (message_obj.content or "").strip()
        prev_answer = self._get_last_answer_text()
        if (
            answer_text
            and prev_answer
            and jaccard_similarity(answer_text, prev_answer)
            >= Config.DUPLICATE_SIMILARITY_THRESHOLD
        ):
            return 'answer', prev_answer
        return None

    def call_with_retries(
        self,
        messages_to_send: list[dict],
        step_prefill: Optional[str] = None,
        prev_response_id: Optional[str] = None,
        all_messages_len: int = 0,
        is_duplicate_fn: Callable = None,
        boost: bool = True,
    ) -> tuple:
        """Проводит до max_generation_attempts попыток генерации (§1), отбрасывая
        дубликаты и активируя буст температуры. Логика повторов/температуры
        собрана здесь, а не размазана по вызывающему коду.

        is_duplicate_fn(message_obj) -> None | ('tool_call', (name, args))
        | ('answer', prev_answer); по умолчанию — self._detect_duplicate.
        boost=False оставляет обнаружение дубликатов, но без буста температуры.
        Возвращает (message_obj, api_error_occurred)."""
        if is_duplicate_fn is None:
            is_duplicate_fn = self._detect_duplicate
        max_generation_attempts = (
            self._max_generation_attempts
            if self._max_generation_attempts is not None
            else Config.MAX_LOOP_RETRIES
        )
        message_obj = None
        last_retry_warning: Optional[str] = None
        dup_watch_target: Optional[str] = None
        api_error_occurred = False

        for attempt in range(max_generation_attempts):
            attempt_params = self._gen_params
            if self._temp_override is not None:
                attempt_params = self._gen_params.with_temp(self._temp_override)
                self._temp_override = None

            active_messages = [dict(msg) for msg in messages_to_send]
            if last_retry_warning and active_messages:
                last_msg = active_messages[-1]
                last_msg["content"] = (last_msg.get("content") or "") + last_retry_warning

            # Транспорт выбирается внутри _call_llm (стриминг или обычный вызов)
            message_obj, err, usage = self._call_llm(
                active_messages,
                prefill=step_prefill,
                previous_response_id=prev_response_id,
                params=attempt_params,
                watch_prefix=dup_watch_target,
                watch_continue_temp=Config.DUPLICATE_CONTINUATION_TEMP if dup_watch_target is not None else None,
            )
            dup_watch_target = None

            if usage:
                self.record_usage(usage)
            if err:
                self.on_system_msg(f"[API Error] {err}")
                api_error_occurred = True
                break

            if not message_obj:
                api_error_occurred = True
                break

            duplicate = is_duplicate_fn(message_obj)
            if duplicate is not None:
                kind, payload = duplicate
                if kind == 'tool_call':
                    dup_name, dup_args = payload
                    last_retry_warning = self._build_duplicate_warning(dup_name, dup_args)
                    tag, reason = 'PROACTIVE LOOP DETECTED', f"Intercepted duplicate call to '{dup_name}'"
                else:
                    last_retry_warning = (
                        f"\n\n{ENVIRONMENT_PREFIX} Your previous answer was the same to the latest one. "
                        f"Please do NOT repeat it again and answer differently."
                        f"{ENVIRONMENT_PREFIX_END}"
                    )
                    dup_watch_target = payload
                    tag, reason = 'DUPLICATE ANSWER DETECTED', 'Model repeated the previous answer verbatim'
                if boost:
                    self._temp_override = Config.BOOST_TEMP
                self.on_system_msg(
                    f"[{tag}] {reason}. "
                    f"Discarding response. Activating temperature boost ({Config.BOOST_TEMP}) "
                    f"and injecting temporary warning. Attempt {attempt + 1}/{max_generation_attempts}."
                )
                continue

            # Ответ принят к обработке — фиксируем позицию контекста для
            # возможного продолжения через previous_response_id.
            if hasattr(message_obj, '_response_id'):
                self._last_response_id = message_obj._response_id
            self._last_sent_msg_count = all_messages_len
            break
        else:
            self.on_system_msg(
                "[PROACTIVE LOOP DETECTOR] Max re-generation attempts reached. Proceeding to execution safety nets."
            )

        return message_obj, api_error_occurred

    def _recover(self, kind: str, retries_left: int = 0, erased_count: int = 0) -> None:
        """Единая точка восстановления после сбоя (§1).

        Сбрасывает позицию контекста (_last_response_id / _last_sent_msg_count),
        чтобы следующая попытка ушла целиком без previous_response_id; для
        'broken_call' и 'tool_error' дополнительно активирует буст температуры.
        kind:
          'broken_call'  — стёрт сломанный ответ ассистента, регенерация
          'broken_fix'   — вставлен фикс-промпт
          'tool_error'   — стёрт неудачный вызов инструмента
          'api_error'    — сдача: нормализация истории и передача управления пользователю
          'broken_giveup'/'error_limit' — сдача после исчерпания попыток"""
        if kind == 'api_error':
            self.history.normalize(is_error_recovery=True)
            self.on_system_msg("⚠️ [RECOVERY] API error occurred. Role sequence restored. Handing control to user.")
            return
        if kind == 'broken_giveup':
            self.history.normalize(is_error_recovery=True)
            self.on_system_msg(
                f"⚠️ [BROKEN CALL] Could not recover malformed tool call after "
                f"{Config.BROKEN_CALL_REGEN_RETRIES + Config.BROKEN_CALL_FIX_RETRIES} attempts. "
                "Handing control to user."
            )
            return
        if kind == 'error_limit':
            self.history.normalize(is_error_recovery=True)
            self.on_system_msg(
                f"⚠️ [LIMIT REACHED] {retries_left} consecutive tool errors. Handing control to user."
            )
            return

        self._last_response_id = None
        self._last_sent_msg_count = 0
        if kind == 'broken_call':
            self._temp_override = Config.ERROR_RECOVERY_TEMP
            self.on_system_msg(
                f"[BROKEN CALL] Detected malformed tool call. "
                f"Regenerating with temp {Config.ERROR_RECOVERY_TEMP} "
                f"(retries left: {retries_left})."
            )
        elif kind == 'broken_fix':
            self.on_system_msg(
                f"[BROKEN CALL] Injected fix prompt for the model "
                f"(retries left: {retries_left})."
            )
        elif kind == 'tool_error':
            self._temp_override = Config.ERROR_RECOVERY_TEMP
            self.on_system_msg(
                f"[TOOL ERROR RECOVERY] Tool error detected. Erased {erased_count} message(s) "
                f"and regenerating with temperature {Config.ERROR_RECOVERY_TEMP} "
                f"(retries left: {retries_left})."
            )

    def chat(self, message: str, max_iter: int = None, prefill: str = None):
        max_iter = max_iter if max_iter is not None else Config.MAX_ITER
        if self.self_consistency_mode:
            return self._chat_self_consistent(message, prefill)

        user_msg = UserMessage(content=message)
        self.history.add(user_msg)
        if self._per_msg_enabled:
            self._maybe_summarize_user_message(user_msg)
        current_prefill = get_effective_prefill(prefill)
        self._last_response_id = None
        self._last_sent_msg_count = 0

        consecutive_errors = 0
        tool_error_retries_left = Config.ERROR_RECOVERY_RETRIES
        broken_regen_left = Config.BROKEN_CALL_REGEN_RETRIES
        broken_fix_left = Config.BROKEN_CALL_FIX_RETRIES

        for i in range(max_iter):
            step_prefill = current_prefill if i == 0 else None
            all_messages = prepare_messages_for_api(self)

            if i == 0 or self._last_response_id is None:
                messages_to_send = all_messages
                prev_response_id = None
            else:
                messages_to_send = all_messages[self._last_sent_msg_count:]
                prev_response_id = self._last_response_id

            message_obj, api_error_occurred = self.call_with_retries(
                messages_to_send,
                step_prefill=step_prefill,
                prev_response_id=prev_response_id,
                all_messages_len=len(all_messages),
            )

            if api_error_occurred or not message_obj:
                self._recover('api_error')
                return ""

            result_text, tool_error_occurred, broken_call = self._process_llm_response(message_obj)

            if broken_call:
                if broken_regen_left > 0:
                    broken_regen_left -= 1
                    self._erase_last_assistant()
                    self._recover('broken_call', retries_left=broken_regen_left)
                    continue
                if broken_fix_left > 0:
                    broken_fix_left -= 1
                    self.history.add(UserMessage(self._build_broken_call_fix()))
                    self.on_render(self.history.get_all()[-1])
                    self._recover('broken_fix', retries_left=broken_fix_left)
                    continue
                self._recover('broken_giveup')
                return ""

            if tool_error_occurred and tool_error_retries_left > 0:
                tool_error_retries_left -= 1
                erased = self._erase_last_failed_tool_call()
                self._recover('tool_error', retries_left=tool_error_retries_left, erased_count=erased)
                continue

            self._compact_completed_tasks()
            if self._get_context_usage_percent() >= Config.AUTO_SUMMARY_THRESHOLD:
                self._auto_summarize_dialogue()

            if tool_error_occurred:
                consecutive_errors += 1
            else:
                consecutive_errors = 0
                tool_error_retries_left = Config.ERROR_RECOVERY_RETRIES

            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                self._recover('error_limit', retries_left=MAX_CONSECUTIVE_ERRORS)
                return ""

            if not message_obj.tool_calls and not tool_error_occurred:
                return result_text