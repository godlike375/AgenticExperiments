from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Union, Callable, Optional

from universal_agents.constants import ENVIRONMENT_PREFIX, ENVIRONMENT_PREFIX_END, INTERRUPT_HEADER
from universal_agents.config import Config
from universal_agents.models import UserMessage, AssistantMessage, ToolResult
from universal_agents.llm_client import LLMClient, TokenUsageTracker, LoopDetector, jaccard_similarity
from universal_agents.history import ChatHistory
from universal_agents.generation import GenerationParams
from universal_agents.tool_manager import ToolManager
from universal_agents.sub_agent import SubAgent
from universal_agents.context_builder import prepare_messages_for_api, get_effective_prefill
from universal_agents.file_states import FileStateTracker
from universal_agents.tool_parsing import tc_name, tc_args
from universal_agents.tools.builtin import answer_to_system

from universal_agents.agent_mixins import (
    ToolsMixin,
    MemoryMixin,
    HistoryMixin,
    StreamingMixin,
    ResponseMixin,
    ExecuteMixin,
    ConsistencyMixin,
)
from universal_agents.exceptions import GenerationInterrupted

# Предел последовательных ошибок инструментов за один chat() до сдачи (§1)
MAX_CONSECUTIVE_ERRORS = 5


@dataclass
class TurnState:
    """Состояние одного хода генерации (_run_turn_loop): счётчики ретраев и prefill.

    Инкапсулирует всю мутируемую state-machine хода, чтобы тело цикла не жонглировало
    счётчиками вперемешку. Поведение идентично прежним локальным переменным.
    """

    prefill: Optional[str] = None
    consecutive_errors: int = 0
    tool_error_retries_left: int = field(default_factory=lambda: Config.ERROR_RECOVERY_RETRIES)
    broken_regen_left: int = field(default_factory=lambda: Config.BROKEN_CALL_REGEN_RETRIES)
    broken_fix_left: int = field(default_factory=lambda: Config.BROKEN_CALL_FIX_RETRIES)
    no_comment_retries_left: int = field(default_factory=lambda: Config.NO_COMMENT_RETRIES)

    def has_prefill(self) -> bool:
        """Есть ли prefill для следующего шага (после [NO COMMENT] или из старта хода)."""
        return self.prefill is not None

    def step_prefill(self) -> Optional[str]:
        """Возвращает prefill для текущего шага (без сброса — живёт до явного clear/set)."""
        return self.prefill

    def set_prefill(self, value: Optional[str]) -> None:
        """Откладывает prefill для следующего шага (из [NO COMMENT] rerun_prefill)."""
        self.prefill = value

    def clear_prefill(self) -> None:
        """Сбрасывает prefill после успешного нетривиального шага."""
        self.prefill = None

    def can_retry_broken_regen(self) -> bool:
        """Остались ли попытки регенерации сломанного вызова; декрементирует счётчик."""
        if self.broken_regen_left > 0:
            self.broken_regen_left -= 1
            return True
        return False

    def can_retry_broken_fix(self) -> bool:
        """Остались ли попытки фикс-инъекции; декрементирует счётчик."""
        if self.broken_fix_left > 0:
            self.broken_fix_left -= 1
            return True
        return False

    def can_retry_tool_error(self) -> bool:
        """Остались ли попытки восстановления после ошибки инструмента; декрементирует счётчик."""
        if self.tool_error_retries_left > 0:
            self.tool_error_retries_left -= 1
            return True
        return False

    def record_tool_error(self) -> None:
        """Фиксирует очередную последовательную ошибку инструмента."""
        self.consecutive_errors += 1

    def record_tool_success(self) -> None:
        """Сбрасывает счётчик последовательных ошибок и возвращает ретраи."""
        self.consecutive_errors = 0
        self.tool_error_retries_left = Config.ERROR_RECOVERY_RETRIES

    @property
    def max_errors_reached(self) -> bool:
        """Достигнут ли лимит последовательных ошибок за один ход."""
        return self.consecutive_errors >= MAX_CONSECUTIVE_ERRORS


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
        autosave_enabled: bool = None,
        autosave_dir: str = None,
        autosave_keep: int = None,
        on_interrupt_check: Callable[[], bool] = None,
    ):
        self.history = ChatHistory(system_prompt)
        self.file_states = FileStateTracker(self.history)
        from universal_agents.archive import HistoryArchive
        self.archive = HistoryArchive()
        self._pending_pins: list[str] = []
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
        self._thinking_enabled: bool = False
        self._thinking_once: bool = False
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
        self._depth: int = 0
        self._disable_per_msg_summarization = disable_per_msg_summarization
        self._compacted_task_ids: set[str] = set()
        self._auto_summarize_suppressed = False
        self.task_plan: list[str] = []
        self.task_plan_map: dict = {}
        self._pending_operation: Optional[dict] = None

        # Авто-сохранение (защита от сбоев): один файл на диалог с меткой времени
        # запуска/сброса; перезаписывается при каждом снимке. Ротация оставляет
        # AUTOSAVE_KEEP последних файлов (по одному на диалог/запуск).
        self.autosave_enabled = Config.AUTOSAVE_ENABLED if autosave_enabled is None else autosave_enabled
        self.autosave_dir = Config.AUTOSAVE_DIR if autosave_dir is None else autosave_dir
        self.autosave_keep = Config.AUTOSAVE_KEEP if autosave_keep is None else autosave_keep
        self._autosave_path = None
        self.reset_autosave_path()
        # Прерывание генерации: Event ставится из фонового монитора (CLI) или on_interrupt_check.
        self.stop_event = threading.Event()
        self.on_interrupt_check = on_interrupt_check if on_interrupt_check is not None else (lambda: False)
        self._stop_check = lambda: self.stop_event.is_set() or bool(self.on_interrupt_check())
        # Пользовательский ввод во время генерации (redirect): когда выполняется инструмент,
        # текст пользователя не прерывает инструмент, а ждёт его завершения, после чего
        # вставляется в историю и запускается новая генерация (сценарий В).
        self.pending_interrupt: Optional[str] = None
        # True, пока поток генерации исполняет инструменты (нужно UI для выбора поведения).
        self._in_tool_execution = False

    @property
    def _per_msg_enabled(self) -> bool:
        """Гейт per-message суммаризации: выключена глобально (Config.PER_MSG_SUMMARIES_ENABLED) или локально (disable_per_msg_summarization)."""
        return Config.PER_MSG_SUMMARIES_ENABLED and not self._disable_per_msg_summarization

    @property
    def _reasoning_effort(self) -> str:
        """Вычисляет текущий reasoning_effort на основе состояния thinking-тогглов: 'low' если
        включён постоянный toggle или задан разовый (_thinking_once); иначе 'none'. Свойство
        чистое — разовый toggle потребляется один раз в начале хода (_run_turn_loop), чтобы
        API и обработчик ответа за весь ход видели один и тот же reasoning_effort."""
        if self._thinking_once or self._thinking_enabled:
            return "low"
        return "none"

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
        include_tools: bool = True,
    ) -> SubAgent:
        """Создаёт SubAgent из текущего агента (наследует системный промпт и историю для KV-cache). denied_tools переопределяет запреты; safe_only запрещает инструменты requires_confirmation/path_safety; include_tools=False убирает схемы инструментов (чистый текстовый субагент)."""
        parent_system_prompt = self.history[0].content if len(self.history) else ""
        parent_history = self.history.get_all()[1:]
        parent_tools = {name: info["handler"] for name, info in self._all_tools.items()} if include_tools else {}
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

    # --------------------------------------------------------
    # Pending operations (для requires_model_confirmation)
    # --------------------------------------------------------
    def set_pending_operation(self, op: dict) -> None:
        """Сохраняет отложенную операцию (handler + args) для подтверждения моделью."""
        self._pending_operation = op

    def pop_pending_operation(self) -> Optional[dict]:
        """Извлекает и удаляет отложенную операцию. Возвращает None, если операций нет."""
        op = self._pending_operation
        self._pending_operation = None
        return op

    # --------------------------------------------------------
    # Авто-сохранение (защита от сбоев)
    # --------------------------------------------------------
    def _build_save_extras(self) -> dict:
        """Собирает служебные данные верхнего уровня для сохранения (архив, план, cwd и т.п.)."""
        from universal_agents.task_tracker import plan_state_to_dict
        from universal_agents.project_root import get_project_root_override
        return {
            "archive": self.archive.to_list(),
            "plan_state": plan_state_to_dict(self),
            "pending_pins": list(getattr(self, "_pending_pins", [])),
            "cwd": os.getcwd(),
            "project_root": get_project_root_override(),
        }

    def save_history(self, path: str, extras: dict = None) -> None:
        """Сохраняет историю (с архивом/планом/состоянием файлов) в файл path."""
        self.history.save(
            path,
            loaded_tools=list(self._all_tools.keys()),
            file_states=self.file_states.to_dict(),
            extras=extras if extras is not None else self._build_save_extras(),
        )

    def _rotate_autosaves(self) -> None:
        """Оставляет только AUTOSAVE_KEEP самых свежих файлов (имя сортируется по времени метки)."""
        if self.autosave_keep <= 0:
            return
        try:
            if not os.path.isdir(self.autosave_dir):
                return
            files = [
                os.path.join(self.autosave_dir, f)
                for f in os.listdir(self.autosave_dir)
                if f.startswith("autosave_") and f.endswith(".json")
            ]
            files.sort()
            excess = files[:-self.autosave_keep]
            for old in excess:
                try:
                    os.remove(old)
                except OSError:
                    pass
        except OSError:
            pass

    def reset_autosave_path(self) -> None:
        """Берёт новую метку (дату/время) для файла авто-сохранения текущего диалога."""
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._autosave_path = os.path.join(self.autosave_dir, f"autosave_{ts}.json")

    def autosave(self) -> None:
        """Снимает снимок истории в файл текущего диалога (перезапись). Безопасно к вызову из любого места."""
        if not self.autosave_enabled:
            return
        try:
            os.makedirs(self.autosave_dir, exist_ok=True)
            path = self._autosave_path or os.path.join(
                self.autosave_dir, f"autosave_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
            )
            self.save_history(path)
            self._rotate_autosaves()
        except Exception as e:
            self.on_system_msg(f"[Autosave] failed: {e}")

    def reset_dialog(self) -> None:
        """Сбрасывает диалог к началу (только системный промпт) и берёт новую метку авто-сохранения."""
        system_prompt = self.history[0].content if len(self.history) else ""
        self.history = ChatHistory(system_prompt)
        from universal_agents.archive import HistoryArchive
        self.archive = HistoryArchive()
        self.task_plan = []
        self.task_plan_map = {}
        self._pending_pins = []
        self.file_states = FileStateTracker(self.history)
        self._tool_usage.clear()
        self.loop_detector = LoopDetector()
        self.token_tracker = TokenUsageTracker(
            system_prompt,
            self.token_tracker.max_context_tokens if self.token_tracker else Config.MAX_CONTEXT_TOKENS,
        )
        self._compacted_task_ids = set()
        self._pending_operation = None
        self.reset_autosave_path()
        self._on_history_changed()

    def _autosave(self) -> None:
        """Точка вызова авто-сохранения из цикла чата."""
        self.autosave()

    # --------------------------------------------------------
    # Прерывание генерации (передача управления пользователю)
    # --------------------------------------------------------
    def request_stop(self) -> None:
        """Запрашивает остановку текущей генерации (проверяется между шагами и внутри стрима)."""
        self.stop_event.set()

    def clear_stop(self) -> None:
        self.stop_event.clear()

    def set_pending_interrupt(self, text: str) -> None:
        """Запоминает текст пользователя, введённый во время выполнения инструмента (сценарий В).
        Поток генерации обработает его на границе хода (после завершения инструмента)."""
        self.pending_interrupt = text

    def take_pending_interrupt(self) -> Optional[str]:
        """Забирает и сбрасывает ожидающий текст прерывания (вызывается UI после завершения потока)."""
        text = self.pending_interrupt
        self.pending_interrupt = None
        return text

    def clear_pending_interrupt(self) -> None:
        self.pending_interrupt = None

    # --------------------------------------------------------

    def _call_llm(
        self,
        messages: list[dict],
        prefill: Optional[str] = None,
        params: GenerationParams = None,
        watch_prefix: Optional[str] = None,
        watch_continue_temp: Optional[float] = None,
        stop_check: Callable[[], bool] = None,
        reasoning_effort: str = "none",
    ) -> tuple:
        """Единая точка транспорта LLM (§2): выбирает стриминг или обычный вызов; возвращает (message_obj, error, usage)."""
        tools = self.tools if self.tools else None
        if self.streaming_enabled and self.on_stream_chunk:
            return self._call_with_streaming(
                messages,
                prefill=prefill,
                tools=tools,
                params=params,
                watch_prefix=watch_prefix,
                watch_continue_temp=watch_continue_temp,
                stop_check=stop_check,
                reasoning_effort=reasoning_effort,
            )
        return LLMClient.call(
            messages,
            tools=tools,
            prefill=prefill,
            params=params,
            stop_check=stop_check,
            reasoning_effort=reasoning_effort,
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
        """Служебный вызов LLM (саммаризация, компактизация, consistency). При заданных on_service_stream_* колбэках стримится в отдельный канал со своей меткой. tools=True — текущий набор инструментов агента. stop_check всегда передаётся (это стоп-событие пользователя), чтобы служебный стрим можно было прервать через watchdog — иначе компакция «глуха» к остановке."""
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
            stop_check=self._stop_check,
        )

    # --------------------------------------------------------
    # Главный цикл
    # --------------------------------------------------------
    def _detect_duplicate(self, message_obj) -> Optional[tuple]:
        """Дефолтный детектор дубликатов (§1): повторный вызов инструмента или повтор текстового ответа; возвращает None или ('tool_call', (name, args))/('answer', prev_answer)."""
        if message_obj.tool_calls:
            current_history = self.history.get_all()
            seen_in_batch: set[str] = set()
            for tc in message_obj.tool_calls:
                norm = self.loop_detector.normalize_args(tc_args(tc))
                # Дубликат внутри текущего пакета (два одинаковых вызова в одном ответе)
                batch_key = f"{tc_name(tc)}:{norm}"
                if batch_key in seen_in_batch:
                    return 'tool_call', (tc_name(tc), tc_args(tc))
                seen_in_batch.add(batch_key)
                # Дубликат с предыдущими вызовами в текущем ходу
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
        is_duplicate_fn: Callable = None,
        boost: bool = True,
        reasoning_effort: str = None,
    ) -> tuple:
        """До max_generation_attempts попыток генерации (§1), отбрасывая дубликаты и бустя температуру; логика повторов собрана здесь. is_duplicate_fn по умолчанию — self._detect_duplicate; boost=False оставляет детект без буста. Возвращает (message_obj, api_error_occurred)."""
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
        effective_reasoning_effort = reasoning_effort if reasoning_effort is not None else self._reasoning_effort

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
                params=attempt_params,
                watch_prefix=dup_watch_target,
                watch_continue_temp=Config.DUPLICATE_CONTINUATION_TEMP if dup_watch_target is not None else None,
                stop_check=self._stop_check,
                reasoning_effort=effective_reasoning_effort,
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

            break
        else:
            self.on_system_msg(
                "[PROACTIVE LOOP DETECTOR] Max re-generation attempts reached. Proceeding to execution safety nets."
            )

        return message_obj, api_error_occurred

    def _recover(self, kind: str, retries_left: int = 0, erased_count: int = 0) -> None:
        """Единая точка восстановления после сбоя (§1): для 'broken_call'/'tool_error' активирует буст температуры. kind: 'broken_call'/'broken_fix'/'tool_error'/'api_error'/'giveup'/'error_limit' — см. код."""
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
        # Новый ход пользователя: снимаем cooldown авто-компакции (следующая
        # компакция снова разрешена один раз), прерванная продолжаться не должна.
        self._auto_summarize_suppressed = False
        self._prepare_turn(message)
        return self._run_turn_loop(max_iter, prefill)

    def inject_user_interrupt(self, user_text: str, max_iter: int = None) -> str:
        """Прерывание генерации с новой информацией (сценарии А/Б/В).

        Сценарий В (после вывода инструмента): шапка и текст пользователя дописываются
        в конец контента ToolResult — это не ломает последовательность ролей в API
        и не требует пустой заглушки ассистента. Сценарии А/Б: сообщение пользователя
        с шапкой добавляется отдельным UserMessage сразу после прерванного фрагмента.
        В обоих случаях сразу запускается новая генерация ассистента.
        """
        max_iter = max_iter if max_iter is not None else Config.MAX_ITER
        if self.self_consistency_mode:
            self.clear_pending_interrupt()
            return self._chat_self_consistent(f"{INTERRUPT_HEADER}\n\n{user_text}", None)
        self.clear_pending_interrupt()
        if not self._append_interrupt_to_tool_result(user_text):
            self._prepare_turn(f"{INTERRUPT_HEADER}\n\n{user_text}")
        return self._run_turn_loop(max_iter)

    def _append_interrupt_to_tool_result(self, user_text: str) -> bool:
        """Сценарий В: дописывает шапку прерывания и текст пользователя в конец вывода
        инструмента (последнего ToolResult в истории). Возвращает True, если дописала."""
        last = self.history.get_last_message()
        if not isinstance(last, ToolResult):
            return False
        last.content = (last.content or "") + f"\n\n{INTERRUPT_HEADER}\n\n{user_text}"
        self._on_history_changed()
        self._autosave()
        self.clear_stop()
        return True

    def _prepare_turn(self, message: str) -> None:
        """Добавляет user-сообщение в историю и готовит позицию контекста перед генерацией.

        Сообщение вставляется сразу после прерванного фрагмента. При необходимости
        добавляется пустая заглушка ассистента (или убираются висящие tool_calls), чтобы
        не сломать последовательность ролей в API-запросе."""
        user_msg = UserMessage(content=message)
        last = self.history.get_last_message()
        if isinstance(last, AssistantMessage) and last.has_tool_calls():
            self.history.pop_pending_tool_calls()
            last = self.history.get_last_message()
        if isinstance(last, UserMessage):
            self.history.add(AssistantMessage(content=""))
        self.history.add(user_msg)
        if self._per_msg_enabled:
            self._maybe_summarize_user_message(user_msg)
        self._autosave()
        self.clear_stop()

    def _run_turn_loop(self, max_iter: int, prefill: str = None) -> str:
        current_prefill = get_effective_prefill(prefill)
        # Разовый thinking-тоггл (_thinking_once) фиксируется ОДИН раз за ход: и API
        # (call_with_retries), и обработчик ответа (_process_llm_response) должны видеть
        # одинаковый reasoning_effort. Раньше геттер _reasoning_effort сбрасывал
        # _thinking_once при первом чтении (сайд-эффект в property), из-за чего второе
        # чтение — в NO COMMENT-ветке обработчика — расходилось с отправленным в API.
        turn_reasoning = self._reasoning_effort
        self._thinking_once = False
        # Состояние хода: prefill для следующего шага ('Assistant:' после [NO COMMENT]
        # или стартовый current_prefill) + счётчики ретраев/ошибок.
        state = TurnState(prefill=current_prefill)

        for _ in range(max_iter):
            # Пользователь ввёл новый текст, пока выполнялся инструмент (сценарий В):
            # завершаем ход на чистой границе истории, чтобы вставка сообщения и новая
            # генерация произошли сразу после результата инструмента.
            if self.pending_interrupt is not None:
                self._autosave()
                return ""

            if self.stop_event.is_set():
                self.on_system_msg("⏹ Generation stopped by user — control returned to you.")
                self.clear_stop()
                self._autosave()
                return ""
            step_prefill = state.step_prefill()
            all_messages = prepare_messages_for_api(
                self, debug_hash_check=Config.DEBUG_PREFIX_HASH_CHECK
            )

            try:
                message_obj, api_error_occurred = self.call_with_retries(
                    all_messages,
                    step_prefill=step_prefill,
                    reasoning_effort=turn_reasoning,
                )
            except GenerationInterrupted:
                self.on_system_msg("⏹ Generation stopped by user — control returned to you.")
                self.clear_stop()
                self._autosave()
                return ""

            if api_error_occurred or not message_obj:
                self._recover('api_error')
                return ""

            try:
                result_text, tool_error_occurred, broken_call, rerun_prefill = self._process_llm_response(
                    message_obj,
                    no_comment_retry_left=state.no_comment_retries_left,
                    reasoning_effort=turn_reasoning,
                )
            except GenerationInterrupted:
                # Пользователь прервал выполнение инструмента: убираем висящий вызов
                # ассистента без результата, чтобы история осталась валидной для API.
                self.history.pop_pending_tool_calls()
                self.history.normalize()
                self._on_history_changed()
                self.on_system_msg("⏹ Generation stopped by user.")
                self._autosave()
                raise

            if rerun_prefill:
                # [NO COMMENT]: модель вызвала инструмент без текста. Перегенерируем
                # следующий ход с prefill, не добавляя пустой ответ в историю (лимит —
                # NO_COMMENT_RETRIES, задаётся в TurnState). Этот блок стоит ДО guard'а
                # pending-операции: иначе guard своим continue съел бы prefill и цикл шёл
                # бы впустую (без prefill). Когда попытки исчерпаны, _process_llm_response
                # возвращает None и вызов исполняется как есть.
                state.no_comment_retries_left -= 1
                state.set_prefill(rerun_prefill)
                continue

            if (self._pending_operation is not None
                    and not message_obj.tool_calls
                    and not tool_error_occurred):
                # Модель ответила текстом, не вызвав 'answer_to_system' — ход неудачный.
                # Стираем неудачный ответ и все предыдущие наги (как при зацикливании),
                # чтобы мусор не копился в контексте, и перегенерируем с повышенной
                # температурой.
                self._erase_last_assistant()
                self._drop_guard_nags()
                self.history.add(UserMessage(
                    f"{ENVIRONMENT_PREFIX} You can't continue unless you answer to the system question using '{answer_to_system.__name__}' tool. "
                    "Call it ritgh now!"
                    f"{ENVIRONMENT_PREFIX_END}"
                ))
                self.on_render(self.history.get_all()[-1])
                self._temp_override = Config.ERROR_RECOVERY_TEMP
                self._on_history_changed()
                continue

            state.clear_prefill()

            if broken_call:
                if state.can_retry_broken_regen():
                    self._erase_last_assistant()
                    self._recover('broken_call', retries_left=state.broken_regen_left)
                    continue
                if state.can_retry_broken_fix():
                    self.history.add(UserMessage(self._build_broken_call_fix()))
                    self.on_render(self.history.get_all()[-1])
                    self._recover('broken_fix', retries_left=state.broken_fix_left)
                    continue
                self._recover('broken_giveup')
                return ""

            if tool_error_occurred and state.can_retry_tool_error():
                erased = self._erase_last_failed_tool_call()
                self._recover('tool_error', retries_left=state.tool_error_retries_left, erased_count=erased)
                continue

            self._compact_completed_tasks()
            if (not getattr(self, '_auto_summarize_suppressed', False)
                    and self._get_context_usage_percent() >= self._current_summary_threshold()):
                if getattr(self, '_is_subagent', False):
                    # Суб-агент: авто-суммаризация бессмысленна (история — клон родителя,
                    # сжатие ломает извлечение ответа и тратит лишние вызовы LLM). При
                    # превышении порога просто завершаемся, возвращая последнее сообщение.
                    return result_text
                self._auto_summarize_dialogue()

            if tool_error_occurred:
                state.record_tool_error()
            else:
                state.record_tool_success()

            if state.max_errors_reached:
                self._recover('error_limit', retries_left=MAX_CONSECUTIVE_ERRORS)
                return ""

            self._autosave()

            if self.stop_event.is_set():
                self.on_system_msg("⏹ Generation stopped by user — control returned to you.")
                self.clear_stop()
                self._autosave()
                return result_text

            if not message_obj.tool_calls and not tool_error_occurred:
                return result_text