from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass, field, replace as dc_replace
from datetime import datetime
from types import SimpleNamespace
from typing import Iterable, Union, Callable, Optional

from universal_agents.constants import ENVIRONMENT_PREFIX, ENVIRONMENT_PREFIX_END, INTERRUPT_HEADER
from universal_agents.config import Config
from universal_agents.models import UserMessage, AssistantMessage, ToolResult
from universal_agents.llm_client import LLMClient, TokenUsageTracker, LoopDetector, jaccard_similarity, text_hash
from universal_agents.history import ChatHistory
from universal_agents.generation import GenerationParams, StructuredOutputConfig, apply_stop_markers
from universal_agents.tool_manager import ToolManager
from universal_agents.sub_agent import SubAgent
from universal_agents.context_builder import prepare_messages_for_api, get_effective_prefill
from universal_agents.file_states import FileStateTracker
from universal_agents.tool_parsing import tc_name, tc_args
from universal_agents.agent_mixins.response_mixin import SimReasoningGroup
from universal_agents.tools.builtin import respond_to_system

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

MAX_CONSECUTIVE_ERRORS = 10

_NO_TOOL_ERROR_SIGNATURE = "<no_tool_call>"

_CONFIRMATION_GUARD_VERDICT = "confirmation_guard"

_REPEAT_ANSWER_NAG = (
    f"\n\n{ENVIRONMENT_PREFIX} Your previous answer was the same to the latest one. "
    f"Please do NOT repeat it again and answer differently."
    f"{ENVIRONMENT_PREFIX_END}"
)

_GUARD_NAG_PREFIX = "You can't continue"

_SYSTEM_HEADER_RE = re.compile(r"^\{<SYSTEM>:.*?\}\n\n", re.DOTALL)


def _strip_system_header(content: Optional[str]) -> Optional[str]:
    if content is None:
        return None
    return _SYSTEM_HEADER_RE.sub("", content, count=1)


@dataclass
class TurnState:

    prefill: Optional[str] = None
    error_counts: dict[str, int] = field(default_factory=dict)
    tool_error_retries_left: int = field(default_factory=lambda: Config.ERROR_RECOVERY_RETRIES)
    broken_regen_left: int = field(default_factory=lambda: Config.BROKEN_CALL_REGEN_RETRIES)
    broken_fix_left: int = field(default_factory=lambda: Config.BROKEN_CALL_FIX_RETRIES)
    no_comment_retries_left: int = field(default_factory=lambda: Config.NO_COMMENT_RETRIES)
    guard_retries_left: int = field(default_factory=lambda: Config.ANSWER_GUARD_MAX_RETRIES)
    structured_output: Optional[StructuredOutputConfig] = None
    structured_output_phase: int = 0

    def step_prefill(self) -> Optional[str]:
        return self.prefill

    def set_prefill(self, value: Optional[str]) -> None:
        self.prefill = value

    def clear_prefill(self) -> None:
        self.prefill = None

    def can_retry_broken_regen(self) -> bool:
        if self.broken_regen_left > 0:
            self.broken_regen_left -= 1
            return True
        return False

    def can_retry_broken_fix(self) -> bool:
        if self.broken_fix_left > 0:
            self.broken_fix_left -= 1
            return True
        return False

    def can_retry_tool_error(self) -> bool:
        if self.tool_error_retries_left > 0:
            self.tool_error_retries_left -= 1
            return True
        return False

    def record_tool_error(self, signature: Optional[str] = None) -> None:
        """Сигнатура (инструмент+аргументы+ошибка) → свой счётчик; чередование [A,B,A,B] тоже лимит."""
        sig = signature if signature is not None else _NO_TOOL_ERROR_SIGNATURE
        self.error_counts[sig] = self.error_counts.get(sig, 0) + 1

    def record_tool_success(self) -> None:
        self.error_counts.clear()
        self.tool_error_retries_left = Config.ERROR_RECOVERY_RETRIES

    def reset_error_counts(self) -> None:
        """Сброс счётчиков после сжатия истории: отсчёт зацикливания заново."""
        self.error_counts.clear()

    def can_retry_answer_guard(self) -> bool:
        if self.guard_retries_left > 0:
            self.guard_retries_left -= 1
            return True
        return False

    @property
    def guard_retries_spent(self) -> int:
        return Config.ANSWER_GUARD_MAX_RETRIES - self.guard_retries_left

    @property
    def max_errors_reached(self) -> bool:
        return any(count >= MAX_CONSECUTIVE_ERRORS for count in self.error_counts.values())


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
        min_p: float = None,
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
        group_manager=None,
    ):
        self.history = ChatHistory(system_prompt)
        self.file_states = FileStateTracker(self.history)
        from universal_agents.archive import HistoryArchive
        self.archive = HistoryArchive()
        self._read_registrations: list[str] = []
        self._gen_params = GenerationParams.from_overrides(
            temp=temp,
            timeout=timeout,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            min_p=min_p,
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
        if token_tracker is not None:
            self.token_tracker = token_tracker
        else:
            self.token_tracker = TokenUsageTracker(system_prompt, max_context_tokens if max_context_tokens is not None else Config.MAX_CONTEXT_TOKENS)
        self.tools_manager = ToolManager(tools_config, external_plugins)
        if denied_tools:
            self.tools_manager.deny(denied_tools)
        from universal_agents.group_worker import GroupManager
        from universal_agents.project_root import get_project_root_override
        if group_manager is not None:
            self.group_manager = group_manager
        else:
            self.group_manager = GroupManager(
                isolated=Config.TOOL_ISOLATION_ENABLED,
                project_root=get_project_root_override(),
                handler_resolver=lambda n: self._all_tools.get(n, {}).get("handler"),
            )
            self.group_manager.proxy_handler = self._plugin_proxy_call
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
        self._guard_nag_message: Optional[UserMessage] = None
        self._confirmation_junk: dict[int, object] = {}

        self.autosave_enabled = Config.AUTOSAVE_ENABLED if autosave_enabled is None else autosave_enabled
        self.autosave_dir = Config.AUTOSAVE_DIR if autosave_dir is None else autosave_dir
        self.autosave_keep = Config.AUTOSAVE_KEEP if autosave_keep is None else autosave_keep
        self._autosave_path = None
        self.reset_autosave_path()
        self.stop_event = threading.Event()
        self.on_interrupt_check = on_interrupt_check if on_interrupt_check is not None else (lambda: False)
        self._stop_check = lambda: self.stop_event.is_set() or bool(self.on_interrupt_check())
        self.pending_interrupt: Optional[str] = None
        self._in_tool_execution = False
        self._in_service_turn = False

    @property
    def _per_msg_enabled(self) -> bool:
        """Гейт per-message суммаризации: выключена глобально/локально/в служебном ходе."""
        return (
            Config.PER_MSG_SUMMARIES_ENABLED
            and not self._disable_per_msg_summarization
            and not self._in_service_turn
        )

    @property
    def _reasoning_effort(self) -> str:
        """'low' если включён toggle или задан разовый; разовый потребляется один раз в начале хода."""
        if self._thinking_once or self._thinking_enabled:
            return "low"
        return "none"

    def _build_duplicate_warning(self, dup_name: str, dup_args: str) -> str:
        return (
            f"\n\n{ENVIRONMENT_PREFIX} Your previous attempt to call tool '{dup_name}' "
            f"with args '{dup_args}' was blocked as a duplicate of a recently already made call."
            f" Do NOT call '{dup_name}' again with same params. "
            f"Use other params or a different tool or write a text answer."
            f"{ENVIRONMENT_PREFIX_END}"
        )

    def _build_broken_call_fix(self) -> str:
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
        """Создаёт субагента: наследует промпт/историю/инструменты для KV-cache, свой трекер токенов."""
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
            group_manager=self.group_manager,
        )

    def _resolve_denied_tools(self, denied_tools) -> set[str]:
        if denied_tools is None:
            return set()
        if isinstance(denied_tools, str):
            if denied_tools in ("*", "all"):
                return set(self._all_tools.keys())
            return {denied_tools}
        return set(denied_tools)

    def _on_history_changed(self) -> None:
        """Очистка после мутации истории: сброс кэша чтений + flush отложенных выгрузок."""
        self.file_states.prune()
        self.tools_manager.flush_pending_unloads()

    def record_usage(self, usage) -> None:
        if usage:
            self.token_tracker.update_from_usage(usage)

    def set_pending_operation(self, op: dict) -> None:
        self._pending_operation = op

    def pop_pending_operation(self) -> Optional[dict]:
        op = self._pending_operation
        self._pending_operation = None
        return op

    def _ensure_guard_nag(self) -> None:
        """Один наг на цикл (§1.11): не пересоздаётся при повторах — байт-стабильный KV-префикс."""
        nag = self._guard_nag_message
        if nag is not None and any(m is nag for m in self.history.get_all()):
            return  # уже в истории — не трогаем
        if nag is None:
            nag = UserMessage(
                f"{ENVIRONMENT_PREFIX} {_GUARD_NAG_PREFIX} unless you answer to the system question using '{respond_to_system.__name__}' tool. "
                "Call it ritgh now!"
                f"{ENVIRONMENT_PREFIX_END}"
            )
            nag._is_guard_nag = True
            self._guard_nag_message = nag
        self.history.add(nag)
        self.on_render(self.history.get_all()[-1])

    def _build_save_extras(self) -> dict:
        from universal_agents.task_tracker import plan_state_to_dict
        from universal_agents.project_root import get_project_root_override
        return {
            "archive": self.archive.to_list(),
            "plan_state": plan_state_to_dict(self),
            "cwd": os.getcwd(),
            "project_root": get_project_root_override(),
        }

    def save_history(self, path: str, extras: dict = None) -> None:
        self.history.save(
            path,
            loaded_tools=list(self._all_tools.keys()),
            file_states=self.file_states.to_dict(),
            extras=extras if extras is not None else self._build_save_extras(),
        )

    def _rotate_autosaves(self) -> None:
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
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._autosave_path = os.path.join(self.autosave_dir, f"autosave_{ts}.json")

    def autosave(self) -> None:
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
        system_prompt = self.history[0].content if len(self.history) else ""
        self.history = ChatHistory(system_prompt)
        from universal_agents.archive import HistoryArchive
        self.archive = HistoryArchive()
        self.task_plan = []
        self.task_plan_map = {}
        self.file_states = FileStateTracker(self.history)
        self._tool_usage.clear()
        self.loop_detector = LoopDetector()
        self.token_tracker = TokenUsageTracker(
            system_prompt,
            self.token_tracker.max_context_tokens if self.token_tracker else Config.MAX_CONTEXT_TOKENS,
        )
        self._compacted_task_ids = set()
        self._pending_operation = None
        self._guard_nag_message = None
        self._confirmation_junk.clear()
        self.reset_autosave_path()
        self._on_history_changed()

    def _autosave(self) -> None:
        self.autosave()

    def close_tool_workers(self) -> None:
        try:
            self.group_manager.close_all()
        except Exception:
            pass

    def _plugin_proxy_call(self, method: str, args=(), kwargs=None):
        """Allow-list API для изолированных плагинов (§9): только явные методы, не объект агента."""
        from universal_agents.project_root import find_project_root

        allowed = {
            "on_system_msg": lambda self, text: self.on_system_msg(str(text)),
            "load_tool": lambda self, name: self.load_tool(str(name)),
            "unload_tool": lambda self, name: self.unload_tool(str(name)),
            "history_len": lambda self: len(self.history.get_all()),
            "project_root": lambda self: find_project_root(),
        }
        handler = allowed.get(method)
        if handler is None:
            raise ValueError(f"method '{method}' is not allowed via plugin proxy")
        return handler(self, *args, **(kwargs or {}))

    def request_stop(self) -> None:
        self.stop_event.set()

    def clear_stop(self) -> None:
        self.stop_event.clear()

    def set_pending_interrupt(self, text: str) -> None:
        self.pending_interrupt = text

    def take_pending_interrupt(self) -> Optional[str]:
        text = self.pending_interrupt
        self.pending_interrupt = None
        return text

    def clear_pending_interrupt(self) -> None:
        self.pending_interrupt = None


    def _call_llm(
        self,
        messages: list[dict],
        prefill: Optional[str] = None,
        params: GenerationParams = None,
        watch_prefix: Optional[str] = None,
        watch_continue_temp: Optional[float] = None,
        stop_check: Callable[[], bool] = None,
        reasoning_effort: str = "none",
        stop_markers: tuple = (),
        prefill_shown: int = 0,
        defer_stream_end: bool = False,
    ) -> tuple:
        """Единая точка транспорта: стриминг или обычный вызов; prefill_shown — уже показанный prefill."""
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
                stop_markers=stop_markers,
                prefill_shown=prefill_shown,
                defer_stream_end=defer_stream_end,
            )
        message_obj, err, usage = LLMClient.call(
            messages,
            tools=tools,
            prefill=prefill,
            params=params,
            stop_check=stop_check,
            reasoning_effort=reasoning_effort,
            stop_markers=stop_markers,
        )
        stopped_at_marker = bool(getattr(message_obj, "stopped_at_marker", False))
        if stop_markers and message_obj is not None and getattr(message_obj, "content", None):
            content, marker_hit = apply_stop_markers(message_obj.content, stop_markers)
            if marker_hit:
                message_obj.content = content
                stopped_at_marker = True
        return message_obj, err, usage, stopped_at_marker

    def service_llm_call(
        self,
        msgs: list[dict],
        temp: Optional[float] = None,
        timeout: Optional[int] = None,
        tools=True,
        prefill: Optional[str] = None,
        params: GenerationParams = None,
        structured_output: Optional[StructuredOutputConfig] = None,
    ) -> tuple:
        """Служебный вызов LLM = один ход через движок: промпт → UserMessage, инструменты не исполняются, откат по умолчанию."""
        prompt = None
        for m in reversed(msgs):
            if isinstance(m, dict) and m.get("role") == "user":
                prompt = m.get("content")
                break
        last_user = self.history.get_last_user_message()
        prompt_clean = _strip_system_header(prompt) if prompt else None
        if prompt_clean is not None and last_user is not None and (last_user.content or "") == prompt_clean:
            append_prompt = False  # промпт уже последнее сообщение в истории (consistency-драфт)
        else:
            append_prompt = True
        result = self._run_service_turn(
            prompt or "",
            append_prompt=append_prompt,
            temp=temp,
            timeout=timeout,
            prefill=prefill,
            params=params,
            structured_output=structured_output,
        )
        if result:
            return SimpleNamespace(content=result, tool_calls=[], reasoning_content=""), None, None
        return None, "Empty response", None

    def _run_service_turn(
        self,
        prompt: str,
        *,
        append_prompt: bool = True,
        keep_in_history: Optional[bool] = None,
        max_iter: Optional[int] = None,
        temp: Optional[float] = None,
        timeout: Optional[int] = None,
        prefill: Optional[str] = None,
        params: GenerationParams = None,
        structured_output: Optional[StructuredOutputConfig] = None,
    ) -> str:
        """Служебный ход: инструменты не исполняются, авто-сейв/компакция выключены, промпт откатывается если не keep."""
        keep = Config.SERVICE_TURN_KEEP_IN_HISTORY if keep_in_history is None else keep_in_history
        max_iter = max_iter if max_iter is not None else Config.SERVICE_TURN_MAX_ITER
        start_len = len(self.history.get_all())
        stream_swap = None
        if self.streaming_enabled and self.on_service_stream_chunk:
            stream_swap = (self.on_stream_start, self.on_stream_chunk, self.on_stream_end)
            self.on_stream_start = self.on_service_stream_start or (lambda: None)
            self.on_stream_chunk = self.on_service_stream_chunk
            self.on_stream_end = self.on_service_stream_end or (lambda: None)

        prev_suppressed = self._auto_summarize_suppressed
        self._auto_summarize_suppressed = True
        self._in_service_turn = True
        prev_params = self._gen_params
        if params is not None or temp is not None or timeout is not None:
            p = params or GenerationParams()
            self._gen_params = dc_replace(
                self._gen_params,
                temp=temp if temp is not None else (p.temp if p.temp is not None else self._gen_params.temp),
                timeout=timeout if timeout is not None else (p.timeout if p.timeout is not None else self._gen_params.timeout),
            )
        try:
            if append_prompt:
                self._append_service_prompt(prompt)
            result = self._run_turn_loop(
                max_iter,
                prefill=prefill,
                structured_output=structured_output,
                service_mode=True,
            ) or ""
        finally:
            if not keep:
                end = len(self.history.get_all())
                if end > start_len:
                    self.history.remove_at(range(start_len, end))
                    self.history.normalize()
                    self._on_history_changed()
            self._gen_params = prev_params
            self._auto_summarize_suppressed = prev_suppressed
            self._in_service_turn = False
            if stream_swap is not None:
                self.on_stream_start, self.on_stream_chunk, self.on_stream_end = stream_swap
        return result

    def _append_service_prompt(self, prompt: str) -> None:
        """Добавляет промпт как UserMessage, не трогая висящие tool_calls текущего хода."""
        last = self.history.get_last_message()
        if isinstance(last, UserMessage):
            self.history.add(AssistantMessage(content=""))
        self.history.add(UserMessage(content=prompt))
        self.clear_stop()

    def _get_prior_text_hashes(self) -> tuple[set[str], set[str]]:
        """Хэши всех AssistantMessage истории: повтор любого текста/reasoning = зацикливание (§1.9)."""
        min_chars = Config.DUPLICATE_TEXT_MIN_CHARS
        text_hashes: set[str] = set()
        reasoning_hashes: set[str] = set()
        for msg in self.history.get_all():
            if not isinstance(msg, AssistantMessage):
                continue
            content = (msg.content or "").strip()
            if content and (min_chars <= 0 or len(content) >= min_chars):
                text_hashes.add(text_hash(content))
            reasoning = (msg.reasoning_content or "").strip()
            if reasoning and (min_chars <= 0 or len(reasoning) >= min_chars):
                reasoning_hashes.add(text_hash(reasoning))
        return text_hashes, reasoning_hashes

    def _detect_duplicate(self, message_obj) -> Optional[tuple]:
        """Детектор дублей: повтор текста/reasoning/вызова. При живом pending guard решает вердикт (§1.11)."""
        content = (message_obj.content or "").strip()
        reasoning = (getattr(message_obj, 'reasoning_content', '') or '').strip()
        pending = self._pending_operation is not None

        if pending and message_obj.tool_calls and any(
            tc_name(tc) == respond_to_system.__name__ for tc in message_obj.tool_calls
        ):
            return None

        min_chars = Config.DUPLICATE_TEXT_MIN_CHARS
        if content or reasoning:
            prior_text, prior_reasoning = self._get_prior_text_hashes()
            if (
                content
                and (min_chars <= 0 or len(content) >= min_chars)
                and text_hash(content) in prior_text
            ):
                if pending and not message_obj.tool_calls:
                    return _CONFIRMATION_GUARD_VERDICT, content
                return 'answer', content
            if (
                reasoning
                and (min_chars <= 0 or len(reasoning) >= min_chars)
                and text_hash(reasoning) in prior_reasoning
            ):
                if pending and not message_obj.tool_calls:
                    return _CONFIRMATION_GUARD_VERDICT, reasoning
                return 'reasoning', reasoning

        if message_obj.tool_calls:
            current_history = self.history.get_all()
            seen_in_batch: set[str] = set()
            for tc in message_obj.tool_calls:
                norm = self.loop_detector.normalize_args(tc_args(tc))
                batch_key = f"{tc_name(tc)}:{norm}"
                if batch_key in seen_in_batch:
                    return 'tool_call', (tc_name(tc), tc_args(tc))
                seen_in_batch.add(batch_key)
                if self.loop_detector.check_duplicate_in_turn(tc_name(tc), tc_args(tc), current_history):
                    return 'tool_call', (tc_name(tc), tc_args(tc))
            return None

        answer_text = content
        prev_answer = self._get_last_answer_text()
        if (
            answer_text
            and prev_answer
            and jaccard_similarity(answer_text, prev_answer)
            >= Config.DUPLICATE_SIMILARITY_THRESHOLD
        ):
            if pending:
                return _CONFIRMATION_GUARD_VERDICT, prev_answer
            return 'answer', prev_answer
        return None

    def call_with_retries(
        self,
        messages_to_send: list[dict],
        step_prefill: Optional[str] = None,
        is_duplicate_fn: Callable = None,
        boost: bool = True,
        reasoning_effort: str = None,
        stop_markers: tuple = (),
        prefill_shown: int = 0,
        defer_stream_end: bool = False,
        gen_params: GenerationParams = None,
    ) -> tuple:
        """До max_generation_attempts попыток: дубликаты отбрасываются с бустом температуры."""
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
        dup_count = 0
        api_error_occurred = False
        stopped_at_marker = False
        effective_reasoning_effort = reasoning_effort if reasoning_effort is not None else self._reasoning_effort

        for attempt in range(max_generation_attempts):
            attempt_params = gen_params if gen_params is not None else self._gen_params
            if self._temp_override is not None:
                attempt_params = attempt_params.with_temp(self._temp_override)
                self._temp_override = None

            active_messages = [dict(msg) for msg in messages_to_send]
            if last_retry_warning and active_messages:
                last_msg = active_messages[-1]
                last_msg["content"] = (last_msg.get("content") or "") + last_retry_warning

            message_obj, err, usage, stopped_at_marker = self._call_llm(
                active_messages,
                prefill=step_prefill,
                params=attempt_params,
                watch_prefix=dup_watch_target,
                watch_continue_temp=Config.DUPLICATE_CONTINUATION_TEMP if dup_watch_target is not None else None,
                stop_check=self._stop_check,
                reasoning_effort=effective_reasoning_effort,
                stop_markers=stop_markers,
                prefill_shown=prefill_shown,
                defer_stream_end=defer_stream_end,
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

            # Детектор дублей проверяется ПЕРВЫМ: правильный respond_to_system
            # с повторённой «пластинкой»-шапкой иначе выбрасывался бы вместе
            # с вызовом, и подтверждение не наступало никогда (§1.11).
            duplicate = is_duplicate_fn(message_obj)
            if duplicate is not None and duplicate[0] == _CONFIRMATION_GUARD_VERDICT:
                duplicate = None
            if duplicate is not None:
                dup_count += 1
                kind, payload = duplicate
                if kind == 'tool_call':
                    dup_name, dup_args = payload
                    tag, reason = 'PROACTIVE LOOP DETECTED', f"Intercepted duplicate call to '{dup_name}'"
                elif kind == 'reasoning':
                    dup_watch_target = None
                    tag, reason = 'REPEATED REASONING DETECTED', 'Model repeated a reasoning block from history'
                else:
                    dup_watch_target = payload
                    tag, reason = 'DUPLICATE ANSWER DETECTED', 'Model repeated the previous answer verbatim'
                if dup_count >= Config.DUPLICATE_NAG_THRESHOLD:
                    last_retry_warning = _REPEAT_ANSWER_NAG if kind != 'tool_call' else self._build_duplicate_warning(dup_name, dup_args)
                else:
                    last_retry_warning = None
                if boost:
                    self._temp_override = Config.BOOST_TEMP
                self.on_system_msg(
                    f"[{tag}] {reason}. "
                    f"Discarding response. Activating temperature boost ({Config.BOOST_TEMP})"
                    + (f" and injecting temporary warning." if last_retry_warning else ".")
                    + f" Attempt {attempt + 1}/{max_generation_attempts}."
                )
                continue

            break
        else:
            message_obj = None
            self.on_system_msg(
                "[PROACTIVE LOOP DETECTOR] Max re-generation attempts reached. Handing control to user."
            )

        return message_obj, api_error_occurred, stopped_at_marker

    def _recover(self, kind: str, retries_left: int = 0, erased_count: int = 0) -> None:
        """Единая точка восстановления: буст температуры или сдача хода пользователю."""
        if kind == 'api_error':
            self.history.normalize(is_error_recovery=True)
            self.on_system_msg("⚠️ [RECOVERY] API error occurred. Role sequence restored. Handing control to user.")
            return
        if kind == 'duplicate_loop':
            self.history.normalize(is_error_recovery=True)
            self.on_system_msg(
                "⚠️ [LOOP DETECTED] Model keeps repeating itself after multiple attempts. "
                "Handing control to user."
            )
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
        if kind == 'answer_guard':
            self.history.normalize(is_error_recovery=True)
            self.on_system_msg(
                f"⚠️ [CONFIRMATION LOOP] Model kept replying in text instead of calling "
                f"{respond_to_system.__name__} ({Config.ANSWER_GUARD_MAX_RETRIES} guard retries exhausted). "
                "Handing control to user."
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

    def chat(self, message: str, max_iter: int = None, prefill: str = None,
             structured_output: Optional[StructuredOutputConfig] = None):
        max_iter = max_iter if max_iter is not None else Config.MAX_ITER
        if self.self_consistency_mode:
            return self._chat_self_consistent(message, prefill)
        self._auto_summarize_suppressed = False
        self._prepare_turn(message)
        return self._run_turn_loop(max_iter, prefill, structured_output=structured_output)

    def inject_user_interrupt(self, user_text: str, max_iter: int = None) -> str:
        """Прерывание: сценарий В — в ToolResult, А/Б — отдельный UserMessage; новая генерация."""
        max_iter = max_iter if max_iter is not None else Config.MAX_ITER
        if self.self_consistency_mode:
            self.clear_pending_interrupt()
            return self._chat_self_consistent(f"{INTERRUPT_HEADER}\n\n{user_text}", None)
        self.clear_pending_interrupt()
        if not self._append_interrupt_to_tool_result(user_text):
            self._prepare_turn(f"{INTERRUPT_HEADER}\n\n{user_text}")
        return self._run_turn_loop(max_iter)

    def _append_interrupt_to_tool_result(self, user_text: str) -> bool:
        """Сценарий В: дописывает шапку+текст в ToolResult, не ломая последовательность ролей."""
        last = self.history.get_last_message()
        if not isinstance(last, ToolResult):
            return False
        last.content = (last.content or "") + f"\n\n{INTERRUPT_HEADER}\n\n{user_text}"
        self._on_history_changed()
        self._autosave()
        self.clear_stop()
        return True

    def _prepare_turn(self, message: str, autosave: bool = True) -> None:
        """Вставка user-сообщения: после прерванного фрагмента; заглушка ассистента для ролей."""
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
        if autosave:
            self._autosave()
        self.clear_stop()

    def _run_turn_loop(self, max_iter: int, prefill: str = None,
                       structured_output: Optional[StructuredOutputConfig] = None,
                       service_mode: bool = False) -> str:
        current_prefill = get_effective_prefill(prefill)
        if structured_output is not None and structured_output.controller is not None:
            controller_prefill = structured_output.controller.initial_prefill()
            if controller_prefill is not None:
                current_prefill = controller_prefill
        turn_reasoning = self._reasoning_effort
        self._thinking_once = False
        state = TurnState(prefill=current_prefill, structured_output=structured_output)

        sim_group = SimReasoningGroup(Config.SIMULATED_REASONING_REPAIRS) if (
                Config.SIMULATED_REASONING_ENABLED
                and turn_reasoning == "none"
                and not service_mode
                and structured_output is None
                and not current_prefill
                and not getattr(self, "_is_subagent", False)
        ) else None
        sim_open = False
        sim_stream_started = False

        def _close_sim_stream():
            nonlocal sim_stream_started
            if sim_stream_started and self.on_stream_end:
                self.on_stream_end()
            sim_stream_started = False

        def _finish_turn(result: str) -> str:
            """Единая точка выхода: закрываем live-блок и сохраняем историю."""
            _close_sim_stream()
            if not service_mode:
                self._autosave()
            return result

        try:
            for _ in range(max_iter):
                if self.pending_interrupt is not None:
                    return _finish_turn("")

                if self.stop_event.is_set():
                    self.on_system_msg("⏹ Generation stopped by user — control returned to you.")
                    self.clear_stop()
                    return _finish_turn("")

                step_prefill = state.step_prefill()
                think_params = None
                if sim_group is not None:
                    if not sim_open:
                        sim_group.reset(Config.SIMULATED_REASONING_REPAIRS)
                        sim_open = True
                        sim_stream_started = False
                    step_prefill = sim_group.prefill()
                    if sim_group.is_think_phase():
                        think_params = sim_group.think_params(self._gen_params)

                if structured_output is not None:
                    effective_markers = structured_output.effective_markers(step_prefill)
                elif sim_group is not None and sim_group.is_think_phase():
                    effective_markers = (f"</{sim_group.tag}>",)
                else:
                    effective_markers = ()

                all_messages = prepare_messages_for_api(
                    self, debug_hash_check=Config.DEBUG_PREFIX_HASH_CHECK
                )

                try:
                    message_obj, api_error_occurred, stopped_at_marker = self.call_with_retries(
                        all_messages,
                        step_prefill=step_prefill,
                        reasoning_effort=turn_reasoning,
                        stop_markers=effective_markers,
                        prefill_shown=sim_group.prefill_shown() if sim_group else 0,
                        defer_stream_end=sim_group is not None,
                        gen_params=think_params,
                    )
                except GenerationInterrupted:
                    _close_sim_stream()
                    self.on_system_msg("⏹ Generation stopped by user — control returned to you.")
                    self.clear_stop()
                    return _finish_turn("")

                if api_error_occurred or not message_obj:
                    _close_sim_stream()
                    self._recover('api_error' if api_error_occurred else 'duplicate_loop')
                    return _finish_turn("")

                controller_verdict = None
                if (structured_output is not None
                        and structured_output.controller is not None
                        and message_obj is not None):
                    controller_verdict = structured_output.controller.advance(message_obj.content or "")
                    if controller_verdict.corrected_content != (message_obj.content or ""):
                        message_obj.content = controller_verdict.corrected_content

                if (controller_verdict is not None
                        and not controller_verdict.complete
                        and not controller_verdict.corrected_content
                        and controller_verdict.next_prefill is not None):
                    if state.structured_output_phase + 1 >= structured_output.max_phases:
                        return _finish_turn(structured_output.controller.document())
                    state.set_prefill(controller_verdict.next_prefill)
                    state.structured_output_phase += 1
                    continue

                sim_broken_scan = None
                if sim_group is not None and message_obj is not None:
                    sim_group.absorb(message_obj, step_prefill)
                    sim_stream_started = True
                    if sim_group.is_think_phase():
                        sim_group.close_section()
                        appended = sim_group.take_appended()
                        if appended and self.on_stream_chunk:
                            self.on_stream_chunk(appended)
                        if sim_group.tool_calls or sim_group.answer_text().strip():
                            sim_msg = self._sim_message(sim_group)
                        else:
                            sim_group.advance_to_answer()
                            continue
                    else:
                        sim_msg = self._sim_verify_phase(sim_group)
                        if sim_msg is None:
                            continue
                    sim_broken_scan = sim_group.answer_text()
                    message_obj = sim_msg
                    sim_open = False
                    _close_sim_stream()

                try:
                    result_text, tool_error_occurred, broken_call, rerun_prefill = self._process_llm_response(
                        message_obj,
                        no_comment_retry_left=state.no_comment_retries_left,
                        reasoning_effort=turn_reasoning,
                        skip_broken_detection=bool(
                            structured_output is not None
                            and (stopped_at_marker or structured_output.controller is not None)
                        ),
                        allow_tools=not service_mode,
                        broken_scan_text=sim_broken_scan,
                    )
                except GenerationInterrupted:
                    self.history.pop_pending_tool_calls()
                    self.history.normalize()
                    self._on_history_changed()
                    self.on_system_msg("⏹ Generation stopped by user.")
                    _close_sim_stream()
                    if not service_mode:
                        self._autosave()
                    raise

                if rerun_prefill:
                    state.no_comment_retries_left -= 1
                    state.set_prefill(rerun_prefill)
                    continue

                if (self._pending_operation is not None
                        and not service_mode
                        and not message_obj.tool_calls
                        and not tool_error_occurred):
                    self._erase_last_assistant()
                    if not state.can_retry_answer_guard():
                        self._recover('answer_guard')
                        return _finish_turn("")
                    self.on_system_msg(
                        f"⏳ [CONFIRMATION PENDING] The model replied with text instead of "
                        f"calling {respond_to_system.__name__} — waiting for the tool call "
                        f"(attempt {state.guard_retries_spent}/{Config.ANSWER_GUARD_MAX_RETRIES})."
                    )
                    self._ensure_guard_nag()
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
                    return _finish_turn("")

                if tool_error_occurred and state.can_retry_tool_error():
                    erased = self._erase_last_failed_tool_call()
                    self._recover('tool_error', retries_left=state.tool_error_retries_left,
                                  erased_count=erased)
                    continue

                compacted = False
                if not service_mode:
                    compacted = self._compact_completed_tasks() > 0
                if (
                        not service_mode
                        and not getattr(self, '_auto_summarize_suppressed', False)
                        and self._get_context_usage_percent() >= self._current_summary_threshold()
                ):
                    if getattr(self, '_is_subagent', False):
                        return _finish_turn(result_text)
                    if self._auto_summarize_dialogue():
                        compacted = True
                        self._auto_summarize_suppressed = True

                if compacted:
                    state.reset_error_counts()

                if tool_error_occurred:
                    state.record_tool_error(self._last_failed_tool_signature())
                else:
                    state.record_tool_success()

                if state.max_errors_reached:
                    self._recover('error_limit', retries_left=MAX_CONSECUTIVE_ERRORS)
                    return _finish_turn("")

                if not service_mode:
                    self._autosave()

                if self.stop_event.is_set():
                    self.on_system_msg("⏹ Generation stopped by user — control returned to you.")
                    self.clear_stop()
                    return _finish_turn(result_text)

                if (
                        structured_output is not None
                        and structured_output.controller is not None
                        and controller_verdict is not None
                ):
                    if controller_verdict.complete:
                        return _finish_turn(structured_output.controller.document())
                    if controller_verdict.next_prefill is not None:
                        if state.structured_output_phase + 1 >= structured_output.max_phases:
                            return _finish_turn(structured_output.controller.document())
                        state.set_prefill(controller_verdict.next_prefill)
                        state.structured_output_phase += 1
                        continue
                    if stopped_at_marker and not message_obj.tool_calls and not tool_error_occurred:
                        state.structured_output_phase += 1
                        continue
                    if not message_obj.tool_calls and not tool_error_occurred:
                        return _finish_turn(structured_output.controller.document())

                if (
                        structured_output is not None
                        and stopped_at_marker
                        and not message_obj.tool_calls
                        and not tool_error_occurred
                        and state.structured_output_phase < len(structured_output.next_prefills)
                        and state.structured_output_phase + 1 < structured_output.max_phases
                ):
                    state.set_prefill(structured_output.next_prefills[state.structured_output_phase])
                    state.structured_output_phase += 1
                    continue

                if not message_obj.tool_calls and not tool_error_occurred:
                    return _finish_turn(result_text)

            return _finish_turn("")
        finally:
            _close_sim_stream()