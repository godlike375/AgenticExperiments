from __future__ import annotations

import json
import os
import threading
from typing import Callable, Iterable, Optional, Union

from universal_agents.config import Config
from universal_agents.constants import (
    ENVIRONMENT_PREFIX,
    ENVIRONMENT_PREFIX_END,
)

MAX_CONSECUTIVE_ERRORS = 10
_GUARD_NAG_PREFIX = "You can't continue" 
from universal_agents.history import ChatHistory
from universal_agents.llm_client import LLMClient
from universal_agents.generation import apply_stop_markers
from universal_agents.llm_client import TokenUsageTracker, LoopDetector
from universal_agents.models import AssistantMessage, ToolCall, UserMessage
from universal_agents.generation import GenerationParams, StructuredOutputConfig
from universal_agents.tool_manager import ToolManager
from universal_agents.tools.builtin import respond_to_system, apply_edit

from universal_agents.agent_mixins import (
    ConsistencyMixin,
    ExecuteMixin,
    HistoryMixin,
    MemoryMixin,
    ResponseMixin,
    StreamingMixin,
    ToolsMixin,
)

from universal_agents.session_storage import SessionStorage
from universal_agents.recovery_policy import (
    RecoveryPolicy,
    _CONFIRMATION_GUARD_VERDICT,
    _REPEAT_ANSWER_NAG,
)
from universal_agents.turn_executor import TurnExecutor, TurnState


class LLMAgent(
    ToolsMixin,
    MemoryMixin,
    HistoryMixin,
    StreamingMixin,
    ResponseMixin,
    ExecuteMixin,
    ConsistencyMixin,
):
    """Главный фасад агента: оркестрация TurnExecutor, RecoveryPolicy и SessionStorage."""

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
            self.token_tracker = TokenUsageTracker(
                system_prompt,
                max_context_tokens if max_context_tokens is not None else Config.MAX_CONTEXT_TOKENS,
            )
        self.tools_manager = ToolManager(tools_config, external_plugins)
        if denied_tools:
            self.tools_manager.deny(denied_tools)
        from universal_agents.tool_worker import ToolWorker

        if group_manager is not None:
            self.tool_worker = group_manager
        else:
            self.tool_worker = ToolWorker(
                isolated=Config.TOOL_ISOLATION_ENABLED,
            )
        self.group_manager = self.tool_worker
        self._auto_trust_git_root()
        self.loop_detector = LoopDetector()
        self._temp_override: Optional[float] = None
        self._original_temp = temp
        self._thinking_enabled: bool = True
        self.on_stream_chunk = on_stream_chunk
        self.on_stream_start = on_stream_start
        self.on_stream_end = on_stream_end
        self.on_reasoning_chunk = on_reasoning_chunk
        self.on_reasoning_start = on_reasoning_start
        self.on_reasoning_end = on_reasoning_end
        self.on_service_stream_start = on_service_stream_start
        self.on_service_stream_chunk = on_service_stream_chunk
        self.on_service_stream_end = on_service_stream_end
        self.streaming_enabled = (
            streaming_enabled if streaming_enabled is not None else Config.STREAM_ENABLED
        )
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

        self.session_storage = SessionStorage(
            self,
            autosave_dir=Config.AUTOSAVE_DIR if autosave_dir is None else autosave_dir,
            autosave_keep=Config.AUTOSAVE_KEEP if autosave_keep is None else autosave_keep,
            enabled=Config.AUTOSAVE_ENABLED if autosave_enabled is None else autosave_enabled,
        )
        self.recovery_policy = RecoveryPolicy(self)
        self.turn_executor = TurnExecutor(self)

        self.stop_event = threading.Event()
        self.on_interrupt_check = on_interrupt_check if on_interrupt_check is not None else (lambda: False)
        self._stop_check = lambda: self.stop_event.is_set() or bool(self.on_interrupt_check())
        self.pending_interrupt: Optional[str] = None
        self._in_tool_execution = False
        self._in_service_turn = False

    @property
    def autosave_enabled(self) -> bool:
        return self.session_storage.autosave_enabled

    @autosave_enabled.setter
    def autosave_enabled(self, val: bool) -> None:
        self.session_storage.autosave_enabled = val

    @property
    def autosave_dir(self) -> str:
        return self.session_storage.autosave_dir

    @autosave_dir.setter
    def autosave_dir(self, val: str) -> None:
        self.session_storage.autosave_dir = val

    @property
    def autosave_keep(self) -> int:
        return self.session_storage.autosave_keep

    @autosave_keep.setter
    def autosave_keep(self, val: int) -> None:
        self.session_storage.autosave_keep = val

    @property
    def _autosave_path(self) -> Optional[str]:
        return self.session_storage._autosave_path

    @_autosave_path.setter
    def _autosave_path(self, val: Optional[str]) -> None:
        self.session_storage._autosave_path = val

    @property
    def _per_msg_enabled(self) -> bool:
        return (
            Config.PER_MSG_SUMMARIES_ENABLED
            and not self._disable_per_msg_summarization
            and not getattr(self, "_in_service_turn", False)
        )

    @property
    def _reasoning_effort(self) -> str:
        if self._thinking_enabled:
            return "low"
        return "none"

    def _build_duplicate_warning(self, kind: str, duplicate_target: str) -> str:
        return self.recovery_policy.build_duplicate_warning(kind, duplicate_target)

    def _build_broken_call_fix(self) -> str:
        return self.recovery_policy.build_broken_call_fix()

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
    ):
        """Создаёт субагента: наследует промпт/историю/инструменты для KV-cache, свой трекер токенов."""
        from universal_agents.sub_agent import SubAgent
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
        if hasattr(self, "_cached_system_message"):
            delattr(self, "_cached_system_message")

    def record_usage(self, usage: dict) -> None:
        if self.token_tracker and usage:
            self.token_tracker.update_from_usage(usage)

    def set_pending_operation(self, op: Union[dict, Callable]) -> None:
        if isinstance(op, dict):
            self._pending_operation = op
        else:
            self._pending_operation = {"resolve": op}

    def pop_pending_operation(self) -> Optional[dict]:
        op = self._pending_operation
        self._pending_operation = None
        self._guard_nag_message = None
        return op

    def _ensure_guard_nag(self) -> None:
        self.recovery_policy.ensure_guard_nag()

    def _build_save_extras(self) -> dict:
        return self.session_storage.build_save_extras()

    def save_history(self, path: str, extras: dict = None) -> None:
        self.session_storage.save_history(path, extras)

    def _rotate_autosaves(self) -> None:
        self.session_storage.rotate_autosaves()

    def reset_autosave_path(self) -> None:
        self.session_storage.reset_autosave_path()

    def autosave(self) -> None:
        self.session_storage.autosave()

    def reset_dialog(self) -> None:
        self.session_storage.reset_dialog()

    def _autosave(self) -> None:
        self.session_storage.autosave()

    def close_tool_workers(self) -> None:
        try:
            if self.tool_worker is not None:
                self.tool_worker.close()
        except Exception as e:
            self.on_system_msg(f"[ToolWorker] Close error: {e}")

    def request_stop(self) -> None:
        self.stop_event.set()
        if self.tool_worker is not None:
            self.tool_worker.kill()

    def clear_stop(self) -> None:
        self.stop_event.clear()

    def set_pending_interrupt(self, text: str) -> None:
        self.pending_interrupt = text

    def take_pending_interrupt(self) -> Optional[str]:
        val = self.pending_interrupt
        self.pending_interrupt = None
        return val

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
        message_obj, err_val, usage = LLMClient.call(
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
        return message_obj, err_val, usage, stopped_at_marker

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
        return self.turn_executor.service_llm_call(
            msgs,
            temp=temp,
            timeout=timeout,
            tools=tools,
            prefill=prefill,
            params=params,
            structured_output=structured_output,
        )

    def _run_service_turn(self, prompt: str, **kwargs) -> str:
        return self.turn_executor.run_service_turn(prompt, **kwargs)

    def _append_service_prompt(self, prompt: str) -> None:
        self.turn_executor.append_service_prompt(prompt)

    def _consecutive_repeats(self, text: str, *, reasoning: bool) -> int:
        return self.recovery_policy.consecutive_repeats(text, reasoning=reasoning)

    def _recent_answer_texts(self, limit: int = 2) -> list[str]:
        return self.recovery_policy.recent_answer_texts(limit)

    def _last_assistant_message(self):
        return self.recovery_policy.last_assistant_message()

    def _call_signature(self, tool_calls) -> tuple:
        return self.recovery_policy.call_signature(tool_calls)

    @staticmethod
    def _actual_tool_call(tc) -> tuple[str, str]:
        return RecoveryPolicy.actual_tool_call(tc)

    @staticmethod
    def _normalize_dispatch_arguments(args: dict) -> str:
        return json.dumps(args, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def _detect_duplicate(self, message_obj) -> Optional[tuple]:
        return self.recovery_policy.detect_duplicate(message_obj)

    def call_with_retries(self, messages_to_send: list[dict], **kwargs) -> tuple:
        return self.turn_executor.call_with_retries(messages_to_send, **kwargs)

    def _recover(self, kind: str, retries_left: int = 0, erased_count: int = 0) -> None:
        self.recovery_policy.recover(kind, retries_left, erased_count)

    def chat(
        self,
        message: str,
        max_iter: int = None,
        prefill: str = None,
        structured_output: Optional[StructuredOutputConfig] = None,
    ):
        max_iter = max_iter if max_iter is not None else Config.MAX_ITER
        if self.self_consistency_mode:
            return self._chat_self_consistent(message, prefill)
        self._auto_summarize_suppressed = False
        self._prepare_turn(message)
        return self._run_turn_loop(max_iter, prefill, structured_output=structured_output)

    def inject_user_interrupt(self, user_text: str, max_iter: int = None) -> str:
        return self.turn_executor.inject_user_interrupt(user_text, max_iter)

    def _append_interrupt_to_tool_result(self, user_text: str) -> bool:
        return self.turn_executor.append_interrupt_to_tool_result(user_text)

    def _prepare_turn(self, message: str, autosave: bool = True) -> None:
        self.turn_executor.prepare_turn(message, autosave)

    def _run_turn_loop(self, max_iter: int, prefill: str = None, **kwargs) -> str:
        return self.turn_executor.run_turn_loop(max_iter, prefill=prefill, **kwargs)
