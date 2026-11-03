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

from universal_agents.history import ChatHistory
from universal_agents.llm_client import LLMClient, TokenUsageTracker, LoopDetector
from universal_agents.models import AssistantMessage, ToolCall, UserMessage
from universal_agents.generation import GenerationParams, StructuredOutputConfig
from universal_agents.tool_manager import ToolManager
from universal_agents.callbacks import AgentCallbacks
from universal_agents.tool_pipeline import ToolPipeline
from universal_agents.compressors import SummaryService
from universal_agents.session_storage import SessionStorage
from universal_agents.recovery_policy import RecoveryPolicy
from universal_agents.turn_executor import TurnExecutor


class LLMAgent:
    """Главный фасад агента: оркестрация TurnExecutor, RecoveryPolicy и SessionStorage."""

    def __init__(
        self,
        system_prompt: str = "You are a helpful assistant",
        temp: float = None,
        timeout: int = None,
        tools_config: Union[list[str], dict, None] = None,
        on_render: Callable = None,
        on_confirm: Callable[[str, dict], bool] = None,
        on_system_msg: Callable[[str], None] = None,
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
        callbacks: Optional[AgentCallbacks] = None,
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
        self.callbacks = callbacks or AgentCallbacks(
            on_render=on_render if on_render is not None else (lambda x: None),
            on_confirm=on_confirm if on_confirm is not None else (lambda n, a: True),
            on_system_msg=on_system_msg if on_system_msg is not None else (lambda x: None),
            on_stream_chunk=on_stream_chunk,
            on_stream_start=on_stream_start,
            on_stream_end=on_stream_end,
            on_reasoning_chunk=on_reasoning_chunk,
            on_reasoning_start=on_reasoning_start,
            on_reasoning_end=on_reasoning_end,
            on_service_stream_start=on_service_stream_start,
            on_service_stream_chunk=on_service_stream_chunk,
            on_service_stream_end=on_service_stream_end,
            on_interrupt_check=on_interrupt_check if on_interrupt_check is not None else (lambda: False),
        )
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
        self.tools_manager.auto_trust_git_root()
        self.loop_detector = LoopDetector()
        self._temp_override: Optional[float] = None
        self._original_temp = temp
        self._thinking_enabled: bool = True
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
        self.tool_pipeline = ToolPipeline(self)

        self.stop_event = threading.Event()
        self._stop_check = lambda: self.stop_event.is_set() or bool(self.callbacks.on_interrupt_check())
        self.pending_interrupt: Optional[str] = None
        self._in_tool_execution = False
        self._in_service_turn = False


    # --- Свойства обратных вызовов (AgentCallbacks) ---
    @property
    def on_render(self):
        return self.callbacks.on_render

    @on_render.setter
    def on_render(self, val):
        self.callbacks.on_render = val

    @property
    def on_confirm(self):
        return self.callbacks.on_confirm

    @on_confirm.setter
    def on_confirm(self, val):
        self.callbacks.on_confirm = val

    @property
    def on_system_msg(self):
        return self.callbacks.on_system_msg

    @on_system_msg.setter
    def on_system_msg(self, val):
        self.callbacks.on_system_msg = val

    @property
    def on_stream_chunk(self):
        return self.callbacks.on_stream_chunk

    @on_stream_chunk.setter
    def on_stream_chunk(self, val):
        self.callbacks.on_stream_chunk = val

    @property
    def on_stream_start(self):
        return self.callbacks.on_stream_start

    @on_stream_start.setter
    def on_stream_start(self, val):
        self.callbacks.on_stream_start = val

    @property
    def on_stream_end(self):
        return self.callbacks.on_stream_end

    @on_stream_end.setter
    def on_stream_end(self, val):
        self.callbacks.on_stream_end = val

    @property
    def on_reasoning_chunk(self):
        return self.callbacks.on_reasoning_chunk

    @on_reasoning_chunk.setter
    def on_reasoning_chunk(self, val):
        self.callbacks.on_reasoning_chunk = val

    @property
    def on_reasoning_start(self):
        return self.callbacks.on_reasoning_start

    @on_reasoning_start.setter
    def on_reasoning_start(self, val):
        self.callbacks.on_reasoning_start = val

    @property
    def on_reasoning_end(self):
        return self.callbacks.on_reasoning_end

    @on_reasoning_end.setter
    def on_reasoning_end(self, val):
        self.callbacks.on_reasoning_end = val

    @property
    def on_service_stream_start(self):
        return self.callbacks.on_service_stream_start

    @on_service_stream_start.setter
    def on_service_stream_start(self, val):
        self.callbacks.on_service_stream_start = val

    @property
    def on_service_stream_chunk(self):
        return self.callbacks.on_service_stream_chunk

    @on_service_stream_chunk.setter
    def on_service_stream_chunk(self, val):
        self.callbacks.on_service_stream_chunk = val

    @property
    def on_service_stream_end(self):
        return self.callbacks.on_service_stream_end

    @on_service_stream_end.setter
    def on_service_stream_end(self, val):
        self.callbacks.on_service_stream_end = val

    @property
    def on_interrupt_check(self):
        return self.callbacks.on_interrupt_check

    @on_interrupt_check.setter
    def on_interrupt_check(self, val):
        self.callbacks.on_interrupt_check = val



    def _call_with_streaming(self, *args, **kwargs):
        return self.turn_executor.call_with_streaming(*args, **kwargs)

    def _process_llm_response(self, *args, **kwargs):
        return self.turn_executor.process_llm_response(*args, **kwargs)

    def _chat_self_consistent(self, message: str, prefill: str = None) -> str:
        return self.turn_executor.chat_self_consistent(message, prefill)

    def rebuild_tool_usage(self) -> None:
        self.tool_pipeline.rebuild_tool_usage()

    # --- Делегирование инструментов (бывший ToolsMixin) ---
    @property
    def _all_tools(self) -> dict:
        return self.tools_manager.tools_map

    @property
    def tools(self) -> list[dict]:
        return self.tools_manager.schemas

    @property
    def _tools_config(self):
        return self.tools_manager.config

    @property
    def trusted_dirs(self) -> set[str]:
        return self.tools_manager.trusted_dirs

    @property
    def trusted_files(self) -> set[str]:
        return self.tools_manager.trusted_files

    def load_tool(self, name: str) -> str:
        return self.tools_manager.load(name)

    def unload_tool(self, name: str) -> str:
        return self.tools_manager.unload(name)

    def is_tool_denied(self, name: str) -> bool:
        return self.tools_manager.is_denied(name)

    def list_available_tools(self) -> str:
        return self.tools_manager.list_available()

    def trust_dir(self, path: str) -> str:
        return self.tools_manager.trust_dir(path)

    def untrust_dir(self, path: str) -> str:
        return self.tools_manager.untrust_dir(path)

    def is_path_trusted(self, path: str) -> bool:
        return self.tools_manager.is_path_trusted(path)

    def is_file_trusted(self, path: str) -> bool:
        return self.tools_manager.is_file_trusted(path)

    def get_context_usage_percent(self) -> float:
        total = self.token_tracker.get_total_context_tokens()
        return (total / self.token_tracker.max_context_tokens) * 100

    def _get_context_usage_percent(self) -> float:
        return self.get_context_usage_percent()

    def _auto_summarize_dialogue(self, force: bool = False) -> bool:
        return SummaryService.compact_segment(self, force=force)

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

    def save_history(self, path: str, extras: dict = None) -> None:
        self.session_storage.save_history(path, extras)

    def reset_autosave_path(self) -> None:
        self.session_storage.reset_autosave_path()

    def autosave(self) -> None:
        self.session_storage.autosave()

    def reset_dialog(self) -> None:
        self.session_storage.reset_dialog()

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

    def _call_llm(self, *args, **kwargs) -> tuple:
        return self.turn_executor.call_llm(*args, **kwargs)

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

    def call_with_retries(self, messages_to_send: list[dict], **kwargs) -> tuple:
        return self.turn_executor.call_with_retries(messages_to_send, **kwargs)

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
