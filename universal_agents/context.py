
"""Протокол AgentContext — минимальный контракт агента для миксинов и хелперов.
Используется только для статической типизации (TYPE_CHECKING); в рантайме — структурная совместимость."""
from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from universal_agents.history import ChatHistory
    from universal_agents.tool_manager import ToolManager
    from universal_agents.llm_client import TokenUsageTracker, LoopDetector
    from universal_agents.file_states import FileStateTracker
    from universal_agents.archive import HistoryArchive
    from universal_agents.generation import GenerationParams


@runtime_checkable
class AgentContext(Protocol):

    history: "ChatHistory"
    tools_manager: "ToolManager"
    token_tracker: "TokenUsageTracker"
    loop_detector: "LoopDetector"
    file_states: "FileStateTracker"
    archive: "HistoryArchive"
    _gen_params: "GenerationParams"
    _all_tools: dict
    _compacted_task_ids: set
    _read_registrations: list
    _auto_summarize_suppressed: bool

    task_plan: list
    task_plan_map: dict

    on_system_msg: callable
    on_render: callable

    stop_event: threading.Event
    temp: float
    timeout: Optional[int]

    @property
    def _per_msg_enabled(self) -> bool: ...

    def _on_history_changed(self) -> None:
        ...

    def service_llm_call(self, *args, **kwargs):
        ...

    def load_tool(self, name: str) -> str:
        ...

    def make_sub_agent(self, *args, **kwargs):
        ...

    def pop_pending_operation(self) -> Optional[dict]:
        ...