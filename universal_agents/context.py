"""Протокол AgentContext — минимальный контракт агента, который используют миксины и хелперы.

Пути для GUI/альтернативных реализаций агента: любая реализация, реализующая этот протокол,
может быть использована миксинами и хелперами без жёсткой зависимости от LLMAgent.
Используется только для статической типизации (TYPE_CHECKING); в рантайме — структурная
совместимость (Protocol).
"""

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
    """Структурный интерфейс агента для миксинов и хелперов.

    Список полей — это union того, что реально используют agent_mixins/*.py,
    compressors.py, task_tracker.py, context_builder.py и tools/*.py.
    """

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
        """Вызывается после любой мутации истории (сброс кэша файлов + flush выгрузок)."""
        ...

    def service_llm_call(self, *args, **kwargs):
        """Служебный вызов LLM (саммаризация, компактизация, consistency)."""
        ...

    def load_tool(self, name: str) -> str:
        """Включить ранее отключённый инструмент; возвращает строку-статус."""
        ...

    def make_sub_agent(self, *args, **kwargs):
        """Создать изолированный субагент с подмножеством инструментов и собственной историей."""
        ...

    def pop_pending_operation(self) -> Optional[dict]:
        """Извлечь и удалить отложенную операцию подтверждения модели."""
        ...