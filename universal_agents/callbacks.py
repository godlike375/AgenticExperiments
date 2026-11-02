from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class AgentCallbacks:
    """Группа функций обратного вызова для стриминга, UI и системных событий."""

    on_stream_start: Optional[Callable[[], None]] = None
    on_stream_chunk: Optional[Callable[[str], None]] = None
    on_stream_end: Optional[Callable[[], None]] = None

    on_reasoning_start: Optional[Callable[[], None]] = None
    on_reasoning_chunk: Optional[Callable[[str], None]] = None
    on_reasoning_end: Optional[Callable[[], None]] = None

    on_service_stream_start: Optional[Callable[[], None]] = None
    on_service_stream_chunk: Optional[Callable[[str], None]] = None
    on_service_stream_end: Optional[Callable[[], None]] = None

    on_render: Callable[[Any], None] = lambda _: None
    on_confirm: Callable[[str, dict], bool] = lambda name, args: True
    on_system_msg: Callable[[str], None] = lambda msg: None
    on_interrupt_check: Callable[[], bool] = lambda: False
