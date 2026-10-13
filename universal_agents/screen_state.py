"""Общее состояние PC-инструментов (скриншот/мышь).

Масштаб последнего скриншота нужен mouse-инструментам для конвертации координат
image→screen. В изолированном режиме (docs/TOOL_ISOLATION_SPEC.md) состояние живёт
в сессии группы desktop (session.last_scale); при прямых вызовах инструментов вне
хода агента сессии нет — используется модульная переменная last_scale (прежнее
поведение, на которое опираются юнит-тесты).
"""

from __future__ import annotations

from universal_agents import tool_session

# Фолбэк для прямых (неизолированных) вызовов: screen_state.last_scale.
last_scale: float | None = None


def get_last_scale() -> float | None:
    """Масштаб последнего скриншота: из сессии desktop, иначе модульный фолбэк."""
    session = tool_session.current_session_or_none()
    if session is not None:
        return session.get("last_scale")
    return last_scale


def set_last_scale(value: float | None) -> None:
    """Сохраняет масштаб в сессию desktop, иначе — в модульный фолбэк."""
    session = tool_session.current_session_or_none()
    if session is not None:
        session["last_scale"] = value
    else:
        global last_scale
        last_scale = value

