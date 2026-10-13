
from __future__ import annotations

from universal_agents import tool_session

last_scale: float | None = None


def get_last_scale() -> float | None:
    session = tool_session.current_session_or_none()
    if session is not None:
        return session.get("last_scale")
    return last_scale


def set_last_scale(value: float | None) -> None:
    session = tool_session.current_session_or_none()
    if session is not None:
        session["last_scale"] = value
    else:
        global last_scale
        last_scale = value

