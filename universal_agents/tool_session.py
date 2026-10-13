
"""Сессия группы инструментов: явное состояние через ContextVar (§1.13).
Воркер ничего не хранит — состояние приходит с каждым вызовом."""
from __future__ import annotations

import contextvars
from typing import Optional

_default_session: dict = {}


def get_session() -> dict:
    """Текущая сессия группы; без контекста — общая дефолтная (тесты/прямые вызовы)."""
    session = _current_session.get()
    return session if session is not None else _default_session


def current_session_or_none() -> Optional[dict]:
    return _current_session.get()


def set_session(session: dict) -> contextvars.Token:
    return _current_session.set(session)


def reset_session(token: contextvars.Token) -> None:
    _current_session.reset(token)


_current_session: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar(
    "tool_group_session", default=None
)
