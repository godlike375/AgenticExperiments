"""Сессия группы инструментов: явное состояние, которое хост передаёт воркеру.

Изоляция инструментов (docs/TOOL_ISOLATION_SPEC.md): состояние группы хранит хост
(GroupManager), воркер ничего не удерживает между вызовами. Перед каждым вызовом
хост кладёт состояние группы в контекст (set_session), инструмент читает/меняет его
через get_session(), после вызова хост забирает обновлённое состояние.

ContextVar (а не модульная глобальная переменная) — потому что состояние явное и
ограничено одним вызовом; прямые вызовы инструментов (юнит-тесты, скрипты) без
сессии получают общий дефолтный словарь, повторяющий прежнее поведение модульных
глобал (например, screen_state.last_scale до первого скриншота).
"""

from __future__ import annotations

import contextvars
from typing import Optional

# Дефолтная сессия для прямых (неизолированных) вызовов инструментов вне хода агента.
_default_session: dict = {}


def get_session() -> dict:
    """Текущая сессия группы. Если контекст не установлен — общая дефолтная сессия
    (обратная совместимость с прямыми вызовами инструментов и юнит-тестами)."""
    session = _current_session.get()
    return session if session is not None else _default_session


def current_session_or_none() -> Optional[dict]:
    """Сессия, установленная на время вызова группы, или None вне хода агента.
    Инструменты, у которых исторически было модульное глобальное состояние, используют
    это, чтобы отличить изолированный вызов (сессия есть) от прямого (тесты)."""
    return _current_session.get()


def set_session(session: dict) -> contextvars.Token:
    """Устанавливает сессию на время вызова инструмента; возвращает токен для reset."""
    return _current_session.set(session)


def reset_session(token: contextvars.Token) -> None:
    """Снимает ранее установленную сессию (парный вызов к set_session)."""
    _current_session.reset(token)


_current_session: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar(
    "tool_group_session", default=None
)
