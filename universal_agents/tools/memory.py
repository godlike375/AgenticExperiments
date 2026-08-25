"""Инструменты долговременной памяти: поиск и чтение архива вытесненных
оригиналов + закрепление фактов в STATE.

Архив наполняется автоматически при каждой компакции (MemoryMixin), поэтому
модель может ответить на вопрос пользователя про любой фрагмент, уже удалённый
из контекста, не полагаясь на собственную дисциплину ведения заметок.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from universal_agents.tool import tool
from universal_agents.constants import ENVIRONMENT_PREFIX, ENVIRONMENT_PREFIX_END, err

if TYPE_CHECKING:
    from universal_agents.agent import LLMAgent


@tool(
    description=(
        "Search the session ARCHIVE of messages that were compressed out of the context. "
        "Use when the user asks about something said/done earlier that is no longer in context. "
        "Returns matching entries with seq numbers; use recall_read to get full originals."
    ),
    short_description="search archived history",
    query=("str", "Words or phrase to search for (identifiers, paths, error text)"),
    role=("str", "Optional filter: user | assistant | tool"),
    tool_name=("str", "Optional filter by tool name for role=tool"),
    limit=("int", "Max matches to show"),
)
def recall_search(agent: LLMAgent, query: str, role: str = "", tool_name: str = "", limit: int = 5) -> str:
    if not hasattr(agent, "archive"):
        return err(": archive is not available in this agent.")
    result = agent.archive.search(query, role=role, tool_name=tool_name, limit=limit)
    return f"{ENVIRONMENT_PREFIX}\n{result}\n{ENVIRONMENT_PREFIX_END}"


@tool(
    description=(
        "Read the FULL original messages seq range from the session archive "
        "(messages previously compressed out of context). Boundaries are inclusive."
    ),
    short_description="read archived originals",
    from_seq=("int", "First message seq"),
    to_seq=("int", "Last message seq (inclusive)"),
)
def recall_read(agent: LLMAgent, from_seq: int, to_seq: int) -> str:
    if not hasattr(agent, "archive"):
        return err(": archive is not available in this agent.")
    result = agent.archive.read_span(from_seq, to_seq)
    return f"{ENVIRONMENT_PREFIX}\n{result}\n{ENVIRONMENT_PREFIX_END}"
