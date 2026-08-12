from __future__ import annotations
from typing import TYPE_CHECKING

from universal_agents.config import Config
from universal_agents.models import AssistantMessage, ToolResult

if TYPE_CHECKING:
    from universal_agents.agent import LLMAgent


def prune_all_failed_tool_calls_except_last(agent: LLMAgent) -> None:
    """Удаляет предыдущий ошибочный вызов, если сразу после него идёт ещё один неудачный вызов.

    Ошибочный вызов очищается только при цепочке "неудачный -> неудачный". При сценарии
    "неудачный -> удачный" предыдущая неудача сохраняется: после удачного вызова может быть
    большой вывод инструмента, и удаление неудачи перед ним сбросило бы KV-кэш, тогда как
    ошибки обычно короткие и сброс не так дорог.
    """
    if len(agent.history) <= Config.AFTER_SYSTEM_PROMPT + 1:
        return

    def is_failed_call(idx: int) -> bool:
        if idx + 1 >= len(agent.history):
            return False
        msg = agent.history[idx]
        if not isinstance(msg, AssistantMessage) or not msg.has_tool_calls():
            return False
        tool_result = agent.history[idx + 1]
        return isinstance(tool_result, ToolResult) and tool_result.is_error and not tool_result.is_user_denied

    indices_to_remove: set[int] = set()
    i = Config.AFTER_SYSTEM_PROMPT
    while i < len(agent.history):
        if is_failed_call(i) and is_failed_call(i + 2):
            indices_to_remove.add(i)
            indices_to_remove.add(i + 1)
            i += 2
            continue
        i += 1

    if indices_to_remove:
        agent.history.remove_at(indices_to_remove)
        agent.on_system_msg(
            f"[CLEANUP] Removed {len(indices_to_remove)} messages "
            f"({len(indices_to_remove) // 2} failed calls)"
        )
        agent.history.normalize()
