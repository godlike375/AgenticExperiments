from __future__ import annotations
from typing import TYPE_CHECKING

from universal_agents.config import Config
from universal_agents.models import AssistantMessage, ToolResult

if TYPE_CHECKING:
    from universal_agents.agent import LLMAgent


def prune_all_failed_tool_calls_except_last(agent: LLMAgent) -> None:
    """Удаляет все ошибочные вызовы инструментов, кроме последнего."""
    if len(agent.history) <= Config.AFTER_SYSTEM_PROMPT + 1:
        return

    last_assistant_idx = -1
    for i in range(len(agent.history) - 1, Config.AFTER_SYSTEM_PROMPT - 1, -1):
        if isinstance(agent.history[i], AssistantMessage):
            last_assistant_idx = i
            break
    if last_assistant_idx == -1:
        return

    indices_to_remove: set[int] = set()
    i = Config.AFTER_SYSTEM_PROMPT
    while i < len(agent.history):
        msg = agent.history[i]
        if isinstance(msg, AssistantMessage) and msg.has_tool_calls():
            if (i + 1 < len(agent.history)
                    and isinstance(agent.history[i + 1], ToolResult)):
                tool_result = agent.history[i + 1]
                if tool_result.is_error and not tool_result.is_user_denied:
                    if i < last_assistant_idx:
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
