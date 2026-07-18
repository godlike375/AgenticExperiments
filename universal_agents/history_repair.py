from __future__ import annotations
from typing import TYPE_CHECKING

from universal_agents.tool import ENVIRONMENT_PREFIX
from universal_agents.config import Config
from universal_agents.models import UserMessage, AssistantMessage, ToolResult

if TYPE_CHECKING:
    from agent import LLMAgent


def break_tool_loop(agent: LLMAgent, tool_name: str, norm_args: str, count: int) -> None:
    """Удаляет повторяющиеся вызовы одного и того же инструмента с одинаковыми аргументами."""
    messages = agent.history._messages
    matched_indices: list[int] = []
    duplicate_tool_ids: set[str] = set()

    for i in range(Config.AFTER_SYSTEM_PROMPT, len(messages)):
        msg = messages[i]
        if not isinstance(msg, AssistantMessage):
            continue
        for tc in msg.tool_calls:
            if (tc.name == tool_name
                    and agent.loop_detector.normalize_args(tc.arguments) == norm_args):
                matched_indices.append(i)
                if len(matched_indices) > 1:
                    duplicate_tool_ids.add(tc.id)

    if len(matched_indices) <= 1:
        return

    to_remove = set(matched_indices[1:])
    for i, msg in enumerate(messages):
        if isinstance(msg, ToolResult) and msg.tool_call_id in duplicate_tool_ids:
            to_remove.add(i)

    for idx in sorted(to_remove, reverse=True):
        del messages[idx]

    warning = (
        f"{ENVIRONMENT_PREFIX} Tool '{tool_name}' with identical args was called a few times. "
        f"The system removed all duplicates."
    )
    messages.append(UserMessage(warning))
    agent.on_system_msg(f"[LOOP DETECTED] '{tool_name}' ×{count}")


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
        for idx in sorted(indices_to_remove, reverse=True):
            del agent.history._messages[idx]
        agent.on_system_msg(
            f"[CLEANUP] Removed {len(indices_to_remove)} messages "
            f"({len(indices_to_remove) // 2} failed calls)"
        )
        agent.history.normalize()
