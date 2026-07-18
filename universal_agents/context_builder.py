from __future__ import annotations
from typing import Optional, TYPE_CHECKING

from universal_agents.models import SystemMessage, UserMessage, AssistantMessage, ToolResult

if TYPE_CHECKING:
    from universal_agents.agent import LLMAgent


def prepare_messages_for_api(agent: LLMAgent) -> list[dict]:
    """Готовит историю диалога для отправки в API."""
    agent.history.normalize()
    api_messages: list[dict] = []

    last_user_idx = None
    last_user_msg = None
    for i in range(len(agent.history) - 1, -1, -1):
        if isinstance(agent.history[i], UserMessage):
            last_user_idx = i
            last_user_msg = agent.history[i]
            break

    for i, msg in enumerate(agent.history):
        if isinstance(msg, SystemMessage):
            api_messages.append(msg.to_api_dict())
        elif isinstance(msg, UserMessage):
            header = agent.token_tracker.format_timestamp_header(msg)
            if i == last_user_idx and last_user_msg:
                header += agent.token_tracker.format_token_header(
                    agent.history[0].content, last_user_msg.content
                )
            header += agent.token_tracker.format_closing_header()
            api_messages.append({
                "role": "user",
                "content": header + msg.content,
            })
        elif isinstance(msg, AssistantMessage):
            api_messages.append(msg.to_api_dict())
        elif isinstance(msg, ToolResult):
            api_messages.append(msg.to_api_dict())
    return api_messages


def get_effective_prefill(custom_prefill: Optional[str]) -> Optional[str]:
    """Возвращает prefill, если задан."""
    if custom_prefill:
        return custom_prefill
    return None
