from __future__ import annotations
from typing import Optional, TYPE_CHECKING

from universal_agents.constants import ENVIRONMENT_PREFIX
from universal_agents.models import SystemMessage, UserMessage, AssistantMessage, ToolResult

if TYPE_CHECKING:
    from universal_agents.agent import LLMAgent


def _format_timestamp_header(msg) -> str:
    """Метка времени из timestamp сообщения."""
    ts = msg.timestamp.strftime("%Y-%m-%d %H:%M:%S")
    return '{' + f"{ENVIRONMENT_PREFIX}:\n{ts} \n"


def _format_token_header(tracker, first_system_message: str = "", last_user_content: str = "") -> str:
    """Только информация о токенах (с учётом последнего сообщения)."""
    total = tracker.get_total_context_tokens(first_system_message, last_user_content)
    remaining = tracker.max_context_tokens - total
    return f"Memory: {remaining} tokens left"


def _format_closing_header() -> str:
    """Закрывающая часть заголовка."""
    return " }\n\n"


def prepare_messages_for_api(agent: LLMAgent) -> list[dict]:
    """Готовит историю диалога для отправки в API."""
    agent.history.normalize()
    api_messages: list[dict] = []

    # «Последним user-сообщением» для токен-заголовка считаем живое сообщение
    # пользователя, а не служебные блоки памяти (STATE/EPISODES).
    last_user_idx = None
    last_user_msg = None
    for i in range(len(agent.history) - 1, -1, -1):
        msg = agent.history[i]
        if isinstance(msg, UserMessage):
            last_user_idx = i
            last_user_msg = msg
            break

    for i, msg in enumerate(agent.history):
        if isinstance(msg, SystemMessage):
            api_messages.append(msg.to_api_dict())
        elif isinstance(msg, UserMessage):
            if msg._cached_header is None:
                header = _format_timestamp_header(msg)
                if i == last_user_idx and last_user_msg:
                    header += _format_token_header(
                        agent.token_tracker, agent.history[0].content, last_user_msg.content
                    )
                header += _format_closing_header()
                msg._cached_header = header
            api_messages.append({
                "role": "user",
                "content": msg._cached_header + msg.content,
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
