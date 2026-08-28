from __future__ import annotations
import hashlib
import json
from typing import Optional, TYPE_CHECKING

from universal_agents.constants import ENVIRONMENT_PREFIX
from universal_agents.config import Config
from universal_agents.models import SystemMessage, UserMessage, AssistantMessage, ToolResult

if TYPE_CHECKING:
    from universal_agents.agent import LLMAgent


def _hash_api_message(d) -> str:
    """Стабильный хэш сериализованного представления, уходящего в LLM.

    Покрывает всё содержимое сообщения (role, content, tool_calls, заголовки
    user-сообщений и т.п.), а также — при передаче списка — схемы инструментов.
    """
    try:
        s = json.dumps(d, ensure_ascii=False, sort_keys=True)
    except TypeError:
        s = repr(d)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _check_prefix_hashes(agent: "LLMAgent", pairs: list) -> None:
    """Сравнивает хеши сообщений по id с прошлой итерацией. Изменение → сброс KV-кэша.
     Также проверяет tools и model — их смена тоже ломает кэш. Хранит состояние
     в _prev_prefix_hashes. Новые сообщения игнорируются.
    """
    prev = getattr(agent, "_prev_prefix_hashes", None)
    if prev is None:
        prev = {}
    curr: dict = {}
    changed: list = []
    for idx, (msg_obj, api_dict) in enumerate(pairs):
        mid = id(msg_obj)
        h = _hash_api_message(api_dict)
        curr[mid] = (idx, api_dict.get("role"), h)
        old = prev.get(mid)
        if old is not None and old[2] != h:
            content = (api_dict.get("content") or "")
            changed.append((idx, api_dict.get("role"), content))

    # Аргументы запроса, влияющие на реальный префикс KV-кэша.
    tools = getattr(agent, "tools", None)
    tools_hash = _hash_api_message(tools) if tools is not None else ""
    model = Config.MODEL_NAME
    model_hash = _hash_api_message({"model": model})
    curr["__tools__"] = (None, "tools", tools_hash)
    curr["__model__"] = (None, "model", model_hash)
    old_tools = prev.get("__tools__")
    if old_tools is not None and old_tools[2] != tools_hash:
        n = len(tools) if tools else 0
        changed.append((None, "tools", f"число схем={n}"))
    old_model = prev.get("__model__")
    if old_model is not None and old_model[2] != model_hash:
        changed.append((None, "model", model))

    agent._prev_prefix_hashes = curr
    for idx, role, content in changed:
        snippet = (content or "")[:80]
        loc = f"# {idx}" if idx is not None else role
        agent.on_system_msg(
            f"[PREFIX-HASH] ⚠️ {loc} (role={role}) изменилось между "
            f"итерациями подготовки → KV-кэш сбросится с этой точки. "
            f"Контент: {snippet!r}"
        )


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


def prepare_messages_for_api(agent: LLMAgent, normalize: bool = True,
                             debug_hash_check: bool = False) -> list[dict]:
    """Готовит историю для API.
`normalize=False` не сбрасывает кэш заголовков user-сообщений (для переиспользования KV).
`debug_hash_check` — сверяет хеши существующих сообщений и выводит предупреждение при изменении.
    """
    if normalize:
        agent.history.normalize()
    api_pairs: list = []  # (объект сообщения, api-словарь)

    # «Последним user-сообщением» для токен-заголовка считаем живое, а не служебные блоки памяти.
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
            api_pairs.append((msg, msg.to_api_dict()))
        elif isinstance(msg, UserMessage):
            if msg._cached_header is None:
                header = _format_timestamp_header(msg)
                if i == last_user_idx and last_user_msg:
                    header += _format_token_header(
                        agent.token_tracker, agent.history[0].content, last_user_msg.content
                    )
                header += _format_closing_header()
                msg._cached_header = header
            api_pairs.append((msg, {
                "role": "user",
                "content": msg._cached_header + msg.content,
            }))
        elif isinstance(msg, AssistantMessage):
            api_pairs.append((msg, msg.to_api_dict()))
        elif isinstance(msg, ToolResult):
            api_pairs.append((msg, msg.to_api_dict()))

    api_messages = [p[1] for p in api_pairs]
    if debug_hash_check:
        _check_prefix_hashes(agent, api_pairs)
    return api_messages


def get_effective_prefill(custom_prefill: Optional[str]) -> Optional[str]:
    """Возвращает prefill, если задан."""
    if custom_prefill:
        return custom_prefill
    return None
