"""Прокси-агент для группы плагинов (docs/TOOL_ISOLATION_SPEC.md §9, этап 7).

Внешние плагины изолированы в отдельной группе `plugin`: произвольный код нельзя
аудить, поэтому он живёт в убиваемом воркер-процессе. Если плагину нужен агент
(история, load/unload, системные сообщения), он обращается к нему не напрямую, а
через этот прокси: вызов уходит по межпроцессному каналу в хост, где ограниченный
набор методов агента исполняется и результат возвращается обратно.

Плагин вызывает::

    from universal_agents import plugin_proxy
    plugin_proxy.call("on_system_msg", "hi from plugin")

Вне воркера плагинов прокси недоступен (нет канала) — вызов бросает RuntimeError.
"""

from __future__ import annotations

import itertools
import threading
from typing import Optional

__all__ = ["call", "set_channel"]


class _Channel:
    """Канал плагин→хост: запрос в response-очередь воркера, ответ — из request-очереди."""

    def __init__(self, request_queue, response_queue):
        self._request_queue = request_queue
        self._response_queue = response_queue
        self._lock = threading.Lock()
        self._counter = itertools.count(1)

    def request(self, method: str, args: tuple, kwargs: dict):
        with self._lock:
            call_id = next(self._counter)
            self._response_queue.put(("proxy", call_id, method, list(args), dict(kwargs)))
            while True:
                item = self._request_queue.get()
                if item is None:
                    raise RuntimeError("host closed the plugin worker")
                if isinstance(item, tuple) and item and item[0] == "proxy_reply" and item[1] == call_id:
                    status, payload = item[2], item[3]
                    if status == "error":
                        raise RuntimeError(payload)
                    return payload


_channel: Optional[_Channel] = None


def set_channel(channel: _Channel) -> None:
    """Устанавливает канал (вызывает воркер плагинов при старте)."""
    global _channel
    _channel = channel


def call(method: str, *args, **kwargs):
    """Вызвать ограниченный метод агента-хоста через прокси. Бросает RuntimeError вне воркера."""
    if _channel is None:
        raise RuntimeError(
            "plugin proxy is not available: this code is not running inside a plugin worker"
        )
    return _channel.request(method, args, kwargs)
