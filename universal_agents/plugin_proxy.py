
from __future__ import annotations

import itertools
import threading
from typing import Optional

__all__ = ["call", "set_channel"]


class _Channel:

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
    global _channel
    _channel = channel


def call(method: str, *args, **kwargs):
    if _channel is None:
        raise RuntimeError(
            "plugin proxy is not available: this code is not running inside a plugin worker"
        )
    return _channel.request(method, args, kwargs)
