from __future__ import annotations

import hashlib
import os
from typing import TYPE_CHECKING, Optional, Dict

if TYPE_CHECKING:
    from universal_agents.history import ChatHistory


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


class FileStateTracker:

    def __init__(self, history: 'ChatHistory' = None):
        self._history = history
        self._states: Dict[str, dict] = {}

    def _key(self, path: str) -> str:
        return os.path.abspath(path)

    def record(self, path: str, disk_hash: str, content_hash: str) -> None:
        self._states[self._key(path)] = {
            "disk_hash": disk_hash,
            "content_hash": content_hash,
            "tool_call_id": None,
        }

    def mark_tool_call(self, path: str, tool_call_id: str) -> None:
        st = self._states.get(self._key(path))
        if st:
            st["tool_call_id"] = tool_call_id

    def _find_read_result(self, tool_call_id: str):
        from universal_agents.tools.fs import read as _read_tool

        if not self._history or not tool_call_id:
            return None
        for msg in self._history.get_all():
            if (
                getattr(msg, 'name', None) == _read_tool.__name__
                and getattr(msg, 'tool_call_id', None) == tool_call_id
            ):
                return msg
        return None

    def _is_history_intact(self, entry: dict) -> bool:
        result = self._find_read_result(entry["tool_call_id"])
        if result is None:
            return False
        return _content_hash(result.content) == entry["content_hash"]

    def should_skip(self, path: str, disk_hash: str) -> bool:
        entry = self._states.get(self._key(path))
        if entry is None or disk_hash != entry["disk_hash"]:
            return False
        return self._is_history_intact(entry)

    def prune(self) -> None:
        stale = [
            k for k, v in self._states.items()
            if v.get("tool_call_id") and not self._is_history_intact(v)
        ]
        for k in stale:
            del self._states[k]

    def clear(self) -> None:
        self._states.clear()

    def to_dict(self) -> dict:
        return {k: dict(v) for k, v in self._states.items()}

    def from_dict(self, data: Optional[dict]) -> None:
        self._states = {}
        if not data:
            return
        for k, v in data.items():
            self._states[k] = {
                "disk_hash": v.get("disk_hash", ""),
                "content_hash": v.get("content_hash", ""),
                "tool_call_id": v.get("tool_call_id"),
            }

    def __len__(self) -> int:
        return len(self._states)
