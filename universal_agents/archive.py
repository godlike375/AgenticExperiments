"""Архив сессии: полные оригиналы сообщений, вытесненных из контекста компакцией.

Инвариант: сжатие истории ничего не уничтожает, а выселяет. Каждый оригинал
дописывается сюда ДО замены диапазона блоками памяти, поэтому агент всегда
может ответить на вопрос пользователя про уже удалённый из контекста фрагмент:
recall_search находит запись по словам, recall_read возвращает дословный кусок.

Адресация — по стабильному Message.seq (переживает /save + /load), не по
индексам списка, которые плавают при каждой модификации истории.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from universal_agents.config import Config

if TYPE_CHECKING:
    from universal_agents.models import Message


class HistoryArchive:
    def __init__(self, entries: list[dict] | None = None):
        self.entries: list[dict] = entries or []

    # ------------------------------------------------------------------
    # Наполнение
    # ------------------------------------------------------------------

    def append_messages(self, msgs: list["Message"]) -> int:
        added = 0
        for m in msgs:
            seq = getattr(m, "seq", None)
            entry: dict = {
                "seq": seq,
                "ts": getattr(m.timestamp, "isoformat", lambda: "")(),
                "role": _role_of(m),
                "content": _content_of(m),
            }
            name = getattr(m, "name", None)
            if isinstance(name, str) and name:
                entry["name"] = name
            tool_calls = []
            for tc in (getattr(m, "tool_calls", None) or []):
                tool_calls.append({
                    "name": tc.name,
                    "arguments": (tc.arguments or "")[:Config.RECALL_MAX_ARG_CHARS],
                })
            if tool_calls:
                entry["tool_calls"] = tool_calls
            self.entries.append(entry)
            added += 1
        return added

    def __len__(self) -> int:
        return len(self.entries)

    def to_list(self) -> list[dict]:
        return self.entries

    @classmethod
    def from_list(cls, data: list[dict] | None) -> "HistoryArchive":
        return cls(list(data) if data else [])

    # ------------------------------------------------------------------
    # Поиск и чтение
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        role: str = "",
        tool_name: str = "",
        limit: int = 5,
    ) -> str:
        """Поиск по архиву. Возвращает отформатированный список находок c seq,
        ролью и сниппетом; результат готов к вставке в контекст."""
        words = [w.lower() for w in re.findall(r"\w+", query or "") if len(w) >= 2]
        if not words:
            return "[recall_search] Empty or too short query."
        limit = max(1, min(int(limit or 5), 20))

        scored: list[tuple[int, int, dict]] = []
        needle = " ".join(words)
        for idx, e in enumerate(self.entries):
            if role and e.get("role") != role.lower():
                continue
            if tool_name and e.get("name", "").lower() != tool_name.lower():
                continue
            text = e.get("content", "").lower()
            hit_words = {w for w in words if w in text}
            if not hit_words:
                continue
            score = len(hit_words) / len(words)
            if needle in text:
                score += 1.0
            scored.append((score, idx, e))

        if not scored:
            return (
                f"[recall_search] No matches in archive ({len(self.entries)} entries) "
                f"for query: '{query}'."
            )
        scored.sort(key=lambda t: (-t[0], -t[1]))

        lines = [
            f"[recall_search] Found {len(scored)} match(es) for '{query}', top {min(limit, len(scored))}:"
        ]
        for score, _, e in scored[:limit]:
            snippet = _snippet(e.get("content", ""), words, Config.RECALL_SNIPPET_CHARS)
            label = f"seq={e.get('seq')}"
            if e.get("role"):
                label += f" {e['role'].upper()}"
            if e.get("name"):
                label += f"[{e['name']}]"
            ts = str(e.get("ts", ""))[:19]
            lines.append(f"- {label} ({ts}): …{snippet}…")
        lines.append("Use recall_read(from_seq, to_seq) to get the full original messages.")
        return "\n".join(lines)

    def read_span(self, from_seq: int, to_seq: int, max_chars: int | None = None) -> str:
        """Дословный кусок переписки по диапазону seq (границы включительно)."""
        max_chars = max_chars or Config.RECALL_READ_MAX_CHARS
        try:
            lo = min(int(from_seq), int(to_seq))
            hi = max(int(from_seq), int(to_seq))
        except (TypeError, ValueError):
            return "[recall_read] from_seq/to_seq must be integers."

        selected = [e for e in self.entries
                    if isinstance(e.get("seq"), int) and lo <= e["seq"] <= hi]
        if not selected:
            known = [e["seq"] for e in self.entries if isinstance(e.get("seq"), int)]
            if known:
                return (
                    f"[recall_read] No archived messages with seq in [{lo}..{hi}]. "
                    f"Archived range covers seq {min(known)}..{max(known)}."
                )
            return "[recall_read] Archive is empty."
        # Порядок записей в архиве — порядок вытеснения; блоки памяти получают
        # seq позже хвоста, поэтому для чтения сортируем по seq явно.
        selected.sort(key=lambda e: e["seq"])

        parts: list[str] = []
        used = 0
        truncated = False
        for e in selected:
            line = _render_entry(e)
            if used + len(line) > max_chars and parts:
                truncated = True
                break
            parts.append(line)
            used += len(line) + 1

        header = f"[recall_read] Original messages seq {lo}..{hi} ({len(parts)} shown):"
        tail = "\n…[truncated; narrow the range]" if truncated else ""
        return header + "\n" + "\n".join(parts) + tail


# ---------------------------------------------------------------------------
# Хелперы
# ---------------------------------------------------------------------------


def _role_of(m) -> str:
    from universal_agents.models import (
        SystemMessage, UserMessage, AssistantMessage, ToolResult,
    )
    if isinstance(m, SystemMessage):
        return "system"
    if isinstance(m, UserMessage):
        return "user"
    if isinstance(m, AssistantMessage):
        return "assistant"
    if isinstance(m, ToolResult):
        return "tool"
    return type(m).__name__


def _content_of(m) -> str:
    from universal_agents.models import AssistantMessage, ToolResult

    content = (getattr(m, "content", "") or "").strip()
    if isinstance(m, ToolResult):
        prefix = f"{m.name}: " if m.name else ""
        return prefix + content
    if isinstance(m, AssistantMessage) and getattr(m, "tool_calls", None):
        calls = "; ".join(f"{tc.name}({tc.arguments})" for tc in m.tool_calls)
        return f"{content}\n[called: {calls}]".strip()
    return content


def _render_entry(e: dict) -> str:
    role = e.get("role", "?")
    seq = e.get("seq")
    name = f"[{e['name']}]" if e.get("name") else ""
    ts = str(e.get("ts", ""))[11:19]
    body = e.get("content", "")
    limit = Config.RECALL_ENTRY_MAX_CHARS
    if len(body) > limit:
        cut = body[:limit]
        nl = cut.rfind("\n")
        if nl > limit // 2:
            cut = cut[:nl]
        body = cut + f"\n…[entry truncated at {limit} chars]"
    return f"#{seq} {role.upper()}{name} ({ts}): {body}"


def _snippet(text: str, words: list[str], size: int) -> str:
    lower = text.lower()
    pos = -1
    for w in words:
        pos = lower.find(w)
        if pos != -1:
            break
    if pos == -1:
        return text[:size].replace("\n", " ")
    start = max(0, pos - size // 3)
    end = min(len(text), start + size)
    chunk = text[start:end].replace("\n", " ")
    return chunk.strip()
