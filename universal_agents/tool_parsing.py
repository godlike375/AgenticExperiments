
from __future__ import annotations

import json
import re

from universal_agents.constants import ENVIRONMENT_PREFIX


def tc_name(tc) -> str:
    func = getattr(tc, 'function', None)
    if func is not None:
        return getattr(func, 'name', None) or ""
    return getattr(tc, 'name', None) or ""


def tc_args(tc) -> str:
    func = getattr(tc, 'function', None)
    if func is not None:
        return getattr(func, 'arguments', None) or ""
    return getattr(tc, 'arguments', None) or ""


def is_error_content(content: str) -> bool:
    s = content.strip()
    while s.startswith(ENVIRONMENT_PREFIX):
        s = s[len(ENVIRONMENT_PREFIX):].strip()
    return s.startswith("Error")


_STRONG_TAGS = ("tool_call", "tool", "function", "call")


_XML_NAME = r'[a-zA-Z_:][a-zA-Z0-9_:.\-]*'

_OPEN_TAG_RE = re.compile(rf'<\s*{_XML_NAME}(?=[\s/>]|$)')


def _has_tag(content: str, tag: str | None = None) -> bool:
    if tag is None:
        return _OPEN_TAG_RE.search(content) is not None
    return re.search(
        rf'<\s*{re.escape(tag)}(?=[\s/>]|$)', content
    ) is not None


_ANY_TAG_RE = re.compile(r'<[^>]*>')


def _strip_tags(content: str) -> str:
    return _ANY_TAG_RE.sub(" ", content)


def _is_standalone_call(content: str, tool_names) -> bool:
    """True, если весь ответ — один вызов известного инструмента (заведомо сломанный синтаксис)."""
    stripped = content.strip()
    if not stripped:
        return False
    return any(
        re.fullmatch(rf'[*\-\u2022>#"\'\]<\[\(]*\s*{re.escape(name)}\s*[\(\{{].*[\)\}}]\s*[.!;:]?',
                     stripped, re.DOTALL)
        for name in tool_names
    )


def detect_broken_call(content: str, tool_names: set[str], require_tag: bool = True) -> bool:
    """True, если ответ похож на нераспарсенный вызов. require_tag=False снимает базовый гейт (симуляция reasoning)."""
    if not content or not content.strip():
        return False

    if any(_has_tag(content, t) for t in _STRONG_TAGS):
        return True

    if _is_standalone_call(content, tool_names):
        return True

    # Обычный ответ с процитированным тегом (напр. <SYSTEM>) и упоминанием инструмента
    # со скобками — не сломанный вызов. Ловим только вызов, обёрнутый в неизвестный тег.
    if require_tag and _has_tag(content):
        return _is_standalone_call(_strip_tags(content), tool_names)

    return False


def build_tool_calls(tool_calls_data: dict) -> list:
    from universal_agents.models import ToolCall

    return [
        ToolCall(
            id=tc_data["id"],
            name=tc_data["function"]["name"],
            arguments=tc_data["function"]["arguments"]
        )
        for tc_data in tool_calls_data.values()
    ]


def try_parse_tool_args(args_str: str):
    if not args_str or args_str.strip() in ("{}", "", "null"):
        return None
    try:
        parsed = json.loads(args_str)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def parse_tool_args(args_str: str) -> dict:
    parsed = try_parse_tool_args(args_str)
    return parsed if parsed is not None else {}


def args_are_valid(args_str: str) -> bool:
    s = (args_str or "").strip()
    if not s or s in ("{}", "null"):
        return True
    try:
        json.loads(s)
        return True
    except Exception:
        return False


def normalize_args(args_str: str) -> str:
    if not args_str or (isinstance(args_str, str) and args_str.strip() in ("{}", "", "null")):
        return ""
    try:
        if isinstance(args_str, dict):
            parsed = args_str
        else:
            parsed = json.loads(args_str)
        return json.dumps(parsed, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    except Exception:
        return str(args_str).strip()