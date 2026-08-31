"""Чистые функции работы с tool call: извлечение имени/аргументов, распознавание сломанных вызовов, парсинг и нормализация JSON-аргументов. Без зависимости от LLMAgent (нет циклических импортов)."""

from __future__ import annotations

import json
import re

from universal_agents.constants import ENVIRONMENT_PREFIX


def tc_name(tc) -> str:
    """Имя tool call независимо от формата (OpenAI/Responses-парсинг)."""
    func = getattr(tc, 'function', None)
    if func is not None:
        return getattr(func, 'name', None) or ""
    return getattr(tc, 'name', None) or ""


def tc_args(tc) -> str:
    """Аргументы tool call независимо от формата (OpenAI/Responses-парсинг)."""
    func = getattr(tc, 'function', None)
    if func is not None:
        return getattr(func, 'arguments', None) or ""
    return getattr(tc, 'arguments', None) or ""


def is_error_content(content: str) -> bool:
    """True, если вывод инструмента — ошибка (по конвенции начинается с 'Error')."""
    s = content.strip()
    while s.startswith(ENVIRONMENT_PREFIX):
        s = s[len(ENVIRONMENT_PREFIX):].strip()
    return s.startswith("Error")


# Сильные признаки XML-вызова: известные имена тегов
_STRONG_TAGS = ("tool_call", "tool", "function", "call")


def detect_broken_call(content: str, tool_names: set[str]) -> bool:
    """True, если ответ похож на нераспарсенный вызов инструмента. Базовый гейт — наличие XML-тега; далее: известные теги (<tool_call>/<tool>/<function>/<call>) либо тег + вызов известного инструмента (напр. read({...})). Без тега не детектится (нет ложных срабатываний)."""
    if not content or not content.strip():
        return False

    # Базовый гейт: хотя бы один открывающийся xml-тег
    if not re.search(r'<\s*[a-zA-Z_][\w-]*', content):
        return False

    # 1) известные имена тегов (полное совпадение имени тега, без продолжения словом)
    has_strong_tag = any(
        re.search(rf'<{re.escape(t)}(?!\w)', content) for t in _STRONG_TAGS
    )
    if has_strong_tag:
        return True

    # 2) тег уже есть (базовый гейт пройден) + признак вызова известного инструмента
    has_tool_call_sign = any(
        re.search(rf'\b{re.escape(name)}\s*[\({{]', content) for name in tool_names
    )
    return bool(has_tool_call_sign)


def build_tool_calls(tool_calls_data: dict) -> list:
    """Собирает ToolCall'ы из накопленных данных стрима."""
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
    """Парсит строку JSON-аргументов. Возвращает dict или None при пустых/невалидных значениях."""
    if not args_str or args_str.strip() in ("{}", "", "null"):
        return None
    try:
        parsed = json.loads(args_str)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def parse_tool_args(args_str: str) -> dict:
    """Парсит строку JSON-аргументов в словарь. Пустые/невалидные значения дают {}."""
    parsed = try_parse_tool_args(args_str)
    return parsed if parsed is not None else {}


def args_are_valid(args_str: str) -> bool:
    """True, если аргументы можно передать инструменту: пустые/`{}`/`null` или валидный JSON-объект."""
    s = (args_str or "").strip()
    if not s or s in ("{}", "null"):
        return True
    try:
        json.loads(s)
        return True
    except Exception:
        return False


def normalize_args(args_str: str) -> str:
    """Канонизирует аргументы для сравнения на дубликаты (порядок ключей и пробелы не важны).

    Принимает строку JSON или dict (на случай, если ToolCall.arguments — dict).
    """
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