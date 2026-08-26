"""Общие константы фреймворка."""

# Маркер, которым системные сообщения помечаются в истории и промптах
ENVIRONMENT_PREFIX = '[[SYSTEM]]'
ENVIRONMENT_PREFIX_END = '[[/SYSTEM]]'

# Инструменты, которые нельзя отключить
CORE_TOOLS = ("load_tool", "unload_tool")


def err(msg: str) -> str:
    """Единый формат сообщения об ошибке (инвариант §2-4). ``msg`` — текст сразу после ``Error`` для побайтово точной замены."""
    return f"{ENVIRONMENT_PREFIX} Error{msg}{ENVIRONMENT_PREFIX_END}"


def ok(msg: str) -> str:
    """Единый формат системного сообщения инструмента. ``msg`` — текст сразу после префикса для побайтово точной замены."""
    return f"{ENVIRONMENT_PREFIX}{msg}{ENVIRONMENT_PREFIX_END}"

# Маркер авто-саммари заменён на метаданные UserMessage.is_summary (текстовых маркеров в контенте больше нет).

# Префиксы плотных per-message саммари (единый источник для MemoryMixin и compressors, чтобы метки не разъехались).
SUMMARY_PREFIX_USER = "USER:"
SUMMARY_PREFIX_AI = "AI:"
SUMMARY_PREFIX_TOOL_CALL = "TOOL:"
SUMMARY_PREFIX_TOOL_RESULT = "RESULT:"
SUMMARY_PREFIX_TOOL_NAMED = "TOOL({name}):"
SUMMARY_MARKER = "[PAST-SUMMARY]"

