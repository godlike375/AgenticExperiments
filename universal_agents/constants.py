"""Общие константы фреймворка."""

# Маркер, которым системные сообщения помечаются в истории и промптах
ENVIRONMENT_PREFIX = '[[SYSTEM]]'
ENVIRONMENT_PREFIX_END = '[[/SYSTEM]]'

# Инструменты, которые нельзя отключить
CORE_TOOLS = ("load_tool", "unload_tool")


def err(msg: str) -> str:
    """Единый формат сообщения об ошибке инструмента (инвариант §2-4).

    ``msg`` — текст, который в исходном коде шёл сразу после слова ``Error``
    (включая ведущий пробел или ``:``), чтобы замена была побайтово точной.
    """
    return f"{ENVIRONMENT_PREFIX} Error{msg}{ENVIRONMENT_PREFIX_END}"


def ok(msg: str) -> str:
    """Единый формат системного (не ошибочного) сообщения инструмента.

    ``msg`` — текст, который в исходном коде шёл сразу после префикса
    (включая ведущий пробел), чтобы замена была побайтово точной.
    """
    return f"{ENVIRONMENT_PREFIX}{msg}{ENVIRONMENT_PREFIX_END}"

# Маркер авто-саммари был заменён на метаданные объекта `UserMessage.is_summary`
# (см. compressors.is_summary_message / history.load). Текстовых маркеров в
# контенте больше не используется.

# Префиксы плотных саммари отдельных сообщений (working memory / сжатие).
# Единый источник истины — используется и в MemoryMixin (запись в рабочую
# память), и в compressors (сборка сжатого диалога), чтобы метки USER:/AI:/
# TOOL:/RESULT: не разъехались между двумя местами.
SUMMARY_PREFIX_USER = "USER:"
SUMMARY_PREFIX_AI = "AI:"
SUMMARY_PREFIX_TOOL_CALL = "TOOL:"
SUMMARY_PREFIX_TOOL_RESULT = "RESULT:"
SUMMARY_PREFIX_TOOL_NAMED = "TOOL({name}):"

