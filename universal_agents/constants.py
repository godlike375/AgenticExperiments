"""Общие константы фреймворка."""

ENVIRONMENT_PREFIX = '<SYSTEM>'
ENVIRONMENT_PREFIX_END = '</SYSTEM>'


def err(msg: str) -> str:
    """Единый формат сообщения об ошибке (инвариант §2-4). ``msg`` — текст сразу после ``Error`` для побайтово точной замены."""
    return f"{ENVIRONMENT_PREFIX} Error{msg}{ENVIRONMENT_PREFIX_END}"


def ok(msg: str) -> str:
    """Единый формат системного сообщения инструмента. ``msg`` — текст сразу после префикса для побайтово точной замены."""
    return f"{ENVIRONMENT_PREFIX} {msg} {ENVIRONMENT_PREFIX_END}"


SUMMARY_PREFIX_USER = "USER:"
SUMMARY_PREFIX_AI = "AI:"
SUMMARY_PREFIX_TOOL_CALL = "TOOL:"
SUMMARY_PREFIX_TOOL_RESULT = "RESULT:"
SUMMARY_PREFIX_TOOL_NAMED = "TOOL({name}):"
SUMMARY_MARKER = "[PAST-SUMMARY]"

INTERRUPT_HEADER = (f"{ENVIRONMENT_PREFIX} User interrupted generation and wants to say something more important. "
                    f"Treat it carefully before continuing {ENVIRONMENT_PREFIX_END}")

