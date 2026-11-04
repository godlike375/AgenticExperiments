from __future__ import annotations

"""Общие константы фреймворка."""
ENVIRONMENT_PREFIX = '<SYSTEM>'
ENVIRONMENT_PREFIX_END = '</SYSTEM>'


def err(msg: str) -> str:
    """Единый формат ошибки (§1.4): msg — текст сразу после Error для побайтовой замены."""
    return f"{ENVIRONMENT_PREFIX} Error{msg}{ENVIRONMENT_PREFIX_END}"


def ok(msg: str) -> str:
    """Единый формат системного сообщения: msg — текст сразу после префикса."""
    return f"{ENVIRONMENT_PREFIX} {msg} {ENVIRONMENT_PREFIX_END}"


SUMMARY_PREFIX_USER = "USER:"
SUMMARY_PREFIX_AI = "AI:"
SUMMARY_PREFIX_TOOL_CALL = "TOOL:"
SUMMARY_PREFIX_TOOL_RESULT = "RESULT:"
SUMMARY_PREFIX_TOOL_NAMED = "TOOL({name}):"
SUMMARY_MARKER = "[PAST-SUMMARY]"

INTERRUPT_HEADER = (f"{ENVIRONMENT_PREFIX} User interrupted generation and wants to say something more important. "
                    f"Treat it carefully before continuing {ENVIRONMENT_PREFIX_END}")

CONFIRMATION_GUARD_VERDICT = "confirmation_guard"
REPEAT_ANSWER_NAG = (
    f"\n\n{ENVIRONMENT_PREFIX} Your previous answer was the same to the latest one. "
    f"Please do NOT repeat it again and answer differently."
    f"{ENVIRONMENT_PREFIX_END}"
)
