from __future__ import annotations

from dataclasses import replace
from typing import Optional

from universal_agents.config import Config
from universal_agents.generation import GenerationParams
from universal_agents.llm_client import apply_prefill
from universal_agents.models import ToolCall

_NO_COMMENT_PREFILL = 'LLM Agent: "'


def sim_schema_text() -> str:
    return "".join(f"<{tag}/>" for tag in Config.model.SIMULATED_REASONING_TAGS)


def tail_after_tag(content: str, tag: str) -> str:
    if not content:
        return ""
    closing = f"</{tag}>"
    idx = content.rfind(closing)
    return content[idx + len(closing):] if idx >= 0 else ""


class SimReasoningGroup:
    """Секция рассуждения (§2.8): prefill открывает тег, закрытие дописываем сами; двухфазный режим при SIMULATED_REASONING_*."""

    def __init__(self, retries: int):
        self.reset(retries)

    def reset(self, retries: int) -> None:
        self.tag = Config.model.SIMULATED_REASONING_TAGS[0]
        self.content = ""
        self.appended = ""
        self.tool_calls: list[ToolCall] = []
        self.reasoning = ""
        self.retries_left = retries
        self._continuation = False
        self.phase = "think" if self._reasoning_overrides() else "single"
        self._shown = 0
        self._injected: list[int] = []

    @staticmethod
    def _reasoning_overrides() -> dict:
        names = ("temp", "top_p", "min_p", "frequency_penalty", "presence_penalty")
        return {
            name: getattr(Config.model, f"SIMULATED_REASONING_{name.upper()}")
            for name in names
            if getattr(Config.model, f"SIMULATED_REASONING_{name.upper()}") is not None
        }

    def two_phase(self) -> bool:
        return self.phase in ("think", "answer")

    def is_think_phase(self) -> bool:
        return self.phase == "think"

    def advance_to_answer(self) -> None:
        if self.phase == "think":
            self.phase = "answer"

    def think_params(self, base: GenerationParams) -> Optional[GenerationParams]:
        overrides = self._reasoning_overrides()
        if not overrides:
            return None
        return replace(base, **overrides)

    def prefill(self) -> str:
        if self._continuation or self.phase == "answer":
            return self.content
        return f"<{self.tag}>"

    def prefill_shown(self) -> int:
        return self._shown

    def absorb(self, message_obj, prefill: str) -> None:
        raw = apply_prefill(message_obj.content or "", prefill)
        self.content = raw
        self.tool_calls.extend(message_obj.tool_calls or [])
        self.reasoning += getattr(message_obj, "reasoning_content", "") or ""
        self._shown = len(self.content)

    def close_section(self) -> None:
        closing = f"</{self.tag}>"
        if self.content and closing not in self.content:
            self.content += closing
            self.appended = closing
        if self.content:
            pos = self.content.rfind(closing) + len(closing)
            if self.content[pos:pos + 1] != "\n":
                self.content = self.content[:pos] + "\n" + self.content[pos:]
                if pos == len(self.content) - 1:
                    self.appended += "\n"
        self._shown = len(self.content)

    def take_appended(self) -> str:
        tail, self.appended = self.appended, ""
        return tail

    def answer_text(self) -> str:
        tail = tail_after_tag(self.content, self.tag)
        head_len = len(self.content) - len(tail)
        return self._strip_injected(tail, head_len)

    def final_content(self) -> str:
        return self._strip_injected(self.content, 0)

    def _strip_injected(self, text: str, offset: int) -> str:
        if not self._injected:
            return text
        skip: set[int] = set()
        for start in self._injected:
            for pos in range(start, min(start + len(_NO_COMMENT_PREFILL), len(self.content))):
                skip.add(pos - offset)
        return "".join(ch for i, ch in enumerate(text) if i not in skip)

    def start_continuation(self) -> None:
        self._shown = len(self.content)
        self._injected.append(len(self.content))
        self.content += _NO_COMMENT_PREFILL
        self.appended = ""
        self.tool_calls = []
        self.reasoning = ""
        self._continuation = True
