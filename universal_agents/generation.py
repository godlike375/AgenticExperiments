
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from universal_agents.config import Config
from universal_agents.controllers import PhaseController


@dataclass
class GenerationParams:

    temp: Optional[float] = None
    timeout: Optional[int] = None
    top_p: Optional[float] = None
    min_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    max_tokens: Optional[int] = None

    @classmethod
    def from_overrides(cls, **overrides) -> "GenerationParams":
        return cls(**overrides).resolved()

    def resolved(self) -> "GenerationParams":
        return GenerationParams(
            temp=Config.TEMP if self.temp is None else self.temp,
            timeout=Config.TIMEOUT if self.timeout is None else self.timeout,
            top_p=Config.TOP_P if self.top_p is None else self.top_p,
            min_p=Config.MIN_P if self.min_p is None else self.min_p,
            frequency_penalty=Config.FREQUENCY_PENALTY if self.frequency_penalty is None else self.frequency_penalty,
            presence_penalty=Config.PRESENCE_PENALTY if self.presence_penalty is None else self.presence_penalty,
            max_tokens=Config.MAX_OUTPUT_TOKENS if self.max_tokens is None else self.max_tokens,
        )

    def with_temp(self, temp: float) -> "GenerationParams":
        return replace(self, temp=temp)


def extract_opening_tag(prefill: str) -> Optional[str]:
    if not prefill:
        return None
    start = prefill.find("<")
    if start < 0:
        return None
    end = prefill.find(">", start)
    if end < 0:
        return None
    token = prefill[start + 1:end].strip()
    if not token or token.startswith("/") or token.endswith("/"):
        return None
    split = token.split()
    tag_name = split[0] if split else token
    if not tag_name:
        return None
    if any(c in tag_name for c in '<>"\'/='):
        return None
    if not all(c.isalnum() or c in "._-:" for c in tag_name):
        return None
    return tag_name


@dataclass(frozen=True)
class StructuredOutputConfig:

    stop_markers: tuple[str, ...] = ()
    next_prefills: tuple[str, ...] = ()
    max_phases: int = 5
    controller: Optional[PhaseController] = None

    @classmethod
    def from_controller(
        cls,
        controller: PhaseController,
        max_phases: int = 5,
    ) -> "StructuredOutputConfig":
        return cls(controller=controller, max_phases=max_phases)

    @classmethod
    def from_prefill(
        cls,
        prefill: str,
        next_prefill: Optional[str] = None,
        max_phases: int = 5,
    ) -> "StructuredOutputConfig":
        tag = extract_opening_tag(prefill)
        markers = (f"</{tag}>",) if tag else ()
        nexts = (next_prefill,) if next_prefill else ()
        return cls(stop_markers=markers, next_prefills=nexts, max_phases=max_phases)

    def effective_markers(self, prefill: Optional[str] = None) -> tuple[str, ...]:
        if self.controller is not None:
            return self.controller.markers()
        if self.stop_markers:
            return self.stop_markers
        if prefill:
            tag = extract_opening_tag(prefill)
            if tag:
                return (f"</{tag}>",)
        return ()


def apply_stop_markers(content: str, stop_markers: tuple[str, ...]) -> tuple[str, bool]:
    best_idx = len(content)
    best_marker = None
    for marker in stop_markers:
        idx = content.find(marker)
        if 0 <= idx < best_idx:
            best_idx = idx
            best_marker = marker
    if best_marker is not None:
        return content[: best_idx + len(best_marker)], True
    return content, False