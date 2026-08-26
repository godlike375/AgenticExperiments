"""Типизированная группа параметров генерации LLM."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from universal_agents.config import Config


@dataclass
class GenerationParams:
    """Параметры генерации; None означает дефолт из Config."""

    temp: Optional[float] = None
    timeout: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    max_tokens: Optional[int] = None

    @classmethod
    def from_overrides(cls, **overrides) -> "GenerationParams":
        """Создаёт полностью разрешённые параметры: None заменяется на дефолты из Config."""
        return cls(**overrides).resolved()

    def resolved(self) -> "GenerationParams":
        """Возвращает копию с подставленными значениями Config вместо None."""
        return GenerationParams(
            temp=Config.TEMP if self.temp is None else self.temp,
            timeout=Config.TIMEOUT if self.timeout is None else self.timeout,
            top_p=Config.TOP_P if self.top_p is None else self.top_p,
            frequency_penalty=Config.FREQUENCY_PENALTY if self.frequency_penalty is None else self.frequency_penalty,
            presence_penalty=Config.PRESENCE_PENALTY if self.presence_penalty is None else self.presence_penalty,
            max_tokens=Config.MAX_OUTPUT_TOKENS if self.max_tokens is None else self.max_tokens,
        )

    def with_temp(self, temp: float) -> "GenerationParams":
        """Возвращает копию с переопределённой температурой."""
        return replace(self, temp=temp)
