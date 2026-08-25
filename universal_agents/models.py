from datetime import datetime
from typing import Optional, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from universal_agents.config import Config

@dataclass
class Message(ABC):
    timestamp: datetime = field(init=False)
    # Стабильный монотонный идентификатор в рамках сессии. Назначается
    # ChatHistory при добавлении; переживёт /save + /load, в API не уходит.
    seq: Optional[int] = field(init=False, default=None, repr=False)

    def __post_init__(self):
        self.timestamp = datetime.now()

    @abstractmethod
    def to_api_dict(self) -> dict[str, Any]:
        pass

    def to_persist_dict(self) -> dict[str, Any]:
        """Словарь для сохранения в историю (JSON). По умолчанию совпадает с
        API-представлением; подклассы могут добавить служебные метаданные
        (например, флаг skip_summarize у ToolResult), которые НЕ должны уходить
        в запрос к LLM, но должны переживать /save + /load."""
        d = self.to_api_dict()
        if self.seq is not None:
            d["_seq"] = self.seq
        return d

@dataclass
class SystemMessage(Message):
    content: str

    def to_api_dict(self) -> dict[str, Any]:
        return {"role": "system", "content": self.content}

@dataclass
class UserMessage(Message):
    content: str
    is_summary: bool = False
    _cached_header: Optional[str] = field(default=None, init=False, repr=False)

    def to_api_dict(self) -> dict[str, Any]:
        return {"role": "user", "content": self.content}

    def to_persist_dict(self) -> dict[str, Any]:
        d = self.to_api_dict()
        d["_is_summary"] = self.is_summary
        return d

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str

    def to_api_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments
            }
        }

@dataclass
class AssistantMessage(Message):
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning_content: str = ""
    streamed: bool = False

    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    def to_api_dict(self) -> dict[str, Any]:
        d = {"role": "assistant", "content": self.content}
        if self.reasoning_content and Config.KEEP_REASONING_CONTENT_IN_HISTORY:
            d["reasoning_content"] = self.reasoning_content
        if self.tool_calls:
            d["tool_calls"] = [tc.to_api_dict() for tc in self.tool_calls]
        return d

@dataclass
class ToolResult(Message):
    tool_call_id: str
    name: str
    content: str
    is_error: bool = False
    is_user_denied: bool = False
    execution_time_ms: Optional[float] = None
    retry_count: int = 0
    skip_summarize: bool = False
    # Эвристическая подсказка для сжатия: результат воспроизводим повторным
    # запуском инструмента (чтение файла, поиск и т.п.) — при компакции его
    # можно сворачивать агрессивнее, чем невосстановимые результаты.
    recoverable_hint: bool = False

    def to_api_dict(self) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "content": self.content
        }

    def to_persist_dict(self) -> dict[str, Any]:
        d = self.to_api_dict()
        # Служебные метаданные (через underscore-префикс, чтобы не конфликтовать
        # с полями API и не уходить в запрос к модели).
        d.update({
            "_is_error": self.is_error,
            "_is_user_denied": self.is_user_denied,
            "_retry_count": self.retry_count,
            "_execution_time_ms": self.execution_time_ms,
            "_skip_summarize": self.skip_summarize,
            "_recoverable_hint": self.recoverable_hint,
        })
        return d

    @classmethod
    def success(cls, tool_call_id: str, name: str, content: str = "Tool executed successfully"):
        return cls(tool_call_id, name, content, is_error=False)

    @classmethod
    def error(cls, tool_call_id: str, name: str, error: str):
        return cls(tool_call_id, name, f"Error: {error}", is_error=True)

    @classmethod
    def user_denied(cls, tool_call_id: str, name: str):
        return cls(
            tool_call_id, name,
            "User denied tool call. ASK them why and what to do next.",
            is_user_denied=True
        )
