from datetime import datetime
from typing import Optional, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from universal_agents.config import Config

@dataclass
class Message(ABC):
    timestamp: datetime = field(init=False)

    def __post_init__(self):
        self.timestamp = datetime.now()

    @abstractmethod
    def to_api_dict(self) -> dict[str, Any]:
        pass

@dataclass
class SystemMessage(Message):
    content: str

    def to_api_dict(self) -> dict[str, Any]:
        return {"role": "system", "content": self.content}

@dataclass
class UserMessage(Message):
    content: str
    _cached_header: Optional[str] = field(default=None, init=False, repr=False)

    def to_api_dict(self) -> dict[str, Any]:
        return {"role": "user", "content": self.content}

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

    def to_api_dict(self) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "content": self.content
        }

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
