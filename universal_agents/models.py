import json
from datetime import datetime
from typing import Optional, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from universal_agents.tool import ENVIRONMENT_PREFIX

@dataclass
class Message(ABC):
    @abstractmethod
    def to_api_dict(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def render(self) -> str:
        pass

@dataclass
class SystemMessage(Message):
    content: str

    def to_api_dict(self) -> dict[str, Any]:
        return {"role": "system", "content": self.content}

    def render(self) -> str:
        return f"[SYSTEM]: {self.content[:100]}..."

@dataclass
class UserMessage(Message):
    content: str

    def to_api_dict(self) -> dict[str, Any]:
        return {"role": "user", "content": self.content}

    def render(self) -> str:
        return f"👤 User: {self.content}"

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

    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    def is_empty_text(self) -> bool:
        return not self.content.strip()

    def to_api_dict(self) -> dict[str, Any]:
        d = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [tc.to_api_dict() for tc in self.tool_calls]
        return d

    def render(self) -> str:
        parts = []
        if self.content.strip():
            parts.append(f"🤖 Agent: {self.content}")
        for tc in self.tool_calls:
            parts.append(f"🛠️ [Tool Call: {tc.name}({tc.arguments})]")
        return "\n".join(parts)

@dataclass
class ToolResult(Message):
    tool_call_id: str
    name: str
    content: str
    is_error: bool = False
    is_user_denied: bool = False
    execution_time_ms: Optional[float] = None
    retry_count: int = 0

    def to_api_dict(self) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "content": self.content
        }

    def render(self) -> str:
        prefix = "❌" if self.is_error else "✅"
        display = str(self.content)
        if len(display) > 300:
            display = display[:300] + "\n... [TRUNCATED]"
        return f"{prefix} [Result '{self.name}']: {display}"

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
