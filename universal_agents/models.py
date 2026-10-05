from datetime import datetime
from typing import Optional, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from universal_agents.config import Config


def multimodal_content(text: str, images: list[str]) -> str | list[dict]:
    """Content для API: строка, когда картинок нет; иначе список частей [text, image_url...].

    Картинки — side-field сообщений (images), в content не хранятся: весь код фреймворка
    работает со строковым content как раньше, а мультимодальность появляется только здесь.
    Сериализация сообщения с картинками детерминирована и никогда не меняется после
    создания — стабильный префикс KV-кэша (§1.2). Часть text присутствует всегда
    (в т.ч. служебная шапка user-сообщения из context_builder)."""
    if not images:
        return text
    parts: list[dict] = [{"type": "text", "text": text or ""}]
    for b64 in images:
        url = f"data:image/jpeg;base64,{b64}"
        parts.append({"type": "image_url", "image_url": {"url": url}})
    return parts


@dataclass
class Message(ABC):
    timestamp: datetime = field(init=False)
    # Стабильный монотонный id сессии; назначается ChatHistory, переживает /save+/load, в API не уходит.
    seq: Optional[int] = field(init=False, default=None, repr=False)

    def __post_init__(self):
        self.timestamp = datetime.now()

    @abstractmethod
    def to_api_dict(self) -> dict[str, Any]:
        pass

    def to_persist_dict(self) -> dict[str, Any]:
        """Словарь для сохранения в историю (JSON). По умолчанию = API-представлению; подклассы добавляют служебные метаданные, которые не уходят в LLM, но переживают /save+/load."""
        d = self.to_api_dict()
        d["_ts"] = self.timestamp.isoformat()
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
    # Base64 JPEG-картинки, приложенные к сообщению (пусто у обычных сообщений).
    # Живут только в памяти и в API; в JSON-файлы попадают лишь при Config.SAVE_IMAGES=True.
    images: list[str] = field(default_factory=list)
    _cached_header: Optional[str] = field(default=None, init=False, repr=False)
    # Метка нага guard'а answer_to_system: в API не уходит, переживает save/load (scrub
    # находит наг по флагу — текст дублируется превью edit'а, матчинг дал бы ложь).
    _is_guard_nag: bool = field(default=False, init=False, repr=False)

    def reset_header_cache(self) -> None:
        """Сбрасывает кэш заголовка user-сообщения: следующая prepare_messages_for_api
        соберёт header заново (актуальный токен-бюджет и т.п.). Единая точка
        инвалидации кэша."""
        self._cached_header = None

    def to_api_dict(self) -> dict[str, Any]:
        return {"role": "user", "content": multimodal_content(self.content, self.images)}

    def to_persist_dict(self) -> dict[str, Any]:
        # content кладём строковым полем (не to_api_dict): файл истории — не запрос к LLM.
        d: dict[str, Any] = {"role": "user", "content": self.content}
        d["_is_summary"] = self.is_summary
        d["_is_guard_nag"] = self._is_guard_nag
        d["_ts"] = self.timestamp.isoformat()
        d["_header"] = self._cached_header
        if Config.SAVE_IMAGES and self.images:
            d["_images"] = list(self.images)
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
    # Подсказка сжатия: воспроизводимый результат (чтение/поиск) сворачивается агрессивнее, чем невосстановимый.
    recoverable_hint: bool = False
    # Base64 JPEG-картинки результата (скриншот): в API уходят как image_url-части
    # рядом с текстом (спайк фазы 0: LM Studio принимает image в role=tool).
    # Поведенчески content остаётся строкой — парсинг ошибок/усечение/саммаризация не меняются.
    images: list[str] = field(default_factory=list)

    def to_api_dict(self) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "content": multimodal_content(self.content, self.images),
        }

    def to_persist_dict(self) -> dict[str, Any]:
        # content кладём строковым полем (не to_api_dict): файл истории — не запрос к LLM.
        d: dict[str, Any] = {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "content": self.content,
        }
        # Служебные метаданные через underscore-префикс — не конфликтуют с API и не уходят в запрос к модели.
        d["_ts"] = self.timestamp.isoformat()
        d.update({
            "_is_error": self.is_error,
            "_is_user_denied": self.is_user_denied,
            "_retry_count": self.retry_count,
            "_execution_time_ms": self.execution_time_ms,
            "_skip_summarize": self.skip_summarize,
            "_recoverable_hint": self.recoverable_hint,
        })
        if Config.SAVE_IMAGES and self.images:
            d["_images"] = list(self.images)
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
