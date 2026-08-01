import json
from typing import Optional, Any, Iterable
from universal_agents.constants import ENVIRONMENT_PREFIX
from universal_agents.config import Config
from universal_agents.models import SystemMessage, UserMessage, AssistantMessage, ToolResult, ToolCall, Message

class ChatHistory:
    def __init__(self, system_prompt: str):
        self._messages: list[Message] = [SystemMessage(system_prompt)]

    def add(self, msg: Message):
        self._messages.append(msg)

    def extend(self, msgs: list[Message]):
        self._messages.extend(msgs)

    def get_all_api(self) -> list[dict[str, Any]]:
        return [msg.to_api_dict() for msg in self._messages]

    def get_all(self) -> list[Message]:
        return self._messages.copy()

    def __len__(self):
        return len(self._messages)

    def __getitem__(self, idx) -> Message:
        return self._messages[idx]

    def __iter__(self):
        return iter(self._messages)

    def get_last_message(self) -> Optional[Message]:
        return self._messages[-1] if len(self._messages) > 1 else None

    def pop_until_user(self) -> Optional[str]:
        user_content = None
        while len(self._messages) > Config.AFTER_SYSTEM_PROMPT:
            last = self._messages[-1]
            if isinstance(last, UserMessage):
                user_content = last.content
                self._messages.pop()
                break
            else:
                self._messages.pop()
        return user_content

    def delete_range(self, start_id: int, end_id: int = -1):
        if not (0 <= start_id < len(self._messages)):
            return f"Error: Invalid start_id {start_id}"
        if end_id == -1 or end_id >= len(self._messages):
            end_id = len(self._messages) - 1
        safe_start = max(start_id, Config.AFTER_SYSTEM_PROMPT)
        safe_end = end_id
        if safe_start > safe_end:
            return f"{ENVIRONMENT_PREFIX} Nothing to delete"
        for msg in self._messages[safe_start:]:
            if isinstance(msg, UserMessage):
                msg._cached_header = None
        del self._messages[safe_start:safe_end + 1]
        
        has_user_message = any(isinstance(m, UserMessage) for m in self._messages)
        if not has_user_message:
            self._messages.append(UserMessage(
                content=f"{ENVIRONMENT_PREFIX} All user messages were deleted. Shortly introduce yourself in Russian."
            ))
        
        return f'{ENVIRONMENT_PREFIX} Successfully deleted messages {start_id} - {end_id}'

    def edit_message(self, idx: int, new_text: str, old_text: str = '') -> str:
        if not (0 <= idx < len(self._messages)):
            return f"{ENVIRONMENT_PREFIX} Error: Invalid message index {idx}"
        msg = self._messages[idx]
        if isinstance(msg, SystemMessage):
            return f"{ENVIRONMENT_PREFIX} Error: Cannot edit system prompt"
        if not old_text.strip():
            msg.content = new_text
        else:
            if old_text not in msg.content:
                return f"Error: Substr '{old_text}' not found in message {idx}"
            msg.content = msg.content.replace(old_text, new_text, 1)
        if isinstance(msg, UserMessage):
            msg._cached_header = None
        if not msg.content.strip() and idx >= Config.AFTER_SYSTEM_PROMPT:
            self.delete_range(idx, idx)
            return 'Replacing to empty text led to deleting the message block.'
        return f'{ENVIRONMENT_PREFIX} Success'

    def normalize(self, is_error_recovery: bool = False):
        if len(self._messages) <= Config.AFTER_SYSTEM_PROMPT:
            return

        raw = self._messages
        valid = [raw[0]]

        # Ищем первое сообщение пользователя
        first_user_idx = Config.AFTER_SYSTEM_PROMPT
        while first_user_idx < len(raw) and not isinstance(raw[first_user_idx], UserMessage):
            first_user_idx += 1

        # Если сообщений пользователя нет вовсе — возвращаем историю к исходному системному промпту
        if first_user_idx >= len(raw):
            self._messages = valid
            return

        valid.append(raw[first_user_idx])

        # Фильтруем и объединяем оставшуюся историю
        for i in range(first_user_idx + 1, len(raw)):
            msg = raw[i]
            last = valid[-1]

            if isinstance(msg, ToolResult):
                if isinstance(last, AssistantMessage) and last.has_tool_calls():
                    call_ids = [tc.id for tc in last.tool_calls]
                    if msg.tool_call_id in call_ids:
                        valid.append(msg)
                continue

            if type(msg) == type(last) and isinstance(msg, (UserMessage, AssistantMessage)):
                last.content = (last.content or "") + "\n\n" + (msg.content or "")
                if isinstance(last, UserMessage):
                    last._cached_header = None
                if isinstance(msg, AssistantMessage) and msg.has_tool_calls():
                    last.tool_calls = last.tool_calls + msg.tool_calls
                continue

            valid.append(msg)

        # Добавляем заглушку ассистента ТОЛЬКО при восстановлении после сбоя
        if is_error_recovery and isinstance(valid[-1], ToolResult):
            valid.append(AssistantMessage(
                content=f"{ENVIRONMENT_PREFIX} This is a message from system because a sequence of failed tool calls was detected and pruned. The system gave control to the user."
            ))

        self._messages = valid

    def save(self, path: str, loaded_tools: list[str] = None):
        payload = {
            "messages": self.get_all_api(),
            "loaded_tools": loaded_tools or [],
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> list[str]:
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        if isinstance(raw, list):
            data_list = raw
            loaded_tools = []
        elif isinstance(raw, dict) and "messages" in raw:
            data_list = raw["messages"]
            loaded_tools = raw.get("loaded_tools", [])
        else:
            raise ValueError(f"⚠️ {ENVIRONMENT_PREFIX} Invalid history format")

        if not isinstance(data_list, list) or not data_list:
            raise ValueError(f"⚠️ {ENVIRONMENT_PREFIX} Invalid history format")

        self._messages = []
        for d in data_list:
            role = d.get("role")
            if role == "system":
                self._messages.append(SystemMessage(d["content"]))
            elif role == "user":
                self._messages.append(UserMessage(d["content"]))
            elif role == "assistant":
                tcs = []
                for tc in d.get("tool_calls", []):
                    tcs.append(ToolCall(
                        id=tc["id"],
                        name=tc["function"]["name"],
                        arguments=tc["function"]["arguments"]
                    ))
                self._messages.append(AssistantMessage(
                    content=d.get("content", ""),
                    tool_calls=tcs
                ))
            elif role == "tool":
                self._messages.append(ToolResult.success(
                    tool_call_id=d["tool_call_id"],
                    name=d.get("name", "unknown"),
                    content=d["content"]
                ))
            else:
                raise ValueError(f"Unknown role: {role}")
        return loaded_tools

    def get_last_user_message(self) -> Optional[UserMessage]:
        for msg in reversed(self._messages):
            if isinstance(msg, UserMessage):
                return msg
        return None

    def remove_at(self, indices: Iterable[int]) -> None:
        """Удаляет сообщения по индексам (порядок не важен)."""
        for idx in sorted(indices, reverse=True):
            if 0 <= idx < len(self._messages):
                del self._messages[idx]

    def content_len(self, start: int, end: int) -> int:
        """Суммарная длина содержимого сообщений в диапазоне [start..end]."""
        return sum(len(getattr(m, 'content', '') or '') for m in self._messages[start:end + 1])

    def replace_range(self, start: int, end: int, replacement: list[Message]) -> None:
        """Заменяет сообщения [start..end] на replacement."""
        self._messages[start:end + 1] = replacement

    def compress_old_messages(self, summary: str, preserve_last: int = 2) -> None:
        """Заменяет старые сообщения summary-сообщением (UserMessage),
        сохраняя system prompt и последние preserve_last сообщений."""
        safe_end = len(self._messages) - preserve_last

        if Config.AFTER_SYSTEM_PROMPT > safe_end:
            return

        summary_msg = UserMessage(
            content=f"{ENVIRONMENT_PREFIX} It's an automatically generated system message. Your previous dialog summary with user:\n{summary}"
        )

        preserved = self._messages[safe_end:]
        for msg in preserved:
            if isinstance(msg, UserMessage):
                msg._cached_header = None
        self._messages = [self._messages[0], summary_msg] + preserved
