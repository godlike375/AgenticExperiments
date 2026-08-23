import json
from typing import Optional, Any, Iterable
from universal_agents.constants import ENVIRONMENT_PREFIX, ENVIRONMENT_PREFIX_END, err, ok
from universal_agents.config import Config
from universal_agents.models import SystemMessage, UserMessage, AssistantMessage, ToolResult, ToolCall, Message

class ChatHistory:
    def __init__(self, system_prompt: str):
        self._messages: list[Message] = [SystemMessage(system_prompt)]
        # Рабочая память (НЕ попадает в контекст): плотные саммари отдельных
        # сообщений (ключ — id(Message)). Используются при сжатии диалога,
        # чтобы из маленьких саммари собрать более короткий новый диалог.
        self._per_msg_summaries: dict[int, str] = {}
        # Dirty-флаг нормализации: normalize() пересобирает список сообщений,
        # что дорого и сбрасывает кэш заголовков; запускаем его только когда
        # история реально менялась с последней нормализации.
        self._needs_normalize: bool = False

    def _mark_dirty(self) -> None:
        """Отмечает историю изменённой — следующая normalize() реально выполнится."""
        self._needs_normalize = True

    def add(self, msg: Message):
        self._messages.append(msg)
        self._mark_dirty()

    # --------------------------------------------------------
    # Рабочая память: плотные саммари отдельных сообщений
    # --------------------------------------------------------
    def set_per_msg_summary(self, msg: Message, summary: str) -> None:
        self._per_msg_summaries[id(msg)] = summary

    def get_per_msg_summary(self, msg: Message) -> Optional[str]:
        return self._per_msg_summaries.get(id(msg))

    def clear_per_msg_summary(self, msg: Message) -> None:
        self._per_msg_summaries.pop(id(msg), None)

    def prune_per_msg_summaries(self) -> None:
        """Убирает из рабочей памяти саммари сообщений, которых больше нет в истории."""
        alive = {id(m) for m in self._messages}
        self._per_msg_summaries = {k: v for k, v in self._per_msg_summaries.items() if k in alive}

    def extend(self, msgs: list[Message]):
        self._messages.extend(msgs)
        self._mark_dirty()

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
                self._mark_dirty()
                break
            else:
                self._messages.pop()
                self._mark_dirty()
        return user_content

    def delete_range(self, start_id: int, end_id: int = -1):
        if not (0 <= start_id < len(self._messages)):
            return err(f": Invalid start_id {start_id}")
        if end_id == -1 or end_id >= len(self._messages):
            end_id = len(self._messages) - 1
        safe_start = max(start_id, Config.AFTER_SYSTEM_PROMPT)
        safe_end = end_id
        if safe_start > safe_end:
            return err(" Nothing to delete")
        for msg in self._messages[safe_start:]:
            if isinstance(msg, UserMessage):
                msg._cached_header = None
        del self._messages[safe_start:safe_end + 1]
        self._mark_dirty()

        has_user_message = any(isinstance(m, UserMessage) for m in self._messages)
        if not has_user_message:
            self._messages.append(UserMessage(
                content=ok(" All user messages were deleted. Shortly introduce yourself in Russian.")
            ))

        return ok(f" Successfully deleted messages {start_id} - {end_id}")

    def edit_message(self, idx: int, new_text: str, old_text: str = '') -> str:
        if not (0 <= idx < len(self._messages)):
            return err(f": Invalid message index {idx}")
        msg = self._messages[idx]
        if isinstance(msg, SystemMessage):
            return err(": Cannot edit system prompt")
        if not old_text.strip():
            msg.content = new_text
        else:
            if old_text not in msg.content:
                return err(f": Substr '{old_text}' not found in message {idx}")
            msg.content = msg.content.replace(old_text, new_text, 1)
        if isinstance(msg, UserMessage):
            msg._cached_header = None
        if not msg.content.strip() and idx >= Config.AFTER_SYSTEM_PROMPT:
            self.delete_range(idx, idx)
            return 'Replacing to empty text led to deleting the message block.'
        return ok(" Success")

    def normalize(self, is_error_recovery: bool = False):
        # Пересборка дорога — только если история менялась (или recovery).
        if not self._needs_normalize and not is_error_recovery:
            return

        if len(self._messages) <= Config.AFTER_SYSTEM_PROMPT:
            self._needs_normalize = False
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
            self._needs_normalize = False
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
                content=f"{ENVIRONMENT_PREFIX} This is a message from system because a sequence of failed tool calls was detected and pruned. The system gave control to the user.{ENVIRONMENT_PREFIX_END}"
            ))

        self._messages = valid
        self._needs_normalize = False

    def remove_failed_call_chains(self) -> int:
        """Удаляет цепочки «неудачный вызов -> неудачный вызов» из истории.

        Ошибочный вызов (AssistantMessage с tool_calls + его error-ToolResult)
        удаляется только если сразу за ним следует ещё один неудачный вызов.
        Сценарий «неудачный -> удачный» сохраняется: после удачного вызова может
        идти большой вывод инструмента, и удаление неудачи перед ним сбросило бы
        KV-кэш, тогда как ошибки обычно короткие.

        Возвращает число удалённых сообщений. Вызывающий код сам решает, звать ли
        normalize()/_on_history_changed().
        """
        n = len(self._messages)
        if n <= Config.AFTER_SYSTEM_PROMPT + 1:
            return 0

        def is_failed_call(idx: int) -> bool:
            if idx + 1 >= len(self._messages):
                return False
            msg = self._messages[idx]
            if not isinstance(msg, AssistantMessage) or not msg.has_tool_calls():
                return False
            tool_result = self._messages[idx + 1]
            return isinstance(tool_result, ToolResult) and tool_result.is_error and not tool_result.is_user_denied

        indices_to_remove: set[int] = set()
        i = Config.AFTER_SYSTEM_PROMPT
        while i < len(self._messages):
            if is_failed_call(i) and is_failed_call(i + 2):
                indices_to_remove.add(i)
                indices_to_remove.add(i + 1)
                i += 2
                continue
            i += 1

        if not indices_to_remove:
            return 0

        removed_count = len(indices_to_remove)
        self.remove_at(indices_to_remove)
        return removed_count

    def save(self, path: str, loaded_tools: list[str] = None, file_states: dict = None):
        # Саммари рабочей памяти сохраняются списком, выровненным по индексам
        # сообщений (ключ id() при перезагрузке не переживёт — объекты создаются заново).
        # Если рабочая память пуста — пишем [], чтобы не засорять файл и сохранить
        # обратно совместимый формат.
        summaries: list[str] = (
            [] if not self._per_msg_summaries
            else [self._per_msg_summaries.get(id(m)) or "" for m in self._messages]
        )
        payload = {
            "messages": [m.to_persist_dict() for m in self._messages],
            "loaded_tools": loaded_tools or [],
            "file_states": file_states or {},
            "per_msg_summaries": summaries,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> tuple[list[str], dict, list[str]]:
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        if isinstance(raw, list):
            data_list = raw
            loaded_tools = []
            file_states = {}
            summaries = []
        elif isinstance(raw, dict) and "messages" in raw:
            data_list = raw["messages"]
            loaded_tools = raw.get("loaded_tools", [])
            file_states = raw.get("file_states", {}) or {}
            summaries = raw.get("per_msg_summaries", []) or []
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
                self._messages.append(UserMessage(
                    d["content"],
                    is_summary=d.get("_is_summary", False),
                ))
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
                self._messages.append(ToolResult(
                    tool_call_id=d["tool_call_id"],
                    name=d.get("name", "unknown"),
                    content=d["content"],
                    is_error=d.get("_is_error", False),
                    is_user_denied=d.get("_is_user_denied", False),
                    retry_count=d.get("_retry_count", 0),
                    execution_time_ms=d.get("_execution_time_ms"),
                    skip_summarize=d.get("_skip_summarize", False),
                ))
            else:
                raise ValueError(f"Unknown role: {role}")

        # Перепривязываем сохранённые саммари рабочей памяти к пересозданным
        # объектам сообщений (ключ id() меняется после загрузки).
        self._per_msg_summaries = {}
        for i, s in enumerate(summaries):
            if s and i < len(self._messages):
                self._per_msg_summaries[id(self._messages[i])] = s
        self._needs_normalize = True
        return loaded_tools, file_states, summaries

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
        self._mark_dirty()

    def pop_pending_tool_calls(self) -> int:
        """Удаляет висящие (pending) вызовы инструментов в конце истории —
        AssistantMessage с tool_calls, у которых ещё нет результата. Возвращает
        число удалённых сообщений.

        Используется перед авто-суммаризацией: незавершённый вызов инструмента
        чище перевызвать уже после сжатия, чем тащить через суммаризацию."""
        popped = 0
        while (
            self._messages
            and isinstance(self._messages[-1], AssistantMessage)
            and self._messages[-1].has_tool_calls()
        ):
            self._messages.pop()
            popped += 1
        if popped:
            self._mark_dirty()
        return popped

    def _message_len(self, msg: Message) -> int:
        """Исчерпывающая длина сообщения: content + reasoning_content + tool_calls."""
        total = len(msg.content or '')
        if isinstance(msg, AssistantMessage):
            total += len(msg.reasoning_content or '')
            for tc in msg.tool_calls:
                total += len(tc.id or '') + len(tc.name or '') + len(tc.arguments or '')
        return total

    def content_len(self, start: int, end: int) -> int:
        """Суммарная исчерпывающая длина сообщений в диапазоне [start..end].

        Учитывает content, reasoning_content (если есть) и tool_calls
        (id + name + arguments), т.к. все они расходуют токены.
        """
        return sum(
            self._message_len(m)
            for m in self._messages[start:end + 1]
        )

    def replace_range(self, start: int, end: int, replacement: list[Message]) -> None:
        """Заменяет сообщения [start..end] на replacement."""
        self._messages[start:end + 1] = replacement
        self._mark_dirty()

    def compress_old_messages(self, summary: str, preserve_last: int = 2) -> None:
        """Заменяет старые сообщения summary-сообщением (одно UserMessage,
        помеченное ENVIRONMENT_PREFIX), сохраняя system prompt и последние
        preserve_last сообщений. Всё (включая выводы инструментов) сворачивается
        в единственное пользовательское сообщение.

        Если граница режет пару «ассистент вызвал инструмент -> ToolResult»
        (первое сохранённое сообщение — ToolResult, а вызвавший его ассистент
        оказывается в свёрнутой области), границу сдвигаем назад, чтобы сохранить
        ассистента вместе с его ToolResult. Иначе ToolResult остаётся без
        предшествующего вызова (invalid для API и теряется в normalize())."""
        safe_end = len(self._messages) - preserve_last

        if Config.AFTER_SYSTEM_PROMPT > safe_end:
            return

        msgs = self._messages
        # Сдвигаем границу назад, пока первое сохраняемое сообщение — ToolResult,
        # а непосредственно перед ним — ассистент с вызовами инструментов.
        while (
            safe_end > Config.AFTER_SYSTEM_PROMPT
            and isinstance(msgs[safe_end], ToolResult)
            and isinstance(msgs[safe_end - 1], AssistantMessage)
            and msgs[safe_end - 1].has_tool_calls()
        ):
            safe_end -= 1

        summary_msg = UserMessage(
            content=f"{ENVIRONMENT_PREFIX}: Your past dialog summary with user:\n{summary}\n{ENVIRONMENT_PREFIX_END}",
            is_summary=True,
        )

        preserved = msgs[safe_end:]
        for msg in preserved:
            if isinstance(msg, UserMessage):
                msg._cached_header = None
        self._messages = [msgs[0], summary_msg] + preserved
        self._mark_dirty()
