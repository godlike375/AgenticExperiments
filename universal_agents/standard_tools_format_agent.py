import json
import os
import shlex
import sys
import importlib.util
import inspect
from collections import deque
from datetime import datetime
from typing import Union, Callable
from dataclasses import dataclass, field
from typing import Optional, Any
from abc import ABC, abstractmethod

from openai import OpenAI
from universal_agents.tool import tool, ENVIRONMENT_PREFIX

def load_external_plugins(plugins_dir="tools"):
    """Загружает .py файлы из директории. Возвращает {имя: функция}."""
    external_tools = {}
    if not os.path.exists(plugins_dir):
        return external_tools

    # Добавляем корень проекта в путь для корректных импортов внутри плагинов
    root_path = os.path.abspath(os.path.join(plugins_dir, ".."))
    if root_path not in sys.path:
        sys.path.insert(0, root_path)

    for filename in os.listdir(plugins_dir):
        if not filename.endswith(".py") or filename.startswith("__"):
            continue

        module_name = filename[:-3]
        file_path = os.path.join(plugins_dir, filename)

        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            for name, obj in inspect.getmembers(module):
                if callable(obj) and hasattr(obj, '_is_tool'):
                    if name in external_tools:
                        print(f"Warning: Duplicate tool name '{name}' in {filename}")
                        continue
                    external_tools[name] = obj
        except Exception as e:
            print(f"Error loading plugin {filename}: {e}")

    return external_tools

class Config:
    API_URL = "http://localhost:1234/v1"
    MODEL_NAME = "local-model"
    AFTER_SYSTEM_PROMPT = 1  # Индекс, после которого начинается диалог (обычно 1, т.к. 0 - system)
    BOOST_TEMP = 0.8


class TokenUsageTracker:
    """Отслеживает использование токенов на основе данных от LM Studio"""
    def __init__(self, max_context_tokens: int = 8192):
        self.max_context_tokens = max_context_tokens
        self.last_usage = None  # { "prompt_tokens": int, "completion_tokens": int, "total_tokens": int }

    def update_from_usage(self, usage: dict):
        self.last_usage = usage

    def get_current_context_tokens(self):
        if self.last_usage:
            return self.last_usage.get("prompt_tokens")
        return None

    def format_info_header(self):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current = self.get_current_context_tokens()
        if current is not None:
            remaining = self.max_context_tokens - current
            token_str = f"Tokens spent: {current} (Remaining: {remaining})"
        else:
            token_str = "Tokens: unknown"
        return f"<<{ENVIRONMENT_PREFIX} === [{timestamp}] | {token_str} ===>>\n\n"


class LoopDetector:
    def __init__(self, threshold: int = 3):
        self.threshold = threshold
        self.recent_calls = deque(maxlen=threshold)

    @staticmethod
    def normalize_args(args_str: str) -> str:
        """Канонизирует JSON строку аргументов."""
        if not args_str or args_str.strip() in ("{}", "", "null"):
            return ""
        try:
            parsed = json.loads(args_str)
            return json.dumps(parsed, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
        except Exception:
            return args_str.strip()

    def add_call(self, tool_name: str, arguments: str):
        norm_args = self.normalize_args(arguments)
        self.recent_calls.append((tool_name, norm_args))

    def detect_loop(self) -> Optional[tuple]:
        """Возвращает (tool_name, norm_args, count), если последние N вызовов идентичны."""
        if len(self.recent_calls) < self.threshold:
            return None

        last_n = list(self.recent_calls)[-self.threshold:]
        first = last_n[0]

        if all(call == first for call in last_n):
            return first[0], first[1], self.threshold
        return None

    def reset(self):
        self.recent_calls.clear()

class LLMClient:
    _client = None

    @classmethod
    def get_client(cls) -> OpenAI:
        if cls._client is None:
            cls._client = OpenAI(api_key="lm-studio", base_url=Config.API_URL)
        return cls._client

    @staticmethod
    def call(messages: list[dict], temp: float, timeout: int, tools: list[dict] = None, prefill: str = None):
        messages_to_send = list(messages)
        if prefill:
            messages_to_send.append({"role": "assistant", "content": prefill})
        try:
            response = LLMClient.get_client().chat.completions.create(
                model=Config.MODEL_NAME,
                messages=messages_to_send,
                temperature=temp,
                max_tokens=7500,
                tools=tools,
                parallel_tool_calls=False,
                timeout=timeout,
                reasoning_effort="none"
            )
            msg = response.choices[0].message
            if prefill and msg.content and not msg.content.startswith(prefill):
                msg.content = prefill + msg.content

            usage = None
            if hasattr(response, 'usage') and response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            return msg, None, usage
        except Exception as e:
            return None, str(e), None

@dataclass
class Message(ABC):
    """Базовый класс для всех сообщений."""

    @abstractmethod
    def to_api_dict(self) -> dict[str, Any]:
        """Рендерит чистый словарь для API (без служебных полей)."""
        pass

    @abstractmethod
    def render(self) -> str:
        """Человекочитаемое представление для UI."""
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
    """Отдельный вызов инструмента внутри AssistantMessage."""
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
        """Используется для определения «пустых» ассистентов,
        которые только вызывали инструменты."""
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
    execution_time_ms: Optional[float] = None  # Пример расширяемости

    # Дополнительные служебные поля, которые НЕ уходят в API
    retry_count: int = 0  # Сколько раз перезапускали этот вызов?

    def to_api_dict(self) -> dict[str, Any]:
        """Только поля, нужные API — никаких is_error, retry_count."""
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


# ─── Обновлённый ChatHistory ───

class ChatHistory:
    def __init__(self, system_prompt: str):
        self._messages: list[Message] = [SystemMessage(system_prompt)]

    def add(self, msg: Message):
        self._messages.append(msg)

    def extend(self, msgs: list[Message]):
        self._messages.extend(msgs)

    def get_all_api(self) -> list[dict[str, Any]]:
        """Отдаёт чистые словари для LLM API."""
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
        """Удаляет сообщения с конца до первого UserMessage."""
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
        """Удаляет диапазон сообщений."""
        if not (0 <= start_id < len(self._messages)):
            return f"Error: Invalid start_id {start_id}"

        if end_id == -1 or end_id >= len(self._messages):
            end_id = len(self._messages) - 1

        safe_start = max(start_id, Config.AFTER_SYSTEM_PROMPT)
        safe_end = end_id

        if safe_start > safe_end:
            return f"{ENVIRONMENT_PREFIX} Success (Nothing to delete)"

        del self._messages[safe_start:safe_end + 1]
        return f'{ENVIRONMENT_PREFIX} Success'

    def edit_message(self, idx: int, new_text: str, old_text: str = '') -> str:
        """Редактирует сообщение по индексу."""
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

        if not msg.content.strip() and idx >= Config.AFTER_SYSTEM_PROMPT:
            self.delete_range(idx, idx)
            return 'Replacing to empty text led to deleting the message block.'

        return f'{ENVIRONMENT_PREFIX} Success'

    def normalize(self):
        """Полная очистка истории для обеспечения валидности структуры API."""
        if len(self._messages) <= Config.AFTER_SYSTEM_PROMPT:
            return

        raw = self._messages
        valid = [raw[0]]  # system

        # Поиск первого UserMessage
        first_user_idx = Config.AFTER_SYSTEM_PROMPT
        while first_user_idx < len(raw) and not isinstance(raw[first_user_idx], UserMessage):
            first_user_idx += 1

        if first_user_idx >= len(raw):
            self._messages = valid
            return

        valid.append(raw[first_user_idx])

        for i in range(first_user_idx + 1, len(raw)):
            msg = raw[i]
            last = valid[-1]

            # ToolResult должен идти после AssistantMessage с tool_calls
            if isinstance(msg, ToolResult):
                if isinstance(last, AssistantMessage) and last.has_tool_calls():
                    # Проверяем, что tool_call_id совпадает
                    call_ids = [tc.id for tc in last.tool_calls]
                    if msg.tool_call_id in call_ids:
                        valid.append(msg)
                continue

            # Склеиваем одинаковые роли
            if type(msg) == type(last) and isinstance(msg, (UserMessage, AssistantMessage)):
                last.content = (last.content or "") + "\n" + (msg.content or "")
                if isinstance(msg, AssistantMessage) and msg.has_tool_calls():
                    last.tool_calls = last.tool_calls + msg.tool_calls
                continue

            valid.append(msg)

        self._messages = valid

    def save(self, path: str):
        """Сохраняет в JSON. Метаданные теряются — только API-поля."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.get_all_api(), f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        """Загружает из JSON (восстанавливает простые типы)."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list) or not data:
            raise ValueError(f"{ENVIRONMENT_PREFIX} Invalid history format")

        self._messages = []
        for d in data:
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
                # При загрузке не знаем, ошибка или нет — считаем успехом
                self._messages.append(ToolResult.success(
                    tool_call_id=d["tool_call_id"],
                    name=d.get("name", "unknown"),
                    content=d["content"]
                ))
            else:
                raise ValueError(f"Unknown role: {role}")

    def find_last_tool_result(self, tool_name: str) -> Optional[ToolResult]:
        """Находит последний результат конкретного инструмента."""
        for msg in reversed(self._messages):
            if isinstance(msg, ToolResult) and msg.name == tool_name:
                return msg
        return None

    def get_messages_by_type(self, msg_type) -> list[Message]:
        """Фильтр сообщений по типу."""
        return [msg for msg in self._messages if isinstance(msg, msg_type)]

class LLMAgent:
    def __init__(
            self,
            system_prompt: str = "You are a helpful assistant",
            temp: float = 0.05,
            timeout: int = 1800,
            tools_config: Union[list[str], dict, None] = None,
            on_render: Callable = lambda x: None,
            on_confirm: Callable[[str, dict], bool] = lambda n, a: True,
            on_system_msg: Callable[[str], None] = lambda x: None,
            external_plugins: dict[str, Callable] = None,
            max_context_tokens: int = 16384
    ):
        self.history = ChatHistory(system_prompt)
        self.temp = temp
        self.timeout = timeout
        self.thinking_enabled = True
        self.on_render = on_render
        self.on_confirm = on_confirm
        self.on_system_msg = on_system_msg
        self.self_consistency_mode = False
        self.sc_samples = 3
        self.token_tracker = TokenUsageTracker(max_context_tokens)

        # Сбор инструментов
        internal_tools = self._collect_internal_tools([self.__class__])

        if external_plugins:
            for name, func in external_plugins.items():
                if name in internal_tools:
                    print(f"Warning: External tool '{name}' conflicts with internal tool. Skipping.")
                    continue
                internal_tools[name] = self._build_tool_dict(func, is_instance_method=False)

        self._all_tools = internal_tools
        self._filter_tools(tools_config)

        self.loop_detector = LoopDetector(threshold=2)
        self._temp_boost_active = False
        self._original_temp = temp

    @staticmethod
    def _build_tool_dict(func: Callable, is_instance_method: bool) -> dict:
        """Хелпер для устранения дублирования при сборке словаря инструмента."""
        return {
            "schema": func._tool_schema,
            "handler": func,
            "is_instance_method": is_instance_method,
            "requires_confirmation": getattr(func, '_requires_confirmation', False)
        }

    def _get_effective_prefill(self, custom_prefill: Optional[str]) -> Optional[str]:
        """Хелпер для устранения дублирования логики подстановки prefill."""
        if custom_prefill:
            return custom_prefill
        if not self.thinking_enabled:
            return "</think>\n\n"
        return None

    @staticmethod
    def _build_assistant_msg(msg_obj, clean_content: str) -> AssistantMessage:
        tool_calls = []
        if msg_obj.tool_calls:
            for tc in msg_obj.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments
                ))
        return AssistantMessage(content=clean_content, tool_calls=tool_calls)

    def _break_tool_loop(self, tool_name: str, norm_args: str, count: int):
        """Исправленная версия для работы с доменными объектами."""
        messages = self.history._messages
        matched_indices = []
        duplicate_tool_ids = set()

        for i in range(Config.AFTER_SYSTEM_PROMPT, len(messages)):
            msg = messages[i]
            if not isinstance(msg, AssistantMessage):
                continue
            for tc in msg.tool_calls:
                if (tc.name == tool_name and
                        self.loop_detector.normalize_args(tc.arguments) == norm_args):
                    matched_indices.append(i)
                    if len(matched_indices) > 1:
                        duplicate_tool_ids.add(tc.id)

        if len(matched_indices) <= 1:
            return

        to_remove = set(matched_indices[1:])

        for i, msg in enumerate(messages):
            if isinstance(msg, ToolResult) and msg.tool_call_id in duplicate_tool_ids:
                to_remove.add(i)

        for idx in sorted(to_remove, reverse=True):
            del messages[idx]

        warning = (
            f"{ENVIRONMENT_PREFIX} Tool '{tool_name}' with identical args "
            f"was called a few times. The system removed all duplicates."
        )
        messages.append(UserMessage(warning))
        self.on_system_msg(f"[LOOP DETECTED] '{tool_name}' ×{count}")

    # ─── Инструменты ───

    @tool(description="Get short indexed current history with ids")
    def get_msg_ids(self, chars_per_message: int = 35):
        history = self.history
        if len(history) <= Config.AFTER_SYSTEM_PROMPT:
            return f"{ENVIRONMENT_PREFIX} История пока пустая."

        lines = ["=== SHORT DIALOG ==="]
        for i in range(Config.AFTER_SYSTEM_PROMPT, len(history)):
            msg = history[i]

            if isinstance(msg, SystemMessage):
                continue
            elif isinstance(msg, UserMessage):
                role = "USER"
                content = msg.content
            elif isinstance(msg, AssistantMessage):
                role = "ASSISTANT"
                content = msg.content
                if msg.has_tool_calls():
                    tc_info = ", ".join(tc.name for tc in msg.tool_calls)
                    content += f" [Tools: {tc_info}]"
            elif isinstance(msg, ToolResult):
                role = "TOOL"
                content = msg.content
                prefix = f"[{msg.name}] " if msg.name else ""
                lines.append(
                    f"[id {i}] {role}: {prefix}{content[:chars_per_message].strip()}"
                )
                continue
            else:
                continue

            if len(content) > chars_per_message:
                content = content[:chars_per_message] + " ..."

            lines.append(f"[id {i}] {role}: {content.strip()}")

        return f"{ENVIRONMENT_PREFIX} Your current history:\n" + "\n".join(lines)

    def _collect_internal_tools(self, classes):
        tools = {}
        for klass in classes:
            for name in dir(klass):
                raw = klass.__dict__.get(name)
                if raw is None:
                    continue

                # Определяем, является ли это методом экземпляра
                is_instance_method = callable(raw) and not isinstance(raw, (staticmethod, classmethod, type))

                # Получаем саму функцию
                func = raw.__func__ if isinstance(raw, staticmethod) else raw

                if hasattr(func, '_is_tool'):
                    tools[func._tool_name] = self._build_tool_dict(func, is_instance_method)
        return tools

    def _generate_draft_with_tool_suggestions(self, draft_messages, prefill, draft_temp, draft_timeout):
        prefill_val = self._get_effective_prefill(prefill)
        for _ in range(3):
            msg_obj, err, _ = LLMClient.call(
                draft_messages, draft_temp, draft_timeout,
                tools=self.tools if self.tools else None,
                prefill=prefill_val
            )
            if msg_obj and not err:
                return msg_obj
        return None

    def _chat_self_consistent(self, message: str, prefill: str = None) -> str:
        user_message = UserMessage(content=message)
        self.history.add(user_message)
        messages_base = self._prepare_messages_for_api()

        self.on_system_msg(f"Generating {self.sc_samples} drafts...")
        drafts = []
        for _ in range(self.sc_samples):
            draft = self._generate_draft_with_tool_suggestions(messages_base, prefill, 0.7, self.timeout)
            if draft:
                drafts.append(draft)

        if not drafts:
            return "Failed to generate any valid draft"

        # Формируем промпт для синтеза
        draft_texts = []
        for i, draft in enumerate(drafts, 1):
            content = draft.content or "(no text)"
            if draft.tool_calls:
                tc_names = [f"{tc.function.name}(...)" for tc in draft.tool_calls]
                content += f"\n[Suggested tools: {', '.join(tc_names)}]"
            draft_texts.append(f"--- Draft {i} ---\n{content}")

        synthesis_prompt = (
                f"{ENVIRONMENT_PREFIX} Here are drafts from multiple reasoning paths:\n" +
                "\n".join(draft_texts) +
                "\n\n Analyse them and synthesize the finishing correct answer, paying attention to suggested tools."
        )

        synthesis_messages = messages_base + [{"role": "user", "content": synthesis_prompt}]
        current_prefill = self._get_effective_prefill(prefill)
        msg_obj, err, usage = LLMClient.call(
            synthesis_messages, temp=0.2, timeout=self.timeout,
            tools=self.tools if self.tools else None,
            prefill=current_prefill
        )
        if usage:
            self.token_tracker.update_from_usage(usage)

        if err or not msg_obj:
            return f"API Error during synthesis: {err}"

        clean_content = msg_obj.content.replace("</think>", "").strip()
        assistant_msg = self._build_assistant_msg(msg_obj, clean_content)

        if not msg_obj.tool_calls:
            self.history.add(assistant_msg)
            self.on_render(assistant_msg)
            return clean_content

        # Выполняем инструменты – получаем список ToolResult
        tool_results = self._execute_tools(assistant_msg.tool_calls)

        self.history.add(assistant_msg)
        self.on_render(assistant_msg)

        self.history.extend(tool_results)
        for tr in tool_results:
            self.on_render(tr)

        # Собираем followup_messages как словари
        followup_dicts = (
                synthesis_messages +
                [assistant_msg.to_api_dict()] +
                [tr.to_api_dict() for tr in tool_results]
        )

        final_obj, final_err, final_usage = LLMClient.call(
            followup_dicts,
            temp=0.1,
            timeout=self.timeout,
            tools=None
        )
        if final_usage:
            self.token_tracker.update_from_usage(final_usage)

        if final_err or not final_obj:
            return clean_content or "Tool executed successfully"

        final_content = final_obj.content.strip()
        final_assistant_msg = self._build_assistant_msg(final_obj, final_content)
        self.history.add(final_assistant_msg)
        self.on_render(final_assistant_msg)
        return final_content

    def _filter_tools(self, config):
        all_names = set(self._all_tools.keys())
        if config is None or config == "all":
            active = all_names
        elif isinstance(config, list):
            active = set(config) & all_names
        elif isinstance(config, dict) and "exclude" in config:
            active = all_names - set(config["exclude"])
        else:
            raise ValueError("Invalid tools_config")

        self._all_tools = {k: v for k, v in self._all_tools.items() if k in active}
        self.tools = [v['schema'] for v in self._all_tools.values()]

    def _prepare_messages_for_api(self) -> list[dict]:
        """Готовит сообщения для API (с заголовками токенов)."""
        self.history.normalize()
        messages = self.history.get_all_api()  # Уже чистые словари!

        for msg in messages:
            if msg["role"] == "user":
                msg["content"] = self.token_tracker.format_info_header() + msg["content"]
            elif msg["role"] == "tool":
                prefix = self.token_tracker.format_info_header() + f"{ENVIRONMENT_PREFIX} [Tool Result]\n"
                msg["content"] = prefix + msg["content"]

        return messages

    def _execute_tools(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Выполняет инструменты, возвращает доменные объекты."""
        results = []
        for tc in tool_calls:
            name = tc.name
            args_str = tc.arguments or "{}"

            self.loop_detector.add_call(name, args_str)

            tool_info = self._all_tools.get(name)
            if not tool_info:
                results.append(ToolResult.error(tc.id, name, f"Unknown tool '{name}'"))
                continue

            args_dict = None
            try:
                args_dict = json.loads(args_str) if args_str != "{}" else {}
            except Exception as e:
                results.append(ToolResult.error(tc.id, name, f"Invalid JSON: {e}"))
                continue

            if tool_info.get('requires_confirmation', False):
                if not self.on_confirm(name, args_dict):
                    results.append(ToolResult.user_denied(tc.id, name))
                    continue

            try:
                handler = tool_info['handler']
                if tool_info['is_instance_method']:
                    full_result = handler(self, **args_dict)
                else:
                    full_result = handler(**args_dict)
                content = str(full_result) if full_result is not None else "Tool executed successfully"
                results.append(ToolResult.success(tc.id, name, content))
            except Exception as e:
                self.on_system_msg(f"[ERROR] Tool '{name}' FAILED: {e}")
                results.append(ToolResult.error(tc.id, name, str(e)))

        loop_info = self.loop_detector.detect_loop()
        if loop_info:
            tool_name, norm_args, count = loop_info
            self._break_tool_loop(tool_name, norm_args, count)
            self._temp_boost_active = True

        return results

    def _process_llm_response(self, message_obj) -> str:
        if not message_obj:
            return "Empty response"

        content = message_obj.content or ""
        clean_content = content.replace("</think>", "").strip()

        assistant_msg = self._build_assistant_msg(message_obj, clean_content)
        self.history.add(assistant_msg)
        self.on_render(assistant_msg)

        if not message_obj.tool_calls:
            return clean_content

        # Выполняем инструменты
        tool_results = self._execute_tools(assistant_msg.tool_calls)
        self.history.extend(tool_results)
        for tr in tool_results:
            self.on_render(tr)

        # Очистка старых ошибок
        self._prune_all_failed_tool_calls_except_last()

        return clean_content

    def _prune_all_failed_tool_calls_except_last(self):
        """Удаляет все неудачные вызовы, кроме последнего обмена."""
        if len(self.history) <= Config.AFTER_SYSTEM_PROMPT + 1:
            return

        # Находим последний AssistantMessage
        last_assistant_idx = -1
        for i in range(len(self.history) - 1, Config.AFTER_SYSTEM_PROMPT - 1, -1):
            if isinstance(self.history[i], AssistantMessage):
                last_assistant_idx = i
                break

        if last_assistant_idx == -1:
            return

        indices_to_remove = set()

        i = Config.AFTER_SYSTEM_PROMPT
        while i < len(self.history):
            msg = self.history[i]

            # Ищем пустой AssistantMessage с tool_calls
            if (isinstance(msg, AssistantMessage) and
                    msg.has_tool_calls()):

                # Проверяем следующее сообщение
                if (i + 1 < len(self.history) and
                        isinstance(self.history[i + 1], ToolResult)):

                    tool_result = self.history[i + 1]

                    # Удаляем только ошибки, не трогая user_denied
                    if tool_result.is_error and not tool_result.is_user_denied:
                        # Не трогаем последний обмен
                        if i < last_assistant_idx:
                            indices_to_remove.add(i)
                            indices_to_remove.add(i + 1)
                            i += 2
                            continue
            i += 1

        if indices_to_remove:
            for idx in sorted(indices_to_remove, reverse=True):
                del self.history._messages[idx]
            self.on_system_msg(
                f"[CLEANUP] Removed {len(indices_to_remove)} messages "
                f"({len(indices_to_remove)//2} failed calls)"
            )
            self.history.normalize()

    def chat(self, message: str, max_iter: int = 5, prefill: str = None) -> str:
        if self.self_consistency_mode:
            return self._chat_self_consistent(message, prefill)

        # Сохраняем исходное сообщение без заголовка (заголовок добавится позже в _prepare_messages_for_api)
        user_msg = UserMessage(content=message)
        self.history.add(user_msg)

        current_prefill = self._get_effective_prefill(prefill)

        for i in range(max_iter):
            step_prefill = current_prefill if i == 0 else None
            messages_to_send = self._prepare_messages_for_api()

            current_temp = self.temp
            if self._temp_boost_active:
                current_temp = Config.BOOST_TEMP
                self._temp_boost_active = False

            message_obj, err, usage = LLMClient.call(
                messages_to_send, current_temp, self.timeout,
                tools=self.tools if self.tools else None,
                prefill=step_prefill
            )

            if usage:
                self.token_tracker.update_from_usage(usage)

            if err:
                return f"API Error: {err}"

            # Обработка ответа
            result_text = self._process_llm_response(message_obj)

            # Если инструментов не было, это финальный ответ
            if not message_obj.tool_calls:
                return result_text

        return "Max iterations reached without final answer"

    # ---------- Инструменты ----------

    @tool(description="Get short indexed current history with ids")
    def get_msg_ids(self, chars_per_message: int = 35):
        history = self.history._messages
        if len(history) <= Config.AFTER_SYSTEM_PROMPT:
            return f"{ENVIRONMENT_PREFIX} История пока пустая."

        lines = ["=== SHORT DIALOG ==="]
        for i in range(Config.AFTER_SYSTEM_PROMPT, len(history)):
            msg = history[i]

            # Определяем роль и контент
            if isinstance(msg, SystemMessage):
                continue
            elif isinstance(msg, UserMessage):
                role = "USER"
                content = msg.content
            elif isinstance(msg, AssistantMessage):
                role = "ASSISTANT"
                content = msg.content or ""
                if msg.has_tool_calls():
                    tc_info = ", ".join(tc.name for tc in msg.tool_calls)
                    content += f" [Tools: {tc_info}]"
            elif isinstance(msg, ToolResult):
                role = "TOOL"
                prefix = f"[{msg.name}] "
                content = prefix + msg.content
                # Добавляем короткий индикатор
                if msg.is_error:
                    content += " ❌"
                elif msg.is_user_denied:
                    content += " 🚫"
            else:
                continue

            if len(content) > chars_per_message:
                content = content[:chars_per_message] + " ..."

            lines.append(f"[id {i}] {role}: {content.strip()}")

        return f"{ENVIRONMENT_PREFIX} Your current history:\n" + "\n".join(lines)

    @tool(description="Edits a specific message in the history",
          requires_confirmation=True,
          id=("int", "ID of the message to edit"),
          old=("str", "Optional exact substr to replace. Empty str replaces whole text"),
          new=("str", "Text to insert in place of old"))
    def edit_message(self, id: int, new: str, old: str = ''):
        return self.history.edit_message(id, new, old)

    @tool(description="Deletes a range of messages from dialog history",
          requires_confirmation=True,
          start_id=("int", "Starting message ID to delete"),
          end_id=("int", "Optional ending message ID (-1 for last)"))
    def delete_messages(self, start_id: int, end_id: int = -1):
        return self.history.delete_range(start_id, end_id)

class ConsoleUI:
    @staticmethod
    def render_message(msg: Message):
        if isinstance(msg, SystemMessage):
            return
        elif isinstance(msg, UserMessage):
            print(f"\n👤 User: {msg.content}")
        elif isinstance(msg, AssistantMessage):
            if msg.content.strip():
                print('\n' + '=' * 15)
                print(f"🤖 Agent: {msg.content}")
                print('=' * 15)
            for tc in msg.tool_calls:
                print(f"🛠️ [Tool Call: {tc.name}({tc.arguments})]")
        elif isinstance(msg, ToolResult):
            display = str(msg.content)
            if len(display) > 300:
                display = display[:300] + "\n... [TRUNCATED]"
            print(f"✅ [Result '{msg.name}']: {display}")

    @staticmethod
    def system_msg(text: str):
        """Выводит системные логи/предупреждения (например, о детекте цикла)."""
        if text:
            print(f"\n⚙️ [System]: {text}")

    @staticmethod
    def confirm_action(name: str, args: dict) -> bool:
        print(f"\n[WARNING] Tool '{name}' modifies state")
        resp = input("Execute? (y/n): ").strip().lower()
        return resp == 'y'


class CLI:
    def __init__(self, agent: LLMAgent):
        self.agent = agent
        self.pending_prefill = None
        self.multiline = False
        self.commands = {
            "/regen": self.cmd_regen,
            "/think_on": self.cmd_think_on,
            "/think_off": self.cmd_think_off,
            "/prefill": self.cmd_prefill,
            "/save": self.cmd_save,
            "/load": self.cmd_load,
            "/consistent": self.cmd_consistent,
            "/multiline": self.cmd_multiline
        }

    def cmd_regen(self, parts: list[str]):
        n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
        for _ in range(n - 1):
            self.agent.history.pop_until_user()
        user_msg = self.agent.history.pop_until_user()

        if not user_msg:
            ConsoleUI.system_msg("Cannot find a preceding user message to regenerate")
            return

        ConsoleUI.system_msg(f"Regenerating response for: '{user_msg}'")
        self.agent.chat(user_msg, max_iter=5, prefill=self.pending_prefill)

    def cmd_think_on(self, parts: list[str]):
        self.agent.thinking_enabled = True
        ConsoleUI.system_msg("Force think ENABLED")

    def cmd_think_off(self, parts: list[str]):
        self.agent.thinking_enabled = False
        ConsoleUI.system_msg("Force think DISABLED (using dirty hack)")

    def cmd_prefill(self, parts: list[str]):
        if len(parts) > 1:
            self.pending_prefill = parts[1]
            ConsoleUI.system_msg(f"Next message will start with prefill: '{self.pending_prefill}'")
        else:
            self.pending_prefill = None
            ConsoleUI.system_msg("Prefill cleared")

    def cmd_save(self, parts: list[str]):
        filename = parts[1] if len(parts) > 1 else "default_history.json"
        try:
            self.agent.history.save(filename)
            ConsoleUI.system_msg(f"History saved to '{filename}'")
        except Exception as e:
            ConsoleUI.system_msg(f"Error saving history: {e}")

    def cmd_load(self, parts: list[str]):
        filename = parts[1] if len(parts) > 1 else "default_history.json"
        if not os.path.exists(filename):
            ConsoleUI.system_msg(f"File '{filename}' not found")
            return
        try:
            self.agent.history.load(filename)
            ConsoleUI.system_msg(f"History loaded. Total messages: {len(self.agent.history)}")
            print("\n" + "="*40 + "\n🔄 LOADED HISTORY:\n" + "="*40)
            for msg in self.agent.history.get_all():
                ConsoleUI.render_message(msg)
        except Exception as e:
            ConsoleUI.system_msg(f"Error loading history: {e}")

    def cmd_consistent(self, parts: list[str]):
        """Toggle self-consistency mode on/off"""
        self.agent.self_consistency_mode = not self.agent.self_consistency_mode
        status = "ON" if self.agent.self_consistency_mode else "OFF"
        ConsoleUI.system_msg(f"Self-consistency mode turned {status}")

    def cmd_multiline(self, parts: list[str]):
        self.multiline = not self.multiline
        status = "ON" if self.multiline else "OFF"
        ConsoleUI.system_msg(f"Multiline input mode turned {status}. Type Ctrl+D to finish the input.")

    def read_until_marker(self, marker="/mm"):
        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                break

            if line.strip() == marker:
                break
            lines.append(line)

        return "\n".join(lines)

    def run(self):
        ConsoleUI.system_msg("Ready. Type 'exit' to quit")
        ConsoleUI.system_msg(f"Commands: {', '.join(self.commands.keys())}")

        while True:
            if self.multiline:
                print("\n👤 User: ")
            inp = self.read_until_marker() if self.multiline else input("\n👤 User: ").strip()
            if self.multiline:
                self.multiline = False
            if not inp: continue
            if inp.lower() in ("exit", "quit"): break

            if inp.startswith("/"):
                try: parts = shlex.split(inp)
                except ValueError as e:
                    ConsoleUI.system_msg(f"Error parsing command: {e}")
                    continue

                handler = self.commands.get(parts[0].lower())
                if handler: handler(parts)
                else: ConsoleUI.system_msg(f"Unknown command: {parts[0]}")
                continue

            self.agent.chat(inp, max_iter=10, prefill=self.pending_prefill)


if __name__ == "__main__":
    # 1. Загружаем внешние плагины
    external_tools = load_external_plugins("tools")
    print(f"Loaded external tools: {list(external_tools.keys())}")

    sys_prompt = (
        "* You are assistant that can call tools. Speak Russian.\n"
        "* You're launched in a special environment.\n"
        f"{ENVIRONMENT_PREFIX} prefix means the system outputs something, it's not what user said.\n"
        f"When remaining tokens amount is about equal to spent tokens, it's time to start cleaning up your context. "
        f"You can compress or delete some messages to free some more tokens to continue working."
    )

    # 2. Передаем их в агент
    agent = LLMAgent(
        system_prompt=sys_prompt,
        tools_config="all",
        external_plugins=external_tools,
        on_render=ConsoleUI.render_message,
        on_confirm=ConsoleUI.confirm_action,
        on_system_msg=ConsoleUI.system_msg,
        max_context_tokens=65535
    )

    cli = CLI(agent)
    cli.run()