import json
import os
import shlex
import sys
import importlib.util
import inspect
from collections import deque
from typing import List, Dict, Union, Callable, Optional

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
    API_URL = "http://192.168.50.196:1234/v1" # "http://localhost:1234/v1"
    MODEL_NAME = "local-model"
    AFTER_SYSTEM_PROMPT = 1  # Индекс, после которого начинается диалог (обычно 1, т.к. 0 - system)

class LoopDetector:
    def __init__(self, max_history: int = 12, threshold: int = 3):
        self.max_history = max_history
        self.threshold = threshold
        self.recent_calls = deque(maxlen=max_history)

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
    def call(messages: List[Dict], temp: float, timeout: int, tools: List[Dict] = None, prefill: str = None):
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

            # Обычно API возвращает полный контент. Если нужно склеить вручную:
            if prefill and msg.content:
                # Проверка, не содержит ли ответ уже префилл (зависит от модели)
                if not msg.content.startswith(prefill):
                    msg.content = prefill + msg.content

            return msg, None
        except Exception as e:
            return None, str(e)

class ChatHistory:
    def __init__(self, system_prompt: str):
        self._messages = [{"role": "system", "content": system_prompt}]

    def add(self, msg: Dict):
        self._messages.append(msg)

    def extend(self, msgs: List[Dict]):
        self._messages.extend(msgs)

    def get_all(self) -> List[Dict]:
        return self._messages

    def __len__(self):
        return len(self._messages)

    def __getitem__(self, idx):
        return self._messages[idx]

    def pop_until_user(self) -> Optional[str]:
        """Удаляет сообщения с конца до первого user message (не включая system)."""
        user_content = None
        # Идем с конца, пока не найдем user или не упремся в system prompt
        while len(self._messages) > Config.AFTER_SYSTEM_PROMPT:
            last_msg = self._messages[-1]
            if last_msg["role"] == "user":
                user_content = self._messages.pop()["content"]
                break
            else:
                self._messages.pop()
        return user_content

    def edit_message(self, idx: int, new_text: str, old_text: str = '') -> str:
        if not (0 <= idx < len(self._messages)):
            return f"{ENVIRONMENT_PREFIX} Error: Invalid message index {idx}"

        msg = self._messages[idx]

        if not old_text.strip():
            msg["content"] = new_text
        else:
            if old_text not in msg["content"]:
                return f"Error: Substr '{old_text}' not found in message {idx}"
            msg["content"] = msg["content"].replace(old_text, new_text, 1)

        # Если сообщение стало пустым, удаляем его (кроме system)
        if not msg["content"].strip() and idx >= Config.AFTER_SYSTEM_PROMPT:
            self.delete_range(idx, idx)
            return 'Replacing to empty text led to deleting the message block.'

        return f'{ENVIRONMENT_PREFIX} Success'

    def delete_range(self, start_id: int, end_id: int = -1):
        if not (0 <= start_id < len(self._messages)):
            return f"Error: Invalid start_id {start_id}"

        if end_id == -1 or end_id >= len(self._messages):
            end_id = len(self._messages) - 1

        # Защита от удаления system prompt
        safe_start = max(start_id, Config.AFTER_SYSTEM_PROMPT)
        safe_end = end_id

        if safe_start > safe_end:
            return f"{ENVIRONMENT_PREFIX} Success (Nothing to delete)"

        # Находим границы блока сообщений пользователя/ассистента, чтобы не ломать структуру пар
        # Простая стратегия: удаляем точно указанный диапазон, но проверяем целостность позже в normalize
        del self._messages[safe_start:safe_end + 1]
        return f'{ENVIRONMENT_PREFIX} Success'

    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self._messages, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list) and data and "role" in data[0]:
            self._messages = data
        else:
            raise ValueError(f"{ENVIRONMENT_PREFIX} Invalid history format")

    def normalize(self):
        """
        Полная очистка истории для обеспечения валидности структуры API.
        """
        if len(self._messages) <= Config.AFTER_SYSTEM_PROMPT:
            return

        raw_msgs = self._messages
        valid_msgs = [raw_msgs[0]]

        # 1. Ищем первое сообщение от пользователя (после system)
        first_user_idx = Config.AFTER_SYSTEM_PROMPT
        while first_user_idx < len(raw_msgs) and raw_msgs[first_user_idx]["role"] != "user":
            first_user_idx += 1

        if first_user_idx >= len(raw_msgs):
            # Если юзер так и не найден, оставляем только системник
            self._messages = valid_msgs
            return

        # Добавляем первого найденного юзера
        valid_msgs.append(raw_msgs[first_user_idx])

        # 2. Проходим по остальным сообщениям
        for i in range(first_user_idx + 1, len(raw_msgs)):
            msg = raw_msgs[i]
            last = valid_msgs[-1]

            # --- ПРАВИЛО 1: Обработка TOOL-сообщений ---
            if msg["role"] == "tool":
                # Tool-сообщение валидно только если ПРЕДЫДУЩЕЕ было Assistant с tool_calls
                # И id этого тула есть в списке вызовов ассистента
                if last["role"] == "assistant" and "tool_calls" in last:
                    call_ids = [tc["id"] for tc in last["tool_calls"]]
                    if msg.get("tool_call_id") in call_ids:
                        valid_msgs.append(msg)
                continue # Если условие не выполнено, тул просто выбрасывается (сирота)

            # --- ПРАВИЛО 2: Чередование USER и ASSISTANT ---
            if msg["role"] == last["role"]:
                if msg["role"] in ["user", "assistant"]:
                    # Склеиваем контент одинаковых ролей, чтобы не нарушать структуру
                    new_content = (last.get("content") or "") + "\n" + (msg.get("content") or "")
                    last["content"] = new_content.strip()
                    # Если у второго ассистента были тул-коллы, переносим их (редкий кейс)
                    if "tool_calls" in msg:
                        last["tool_calls"] = last.get("tool_calls", []) + msg["tool_calls"]
                continue

            valid_msgs.append(msg)

        self._messages = valid_msgs

class LLMAgent:
    def __init__(
            self,
            system_prompt: str = "You are a helpful assistant",
            temp: float = 0.05,
            timeout: int = 1800,
            tools_config: Union[List[str], Dict, None] = None,
            on_render: Callable[[Dict], None] = lambda x: None,
            on_confirm: Callable[[str, Dict], bool] = lambda n, a: True,
            on_system_msg: Callable[[str], None] = lambda x: None,
            external_plugins: Dict[str, Callable] = None
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

        self.loop_detector = LoopDetector(max_history=12, threshold=3)
        self._temp_boost_active = False
        self._original_temp = temp

    @staticmethod
    def _build_tool_dict(func: Callable, is_instance_method: bool) -> Dict:
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
    def _build_assistant_msg(msg_obj, clean_content: str) -> Dict:
        """Хелпер для устранения дублирования при сборке сообщения ассистента с инструментами."""
        assistant_msg = {"role": "assistant", "content": clean_content}
        if msg_obj.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                } for tc in msg_obj.tool_calls
            ]
        return assistant_msg

    def _break_tool_loop(self, tool_name: str, norm_args: str, count: int):
        """Удаляет дублирующиеся вызовы инструментов, оставляя только первый."""
        messages = self.history._messages
        matched_indices = [] # Индексы сообщений assistant с этим tool_call
        duplicate_tool_ids = set()

        # 1. Находим все assistant сообщения с нужным tool call
        for i in range(Config.AFTER_SYSTEM_PROMPT, len(messages)):
            msg = messages[i]
            if msg["role"] != "assistant":
                continue
            for tc in msg.get("tool_calls", []):
                if (tc["function"]["name"] == tool_name and
                        self.loop_detector.normalize_args(tc["function"].get("arguments", "{}")) == norm_args):
                    matched_indices.append(i)
                    # Собираем ID всех найденных вызовов, кроме первого
                    if len(matched_indices) > 1:
                        duplicate_tool_ids.add(tc["id"])

        if len(matched_indices) <= 1:
            return

        # 2. Собираем индексы на удаление (дубликаты assistant + их результаты tool)
        to_remove = set(matched_indices[1:]) # Все assistant кроме первого

        # Добавляем результаты инструментов для удаленных вызовов
        for i, msg in enumerate(messages):
            if msg["role"] == "tool" and msg.get("tool_call_id") in duplicate_tool_ids:
                to_remove.add(i)

        # 3. Удаляем с конца, чтобы не сбить индексы
        for idx in sorted(to_remove, reverse=True):
            del messages[idx]

        warning = (
            f"{ENVIRONMENT_PREFIX} LOOP DETECTED! Tool '{tool_name}' with identical parameters "
            f"was called {count} times. I kept only the FIRST execution and removed duplicates."
        )
        messages.append({"role": "user", "content": warning})
        self.on_system_msg(f"[LOOP DETECTED] '{tool_name}' ×{count}")

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
            msg_obj, err = LLMClient.call(
                draft_messages, draft_temp, draft_timeout,
                tools=self.tools if self.tools else None,
                prefill=prefill_val
            )
            if msg_obj and not err:
                return msg_obj
        return None

    def _chat_self_consistent(self, message: str, prefill: str = None) -> str:
        user_message = {"role": "user", "content": message}
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

        msg_obj, err = LLMClient.call(
            synthesis_messages, temp=0.2, timeout=self.timeout,
            tools=self.tools if self.tools else None,
            prefill=current_prefill
        )

        if err or not msg_obj:
            return f"API Error during synthesis: {err}"

        clean_content = msg_obj.content.replace("</think>", "").strip()
        assistant_msg = self._build_assistant_msg(msg_obj, clean_content)

        # 1. Если модель сразу ответила текстом без инструментов
        if not msg_obj.tool_calls:
            self.history.add(assistant_msg)
            self.on_render(assistant_msg)
            return clean_content

        # 2. Если модель решила вызвать инструменты
        tool_results = self._execute_tools(msg_obj.tool_calls)

        # Сохраняем вызов инструментов и их результаты в основную историю
        self.history.add(assistant_msg)
        self.on_render(assistant_msg)

        self.history.extend(tool_results)
        for tr in tool_results:
            self.on_render(tr)

        followup_messages = synthesis_messages + [assistant_msg] + tool_results

        final_obj, final_err = LLMClient.call(
            followup_messages,
            temp=0.1,
            timeout=self.timeout,
            tools=None  # На финальном этапе инструменты уже не нужны
        )

        if final_err or not final_obj:
            return clean_content or "Tool executed successfully"

        # Сохраняем финальный ответ в основную историю
        final_content = final_obj.content.strip()
        final_assistant_msg = {"role": "assistant", "content": final_content}
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

    def _prepare_messages_for_api(self) -> List[Dict]:
        self.history.normalize()
        # Копируем сообщения, чтобы не мутировать историю при отправке (если вдруг API что-то меняет)
        return [msg.copy() for msg in self.history.get_all()]

    def _execute_tools(self, tool_calls) -> List[Dict]:
        results = []
        for tc in tool_calls:
            name = tc.function.name
            args_str = tc.function.arguments or "{}"

            self.loop_detector.add_call(name, args_str)

            tool_info = self._all_tools.get(name)
            if not tool_info:
                results.append({
                    "tool_call_id": tc.id, "role": "tool", "name": name,
                    "content": f"Error: Unknown tool '{name}'"
                })
                continue

            # Парсим аргументы один раз, сохраняя оригинальное поведение при ошибках парсинга
            args_dict = None
            parse_err = None
            try:
                args_dict = json.loads(args_str) if args_str != "{}" else {}
            except Exception as e:
                parse_err = e

            # Подтверждение
            if tool_info.get('requires_confirmation', False):
                if parse_err:
                    raise parse_err  # Сохраняем оригинальное поведение падения
                if not self.on_confirm(name, args_dict):
                    results.append({
                        "tool_call_id": tc.id, "role": "tool", "name": name,
                        "content": f"User denied tool call remotely. ASK them why they denied and what to do next."
                    })
                    continue

            # Выполнение
            try:
                if parse_err:
                    raise parse_err
                handler = tool_info['handler']
                if tool_info['is_instance_method']:
                    full_result = handler(self, **args_dict)
                else:
                    full_result = handler(**args_dict)
                content = str(full_result) if full_result is not None else "Tool executed successfully"
            except Exception as e:
                self.on_system_msg(f"[ERROR] Tool '{name}' FAILED: {e}")
                content = f"Error: {e}"

            results.append({
                "tool_call_id": tc.id, "role": "tool", "name": name, "content": content
            })

        # Проверка на цикл после выполнения всех инструментов в пакете
        loop_info = self.loop_detector.detect_loop()
        if loop_info:
            tool_name, norm_args, count = loop_info
            self._break_tool_loop(tool_name, norm_args, count)
            self._temp_boost_active = True

        return results

    def _process_llm_response(self, message_obj) -> str:
        """
        Универсальный обработчик ответа LLM.
        Добавляет сообщение в историю, рендерит, выполняет инструменты.
        Возвращает финальный текст ответа или текст после выполнения инструментов.
        """
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
        tool_results = self._execute_tools(message_obj.tool_calls)
        self.history.extend(tool_results)
        for tr in tool_results:
            self.on_render(tr)
        return clean_content

    def chat(self, message: str, max_iter: int = 5, prefill: str = None) -> str:
        if self.self_consistency_mode:
            return self._chat_self_consistent(message, prefill)

        user_msg = {"role": "user", "content": message}
        self.history.add(user_msg)

        current_prefill = self._get_effective_prefill(prefill)

        for i in range(max_iter):
            step_prefill = current_prefill if i == 0 else None
            messages_to_send = self._prepare_messages_for_api()

            # Управление температурой при детекте цикла
            current_temp = self.temp
            if self._temp_boost_active:
                current_temp = min(0.9, self.temp + 0.4)
                self._temp_boost_active = False

            message_obj, err = LLMClient.call(
                messages_to_send, current_temp, self.timeout,
                tools=self.tools if self.tools else None,
                prefill=step_prefill
            )

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
        history = self.history.get_all()
        if len(history) <= Config.AFTER_SYSTEM_PROMPT:
            return f"{ENVIRONMENT_PREFIX} История пока пустая."

        lines = ["=== SHORT DIALOG ==="]
        for i in range(Config.AFTER_SYSTEM_PROMPT, len(history)):
            msg = history[i]
            role = msg["role"]
            content = msg.get("content", "") or ""

            if len(content) > chars_per_message:
                content = content[:chars_per_message] + " ..."

            if role == "assistant" and msg.get("tool_calls"):
                tc_info = ", ".join(tc["function"]["name"] for tc in msg["tool_calls"])
                content += f" [Tools: {tc_info}]"

            tool_name = msg.get("name", "")
            prefix = f"[{tool_name}] " if tool_name else ""
            lines.append(f"[id {i}] {role.upper()}: {prefix}{content.strip()}")

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
    def render_message(msg: Dict):
        role = msg.get("role")
        content = msg.get("content")

        if role == "system":
            return
        elif role == "user":
            print(f"\n👤 User: {content}")
        elif role == "assistant":
            if content:
                print('\n' + '=' * 15)
                print(f"🤖 Agent: {content}")
                print('=' * 15)
            for tc in msg.get("tool_calls", []):
                func = tc.get("function", {})
                print(f"🛠️ [Tool Call: {func.get('name')}({func.get('arguments')})]")
        elif role == "tool":
            display = str(content)
            if len(display) > 300:
                display = display[:300] + "\n... [TRUNCATED]"
            print(f"✅ [Result '{msg.get('name')}']: {display}")

    @staticmethod
    def system_msg(text: str):
        """Выводит системные логи/предупреждения (например, о детекте цикла)."""
        if text:
            print(f"\n⚙️ [System]: {text}")

    @staticmethod
    def confirm_action(name: str, args: Dict) -> bool:
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

    def cmd_regen(self, parts: List[str]):
        n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
        for _ in range(n - 1):
            self.agent.history.pop_until_user()
        user_msg = self.agent.history.pop_until_user()

        if not user_msg:
            ConsoleUI.system_msg("Cannot find a preceding user message to regenerate")
            return

        ConsoleUI.system_msg(f"Regenerating response for: '{user_msg}'")
        self.agent.chat(user_msg, max_iter=5, prefill=self.pending_prefill)

    def cmd_think_on(self, parts: List[str]):
        self.agent.thinking_enabled = True
        ConsoleUI.system_msg("Force think ENABLED")

    def cmd_think_off(self, parts: List[str]):
        self.agent.thinking_enabled = False
        ConsoleUI.system_msg("Force think DISABLED (using dirty hack)")

    def cmd_prefill(self, parts: List[str]):
        if len(parts) > 1:
            self.pending_prefill = parts[1]
            ConsoleUI.system_msg(f"Next message will start with prefill: '{self.pending_prefill}'")
        else:
            self.pending_prefill = None
            ConsoleUI.system_msg("Prefill cleared")

    def cmd_save(self, parts: List[str]):
        filename = parts[1] if len(parts) > 1 else "default_history.json"
        try:
            self.agent.history.save(filename)
            ConsoleUI.system_msg(f"History saved to '{filename}'")
        except Exception as e:
            ConsoleUI.system_msg(f"Error saving history: {e}")

    def cmd_load(self, parts: List[str]):
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

    def cmd_consistent(self, parts: List[str]):
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
        "You are assistant that can call tools. Speak Russian. You're launched in a special environment."
        f" {ENVIRONMENT_PREFIX} prefix means the system tells you something, it's not what user said."
    )

    # 2. Передаем их в агент
    agent = LLMAgent(
        system_prompt=sys_prompt,
        tools_config="all",
        external_plugins=external_tools,
        on_render=ConsoleUI.render_message,
        on_confirm=ConsoleUI.confirm_action,
        on_system_msg=ConsoleUI.system_msg
    )

    cli = CLI(agent)
    cli.run()