import json
import shlex
import importlib
import inspect
import os
import sys
from collections import deque
from typing import List, Dict, Union, Callable, Optional

from openai import OpenAI

from universal_agents.tool import tool, ENVIRONMENT_PREFIX


def load_external_plugins(plugins_dir="tools"):
    """
    Загружает все .py файлы из директории и возвращает словарь
    {имя_функции: функция}, где функции помечены декоратором @tool.
    """
    external_tools = {}

    if not os.path.exists(plugins_dir):
        return external_tools

    # Добавляем корень проекта в путь, чтобы импорты внутри плагинов работали
    root_path = os.path.abspath(os.path.join(plugins_dir, ".."))
    if root_path not in sys.path:
        sys.path.insert(0, root_path)

    for filename in os.listdir(plugins_dir):
        if filename.endswith(".py") and not filename.startswith("__"):
            module_name = filename[:-3]
            try:
                spec = importlib.util.spec_from_file_location(module_name, os.path.join(plugins_dir, filename))
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Ищем функции с атрибутом _is_tool
                for name, obj in inspect.getmembers(module):
                    if hasattr(obj, '_is_tool') and callable(obj):
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
    AFTER_SYSTEM_PROMPT = 1


class LoopDetector:
    def __init__(self, max_history: int = 12, threshold: int = 3):
        self.max_history = max_history
        self.threshold = threshold
        self.recent_calls = deque(maxlen=max_history)  # (tool_name, normalized_args)

    def normalize_args(self, args_str: str) -> str:
        """Приводим аргументы к каноническому виду (порядок ключей неважен)"""
        if not args_str or args_str.strip() in ("{}", "", "null"):
            return ""
        try:
            parsed = json.loads(args_str)
            # sort_keys=True + separators — гарантирует идентичную строку при одинаковых значениях
            return json.dumps(parsed, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
        except Exception:
            return args_str.strip()

    def add_call(self, tool_name: str, arguments: str):
        norm_args = self.normalize_args(arguments)
        self.recent_calls.append((tool_name, norm_args))

    def detect_loop(self) -> Optional[tuple[str, str, int]]:
        """
        Возвращает (tool_name, normalized_args, count) если один и тот же вызов
        с полностью одинаковыми параметрами повторился threshold раз подряд.
        """
        if len(self.recent_calls) < self.threshold:
            return None

        last_calls = list(self.recent_calls)

        # Проверяем последние threshold вызовов — все ли они полностью одинаковые
        window = last_calls[-self.threshold:]
        first_call = window[0]

        if all(call == first_call for call in window):
            tool_name, norm_args = first_call
            return tool_name, norm_args, self.threshold

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
    def call(messages: List[Dict], temp: float, timeout: int,
             tools: List[Dict] = None, prefill: str = None):
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

            if prefill and msg.content:
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
        user_msg = None
        while len(self._messages) > Config.AFTER_SYSTEM_PROMPT and self._messages[-1]["role"] != "user":
            self._messages.pop()
        if len(self._messages) > Config.AFTER_SYSTEM_PROMPT and self._messages[-1]["role"] == "user":
            user_msg = self._messages.pop()["content"]
        return user_msg

    def edit_message(self, idx: int, new_text: str, old_text: str = '') -> str:
        if not (0 <= idx < len(self._messages)):
            return f"{ENVIRONMENT_PREFIX} Error: Invalid message index {idx}"

        msg = self._messages[idx]
        if not old_text.strip():
            msg["content"] = new_text
        elif old_text not in msg["content"]:
            return f"Error: Substr '{old_text}' not found in message {idx}"
        else:
            msg["content"] = msg["content"].replace(old_text, new_text, 1)

        if not msg["content"].strip():
            self.delete_range(idx, idx)
            return 'Replacing to empty text led to deleting the message block.'
        return f'{ENVIRONMENT_PREFIX} Success'

    def delete_range(self, start_id: int, end_id: int = -1):
        if not (0 <= start_id < len(self._messages)):
            return f"Error: Invalid start_id {start_id}"

        if end_id == -1 or end_id >= len(self._messages):
            end_id = len(self._messages) - 1

        start_id = max(start_id, Config.AFTER_SYSTEM_PROMPT)
        if start_id > end_id:
            start_id, end_id = end_id, start_id
        if start_id == end_id:
            end_id += 1

        actual_start = start_id
        while actual_start > Config.AFTER_SYSTEM_PROMPT and self._messages[actual_start]["role"] != "user":
            actual_start -= 1

        actual_end = end_id
        while actual_end < len(self._messages) and self._messages[actual_end]["role"] != "user":
            actual_end += 1

        del self._messages[actual_start:actual_end]
        return f'{ENVIRONMENT_PREFIX} Success'

    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self._messages, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list) and len(data) > 0 and "role" in data[0]:
            self._messages = data
        else:
            raise ValueError(f"{ENVIRONMENT_PREFIX} Invalid history format")

    def normalize(self):
        "После system всегда должен быть user, не tool и не assistant"
        if len(self._messages) <= Config.AFTER_SYSTEM_PROMPT:
            return

        i = Config.AFTER_SYSTEM_PROMPT
        while i < len(self._messages) and self._messages[i]["role"] != "user":
            i += 1
        if i > Config.AFTER_SYSTEM_PROMPT:
            del self._messages[Config.AFTER_SYSTEM_PROMPT:i]


class LLMAgent:
    def __init__(self,
                 system_prompt: str = "You are a helpful assistant",
                 temp: float = 0.25,
                 timeout: int = 1800,
                 tools_config: Union[List[str], Dict, None] = None,
                 on_render: Callable[[Dict], None] = lambda x: None,
                 on_confirm: Callable[[str, Dict], bool] = lambda n, a: True,
                 on_system_msg: Callable[[str], None] = lambda x: None,
                 external_plugins: Dict[str, Callable] = None):

        self.history = ChatHistory(system_prompt)
        self.temp = temp
        self.timeout = timeout
        self.thinking_enabled = True

        self.on_render = on_render
        self.on_confirm = on_confirm
        self.on_system_msg = on_system_msg

        self.self_consistency_mode = False
        self.sc_samples = 3

        # 1. Собираем внутренние инструменты (из классов FS и самого агента)
        internal_tools = self._collect_internal_tools([self.__class__])

        # 2. Обрабатываем внешние плагины
        if external_plugins:
            for name, func in external_plugins.items():
                # Проверяем, не конфликтует ли имя
                if name in internal_tools:
                    print(f"Warning: External tool '{name}' conflicts with internal tool. Skipping.")
                    continue

                internal_tools[name] = {
                    "schema": func._tool_schema,
                    "handler": func,
                    "is_instance_method": False, # Внешние функции обычно статические
                    "requires_confirmation": getattr(func, '_requires_confirmation', False)
                }

        self._all_tools = internal_tools
        self._filter_tools(tools_config)

        self.loop_detector = LoopDetector(max_history=12, threshold=3)
        self._temp_boost_active = False
        self._next_prefill = None
        self._original_temp = temp

    def _break_tool_loop(self, tool_name: str, norm_args: str, count: int):
        """
        Оставляем только ПЕРВЫЙ tool_call с такими args.
        Удаляем все последующие assistant/tool сообщения,
        связанные через tool_call_id.
        """

        messages = self.history._messages

        matched_calls = []

        # Ищем assistant tool_calls
        for i in range(Config.AFTER_SYSTEM_PROMPT, len(messages)):
            msg = messages[i]

            if msg["role"] != "assistant":
                continue

            for tc in msg.get("tool_calls", []):
                if (
                        tc["function"]["name"] == tool_name
                        and self.loop_detector.normalize_args(
                    tc["function"].get("arguments", "{}")
                ) == norm_args
                ):
                    matched_calls.append({
                        "msg_index": i,
                        "tool_call_id": tc["id"]
                    })

        if len(matched_calls) <= 1:
            return

        # Оставляем первый вызов
        keep = matched_calls[0]
        duplicates = matched_calls[1:]

        to_remove = set()

        # assistant messages
        for dup in duplicates:
            to_remove.add(dup["msg_index"])

        # tool results по tool_call_id
        duplicate_ids = {d["tool_call_id"] for d in duplicates}

        for i, msg in enumerate(messages):
            if (
                    msg["role"] == "tool"
                    and msg.get("tool_call_id") in duplicate_ids
            ):
                to_remove.add(i)

        # Удаляем с конца
        for idx in sorted(to_remove, reverse=True):
            del messages[idx]

        self.on_system_msg(
            f"[LOOP] Kept first call "
            f"(tool_call_id={keep['tool_call_id']}), "
            f"removed {len(to_remove)} messages"
        )

        warning = f"""{ENVIRONMENT_PREFIX} LOOP DETECTED!
    
    Tool '{tool_name}' with identical parameters was called {count} times.
    
    I kept only the FIRST execution and removed all subsequent duplicates.
    
    Do not repeat this tool call with the same parameters again.
    """

        messages.append({
            "role": "user",
            "content": warning
        })

        self.on_system_msg(
            f"[LOOP DETECTED] '{tool_name}' ×{count}"
        )

    def _collect_internal_tools(self, classes):
        """Собирает инструменты только из переданных классов (внутренние)"""
        tools = {}
        for klass in classes:
            for name in dir(klass):
                raw = klass.__dict__.get(name)
                if raw is None: continue

                is_instance_method = callable(raw) and not isinstance(raw, (staticmethod, classmethod, type))
                func = raw.__func__ if isinstance(raw, staticmethod) else raw

                if not hasattr(func, '_is_tool'): continue

                tools[func._tool_name] = {
                    "schema": func._tool_schema,
                    "handler": func,
                    "is_instance_method": is_instance_method,
                    "requires_confirmation": getattr(func, '_requires_confirmation', False)
                }
        return tools

    def _generate_draft_with_tool_suggestions(self, draft_messages, prefill, draft_temp, draft_timeout):
        """
        Генерирует черновик с полным доступом к схемам инструментов,
        но не исполняет tool_calls. Возвращает полный message_obj.
        """
        prefill = prefill or ("</think>\n\n" if not self.thinking_enabled else None)

        for attempt in range(3):
            msg_obj, err = LLMClient.call(
                draft_messages, draft_temp, draft_timeout,
                tools=self.tools if self.tools else None,
                prefill=prefill
            )
            if msg_obj and not err:
                return msg_obj
        return None

    def _chat_self_consistent(self, message: str, prefill: str = None) -> str:
        user_message = {"role": "user", "content": message}
        self.history.add(user_message)
        messages_base = self._prepare_messages_for_api()

        self.on_system_msg(f"Generating {self.sc_samples} drafts with tool suggestions..")
        drafts = []

        for i in range(self.sc_samples):
            draft = self._generate_draft_with_tool_suggestions(
                messages_base, prefill, 0.85, self.timeout)
            if draft:
                drafts.append(draft)

        if not drafts:
            return "Failed to generate any valid draft"

        draft_texts = []
        for i, draft in enumerate(drafts, 1):
            content = draft.content or "(no text)"
            if draft.tool_calls:
                tc_names = [f"{tc.function.name}({tc.function.arguments})" for tc in draft.tool_calls]
                content += f"\n[Suggested tool calls: {', '.join(tc_names)}]"
            draft_texts.append(f"--- Draft {i} ---\n{content}")

        synthesis_prompt = (
                f"[SYSTEM] Here are drafts from multiple reasoning paths"
                + "\n".join(draft_texts) +
                "\n\nBased on the drafts and the user's request, provide the final answer. "
        )

        synthesis_messages = messages_base + [{"role": "user", "content": synthesis_prompt}]
        current_prefill = prefill or None
        current_prefill = current_prefill or ("</think>\n\n" if not self.thinking_enabled else None)

        msg_obj, err = LLMClient.call(
            synthesis_messages,
            temp=0.1,
            timeout=self.timeout,
            tools=self.tools if self.tools else None,
            prefill=current_prefill
        )

        if err:
            return f"API Error during synthesis: {err}"
        if not msg_obj:
            return "Empty response during synthesis"

        # Если модель сразу ответила без инструментов
        if not msg_obj.tool_calls:
            final_content = msg_obj.content.replace("</think>", "").strip()
            assistant_msg = {"role": "assistant", "content": final_content}
            self.history.add(assistant_msg)
            self.on_render(assistant_msg)
            return final_content

        tool_results = self._execute_tools(msg_obj.tool_calls)

        clean_content = msg_obj.content.replace("</think>", "").strip()

        assistant_msg = {
            "role": "assistant",
            "content": clean_content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                } for tc in msg_obj.tool_calls
            ]
        }
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
            tools=None
        )

        if final_err or not final_obj:
            return clean_content or "Tool executed successfully"

        final_content = final_obj.content.strip()
        final_assistant_msg = {"role": "assistant", "content": final_content}
        self.history.add(final_assistant_msg)
        self.on_render(final_assistant_msg)
        return final_content

    def _filter_tools(self, config):
        all_names = set(self._all_tools.keys())
        if config is None or config == "all": active = all_names
        elif isinstance(config, list): active = set(config) & all_names
        elif isinstance(config, dict) and "exclude" in config: active = all_names - set(config["exclude"])
        else: raise ValueError("Invalid tools_config")

        self._all_tools = {k: v for k, v in self._all_tools.items() if k in active}
        self.tools = [v['schema'] for v in self._all_tools.values()]

    def _prepare_messages_for_api(self) -> List[Dict]:
        self.history.normalize()
        messages_to_send = [self.history[0].copy()]
        for idx, msg in enumerate(self.history.get_all()[Config.AFTER_SYSTEM_PROMPT:]):
            copy = msg.copy()
            messages_to_send.append(copy)

        return messages_to_send

    def _execute_tools(self, tool_calls) -> List[Dict]:
        results = []

        for tc in tool_calls:
            name = tc.function.name
            args_str = tc.function.arguments or "{}"

            # === Регистрируем вызов для обнаружения петли ===
            self.loop_detector.add_call(name, args_str)

            # === Выполнение инструмента ===
            tool_info = self._all_tools.get(name)
            if not tool_info:
                results.append({
                    "tool_call_id": tc.id,
                    "role": "tool",
                    "name": name,
                    "content": f"Error: Unknown tool '{name}'"
                })
                continue

            # Проверка подтверждения (если требуется)
            if tool_info.get('requires_confirmation', False):
                args_dict = json.loads(args_str) if args_str != "{}" else {}
                if not self.on_confirm(name, args_dict):
                    results.append({
                        "tool_call_id": tc.id,
                        "role": "tool",
                        "name": name,
                        "content": f"{ENVIRONMENT_PREFIX} Execution cancelled by user"
                    })
                    continue

            # Вызов обработчика
            try:
                args_dict = json.loads(args_str) if args_str != "{}" else {}
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
                "tool_call_id": tc.id,
                "role": "tool",
                "name": name,
                "content": content
            })


        loop_info = self.loop_detector.detect_loop()
        if loop_info:
            tool_name, norm_args, count = loop_info
            self._break_tool_loop(tool_name, norm_args, count)
            self._temp_boost_active = True

        return results

    # ---------- Инструменты ----------
    @tool(description="Get short indexed current history with ids")
    def get_msg_ids(self, chars_per_message: int = 35):
        history = self.history.get_all()

        if len(history) <= Config.AFTER_SYSTEM_PROMPT:
            return f"{ENVIRONMENT_PREFIX} История пока пустая (только system prompt)."

        lines = ["=== SHORT DIALOG ==="]

        for i in range(Config.AFTER_SYSTEM_PROMPT, len(history)):
            msg = history[i]
            role = msg["role"]
            content = msg.get("content", "") or ""

            # Сокращаем длинные сообщения
            if len(content) > chars_per_message:
                content = content[:chars_per_message] + " ..."

            # Для tool calls добавляем информацию
            if role == "assistant" and msg.get("tool_calls"):
                tc_info = ", ".join(tc["function"]["name"] for tc in msg["tool_calls"])
                content += f"  [Tool calls: {tc_info}]"

            # Для tool results показываем имя инструмента
            tool_name = msg.get("name", "")
            if tool_name:
                prefix = f"[{tool_name}] "
            else:
                prefix = ""

            lines.append(f"[id {i}] {role.upper()}: {prefix}{content.strip()}")

        result = "\n".join(lines)
        return f"{ENVIRONMENT_PREFIX} Your current history:\n{result}\n\nUse these IDs to edit & delete messages."

    @tool(description="Edits a specific message in the history",
          requires_confirmation=True,
          id=("int", "ID of the message to edit"),
          old=("str", "Optional exact substr to replace. Empty str replaces whole text"),
          new=("str", "Text to insert in place of old"))
    def edit_message(self, id: int, new: str, old: str = ''):
        res = self.history.edit_message(id, new, old)
        return res

    @tool(description="Deletes a range of messages from dialog history",
          requires_confirmation=True,
          start_id=("int", "Starting message ID to delete"),
          end_id=("int", "Optional ending message ID (-1 for last)"))
    def delete_messages(self, start_id: int, end_id: int = -1):
        err = self.history.delete_range(start_id, end_id)
        return err

    def chat(self, message: str, max_iter: int = 5, prefill: str = None) -> str:
        if self.self_consistency_mode:
            return self._chat_self_consistent(message, prefill)
        user_msg = {"role": "user", "content": message}
        self.history.add(user_msg)

        current_prefill = prefill
        if not self.thinking_enabled and not current_prefill:
            current_prefill = "</think>\n\n"

        for i in range(max_iter):
            step_prefill = current_prefill if i == 0 else None
            messages_to_send = self._prepare_messages_for_api()

            current_prefill = self._next_prefill or (step_prefill if i == 0 else None)
            current_temp = self.temp
            if self._temp_boost_active:
                current_temp = min(0.9, self.temp + 0.4)
                self._temp_boost_active = False

            message_obj, err = LLMClient.call(
                messages_to_send,
                current_temp,
                self.timeout,
                tools=self.tools if self.tools else None,
                prefill=current_prefill
            )

            self._next_prefill = None   # сбрасываем после использования

            if err: return f"API Error: {err}"
            if not message_obj: return "Empty response"

            content = message_obj.content or ""
            clean_content = content.replace("</think>", "").strip()

            assistant_msg = {"role": "assistant", "content": clean_content}
            if message_obj.tool_calls:
                assistant_msg["tool_calls"] = [
                    {"id": tc.id, "type": tc.type, "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in message_obj.tool_calls
                ]

            self.history.add(assistant_msg)
            self.on_render(assistant_msg)

            if not message_obj.tool_calls:
                return clean_content

            tool_results = self._execute_tools(message_obj.tool_calls)
            self.history.extend(tool_results)
            for tr in tool_results:
                self.on_render(tr)

        return "Max iterations reached without final answer"


class ConsoleUI:
    @staticmethod
    def render_message(msg: Dict):
        role = msg.get("role")
        content = msg.get("content")

        if role == "system": return
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
            if len(display) > 300: display = display[:300] + "\n... [TRUNCATED]"
            print(f"✅ [Result '{msg.get('name')}]: {display}")

    @staticmethod
    def confirm_action(name: str, args: Dict) -> bool:
        print(f"\n[WARNING] Tool '{name}' modifies state")
        return input("Execute? (y/N): ").strip().lower() == 'y'

    @staticmethod
    def system_msg(msg: str):
        print(f"[System] {msg}")


class CLI:
    def __init__(self, agent: LLMAgent):
        self.agent = agent
        self.pending_prefill = None
        self.commands = {
            "/regen": self.cmd_regen,
            "/think_on": self.cmd_think_on,
            "/think_off": self.cmd_think_off,
            "/prefill": self.cmd_prefill,
            "/save": self.cmd_save,
            "/load": self.cmd_load,
            "/consistent": self.cmd_consistent,
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

    def run(self):
        ConsoleUI.system_msg("Ready. Type 'exit' to quit")
        ConsoleUI.system_msg(f"Commands: {', '.join(self.commands.keys())}")

        while True:
            inp = input("\n👤 User: ").strip()
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