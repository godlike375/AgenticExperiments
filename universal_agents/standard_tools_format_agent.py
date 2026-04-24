import datetime
import json
import os
import re
import shlex
from collections import defaultdict
from typing import List, Dict, Union, Any

from openai import OpenAI

class Config:
    API_URL = "http://localhost:1234/v1"
    MODEL_NAME = "local-model"
    AFTER_SYSTEM_PROMPT = 1


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
                max_tokens=11500,
                tools=tools,
                timeout=timeout,
                reasoning_effort="none"
            )
            return response.choices[0].message, None
        except Exception as e:
            return None, str(e)


# Универсальный декоратор для описания инструментов
def tool(description="", **params):
    def decorator(func):
        func._is_tool = True
        func._tool_name = func.__name__
        func._tool_desc = description or (func.__doc__ or "").split("\n")[0].strip()
        func._tool_params = params
        return func
    return decorator


class FS:
    @staticmethod
    def _format_size(size_bytes: int) -> str:
        if size_bytes < 1024:
            return f"{size_bytes}B"
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f}{unit}" if unit != 'B' else f"{size_bytes}{unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f}TB"

    @staticmethod
    def _count_hidden_size(root_path: str) -> int:
        total = 0
        try:
            for entry in os.scandir(root_path):
                try:
                    if entry.is_file():
                        total += entry.stat().st_size
                    elif entry.is_dir():
                        total += FS._count_hidden_size(entry.path)
                except PermissionError:
                    continue
        except (PermissionError, FileNotFoundError):
            pass
        return total

    @staticmethod
    def _build_tree(root_path: str, depth: int = 0, density: int = 4) -> str:
        try:
            entries = list(os.scandir(root_path))
        except PermissionError:
            return f"{'  ' * depth}[Permission Denied]"
        except FileNotFoundError:
            return f"{'  ' * depth}[Path Not Found]"

        if depth > 0 and len(entries) > density:
            size = FS._format_size(FS._count_hidden_size(root_path))
            return f"{'  ' * depth}[{len(entries)} items TRUNCATED, {size}]"

        dirs = sorted([e for e in entries if e.is_dir()], key=lambda x: x.name.lower())
        files = sorted([e for e in entries if e.is_file()], key=lambda x: x.name.lower())

        lines = []
        for entry in dirs + files:
            mtime = datetime.datetime.fromtimestamp(entry.stat().st_mtime).strftime("%Y-%m-%d")
            prefix = f"{'  ' * depth}"
            if entry.is_dir():
                lines.append(f"{prefix}{entry.name}/ ({mtime})")
                if sub := FS._build_tree(entry.path, depth + 1, density):
                    lines.append(sub)
            else:
                lines.append(f"{prefix}{entry.name} ({FS._format_size(entry.stat().st_size)})")
        return "\n".join(lines)

    @staticmethod
    @tool(description="Gets file content or directory tree.",
          path=("str", "Optional path to file/directory (default '.'). Use '..' for parent."))
    def open(path: str = '.'):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path not found: {path}")
        try:
            mtime = datetime.datetime.fromtimestamp(os.stat(path).st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            if os.path.isfile(path):
                try:
                    with open(path, 'r', encoding='utf-8', errors='strict') as f:
                        content = f.read()
                    return f"File: {path}\nModified: {mtime}\nContent:\n---\n{content}"
                except UnicodeDecodeError:
                    return "Error: Cannot read binary files (failed UTF-8 decode)."
            elif os.path.isdir(path):
                return f"Directory Tree: {os.path.abspath(path)}\nModified: {mtime}\n\n{FS._build_tree(path)}"
            raise RuntimeError("Unexpected file type")
        except Exception as e:
            raise PermissionError(f"Error accessing {path}: {e}")

    @staticmethod
    @tool(description="Searches files by regex pattern.",
          pattern=("str", "Regex pattern to search for."),
          path=("str", "Optional base directory (default current)."))
    def search_files(pattern: str, path: str = "."):
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Base path not found: {path}")
        folders_map = defaultdict(list)
        for root, _, files in os.walk(path):
            for f in files:
                if re.search(pattern, f):
                    rel = os.path.relpath(root, path)
                    folders_map["." if rel == "." else rel].append(f)
        if not folders_map:
            return "No matches found."
        lines = []
        for folder in sorted(folders_map.keys()):
            lines.append(f"{folder}/:")
            for fname in sorted(folders_map[folder]):
                lines.append(f"  - {fname}")
        return "\n".join(lines)

    @staticmethod
    @tool(description="Gets or changes current working directory.",
          path=("str", "Optional new working directory. Use '..' for parent."))
    def cwd(path: str = None):
        if path:
            try:
                os.chdir(path)
                return 'Success'
            except Exception as e:
                return f"Error changing directory: {e}"
        return os.getcwd()


class LLMAgent:
    def __init__(self, system_prompt: str = "You are a helpful assistant.",
                 temp: float = 0.25, timeout: int = 1800,
                 tools_config: Union[List[str], Dict, None] = None):
        # 1. Сбор ВСЕХ доступных инструментов из FS и собственных методов
        self._all_tools = self._collect_tools([FS, self.__class__])

        # 2. Фильтрация инструментов
        all_names = set(self._all_tools.keys())
        if tools_config is None or tools_config == "all":
            active = all_names
        elif isinstance(tools_config, list):
            active = set(tools_config) & all_names
        elif isinstance(tools_config, dict) and "exclude" in tools_config:
            active = all_names - set(tools_config["exclude"])
        else:
            raise ValueError("Invalid tools_config")

        self._all_tools = {k: v for k, v in self._all_tools.items() if k in active}
        self.tools = [v['schema'] for v in self._all_tools.values()]

        # 3. Инициализация состояния
        self.history = [{"role": "system", "content": system_prompt}]
        self.temp = temp
        self.timeout = timeout
        self.edit_mode = False
        self.thinking_enabled = True

    @staticmethod
    def _collect_tools(classes):
        """Сканирует переданные классы и собирает все @tool-декорированные функции."""
        tools = {}
        for klass in classes:
            for name in dir(klass):
                # Используем __dict__ для точного определения типа дескриптора
                raw = klass.__dict__.get(name)
                if raw is None:
                    continue

                if isinstance(raw, staticmethod):
                    func = raw.__func__
                    is_instance_method = False
                elif isinstance(raw, classmethod):
                    continue
                elif callable(raw) and not isinstance(raw, type):
                    func = raw
                    is_instance_method = True
                else:
                    continue

                if not hasattr(func, '_is_tool'):
                    continue

                # Строим схему
                params_desc = func._tool_params
                properties = {}
                required = []
                for pname, (ptype, pdesc) in params_desc.items():
                    properties[pname] = {"type": ptype, "description": pdesc}
                    if not pdesc.lower().startswith("optional"):
                        required.append(pname)

                schema = {
                    "type": "function",
                    "function": {
                        "name": func._tool_name,
                        "description": func._tool_desc,
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required
                        }
                    }
                }
                tools[func._tool_name] = {
                    "schema": schema,
                    "handler": func,
                    "is_instance_method": is_instance_method
                }
        return tools

    # ---------- Вспомогательные методы ----------
    def hide_messages_id(self):
        self.edit_mode = False

    def _handle_regen(self, num_messages=1, pending_prefill=''):
        user_message_to_resend = None
        for _ in range(num_messages):
            while self.history and self.history[-1]["role"] != "user":
                self.history.pop()
            if self.history and self.history[-1]["role"] == "user":
                user_message_to_resend = self.history.pop()["content"]
        if not user_message_to_resend:
            return "Cannot find a preceding user message to regenerate."
        print(f"\n[REGEN] Regenerating response for: '{user_message_to_resend}'")
        return self.chat(user_message_to_resend, max_iter=5, prefill=pending_prefill)

    def _execute_tools(self, tool_calls):
        results = []
        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                tool_info = self._all_tools.get(name)
                if not tool_info:
                    msg = f"Error: Unknown tool '{name}'"
                    results.append({
                        "tool_call_id": tc.id, "role": "tool",
                        "name": name, "content": msg
                    })
                    continue

                handler = tool_info['handler']
                if tool_info['is_instance_method']:
                    full_result = handler(self, **args)
                else:
                    full_result = handler(**args)

                print(f"[Tool] {name}({args}) -> {str(full_result)}")
                if full_result is not None:
                    results.append({
                        "tool_call_id": tc.id, "role": "tool",
                        "name": name, "content": str(full_result)
                    })
            except Exception as e:
                error_msg = f"Tool '{name}' FAILED: {e}"
                print(f"[ERROR] {error_msg}")
                results.append({
                    "tool_call_id": tc.id, "role": "tool",
                    "name": name, "content": error_msg
                })
        return results

    # ---------- Инструменты (имя = имя метода) ----------
    @tool(description="Enables visibility of message IDs.")
    def get_msg_ids(self):
        self.edit_mode = True
        return 'Success'

    @tool(description="Edits a specific message in the history.",
          id=("int", "ID of the message to edit."),
          old=("str", "Optional exact substr to replace. Empty str replaces whole text."),
          new=("str", "Text to insert in place of old."))
    def edit_message(self, id: int, new: str, old: str = ''):
        if not (0 <= id < len(self.history)):
            return f"Error: Invalid message index {id}."
        msg = self.history[id]
        if not old.strip():
            msg["content"] = new
        elif old not in msg["content"]:
            return f"Error: Substr '{old}' not found in message {id}."
        else:
            msg["content"] = msg["content"].replace(old, new, 1)

        if not msg["content"].strip():
            self.delete_messages(id, id)
            self.hide_messages_id()
            return 'Replacing to empty text led to deleting the message block.'
        self.hide_messages_id()
        return 'Success'

    @tool(description="Deletes a range of messages from dialog history.",
          start_id=("int", "Starting message ID to delete."),
          end_id=("int", "Optional ending message ID (-1 for last)."))
    def delete_messages(self, start_id: int, end_id: int = -1):
        if not (0 <= start_id < len(self.history)):
            return f"Error: Invalid start_id {start_id}."

        if end_id == -1 or end_id >= len(self.history):
            end_id = len(self.history) - 1

        start_id = max(start_id, Config.AFTER_SYSTEM_PROMPT)

        if start_id > end_id:
            start_id, end_id = end_id, start_id
        if start_id == end_id:
            end_id += 1

        actual_start = start_id
        while actual_start > 1 and self.history[actual_start]["role"] != "user":
            actual_start -= 1
        actual_end = end_id
        while actual_end < len(self.history) and self.history[actual_end]["role"] != "user":
            actual_end += 1

        del self.history[actual_start:actual_end]
        self.hide_messages_id()
        return None

    # ---------- Основной цикл общения ----------
    def chat(self, message: str, max_iter: int = 5, prefill: str = None) -> str:
        self.history.append({"role": "user", "content": message})

        current_prefill = prefill
        if not self.thinking_enabled and not current_prefill:
            current_prefill = "</think>\n\n"

        for i in range(max_iter):
            step_prefill = current_prefill if i == 0 else None

            messages_to_send = [self.history[0].copy()]
            for idx, msg in enumerate(self.history[Config.AFTER_SYSTEM_PROMPT:]):
                copy = msg.copy()
                if self.edit_mode:
                    copy["content"] = f"[id {idx + Config.AFTER_SYSTEM_PROMPT}]\n{copy['content']}"
                messages_to_send.append(copy)

            message_obj, err = LLMClient.call(
                messages_to_send,
                self.temp, self.timeout,
                tools=self.tools if self.tools else None,
                prefill=step_prefill
            )

            if err:
                return f"API Error: {err}"
            if not message_obj:
                return "Empty response"

            content = message_obj.content or ""
            full_content = (step_prefill + content) if step_prefill else content
            clean_content = full_content.replace("</think>", "").strip()

            if clean_content:
                print('\n' + '=' * 15)
                print(f"Agent: {clean_content}")
                print('=' * 15)

            assistant_msg = {"role": "assistant", "content": clean_content}
            if message_obj.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                    }
                    for tc in message_obj.tool_calls
                ]
            self.history.append(assistant_msg)

            if not message_obj.tool_calls:
                return clean_content

            tool_results = self._execute_tools(message_obj.tool_calls)
            self.history.extend(tool_results)

        return "Max iterations reached without final answer."


if __name__ == "__main__":
    sys_prompt = (
        "You are a special tool-calling assistant. Use tools to fulfill user requests. "
        "Ask user's confirmation before calling tools. Speak Russian."
    )

    agent = LLMAgent(
        system_prompt=sys_prompt,
        tools_config="all"
    )
    print("Ready. Type 'exit' to quit.")
    print("Commands: /regen, /think_on, /think_off, /prefill <text>")

    pending_prefill = None

    while True:
        inp = input("\nUser: ").strip()
        if not inp:
            continue

        if inp.startswith("/"):
            try:
                parts = shlex.split(inp)
            except ValueError as e:
                print(f"[System] Error parsing command: {e}")
                continue

            cmd = parts[0].lower()

            if cmd == "/regen":
                n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
                agent._handle_regen(n, pending_prefill)
            elif cmd == "/think_on":
                agent.thinking_enabled = True
                print("[System] Force think ENABLED.")
            elif cmd == "/think_off":
                agent.thinking_enabled = False
                print("[System] Force think DISABLED (using dirty hack).")
            elif cmd == "/prefill":
                if len(parts) > 1:
                    pending_prefill = parts[1]
                    print(f"[System] Next message will start with prefill: '{pending_prefill}'")
                else:
                    pending_prefill = None
                    print("[System] Prefill cleared.")
            else:
                print(f"Unknown command: {cmd}")
            continue

        if inp.lower() in ("exit", "quit"):
            break

        agent.chat(inp, max_iter=5, prefill=pending_prefill)