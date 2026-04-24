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
            cls._client = OpenAI(
                api_key="lm-studio",
                base_url=Config.API_URL
            )
        return cls._client

    @staticmethod
    def call(messages: List[Dict], temp: float, timeout: int, tools: List[Dict] = None, prefill: str = None):
        # Создаем копию истории для отправки
        messages_to_send = list(messages)

        if prefill:
            messages_to_send.append({"role": "assistant", "content": prefill})

        try:
            client = LLMClient.get_client()
            response = client.chat.completions.create(
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

class FS:
    @staticmethod
    def _format_size(size_bytes: int) -> str:
        if size_bytes < 1024: return f"{size_bytes}B"
        units = ['B', 'KB', 'MB', 'GB', 'TB']
        unit_index = 0
        current_size = float(size_bytes)
        while current_size >= 1024 and unit_index < len(units) - 1:
            current_size /= 1024
            unit_index += 1
        return f"{current_size:.2f}".rstrip('0').rstrip('.') + units[unit_index]

    @staticmethod
    def _count_hidden_size(root_path: str) -> int:
        total_size = 0
        try:
            for entry in os.scandir(root_path):
                try:
                    if entry.is_file(): total_size += entry.stat().st_size
                    elif entry.is_dir(): total_size += FS._count_hidden_size(entry.path)
                except PermissionError: continue
        except (PermissionError, FileNotFoundError): pass
        return total_size

    @staticmethod
    def _build_tree(root_path: str, current_depth: int = 0) -> str:
        DENSITY_LIMIT = 4
        result_lines = []
        try:
            entries = list(os.scandir(root_path))
        except PermissionError: return f"{'  ' * current_depth}[Permission Denied]"
        except FileNotFoundError: return f"{'  ' * current_depth}[Path Not Found]"

        if current_depth > 0 and len(entries) > DENSITY_LIMIT:
            size_str = FS._format_size(FS._count_hidden_size(root_path))
            return f"{'  ' * current_depth}[{len(entries)} nested items TRUNCATED, size={size_str}]"

        dirs = sorted([e for e in entries if e.is_dir()], key=lambda x: x.name.lower())
        files = sorted([e for e in entries if e.is_file()], key=lambda x: x.name.lower())

        for entry in dirs + files:
            stat_info = entry.stat()
            mtime = datetime.datetime.fromtimestamp(stat_info.st_mtime).strftime("%Y-%m-%d")
            prefix = f"{'  ' * current_depth}"
            if entry.is_dir():
                result_lines.append(f"{prefix}{entry.name}/ ({mtime})")
                sub_result = FS._build_tree(entry.path, current_depth + 1)
                if sub_result: result_lines.append(sub_result)
            else:
                size_str = FS._format_size(stat_info.st_size)
                result_lines.append(f"{prefix}{entry.name} ({size_str})")
        return "\n".join(result_lines)

    @staticmethod
    def open(path: str = None) -> str:
        path = path or '.'
        if not os.path.exists(path): raise FileNotFoundError(f"Path not found: {path}")
        try:
            mtime = datetime.datetime.fromtimestamp(os.stat(path).st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            if os.path.isfile(path):
                try:
                    with open(path, 'r', encoding='utf-8', errors='strict') as f: content = f.read()
                    return f"File: {path}\nModified: {mtime}\nContent:\n---\n{content}"
                except UnicodeDecodeError:
                    return f"Error: Cannot read binary files (failed UTF-8 decode)"
            elif os.path.isdir(path):
                return f"Directory Tree: {os.path.abspath(path)}\nModified: {mtime}\n\n{FS._build_tree(path, 0)}"
            raise UnexpectedException("Something went wrong")
        except Exception as e:
            raise PermissionError(f"Error accessing {path}: {e}") from e

    @staticmethod
    def search_files(pattern: str, path: str = ".") -> str:
        if not os.path.isdir(path): raise FileNotFoundError(f"Base path not found: {path}")
        matches = []
        for root, _, files in os.walk(path):
            for f in files:
                if re.search(pattern, f): matches.append(os.path.join(root, f))
        if not matches: return "No matches found."

        folders_map = defaultdict(list)
        for full_path in matches:
            parent_dir, filename = os.path.split(full_path)
            rel_parent = os.path.relpath(parent_dir, path)
            folders_map["." if rel_parent == "." else rel_parent].append(filename)

        lines = []
        for folder in sorted(folders_map.keys()):
            lines.append(f"{folder}/:")
            for file_name in sorted(folders_map[folder]): lines.append(f"  - {file_name}")
        return "\n".join(lines)

    @staticmethod
    @staticmethod
    def cwd(path: str = None) -> str:
        if path:
            try:
                os.chdir(path)
                return 'Success'
            except Exception as e:
                return f"Error changing directory: {e}"
        return f"{os.getcwd()}"

TOOLS_CONFIG = {
    "open": {
        "type": "function",
        "function": {
            "name": "open",
            "description": "Gets file or directory content.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Optional path to file or directory. If not provided then default is '.'. To quick open top (parent) directory use '..' argument."}},
                "required": ["path"],
            },
        },
    },
    "search_files": {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Searches for files by pattern in path",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for"},
                    "path": {"type": "string", "description": "Base directory to search in"}
                },
                "required": ["pattern", "path"],
            },
        },
    },
    "cwd": {
        "type": "function",
        "function": {
            "name": "cwd",
            "description": "Gets current working directory or changes it if 'path' argument is provided",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Optional path to change cwd to. To quick change cwd to top (parent) directory use '..'"}},
            },
        },
    },
    "edit_message": {
        "type": "function",
        "function": {
            "name": "edit_message",
            "description": "Edits specific message in the history by replacing a part of it or the whole text with a substring.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "description": "id of the message to edit"},
                    "old": {"type": "string", "description": "Exact substring to be replaced. Make sure you type it without '[id X]'. If you wanna replace whole text you can pass '' (empty string) to old argument."},
                    "new": {"type": "string", "description": "Text to insert in place of old"}
                },
                "required": ["id", "old", "new"],
            },
        },
    },
    "delete_messages": {
        "type": "function",
        "function": {
            "name": "delete_messages",
            "description": "Deletes a range of messages from dialog history.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_id": {"type": "integer", "description": "Starting message id to delete"},
                    "end_id": {"type": "integer", "description": "Ending inclusive message id to delete. You can type -1 like in Python to delete up to the last."}
                },
                "required": ["start_id", "end_id"],
            },
        },
    },
    "show_msg_ids": {
        "type": "function",
        "function": {
            "name": "show_msg_ids",
            "description": f"Enables the visibility ids for messages. Ids start with {Config.AFTER_SYSTEM_PROMPT}.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
}

FUNCTIONS_REGISTRY = {
    "open": FS.open,
    "search_files": FS.search_files,
    "cwd": FS.cwd,
    # edit_message, delete_messages, show_msg_ids привязываются динамически в __init__
}

class LLMAgent:
    def __init__(self, system_prompt: str = "You are a helpful assistant.", temp: float = 0.25, timeout: int = 1800,
                 tools_config: Union[List[str], Dict, None] = None):

        all_groups = set(TOOLS_CONFIG.keys())
        if tools_config is None or tools_config == "all":
            active_groups = all_groups
        elif isinstance(tools_config, list):
            active_groups = set(tools_config) & all_groups
        elif isinstance(tools_config, dict) and "exclude" in tools_config:
            active_groups = all_groups - set(tools_config["exclude"])
        else:
            raise ValueError("Invalid tools_config")

        self.tools = [TOOLS_CONFIG[name] for name in active_groups]
        self.edit_mode = False  # Флаг режима редактирования

        # Привязываем функции.
        self.functions = {}
        for name in active_groups:
            if name == "edit_message":
                self.functions[name] = self.edit_message
            elif name == "delete_messages":
                self.functions[name] = self.delete_messages
            elif name == "show_msg_ids":
                self.functions[name] = self.show_msg_ids
            else:
                self.functions[name] = FUNCTIONS_REGISTRY[name]

        self.history = [{"role": "system", "content": system_prompt}]
        self.temp = temp
        self.timeout = timeout
        self.thinking_enabled = True

    def show_msg_ids(self) -> str:
        """Метод для включения видимости ID сообщений"""
        self.edit_mode = True
        return 'Success'

    def hide_messages_id(self) -> str:
        """Метод для выключения видимости ID сообщений"""
        self.edit_mode = False

    def delete_messages(self, start_id: int = 1, end_id: int = -1) -> str:
        """Удаляет диапазон сообщений, расширяя его до полных блоков (user -> assistant -> tools)"""
        if not (0 <= start_id < len(self.history)):
            return f"Error: Invalid start_id {start_id}."

        if end_id == -1 or end_id >= len(self.history):
            end_id = len(self.history) - 1
            # TODO: добавить подтверждение от LLM если он перепутал аргументы и собирается удалить всю историю

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

    def edit_message(self, id: int, new: str, old: str = '') -> str:
        """Метод для редактирования истории самим LLM"""
        if not (0 <= id < len(self.history) - 1):
            return f"Error: Invalid message index {id}."

        msg = self.history[id]

        if not old.strip():
            msg["content"] = new
        elif old not in msg["content"]:
            return f"Error: Substring '{old}' not found in message {id}. Make sure you typed exact substring without [id X] prefix."
        else:
            # Заменяем только первое вхождение
            msg["content"] = msg["content"].replace(old, new, 1)

        # Если после редактирования сообщение стало пустым - удаляем весь блок
        if not msg["content"].strip():
            self.delete_messages(id, id)
            self.hide_messages_id()
            return 'Replacing to empty text leaded to deleting messages block'

        self.hide_messages_id()
        return 'Success'

    def _handle_regen(self, num_messages: int = 1, pending_prefill: str = '') -> str:
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

    def _execute_tools(self, tool_calls: List[Any]) -> List[Dict]:
        results = []
        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                print(f"[Tool] Calling {name}({args})")

                if name in self.functions:
                    full_result = self.functions[name](**args)
                    if full_result is not None:
                        print(f"[RESULT] {str(full_result)[:200]}...")
                        results.append({"tool_call_id": tc.id, "role": "tool", "name": name, "content": str(full_result)})
                else:
                    error_msg = f"Error: Unknown tool '{name}'"
                    results.append({"tool_call_id": tc.id, "role": "tool", "name": name, "content": error_msg})
            except Exception as e:
                error_msg = f"Tool '{name}' FAILED: {e}"
                print(f"[ERROR] {error_msg}")
                results.append({"tool_call_id": tc.id, "role": "tool", "name": name, "content": error_msg})
        return results

    def chat(self, message: str, max_iter: int = 5, prefill: str = None) -> str:
        self.history.append({"role": "user", "content": message})

        current_prefill = prefill
        if not self.thinking_enabled and not current_prefill:
            current_prefill = "</think>\n\n"

        for i in range(max_iter):
            step_prefill = current_prefill if i == 0 else None

            # --- ИНЖЕКЦИЯ НОМЕРОВ СООБЩЕНИЙ (ЗАВИСИТ ОТ self.edit_mode) ---
            messages_to_send = []
            msg_copy = self.history[0].copy()
            messages_to_send.append(msg_copy)
            for idx, msg in enumerate(self.history[Config.AFTER_SYSTEM_PROMPT:]):
                msg_copy = msg.copy()
                if self.edit_mode:
                    msg_copy["content"] = f"[id {idx + Config.AFTER_SYSTEM_PROMPT}]\n{msg_copy['content']}"
                messages_to_send.append(msg_copy)

            message_obj, err = LLMClient.call(
                messages_to_send,
                self.temp, self.timeout,
                tools=self.tools if self.tools else None,
                prefill=step_prefill
            )

            if err: return f"API Error: {err}"
            if not message_obj: return "Empty response"

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
                    {"id": tc.id, "type": tc.type, "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
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
        "You are a special tool-calling assistant. Use tools to fulfill user requests. Ask user's confirmation before calling tools. Speak Russian."
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
        if not inp: continue

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