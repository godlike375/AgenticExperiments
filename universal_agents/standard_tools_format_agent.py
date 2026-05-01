import datetime
import json
import os
import re
import shlex
from collections import defaultdict
from typing import List, Dict, Union, Callable, Optional

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


def tool(description="", requires_confirmation=False, **params):
    def decorator(func):
        func._is_tool = True
        func._tool_name = func.__name__
        func._requires_confirmation = requires_confirmation

        properties = {}
        required = []
        for pname, (ptype, pdesc) in params.items():
            properties[pname] = {"type": ptype, "description": pdesc}
            if not pdesc.lower().startswith("optional"):
                required.append(pname)

        func._tool_schema = {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": description or (func.__doc__ or "").split("\n")[0].strip(),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }
        return func
    return decorator


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
            return f"Error: Invalid message index {idx}."

        msg = self._messages[idx]
        if not old_text.strip():
            msg["content"] = new_text
        elif old_text not in msg["content"]:
            return f"Error: Substr '{old_text}' not found in message {idx}."
        else:
            msg["content"] = msg["content"].replace(old_text, new_text, 1)

        if not msg["content"].strip():
            self.delete_range(idx, idx)
            return 'Replacing to empty text led to deleting the message block.'
        return 'Success'

    def delete_range(self, start_id: int, end_id: int = -1):
        if not (0 <= start_id < len(self._messages)):
            return f"Error: Invalid start_id {start_id}."

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
        return None

    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self._messages, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list) and len(data) > 0 and "role" in data[0]:
            self._messages = data
        else:
            raise ValueError("Invalid history format")


class FS:
    @staticmethod
    def _format_size(size_bytes: int) -> str:
        if size_bytes < 1024: return f"{size_bytes}B"
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
                    if entry.is_file(): total += entry.stat().st_size
                    elif entry.is_dir(): total += FS._count_hidden_size(entry.path)
                except PermissionError: continue
        except (PermissionError, FileNotFoundError): pass
        return total

    @staticmethod
    def _build_tree(root_path: str, depth: int = 0, density: int = 4) -> str:
        try: entries = list(os.scandir(root_path))
        except PermissionError: return f"{'  ' * depth}[Permission Denied]"
        except FileNotFoundError: return f"{'  ' * depth}[Path Not Found]"

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
    @tool(description="Gets file content or dir tree.",
          path=("str", "Optional path to file/dir (default '.'). Use '..' for parent."))
    def open(path: str = '.'):
        if not os.path.exists(path): raise FileNotFoundError(f"Path not found: {path}")
        try:
            mtime = datetime.datetime.fromtimestamp(os.stat(path).st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            if os.path.isfile(path):
                try:
                    with open(path, 'r', encoding='utf-8', errors='strict') as f:
                        return f"File: {path}\nModified: {mtime}\nContent:\n---\n{f.read()}"
                except UnicodeDecodeError:
                    return "Error: Cannot read binary files (failed UTF-8 decode)."
            elif os.path.isdir(path):
                return f"Directory Tree: {os.path.abspath(path)}\nModified: {mtime}\n\n{FS._build_tree(path)}"
            raise RuntimeError("Unexpected file type")
        except Exception as e:
            raise PermissionError(f"Error accessing {path}: {e}")

    @staticmethod
    @tool(description="Searches files by regex.",
          pattern=("str", "Regex to search for."),
          path=("str", "Optional base dir (default '.')."))
    def search_files(pattern: str, path: str = "."):
        if not os.path.isdir(path): raise FileNotFoundError(f"Base path not found: {path}")
        folders_map = defaultdict(list)
        for root, _, files in os.walk(path):
            for f in files:
                if re.search(pattern, f):
                    rel = os.path.relpath(root, path)
                    folders_map["." if rel == "." else rel].append(f)
        if not folders_map: return "No matches."

        lines = []
        for folder in sorted(folders_map.keys()):
            lines.append(f"{folder}/:")
            for fname in sorted(folders_map[folder]):
                lines.append(f"  - {fname}")
        return "\n".join(lines)

    @staticmethod
    @tool(description="Gets or changes current working dir.",
          path=("str", "Optional new working dir. Use '..' for parent."))
    def cwd(path: str = None):
        if path:
            try:
                os.chdir(path)
                return 'Success'
            except Exception as e:
                return f"Error changing cwd: {e}"
        return os.getcwd()


class LLMAgent:
    def __init__(self,
                 system_prompt: str = "You are a helpful assistant.",
                 temp: float = 0.25,
                 timeout: int = 1800,
                 tools_config: Union[List[str], Dict, None] = None,
                 on_render: Callable[[Dict], None] = lambda x: None,
                 on_confirm: Callable[[str, Dict], bool] = lambda n, a: True,
                 on_system_msg: Callable[[str], None] = lambda x: None):

        self.history = ChatHistory(system_prompt)
        self.temp = temp
        self.timeout = timeout
        self.edit_mode = False
        self.thinking_enabled = True

        self.on_render = on_render
        self.on_confirm = on_confirm
        self.on_system_msg = on_system_msg

        self._all_tools = self._collect_tools([FS, self.__class__])
        self._filter_tools(tools_config)

    def _collect_tools(self, classes):
        tools = {}
        for klass in classes:
            for name in dir(klass):
                raw = klass.__dict__.get(name)
                if raw is None: continue

                # Хак для определения, нужно ли передавать self (агента) в инструмент
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

    def _filter_tools(self, config):
        all_names = set(self._all_tools.keys())
        if config is None or config == "all": active = all_names
        elif isinstance(config, list): active = set(config) & all_names
        elif isinstance(config, dict) and "exclude" in config: active = all_names - set(config["exclude"])
        else: raise ValueError("Invalid tools_config")

        self._all_tools = {k: v for k, v in self._all_tools.items() if k in active}
        self.tools = [v['schema'] for v in self._all_tools.values()]

    def _prepare_messages_for_api(self, step_prefill: str) -> List[Dict]:
        messages_to_send = [self.history[0].copy()]
        for idx, msg in enumerate(self.history.get_all()[Config.AFTER_SYSTEM_PROMPT:]):
            copy = msg.copy()
            if self.edit_mode:
                copy["content"] = f"[id {idx + Config.AFTER_SYSTEM_PROMPT}]\n{copy['content']}"
            messages_to_send.append(copy)
        return messages_to_send

    def _execute_tools(self, tool_calls) -> List[Dict]:
        results = []
        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                tool_info = self._all_tools.get(name)

                if not tool_info:
                    results.append({"tool_call_id": tc.id, "role": "tool", "name": name, "content": f"Error: Unknown tool '{name}'"})
                    continue

                if tool_info['requires_confirmation'] and not self.on_confirm(name, args):
                    results.append({"tool_call_id": tc.id, "role": "tool", "name": name, "content": "Execution cancelled by user."})
                    self.edit_mode = False
                    continue

                handler = tool_info['handler']
                full_result = handler(self, **args) if tool_info['is_instance_method'] else handler(**args)

                if full_result is not None:
                    results.append({"tool_call_id": tc.id, "role": "tool", "name": name, "content": str(full_result)})
            except Exception as e:
                self.on_system_msg(f"[ERROR] Tool '{name}' FAILED: {e}")
                results.append({"tool_call_id": tc.id, "role": "tool", "name": name, "content": f"Error: {e}"})
        return results

    # ---------- Инструменты ----------
    @tool(description="Enables visibility of message IDs.")
    def get_msg_ids(self):
        self.edit_mode = True
        return 'Success'

    @tool(description="Edits a specific message in the history.",
          requires_confirmation=True,
          id=("int", "ID of the message to edit."),
          old=("str", "Optional exact substr to replace. Empty str replaces whole text."),
          new=("str", "Text to insert in place of old."))
    def edit_message(self, id: int, new: str, old: str = ''):
        res = self.history.edit_message(id, new, old)
        return res

    @tool(description="Deletes a range of messages from dialog history.",
          requires_confirmation=True,
          start_id=("int", "Starting message ID to delete."),
          end_id=("int", "Optional ending message ID (-1 for last)."))
    def delete_messages(self, start_id: int, end_id: int = -1):
        err = self.history.delete_range(start_id, end_id)
        return err

    def chat(self, message: str, max_iter: int = 5, prefill: str = None) -> str:
        user_msg = {"role": "user", "content": message}
        self.history.add(user_msg)

        current_prefill = prefill
        if not self.thinking_enabled and not current_prefill:
            current_prefill = "</think>\n\n"

        for i in range(max_iter):
            step_prefill = current_prefill if i == 0 else None
            messages_to_send = self._prepare_messages_for_api(step_prefill)

            message_obj, err = LLMClient.call(
                messages_to_send, self.temp, self.timeout,
                tools=self.tools if self.tools else None, prefill=step_prefill
            )

            if err: return f"API Error: {err}"
            if not message_obj: return "Empty response"

            self.edit_mode = False

            content = message_obj.content or ""
            clean_content = ((step_prefill + content) if step_prefill else content).replace("</think>", "").strip()

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

        return "Max iterations reached without final answer."


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
            print(f" ✅ [Result '{msg.get('name')}']: {display}")

    @staticmethod
    def confirm_action(name: str, args: Dict) -> bool:
        print(f"\n[WARNING] Tool '{name}' modifies state.")
        print(f"Arguments: {json.dumps(args, ensure_ascii=False)}")
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
            "/load": self.cmd_load
        }

    def cmd_regen(self, parts: List[str]):
        n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
        for _ in range(n - 1):
            self.agent.history.pop_until_user()
        user_msg = self.agent.history.pop_until_user()

        if not user_msg:
            ConsoleUI.system_msg("Cannot find a preceding user message to regenerate.")
            return

        ConsoleUI.system_msg(f"Regenerating response for: '{user_msg}'")
        self.agent.chat(user_msg, max_iter=5, prefill=self.pending_prefill)

    def cmd_think_on(self, parts: List[str]):
        self.agent.thinking_enabled = True
        ConsoleUI.system_msg("Force think ENABLED.")

    def cmd_think_off(self, parts: List[str]):
        self.agent.thinking_enabled = False
        ConsoleUI.system_msg("Force think DISABLED (using dirty hack).")

    def cmd_prefill(self, parts: List[str]):
        if len(parts) > 1:
            self.pending_prefill = parts[1]
            ConsoleUI.system_msg(f"Next message will start with prefill: '{self.pending_prefill}'")
        else:
            self.pending_prefill = None
            ConsoleUI.system_msg("Prefill cleared.")

    def cmd_save(self, parts: List[str]):
        filename = parts[1] if len(parts) > 1 else "default_history.json"
        try:
            self.agent.history.save(filename)
            ConsoleUI.system_msg(f"History saved to '{filename}'.")
        except Exception as e:
            ConsoleUI.system_msg(f"Error saving history: {e}")

    def cmd_load(self, parts: List[str]):
        filename = parts[1] if len(parts) > 1 else "default_history.json"
        if not os.path.exists(filename):
            ConsoleUI.system_msg(f"File '{filename}' not found.")
            return
        try:
            self.agent.history.load(filename)
            ConsoleUI.system_msg(f"History loaded. Total messages: {len(self.agent.history)}")
            print("\n" + "="*40 + "\n🔄 LOADED HISTORY:\n" + "="*40)
            for msg in self.agent.history.get_all():
                ConsoleUI.render_message(msg)
        except Exception as e:
            ConsoleUI.system_msg(f"Error loading history: {e}")

    def run(self):
        ConsoleUI.system_msg("Ready. Type 'exit' to quit.")
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
    sys_prompt = (
        "You are a special tool-calling assistant. Use tools to fulfill user requests. "
        "Ask user's confirmation before calling tools. Speak Russian."
    )

    agent = LLMAgent(
        system_prompt=sys_prompt,
        tools_config="all",
        on_render=ConsoleUI.render_message,
        on_confirm=ConsoleUI.confirm_action,
        on_system_msg=ConsoleUI.system_msg
    )

    cli = CLI(agent)
    cli.run()