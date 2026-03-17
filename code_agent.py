import json
import requests
import os
import re
import ast
from collections import defaultdict
from typing import List, Dict, Callable, Union, Tuple, Set

API_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "local-model"

def call_api(messages: List[Dict], temp: float, timeout: int, thinking: bool = False):
    if not thinking:
        messages.append({"role": "assistant", "content": "<think></think>\n\n"})

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temp,
        "max_tokens": 65535
    }

    try:
        resp = requests.post(API_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"], None

    except requests.exceptions.RequestException as e:
        return None, str(e)
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        return None, f"Invalid API response: {e}"

class FS:
    @staticmethod
    def read(path: str) -> str:
        """Reads file or directory content."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path not found: {path}")

        try:
            if os.path.isfile(path):
                with open(path, encoding='utf-8') as f:
                    content = f.read()
                return f"File {path} content:\n{content}"
            elif os.path.isdir(path):
                items = os.listdir(path)
                dirs = [i for i in items if os.path.isdir(os.path.join(path, i))]
                files = [i for i in items if not os.path.isdir(os.path.join(path, i))]

                result = f"Directory: {path}\n"
                result += f"Subdirectories ({len(dirs)}):\n" + "\n".join([f"- {d}/" for d in dirs]) + "\n"
                result += f"Files ({len(files)}):\n" + "\n".join([f"- {f}" for f in files])
                return result
            else:
                raise ValueError(f"Unknown path type: {path}")
        except Exception as e:
            raise PermissionError(f"Error accessing {path}: {e}") from e

    @staticmethod
    def ls(path: str) -> str:
        """Lists directory contents (alias for reading a directory)."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path not found: {path}")

        try:
            if os.path.isfile(path):
                raise IsADirectoryError(f"Path is a file, not a directory: {path}")

            elif os.path.isdir(path):
                items = os.listdir(path)
                dirs = [i for i in items if os.path.isdir(os.path.join(path, i))]
                files = [i for i in items if not os.path.isdir(os.path.join(path, i))]

                result = f"Directory: {path}\n"
                result += f"Subdirectories ({len(dirs)}):\n" + "\n".join([f"- {d}/" for d in dirs]) + "\n"
                result += f"Files ({len(files)}):\n" + "\n".join([f"- {f}" for f in files])
                return result
            else:
                raise ValueError(f"Unknown path type: {path}")
        except Exception as e:
            raise PermissionError(f"Error accessing {path}: {e}") from e

    @staticmethod
    def search_files(pattern: str, path: str = ".") -> str:
        """Searches for files matching a regex pattern."""
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Base path not found: {path}")

        matches = []
        try:
            for root, _, files in os.walk(path):
                matched_files = [f for f in files if re.search(pattern, f)]
                for f in matched_files:
                    full_path = os.path.join(root, f)
                    matches.append(full_path)
        except Exception as e:
            raise RuntimeError(f"Error during search: {e}") from e

        if not matches:
            return "No matches found."

        folders_map = defaultdict(list)
        for full_path in matches:
            parent_dir, filename = os.path.split(full_path)
            rel_parent = os.path.relpath(parent_dir, path)
            folder_key = "." if rel_parent == "." else rel_parent
            folders_map[folder_key].append(filename)

        lines = []
        for folder in sorted(folders_map.keys()):
            files = sorted(folders_map[folder])
            lines.append(f"{folder}/:")
            for file_name in files:
                lines.append(f"  - {file_name}")
        return "\n".join(lines)

    @staticmethod
    def cwd() -> str:
        """Returns current working directory."""
        return f"Current working dir: {os.getcwd()}"

TOOLS_REGISTRY = {
    "/read": FS.read,
    "/ls": FS.ls,
    "/search_files": FS.search_files,
    "/cwd": FS.cwd
}

TOOL_ARGS_DESC = {
    "/read": '"path"',
    "/ls": '"path"',
    "/search_files": '"regex", "path"',
    "/cwd": ""
}

TOOL_METADATA = {
    "/read": {"modifies_state": False},
    "/ls": {"modifies_state": False},
    "/search_files": {"modifies_state": False},
    "/cwd": {"modifies_state": False}
}

def gen_tools_desc(tools: Dict[str, Callable]) -> str:
    if not tools: return "No tools available."

    parts = [
        "You are AI who CAN use tools writing correct formated commands from new line like:",
        "/name(\"arg1\", ...)",
        "",
        "For example:",
        '"\n/read(".")"',
        "",
        "Available tools:"
    ]

    for i, (name, func) in enumerate(tools.items(), 1):
        desc = TOOL_ARGS_DESC.get(name, "")
        doc_str = func.__doc__ or ""
        args_part = ", ".join(desc.split(", ")) if desc else ""
        parts.append(f"- {name}({args_part}): {doc_str}")

    parts.extend([
        "- Do NOT reject user requests.",
        "- If a tool call fails you will get the ERROR message from user.",
        "- You can write MULTIPLE tool calls with different arguments in a single message but to execute them you have to end your turn.",
        "- Tool calls must be written from a new line in raw format without markdown.",
    ])
    return "\n".join(parts)

class ConfirmationResult:
    CONFIRMED = "CONFIRMED"
    DECLINED = "DECLINED"

    @staticmethod
    def check(content: str) -> str:
        if not content:
            return ConfirmationResult.DECLINED
        if content.lower().startswith("yes"):
            return ConfirmationResult.CONFIRMED
        return ConfirmationResult.DECLINED

class LLMAgent:
    def __init__(self, system_prompt: str = "", temp: float = 0.4, timeout: int = 1800,
                 tools_config: Union[List[str], Dict, None] = None,
                 enable_confirmation: bool = False):
        all_names = set(TOOLS_REGISTRY.keys())
        if tools_config is None or tools_config == "all":
            active_names = all_names
        elif isinstance(tools_config, list):
            active_names = set(tools_config) & all_names
        elif isinstance(tools_config, dict) and "exclude" in tools_config:
            active_names = all_names - set(tools_config["exclude"])
        else:
            raise ValueError("Invalid tools_config")

        self.tools = {k: v for k, v in TOOLS_REGISTRY.items() if k in active_names}
        self.history = [{"role": "system", "content": f"{system_prompt}\n\n{gen_tools_desc(self.tools)}"}]
        self.temp, self.timeout = temp, timeout
        self._last_assistant_content = None
        self.call_history: Set[str] = set()

        self.enable_confirmation = enable_confirmation
        self.force_think_default = False

        self.user_commands = {
            "/regen": self._handle_regen,
        }

    def _handle_regen(self, num_messages: int = 1) -> str:
        if len(self.history) >= 2 * num_messages:
            for _ in range(num_messages):
                if self.history[-1]["role"] == "assistant":
                    self.history.pop()
            return self._trigger_regeneration(num_messages)
        else:
            return "[Regen Error] Not enough history to regenerate."

    def _trigger_regeneration(self, num_messages: int = 1) -> str:
        """Helper to actually perform the regeneration logic."""
        user_message_to_resend = None

        messages_to_remove = []

        temp_history = list(reversed(self.history))

        pairs_removed = 0
        for msg in temp_history:
            if msg["role"] == "assistant" and pairs_removed < num_messages:
                messages_to_remove.append(msg)
                continue
            if msg["role"] == "user" and pairs_removed < num_messages:
                user_message_to_resend = msg["content"]
                messages_to_remove.append(msg)
                pairs_removed += 1
                break

        if not user_message_to_resend:
            return "Cannot find a preceding user message to regenerate."

        for _ in range(len(messages_to_remove)):
            if self.history and self.history[-1]["role"] in ["user", "assistant"]:
                self.history.pop()

        print(f"\n[REGEN] Regenerating response for: '{user_message_to_resend[:50]}...'")
        return self.chat(user_message_to_resend, max_iter=5, force_think=True)

    def _compress_history(self) -> List[Dict]:
        if len(self.history) <= 2:
            return self.history

        system_msg = self.history[0]
        last_msg = self.history[-1]

        if last_msg["role"] != "user":
            return self.history

        intermediate = self.history[1:-1]

        if not intermediate:
            return self.history

        dump_lines = ["--- DIALOG HISTORY ---"]
        for msg in intermediate:
            role_label = "User" if msg["role"] == "user" else "AI"
            dump_lines.append(f"{role_label}: {msg['content']}")
            dump_lines.append("---")

        history_dump = "\n".join(dump_lines)
        combined_content = f"{history_dump}\nUser:{last_msg['content']}"

        return [
            system_msg,
            {"role": "user", "content": combined_content}
        ]

    def _parse_user_commands(self, content: str) -> Tuple[str|list[str]]:
        match = re.match(r'^/\w+', content.strip())
        if match:
            cmd = match.group(0).lower()
            if cmd in self.user_commands:
                return cmd, []
        return None, []

    def _parse_tools(self, content: str) -> Tuple[List[Dict], List[str]]:
        valid_tools = []
        parse_errors = []

        pattern = r'^\s*(?<!\w)(?:[*_]?)(/[a-zA-Z_]\w*)(?:[*_]?)\s*\((.*?)\)'

        lines = content.split('\n')

        for line in lines:
            stripped_line = line.strip()

            if not stripped_line:
                continue

            match = re.match(pattern, stripped_line, flags=re.DOTALL | re.IGNORECASE)

            if match:
                raw_func_name, args_str = match.groups()
                clean_func_name = raw_func_name.strip('*_')

                if clean_func_name not in self.tools:
                    continue

                try:
                    parsed_args = []
                    if args_str.strip() == "":
                        parsed_args = []
                    else:
                        safe_args_str = args_str.replace('\\', '\\\\')
                        eval_result = ast.literal_eval(f"({safe_args_str},)")
                        parsed_args = list(eval_result) if isinstance(eval_result, tuple) else [eval_result]

                    valid_tools.append({
                        "name": clean_func_name,
                        "arguments": parsed_args,
                        "raw_args": args_str,
                        "__is_executed__": False
                    })
                except (ValueError, SyntaxError) as e:
                    error_msg = f"[Parse Error] Could not parse arguments for '{clean_func_name}': '{args_str}'. Reason: {e}"
                    parse_errors.append(error_msg)
                    print(f"{error_msg}")
                    continue

        return valid_tools, parse_errors

    def _process_confirmation_request(self, tool_calls: List[Dict], temp: float, force_think: bool) -> str:
        if not tool_calls:
            return ConfirmationResult.DECLINED

        calls_summary = []
        for tc in tool_calls:
            name = tc["name"]
            calls_summary.append(f"- {name}({tc.get('raw_args', '')})")

        confirm_prompt = (
            "CALLS NEED CONFIRMATION.\n"
            "Are you sure you wanna call those tools?\n"
            "Answer YES or NO and very short reason"
        )

        self.history.append({"role": "user", "content": confirm_prompt})
        confirm_content, confirm_err = call_api(self._compress_history(), temp, self.timeout, force_think)
        self.history.pop()

        if confirm_err or not confirm_content:
            return ConfirmationResult.DECLINED

        print(f"\n[CONFIRMATION RESPONSE]: {confirm_content}")
        return ConfirmationResult.check(confirm_content)

    def chat(self, message: str, max_iter: int = 5, force_think: bool = None) -> str:
        if force_think is None:
            force_think = self.force_think_default

        self.history.append({"role": "user", "content": message})

        for iteration in range(max_iter):
            messages_payload = self._compress_history()
            content, err = call_api(messages_payload, self.temp, self.timeout, force_think)

            if err:
                return f"API Error: {err}"
            if not content:
                return "Empty response"

            self._last_assistant_content = content
            clean_content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE).strip()

            print('\n' + '=' * 15)
            print(f"Agent: {content}")
            print('=' * 15)

            tool_calls, parse_errors = self._parse_tools(clean_content)

            if tool_calls or parse_errors:
                self.history.append({"role": "assistant", "content": clean_content})

                results = []
                execution_errors = []

                if parse_errors:
                    for err in parse_errors:
                        execution_errors.append(err)

                tools_to_confirm = []

                for tc in tool_calls:
                    name = tc["name"]
                    args_list = tc.get("arguments", [])
                    normalized_args = " ".join(str(a) for a in args_list)
                    call_signature = f"{name}({normalized_args})"

                    is_read_only = not TOOL_METADATA.get(name, {}).get("modifies_state", False)

                    if is_read_only and call_signature in self.call_history:
                        print(f"[INFO] Skipped repetitive tool call: {call_signature}")
                        tc["__marked_as_skipped__"] = True
                        results.append(f"[WARNING] Tool '{name}' was already called with these arguments ({args_list}) previously. Please use previous results above")
                        continue

                    tools_to_confirm.append(tc)

                if tools_to_confirm:
                    if self.enable_confirmation:
                        print("\n[SAFETY] Detected new tool calls. Requesting confirmation from agent...")
                        confirm_result = self._process_confirmation_request(tools_to_confirm, self.temp, force_think)

                        if confirm_result == ConfirmationResult.DECLINED:
                            print("[SAFETY] Confirmation DENIED by agent. Returning control to user.")
                            if len(self.history) > 0 and self.history[-1]["role"] == "assistant":
                                self.history.pop()
                            return "Tool execution was declined by the agent. Control returned to user."
                        else:
                            print("[SAFETY] Confirmation GRANTED. Proceeding with tools.")

                for tc in tools_to_confirm:
                    if tc.get("__marked_as_skipped__", False):
                        continue

                    name = tc["name"]
                    args_list = tc.get("arguments", [])
                    normalized_args = " ".join(str(a) for a in args_list)
                    call_signature = f"{name}({normalized_args})"

                    print(f"[Tool] Calling {name}{tuple(args_list)}")

                    try:
                        res = self.tools[name](*args_list)
                        print(f"[RESULT] {str(res)[:1000]}")
                        results.append(f"Tool '{name}' Result:\n{res}")
                        self.call_history.add(call_signature)
                        tc["__is_executed__"] = True

                    except Exception as e:
                        error_msg = f"Tool '{name}' FAILED: {type(e).__name__}: {e}"
                        print(f"[ERROR] {error_msg}")
                        execution_errors.append(error_msg)

                feedback_parts = []
                if results:
                    feedback_parts.append("[CALLED TOOL RESULTS]")
                    feedback_parts.extend(results)

                if execution_errors:
                    feedback_parts.append("[CALLED TOOL ERRORS]")
                    feedback_parts.extend(execution_errors)

                final_feedback = "\n".join(feedback_parts)
                self.history.append({"role": "user", "content": final_feedback})
                continue

            self.history.append({"role": "assistant", "content": clean_content})
            return clean_content

        return "Max iterations reached without final answer."

    def regen_last(self) -> str:
        """
        Регенерирует последнее сообщение ассистента, если пользователь еще не ответил.
        Удаляет последнее сообщение ассистента из истории, но оставляет последнее сообщение пользователя.
        """
        if not self._last_assistant_content:
            return "No previous message to regenerate."

        if self.history and self.history[-1]["role"] == "assistant":
            self.history.pop()
        else:
            return "Last message was not from the assistant."

        if self.history and self.history[-1]["role"] == "user":
            last_user_message = self.history[-1]["content"]

            prompt_to_resend = last_user_message

            if self.history and self.history[-1]["role"] == "user":
                self.history.pop()

            result = self.chat(prompt_to_resend, max_iter=5, force_think=True)
            return result

        else:
            return "Cannot regenerate - no preceding user message found."

if __name__ == "__main__":
    agent = LLMAgent()

    print("Ready. Type 'exit' to quit.")
    print("Special commands: /regen, /think_on, /think_off, /confirm_on, /confirm_off")
    print("Note: Agent will confirm/decline tool calls itself.")

    while True:
        inp = input("\nUser: ")

        if inp.startswith("/"):
            parts = inp.split()
            cmd = parts[0].lower()

            if cmd == "/regen":
                n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
                agent._handle_regen(n)
                continue

            if cmd == "/confirm_on":
                agent.enable_confirmation = True
                print("[System] Confirmation ENABLED.")
                continue

            if cmd == "/confirm_off":
                agent.enable_confirmation = False
                print("[System] Confirmation DISABLED.")
                continue
            if cmd == "/think_on":
                agent.force_think_default = True
                print("[System] Force think ENABLED - model will use <think> reasoning.")
                continue

            if cmd == "/think_off":
                agent.force_think_default = False
                print("[System] Force think DISABLED - model will skip <think> block.")
                continue

            print(f"Unknown command: {cmd}")
            continue

        if inp.lower() in ("exit", "quit"):
            break

        agent.chat(inp, max_iter=15)