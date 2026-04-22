import datetime
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Union, Tuple, Set, Any, Optional

import requests
import os
import re
import ast

API_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "local-model"


def call_api(messages: List[Dict], temp: float, timeout: int, thinking: bool = False):
    if not thinking:
        messages.append({"role": "assistant", "content": "</think>\n\n"})


    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temp,
        "max_tokens": 11500
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


MAX_DEPTH = 1


@dataclass
class ToolDefinition:
    func: Callable
    args_desc: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    aliases: List[str] = field(default_factory=list)


def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes}B"

    units = ['B', 'KB', 'MB', 'GB', 'TB']
    unit_index = 0
    current_size = float(size_bytes)

    while current_size >= 1024 and unit_index < len(units) - 1:
        current_size /= 1024
        unit_index += 1

    formatted_size = f"{current_size:.2f}".rstrip('0').rstrip('.')
    return f"{formatted_size}{units[unit_index]}"


def _count_hidden_size(root_path: str) -> int:
    total_size = 0
    try:
        entries = os.scandir(root_path)
        for entry in entries:
            try:
                if entry.is_file():
                    total_size += entry.stat().st_size
                elif entry.is_dir():
                    total_size += _count_hidden_size(entry.path)
            except PermissionError:
                continue
    except (PermissionError, FileNotFoundError):
        pass
    return total_size


def _build_tree(root_path: str, current_depth: int = 0) -> str:
    DENSITY_LIMIT = 4
    result_lines = []

    try:
        entries = list(os.scandir(root_path))
    except PermissionError:
        return f"{'  ' * current_depth}[Permission Denied]"
    except FileNotFoundError:
        return f"{'  ' * current_depth}[Path Not Found]"

    total_count = len(entries)

    if current_depth > 0 and total_count > DENSITY_LIMIT:
        indent = '  ' * current_depth
        hidden_size_bytes = _count_hidden_size(root_path)
        size_str = _format_size(hidden_size_bytes)
        return f"{indent}[{total_count} nested items were TRUNCATED, size={size_str}]"

    dirs = [e for e in entries if e.is_dir()]
    files = [e for e in entries if e.is_file()]

    dirs.sort(key=lambda x: x.name.lower())
    files.sort(key=lambda x: x.name.lower())
    items_to_show = dirs + files

    for entry in items_to_show:
        stat_info = entry.stat()
        mtime = datetime.datetime.fromtimestamp(stat_info.st_mtime).strftime("%Y-%m-%d")
        prefix = f"{'  ' * current_depth}"

        if entry.is_dir():
            line_str = f"{prefix}{entry.name}/ ({mtime})"
            result_lines.append(line_str)
            sub_result = _build_tree(entry.path, current_depth + 1)
            if sub_result:
                result_lines.append(sub_result)
        else:
            size_str = _format_size(stat_info.st_size)
            line_str = f"{prefix}{entry.name} ({size_str})"
            result_lines.append(line_str)

    return "\n".join(result_lines)


class FS:
    @staticmethod
    def open(path: str) -> str:
        """Gets file or directory content."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path not found: {path}")

        try:
            stat_info = os.stat(path)
            mtime = datetime.datetime.fromtimestamp(stat_info.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            size = stat_info.st_size

            if os.path.isfile(path):
                try:
                    with open(path, 'r', encoding='utf-8', errors='strict') as f:
                        content = f.read()
                    is_text_file = True
                except UnicodeDecodeError:
                    is_text_file = False
                    content = None
                except Exception as e:
                    return f"Error reading file {path}: {e}"

                if not is_text_file:
                    _, ext = os.path.splitext(path)
                    return f"Error reading file {path}: Access denied. Cannot read binary files (failed UTF-8 decode)"

                return f"File: {path}\nModified: {mtime}\nContent:\n---\n{content}"

            elif os.path.isdir(path):
                tree_content = _build_tree(path, 0)
                header = f"Directory Tree: {path}\nModified: {mtime}\n"
                return f"{header}\n{tree_content}"
            else:
                raise ValueError(f"Unknown path type: {path}")

        except Exception as e:
            raise PermissionError(f"Error accessing {path}: {e}") from e

    @staticmethod
    def search_files(pattern: str, path: str = ".") -> str:
        """Searches for files by pattern in path"""
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
        """Gets current working directory"""
        return f"Current working dir: {os.getcwd()}"


TOOLS_CONFIG = {
    "read_group": ToolDefinition(
        func=FS.open,
        args_desc='"path"',
        metadata={"modifies_state": False, "type": "read"},
        aliases=["/open"]
    ),
    "/search_files": ToolDefinition(
        func=FS.search_files,
        args_desc='"regex", "path"',
        metadata={"modifies_state": False},
        aliases=[]
    ),
    "/cwd": ToolDefinition(
        func=FS.cwd,
        args_desc="",
        metadata={"modifies_state": False},
        aliases=[]
    )
}


def gen_tools_desc(tools_config: Dict[str, ToolDefinition]) -> str:
    if not tools_config:
        return "No tools available."
    parts = [
        "You are a special AI that uses tools writing correct commands from a new line as:",
        "\n/name(\"arg1\", ...)",
        "",
        "Aka:",
        '"\n/open("./")"',
        "",
        "All available tools:"
    ]

    seen_funcs = set()

    for i, (group_name, tool_def) in enumerate(tools_config.items(), 1):
        func_id = id(tool_def.func)
        if func_id in seen_funcs:
            continue
        seen_funcs.add(func_id)

        desc = tool_def.args_desc
        func = tool_def.func
        doc_str = func.__doc__ or ""
        args_part = ", ".join(desc.split(", ")) if desc else ""

        names_str = ", ".join(tool_def.aliases) if tool_def.aliases else group_name

        parts.append(f"- {names_str}({args_part}): {doc_str}")

    return "\n".join(parts)
    parts.extend([
        "NEVER show usage examples of your tools."
        "AI can write up to 4 tool calls in 1 message.",
        "Each tool must be on new line in raw text format.",
        "To show content of truncated items call them individually."
        "Use '/' in paths."
    ])


class ConfirmationResult:
    CONFIRMED = "CONFIRMED"
    DECLINED = "DECLINED"
    KEEP = "KEEP"
    DISCARD = "DISCARD"

    @staticmethod
    def check(content: str) -> str:
        if not content:
            return ConfirmationResult.DECLINED
        if content.lower().startswith("yes"):
            return ConfirmationResult.CONFIRMED
        return ConfirmationResult.DECLINED

    @staticmethod
    def check_retention(content: str) -> str:
        if not content:
            return ConfirmationResult.DISCARD
        if content.strip().upper().startswith("SAVE"):
            return ConfirmationResult.KEEP
        return ConfirmationResult.DISCARD


class LLMAgent:
    def __init__(self, system_prompt: str = "", temp: float = 0.65, timeout: int = 1800,
                 tools_config: Union[List[str], Dict, None] = None,
                 enable_confirmation: bool = False):

        all_groups = set(TOOLS_CONFIG.keys())

        if tools_config is None or tools_config == "all":
            active_groups = all_groups
        elif isinstance(tools_config, list):
            active_groups = set(tools_config) & all_groups
        elif isinstance(tools_config, dict) and "exclude" in tools_config:
            active_groups = all_groups - set(tools_config["exclude"])
        else:
            raise ValueError("Invalid tools_config")

        self.tools_registry = {}
        self.tools_metadata = {}
        self.alias_map = {}

        for group_name in active_groups:
            tool_def = TOOLS_CONFIG[group_name]
            func = tool_def.func

            names_to_register = tool_def.aliases if tool_def.aliases else [group_name]

            for name in names_to_register:
                self.tools_registry[name] = func
                self.tools_metadata[name] = tool_def.metadata
                self.alias_map[name] = func

        active_config_subset = {k: v for k, v in TOOLS_CONFIG.items() if k in active_groups}

        self.history = [{"role": "system", "content": f"{system_prompt}\n\n{gen_tools_desc(active_config_subset)}"}]
        self.temp, self.timeout = temp, timeout
        self._last_assistant_content = None

        self.confirmed_reads: Set[str] = set()

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

        print(f"\n[REGEN] Regenerating response for: '{user_message_to_resend}'")
        return self.chat(user_message_to_resend, max_iter=5)

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
        dump_lines = ["--- FULL DIALOG HISTORY ---"]
        for msg in intermediate:
            role_label = "User" if msg["role"] == "user" else "AI"
            dump_lines.append(f"{role_label}: \"{msg['content']}\"\n")
            dump_lines.append("---")
        history_dump = "\n".join(dump_lines)
        combined_content = f"{history_dump}\nUser:{last_msg['content']}"
        return [system_msg, {"role": "user", "content": combined_content}]

    def _parse_user_commands(self, content: str) -> Tuple[Optional[str], List[str]]:
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

                if clean_func_name not in self.tools_registry:
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

    def _query_agent_decision(self, prompt_question: str, context_data: Optional[str] = None, force_think: bool = False) -> str:
        base_messages = self._compress_history()
        temp_messages = [msg.copy() for msg in base_messages]

        user_content = ""
        if context_data:
            user_content += context_data
        user_content += prompt_question

        temp_messages.append({"role": "user", "content": user_content})

        response_content, err = call_api(temp_messages, self.temp, self.timeout, force_think)

        if err or not response_content:
            print(f"[DECISION API ERROR]: {err}")
            return ConfirmationResult.DISCARD

        print(f"[AGENT DECISION RESPONSE]: {response_content}")
        return response_content

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
            "Say only YES or NO"
        )
        confirm_content = self._query_agent_decision(confirm_prompt, context_data="\n".join(calls_summary), force_think=force_think)

        return ConfirmationResult.check(confirm_content)

    def _get_canonical_name_and_signature(self, name: str, args_list: List[Any]) -> Tuple[str, str]:
        target_func = self.tools_registry.get(name)
        canonical_name = name
        if target_func:
            for alias, func in self.alias_map.items():
                if func == target_func:
                    canonical_name = alias
                    break

        normalized_args = " ".join(str(a) for a in args_list)
        call_signature = f"{canonical_name}({normalized_args})"
        return canonical_name, call_signature

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
                skipped_results = []

                seen_in_current_batch = set()

                for tc in tool_calls:
                    name = tc["name"]
                    args_list = tc.get("arguments", [])

                    canonical_name, call_signature = self._get_canonical_name_and_signature(name, args_list)

                    if call_signature in self.confirmed_reads:
                        print(f"[INFO] Skipped globally confirmed read: {call_signature}")
                        tc["__marked_as_skipped__"] = True
                        skipped_results.append(
                            f"[WARNING] Tool '{name}' (alias for {canonical_name}) was already processed and content kept in memory. No need to reopen.")
                        continue

                    if call_signature in seen_in_current_batch:
                        print(f"[INFO] Skipped repetitive tool call within same message: {call_signature}")
                        tc["__marked_as_skipped__"] = True
                        skipped_results.append(
                            f"[WARNING] Tool '{name}' (alias for {canonical_name}) was already called with these arguments ({args_list}) in this message. Skipping duplicate execution.")
                        continue

                    seen_in_current_batch.add(call_signature)
                    tools_to_confirm.append(tc)

                if skipped_results:
                    results.extend(skipped_results)

                if tools_to_confirm:
                    if self.enable_confirmation:
                        print("\n[SAFETY] Detected new tool calls. Requesting confirmation from agent")
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

                    canonical_name, call_signature = self._get_canonical_name_and_signature(name, args_list)

                    print(f"[Tool] Calling {name}{tuple(args_list)}")

                    try:
                        full_result = self.tools_registry[name](*args_list)

                        tool_meta = self.tools_metadata.get(name, {})
                        is_file_read = tool_meta.get("type") == "read"

                        should_add_to_global_history = False
                        important = '[IMPORTANT]'
                        question = 'Is this content useful to remember?\nSay only SAVE or FORGET. Nothing else. No comms. Plain text!'

                        if is_file_read and isinstance(full_result, str) and full_result.startswith("File:"):
                            lines = full_result.split('\n', 3)
                            if len(lines) >= 4 and lines[0].startswith("File:"):
                                file_path_line = lines[0]
                                file_path = file_path_line.replace("File: ", "").strip()
                                content_start_marker = "\nContent:\n"
                                marker_pos = full_result.find(content_start_marker)

                                if marker_pos != -1:
                                    full_file_content = '```\n'+full_result[marker_pos + len(content_start_marker):]
                                    decision_prompt = (
                                        f"\n```\n{important} This is '{file_path}' content sent to you by tool.\n{question}"
                                    )

                                    decision_response = self._query_agent_decision(
                                        decision_prompt,
                                        context_data=full_file_content,
                                        force_think=force_think
                                    )

                                    decision = ConfirmationResult.check_retention(decision_response)

                                    if decision == ConfirmationResult.KEEP:
                                        print(f"[RETENTION] Agent decided to KEEP full content of {file_path}. Adding to global blocklist.")
                                        final_result_text = full_result
                                        should_add_to_global_history = True
                                    else:
                                        print(f"[RETENTION] Agent decided to DISCARD full content of {file_path}. reopen allowed later.")

                                        stat_info = os.stat(file_path)
                                        mtime = datetime.datetime.fromtimestamp(stat_info.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                                        size = stat_info.st_size
                                        final_result_text = (
                                            f"File: {file_path}\n"
                                            f"Modified: {mtime}\n"
                                            f"[You (AI) cancelled file loading by saying 'FORGET']\n"
                                            f"You can reopen file again if you mistook."
                                        )
                                else:
                                    final_result_text = full_result
                            else:
                                final_result_text = full_result
                        else:
                            final_result_text = full_result

                        print(f"[RESULT] {str(final_result_text)[:2000]}")
                        results.append(f"Tool '{name}' Result:\n{final_result_text}")

                        if should_add_to_global_history:
                            self.confirmed_reads.add(call_signature)

                        tc["__is_executed__"] = True

                    except Exception as e:
                        error_msg = f"Tool '{name}' FAILED: {type(e).__name__}: {e}"
                        print(f"[ERROR] {error_msg}")
                        execution_errors.append(error_msg)

                feedback_parts = []
                if results:
                    feedback_parts.append("[TOOLS RESULTS]")
                    feedback_parts.extend(results)

                if execution_errors:
                    feedback_parts.append("[TOOLS ERRORS]")
                    feedback_parts.extend(execution_errors)

                final_feedback = "\n".join(feedback_parts)
                self.history.append({"role": "user", "content": final_feedback})
                continue

            self.history.append({"role": "assistant", "content": clean_content})
            return clean_content

        return "Max iterations reached without final answer."

    def regen_last(self) -> str:
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
            result = self.chat(prompt_to_resend, max_iter=5)
            return result
        else:
            return "Cannot regenerate - no preceding user message found."


if __name__ == "__main__":
    agent = LLMAgent(tools_config={"exclude": ["/search_files"]})
    print("Ready. Type 'exit' to quit.")
    print("Special commands: /regen, /think_on, /think_off, /confirm_on, /confirm_off")
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
                print("[System] Force think ENABLED.")
                continue
            if cmd == "/think_off":
                agent.force_think_default = False
                print("[System] Force think DISABLED.")
                continue
            print(f"Unknown command: {cmd}")
            continue
        if inp.lower() in ("exit", "quit"):
            break
        agent.chat(inp, max_iter=25)