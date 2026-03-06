import json
import urllib.request
import os
import re
import ast
from typing import List, Dict, Callable, Union, Tuple, Set

# --- КОНФИГУРАЦИЯ ---
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

    req = urllib.request.Request(
        API_URL, data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'}, method='POST'
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
            return data["choices"][0]["message"]["content"], None
    except Exception as e:
        return None, str(e)

class FS:
    @staticmethod
    def read(path: str) -> str:
        """Returns content of either file or directory"""
        try:
            if not os.path.exists(path):
                return f"Path not found: {path}"

            if os.path.isfile(path):
                return f"File {path} content: {open(path, encoding='utf-8').read()}"
            elif os.path.isdir(path):
                items = os.listdir(path)
                dirs = [i for i in items if os.path.isdir(os.path.join(path, i))]
                files = [i for i in items if not os.path.isdir(os.path.join(path, i))]

                result = f"Directory: {path}\n"
                result += f"Subdirectories ({len(dirs)}):\n" + "\n".join([f"- {d}/" for d in dirs]) + "\n"
                result += f"Files ({len(files)}):\n" + "\n".join([f"- {f}" for f in files])
                return result
            else:
                return f"Unknown path type: {path}"
        except Exception as e:
            return f"Error accessing {path}: {e}"

    @staticmethod
    def search_files(pattern: str, path: str = ".") -> str:
        """Searches for files recursively"""
        try:
            if not os.path.isdir(path): return f"Base path not found: {path}"
            matches = []
            for root, _, files in os.walk(path):
                matches.extend([os.path.join(root, f) for f in files if re.search(pattern, f)])
            return "\n".join(matches) or "No matches found."
        except Exception as e: return f"Error: {e}"

    @staticmethod
    def cwd() -> str:
        """Returns current working directory path like '.'"""
        return f"Current working dir: {os.getcwd()}"

TOOLS_REGISTRY = {
    "/read": FS.read,
    "/search_files": FS.search_files,
    "/cwd": FS.cwd
}

TOOL_ARGS_DESC = {
    "/read": "path",
    "/search_files": "regex, path",
    "/cwd": ""
}

TOOL_METADATA = {
    "/read": {"modifies_state": False},
    "/search_files": {"modifies_state": False},
    "/cwd": {"modifies_state": False}
}

def gen_tools_desc(tools: Dict[str, Callable]) -> str:
    if not tools: return "No tools available."

    parts = [
        "You are tool-using AI that must use tools by writing like:",
        "/name(args)",
        "",
        "Example:",
        "/read(\".\")",
        "",
        "Your available tools:"
    ]

    for i, (name, func) in enumerate(tools.items(), 1):
        desc = TOOL_ARGS_DESC.get(name, "")
        doc_str = func.__doc__ or ""
        args_part = ", ".join(desc.split(", ")) if desc else ""
        parts.append(f"- {name}({args_part}): {doc_str}")

    parts.extend([
        "",
        "- You can interact with file system using tools.",
        "- You can write MULTIPLE tool calls with different arguments.",
    ])
    return "\n".join(parts)

class LLMAgent:
    def __init__(self, system_prompt: str = "", temp: float = 0.15, timeout: int = 1800,
                 tools_config: Union[List[str], Dict, None] = None):

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

        self.user_commands = {
            "/regen": self._handle_regen,
        }

    def _handle_regen(self, num_messages: int = 1) -> str:
        if len(self.history) >= 2 * num_messages:
            for _ in range(num_messages):
                if self.history[-1]["role"] == "user":
                    self.history.pop()
                if self.history[-1]["role"] == "assistant":
                    self.history.pop()

            self._last_assistant_content = None
            return f"Removed last {num_messages} assistant messages. Ready to regenerate.\nPlease send your request again."
        else:
            return "[Regen Error] Not enough history to regenerate."

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

        # Паттерн ищет функцию и всё внутри скобок (включая пустоту)
        pattern = r"(/\w+)\s*\((.*?)\)"
        matches = re.findall(pattern, content, re.DOTALL)

        for func_name, args_str in matches:
            if func_name not in self.tools:
                continue

            try:
                parsed_args = []

                if args_str.strip() == "":
                    parsed_args = []
                else:
                    eval_result = ast.literal_eval(f"({args_str},)")
                    parsed_args = list(eval_result) if isinstance(eval_result, tuple) else [eval_result]

                valid_tools.append({
                    "name": func_name,
                    "arguments": parsed_args,
                    "raw_args": args_str
                })
            except (ValueError, SyntaxError) as e:
                error_msg = f"[Parse Error] Could not parse arguments for '{func_name}': '{args_str}'. Reason: {e}"
                parse_errors.append(error_msg)
                print(f"{error_msg}")
                continue

        return valid_tools, parse_errors

    def chat(self, message: str, max_iter: int = 5, force_think: bool = False) -> str:
        self.history.append({"role": "user", "content": message})

        for iteration in range(max_iter):
            content, err = call_api(self.history, self.temp, self.timeout, force_think)
            if err: return f"API Error: {err}"
            if not content: return "Empty response"

            self._last_assistant_content = content

            clean_content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE).strip()

            print(f"Agent: {content}")

            tool_calls, parse_errors = self._parse_tools(clean_content)

            if tool_calls or parse_errors:
                self.history.append({"role": "assistant", "content": clean_content})

                results = []
                execution_errors = []
                has_skipped_tool = False

                if parse_errors:
                    for err in parse_errors:
                        execution_errors.append(err)

                for tc in tool_calls:
                    name = tc["name"]
                    args_list = tc.get("arguments", [])

                    normalized_args = " ".join(str(a) for a in args_list)
                    call_signature = f"{name}({normalized_args})"

                    is_read_only = not TOOL_METADATA.get(name, {}).get("modifies_state", False)

                    if is_read_only and call_signature in self.call_history:
                        msg = "[WARNING] Please don't repeat tool calls that are not state modifying with the same arguments"
                        results.append(f"Tool '{name}':\n{msg}")
                        has_skipped_tool = True
                        continue

                    print(f"[Tool] Calling {name}{tuple(args_list)}")

                    try:
                        res = self.tools[name](*args_list)
                        print(f"[RESULT] {str(res)[:500]}")
                        results.append(f"Tool '{name}':\n{res}")
                        self.call_history.add(call_signature)
                    except Exception as e:
                        error_msg = f"Runtime Error in {name}: {e}"
                        print(f"[ERROR] {error_msg}")
                        execution_errors.append(error_msg)

                feedback_parts = []
                if results:
                    feedback_parts.append("--- SUCCESSFUL TOOL OUTPUTS ---")
                    feedback_parts.extend(results)
                if execution_errors:
                    feedback_parts.append("--- ERRORS AND PARSE MESSAGES ---")
                    feedback_parts.extend(execution_errors)

                final_feedback = "\n".join(feedback_parts)
                self.history.append({"role": "user", "content": final_feedback})

                if has_skipped_tool:
                    print("\n[INFO] Skipped repetitive tool call")

                continue

            self.history.append({"role": "assistant", "content": clean_content})
            return clean_content

        return "Max iterations reached without final answer."

    def regen_last(self) -> str:
        if not self._last_assistant_content:
            return "No previous message to regenerate."

        while self.history and self.history[-1]["role"] == "assistant":
            self.history.pop()
        if self.history and self.history[-1]["role"] == "user":
            self.history.pop()

        if self.history:
            last_content = self.history[-1]["content"] if self.history[-1]["role"] == "user" else None
            if last_content:
                self.chat(last_content, force_think=True)
                return "Regenerated last message."
        return "Cannot regenerate - no history."

if __name__ == "__main__":
    agent = LLMAgent()
    print("Ready. Type 'exit' to quit.")
    print("Special commands: /regen")

    while True:
        inp = input("\nUser: ")

        if inp.startswith("/"):
            parts = inp.split()
            cmd = parts[0].lower()

            if cmd == "/regen":
                n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
                print(agent._handle_regen(n))
                continue

            print(f"Unknown command: {cmd}")
            continue

        if inp.lower() in ("exit", "quit"):
            break

        agent.chat(inp, max_iter=10, force_think=True)