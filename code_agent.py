import json
import urllib.request
import os
import re
import ast
from collections import defaultdict
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
    def search_files(pattern: str, path: str = ".") -> str:
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Base path not found: {path}")

        matches = []
        try:
            for root, _, files in os.walk(path):
                # Фильтруем файлы по имени
                matched_files = [f for f in files if re.search(pattern, f)]
                for f in matched_files:
                    full_path = os.path.join(root, f)
                    matches.append(full_path)
        except Exception as e:
            raise RuntimeError(f"Error during search: {e}") from e

        if not matches:
            return "No matches found."

        # Группируем файлы по родительским папкам
        folders_map = defaultdict(list)

        for full_path in matches:
            parent_dir, filename = os.path.split(full_path)

            # Вычисляем относительный путь от базы поиска до папки файла
            rel_parent = os.path.relpath(parent_dir, path)

            # Если файл лежит прямо в корне поиска, используем "."
            if rel_parent == ".":
                folder_key = "."
            else:
                folder_key = rel_parent

            folders_map[folder_key].append(filename)

        # Формируем вывод
        lines = []
        # Сортируем папки для предсказуемого порядка вывода
        for folder in sorted(folders_map.keys()):
            files = sorted(folders_map[folder])

            # Добавляем заголовок папки
            lines.append(f"{folder}/:")

            # Добавляем файлы внутри папки с отступом
            for file_name in files:
                lines.append(f"  - {file_name}")

        return "\n".join(lines)

    @staticmethod
    def cwd() -> str:
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
        "- User can not use these tools but you can.",
        "- If a tool fails due to syntax or arguments - ERROR message will be returned to you",
        "- You can write MULTIPLE tool calls with different arguments.",
    ])
    return "\n".join(parts)

class ConfirmationResult:
    """Результат проверки подтверждения"""
    CONFIRMED = "CONFIRMED"
    DECLINED = "DECLINED"

    @staticmethod
    def check(content: str) -> str:
        """Проверяет ответ агента на подтверждение/отклонение"""
        if not content:
            return ConfirmationResult.DECLINED

        # "yes" - подтверждено
        if content.lower().startswith("yes"):
            return ConfirmationResult.CONFIRMED

        # Любое другое - отклонено
        return ConfirmationResult.DECLINED


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

        dump_lines = ["---DIALOG HISTORY---"]
        for msg in intermediate:
            role_label = "User" if msg["role"] == "user" else "Assistant(you)"
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
                    safe_args_str = args_str.replace('\\', '\\\\')
                    eval_result = ast.literal_eval(f"({safe_args_str},)")
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

    def _process_confirmation_request(self, tool_calls: List[Dict], temp: float, force_think: bool) -> str:
        """Запрашивает подтверждение у агента и получает результат"""
        if not tool_calls:
            return ConfirmationResult.DECLINED

        # Формируем запрос подтверждения БЕЗ добавления в историю
        calls_summary = []
        for tc in tool_calls:
            name = tc["name"]
            calls_summary.append(f"- {name}({tc.get('raw_args', '')})")

        summary_text = "\n".join(calls_summary)
        confirm_prompt = (
            "SAFETY CHECK: \n"
            "You (AI) just wrote these tool calls in your previous message:\n"
            f"{summary_text}\n\n"
            "Now you decide if you really wanna call them now. Start answering with YES or NO and short reason"
        )

        # Добавляем временное сообщение для получения ответа
        self.history.append({"role": "user", "content": confirm_prompt})

        # Вызываем API
        confirm_content, confirm_err = call_api(self._compress_history(), temp, self.timeout, force_think)

        # Сразу убираем из истории, если что-то пошло не так
        self.history.pop()

        if confirm_err or not confirm_content:
            return ConfirmationResult.DECLINED

        print(f"\n[CONFIRMATION RESPONSE]: {confirm_content}")

        return ConfirmationResult.check(confirm_content)

    def chat(self, message: str, max_iter: int = 5, force_think: bool = False) -> str:
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

            print('\n'+'='*15)
            print(f"Agent: {content}")
            print('='*15)

            tool_calls, parse_errors = self._parse_tools(clean_content)

            if tool_calls or parse_errors:
                # Сохраняем сообщение ассистента с вызовами инструментов
                self.history.append({"role": "assistant", "content": clean_content})

                results = []
                execution_errors = []
                has_skipped_tool = False

                if parse_errors:
                    for err in parse_errors:
                        execution_errors.append(err)

                # --- НОВАЯ ЛОГИКА ПОДТВЕРЖДЕНИЯ ---
                if tool_calls:
                    print("\n[SAFETY] Detected tool calls. Requesting confirmation from agent...")

                    # Получаем результат подтверждения БЕЗ сохранения в историю
                    confirm_result = self._process_confirmation_request(tool_calls, self.temp, force_think)

                    if confirm_result == ConfirmationResult.DECLINED:
                        print("[SAFETY] Confirmation DENIED by agent. Returning control to user.")
                        # УДАЛЯЕМ сообщение ассистента с инструментами из истории
                        # Так как отклонение НЕ должно сохраняться
                        if len(self.history) > 0 and self.history[-1]["role"] == "assistant":
                            self.history.pop()

                        # Возвращаем управление пользователю - выходим из цикла
                        return "Tool execution was declined by the agent. Control returned to user."

                    else:
                        print("[SAFETY] Confirmation GRANTED. Proceeding with tools.")

                # Выполнение инструментов (если подтверждено)
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
                        print(f"[RESULT] {str(res)[:1000]}")
                        results.append(f"Tool '{name}' Result:\n{res}")
                        self.call_history.add(call_signature)

                    except Exception as e:
                        error_msg = f"Tool '{name}' FAILED: {type(e).__name__}: {e}"
                        print(f"[ERROR] {error_msg}")
                        execution_errors.append(error_msg)

                feedback_parts = []

                if results:
                    feedback_parts.append("--- TOOL EXECUTION RESULTS ---")
                    feedback_parts.extend(results)

                if execution_errors:
                    feedback_parts.append("--- TOOL EXECUTION ERRORS ---")
                    feedback_parts.extend(execution_errors)
                    feedback_parts.append("\nIMPORTANT: One or more tools failed. Please try again with corrected arguments")

                final_feedback = "\n".join(feedback_parts)
                self.history.append({"role": "user", "content": final_feedback})

                if has_skipped_tool:
                    print("\n[INFO] Skipped repetitive tool call")

                continue

            # Если инструментов нет, возвращаем финальный ответ
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
    print("Note: Agent will confirm/decline tool calls itself. Declinations won't be stored in history.")

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

        agent.chat(inp, max_iter=5, force_think=True)