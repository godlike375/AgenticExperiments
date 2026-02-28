import json
import urllib.request
import os
import re
from typing import List, Dict, Tuple, Optional, Callable

# =============================================================================
# КОНФИГУРАЦИЯ И УТИЛИТЫ
# =============================================================================

API_URL = "http://192.168.50.196:1234/v1/chat/completions"
MODEL_NAME = "local-model"

def call_api(messages: List[Dict], temperature: float, timeout: int, thinking: bool = False) -> Tuple[Optional[str], Optional[str]]:
    """
    Базовая функция вызова API LM Studio.
    Включает трюк с </think>, если thinking=False.
    """
    if not thinking:
        # Трюк для принудительного закрытия блока мыслей и начала ответа
        messages.append({"role": "assistant", "content": "<think></think>\n\n"})

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 65535
    }

    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST'
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode('utf-8'))
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return content, None
    except Exception as e:
        return None, f"API Error: {str(e)}"

# =============================================================================
# ИНСТРУМЕНТЫ (TOOLS)
# =============================================================================

class FileSystemTools:
    @staticmethod
    def read_file(path: str) -> str:
        try:
            if not os.path.exists(path):
                return f"Error: File not found: {path}"
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"

    @staticmethod
    def list_dir(path: str) -> str:
        try:
            if not os.path.isdir(path):
                return f"Error: Directory not found: {path}"

            items = os.listdir(path)
            dirs: List[str] = []
            files: List[str] = []

            for item in items:
                full_path = os.path.join(path, item)
                if os.path.isdir(full_path):
                    dirs.append(item)
                else:
                    files.append(item)

            return f"Directories: {dirs};\nFiles: {files}"

        except Exception as e:
            return f"Error listing directory: {str(e)}"

    @staticmethod
    def search_files(pattern: str, base_path: str = ".") -> str:
        try:
            if not os.path.isdir(base_path):
                return f"Error: Base path not found: {base_path}"

            matches = []
            for root, dirs, files in os.walk(base_path):
                for filename in files:
                    if re.search(pattern, filename):
                        matches.append(os.path.join(root, filename))

            if not matches:
                return "No files found matching the pattern."

            return "\n".join(matches)
        except Exception as e:
            return f"Error searching files: {str(e)}"

TOOLS_DESCRIPTION = """
You have access to the following tools. To use a tool, append the following block:
<tool_call>
{"name": "tool_name", "arguments": {"arg1": "value1"}}
</tool_call>

Available tools:
1. read_file: Reads content from a file. Args: {"path": "string"}
2. list_dir: Lists files in a directory. Args: {"path": "string"}
3. search_files: Searches files by regex name. Args: {"pattern": "regex_string", "base_path": "string (optional, default '.')"}

IMPORTANT: 
- You can output text explaining your actions BEFORE the tool calls.
- You can make MULTIPLE tool calls in one message.
- Stop generating text immediately after the last tool call block.
"""

AVAILABLE_TOOLS: Dict[str, Callable] = {
    "read_file": FileSystemTools.read_file,
    "list_dir": FileSystemTools.list_dir,
    "search_files": FileSystemTools.search_files,
}

# =============================================================================
# АГЕНТ (CORE LOGIC)
# =============================================================================

class LLMAgent:
    def __init__(self, system_prompt: Optional[str] = None, temperature: float = 0.7, timeout: int = 1800):
        self.history: List[Dict] = []
        self.temperature = temperature
        self.timeout = timeout

        base_system = "You are a helpful AI assistant with access to file system tools."
        if system_prompt:
            base_system += f"\n\n{system_prompt}"

        full_system = f"{base_system}\n\n{TOOLS_DESCRIPTION}"
        self.history.append({"role": "system", "content": full_system})

    def _parse_tool_calls(self, content: str) -> List[Dict]:
        """Ищет ВСЕ блоки <tool>...</tool_call> в ответе."""
        pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
        matches = re.findall(pattern, content, re.DOTALL)

        tools = []
        for json_str in matches:
            try:
                tool_data = json.loads(json_str)
                tools.append(tool_data)
            except json.JSONDecodeError:
                continue
        return tools

    def _execute_tool(self, tool_name: str, arguments: Dict) -> str:
        """Выполняет инструмент."""
        if tool_name not in AVAILABLE_TOOLS:
            return f"Error: Unknown tool '{tool_name}'."

        func = AVAILABLE_TOOLS[tool_name]
        try:
            result = func(**arguments)
            return str(result)
        except TypeError as e:
            return f"Error: Invalid arguments for tool '{tool_name}': {str(e)}"
        except Exception as e:
            return f"Error executing tool: {str(e)}"

    def _truncate_text(self, text: str, max_len: int = 80) -> str:
        """Обрезает текст до max_len символов."""
        if len(text) <= max_len:
            return text
        return text[:max_len] + "..."

    def _extract_thoughts(self, content: str) -> Tuple[str, str]:
        """
        Извлекает блок мыслей <think>...</think>.
        Возвращает кортеж: (текст_мыслей, очищенный_контент).
        Если мыслей нет, возвращает ("", исходный_контент).
        """
        # Паттерн ищет <think>, затем любое количество символов (нежадно или жадно до первого закрывающего тега), затем </think>
        # Используем DOTALL, чтобы точка匹配ала переносы строк
        pattern = r"<think>(.*?)</think>"
        match = re.search(pattern, content, re.DOTALL)

        if match:
            thoughts = match.group(1).strip()
            # Удаляем весь блок мыслей из контента
            clean_content = content[:match.start()] + content[match.end():]
            return thoughts, clean_content
        else:
            return "", content

    def chat(self, user_message: str, max_iterations: int = 5, force_thinking_step: bool = False) -> str:
        self.history.append({"role": "user", "content": user_message})

        current_iteration = 0
        final_response = ""

        while current_iteration < max_iterations:
            current_iteration += 1

            content, error = call_api(self.history, self.temperature, self.timeout, thinking=force_thinking_step)

            if error:
                return f"System Error: {error}"

            if not content:
                return "Empty response from model."

            # 1. ПАРСИНГ МЫСЛЕЙ
            thoughts, clean_content = self._extract_thoughts(content)

            if thoughts:
                print("\n" + "="*40)
                print("[Agent thoughts]")
                print("-"*40)
                print(thoughts)
                print("="*40 + "\n")

            # Далее работаем только с очищенным контентом (без мыслей)
            content_to_process = clean_content

            tool_calls = self._parse_tool_calls(content_to_process)

            if tool_calls:
                # 1. Обработка текста ПЕРЕД инструментами
                first_tool_match = re.search(r"<tool_call>\s*.*?\s*</tool_call>", content_to_process, re.DOTALL)
                text_before_tools = ""

                if first_tool_match:
                    text_before_tools = content_to_process[:first_tool_match.start()].strip()

                # Если есть текст перед инструментами, выводим его и сохраняем в историю
                if text_before_tools:
                    print(f"Agent: {text_before_tools}")
                    self.history.append({"role": "assistant", "content": text_before_tools})

                # 2. Выполнение всех инструментов
                tool_results_output = []

                for tool_call in tool_calls:
                    tool_name = tool_call.get("name")
                    args = tool_call.get("arguments", {})

                    print(f"[Agent] Calling tool: {tool_name} with args: {args}")

                    raw_result = self._execute_tool(tool_name, args)

                    # Создаем краткую версию для вывода и истории
                    short_result = self._truncate_text(raw_result, 80)

                    print(f"[Tool Result]: {short_result}")

                    tool_results_output.append(f"Tool '{tool_name}' result:\n{raw_result}")

                # 3. Отправка результатов модели
                combined_results = "\n\n---\n\n".join(tool_results_output)
                self.history.append({"role": "user", "content": f"[TOOL OUTPUTS]:\n{combined_results}"})

                continue
            else:
                # Финальный ответ без инструментов
                final_response = content_to_process
                print(f"Agent: {content_to_process}")
                self.history.append({"role": "assistant", "content": content_to_process})
                break

        return final_response

# =============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# =============================================================================

if __name__ == "__main__":
    agent = LLMAgent(system_prompt="Be concise and accurate when handling file paths.")

    print("=== Агент запущен. Введите команду или 'exit' ===")

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        response = agent.chat(user_input, max_iterations=30, force_thinking_step=True)