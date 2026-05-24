import json
from collections import deque
from datetime import datetime
from typing import Optional
from openai import OpenAI
from universal_agents.tool import ENVIRONMENT_PREFIX
from config import Config

class TokenUsageTracker:
    def __init__(self, max_context_tokens: int = 8192):
        self.max_context_tokens = max_context_tokens
        self.last_usage = None

    def update_from_usage(self, usage: dict):
        self.last_usage = usage

    def estimate_tokens(self, text: str) -> int:
        """Грубая оценка токенов: символы / 2.5"""
        return int(len(text) / 2.5)

    def get_total_context_tokens(self, first_system_message: str = "", last_user_content: str = "") -> int:
        known = self.estimate_tokens(first_system_message)
        if self.last_usage:
            known = self.last_usage.get("prompt_tokens", 0)
        if last_user_content:
            known += self.estimate_tokens(last_user_content)
        return known

    def format_timestamp_header(self, msg) -> str:
        """Метка времени из timestamp сообщения."""
        ts = msg.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        return f"<<{ENVIRONMENT_PREFIX} === [{ts}]"

    def format_token_header(self, first_system_message: str = "", last_user_content: str = "") -> str:
        """Только информация о токенах (с учётом последнего сообщения)."""
        total = self.get_total_context_tokens(first_system_message, last_user_content)
        remaining = self.max_context_tokens - total
        return f" | Tokens spent: {total} (Remaining: {remaining})"

    def format_closing_header(self) -> str:
        """Закрывающая часть заголовка."""
        return f" ===>>\n\n"

class LoopDetector:
    def __init__(self):
        self.threshold = 1

    @staticmethod
    def normalize_args(args_str: str) -> str:
        if not args_str or args_str.strip() in ("{}", "", "null"):
            return ""
        try:
            parsed = json.loads(args_str)
            return json.dumps(parsed, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
        except Exception:
            return args_str.strip()

    def check_duplicate_in_turn(self, tool_name: str, arguments: str, messages: list) -> bool:
        """
        Проверяет, вызывался ли уже этот инструмент с такими же параметрами
        после последнего сообщения пользователя (в рамках текущего хода).
        """
        norm_args = self.normalize_args(arguments)

        # Идем с конца истории сообщений
        for msg in reversed(messages):
            # Если дошли до сообщения пользователя, значит этот ход начался здесь.
            # Всё, что было до него, не считается повтором в текущем ходу.
            from models import UserMessage, AssistantMessage
            if isinstance(msg, UserMessage):
                break

            if isinstance(msg, AssistantMessage):
                for tc in msg.tool_calls:
                    if tc.name == tool_name:
                        if self.normalize_args(tc.arguments) == norm_args:
                            return True
        return False

class LLMClient:
    _client = None

    @classmethod
    def get_client(cls) -> OpenAI:
        if cls._client is None:
            cls._client = OpenAI(api_key="lm-studio", base_url=Config.API_URL)
        return cls._client

    @staticmethod
    def call(messages: list[dict], temp: float, timeout: int, tools: list[dict] = None, prefill: str = None):
        messages_to_send = list(messages)
        if prefill:
            messages_to_send.append({"role": "assistant", "content": prefill})
        try:
            response = LLMClient.get_client().chat.completions.create(
                model=Config.MODEL_NAME,
                messages=messages_to_send,
                temperature=temp,
                max_tokens=12000,
                tools=tools,
                parallel_tool_calls=False,
                timeout=timeout,
                reasoning_effort="none"
            )
            msg = response.choices[0].message
            if prefill and msg.content and not msg.content.startswith(prefill):
                msg.content = prefill + msg.content

            usage = None
            if hasattr(response, 'usage') and response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            return msg, None, usage
        except Exception as e:
            return None, str(e), None
