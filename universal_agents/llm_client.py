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

    def get_current_context_tokens(self):
        if self.last_usage:
            return self.last_usage.get("prompt_tokens")
        return None

    def format_info_header(self):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current = self.get_current_context_tokens()
        if current is not None:
            remaining = self.max_context_tokens - current
            token_str = f"Tokens spent: {current} (Remaining: {remaining})"
        else:
            token_str = "Tokens: unknown"
        return f"<<{ENVIRONMENT_PREFIX} === [{timestamp}] | {token_str} ===>>\n\n"

class LoopDetector:
    def __init__(self, threshold: int = 3):
        self.threshold = threshold
        self.recent_calls = deque(maxlen=threshold)

    @staticmethod
    def normalize_args(args_str: str) -> str:
        if not args_str or args_str.strip() in ("{}", "", "null"):
            return ""
        try:
            parsed = json.loads(args_str)
            return json.dumps(parsed, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
        except Exception:
            return args_str.strip()

    def add_call(self, tool_name: str, arguments: str):
        norm_args = self.normalize_args(arguments)
        self.recent_calls.append((tool_name, norm_args))

    def detect_loop(self) -> Optional[tuple]:
        if len(self.recent_calls) < self.threshold:
            return None
        last_n = list(self.recent_calls)[-self.threshold:]
        first = last_n[0]
        if all(call == first for call in last_n):
            return first[0], first[1], self.threshold
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
    def call(messages: list[dict], temp: float, timeout: int, tools: list[dict] = None, prefill: str = None):
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
