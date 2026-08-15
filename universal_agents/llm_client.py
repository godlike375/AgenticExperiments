import json
from typing import Optional
from openai import OpenAI
from universal_agents.config import Config, CHARS_PER_TOKEN
from universal_agents.generation import GenerationParams


def build_usage_dict(prompt_tokens: int, completion_tokens: int, total_tokens: Optional[int] = None) -> dict:
    """Собирает usage-словарь в едином формате для агента."""
    if total_tokens is None:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def apply_prefill(content: Optional[str], prefill: Optional[str]) -> str:
    """Добавляет prefill в начало содержимого, если его там ещё нет."""
    if prefill is None:
        return content or ""
    content = content or ""
    if content.startswith(prefill):
        return content
    return prefill + content


class TokenUsageTracker:
    def __init__(self, system_prompt: str, max_context_tokens: int = 8192):
        self.max_context_tokens = max_context_tokens
        self.last_usage = None
        self.system_prompt = system_prompt

    def update_from_usage(self, usage: dict):
        self.last_usage = usage

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Грубая оценка токенов: символы / CHARS_PER_TOKEN"""
        return int(len(text) / CHARS_PER_TOKEN)

    def get_total_context_tokens(self, first_system_message: str = "", last_user_content: str = "") -> int:
        known = self.estimate_tokens(first_system_message)
        if self.last_usage:
            known = self.last_usage.get("prompt_tokens", 0)
        if last_user_content:
            known += self.estimate_tokens(last_user_content)
        return known

    def get_remaining(self, last_user_content: str = ""):
        total = self.get_total_context_tokens(self.system_prompt, last_user_content)
        remaining = self.max_context_tokens - total
        return remaining

    def format_user_token_info(self) -> str:
        """Информация о токенах для отображения пользователю."""
        if not self.last_usage:
            return ""
        total = self.last_usage.get("prompt_tokens", 0)
        remaining = self.max_context_tokens - total
        return f"Tokens spent: {total} (Remaining: {remaining})"

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
            from universal_agents.models import UserMessage, AssistantMessage
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
    def call(
        messages: list[dict],
        temp: float = None,
        timeout: int = None,
        tools: list[dict] = None,
        prefill: str = None,
        top_p: float = None,
        frequency_penalty: float = None,
        presence_penalty: float = None,
        max_tokens: int = None,
        previous_response_id: str = None,
        params: GenerationParams = None,
    ):
        temp, timeout, top_p, frequency_penalty, presence_penalty, max_tokens = LLMClient._resolve_params(
            params, temp, timeout, top_p, frequency_penalty, presence_penalty, max_tokens
        )

        messages_to_send = list(messages)
        if prefill:
            messages_to_send.append({"role": "assistant", "content": prefill})

        if Config.USE_RESPONSES_API:
            if previous_response_id is not None:
                msg, err, usage = LLMClient._call_responses_api(
                    messages_to_send, temp, timeout, tools, top_p,
                    frequency_penalty, presence_penalty, max_tokens, previous_response_id
                )
                if not err and msg and (msg.content or msg.tool_calls):
                    return msg, err, usage
            msg, err, usage = LLMClient._call_responses_api_full(
                messages_to_send, temp, timeout, tools, top_p,
                frequency_penalty, presence_penalty, max_tokens
            )
            if not err and msg and (msg.content or msg.tool_calls):
                return msg, err, usage

        return LLMClient._call_chat_completions(
            messages_to_send, temp, timeout, tools, prefill, top_p,
            frequency_penalty, presence_penalty, max_tokens
        )

    @staticmethod
    def _resolve_params(params, temp, timeout, top_p, frequency_penalty, presence_penalty, max_tokens):
        if params is not None:
            p = params.resolved()
            temp = p.temp if temp is None else temp
            timeout = p.timeout if timeout is None else timeout
            top_p = p.top_p if top_p is None else top_p
            frequency_penalty = p.frequency_penalty if frequency_penalty is None else frequency_penalty
            presence_penalty = p.presence_penalty if presence_penalty is None else presence_penalty
            max_tokens = p.max_tokens if max_tokens is None else max_tokens
        return temp, timeout, top_p, frequency_penalty, presence_penalty, max_tokens

    @staticmethod
    def _call_chat_completions(messages_to_send, temp, timeout, tools, prefill, top_p,
                               frequency_penalty, presence_penalty, max_tokens):
        try:
            response = LLMClient.get_client().chat.completions.create(
                model=Config.MODEL_NAME,
                messages=messages_to_send,
                temperature=temp if temp is not None else Config.TEMP,
                max_tokens=max_tokens if max_tokens is not None else Config.MAX_OUTPUT_TOKENS,
                tools=tools,
                parallel_tool_calls=False,
                timeout=timeout if timeout is not None else Config.TIMEOUT,
                reasoning_effort="none",
                frequency_penalty=frequency_penalty if frequency_penalty is not None else Config.FREQUENCY_PENALTY,
                presence_penalty=presence_penalty if presence_penalty is not None else Config.PRESENCE_PENALTY,
                top_p=top_p if top_p is not None else Config.TOP_P,
            )
            msg = response.choices[0].message
            msg.content = apply_prefill(msg.content, prefill)

            usage = None
            if hasattr(response, 'usage') and response.usage:
                usage = build_usage_dict(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                    response.usage.total_tokens,
                )
            return msg, None, usage
        except Exception as e:
            return None, str(e), None

    @staticmethod
    def _call_responses_api_full(messages_to_send, temp, timeout, tools, top_p,
                                 frequency_penalty, presence_penalty, max_tokens):
        try:
            kwargs = {
                "model": Config.MODEL_NAME,
                "input": messages_to_send,
                "temperature": temp if temp is not None else Config.TEMP,
                "max_output_tokens": max_tokens if max_tokens is not None else Config.MAX_OUTPUT_TOKENS,
                "timeout": timeout if timeout is not None else Config.TIMEOUT,
                "reasoning_effort": "none",
            }
            if tools:
                kwargs["tools"] = tools
            if top_p is not None:
                kwargs["top_p"] = top_p

            response = LLMClient.get_client().responses.create(**kwargs)
            msg = LLMClient._parse_responses_output(response)
            if msg and prefill:
                msg.content = apply_prefill(msg.content, prefill)
            return msg, None, LLMClient._extract_responses_usage(response)
        except Exception as e:
            return None, str(e), None

    @staticmethod
    def _call_responses_api(messages_to_send, temp, timeout, tools, top_p,
                            frequency_penalty, presence_penalty, max_tokens, previous_response_id):
        try:
            kwargs = {
                "model": Config.MODEL_NAME,
                "input": messages_to_send,
                "previous_response_id": previous_response_id,
                "temperature": temp if temp is not None else Config.TEMP,
                "max_output_tokens": max_tokens if max_tokens is not None else Config.MAX_OUTPUT_TOKENS,
                "timeout": timeout if timeout is not None else Config.TIMEOUT,
                "reasoning_effort": "none",
            }
            if tools:
                kwargs["tools"] = tools
            if top_p is not None:
                kwargs["top_p"] = top_p

            response = LLMClient.get_client().responses.create(**kwargs)
            msg = LLMClient._parse_responses_output(response)
            if msg and prefill:
                msg.content = apply_prefill(msg.content, prefill)
            return msg, None, LLMClient._extract_responses_usage(response)
        except Exception as e:
            return None, str(e), None

    @staticmethod
    def _parse_responses_output(response):
        from types import SimpleNamespace
        text_content = ""
        tool_calls = []
        for item in response.output:
            if item.type == "message":
                if hasattr(item, 'content') and item.content:
                    if isinstance(item.content, list):
                        for part in item.content:
                            if hasattr(part, 'type') and part.type == "output_text":
                                text_content += part.text
                            elif hasattr(part, 'text'):
                                text_content += part.text
                    elif isinstance(item.content, str):
                        text_content += item.content
            elif item.type == "function_call":
                tool_calls.append(SimpleNamespace(
                    id=item.call_id,
                    name=item.name,
                    arguments=item.arguments,
                    function=SimpleNamespace(name=item.name, arguments=item.arguments)
                ))

        msg = SimpleNamespace(
            content=text_content,
            tool_calls=tool_calls if tool_calls else None,
        )
        msg._response_id = response.id
        return msg

    @staticmethod
    def _extract_responses_usage(response):
        if not hasattr(response, 'usage') or not response.usage:
            return None
        usage = response.usage
        return build_usage_dict(
            getattr(usage, 'input_tokens', 0),
            getattr(usage, 'output_tokens', 0),
        )


    @staticmethod
    def stream(
        messages: list[dict],
        temp: float = None,
        timeout: int = None,
        tools: list[dict] = None,
        prefill: str = None,
        top_p: float = None,
        frequency_penalty: float = None,
        presence_penalty: float = None,
        max_tokens: int = None,
        previous_response_id: str = None,
        params: GenerationParams = None,
    ):
        """Streaming version of call() - returns generator of chunks.
        Note: previous_response_id is ignored for streaming (Responses API streaming not yet supported).
        """
        temp, timeout, top_p, frequency_penalty, presence_penalty, max_tokens = LLMClient._resolve_params(
            params, temp, timeout, top_p, frequency_penalty, presence_penalty, max_tokens
        )

        messages_to_send = list(messages)
        if prefill:
            messages_to_send.append({"role": "assistant", "content": prefill})
        
        try:
            stream = LLMClient.get_client().chat.completions.create(
                model=Config.MODEL_NAME,
                messages=messages_to_send,
                temperature=temp if temp is not None else Config.TEMP,
                max_tokens=max_tokens if max_tokens is not None else Config.MAX_OUTPUT_TOKENS,
                tools=tools,
                parallel_tool_calls=False,
                timeout=timeout if timeout is not None else Config.TIMEOUT,
                reasoning_effort="none",
                frequency_penalty=frequency_penalty if frequency_penalty is not None else Config.FREQUENCY_PENALTY,
                presence_penalty=presence_penalty if presence_penalty is not None else Config.PRESENCE_PENALTY,
                top_p=top_p if top_p is not None else Config.TOP_P,
                stream=True,
                stream_options={"include_usage": True},
            )
            return stream
        except Exception as e:
            # Для streaming ошибки возвращаем генератор с ошибкой
            def error_generator(err=str(e)):
                yield {"error": err}
            return error_generator()
