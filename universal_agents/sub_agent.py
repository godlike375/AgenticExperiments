from typing import Optional, Callable, Union
from universal_agents.models import AssistantMessage, ToolResult
from universal_agents.llm_client import TokenUsageTracker
from universal_agents.config import Config
import sys

MAX_SUB_AGENT_DEPTH = 1


class SubAgent:
    """
    Субагент на базе LLMAgent.
    - Наследует системный промпт, историю и набор инструментов родителя для KV-cache reuse.
    - Изолированный трекер токенов.
    - Рекурсия предотвращается через depth check.
    """

    def __init__(
            self,
            system_prompt: str = 'You are a sub-agent. Always respond in "<sub-agent>" tags.',
            max_context_tokens: int = None,
            tools_config: Union[list[str], dict, None] = None,
            external_plugins: Optional[dict] = None,
            safe_only: bool = True,
            max_iter: int = None,
            temp: float = None,
            on_log: Callable[[str], None] = lambda x: None,
            top_p: float = None,
            frequency_penalty: float = None,
            presence_penalty: float = None,
            max_tokens: int = None,
            timeout: int = None,
            # KV-cache: наследование от родителя
            parent_history: Optional[list] = None,
            parent_system_prompt: Optional[str] = None,
            # Recursion prevention
            depth: int = 0,
            # Streaming
            on_stream_chunk: Optional[Callable[[str], None]] = None,
            on_stream_start: Optional[Callable[[], None]] = None,
            on_stream_end: Optional[Callable[[], None]] = None,
    ):
        from agent import LLMAgent

        self._max_iter = max_iter if max_iter is not None else Config.MAX_ITER
        self._on_log = on_log
        self._depth = depth

        # Если указан системный промпт родителя — используем его для разделения KV-кеша
        effective_system_prompt = parent_system_prompt if parent_system_prompt is not None else system_prompt

        # Фильтрация опасных инструментов
        safe_plugins = external_plugins
        if safe_only and external_plugins:
            safe_plugins = {
                name: func for name, func in external_plugins.items()
                if not getattr(func, '_requires_confirmation', False)
            }

        # Полный контекст: без деления бюджета — сервер сам управляет limits
        effective_max_context_tokens = max_context_tokens if max_context_tokens is not None else Config.MAX_CONTEXT_TOKENS
        self._own_tracker = TokenUsageTracker(effective_system_prompt, effective_max_context_tokens)

        def _render_subagent(msg):
            output = msg.render(label="[sub]")
            if output:
                on_log(output)

        # Streaming для субагента (по умолчанию — вывод в stdout)
        if on_stream_chunk is None:
            def _sub_stream_start():
                sys.stdout.write("[sub] ")
                sys.stdout.flush()
            def _sub_stream_chunk(chunk):
                sys.stdout.write(chunk)
                sys.stdout.flush()
            def _sub_stream_end():
                sys.stdout.write("\n")
                sys.stdout.flush()
            on_stream_start = _sub_stream_start
            on_stream_chunk = _sub_stream_chunk
            on_stream_end = _sub_stream_end

        self._agent = LLMAgent(
            system_prompt=effective_system_prompt,
            temp=temp if temp is not None else Config.TEMP,
            timeout=timeout if timeout is not None else 60,
            tools_config=tools_config,
            external_plugins=safe_plugins,
            on_render=_render_subagent,
            on_confirm=lambda n, a: True,
            on_system_msg=on_log,
            max_context_tokens=effective_max_context_tokens,
            top_p=top_p if top_p is not None else Config.TOP_P,
            frequency_penalty=frequency_penalty if frequency_penalty is not None else Config.FREQUENCY_PENALTY,
            presence_penalty=presence_penalty if presence_penalty is not None else Config.PRESENCE_PENALTY,
            max_tokens=max_tokens if max_tokens is not None else Config.MAX_OUTPUT_TOKENS,
            max_generation_attempts=1,
            streaming_enabled=True,
            on_stream_chunk=on_stream_chunk,
            on_stream_start=on_stream_start,
            on_stream_end=on_stream_end,
        )
        self._agent.token_tracker = self._own_tracker
        self._agent._depth = depth

        # Клонируем историю родителя как префикс для KV-cache reuse
        if parent_history:
            self._agent.history.extend(parent_history)

    def run(self, task: str, prefill: str = None) -> str:
        """Выполняет задачу и возвращает финальный текстовый ответ."""
        self._agent.chat(task, max_iter=self._max_iter, prefill=prefill)
        last_msg = self._agent.history.get_last_message()
        if isinstance(last_msg, AssistantMessage):
            return last_msg.content or ""
        return ""

    def get_last_tool_call(self):
        """Возвращает последний tool_call (для structured output)."""
        for msg in reversed(self._agent.history.get_all()):
            if isinstance(msg, AssistantMessage) and msg.has_tool_calls():
                return msg.tool_calls[-1]
        return None

    @property
    def tokens_spent(self) -> int:
        if self._own_tracker.last_usage:
            return self._own_tracker.last_usage.get("total_tokens", 0)
        return 0
