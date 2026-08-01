import sys
from dataclasses import asdict
from typing import Optional, Callable, Union
from universal_agents.models import AssistantMessage
from universal_agents.rendering import render_message
from universal_agents.llm_client import TokenUsageTracker
from universal_agents.config import Config
from universal_agents.generation import GenerationParams

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
            parent_history: Optional[list] = None,
            parent_system_prompt: Optional[str] = None,
            depth: int = 0,
            on_stream_chunk: Optional[Callable[[str], None]] = None,
            on_stream_start: Optional[Callable[[], None]] = None,
            on_stream_end: Optional[Callable[[], None]] = None,
            on_reasoning_chunk: Optional[Callable[[str], None]] = None,
            on_reasoning_start: Optional[Callable[[], None]] = None,
            on_reasoning_end: Optional[Callable[[], None]] = None,
    ):
        from universal_agents.agent import LLMAgent

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
            output = render_message(msg, label="[🤖sub-agent]")
            if output:
                on_log(output)

        # Streaming для субагента (по умолчанию — вывод в stdout)
        if on_stream_chunk is None:
            def _sub_stream_start():
                sys.stdout.write("[🤖sub-agent] ")
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

        if on_reasoning_chunk is None:
            def _sub_reasoning_start():
                sys.stdout.write("[🤖sub-agent] 📝[reasoning] ")
                sys.stdout.flush()
            def _sub_reasoning_chunk(chunk):
                sys.stdout.write(chunk)
                sys.stdout.flush()
            def _sub_reasoning_end():
                sys.stdout.write("\n")
                sys.stdout.flush()
            on_reasoning_start = _sub_reasoning_start
            on_reasoning_chunk = _sub_reasoning_chunk
            on_reasoning_end = _sub_reasoning_end

        params = GenerationParams.from_overrides(
            temp=temp,
            timeout=timeout if timeout is not None else 60,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            max_tokens=max_tokens,
        )

        self._agent = LLMAgent(
            system_prompt=effective_system_prompt,
            **asdict(params),
            tools_config=tools_config,
            external_plugins=safe_plugins,
            on_render=_render_subagent,
            on_confirm=lambda n, a: True,
            on_system_msg=on_log,
            max_context_tokens=effective_max_context_tokens,
            max_generation_attempts=1,
            streaming_enabled=True,
            on_stream_chunk=on_stream_chunk,
            on_stream_start=on_stream_start,
            on_stream_end=on_stream_end,
            on_reasoning_chunk=on_reasoning_chunk,
            on_reasoning_start=on_reasoning_start,
            on_reasoning_end=on_reasoning_end,
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

    @property
    def tokens_spent(self) -> int:
        if self._own_tracker.last_usage:
            return self._own_tracker.last_usage.get("total_tokens", 0)
        return 0
