from typing import Optional, Callable, Union
from universal_agents.models import AssistantMessage
from universal_agents.llm_client import TokenUsageTracker
from universal_agents.config import Config


class SubAgent:
    """
    Универсальный субагент на базе LLMAgent.
    - Изолированная история и изолированный трекер токенов.
    - В safe-режиме не получает инструментов, требующих подтверждения.
    - Контекст передаётся явно в задаче, а не через общую историю.
    """

    def __init__(
            self,
            system_prompt: str,
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
    ):
        # Отложенный импорт для разрыва цикла agent ↔ sub_agent
        from agent import LLMAgent

        self._max_iter = max_iter if max_iter is not None else Config.MAX_ITER
        self._on_log = on_log

        # Фильтрация опасных инструментов
        safe_plugins = external_plugins
        if safe_only and external_plugins:
            safe_plugins = {
                name: func for name, func in external_plugins.items()
                if not getattr(func, '_requires_confirmation', False)
            }

        # Изолированный трекер: траты субагента НЕ влияют на бюджет основного агента
        effective_max_context_tokens = max_context_tokens if max_context_tokens is not None else Config.MAX_CONTEXT_TOKENS
        self._own_tracker = TokenUsageTracker(system_prompt, effective_max_context_tokens)

        self._agent = LLMAgent(
            system_prompt=system_prompt,
            temp=temp if temp is not None else Config.TEMP,
            timeout=timeout if timeout is not None else 60,
            tools_config=tools_config,
            external_plugins=safe_plugins,
            on_render=lambda msg: None,
            on_confirm=lambda n, a: True,
            on_system_msg=on_log,
            max_context_tokens=effective_max_context_tokens,
            _create_judge=False,  # <-- предотвращает рекурсию
            top_p=top_p if top_p is not None else Config.TOP_P,
            frequency_penalty=frequency_penalty if frequency_penalty is not None else Config.FREQUENCY_PENALTY,
            presence_penalty=presence_penalty if presence_penalty is not None else Config.PRESENCE_PENALTY,
            max_tokens=max_tokens if max_tokens is not None else Config.MAX_TOKENS,
        )
        self._agent.token_tracker = self._own_tracker

    def run(self, task: str) -> str:
        """Выполняет задачу и возвращает финальный текстовый ответ."""
        self._agent.chat(task, max_iter=self._max_iter)
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
