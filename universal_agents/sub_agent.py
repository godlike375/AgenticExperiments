from typing import Optional, Callable, Union
from models import AssistantMessage
from llm_client import TokenUsageTracker


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
            max_context_tokens: int = 8192,
            tools_config: Union[list[str], dict, None] = None,
            external_plugins: Optional[dict] = None,
            safe_only: bool = True,
            max_iter: int = 5,
            temp: float = 0.2,
            on_log: Callable[[str], None] = lambda x: None,
    ):
        # Отложенный импорт для разрыва цикла agent ↔ sub_agent
        from agent import LLMAgent

        self._max_iter = max_iter
        self._on_log = on_log

        # Фильтрация опасных инструментов
        safe_plugins = external_plugins
        if safe_only and external_plugins:
            safe_plugins = {
                name: func for name, func in external_plugins.items()
                if not getattr(func, '_requires_confirmation', False)
            }

        # Изолированный трекер: траты субагента НЕ влияют на бюджет основного агента
        self._own_tracker = TokenUsageTracker(system_prompt, max_context_tokens)

        self._agent = LLMAgent(
            system_prompt=system_prompt,
            temp=temp,
            timeout=60,
            tools_config=tools_config,
            external_plugins=safe_plugins,
            on_render=lambda msg: None,
            on_confirm=lambda n, a: True,
            on_system_msg=on_log,
            max_context_tokens=max_context_tokens,
            _create_judge=False,  # <-- предотвращает рекурсию
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
