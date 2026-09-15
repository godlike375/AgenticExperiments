"""Общие фикстуры и фабрики тестового окружения (pytest).

Правило AGENTS.md §5: тестового агента не создавать в каждом тесте —
использовать make_agent отсюда. Для юнит-тестов хелперов (task_tracker,
compressors) по-прежнему допустимы лёгкие специализированные фейки
(SimpleNamespace / MemoryMixin-only), когда полный LLMAgent не нужен.
"""

from __future__ import annotations

from universal_agents.agent import LLMAgent


def make_agent(
    system_prompt: str = "You are a helpful assistant",
    tools_config=None,
    external_plugins: dict | None = None,
    disable_per_msg_summarization: bool = True,
    autosave_enabled: bool = False,
    **kwargs,
) -> LLMAgent:
    """Фабрика реального LLMAgent для поведенческих тестов (пункт AGENTS.md §5).

    Отключает per-message суммаризацию и авто-сохранение по умолчанию
    (чистый, детерминированный диалог). Каждый тест сам добавляет доверенные
    папки и внешние инструменты под свою задачу."""
    return LLMAgent(
        system_prompt=system_prompt,
        tools_config=tools_config,
        external_plugins=external_plugins,
        disable_per_msg_summarization=disable_per_msg_summarization,
        autosave_enabled=autosave_enabled,
        **kwargs,
    )