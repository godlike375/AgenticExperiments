"""Общие фикстуры и фабрики тестового окружения (pytest).

Правило AGENTS.md §5: тестового агента не создавать в каждом тесте —
использовать make_agent отсюда. Для юнит-тестов хелперов (task_tracker,
compressors) по-прежнему допустимы лёгкие специализированные фейки
(SimpleNamespace / MemoryMixin-only), когда полный LLMAgent не нужен.
"""

from __future__ import annotations

import pytest
from types import SimpleNamespace

from universal_agents.agent import LLMAgent
from universal_agents.config import Config
from universal_agents.tool import tool


@tool(description="double a value")
def double_me(agent, value: int) -> str:
    return str(value * 2)


@tool(description="always fails")
def fail_me(agent, value: int) -> str:
    raise ValueError("boom")


def stream_chunk(delta=None, usage=None, choices=None):
    """Фейк чанка стрима: единая форма для всех streaming-тестов."""
    if choices is None:
        choices = [SimpleNamespace(delta=delta)] if delta is not None else []
    return SimpleNamespace(choices=choices, usage=usage)


def stream_delta(content=None, tool_calls=None, reasoning_content=None):
    """Фейк дельты стрима: единая форма для всех streaming-тестов."""
    return SimpleNamespace(content=content, tool_calls=tool_calls, reasoning_content=reasoning_content)


@pytest.fixture(autouse=True)
def sim_reasoning_off(monkeypatch):
    """По умолчанию выключает симуляцию reasoning (Config.SIMULATED_REASONING_ENABLED).

    Фичи, навязывающие формат ответа, ломают фейки, которые отдают «голый» текст:
    каждый такой ответ трактовался бы как нарушение формата и добавлял лишние вызовы
    LLM. Тесты, которым формат нужен (tests/test_simulated_reasoning.py), включают его
    обратно через monkeypatch.setattr — он перебивает эту фикстуру."""
    monkeypatch.setattr(Config, "SIMULATED_REASONING_ENABLED", False)


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