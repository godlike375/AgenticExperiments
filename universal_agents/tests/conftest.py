
from __future__ import annotations

import pytest
from types import SimpleNamespace

from universal_agents.agent import LLMAgent
from universal_agents.config import Config
from universal_agents.llm_client import LLMClient
from universal_agents.tool import tool


@tool(description="double a value")
def double_me(agent, value: int) -> str:
    return str(value * 2)


@tool(description="always fails")
def fail_me(agent, value: int) -> str:
    raise ValueError("boom")


def stream_chunk(delta=None, usage=None, choices=None):
    if choices is None:
        choices = [SimpleNamespace(delta=delta)] if delta is not None else []
    return SimpleNamespace(choices=choices, usage=usage)


def stream_delta(content=None, tool_calls=None, reasoning_content=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls, reasoning_content=reasoning_content)


@pytest.fixture(autouse=True)
def sim_reasoning_off(monkeypatch):
    monkeypatch.setattr(Config, "SIMULATED_REASONING_ENABLED", False)


@pytest.fixture(autouse=True)
def no_real_llm_transport(monkeypatch, request):
    if "test_in_process_stream" in str(request.node.fspath):
        return

    def _no_transport(*args, **kwargs):
        raise RuntimeError("LLM transport not patched in test (patch LLMClient.open_stream)")

    monkeypatch.setattr(LLMClient, "open_stream", staticmethod(_no_transport))


@pytest.fixture(autouse=True)
def in_process_tool_isolation(monkeypatch):
    monkeypatch.setattr(Config, "TOOL_ISOLATION_ENABLED", False)


def make_agent(
    system_prompt: str = "You are a helpful assistant",
    tools_config=None,
    external_plugins: dict | None = None,
    disable_per_msg_summarization: bool = True,
    autosave_enabled: bool = False,
    **kwargs,
) -> LLMAgent:
    return LLMAgent(
        system_prompt=system_prompt,
        tools_config=tools_config,
        external_plugins=external_plugins,
        disable_per_msg_summarization=disable_per_msg_summarization,
        autosave_enabled=autosave_enabled,
        **kwargs,
    )