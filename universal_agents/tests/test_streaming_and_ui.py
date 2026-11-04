from __future__ import annotations

import time
from unittest import mock
import pytest

from universal_agents.constants import INTERRUPT_HEADER
from universal_agents.models import AssistantMessage, UserMessage, ToolCall, ToolResult
from universal_agents.exceptions import GenerationInterrupted
from universal_agents.ui import CLI
from tests.conftest import make_agent, stream_chunk, stream_delta, double_me


def test_live_streaming_chunks():
    agent = make_agent()
    received_content = []
    received_reasoning = []

    agent.on_stream_chunk = received_content.append
    agent.on_reasoning_chunk = received_reasoning.append

    chunks = [
        stream_chunk(stream_delta(reasoning_content="think1 ")),
        stream_chunk(stream_delta(reasoning_content="think2 ")),
        stream_chunk(stream_delta(content="answer1 ")),
        stream_chunk(stream_delta(content="answer2")),
    ]

    def fake_stream(*args, **kwargs):
        for c in chunks:
            yield c

    with mock.patch("universal_agents.llm_client.LLMClient.open_stream", side_effect=fake_stream):
        reply = agent.chat("test streaming")

    assert reply == "answer1 answer2"
    assert "".join(received_content) == "answer1 answer2"
    assert "".join(received_reasoning) == "think1 think2 "


def test_user_stop_interrupts_generation():
    agent = make_agent()

    def fake_call(*args, **kwargs):
        # Симулируем сигнал остановки
        agent.request_stop()
        if agent.stop_event.is_set():
            raise GenerationInterrupted("Stopped by user")
        return AssistantMessage(content="normal"), None, None

    with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake_call):
        res = agent.chat("run long task")

    assert "прерван" in res.lower() or "interrupted" in res.lower() or res == ""


def test_cli_commands():
    agent = make_agent()
    cli = CLI(agent)

    # Проверяем команду /think_on и /think_off
    agent._thinking_enabled = False
    cli.commands["/think_on"](["/think_on"])
    assert agent._thinking_enabled is True
    assert agent._reasoning_effort == "low"

    cli.commands["/think_off"](["/think_off"])
    assert agent._thinking_enabled is False
    assert agent._reasoning_effort == "none"

    # Проверяем /clear
    agent.history.add(AssistantMessage(content="hello"))
    cli.commands["/clear"](["/clear"])
    assert len(agent.history) == 1  # Только system prompt


def test_inject_user_interrupt_live():
    # 1. Инъекция в середину диалога создает UserMessage с заголовком INTERRUPT_HEADER
    agent = make_agent()
    fake_reply = AssistantMessage(content="redirected answer")
    with mock.patch("universal_agents.agent.LLMClient.call", return_value=(fake_reply, None, None)):
        res = agent.inject_user_interrupt("new goal")

    assert res == "redirected answer"
    user_msgs = [m.content for m in agent.history.get_all() if isinstance(m, UserMessage)]
    assert any(INTERRUPT_HEADER in c for c in user_msgs)
    assert any("new goal" in c for c in user_msgs)

    # 2. Если прерывание пришло сразу после ToolResult, оно подклеивается в конец ToolResult
    agent2 = make_agent(tools_config=["double_me"], external_plugins={"double_me": double_me})
    agent2.history.add(UserMessage("calculate"))
    agent2.history.add(AssistantMessage(
        content="",
        tool_calls=[ToolCall(id="t1", name="double_me", arguments='{"value": 10}')],
    ))
    tr = ToolResult.success(tool_call_id="t1", name="double_me", content="20")
    agent2.history.add(tr)

    with mock.patch("universal_agents.agent.LLMClient.call", return_value=(fake_reply, None, None)):
        agent2.inject_user_interrupt("stop and rethink")

    assert INTERRUPT_HEADER in tr.content
    assert "stop and rethink" in tr.content
