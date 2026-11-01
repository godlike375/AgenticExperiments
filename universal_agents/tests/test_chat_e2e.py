from __future__ import annotations

from unittest import mock
import pytest

from universal_agents.agent import LLMAgent
from universal_agents.models import AssistantMessage, ToolCall, UserMessage, ToolResult
from universal_agents.generation import StructuredOutputConfig
from tests.conftest import make_agent, double_me


def test_chat_single_turn_with_thinking_default():
    agent = make_agent()
    # Thinking по умолчанию включено (reasoning_effort == "low")
    assert agent._thinking_enabled is True
    assert agent._reasoning_effort == "low"

    calls = []
    reply = AssistantMessage(content="Hello world", reasoning_content="Thinking step")

    def fake_call(messages, reasoning_effort=None, **kwargs):
        calls.append(reasoning_effort)
        return reply, None, None

    with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake_call):
        res = agent.chat("Hi")

    assert res == "Hello world"
    assert calls == ["low"]
    # Проверяем сохранение истории
    msgs = agent.history.get_all()
    assert len(msgs) == 3  # [system, user, assistant]
    assert msgs[1].content == "Hi"
    assert msgs[2].content == "Hello world"
    assert msgs[2].reasoning_content == "Thinking step"


def test_chat_turn_with_tool_call_and_result():
    agent = make_agent(tools_config=["double_me"], external_plugins={"double_me": double_me})

    tc = ToolCall(id="call_1", name="double_me", arguments='{"value": 21}')
    first_reply = AssistantMessage(content="", tool_calls=[tc], reasoning_content="Calculating")
    final_reply = AssistantMessage(content="Result is 42", reasoning_content="Done")

    responses = [(first_reply, None, None), (final_reply, None, None)]
    seen_calls = []

    def fake_call(messages, reasoning_effort=None, **kwargs):
        seen_calls.append(reasoning_effort)
        return responses.pop(0)

    with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake_call):
        res = agent.chat("double 21")

    assert res == "Result is 42"
    assert len(seen_calls) == 2
    # Проверяем структуру сообщений по правилам API
    roles = [m.to_api_dict()["role"] for m in agent.history.get_all()]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    tool_msg = agent.history.get_all()[3]
    assert tool_msg.content == "42"
    assert tool_msg.tool_call_id == "call_1"


def test_simulated_reasoning_fallback(monkeypatch):
    from universal_agents.config import Config
    monkeypatch.setattr(Config, "SIMULATED_REASONING_ENABLED", True)

    agent = make_agent()
    agent._thinking_enabled = False
    assert agent._reasoning_effort == "none"

    tag = Config.SIMULATED_REASONING_TAGS[0]
    expected_result = f"<{tag}>мысли</{tag}>\n42"

    def fake_call(messages, prefill=None, stop_markers=(), **kwargs):
        return AssistantMessage(content=expected_result), None, None

    with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake_call):
        res = agent.chat("сколько будет 21*2?")

    assert res == expected_result
    last_msg = agent.history.get_all()[-1]
    assert last_msg.content == expected_result


def test_structured_output_multi_phase():
    phase1 = AssistantMessage(content="<structure>\nL1-5 foo\n</structure>\nignored text")
    phase2 = AssistantMessage(content="<details>\nitem1\n</details>")
    responses = [(phase1, None, None), (phase2, None, None)]
    call_idx = [0]

    def fake_call(messages, prefill=None, **kwargs):
        idx = call_idx[0]
        call_idx[0] += 1
        return responses[idx]

    agent = make_agent()
    cfg = StructuredOutputConfig.from_prefill(
        "<structure>\n", next_prefill="<details>\n", max_phases=5
    )

    with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake_call):
        result = agent.chat("Give me the structure", structured_output=cfg)

    assert "item1" in result
    assert "<details>" in result
    assert call_idx[0] == 2


def test_multimodal_images_in_user_and_tool_messages():
    b64_img = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    agent = make_agent()

    # User message с изображением
    agent.history.add(UserMessage("Look at this image", images=[b64_img]))
    # Tool result с изображением
    agent.history.add(ToolResult.success(tool_call_id="t1", name="screenshot", content="captured"))
    agent.history.get_all()[-1].images = [b64_img]

    user_api = agent.history.get_all()[1].to_api_dict()
    assert user_api["role"] == "user"
    assert isinstance(user_api["content"], list)
    assert any(part.get("type") == "image_url" for part in user_api["content"])

    tool_api = agent.history.get_all()[2].to_api_dict()
    assert tool_api["role"] == "tool"
    assert isinstance(tool_api["content"], list)
    assert any(part.get("type") == "image_url" for part in tool_api["content"])


def test_message_representations_api_vs_persist_invariant():
    # Инвариант §1.6: to_api_dict() содержит только нужное LLM, to_persist_dict() сохраняет все
    msg = AssistantMessage(
        content="Answer",
        reasoning_content="Thought",
        tool_calls=[ToolCall(id="c1", name="read", arguments='{"path": "a.py"}')],
    )

    api_dict = msg.to_api_dict()
    persist_dict = msg.to_persist_dict()

    assert api_dict["role"] == "assistant"
    assert api_dict["content"] == "Answer"
    assert "tool_calls" in api_dict
    assert api_dict["tool_calls"][0]["function"]["name"] == "read"

    # persist содержит полную сериализацию для JSON (tool_calls как словари)
    assert persist_dict["content"] == "Answer"
    assert persist_dict["reasoning_content"] == "Thought"
    assert persist_dict["tool_calls"][0]["function"]["name"] == "read"
