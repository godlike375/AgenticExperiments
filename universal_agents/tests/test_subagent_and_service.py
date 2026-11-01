from __future__ import annotations

from unittest import mock
import pytest

from universal_agents.models import AssistantMessage, UserMessage
from tests.conftest import make_agent, double_me


def test_subagent_inherits_context_and_limits():
    parent = make_agent(
        system_prompt="Parent system prompt",
        tools_config=["double_me"],
        external_plugins={"double_me": double_me},
    )
    parent.history.add(UserMessage("Previous message"))
    parent.history.add(AssistantMessage(content="Previous answer"))

    sub = parent.make_sub_agent(safe_only=False, max_iter=5)

    # Проверяем наследование промпта и доступ к инструментам внутреннего агента
    assert sub._agent.history[0].content == "Parent system prompt"
    assert "double_me" in sub._agent._all_tools
    # Трекер токенов должен быть независимым
    assert sub._own_tracker is not parent.token_tracker


def test_service_llm_call_isolated_from_history():
    agent = make_agent()
    agent.history.add(UserMessage("Chat question"))
    agent.history.add(AssistantMessage(content="Chat answer"))

    initial_len = len(agent.history)

    # Служебный вызов LLM (например, внутренняя классификация или суммаризация)
    service_reply = AssistantMessage(content="Service internal result")
    with mock.patch("universal_agents.agent.LLMClient.call", return_value=(service_reply, None, None)):
        msg_obj, err, _ = agent.service_llm_call([{"role": "user", "content": "analyze this"}])

    assert msg_obj.content == "Service internal result"
    # Служебный вызов НЕ должен менять постоянную историю диалога (§2.4)
    assert len(agent.history) == initial_len
    assert agent.history.get_all()[-1].content == "Chat answer"
