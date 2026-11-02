from __future__ import annotations

import os
import shutil
import tempfile
from unittest import mock
import pytest

from universal_agents.models import AssistantMessage, UserMessage, ToolCall
from universal_agents.task_tracker import complete_milestone, set_plan
from universal_agents.context_builder import prepare_messages_for_api
from tests.conftest import make_agent, double_me


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_auto_summarize_when_budget_exceeded():
    agent = make_agent(disable_per_msg_summarization=False)

    # Наполняем историю длинными сообщениями
    for i in range(10):
        agent.history.add(UserMessage(f"Message {i} " + "content " * 50))
        agent.history.add(AssistantMessage(content=f"Reply {i} " + "details " * 50))

    initial_len = len(agent.history)
    assert initial_len >= 20

    # Запускаем принудительное сжатие
    summary_mock = AssistantMessage(content="Summary of previous discussion")
    with mock.patch("universal_agents.turn_executor.TurnExecutor.service_llm_call", return_value=(summary_mock, None, None)):
        compacted = agent._auto_summarize_dialogue(force=True)

    assert compacted is True
    # Проверяем, что в начале теперь саммари, а хвост сохранен
    assert len(agent.history) < initial_len
    roles = [m.to_api_dict()["role"] for m in agent.history.get_all()]
    assert roles[0] == "system"


def test_milestone_compaction():
    agent = make_agent(tools_config=["double_me"], external_plugins={"double_me": double_me})

    # Симулируем предыдущую работу
    agent.history.add(UserMessage("Do step 1"))
    agent.history.add(AssistantMessage(content="Doing step 1", tool_calls=[ToolCall(id="1", name="double_me", arguments='{"value": 1}')]))
    agent.history.add(UserMessage("Do step 2"))
    agent.history.add(AssistantMessage(content="Doing step 2"))

    len_before = len(agent.history)

    # Завершаем этап через complete_milestone
    result = complete_milestone(
        agent,
        stage_name="Initial Setup",
        summary_of_work="Completed step 1 and step 2 successfully",
        next_goals="Proceed to testing",
    )

    assert "Initial Setup" in result
    assert "compacted" in result.lower()
    # История уплотнена
    assert len(agent.history) <= len_before
    # Последнее сообщение содержит саммари вехи
    last_msg = agent.history.get_all()[-1]
    assert "Initial Setup" in last_msg.content
    assert "Completed step 1 and step 2" in last_msg.content


def test_kv_cache_prefix_stability():
    agent = make_agent()
    agent.history.add(UserMessage("turn 1"))
    agent.history.add(AssistantMessage(content="answer 1"))

    # Сохраняем представление префикса для API
    api_msgs_1 = [m.to_api_dict() for m in agent.history.get_all()]

    # Следующий ход
    agent.history.add(UserMessage("turn 2"))
    agent.history.add(AssistantMessage(content="answer 2"))
    api_msgs_2 = [m.to_api_dict() for m in agent.history.get_all()]

    # Префикс (первые 3 сообщения) должен оставаться побайтово идентичным
    for i in range(len(api_msgs_1)):
        assert api_msgs_1[i] == api_msgs_2[i]


def test_user_header_budget_cache_policy():
    # Инвариант §1.2: динамическая шапка с остатком бюджета контекста пишется только последнему UserMessage,
    # а у предыдущих сообщений кэшируется, чтобы не портить KV-cache.
    agent = make_agent()
    agent.history.add(UserMessage("First user question"))
    agent.history.add(AssistantMessage(content="First answer"))
    agent.history.add(UserMessage("Second user question"))

    msgs = prepare_messages_for_api(agent)
    # Первое сообщение юзера получает фиксированный snapshot шапки
    first_user_content = msgs[1]["content"]
    assert "First user question" in first_user_content

    # Меняем расход токенов
    agent.token_tracker.update_from_usage({"prompt_tokens": 80_000, "total_tokens": 80_000})
    msgs_after = prepare_messages_for_api(agent)

    # Шапка первого сообщения юзера не изменилась ни на один байт!
    assert msgs_after[1]["content"] == first_user_content


def test_history_auto_repair_consecutive_users_and_hanging_tools():
    # Инвариант §0 и §1.7: авто-ремонт истории
    agent = make_agent(tools_config=["double_me"], external_plugins={"double_me": double_me})

    # 1. Два подряд UserMessage: prepare_turn должен вставить пустой AssistantMessage между ними
    agent.history.add(UserMessage("first question"))
    agent._prepare_turn("second question")

    roles = [m.to_api_dict()["role"] for m in agent.history.get_all()]
    assert roles == ["system", "user", "assistant", "user"]
    stub = agent.history.get_all()[2]
    assert isinstance(stub, AssistantMessage)
    assert stub.content == ""

    # 2. Зависший вызов тула без ToolResult срезается
    agent2 = make_agent()
    agent2.history.add(UserMessage("question"))
    agent2.history.add(AssistantMessage(
        content="executing",
        tool_calls=[ToolCall(id="t_hanging", name="read", arguments='{}')],
    ))
    agent2._prepare_turn("new turn")

    roles2 = [m.to_api_dict()["role"] for m in agent2.history.get_all()]
    assert roles2 == ["system", "user", "assistant", "user"]
    msgs = agent2.history.get_all()
    # У первого ассистента срезан зависший tool_call
    assert not msgs[1].has_tool_calls() if hasattr(msgs[1], "has_tool_calls") else True


def test_persistence_save_and_load_roundtrip(temp_dir):
    save_file = os.path.join(temp_dir, "saved_dialog.json")
    agent = make_agent(tools_config=["double_me"], external_plugins={"double_me": double_me})

    agent.history.add(UserMessage("Remember 42"))
    agent.history.add(AssistantMessage(content="Saved 42"))
    set_plan(agent, "1. Think\n2. Act")

    # Сохранение истории
    agent.save_history(save_file)
    assert os.path.exists(save_file)

    # Загрузка в нового агента
    new_agent = make_agent()
    loaded_tools, _, _ = new_agent.history.load(save_file)

    assert len(new_agent.history) == len(agent.history)
    assert new_agent.history.get_all()[1].content == "Remember 42"
    assert new_agent.history.get_all()[2].content == "Saved 42"


def test_compaction_kv_cache_prefix_stability_and_clean_headers():
    agent = make_agent(tools_config=["double_me"], external_plugins={"double_me": double_me})

    for i in range(1, 4):
        agent.history.add(UserMessage(f"User turn {i}"))
        agent.history.add(AssistantMessage(content=f"Assistant answer {i}"))

    summary_content = (
        "* 1.1 Context: test\n"
        "* 1.2 Terms: none\n"
        "* 1.3 Main task: checklist\n"
        "* 1.4 Constraints: none\n"
        "* 1.5 Preferences: none\n"
        "* 2.1 Completed: did X\n"
        "* 2.2 Next steps: step Y\n"
        "* 2.3 Decisions: decision Z\n"
        "* 3.1 Pending: none\n"
        "* 4.1 Registry: [C1] fact\n"
        "* 5.1 Metadata: test\n"
        "* 6.1 Resources: test.py\n"
        "* 7.1 Hints: none\n"
        "* 7.2 Other: none"
    )
    summary_mock = AssistantMessage(content=summary_content)
    with mock.patch("universal_agents.turn_executor.TurnExecutor.service_llm_call", return_value=(summary_mock, None, None)):
        compacted = agent._auto_summarize_dialogue(force=True)

    assert compacted is True

    api_msgs = prepare_messages_for_api(agent)
    assert len(api_msgs) >= 3
    assert api_msgs[0]["role"] == "system"

    # summary_msg at index 1 is auxiliary: no volatile timestamp or memory budget
    summary_msg = api_msgs[1]
    assert summary_msg["role"] == "user"
    assert "PAST-SUMMARY" in summary_msg["content"]
    assert "Memory:" not in summary_msg["content"]
    assert "{" not in summary_msg["content"][:10]

    # Subsequent turn retains identical prefix
    agent._prepare_turn("User turn 4")
    api_msgs_turn4 = prepare_messages_for_api(agent)
    assert api_msgs_turn4[0] == api_msgs[0]
    assert api_msgs_turn4[1] == api_msgs[1]
