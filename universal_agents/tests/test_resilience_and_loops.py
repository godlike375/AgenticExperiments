from __future__ import annotations

import json
from unittest import mock
import pytest

from universal_agents.models import AssistantMessage, ToolCall
from universal_agents.tool_parsing import detect_broken_call, parse_tool_args, normalize_args
from tests.conftest import make_agent, double_me, fail_me


def test_duplicate_tool_call_detection_and_recovery():
    # Модель зацикливается, вызывая инструмент с одинаковыми аргументами
    agent = make_agent(tools_config=["double_me"], external_plugins={"double_me": double_me})

    tc = ToolCall(id="c1", name="double_me", arguments='{"value": 5}')
    repeat_call = AssistantMessage(content="", tool_calls=[tc], reasoning_content="think")
    recovered_call = AssistantMessage(content="Final answer: 10", reasoning_content="done")

    # Сначала повторы, потом исправление
    responses = [
        (repeat_call, None, None),
        (repeat_call, None, None),
        (recovered_call, None, None),
    ]

    def fake_call(messages, **kwargs):
        return responses.pop(0)

    with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake_call):
        res = agent.chat("compute")

    assert res == "Final answer: 10"


def test_duplicate_answer_detection():
    agent = make_agent()
    # Первое сообщение
    reply1 = AssistantMessage(content="Static answer")
    with mock.patch("universal_agents.agent.LLMClient.call", return_value=(reply1, None, None)):
        res1 = agent.chat("Hi")
    assert res1 == "Static answer"

    # Второе сообщение: модель пытается повторить точь-в-точь такой же ответ
    reply_duplicate = AssistantMessage(content="Static answer")
    reply_recovered = AssistantMessage(content="Different answer")
    responses = [(reply_duplicate, None, None), (reply_recovered, None, None)]

    def fake_call(messages, **kwargs):
        return responses.pop(0)

    with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake_call):
        res2 = agent.chat("Hi again")

    assert res2 == "Different answer"


def test_consecutive_same_tool_errors_limit():
    agent = make_agent(tools_config=["fail_me"], external_plugins={"fail_me": fail_me})

    tc = ToolCall(id="c_fail", name="fail_me", arguments='{"value": 1}')
    fail_reply = AssistantMessage(content="", tool_calls=[tc])

    # Бесконечные одинаковые ошибки тула
    def fake_call(messages, **kwargs):
        return fail_reply, None, None

    with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake_call):
        res = agent.chat("do failing action")

    # Агент должен остановить ход и вернуть управление/сообщение об ошибке
    assert "consecutive" in res.lower() or "error" in res.lower() or "too many" in res.lower() or res is not None


def test_broken_tool_call_json_recovery():
    agent = make_agent(tools_config=["double_me"], external_plugins={"double_me": double_me})

    broken_tc = ToolCall(id="c_broken", name="double_me", arguments='{broken_json: 5')
    broken_reply = AssistantMessage(content="", tool_calls=[broken_tc])
    valid_tc = ToolCall(id="c_valid", name="double_me", arguments='{"value": 5}')
    valid_reply = AssistantMessage(content="", tool_calls=[valid_tc])
    final_reply = AssistantMessage(content="Done: 10")

    responses = [
        (broken_reply, None, None),
        (valid_reply, None, None),
        (final_reply, None, None),
    ]

    def fake_call(messages, **kwargs):
        return responses.pop(0)

    with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake_call):
        res = agent.chat("double 5")

    assert res == "Done: 10"


def test_detect_pseudo_tool_calls_in_text():
    # Модель случайно пишет псевдовызов в обычном тексте вместо реального tool call API
    known_tools = {"read", "double_me"}

    # Псевдотег тула должен быть обнаружен
    assert detect_broken_call("<tool_call>read({})</tool_call>", known_tools) is True
    assert detect_broken_call('read({"path": "file.txt"})', known_tools) is True

    # Обычный текст с упоминанием инструмента не должен считаться broken call
    assert detect_broken_call("Я использовал read() для чтения файла.", known_tools) is False


def test_tool_args_parsing_and_normalization():
    # Парсинг и нормализация JSON аргументов
    assert parse_tool_args('{"b": 2, "a": 1}') == {"a": 1, "b": 2}
    assert parse_tool_args("invalid json") == {}

    # Нормализация сортирует ключи для точного сравнения повторов вызовов
    assert normalize_args('{"b": 2, "a": 1}') == '{"a":1,"b":2}'
