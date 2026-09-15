"""Тесты детектора повторов текста/reasoning по ВСЕЙ истории.

Проверяет новые возможности _detect_duplicate / _get_prior_text_hashes:
- точное совпадение текста ответа или reasoning-блока где угодно в истории
  (не только у предыдущего сообщения) → отброс ответа и регенерация;
- общая приставка (prefill) НЕ является ложным срабатыванием.
"""

from __future__ import annotations

import unittest
from unittest import mock

from universal_agents.agent import LLMAgent
from universal_agents.models import AssistantMessage, ToolCall, ToolResult, UserMessage
from universal_agents.llm_client import text_hash
from universal_agents.tool import tool


class TestTextHash(unittest.TestCase):
    def test_strips_whitespace(self):
        self.assertEqual(text_hash("  hello  "), text_hash("hello"))

    def test_different_texts_different_hashes(self):
        self.assertNotEqual(text_hash("abc"), text_hash("def"))


class TestPriorTextHashes(unittest.TestCase):
    def test_empty_history(self):
        agent = LLMAgent(system_prompt="sys")
        t, r = agent._get_prior_text_hashes()
        self.assertEqual(t, set())
        self.assertEqual(r, set())

    def test_includes_only_assistant_messages(self):
        agent = LLMAgent(system_prompt="sys")
        agent.history.add(UserMessage("q"))
        agent.history.add(AssistantMessage(content="ans", reasoning_content="think"))
        agent.history.add(ToolResult.success("t1", "read", "ok"))
        t, r = agent._get_prior_text_hashes()
        self.assertEqual(t, {text_hash("ans")})
        self.assertEqual(r, {text_hash("think")})

    def test_ignores_empty_content_and_reasoning(self):
        agent = LLMAgent(system_prompt="sys")
        agent.history.add(UserMessage("q"))
        agent.history.add(AssistantMessage(content="", reasoning_content=""))
        t, r = agent._get_prior_text_hashes()
        self.assertEqual(t, set())
        self.assertEqual(r, set())


class TestDuplicateTextAcrossHistory(unittest.TestCase):
    """Сценарий пользователя: модель пишет один и тот же комментарий перед каждым
    вызовом read (с prefill 'LLM:\\n"') — пусть инструменты разные."""

    def test_repeated_comment_with_different_tool_call_triggers_regen(self):
        agent = LLMAgent(
            system_prompt="sys",
            tools_config=["double_me"],
            external_plugins={"double_me": _double_me},
            max_generation_attempts=3,
            autosave_enabled=False,
        )
        old_comment = 'LLM:\n"I will compute the value by doubling it."'
        agent.history.add(UserMessage("first"))
        agent.history.add(AssistantMessage(
            content=old_comment,
            tool_calls=[ToolCall(id="t1", name="double_me", arguments='{"value": 2}')],
        ))
        agent.history.add(ToolResult.success("t1", "double_me", "4"))
        # тот же комментарий, но другой вызов
        dup = AssistantMessage(
            content=old_comment,
            tool_calls=[ToolCall(id="t2", name="double_me", arguments='{"value": 3}')],
        )
        fresh = AssistantMessage(content="6")
        with mock.patch(
            "universal_agents.agent.LLMClient.call",
            side_effect=[(dup, None, None), (fresh, None, None)],
        ):
            result = agent.chat("more")
        self.assertEqual(result, "6")

    def test_repeated_reasoning_across_history_triggers_regen(self):
        agent = LLMAgent(
            system_prompt="sys",
            max_generation_attempts=3,
            autosave_enabled=False,
        )
        old_reasoning = "I should double-check the value before answering."
        agent.history.add(UserMessage("first"))
        agent.history.add(AssistantMessage(content="something", reasoning_content=old_reasoning))
        agent.history.add(UserMessage("second"))
        agent.history.add(AssistantMessage(content="other", reasoning_content="different"))
        dup = AssistantMessage(content="entirely new text", reasoning_content=old_reasoning)
        fresh = AssistantMessage(content="final")
        with mock.patch(
            "universal_agents.agent.LLMClient.call",
            side_effect=[(dup, None, None), (fresh, None, None)],
        ):
            result = agent.chat("third")
        self.assertEqual(result, "final")
        # в истории остался только тот old_reasoning, что мы сами подложили в setup
        count = sum(
            1 for m in agent.history.get_all()
            if getattr(m, "reasoning_content", "") == old_reasoning
        )
        self.assertEqual(count, 1)

    def test_different_continuation_same_prefill_not_blocked(self):
        """Одна приставка, другой текст после неё — не ложное срабатывание."""
        agent = LLMAgent(
            system_prompt="sys",
            max_generation_attempts=3,
            autosave_enabled=False,
        )
        agent.history.add(UserMessage("first"))
        agent.history.add(AssistantMessage(content='LLM:\n"Read the file."'))
        agent.history.add(UserMessage("second"))
        fresh = AssistantMessage(content='LLM:\n"Read the file now."')
        with mock.patch(
            "universal_agents.agent.LLMClient.call",
            return_value=(fresh, None, None),
        ):
            result = agent.chat("third")
        self.assertEqual(result, 'LLM:\n"Read the file now."')


_NAG_TEXT = "previous answer was the same to the latest one"


class TestDuplicateEscalation(unittest.TestCase):
    """Эскалация повторов: первые дубли — просто отброс + буст температуры (без NAG),
    после Config.DUPLICATE_NAG_THRESHOLD подряд — вставка NAG в контекст, а при
    исчерпании MAX_LOOP_RETRIES — сдача хода пользователю (дубль НЕ пропускается)."""

    def test_single_duplicate_then_fresh_no_nag(self):
        """Один дубль → отброс + буст, без NAG в запросах; свежий ответ принимается."""
        agent = LLMAgent(
            system_prompt="sys",
            max_generation_attempts=3,
            autosave_enabled=False,
        )
        agent.history.add(UserMessage("q"))
        agent.history.add(AssistantMessage(content="same"))
        agent.history.add(UserMessage("q2"))
        dup = AssistantMessage(content="same")
        fresh = AssistantMessage(content="new")
        captured: list[tuple] = []

        def fake_call(messages, **kwargs):
            captured.append([(m.get("role"), m.get("content", "")) for m in messages])
            if len(captured) == 1:
                return (dup, None, None)
            return (fresh, None, None)

        with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake_call):
            result = agent.chat("q3")
        self.assertEqual(result, "new")
        # NAG не вставлялся ни в один запрос (один дубль ниже порога) — проверяем
        # последнее сообщение, куда NAG вшивается
        for msgs in captured:
            self.assertNotIn(_NAG_TEXT, msgs[-1][1])

    def test_nag_injected_only_after_threshold(self):
        """NAG появляется в контексте только после DUPLICATE_NAG_THRESHOLD дублей."""
        agent = LLMAgent(
            system_prompt="sys",
            max_generation_attempts=4,
            autosave_enabled=False,
        )
        agent.history.add(UserMessage("q"))
        agent.history.add(AssistantMessage(content="same"))
        dup = AssistantMessage(content="same")
        captured: list[tuple] = []

        def fake_call(messages, **kwargs):
            captured.append([(m.get("role"), m.get("content", "")) for m in messages])
            return (dup, None, None)

        with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake_call):
            agent.chat("q2")

        # первые два запроса — без NAG, с порога — NAG в контексте (в последнем сообщении)
        def _has_nag(msgs):
            return _NAG_TEXT in msgs[-1][1]

        self.assertFalse(_has_nag(captured[0]))
        self.assertFalse(_has_nag(captured[1]))
        self.assertTrue(_has_nag(captured[2]))
        self.assertTrue(_has_nag(captured[3]))

    def test_max_retries_hands_control_to_user(self):
        """Дубль, не вылеченный за все попытки, НЕ пропускается: ход отдаётся пользователю."""
        agent = LLMAgent(
            system_prompt="sys",
            max_generation_attempts=2,
            autosave_enabled=False,
        )
        agent.history.add(UserMessage("q"))
        agent.history.add(AssistantMessage(content="same"))
        dup = AssistantMessage(content="same")
        with mock.patch(
            "universal_agents.agent.LLMClient.call",
            return_value=(dup, None, None),
        ):
            result = agent.chat("q2")
        self.assertEqual(result, "")
        # отброшенный дубль ни разу НЕ попал в историю — только наш setup
        count = sum(
            1 for m in agent.history.get_all()
            if isinstance(m, AssistantMessage) and m.content == "same"
        )
        self.assertEqual(count, 1)


@tool(description="double a value")
def _double_me(agent, value: int) -> str:
    return str(value * 2)
