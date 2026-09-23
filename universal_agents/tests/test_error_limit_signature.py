"""Тесты лимита ошибок инструментов (§1.10).

Лимит (MAX_CONSECUTIVE_ERRORS) срабатывает, когда ЛЮБАЯ сигнатура сбоя
(инструмент + аргументы + текст ошибки) повторилась за ход MAX раз — независимо
от того, шли повторы подряд или чередовались с другими сбоями ([A,B,A,B,...] —
тоже зацикливание, раньше такой цикл шёл бесконечно). Разные сбои по отдельности
лимита не дают, пока ни одна сигнатура не набрала MAX (модель диагностирует).
Счётчики сбрасываются на успешном инструменте и при сжатии истории.
"""

from __future__ import annotations

import unittest
from unittest import mock

from universal_agents.agent import LLMAgent, MAX_CONSECUTIVE_ERRORS
from universal_agents.constants import err
from universal_agents.models import AssistantMessage, ToolCall, ToolResult
from universal_agents.tool import tool


def _tool_call(name: str, args: str) -> AssistantMessage:
    return AssistantMessage(content="", tool_calls=[ToolCall(id="c1", name=name, arguments=args)])


@tool(description="always fails")
def _failing_tool(message: str) -> str:
    return err(f": {message}")


def _make_agent(on_system_msg):
    return LLMAgent(
        system_prompt="sys",
        tools_config=[_failing_tool.__name__],
        external_plugins={_failing_tool.__name__: _failing_tool},
        disable_per_msg_summarization=True,
        autosave_enabled=False,
        on_system_msg=on_system_msg,
    )


class TestIdenticalErrorLimit(unittest.TestCase):
    """Одинаковые инструмент + аргументы + текст ошибки подряд — это зацикливание,
    лимит достигается, ход сдаётся пользователю (прежнее поведение сохраняется)."""

    def test_identical_errors_reach_limit(self):
        calls = []
        agent = _make_agent(calls.append)
        agent._thinking_enabled = True
        fail = _tool_call(_failing_tool.__name__, '{"message": "boom"}')
        # ровно MAX_CONSECUTIVE_ERRORS одинаковых ошибок — на последней лимит срабатывает
        with mock.patch(
            "universal_agents.agent.LLMClient.call",
            return_value=(fail, None, None),
        ):
            result = agent.chat("сделай что-нибудь", max_iter=10)
        self.assertEqual(result, "")
        self.assertTrue(any("LIMIT REACHED" in m for m in calls))
        self.assertTrue(any(f"{MAX_CONSECUTIVE_ERRORS} consecutive tool errors" in m for m in calls))

    def test_fewer_identical_errors_do_not_reach_limit(self):
        calls = []
        agent = _make_agent(calls.append)
        agent._thinking_enabled = True
        fail = _tool_call(_failing_tool.__name__, '{"message": "boom"}')
        final = AssistantMessage(content="сдаюсь")
        responses = [
            (fail, None, None),
            (fail, None, None),
            (fail, None, None),
            (final, None, None),
        ]
        with mock.patch(
            "universal_agents.agent.LLMClient.call",
            side_effect=responses,
        ):
            result = agent.chat("сделай что-нибудь", max_iter=10)
        # меньше порога — не прерываем, ход завершается нормальным ответом
        self.assertEqual(result, "сдаюсь")
        self.assertFalse(any("LIMIT REACHED" in m for m in calls))


class TestVaryingErrorNoLimit(unittest.TestCase):
    """Разные ошибки (другой текст, другие аргументы) — это диагностика, а не
    зацикливание: даже больше порога ошибок подряд лимит НЕ достигается."""

    def test_varying_error_text_does_not_reach_limit(self):
        calls = []
        agent = _make_agent(calls.append)
        agent._thinking_enabled = True
        # каждый префиксный проход возвращает новую ошибку — постоянно «диагностирует»
        responses = [
            (_tool_call(_failing_tool.__name__, '{"message": "a"}'), None, None),
            (_tool_call(_failing_tool.__name__, '{"message": "b"}'), None, None),
            (_tool_call(_failing_tool.__name__, '{"message": "c"}'), None, None),
            (_tool_call(_failing_tool.__name__, '{"message": "d"}'), None, None),
            (_tool_call(_failing_tool.__name__, '{"message": "e"}'), None, None),
            (_tool_call(_failing_tool.__name__, '{"message": "f"}'), None, None),
            (AssistantMessage(content="наконец-то"), None, None),
        ]
        with mock.patch(
            "universal_agents.agent.LLMClient.call",
            side_effect=responses,
        ):
            result = agent.chat("сделай что-нибудь", max_iter=20)
        self.assertEqual(result, "наконец-то")
        self.assertFalse(any("LIMIT REACHED" in m for m in calls))

    def test_one_different_error_resets_series(self):
        """Если среди одинаковых ошибок вклинилась другая — серия начинается заново:
        одиночные совпадения по разные стороны разрыва в лимит не складываются."""
        calls = []
        agent = _make_agent(calls.append)
        agent._thinking_enabled = True
        boom = _tool_call(_failing_tool.__name__, '{"message": "boom"}')
        glitch = _tool_call(_failing_tool.__name__, '{"message": "glitch"}')
        # 3 одинаковых (серия=3) → 1 другая (сброс) → ещё 3 одинаковых (серия=3)
        # суммарно 3+1+3=7 ошибок, но ни одна серия не достигает порога 5
        responses = [(boom, None, None)] * 3 + [(glitch, None, None)] + [(boom, None, None)] * 3
        responses.append((AssistantMessage(content="итог"), None, None))
        with mock.patch(
            "universal_agents.agent.LLMClient.call",
            side_effect=responses,
        ):
            result = agent.chat("сделай что-нибудь", max_iter=20)
        self.assertEqual(result, "итог")
        self.assertFalse(any("LIMIT REACHED" in m for m in calls))


class TestSignatureDerivation(unittest.TestCase):
    """Юнит-проверка сигнатуры ошибки: одинаковые вызовы дают одинаковую сигнатуру,
    отличаться должен любой из трёх компонентов (инструмент/аргументы/ошибка)."""

    def _signature(self, name: str, args: str, error_content: str) -> str:
        agent = _make_agent(lambda x: None)
        call = AssistantMessage(
            content="",
            tool_calls=[ToolCall(id="t1", name=name, arguments=args)],
        )
        agent.history.add(call)
        agent.history.add(ToolResult.error("t1", name, error_content))
        return agent._last_failed_tool_signature()

    def test_same_failure_same_signature(self):
        s1 = self._signature(_failing_tool.__name__, '{"message": "boom"}', "boom")
        s2 = self._signature(_failing_tool.__name__, '{"message": "boom"}', "boom")
        self.assertEqual(s1, s2)
        self.assertIsNotNone(s1)

    def test_different_error_text_differs(self):
        s1 = self._signature(_failing_tool.__name__, '{"message": "boom"}', "boom")
        s2 = self._signature(_failing_tool.__name__, '{"message": "boom"}', "kaput")
        self.assertNotEqual(s1, s2)

    def test_different_args_differs(self):
        s1 = self._signature(_failing_tool.__name__, '{"message": "boom"}', "boom")
        s2 = self._signature(_failing_tool.__name__, '{"message": "other"}', "boom")
        self.assertNotEqual(s1, s2)

    def test_no_failed_tool_call_returns_none(self):
        agent = _make_agent(lambda x: None)
        agent.history.add(AssistantMessage(content="просто текст"))
        self.assertIsNone(agent._last_failed_tool_signature())


class TestTurnStateSignature(unittest.TestCase):
    """Поведение счётчика: счёт по-сигнатурный. Каждая сигнатура растёт независимо —
    чередование [A,B,A,B,...] тоже зацикливание и даёт лимит по любой из них."""

    def test_series_counts_only_identical(self):
        from universal_agents.agent import TurnState

        state = TurnState()
        self.assertFalse(state.max_errors_reached)
        sig = "tool|args|errhash"
        for _ in range(MAX_CONSECUTIVE_ERRORS - 1):
            state.record_tool_error(sig)
            self.assertFalse(state.max_errors_reached)
        state.record_tool_error(sig)
        self.assertTrue(state.max_errors_reached)

    def test_alternating_signatures_accumulate_independently(self):
        """[A,B,A,B,...]: A и B растут по отдельности — чередование НЕ сбрасывает.
        До достижения одной из них MAX лимит не наступает."""
        from universal_agents.agent import TurnState

        state = TurnState()
        for i in range(2 * MAX_CONSECUTIVE_ERRORS - 2):
            state.record_tool_error("A" if i % 2 == 0 else "B")
        # каждая сигнатура вернулась MAX-1 раз, лимита ещё нет
        self.assertEqual(state.error_counts["A"], MAX_CONSECUTIVE_ERRORS - 1)
        self.assertEqual(state.error_counts["B"], MAX_CONSECUTIVE_ERRORS - 1)
        self.assertFalse(state.max_errors_reached)

    def test_alternating_signatures_reach_limit_at_2x(self):
        """Половина повторений на сигнатуру — порог вдвое больше, но он достижим:
        когда ЛЮБАЯ из сигнатур добирает MAX, ход прерывается.

        (Это тот случай, что раньше уходил в бесконечный цикл: серия «сбрасывалась»
        на каждой смене сигнатуры и лимит никогда не наступал.)"""
        from universal_agents.agent import TurnState

        state = TurnState()
        for i in range(2 * MAX_CONSECUTIVE_ERRORS - 1):
            state.record_tool_error("A" if i % 2 == 0 else "B")
        # (2*MAX-1)-я запись — нечётная позиция (B), A уже набрала MAX
        self.assertEqual(state.error_counts["A"], MAX_CONSECUTIVE_ERRORS)
        self.assertEqual(state.error_counts["B"], MAX_CONSECUTIVE_ERRORS - 1)
        self.assertTrue(state.max_errors_reached)

    def test_one_different_error_does_not_reset_accumulated_count(self):
        """Другая сигнатура между повторами одной и той же — не сброс: модель могла
        диагностировать, но точный сбой A всё равно повторился MAX раз."""
        from universal_agents.agent import TurnState

        state = TurnState()
        for _ in range(MAX_CONSECUTIVE_ERRORS - 1):
            state.record_tool_error("A")
            state.record_tool_error("B")  # диагностика между повторами A
        state.record_tool_error("A")      # A набрала MAX
        self.assertTrue(state.max_errors_reached)

    def test_no_tool_error_signature_is_stable_bucket(self):
        from universal_agents.agent import TurnState

        state = TurnState()
        for _ in range(MAX_CONSECUTIVE_ERRORS):
            state.record_tool_error(None)  # пустой ответ без инструмента
        self.assertTrue(state.max_errors_reached)

    def test_success_resets_all_counts(self):
        from universal_agents.agent import TurnState

        state = TurnState()
        state.record_tool_error("A")
        state.record_tool_error("A")
        state.record_tool_success()
        state.record_tool_error("A")
        self.assertEqual(state.error_counts["A"], 1)
        self.assertFalse(state.max_errors_reached)

    def test_reset_error_counts_clears_everything(self):
        from universal_agents.agent import TurnState

        state = TurnState()
        state.record_tool_error("A")
        state.record_tool_error("A")
        state.record_tool_error("B")
        state.reset_error_counts()  # история сжата — старые падения в контексте больше нет
        self.assertEqual(state.error_counts, {})
        self.assertFalse(state.max_errors_reached)


class TestAlternatingErrorsReachLimit(unittest.TestCase):
    """Чередующиеся разные сигнатуры [A,B,A,B,...] — реальное зацикливание: раньше
    серия сбрасывалась на каждой смене и цикл шёл бесконечно, теперь ЛЮБАЯ сигнатура,
    набравшая MAX_CONSECUTIVE_ERRORS, прерывает ход."""

    def test_alternating_failures_are_interrupted(self):
        calls = []
        agent = _make_agent(calls.append)
        agent._thinking_enabled = True
        sa = _tool_call(_failing_tool.__name__, '{"message": "a"}')
        sb = _tool_call(_failing_tool.__name__, '{"message": "b"}')
        # A,B повторяются 2*MAX раз: каждая сигнатура достигает MAX → лимит срабатывает
        responses = [(sa, None, None), (sb, None, None)] * (2 * MAX_CONSECUTIVE_ERRORS)
        responses.append((AssistantMessage(content="недостижимо"), None, None))
        with mock.patch(
            "universal_agents.agent.LLMClient.call",
            side_effect=responses,
        ):
            result = agent.chat("сделай что-нибудь", max_iter=50)
        self.assertEqual(result, "")
        self.assertTrue(any("LIMIT REACHED" in m for m in calls))


class TestCompressionResetsCounters(unittest.TestCase):
    """Сжатие истории убирает старые ошибки из контекста — счётчики повторов (§1.10)
    сбрасываются, отсчёт зацикливания идёт по актуальной картине."""

    def _agent_compressing_once(self):
        calls = []
        agent = _make_agent(calls.append)
        agent._thinking_enabled = True
        state = {"usage": 0, "compressed": 0}

        def fake_usage() -> float:
            # порог переходим только на 4-й итерации (после 3 ошибок)
            state["usage"] += 1
            return 100.0 if state["usage"] == 4 else 0.0

        def fake_compress():
            if state["compressed"] == 0:
                state["compressed"] += 1
                agent.history.compress_old_messages("сжатая история", preserve_last=2)
                agent.history.normalize()
                agent._on_history_changed()
                return True
            return False

        mock.patch.object(agent, "_get_context_usage_percent", side_effect=fake_usage).start()
        mock.patch.object(agent, "_auto_summarize_dialogue", side_effect=fake_compress).start()
        self.addCleanup(mock.patch.stopall)
        return agent

    def test_compression_resets_accumulated_counts(self):
        calls = []
        agent = self._agent_compressing_once()
        # фиксируем callback системных сообщений, чтобы проверить отсутствие LIMIT
        agent.on_system_msg = calls.append
        fail = _tool_call(_failing_tool.__name__, '{"message": "boom"}')
        # 6 одинаковых ошибок + успех. Без сброса лимит наступил бы на 5-й.
        # Компакция на 4-й итерации стирает старое — к моменту успеха ни одна
        # сигнатура не набрала MAX.
        responses = [(fail, None, None)] * 6
        responses.append((AssistantMessage(content="итог"), None, None))
        with mock.patch(
            "universal_agents.agent.LLMClient.call",
            side_effect=responses,
        ):
            result = agent.chat("сделай что-нибудь", max_iter=20)
        self.assertEqual(result, "итог")
        self.assertFalse(any("LIMIT REACHED" in m for m in calls))

    def test_compression_does_not_reset_without_compaction(self):
        """Если сжатия фактически не было (низкая занятость контекста) — счётчики
        не трогаются и серия из MAX одинаковых ошибок всё равно прерывается."""
        calls = []
        agent = _make_agent(calls.append)
        agent._thinking_enabled = True
        fail = _tool_call(_failing_tool.__name__, '{"message": "boom"}')
        responses = [(fail, None, None)] * MAX_CONSECUTIVE_ERRORS
        responses.append((AssistantMessage(content="недостижимо"), None, None))
        with mock.patch(
            "universal_agents.agent.LLMClient.call",
            side_effect=responses,
        ):
            result = agent.chat("сделай что-нибудь", max_iter=20)
        self.assertEqual(result, "")
        self.assertTrue(any("LIMIT REACHED" in m for m in calls))


if __name__ == "__main__":
    unittest.main()