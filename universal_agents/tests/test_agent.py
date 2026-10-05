import unittest
from types import SimpleNamespace
from unittest import mock

from universal_agents.agent import LLMAgent, MAX_CONSECUTIVE_ERRORS, _REPEAT_ANSWER_NAG
from universal_agents.config import Config
from universal_agents.constants import INTERRUPT_HEADER, err
from universal_agents.llm_client import text_hash
from universal_agents.models import AssistantMessage, ToolCall, ToolResult, UserMessage
from universal_agents.tool import tool

from tests.conftest import double_me


class TestAgentTurnLoop(unittest.TestCase):
    def test_chat_returns_plain_answer_to_system(self):
        agent = LLMAgent(system_prompt="sys")
        fake = AssistantMessage(content="hello back")
        with mock.patch("universal_agents.agent.LLMClient.call", return_value=(fake, None, None)):
            result = agent.chat("hello")
        self.assertIn("hello back", result)
        roles = [m.to_api_dict()["role"] for m in agent.history]
        self.assertEqual(roles, ["system", "user", "assistant"])

    def test_inject_user_interrupt_adds_header_and_generates(self):
        agent = LLMAgent(system_prompt="sys")
        fake = AssistantMessage(content="redirected answer")
        with mock.patch("universal_agents.agent.LLMClient.call", return_value=(fake, None, None)):
            result = agent.inject_user_interrupt("new direction")
        self.assertIn("redirected answer", result)
        user_msgs = [m.content for m in agent.history.get_all() if isinstance(m, UserMessage)]
        self.assertTrue(any(INTERRUPT_HEADER in c for c in user_msgs))
        self.assertTrue(any("new direction" in c for c in user_msgs))
        roles = [m.to_api_dict()["role"] for m in agent.history]
        self.assertEqual(roles, ["system", "user", "assistant"])

    def test_inject_user_interrupt_appends_to_tool_result(self):
        agent = LLMAgent(system_prompt="sys")
        agent.history.add(UserMessage("comp"))
        agent.history.add(AssistantMessage(
            content="calling",
            tool_calls=[ToolCall(id="t1", name="double_me", arguments='{"value": 2}')],
        ))
        tr = ToolResult(tool_call_id="t1", name="double_me", content="4")
        agent.history.add(tr)
        fake = AssistantMessage(content="ok")
        with mock.patch("universal_agents.agent.LLMClient.call", return_value=(fake, None, None)):
            agent.inject_user_interrupt("stop, instead do X")
        # шапка и текст пользователя дописаны в конец вывода инструмента
        self.assertIn(INTERRUPT_HEADER, tr.content)
        self.assertIn("stop, instead do X", tr.content)
        # отдельного user-сообщения с текстом прерывания нет, пустой заглушки ассистента нет
        self.assertFalse(
            any(isinstance(m, UserMessage) and "stop, instead do X" in m.content for m in agent.history.get_all())
        )
        self.assertFalse(
            any(
                isinstance(m, AssistantMessage) and m.content == "" and not m.has_tool_calls()
                for m in agent.history.get_all()
            )
        )
        roles = [m.to_api_dict()["role"] for m in agent.history]
        self.assertEqual(roles, ["system", "user", "assistant", "tool", "assistant"])

    def test_pending_interrupt_stops_turn_without_llm_call(self):
        agent = LLMAgent(system_prompt="sys")
        agent.pending_interrupt = "user typed during tool execution"
        with mock.patch("universal_agents.agent.LLMClient.call") as mocked:
            result = agent._run_turn_loop(5)
        mocked.assert_not_called()
        self.assertEqual(result, "")
        self.assertEqual(agent.pending_interrupt, "user typed during tool execution")

    def test_prepare_turn_inserts_stub_between_consecutive_user_messages(self):
        agent = LLMAgent(system_prompt="sys")
        agent.history.add(UserMessage("first"))
        agent._prepare_turn("second")
        roles = [m.to_api_dict()["role"] for m in agent.history]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])
        stub = agent.history.get_all()[2]
        self.assertIsInstance(stub, AssistantMessage)
        self.assertEqual(stub.content, "")

    def test_prepare_turn_cleans_hanging_tool_call(self):
        agent = LLMAgent(system_prompt="sys")
        agent.history.add(UserMessage("comp"))
        agent.history.add(AssistantMessage(
            content="calling",
            tool_calls=[ToolCall(id="t1", name="double_me", arguments='{"value": 2}')],
        ))
        # висящий вызов без результата — должен быть удалён при вставке нового сообщения
        agent._prepare_turn("redirect")
        roles = [m.to_api_dict()["role"] for m in agent.history]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])
        msgs = agent.history.get_all()
        self.assertFalse(msgs[1].has_tool_calls() if hasattr(msgs[1], 'has_tool_calls') else False)
        stub = msgs[2]
        self.assertIsInstance(stub, AssistantMessage)
        self.assertEqual(stub.content, "")

    def test_chat_executes_tool_and_finishes(self):
        agent = LLMAgent(
            system_prompt="sys",
            tools_config=["double_me"],
            external_plugins={"double_me": double_me},
        )
        first = AssistantMessage(content="Let me compute 21 * 2", tool_calls=[ToolCall(id="t1", name="double_me", arguments='{"value": 21}')])
        second = AssistantMessage(content="final answer 42")
        with mock.patch(
            "universal_agents.agent.LLMClient.call",
            side_effect=[(first, None, None), (second, None, None)],
        ):
            result = agent.chat("compute", max_iter=5)
        self.assertIn("final answer 42", result)
        roles = [m.to_api_dict()["role"] for m in agent.history]
        self.assertIn("tool", roles)
        self.assertIn("assistant", roles)

    def test_broken_call_triggers_regen(self):
        agent = LLMAgent(system_prompt="sys", max_generation_attempts=3)
        broken = AssistantMessage(content="Please use <tool_call>read</tool_call>")
        fixed = AssistantMessage(content="ok done")
        with mock.patch(
            "universal_agents.agent.LLMClient.call",
            side_effect=[(broken, None, None), (fixed, None, None)],
        ):
            result = agent.chat("do x")
        self.assertEqual(result, "ok done")
        # сломанное сообщение стёрто из истории, финальный ответ остался
        last = agent.history.get_last_message()
        self.assertEqual(last.content, "ok done")


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
            external_plugins={"double_me": double_me},
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
        fresh = AssistantMessage(content='LLM:\n"Search for the config."')
        with mock.patch(
            "universal_agents.agent.LLMClient.call",
            return_value=(fresh, None, None),
        ):
            result = agent.chat("third")
        self.assertEqual(result, 'LLM:\n"Search for the config."')



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
            self.assertNotIn(_REPEAT_ANSWER_NAG, msgs[-1][1])

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

        # до порога дублей подряд — без NAG, с порога — NAG в контексте (в последнем сообщении)
        def _has_nag(msgs):
            return _REPEAT_ANSWER_NAG in msgs[-1][1]

        for i in range(Config.DUPLICATE_NAG_THRESHOLD - 1):
            self.assertFalse(_has_nag(captured[i]))
        for i in range(Config.DUPLICATE_NAG_THRESHOLD, len(captured)):
            self.assertTrue(_has_nag(captured[i]))

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

    def test_duplicate_retry_keeps_prefix_byte_identical(self):
        """Регрессия поломки KV-кэша: перегенерация после DUPLICATE ANSWER DETECTED
        должна уходить с байт-идентичным префиксом (модель не перечитывает контекст
        с нуля). Раньше watch-достройка прерывала стрим на расхождении и достраивала
        ответ по partial — сервер кэшировал дубль как «префикс» и сломал переиспользование."""
        agent = LLMAgent(
            system_prompt="sys",
            max_generation_attempts=3,
            autosave_enabled=False,
            streaming_enabled=True,
            on_stream_chunk=lambda _: None,
        )
        agent.history.add(UserMessage("q"))
        agent.history.add(AssistantMessage(content="same answer text"))

        streamed: list[str] = []

        def fake_stream(messages, **kwargs):
            # каждое повторение ответа стримится целиком — никакого прерывания по расхождению
            content = "same answer text" if len(streamed) == 0 else "a different answer"
            streamed.append(content)
            for piece in [content[:10], content[10:20], content[20:]]:
                if piece:
                    delta = SimpleNamespace(content=piece, tool_calls=None, reasoning_content=None)
                    yield SimpleNamespace(choices=[SimpleNamespace(delta=delta)])

        with mock.patch("universal_agents.agent.LLMClient.stream", side_effect=fake_stream):
            result = agent.chat("q2")
        self.assertEqual(result, "a different answer")
        # стрим открывался на каждую попытку (дубль + свежий ответ), дубль отброшен без достройки
        self.assertEqual(streamed, ["same answer text", "a different answer"])


def _error_tool_call(name: str, args: str) -> AssistantMessage:
    return AssistantMessage(content="", tool_calls=[ToolCall(id="c1", name=name, arguments=args)])


@tool(description="always fails")
def _failing_tool(message: str) -> str:
    return err(f": {message}")


def _make_failing_agent(on_system_msg):
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
        agent = _make_failing_agent(calls.append)
        agent._thinking_enabled = True
        fail = _error_tool_call(_failing_tool.__name__, '{"message": "boom"}')
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
        agent = _make_failing_agent(calls.append)
        agent._thinking_enabled = True
        fail = _error_tool_call(_failing_tool.__name__, '{"message": "boom"}')
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
        agent = _make_failing_agent(calls.append)
        agent._thinking_enabled = True
        # каждый префиксный проход возвращает новую ошибку — постоянно «диагностирует»
        responses = [
            (_error_tool_call(_failing_tool.__name__, '{"message": "a"}'), None, None),
            (_error_tool_call(_failing_tool.__name__, '{"message": "b"}'), None, None),
            (_error_tool_call(_failing_tool.__name__, '{"message": "c"}'), None, None),
            (_error_tool_call(_failing_tool.__name__, '{"message": "d"}'), None, None),
            (_error_tool_call(_failing_tool.__name__, '{"message": "e"}'), None, None),
            (_error_tool_call(_failing_tool.__name__, '{"message": "f"}'), None, None),
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
        agent = _make_failing_agent(calls.append)
        agent._thinking_enabled = True
        boom = _error_tool_call(_failing_tool.__name__, '{"message": "boom"}')
        glitch = _error_tool_call(_failing_tool.__name__, '{"message": "glitch"}')
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
        agent = _make_failing_agent(lambda x: None)
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
        agent = _make_failing_agent(lambda x: None)
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
        for _ in range(MAX_CONSECUTIVE_ERRORS):
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
        agent = _make_failing_agent(calls.append)
        agent._thinking_enabled = True
        sa = _error_tool_call(_failing_tool.__name__, '{"message": "a"}')
        sb = _error_tool_call(_failing_tool.__name__, '{"message": "b"}')
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
        agent = _make_failing_agent(calls.append)
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
        fail = _error_tool_call(_failing_tool.__name__, '{"message": "boom"}')
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
        agent = _make_failing_agent(calls.append)
        agent._thinking_enabled = True
        fail = _error_tool_call(_failing_tool.__name__, '{"message": "boom"}')
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
