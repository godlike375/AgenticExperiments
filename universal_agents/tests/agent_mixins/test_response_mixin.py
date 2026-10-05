"""Симуляция reasoning при выключенном thinking (Config.SIMULATED_REASONING_*).

Ответ обязан начаться с секции рассуждения <short_think>…</short_think>, а всё, что после
её закрытия, — свободный ответ без тегов. Формат навязывает SimReasoningGroup ОДНИМ
вызовом LLM: prefill открывает секцию, стоп-маркеров нет (модель закрывает секцию сама
и дописывает ответ в том же вызове), а незакрытую секцию закрывает агент.
"""

import unittest
from unittest import mock

from universal_agents.agent import LLMAgent
from universal_agents.agent_mixins.response_mixin import _NO_COMMENT_PREFILL, sim_schema_text
from universal_agents.config import Config
from universal_agents.controllers import XMLStructureController
from universal_agents.models import AssistantMessage, ToolCall, ToolResult, UserMessage

from tests.conftest import double_me, stream_chunk, stream_delta

SIM_SCHEMA = sim_schema_text()
TAG = Config.SIMULATED_REASONING_TAGS[0]


class _Call:
    """Запись одного вызова транспорта (что ушло на сервер)."""

    def __init__(self, prefill, markers, params=None):
        self.prefill = prefill
        self.markers = markers
        self.params = params


def _transport(script):
    """Фейк транспорта: script = [(delta, tool_calls), ...] по вызовам.

    Имитирует реальный транспорт: prefill приклеивается к контенту, стоп-маркеры (их у
    симуляции нет) режут хвост. Возвращает (фейк, список вызовов)."""
    calls = []

    def fake_call(messages, prefill=None, stop_markers=(), **kwargs):
        calls.append(_Call(prefill, stop_markers, kwargs.get("params")))
        delta, tool_calls = script[len(calls) - 1]
        content = (prefill or "") + (delta or "")
        return AssistantMessage(content=content, tool_calls=list(tool_calls or [])), None, None

    return fake_call, calls


def _plain_transport(responses):
    """Фейк транспорта для одиночных ответов (формат выключен): список строк контента."""
    calls = []

    def fake_call(messages, prefill=None, stop_markers=(), **kwargs):
        calls.append(prefill)
        content = responses[len(calls) - 1]
        if prefill and not content.startswith(prefill):
            content = prefill + content
        return AssistantMessage(content=content), None, None

    return fake_call, calls


class SimReasoningTestCase(unittest.TestCase):
    def setUp(self):
        self._prev_sim_settings = {
            name: getattr(Config, name) for name in (
                "SIMULATED_REASONING_ENABLED",
                "SIMULATED_REASONING_REPAIRS",
                "SIMULATED_REASONING_TEMP",
                "SIMULATED_REASONING_TOP_P",
                "SIMULATED_REASONING_MIN_P",
                "SIMULATED_REASONING_FREQUENCY_PENALTY",
                "SIMULATED_REASONING_PRESENCE_PENALTY",
            )
        }
        Config.SIMULATED_REASONING_ENABLED = True
        # Герметичность: тесты исходят из дефолта (однофазный режим), двухфазные
        # включают свои настройки явно. Иначе локальный config.py с заданными
        # SIMULATED_REASONING_* роняет тесты, написанные под один вызов.
        Config.SIMULATED_REASONING_TEMP = None
        Config.SIMULATED_REASONING_TOP_P = None
        Config.SIMULATED_REASONING_MIN_P = None
        Config.SIMULATED_REASONING_FREQUENCY_PENALTY = None
        Config.SIMULATED_REASONING_PRESENCE_PENALTY = None

    def tearDown(self):
        for name, value in self._prev_sim_settings.items():
            setattr(Config, name, value)

    def _agent(self, tools=False, system_msgs=None):
        kwargs = {}
        if tools:
            kwargs = {
                "tools_config": ["double_me"],
                "external_plugins": {"double_me": double_me},
            }
        agent = LLMAgent(
            system_prompt="sys",
            disable_per_msg_summarization=True,
            autosave_enabled=False,
            **kwargs,
        )
        if system_msgs is not None:
            agent.on_system_msg = system_msgs.append
        return agent

    def _streaming_agent(self, chunks, events):
        return LLMAgent(
            system_prompt="sys",
            disable_per_msg_summarization=True,
            autosave_enabled=False,
            streaming_enabled=True,
            on_stream_chunk=lambda c: (chunks.append(c), events.append(("chunk", c))),
            on_stream_start=lambda: events.append(("start", "")),
            on_stream_end=lambda: events.append(("end", "")),
        )

    def _scripted_stream(self, script):
        """Фейк стрима по вызовам: script = [[куски текста], ...]."""
        calls = []

        def stream(*args, **kwargs):
            calls.append(kwargs.get("prefill"))
            for text in script[len(calls) - 1]:
                yield stream_chunk(stream_delta(content=text))

        return stream, calls

    def _assistant_contents(self, agent):
        return [m.content for m in agent.history.get_all() if isinstance(m, AssistantMessage)]

    def _tool_results(self, agent):
        return [m for m in agent.history.get_all() if isinstance(m, ToolResult)]


# ── Схема формата ─────────────────────────────────────────────────────────

class TestSimSchema(unittest.TestCase):
    def test_schema_is_single_section(self):
        self.assertEqual(SIM_SCHEMA, "<short_think/>")
        self.assertEqual(SIM_SCHEMA, "".join(
            f"<{tag}/>" for tag in Config.SIMULATED_REASONING_TAGS
        ))

    def test_controller_generic_api_unchanged(self):
        # Контроллер больше не используется симуляцией, но остаётся общим механизмом.
        ctrl = XMLStructureController.from_schema_text("<root><a/></root>")
        self.assertEqual(ctrl.initial_prefill(), "<root>")
        self.assertEqual(ctrl.markers(), ("</root>", "</a>"))


# ── Один вызов: модель сама закрывает секцию и пишет ответ ────────────────

class TestSingleCallFlow(SimReasoningTestCase):
    def test_self_closed_section_keeps_answer_and_uses_no_stop_markers(self):
        agent = self._agent()
        fake, calls = _transport([("мысли</short_think>42", None)])
        with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake):
            result = agent.chat("сколько будет 21*2?")

        # Один вызов, prefill открывает секцию, стоп-маркеров нет: закрыв секцию,
        # модель дописывает ответ в том же потоке, и мы его не режем.
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].prefill, "<short_think>")
        self.assertEqual(calls[0].markers, ())
        self.assertEqual(result, "<short_think>мысли</short_think>\n42")
        self.assertEqual(self._assistant_contents(agent), [result])

    def test_answer_keeps_text_after_closing_tag(self):
        agent = self._agent()
        fake, _calls = _transport([
            ("мысли</short_think>Готово: 42 — это ответ.", None),
        ])
        with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake):
            result = agent.chat("сколько будет 21*2?")

        self.assertEqual(result, "<short_think>мысли</short_think>\nГотово: 42 — это ответ.")

    def test_unclosed_section_is_closed_by_agent(self):
        agent = self._agent()
        fake, calls = _transport([("мысли", None), ("42", None)])
        # Даже когда секция не закрыта, ответа в первом вызове нет — продолжим.
        with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake):
            result = agent.chat("сколько будет 21*2?")

        self.assertEqual(calls[0].prefill, "<short_think>")
        self.assertIn("<short_think>мысли</short_think>", result)

    def test_multi_paragraph_answer_not_cut(self):
        agent = self._agent()
        answer = "Первый абзац.\n\nВторой абзац.\n- пункт\n- пункт"
        fake, calls = _transport([(f"мысли</short_think>{answer}", None)])
        with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake):
            result = agent.chat("что-то")

        # Стоп-маркеров нет: ничего после закрытия секции не выбрасывается.
        self.assertEqual(calls[0].markers, ())
        self.assertEqual(result, f"<short_think>мысли</short_think>\n{answer}")


# ── Живой вывод: блок один, закрыт один раз, автозакрытие видно ───────────

class TestSimStreaming(SimReasoningTestCase):
    def test_unclosed_section_is_echoed_in_live_output(self):
        chunks = []
        events = []
        agent = self._streaming_agent(chunks, events)
        # Первый вызов: секция не закрыта → автозакрытие + продолжение пустого ответа.
        script = [["мысли"], ["теперь скажу"]]
        stream, calls = self._scripted_stream(script)

        with mock.patch("universal_agents.agent.LLMClient.stream", side_effect=stream), \
             mock.patch("universal_agents.llm_client.LLMClient.close_stream"):
            result = agent.chat("сколько будет 21*2?")

        # Дописанный закрывающий тег виден в консоли ровно один раз, иначе он выглядит
        # незакрытым (или задваивается при автозакрытии в следующем вызове).
        # Перевод строки после закрытия — тоже часть эха.
        self.assertEqual(sum(c.count("</short_think>") for c in chunks), 1)
        self.assertTrue(result.startswith("<short_think>мысли</short_think>\n"))
        self.assertIn("теперь скажу", result)
        self.assertEqual(events[-1][0], "end")
        self.assertEqual(len(calls), 2)
        self.assertEqual(
            calls[1], "<short_think>мысли</short_think>\n" + _NO_COMMENT_PREFILL
        )

    def test_single_call_stream_block_opens_and_closes_once(self):
        chunks = []
        events = []
        agent = self._streaming_agent(chunks, events)
        script = [["мысли</short_think>", "42"]]
        stream, calls = self._scripted_stream(script)

        with mock.patch("universal_agents.agent.LLMClient.stream", side_effect=stream), \
             mock.patch("universal_agents.llm_client.LLMClient.close_stream"):
            result = agent.chat("сколько будет 21*2?")

        kinds = [kind for kind, _ in events]
        self.assertEqual(kinds.count("start"), 1)
        self.assertEqual(kinds.count("end"), 1)
        self.assertEqual(kinds[0], "start")
        self.assertEqual(kinds[-1], "end")
        # Ответ уже выведен дальше тега в том же стриме: перевод строки есть
        # в истории, но в живой вывод задним числом не вставляется.
        self.assertEqual("".join(chunks), "<short_think>мысли</short_think>42")
        self.assertEqual(result, "<short_think>мысли</short_think>\n42")
        self.assertEqual(calls, ["<short_think>"])

    def test_continuation_does_not_reprint_prefill(self):
        chunks = []
        events = []
        agent = self._streaming_agent(chunks, events)
        script = [["мысли</short_think>"], ["теперь скажу"]]
        stream, calls = self._scripted_stream(script)

        with mock.patch("universal_agents.agent.LLMClient.stream", side_effect=stream), \
             mock.patch("universal_agents.llm_client.LLMClient.close_stream"):
            result = agent.chat("сколько будет 21*2?")

        # Накопленный prefill не перепечатывается (открывающий тег ровно один раз,
        # блок не переоткрывается), но новая приставка NO COMMENT видна в живом
        # выводе как часть сообщения модели — в историю она не попадает.
        kinds = [kind for kind, _ in events]
        self.assertEqual(kinds.count("start"), 1)
        self.assertEqual(kinds.count("end"), 1)
        self.assertEqual(chunks.count("<short_think>"), 1)
        self.assertEqual(
            "".join(chunks),
            "<short_think>мысли</short_think>\n" + _NO_COMMENT_PREFILL + "теперь скажу",
        )
        self.assertEqual(result, "<short_think>мысли</short_think>\nтеперь скажу")


# ── Пустой ответ: продолжение с _NO_COMMENT_PREFILL ───────────────────────

class TestEmptyAnswerContinuation(SimReasoningTestCase):
    def test_empty_answer_continues_with_no_comment_prefill(self):
        msgs = []
        agent = self._agent(system_msgs=msgs)
        fake, calls = _transport([
            ("мысли</short_think>", None),
            ("теперь скажу", None),
        ])
        with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake):
            result = agent.chat("сколько будет 21*2?")

        # Рассуждение не перегенерируется: продолжаем ровно с его конца.
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1].prefill, "<short_think>мысли</short_think>\n" + _NO_COMMENT_PREFILL)
        # Приставка была нашим prefill — в ответ и в историю она не попадает.
        self.assertEqual(result, "<short_think>мысли</short_think>\nтеперь скажу")
        self.assertTrue(any("No answer text" in m for m in msgs), msgs)
        self.assertEqual(self._assistant_contents(agent), [result])

    def test_continuation_budget_is_bounded(self):
        msgs = []
        agent = self._agent(system_msgs=msgs)
        fake, calls = _transport([
            ("мысли</short_think>", None),
            ("", None),
            ("всё равно пусто", None),
        ])
        with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake):
            result = agent.chat("сколько будет 21*2?")

        # Ровно SIMULATED_REASONING_REPAIRS продолжений: бюджет не даёт зациклиться.
        self.assertEqual(len(calls), 3)
        self.assertIn("retries left: 1", msgs[0])
        self.assertIn("retries left: 0", msgs[1])
        self.assertIn("всё равно пусто", result)
        # Вклинённая приставка — наш prefill, в ответе её быть не должно.
        self.assertNotIn("LLM Assistant", result)

    def test_zero_repair_budget_accepts_empty_answer(self):
        Config.SIMULATED_REASONING_REPAIRS = 0
        msgs = []
        agent = self._agent(system_msgs=msgs)
        fake, calls = _transport([("мысли</short_think>", None)])
        with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake):
            result = agent.chat("сколько будет 21*2?")

        self.assertEqual(len(calls), 1)
        # Ответ пуст — висячий перевод строки срезан нормализацией контента
        # (_process_llm_response: content.strip()); за тегом всё равно ничего нет.
        self.assertEqual(result, "<short_think>мысли</short_think>")
        self.assertTrue(any("budget exhausted" in m for m in msgs))


class TestTwoPhaseReasoning(SimReasoningTestCase):
    """Двухфазный режим: задана хотя бы одна SIMULATED_REASONING_*-настройка —
    секция размышлений генерируется отдельно (свои параметры + стоп-маркер),
    ответ — вторым вызовом с обычными параметрами."""

    def _enable_overrides(self):
        Config.SIMULATED_REASONING_TEMP = 0.9
        Config.SIMULATED_REASONING_TOP_P = 0.5
        Config.SIMULATED_REASONING_MIN_P = 0.1
        Config.SIMULATED_REASONING_FREQUENCY_PENALTY = 0.3
        Config.SIMULATED_REASONING_PRESENCE_PENALTY = 0.4

    def test_think_and_answer_use_different_params(self):
        self._enable_overrides()
        agent = self._agent()
        fake, calls = _transport([
            ("мысли</short_think>", None),
            ("готовый ответ", None),
        ])
        with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake):
            result = agent.chat("сколько будет 21*2?")

        self.assertEqual(len(calls), 2)
        # Фаза размышлений: открывающий prefill, стоп-маркер закрытия, свои параметры.
        self.assertEqual(calls[0].prefill, "<short_think>")
        self.assertEqual(calls[0].markers, ("</short_think>",))
        think = calls[0].params
        self.assertEqual(
            (think.temp, think.top_p, think.min_p,
             think.frequency_penalty, think.presence_penalty),
            (0.9, 0.5, 0.1, 0.3, 0.4),
        )
        # Фаза ответа: накопленный prefill, без маркеров, обычные параметры.
        self.assertEqual(calls[1].prefill, "<short_think>мысли</short_think>\n")
        self.assertEqual(calls[1].markers, ())
        answer = calls[1].params
        self.assertEqual(answer.temp, Config.TEMP)
        # min_p не задан для размышлений — унаследован глобальный (по умолчанию None).
        self.assertEqual(answer.min_p, Config.MIN_P)
        self.assertNotEqual(answer.temp, think.temp)
        self.assertEqual(result, "<short_think>мысли</short_think>\nготовый ответ")
        self.assertEqual(self._assistant_contents(agent), [result])

    def test_partial_overrides_inherit_the_rest(self):
        Config.SIMULATED_REASONING_TEMP = 0.9
        agent = self._agent()
        fake, calls = _transport([
            ("мысли</short_think>", None),
            ("готовый ответ", None),
        ])
        with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake):
            agent.chat("сколько будет 21*2?")

        self.assertEqual(len(calls), 2)
        think = calls[0].params
        self.assertEqual(think.temp, 0.9)
        # Остальное унаследовано от базовых параметров хода.
        self.assertEqual(think.top_p, Config.TOP_P)
        self.assertEqual(think.min_p, Config.MIN_P)

    def test_think_phase_tool_call_accepted_immediately(self):
        self._enable_overrides()
        agent = self._agent(tools=True)
        fake, calls = _transport([
            ("надо посмотреть</short_think>",
             [ToolCall(id="t1", name="double_me", arguments='{"value": 21}')]),
            ("посмотрел</short_think>", None),
            ("Готово: 42", None),
        ])
        with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake):
            result = agent.chat("удвой 21")

        # Инструмент из фазы размышлений исполнен сразу, без фазы ответа;
        # следующее сообщение — новая think-фаза с теми же параметрами.
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0].markers, ("</short_think>",))
        self.assertEqual(calls[1].prefill, "<short_think>")
        self.assertEqual(calls[1].params.temp, 0.9)
        self.assertEqual(calls[2].prefill, "<short_think>посмотрел</short_think>\n")
        self.assertEqual([m.content for m in self._tool_results(agent)], ["42"])
        self.assertIn("Готово: 42", result)

    def test_empty_think_still_gets_answer_phase(self):
        self._enable_overrides()
        agent = self._agent()
        fake, calls = _transport([
            ("", None),
            ("ответ без размышлений", None),
        ])
        with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake):
            result = agent.chat("сколько будет 21*2?")

        self.assertEqual(len(calls), 2)
        self.assertEqual(result, "<short_think></short_think>\nответ без размышлений")

    def test_two_phase_empty_answer_uses_continuation(self):
        self._enable_overrides()
        msgs = []
        agent = self._agent(system_msgs=msgs)
        fake, calls = _transport([
            ("мысли</short_think>", None),
            ("", None),
            ("нашёлся ответ", None),
        ])
        with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake):
            result = agent.chat("сколько будет 21*2?")

        # think + пустой answer + одно NO COMMENT-продолжение.
        self.assertEqual(len(calls), 3)
        self.assertTrue(any("No answer text" in m for m in msgs), msgs)
        self.assertEqual(result, "<short_think>мысли</short_think>\nнашёлся ответ")
        self.assertNotIn("LLM Assistant", result)

    def test_two_phase_streaming_keeps_single_block(self):
        self._enable_overrides()
        chunks = []
        events = []
        agent = self._streaming_agent(chunks, events)
        script = [["мысли", "</short_think>"], ["готовый ответ"]]
        stream, calls = self._scripted_stream(script)

        with mock.patch("universal_agents.agent.LLMClient.stream", side_effect=stream), \
             mock.patch("universal_agents.llm_client.LLMClient.close_stream"):
            result = agent.chat("сколько будет 21*2?")

        # Две фазы — один блок вывода: мысли и ответ идут подряд без дублей.
        kinds = [kind for kind, _ in events]
        self.assertEqual(kinds.count("start"), 1)
        self.assertEqual(kinds.count("end"), 1)
        self.assertEqual("".join(chunks), "<short_think>мысли</short_think>\nготовый ответ")
        self.assertEqual(result, "<short_think>мысли</short_think>\nготовый ответ")


# ── Инструменты: группа перезапускается, один вызов за сообщение ───────────

class TestSimulatedReasoningWithTools(SimReasoningTestCase):
    def _tool_call(self, value=21):
        return [ToolCall(id="t1", name="double_me", arguments='{"value": %d}' % value)]

    def test_tool_call_keeps_answer_and_group_restarts(self):
        agent = self._agent(tools=True)
        fake, calls = _transport([
            ("удваиваю</short_think>", self._tool_call()),
            ("проверяю</short_think>Получилось 42", None),
        ])
        with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake):
            result = agent.chat("удвой 21")

        # Структура начинается заново на каждом assistant-сообщении.
        self.assertEqual([c.prefill for c in calls], ["<short_think>", "<short_think>"])
        self.assertEqual(result, "<short_think>проверяю</short_think>\nПолучилось 42")
        self.assertEqual([m.content for m in self._tool_results(agent)], ["42"])

    def test_tool_call_without_answer_text_is_accepted(self):
        msgs = []
        agent = self._agent(tools=True, system_msgs=msgs)
        fake, calls = _transport([
            ("сразу беру инструмент</short_think>", self._tool_call()),
            ("инструмент отработал</short_think>42", None),
        ])
        with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake):
            result = agent.chat("удвой 21")

        # Модели нечего сказать — продолжение тут только зациклило бы ход (§2.8).
        self.assertEqual(len(calls), 2)
        self.assertEqual([m.content for m in self._tool_results(agent)], ["42"])
        self.assertTrue(any("tool call accepted as-is" in m for m in msgs), msgs)
        self.assertFalse(any("[NO COMMENT]" in m for m in msgs), msgs)
        self.assertEqual(
            self._assistant_contents(agent)[0],
            # Пустой ответ: висячий перевод срезан нормализацией (см. выше).
            "<short_think>сразу беру инструмент</short_think>",
        )
        self.assertEqual(result, "<short_think>инструмент отработал</short_think>\n42")

    def test_one_tool_call_per_message(self):
        agent = self._agent(tools=True)
        two = self._tool_call() + [ToolCall(id="t2", name="double_me", arguments='{"value": 5}')]
        fake, _calls = _transport([
            ("считаю</short_think>", two),
            ("готово</short_think>всё", None),
        ])
        with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake):
            agent.chat("удвой 21", max_iter=5)

        # Инвариант §1.3: в сообщение попадает ровно один вызов.
        tool_call_counts = [len(m.tool_calls) for m in agent.history.get_all()
                            if isinstance(m, AssistantMessage)]
        self.assertTrue(all(c <= 1 for c in tool_call_counts), tool_call_counts)


# ── Дефолтный выключатель формата ─────────────────────────────────────────

class TestSimulatedReasoningDisabled(SimReasoningTestCase):
    def test_thinking_on_keeps_plain_answers(self):
        agent = self._agent()
        agent._thinking_enabled = True
        fake, calls = _plain_transport(["просто текст"])
        with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake):
            result = agent.chat("привет")

        self.assertEqual(calls, [None])
        self.assertEqual(result, "просто текст")

    def test_user_prefill_wins_over_format(self):
        agent = self._agent()
        fake, calls = _plain_transport(["42"])
        with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake):
            result = agent.chat("2*21?", prefill="The answer is ")

        self.assertEqual(calls, ["The answer is "])
        self.assertEqual(result, "The answer is 42")

    def test_service_turn_is_not_formatted(self):
        agent = self._agent()
        fake, calls = _plain_transport(["итог саммари"])
        with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake):
            msg, err, _ = agent.service_llm_call([{"role": "user", "content": "суммируй диалог"}])

        self.assertEqual(calls, [None])
        self.assertEqual(msg.content, "итог саммари")


# ── Детекция сломанного вызова не ловит прозу ─────────────────────────────

class TestBrokenCallScanning(SimReasoningTestCase):
    def test_tool_mention_in_reasoning_is_not_broken_call(self):
        msgs = []
        agent = self._agent(tools=True, system_msgs=msgs)
        fake, _calls = _transport([
            ("нужно вызвать double_me(21)</short_think>сделаю это", None),
        ])
        with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake):
            result = agent.chat("удвой 21")

        self.assertIn("сделаю это", result)
        self.assertFalse(any("[BROKEN CALL]" in m for m in msgs))

    def test_tool_mention_in_prose_answer_is_not_broken_call(self):
        msgs = []
        agent = self._agent(tools=True, system_msgs=msgs)
        # Именно этот сценарий ловил ложный BROKEN CALL: проза с упоминанием read().
        answer = "read() выполнен: показана структура каталога. Вызываю read() для деталей."
        fake, _calls = _transport([(f"посмотрел</short_think>{answer}", None)])
        with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake):
            result = agent.chat("что в папке?")

        self.assertIn(answer, result)
        self.assertFalse(any("[BROKEN CALL]" in m for m in msgs), msgs)

    def test_standalone_call_in_answer_is_detected(self):
        msgs = []
        agent = self._agent(tools=True, system_msgs=msgs)
        fake, _calls = _transport([
            ("хм</short_think>double_me(21)", None),
            ("хм, переделаю</short_think>думаю, этого хватит", None),
        ])
        with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake):
            agent.chat("удвой 21", max_iter=5)

        self.assertTrue(any("[BROKEN CALL]" in m for m in msgs), msgs)

    def test_tool_mention_prose_variants_are_not_broken_call(self):
        # Регрессия: реальный баг — гейт «есть XML-тег» выполнялся по контенту ВМЕСТЕ
        # с тегом секции, поэтому любая проза про инструмент давала ложный BROKEN CALL.
        variants = [
            "read() выполнен: показана структура каталога.",
            "Вызываю read() для просмотра содержимого текущей директории:",
            "Сделаю read(), потом напишу итог.",
            "Сначала double_me(21), затем поиск.",
            "Инструменты: read(path), write(path).",
            "Ничего не делаю.",
        ]
        for answer in variants:
            with self.subTest(answer=answer):
                msgs = []
                agent = self._agent(tools=True, system_msgs=msgs)
                fake, _calls = _transport([(f"посмотрел</short_think>{answer}", None)])
                with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake):
                    result = agent.chat("что в папке?")

                self.assertIn(answer, result)
                self.assertFalse(any("[BROKEN CALL]" in m for m in msgs), msgs)

    def test_section_tags_alone_do_not_enable_the_gate(self):
        # Теги секции не должны открывать гейт скана: скан идёт по ответу, а не по
        # контенту вместе с <short_think>…</short_think>.
        msgs = []
        agent = self._agent(tools=True, system_msgs=msgs)
        fake, _calls = _transport([("мысли про read()</short_think>итог без вызовов", None)])
        with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake):
            agent.chat("удвой 21")

        self.assertFalse(any("[BROKEN CALL]" in m for m in msgs), msgs)

    def test_stray_tool_tag_in_answer_is_detected(self):
        msgs = []
        agent = self._agent(tools=True, system_msgs=msgs)
        fake, _calls = _transport([
            ("хм</short_think><tool_call>мусор</tool_call>", None),
            ("хм, переделаю</short_think>готово", None),
        ])
        with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake):
            agent.chat("удвой 21", max_iter=5)

        self.assertTrue(any("[BROKEN CALL]" in m for m in msgs), msgs)


class TestNoCommentResponse(unittest.TestCase):
    def test_no_comment_rerun_when_reasoning_off(self):
        agent = LLMAgent(system_prompt="sys")
        agent.history.add(UserMessage("compute"))
        msg = AssistantMessage(
            content="",
            tool_calls=[ToolCall(id="t1", name="double_me", arguments='{"value": 2}')],
        )
        text, tool_err, broken, rerun = agent._process_llm_response(msg)
        self.assertEqual(rerun, _NO_COMMENT_PREFILL)
        self.assertEqual(text, "")
        # пустой ответ не попал в историю, инструмент не исполнялся
        roles = [m.to_api_dict()["role"] for m in agent.history]
        self.assertEqual(roles, ["system", "user"])

    def test_no_comment_accepted_when_reasoning_on(self):
        agent = LLMAgent(
            system_prompt="sys",
            tools_config=["double_me"],
            external_plugins={"double_me": double_me},
        )
        agent._thinking_enabled = True
        msg = AssistantMessage(
            content="",
            tool_calls=[ToolCall(id="t1", name="double_me", arguments='{"value": 21}')],
        )
        text, tool_err, broken, rerun = agent._process_llm_response(msg)
        self.assertIsNone(rerun)
        roles = [m.to_api_dict()["role"] for m in agent.history]
        self.assertIn("tool", roles)
        tr = [m for m in agent.history.get_all() if m.to_api_dict()["role"] == "tool"][-1]
        self.assertEqual(tr.content, "42")


if __name__ == "__main__":
    unittest.main()