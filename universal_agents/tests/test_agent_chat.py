import os
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

from universal_agents.agent import LLMAgent
from universal_agents.agent_mixins.response_mixin import _NO_COMMENT_PREFILL
from universal_agents.models import AssistantMessage, ToolCall, ToolResult, UserMessage
from universal_agents.tool import tool
from universal_agents.config import Config
from universal_agents.constants import INTERRUPT_HEADER


@tool(description="double a value")
def double_me(agent, value: int) -> str:
    return str(value * 2)


@tool(description="always fails")
def fail_me(agent, value: int) -> str:
    raise ValueError("boom")


def _chunk(delta=None, usage=None, choices=None):
    if choices is None:
        choices = [SimpleNamespace(delta=delta)] if delta is not None else []
    return SimpleNamespace(choices=choices, usage=usage)


def _delta(content=None, tool_calls=None, reasoning_content=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls, reasoning_content=reasoning_content)


def _tc_delta(index, id=None, name=None, arguments=None):
    return SimpleNamespace(
        index=index,
        id=id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class TestAgentChat(unittest.TestCase):
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

    def test_chat_streaming_executes_tool(self):
        agent = LLMAgent(
            system_prompt="sys",
            tools_config=["double_me"],
            external_plugins={"double_me": double_me},
            streaming_enabled=True,
            on_stream_chunk=lambda _: None,
        )

        def stream1(*args, **kwargs):
            yield _chunk(_delta(content="Let me compute "))
            yield _chunk(_delta(content="21 * 2 "))
            yield _chunk(_delta(tool_calls=[_tc_delta(0, id="t1", name="double_me", arguments='{"value": ')],
                                content=""))
            yield _chunk(_delta(tool_calls=[_tc_delta(0, arguments='21}')]))
            yield _chunk(_delta(tool_calls=[_tc_delta(0, id="t1")]))
            yield _chunk(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15))

        def stream2(*args, **kwargs):
            yield _chunk(_delta(content="final "))
            yield _chunk(_delta(content="answer 42"))
            yield _chunk(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15))

        with mock.patch(
            "universal_agents.agent.LLMClient.stream",
            side_effect=[stream1(), stream2()],
        ):
            result = agent.chat("compute", max_iter=5)
        self.assertIn("final answer 42", result)
        roles = [m.to_api_dict()["role"] for m in agent.history]
        self.assertIn("tool", roles)


    def test_chat_streaming_applies_prefill_and_emits_it(self):
        seen = []

        def on_chunk(chunk):
            seen.append(chunk)

        agent = LLMAgent(
            system_prompt="sys",
            streaming_enabled=True,
            on_stream_chunk=on_chunk,
        )

        def stream(*args, **kwargs):
            yield _chunk(_delta(content="hello back"))

        with mock.patch("universal_agents.agent.LLMClient.stream", return_value=stream()):
            result = agent.chat("hello", prefill="<start>")

        self.assertTrue(result.startswith("<start>"), f"result should start with prefill: {result!r}")
        self.assertEqual(seen[0], "<start>", "prefill should be the first streamed chunk")
        self.assertEqual("".join(seen), "<start>hello back", f"unexpected stream chunks: {seen!r}")

    def test_chat_streaming_emits_prefill_after_reasoning(self):
        seen = []

        def on_chunk(chunk):
            seen.append(chunk)

        agent = LLMAgent(
            system_prompt="sys",
            streaming_enabled=True,
            on_stream_chunk=on_chunk,
        )

        def stream(*args, **kwargs):
            yield _chunk(_delta(reasoning_content="think..."))
            yield _chunk(_delta(content="hello back"))

        with mock.patch("universal_agents.agent.LLMClient.stream", return_value=stream()):
            result = agent.chat("hello", prefill="<start>")

        self.assertTrue(result.startswith("<start>"))
        self.assertEqual(seen, ["<start>", "hello back"], f"unexpected stream chunks: {seen!r}")

    def test_chat_streaming_prefill_with_empty_content(self):
        agent = LLMAgent(
            system_prompt="sys",
            streaming_enabled=True,
            on_stream_chunk=lambda _: None,
        )

        def stream(*args, **kwargs):
            yield _chunk(_delta(content=""))
            yield _chunk(usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2))

        with mock.patch("universal_agents.agent.LLMClient.stream", return_value=stream()):
            result = agent.chat("hello", prefill="X")

        self.assertEqual(result, "X")

    def test_auto_trust_git_root(self):
        with tempfile.TemporaryDirectory() as repo:
            os.makedirs(os.path.join(repo, ".git"))
            open(os.path.join(repo, ".git", "HEAD"), "w").close()
            with mock.patch("universal_agents.agent_mixins.tools_mixin.find_project_root", return_value=repo):
                agent = LLMAgent(system_prompt="sys")
            self.assertIn(os.path.abspath(repo), agent.trusted_dirs)
            # файлы внутри корня считаются доверенными
            self.assertTrue(agent.is_path_trusted(os.path.join(repo, "src", "index.html")))

    def test_auto_trust_skipped_when_no_git(self):
        with tempfile.TemporaryDirectory() as nodir:
            with mock.patch("universal_agents.agent_mixins.tools_mixin.find_project_root", return_value=None):
                agent = LLMAgent(system_prompt="sys")
            self.assertEqual(agent.trusted_dirs, set())

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

    def test_tool_error_triggers_recovery(self):
        agent = LLMAgent(
            system_prompt="sys",
            tools_config=["fail_me"],
            external_plugins={"fail_me": fail_me},
            max_generation_attempts=3,
        )
        err_call = AssistantMessage(content="", tool_calls=[ToolCall(id="t1", name="fail_me", arguments='{"value": 1}')])
        fixed = AssistantMessage(content="recovered answer")
        with mock.patch(
            "universal_agents.agent.LLMClient.call",
            side_effect=[(err_call, None, None), (fixed, None, None)],
        ):
            result = agent.chat("do it")
        self.assertEqual(result, "recovered answer")
        # не найдено ни одного error-result в итоговой истории
        self.assertFalse(any(getattr(m, 'is_error', False) for m in agent.history.get_all()))

    def test_consecutive_tool_errors_hit_limit(self):
        old_retries = Config.ERROR_RECOVERY_RETRIES
        Config.ERROR_RECOVERY_RETRIES = 0
        try:
            agent = LLMAgent(
                system_prompt="sys",
                tools_config=["fail_me"],
                external_plugins={"fail_me": fail_me},
                max_generation_attempts=1,
            )
            err_call = AssistantMessage(
                content="",
                tool_calls=[ToolCall(id="t1", name="fail_me", arguments='{"value": 1}')],
            )
            with mock.patch(
                "universal_agents.agent.LLMClient.call",
                return_value=(err_call, None, None),
            ):
                result = agent.chat("loop", max_iter=10)
            self.assertEqual(result, "")
        finally:
            Config.ERROR_RECOVERY_RETRIES = old_retries

    def test_duplicate_answer_triggers_regen(self):
        agent = LLMAgent(system_prompt="sys", max_generation_attempts=3)
        # предыдущий ответ уже есть в истории
        agent.history.add(UserMessage("first"))
        agent.history.add(AssistantMessage(content="same answer"))
        dup = AssistantMessage(content="same answer")
        fresh = AssistantMessage(content="new answer")
        with mock.patch(
            "universal_agents.agent.LLMClient.call",
            side_effect=[(dup, None, None), (fresh, None, None)],
        ):
            result = agent.chat("second")
        self.assertEqual(result, "new answer")

    def test_duplicate_answer_with_paraphrase_triggers_regen(self):
        # ровно те сообщения, что агент прислал пользователю в заевшей «пластинке»
        first = (
            "Я проверил файл `main.py` и нашёл секрет!\n"
            "Секрет находится в первой строчке файла (строка 1). В начале кода есть "
            "отключающая переменная, которая сбрасывает все настройки к умолчаниювым "
            "значениям при загрузке скрипта. Это часто делается для обеспечения "
            "стабильного поведения модели или выполнения по специфическим инструкциям "
            "без влияния внешних настроек.\n"
            "Если вы хотите убрать этот секрет и вернуть исходные настройки, "
            "отредактируйте первый ряд кода файла `main.py`."
        )
        repeated = (
            "Я проверил содержимое файла `main.py` и нашёл секрет!\n"
            "Секрет находится в первой строке (строка 1). В начале кода есть "
            "отключающая переменная, которая сбрасывает все настройки к умолчаниювым "
            "значениям при загрузке скрипта. Это часто делается для обеспечения "
            "стабильного поведения модели или выполнения по специфическим инструкциям "
            "без влияния внешних настроек.\n"
            "Если вы хотите убрать этот секрет и вернуть исходные настройки, "
            "отредактируйте первый ряд кода файла `main.py`."
        )
        agent = LLMAgent(system_prompt="sys", max_generation_attempts=3)
        agent.history.add(UserMessage("first"))
        agent.history.add(AssistantMessage(content=first))
        fresh = AssistantMessage(content="new answer")
        with mock.patch(
            "universal_agents.agent.LLMClient.call",
            side_effect=[(AssistantMessage(content=repeated), None, None), (fresh, None, None)],
        ):
            result = agent.chat("second")
        # повтор был отброшен, вернулся свежий ответ
        self.assertEqual(result, "new answer")

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

    def test_thinking_once_consistent_across_whole_turn(self):
        """Разовый /think применяется ко всему ходу: и API, и обработчик ответа видят
        reasoning 'low', поэтому пустой tool-call исполняется без NO COMMENT-перегенераций.
        Регрессия: раньше _reasoning_effort сбрасывал _thinking_once при первом чтении,
        и второе чтение (NO COMMENT-ветка) расходилось с отправленным в API."""
        agent = LLMAgent(
            system_prompt="sys",
            tools_config=["double_me"],
            external_plugins={"double_me": double_me},
            autosave_enabled=False,
        )
        bare_call = AssistantMessage(
            content="",
            tool_calls=[ToolCall(id="t1", name="double_me", arguments='{"value": 21}')],
        )
        final_reply = AssistantMessage(content="Готово.")

        seen = []
        responses = [(bare_call, None, None), (final_reply, None, None)]

        def fake_call(messages, reasoning_effort=None, **kwargs):
            seen.append(reasoning_effort)
            return responses.pop(0)

        with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake_call):
            agent._thinking_once = True
            result = agent.chat("compute")

        self.assertEqual(result, "Готово.")
        # Все API-вызовы хода получили reasoning 'low' (разовый тоггл активен):
        # ни один обработчик не увидел 'none' — регрессия на рассинхрон между API и NO COMMENT-веткой.
        self.assertTrue(seen, "должен быть хотя бы один API-вызов")
        self.assertTrue(all(e == "low" for e in seen), seen)
        # Голый вызов исполнен сразу (reasoning 'low' для обработчика) — без
        # NO COMMENT-перегенераций: в истории есть tool-результат.
        roles = [m.to_api_dict()["role"] for m in agent.history]
        self.assertIn("tool", roles)
        tr = [m for m in agent.history.get_all() if m.to_api_dict()["role"] == "tool"][-1]
        self.assertEqual(tr.content, "42")

    def test_forced_compaction_short_history_returns_false(self):
        agent = LLMAgent(system_prompt="sys")
        with mock.patch("universal_agents.compressors.summarize_history_plain") as summ:
            result = agent._auto_summarize_dialogue(force=True)
        self.assertFalse(result)
        summ.assert_not_called()

    def test_forced_compaction_writes_summary(self):
        agent = LLMAgent(system_prompt="sys")
        agent.history.add(UserMessage("question one"))
        agent.history.add(AssistantMessage(content="answer one"))
        agent.history.add(UserMessage("question two"))
        agent.history.add(AssistantMessage(content="answer two"))
        ids = ["1.1", "1.2", "1.3", "1.4", "1.5", "2.1", "2.2", "2.3",
               "3.1", "4.1", "4.2", "4.3", "5.1", "6.1", "6.2", "7.1", "7.2"]
        summary_body = "".join(f"{section_id} content. " for section_id in ids) + "Compressed."
        with mock.patch(
            "universal_agents.compressors.summarize_history_plain",
            return_value=summary_body,
        ):
            result = agent._auto_summarize_dialogue(force=True)
        self.assertTrue(result)
        user_msgs = [m for m in agent.history.get_all() if isinstance(m, UserMessage)]
        self.assertTrue(any("Compressed" in (m.content or "") for m in user_msgs))

    def test_auto_summarize_bails_on_user_stop(self):
        agent = LLMAgent(system_prompt="sys")
        agent.history.add(UserMessage("q"))
        agent.history.add(AssistantMessage(content="a"))
        agent.stop_event.set()
        with mock.patch("universal_agents.compressors.summarize_history_plain") as summ:
            result = agent._auto_summarize_dialogue(force=True)
        self.assertFalse(result)
        summ.assert_not_called()
        # история не изменена: роли те же, summary-сообщение не добавлено
        roles = [m.to_api_dict()["role"] for m in agent.history]
        self.assertEqual(roles, ["system", "user", "assistant"])

    def test_auto_summarize_sets_suppression_on_user_stop(self):
        agent = LLMAgent(system_prompt="sys")
        agent.history.add(UserMessage("q"))
        agent.history.add(AssistantMessage(content="a"))
        agent.stop_event.set()
        with mock.patch("universal_agents.compressors.summarize_history_plain") as summ:
            result = agent._auto_summarize_dialogue(force=True)
        self.assertFalse(result)
        self.assertTrue(agent._auto_summarize_suppressed)

    def test_auto_summarize_clears_suppression_on_success(self):
        agent = LLMAgent(system_prompt="sys")
        agent._auto_summarize_suppressed = True
        agent.history.add(UserMessage("question one"))
        agent.history.add(AssistantMessage(content="answer one"))
        agent.history.add(UserMessage("question two"))
        agent.history.add(AssistantMessage(content="answer two"))
        ids = ["1.1", "1.2", "1.3", "1.4", "1.5", "2.1", "2.2", "2.3",
               "3.1", "4.1", "4.2", "4.3", "5.1", "6.1", "6.2", "7.1", "7.2"]
        summary_body = "".join(f"{section_id} content. " for section_id in ids) + "Compressed."
        with mock.patch(
            "universal_agents.compressors.summarize_history_plain",
            return_value=summary_body,
        ):
            result = agent._auto_summarize_dialogue(force=True)
        self.assertTrue(result)
        self.assertFalse(agent._auto_summarize_suppressed)

    def test_auto_summarize_suppression_skips_and_chat_resets(self):
        agent = LLMAgent(system_prompt="sys")
        fake = AssistantMessage(content="answer")
        # История над порогом, но компакция подавлена прерванной попыткой.
        agent._auto_summarize_suppressed = True
        with mock.patch("universal_agents.agent.LLMClient.call", return_value=(fake, None, None)), \
                mock.patch.object(agent, "_auto_summarize_dialogue") as auto, \
                mock.patch.object(agent, "_get_context_usage_percent", return_value=99.0), \
                mock.patch.object(agent, "_autosave"):
            agent.history.add(UserMessage("q"))
            agent.history.add(AssistantMessage(content="a"))
            agent._run_turn_loop(max_iter=1, prefill="")
        auto.assert_not_called()
        # Новый ход пользователя через chat() снимает cooldown.
        fake2 = AssistantMessage(content="fresh answer")
        with mock.patch("universal_agents.agent.LLMClient.call", return_value=(fake2, None, None)), \
                mock.patch.object(agent, "_autosave"):
            result = agent.chat("next user turn")
        self.assertIn("fresh answer", result)
        self.assertFalse(agent._auto_summarize_suppressed)

    def test_service_llm_call_passes_stop_check(self):
        agent = LLMAgent(system_prompt="sys")
        with mock.patch("universal_agents.agent.LLMClient.call") as mocked:
            agent.service_llm_call([{"role": "user", "content": "hi"}])
        _, kwargs = mocked.call_args
        self.assertIs(kwargs["stop_check"], agent._stop_check)


if __name__ == "__main__":
    unittest.main()
