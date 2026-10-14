import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from universal_agents.agent import LLMAgent, _GUARD_NAG_PREFIX
from universal_agents.agent_mixins.response_mixin import _NO_COMMENT_PREFILL
from universal_agents.config import Config
from universal_agents.models import AssistantMessage, ToolCall, ToolResult, UserMessage
from universal_agents.tool import tool, ToolOutput
from universal_agents.tools.fs import line_range_edit
from universal_agents.tools.builtin import respond_to_system
from universal_agents.tools.fs import read as _read_tool

from tests.conftest import make_agent as make_test_agent

B64 = "aGVsbG8taW1hZ2U="

answer_tool_name = respond_to_system.__name__
line_range_edit_tool_name = line_range_edit.__name__


def nag_contents(msgs) -> list:
    return [m.content for m in msgs
            if isinstance(m, UserMessage) and _GUARD_NAG_PREFIX in (m.content or "")]


class TestAnswerRequiredGuard(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)

    def make_agent(self):
        agent = make_test_agent(
            system_prompt="You edit files. Always call answer to confirm edits.",
            external_plugins={line_range_edit_tool_name: line_range_edit, answer_tool_name: respond_to_system},
        )
        agent.trust_dir(self._tmp)
        return agent

    def test_repeated_text_answer_is_guard_violation_not_duplicate(self):
        path = os.path.join(self._tmp, "hello.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("hello\n")

        prior_answer = AssistantMessage(content="Подтверждаю.")
        edit_call = AssistantMessage(
            content="Отредактирую файл.",
            tool_calls=[ToolCall(id="c1", name=line_range_edit_tool_name, arguments=json.dumps({
                "path": path, "new_text": "world\n", "start_line": 1, "end_line": 1,
            }))],
        )
        answer_call = AssistantMessage(
            content="Подтверждаю.",
            tool_calls=[ToolCall(id="c2", name=answer_tool_name, arguments='{"text": "yes"}')],
        )
        final_reply = AssistantMessage(content="Операция завершена.")

        sysmsgs = []
        responses = [
            (prior_answer, None, None),      # ход 1: обычный текст, в историю
            (edit_call, None, None),         # ход 2: edit → превью, pending + наг
            (prior_answer, None, None),      # та же фраза: раньше это был DUPLICATE
            (answer_call, None, None),       # модель вспомнила протокол
            (final_reply, None, None),
        ]

        def fake_call(messages, prefill=None, **kwargs):
            return responses.pop(0)

        with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake_call):
            agent = self.make_agent()
            agent.on_system_msg = sysmsgs.append
            agent.chat("Привет", max_iter=5)
            result = agent.chat("Отрезай hello", max_iter=10)

        self.assertIn("Операция завершена", result)
        self.assertEqual([m for m in sysmsgs if "DUPLICATE ANSWER DETECTED" in m], [],
                         "При висящем pending текст должен разбираться guard'ом, не детектором дублей")
        notices = [m for m in sysmsgs if "CONFIRMATION PENDING" in m]
        self.assertEqual(len(notices), 1, "Пользователь должен быть предупреждён ровно один раз за ход")
        self.assertIn("attempt 1/5", notices[0])
        self.assertIsNone(agent._pending_operation)
        with open(path, encoding="utf-8") as f:
            self.assertEqual(f.read(), "world\n")

    def test_text_answer_without_answer_tool_then_answer_yes(self):
        path = os.path.join(self._tmp, "hello.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("hello\n")

        edit_call = AssistantMessage(
            content="Отредактирую файл.",
            tool_calls=[ToolCall(id="c1", name=line_range_edit_tool_name, arguments=json.dumps({
                "path": path,
                "new_text": "world\n",
                "start_line": 1,
                "end_line": 1,
            }))],
        )
        text_turn = AssistantMessage(content="Я отредактирую файл и подтверждаю правку, всё хорошо.")
        answer_call = AssistantMessage(
            content="Подтверждаю.",
            tool_calls=[ToolCall(id="c2", name=answer_tool_name, arguments='{"text": "yes"}')],
        )
        final_reply = AssistantMessage(content="Операция завершена.")

        seen = []
        responses = [
            (edit_call, None, None),
            (text_turn, None, None),
            (answer_call, None, None),
            (final_reply, None, None),
        ]

        def fake_call(messages, prefill=None, **kwargs):
            seen.append([m.get("content", "") for m in messages if m.get("role") == "user"])
            return responses.pop(0)

        with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake_call):
            agent = self.make_agent()
            result = agent.chat("Отредактируй hello.txt: замени hello на world", max_iter=10)

        self.assertIn("Операция завершена", result)
        self.assertIsNone(agent._pending_operation)
        with open(path, encoding="utf-8") as f:
            self.assertEqual(f.read(), "world\n")

        self.assertTrue(any(_GUARD_NAG_PREFIX in t for content in seen for t in content),
                        "Модель должна была получить сообщение-ошибку о вызове answer")
        msgs = agent.history.get_all()
        self.assertEqual(nag_contents(msgs), [],
                         "После успешного respond_to_system наг должен быть вычищен из истории")

    def test_text_answer_without_answer_tool_then_text_again_no_message(self):
        path = os.path.join(self._tmp, "x.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("a\n")

        edit_call = AssistantMessage(
            content="Отредактирую файл.",
            tool_calls=[ToolCall(id="c1", name=line_range_edit_tool_name, arguments=json.dumps({
                "path": path,
                "new_text": "b\n",
                "start_line": 1,
                "end_line": 1,
            }))],
        )
        text1 = AssistantMessage(content="Не думаю, что нужно менять.")
        text2 = AssistantMessage(content="Ладно, но вызвать инструмент не буду.")
        answer_no = AssistantMessage(
            content="Отменяю.",
            tool_calls=[ToolCall(id="c2", name=answer_tool_name, arguments='{"text": "no"}')],
        )
        final_reply = AssistantMessage(content="Готово.")
        seen = []
        responses = [
            (edit_call, None, None),
            (text1, None, None),
            (text2, None, None),
            (answer_no, None, None),
            (final_reply, None, None),
        ]

        def fake_call(messages, prefill=None, **kwargs):
            seen.append([m.get("content", "") for m in messages if m.get("role") == "user"])
            return responses.pop(0)

        with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake_call):
            agent = self.make_agent()
            result = agent.chat("Измени файл", max_iter=10)

        self.assertIn("Готово", result)
        with open(path, encoding="utf-8") as f:
            self.assertEqual(f.read(), "a\n")
        self.assertIsNone(agent._pending_operation)
        self.assertTrue(any(_GUARD_NAG_PREFIX in t for content in seen for t in content))
        msgs = agent.history.get_all()
        self.assertEqual(nag_contents(msgs), [])

    def test_bare_answer_is_regenerated_until_model_adds_comment(self):
        path = os.path.join(self._tmp, "k.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("a\n")

        edit_call = AssistantMessage(
            content="Отредактирую файл.",
            tool_calls=[ToolCall(id="c1", name=line_range_edit_tool_name, arguments=json.dumps({
                "path": path, "new_text": "b\n", "start_line": 1, "end_line": 1}))],
        )
        bare_answer = AssistantMessage(
            content="",
            tool_calls=[ToolCall(id="c2", name=answer_tool_name, arguments='{"text": "yes"}')],
        )
        aware_answer = AssistantMessage(
            content="Подтверждаю правку.",
            tool_calls=[ToolCall(id="c3", name=answer_tool_name, arguments='{"text": "yes"}')],
        )
        final_reply = AssistantMessage(content="Готово.")

        seen = []
        responses = [
            (edit_call, None, None),
            (bare_answer, None, None),
            (aware_answer, None, None),
            (final_reply, None, None),
        ]

        def fake_call(messages, prefill=None, **kwargs):
            seen.append(prefill)
            return responses.pop(0)

        with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake_call):
            agent = self.make_agent()
            result = agent.chat("Измени файл", max_iter=10)

        self.assertIn("Готово", result)
        with open(path, encoding="utf-8") as f:
            self.assertEqual(f.read(), "b\n")
        self.assertIsNone(agent._pending_operation)
        self.assertIn(_NO_COMMENT_PREFILL, seen)
        msgs = agent.history.get_all()
        texts = [getattr(m, "content", "") or "" for m in msgs]
        self.assertTrue(any("Подтверждаю правку" in t for t in texts))

    def test_rerun_prefill_survives_pending_operation(self):
        path = os.path.join(self._tmp, "l.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("a\n")

        edit_call = AssistantMessage(
            content="Отредактирую файл.",
            tool_calls=[ToolCall(id="c1", name=line_range_edit_tool_name, arguments=json.dumps({
                "path": path, "new_text": "b\n", "start_line": 1, "end_line": 1}))],
        )
        bare_edit = AssistantMessage(
            content="",
            tool_calls=[ToolCall(id="c2", name=line_range_edit_tool_name, arguments=json.dumps({
                "path": path, "new_text": "c\n", "start_line": 1, "end_line": 1}))],
        )
        text_no_answer = AssistantMessage(content="Продолжу без ответа.")
        answer_call = AssistantMessage(
            content="да", tool_calls=[ToolCall(id="c3", name=answer_tool_name, arguments='{"text": "yes"}')],
        )
        final_reply = AssistantMessage(content="Готово.")

        seen = []
        seen_content = []
        responses = [
            (edit_call, None, None),
            (bare_edit, None, None),
            (text_no_answer, None, None),
            (answer_call, None, None),
            (final_reply, None, None),
        ]

        def fake_call(messages, prefill=None, **kwargs):
            seen.append(prefill)
            seen_content.append([m.get("content", "") for m in messages if m.get("role") == "user"])
            return responses.pop(0)

        with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake_call):
            agent = self.make_agent()
            result = agent.chat("Измени файл", max_iter=10)

        self.assertIn("Готово", result)
        with open(path, encoding="utf-8") as f:
            self.assertEqual(f.read(), "b\n")
        self.assertIsNone(agent._pending_operation)
        self.assertIn(_NO_COMMENT_PREFILL, seen)
        self.assertTrue(any(_GUARD_NAG_PREFIX in t for content in seen_content for t in content))
        msgs = agent.history.get_all()
        self.assertEqual(nag_contents(msgs), [])

    def test_prefill_marker_alone_is_not_an_explanation(self):
        path = os.path.join(self._tmp, "m.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("a\n")

        edit_call = AssistantMessage(
            content="Отредактирую файл.",
            tool_calls=[ToolCall(id="c1", name=line_range_edit_tool_name, arguments=json.dumps({
                "path": path, "new_text": "b\n", "start_line": 1, "end_line": 1}))],
        )
        bare_answer = AssistantMessage(
            content="",
            tool_calls=[ToolCall(id="c2", name=answer_tool_name, arguments='{"text": "yes"}')],
        )
        prefilled_bare = AssistantMessage(
            content=_NO_COMMENT_PREFILL,
            tool_calls=[ToolCall(id="c3", name=answer_tool_name, arguments='{"text": "yes"}')],
        )
        aware_answer = AssistantMessage(
            content="Подтверждаю правку.",
            tool_calls=[ToolCall(id="c4", name=answer_tool_name, arguments='{"text": "yes"}')],
        )
        final_reply = AssistantMessage(content="Готово.")

        responses = [
            (edit_call, None, None),
            (bare_answer, None, None),
            (prefilled_bare, None, None),
            (aware_answer, None, None),
            (final_reply, None, None),
        ]

        with mock.patch(
            "universal_agents.agent.LLMClient.call",
            side_effect=responses[:],
        ):
            agent = self.make_agent()
            result = agent.chat("Измени файл", max_iter=10)

        self.assertIn("Готово", result)
        with open(path, encoding="utf-8") as f:
            self.assertEqual(f.read(), "b\n")
        msgs = agent.history.get_all()
        answers = [m for m in msgs if m.to_api_dict()["role"] == "tool" and m.name == answer_tool_name]
        self.assertEqual(len(answers), 1)

    def test_answer_without_pending_returns_error(self):
        agent = LLMAgent(
            system_prompt="You are helpful.",
            external_plugins={answer_tool_name: respond_to_system},
            disable_per_msg_summarization=True,
            autosave_enabled=False,
        )
        rogue_answer = AssistantMessage(
            content="Помогу!",
            tool_calls=[ToolCall(id="c1", name=answer_tool_name, arguments='{"text": "Привет"}')],
        )
        final_reply = AssistantMessage(content="Я умею работать с файлами.")

        with mock.patch(
            "universal_agents.agent.LLMClient.call",
            side_effect=[(rogue_answer, None, None), (final_reply, None, None)],
        ):
            result = agent.chat("Расскажи что умеешь", max_iter=10)

        self.assertIn("Я умею работать с файлами", result)
        msgs = agent.history.get_all()
        tool_results = [m.content for m in msgs if m.to_api_dict()["role"] == "tool"]
        self.assertTrue(any("requires a pending" in c for c in tool_results))
        self.assertTrue(all("Recorded" not in c for c in tool_results))

    def test_bare_answer_without_pending_does_not_loop(self):
        agent = LLMAgent(
            system_prompt="You are helpful.",
            external_plugins={answer_tool_name: respond_to_system},
            disable_per_msg_summarization=True,
            autosave_enabled=False,
        )
        bare_answer_tpl = {
            "content": "",
            "tool_calls": [ToolCall(id="c1", name=answer_tool_name, arguments='{"text": "yes"}')],
        }
        final_reply = AssistantMessage(content="Всё, готово.")

        seen = []
        retries = Config.NO_COMMENT_RETRIES

        def fresh_bare_answer():
            return (AssistantMessage(content=bare_answer_tpl["content"], tool_calls=list(bare_answer_tpl["tool_calls"])), None, None)

        responses = [fresh_bare_answer() for _ in range(retries + 1)] + [(final_reply, None, None)]

        def fake_call(messages, prefill=None, **kwargs):
            seen.append(prefill)
            return responses.pop(0)

        with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake_call):
            result = agent.chat("Привет", max_iter=10)

        self.assertIn("Всё, готово", result)
        self.assertEqual(seen.count(_NO_COMMENT_PREFILL), retries)
        msgs = agent.history.get_all()
        answers = [m for m in msgs if m.to_api_dict()["role"] == "tool" and m.name == answer_tool_name]
        self.assertEqual(len(answers), 1)
        self.assertTrue(any("requires a pending" in m.content for m in answers))

    def test_answer_with_pending_still_works(self):
        path = os.path.join(self._tmp, "p.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("a\n")

        edit_call = AssistantMessage(
            content="Отредактирую.",
            tool_calls=[ToolCall(id="c1", name=line_range_edit_tool_name, arguments=json.dumps({
                "path": path, "new_text": "b\n", "start_line": 1, "end_line": 1}))],
        )
        answer_yes = AssistantMessage(
            content="Подтверждаю.",
            tool_calls=[ToolCall(id="c2", name=answer_tool_name, arguments='{"text": "yes"}')],
        )
        final_reply = AssistantMessage(content="Готово.")

        with mock.patch(
            "universal_agents.agent.LLMClient.call",
            side_effect=[(edit_call, None, None), (answer_yes, None, None), (final_reply, None, None)],
        ):
            agent = self.make_agent()
            result = agent.chat("Измени p.txt", max_iter=10)

        self.assertIn("Готово", result)
        with open(path, encoding="utf-8") as f:
            self.assertEqual(f.read(), "b\n")
        msgs = agent.history.get_all()
        answers = [m for m in msgs if m.to_api_dict()["role"] == "tool" and m.name == answer_tool_name]
        self.assertEqual(len(answers), 1)
        self.assertFalse(any("requires a pending" in m.content for m in answers))

    def test_answer_guard_gives_up_after_max_retries(self):
        path = os.path.join(self._tmp, "g.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("a\n")

        edit_call = AssistantMessage(
            content="Отредактирую.",
            tool_calls=[ToolCall(id="c1", name=line_range_edit_tool_name, arguments=json.dumps({
                "path": path, "new_text": "b\n", "start_line": 1, "end_line": 1}))],
        )
        n_text = Config.ANSWER_GUARD_MAX_RETRIES + 1  # 6 текстов: 1-й..5-й перегенерируются, 6-й → сдача

        def fresh_text(i):
            return (AssistantMessage(content=f"Отвечу текстом, попытка {i}, без вызова инструмента."), None, None)

        responses = [(edit_call, None, None)] + [fresh_text(i) for i in range(1, n_text + 1)]
        calls = []

        def fake_call(messages, prefill=None, **kwargs):
            calls.append(messages)
            return responses.pop(0)

        with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake_call):
            agent = self.make_agent()
            result = agent.chat("Измени файл", max_iter=50)

        self.assertEqual(result, "")
        self.assertEqual(len(calls), 1 + n_text)
        self.assertIsNotNone(agent._pending_operation)
        msgs = agent.history.get_all()
        nags = [m for m in msgs if isinstance(m, UserMessage) and _GUARD_NAG_PREFIX in (m.content or "")]
        self.assertEqual(len(nags), 1)

    def test_wrong_attempt_scrubbed_after_success(self):
        path = os.path.join(self._tmp, "w.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("a\n")

        edit_call = AssistantMessage(
            content="Отредактирую.",
            tool_calls=[ToolCall(id="c1", name=line_range_edit_tool_name, arguments=json.dumps({
                "path": path, "new_text": "b\n", "start_line": 1, "end_line": 1}))],
        )
        wrong_call = AssistantMessage(
            content="Проверю файл перед подтверждением.",
            tool_calls=[ToolCall(id="c2", name=_read_tool.__name__, arguments=json.dumps({"path": path}))],
        )
        answer_call = AssistantMessage(
            content="Подтверждаю.",
            tool_calls=[ToolCall(id="c3", name=answer_tool_name, arguments='{"text": "yes"}')],
        )
        final_reply = AssistantMessage(content="Готово.")

        with mock.patch(
            "universal_agents.agent.LLMClient.call",
            side_effect=[(edit_call, None, None), (wrong_call, None, None), (answer_call, None, None), (final_reply, None, None)],
        ):
            agent = self.make_agent()
            result = agent.chat("Измени файл", max_iter=10)

        self.assertIn("Готово", result)
        self.assertIsNone(agent._pending_operation)
        msgs = agent.history.get_all()
        contents = [getattr(m, "content", "") or "" for m in msgs]
        self.assertTrue(any("ATTENTION" in c for c in contents), "Превью-превью правки должно остаться")
        self.assertTrue(any("Подтверждаю" in c for c in contents))
        wrong_assistant = [m for m in msgs
                           if isinstance(m, AssistantMessage) and any(tc.name == _read_tool.__name__ for tc in m.tool_calls)]
        wrong_results = [m for m in msgs if isinstance(m, ToolResult) and m.name in (_read_tool.__name__,)]
        self.assertEqual(wrong_assistant, [], "Вызов чужого инструмента при pending должен быть вычищен")
        self.assertEqual(wrong_results, [], "Результат чужого инструмента при pending должен быть вычищен")
        self.assertEqual(nag_contents(msgs), [])

    def test_guard_nag_is_byte_stable_between_llm_calls(self):
        path = os.path.join(self._tmp, "s.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("a\n")

        edit_call = AssistantMessage(
            content="Отредактирую.",
            tool_calls=[ToolCall(id="c1", name=line_range_edit_tool_name, arguments=json.dumps({
                "path": path, "new_text": "b\n", "start_line": 1, "end_line": 1}))],
        )
        text1 = AssistantMessage(content="Не вызываю инструмент.")
        text2 = AssistantMessage(content="Всё ещё не вызываю.")
        answer_call = AssistantMessage(
            content="Подтверждаю.",
            tool_calls=[ToolCall(id="c3", name=answer_tool_name, arguments='{"text": "yes"}')],
        )
        final_reply = AssistantMessage(content="Готово.")

        user_contents = []
        responses = [
            (edit_call, None, None),
            (text1, None, None),
            (text2, None, None),
            (answer_call, None, None),
            (final_reply, None, None),
        ]

        def fake_call(messages, prefill=None, **kwargs):
            user_contents.append([m.get("content", "") for m in messages if m.get("role") == "user"])
            return responses.pop(0)

        with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake_call):
            agent = self.make_agent()
            result = agent.chat("Измени файл", max_iter=10)

        self.assertIn("Готово", result)
        self.assertEqual(len(user_contents), 5)
        self.assertGreaterEqual(len(user_contents[2]), 1)
        self.assertEqual(user_contents[2], user_contents[3],
                         "User-часть префикса должна быть байт-стабильной между срабатываниями guard'а")
        for content in user_contents:
            nags = [t for t in content if _GUARD_NAG_PREFIX in t]
            self.assertLessEqual(len(nags), 1, "Наг не должен накапливаться")

    def test_scrub_after_answer_keeps_prefix_byte_stable(self):
        path = os.path.join(self._tmp, "p.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("a\n")

        edit_call = AssistantMessage(
            content="Отредактирую.",
            tool_calls=[ToolCall(id="c1", name=line_range_edit_tool_name, arguments=json.dumps({
                "path": path, "new_text": "b\n", "start_line": 1, "end_line": 1}))],
        )
        text_turn = AssistantMessage(content="Подтверждаю правку.")
        answer_call = AssistantMessage(
            content="Подтверждаю.",
            tool_calls=[ToolCall(id="c2", name=answer_tool_name, arguments='{"text": "yes"}')],
        )
        final_reply = AssistantMessage(content="Готово.")

        user_contents = []
        responses = [
            (edit_call, None, None),
            (text_turn, None, None),
            (answer_call, None, None),
            (final_reply, None, None),
        ]
        attempt = {"n": 0}

        def fake_call(messages, prefill=None, **kwargs):
            user_contents.append([m.get("content", "") for m in messages if m.get("role") == "user"])
            msg, err, _ = responses.pop(0)
            attempt["n"] += 1
            return msg, err, {"prompt_tokens": 1000 * attempt["n"], "completion_tokens": 10,
                              "total_tokens": 1000 * attempt["n"]}

        system_msgs = []
        with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake_call):
            agent = self.make_agent()
            agent.on_system_msg = system_msgs.append
            result = agent.chat("Измени файл", max_iter=10)

        self.assertIn("Готово", result)
        self.assertEqual([m for m in system_msgs if "[PREFIX-HASH]" in m], [],
                         "Вычистка нага не должна сдвигать префикс")
        self.assertEqual(len(user_contents), 4)
        self.assertEqual(user_contents[0], user_contents[1])
        self.assertEqual(user_contents[1][0], user_contents[3][0],
                         "User-сообщение задачи должно вернуться байт-в-байт после вычистки нага")
        self.assertEqual(nag_contents(agent.history.get_all()), [])

    def test_guard_nag_flag_roundtrip_through_save_load(self):
        from universal_agents.history import ChatHistory

        hist = ChatHistory("sys")
        hist.add(UserMessage("q"))
        nag = UserMessage("You can't continue — use answer tool")
        nag._is_guard_nag = True
        hist.add(nag)

        self.assertNotIn("_is_guard_nag", nag.to_api_dict())
        self.assertTrue(nag.to_persist_dict()["_is_guard_nag"])

        payload = {
            "messages": [m.to_persist_dict() for m in hist.get_all()],
            "loaded_tools": [], "file_states": {}, "per_msg_summaries": [], "next_seq": 4, "extras": {},
        }
        hist2 = ChatHistory("sys")
        hist2.load_from_payload(payload)
        self.assertTrue(
            any(isinstance(m, UserMessage) and m._is_guard_nag for m in hist2.get_all()),
            "Флаг нага должен пережить /load",
        )

        agent = self.make_agent()
        agent.history.load_from_payload(payload)
        removed = agent._scrub_confirmation_trail()
        self.assertGreater(removed, 0)
        self.assertFalse(
            any(isinstance(m, UserMessage) and m._is_guard_nag for m in agent.history.get_all()),
            "После успешного answer скраб удаляет загруженный наг",
        )


@tool(description="fake screenshot tool")
def fake_screenshot() -> ToolOutput:
    return ToolOutput(text="Скриншот 640x480", images=[B64])


@tool(description="plain text tool")
def plain_tool() -> str:
    return "просто текст"


class TestToolOutputExecution(unittest.TestCase):

    def _agent(self):
        return make_test_agent(
            tools_config=["fake_screenshot", "plain_tool"],
            external_plugins={"fake_screenshot": fake_screenshot, "plain_tool": plain_tool},
        )

    def test_tool_output_unwrapped_into_tool_result(self):
        agent = self._agent()
        results = agent._execute_tools([ToolCall(id="t1", name="fake_screenshot", arguments="{}")])
        tr = results[0]
        self.assertFalse(tr.is_error)
        self.assertEqual(tr.content, "Скриншот 640x480")
        self.assertEqual(tr.images, [B64])
        self.assertTrue(tr.skip_summarize)

    def test_plain_str_tool_unchanged(self):
        agent = self._agent()
        results = agent._execute_tools([ToolCall(id="t1", name="plain_tool", arguments="{}")])
        tr = results[0]
        self.assertEqual(tr.content, "просто текст")
        self.assertEqual(tr.images, [])

    def test_tool_output_flows_to_api_dict(self):
        agent = self._agent()
        results = agent._execute_tools([ToolCall(id="t1", name="fake_screenshot", arguments="{}")])
        d = results[0].to_api_dict()
        self.assertIsInstance(d["content"], list)
        self.assertEqual(d["content"][0]["type"], "text")
        self.assertEqual(d["content"][1]["type"], "image_url")


if __name__ == "__main__":
    unittest.main()