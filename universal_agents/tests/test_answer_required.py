import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from universal_agents.agent import LLMAgent
from universal_agents.models import AssistantMessage, ToolCall
from universal_agents.tools.fs import edit_file
from universal_agents.tools.builtin import answer


class TestAnswerRequiredGuard(unittest.TestCase):
    """Если после edit_file модель ответила текстом без вызова 'answer',
    цикл должен вколоть ошибку и продолжить, пока answer не будет вызван."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)

    def make_agent(self):
        agent = LLMAgent(
            system_prompt="You edit files. Always call answer to confirm edits.",
            tools_config=None,
            external_plugins={"edit_file": edit_file, "answer": answer},
            disable_per_msg_summarization=True,
            autosave_enabled=False,
        )
        agent.trust_dir(self._tmp)
        return agent

    def test_text_answer_without_answer_tool_then_answer_yes(self):
        path = os.path.join(self._tmp, "hello.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("hello\n")

        edit_call = AssistantMessage(
            content="Отредактирую файл.",
            tool_calls=[ToolCall(id="c1", name="edit_file", arguments=json.dumps({
                "path": path,
                "new_text": "world\n",
                "start_line": 1,
                "end_line": 1,
            }))],
        )
        text_turn = AssistantMessage(content="Я отредактирую файл и подтверждаю правку, всё хорошо.")
        answer_call = AssistantMessage(
            content="Подтверждаю.",
            tool_calls=[ToolCall(id="c2", name="answer", arguments='{"text": "yes"}')],
        )

        final_reply = AssistantMessage(content="Операция завершена.")
        with mock.patch(
            "universal_agents.agent.LLMClient.call",
            side_effect=[(edit_call, None, None), (text_turn, None, None), (answer_call, None, None), (final_reply, None, None)],
        ):
            agent = self.make_agent()
            result = agent.chat("Отредактируй hello.txt: замени hello на world", max_iter=10)

        self.assertIn("Операция завершена", result)
        self.assertIsNone(agent._pending_operation)
        # Правка применена только после настоящего вызова answer.
        with open(path, encoding="utf-8") as f:
            self.assertEqual(f.read(), "world")

        # В истории есть user-сообщение-ошибка «call answer» и системный алерт.
        msgs = agent.history.get_all()
        texts = [getattr(m, "content", "") or "" for m in msgs]
        self.assertTrue(any("answer" in t and "You can't continue" in t for t in texts),
                        "Модель должна была получить сообщение-ошибку о вызове answer")

    def test_text_answer_without_answer_tool_then_text_again_no_message(self):
        """Если модель упорно не вызывает answer, цикл продолжается (не обрывается),
        пока она наконец не вызовет answer — здесь answer('no'), правка отменяется."""
        path = os.path.join(self._tmp, "x.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("a\n")

        edit_call = AssistantMessage(
            content="Отредактирую файл.",
            tool_calls=[ToolCall(id="c1", name="edit_file", arguments=json.dumps({
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
            tool_calls=[ToolCall(id="c2", name="answer", arguments='{"text": "no"}')],
        )
        final_reply = AssistantMessage(content="Готово.")

        with mock.patch(
            "universal_agents.agent.LLMClient.call",
            side_effect=[(edit_call, None, None), (text1, None, None), (text2, None, None), (answer_no, None, None), (final_reply, None, None)],
        ):
            agent = self.make_agent()
            result = agent.chat("Измени файл", max_iter=10)

        self.assertIn("Готово", result)
        # Отмена: файл не тронут, pending-операция снята.
        with open(path, encoding="utf-8") as f:
            self.assertEqual(f.read(), "a\n")
        self.assertIsNone(agent._pending_operation)
        # Модель получала наг «call answer» минимум один раз.
        msgs = agent.history.get_all()
        texts = [getattr(m, "content", "") or "" for m in msgs]
        self.assertGreaterEqual(
            sum(1 for t in texts if "You can't continue" in t and "answer" in t), 1)

    def test_bare_answer_is_regenerated_until_model_adds_comment(self):
        """Модель ОБЯЗАНА написать текст перед вызовом answer (ответ системе —
        осознанный). Голый answer перегенерируется с 'Assistant:', а исполняется
        только тот, что сопровождается реальным текстом."""
        path = os.path.join(self._tmp, "k.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("a\n")

        edit_call = AssistantMessage(
            content="Отредактирую файл.",
            tool_calls=[ToolCall(id="c1", name="edit_file", arguments=json.dumps({
                "path": path, "new_text": "b\n", "start_line": 1, "end_line": 1}))],
        )
        bare_answer = AssistantMessage(
            content="",
            tool_calls=[ToolCall(id="c2", name="answer", arguments='{"text": "yes"}')],
        )
        aware_answer = AssistantMessage(
            content="Подтверждаю правку.",
            tool_calls=[ToolCall(id="c3", name="answer", arguments='{"text": "yes"}')],
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
            self.assertEqual(f.read(), "b")
        self.assertIsNone(agent._pending_operation)
        # Голый answer был перегенерирован: до следующего вызова LLM дошёл prefill 'Assistant:'.
        self.assertIn("Assistant:", seen)
        # В истории есть только осознанный ответ модели (с текстом), а не голый вызов.
        msgs = agent.history.get_all()
        texts = [getattr(m, "content", "") or "" for m in msgs]
        self.assertTrue(any("Подтверждаю правку" in t for t in texts))

    def test_rerun_prefill_survives_pending_operation(self):
        """Перегенерация [NO COMMENT] с 'Assistant:' не должна съедаться guard'ом
        pending-операции: prefill обязан дойти до следующего вызова LLM, и только
        потом guard напомнит про answer (если модель так и не ответила)."""
        path = os.path.join(self._tmp, "l.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("a\n")

        edit_call = AssistantMessage(
            content="Отредактирую файл.",
            tool_calls=[ToolCall(id="c1", name="edit_file", arguments=json.dumps({
                "path": path, "new_text": "b\n", "start_line": 1, "end_line": 1}))],
        )
        bare_edit = AssistantMessage(
            content="",
            tool_calls=[ToolCall(id="c2", name="edit_file", arguments=json.dumps({
                "path": path, "new_text": "c\n", "start_line": 1, "end_line": 1}))],
        )
        text_no_answer = AssistantMessage(content="Продолжу без ответа.")
        answer_call = AssistantMessage(
            content="да", tool_calls=[ToolCall(id="c3", name="answer", arguments='{"text": "yes"}')],
        )
        final_reply = AssistantMessage(content="Готово.")

        seen = []
        responses = [
            (edit_call, None, None),
            (bare_edit, None, None),
            (text_no_answer, None, None),
            (answer_call, None, None),
            (final_reply, None, None),
        ]

        def fake_call(messages, prefill=None, **kwargs):
            seen.append(prefill)
            return responses.pop(0)

        with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake_call):
            agent = self.make_agent()
            result = agent.chat("Измени файл", max_iter=10)

        self.assertIn("Готово", result)
        # Применена первая правка (ответ 'yes'), голый повторный edit_file не исполнился.
        with open(path, encoding="utf-8") as f:
            self.assertEqual(f.read(), "b")
        self.assertIsNone(agent._pending_operation)
        # Prefill 'Assistant:' дошёл до следующего вызова LLM — guard его не съел.
        self.assertIn("Assistant:", seen)
        # Guard всё равно сработал после перегенерации (модель так и не ответила).
        msgs = agent.history.get_all()
        texts = [getattr(m, "content", "") or "" for m in msgs]
        self.assertTrue(any("You can't continue" in t and "answer" in t for t in texts))

    def test_prefill_marker_alone_is_not_an_explanation(self):
        """Впрыснутый 'Assistant:' сам по себе не считается пояснением: если модель
        после prefill снова выдаёт голый answer (content='Assistant:'), вызов снова
        отвергается — исполняется только answer с настоящим текстом."""
        path = os.path.join(self._tmp, "m.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("a\n")

        edit_call = AssistantMessage(
            content="Отредактирую файл.",
            tool_calls=[ToolCall(id="c1", name="edit_file", arguments=json.dumps({
                "path": path, "new_text": "b\n", "start_line": 1, "end_line": 1}))],
        )
        bare_answer = AssistantMessage(
            content="",
            tool_calls=[ToolCall(id="c2", name="answer", arguments='{"text": "yes"}')],
        )
        prefilled_bare = AssistantMessage(
            content="Assistant:",
            tool_calls=[ToolCall(id="c3", name="answer", arguments='{"text": "yes"}')],
        )
        aware_answer = AssistantMessage(
            content="Подтверждаю правку.",
            tool_calls=[ToolCall(id="c4", name="answer", arguments='{"text": "yes"}')],
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
            self.assertEqual(f.read(), "b")
        # Исполнился ровно один answer — только осознанный (с настоящим текстом).
        msgs = agent.history.get_all()
        answers = [m for m in msgs if m.to_api_dict()["role"] == "tool" and m.name == "answer"]
        self.assertEqual(len(answers), 1)

if __name__ == "__main__":
    unittest.main()