import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from universal_agents.agent import LLMAgent
from universal_agents.agent_mixins.response_mixin import _NO_COMMENT_PREFILL
from universal_agents.config import Config
from universal_agents.models import AssistantMessage, ToolCall
from universal_agents.tools.fs import line_range_edit
from universal_agents.tools.builtin import answer_to_system

from tests.conftest import make_agent as make_test_agent

answer_tool_name = answer_to_system.__name__
line_range_edit_tool_name = line_range_edit.__name__


class TestAnswerRequiredGuard(unittest.TestCase):
    """Если после edit-инструмента модель ответила текстом без вызова 'answer_to_system',
    цикл должен вколоть ошибку и продолжить, пока answer не будет вызван."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)

    def make_agent(self):
        agent = make_test_agent(
            system_prompt="You edit files. Always call answer to confirm edits.",
            external_plugins={line_range_edit_tool_name: line_range_edit, answer_tool_name: answer_to_system},
        )
        agent.trust_dir(self._tmp)
        return agent

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
        пока она наконец не вызовет answer — здесь answer_to_system('no'), правка отменяется."""
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
            self.assertEqual(f.read(), "b")
        self.assertIsNone(agent._pending_operation)
        # Голый answer был перегенерирован: до следующего вызова LLM дошёл prefill 'Assistant:'.
        self.assertIn(_NO_COMMENT_PREFILL, seen)
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
        # Применена первая правка (ответ 'yes'), голый повторный line_range_edit не исполнился.
        with open(path, encoding="utf-8") as f:
            self.assertEqual(f.read(), "b")
        self.assertIsNone(agent._pending_operation)
        # Prefill 'Assistant:' дошёл до следующего вызова LLM — guard его не съел.
        self.assertIn(_NO_COMMENT_PREFILL, seen)
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
            self.assertEqual(f.read(), "b")
        # Исполнился ровно один answer — только осознанный (с настоящим текстом).
        msgs = agent.history.get_all()
        answers = [m for m in msgs if m.to_api_dict()["role"] == "tool" and m.name == answer_tool_name]
        self.assertEqual(len(answers), 1)

    def test_answer_without_pending_returns_error(self):
        """answer_to_system() без вопроса от системы — ошибка, а не молчаливое 'Recorded'.
        Модель должна видеть явный отказ и ответить текстом."""
        agent = LLMAgent(
            system_prompt="You are helpful.",
            external_plugins={answer_tool_name: answer_to_system},
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
        # В истории остался результат инструмента с явной ошибкой (не 'Recorded').
        msgs = agent.history.get_all()
        tool_results = [m.content for m in msgs if m.to_api_dict()["role"] == "tool"]
        self.assertTrue(any("requires a pending" in c for c in tool_results))
        self.assertTrue(all("Recorded" not in c for c in tool_results))

    def test_bare_answer_without_pending_does_not_loop(self):
        """Голый answer_to_system() без pending-операции перегенерируется через
        [NO COMMENT] не больше NO_COMMENT_RETRIES раз, затем принимается как есть:
        инструмент возвращает ошибку, и цикл завершается текстовым ответом — без
        бесконечного зацикливания."""
        agent = LLMAgent(
            system_prompt="You are helpful.",
            external_plugins={answer_tool_name: answer_to_system},
            disable_per_msg_summarization=True,
            autosave_enabled=False,
        )
        bare_answer_tpl = {
            "content": "",
            "tool_calls": [ToolCall(id="c1", name=answer_tool_name, arguments='{"text": "yes"}')],
        }
        final_reply = AssistantMessage(content="Всё, готово.")

        seen = []
        # Первые (NO_COMMENT_RETRIES+1) сообщений — голый answer: первые N отвергаются
        # и перегенерируются с prefill, последний исполняется как есть (ретраи исчерпаны)
        # и возвращает ошибку инструмента. Объекты создаём заново: обработчик мутирует
        # message_obj (стирает tool_calls) при каждом rerun.
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
        # Голый answer перегенерировался ровно NO_COMMENT_RETRIES раз с prefill.
        self.assertEqual(seen.count(_NO_COMMENT_PREFILL), retries)
        # Инструмент исполнился один раз (когда ретраи кончились) и вернул ошибку.
        msgs = agent.history.get_all()
        answers = [m for m in msgs if m.to_api_dict()["role"] == "tool" and m.name == answer_tool_name]
        self.assertEqual(len(answers), 1)
        self.assertTrue(any("requires a pending" in m.content for m in answers))

    def test_answer_with_pending_still_works(self):
        """answer_to_system() после вопроса системы (pending) работает как раньше."""
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
            self.assertEqual(f.read(), "b")
        # Правка применена — answer с pending не вернул ошибку.
        msgs = agent.history.get_all()
        answers = [m for m in msgs if m.to_api_dict()["role"] == "tool" and m.name == answer_tool_name]
        self.assertEqual(len(answers), 1)
        self.assertFalse(any("requires a pending" in m.content for m in answers))

if __name__ == "__main__":
    unittest.main()