import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from universal_agents.agent import LLMAgent
from universal_agents.agent_mixins.response_mixin import _NO_COMMENT_PREFILL
from universal_agents.config import Config
from universal_agents.models import AssistantMessage, ToolCall, ToolResult, UserMessage
from universal_agents.tools.fs import line_range_edit
from universal_agents.tools.builtin import answer_to_system

from tests.conftest import make_agent as make_test_agent

answer_tool_name = answer_to_system.__name__
line_range_edit_tool_name = line_range_edit.__name__


def nag_contents(msgs) -> list:
    """Контент нагов guard'а (UserMessage «You can't continue...») в истории.
    Превью edit-инструмента содержит «You can't continue with common prose», но это
    ToolResult, а не наг — поэтому смотрим только на UserMessage."""
    return [m.content for m in msgs
            if isinstance(m, UserMessage) and "You can't continue" in (m.content or "")]


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
        # Правка применена только после настоящего вызова answer.
        with open(path, encoding="utf-8") as f:
            self.assertEqual(f.read(), "world")

        # Наг guard'а МОДЕЛЬ видела (между срабатыванием и успехом), но после успеха
        # он вычищен — в контексте остаётся только правильный путь подтверждения (§1.11).
        self.assertTrue(any("You can't continue" in t for content in seen for t in content),
                        "Модель должна была получить сообщение-ошибку о вызове answer")
        msgs = agent.history.get_all()
        self.assertEqual(nag_contents(msgs), [],
                         "После успешного answer_to_system наг должен быть вычищен из истории")

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
        # Отмена: файл не тронут, pending-операция снята.
        with open(path, encoding="utf-8") as f:
            self.assertEqual(f.read(), "a\n")
        self.assertIsNone(agent._pending_operation)
        # Модель получала наг «call answer» минимум один раз...
        self.assertTrue(any("You can't continue" in t for content in seen for t in content))
        # ...но после успешного подтверждения (даже 'no') наг вычищен из истории (§1.11).
        msgs = agent.history.get_all()
        self.assertEqual(nag_contents(msgs), [])

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
        # Применена первая правка (ответ 'yes'), голый повторный line_range_edit не исполнился.
        with open(path, encoding="utf-8") as f:
            self.assertEqual(f.read(), "b")
        self.assertIsNone(agent._pending_operation)
        # Prefill 'Assistant:' дошёл до следующего вызова LLM — guard его не съел.
        self.assertIn(_NO_COMMENT_PREFILL, seen)
        # Guard всё равно сработал после перегенерации (модель так и не ответила)
        # и был вычищен после успешного подтверждения (§1.11).
        self.assertTrue(any("You can't continue" in t for content in seen_content for t in content))
        msgs = agent.history.get_all()
        self.assertEqual(nag_contents(msgs), [])

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

    def test_answer_guard_gives_up_after_max_retries(self):
        """Guard без лимита зацикливал ход вечно (текст → наг → текст → ... до max_iter).
        Теперь после ANSWER_GUARD_MAX_RETRIES срабатываний ход сдаётся пользователю:
        result == '', лишних вызовов LLM нет (ровно edit + MAX_RETRIES+1 текстов)."""
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

        # Guard сдал ход пользователю ровно после MAX_RETRIES+1 текстового ответа.
        self.assertEqual(result, "")
        self.assertEqual(len(calls), 1 + n_text)
        # Операция так и не подтверждена — pending висит, решать будет пользователь.
        self.assertIsNotNone(agent._pending_operation)
        # В истории ровно ОДИН наг (переиспользуется, не копится между срабатываниями).
        msgs = agent.history.get_all()
        nags = [m for m in msgs if isinstance(m, UserMessage) and "You can't continue" in (m.content or "")]
        self.assertEqual(len(nags), 1)

    def test_wrong_attempt_scrubbed_after_success(self):
        """Неверная попытка (чужой инструмент при висящем pending) и наг вычищаются после
        успешного answer_to_system: в истории остаётся только правильный путь подтверждения
        (превью edit → answer → результат), а не чередование ошибок (§1.11)."""
        path = os.path.join(self._tmp, "w.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("a\n")

        edit_call = AssistantMessage(
            content="Отредактирую.",
            tool_calls=[ToolCall(id="c1", name=line_range_edit_tool_name, arguments=json.dumps({
                "path": path, "new_text": "b\n", "start_line": 1, "end_line": 1}))],
        )
        # Модель застряла: вместо answer вызывает чужой инструмент (read).
        wrong_call = AssistantMessage(
            content="Проверю файл перед подтверждением.",
            tool_calls=[ToolCall(id="c2", name="read", arguments=json.dumps({"path": path}))],
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
        # Правильный путь подтверждения остался: превью edit ("ATTENTION") и answer.
        msgs = agent.history.get_all()
        contents = [getattr(m, "content", "") or "" for m in msgs]
        self.assertTrue(any("ATTENTION" in c for c in contents), "Превью-превью правки должно остаться")
        self.assertTrue(any("Подтверждаю" in c for c in contents))
        # Мусор вычищен: чужих вызовов read (assistant + результат) и нагов нет.
        wrong_assistant = [m for m in msgs
                           if isinstance(m, AssistantMessage) and any(tc.name == "read" for tc in m.tool_calls)]
        wrong_results = [m for m in msgs if isinstance(m, ToolResult) and m.name in ("read",)]
        self.assertEqual(wrong_assistant, [], "Вызов чужого инструмента при pending должен быть вычищен")
        self.assertEqual(wrong_results, [], "Результат чужого инструмента при pending должен быть вычищен")
        self.assertEqual(nag_contents(msgs), [])

    def test_guard_nag_is_byte_stable_between_llm_calls(self):
        """Повторные срабатывания guard'а переиспользуют ОДИН объект нага: user-часть
        префикса байт-идентична между вызовами LLM (header-кэш не сбрасывается, ложных
        [PREFIX-HASH] нет). Прежний _drop_guard_nags плодил новый UserMessage со свежим
        timestamp — содержимое user-сообщений между вызовами расходилось."""
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
        # Вызовы: 1=edit, 2=text1, 3=text2, 4=answer, 5=final. Наг добавлен после вызова 2
        # (guard №1), переиспользован перед вызовом 4 (guard №2) — user-префикс обязан
        # быть байт-идентичным между вызовами 3 и 4.
        self.assertEqual(len(user_contents), 5)
        self.assertGreaterEqual(len(user_contents[2]), 1)
        self.assertEqual(user_contents[2], user_contents[3],
                         "User-часть префикса должна быть байт-стабильной между срабатываниями guard'а")
        # И сам наг в префиксе ровно один (не накапливается).
        for content in user_contents:
            nags = [t for t in content if "You can't continue" in t]
            self.assertLessEqual(len(nags), 1, "Наг не должен накапливаться")

    def test_guard_nag_flag_roundtrip_through_save_load(self):
        """Флаг _is_guard_nag не уходит в API, но переживает save/load — скраб находит
        наг и после перезагрузки (объекты пересозданы, identity-проверка бы не сработала)."""
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

if __name__ == "__main__":
    unittest.main()