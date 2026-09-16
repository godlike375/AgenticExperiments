from __future__ import annotations

"""Тесты унифицированного служебного хода (service_llm_call = один ход через общий движок).

Проверяются инварианты решения пользователя:
- промпт добавляется в историю как UserMessage, генерируется одно ассистентское сообщение;
- инструменты в служебном ходе НЕ исполняются (текст сохраняется, голый вызов — перегенерация);
- автозапись в служебном ходе отключена;
- after завершения промпт (и ответ) откатываются из истории (Config.SERVICE_TURN_KEEP_IN_HISTORY=>>> get)
- consistent: история байт-точно восстанавливается, объединённый префикс для KV-cache не страдает.
"""

import unittest
from unittest import mock

from universal_agents.agent import LLMAgent
from universal_agents.config import Config
from universal_agents.context_builder import prepare_messages_for_api
from universal_agents.models import AssistantMessage, ToolCall, ToolResult, UserMessage

from tests.conftest import make_agent


class TestServiceTurn(unittest.TestCase):

    def _base_history(self, agent: LLMAgent) -> None:
        """Подготовка стабильного состояния истории (системный промпт уже на месте)."""
        agent.history.add(UserMessage("вопрос"))
        agent.history.add(AssistantMessage(content="ответ"))

    def test_one_generation_and_rollback(self):
        agent = make_agent()
        self._base_history(agent)
        hist_before = [m.content for m in agent.history.get_all()]
        with mock.patch("universal_agents.agent.LLMClient.call",
                        return_value=(AssistantMessage(content="выжимка"), None, None)):
            msg_obj, err, usage = agent.service_llm_call([{"role": "user", "content": "sami"}])
        self.assertIsNotNone(msg_obj)
        self.assertEqual(msg_obj.content, "выжимка")
        self.assertIsNone(err)
        self.assertIsNone(usage)
        hist_after = [m.content for m in agent.history.get_all()]
        # Default (Config.SERVICE_TURN_KEEP_IN_HISTORY=False): промпт и ответ откачены.
        self.assertEqual(hist_after, hist_before)

    def test_keep_in_history_via_config(self):
        agent = make_agent()
        self._base_history(agent)
        with mock.patch.object(Config, "SERVICE_TURN_KEEP_IN_HISTORY", True), \
                mock.patch("universal_agents.agent.LLMClient.call",
                           return_value=(AssistantMessage(content="выжимка"), None, None)):
            msg_obj, err, _ = agent.service_llm_call([{"role": "user", "content": "sami"}])
        self.assertEqual(msg_obj.content, "выжимка")
        contents = [m.content for m in agent.history.get_all()]
        # Промпт сервисного вызова остался в истории после завершения.
        self.assertIn("sami", contents)
        self.assertIn("выжимка", contents)

    def test_append_prompt_dedup_when_already_last_user(self):
        # Consistency-драфт: промпт равен последнему user-сообщению истории —
        # он не должен дублироваться (иначе normalize склеит и испортит контент).
        agent = make_agent()
        agent.history.add(UserMessage("текущий вопрос"))
        agent.history.add(AssistantMessage(content="ответ на текущий вопрос"))
        with mock.patch("universal_agents.agent.LLMClient.call",
                        return_value=(AssistantMessage(content="draft"), None, None)) as mocked:
            msg_obj, _, _ = agent.service_llm_call([
                {"role": "user", "content": "текущий вопрос"},
            ])
        self.assertEqual(msg_obj.content, "draft")
        sent_messages = mocked.call_args[0][0]
        user_contents = [m["content"] for m in sent_messages if m["role"] == "user"]
        # Промпт не продублирован: в API-запросе ровно один user-контент, содержащий промпт
        # (шапка <SYSTEM> в начале контента не должна считаться отдельным сообщением).
        self.assertEqual(sum("текущий вопрос" in (c or "") for c in user_contents), 1)

    def test_tools_never_executed_text_kept(self):
        # Модель вернула tool_calls вместе с текстом: инструменты не исполняются,
        # остаётся только текст (решение пользователя).
        agent = make_agent()
        self._base_history(agent)
        tc = ToolCall(id="call_1", name="some_tool", arguments="{}")
        with mock.patch("universal_agents.agent.LLMClient.call",
                        return_value=(AssistantMessage(content="текст с объяснением", tool_calls=[tc]), None, None)), \
                mock.patch.object(agent, "_execute_tools", side_effect=AssertionError("tools executed")) as ex:
            msg_obj, err, _ = agent.service_llm_call([{"role": "user", "content": "prompt"}])
        ex.assert_not_called()
        self.assertEqual(msg_obj.content, "текст с объяснением")
        self.assertIsNone(err)
        # Никаких ToolResult в истории от служебного хода.
        self.assertFalse(any(isinstance(m, ToolResult) for m in agent.history.get_all()))

    def test_bare_tool_call_regenerates(self):
        # Голый вызов инструмента без текста: NO COMMENT-механика перегенерирует,
        # после чего возвращается итоговый текст.
        agent = make_agent()
        self._base_history(agent)
        tc = ToolCall(id="call_1", name="some_tool", arguments="{}")
        responses = [
            (AssistantMessage(content="", tool_calls=[tc]), None, None),
            (AssistantMessage(content="финальный текст"), None, None),
        ]

        def fake_call(messages, prefill=None, **kwargs):
            return responses.pop(0)

        with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake_call):
            msg_obj, err, _ = agent.service_llm_call([{"role": "user", "content": "prompt"}])
        self.assertEqual(msg_obj.content, "финальный текст")
        self.assertIsNone(err)
        # Ответы служебного хода откачены — история вернулась к базовой (системный + вопрос + ответ).
        self.assertEqual([m.content for m in agent.history.get_all()],
                         ["You are a helpful assistant", "вопрос", "ответ"])

    def test_autosave_and_per_msg_summarization_disabled(self):
        agent = make_agent()
        self._base_history(agent)
        with mock.patch("universal_agents.agent.LLMClient.call",
                        return_value=(AssistantMessage(content="выжимка"), None, None)), \
                mock.patch.object(agent, "autosave") as save, \
                mock.patch.object(agent, "_maybe_summarize_user_message") as summ:
            agent.service_llm_call([{"role": "user", "content": "prompt"}])
        save.assert_not_called()
        summ.assert_not_called()

    def test_rollback_preserves_last_user_header_cache(self):
        # Регрессия [PREFIX-HASH]: откат служебного хода через history.remove_at
        # сбрасывал кэш-заголовок последнего user-сообщения (_resync_last_user_header
        # видел «смену последности», хотя история вернулась байт-идентично), и
        # следующий prepare_messages_for_api пересобирал шапку с новым токен-бюджетом
        # → hash сообщения менялся → KV-кэш пересобирался.
        agent = make_agent()
        self._base_history(agent)
        # Тёплый кэш: последний user («вопрос») уже получил шапку (момент А).
        msgs_a = prepare_messages_for_api(agent, debug_hash_check=True)
        last_user_a = msgs_a[1]["content"]
        with mock.patch("universal_agents.agent.LLMClient.call",
                        return_value=(AssistantMessage(content="выжимка"), None, None)):
            agent.service_llm_call([{"role": "user", "content": "sami"}])
        # После отката prepare даёт байт-идентичный префикс: шапка не пересобрана.
        msgs_b = prepare_messages_for_api(agent, debug_hash_check=True)
        self.assertEqual(msgs_b[1]["content"], last_user_a)


if __name__ == "__main__":
    unittest.main()