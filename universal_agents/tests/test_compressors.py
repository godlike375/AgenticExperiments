import unittest
from unittest import mock
from types import SimpleNamespace
from universal_agents.compressors import (
    is_summary_message,
    _find_existing_summary,
    _build_draft_prompt,
    summarize_dialogue,
)
from universal_agents.history import ChatHistory
from universal_agents.models import UserMessage, AssistantMessage, ToolResult
from universal_agents.config import Config
from universal_agents.llm_client import LLMClient


def _fake_msg(content="... summary ..."):
    return SimpleNamespace(content=content, tool_calls=None)


def _history():
    h = ChatHistory("sys")
    h.add(UserMessage("task: create readme"))
    h.add(AssistantMessage(content="a1"))
    h.add(ToolResult.success("t1", "read", "some file content"))
    h.add(AssistantMessage(content="a2"))
    h.add(UserMessage("please summarize"))
    return h


def _history_with_summary():
    h = ChatHistory("sys")
    h.add(UserMessage("old facts\nKEY FACTS: src/main.py\nold decision X", is_summary=True))
    h.add(UserMessage("new: fix bug in main.py"))
    h.add(AssistantMessage(content="fixed"))
    return h


class TestSummaryHelpers(unittest.TestCase):
    def test_is_summary_message(self):
        # Детекция только по метаданным объекта, без текстовых маркеров
        self.assertFalse(is_summary_message(UserMessage("xyz")))
        self.assertFalse(is_summary_message(UserMessage("[SUMMARY of messages 1-5]: xyz")))
        self.assertFalse(is_summary_message(UserMessage("It's an auto-generated text. Your past dialog summary")))
        self.assertTrue(is_summary_message(UserMessage("flagged", is_summary=True)))
        self.assertFalse(is_summary_message(UserMessage("normal user message")))
        self.assertFalse(is_summary_message(UserMessage("")))

    def test_find_existing_summary(self):
        msgs = _history().get_all()
        self.assertIsNone(_find_existing_summary(msgs, len(msgs) - 1))

        h = _history()
        h.replace_range(2, 2, [UserMessage("earlier facts\npath /etc/x", is_summary=True)])
        msgs = h.get_all()
        found = _find_existing_summary(msgs, len(msgs) - 1)
        self.assertIsNotNone(found)
        self.assertEqual(found["index"], 2)
        # body — это текст после первой строки-заголовка
        self.assertIn("path /etc/x", found["body"])
        self.assertIn("earlier facts", found["full_content"])

    def test_build_draft_prompt_has_no_last_user_content(self):
        p = _build_draft_prompt(existing=None)
        self.assertIn("KEY FACTS", p)
        # явный verbatim-прогон последнего сообщения в промпт не вставляем
        self.assertNotIn("LAST USER REQUEST verbatim", p)

    def test_build_draft_prompt_with_existing_summary(self):
        p = _build_draft_prompt(existing={"body": "old facts"})
        self.assertIn("EXISTING auto-summary", p)
        self.assertIn("MERGE", p)


class TestSummarizeDialogue(unittest.TestCase):
    def _agent(self, history):
        agent = mock.Mock()
        agent.history = history
        agent.token_tracker = mock.Mock()

        # Реальный service_llm_call поверх мок-агента: патчится LLMClient.call,
        # чтобы перехватывать транспорт (как это делает настоящий агент).
        def _service(msgs, temp=None, timeout=None, tools=True, prefill=None, params=None):
            return LLMClient.call(msgs, temp=temp, timeout=timeout,
                                  prefill=prefill, params=params)

        agent.service_llm_call.side_effect = _service
        # per-message режим читает рабочую память (agent.history._per_msg_summaries);
        # по умолчанию она пуста, поэтому суммаризация падает на однофазный
        # (draft+review) путь.
        return agent

    def test_system_prompt_is_first_in_draft_and_review(self):
        old = Config.AUTO_SUMMARY_REVIEW_PASS
        Config.AUTO_SUMMARY_REVIEW_PASS = True
        try:
            h = _history()
            expected_system = h[0].content  # "sys"
            agent = self._agent(h)
            captured = []

            def fake_call(msgs, **kwargs):
                captured.append(msgs)
                content = msgs[-1]["content"]
                if "DRAFT SUMMARY" in content:
                    return (_fake_msg("TASK: create readme\nPROGRESS: done\nKEY FACTS: fixed"), None, None)
                return (_fake_msg("TASK: create readme\nPROGRESS: done"), None, None)

            with mock.patch("universal_agents.compressors.LLMClient.call", side_effect=fake_call):
                summarize_dialogue(agent)
            # draft + review: в обоих вызовах 0-е сообщение — системный промпт
            self.assertEqual(len(captured), 2)
            for msgs in captured:
                self.assertEqual(msgs[0]["role"], "system")
                self.assertIn(expected_system, msgs[0]["content"])
        finally:
            Config.AUTO_SUMMARY_REVIEW_PASS = old

    def test_draft_only_without_review(self):
        old = Config.AUTO_SUMMARY_REVIEW_PASS
        Config.AUTO_SUMMARY_REVIEW_PASS = False
        try:
            agent = self._agent(_history())
            with mock.patch(
                "universal_agents.compressors.LLMClient.call",
                return_value=(_fake_msg("TASK: create readme\nPROGRESS: done"), None, None),
            ) as call_mock:
                res = summarize_dialogue(agent)
            self.assertIn("TASK: create readme", res)
            self.assertEqual(call_mock.call_count, 1)
        finally:
            Config.AUTO_SUMMARY_REVIEW_PASS = old

    def test_review_pass_runs_after_draft(self):
        old = Config.AUTO_SUMMARY_REVIEW_PASS
        Config.AUTO_SUMMARY_REVIEW_PASS = True
        try:
            agent = self._agent(_history())
            calls = []

            def fake_call(msgs, **kwargs):
                calls.append(msgs)
                content = msgs[-1]["content"]
                if "DRAFT SUMMARY" in content:
                    # review-проход: убрал obsolete-деталь, вернул полную версию
                    return (_fake_msg("TASK: create readme\nPROGRESS: done\nKEY FACTS: fixed"), None, None)
                return (_fake_msg("TASK: create readme\nPROGRESS: done\nOBSOLETE: removed\nKEY FACTS: fixed"), None, None)

            with mock.patch("universal_agents.compressors.LLMClient.call", side_effect=fake_call):
                res = summarize_dialogue(agent)
            # draft + review
            self.assertEqual(len(calls), 2)
            self.assertIn("KEY FACTS: fixed", res)
            # итоговое саммари — из review-прохода (убрал obsolete)
            self.assertNotIn("OBSOLETE", res)
        finally:
            Config.AUTO_SUMMARY_REVIEW_PASS = old

    def test_review_prompt_prunes_and_adds(self):
        old = Config.AUTO_SUMMARY_REVIEW_PASS
        Config.AUTO_SUMMARY_REVIEW_PASS = True
        try:
            agent = self._agent(_history_with_summary())
            review_prompt_captured = []

            def fake_call(msgs, **kwargs):
                content = msgs[-1]["content"]
                if "DRAFT SUMMARY" in content:
                    review_prompt_captured.append(content)
                    return (_fake_msg("TASK: fixed\nKEY FACTS: src/main.py"), None, None)
                return (_fake_msg("TASK: fix bug\nKEY FACTS: src/main.py"), None, None)

            with mock.patch("universal_agents.compressors.LLMClient.call", side_effect=fake_call):
                res = summarize_dialogue(agent)
            self.assertEqual(len(review_prompt_captured), 1)
            # review просит и прунить устаревшее, и добавлять недостающее
            self.assertIn("PRUNE", review_prompt_captured[0])
            self.assertIn("ADD", review_prompt_captured[0])
            self.assertIn("EXISTING auto-summary", review_prompt_captured[0])
            self.assertIn("KEY FACTS: src/main.py", res)
        finally:
            Config.AUTO_SUMMARY_REVIEW_PASS = old


if __name__ == "__main__":
    unittest.main()