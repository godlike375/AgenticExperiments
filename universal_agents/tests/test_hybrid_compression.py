import unittest
from unittest import mock
from types import SimpleNamespace

from universal_agents.history import ChatHistory
from universal_agents.llm_client import TokenUsageTracker
from universal_agents.agent_mixins.memory_mixin import MemoryMixin
from universal_agents.models import (
    AssistantMessage,
    ToolResult,
    UserMessage,
)
from universal_agents.archive import HistoryArchive


SUMMARY_TEXT = (
    "### 1. Постановка задачи и ожидания\n"
    "Задача: отчёт о дельфинах. Строго: только проверенные факты.\n"
    "### 2. Ход работы и принятые решения\n"
    "Введение написано. Решение: latex, потому что требует площадка.\n"
    "### 3. Текущий статус, блокировки и черновики\n"
    "Блокировок нет. Следующий шаг: раздел про питание.\n"
)


class FakeAgent(MemoryMixin):
    def __init__(self):
        self.history = ChatHistory("sys")
        self.archive = HistoryArchive()
        self.on_system_msg = lambda x: None
        self.token_tracker = TokenUsageTracker("sys", 50000)
        self.task_plan = []
        self.task_plan_map = {}
        self._compacted_task_ids = set()
        # Реальный service_llm_call поверх мок-агента: транспорт перехватывается
        # через mock.patch("...LLMClient.call"), как в test_compressors.py.
        from universal_agents.llm_client import LLMClient

        def _service(msgs, temp=None, timeout=None, tools=True, prefill=None, params=None):
            return LLMClient.call(msgs, temp=temp, timeout=timeout,
                                  prefill=prefill, params=params)

        self.service_llm_call = mock.Mock(side_effect=_service)

    def _on_history_changed(self):
        pass

    def _prune_per_msg_summaries(self):
        self.history.prune_per_msg_summaries()


def _compression_router(summary_text=SUMMARY_TEXT, calls=None):
    """Перехват транспорта: на вызов компакции отвечает заготовленным текстом
    (полная перезапись заметок по 7 разделам). Захватывает весь список
    сообщений (полная история + инструкция), а не только инструкцию."""
    def fake_call(msgs, **kwargs):
        if calls is not None:
            calls.append(msgs)
        prompt = msgs[-1]["content"]
        if "Write the full session summary from the beginning" in prompt:
            return (SimpleNamespace(content=summary_text, tool_calls=None), None, None)
        return (SimpleNamespace(content="ok", tool_calls=None), None, None)
    return fake_call


def _seed_dialog(agent: FakeAgent):
    h = agent.history
    h.add(UserMessage("TASK: write the report about dolphins"))
    a1 = AssistantMessage(content="long reasoning " * 80)
    h.add(a1)
    h.add(ToolResult.success("t1", "search", "found 3 sources"))
    h.add(UserMessage("continue please"))


def _summary_msgs(history) -> list:
    return [m for m in history.get_all()
            if isinstance(m, UserMessage) and m.is_summary]


class TestFirstCompaction(unittest.TestCase):
    def setUp(self):
        self.agent = FakeAgent()
        _seed_dialog(self.agent)

    def _run(self):
        with mock.patch(
            "universal_agents.compressors.LLMClient.call",
            side_effect=_compression_router(),
        ):
            self.agent._auto_summarize_dialogue()

    def test_layout_summary_then_tail(self):
        self._run()
        msgs = self.agent.history.get_all()
        # [system, саммари-user, хвост] — старые сообщения удалены
        self.assertEqual(len(msgs), 3)
        self.assertIsInstance(msgs[1], UserMessage)
        self.assertTrue(msgs[1].is_summary)
        self.assertIn(SUMMARY_TEXT, msgs[1].content)
        self.assertEqual(msgs[2].content, "continue please")

    def test_archive_got_originals(self):
        self._run()
        self.assertEqual(len(self.agent.archive), 3)

    def test_prompt_is_plain_text_with_seven_sections(self):
        captured = []
        with mock.patch(
            "universal_agents.compressors.LLMClient.call",
            side_effect=_compression_router(calls=captured),
        ):
            self.agent._auto_summarize_dialogue()
        # единственный вызов компакции — полная история + инструкция
        self.assertEqual(len(captured), 1)
        msgs = captured[0]
        # system prompt — первое сообщение (префикс для KV-cache)
        self.assertEqual(msgs[0]["role"], "system")
        # инструкция суммаризации — последнее сообщение
        instruction = msgs[-1]["content"]
        self.assertIn("Write the full session summary from the beginning", instruction)
        for marker in (
            "### 1. Постановка задачи", "### 2. Ход работы",
            "### 3. Текущий статус", "### 4. Знания, гипотезы",
            "### 5. Проектная специфика", "### 6. Ресурсы и артефакты",
            "### 7. Инструкции для передачи контекста",
        ):
            self.assertIn(marker, instruction)
        # полная история передана как структурированные сообщения, а не текст-транскрипция
        full_text = "\n".join(
            m["content"] for m in msgs
            if isinstance(m.get("content"), str)
        )
        self.assertIn("TASK: write the report about dolphins", full_text)
        self.assertIn("found 3 sources", full_text)

    def test_seq_assigned_unique_and_stable(self):
        self._run()
        msgs = self.agent.history.get_all()
        seqs = [m.seq for m in msgs[1:]]
        self.assertTrue(all(isinstance(s, int) for s in seqs), seqs)
        self.assertEqual(len(seqs), len(set(seqs)), seqs)
        self.agent.history.add(UserMessage("after compaction"))
        last = self.agent.history.get_all()[-1]
        self.assertGreater(last.seq, max(seqs))

    def test_failure_leaves_history_untouched(self):
        notes = []
        self.agent.on_system_msg = notes.append

        def broken(msgs, **kwargs):
            return (SimpleNamespace(content="", tool_calls=None), "boom", None)

        before = len(self.agent.history)
        with mock.patch("universal_agents.compressors.LLMClient.call", side_effect=broken):
            self.agent._auto_summarize_dialogue()
        self.assertEqual(len(self.agent.history), before)
        self.assertFalse(_summary_msgs(self.agent.history))
        self.assertTrue(any("Compression call failed" in n for n in notes))


class TestSecondCompaction(unittest.TestCase):
    def setUp(self):
        self.agent = FakeAgent()
        _seed_dialog(self.agent)
        with mock.patch(
            "universal_agents.compressors.LLMClient.call",
            side_effect=_compression_router(),
        ):
            self.agent._auto_summarize_dialogue()
        self.agent.history.add(AssistantMessage(content="ok"))
        self.agent.history.add(ToolResult.success("t2", "cwd", "/proj"))
        self.agent.history.add(UserMessage("go on"))

    def test_single_summary_after_second_run(self):
        with mock.patch(
            "universal_agents.compressors.LLMClient.call",
            side_effect=_compression_router(),
        ):
            self.agent._auto_summarize_dialogue()
        msgs = self.agent.history.get_all()
        self.assertEqual([type(m).__name__ for m in msgs],
                         ["SystemMessage", "UserMessage", "UserMessage"])
        self.assertTrue(msgs[1].is_summary)
        # Заметки переписаны с нуля тем же заготовленным текстом
        self.assertIn(SUMMARY_TEXT, msgs[1].content)

    def test_prompt_includes_existing_notes_and_segment(self):
        captured = []
        with mock.patch(
            "universal_agents.compressors.LLMClient.call",
            side_effect=_compression_router(calls=captured),
        ):
            self.agent._auto_summarize_dialogue()
        msgs = captured[-1]
        instruction = msgs[-1]["content"]
        # Старые заметки поданы отдельным блоком для слияния
        self.assertIn("EXISTING SUMMARY:", instruction)
        self.assertIn("Задача: отчёт о дельфинах.", instruction)
        # Полная история (включая новый сегмент) — структурированными сообщениями
        full_text = "\n".join(
            m["content"] for m in msgs
            if isinstance(m.get("content"), str)
        )
        self.assertIn("go on", full_text)
        # инструкция не дублирует обёртку старого саммари
        self.assertNotIn("Your past dialog summary with user:", instruction)

    def test_old_summary_archived_before_rewrite(self):
        archive_len_before = len(self.agent.archive)
        with mock.patch(
            "universal_agents.compressors.LLMClient.call",
            side_effect=_compression_router(),
        ):
            self.agent._auto_summarize_dialogue()
        self.assertGreater(len(self.agent.archive), archive_len_before)
        self.assertIn("latex", self.agent.archive.search("latex"))

    def test_failure_leaves_history_untouched(self):
        notes = []
        self.agent.on_system_msg = notes.append
        before = len(self.agent.history)
        with mock.patch(
            "universal_agents.compressors.LLMClient.call",
            side_effect=lambda msgs, **kw: (None, "boom", None),
        ):
            self.agent._auto_summarize_dialogue()
        self.assertEqual(len(self.agent.history), before)
        # существующее саммари не тронуто при неудаче служебного вызова
        self.assertTrue(_summary_msgs(self.agent.history))
        self.assertIn(SUMMARY_TEXT, self.agent.history.get_all()[1].content)
        self.assertTrue(any("Compression call failed" in n for n in notes))


class TestUnsafeBoundary(unittest.TestCase):
    def test_no_compaction_on_assistant_tail(self):
        agent = FakeAgent()
        _seed_dialog(agent)
        agent.history.add(AssistantMessage(content="text answer only"))
        before = len(agent.history)
        with mock.patch("universal_agents.compressors.LLMClient.call", side_effect=_compression_router()):
            agent._auto_summarize_dialogue()
        self.assertEqual(len(agent.history), before)


class TestPersistenceRoundTrip(unittest.TestCase):
    def test_summary_and_extras_survive_save_load(self):
        import tempfile, os
        agent = FakeAgent()
        _seed_dialog(agent)
        with mock.patch("universal_agents.compressors.LLMClient.call", side_effect=_compression_router()):
            agent._auto_summarize_dialogue()
        agent.archive.append_messages(agent.history.get_all()[1:2])
        agent.task_plan = ["t1"]
        agent.task_plan_map = {"t1": {"title": "audit"}}
        agent._compacted_task_ids = {"t1"}

        from universal_agents.task_tracker import plan_state_to_dict, restore_plan_state
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "h.json")
            agent.history.save(
                path, loaded_tools=["read"], file_states={},
                extras={"archive": agent.archive.to_list(), "plan_state": plan_state_to_dict(agent)},
            )
            fresh = ChatHistory("sys")
            tools, fs, summaries = fresh.load(path)
            extras = fresh.last_loaded_extras

            # саммари восстановлено как user-сообщение с флагом
            smry = [m for m in fresh.get_all() if isinstance(m, UserMessage) and m.is_summary]
            self.assertEqual(len(smry), 1)
            self.assertIn(SUMMARY_TEXT, smry[0].content)
            self.assertEqual(len(extras["archive"]), 4)

            restored = HistoryArchive.from_list(extras["archive"])
            self.assertIn("dolphins", restored.search("dolphins"))

            new_agent = FakeAgent()
            restore_plan_state(new_agent, extras["plan_state"])
            self.assertEqual(new_agent.task_plan_map["t1"]["title"], "audit")

            nxt = fresh.next_seq
            fresh.add(UserMessage("post-load"))
            self.assertEqual(fresh.get_all()[-1].seq, nxt)


class TestRecallTools(unittest.TestCase):
    def test_recall_tools_end_to_end(self):
        import importlib.util, os
        spec = importlib.util.spec_from_file_location(
            "mem_tools",
            os.path.join(os.path.dirname(__file__), "..", "tools", "memory.py"),
        )
        mem_tools = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mem_tools)

        agent = FakeAgent()
        _seed_dialog(agent)
        with mock.patch("universal_agents.compressors.LLMClient.call", side_effect=_compression_router()):
            agent._auto_summarize_dialogue()

        out = mem_tools.recall_search(agent, query="dolphins")
        self.assertIn("[[SYSTEM]]", out)
        self.assertIn("seq=", out)
        out_read = mem_tools.recall_read(agent, from_seq=2, to_seq=2)
        self.assertIn("#2", out_read)
        self.assertIn("long reasoning", out_read.replace("\n", " "))


if __name__ == "__main__":
    unittest.main()
