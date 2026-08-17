import json
import unittest
from types import SimpleNamespace
from unittest import mock

from universal_agents.task_tracker import (
    plan_leaf_sequence,
    validate_task_mark_call,
    set_plan,
    mark_task_done,
    compact_completed_tasks,
)
from universal_agents.constants import SUMMARY_MARKER
from universal_agents.history import ChatHistory
from universal_agents.models import UserMessage, AssistantMessage, ToolResult, ToolCall


def _done(tid, summary=""):
    args = {"id": tid, "summary": summary}
    return ToolCall(
        id=f"call_done_{tid}",
        name="have_done",
        arguments=json.dumps(args, ensure_ascii=False),
    )


def _assistant_done(tid, summary=""):
    return AssistantMessage(content="", tool_calls=[_done(tid, summary)])


def _done_result(tid):
    return ToolResult.success(f"call_done_{tid}", "have_done", "ok")


def _plan_call(plan_list):
    return ToolCall(
        id="call_plan",
        name="make_plan",
        arguments=json.dumps({"plan": plan_list}, ensure_ascii=False),
    )


def _work(path="a"):
    """Сообщение с реальным вызовом инструмента (read)."""
    return AssistantMessage(content="", tool_calls=[ToolCall(f"r_{path}", "read", json.dumps({"path": path}))])


def _work_result(path="a"):
    return ToolResult.success(f"r_{path}", "read", "file contents")


def _make_agent():
    h = ChatHistory("sys")
    h.add(UserMessage("do X"))
    h.add(AssistantMessage(content="", tool_calls=[_plan_call([
        {"id": "A1", "title": "A1"},
        {"id": "A2", "title": "A2"},
        {"id": "B1", "title": "B1"},
    ])]))
    h.add(ToolResult.success("c0", "make_plan", "Plan set"))
    h.add(AssistantMessage(content=("A1 work (long content) " * 40)))
    h.add(_assistant_done("A1", "done A1"))
    h.add(_done_result("A1"))
    h.add(AssistantMessage(content=("A2 work (long content) " * 40)))
    h.add(_assistant_done("A2", "done A2"))
    h.add(_done_result("A2"))
    agent = SimpleNamespace(
        history=h,
        task_plan=[],
        task_plan_map={},
        _compacted_task_ids=set(),
        file_states=SimpleNamespace(prune=lambda: None),
        on_system_msg=lambda *a, **k: None,
    )
    set_plan(agent, [
        {"id": "A1", "title": "A1"},
        {"id": "A2", "title": "A2"},
        {"id": "B1", "title": "B1"},
    ])
    return agent


class TestPlan(unittest.TestCase):
    def test_set_plan_stores_order(self):
        agent = SimpleNamespace(task_plan=[], task_plan_map={}, _compacted_task_ids=set())
        res = set_plan(agent, [
            {"id": "A", "title": "A"},
            {"id": "B", "title": "B"},
            {"id": "C", "title": "C"},
        ])
        self.assertIn("Execution order", res)
        self.assertEqual(agent.task_plan, ["A", "B", "C"])
        self.assertEqual(plan_leaf_sequence(agent.task_plan_map), ["A", "B", "C"])

    def test_set_plan_rejects_duplicate(self):
        agent = SimpleNamespace(task_plan=[], task_plan_map={}, _compacted_task_ids=set())
        res = set_plan(agent, [{"id": "A"}, {"id": "A"}])
        self.assertIn("duplicate", res)
        self.assertEqual(agent.task_plan, [])

    def test_set_plan_rejects_non_dict(self):
        agent = SimpleNamespace(task_plan=[], task_plan_map={}, _compacted_task_ids=set())
        res = set_plan(agent, ["A"])
        self.assertIn("object", res)

    def test_set_plan_rejects_empty(self):
        agent = SimpleNamespace(task_plan=[], task_plan_map={}, _compacted_task_ids=set())
        res = set_plan(agent, [])
        self.assertIn("non-empty", res)


class TestMarkDone(unittest.TestCase):
    def test_mark_done(self):
        agent = _make_agent()
        res = mark_task_done(agent, "A1", "ok")
        self.assertIn("marked done", res)
        self.assertIn("A1", res)

    def test_mark_unknown_id(self):
        agent = _make_agent()
        res = mark_task_done(agent, "ZZZ", "x")
        self.assertIn("not in the plan", res)

    def test_mark_empty_id(self):
        agent = _make_agent()
        res = mark_task_done(agent, "", "x")
        self.assertIn("Error", res)


class TestOrderValidation(unittest.TestCase):
    def _history(self, plan_list):
        h = ChatHistory("sys")
        h.add(UserMessage("do X"))
        h.add(AssistantMessage(content="", tool_calls=[_plan_call(plan_list)]))
        h.add(ToolResult.success("c0", "make_plan", "Plan set"))
        plan_map = {e["id"]: {"title": e.get("title", "")} for e in plan_list}
        return h, plan_map

    PLAN = [{"id": "A1"}, {"id": "A2"}, {"id": "B1"}]

    def test_valid_sequence(self):
        h, pm = self._history(self.PLAN)
        h.add(_work("a"))
        h.add(_work_result("a"))
        self.assertIsNone(validate_task_mark_call(h.get_all(), {"id": "A1"}, pm, set()))
        h.add(_assistant_done("A1"))
        h.add(_done_result("A1"))
        h.add(_work("b"))
        h.add(_work_result("b"))
        self.assertIsNone(validate_task_mark_call(h.get_all(), {"id": "A2"}, pm, set()))
        h.add(_assistant_done("A2"))
        h.add(_done_result("A2"))
        h.add(_work("c"))
        h.add(_work_result("c"))
        self.assertIsNone(validate_task_mark_call(h.get_all(), {"id": "B1"}, pm, set()))
        h.add(_assistant_done("B1"))
        h.add(_done_result("B1"))

    def test_requires_plan(self):
        h, _ = self._history(self.PLAN)
        err = validate_task_mark_call(h.get_all(), {"id": "A1"}, {}, set())
        self.assertIsNotNone(err)
        self.assertIn("no task plan", err)

    def test_cannot_mark_unknown(self):
        h, pm = self._history(self.PLAN)
        err = validate_task_mark_call(h.get_all(), {"id": "X"}, pm, set())
        self.assertIsNotNone(err)
        self.assertIn("not in the plan", err)

    def test_cannot_skip_ahead(self):
        h, pm = self._history(self.PLAN)
        h.add(_assistant_done("A1"))
        h.add(_done_result("A1"))
        err = validate_task_mark_call(h.get_all(), {"id": "B1"}, pm, set())
        self.assertIsNotNone(err)
        self.assertIn("OUT-OF-ORDER", err)
        self.assertIn("'A2'", err)

    def test_revision_allows_start_from_arbitrary_step(self):
        h, _ = self._history(self.PLAN)
        new_pm = {"B1": {"title": ""}}
        h.add(_work("x"))
        h.add(_work_result("x"))
        self.assertIsNone(validate_task_mark_call(h.get_all(), {"id": "B1"}, new_pm, set()))
        h.add(_assistant_done("B1"))

    def test_revision_reused_ids_ignores_old_plan_done_markers(self):
        h = ChatHistory("sys")
        h.add(UserMessage("test"))
        planA = [{"id": "A1"}, {"id": "A2"}, {"id": "A3"}]
        h.add(AssistantMessage(content="", tool_calls=[_plan_call(planA)]))
        h.add(ToolResult.success("c", "make_plan", "Plan set"))
        h.add(_assistant_done("A1"))
        h.add(ToolResult.success("c", "have_done", "ok"))
        # Новый план переиспользует те же id, но порядок: A3,A1,A2
        planB = [{"id": "A3"}, {"id": "A1"}, {"id": "A2"}]
        pm_b = {e["id"]: {"title": ""} for e in planB}
        h.add(AssistantMessage(content="", tool_calls=[_plan_call(planB)]))
        h.add(ToolResult.success("c", "make_plan", "Plan set"))
        h.add(_work("w1"))
        h.add(_work_result("w1"))
        self.assertIsNone(validate_task_mark_call(h.get_all(), {"id": "A3"}, pm_b, set()))
        h.add(_assistant_done("A3"))
        h.add(_done_result("A3"))
        # по новому плану следующая — A1, несмотря на done-маркер из старого плана
        h.add(_work("w2"))
        h.add(_work_result("w2"))
        self.assertIsNone(validate_task_mark_call(h.get_all(), {"id": "A1"}, pm_b, set()))

    def test_all_done(self):
        h, pm = self._history(self.PLAN)
        for tid in ("A1", "A2", "B1"):
            h.add(_assistant_done(tid))
            h.add(_done_result(tid))
        err = validate_task_mark_call(h.get_all(), {"id": "B1"}, pm, set())
        self.assertIsNotNone(err)
        self.assertIn("already done", err)

    def test_empty_id_rejected(self):
        h, pm = self._history(self.PLAN)
        err = validate_task_mark_call(h.get_all(), {"id": ""}, pm, set())
        self.assertIsNotNone(err)

    def test_rejects_done_without_real_work(self):
        # Сразу после make_plan помечаем задачу выполненной без реальных инструментов
        h, pm = self._history(self.PLAN)
        err = validate_task_mark_call(h.get_all(), {"id": "A1"}, pm, set())
        self.assertIsNotNone(err)
        self.assertIn("NO-WORK-DONE", err)

    def test_rejected_done_does_not_mark_task_complete(self):
        # Ошибочный (отклонённый) have_done не должен считаться выполненным:
        # после него следующей задачей всё ещё остаётся A1, а не A2.
        h, pm = self._history(self.PLAN)
        h.add(_assistant_done("A1"))  # отклонённый вызов (без успешного ToolResult)
        h.add(ToolResult.error("x", "have_done", "NO-WORK-DONE"))
        err = validate_task_mark_call(h.get_all(), {"id": "A1"}, pm, set())
        # всё ещё A1 (нет успешного done-маркера), а не OUT-OF-ORDER на A2
        self.assertIsNotNone(err)
        self.assertIn("NO-WORK-DONE", err)
        self.assertNotIn("OUT-OF-ORDER", err)

    def test_accepts_done_after_real_work(self):
        h, pm = self._history(self.PLAN)
        # реальная работа: вызов read между планом и have_done
        h.add(AssistantMessage(content="", tool_calls=[ToolCall("r", "read", '{"path":"a"}')]))
        h.add(ToolResult.success("r", "read", "file contents"))
        self.assertIsNone(validate_task_mark_call(h.get_all(), {"id": "A1"}, pm, set()))
        h.add(_assistant_done("A1"))
        h.add(_done_result("A1"))
        # следующая задача без работы — снова отклоняется
        err = validate_task_mark_call(h.get_all(), {"id": "A2"}, pm, set())
        self.assertIsNotNone(err)
        self.assertIn("NO-WORK-DONE", err)
        # добавляем работу для A2
        h.add(AssistantMessage(content="", tool_calls=[ToolCall("r", "read", '{"path":"b"}')]))
        h.add(ToolResult.success("r", "read", "more"))
        self.assertIsNone(validate_task_mark_call(h.get_all(), {"id": "A2"}, pm, set()))


class TestCompaction(unittest.TestCase):
    def test_compaction_compresses_done_tasks(self):
        agent = _make_agent()
        before = agent.history.content_len(0, len(agent.history) - 1)
        with mock.patch(
            "universal_agents.task_tracker.summarize_task_segment",
            return_value="dense summary of task",
        ):
            n = compact_completed_tasks(agent)
        self.assertGreater(n, 0)
        after = agent.history.content_len(0, len(agent.history) - 1)
        self.assertLess(after, before)
        summary_msgs = [m for m in agent.history if isinstance(m, UserMessage) and SUMMARY_MARKER in m.content]
        self.assertTrue(summary_msgs)
        # маркеры make_plan и have_done сохраняются (не компактизируются)
        remaining = [m for m in agent.history if isinstance(m, AssistantMessage) and m.has_tool_calls()]
        self.assertTrue(
            any(any(tc.name == "make_plan" for tc in m.tool_calls) for m in remaining)
        )
        done_calls = [
            tc.name for m in remaining for tc in m.tool_calls if tc.name == "have_done"
        ]
        # A1 и A2 размечены и их have_done-маркеры остаются в истории
        self.assertEqual(len(done_calls), 2)

    def test_compaction_trims_bloated_have_done_summary(self):
        h = ChatHistory("sys")
        h.add(UserMessage("do X"))
        h.add(AssistantMessage(content="", tool_calls=[_plan_call([
            {"id": "A1", "title": "A1"},
            {"id": "B1", "title": "B1"},
        ])]))
        h.add(ToolResult.success("c0", "make_plan", "Plan set"))
        h.add(AssistantMessage(content=("A1 work " * 500)))
        long_summary = "x" * 5000
        h.add(_assistant_done("A1", long_summary))
        h.add(ToolResult.success(
            "call_done_A1", "have_done",
            f"Task 'A1' marked done. Summary recorded: {long_summary}",
        ))
        agent = SimpleNamespace(
            history=h,
            task_plan=[],
            task_plan_map={},
            _compacted_task_ids=set(),
            file_states=SimpleNamespace(prune=lambda: None),
            on_system_msg=lambda *a, **k: None,
        )
        set_plan(agent, [{"id": "A1", "title": "A1"}, {"id": "B1", "title": "B1"}])
        with mock.patch(
            "universal_agents.task_tracker.summarize_task_segment",
            return_value="dense",
        ):
            compact_completed_tasks(agent)
        hd_msgs = [m for m in agent.history if isinstance(m, ToolResult) and m.name == "have_done"]
        self.assertEqual(len(hd_msgs), 1)
        self.assertIn("summary compacted", hd_msgs[0].content)
        self.assertLess(len(hd_msgs[0].content), 100)

    def test_compaction_marks_compacted_ids(self):
        agent = _make_agent()
        with mock.patch(
            "universal_agents.task_tracker.summarize_task_segment",
            return_value="dense summary",
        ):
            compact_completed_tasks(agent)
        self.assertIn("A1", agent._compacted_task_ids)
        self.assertIn("A2", agent._compacted_task_ids)

    def test_compaction_noop_when_disabled(self):
        agent = _make_agent()
        with mock.patch("universal_agents.config.Config.TASK_COMPACTION_ENABLED", False):
            n = compact_completed_tasks(agent)
        self.assertEqual(n, 0)

    def test_compaction_ignores_incomplete_tasks(self):
        agent = _make_agent()
        with mock.patch(
            "universal_agents.task_tracker.summarize_task_segment",
            return_value="dense summary",
        ):
            n = compact_completed_tasks(agent)
        self.assertGreater(n, 0)
        self.assertNotIn("B1", agent._compacted_task_ids)

    def test_compaction_does_not_swallow_plan(self):
        agent = _make_agent()
        with mock.patch(
            "universal_agents.task_tracker.summarize_task_segment",
            return_value="dense summary",
        ):
            compact_completed_tasks(agent)
        msgs = list(agent.history)
        self.assertTrue(any(isinstance(m, UserMessage) for m in msgs))
        self.assertTrue(
            any(isinstance(m, AssistantMessage) and m.has_tool_calls()
                and any(tc.name == "make_plan" for tc in m.tool_calls)
                for m in msgs)
        )


if __name__ == "__main__":
    unittest.main()
