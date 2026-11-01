from __future__ import annotations

import json
import os
import shutil
import tempfile
import textwrap
import threading
import time
import unittest
from unittest import mock

from universal_agents.config import Config
from universal_agents.models import AssistantMessage, ToolCall
from universal_agents.tool import tool
from universal_agents.tool_manager import ToolManager
from universal_agents.tool_registry import build_tool_dict
from universal_agents.tool_worker import ToolWorker

from tests.conftest import make_agent as make_test_agent


@tool(description="no-agent external", short_description="x")
def _ungrouped_external(value: int) -> str:
    return str(value)


@tool(description="agent tool")
def _with_agent(agent, value: int) -> str:
    return str(value)


@tool(description="hidden low-level", hidden=True)
def _hidden_tool(value: int) -> str:
    return str(value)


class TestToolMetadata(unittest.TestCase):
    def test_hidden_flag(self):
        @tool(description="d", hidden=True)
        def sample(x: int) -> str:
            return str(x)
        self.assertTrue(sample._tool_hidden)

    def test_build_tool_dict_derives_kind(self):
        external = build_tool_dict(_ungrouped_external, is_instance_method=False)
        self.assertTrue(external["is_external"])

        agent_tool = build_tool_dict(_with_agent, is_instance_method=False)
        self.assertFalse(agent_tool["is_external"], "инструмент с агентом всегда внутренний")

        hidden = build_tool_dict(_hidden_tool, is_instance_method=False)
        self.assertTrue(hidden["hidden"])
        self.assertTrue(hidden["is_external"])


class TestHiddenToolsHiddenFromModel(unittest.TestCase):
    def test_hidden_excluded_from_loadable_and_schemas(self):
        tm = ToolManager(tools_config=None, external_plugins={_hidden_tool.__name__: _hidden_tool})
        self.assertNotIn(_hidden_tool.__name__, tm.all_known_names)
        self.assertNotIn(_hidden_tool.__name__, tm.tools_map)

    def test_hidden_cannot_be_loaded(self):
        tm = ToolManager(tools_config=None, external_plugins={_hidden_tool.__name__: _hidden_tool})
        result = tm.load(_hidden_tool.__name__)
        self.assertIn("Error", result)


class TestInProcessWorker(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)
        self._prev = os.getcwd()
        self.addCleanup(os.chdir, self._prev)

    def test_cwd_updates_host(self):
        sub = os.path.join(self._tmp, "sub")
        os.makedirs(sub)
        os.chdir(self._tmp)
        worker = ToolWorker(isolated=False)
        try:
            status, val = worker.call("cwd", {"path": "sub"})
            self.assertEqual(status, "ok")
            self.assertEqual(os.path.abspath(os.getcwd()), os.path.abspath(sub))
        finally:
            worker.close()


class TestIsolatedWorker(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)
        self._prev = os.getcwd()
        self.addCleanup(os.chdir, self._prev)

    def test_cwd_round_trip_and_error(self):
        sub = os.path.join(self._tmp, "sub")
        os.makedirs(sub)
        os.chdir(self._tmp)
        worker = ToolWorker(isolated=True)
        try:
            status, val = worker.call("cwd", {})
            self.assertEqual(status, "ok")
            status, val = worker.call("cwd", {"path": "sub"})
            self.assertEqual(status, "ok")
            self.assertEqual(os.path.abspath(os.getcwd()), os.path.abspath(sub))

            status, val = worker.call("cwd", {"path": "does_not_exist_xyz"})
            self.assertEqual(status, "error")

            status, val = worker.call("nope_not_a_tool", {})
            self.assertEqual(status, "error")
        finally:
            worker.close()

    def test_wrapped_edit_dry_run_then_apply(self):
        f = os.path.join(self._tmp, "a.py")
        with open(f, "w", encoding="utf-8") as fh:
            fh.write("l1\nl2\nl3\nl4\n")
        os.chdir(self._tmp)
        worker = ToolWorker(isolated=True)
        try:
            args = {"path": f, "new_text": "X\n", "start_line": 2, "end_line": 2}
            status, val = worker.call("line_range_edit", args, dry_run=True)
            self.assertEqual(status, "wrapped")
            preview, _ask = val
            self.assertIn("-l2", preview)

            status_apply, val_apply = worker.call("line_range_edit", args, dry_run=False)
            self.assertEqual(status_apply, "ok")
            with open(f, encoding="utf-8") as fh:
                self.assertEqual(fh.read(), "l1\nX\nl3\nl4\n")
        finally:
            worker.close()


class TestIsolatedStop(unittest.TestCase):
    def test_stop_kills_blocking_worker_fast(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        with open(os.path.join(tmp, "blocker_plugin.py"), "w", encoding="utf-8") as f:
            f.write(textwrap.dedent('''
                import time
                from universal_agents.tool import tool

                @tool(description="block")
                def blocker(seconds: int) -> str:
                    time.sleep(seconds)
                    return "done"
            '''))
        worker = ToolWorker(isolated=True, tools_dir=tmp)
        stop = threading.Event()
        timer = threading.Timer(0.3, stop.set)
        timer.start()
        started = time.monotonic()
        try:
            status, val = worker.call("blocker", {"seconds": 60}, stop_check=stop.is_set)
        finally:
            timer.cancel()
            worker.close()
        elapsed = time.monotonic() - started
        self.assertEqual(status, "error")
        self.assertIn("stopped", val.lower())
        self.assertLess(elapsed, 5.0, "остановка должна убивать воркер, а не ждать")


class TestWorkerProtocolRobustness(unittest.TestCase):
    def test_tool_stdout_does_not_corrupt_protocol(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        with open(os.path.join(tmp, "noisy_plugin.py"), "w", encoding="utf-8") as f:
            f.write(textwrap.dedent('''
                from universal_agents.tool import tool

                @tool(description="prints to stdout")
                def noisy(value: str) -> str:
                    print("noise on stdout")
                    return f"echo={value}"
            '''))
        worker = ToolWorker(isolated=True, tools_dir=tmp)
        try:
            status1, val1 = worker.call("noisy", {"value": "a"})
            status2, val2 = worker.call("noisy", {"value": "b"})
        finally:
            worker.close()
        self.assertEqual(status1, "ok")
        self.assertEqual(val1, "echo=a")
        self.assertEqual(status2, "ok")
        self.assertEqual(val2, "echo=b")


class TestAgentEndToEndIsolated(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)
        self._prev = os.getcwd()
        self.addCleanup(os.chdir, self._prev)
        os.chdir(self._tmp)
        self._patch = mock.patch.object(Config, "TOOL_ISOLATION_ENABLED", True)
        self._patch.start()
        self.addCleanup(self._patch.stop)

    def _run(self, responses, plugins):
        agent = make_test_agent(external_plugins=plugins)
        seen = []

        def fake_call(messages, prefill=None, **kwargs):
            seen.append([m.get("content", "") for m in messages if m.get("role") == "tool"])
            return responses.pop(0), None, None

        with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake_call):
            result = agent.chat("do it", max_iter=10)
        agent.close_tool_workers()
        return agent, result, seen

    def test_search_uses_worker(self):
        from universal_agents.tools.fs import search
        f = os.path.join(self._tmp, "hello.txt")
        with open(f, "w", encoding="utf-8") as fh:
            fh.write("alpha\nbeta\n")
        responses = [
            AssistantMessage(content="Ищу файл.", tool_calls=[
                ToolCall(id="c1", name=search.__name__, arguments=json.dumps({"pattern": "alpha", "path": self._tmp}))]),
            AssistantMessage(content="Готово."),
        ]
        _agent, result, seen = self._run(responses, {search.__name__: search})
        self.assertIn("Готово", result)
        flat = [c for turn in seen for c in turn]
        self.assertTrue(any("alpha" in c for c in flat), flat)

    def test_isolated_wrapped_edit_applies_after_confirmation(self):
        from universal_agents.tools.fs import line_range_edit
        f = os.path.join(self._tmp, "edit.py")
        with open(f, "w", encoding="utf-8") as fh:
            fh.write("a\nb\nc\n")
        responses = [
            AssistantMessage(content="Меняю строку.", tool_calls=[
                ToolCall(id="c1", name=line_range_edit.__name__, arguments=json.dumps({
                    "path": f, "new_text": "X\n", "start_line": 2, "end_line": 2}))]),
            AssistantMessage(content="Подтверждаю.", tool_calls=[
                ToolCall(id="c2", name="respond_to_system", arguments=json.dumps({"text": "yes"}))]),
            AssistantMessage(content="Правка внесена."),
        ]
        from universal_agents.tools.builtin import respond_to_system
        agent, result, _seen = self._run(responses, {
            line_range_edit.__name__: line_range_edit,
            respond_to_system.__name__: respond_to_system,
        })
        self.assertIn("Правка внесена", result)
        with open(f, encoding="utf-8") as fh:
            self.assertEqual(fh.read(), "a\nX\nc\n")


if __name__ == "__main__":
    unittest.main()
