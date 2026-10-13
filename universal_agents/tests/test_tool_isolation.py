
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
from universal_agents.group_worker import GroupManager, make_worker_resolve
from universal_agents.models import AssistantMessage, ToolCall
from universal_agents.tool import tool
from universal_agents.tool_manager import ToolManager
from universal_agents.tool_registry import build_tool_dict

from tests.conftest import make_agent as make_test_agent


@tool(description="no-agent, default group", short_description="x")
def _ungrouped_external(value: int) -> str:
    return str(value)


@tool(description="agent tool", group="desktop")
def _with_agent(agent, value: int) -> str:
    return str(value)


@tool(description="hidden low-level", group="workspace", hidden=True)
def _hidden_tool(value: int) -> str:
    return str(value)


class TestToolMetadata(unittest.TestCase):
    def test_group_and_hidden_flags(self):
        @tool(description="d", group="sandbox", hidden=True)
        def sample(x: int) -> str:
            return str(x)
        self.assertEqual(sample._tool_group, "sandbox")
        self.assertTrue(sample._tool_hidden)

    def test_build_tool_dict_derives_kind(self):
        external = build_tool_dict(_ungrouped_external, is_instance_method=False)
        self.assertEqual(external["group"], "internal")
        self.assertFalse(external["is_external"])

        agent_tool = build_tool_dict(_with_agent, is_instance_method=False)
        self.assertEqual(agent_tool["group"], "desktop")
        self.assertFalse(agent_tool["is_external"], "инструмент с агентом всегда внутренний")

        hidden = build_tool_dict(_hidden_tool, is_instance_method=False)
        self.assertTrue(hidden["hidden"])
        self.assertTrue(hidden["is_external"])


class TestHiddenToolsHiddenFromModel(unittest.TestCase):
    def test_hidden_excluded_from_loadable_and_schemas(self):
        from universal_agents.tools.fs import _read_file_raw
        tm = ToolManager(tools_config=None)
        self.assertNotIn(_read_file_raw.__name__, tm.all_known_names)
        self.assertNotIn(_read_file_raw.__name__, tm.tools_map)

    def test_hidden_cannot_be_loaded(self):
        from universal_agents.tools.fs import _read_file_raw
        tm = ToolManager(tools_config=None)
        result = tm.load(_read_file_raw.__name__)
        self.assertIn("Error", result)


class TestMislabeledGroupWarning(unittest.TestCase):
    def test_no_agent_internal_warns(self):
        import io
        import contextlib
        tm = ToolManager(tools_config=None, external_plugins={_ungrouped_external.__name__: _ungrouped_external})
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            tm._warn_mislabeled_groups()
        self.assertIn(_ungrouped_external.__name__, buf.getvalue())


class TestInProcessGroupSession(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)
        self._prev = os.getcwd()
        self.addCleanup(os.chdir, self._prev)

    def test_cwd_updates_session_and_host(self):
        sub = os.path.join(self._tmp, "sub")
        os.makedirs(sub)
        os.chdir(self._tmp)
        mgr = GroupManager(isolated=False)
        try:
            result = mgr.call("workspace", "cwd", {"path": "sub"})
            self.assertEqual(result.status, "ok")
            self.assertEqual(os.path.abspath(mgr.session("workspace")["cwd"]), os.path.abspath(sub))
            self.assertEqual(os.path.abspath(os.getcwd()), os.path.abspath(sub))
        finally:
            mgr.close_all()


class TestIsolatedWorkspace(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)
        self._prev = os.getcwd()
        self.addCleanup(os.chdir, self._prev)

    def test_cwd_round_trip_and_error(self):
        sub = os.path.join(self._tmp, "sub")
        os.makedirs(sub)
        os.chdir(self._tmp)
        mgr = GroupManager(isolated=True)
        try:
            r = mgr.call("workspace", "cwd", {})
            self.assertEqual(r.status, "ok")
            r = mgr.call("workspace", "cwd", {"path": "sub"})
            self.assertEqual(r.status, "ok")
            self.assertEqual(os.path.abspath(mgr.session("workspace")["cwd"]), os.path.abspath(sub))

            r = mgr.call("workspace", "cwd", {"path": "does_not_exist_xyz"})
            self.assertEqual(r.status, "error")

            r = mgr.call("workspace", "nope_not_a_tool", {})
            self.assertEqual(r.status, "error")
        finally:
            mgr.close_all()

    def test_wrapped_edit_dry_run_then_apply(self):
        f = os.path.join(self._tmp, "a.py")
        with open(f, "w", encoding="utf-8") as fh:
            fh.write("l1\nl2\nl3\nl4\n")
        os.chdir(self._tmp)
        mgr = GroupManager(isolated=True)
        try:
            args = {"path": f, "new_text": "X\n", "start_line": 2, "end_line": 2}
            r = mgr.call("workspace", "line_range_edit", args, dry_run=True)
            self.assertEqual(r.status, "wrapped")
            preview, _ask = r.value
            self.assertIn("-l2", preview)

            resolve = make_worker_resolve(mgr, "workspace", "line_range_edit", args)
            self.assertIsNone(resolve(None, "no"))
            applied = resolve(None, "yes")
            self.assertIn("L2-2", applied)
            with open(f, encoding="utf-8") as fh:
                self.assertEqual(fh.read(), "l1\nX\nl3\nl4")
        finally:
            mgr.close_all()


class TestIsolatedStop(unittest.TestCase):
    def test_stop_kills_blocking_worker_fast(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        self._write_plugin(tmp, "blocker_plugin.py", textwrap.dedent('''
            import time
            from universal_agents.tool import tool

            @tool(description="block", group="slow")
            def blocker(seconds: int) -> str:
                time.sleep(seconds)
                return "done"
        '''))
        mgr = GroupManager(isolated=True, tools_dir=tmp)
        stop = threading.Event()
        timer = threading.Timer(0.3, stop.set)
        timer.start()
        started = time.monotonic()
        try:
            r = mgr.call("slow", "blocker", {"seconds": 60}, stop_check=stop.is_set)
        finally:
            timer.cancel()
            mgr.close_all()
        elapsed = time.monotonic() - started
        self.assertEqual(r.status, "error")
        self.assertIn("stopped", r.value)
        self.assertLess(elapsed, 5.0, "остановка должна убивать воркер, а не ждать")

    @staticmethod
    def _write_plugin(directory: str, filename: str, source: str) -> None:
        with open(os.path.join(directory, filename), "w", encoding="utf-8") as f:
            f.write(source)


def _dying_worker(*args, **kwargs):
    os._exit(1)


class TestSpawnFailureFallback(unittest.TestCase):
    def test_degrade_to_in_process_on_worker_death(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        with open(os.path.join(tmp, "fragile_plugin.py"), "w", encoding="utf-8") as f:
            f.write(textwrap.dedent('''
                from universal_agents.tool import tool

                @tool(description="ping", group="fragile")
                def ping() -> str:
                    return "pong"
            '''))
        mgr = GroupManager(isolated=True, tools_dir=tmp)
        try:
            with mock.patch.dict(os.environ, {"UA_TOOL_WORKER_TARGET": f"{__name__}:_dying_worker"}):
                result = mgr.call("fragile", "ping", {})
            self.assertEqual(result.status, "ok")
            self.assertEqual(result.value, "pong")
            self.assertFalse(mgr.isolated, "после сбоя spawn изоляция отключается")
        finally:
            mgr.close_all()


class TestPluginProxy(unittest.TestCase):
    def test_plugin_calls_host_via_proxy(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        with open(os.path.join(tmp, "proxy_plugin.py"), "w", encoding="utf-8") as f:
            f.write(textwrap.dedent('''
                from universal_agents.tool import tool
                from universal_agents import plugin_proxy

                @tool(description="plugin that talks to host", group="plugin")
                def chatty() -> str:
                    plugin_proxy.call("on_system_msg", "hello from plugin")
                    return f"host_says={plugin_proxy.call('history_len')}"
            '''))
        seen = []

        def proxy_handler(method, args, kwargs):
            if method == "on_system_msg":
                seen.append(args[0])
                return None
            if method == "history_len":
                return 7
            raise ValueError("unknown")

        mgr = GroupManager(isolated=True, tools_dir=tmp)
        mgr.proxy_handler = proxy_handler
        try:
            r = mgr.call("plugin", "chatty", {})
        finally:
            mgr.close_all()
        self.assertEqual(r.status, "ok")
        self.assertEqual(r.value, "host_says=7")
        self.assertEqual(seen, ["hello from plugin"])


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

    def test_read_uses_workspace_worker(self):
        from universal_agents.tools.fs import read
        f = os.path.join(self._tmp, "hello.txt")
        with open(f, "w", encoding="utf-8") as fh:
            fh.write("alpha\nbeta\n")
        responses = [
            AssistantMessage(content="Читаю файл.", tool_calls=[
                ToolCall(id="c1", name=read.__name__, arguments=json.dumps({"path": f}))]),
            AssistantMessage(content="Готово."),
        ]
        _agent, result, seen = self._run(responses, {read.__name__: read})
        self.assertIn("Готово", result)
        flat = [c for turn in seen for c in turn]
        self.assertTrue(any("alpha" in c and "beta" in c for c in flat), flat)

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
            self.assertEqual(fh.read(), "a\nX\nc")


if __name__ == "__main__":
    unittest.main()