
from __future__ import annotations

import dataclasses
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import namedtuple
from typing import Callable, Optional

from universal_agents import tool_session
from universal_agents.config import Config
from universal_agents.tool_registry import load_external_plugins
from universal_agents.tool_worker import invoke_tool

__all__ = ["GroupManager", "WorkerCallResult"]


WorkerCallResult = namedtuple("WorkerCallResult", "status value session")


def _worker_died_message(group: str) -> str:
    return f"worker for group '{group}' died"


def _no_window_kwargs() -> dict:
    if sys.platform == "win32":
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        return {"startupinfo": si, "creationflags": 0x08000000}  # CREATE_NO_WINDOW
    return {}


def tool_config_snapshot() -> dict:
    snapshot = {}
    for field in dataclasses.fields(Config):
        snapshot[field.name] = getattr(Config, field.name)
    return snapshot


def default_group_session(group: str) -> dict:
    if group == "workspace":
        return {"cwd": os.getcwd()}
    if group == "desktop":
        return {"last_scale": None}
    if group == "sandbox":
        return {"container": None, "repo_path": None}
    return {}


class SubprocessGroupWorker:

    def __init__(
            self,
            group: str,
            session: dict,
            config_overrides: dict,
            project_root: Optional[str],
            tools_dir: Optional[str],
            poll_interval: float = None,
            kill_grace: float = None,
            proxy_handler: Optional[Callable] = None,
            startup_timeout: float = 30.0,
    ):
        self.group = group
        self._session = session if isinstance(session, dict) else {}
        self._proxy_handler = proxy_handler
        self._poll_interval = (
            poll_interval if poll_interval is not None else Config.TOOL_WORKER_POLL_INTERVAL
        )
        self._kill_grace = (
            kill_grace if kill_grace is not None else Config.TOOL_WORKER_KILL_GRACE
        )
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._stderr_thread: Optional[threading.Thread] = None
        overrides_json = json.dumps(config_overrides or {}, ensure_ascii=False)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["UA_WORKER_CONFIG_OVERRIDES"] = overrides_json

        tools_dir = tools_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "tools"
        )
        args = [
            sys.executable, "-m", "universal_agents.subprocess_worker",
            "--group", group,
            "--tools-dir", tools_dir,
        ]
        if project_root:
            args += ["--project-root", project_root]

        try:
            self._proc = subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                env=env,
                **_no_window_kwargs(),
            )
        except Exception as e:
            raise RuntimeError(f"failed to launch worker for group '{group}': {e}")

        self._stderr_thread = threading.Thread(target=self._pump_stderr, daemon=True)
        self._stderr_thread.start()

        self._warmup(timeout=startup_timeout)


    def _pump_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        try:
            for line in proc.stderr:
                sys.stderr.write(line)
                sys.stderr.flush()
        except Exception:
            pass


    def _warmup(self, timeout: float) -> None:
        try:
            self.call("__noop__", {}, self._session, timeout=timeout)
        except Exception:
            pass

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def kill(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=self._kill_grace)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            try:
                if stream is not None:
                    stream.close()
            except Exception:
                pass


    def call(
            self,
            name: str,
            args: dict,
            session: dict,
            dry_run: bool = False,
            stop_check: Optional[Callable[[], bool]] = None,
            timeout: Optional[int] = None,
    ) -> WorkerCallResult:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return WorkerCallResult("error", _worker_died_message(self.group), session)
        assert proc.stdin is not None and proc.stdout is not None

        req = json.dumps(
            {"name": name, "args": args, "session": session, "dry_run": dry_run},
            ensure_ascii=False,
        ) + "\n"

        deadline = time.time() + (timeout if timeout is not None else Config.TOOL_WORKER_TIMEOUT)

        with self._lock:
            try:
                proc.stdin.write(req)
                proc.stdin.flush()
            except (BrokenPipeError, OSError):
                return WorkerCallResult("error", _worker_died_message(self.group), session)

            box: list = []
            done = threading.Event()

            def _read() -> None:
                try:
                    line = proc.stdout.readline()
                except Exception:
                    line = ""
                box.append(line)
                done.set()

            threading.Thread(target=_read, daemon=True).start()

            while True:
                if done.wait(timeout=self._poll_interval):
                    break
                if stop_check is not None and stop_check():
                    self.kill()
                    return WorkerCallResult("error", "stopped", session)
                if time.time() > deadline:
                    self.kill()
                    return WorkerCallResult("error", f"tool '{name}' timed out", session)

        line = box[0] if box else ""
        if not line:
            return WorkerCallResult("error", _worker_died_message(self.group), session)
        try:
            resp = json.loads(line)
        except json.JSONDecodeError as e:
            return WorkerCallResult("error", f"bad response from worker: {e}", session)
        return WorkerCallResult(
            resp.get("status", "error"),
            resp.get("value"),
            resp.get("session", session),
        )


class LocalWorker:

    def __init__(
            self,
            group: str,
            handler_resolver: Optional[Callable[[str], Optional[Callable]]] = None,
            tools_dir: Optional[str] = None,
    ):
        self.group = group
        self._handler_resolver = handler_resolver
        self._tools_dir = tools_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "tools"
        )
        self._funcs_cache: Optional[dict] = None

    def is_alive(self) -> bool:
        return True

    def call(
            self,
            name: str,
            args: dict,
            session: dict,
            dry_run: bool = False,
            stop_check: Optional[Callable[[], bool]] = None,
            timeout: Optional[int] = None,
    ) -> WorkerCallResult:
        func = self._resolve_func(name)
        if func is None:
            return WorkerCallResult(
                "error", f"tool '{name}' not found for in-process execution", session
            )
        token = tool_session.set_session(session)
        try:
            cwd = session.get("cwd")
            if cwd and os.path.isdir(cwd):
                try:
                    os.chdir(cwd)
                except Exception:
                    pass
            status, payload = invoke_tool(func, args, dry_run)
            return WorkerCallResult(status, payload, session)
        except Exception as e:
            return WorkerCallResult("error", f"{type(e).__name__}: {e}", session)
        finally:
            tool_session.reset_session(token)

    def kill(self) -> None:
        pass

    def _resolve_func(self, name: str) -> Optional[Callable]:
        if self._handler_resolver is not None:
            handler = self._handler_resolver(name)
            if handler is not None:
                return handler
        if self._funcs_cache is None:
            self._funcs_cache = load_external_plugins(self._tools_dir)
        return self._funcs_cache.get(name)


class GroupManager:

    def __init__(
            self,
            isolated: bool,
            config_overrides: Optional[dict] = None,
            project_root: Optional[str] = None,
            tools_dir: Optional[str] = None,
            handler_resolver: Optional[Callable[[str], Optional[Callable]]] = None,
    ):
        self.isolated = isolated
        self._config_overrides = (
            config_overrides if config_overrides is not None else tool_config_snapshot()
        )
        self._project_root = project_root
        self._tools_dir = tools_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "tools"
        )
        self.handler_resolver = handler_resolver
        self.proxy_handler: Optional[Callable] = None
        self._sessions: dict[str, dict] = {}
        self._workers: dict[str, object] = {}


    def session(self, group: str) -> dict:
        session = self._sessions.get(group)
        if session is None:
            session = default_group_session(group)
            self._sessions[group] = session
        return session


    def call(
            self,
            group: str,
            name: str,
            args: dict,
            dry_run: bool = False,
            stop_check: Optional[Callable[[], bool]] = None,
            timeout: Optional[int] = None,
    ) -> WorkerCallResult:
        worker = self._get_worker(group)
        result = worker.call(name, args, self.session(group), dry_run, stop_check, timeout)
        if self.isolated and result.status == "error" and result.value == _worker_died_message(group):
            self._workers.pop(group, None)
            self._degrade_to_in_process("worker failed to start")
            worker = self._get_worker(group)
            result = worker.call(name, args, self.session(group), dry_run, stop_check, timeout)
        if isinstance(result.session, dict):
            self._sessions[group] = result.session
        self._sync_host_cwd(group)
        return result

    def _get_worker(self, group: str):
        worker = self._workers.get(group)
        if worker is not None and worker.is_alive():
            return worker
        try:
            worker = self._make_worker(group)
        except Exception as e:
            self._degrade_to_in_process(f"worker start error: {e}")
            worker = self._make_worker(group)
        self._workers[group] = worker
        return worker

    def _make_worker(self, group: str):
        if self.isolated:
            return SubprocessGroupWorker(
                group,
                session=self.session(group),
                config_overrides=self._config_overrides,
                project_root=self._project_root,
                tools_dir=self._tools_dir,
                proxy_handler=self.proxy_handler,
            )
        return LocalWorker(
            group,
            handler_resolver=self.handler_resolver,
            tools_dir=self._tools_dir,
        )

    def _sync_host_cwd(self, group: str) -> None:
        cwd = self._sessions.get(group, {}).get("cwd")
        if cwd and os.path.isdir(cwd):
            try:
                if os.path.abspath(cwd) != os.getcwd():
                    os.chdir(cwd)
            except Exception:
                pass

    def _degrade_to_in_process(self, reason: str) -> None:
        if self.isolated:
            self.isolated = False
            print(
                f"[tool isolation] {reason}; falling back to in-process tool execution "
                f"for this session."
            )


    def kill(self, group: str) -> None:
        worker = self._workers.pop(group, None)
        if worker is not None:
            worker.kill()

    def close_all(self) -> None:
        for worker in list(self._workers.values()):
            worker.kill()
        self._workers.clear()


def make_worker_resolve(
        manager: GroupManager,
        group: str,
        name: str,
        args: dict,
        stop_check: Optional[Callable[[], bool]] = None,
):

    def resolve(agent, text):
        raw = (text or "").strip().lower()
        first = re.sub(r"[^\w]+", "", raw.split(" ", 1)[0]) if raw else ""
        if first in ("yes", "y", "true", "да"):
            result = manager.call(group, name, args, dry_run=False, stop_check=stop_check)
            if result.status == "error":
                return f"Error: {result.value}"
            return result.value
        return None

    return resolve