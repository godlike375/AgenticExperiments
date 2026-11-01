from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Optional, TextIO

from universal_agents.config import Config
from universal_agents.tool import ToolOutput
from universal_agents.tool_registry import load_external_plugins

__all__ = ["ToolWorker", "invoke_tool"]


def _no_window_kwargs() -> dict:
    if sys.platform == "win32":
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        return {"startupinfo": si, "creationflags": 0x08000000}  # CREATE_NO_WINDOW
    return {}


def invoke_tool(func: Callable, args: dict, dry_run: bool = False) -> tuple[str, Any]:
    if dry_run:
        result = func(**args, dry_run=True)
        if isinstance(result, tuple) and len(result) == 3:
            preview, _resolve, ask = result
            return ("wrapped", (preview, ask))
        return ("ok", result)
    return ("ok", func(**args))


class ToolWorker:
    """Управляет единым изолированным воркером инструментов в отдельном дочернем процессе."""

    def __init__(
        self,
        tools_dir: Optional[str] = None,
        isolated: bool = True,
        poll_interval: float = 0.05,
    ) -> None:
        self.tools_dir = tools_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "tools"
        )
        self.isolated = isolated
        self._poll_interval = poll_interval
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._stderr_thread: Optional[threading.Thread] = None
        self._local_tools: Optional[dict] = None

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _ensure_proc(self) -> subprocess.Popen:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        project_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        current_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = project_parent + (os.pathsep + current_pp if current_pp else "")

        args = [
            sys.executable,
            "-m",
            "universal_agents.tool_worker",
            "--tools-dir",
            self.tools_dir,
        ]

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

        self._stderr_thread = threading.Thread(target=self._pump_stderr, daemon=True)
        self._stderr_thread.start()

        # Warmup ping
        try:
            self._send_raw({"name": "__ping__", "args": {}})
            self._read_line(timeout=5.0)
        except Exception as e:
            self.kill()
            raise RuntimeError(f"Failed to start tool worker process: {e}") from e

        return self._proc

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

    def _send_raw(self, payload: dict) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.poll() is not None:
            raise BrokenPipeError("Worker process is not running")
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        proc.stdin.write(line)
        proc.stdin.flush()

    def _read_line(self, timeout: float = 30.0, stop_check: Optional[Callable[[], bool]] = None) -> str:
        proc = self._proc
        if proc is None or proc.stdout is None:
            raise BrokenPipeError("Worker process stdout is unavailable")

        box: list[str] = []
        done = threading.Event()

        def _read() -> None:
            try:
                line = proc.stdout.readline()
                box.append(line)
            except Exception:
                box.append("")
            finally:
                done.set()

        reader = threading.Thread(target=_read, daemon=True)
        reader.start()

        deadline = time.time() + timeout
        while True:
            if done.wait(timeout=self._poll_interval):
                break
            if stop_check is not None and stop_check():
                self.kill()
                raise InterruptedError("Tool execution stopped by user")
            if time.time() > deadline:
                self.kill()
                raise TimeoutError(f"Tool execution timed out after {timeout}s")
            if proc.poll() is not None:
                raise BrokenPipeError(f"Worker died with code {proc.returncode}")

        if not box or not box[0]:
            raise BrokenPipeError("Empty response from worker process")
        return box[0]

    def call(
        self,
        name: str,
        args: dict,
        dry_run: bool = False,
        stop_check: Optional[Callable[[], bool]] = None,
        timeout: Optional[int] = None,
    ) -> tuple[str, Any]:
        """Вызов инструмента. Возвращает (status, value), где status: 'ok' | 'wrapped' | 'error'."""
        if not self.isolated:
            return self._call_local(name, args, dry_run)

        timeout_sec = timeout if timeout is not None else Config.TOOL_WORKER_TIMEOUT
        with self._lock:
            try:
                self._ensure_proc()
                req = {
                    "name": name,
                    "args": args,
                    "dry_run": dry_run,
                    "cwd": os.getcwd(),
                }
                self._send_raw(req)
                resp_line = self._read_line(timeout=timeout_sec, stop_check=stop_check)
                resp = json.loads(resp_line)

                # Синхронизация CWD с хостом
                new_cwd = resp.get("cwd")
                if new_cwd and os.path.isdir(new_cwd) and os.path.abspath(new_cwd) != os.getcwd():
                    try:
                        os.chdir(new_cwd)
                    except Exception:
                        pass

                return resp.get("status", "error"), resp.get("value")
            except InterruptedError as e:
                return "error", str(e)
            except TimeoutError as e:
                return "error", str(e)
            except Exception as e:
                self.kill()
                return "error", f"Tool worker failure: {e}"

    def _call_local(self, name: str, args: dict, dry_run: bool) -> tuple[str, Any]:
        if self._local_tools is None:
            self._local_tools = load_external_plugins(self.tools_dir)
        func = self._local_tools.get(name)
        if func is None:
            return "error", f"Tool '{name}' not found"
        try:
            return invoke_tool(func, args, dry_run=dry_run)
        except Exception as e:
            return "error", f"{type(e).__name__}: {e}"

    def kill(self) -> None:
        """Мгновенно уничтожает процесс воркера при зависании или отмене."""
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=0.3)
                except Exception:
                    proc.kill()
        except Exception:
            pass
        finally:
            for s in (proc.stdin, proc.stdout, proc.stderr):
                if s is not None:
                    try:
                        s.close()
                    except Exception:
                        pass

    def close(self) -> None:
        self.kill()


def _open_protocol_stream() -> TextIO:
    """Отделяет протокольный stdout от любого стороннего вывода инструментов/принтов."""
    try:
        fd = os.dup(sys.stdout.fileno())
        protocol = os.fdopen(fd, "w", encoding="utf-8", newline="\n")
        sys.stdout.flush()
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        sys.stdout = sys.stderr
        return protocol
    except (OSError, AttributeError, ValueError):
        return sys.stdout


def _worker_main() -> None:
    protocol = _open_protocol_stream()
    parser = argparse.ArgumentParser()
    parser.add_argument("--tools-dir", required=True)
    opts = parser.parse_args()

    tools = load_external_plugins(opts.tools_dir)
    sys.stderr.write(f"[tool_worker] ready with {len(tools)} tools: {sorted(tools.keys())}\n")
    sys.stderr.flush()

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        try:
            req = json.loads(line)
        except Exception:
            break

        name = req.get("name")
        if name == "__ping__":
            protocol.write(json.dumps({"status": "ok", "value": "pong"}) + "\n")
            protocol.flush()
            continue

        req_cwd = req.get("cwd")
        if req_cwd and os.path.isdir(req_cwd):
            try:
                os.chdir(req_cwd)
            except Exception:
                pass

        func = tools.get(name)
        if func is None:
            resp = {"status": "error", "value": f"Unknown tool '{name}'", "cwd": os.getcwd()}
        else:
            try:
                status, val = invoke_tool(func, req.get("args", {}), req.get("dry_run", False))
                images = []
                if isinstance(val, ToolOutput):
                    images = val.images
                    val = val.text
                resp = {"status": status, "value": val, "images": images, "cwd": os.getcwd()}
            except Exception as e:
                resp = {"status": "error", "value": f"{type(e).__name__}: {e}", "cwd": os.getcwd()}

        protocol.write(json.dumps(resp, ensure_ascii=False) + "\n")
        protocol.flush()


if __name__ == "__main__":
    _worker_main()
