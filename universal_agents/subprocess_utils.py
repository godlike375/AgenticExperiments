
from __future__ import annotations

import ctypes
import os
import subprocess
import threading
import time

from universal_agents.config import Config
from universal_agents.constants import ENVIRONMENT_PREFIX, ENVIRONMENT_PREFIX_END
from universal_agents.exceptions import GenerationInterrupted

_IS_WINDOWS = os.name == "nt"

if _IS_WINDOWS:
    _kernel32 = ctypes.windll.kernel32

    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x08

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", ctypes.c_uint32),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", ctypes.c_uint32),
            ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
            ("PriorityClass", ctypes.c_uint32),
            ("SchedulingClass", ctypes.c_uint32),
        ]

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]


def _create_kill_job():
    if not _IS_WINDOWS:
        return None, None
    try:
        job = _kernel32.CreateJobObjectW(None, None)
        if not job:
            return None, None

        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        info.BasicLimitInformation.ActiveProcessLimit = 512

        result = _kernel32.SetInformationJobObject(
            job, 2,  # JobObjectExtendedLimitInformation
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not result:
            _kernel32.CloseHandle(job)
            return None, None

        def assign(proc_handle):
            _kernel32.AssignProcessToJobObject(job, proc_handle)

        return job, assign
    except Exception:
        return None, None


def _kill_process_tree(pid: int) -> None:
    try:
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(pid)],
            capture_output=True,
            timeout=5,
        )
    except Exception:
        pass


def _safe_close_job(job) -> None:
    if job and _IS_WINDOWS:
        try:
            _kernel32.CloseHandle(job)
        except Exception:
            pass


def _force_kill_proc(proc, job_handle) -> None:
    try:
        proc.terminate()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass

    if _IS_WINDOWS:
        _safe_close_job(job_handle)
        try:
            _kill_process_tree(proc.pid)
        except Exception:
            pass
        try:
            proc.kill()
        except Exception:
            pass

_interrupt_event = None
_interrupt_lock = threading.Lock()


def set_interrupt_event(ev) -> None:
    global _interrupt_event
    with _interrupt_lock:
        _interrupt_event = ev


def clear_interrupt_event() -> None:
    global _interrupt_event
    with _interrupt_lock:
        _interrupt_event = None


def _interrupt_requested() -> bool:
    with _interrupt_lock:
        ev = _interrupt_event
    return ev is not None and ev.is_set()


def truncate_middle(
    text: str,
    max_chars: int = None,
    max_lines: int = None,
) -> str:
    max_chars = max_chars or Config.MAX_READ_CHARS_PER_CALL
    max_lines = max_lines or Config.MAX_READ_LINES_PER_CALL

    lines = text.split("\n")
    total_chars = len(text)
    total_lines = len(lines)

    if total_chars <= max_chars and total_lines <= max_lines:
        return text

    chars_ratio = total_chars / max_chars if max_chars else 0
    lines_ratio = total_lines / max_lines if max_lines else 0

    if chars_ratio >= lines_ratio:
        head_budget = int(max_chars * 0.4)
        tail_budget = int(max_chars * 0.4)
    else:
        avg_line_len = total_chars / max(total_lines, 1)
        head_budget = int(max_lines * 0.4 * avg_line_len)
        tail_budget = int(max_lines * 0.4 * avg_line_len)

    head_lines: list[str] = []
    head_chars = 0
    for line in lines:
        cost = len(line) + 1  # +1 за '\n'
        if head_chars + cost > head_budget:
            break
        head_lines.append(line)
        head_chars += cost

    tail_lines: list[str] = []
    tail_chars = 0
    for line in reversed(lines):
        cost = len(line) + 1
        if tail_chars + cost > tail_budget:
            break
        tail_lines.append(line)
        tail_chars += cost
    tail_lines.reverse()

    head_end_idx = len(head_lines)
    tail_start_idx = len(lines) - len(tail_lines)
    if head_end_idx > tail_start_idx:
        mid = len(lines) // 2
        head_lines = lines[: mid // 2]
        tail_lines = lines[-(mid // 2):]
        head_end_idx = len(head_lines)
        tail_start_idx = len(lines) - len(tail_lines)

    skipped_lines = tail_start_idx - head_end_idx
    skipped_chars = sum(len(lines[i]) + 1 for i in range(head_end_idx, tail_start_idx))

    marker = (
        f"\n{ENVIRONMENT_PREFIX} ... [{skipped_lines} lines / ~{skipped_chars} chars skipped] ...{ENVIRONMENT_PREFIX_END}\n"
    )

    result = "\n".join(head_lines) + marker + "\n".join(tail_lines)
    return result


def _decode_bytes(data: bytes, encoding: str, fallback_encoding: str | None) -> str:
    try:
        return data.decode(encoding)
    except (UnicodeDecodeError, LookupError):
        if fallback_encoding:
            try:
                return data.decode(fallback_encoding, errors="replace")
            except LookupError:
                pass
        return data.decode(encoding, errors="replace")


def run_capture(
    cmd: list[str],
    timeout: int,
    cwd: str = None,
    stdin: str = None,
    encoding: str = "utf-8",
    fallback_encoding: str | None = None,
) -> str:
    effective_timeout = timeout if timeout and timeout > 0 else 60
    stdin_stream = subprocess.PIPE if stdin is not None else subprocess.DEVNULL
    raw_mode = fallback_encoding is not None

    job_handle, job_assign = _create_kill_job()

    if raw_mode:
        proc = subprocess.Popen(
            cmd,
            stdin=stdin_stream,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            cwd=cwd,
        )
    else:
        proc = subprocess.Popen(
            cmd,
            stdin=stdin_stream,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding=encoding,
            errors="replace",
            cwd=cwd,
        )

    if job_assign is not None:
        try:
            job_assign(proc._handle)
        except Exception:
            pass

    out_chunks: list = []
    err_chunks: list = []

    def _reader(stream, sink: list) -> None:
        try:
            for chunk in iter(stream.readline, b'' if raw_mode else ''):
                sink.append(chunk)
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass

    t_out = threading.Thread(target=_reader, args=(proc.stdout, out_chunks), daemon=True)
    t_err = threading.Thread(target=_reader, args=(proc.stderr, err_chunks), daemon=True)
    t_out.start()
    t_err.start()

    if stdin is not None:
        try:
            proc.stdin.write(stdin.encode(encoding) if raw_mode else stdin)
        except Exception:
            pass
        try:
            proc.stdin.close()
        except Exception:
            pass

    interrupted = False
    timed_out = False
    deadline = time.time() + effective_timeout
    try:
        while True:
            rc = proc.poll()
            if rc is not None:
                break
            if _interrupt_requested():
                interrupted = True
                _force_kill_proc(proc, job_handle)
                break
            if time.time() > deadline:
                timed_out = True
                _force_kill_proc(proc, job_handle)
                break
            time.sleep(0.05)
    finally:
        t_out.join(timeout=5)
        t_err.join(timeout=5)
        _safe_close_job(job_handle)

    if interrupted:
        raise GenerationInterrupted("Command interrupted by user")
    if timed_out:
        raise TimeoutError(f"Command timed out after {effective_timeout}s")

    if raw_mode:
        output = _decode_bytes(b"".join(out_chunks), encoding, fallback_encoding)
        err = _decode_bytes(b"".join(err_chunks), encoding, fallback_encoding)
    else:
        output = "".join(out_chunks)
        err = "".join(err_chunks)
    if err:
        output += f"\n[stderr]\n{err}"
    output += f"\n[exit_code]: {proc.returncode}"
    if proc.returncode != 0:
        raise RuntimeError(output)
    return output
