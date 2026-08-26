"""Общий помощник для запуска внешних команд (host-shell и sandbox).

Централизует сборку текстового результата и обработку ошибок, которые раньше
дублировались в `tools/host_shell.py` (`_run`) и `tools/sandbox.py` (`_run_cmd`).

Важно: запуск через Popen с опросом, чтобы можно было прервать выполнение по
требованию пользователя (см. set_interrupt_event / _interrupt_requested) — иначе
долгий вызов инструмента блокирует поток и остановка оказывается невозможной.
"""

from __future__ import annotations

import subprocess
import threading
import time

from universal_agents.exceptions import GenerationInterrupted

# Текущий "прерыватель" (обычно stop_event агента), устанавливается на время
# выполнения инструмента и сбрасывается после. Гонок нет: инструменты агента
# исполняются одним рабочим потоком.
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


def run_capture(
    cmd: list[str],
    timeout: int,
    cwd: str = None,
    stdin: str = None,
) -> str:
    """Запускает ``cmd`` и возвращает текст результата.

    Собирает stdout + stderr + строку с кодом завершения в единый текст.
    При ненулевом коде завершения бросает ``RuntimeError`` с этим же текстом
    (чтобы вызывающая сторона могла трактовать результат как ошибку).
    Если во время выполнения запрошена остановка (interrupt event), процесс
    принудительно завершается и бросается ``GenerationInterrupted``.
    """
    effective_timeout = timeout if timeout and timeout > 0 else 60
    stdin_stream = subprocess.PIPE if stdin is not None else subprocess.DEVNULL

    proc = subprocess.Popen(
        cmd,
        stdin=stdin_stream,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=cwd,
        bufsize=1,
    )

    out_chunks: list[str] = []
    err_chunks: list[str] = []

    def _reader(stream, sink: list[str]) -> None:
        try:
            for chunk in iter(stream.readline, ''):
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
            proc.stdin.write(stdin)
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
                proc.kill()
                break
            if time.time() > deadline:
                timed_out = True
                proc.kill()
                break
            time.sleep(0.05)
    finally:
        t_out.join()
        t_err.join()

    if interrupted:
        raise GenerationInterrupted("Command interrupted by user")
    if timed_out:
        raise TimeoutError(f"Command timed out after {effective_timeout}s")

    output = "".join(out_chunks)
    err = "".join(err_chunks)
    if err:
        output += f"\n[stderr]\n{err}"
    output += f"\n[exit_code]: {proc.returncode}"
    if proc.returncode != 0:
        raise RuntimeError(output)
    return output
