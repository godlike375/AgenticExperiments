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

from universal_agents.config import Config
from universal_agents.constants import ENVIRONMENT_PREFIX, ENVIRONMENT_PREFIX_END
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


def truncate_middle(
    text: str,
    max_chars: int = None,
    max_lines: int = None,
) -> str:
    """Обрезает середину вывода, оставляя начало и конец.

    Если ``text`` превышает ``max_chars`` символов ИЛИ ``max_lines`` строк,
    из середины удаляется блок, а вместо него вставляется маркер
    ``... [N lines / M chars skipped] ...``.  Началу выделяется ~40 %
    лимита, концу — ~40 %, маркер занимает оставшиеся ~20 %.
    """
    max_chars = max_chars or Config.MAX_READ_CHARS_PER_CALL
    max_lines = max_lines or Config.MAX_READ_LINES_PER_CALL

    lines = text.split("\n")
    total_chars = len(text)
    total_lines = len(lines)

    # Ничего обрезать не нужно
    if total_chars <= max_chars and total_lines <= max_lines:
        return text

    # --- Определяем, что является «лимитирующим» фактором ---
    # Используем тот лимит, который нарушается сильнее.
    chars_ratio = total_chars / max_chars if max_chars else 0
    lines_ratio = total_lines / max_lines if max_lines else 0

    if chars_ratio >= lines_ratio:
        # Лимит по символам
        head_budget = int(max_chars * 0.4)
        tail_budget = int(max_chars * 0.4)
    else:
        # Лимит по строкам — переводим в «виртуальные» символы
        avg_line_len = total_chars / max(total_lines, 1)
        head_budget = int(max_lines * 0.4 * avg_line_len)
        tail_budget = int(max_lines * 0.4 * avg_line_len)

    # --- Собираем голову ---
    head_lines: list[str] = []
    head_chars = 0
    for line in lines:
        cost = len(line) + 1  # +1 за '\n'
        if head_chars + cost > head_budget:
            break
        head_lines.append(line)
        head_chars += cost

    # --- Собираем хвост ---
    tail_lines: list[str] = []
    tail_chars = 0
    for line in reversed(lines):
        cost = len(line) + 1
        if tail_chars + cost > tail_budget:
            break
        tail_lines.append(line)
        tail_chars += cost
    tail_lines.reverse()

    # Не допускаем, чтобы голова и хвост накладывались
    head_end_idx = len(head_lines)
    tail_start_idx = len(lines) - len(tail_lines)
    if head_end_idx > tail_start_idx:
        # Перераспределяем поровну
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
