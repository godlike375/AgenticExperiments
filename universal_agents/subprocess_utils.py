"""Общий помощник для запуска внешних команд (host-shell и sandbox).

Централизует сборку текстового результата и обработку ошибок, которые раньше
дублировались в `tools/host_shell.py` (`_run`) и `tools/sandbox.py` (`_run_cmd`).
"""

from __future__ import annotations

import subprocess


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
    """
    try:
        result = subprocess.run(
            cmd,
            input=stdin,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            cwd=cwd,
        )
    except subprocess.TimeoutExpired:
        raise TimeoutError(f"Command timed out after {timeout}s")
    except FileNotFoundError as e:
        raise RuntimeError(f"Executable not found: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during command execution: {e}")

    output = result.stdout or ""
    if result.stderr:
        output += f"\n[stderr]\n{result.stderr}"
    output += f"\n[exit_code]: {result.returncode}"
    if result.returncode != 0:
        raise RuntimeError(output)
    return output
