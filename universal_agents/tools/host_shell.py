"""Инструменты исполнения shell-команд на хосте (без Docker-контейнера).

В отличие от sandbox.py (который гоняет команды через docker + bash),
эти инструменты исполняют команды напрямую на хост-машине:
- run_powershell — PowerShell (Windows)
- run_bash_host — bash (на системах, где он есть; на Windows можно через git-bash/WSL)

Оба помечены path_safety=True: если в команде встречается путь вне корня
проекта (папки с .git), агент запросит подтверждение у пользователя.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time

from universal_agents.tool import tool
from universal_agents.constants import ENVIRONMENT_PREFIX, ENVIRONMENT_PREFIX_END
from universal_agents.config import Config


def _run(cmd_list: list[str], timeout: int, cwd: str = None) -> str:
    """Выполняет команду и форматирует результат как текст."""
    try:
        result = subprocess.run(
            cmd_list,
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

    output = result.stdout or ""
    if result.stderr:
        output += f"\n[stderr]\n{result.stderr}"
    output += f"\n[exit_code]: {result.returncode}"
    if result.returncode != 0:
        raise RuntimeError(output)
    return output


def _git_bash_candidates() -> list[str]:
    """Стандартные места установки Git Bash на Windows."""
    candidates = []
    if Config.GIT_BASH_PATH:
        candidates.append(Config.GIT_BASH_PATH)
    candidates.extend([
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files\Git\usr\bin\bash.exe",
        r"C:\Program Files (x86)\Git\bin\bash.exe",
        r"C:\Program Files (x86)\Git\usr\bin\bash.exe",
    ])
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidates.append(os.path.join(local, "Programs", "Git", "bin", "bash.exe"))
        candidates.append(os.path.join(local, "Programs", "Git", "usr", "bin", "bash.exe"))
    return [c for c in candidates if c and os.path.isfile(c)]


def _is_windows() -> bool:
    return os.name == "nt" or sys.platform.startswith("win")


def _resolve_bash_backend() -> str:
    """Определяет, каким bash исполнять, согласно Config.BASH_BACKEND."""
    backend = (Config.BASH_BACKEND or "auto").strip().lower()
    if backend == "gitbash":
        return "gitbash"
    if backend == "wsl":
        return "wsl"
    if backend == "system":
        return "system"
    # auto: при наличии Git Bash предпочитаем его, иначе WSL
    if _git_bash_candidates():
        return "gitbash"
    if shutil.which("bash"):
        return "system"
    if _is_windows() and shutil.which("wsl"):
        return "wsl"
    return "system"


def _build_bash_command(command: str) -> list[str]:
    """Собирает argv для запуска bash-команды согласно выбранному бэкенду."""
    backend = _resolve_bash_backend()

    if backend == "gitbash":
        bash = _git_bash_candidates()[0] if _git_bash_candidates() else None
        if not bash:
            raise RuntimeError("Git Bash not found. Install Git or set Config.GIT_BASH_PATH")
        # Git Bash: -c запускает команду; --login не нужен
        return [bash, "-c", command]

    if backend == "wsl":
        wsl = shutil.which("wsl") or shutil.which("bash")
        if not wsl:
            raise RuntimeError("WSL not found")
        if "wsl.exe" in os.path.basename(wsl).lower():
            return [wsl, "bash", "-c", command]
        return [wsl, "-c", command]

    # system: просто bash -c
    bash = shutil.which("bash")
    if not bash:
        raise RuntimeError("bash executable not found on this host")
    return [bash, "-c", command]


@tool(
    description="Execute a PowerShell command directly on the host machine (Windows), without a Docker container. "
                "Any filesystem path mentioned in the command that lies OUTSIDE the project root (folder with .git) "
                "will trigger a user confirmation prompt for safety. Paths inside the project run without prompting.",
    short_description="run powershell on host",
    path_safety=True,
    command=("str", "The PowerShell command to run"),
    timeout=("int", "Optional timeout in seconds (default 60)"),
)
def run_powershell(command: str, timeout: int = 60) -> str:
    pwsh = shutil.which("pwsh") or shutil.which("powershell")
    if not pwsh:
        return f"{ENVIRONMENT_PREFIX} Error PowerShell executable not found{ENVIRONMENT_PREFIX_END}"
    # -NoProfile: без профилей пользователя; -NonInteractive: без интерактива
    return _run([pwsh, "-NoProfile", "-NonInteractive", "-Command", command], timeout)


@tool(
    description="Execute a bash command directly on the host machine, without a Docker container. "
                "On Windows uses Git Bash or WSL depending on Config.BASH_BACKEND. "
                "Any filesystem path mentioned in the command that lies OUTSIDE the project root (folder with .git) "
                "will trigger a user confirmation prompt for safety. Paths inside the project run without prompting.",
    short_description="run bash on host",
    path_safety=True,
    command=("str", "The bash command to run"),
    timeout=("int", "Optional timeout in seconds (default 60)"),
)
def run_bash_host(command: str, timeout: int = 60) -> str:
    try:
        argv = _build_bash_command(command)
    except RuntimeError as e:
        return f"{ENVIRONMENT_PREFIX} Error {e}{ENVIRONMENT_PREFIX_END}"
    return _run(argv, timeout)