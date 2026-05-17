import os
import subprocess
from universal_agents.tool import tool


class UnifiedDockerAgent:
    CONTAINER_NAME = "llm-unified-agent"
    IMAGE = "python:3.11-slim"
    # Запоминаем последний путь, по умолчанию - текущая папка
    _current_repo_path = os.getcwd()

    @classmethod
    def _is_container_running(cls) -> bool:
        """Проверяет, существует ли и запущен ли контейнер."""
        cmd = ["docker", "inspect", "-f", "{{.State.Running}}", cls.CONTAINER_NAME]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0 and result.stdout.strip() == "true"

    @classmethod
    def start(cls, repo_path: str) -> str:
        cls._current_repo_path = os.path.abspath(repo_path)

        if not os.path.isdir(cls._current_repo_path):
            return f"Error: repo not found: {cls._current_repo_path}"

        # Если контейнер уже запущен, просто возвращаем статус
        if cls._is_container_running():
            return f"Container {cls.CONTAINER_NAME} is already running at {cls._current_repo_path}."

        # Если контейнер существует, но остановлен — удалим его, чтобы пересоздать с новым путем
        subprocess.run(["docker", "rm", "-f", cls.CONTAINER_NAME], capture_output=True)

        # Поиск .git
        git_dir = None
        current_path = cls._current_repo_path
        while True:
            potential_git = os.path.join(current_path, ".git")
            if os.path.isdir(potential_git):
                git_dir = potential_git
                break
            parent = os.path.dirname(current_path)
            if parent == current_path:
                break
            current_path = parent

        cmd = [
            "docker", "run", "-dit",
            "--name", cls.CONTAINER_NAME,
            "--network", "none",
            "--memory", "1g",
            "--cpus", "1",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "-v", f"{cls._current_repo_path}:/workspace",
            "-w", "/workspace",
        ]

        if git_dir:
            cmd.extend(["-v", f"{git_dir}:/workspace/.git:ro"])

        cmd.extend([cls.IMAGE, "sleep", "infinity"])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return f"Error starting container: {result.stderr}"

        return f"Unified agent started. Workspace: {cls._current_repo_path}"

    @classmethod
    def _ensure_container(cls):
        """Гарантирует, что контейнер запущен перед выполнением команд."""
        if not cls._is_container_running():
            cls.start(cls._current_repo_path)

    @classmethod
    def execute_bash(cls, command: str, timeout: int = 60) -> str:
        cls._ensure_container()
        cmd = ["docker", "exec", cls.CONTAINER_NAME, "bash", "-c", command]
        return cls._run_cmd(cmd, timeout)

    @classmethod
    def execute_python(cls, code: str, timeout: int = 60) -> str:
        cls._ensure_container()
        cmd = ["docker", "exec", "-i", cls.CONTAINER_NAME, "python3", "-"]
        return cls._run_cmd(cmd, timeout, stdin_input=code)

    @classmethod
    def _run_cmd(cls, cmd: list, timeout: int, stdin_input: str = None) -> str:
        try:
            result = subprocess.run(
                cmd,
                input=stdin_input,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=timeout
            )

            output = result.stdout or ""

            if result.returncode != 0:
                error_msg = result.stderr or result.stdout or "Unknown error"
                raise RuntimeError(
                    f"Command failed with exit code {result.returncode}:\n{error_msg}"
                )

            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}"

            output += f"\n[exit_code]: {result.returncode}"
            return output

        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Command timed out after {timeout}s")
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Unexpected error during command execution: {e}")

        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Command timed out after {timeout}s")
        except RuntimeError:
            # Пробрасываем наши ошибки дальше
            raise
        except Exception as e:
            raise RuntimeError(f"Unexpected error during command execution: {e}")

    @classmethod
    def stop(cls) -> str:
        subprocess.run(["docker", "rm", "-f", cls.CONTAINER_NAME], capture_output=True)
        return "Unified agent stopped."


@tool(
    description="Start a persistent sandbox container for a repository. Call this to set the working directory.",
    repo_path=("str", "Absolute or relative path to the git repository on the host")
)
def start_sandbox(repo_path: str):
    return UnifiedDockerAgent.start(repo_path)


@tool(
    description="Execute a bash command. Auto-starts container in current directory if not already running.",
    command=("str", "The bash command to run"),
    timeout=("int", "Timeout in seconds (default 60)")
)
def run_bash(command: str, timeout: int = 60):
    return UnifiedDockerAgent.execute_bash(command, timeout)


@tool(
    description="Execute Python code. Auto-starts container in current directory if not already running.",
    code=("str", "The Python code to execute"),
    timeout=("int", "Timeout in seconds (default 60)")
)
def run_python(code: str, timeout: int = 60):
    return UnifiedDockerAgent.execute_python(code, timeout)


@tool(
    description="Stop and remove the persistent sandbox container."
)
def stop_sandbox():
    return UnifiedDockerAgent.stop()