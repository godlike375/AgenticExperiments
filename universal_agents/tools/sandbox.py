import os
import subprocess
from universal_agents.tool import tool
from universal_agents.constants import err
from universal_agents.subprocess_utils import run_capture
from universal_agents import tool_session


class UnifiedDockerAgent:
    CONTAINER_NAME = "llm-unified-agent"
    IMAGE = "python:3.11-slim"

    @classmethod
    def _session(cls) -> dict:
        """Состояние группы sandbox (docs/TOOL_ISOLATION_SPEC.md §5.3): имя контейнера и
        рабочая папка репозитория. Воркер без памяти — состояние приходит с каждым вызовом."""
        return tool_session.get_session()

    @classmethod
    def _container_name(cls) -> str:
        return cls._session().get("container") or cls.CONTAINER_NAME

    @classmethod
    def _is_container_running(cls) -> bool:
        """Проверяет, существует ли и запущен ли контейнер."""
        cmd = ["docker", "inspect", "-f", "{{.State.Running}}", cls._container_name()]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0 and result.stdout.strip() == "true"

    @classmethod
    def start(cls, repo_path: str) -> str:
        session = cls._session()
        repo_path = os.path.abspath(repo_path)
        session["repo_path"] = repo_path
        session["container"] = cls.CONTAINER_NAME

        if not os.path.isdir(repo_path):
            return err(f": repo not found: {repo_path}")

        if cls._is_container_running():
            return f"Container {cls.CONTAINER_NAME} is already running at {repo_path}."

        subprocess.run(["docker", "rm", "-f", cls.CONTAINER_NAME], capture_output=True)

        git_dir = None
        current_path = repo_path
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
            "-v", f"{repo_path}:/workspace",
            "-w", "/workspace",
        ]

        if git_dir:
            cmd.extend(["-v", f"{git_dir}:/workspace/.git:ro"])

        cmd.extend([cls.IMAGE, "sleep", "infinity"])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return err(f" starting container: {result.stderr}")

        return f"Unified agent started. Workspace: {repo_path}"

    @classmethod
    def _ensure_container(cls):
        """Гарантирует, что контейнер запущен перед выполнением команд."""
        if not cls._is_container_running():
            cls.start(cls._session().get("repo_path") or os.getcwd())

    @classmethod
    def execute_bash(cls, command: str, timeout: int = 60) -> str:
        cls._ensure_container()
        cmd = ["docker", "exec", cls._container_name(), "bash", "-c", command]
        return run_capture(cmd, timeout)

    @classmethod
    def execute_python(cls, code: str, timeout: int = 60) -> str:
        cls._ensure_container()
        cmd = ["docker", "exec", "-i", cls._container_name(), "python3", "-"]
        return run_capture(cmd, timeout, stdin=code)

    @classmethod
    def stop(cls) -> str:
        subprocess.run(["docker", "rm", "-f", cls._container_name()], capture_output=True)
        cls._session()["container"] = None
        return "Unified agent stopped."


@tool(
    description="Start a persistent sandbox container for a repository. Call this to set the working directory.",
    short_description="start sandbox container",
    group="sandbox",
    repo_path=("str", "Absolute or relative path to the git repository on the host")
)
def start_sandbox(repo_path: str):
    return UnifiedDockerAgent.start(repo_path)


@tool(
    description="Execute a bash command. Auto-starts container in current directory if not already running.",
    short_description="run bash command",
    group="sandbox",
    command=("str", "The bash command to run"),
    timeout=("int", "Timeout in seconds (default 60)")
)
def run_bash(command: str, timeout: int = 60):
    return UnifiedDockerAgent.execute_bash(command, timeout)


@tool(
    description="Execute Python code. Auto-starts container in current directory if not already running.",
    short_description="run python code",
    group="sandbox",
    code=("str", "The Python code to execute"),
    timeout=("int", "Timeout in seconds (default 60)")
)
def run_python(code: str, timeout: int = 60):
    return UnifiedDockerAgent.execute_python(code, timeout)


@tool(
    description="Stop and remove the persistent sandbox container.",
    short_description="stop sandbox container",
    group="sandbox"
)
def stop_sandbox():
    return UnifiedDockerAgent.stop()