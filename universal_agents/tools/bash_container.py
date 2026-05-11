import os
import subprocess

from universal_agents.tool import tool


class PersistentDockerAgent:
    CONTAINER_NAME = "llm-agent-box"

    # Очень лёгкий image с полноценным bash
    IMAGE = "bash:5.2"

    @classmethod
    def start(cls, repo_path: str) -> str:

        repo_path = os.path.abspath(repo_path)

        if not os.path.isdir(repo_path):
            return f"Error: repo not found: {repo_path}"

        current_path = os.path.abspath(repo_path)

        while True:
            git_dir = os.path.join(current_path, ".git")
            if os.path.isdir(git_dir):
                break

            parent = os.path.dirname(current_path)
            if parent == current_path:  # корень файловой системы
                return f"Error: not a git repository: {repo_path}"

            current_path = parent

        # Проверяем существует ли контейнер
        check_cmd = [
            "docker",
            "ps",
            "-a",
            "--filter",
            f"name={cls.CONTAINER_NAME}",
            "--format",
            "{{.Names}}"
        ]

        existing = subprocess.run(
            check_cmd,
            capture_output=True,
            text=True
        )

        if cls.CONTAINER_NAME in existing.stdout:
            return f"Container already exists: {cls.CONTAINER_NAME}"

        cmd = [
            "docker", "run",
            "-dit",

            "--name", cls.CONTAINER_NAME,

            # sandbox
            "--network", "none",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",

            # лимиты
            "--memory", "2g",
            "--cpus", "2",
            "--pids-limit", "256",

            # project writable
            "-v", f"{repo_path}:/workspace",

            # .git readonly
            "-v", f"{git_dir}:/workspace/.git:ro",

            "-w", "/workspace",

            cls.IMAGE,

            "sleep", "infinity"
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            return result.stderr

        return f"Started container: {cls.CONTAINER_NAME}"

    @classmethod
    def execute(cls, command: str, timeout: int = 60) -> str:

        cmd = [
            "docker",
            "exec",
            cls.CONTAINER_NAME,
            "bash",
            "-lc",
            command
        ]

        try:

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=timeout
            )

            output = ""

            if result.stdout:
                output += result.stdout

            if result.stderr:
                output += "\n[stderr]\n"
                output += result.stderr

            output += f"\n[exit_code]: {result.returncode}"

            return output

        except subprocess.TimeoutExpired:
            return f"Error: timeout after {timeout}s"

        except Exception as e:
            return f"Error: {e}"

    @classmethod
    def stop(cls) -> str:

        cmd = [
            "docker",
            "rm",
            "-f",
            cls.CONTAINER_NAME
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            return result.stderr

        return f"Stopped container: {cls.CONTAINER_NAME}"


@tool(
    description="Start persistent docker sandbox for git repository",
    repo_path=("string", "Absolute or relative path to git repository")
)
def start_agent_container(repo_path: str):
    return PersistentDockerAgent.start(repo_path)


@tool(
    description="Execute bash command inside persistent agent container",
    command=("string", "Bash command to execute"),
    timeout=("integer", "Optional timeout in seconds")
)
def agent_exec(command: str, timeout: int = 60):
    return PersistentDockerAgent.execute(
        command=command,
        timeout=timeout
    )


@tool(
    description="Stop and remove persistent agent container"
)
def stop_agent_container():
    return PersistentDockerAgent.stop()