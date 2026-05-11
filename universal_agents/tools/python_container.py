import os
import subprocess
import tempfile

from universal_agents.tool import tool


class DockerSandbox:
    _ALLOWED_MOUNTS = set()  # Пути на хосте, разрешённые для монтирования

    @classmethod
    def set_allowed_mounts(cls, paths: list[str]):
        cls._ALLOWED_MOUNTS = set(os.path.abspath(p) for p in paths)

    @staticmethod
    def run_python(code: str, timeout: int = 30) -> str:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Записываем код во временную папку
            code_path = os.path.join(tmpdir, "script.py")
            with open(code_path, 'w', encoding='utf-8') as f:
                f.write(code)

            # Формируем docker команду
            cmd = [
                "docker", "run", "--rm",
                "--network", "none",  # без сети
                "--memory", "512m",   # ограничение памяти
                "--cpus", "1",        # ограничение CPU
                "-v", f"{tmpdir}:/code:ro",  # код только для чтения
            ]

            # Монтируем только разрешённые папки
            for mount_path in DockerSandbox._ALLOWED_MOUNTS:
                if os.path.exists(mount_path):
                    cmd.extend(["-v", f"{mount_path}:{mount_path}:ro"])

            cmd.extend([
                "python:3.11-slim",
                "python", "/code/script.py"
            ])

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    timeout=timeout
                )
                output = result.stdout
                if result.stderr:
                    output += f"\n[stderr]:\n{result.stderr}"
                return output
            except subprocess.TimeoutExpired:
                return f"Error: execution timed out after {timeout}s"
            except Exception as e:
                return f"Error running code in container: {e}"


@tool(description="Runs Python code in an isolated container with no host file access",
      code=("str", "Python code to run"))
def run_python(code: str):
    return DockerSandbox.run_python(code)