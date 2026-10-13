
import os

GIT_MARKER = ".git"

_OVERRIDE_ROOT: str | None = None


def set_project_root(path: str | None) -> None:
    global _OVERRIDE_ROOT
    _OVERRIDE_ROOT = os.path.abspath(path) if path else None


def get_project_root_override() -> str | None:
    return _OVERRIDE_ROOT


def _is_real_git_dir(path: str) -> bool:
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read(64).lstrip().startswith("gitdir:")
        except OSError:
            return False
    if os.path.isdir(path):
        return os.path.isfile(os.path.join(path, "HEAD"))
    return False


def find_project_root(start: str = None) -> str | None:
    if _OVERRIDE_ROOT is not None:
        return _OVERRIDE_ROOT
    current = os.path.abspath(start) if start else os.path.abspath(os.getcwd())
    while True:
        git_marker = os.path.join(current, GIT_MARKER)
        if _is_real_git_dir(git_marker):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent


def is_within(path: str, root: str) -> bool:
    abs_path = os.path.abspath(path)
    abs_root = os.path.abspath(root)
    if abs_path == abs_root:
        return True
    return abs_path.startswith(abs_root + os.sep)


def external_paths(paths: list[str], root: str | None) -> list[str]:
    if not root:
        return list(paths)
    return [p for p in paths if not is_within(p, root)]