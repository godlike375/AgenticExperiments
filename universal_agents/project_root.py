"""Поиск корня проекта (ближайшая папка с .git, поднимаясь вверх) и проверка вхождения путей."""

import os

GIT_MARKER = ".git"


def _is_real_git_dir(path: str) -> bool:
    """True, если `.git` — настоящий репозиторий: файл-указатель (gitdir: ...)
    или каталог, содержащий HEAD (пустой каталог .git не считается репозиторием)."""
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
    """Возвращает путь к корню проекта — ближайшей папке с валидным `.git`
    при подъёме вверх от `start`. Пустые/невалидные каталоги `.git` пропускаются.

    Если `start` не задан — используется текущая рабочая директория.
    Если валидный `.git` не найден вплоть до корня ФС — возвращается None.
    """
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
    """True, если `path` находится внутри `root` (или равен ему)."""
    abs_path = os.path.abspath(path)
    abs_root = os.path.abspath(root)
    if abs_path == abs_root:
        return True
    return abs_path.startswith(abs_root + os.sep)


def external_paths(paths: list[str], root: str | None) -> list[str]:
    """Возвращает подмножество `paths`, которые лежат вне `root`."""
    if not root:
        return list(paths)
    return [p for p in paths if not is_within(p, root)]