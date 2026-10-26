
from __future__ import annotations

import os

from universal_agents.config import Config
from universal_agents.command_paths import extract_paths
from universal_agents.project_root import find_project_root, external_paths
from universal_agents.tools.fs import is_protected_path


class ToolsMixin:
    """Обвязка поверх ToolManager: load/unload/trust и проверка путей."""

    @property
    def _all_tools(self) -> dict:
        return self.tools_manager.tools_map

    @property
    def tools(self) -> list[dict]:
        return self.tools_manager.schemas

    @property
    def _tools_config(self):
        return self.tools_manager.config

    @property
    def trusted_dirs(self) -> set[str]:
        return self.tools_manager.trusted_dirs

    @property
    def trusted_files(self) -> set[str]:
        return self.tools_manager.trusted_files

    def load_tool(self, name: str) -> str:
        return self.tools_manager.load(name)

    def unload_tool(self, name: str) -> str:
        return self.tools_manager.unload(name)

    def is_tool_denied(self, name: str) -> bool:
        return self.tools_manager.is_denied(name)

    def list_available_tools(self) -> str:
        return self.tools_manager.list_available()

    def trust_dir(self, path: str) -> str:
        return self.tools_manager.trust_dir(path)

    def untrust_dir(self, path: str) -> str:
        return self.tools_manager.untrust_dir(path)

    def is_path_trusted(self, path: str) -> bool:
        return self.tools_manager.is_path_trusted(path)

    def is_file_trusted(self, path: str) -> bool:
        return os.path.abspath(path) in self.tools_manager.trusted_files

    def _protected_refs(self, name: str, args: dict, tool_info: dict) -> list[str]:
        """Защищённые файлы, на которые ссылается вызов: явный path или пути из команды."""
        if not args:
            return []
        candidates: list[str] = []
        if isinstance(args.get("path"), str) and name != "cwd":
            candidates.append(args["path"])
        if tool_info.get("path_safety"):
            for key in ("command", "cmd", "script"):
                value = args.get(key)
                if isinstance(value, str) and value.strip():
                    candidates.extend(extract_paths(value))
        found = [os.path.abspath(p) for p in candidates if os.path.isfile(p) and is_protected_path(p)]
        return list(dict.fromkeys(found))

    def _auto_trust_git_root(self) -> None:
        """При наличии .git доверяем корень проекта: редакторы внутри него не спрашивают подтверждение."""
        if not Config.AUTO_TRUST_GIT_ROOT:
            return
        root = find_project_root()
        if root:
            self.tools_manager.trusted_dirs.add(os.path.abspath(root))

    def _check_external_paths(self, name: str, args: dict) -> list[str]:
        """Для path_safety: извлекает пути вне корня проекта (.git); только существующие."""
        if not args:
            return []
        root = find_project_root()
        if not root:
            return []
        external: list[str] = []
        for key in ("command", "cmd", "script", "path"):
            value = args.get(key)
            if isinstance(value, str) and value.strip():
                paths = extract_paths(value)
                existing = [p for p in paths if os.path.exists(p)]
                external.extend(external_paths(existing, root))
        return list(dict.fromkeys(external))

    def _known_tool_names(self) -> set[str]:
        names = set(self._all_tools.keys())
        names |= self.tools_manager.all_known_names
        return names
