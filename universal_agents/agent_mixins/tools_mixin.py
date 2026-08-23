"""Mixin управления инструментами LLMAgent (делегирование в ToolManager + path-safety)."""

from __future__ import annotations

import os

from universal_agents.config import Config
from universal_agents.command_paths import extract_paths
from universal_agents.project_root import find_project_root, external_paths


class ToolsMixin:
    """Публичная обвязка поверх ToolManager: load/unload/trust и проверка путей."""

    @property
    def _all_tools(self) -> dict:
        """Карта активных инструментов (схемы + обработчики)."""
        return self.tools_manager.tools_map

    @property
    def tools(self) -> list[dict]:
        """JSON-схемы активных инструментов для API."""
        return self.tools_manager.schemas

    @property
    def _tools_config(self):
        return self.tools_manager.config

    @property
    def trusted_dirs(self) -> set[str]:
        return self.tools_manager.trusted_dirs

    def load_tool(self, name: str) -> str:
        """Enable a previously disabled tool by name."""
        return self.tools_manager.load(name)

    def unload_tool(self, name: str) -> str:
        """Disable a tool by name, removing it from available tools."""
        return self.tools_manager.unload(name)

    def is_tool_denied(self, name: str) -> bool:
        """True, если вызов инструмента запрещён запретительным конфигом."""
        return self.tools_manager.is_denied(name)

    def list_available_tools(self) -> str:
        """List all available (loadable) tools from plugins directory."""
        return self.tools_manager.list_available()

    def trust_dir(self, path: str) -> str:
        """Add a directory to trusted dirs (edit_file skips confirmation)."""
        return self.tools_manager.trust_dir(path)

    def untrust_dir(self, path: str) -> str:
        """Remove a directory from trusted dirs."""
        return self.tools_manager.untrust_dir(path)

    def is_path_trusted(self, path: str) -> bool:
        """Check if path is inside a trusted directory."""
        return self.tools_manager.is_path_trusted(path)

    def _auto_trust_git_root(self) -> None:
        """Если в текущей папке (или выше) есть валидный `.git` — доверяем корень
        проекта по умолчанию: edit_file внутри него не запрашивает подтверждение
        (git позволяет откатить любые правки)."""
        if not Config.AUTO_TRUST_GIT_ROOT:
            return
        root = find_project_root()
        if root:
            self.tools_manager.trusted_dirs.add(os.path.abspath(root))

    def _check_external_paths(self, name: str, args: dict) -> list[str]:
        """Для инструментов с path_safety извлекает из аргументов команды пути,
        выходящие за пределы корня проекта (.git). Возвращает список внешних путей.

        Триггерятся только СУЩЕСТВУЮЩИЕ пути: несуществующий путь — это скорее
        создание, чем изменение существующих файлов, и потому менее критичен."""
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
        """Имена всех инструментов, о которых модель может знать (загруженных и доступных к загрузке)."""
        names = set(self._all_tools.keys())
        names |= self.tools_manager.all_known_names
        return names
