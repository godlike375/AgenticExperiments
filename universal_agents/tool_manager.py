
import os
from typing import Callable, Iterable, Union, Optional

from universal_agents.tool_registry import load_external_plugins, build_tool_dict
from universal_agents.tools.builtin import (
    have_done as _have_done_tool,
    load_tool as _load_tool_tool,
    unload_tool as _unload_tool_tool,
)

CORE_TOOLS = (_load_tool_tool.__name__, _unload_tool_tool.__name__)


def _tools_directory() -> str:
    return os.path.join(os.path.dirname(__file__), "tools")


MANAGED_TOOLS = {_have_done_tool.__name__}


class ToolManager:

    def __init__(
        self,
        tools_config: Union[list[str], dict, None] = None,
        external_plugins: dict[str, Callable] = None,
    ):
        self._tools_config = tools_config
        self._tools_map: dict[str, dict] = {}
        self._all_known_names: Optional[set[str]] = None
        self._pending_unload: set[str] = set()
        self._denied: set[str] = set()
        if external_plugins:
            for name, func in external_plugins.items():
                self._tools_map[name] = build_tool_dict(func, is_instance_method=False)
        self._filter()
        self.trusted_dirs: set[str] = set()

    @property
    def tools_map(self) -> dict[str, dict]:
        return self._tools_map

    @property
    def all_known_names(self) -> set[str]:
        if self._all_known_names is None:
            self._all_known_names = {
                name for name, func in load_external_plugins(_tools_directory()).items()
                if not getattr(func, '_tool_hidden', False)
            }
        return self._all_known_names

    @property
    def schemas(self) -> list[dict]:
        return [self._tools_map[n]["schema"] for n in sorted(self._tools_map)]

    @property
    def config(self):
        return self._tools_config

    def _filter(self) -> None:
        all_names = {k for k, v in self._tools_map.items() if not v.get("hidden")}
        if self._tools_config is None:
            active = all_names
        elif isinstance(self._tools_config, list):
            active = set(self._tools_config) & all_names
        elif isinstance(self._tools_config, dict) and "exclude" in self._tools_config:
            active = all_names - set(self._tools_config["exclude"])
        else:
            raise ValueError("Invalid tools_config")
        self._tools_map = {k: v for k, v in self._tools_map.items() if k in active}
        self._warn_mislabeled_groups()

    def _warn_mislabeled_groups(self) -> None:
        """Мягкий контроль разметки: инструмент без agent в internal группе — вероятная ошибка конфигурации."""
        for name, info in self._tools_map.items():
            if not info.get("has_agent_param") and info.get("group", "internal") == "internal":
                print(
                    f"Warning: tool '{name}' has no 'agent' param but is labeled group='internal'. "
                    f"It will run in the host process; give it an external group to isolate it."
                )

    def is_tool_allowed(self, name: str) -> bool:
        if self._tools_config is None:
            return True
        if isinstance(self._tools_config, list):
            return name in self._tools_config
        if isinstance(self._tools_config, dict) and "exclude" in self._tools_config:
            return name not in self._tools_config["exclude"]
        return True

    def load(self, name: str) -> str:
        if name in self._pending_unload:
            self._pending_unload.discard(name)
            return f"'{name}' re-enabled (unload cancelled)."

        if name in self._tools_map:
            return f"Error '{name}' is already loaded."

        if name in MANAGED_TOOLS:
            return (f"Error '{name}' is managed by the system and cannot be loaded manually. "
                    f"It is attached automatically when needed.")

        if not self.is_tool_allowed(name):
            return f"Error '{name}' is not allowed by tools_config."

        external_tools = load_external_plugins(_tools_directory())
        if name in external_tools:
            if getattr(external_tools[name], '_tool_hidden', False):
                return f"Error '{name}' is an internal low-level tool and cannot be loaded."
            self._tools_map[name] = build_tool_dict(external_tools[name], is_instance_method=False)

            non_core = [n for n in self._tools_map if n not in CORE_TOOLS]
            unload_name = _unload_tool_tool.__name__
            if len(non_core) >= 1 and unload_name not in self._tools_map and unload_name in external_tools:
                self._tools_map[unload_name] = build_tool_dict(external_tools[unload_name], is_instance_method=False)

            return f"'{name}' loaded."

        return f"Error '{name}' not found in loadable tools"

    def force_load(self, name: str) -> str:
        if name in self._tools_map:
            return f"'{name}' is already loaded."
        external_tools = load_external_plugins(_tools_directory())
        if name in external_tools:
            if getattr(external_tools[name], '_tool_hidden', False):
                return f"Error '{name}' is an internal low-level tool and cannot be loaded."
            self._tools_map[name] = build_tool_dict(external_tools[name], is_instance_method=False)
            non_core = [n for n in self._tools_map if n not in CORE_TOOLS]
            unload_name = _unload_tool_tool.__name__
            if len(non_core) >= 1 and unload_name not in self._tools_map and unload_name in external_tools:
                self._tools_map[unload_name] = build_tool_dict(external_tools[unload_name], is_instance_method=False)
            return f"'{name}' loaded."
        return f"Error '{name}' not found in loadable tools"

    def unload(self, name: str) -> str:
        """Откладывает отключение до следующего изменения истории (KV-cache safe); реальное удаление — flush_pending_unloads()."""
        if name in CORE_TOOLS:
            return f"Error Cannot disable built-in tool '{name}'."

        if name not in self._tools_map:
            return f"Error Tool '{name}' is not loaded yet."

        if name in self._pending_unload:
            return f"'{name}' is already marked for unload."

        self._pending_unload.add(name)
        return (f"'{name}' will be unloaded on the next history change (KV-cache safe). "
                f"Until then it stays available but calling it returns an error; "
                f"if you still need it, call load_tool('{name}') first.")

    def is_pending_unload(self, name: str) -> bool:
        return name in self._pending_unload

    @property
    def pending_unload(self) -> set[str]:
        return set(self._pending_unload)


    def deny(self, names: Union[str, Iterable[str]]) -> None:
        if isinstance(names, str):
            names = list(self._tools_map.keys()) if names in ("*", "all") else [names]
        self._denied.update(names)

    def is_denied(self, name: str) -> bool:
        return name in self._denied

    @property
    def denied(self) -> set[str]:
        return set(self._denied)

    def flush_pending_unloads(self) -> list[str]:
        """Физически удаляет помеченные unload инструменты (только при изменении истории)."""
        flushed = [n for n in self._pending_unload if n in self._tools_map]
        for name in flushed:
            del self._tools_map[name]
        self._pending_unload.clear()

        non_core = [n for n in self._tools_map if n not in CORE_TOOLS]
        unload_name = _unload_tool_tool.__name__
        if len(non_core) == 0 and unload_name in self._tools_map:
            del self._tools_map[unload_name]

        return flushed

    def list_loaded(self) -> str:
        lines = []
        for name in sorted(self._tools_map):
            func = self._tools_map[name].get("handler")
            desc = getattr(func, '_short_description', '') if func else ''
            lines.append(f'"{name}" ({desc});' if desc else name)
        return "LOADED TOOLS:\n" + "\n".join(lines)

    def list_available(self) -> str:
        external_tools = load_external_plugins(_tools_directory())
        enabled = set(self._tools_map.keys())
        available = set(external_tools.keys()) - enabled - MANAGED_TOOLS
        lines = []
        for name in sorted(available):
            if getattr(external_tools[name], '_tool_hidden', False):
                continue
            if not self.is_tool_allowed(name):
                continue
            func = external_tools[name]
            desc = getattr(func, '_short_description', '')
            lines.append(f'"{name}" ({desc});' if desc else name)
        return "\n".join(lines)

    def trust_dir(self, path: str) -> str:
        abs_path = os.path.abspath(path)
        if not os.path.isdir(abs_path):
            return f"Error '{path}' is not a directory"
        self.trusted_dirs.add(abs_path)
        return f"Trusted: {abs_path}"

    def untrust_dir(self, path: str) -> str:
        abs_path = os.path.abspath(path)
        if abs_path in self.trusted_dirs:
            self.trusted_dirs.discard(abs_path)
            return f"Untrusted: {abs_path}"
        return f"Error '{path}' was not trusted"

    def is_path_trusted(self, path: str) -> bool:
        abs_path = os.path.abspath(path)
        for trusted in self.trusted_dirs:
            if abs_path.startswith(trusted + os.sep) or abs_path == trusted:
                return True
        return False
