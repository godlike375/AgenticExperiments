"""Управление инструментами агента: регистрация, фильтрация, доверенные папки."""

import os
from typing import Callable, Iterable, Union, Optional

from universal_agents.constants import CORE_TOOLS
from universal_agents.tool_registry import load_external_plugins, build_tool_dict


def _tools_directory() -> str:
    """Путь к директории с инструментами-плагинами."""
    return os.path.join(os.path.dirname(__file__), "tools")


# Инструменты, управляемые системой: модель не может загрузить их сама через
# load_tool — они подключаются автоматически в нужный момент (have_done —
# только после успешного make_plan).
MANAGED_TOOLS = {"have_done"}


class ToolManager:
    """Владеет набором инструментов: схемами, обработчиками и доверенными папками."""

    def __init__(
        self,
        tools_config: Union[list[str], dict, None] = None,
        external_plugins: dict[str, Callable] = None,
    ):
        self._tools_config = tools_config
        self._tools_map: dict[str, dict] = {}
        self._all_known_names: Optional[set[str]] = None
        # Инструменты, по которым вызвали unload_tool, но которые ещё не выгружены
        # физически: они остаются в префиксе (описание в системном промпте), чтобы не
        # ломать KV-кэш, пока история не изменится. Вызов такого инструмента моделью
        # возвращает ошибку «was unloaded». Реальное удаление — flush_pending_unloads().
        self._pending_unload: set[str] = set()
        # Запретительные конфиг (например, для суб-агентов): инструменты, чей вызов
        # запрещён. Как и pending_unload — KV-cache safe: схемы остаются в префиксе,
        # но попытка вызова возвращает ошибку «forbidden» (см. ExecuteMixin).
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
        """Имена всех инструментов, доступных к загрузке (даже ещё не загруженных)."""
        if self._all_known_names is None:
            self._all_known_names = set(load_external_plugins(_tools_directory()).keys())
        return self._all_known_names

    @property
    def schemas(self) -> list[dict]:
        return [v["schema"] for v in self._tools_map.values()]

    @property
    def config(self):
        return self._tools_config

    def _filter(self) -> None:
        all_names = set(self._tools_map.keys())
        if self._tools_config is None:
            active = all_names
        elif isinstance(self._tools_config, list):
            active = set(self._tools_config) & all_names
        elif isinstance(self._tools_config, dict) and "exclude" in self._tools_config:
            active = all_names - set(self._tools_config["exclude"])
        else:
            raise ValueError("Invalid tools_config")
        self._tools_map = {k: v for k, v in self._tools_map.items() if k in active}

    def is_tool_allowed(self, name: str) -> bool:
        """Проверяет, разрешён ли инструмент tools_config."""
        if self._tools_config is None:
            return True
        if isinstance(self._tools_config, list):
            return name in self._tools_config
        if isinstance(self._tools_config, dict) and "exclude" in self._tools_config:
            return name not in self._tools_config["exclude"]
        return True

    def load(self, name: str) -> str:
        """Включает ранее отключённый инструмент по имени."""
        # Если инструмент был помечен на отложенную выгрузку — просто снимаем метку:
        # он всё ещё в _tools_map (префикс не менялся), так что повторно не грузим.
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
            self._tools_map[name] = build_tool_dict(external_tools[name], is_instance_method=False)

            non_core = [n for n in self._tools_map if n not in CORE_TOOLS]
            if len(non_core) >= 1 and "unload_tool" not in self._tools_map and "unload_tool" in external_tools:
                self._tools_map["unload_tool"] = build_tool_dict(external_tools["unload_tool"], is_instance_method=False)

            return f"'{name}' loaded."

        return f"Error '{name}' not found in loadable tools"

    def force_load(self, name: str) -> str:
        """Внутренняя загрузка управляемого инструмента в обход allow-list.

        Для инструментов, которые модель НЕ должна грузить сама через load_tool
        (например, have_done подключается автоматически только после make_plan).
        """
        if name in self._tools_map:
            return f"'{name}' is already loaded."
        external_tools = load_external_plugins(_tools_directory())
        if name in external_tools:
            self._tools_map[name] = build_tool_dict(external_tools[name], is_instance_method=False)
            non_core = [n for n in self._tools_map if n not in CORE_TOOLS]
            if len(non_core) >= 1 and "unload_tool" not in self._tools_map and "unload_tool" in external_tools:
                self._tools_map["unload_tool"] = build_tool_dict(external_tools["unload_tool"], is_instance_method=False)
            return f"'{name}' loaded."
        return f"Error '{name}' not found in loadable tools"

    def unload(self, name: str) -> str:
        """Откладывает отключение инструмента до следующего изменения истории.

        Инструмент остаётся в префиксе (описание в системном промпте/схемах), чтобы не
        ломать KV-кэш. Если модель попытается вызвать его до фактической выгрузки — вернётся
        ошибка «was unloaded». Реальное удаление происходит в flush_pending_unloads().
        """
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
        """True, если по инструменту вызвали unload_tool, но он ещё не выгружен физически."""
        return name in self._pending_unload

    @property
    def pending_unload(self) -> set[str]:
        """Набор инструментов, ожидающих физической выгрузки при изменении истории."""
        return set(self._pending_unload)

    # --------------------------------------------------------
    # Запретительный конфиг (denied tools)
    # --------------------------------------------------------

    def deny(self, names: Union[str, Iterable[str]]) -> None:
        """Запрещает вызов инструментов по имени.

        KV-cache safe: схемы запрещённых инструментов НЕ удаляются из контекста,
        меняется только поведение при вызове — ExecuteMixin вернёт ошибку
        «forbidden». Строка '*' или 'all' запрещает все загруженные инструменты.
        """
        if isinstance(names, str):
            names = list(self._tools_map.keys()) if names in ("*", "all") else [names]
        self._denied.update(names)

    def is_denied(self, name: str) -> bool:
        """True, если вызов инструмента запрещён запретительным конфигом."""
        return name in self._denied

    @property
    def denied(self) -> set[str]:
        """Текущий набор запрещённых инструментов."""
        return set(self._denied)

    def flush_pending_unloads(self) -> list[str]:
        """Физически удаляет инструменты, помеченные unload_tool, и очищает метки.

        Вызывается только при реальном изменении истории (сжатие/удаление/правка), когда
        префикс и так пересобирается и KV-кэш всё равно инвалидируется.
        """
        flushed = [n for n in self._pending_unload if n in self._tools_map]
        for name in flushed:
            del self._tools_map[name]
        self._pending_unload.clear()

        non_core = [n for n in self._tools_map if n not in CORE_TOOLS]
        if len(non_core) == 0 and "unload_tool" in self._tools_map:
            del self._tools_map["unload_tool"]

        return flushed

    def list_available(self) -> str:
        """Список загружаемых (неактивных) инструментов из директории плагинов."""
        external_tools = load_external_plugins(_tools_directory())
        enabled = set(self._tools_map.keys())
        available = set(external_tools.keys()) - enabled - MANAGED_TOOLS
        lines = ["LOADABLE TOOLS:\n"]
        for name in sorted(available):
            if not self.is_tool_allowed(name):
                continue
            func = external_tools[name]
            desc = getattr(func, '_short_description', '')
            lines.append(f'"{name}" ({desc});' if desc else name)
        lines.append(f'\nTo load a concrete tool use "load_tool" + "name" arg')
        return "\n".join(lines)

    # --------------------------------------------------------
    # Доверенные папки
    # --------------------------------------------------------
    def trust_dir(self, path: str) -> str:
        """Добавляет директорию в доверенные (edit_file пропускает подтверждение)."""
        abs_path = os.path.abspath(path)
        if not os.path.isdir(abs_path):
            return f"Error '{path}' is not a directory"
        self.trusted_dirs.add(abs_path)
        return f"Trusted: {abs_path}"

    def untrust_dir(self, path: str) -> str:
        """Убирает директорию из доверенных."""
        abs_path = os.path.abspath(path)
        if abs_path in self.trusted_dirs:
            self.trusted_dirs.discard(abs_path)
            return f"Untrusted: {abs_path}"
        return f"Error '{path}' was not trusted"

    def is_path_trusted(self, path: str) -> bool:
        """Проверяет, находится ли путь внутри доверенной директории."""
        abs_path = os.path.abspath(path)
        for trusted in self.trusted_dirs:
            if abs_path.startswith(trusted + os.sep) or abs_path == trusted:
                return True
        return False
