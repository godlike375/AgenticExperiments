import os
import sys
import importlib.util
import inspect
from typing import Callable


# Кэш уже загруженных плагинов (по абсолютному пути файла). Повторный exec_module
# плодил бы новые объекты функций при каждом load_tool и прогонял module-level код;
# здесь существующие файлы переиспользуются (identity хэндлеров стабильна), НОВЫЕ
# файлы подхватываются автоматически, а удалённые вычищаются из кэша.
_PLUGIN_CACHE: dict[str, dict[str, Callable]] = {}


def _load_plugin_module(file_path: str) -> dict[str, Callable]:
    """Выполняет один файл-плагин и возвращает словарь «имя → @tool-функция»."""
    filename = os.path.basename(file_path)
    module_name = filename[:-3]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {
        name: obj
        for name, obj in inspect.getmembers(module)
        if callable(obj) and hasattr(obj, '_is_tool')
    }


def load_external_plugins(plugins_dir: str = "tools") -> dict[str, Callable]:
    """Загружает все функции с декоратором @tool из .py-файлов указанной директории.

    Загруженные файлы кэшируются (без повторного exec_module); при повторных вызовах
    подхватываются только новые файлы. Внутри одного процесса хэндлеры инструмента
    сохраняют identity — а значит и ссылку на кэшированные заголовки сообщений."""
    external_tools: dict[str, Callable] = {}
    if not os.path.exists(plugins_dir):
        return external_tools

    root_path = os.path.abspath(os.path.join(plugins_dir, ".."))
    if root_path not in sys.path:
        sys.path.insert(0, root_path)

    for filename in os.listdir(plugins_dir):
        if not filename.endswith(".py") or filename.startswith("__"):
            continue
        file_path = os.path.abspath(os.path.join(plugins_dir, filename))
        try:
            if file_path not in _PLUGIN_CACHE:
                _PLUGIN_CACHE[file_path] = _load_plugin_module(file_path)
            for name, obj in _PLUGIN_CACHE[file_path].items():
                if name in external_tools:
                    print(f"Warning: Duplicate tool name '{name}' in {filename}")
                    continue
                external_tools[name] = obj
        except Exception as e:
            print(f"Error loading plugin {filename}: {e}")

    # Вычищаем кэш от файлов, которых больше нет в директории.
    for cached_path in list(_PLUGIN_CACHE):
        if not os.path.exists(cached_path):
            del _PLUGIN_CACHE[cached_path]
    return external_tools


def build_tool_dict(func: Callable, is_instance_method: bool) -> dict:
    """Создаёт словарь-описание инструмента из функции, помеченной @tool."""
    has_agent = getattr(func, '_has_agent_param', False)
    group = getattr(func, '_tool_group', 'internal')
    return {
        "schema": func._tool_schema,
        "handler": func,
        "is_instance_method": is_instance_method,
        "has_agent_param": has_agent,
        "requires_confirmation": getattr(func, '_requires_confirmation', False),
        "requires_model_confirmation": getattr(func, '_requires_model_confirmation', False),
        "path_safety": getattr(func, '_path_safety', False),
        "safe_in_trusted": getattr(func, '_safe_in_trusted', False),
        "self_truncating": getattr(func, '_self_truncating', False),
        "group": group,
        "hidden": getattr(func, '_tool_hidden', False),
        # Внешний инструмент (docs/TOOL_ISOLATION_SPEC.md §4): нет параметра-агента и группа
        # не internal → тело исполняется в воркере группы, а не в процессе хоста.
        "is_external": (not has_agent) and group != "internal",
    }
