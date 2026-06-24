import os
import sys
import importlib.util
import inspect
from typing import Callable


def load_external_plugins(plugins_dir: str = "tools") -> dict[str, Callable]:
    """
    Загружает все функции, помеченные декоратором @tool, из .py-файлов
    в указанной директории.
    """
    external_tools: dict[str, Callable] = {}
    if not os.path.exists(plugins_dir):
        return external_tools

    root_path = os.path.abspath(os.path.join(plugins_dir, ".."))
    if root_path not in sys.path:
        sys.path.insert(0, root_path)

    for filename in os.listdir(plugins_dir):
        if not filename.endswith(".py") or filename.startswith("__"):
            continue
        module_name = filename[:-3]
        file_path = os.path.join(plugins_dir, filename)
        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for name, obj in inspect.getmembers(module):
                if callable(obj) and hasattr(obj, '_is_tool'):
                    if name in external_tools:
                        print(f"Warning: Duplicate tool name '{name}' in {filename}")
                        continue
                    external_tools[name] = obj
        except Exception as e:
            print(f"Error loading plugin {filename}: {e}")
    return external_tools


def build_tool_dict(func: Callable, is_instance_method: bool) -> dict:
    """Создаёт словарь-описание инструмента из функции, помеченной @tool."""
    return {
        "schema": func._tool_schema,
        "handler": func,
        "is_instance_method": is_instance_method,
        "requires_confirmation": getattr(func, '_requires_confirmation', False),
    }
