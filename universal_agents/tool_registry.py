from __future__ import annotations

import os
import sys
import importlib.util
import inspect
from typing import Callable


_PLUGIN_CACHE: dict[str, tuple[float, dict[str, Callable]]] = {}


def _load_plugin_module(file_path: str) -> dict[str, Callable]:
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
            mtime = os.path.getmtime(file_path)
            cached_mtime, cached_tools = _PLUGIN_CACHE.get(file_path, (0.0, {}))
            if file_path not in _PLUGIN_CACHE or cached_mtime != mtime:
                loaded = _load_plugin_module(file_path)
                _PLUGIN_CACHE[file_path] = (mtime, loaded)
            else:
                loaded = cached_tools

            for name, obj in loaded.items():
                if name in external_tools:
                    print(f"Warning: Duplicate tool name '{name}' in {filename}")
                    continue
                external_tools[name] = obj
        except Exception as e:
            print(f"Error loading plugin {filename}: {e}")

    for cached_path in list(_PLUGIN_CACHE):
        if not os.path.exists(cached_path):
            del _PLUGIN_CACHE[cached_path]
    return external_tools


def build_tool_dict(func: Callable, is_instance_method: bool) -> dict:
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
        "is_external": not has_agent and not is_instance_method,
    }
