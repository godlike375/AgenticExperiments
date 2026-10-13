
"""Универсальный воркер группы: одна точка входа (worker_main) для всех групп.
Воркер ничего не хранит между вызовами; Windows — spawn (нет наследования памяти)."""
from __future__ import annotations

import os
import sys
from typing import Optional

from universal_agents import tool_session
from universal_agents.config import Config
from universal_agents.tool_registry import load_external_plugins

__all__ = ["invoke_tool"]



def invoke_tool(func, args: dict, dry_run: bool) -> tuple:
    if dry_run:
        result = func(**args, dry_run=True)
        if isinstance(result, tuple) and len(result) == 3:
            preview, _resolve, ask = result
            return ("wrapped", (preview, ask))
        return ("ok", result)
    return ("ok", func(**args))


if __name__ == "__main__":  # pragma: no cover - ручной запуск/отладка
    sys.stderr.write("tool_worker is an entry point for GroupManager; do not run directly.\n")
