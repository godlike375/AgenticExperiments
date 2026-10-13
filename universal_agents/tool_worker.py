"""Универсальный воркер группы инструментов.

Точка входа ОДНА для всех групп (`worker_main`): воркер импортирует инструменты,
отбирает свои по метке группы из метаданных и вызывает нужный по имени. Отдельных
классов под группы нет — хост поднимает один и тот же воркер на каждую группу.

Воркер ничего не хранит между вызовами: состояние группы (session) приходит с каждым
запросом и уходит обратно в ответе. На Windows процесс запускается через spawn и не
наследует память родителя, поэтому: (а) точка входа — обычная функция модуля,
(б) значения Config и корень проекта передаются снапшотом и применяются поверх
свежего импорта, (в) инструменты импортируются заново в дочернем процессе.

Остановка/таймаут — принудительное завершение процесса со стороны хоста
(GroupManager.ProcessGroupWorker). Отдельного сигнала остановки внутрь воркера нет:
убьём процесс — умрёт и всё дерево запущенных им процессов оболочки (Job Object).
"""

from __future__ import annotations

import os
import sys
from typing import Optional

from universal_agents import tool_session
from universal_agents.config import Config
from universal_agents.tool_registry import load_external_plugins

__all__ = ["invoke_tool"]



def invoke_tool(func, args: dict, dry_run: bool) -> tuple:
    """Вызывает инструмент; возвращает (status, payload).

    Общий хелпер для воркера группы (процесса) и локального воркера хоста: для dry_run
    обёрнутого инструмента (requires_model_confirmation) даёт ('wrapped', (preview, ask)),
    иначе ('ok', результат). Инструмент обязан вернуть (preview, resolve, ask) в dry_run."""
    if dry_run:
        result = func(**args, dry_run=True)
        if isinstance(result, tuple) and len(result) == 3:
            preview, _resolve, ask = result
            return ("wrapped", (preview, ask))
        return ("ok", result)
    return ("ok", func(**args))


if __name__ == "__main__":  # pragma: no cover - ручной запуск/отладка
    sys.stderr.write("tool_worker is an entry point for GroupManager; do not run directly.\n")
