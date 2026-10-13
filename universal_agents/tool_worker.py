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

__all__ = ["worker_main", "invoke_tool"]


def _apply_config_overrides(overrides: Optional[dict]) -> None:
    """Применяет снапшот Config в дочернем процессе (spawn не наследует память родителя)."""
    for key, value in (overrides or {}).items():
        try:
            setattr(Config, key, value)
        except Exception:
            pass


def _collect_group_tools(group_name: str, tools_dir: str) -> dict:
    """Инструменты, помеченные группой group_name (по метаданным, а не списком хоста)."""
    plugins = load_external_plugins(tools_dir)
    return {
        name: func
        for name, func in plugins.items()
        if getattr(func, "_tool_group", "internal") == group_name
    }


def worker_main(
    group_name: str,
    request_queue,
    response_queue,
    config_overrides: Optional[dict] = None,
    project_root: Optional[str] = None,
    tools_dir: Optional[str] = None,
    initial_cwd: Optional[str] = None,
) -> None:
    """Цикл воркера группы: принять запрос → вызвать инструмент → вернуть ответ.

    Протокол запроса: (cid, name, args: dict, session: dict, dry_run: bool)
    Протокол ответа:  (cid, status, payload, session)
      status "ok"      — payload = результат инструмента;
      status "error"   — payload = текст ошибки;
      status "wrapped" — payload = (preview, ask) при dry_run обёрнутого инструмента.
    """
    _apply_config_overrides(config_overrides)
    if project_root:
        from universal_agents.project_root import set_project_root
        try:
            set_project_root(project_root)
        except Exception:
            pass
    if initial_cwd and os.path.isdir(initial_cwd):
        try:
            os.chdir(initial_cwd)
        except Exception:
            pass

    tools_dir = tools_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools")
    group_tools = _collect_group_tools(group_name, tools_dir)

    # Группа плагинов: подключаем прокси-агент — плагин обращается к хосту через канал.
    if group_name == "plugin":
        from universal_agents import plugin_proxy
        plugin_proxy.set_channel(plugin_proxy._Channel(request_queue, response_queue))

    while True:
        try:
            request = request_queue.get()
        except (EOFError, OSError):
            break
        if request is None:
            break

        cid, name, args, session, dry_run = request
        session = session if isinstance(session, dict) else {}
        token = tool_session.set_session(session)
        try:
            func = group_tools.get(name)
            if func is None:
                response_queue.put((cid, "error", f"tool '{name}' is not available in group '{group_name}'", session))
                continue

            # Применяем рабочую папку из состояния группы (воркер без собственной памяти:
            # сессия — единственный источник истины, она приходит с каждым запросом).
            cwd = session.get("cwd")
            if cwd and os.path.isdir(cwd):
                try:
                    os.chdir(cwd)
                except Exception:
                    pass

            response_queue.put((cid,) + invoke_tool(func, args, dry_run) + (session,))
        except Exception as e:
            response_queue.put((cid, "error", f"{type(e).__name__}: {e}", session))
        finally:
            tool_session.reset_session(token)


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
