from __future__ import annotations

import json
import os
import shutil
import tempfile
from unittest import mock
import pytest

from universal_agents.config import Config
from universal_agents.models import AssistantMessage, ToolCall
from universal_agents.tools.fs import (
    read as read_tool,
    line_range_edit,
    match_replace_edit,
    is_protected_path,
)
from universal_agents.tools.host_shell import run_powershell
from universal_agents.tools.builtin import apply_edit
from universal_agents.tool_worker import ToolWorker
from tests.conftest import make_agent, double_me


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_fs_read_lifecycle(temp_dir):
    agent = make_agent()
    test_file = os.path.join(temp_dir, "sample.txt")
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("line 1\nline 2\nline 3\n")

    # 1. Чтение файла целиком
    read_res = read_tool(agent, path=test_file)
    assert "1 line 1\n2 line 2\n3 line 3" in read_res

    # 2. Чтение диапазона строк
    partial = read_tool(agent, path=test_file, start_line=2, end_line=2)
    assert "2 line 2" in partial
    assert "1 line 1" not in partial


def test_fs_edit_confirmation_flow(temp_dir):
    agent = make_agent()
    test_file = os.path.join(temp_dir, "code.py")
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("def foo():\n    return 1\n")

    # Вызов match_replace_edit с dry_run=True возвращает (preview, resolve, ask)
    preview, resolve, ask = match_replace_edit(
        path=test_file,
        old="return 1",
        new_text="return 42",
        dry_run=True,
    )
    assert "L" in preview  # Проверяем номера строк в diff: L   2: +
    assert "-    return 1" in preview
    assert "+    return 42" in preview

    # Модель подтверждает правку через apply_edit
    agent.set_pending_operation({
        "resolve": resolve,
        "ask": ask,
        "summary": "Edit file",
        "operation": "match_replace_edit",
    })
    apply_res = apply_edit(agent, confirm=True)
    assert "Replaced" in apply_res
    assert agent._pending_operation is None

    # Проверяем, что файл действительно изменился на диске
    with open(test_file, "r", encoding="utf-8") as f:
        content = f.read()
    assert "return 42" in content


def test_path_protection_and_trust(temp_dir, monkeypatch):
    monkeypatch.setattr(Config, "API_URL", "https://api.example.com/v1")
    # .env и config файлы защищены от записи при удаленном API_URL
    env_file = os.path.join(temp_dir, ".env")
    with open(env_file, "w", encoding="utf-8") as f:
        f.write("SECRET=1")
    cfg_file = os.path.join(temp_dir, "config.ini")
    with open(cfg_file, "w", encoding="utf-8") as f:
        f.write("[test]")

    assert is_protected_path(env_file) is True
    assert is_protected_path(cfg_file) is True

    # Обычный файл в проекте не защищен
    normal_file = os.path.join(temp_dir, "normal.txt")
    with open(normal_file, "w", encoding="utf-8") as f:
        f.write("plain")
    assert is_protected_path(normal_file) is False


def test_tool_worker_subprocess_isolation_and_kill():
    # Тестируем запуск изолированного процесса воркера и завершение через kill()
    worker = ToolWorker(isolated=True)
    try:
        status, value = worker.call("read", {"path": "nonexistent_file_for_test.txt"})
        assert status in ("ok", "error")

        # Завершение воркера (kill)
        worker.kill()
        assert worker._proc is None
    finally:
        worker.kill()


def test_powershell_execution_and_shell_fallback():
    # Тестируем выбор powershell / pwsh и fallback на OEM для Windows
    captured = {}

    def fake_which(name):
        # Имитируем отсутствие pwsh и наличие legacy powershell
        if name == "powershell":
            return r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
        return None

    def fake_run_capture(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "Тестовый вывод"

    with mock.patch("universal_agents.tools.host_shell.shutil.which", side_effect=fake_which), \
         mock.patch("universal_agents.tools.host_shell.run_capture", side_effect=fake_run_capture):
        res = run_powershell("Write-Output 'Привет'")

    assert res == "Тестовый вывод"
    assert captured["kwargs"].get("fallback_encoding") == "oem"

    # Проверяем корректную ошибку, если шелл не найден
    with mock.patch("universal_agents.tools.host_shell.shutil.which", return_value=None):
        err_res = run_powershell("Get-Process")
        assert "not found" in err_res.lower()


def test_tool_deferred_unload_preserves_schema_order():
    # Инвариант §1.2: выгрузка инструмента не должна мгновенно удалять инструмент из памяти.
    # Инструмент помечается как pending_unload.
    agent = make_agent(tools_config=["double_me"], external_plugins={"double_me": double_me})

    assert "double_me" in agent.tools_manager.tools_map
    assert not agent.tools_manager.is_pending_unload("double_me")

    # Выгружаем double_me
    msg = agent.tools_manager.unload("double_me")
    assert "will be unloaded on the next history change" in msg
    assert agent.tools_manager.is_pending_unload("double_me")

    # Инструмент все еще в tools_map (KV-cache safe)
    assert "double_me" in agent.tools_manager.tools_map

    # Только после явного сброса (flush_pending_unloads) инструмент физически удаляется
    flushed = agent.tools_manager.flush_pending_unloads()
    assert "double_me" in flushed
    assert "double_me" not in agent.tools_manager.tools_map
