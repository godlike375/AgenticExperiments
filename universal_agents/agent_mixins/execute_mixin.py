"""Mixin выполнения инструментов LLMAgent: валидация, подтверждение, запуск обработчиков."""

from __future__ import annotations

from universal_agents.config import Config
from universal_agents.constants import (
    ENVIRONMENT_PREFIX,
    ENVIRONMENT_PREFIX_END,
    err,
)
from universal_agents.compressors import auto_compress_tool_result
from universal_agents.models import ToolCall, ToolResult
from universal_agents.task_tracker import DONE_TOOL, validate_task_mark_call
from universal_agents.tool_parsing import parse_tool_args, is_error_content
from universal_agents.subprocess_utils import set_interrupt_event, clear_interrupt_event
from universal_agents.exceptions import GenerationInterrupted


class ExecuteMixin:
    """Выполняет ToolCall'ы: защита от дубликатов, контроль порядка have_done, запуск."""

    def _execute_tools(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        results = []
        history_before_current_turn = self.history.get_all()[:-1]
        seen_in_batch: set[str] = set()

        # На время выполнения инструментов регистрируем флаг прерывания, чтобы
        # долгий subprocess можно было принудительно убить по запросу пользователя.
        set_interrupt_event(self.stop_event)
        try:
            for tc in tool_calls:
                name = tc.name
                args_str = tc.arguments or "{}"

                # Инструмент запрещён для суб-агента: схема в префиксе ради KV-cache, но вызов отклоняется.
                if self.tools_manager.is_denied(name):
                    err_msg = err(
                        f": tool '{name}' is forbidden for this sub-agent. "
                        f"Use other tools or answer yourself."
                    )
                    self.on_system_msg(f"[TOOL DENIED] Model tried to call forbidden tool '{name}'.")
                    results.append(ToolResult.error(tc.id, name, err_msg))
                    continue

                # Инструмент помечен на отложенную выгрузку (unload_tool), но ещё не убран из
                # префикса ради KV-кэша. Модель пытается его вызвать — возвращаем явную ошибку.
                if self.tools_manager.is_pending_unload(name):
                    err_msg = err(
                        f": tool '{name}' was unloaded. "
                        f"If you still wanna use it then load_tool('{name}') first."
                    )
                    self.on_system_msg(f"[TOOL UNLOADED] Model tried to call unloaded tool '{name}'.")
                    results.append(ToolResult.error(tc.id, name, err_msg))
                    continue

                # Дубликат внутри текущего пакета (два одинаковых вызова в одном ответе)
                norm = self.loop_detector.normalize_args(args_str)
                batch_key = f"{name}:{norm}"
                if batch_key in seen_in_batch:
                    warning_msg = (
                        f"{ENVIRONMENT_PREFIX} System rejected duplicate call of tool '{name}'. "
                        f"This tool was just called with the exact same parameters in the previous step. "
                        f"Do NOT call it again in the current moment even if user asked to. Try a different approach, use other parameters, "
                        f"or complete your response with the final answer."
                        f"{ENVIRONMENT_PREFIX_END}"
                    )
                    self.on_system_msg(f"[LOOP PREVENTED] Blocked repeated call to '{name}' during execution.")
                    results.append(ToolResult.error(tc.id, name, warning_msg))
                    continue
                seen_in_batch.add(batch_key)

                if self.loop_detector.check_duplicate_in_turn(name, args_str, history_before_current_turn):
                    warning_msg = (
                        f"{ENVIRONMENT_PREFIX} System rejected duplicate call of tool '{name}'. "
                        f"This tool was just called with the exact same parameters in the previous step. "
                        f"Do NOT call it again in the current moment even if user asked to. Try a different approach, use other parameters, "
                        f"or complete your response with the final answer."
                        f"{ENVIRONMENT_PREFIX_END}"
                    )
                    self.on_system_msg(f"[LOOP PREVENTED] Blocked repeated call to '{name}' during execution.")
                    results.append(ToolResult.error(tc.id, name, warning_msg))
                    continue

                tool_info = self._all_tools.get(name)
                if not tool_info:
                    results.append(ToolResult.error(tc.id, name, f"Unknown tool '{name}'. It must be loaded first or probably misspelled."))
                    continue

                args_dict = parse_tool_args(args_str)

                # Принудительный порядок декомпозиции: неверный вызов have_done
                # → ошибка → существующий механизм перегенерации ответа модели.
                if name == DONE_TOOL:
                    order_err = validate_task_mark_call(
                        history_before_current_turn, args_dict, self.task_plan_map, self._compacted_task_ids
                    )
                    if order_err:
                        self.on_system_msg(f"[TASK ORDER] Rejected out-of-order have_done call: {order_err}")
                        results.append(ToolResult.error(tc.id, name, f"{ENVIRONMENT_PREFIX} {order_err}{ENVIRONMENT_PREFIX_END}"))
                        continue

                if tool_info.get('requires_confirmation', False) or tool_info.get('path_safety', False):
                    skip_confirm = bool(
                        tool_info.get('safe_in_trusted')
                        and "path" in args_dict
                        and self.is_path_trusted(args_dict["path"])
                    )
                    external = self._check_external_paths(name, args_dict)
                    if not skip_confirm and (external or not tool_info.get('path_safety', False)):
                        if external:
                            self.on_system_msg(
                                f"🚫 Command references path(s) OUTSIDE project root: {', '.join(external)}"
                            )
                        if not self.on_confirm(name, args_dict):
                            results.append(ToolResult.user_denied(tc.id, name))
                            continue

                try:
                    handler = tool_info['handler']
                    if tool_info.get('has_agent_param') or tool_info.get('is_instance_method'):
                        full_result = handler(self, **args_dict)
                    else:
                        full_result = handler(**args_dict)
                    content = str(full_result) if full_result is not None else "Tool executed successfully"
                    if is_error_content(content):
                        tr = ToolResult(tc.id, name, content, is_error=True)
                    else:
                        # Жёсткий лимит вывода любого инструмента — держим контекст в рамках.
                        # `read`, `search`, `run_powershell`, `run_bash_host` уже сами режут вывод
                        # (с подсказками по продолжению), поэтому их не трогаем.
                        _self_truncating = ('read', 'search', 'run_powershell', 'run_bash_host')
                        if name not in _self_truncating and len(content) > Config.MAX_READ_CHARS_PER_CALL:
                            content = (
                                content[:Config.MAX_READ_CHARS_PER_CALL]
                                + f"\n{ENVIRONMENT_PREFIX} Output truncated to {Config.MAX_READ_CHARS_PER_CALL} "
                                f"chars per tool call — narrow your request/parameters.{ENVIRONMENT_PREFIX_END}"
                            )
                        tr = ToolResult.success(tc.id, name, content)
                        self._tool_usage[name] = self._tool_usage.get(name, 0) + 1
                        if name == 'read' and self._read_registrations:
                            self.file_states.mark_tool_call(self._read_registrations.pop(0), tr.tool_call_id)
                        # Чтение не пересжимаем в память: скелет уже компактный, диапазон/маленький
                        # файл — сырые строки, которые должны остаться в контексте как есть.
                        if name == 'read':
                            tr.skip_summarize = True

                    if not getattr(tr, 'skip_summarize', False):
                        auto_compress_tool_result(self, tr)
                    results.append(tr)
                except GenerationInterrupted:
                    raise
                except Exception as e:
                    self.on_system_msg(f"⚠️ [ERROR] Tool '{name}' FAILED: {e}")
                    results.append(ToolResult.error(tc.id, name, str(e)))

        finally:
            clear_interrupt_event()

        return results

    def rebuild_tool_usage(self):
        """Пересчитывает статистику использования инструментов по успешным результатам истории."""
        self._tool_usage.clear()
        for msg in self.history.get_all():
            if isinstance(msg, ToolResult) and not msg.is_error and not msg.is_user_denied:
                self._tool_usage[msg.name] = self._tool_usage.get(msg.name, 0) + 1