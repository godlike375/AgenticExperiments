"""Mixin выполнения инструментов LLMAgent: валидация, подтверждение, запуск обработчиков."""

from __future__ import annotations

from universal_agents.constants import ENVIRONMENT_PREFIX, ENVIRONMENT_PREFIX_END
from universal_agents.compressors import auto_compress_tool_result
from universal_agents.models import ToolCall, ToolResult
from universal_agents.task_tracker import DONE_TOOL, validate_task_mark_call
from universal_agents.tool_parsing import parse_tool_args, is_error_content


def _read_result_already_compressed(content: str) -> bool:
    """True, если результат read уже обработан логикой больших файлов
    (интерактивная выемка most_relevant_lines со скелетом либо её фолбэк —
    усечённый префикс) и не должен повторно суммаризироваться
    пер-сообщенческой суммаризацией."""
    return (
        "Most important file lines (for memory saving):" in content
        or "Interactive extraction failed; showing truncated prefix" in content
    )


class ExecuteMixin:
    """Выполняет ToolCall'ы: защита от дубликатов, контроль порядка have_done, запуск."""

    def _execute_tools(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        results = []
        history_before_current_turn = self.history.get_all()[:-1]

        for tc in tool_calls:
            name = tc.name
            args_str = tc.arguments or "{}"

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
                skip_confirm = (name == "edit_file" and "path" in args_dict and self.is_path_trusted(args_dict["path"]))
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
                    tr = ToolResult.success(tc.id, name, content)
                    self._tool_usage[name] = self._tool_usage.get(name, 0) + 1
                    if name == 'read' and self._read_registrations:
                        self.file_states.mark_tool_call(self._read_registrations.pop(0), tr.tool_call_id)
                    # Обработка больших файлов уже вернула сжатый результат
                    # (most_relevant_lines со скелетом либо усечённый префикс при
                    # провале выемки) — не даём пер-сообщенческой суммаризации
                    # сжимать его повторно.
                    if name == 'read' and _read_result_already_compressed(content):
                        tr.skip_summarize = True

                auto_compress_tool_result(self, tr)
                results.append(tr)
            except Exception as e:
                self.on_system_msg(f"⚠️ [ERROR] Tool '{name}' FAILED: {e}")
                results.append(ToolResult.error(tc.id, name, str(e)))

        return results

    def rebuild_tool_usage(self):
        """Пересчитывает статистику использования инструментов по успешным результатам истории."""
        self._tool_usage.clear()
        for msg in self.history.get_all():
            if isinstance(msg, ToolResult) and not msg.is_error and not msg.is_user_denied:
                self._tool_usage[msg.name] = self._tool_usage.get(msg.name, 0) + 1