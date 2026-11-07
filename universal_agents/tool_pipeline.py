from __future__ import annotations

import json
from typing import TYPE_CHECKING, Optional

from universal_agents.config import Config
from universal_agents.constants import (
    ENVIRONMENT_PREFIX,
    ENVIRONMENT_PREFIX_END,
    err,
)
from universal_agents.compressors import auto_compress_tool_result
from universal_agents.models import ToolCall, ToolResult
from universal_agents.task_tracker import DONE_TOOL, validate_task_mark_call
from universal_agents.tool_parsing import parse_tool_args, is_error_content, unwrap_dispatch_call
from universal_agents.tool_manager import DISPATCH_TOOL_NAME
from universal_agents.tool import ToolOutput
from universal_agents.subprocess_utils import set_interrupt_event, clear_interrupt_event
from universal_agents.exceptions import GenerationInterrupted
from universal_agents.tools.fs import read as _read_tool

if TYPE_CHECKING:
    from universal_agents.agent import LLMAgent


def _make_tool_resolve(worker, name: str, args: dict, stop_check=None):
    def resolve(agent, text):
        raw = (text or "").strip().lower()
        first = "".join(c for c in (raw.split(" ", 1)[0] if raw else "") if c.isalnum())
        if first in ("yes", "y", "true", "да"):
            status, val = worker.call(name, args, dry_run=False, stop_check=stop_check)
            if status == "error":
                return f"Error: {val}"
            return val
        return None
    return resolve


class ToolPipeline:
    """Конвейер исполнения инструментов: валидация, безопасность путей, подтверждения, вызов и сжатие."""

    def __init__(self, agent: LLMAgent):
        self.agent = agent

    def execute(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        results: list[ToolResult] = []
        history_before = self.agent.history.get_all()[:-1]
        seen_in_batch: set[str] = set()

        set_interrupt_event(self.agent.stop_event)
        self.agent._in_tool_execution = True
        try:
            for tc in tool_calls:
                res = self._process_tool_call(tc, history_before, seen_in_batch)
                results.append(res)
                try:
                    self.agent.autosave()
                except Exception:
                    pass
        finally:
            clear_interrupt_event()
            self.agent._in_tool_execution = False

        return results

    def _process_tool_call(
        self,
        tc: ToolCall,
        history_before: list,
        seen_in_batch: set[str],
    ) -> ToolResult:
        name = tc.name
        args_str = tc.arguments or "{}"

        # 1. Распаковка виртуального диспатчера (invoke_function / call_tool)
        if name in (DISPATCH_TOOL_NAME, "call_tool"):
            unwrapped_name, nested_args = unwrap_dispatch_call(name, args_str)
            if unwrapped_name in (DISPATCH_TOOL_NAME, "call_tool"):
                return ToolResult.error(
                    tc.id, tc.name, err(": invoke_function requires function_name and an args object.")
                )
            name = unwrapped_name
            args_str = json.dumps(nested_args, ensure_ascii=False)
        result_name = name

        # 2. Проверка ограничений (запреты и отложенная выгрузка)
        if self.agent.tools_manager.is_denied(name):
            err_msg = err(
                f": tool '{name}' is forbidden for this sub-agent. "
                f"Use other tools or answer yourself."
            )
            self.agent.on_system_msg(f"[TOOL DENIED] Model tried to call forbidden tool '{name}'.")
            return ToolResult.error(tc.id, result_name, err_msg)

        if self.agent.tools_manager.is_pending_unload(name):
            err_msg = err(
                f": tool '{name}' was unloaded. "
                f"If you still wanna use it then load_tool('{name}') first."
            )
            self.agent.on_system_msg(f"[TOOL UNLOADED] Model tried to call unloaded tool '{name}'.")
            return ToolResult.error(tc.id, result_name, err_msg)

        # 3. Детекция повторов вызова (§1.9)
        loop_warning = self.agent.recovery_policy.check_tool_loop(name, args_str, seen_in_batch, history_before)
        if loop_warning:
            args_preview = f" with arguments {args_str}" if args_str and args_str != "{}" else ""
            self.agent.on_system_msg(f"[LOOP PREVENTED] Blocked repeated call to '{name}'{args_preview} during execution.")
            return ToolResult.error(tc.id, result_name, loop_warning)

        # 4. Проверка наличия инструмента в схемах
        tool_info = self.agent._all_tools.get(name)
        if not tool_info:
            return ToolResult.error(
                tc.id,
                result_name,
                f"Unknown tool '{name}'. It must be loaded first or probably misspelled.",
            )

        args_dict = parse_tool_args(args_str)

        # 5. Проверка порядка задач для have_done
        if name == DONE_TOOL:
            order_err = validate_task_mark_call(
                history_before, args_dict, self.agent.task_plan_map, self.agent._compacted_task_ids
            )
            if order_err:
                self.agent.on_system_msg(f"[TASK ORDER] Rejected out-of-order have_done call: {order_err}")
                return ToolResult.error(tc.id, result_name, f"{ENVIRONMENT_PREFIX} {order_err} {ENVIRONMENT_PREFIX_END}")

        # 6. Авторизация и безопасность путей (path_safety / confirmation)
        if tool_info.get("requires_confirmation", False) or tool_info.get("path_safety", False):
            skip_confirm = bool(
                tool_info.get("safe_in_trusted")
                and "path" in args_dict
                and self.agent.is_path_trusted(args_dict["path"])
            )
            external = self.agent.tools_manager.check_external_paths(name, args_dict)
            if not skip_confirm and (external or not tool_info.get("path_safety", False)):
                if external:
                    self.agent.on_system_msg(
                        f"🚫 Command references path(s) OUTSIDE project root: {', '.join(external)}"
                    )
                if not self.agent.on_confirm(name, args_dict):
                    return ToolResult.user_denied(tc.id, result_name)

        protected = self.agent.tools_manager.protected_refs(name, args_dict, tool_info)
        if protected and not all(self.agent.is_file_trusted(p) for p in protected):
            self.agent.on_system_msg(f"🔒 Protected path(s) referenced: {', '.join(protected)}")
            if not self.agent.on_confirm(name, args_dict):
                return ToolResult.user_denied(tc.id, result_name)

        # 7. Подтверждение моделью (requires_model_confirmation)
        if tool_info.get("requires_model_confirmation", False):
            return self._handle_model_confirmation(tc.id, result_name, name, args_dict, tool_info)

        # 8. Исполнение инструмента
        return self._execute_and_postprocess(tc.id, result_name, name, args_dict, tool_info)

    def _handle_model_confirmation(
        self,
        tc_id: str,
        result_name: str,
        name: str,
        args_dict: dict,
        tool_info: dict,
    ) -> ToolResult:
        if tool_info.get("is_external") and getattr(self.agent, "tool_worker", None) and self.agent.tool_worker.isolated:
            status, val = self.agent.tool_worker.call(
                name, args_dict, dry_run=True, stop_check=self.agent._stop_check
            )
            if status == "error":
                self.agent.on_system_msg(f"⚠️ [ERROR] Tool '{name}' dry_run failed: {val}")
                return ToolResult.error(tc_id, result_name, val)
            if status == "ok":
                text = val
                if isinstance(text, str) and is_error_content(text):
                    return ToolResult(tc_id, result_name, text, is_error=True)
                return ToolResult.success(tc_id, result_name, str(text))

            preview, ask = val
            self.agent.set_pending_operation({
                "resolve": _make_tool_resolve(
                    self.agent.tool_worker, name, args_dict, self.agent._stop_check
                ),
                "tool_name": name,
            })
            confirm_msg = f"{preview}\n\n{ask}"
            self.agent.on_system_msg(f"[CONFIRM] Tool '{name}' awaits LLM confirmation.")
            return ToolResult(tc_id, result_name, confirm_msg)

        handler = tool_info["handler"]
        dry_run_args = {**args_dict, "dry_run": True}
        try:
            if tool_info.get("has_agent_param") or tool_info.get("is_instance_method"):
                result = handler(self.agent, **dry_run_args)
            else:
                result = handler(**dry_run_args)
        except GenerationInterrupted:
            raise
        except Exception as e:
            self.agent.on_system_msg(f"⚠️ [ERROR] Tool '{name}' dry_run failed: {e}")
            return ToolResult.error(tc_id, result_name, str(e))

        if isinstance(result, str):
            if is_error_content(result):
                return ToolResult(tc_id, result_name, result, is_error=True)
            return ToolResult.success(tc_id, result_name, result)

        if not (isinstance(result, tuple) and len(result) == 3):
            self.agent.on_system_msg(
                f"⚠️ [ERROR] Tool '{name}' with requires_model_confirmation must return (preview, resolve, ask) from dry_run."
            )
            return ToolResult.error(
                tc_id, result_name, "Tool configuration error: dry_run must return (preview, resolve, ask)."
            )

        preview, resolve, ask = result
        self.agent.set_pending_operation({
            "resolve": resolve,
            "tool_name": name,
        })
        confirm_msg = f"{preview}\n\n{ask}"
        self.agent.on_system_msg(f"[CONFIRM] Tool '{name}' awaits LLM confirmation.")
        return ToolResult(tc_id, result_name, confirm_msg)

    def _execute_and_postprocess(
        self,
        tc_id: str,
        result_name: str,
        name: str,
        args_dict: dict,
        tool_info: dict,
    ) -> ToolResult:
        try:
            handler = tool_info["handler"]
            if tool_info.get("has_agent_param") or tool_info.get("is_instance_method"):
                full_result = handler(self.agent, **args_dict)
            elif tool_info.get("is_external") and getattr(self.agent, "tool_worker", None) and self.agent.tool_worker.isolated:
                status, val = self.agent.tool_worker.call(name, args_dict, stop_check=self.agent._stop_check)
                if status == "error":
                    raise RuntimeError(val)
                else:
                    full_result = val
            else:
                full_result = handler(**args_dict)

            result_images: list[str] = []
            if isinstance(full_result, ToolOutput):
                result_images = full_result.images
                full_result = full_result.text

            content = str(full_result) if full_result is not None else "Tool executed successfully"
            if is_error_content(content):
                tr = ToolResult(tc_id, result_name, content, is_error=True)
            else:
                self_truncating = bool(
                    self.agent.tools_manager.tools_map.get(name, {}).get("self_truncating")
                )
                if not self_truncating and len(content) > Config.tools.MAX_READ_CHARS_PER_CALL:
                    content = (
                        content[:Config.tools.MAX_READ_CHARS_PER_CALL]
                        + f"\n{ENVIRONMENT_PREFIX} Output truncated to {Config.tools.MAX_READ_CHARS_PER_CALL} "
                        f"chars per tool call — narrow your request/parameters. {ENVIRONMENT_PREFIX_END}"
                    )
                tr = ToolResult.success(tc_id, result_name, content)
                self.agent._tool_usage[name] = self.agent._tool_usage.get(name, 0) + 1
                if name == _read_tool.__name__:
                    tr.skip_summarize = True

            if result_images:
                tr.images = list(result_images)
                tr.skip_summarize = True

            if not getattr(tr, "skip_summarize", False):
                auto_compress_tool_result(self.agent, tr)

            return tr
        except GenerationInterrupted:
            raise
        except Exception as e:
            return ToolResult.error(tc_id, result_name, str(e))

    def rebuild_tool_usage(self) -> None:
        self.agent._tool_usage.clear()
        for msg in self.agent.history.get_all():
            if isinstance(msg, ToolResult) and not msg.is_error and not msg.is_user_denied:
                self.agent._tool_usage[msg.name] = self.agent._tool_usage.get(msg.name, 0) + 1
