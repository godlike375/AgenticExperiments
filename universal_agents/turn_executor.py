from __future__ import annotations

import re
from dataclasses import dataclass, field, replace as dc_replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Optional, Callable

from universal_agents.config import Config
from universal_agents.constants import (
    INTERRUPT_HEADER,
    ENVIRONMENT_PREFIX,
    ENVIRONMENT_PREFIX_END,
)

MAX_CONSECUTIVE_ERRORS = 10
from universal_agents.models import UserMessage, AssistantMessage, ToolResult
from universal_agents.generation import GenerationParams
from universal_agents.llm_client import LLMClient
from universal_agents.generation import apply_stop_markers
from universal_agents.exceptions import GenerationInterrupted
from universal_agents.agent_mixins.response_mixin import SimReasoningGroup
from universal_agents.context_builder import prepare_messages_for_api, get_effective_prefill
from universal_agents.generation import StructuredOutputConfig
from universal_agents.tools.builtin import respond_to_system, apply_edit

if TYPE_CHECKING:
    from universal_agents.agent import LLMAgent

_NO_TOOL_ERROR_SIGNATURE = "<no_tool_call>"
_CONFIRMATION_GUARD_VERDICT = "confirmation_guard"
_SYSTEM_HEADER_RE = re.compile(r"^\{<SYSTEM>:.*?\}\n\n", re.DOTALL)
_REPEAT_ANSWER_NAG = (
    f"\n\n{ENVIRONMENT_PREFIX} Your previous answer was the same to the latest one. "
    f"Please do NOT repeat it again and answer differently."
    f"{ENVIRONMENT_PREFIX_END}"
)


def _strip_system_header(content: Optional[str]) -> Optional[str]:
    if content is None:
        return None
    return _SYSTEM_HEADER_RE.sub("", content, count=1)


@dataclass
class TurnState:
    prefill: Optional[str] = None
    error_counts: dict[str, int] = field(default_factory=dict)
    tool_error_retries_left: int = field(default_factory=lambda: Config.ERROR_RECOVERY_RETRIES)
    broken_regen_left: int = field(default_factory=lambda: Config.BROKEN_CALL_REGEN_RETRIES)
    broken_fix_left: int = field(default_factory=lambda: Config.BROKEN_CALL_FIX_RETRIES)
    no_comment_retries_left: int = field(default_factory=lambda: Config.NO_COMMENT_RETRIES)
    guard_retries_left: int = field(default_factory=lambda: Config.ANSWER_GUARD_MAX_RETRIES)
    structured_output: Optional[StructuredOutputConfig] = None
    structured_output_phase: int = 0

    def step_prefill(self) -> Optional[str]:
        return self.prefill

    def set_prefill(self, value: Optional[str]) -> None:
        self.prefill = value

    def clear_prefill(self) -> None:
        self.prefill = None

    def can_retry_broken_regen(self) -> bool:
        if self.broken_regen_left > 0:
            self.broken_regen_left -= 1
            return True
        return False

    def can_retry_broken_fix(self) -> bool:
        if self.broken_fix_left > 0:
            self.broken_fix_left -= 1
            return True
        return False

    def can_retry_tool_error(self) -> bool:
        if self.tool_error_retries_left > 0:
            self.tool_error_retries_left -= 1
            return True
        return False

    def record_tool_error(self, signature: Optional[str] = None) -> None:
        sig = signature if signature is not None else _NO_TOOL_ERROR_SIGNATURE
        self.error_counts[sig] = self.error_counts.get(sig, 0) + 1

    def record_tool_success(self) -> None:
        self.error_counts.clear()
        self.tool_error_retries_left = Config.ERROR_RECOVERY_RETRIES

    def reset_error_counts(self) -> None:
        self.error_counts.clear()

    def can_retry_answer_guard(self) -> bool:
        if self.guard_retries_left > 0:
            self.guard_retries_left -= 1
            return True
        return False

    @property
    def guard_retries_spent(self) -> int:
        return Config.ANSWER_GUARD_MAX_RETRIES - self.guard_retries_left

    @property
    def max_errors_reached(self) -> bool:
        return any(count >= MAX_CONSECUTIVE_ERRORS for count in self.error_counts.values())


class TurnExecutor:
    """Исполнитель ходов диалога и служебных вызовов к LLM (§1.3, §1.7, §2.4)."""

    def __init__(self, agent: LLMAgent):
        self.agent = agent

    def prepare_turn(self, message: str, autosave: bool = True) -> None:
        user_msg = UserMessage(content=message)
        last = self.agent.history.get_last_message()
        if isinstance(last, AssistantMessage) and last.has_tool_calls():
            self.agent.history.pop_pending_tool_calls()
            last = self.agent.history.get_last_message()
        if isinstance(last, UserMessage):
            self.agent.history.add(AssistantMessage(content=""))
        self.agent.history.add(user_msg)
        if self.agent._per_msg_enabled:
            self.agent._maybe_summarize_user_message(user_msg)
        if autosave:
            self.agent.autosave()
        self.agent.clear_stop()

    def call_with_retries(
        self,
        messages_to_send: list[dict],
        step_prefill: Optional[str] = None,
        is_duplicate_fn: Callable = None,
        boost: bool = True,
        reasoning_effort: str = None,
        stop_markers: tuple = (),
        prefill_shown: int = 0,
        defer_stream_end: bool = False,
        gen_params: GenerationParams = None,
    ) -> tuple:
        if is_duplicate_fn is None:
            is_duplicate_fn = self.agent.recovery_policy.detect_duplicate
        max_generation_attempts = (
            self.agent._max_generation_attempts
            if self.agent._max_generation_attempts is not None
            else max(
                Config.MAX_LOOP_RETRIES,
                Config.DUPLICATE_REGENERATIONS_BEFORE_SYSTEM + Config.DUPLICATE_SYSTEM_WARNINGS + 1,
            )
        )
        message_obj = None
        duplicate_system_warning: Optional[str] = None
        dup_watch_target: Optional[str] = None
        dup_count = 0
        api_error_occurred = False
        stopped_at_marker = False
        effective_reasoning_effort = (
            reasoning_effort if reasoning_effort is not None else self.agent._reasoning_effort
        )

        for attempt in range(max_generation_attempts):
            attempt_params = gen_params if gen_params is not None else self.agent._gen_params
            if self.agent._temp_override is not None:
                attempt_params = attempt_params.with_temp(self.agent._temp_override)
                self.agent._temp_override = None

            active_messages = [dict(msg) for msg in messages_to_send]
            if duplicate_system_warning:
                active_messages.append({"role": "system", "content": duplicate_system_warning})

            message_obj, err_val, usage, stopped_at_marker = self.agent._call_llm(
                active_messages,
                prefill=step_prefill,
                params=attempt_params,
                watch_prefix=dup_watch_target,
                watch_continue_temp=Config.DUPLICATE_CONTINUATION_TEMP if dup_watch_target is not None else None,
                stop_check=self.agent._stop_check,
                reasoning_effort=effective_reasoning_effort,
                stop_markers=stop_markers,
                prefill_shown=prefill_shown,
                defer_stream_end=defer_stream_end,
            )
            dup_watch_target = None

            if usage:
                self.agent.record_usage(usage)
            if err_val:
                self.agent.on_system_msg(f"[API Error] {err_val}")
                api_error_occurred = True
                break

            if not message_obj:
                api_error_occurred = True
                break

            duplicate = is_duplicate_fn(message_obj)
            if duplicate is not None and duplicate[0] == _CONFIRMATION_GUARD_VERDICT:
                duplicate = None
            if duplicate is not None:
                dup_count += 1
                kind, payload = duplicate
                if kind == 'tool_call':
                    dup_name, dup_args = payload
                    tag, reason = 'PROACTIVE LOOP DETECTED', f"Intercepted duplicate call to '{dup_name}'"
                elif kind == 'reasoning':
                    dup_watch_target = None
                    tag, reason = 'REPEATED REASONING DETECTED', 'Model repeated a reasoning block from history'
                else:
                    dup_watch_target = payload
                    tag, reason = 'DUPLICATE ANSWER DETECTED', 'Model repeated the previous answer verbatim'
                dup_budget = (
                    Config.DUPLICATE_REGENERATIONS_BEFORE_SYSTEM + Config.DUPLICATE_SYSTEM_WARNINGS
                )
                if dup_count > dup_budget:
                    message_obj = None
                    self.agent.on_system_msg(
                        f"[{tag}] {reason}. Duplicate still present after "
                        f"{Config.DUPLICATE_REGENERATIONS_BEFORE_SYSTEM} regenerations and "
                        f"{Config.DUPLICATE_SYSTEM_WARNINGS} attempts with a system warning. "
                        "Handing control to user."
                    )
                    return message_obj, api_error_occurred, stopped_at_marker
                if dup_count > Config.DUPLICATE_REGENERATIONS_BEFORE_SYSTEM:
                    duplicate_system_warning = (
                        self.agent.recovery_policy.build_duplicate_warning(dup_name, dup_args)
                        if kind == 'tool_call' else _REPEAT_ANSWER_NAG
                    )
                if boost:
                    self.agent._temp_override = Config.BOOST_TEMP
                self.agent.on_system_msg(
                    f"[{tag}] {reason}. "
                    f"Discarding response. Activating temperature boost ({Config.BOOST_TEMP})"
                    + (" and injecting system warning." if duplicate_system_warning else ".")
                    + f" Attempt {attempt + 1}/{max_generation_attempts}."
                )
                continue

            break
        else:
            message_obj = None
            self.agent.on_system_msg(
                "[PROACTIVE LOOP DETECTOR] Max re-generation attempts reached. Handing control to user."
            )

        return message_obj, api_error_occurred, stopped_at_marker

    def run_turn_loop(
        self,
        max_iter: int,
        prefill: str = None,
        structured_output: Optional[StructuredOutputConfig] = None,
        service_mode: bool = False,
    ) -> str:
        current_prefill = get_effective_prefill(prefill)
        if structured_output is not None and structured_output.controller is not None:
            controller_prefill = structured_output.controller.initial_prefill()
            if controller_prefill is not None:
                current_prefill = controller_prefill
        turn_reasoning = self.agent._reasoning_effort
        self.agent._thinking_once = False
        state = TurnState(prefill=current_prefill, structured_output=structured_output)

        sim_group = SimReasoningGroup(Config.SIMULATED_REASONING_REPAIRS) if (
            Config.SIMULATED_REASONING_ENABLED
            and turn_reasoning == "none"
            and not service_mode
            and structured_output is None
            and not current_prefill
            and not getattr(self.agent, "_is_subagent", False)
        ) else None
        sim_open = False
        sim_stream_started = False

        def _close_sim_stream():
            nonlocal sim_stream_started
            if sim_stream_started and self.agent.on_stream_end:
                self.agent.on_stream_end()
            sim_stream_started = False

        def _finish_turn(result: str) -> str:
            _close_sim_stream()
            if not service_mode:
                self.agent.autosave()
            return result

        try:
            for _ in range(max_iter):
                if self.agent.pending_interrupt is not None:
                    return _finish_turn("")

                if self.agent.stop_event.is_set():
                    self.agent.on_system_msg("⏹ Generation stopped by user — control returned to you.")
                    self.agent.clear_stop()
                    return _finish_turn("")

                step_prefill = state.step_prefill()
                think_params = None
                if sim_group is not None:
                    if not sim_open:
                        sim_group.reset(Config.SIMULATED_REASONING_REPAIRS)
                        sim_open = True
                        sim_stream_started = False
                    step_prefill = sim_group.prefill()
                    if sim_group.is_think_phase():
                        think_params = sim_group.think_params(self.agent._gen_params)

                if structured_output is not None:
                    effective_markers = structured_output.effective_markers(step_prefill)
                elif sim_group is not None and sim_group.is_think_phase():
                    effective_markers = (f"</{sim_group.tag}>",)
                else:
                    effective_markers = ()

                self.agent.history.repair_tail()
                all_messages = prepare_messages_for_api(
                    self.agent, debug_hash_check=Config.DEBUG_PREFIX_HASH_CHECK
                )

                try:
                    message_obj, api_error_occurred, stopped_at_marker = self.call_with_retries(
                        all_messages,
                        step_prefill=step_prefill,
                        reasoning_effort=turn_reasoning,
                        stop_markers=effective_markers,
                        prefill_shown=sim_group.prefill_shown() if sim_group else 0,
                        defer_stream_end=sim_group is not None,
                        gen_params=think_params,
                    )
                except GenerationInterrupted:
                    _close_sim_stream()
                    self.agent.on_system_msg("⏹ Generation stopped by user — control returned to you.")
                    self.agent.clear_stop()
                    return _finish_turn("")

                if api_error_occurred or not message_obj:
                    _close_sim_stream()
                    self.agent.recovery_policy.recover('api_error' if api_error_occurred else 'duplicate_loop')
                    return _finish_turn("")

                controller_verdict = None
                if (
                    structured_output is not None
                    and structured_output.controller is not None
                    and message_obj is not None
                ):
                    controller_verdict = structured_output.controller.advance(message_obj.content or "")
                    if controller_verdict.corrected_content != (message_obj.content or ""):
                        message_obj.content = controller_verdict.corrected_content

                if (
                    controller_verdict is not None
                    and not controller_verdict.complete
                    and not controller_verdict.corrected_content
                    and controller_verdict.next_prefill is not None
                ):
                    if state.structured_output_phase + 1 >= structured_output.max_phases:
                        return _finish_turn(structured_output.controller.document())
                    state.set_prefill(controller_verdict.next_prefill)
                    state.structured_output_phase += 1
                    continue

                sim_broken_scan = None
                if sim_group is not None and message_obj is not None:
                    sim_group.absorb(message_obj, step_prefill)
                    sim_stream_started = True
                    if sim_group.is_think_phase():
                        sim_group.close_section()
                        appended = sim_group.take_appended()
                        if appended and self.agent.on_stream_chunk:
                            self.agent.on_stream_chunk(appended)
                        if sim_group.tool_calls or sim_group.answer_text().strip():
                            sim_msg = self.agent._sim_message(sim_group)
                        else:
                            sim_group.advance_to_answer()
                            continue
                    else:
                        sim_msg = self.agent._sim_verify_phase(sim_group)
                        if sim_msg is None:
                            continue
                    sim_broken_scan = sim_group.answer_text()
                    message_obj = sim_msg
                    sim_open = False
                    _close_sim_stream()

                try:
                    result_text, tool_error_occurred, broken_call, rerun_prefill = self.agent._process_llm_response(
                        message_obj,
                        no_comment_retry_left=state.no_comment_retries_left,
                        reasoning_effort=turn_reasoning,
                        skip_broken_detection=bool(
                            structured_output is not None
                            and (stopped_at_marker or structured_output.controller is not None)
                        ),
                        allow_tools=not service_mode,
                        broken_scan_text=sim_broken_scan,
                    )
                except GenerationInterrupted:
                    self.agent.history.pop_pending_tool_calls()
                    self.agent.history.normalize()
                    self.agent._on_history_changed()
                    self.agent.on_system_msg("⏹ Generation stopped by user.")
                    _close_sim_stream()
                    if not service_mode:
                        self.agent.autosave()
                    raise

                if rerun_prefill:
                    state.no_comment_retries_left -= 1
                    state.set_prefill(rerun_prefill)
                    continue

                if (
                    self.agent._pending_operation is not None
                    and not service_mode
                    and not message_obj.tool_calls
                    and not tool_error_occurred
                ):
                    self.agent._erase_last_assistant()
                    if not state.can_retry_answer_guard():
                        self.agent.recovery_policy.recover('answer_guard')
                        return _finish_turn("")
                    self.agent.on_system_msg(
                        f"⏳ [CONFIRMATION PENDING] The model replied with text instead of "
                        f"calling {apply_edit.__name__} or {respond_to_system.__name__} — waiting for the tool call "
                        f"(attempt {state.guard_retries_spent}/{Config.ANSWER_GUARD_MAX_RETRIES})."
                    )
                    self.agent.recovery_policy.ensure_guard_nag()
                    self.agent._temp_override = Config.ERROR_RECOVERY_TEMP
                    self.agent._on_history_changed()
                    continue

                state.clear_prefill()

                if broken_call:
                    if state.can_retry_broken_regen():
                        self.agent._erase_last_assistant()
                        self.agent.recovery_policy.recover('broken_call', retries_left=state.broken_regen_left)
                        continue
                    if state.can_retry_broken_fix():
                        self.agent.history.add(UserMessage(self.agent.recovery_policy.build_broken_call_fix()))
                        self.agent.on_render(self.agent.history.get_all()[-1])
                        self.agent.recovery_policy.recover('broken_fix', retries_left=state.broken_fix_left)
                        continue
                    self.agent.recovery_policy.recover('broken_giveup')
                    return _finish_turn("")

                if tool_error_occurred and state.can_retry_tool_error():
                    erased = self.agent._erase_last_failed_tool_call()
                    self.agent.recovery_policy.recover(
                        'tool_error',
                        retries_left=state.tool_error_retries_left,
                        erased_count=erased,
                    )
                    continue

                compacted = False
                if not service_mode:
                    compacted = self.agent._compact_completed_tasks() > 0
                if (
                    not service_mode
                    and not getattr(self.agent, '_auto_summarize_suppressed', False)
                    and self.agent._get_context_usage_percent() >= self.agent._current_summary_threshold()
                ):
                    if getattr(self.agent, '_is_subagent', False):
                        return _finish_turn(result_text)
                    if self.agent._auto_summarize_dialogue():
                        compacted = True
                        self.agent._auto_summarize_suppressed = True

                if compacted:
                    state.reset_error_counts()

                if tool_error_occurred:
                    state.record_tool_error(self.agent._last_failed_tool_signature())
                else:
                    state.record_tool_success()

                if state.max_errors_reached:
                    self.agent.recovery_policy.recover('error_limit', retries_left=MAX_CONSECUTIVE_ERRORS)
                    return _finish_turn("")

                if (
                    structured_output is not None
                    and structured_output.controller is not None
                    and controller_verdict is not None
                ):
                    if controller_verdict.complete:
                        return _finish_turn(structured_output.controller.document())
                    if controller_verdict.next_prefill is not None:
                        if state.structured_output_phase + 1 >= structured_output.max_phases:
                            return _finish_turn(structured_output.controller.document())
                        state.set_prefill(controller_verdict.next_prefill)
                        state.structured_output_phase += 1
                        continue
                    if stopped_at_marker and not message_obj.tool_calls and not tool_error_occurred:
                        state.structured_output_phase += 1
                        continue
                    if not message_obj.tool_calls and not tool_error_occurred:
                        return _finish_turn(structured_output.controller.document())

                if (
                    structured_output is not None
                    and stopped_at_marker
                    and not message_obj.tool_calls
                    and not tool_error_occurred
                    and state.structured_output_phase < len(structured_output.next_prefills)
                    and state.structured_output_phase + 1 < structured_output.max_phases
                ):
                    state.set_prefill(structured_output.next_prefills[state.structured_output_phase])
                    state.structured_output_phase += 1
                    continue

                if not message_obj.tool_calls:
                    return _finish_turn(result_text)

            self.agent.on_system_msg(f"⚠️ Reached max iterations ({max_iter}).")
            return _finish_turn("")
        finally:
            _close_sim_stream()

    def service_llm_call(
        self,
        msgs: list[dict],
        temp: Optional[float] = None,
        timeout: Optional[int] = None,
        tools=True,
        prefill: Optional[str] = None,
        params: GenerationParams = None,
        structured_output: Optional[StructuredOutputConfig] = None,
    ) -> tuple:
        prompt = None
        for m in reversed(msgs):
            if isinstance(m, dict) and m.get("role") == "user":
                prompt = m.get("content")
                break
        last_user = self.agent.history.get_last_user_message()
        prompt_clean = _strip_system_header(prompt) if prompt else None
        if prompt_clean is not None and last_user is not None and (last_user.content or "") == prompt_clean:
            append_prompt = False
        else:
            append_prompt = True
        result = self.run_service_turn(
            prompt or "",
            append_prompt=append_prompt,
            temp=temp,
            timeout=timeout,
            prefill=prefill,
            params=params,
            structured_output=structured_output,
        )
        if result:
            return SimpleNamespace(content=result, tool_calls=[], reasoning_content=""), None, None
        return None, "Empty response", None

    def run_service_turn(
        self,
        prompt: str,
        *,
        append_prompt: bool = True,
        keep_in_history: Optional[bool] = None,
        max_iter: Optional[int] = None,
        temp: Optional[float] = None,
        timeout: Optional[int] = None,
        prefill: Optional[str] = None,
        params: GenerationParams = None,
        structured_output: Optional[StructuredOutputConfig] = None,
    ) -> str:
        keep = Config.SERVICE_TURN_KEEP_IN_HISTORY if keep_in_history is None else keep_in_history
        max_iter = max_iter if max_iter is not None else Config.SERVICE_TURN_MAX_ITER
        start_len = len(self.agent.history.get_all())
        usage_before = self.agent.token_tracker.last_usage
        stream_swap = None
        if self.agent.streaming_enabled and self.agent.on_service_stream_chunk:
            stream_swap = (self.agent.on_stream_start, self.agent.on_stream_chunk, self.agent.on_stream_end)
            self.agent.on_stream_start = self.agent.on_service_stream_start or (lambda: None)
            self.agent.on_stream_chunk = self.agent.on_service_stream_chunk
            self.agent.on_stream_end = self.agent.on_service_stream_end or (lambda: None)

        prev_suppressed = self.agent._auto_summarize_suppressed
        self.agent._auto_summarize_suppressed = True
        self.agent._in_service_turn = True
        prev_params = self.agent._gen_params
        if params is not None or temp is not None or timeout is not None:
            p = params or GenerationParams()
            self.agent._gen_params = dc_replace(
                self.agent._gen_params,
                temp=temp if temp is not None else (p.temp if p.temp is not None else self.agent._gen_params.temp),
                timeout=timeout if timeout is not None else (p.timeout if p.timeout is not None else self.agent._gen_params.timeout),
            )
        try:
            if append_prompt:
                self.append_service_prompt(prompt)
            result = self.run_turn_loop(
                max_iter,
                prefill=prefill,
                structured_output=structured_output,
                service_mode=True,
            ) or ""
        finally:
            if not keep:
                end = len(self.agent.history.get_all())
                if end > start_len:
                    self.agent.history.remove_at(range(start_len, end))
                    self.agent.history.normalize()
                    self.agent._on_history_changed()
                self.agent.token_tracker.last_usage = usage_before
            self.agent._gen_params = prev_params
            self.agent._auto_summarize_suppressed = prev_suppressed
            self.agent._in_service_turn = False
            if stream_swap is not None:
                self.agent.on_stream_start, self.agent.on_stream_chunk, self.agent.on_stream_end = stream_swap
        return result

    def append_service_prompt(self, prompt: str) -> None:
        last = self.agent.history.get_last_message()
        if isinstance(last, UserMessage):
            self.agent.history.add(AssistantMessage(content=""))
        self.agent.history.add(UserMessage(content=prompt))

    def inject_user_interrupt(self, user_text: str, max_iter: int = None) -> str:
        max_iter = max_iter if max_iter is not None else Config.MAX_ITER
        if self.agent.self_consistency_mode:
            self.agent.clear_pending_interrupt()
            return self.agent._chat_self_consistent(f"{INTERRUPT_HEADER}\n\n{user_text}", None)
        self.agent.clear_pending_interrupt()
        if not self.append_interrupt_to_tool_result(user_text):
            self.prepare_turn(f"{INTERRUPT_HEADER}\n\n{user_text}")
        return self.run_turn_loop(max_iter)

    def append_interrupt_to_tool_result(self, user_text: str) -> bool:
        last = self.agent.history.get_last_message()
        if not isinstance(last, ToolResult):
            return False
        last.content = (last.content or "") + f"\n\n{INTERRUPT_HEADER}\n\n{user_text}"
        self.agent._on_history_changed()
        self.agent.autosave()
        self.agent.clear_stop()
        return True
