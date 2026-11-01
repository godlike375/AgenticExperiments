from __future__ import annotations
import json
from typing import TYPE_CHECKING, Optional, Callable

from universal_agents.config import Config
from universal_agents.constants import ENVIRONMENT_PREFIX, ENVIRONMENT_PREFIX_END

_GUARD_NAG_PREFIX = "You can't continue" 
from universal_agents.models import AssistantMessage, UserMessage
from universal_agents.llm_client import jaccard_similarity, text_hash
from universal_agents.tool_parsing import parse_tool_args, tc_name, tc_args
from universal_agents.tool_manager import DISPATCH_TOOL_NAME
from universal_agents.tools.builtin import respond_to_system, apply_edit

if TYPE_CHECKING:
    from universal_agents.agent import LLMAgent

_CONFIRMATION_GUARD_VERDICT = "confirmation_guard"

_REPEAT_ANSWER_NAG = (
    f"\n\n{ENVIRONMENT_PREFIX} Your previous answer was the same to the latest one. "
    f"Please do NOT repeat it again and answer differently."
    f"{ENVIRONMENT_PREFIX_END}"
)


class RecoveryPolicy:
    """Политика восстановления после сбоев и детекции зацикливаний (§1.7, §1.9, §1.10, §1.11, §2.6)."""

    def __init__(self, agent: LLMAgent):
        self.agent = agent

    def build_duplicate_warning(self, kind: str, duplicate_target: str) -> str:
        prompt_override = getattr(Config, 'DUPLICATE_PROMPT_OVERRIDE', None)
        if prompt_override:
            return prompt_override(kind, duplicate_target)
        if kind == 'tool_call':
            tool_name = duplicate_target[0] if isinstance(duplicate_target, (list, tuple)) else duplicate_target
            return (
                f"{ENVIRONMENT_PREFIX} SYSTEM ALERT: You have called tool '{tool_name}' "
                f"with the exact same arguments repeatedly. DO NOT make this exact tool call again. "
                f"Change arguments, try another tool, or reply with text. {ENVIRONMENT_PREFIX_END}"
            )
        return (
            f"{ENVIRONMENT_PREFIX} SYSTEM ALERT: You are repeating your previous response. "
            f"Do NOT output the same text again. Provide a different answer or proceed with next steps. "
            f"{ENVIRONMENT_PREFIX_END}"
        )

    def build_broken_call_fix(self) -> str:
        return (
            f"{ENVIRONMENT_PREFIX} SYSTEM: Your previous tool call was malformed and could not be parsed. "
            f"Please call the tool again with valid arguments adhering to the tool schema. {ENVIRONMENT_PREFIX_END}"
        )

    def ensure_guard_nag(self) -> None:
        """Один наг на цикл (§1.11): не пересоздаётся при повторах — байт-стабильный KV-префикс."""
        nag = self.agent._guard_nag_message
        if nag is not None and any(m is nag for m in self.agent.history.get_all()):
            return
        if nag is None:
            nag = UserMessage(
                f"{ENVIRONMENT_PREFIX} {_GUARD_NAG_PREFIX} unless you answer to the system question using '{apply_edit.__name__}' "
                f"(or '{respond_to_system.__name__}') tool. Call it right now! {ENVIRONMENT_PREFIX_END}"
            )
            nag._is_guard_nag = True
            self.agent._guard_nag_message = nag
        self.agent.history.add(nag)
        self.agent.on_render(self.agent.history.get_all()[-1])

    def consecutive_repeats(self, text: str, *, reasoning: bool) -> int:
        target = text_hash(text)
        count = 0
        for msg in reversed(self.agent.history.get_all()):
            if not isinstance(msg, AssistantMessage):
                continue
            value = ((msg.reasoning_content or "") if reasoning else (msg.content or "")).strip()
            if not value:
                continue
            if text_hash(value) != target:
                break
            count += 1
        return count

    def recent_answer_texts(self, limit: int = 2) -> list[str]:
        texts: list[str] = []
        for msg in reversed(self.agent.history.get_all()):
            if isinstance(msg, AssistantMessage) and (msg.content or "").strip():
                texts.append(msg.content.strip())
                if len(texts) == limit:
                    break
        return texts

    def last_assistant_message(self) -> Optional[AssistantMessage]:
        for msg in reversed(self.agent.history.get_all()):
            if isinstance(msg, AssistantMessage):
                return msg
        return None

    def call_signature(self, tool_calls) -> tuple:
        return tuple(sorted(
            (name, self.agent.loop_detector.normalize_args(args))
            for name, args in (self.actual_tool_call(tc) for tc in (tool_calls or []))
        ))

    @staticmethod
    def actual_tool_call(tc) -> tuple[str, str]:
        name = tc_name(tc)
        args = tc_args(tc)
        if name == DISPATCH_TOOL_NAME:
            dispatch_args = parse_tool_args(args)
            name = dispatch_args.get("name", "")
            args = dispatch_args.get("arguments", "{}")
            if isinstance(args, dict):
                args = json.dumps(args, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return name, args

    def detect_duplicate(self, message_obj) -> Optional[tuple]:
        content = (message_obj.content or "").strip()
        reasoning = (getattr(message_obj, 'reasoning_content', '') or '').strip()
        pending = self.agent._pending_operation is not None

        if pending and message_obj.tool_calls and any(
            tc_name(tc) in (respond_to_system.__name__, apply_edit.__name__)
            or (
                tc_name(tc) == DISPATCH_TOOL_NAME
                and parse_tool_args(tc_args(tc)).get("name") in (respond_to_system.__name__, apply_edit.__name__)
            )
            for tc in message_obj.tool_calls
        ):
            return None

        args_changed = False
        if message_obj.tool_calls:
            prev = self.last_assistant_message()
            args_changed = bool(
                prev is not None
                and prev.tool_calls
                and self.call_signature(prev.tool_calls) != self.call_signature(message_obj.tool_calls)
            )

        min_chars = Config.DUPLICATE_TEXT_MIN_CHARS
        reasoning_min_chars = Config.DUPLICATE_REASONING_MIN_CHARS
        required_prior = max(2, Config.DUPLICATE_RUN_LENGTH) - 1 if message_obj.tool_calls else 1
        if not args_changed and (content or reasoning):
            if (
                content
                and (min_chars <= 0 or len(content) >= min_chars)
                and self.consecutive_repeats(content, reasoning=False) >= required_prior
            ):
                if pending and not message_obj.tool_calls:
                    return _CONFIRMATION_GUARD_VERDICT, content
                return 'answer', content
            if (
                reasoning
                and (reasoning_min_chars <= 0 or len(reasoning) >= reasoning_min_chars)
                and self.consecutive_repeats(reasoning, reasoning=True) >= required_prior
            ):
                if pending and not message_obj.tool_calls:
                    return _CONFIRMATION_GUARD_VERDICT, reasoning
                return 'reasoning', reasoning

        if message_obj.tool_calls:
            current_history = self.agent.history.get_all()
            seen_in_batch: set[str] = set()
            for tc in message_obj.tool_calls:
                actual_name, actual_args = self.actual_tool_call(tc)
                norm = self.agent.loop_detector.normalize_args(actual_args)
                batch_key = f"{actual_name}:{norm}"
                if batch_key in seen_in_batch:
                    return 'tool_call', (actual_name, actual_args)
                seen_in_batch.add(batch_key)
                if self.agent.loop_detector.check_duplicate_in_turn(actual_name, actual_args, current_history):
                    return 'tool_call', (actual_name, actual_args)
            return None

        answer_text = content
        recent_answers = self.recent_answer_texts(1)
        if (
            answer_text
            and recent_answers
            and jaccard_similarity(answer_text, recent_answers[0])
            >= Config.DUPLICATE_SIMILARITY_THRESHOLD
        ):
            if pending:
                return _CONFIRMATION_GUARD_VERDICT, recent_answers[0]
            return 'answer', recent_answers[0]
        return None

    def recover(self, kind: str, retries_left: int = 0, erased_count: int = 0) -> None:
        if kind == 'api_error':
            self.agent.history.normalize(is_error_recovery=True)
            self.agent.on_system_msg("⚠️ [RECOVERY] API error occurred. Role sequence restored. Handing control to user.")
            return
        if kind == 'duplicate_loop':
            self.agent.history.normalize(is_error_recovery=True)
            self.agent.on_system_msg(
                "⚠️ [LOOP DETECTED] Model keeps repeating itself after multiple attempts. "
                "Handing control to user."
            )
            return
        if kind == 'broken_giveup':
            self.agent.history.normalize(is_error_recovery=True)
            self.agent.on_system_msg(
                f"⚠️ [BROKEN CALL] Could not recover malformed tool call after "
                f"{Config.BROKEN_CALL_REGEN_RETRIES + Config.BROKEN_CALL_FIX_RETRIES} attempts. "
                "Handing control to user."
            )
            return
        if kind == 'error_limit':
            self.agent.history.normalize(is_error_recovery=True)
            self.agent.on_system_msg(
                f"⚠️ [LIMIT REACHED] {retries_left} consecutive tool errors. Handing control to user."
            )
            return
        if kind == 'answer_guard':
            self.agent.history.normalize(is_error_recovery=True)
            self.agent.on_system_msg(
                f"⚠️ [CONFIRMATION LOOP] Model kept replying in text instead of calling "
                f"{apply_edit.__name__} or {respond_to_system.__name__} "
                f"({Config.ANSWER_GUARD_MAX_RETRIES} guard retries exhausted). Handing control to user."
            )
            return

        if kind == 'broken_call':
            self.agent._temp_override = Config.ERROR_RECOVERY_TEMP
            self.agent.on_system_msg(
                f"[BROKEN CALL] Detected malformed tool call. "
                f"Regenerating with temp {Config.ERROR_RECOVERY_TEMP} "
                f"(retries left: {retries_left})."
            )
        elif kind == 'broken_fix':
            self.agent.on_system_msg(
                f"[BROKEN CALL] Injected fix prompt for the model "
                f"(retries left: {retries_left})."
            )
        elif kind == 'tool_error':
            self.agent._temp_override = Config.ERROR_RECOVERY_TEMP
            self.agent.on_system_msg(
                f"[TOOL ERROR RECOVERY] Tool error detected. Erased {erased_count} message(s) "
                f"and regenerating with temperature {Config.ERROR_RECOVERY_TEMP} "
                f"(retries left: {retries_left})."
            )
