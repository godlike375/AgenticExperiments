from __future__ import annotations

from typing import Union, Callable, Optional

from universal_agents.constants import ENVIRONMENT_PREFIX, ENVIRONMENT_PREFIX_END
from universal_agents.config import Config
from universal_agents.models import UserMessage
from universal_agents.llm_client import LLMClient, TokenUsageTracker, LoopDetector, jaccard_similarity
from universal_agents.history import ChatHistory
from universal_agents.generation import GenerationParams
from universal_agents.tool_manager import ToolManager
from universal_agents.sub_agent import SubAgent
from universal_agents.context_builder import prepare_messages_for_api, get_effective_prefill
from universal_agents.file_states import FileStateTracker
from universal_agents.tool_parsing import tc_name, tc_args

from universal_agents.agent_mixins import (
    ToolsMixin,
    MemoryMixin,
    HistoryMixin,
    StreamingMixin,
    ResponseMixin,
    ExecuteMixin,
    ConsistencyMixin,
)


class LLMAgent(
    ToolsMixin,
    MemoryMixin,
    HistoryMixin,
    StreamingMixin,
    ResponseMixin,
    ExecuteMixin,
    ConsistencyMixin,
):
    def __init__(
        self,
        system_prompt: str = "You are a helpful assistant",
        temp: float = None,
        timeout: int = None,
        tools_config: Union[list[str], dict, None] = None,
        on_render: Callable = lambda x: None,
        on_confirm: Callable[[str, dict], bool] = lambda n, a: True,
        on_system_msg: Callable[[str], None] = lambda x: None,
        external_plugins: dict[str, Callable] = None,
        max_context_tokens: int = None,
        top_p: float = None,
        frequency_penalty: float = None,
        presence_penalty: float = None,
        max_tokens: int = None,
        on_stream_chunk: Callable[[str], None] = None,
        on_stream_start: Callable[[], None] = None,
        on_stream_end: Callable[[], None] = None,
        on_reasoning_chunk: Callable[[str], None] = None,
        on_reasoning_start: Callable[[], None] = None,
        on_reasoning_end: Callable[[], None] = None,
        streaming_enabled: bool = None,
        max_generation_attempts: int = None,
        disable_per_msg_summarization: bool = False,
    ):
        self.history = ChatHistory(system_prompt)
        self.file_states = FileStateTracker(self.history)
        self._read_registrations: list[str] = []
        self._gen_params = GenerationParams.from_overrides(
            temp=temp,
            timeout=timeout,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            max_tokens=max_tokens,
        )
        self.temp = self._gen_params.temp
        self.timeout = self._gen_params.timeout
        self.top_p = self._gen_params.top_p
        self.frequency_penalty = self._gen_params.frequency_penalty
        self.presence_penalty = self._gen_params.presence_penalty
        self.max_tokens = self._gen_params.max_tokens
        self.on_render = on_render
        self.on_confirm = on_confirm
        self.on_system_msg = on_system_msg
        self.self_consistency_mode = False
        self.sc_samples = 3
        self.token_tracker = TokenUsageTracker(system_prompt, max_context_tokens if max_context_tokens is not None else Config.MAX_CONTEXT_TOKENS)
        self.tools_manager = ToolManager(tools_config, external_plugins)
        self._auto_trust_git_root()
        self.loop_detector = LoopDetector()
        self._temp_override: Optional[float] = None
        self._original_temp = temp
        self.on_stream_chunk = on_stream_chunk
        self.on_stream_start = on_stream_start
        self.on_stream_end = on_stream_end
        self.on_reasoning_chunk = on_reasoning_chunk
        self.on_reasoning_start = on_reasoning_start
        self.on_reasoning_end = on_reasoning_end
        self.streaming_enabled = streaming_enabled if streaming_enabled is not None else Config.STREAM_ENABLED
        self._tool_usage: dict[str, int] = {}
        self._max_generation_attempts = max_generation_attempts
        self._last_response_id: Optional[str] = None
        self._last_sent_msg_count: int = 0
        self._depth: int = 0
        self._disable_per_msg_summarization = disable_per_msg_summarization
        self._compacted_task_ids: set[str] = set()
        self.task_plan: list[str] = []
        self.task_plan_map: dict = {}
        # Рабочая память (НЕ попадает в контекст): плотные саммари отдельных
        # сообщений (ключ — id(Message)). Используются при сжатии диалога,
        # чтобы из маленьких саммари собрать более короткий новый диалог.
        self._per_msg_summaries: dict[int, str] = {}

    def make_sub_agent(
        self,
        *,
        tools_config=None,
        external_plugins=None,
        safe_only: bool = True,
        max_iter: int = None,
        temp: float = None,
        on_log: Callable = None,
        depth: int = 0,
        disable_per_msg_summarization: bool = True,
    ) -> SubAgent:
        """Создаёт SubAgent, наследуя историю и бюджет токенов родителя."""
        parent_system_prompt = self.history[0].content if len(self.history) else ""
        parent_history = self.history.get_all()[1:]
        return SubAgent(
            parent_system_prompt=parent_system_prompt,
            parent_history=parent_history,
            max_context_tokens=self.token_tracker.max_context_tokens,
            tools_config=tools_config,
            external_plugins=external_plugins,
            safe_only=safe_only,
            max_iter=max_iter,
            temp=temp,
            on_log=on_log if on_log is not None else (lambda x: None),
            depth=depth,
            disable_per_msg_summarization=disable_per_msg_summarization,
        )

    # --------------------------------------------------------
    # Главный цикл
    # --------------------------------------------------------
    def _generate_response(
        self,
        messages_to_send: list[dict],
        step_prefill: Optional[str],
        prev_response_id: Optional[str],
        max_generation_attempts: int,
        all_messages_len: int,
    ) -> tuple:
        """Проводит до max_generation_attempts попыток генерации, отбрасывая
        дубликаты (повторный tool call / повтор ответа) и активируя буст
        температуры. Возвращает (message_obj, api_error_occurred)."""
        message_obj = None
        last_retry_warning: Optional[str] = None
        dup_watch_target: Optional[str] = None
        api_error_occurred = False

        for attempt in range(max_generation_attempts):
            attempt_params = self._gen_params
            if self._temp_override is not None:
                attempt_params = self._gen_params.with_temp(self._temp_override)
                self._temp_override = None

            active_messages = [dict(msg) for msg in messages_to_send]

            if last_retry_warning:
                if active_messages:
                    last_msg = active_messages[-1]
                    last_msg["content"] = (last_msg.get("content") or "") + last_retry_warning

            # Streaming или обычный режим
            if self.streaming_enabled and self.on_stream_chunk:
                message_obj, err, usage = self._call_with_streaming(
                    active_messages,
                    step_prefill,
                    tools=self.tools if self.tools else None,
                    previous_response_id=prev_response_id,
                    params=attempt_params,
                    watch_prefix=dup_watch_target,
                    watch_continue_temp=Config.DUPLICATE_CONTINUATION_TEMP if dup_watch_target is not None else None,
                )
            else:
                message_obj, err, usage = LLMClient.call(
                    active_messages,
                    tools=self.tools if self.tools else None,
                    prefill=step_prefill,
                    previous_response_id=prev_response_id,
                    params=attempt_params,
                )
            dup_watch_target = None

            if usage:
                self.token_tracker.update_from_usage(usage)
            if err:
                self.on_system_msg(f"[API Error] {err}")
                api_error_occurred = True
                break

            if not message_obj:
                api_error_occurred = True
                break

            if message_obj.tool_calls:
                has_duplicate = False
                current_history = self.history.get_all()
                for tc in message_obj.tool_calls:
                    if self.loop_detector.check_duplicate_in_turn(tc_name(tc), tc_args(tc), current_history):
                        has_duplicate = True
                        dup_name, dup_args = tc_name(tc), tc_args(tc)
                        last_retry_warning = (
                            f"\n\n{ENVIRONMENT_PREFIX} Your previous attempt to call tool '{dup_name}' "
                            f"with args '{dup_args}' was blocked as a duplicate of a recently already made call."
                            f" Do NOT call '{dup_name}' again with same params. "
                            f"Use other params or a different tool or write a text answer."
                            f"{ENVIRONMENT_PREFIX_END}"
                        )
                        self.on_system_msg(
                            f"[PROACTIVE LOOP DETECTED] Intercepted duplicate call to '{dup_name}'. "
                            f"Discarding response. Activating temperature boost ({Config.BOOST_TEMP}) "
                            f"and injecting temporary warning. Attempt {attempt + 1}/{max_generation_attempts}."
                        )
                        break

                if has_duplicate:
                    self._temp_override = Config.BOOST_TEMP
                    continue

            else:
                answer_text = (message_obj.content or "").strip()
                prev_answer = self._get_last_answer_text()
                is_duplicate = (
                    answer_text
                    and prev_answer
                    and jaccard_similarity(answer_text, prev_answer)
                    >= Config.DUPLICATE_SIMILARITY_THRESHOLD
                )
                if is_duplicate:
                    last_retry_warning = (
                        f"\n\n{ENVIRONMENT_PREFIX} Your previous answer was the same to the latest one. "
                        f"Please do NOT repeat it again and answer differently."
                        f"{ENVIRONMENT_PREFIX_END}"
                    )
                    dup_watch_target = prev_answer
                    self.on_system_msg(
                        f"[DUPLICATE ANSWER DETECTED] Model repeated the previous answer verbatim. "
                        f"Discarding response. Activating temperature boost ({Config.BOOST_TEMP}) "
                        f"and injecting temporary warning. Attempt {attempt + 1}/{max_generation_attempts}."
                    )
                    self._temp_override = Config.BOOST_TEMP
                    continue

            # Ответ принят к обработке — фиксируем позицию контекста для
            # возможного продолжения через previous_response_id.
            if hasattr(message_obj, '_response_id'):
                self._last_response_id = message_obj._response_id
            self._last_sent_msg_count = all_messages_len
            break
        else:
            self.on_system_msg(
                "[PROACTIVE LOOP DETECTOR] Max re-generation attempts reached. Proceeding to execution safety nets."
            )

        return message_obj, api_error_occurred

    def chat(self, message: str, max_iter: int = None, prefill: str = None):
        max_iter = max_iter if max_iter is not None else Config.MAX_ITER
        if self.self_consistency_mode:
            return self._chat_self_consistent(message, prefill)

        user_msg = UserMessage(content=message)
        self.history.add(user_msg)
        if not (Config.DISABLE_PER_MESSAGE_SUMMARIZATION or self._disable_per_msg_summarization):
            self._maybe_summarize_user_message(user_msg)
        current_prefill = get_effective_prefill(prefill)
        self._last_response_id = None
        self._last_sent_msg_count = 0

        consecutive_errors = 0
        MAX_CONSECUTIVE_ERRORS = 5
        tool_error_retries_left = Config.ERROR_RECOVERY_RETRIES
        broken_regen_left = Config.BROKEN_CALL_REGEN_RETRIES
        broken_fix_left = Config.BROKEN_CALL_FIX_RETRIES

        for i in range(max_iter):
            step_prefill = current_prefill if i == 0 else None
            all_messages = prepare_messages_for_api(self)

            if i == 0 or self._last_response_id is None:
                messages_to_send = all_messages
                prev_response_id = None
            else:
                messages_to_send = all_messages[self._last_sent_msg_count:]
                prev_response_id = self._last_response_id

            max_generation_attempts = self._max_generation_attempts if self._max_generation_attempts is not None else Config.MAX_LOOP_RETRIES

            message_obj, api_error_occurred = self._generate_response(
                messages_to_send,
                step_prefill,
                prev_response_id,
                max_generation_attempts,
                len(all_messages),
            )

            if api_error_occurred or not message_obj:
                self.history.normalize(is_error_recovery=True)
                self.on_system_msg("⚠️ [RECOVERY] API error occurred. Role sequence restored. Handing control to user.")
                return ""

            result_text, tool_error_occurred, broken_call = self._process_llm_response(message_obj)

            if broken_call:
                if broken_regen_left > 0:
                    broken_regen_left -= 1
                    self._erase_last_assistant()
                    self._temp_override = Config.ERROR_RECOVERY_TEMP
                    self._last_response_id = None
                    self._last_sent_msg_count = 0
                    self.on_system_msg(
                        f"[BROKEN CALL] Detected malformed tool call. "
                        f"Regenerating with temp {Config.ERROR_RECOVERY_TEMP} "
                        f"(retries left: {broken_regen_left})."
                    )
                    continue
                if broken_fix_left > 0:
                    broken_fix_left -= 1
                    self.history.add(UserMessage(
                        f"{ENVIRONMENT_PREFIX} Your previous message looked like a tool call "
                        f"but was not successfully parsed by the API. Try to write it properly now."
                        f"{ENVIRONMENT_PREFIX_END}"
                    ))
                    self.on_render(self.history.get_all()[-1])
                    self._last_response_id = None
                    self._last_sent_msg_count = 0
                    self.on_system_msg(
                        f"[BROKEN CALL] Injected fix prompt for the model "
                        f"(retries left: {broken_fix_left})."
                    )
                    continue
                self.history.normalize(is_error_recovery=True)
                self.on_system_msg(
                    f"⚠️ [BROKEN CALL] Could not recover malformed tool call after "
                    f"{Config.BROKEN_CALL_REGEN_RETRIES + Config.BROKEN_CALL_FIX_RETRIES} attempts. "
                    "Handing control to user."
                )
                return ""

            if tool_error_occurred and tool_error_retries_left > 0:
                tool_error_retries_left -= 1
                erased = self._erase_last_failed_tool_call()
                self._temp_override = Config.ERROR_RECOVERY_TEMP
                self._last_response_id = None
                self._last_sent_msg_count = 0
                self.on_system_msg(
                    f"[TOOL ERROR RECOVERY] Tool error detected. Erased {erased} message(s) "
                    f"and regenerating with temperature {Config.ERROR_RECOVERY_TEMP} "
                    f"(retries left: {tool_error_retries_left})."
                )
                continue

            self._compact_completed_tasks()
            if self._get_context_usage_percent() >= Config.AUTO_SUMMARY_THRESHOLD:
                self._auto_summarize_dialogue()

            if tool_error_occurred:
                consecutive_errors += 1
            else:
                consecutive_errors = 0
                tool_error_retries_left = Config.ERROR_RECOVERY_RETRIES

            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                self.history.normalize(is_error_recovery=True)
                self.on_system_msg(
                    f"⚠️ [LIMIT REACHED] {MAX_CONSECUTIVE_ERRORS} consecutive tool errors. Handing control to user."
                )
                return ""

            if not message_obj.tool_calls and not tool_error_occurred:
                return result_text