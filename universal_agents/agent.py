from __future__ import annotations

import json
from typing import Union, Callable, Optional

from universal_agents.constants import ENVIRONMENT_PREFIX
from universal_agents.config import Config
from universal_agents.models import UserMessage, AssistantMessage, ToolCall, ToolResult
from universal_agents.llm_client import LLMClient, TokenUsageTracker, LoopDetector, apply_prefill, build_usage_dict
from universal_agents.history import ChatHistory
from universal_agents.generation import GenerationParams
from universal_agents.tool_manager import ToolManager
from universal_agents.sub_agent import SubAgent

from universal_agents.compressors import auto_compress_tool_result, summarize_dialogue
from universal_agents.context_builder import prepare_messages_for_api, get_effective_prefill
from universal_agents.history_repair import prune_all_failed_tool_calls_except_last


def _tc_name(tc) -> str:
    """Имя tool call независимо от формата (OpenAI/Responses-парсинг)."""
    func = getattr(tc, 'function', None)
    if func is not None:
        return getattr(func, 'name', None) or ""
    return getattr(tc, 'name', None) or ""


def _tc_args(tc) -> str:
    """Аргументы tool call независимо от формата (OpenAI/Responses-парсинг)."""
    func = getattr(tc, 'function', None)
    if func is not None:
        return getattr(func, 'arguments', None) or ""
    return getattr(tc, 'arguments', None) or ""


class LLMAgent:
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
    ):
        self.history = ChatHistory(system_prompt)
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
        self.loop_detector = LoopDetector()
        self._temp_boost_active = False
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

    # --------------------------------------------------------
    # Делегирование управления инструментами в ToolManager
    # --------------------------------------------------------
    @property
    def _all_tools(self) -> dict:
        """Карта активных инструментов (схемы + обработчики)."""
        return self.tools_manager.tools_map

    @property
    def tools(self) -> list[dict]:
        """JSON-схемы активных инструментов для API."""
        return self.tools_manager.schemas

    @property
    def _tools_config(self):
        return self.tools_manager.config

    @property
    def trusted_dirs(self) -> set[str]:
        return self.tools_manager.trusted_dirs

    def load_tools(self, name: str) -> str:
        """Enable a previously disabled tool by name."""
        return self.tools_manager.load(name)

    def unload_tool(self, name: str) -> str:
        """Disable a tool by name, removing it from available tools."""
        return self.tools_manager.unload(name)

    def list_available_tools(self) -> str:
        """List all available (loadable) tools from plugins directory."""
        return self.tools_manager.list_available()

    def trust_dir(self, path: str) -> str:
        """Add a directory to trusted dirs (edit_file skips confirmation)."""
        return self.tools_manager.trust_dir(path)

    def untrust_dir(self, path: str) -> str:
        """Remove a directory from trusted dirs."""
        return self.tools_manager.untrust_dir(path)

    def is_path_trusted(self, path: str) -> bool:
        """Check if path is inside a trusted directory."""
        return self.tools_manager.is_path_trusted(path)

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
        )

    def _emit_token_info(self):
        parts = []
        if self.token_tracker.last_usage:
            parts.append(self.token_tracker.format_user_token_info())
        if self._tool_usage:
            parts.append(self._format_tool_stats())
        if parts:
            self.on_system_msg(" | ".join(parts))

    def _append_assistant(self, msg: AssistantMessage) -> None:
        """Добавляет сообщение ассистента в историю и рендерит его."""
        self.history.add(msg)
        self.on_render(msg)
        self._emit_token_info()

    def _append_tool_results(self, results: list[ToolResult]) -> None:
        """Добавляет результаты инструментов в историю и рендерит их."""
        self.history.extend(results)
        for tr in results:
            self.on_render(tr)
            self._emit_token_info()

    def _format_tool_stats(self) -> str:
        total = sum(self._tool_usage.values())
        items = " · ".join(f"{name} ×{count}" for name, count in sorted(self._tool_usage.items(), key=lambda x: -x[1]))
        return f"Tools: {items} ({total} total)"

    def _get_context_usage_percent(self) -> float:
        """Процент заполнения контекста по фактическому расходу из API (тот же
        источник, что и заголовок \"Tokens spent / Remaining\")."""
        total = self.token_tracker.get_total_context_tokens()
        return (total / self.token_tracker.max_context_tokens) * 100

    def _auto_summarize_dialogue(self) -> None:
        """Автоматическая суммаризация диалога при превышении порога контекста."""
        preserve_last = Config.AUTO_SUMMARY_PRESERVE_LAST
        total = len(self.history)

        if total <= Config.AFTER_SYSTEM_PROMPT + preserve_last:
            return

        end_id = total - 1 - preserve_last
        start_id = Config.AFTER_SYSTEM_PROMPT

        if start_id > end_id:
            return

        original_len = self.history.content_len(start_id, end_id)

        summary = summarize_dialogue(
            self, start_id=start_id, end_id=end_id,
        )

        if not summary or len(summary) >= original_len:
            return

        self.history.compress_old_messages(summary, preserve_last=preserve_last)
        self.on_system_msg(
            f"[AUTO-SUMMARY] Context compressed ({int(original_len / Config.CHARS_PER_TOKEN)} -> {int(len(summary) / Config.CHARS_PER_TOKEN)} tokens)"
        )

    def rebuild_tool_usage(self):
        self._tool_usage.clear()
        for msg in self.history.get_all():
            if isinstance(msg, ToolResult) and not msg.is_error and not msg.is_user_denied:
                self._tool_usage[msg.name] = self._tool_usage.get(msg.name, 0) + 1

    # --------------------------------------------------------
    # Подготовка сообщений (делегаты)
    # --------------------------------------------------------

    def _prepare_messages_for_api(self) -> list[dict]:
        return prepare_messages_for_api(self)

    def _get_effective_prefill(self, custom_prefill: Optional[str]) -> Optional[str]:
        return get_effective_prefill(custom_prefill)

    # --------------------------------------------------------
    # Выполнение инструментов
    # --------------------------------------------------------
    def _execute_tools(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        results = []
        history_before_current_turn = self.history.get_all()[:-1]

        for tc in tool_calls:
            name = tc.name
            self._tool_usage[name] = self._tool_usage.get(name, 0) + 1
            args_str = tc.arguments or "{}"

            if self.loop_detector.check_duplicate_in_turn(name, args_str, history_before_current_turn):
                warning_msg = (
                    f"{ENVIRONMENT_PREFIX} System rejected duplicate call of tool '{name}'. "
                    f"This tool was just called with the exact same parameters in the previous step. "
                    f"Do NOT call it again in the current moment even if user asked to. Try a different approach, use other parameters, "
                    f"or complete your response with the final answer."
                )
                self.on_system_msg(f"[LOOP PREVENTED] Blocked repeated call to '{name}' during execution.")
                results.append(ToolResult.error(tc.id, name, warning_msg))
                continue

            tool_info = self._all_tools.get(name)
            if not tool_info:
                results.append(ToolResult.error(tc.id, name, f"Unknown tool '{name}'. It must be loaded first or probably misspelled."))
                continue

            args_dict = None
            try:
                args_dict = json.loads(args_str) if args_str != "{}" else {}
            except Exception as e:
                results.append(ToolResult.error(tc.id, name, f"Invalid JSON: {e}"))
                continue

            if tool_info.get('requires_confirmation', False):
                skip_confirm = (name == "edit_file" and "path" in args_dict and self.is_path_trusted(args_dict["path"]))
                if not skip_confirm and not self.on_confirm(name, args_dict):
                    results.append(ToolResult.user_denied(tc.id, name))
                    continue

            try:
                handler = tool_info['handler']
                if tool_info.get('has_agent_param') or tool_info.get('is_instance_method'):
                    full_result = handler(self, **args_dict)
                else:
                    full_result = handler(**args_dict)
                content = str(full_result) if full_result is not None else "Tool executed successfully"
                tr = ToolResult.success(tc.id, name, content)

                auto_compress_tool_result(self, tr)
                results.append(tr)
            except Exception as e:
                self.on_system_msg(f"⚠️ [ERROR] Tool '{name}' FAILED: {e}")
                results.append(ToolResult.error(tc.id, name, str(e)))

        return results

    # --------------------------------------------------------
    # Обработка ответа LLM
    # --------------------------------------------------------
    @staticmethod
    def _build_assistant_msg(msg_obj, clean_content: str) -> AssistantMessage:
        tool_calls = []
        if msg_obj.tool_calls:
            for tc in msg_obj.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=_tc_name(tc),
                    arguments=_tc_args(tc),
                ))
        result = AssistantMessage(
            content=clean_content,
            tool_calls=tool_calls,
            reasoning_content=getattr(msg_obj, 'reasoning_content', ''),
            streamed=bool(getattr(msg_obj, 'streamed', False) or getattr(msg_obj, '_streamed', False)),
        )
        return result

    def _call_with_streaming(
        self,
        messages: list[dict],
        prefill: str = None,
        tools: list[dict] = None,
        previous_response_id: str = None,
        params: GenerationParams = None,
    ) -> tuple:
        """
        Вызов LLM с streaming для текста.
        Возвращает (message_obj, error, usage) как обычный LLMClient.call().
        """
        try:
            stream = LLMClient.stream(
                messages,
                tools=tools,
                prefill=prefill,
                previous_response_id=previous_response_id,
                params=params,
            )
            
            tool_calls_data: dict = {}

            stream_state = {
                "full_content": "",
                "full_reasoning": "",
                "usage": None,
                "reasoning_started": False,
            }

            first_chunk = next(stream)
            if isinstance(first_chunk, dict) and "error" in first_chunk:
                return None, first_chunk["error"], None

            if self.on_stream_start:
                self.on_stream_start()

            self._process_stream_chunk(first_chunk, tool_calls_data)
            self._apply_stream_chunk(first_chunk, stream_state)

            for chunk in stream:
                self._process_stream_chunk(chunk, tool_calls_data)
                self._apply_stream_chunk(chunk, stream_state)

            if self.on_stream_end:
                self.on_stream_end()
            if stream_state["reasoning_started"] and self.on_reasoning_end:
                self.on_reasoning_end()

            tool_calls = []
            if tool_calls_data:
                for tc_data in tool_calls_data.values():
                    tool_calls.append(ToolCall(
                        id=tc_data["id"],
                        name=tc_data["function"]["name"],
                        arguments=tc_data["function"]["arguments"]
                    ))

            message_obj = AssistantMessage(
                content=stream_state["full_content"],
                tool_calls=tool_calls,
                reasoning_content=stream_state["full_reasoning"],
                streamed=True,
            )

            message_obj.content = apply_prefill(message_obj.content, prefill)

            return message_obj, None, stream_state["usage"]
            
        except Exception as e:
            return None, str(e), None
    
    def _apply_stream_chunk(self, chunk, state: dict) -> None:
        """Применяет чанк стрима: usage, reasoning и текстовый контент."""
        usage = getattr(chunk, 'usage', None)
        if usage:
            state["usage"] = build_usage_dict(
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens,
            )
        if not getattr(chunk, 'choices', None):
            return
        delta = chunk.choices[0].delta
        rc = getattr(delta, 'reasoning_content', None)
        if rc:
            state["full_reasoning"] += rc
            if not state["reasoning_started"]:
                state["reasoning_started"] = True
                if self.on_reasoning_start:
                    self.on_reasoning_start()
            if self.on_reasoning_chunk:
                self.on_reasoning_chunk(rc)
        if delta.content:
            state["full_content"] += delta.content
            if self.on_stream_chunk:
                self.on_stream_chunk(delta.content)

    def _process_stream_chunk(self, chunk, tool_calls_data: dict):
        """Обработка чанка для сбора tool calls"""
        if not chunk.choices:
            return
            
        delta = chunk.choices[0].delta
        if not delta.tool_calls:
            return
            
        for tc in delta.tool_calls:
            idx = tc.index
            if idx not in tool_calls_data:
                tool_calls_data[idx] = {
                    "id": tc.id or "",
                    "type": "function",
                    "function": {
                        "name": tc.function.name if tc.function and tc.function.name else "",
                        "arguments": tc.function.arguments if tc.function and tc.function.arguments else ""
                    }
                }
            else:
                if tc.id:
                    tool_calls_data[idx]["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        tool_calls_data[idx]["function"]["name"] += tc.function.name
                    if tc.function.arguments:
                        tool_calls_data[idx]["function"]["arguments"] += tc.function.arguments
    
    def _process_llm_response(self, message_obj) -> tuple[str, bool]:
        if not message_obj:
            return "Empty response", True

        content = message_obj.content or ""
        clean_content = content.strip()
        assistant_msg = self._build_assistant_msg(message_obj, clean_content)

        if assistant_msg.has_tool_calls():
            valid_tc = None
            fallback_tc = assistant_msg.tool_calls[0]

            for tc in assistant_msg.tool_calls:
                if tc.name in self._all_tools:
                    args_str = tc.arguments or "{}"
                    try:
                        if args_str != "{}":
                            json.loads(args_str)
                        valid_tc = tc
                        break
                    except Exception:
                        pass

            chosen_tc = valid_tc if valid_tc else fallback_tc

            if len(assistant_msg.tool_calls) > 1:
                self.on_system_msg(f"[MULTIPLE TOOLS DETECTED] Kept only '{chosen_tc.name}', removed others.")
                assistant_msg.tool_calls = [chosen_tc]
                if message_obj.tool_calls:
                    message_obj.tool_calls = [tc for tc in message_obj.tool_calls if tc.id == chosen_tc.id]

        if not clean_content and not assistant_msg.has_tool_calls():
            self.on_system_msg("[EMPTY RESPONSE] Model returned no content. Discarding and retrying...")
            return clean_content, True

        self._append_assistant(assistant_msg)

        if not assistant_msg.has_tool_calls():
            return clean_content, False

        tool_results = self._execute_tools(assistant_msg.tool_calls)
        self._append_tool_results(tool_results)
        prune_all_failed_tool_calls_except_last(self)

        tool_error_occurred = any(tr.is_error and not tr.is_user_denied for tr in tool_results)
        return clean_content, tool_error_occurred

    # --------------------------------------------------------
    # Self-consistency
    # --------------------------------------------------------
    def _generate_draft_with_tool_suggestions(self, draft_messages, prefill, draft_temp):
        prefill_val = self._get_effective_prefill(prefill)
        params = self._gen_params.with_temp(draft_temp)
        for _ in range(3):
            msg_obj, err, _ = LLMClient.call(
                draft_messages,
                tools=self.tools if self.tools else None,
                prefill=prefill_val,
                params=params,
            )
            if msg_obj and not err:
                return msg_obj
        return None

    def _chat_self_consistent(self, message: str, prefill: str = None) -> str:
        user_message = UserMessage(content=message)
        self.history.add(user_message)
        messages_base = self._prepare_messages_for_api()

        self.on_system_msg(f"Generating {self.sc_samples} drafts...")
        drafts = []
        for _ in range(self.sc_samples):
            draft = self._generate_draft_with_tool_suggestions(messages_base, prefill, 0.7)
            if draft:
                drafts.append(draft)
        if not drafts:
            return "Failed to generate any valid draft"

        draft_texts = []
        for i, draft in enumerate(drafts, 1):
            content = draft.content or "(no text)"
            if draft.tool_calls:
                tc_names = [f"{_tc_name(tc)}(...)" for tc in draft.tool_calls]
                content += f"\n[Suggested tools: {', '.join(tc_names)}]"
            draft_texts.append(f"--- Draft {i} ---\n{content}")

        synthesis_prompt = (
            f"{ENVIRONMENT_PREFIX} Here are drafts from multiple reasoning paths:\n"
            + "\n".join(draft_texts)
            + "\n\n Analyse them and synthesize the finishing correct answer, paying attention to suggested tools. Output only the final synthesized answer."
        )
        synthesis_messages = messages_base + [{"role": "user", "content": synthesis_prompt}]
        current_prefill = self._get_effective_prefill(prefill)
        msg_obj, err, usage = LLMClient.call(
            synthesis_messages,
            tools=self.tools if self.tools else None,
            prefill=current_prefill,
            params=self._gen_params.with_temp(0.2),
        )
        if usage:
            self.token_tracker.update_from_usage(usage)
        if err or not msg_obj:
            error = f"⚠️ API Error during synthesis: {err}"
            self.on_system_msg(error)
            return error

        assistant_msg = self._build_assistant_msg(msg_obj, msg_obj.content)
        if not msg_obj.tool_calls:
            self._append_assistant(assistant_msg)
            return msg_obj.content

        tool_results = self._execute_tools(assistant_msg.tool_calls)
        self._append_assistant(assistant_msg)
        self._append_tool_results(tool_results)

        followup_dicts = (
            synthesis_messages
            + [assistant_msg.to_api_dict()]
            + [tr.to_api_dict() for tr in tool_results]
        )
        final_obj, final_err, final_usage = LLMClient.call(
            followup_dicts,
            tools=None,
            params=self._gen_params.with_temp(0.1),
        )
        if final_usage:
            self.token_tracker.update_from_usage(final_usage)
        if final_err or not final_obj:
            return msg_obj.content or "Tool executed successfully"

        final_content = final_obj.content.strip()
        final_assistant_msg = self._build_assistant_msg(final_obj, final_content)
        self._append_assistant(final_assistant_msg)
        return final_content

    # --------------------------------------------------------
    # Главный цикл
    # --------------------------------------------------------
    def chat(self, message: str, max_iter: int = None, prefill: str = None):
        max_iter = max_iter if max_iter is not None else Config.MAX_ITER
        if self.self_consistency_mode:
            return self._chat_self_consistent(message, prefill)

        user_msg = UserMessage(content=message)
        self.history.add(user_msg)
        current_prefill = self._get_effective_prefill(prefill)
        self._last_response_id = None
        self._last_sent_msg_count = 0

        consecutive_errors = 0
        MAX_CONSECUTIVE_ERRORS = 5

        for i in range(max_iter):
            step_prefill = current_prefill if i == 0 else None
            all_messages = self._prepare_messages_for_api()

            if i == 0 or self._last_response_id is None:
                messages_to_send = all_messages
                prev_response_id = None
            else:
                messages_to_send = all_messages[self._last_sent_msg_count:]
                prev_response_id = self._last_response_id

            max_generation_attempts = self._max_generation_attempts if self._max_generation_attempts is not None else 2
            message_obj = None
            last_duplicate_info = None
            api_error_occurred = False

            for attempt in range(max_generation_attempts):
                attempt_params = self._gen_params
                if self._temp_boost_active:
                    attempt_params = self._gen_params.with_temp(Config.BOOST_TEMP)
                    self._temp_boost_active = False

                active_messages = [dict(msg) for msg in messages_to_send]

                if last_duplicate_info:
                    dup_name, dup_args = last_duplicate_info
                    warning_text = (
                        f"\n\n{ENVIRONMENT_PREFIX} Your previous attempt to call tool '{dup_name}' "
                        f"with arguments '{dup_args}' was blocked because it is a duplicate of a call already made "
                        f"in this turn. Do NOT call '{dup_name}' again with the same parameters. "
                        f"Use other parameters, call a different tool, or provide your final response."
                    )
                    if active_messages:
                        last_msg = active_messages[-1]
                        last_msg["content"] = (last_msg.get("content") or "") + warning_text

                # Streaming или обычный режим
                if self.streaming_enabled and self.on_stream_chunk:
                    message_obj, err, usage = self._call_with_streaming(
                        active_messages,
                        step_prefill,
                        tools=self.tools if self.tools else None,
                        previous_response_id=prev_response_id,
                        params=attempt_params,
                    )
                else:
                    message_obj, err, usage = LLMClient.call(
                        active_messages,
                        tools=self.tools if self.tools else None,
                        prefill=step_prefill,
                        previous_response_id=prev_response_id,
                        params=attempt_params,
                    )
                
                if usage:
                    self.token_tracker.update_from_usage(usage)
                if err:
                    self.on_system_msg(f"[API Error] {err}")
                    api_error_occurred = True
                    break

                if not message_obj:
                    api_error_occurred = True
                    break

                if hasattr(message_obj, '_response_id'):
                    self._last_response_id = message_obj._response_id
                self._last_sent_msg_count = len(all_messages)

                if message_obj.tool_calls:
                    has_duplicate = False
                    current_history = self.history.get_all()
                    for tc in message_obj.tool_calls:
                        if self.loop_detector.check_duplicate_in_turn(_tc_name(tc), _tc_args(tc), current_history):
                            has_duplicate = True
                            last_duplicate_info = (_tc_name(tc), _tc_args(tc))
                            self.on_system_msg(
                                f"[PROACTIVE LOOP DETECTED] Intercepted duplicate call to '{_tc_name(tc)}'. "
                                f"Discarding response. Activating temperature boost ({Config.BOOST_TEMP}) "
                                f"and injecting temporary warning. Attempt {attempt + 1}/{max_generation_attempts}."
                            )
                            break

                    if has_duplicate:
                        self._temp_boost_active = True
                        continue

                break
            else:
                self.on_system_msg(
                    "[PROACTIVE LOOP DETECTOR] Max re-generation attempts reached. Proceeding to execution safety nets."
                )

            if api_error_occurred or not message_obj:
                self.history.normalize(is_error_recovery=True)
                self.on_system_msg("⚠️ [RECOVERY] API error occurred. Role sequence restored. Handing control to user.")
                return ""

            result_text, tool_error_occurred = self._process_llm_response(message_obj)

            if self._get_context_usage_percent() >= Config.AUTO_SUMMARY_THRESHOLD:
                self._auto_summarize_dialogue()

            if tool_error_occurred:
                consecutive_errors += 1
            else:
                consecutive_errors = 0

            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                self.history.normalize(is_error_recovery=True)
                self.on_system_msg(
                    f"⚠️ [LIMIT REACHED] {MAX_CONSECUTIVE_ERRORS} consecutive tool errors. Handing control to user."
                )
                return ""

            if not message_obj.tool_calls and not tool_error_occurred:
                return result_text
