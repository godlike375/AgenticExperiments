from __future__ import annotations

import json
from typing import Union, Callable, Optional

from universal_agents.tool import ENVIRONMENT_PREFIX
from config import Config
from models import UserMessage, AssistantMessage, ToolCall, ToolResult
from llm_client import LLMClient, TokenUsageTracker, LoopDetector
from history import ChatHistory

from compressors import auto_compress_tool_result
from context_builder import prepare_messages_for_api, get_effective_prefill
from history_repair import prune_all_failed_tool_calls_except_last


class LLMAgent:
    def __init__(
        self,
        system_prompt: str = "You are a helpful assistant",
        temp: float = 0.3,
        timeout: int = 1800,
        tools_config: Union[list[str], dict, None] = None,
        on_render: Callable = lambda x: None,
        on_confirm: Callable[[str, dict], bool] = lambda n, a: True,
        on_system_msg: Callable[[str], None] = lambda x: None,
        external_plugins: dict[str, Callable] = None,
        max_context_tokens: int = 16384,
        _create_judge: bool = True,
    ):
        self.history = ChatHistory(system_prompt)
        self.temp = temp
        self.timeout = timeout
        self.on_render = on_render
        self.on_confirm = on_confirm
        self.on_system_msg = on_system_msg
        self.self_consistency_mode = False
        self.sc_samples = 3
        self.token_tracker = TokenUsageTracker(system_prompt, max_context_tokens)
        self._all_tools = {}
        if external_plugins:
            for name, func in external_plugins.items():
                self._all_tools[name] = self._build_tool_dict(func, is_instance_method=False)
        self._filter_tools(tools_config)
        self.loop_detector = LoopDetector()
        self._temp_boost_active = False
        self._original_temp = temp

    # --------------------------------------------------------
    # Фильтрация инструментов
    # --------------------------------------------------------
    def _filter_tools(self, config):
        all_names = set(self._all_tools.keys())
        if config is None:
            active = all_names
        elif isinstance(config, list):
            active = set(config) & all_names
        elif isinstance(config, dict) and "exclude" in config:
            active = all_names - set(config["exclude"])
        else:
            raise ValueError("Invalid tools_config")
        self._all_tools = {k: v for k, v in self._all_tools.items() if k in active}
        self.tools = [v['schema'] for v in self._all_tools.values()]

    # --------------------------------------------------------
    # Подготовка сообщений (делегаты)
    # --------------------------------------------------------

    @staticmethod
    def _build_tool_dict(func: Callable, is_instance_method: bool) -> dict:
        return {
            "schema": func._tool_schema,
            "handler": func,
            "is_instance_method": is_instance_method,
            "requires_confirmation": getattr(func, '_requires_confirmation', False)
        }

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
                results.append(ToolResult.error(tc.id, name, f"Unknown tool '{name}'"))
                continue

            args_dict = None
            try:
                args_dict = json.loads(args_str) if args_str != "{}" else {}
            except Exception as e:
                results.append(ToolResult.error(tc.id, name, f"Invalid JSON: {e}"))
                continue

            if tool_info.get('requires_confirmation', False):
                if not self.on_confirm(name, args_dict):
                    results.append(ToolResult.user_denied(tc.id, name))
                    continue

            try:
                handler = tool_info['handler']
                if tool_info['is_instance_method']:
                    full_result = handler(self, **args_dict)
                else:
                    full_result = handler(**args_dict) if 'agent' not in handler.__code__.co_varnames[:1] else handler(self, **args_dict)
                content = str(full_result) if full_result is not None else "Tool executed successfully"
                tr = ToolResult.success(tc.id, name, content)

                auto_compress_tool_result(self, tr)
                results.append(tr)
            except Exception as e:
                self.on_system_msg(f"[ERROR] Tool '{name}' FAILED: {e}")
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
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                ))
        return AssistantMessage(content=clean_content, tool_calls=tool_calls)

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
                if hasattr(message_obj, 'tool_calls') and message_obj.tool_calls:
                    message_obj.tool_calls = [tc for tc in message_obj.tool_calls if tc.id == chosen_tc.id]

        if not clean_content and not assistant_msg.has_tool_calls():
            self.on_system_msg("[EMPTY RESPONSE] Model returned no content. Discarding and retrying with temp boost...")
            self._temp_boost_active = True
            return clean_content, True

        self.history.add(assistant_msg)
        self.on_render(assistant_msg)

        if not assistant_msg.has_tool_calls():
            return clean_content, False

        tool_results = self._execute_tools(assistant_msg.tool_calls)
        self.history.extend(tool_results)
        for tr in tool_results:
            self.on_render(tr)
        prune_all_failed_tool_calls_except_last(self)

        tool_error_occurred = any(tr.is_error and not tr.is_user_denied for tr in tool_results)
        return clean_content, tool_error_occurred

    # --------------------------------------------------------
    # Self-consistency
    # --------------------------------------------------------
    def _generate_draft_with_tool_suggestions(self, draft_messages, prefill, draft_temp, draft_timeout):
        prefill_val = self._get_effective_prefill(prefill)
        for _ in range(3):
            msg_obj, err, _ = LLMClient.call(
                draft_messages, draft_temp, draft_timeout,
                tools=self.tools if self.tools else None,
                prefill=prefill_val,
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
            draft = self._generate_draft_with_tool_suggestions(messages_base, prefill, 0.7, self.timeout)
            if draft:
                drafts.append(draft)
        if not drafts:
            return "Failed to generate any valid draft"

        draft_texts = []
        for i, draft in enumerate(drafts, 1):
            content = draft.content or "(no text)"
            if draft.tool_calls:
                tc_names = [f"{tc.function.name}(...)" for tc in draft.tool_calls]
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
            synthesis_messages, temp=0.2, timeout=self.timeout,
            tools=self.tools if self.tools else None,
            prefill=current_prefill,
        )
        if usage:
            self.token_tracker.update_from_usage(usage)
        if err or not msg_obj:
            error = f"API Error during synthesis: {err}"
            self.on_system_msg(error)
            return error

        assistant_msg = self._build_assistant_msg(msg_obj, msg_obj.content)
        if not msg_obj.tool_calls:
            self.history.add(assistant_msg)
            self.on_render(assistant_msg)
            return msg_obj.content

        tool_results = self._execute_tools(assistant_msg.tool_calls)
        self.history.add(assistant_msg)
        self.on_render(assistant_msg)
        self.history.extend(tool_results)
        for tr in tool_results:
            self.on_render(tr)

        followup_dicts = (
            synthesis_messages
            + [assistant_msg.to_api_dict()]
            + [tr.to_api_dict() for tr in tool_results]
        )
        final_obj, final_err, final_usage = LLMClient.call(
            followup_dicts, temp=0.1, timeout=self.timeout, tools=None,
        )
        if final_usage:
            self.token_tracker.update_from_usage(final_usage)
        if final_err or not final_obj:
            return msg_obj.content or "Tool executed successfully"

        final_content = final_obj.content.strip()
        final_assistant_msg = self._build_assistant_msg(final_obj, final_content)
        self.history.add(final_assistant_msg)
        self.on_render(final_assistant_msg)
        return final_content

    # --------------------------------------------------------
    # Главный цикл
    # --------------------------------------------------------
    def chat(self, message: str, max_iter: int = 5, prefill: str = None):
        if self.self_consistency_mode:
            return self._chat_self_consistent(message, prefill)

        user_msg = UserMessage(content=message)
        self.history.add(user_msg)
        current_prefill = self._get_effective_prefill(prefill)

        consecutive_errors = 0
        MAX_CONSECUTIVE_ERRORS = 5

        for i in range(max_iter):
            step_prefill = current_prefill if i == 0 else None
            messages_to_send = self._prepare_messages_for_api()

            max_generation_attempts = 3
            message_obj = None
            last_duplicate_info = None
            api_error_occurred = False

            for attempt in range(max_generation_attempts):
                current_temp = self.temp
                if self._temp_boost_active:
                    current_temp = Config.BOOST_TEMP
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

                message_obj, err, usage = LLMClient.call(
                    active_messages, current_temp, self.timeout,
                    tools=self.tools if self.tools else None,
                    prefill=step_prefill,
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

                if message_obj.tool_calls:
                    has_duplicate = False
                    current_history = self.history.get_all()
                    for tc in message_obj.tool_calls:
                        tc_name = tc.function.name
                        tc_args = tc.function.arguments
                        if self.loop_detector.check_duplicate_in_turn(tc_name, tc_args, current_history):
                            has_duplicate = True
                            last_duplicate_info = (tc_name, tc_args)
                            self.on_system_msg(
                                f"[PROACTIVE LOOP DETECTED] Intercepted duplicate call to '{tc_name}'. "
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
                self.on_system_msg("[RECOVERY] API error occurred. Role sequence restored. Handing control to user.")
                return

            result_text, tool_error_occurred = self._process_llm_response(message_obj)

            if tool_error_occurred:
                consecutive_errors += 1
            else:
                consecutive_errors = 0

            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                self.history.normalize(is_error_recovery=True)
                self.on_system_msg(
                    f"[LIMIT REACHED] {MAX_CONSECUTIVE_ERRORS} consecutive tool errors. Handing control to user."
                )
                return

            if not message_obj.tool_calls and not tool_error_occurred:
                return
