import json
import os
import sys
import importlib.util
import inspect
from typing import Union, Callable, Optional
from universal_agents.tool import tool, ENVIRONMENT_PREFIX
from config import Config
from models import UserMessage, AssistantMessage, ToolCall, ToolResult, Message, SystemMessage
from llm_client import LLMClient, TokenUsageTracker, LoopDetector
from history import ChatHistory

def load_external_plugins(plugins_dir="tools"):
    external_tools = {}
    if not os.path.exists(plugins_dir):
        return external_tools
    root_path = os.path.abspath(os.path.join(plugins_dir, ".."))
    if root_path not in sys.path:
        sys.path.insert(0, root_path)
    for filename in os.listdir(plugins_dir):
        if not filename.endswith(".py") or filename.startswith("__"):
            continue
        module_name = filename[:-3]
        file_path = os.path.join(plugins_dir, filename)
        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for name, obj in inspect.getmembers(module):
                if callable(obj) and hasattr(obj, '_is_tool'):
                    if name in external_tools:
                        print(f"Warning: Duplicate tool name '{name}' in {filename}")
                        continue
                    external_tools[name] = obj
        except Exception as e:
            print(f"Error loading plugin {filename}: {e}")
    return external_tools

class LLMAgent:
    def __init__(
            self,
            system_prompt: str = "You are a helpful assistant",
            temp: float = 0.05,
            timeout: int = 1800,
            tools_config: Union[list[str], dict, None] = None,
            on_render: Callable = lambda x: None,
            on_confirm: Callable[[str, dict], bool] = lambda n, a: True,
            on_system_msg: Callable[[str], None] = lambda x: None,
            external_plugins: dict[str, Callable] = None,
            max_context_tokens: int = 16384
    ):
        self.history = ChatHistory(system_prompt)
        self.temp = temp
        self.timeout = timeout
        self.on_render = on_render
        self.on_confirm = on_confirm
        self.on_system_msg = on_system_msg
        self.self_consistency_mode = False
        self.sc_samples = 3
        self.token_tracker = TokenUsageTracker(max_context_tokens)

        internal_tools = self._collect_internal_tools([self.__class__])
        if external_plugins:
            for name, func in external_plugins.items():
                if name in internal_tools:
                    print(f"Warning: External tool '{name}' conflicts with internal tool. Skipping.")
                    continue
                internal_tools[name] = self._build_tool_dict(func, is_instance_method=False)
        self._all_tools = internal_tools
        self._filter_tools(tools_config)
        self.loop_detector = LoopDetector(threshold=2)
        self._temp_boost_active = False
        self._original_temp = temp

    @staticmethod
    def _build_tool_dict(func: Callable, is_instance_method: bool) -> dict:
        return {
            "schema": func._tool_schema,
            "handler": func,
            "is_instance_method": is_instance_method,
            "requires_confirmation": getattr(func, '_requires_confirmation', False)
        }

    def _get_effective_prefill(self, custom_prefill: Optional[str]) -> Optional[str]:
        if custom_prefill:
            return custom_prefill
        return None

    @staticmethod
    def _build_assistant_msg(msg_obj, clean_content: str) -> AssistantMessage:
        tool_calls = []
        if msg_obj.tool_calls:
            for tc in msg_obj.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments
                ))
        return AssistantMessage(content=clean_content, tool_calls=tool_calls)

    def _break_tool_loop(self, tool_name: str, norm_args: str, count: int):
        messages = self.history._messages
        matched_indices = []
        duplicate_tool_ids = set()
        for i in range(Config.AFTER_SYSTEM_PROMPT, len(messages)):
            msg = messages[i]
            if not isinstance(msg, AssistantMessage):
                continue
            for tc in msg.tool_calls:
                if (tc.name == tool_name and
                        self.loop_detector.normalize_args(tc.arguments) == norm_args):
                    matched_indices.append(i)
                    if len(matched_indices) > 1:
                        duplicate_tool_ids.add(tc.id)
        if len(matched_indices) <= 1:
            return
        to_remove = set(matched_indices[1:])
        for i, msg in enumerate(messages):
            if isinstance(msg, ToolResult) and msg.tool_call_id in duplicate_tool_ids:
                to_remove.add(i)
        for idx in sorted(to_remove, reverse=True):
            del messages[idx]
        warning = f"{ENVIRONMENT_PREFIX} Tool '{tool_name}' with identical args was called a few times. The system removed all duplicates."
        messages.append(UserMessage(warning))
        self.on_system_msg(f"[LOOP DETECTED] '{tool_name}' ×{count}")

    # ---------- Tools ----------
    @tool(description="Get short indexed current history with ids")
    def get_msg_ids(self, chars_per_message: int = 35):
        history = self.history
        if len(history) <= Config.AFTER_SYSTEM_PROMPT:
            return f"{ENVIRONMENT_PREFIX} История пока пустая."
        lines = ["=== SHORT DIALOG ==="]
        for i in range(Config.AFTER_SYSTEM_PROMPT, len(history)):
            msg = history[i]
            if isinstance(msg, SystemMessage):
                continue
            elif isinstance(msg, UserMessage):
                role = "USER"
                content = msg.content
            elif isinstance(msg, AssistantMessage):
                role = "ASSISTANT"
                content = msg.content
                if msg.has_tool_calls():
                    tc_info = ", ".join(tc.name for tc in msg.tool_calls)
                    content += f" [Tools: {tc_info}]"
            elif isinstance(msg, ToolResult):
                role = "TOOL"
                prefix = f"[{msg.name}] "
                content = prefix + msg.content
                if msg.is_error:
                    content += " ❌"
                elif msg.is_user_denied:
                    content += " 🚫"
            else:
                continue
            if len(content) > chars_per_message:
                content = content[:chars_per_message] + " ..."
            lines.append(f"[id {i}] {role}: {content.strip()}")
        return f"{ENVIRONMENT_PREFIX} Your current history:\n" + "\n".join(lines)

    @tool(description="Edits a specific message in the history",
          requires_confirmation=True,
          id=("int", "ID of the message to edit"),
          old=("str", "Optional exact substr to replace. Empty str replaces whole text"),
          new=("str", "Text to insert in place of old"))
    def edit_message(self, id: int, new: str, old: str = ''):
        return self.history.edit_message(id, new, old)

    @tool(description="Deletes a range of messages from dialog history",
          requires_confirmation=True,
          start_id=("int", "Starting message ID to delete"),
          end_id=("int", "Optional ending message ID (-1 for last)"))
    def delete_messages(self, start_id: int, end_id: int = -1):
        return self.history.delete_range(start_id, end_id)

    def _collect_internal_tools(self, classes):
        tools = {}
        for klass in classes:
            for name in dir(klass):
                raw = klass.__dict__.get(name)
                if raw is None:
                    continue
                is_instance_method = callable(raw) and not isinstance(raw, (staticmethod, classmethod, type))
                func = raw.__func__ if isinstance(raw, staticmethod) else raw
                if hasattr(func, '_is_tool'):
                    tools[func._tool_name] = self._build_tool_dict(func, is_instance_method)
        return tools

    def _generate_draft_with_tool_suggestions(self, draft_messages, prefill, draft_temp, draft_timeout):
        prefill_val = self._get_effective_prefill(prefill)
        for _ in range(3):
            msg_obj, err, _ = LLMClient.call(
                draft_messages, draft_temp, draft_timeout,
                tools=self.tools if self.tools else None,
                prefill=prefill_val
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
                f"{ENVIRONMENT_PREFIX} Here are drafts from multiple reasoning paths:\n" +
                "\n".join(draft_texts) +
                "\n\n Analyse them and synthesize the finishing correct answer, paying attention to suggested tools."
        )
        synthesis_messages = messages_base + [{"role": "user", "content": synthesis_prompt}]
        current_prefill = self._get_effective_prefill(prefill)
        msg_obj, err, usage = LLMClient.call(
            synthesis_messages, temp=0.2, timeout=self.timeout,
            tools=self.tools if self.tools else None,
            prefill=current_prefill
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
                synthesis_messages +
                [assistant_msg.to_api_dict()] +
                [tr.to_api_dict() for tr in tool_results]
        )
        final_obj, final_err, final_usage = LLMClient.call(
            followup_dicts,
            temp=0.1,
            timeout=self.timeout,
            tools=None
        )
        if final_usage:
            self.token_tracker.update_from_usage(final_usage)
        if final_err or not final_obj:
            return clean_content or "Tool executed successfully"
        final_content = final_obj.content.strip()
        final_assistant_msg = self._build_assistant_msg(final_obj, final_content)
        self.history.add(final_assistant_msg)
        self.on_render(final_assistant_msg)
        return final_content

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

    def _prepare_messages_for_api(self) -> list[dict]:
        self.history.normalize()
        api_messages = []

        last_user_idx = None
        last_user_msg = None
        for i in range(len(self.history) - 1, -1, -1):
            if isinstance(self.history[i], UserMessage):
                last_user_idx = i
                last_user_msg = self.history[i]
                break

        for i, msg in enumerate(self.history):
            if isinstance(msg, SystemMessage):
                api_messages.append(msg.to_api_dict())
            elif isinstance(msg, UserMessage):
                header = self.token_tracker.format_timestamp_header(msg)
                if i == last_user_idx and last_user_msg:
                    header += self.token_tracker.format_token_header(self.history[0].content, last_user_msg.content)
                header += self.token_tracker.format_closing_header()
                api_messages.append({
                    "role": "user",
                    "content": header + msg.content
                })
            elif isinstance(msg, AssistantMessage):
                api_messages.append(msg.to_api_dict())
            elif isinstance(msg, ToolResult):
                api_messages.append(msg.to_api_dict())
        return api_messages

    def _execute_tools(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        results = []
        for tc in tool_calls:
            name = tc.name
            args_str = tc.arguments or "{}"
            self.loop_detector.add_call(name, args_str)
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
                    full_result = handler(**args_dict)
                content = str(full_result) if full_result is not None else "Tool executed successfully"
                results.append(ToolResult.success(tc.id, name, content))
            except Exception as e:
                self.on_system_msg(f"[ERROR] Tool '{name}' FAILED: {e}")
                results.append(ToolResult.error(tc.id, name, str(e)))
        loop_info = self.loop_detector.detect_loop()
        if loop_info:
            tool_name, norm_args, count = loop_info
            self._break_tool_loop(tool_name, norm_args, count)
            self._temp_boost_active = True
        return results

    def _process_llm_response(self, message_obj) -> str:
        if not message_obj:
            return "Empty response"
        content = message_obj.content or ""
        clean_content = content.replace("</think>", "").strip()
        assistant_msg = self._build_assistant_msg(message_obj, clean_content)
        self.history.add(assistant_msg)
        self.on_render(assistant_msg)
        if not message_obj.tool_calls:
            return clean_content
        tool_results = self._execute_tools(assistant_msg.tool_calls)
        self.history.extend(tool_results)
        for tr in tool_results:
            self.on_render(tr)
        self._prune_all_failed_tool_calls_except_last()
        return clean_content

    def _prune_all_failed_tool_calls_except_last(self):
        if len(self.history) <= Config.AFTER_SYSTEM_PROMPT + 1:
            return
        last_assistant_idx = -1
        for i in range(len(self.history) - 1, Config.AFTER_SYSTEM_PROMPT - 1, -1):
            if isinstance(self.history[i], AssistantMessage):
                last_assistant_idx = i
                break
        if last_assistant_idx == -1:
            return
        indices_to_remove = set()
        i = Config.AFTER_SYSTEM_PROMPT
        while i < len(self.history):
            msg = self.history[i]
            if (isinstance(msg, AssistantMessage) and
                    msg.has_tool_calls()):
                if (i + 1 < len(self.history) and
                        isinstance(self.history[i + 1], ToolResult)):
                    tool_result = self.history[i + 1]
                    if tool_result.is_error and not tool_result.is_user_denied:
                        if i < last_assistant_idx:
                            indices_to_remove.add(i)
                            indices_to_remove.add(i + 1)
                            i += 2
                            continue
            i += 1
        if indices_to_remove:
            for idx in sorted(indices_to_remove, reverse=True):
                del self.history._messages[idx]
            self.on_system_msg(
                f"[CLEANUP] Removed {len(indices_to_remove)} messages "
                f"({len(indices_to_remove)//2} failed calls)"
            )
            self.history.normalize()

    def chat(self, message: str, max_iter: int = 5, prefill: str = None) -> str:
        if self.self_consistency_mode:
            return self._chat_self_consistent(message, prefill)
        user_msg = UserMessage(content=message)
        self.history.add(user_msg)
        current_prefill = self._get_effective_prefill(prefill)
        for i in range(max_iter):
            step_prefill = current_prefill if i == 0 else None
            messages_to_send = self._prepare_messages_for_api()
            current_temp = self.temp
            if self._temp_boost_active:
                current_temp = Config.BOOST_TEMP
                self._temp_boost_active = False
            message_obj, err, usage = LLMClient.call(
                messages_to_send, current_temp, self.timeout,
                tools=self.tools if self.tools else None,
                prefill=step_prefill
            )
            if usage:
                self.token_tracker.update_from_usage(usage)
            if err:
                error = f"[API Error] {err}"
                self.on_system_msg(error)
                return error
            result_text = self._process_llm_response(message_obj)
            if not message_obj.tool_calls:
                return result_text
        return "Max iterations reached without final answer"
