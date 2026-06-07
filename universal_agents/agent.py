import json
import os
import sys
import importlib.util
import inspect
from typing import Union, Callable, Optional
from universal_agents.tool import tool, ENVIRONMENT_PREFIX
from config import Config
from models import UserMessage, AssistantMessage, ToolCall, ToolResult, SystemMessage
from llm_client import LLMClient, TokenUsageTracker, LoopDetector
from history import ChatHistory
from sub_agent import SubAgent


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
            temp: float = 0.15,
            timeout: int = 1800,
            tools_config: Union[list[str], dict, None] = None,
            on_render: Callable = lambda x: None,
            on_confirm: Callable[[str, dict], bool] = lambda n, a: True,
            on_system_msg: Callable[[str], None] = lambda x: None,
            external_plugins: dict[str, Callable] = None,
            max_context_tokens: int = 16384,
            _create_judge: bool = True
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

        internal_tools = self._collect_internal_tools([self.__class__])
        if external_plugins:
            for name, func in external_plugins.items():
                if name in internal_tools:
                    print(f"Warning: External tool '{name}' conflicts with internal tool. Skipping.")
                    continue
                internal_tools[name] = self._build_tool_dict(func, is_instance_method=False)
        self._all_tools = internal_tools
        self._filter_tools(tools_config)
        self.loop_detector = LoopDetector()
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

    # === Tools ===

    @tool(description="Get short indexed current history with ids")
    def get_messages(self, chars_per_message: int = 30):
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
                role = "AI"
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
            lines.append(f"{i}. {role}: {content.strip()}")
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

    @tool(
        description="Summarizes a range of dialog messages into a single concise UserMessage. "
                    "Use to free context tokens. Cannot summarize system prompt. ",
        requires_confirmation=True,
        start_id=("int", "Start index of messages to summarize"),
        end_id=("int", "End index (inclusive). Use -1 for last message")
    )
    def summarize_messages(self, start_id: int, end_id: int = -1) -> str:
        history = self.history
        if end_id == -1 or end_id >= len(history):
            end_id = len(history) - 3

        safe_start = max(start_id, Config.AFTER_SYSTEM_PROMPT)
        safe_end = min(end_id, len(history) - 3)

        if safe_start > safe_end:
            return (f"{ENVIRONMENT_PREFIX} Cannot summarize: range [{start_id}:{end_id}] "
                    f"is invalid or overlaps with protected last 2 messages.")

        lines = []
        for i in range(safe_start, safe_end + 1):
            msg = history[i]
            role = type(msg).__name__.replace("Message", "").upper()
            role = role if role!='ASSISTANT' else 'AI'
            content = getattr(msg, 'content', str(msg))
            lines.append(f"{role}: {content}")

        raw_text = "\n".join(lines)
        summary = self._summarize_text(
            raw_text
        )
        if not summary:
            return f"{ENVIRONMENT_PREFIX} Summarization failed (empty response or error)."

        summary_content = (f"{ENVIRONMENT_PREFIX} [SUMMARY of messages {safe_start}-{safe_end}]: "
                           f"{summary}")
        del history._messages[safe_start:safe_end + 1]
        history._messages.insert(safe_start, UserMessage(content=summary_content))
        history.normalize()

        freed = len(raw_text) - len(summary_content)
        return (f"{ENVIRONMENT_PREFIX} Successfully summarized "
                f"{safe_end - safe_start + 1} messages into 1. Freed ~{freed} chars.")

    @tool(
        description="Delegates a task to a limited sub-agent that has its own dialog history and just read-only tools unlike you. "
                    "You can delegate to it, for example, 1 step of a multi-step task. "
                    "Include necessary context for execution in task description. "
                    "The tool returns only the final result of a task.",
        task=("str", "Clear task description with all necessary context"),
        max_iter=("int", "Optional max tool calls for sub-agent (default 5)")
    )
    def delegate_to_subagent(self, task: str, max_iter: int = 5) -> str:
        # Сбор всех зарегистрированных обработчиков инструментов главного агента
        sub_plugins = {}
        for name, tool_info in self._all_tools.items():
            # Исключаем сам инструмент делегирования для предотвращения неконтролируемой рекурсии
            if name == "delegate_to_subagent":
                continue
            sub_plugins[name] = tool_info["handler"]

        sub = SubAgent(
            system_prompt=(
                "You are a sub-agent working on a specific subtask. "
                "You have access to read-only tools. "
                "Complete the task using tools if needed and provide a final answer. "
                "Do NOT ever ask clarifying questions — work with what you have."
            ),
            max_context_tokens=self.token_tracker.max_context_tokens // 3,
            tools_config={"exclude": [
                "edit_message", "delete_messages", "summarize_messages",
                "delegate_to_subagent"
            ]},
            external_plugins=sub_plugins, # Пробрасываем собранные инструменты
            safe_only=True,               # Автоматически отсекает инструменты с requires_confirmation=True
            max_iter=max_iter,
            temp=0.1,
            on_log=self.on_system_msg,
        )

        self.on_system_msg(f"[DELEGATE] Starting sub-agent for: {task[:100]}...")
        result = sub.run(task)
        self.on_system_msg(f"[DELEGATE] Completed. Tokens spent by sub-agent: {sub.tokens_spent}")

        if not result.strip():
            return f"{ENVIRONMENT_PREFIX} Sub-agent returned empty result."
        return f"{ENVIRONMENT_PREFIX} Sub-agent result:\n{result}"

    # === Internal ===

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
            return msg_obj.content or "Tool executed successfully"
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
                self.on_system_msg(
                    f"[LOOP PREVENTED] Blocked repeated call to '{name}' during execution."
                )
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
                    full_result = handler(**args_dict)
                content = str(full_result) if full_result is not None else "Tool executed successfully"
                tr = ToolResult.success(tc.id, name, content)

                self._auto_compress_tool_result(tr)

                results.append(tr)
            except Exception as e:
                self.on_system_msg(f"[ERROR] Tool '{name}' FAILED: {e}")
                results.append(ToolResult.error(tc.id, name, str(e)))

        return results

    # === Auto-compression ===

    def _summarize_text(self, text: str) -> Optional[str]:
        """Универсальная LLM-суммаризация произвольного текста."""
        prompt = (
            f"{ENVIRONMENT_PREFIX} Summarize the following text. "
            f"Preserve all key facts, specific data, decisions, etc. "
            f"Remove temporal reasoning and other things that aren't important anymore."
            f"Output ONLY the concise summary text.\n\n"
            f"The original text:\n```\n{text}\n```\n "
            f"*Use the following text's language!* "
            f"If you see the dialog of AI and User, treat it as your own dialog with User!"
        )
        msgs = [{"role": "user", "content": prompt}]
        msg_obj, err, usage = LLMClient.call(msgs, temp=0.1, timeout=60, tools=None)
        if usage:
            self.token_tracker.update_from_usage(usage)
        if err or not msg_obj or not msg_obj.content:
            return None
        return msg_obj.content.strip()

    def _synthesize_task_goal(self, tool_name: str) -> str:
        """
        Анализирует всю историю диалога через LLM и формулирует точную цель
        для анализа вывода конкретного инструмента.
        """
        self.on_system_msg(f"[GOAL SYNTHESIS] Analyzing conversation history to formulate goal for '{tool_name}'...")

        # Получаем подготовленные сообщения для API (включая системный промпт)
        messages_base = self._prepare_messages_for_api()[:-1]

        # Инструкция для мета-анализа истории
        synthesis_prompt = (
            f"{ENVIRONMENT_PREFIX}\n"
            f"Based on the current conversation context above create a tip for a sub-agent who will parse the output "
            f"of tool '{tool_name}' and summarize it for you because tool output is too long for your memory. "
            f"You only will read the summarization of the sub-agent so you need to note specific things that sub-agent "
            f"must pay attention to.\n"
            f"It must be a clear relatively concise instruction.\n"
            f"Output ONLY the formulated instruction for sub-agent."
        )

        # Добавляем инструкцию к контексту
        synthesis_messages = messages_base + [{"role": "user", "content": synthesis_prompt}]

        # Делаем быстрый запрос с низкой температурой для точности
        msg_obj, err, usage = LLMClient.call(
            synthesis_messages,
            temp=self.temp,
            timeout=self.timeout,
            tools=None
        )

        if usage:
            self.token_tracker.update_from_usage(usage)

        if err or not msg_obj or not msg_obj.content:
            # Резервный вариант при сбое (откат к последней реплике пользователя)
            self.on_system_msg("[GOAL SYNTHESIS] Failed to synthesize goal via LLM. Falling back to last user message.")
            for msg in reversed(self.history.get_all()):
                if isinstance(msg, UserMessage) and not msg.content.startswith(ENVIRONMENT_PREFIX):
                    return msg.content
            return "Extract any useful facts and errors relevant to the general task."

        synthesized_goal = msg_obj.content.strip()
        self.on_system_msg(f"[GOAL SYNTHESIS] Synthesized objective: \"{synthesized_goal}\"")
        return synthesized_goal

    def _auto_compress_tool_result(self, tool_result: ToolResult):
        """
        Автоматически сжимает вывод инструмента перед добавлением в историю,
        если он длинный, используя порционный анализ и динамический синтез цели.
        """
        if tool_result.is_error or tool_result.is_user_denied:
            return

        remaining = self.token_tracker.get_remaining(self.history.get_last_user_message().content)

        # Если вывод небольшой, оставляем его без изменений
        if TokenUsageTracker.estimate_tokens(tool_result.content) < remaining / 6:
            return

        # Интеллектуально синтезируем цель анализа на основе всей истории через LLM
        task_goal = self._synthesize_task_goal(tool_result.name)

        # Запускаем итеративную обработку порциями с вычисленной целью
        compressed_output = self._chunk_and_summarize_large_text(
            text=tool_result.content,
            tool_name=tool_result.name,
            task_goal=task_goal
        )

        original_len = len(tool_result.content)
        tool_result.content = (f"{ENVIRONMENT_PREFIX} Tool result content is too large so it was summarized automatically. "
                               f"Don't repeat reading the file, it will lead to the same result and won't help to change anything. Summarization: \n{compressed_output}")

        self.on_system_msg(
            f"[AUTO-COMPRESS] Summarized '{tool_result.name}' output: "
            f"{original_len} → {len(tool_result.content)} chars"
        )

    def _chunk_and_summarize_large_text(self, text: str, tool_name: str, task_goal: str) -> str:
        """
        Инкрементально собирает факты по каждому чанку и синтезирует их в единый связный отчет.
        Гарантирует отсутствие потери информации из ранних порций данных.
        """
        self.on_system_msg(f"[CHUNK ANALYZER] Starting chunked analysis of {len(text)} chars for tool '{tool_name}'...")

        token_limit = self.token_tracker.max_context_tokens
        token_chunk_size = max(int(token_limit / 3.5), 3000)
        chunk_size = int(token_chunk_size * 2.5)

        chunks = []
        pos = 0
        while pos < len(text):
            if pos + chunk_size >= len(text):
                chunks.append(text[pos:])
                break
            split_pos = text.rfind('\n', pos, pos + chunk_size)
            if split_pos == -1 or split_pos <= pos:
                split_pos = pos + chunk_size
            chunks.append(text[pos:split_pos])
            pos = split_pos

        total_chunks = len(chunks)

        # Список для накопления уникальных находок из каждого чанка в Python
        findings_by_portion = []

        # Переменная для фиксации решения субагента на каждом шаге
        decision_data = {"chunk_findings": "", "decision": "continue", "reason": ""}

        # Объявляем точечный инструмент сбора информации с более жесткими описаниями решений
        @tool(
            description="Report findings from the current chunk and decide about the next step.",
            chunk_findings=("str", "Key facts, data, or errors extracted from this current chunk that are highly relevant to the goal/task. Be precise and concrete. Write 'None' if nothing useful found."),
            reason=("str", "Very brief explanation for your decision"),
            decision=("str", "One of: "
                             "'continue' (select this if think looking at remaining portions is perspective/useful), "
                             "'stop_found' (select this if you have already extracted what is needed to satisfy the goal), "
                             "'stop_useless' (select this ONLY if the entire file/output is completely unrelated to the task and cannot help at all)"),

        )
        def report_step(chunk_findings: str, decision: str, reason: str = "") -> str:
            decision_data["chunk_findings"] = chunk_findings
            decision_data["decision"] = decision.strip().lower()
            decision_data["reason"] = reason
            return f"Step recorded."

        # 2. Итеративно собираем факты
        for idx, chunk in enumerate(chunks):
            current_num = idx + 1
            self.on_system_msg(f"[CHUNK ANALYZER] Processing portion {current_num}/{total_chunks}...")

            # Формируем историю находок для контекста субагента, чтобы он не дублировал информацию
            history_str = "\n".join(findings_by_portion) if findings_by_portion else "No findings yet."

            step_agent = SubAgent(
                system_prompt=(
                    "You're a info extractor sub-agent. Your main job is to extract and preserve most useful highly relevant to "
                    "the goal info from portions of text. Try to separate the useful signal from the noise, keeping only the signal. "
                    "You basically need to intelligently summarize what you read. YOU MUST CITE PORTION TEXT. "
                    "Do NOT duplicate what has already been found in previous portions.\n"
                ),
                max_context_tokens=token_chunk_size * 2,
                tools_config=["report_step"],  # Разрешен только инструмент сбора
                external_plugins={"report_step": report_step},
                safe_only=True,
                max_iter=1,
                temp=0.0,
                on_log=lambda x: None
            )

            prompt = (
                f"MAIN GOAL: {task_goal}\n"
                f"Instructions:\n"
                f"Just call `report_step` with only fresh new very detailed findings and citations from {current_num} portion, very brief reasoning and final decision.\n"
                f"YOUR FINDINGS FROM PREVIOUS PORTIONS:\n{history_str}\n\n"
                f"--- PORTION ({current_num} / {total_chunks}) ---\n"
                f"{chunk}\n---\n\n"
                f"Consider that your findings will be read by another agent that can't read the original text unlike you. "
                f"So preserve many details except useless noise. Basically summarize the portions considering the goal."
            )

            step_agent.run(prompt)

            tc = step_agent.get_last_tool_call()
            if not tc or tc.name != "report_step":
                self.on_system_msg(f"[CHUNK ANALYZER] Warning: Subagent missed tool call at portion {current_num}. Skipping.")
                last_msg = step_agent._agent.history.get_last_message()
                if last_msg and last_msg.content:
                    findings_by_portion.append(f"\n[Portion {current_num}]: {last_msg.content}")
                continue

            findings = decision_data["chunk_findings"].strip()
            decision = decision_data["decision"]
            reason = decision_data["reason"]

            # Сохраняем находки, если они содержат полезную информацию
            if findings and findings.lower() != "none":
                findings_by_portion.append(f"- [Portion {current_num}]: {findings}")

            self.on_system_msg(f"[CHUNK ANALYZER] Portion {current_num} decision: '{decision}' ({reason})")

            # Логика остановки
            if decision == 'stop_found':
                self.on_system_msg(f"[CHUNK ANALYZER] Early stop: Target located. Proceeding to synthesis...")
                break

            elif decision == 'stop_useless':
                # Если модель решила, что файл бесполезен, мы в любом случае останавливаем чтение
                self.on_system_msg(f"[CHUNK ANALYZER] Early stop: Source determined irrelevant. Reason: {reason}")

                # Но если мы УЖЕ успели что-то найти ранее, мы не выбрасываем это, а идем на финальный синтез
                if len(findings_by_portion) == 0:
                    return f"[ANALYSIS ABORTED] Source output is irrelevant to the task. Reason: {reason}"
                break

        # 3. Финальный синтез результатов
        if not findings_by_portion:
            return "No relevant information found in the tool output."

        raw_accumulated_findings = "\n".join(findings_by_portion)
        self.on_system_msg(f"[CHUNK ANALYZER] Synthesizing final response from all collected portions: {raw_accumulated_findings}")

        return raw_accumulated_findings

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
        self._prune_all_failed_tool_calls_except_last()

        tool_error_occurred = any(tr.is_error and not tr.is_user_denied for tr in tool_results)

        return clean_content, tool_error_occurred

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
                    prefill=step_prefill
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

        return