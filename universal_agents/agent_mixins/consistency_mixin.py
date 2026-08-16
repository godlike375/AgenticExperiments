"""Mixin self-consistency генерации LLMAgent: несколько черновиков + синтез."""

from __future__ import annotations

from universal_agents.config import Config
from universal_agents.constants import ENVIRONMENT_PREFIX
from universal_agents.llm_client import LLMClient
from universal_agents.models import UserMessage
from universal_agents.context_builder import prepare_messages_for_api, get_effective_prefill
from universal_agents.tool_parsing import tc_name


class ConsistencyMixin:
    """Режим self-consistency: генерирует несколько черновиков и синтезирует итог."""

    def _generate_draft_with_tool_suggestions(self, draft_messages, prefill, draft_temp):
        prefill_val = get_effective_prefill(prefill)
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
        if not Config.DISABLE_PER_MESSAGE_SUMMARIZATION:
            self._maybe_summarize_user_message(user_message)
        messages_base = prepare_messages_for_api(self)

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
                tc_names = [f"{tc_name(tc)}(...)" for tc in draft.tool_calls]
                content += f"\n[Suggested tools: {', '.join(tc_names)}]"
            draft_texts.append(f"--- Draft {i} ---\n{content}")

        synthesis_prompt = (
            f"{ENVIRONMENT_PREFIX} Here are drafts from multiple reasoning paths:\n"
            + "\n".join(draft_texts)
            + "\n\n Analyse them and synthesize the finishing correct answer, paying attention to suggested tools. Output only the final synthesized answer."
        )
        synthesis_messages = messages_base + [{"role": "user", "content": synthesis_prompt}]
        current_prefill = get_effective_prefill(prefill)
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