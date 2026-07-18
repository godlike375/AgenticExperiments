from __future__ import annotations
from typing import Optional, TYPE_CHECKING

from universal_agents.tool import tool, ENVIRONMENT_PREFIX
from universal_agents.models import UserMessage, ToolResult
from universal_agents.llm_client import LLMClient, TokenUsageTracker

if TYPE_CHECKING:
    from universal_agents.agent import LLMAgent


# ============================================================
# Автокомпрессия длинных результатов инструментов
# ============================================================

def summarize_text(agent: LLMAgent, text: str) -> Optional[str]:
    """Универсальная LLM-суммаризация произвольного текста."""
    prompt = (
        f"{ENVIRONMENT_PREFIX} Summarize the following text. "
        f"Preserve all key facts, specific data, decisions, etc. "
        f"Remove temporal reasoning and other things that aren't important anymore. "
        f"Output ONLY the concise summary text.\n\n"
        f"The original text:\n```\n{text}\n```\n "
        f"*Use the following text's language!* "
        f"If you see the dialog of AI and User, treat it as your own dialog with User!"
    )
    msgs = [{"role": "user", "content": prompt}]
    msg_obj, err, usage = LLMClient.call(msgs, temp=0.1, timeout=60, tools=None)
    if usage:
        agent.token_tracker.update_from_usage(usage)
    if err or not msg_obj or not msg_obj.content:
        return None
    return msg_obj.content.strip()


def synthesize_task_goal(agent: LLMAgent, tool_name: str) -> str:
    """
    Анализирует всю историю диалога через LLM и формулирует точную цель
    для анализа вывода конкретного инструмента.
    """
    agent.on_system_msg(f"[GOAL SYNTHESIS] Analyzing conversation history to formulate goal for '{tool_name}'...")

    messages_base = agent._prepare_messages_for_api()[:-1]

    synthesis_prompt = (
        f"{ENVIRONMENT_PREFIX}\n"
        f"Based on the current conversation context above create a tip for a sub-agent who will parse the output "
        f"of tool '{tool_name}' and summarize it for you because tool output is too long for your memory. "
        f"You only will read the summarization of the sub-agent so you need to note specific things that sub-agent "
        f"must pay attention to.\n"
        f"It must be a clear relatively concise instruction.\n"
        f"Output ONLY the formulated instruction for sub-agent."
    )

    synthesis_messages = messages_base + [{"role": "user", "content": synthesis_prompt}]

    msg_obj, err, usage = LLMClient.call(
        synthesis_messages,
        temp=agent.temp,
        timeout=agent.timeout,
        tools=None
    )

    if usage:
        agent.token_tracker.update_from_usage(usage)

    if err or not msg_obj or not msg_obj.content:
        agent.on_system_msg("[GOAL SYNTHESIS] Failed to synthesize goal via LLM. Falling back to last user message.")
        for msg in reversed(agent.history.get_all()):
            if isinstance(msg, UserMessage) and not msg.content.startswith(ENVIRONMENT_PREFIX):
                return msg.content
        return "Extract any useful facts and errors relevant to the general task."

    synthesized_goal = msg_obj.content.strip()
    agent.on_system_msg(f"[GOAL SYNTHESIS] Synthesized objective: \"{synthesized_goal}\"")
    return synthesized_goal


def auto_compress_tool_result(agent: LLMAgent, tool_result: ToolResult) -> None:
    """
    Автоматически сжимает вывод инструмента перед добавлением в историю,
    если он длинный, используя порционный анализ и динамический синтез цели.
    """
    if tool_result.is_error or tool_result.is_user_denied:
        return

    last_user = agent.history.get_last_user_message()
    if last_user is None:
        return
    remaining = agent.token_tracker.get_remaining(last_user.content)

    if TokenUsageTracker.estimate_tokens(tool_result.content) < remaining / 6:
        return

    task_goal = synthesize_task_goal(agent, tool_result.name)
    compressed_output = chunk_and_summarize_large_text(agent, tool_result.content, tool_result.name, task_goal)

    original_len = len(tool_result.content)
    tool_result.content = (
        f"{ENVIRONMENT_PREFIX} Tool result content is too large so it was summarized automatically. "
        f"Don't repeat reading the file, it will lead to the same result and won't help to change anything. "
        f"Summarization: \n{compressed_output}"
    )

    agent.on_system_msg(
        f"[AUTO-COMPRESS] Summarized '{tool_result.name}' output: "
        f"{original_len} → {len(tool_result.content)} chars"
    )


def chunk_and_summarize_large_text(agent: LLMAgent, text: str, tool_name: str, task_goal: str) -> str:
    """
    Инкрементально собирает факты по каждому чанку и синтезирует их в единый связный отчет.
    """
    agent.on_system_msg(f"[CHUNK ANALYZER] Starting chunked analysis of {len(text)} chars for tool '{tool_name}'...")

    token_limit = agent.token_tracker.max_context_tokens
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
    findings_by_portion: list[str] = []
    decision_data = {"chunk_findings": "", "decision": "continue", "reason": ""}

    from universal_agents.sub_agent import SubAgent

    @tool(
        description="Report findings from the current chunk and decide about the next step.",
        chunk_findings=("str", "Key facts... Write 'None' if nothing useful found."),
        reason=("str", "Very brief explanation for your decision"),
        decision=("str", "One of: 'continue', 'stop_found', 'stop_useless'"),
    )
    def report_step(chunk_findings: str, decision: str, reason: str = "") -> str:
        decision_data["chunk_findings"] = chunk_findings
        decision_data["decision"] = decision.strip().lower()
        decision_data["reason"] = reason
        return "Step recorded."

    for idx, chunk in enumerate(chunks):
        current_num = idx + 1
        agent.on_system_msg(f"[CHUNK ANALYZER] Processing portion {current_num}/{total_chunks}...")

        history_str = "\n".join(findings_by_portion) if findings_by_portion else "No findings yet."

        step_agent = SubAgent(
            system_prompt=(
                "You're a info extractor sub-agent. Your main job is to extract and preserve most useful highly relevant to "
                "the goal info from portions of text. Try to separate the useful signal from the noise, keeping only the signal. "
                "You basically need to intelligently summarize what you read. YOU MUST CITE PORTION TEXT. "
                "Do NOT duplicate what has already been found in previous portions.\n"
            ),
            max_context_tokens=token_chunk_size * 2,
            tools_config=["report_step"],
            external_plugins={"report_step": report_step},
            safe_only=True,
            max_iter=1,
            temp=0.0,
            on_log=lambda x: None,
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
            agent.on_system_msg(f"[CHUNK ANALYZER] Warning: Subagent missed tool call at portion {current_num}. Skipping.")
            last_msg = step_agent._agent.history.get_last_message()
            if last_msg and last_msg.content:
                findings_by_portion.append(f"\n[Portion {current_num}]: {last_msg.content}")
            continue

        findings = decision_data["chunk_findings"].strip()
        decision = decision_data["decision"]
        reason = decision_data["reason"]

        if findings and findings.lower() != "none":
            findings_by_portion.append(f"- [Portion {current_num}]: {findings}")

        agent.on_system_msg(f"[CHUNK ANALYZER] Portion {current_num} decision: '{decision}' ({reason})")

        if decision == 'stop_found':
            agent.on_system_msg(f"[CHUNK ANALYZER] Early stop: Target located. Proceeding to synthesis...")
            break
        elif decision == 'stop_useless':
            agent.on_system_msg(f"[CHUNK ANALYZER] Early stop: Source determined irrelevant. Reason: {reason}")
            if len(findings_by_portion) == 0:
                return f"[ANALYSIS ABORTED] Source output is irrelevant to the task. Reason: {reason}"
            break

    if not findings_by_portion:
        return "No relevant information found in the tool output."

    raw_accumulated_findings = "\n".join(findings_by_portion)
    agent.on_system_msg(f"[CHUNK ANALYZER] Synthesizing final response from all collected portions: {raw_accumulated_findings}")
    return raw_accumulated_findings
