from __future__ import annotations
from typing import Optional, TYPE_CHECKING

from universal_agents.config import Config, CHARS_PER_TOKEN, MIN_TOKENS_TO_SUMMARIZE
from universal_agents.constants import ENVIRONMENT_PREFIX
from universal_agents.models import UserMessage, ToolResult
from universal_agents.llm_client import LLMClient, TokenUsageTracker

if TYPE_CHECKING:
    from universal_agents.agent import LLMAgent


# ============================================================
# Автокомпрессия длинных результатов инструментов
# ============================================================

def summarize_dialogue(
    agent: LLMAgent,
    start_id: Optional[int] = None,
    end_id: Optional[int] = None,
) -> Optional[str]:
    """
    Суммаризация диалога через LLM.
    mode='single' — один вызов LLM со всей историей как structured prefix.
    mode='batch' — разбивка на пачки по MIN_TOKENS_TO_SUMMARIZE, каждая
                   суммаризируется отдельно, результаты склеиваются.
    """
    messages = agent.history.get_all()

    if start_id is None:
        start_id = Config.AFTER_SYSTEM_PROMPT
    if end_id is None:
        end_id = len(messages) - 1

    if start_id > end_id:
        return None

    # Структурированные сообщения истории для KV-cache reuse
    history_msgs = [msg.to_api_dict() for msg in messages[start_id:end_id + 1]]

    # Последнее user-сообщение в диапазоне для акцента
    last_user_content = ""
    for i in range(end_id, start_id - 1, -1):
        if isinstance(messages[i], UserMessage):
            last_user_content = messages[i].content
            break

    # Вся история как prefix для KV-cache reuse
    full_history_msgs = [msg.to_api_dict() for msg in messages]

    if Config.DISABLE_PER_MESSAGE_SUMMARIZATION:
        # Сразу суммаризируем весь диалог целиком, не трогая отдельные сообщения
        summary = _single_phase_summarize(agent, history_msgs, last_user_content)
    else:
        summary = _batch_summarize(agent, history_msgs, full_history_msgs) or _single_phase_summarize(agent, history_msgs, last_user_content)

    return summary


def _single_phase_summarize(
    agent: LLMAgent, history_msgs: list[dict], last_user_content: str = ""
) -> Optional[str]:
    """
    Однофазная суммаризация — один вызов LLM.
    Промпт суммаризации добавляется в конец structured истории.
    """
    prompt = (
        f"{ENVIRONMENT_PREFIX} Based on the conversation above "
        f"write very dense and detailed summarized dialog using roles User, AI, tool.\n"
        f"Ensure preserving:"
        f"1. The last user message and the original user task.\n"
        f"2. Key decisions made\n"
        f"3. Critical details for further conversation/work\n"
        f"4. Current task state (what's done, what's pending)\n"
        f"Remove reasoning chains and redundant info. "
        f"Output ONLY the dense summary in dialog format (like 'AI: ...\\n User: ...' and so on)."
    )
    msgs = history_msgs + [{"role": "user", "content": prompt}]
    msg_obj, err, usage = LLMClient.call(msgs, temp=0.1, timeout=60, tools=None)
    if usage:
        agent.token_tracker.update_from_usage(usage)
    if err or not msg_obj or not msg_obj.content:
        return None
    return msg_obj.content.strip()


def _summarize_batch(agent: LLMAgent, full_history_msgs: list[dict], target_content: str) -> Optional[str]:
    """
    Суммаризация одного длинного сообщения (до 2 попыток).
    full_history_msgs + prompt(с кусочком target) → summary.
    Если summary не короче оригинала после 2 попыток — возвращает None.
    """
    original_chars = len(target_content)

    for attempt in range(2):
        snippet = target_content[:100]
        prompt = (
            f"{ENVIRONMENT_PREFIX} Summarize that 1 message above in a very dense detailed way."
            f"Preserve critical details for further dialog/work. "
            f"Remove: reasoning chains / redundant info. "
            f"Output ONLY the dense detailed version of the message above."
            f"\n\nTarget message begins with:\n{snippet}"
        )

        msgs = full_history_msgs + [{"role": "user", "content": prompt}]
        msg_obj, err, usage = LLMClient.call(
            msgs, temp=0.2 if attempt == 0 else 0.45,
        )
        if usage:
            agent.token_tracker.update_from_usage(usage)

        if err or not msg_obj or not msg_obj.content:
            if attempt == 1:
                return None
            continue

        summary = msg_obj.content.strip()
        if len(summary) < original_chars:
            return summary

    return None


def _batch_summarize(
    agent: LLMAgent, history_msgs: list[dict], full_history_msgs: list[dict]
) -> Optional[str]:
    """
    Проходит по сообщениям, каждое длинное (> MIN_TOKENS_TO_SUMMARIZE)
    суммаризирует отдельно через _summarize_batch.
    Короткие сообщения оставляет как есть.
    Результаты склеиваются в один текст через разделитель.
    """
    MIN_CHARS = int(MIN_TOKENS_TO_SUMMARIZE * CHARS_PER_TOKEN)

    summarized = 0

    parts: list[str] = []
    for msg_dict in history_msgs:
        content = msg_dict.get('content', '') or ''
        if not content.strip():
            continue

        role = msg_dict.get('role', 'user')
        role = role if role != 'assistant' else 'AI'

        if len(content) >= MIN_CHARS:
            batch = _summarize_batch(agent, full_history_msgs, content)
            if batch:
                parts.append(role + ': ' + batch)
                summarized += 1
                continue

        parts.append(role + ': ' + content)

    return "\n\n".join(parts) if summarized > 0 and parts else None


def synthesize_task_goal(agent: LLMAgent, tool_name: str) -> str:
    """
    Анализирует всю историю диалога через LLM и формулирует точную цель
    для анализа вывода конкретного инструмента.
    """
    agent.on_system_msg(f"[GOAL SYNTHESIS] Analyzing conversation history to formulate goal for '{tool_name}'...")

    messages_base = agent._prepare_messages_for_api()[:-1]

    synthesis_prompt = (
        f"{ENVIRONMENT_PREFIX}\n"
        f"Based on the dialog create a very concise goal for a sub-agent who will summarize the output "
        f"of '{tool_name}' tool for you because it's too large to fit in your memory. "
        f"After that you will only read the summary so you need to list concrete things sub-agent "
        f"must pay attention to. Output ONLY very short instruction for sub-agent."
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
            if isinstance(msg, UserMessage):
                return msg.content
        return "Extract any useful info relevant to the general task."

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

    if Config.DISABLE_TOOL_AUTO_SUMMARIZATION:
        return

    last_user = agent.history.get_last_user_message()
    if last_user is None:
        return
    remaining = agent.token_tracker.get_remaining(last_user.content)

    if TokenUsageTracker.estimate_tokens(tool_result.content) < remaining // Config.SUMMARIZATION_THRESHOLD_DIVIDER and remaining > Config.MAX_CONTEXT_TOKENS // Config.SUMMARIZATION_THRESHOLD_DIVIDER:
        return

    task_goal = synthesize_task_goal(agent, tool_result.name)
    compressed_output = chunk_and_summarize_large_text(agent, tool_result.content, tool_result.name, task_goal)

    original_len = len(tool_result.content)

    new_tool_result_content = (
        f"{ENVIRONMENT_PREFIX}\nTool result content was auto-summarized because of size. "
        f"Don't repeat call this tool with same args - you'll get same result.\n"
        f"Summary: \n{compressed_output}"
    )

    if len(new_tool_result_content) < original_len * 0.95:
        tool_result.content = new_tool_result_content
        agent.on_system_msg(
            f"[AUTO-COMPRESS] Summarized '{tool_result.name}' output: "
            f"{original_len} → {len(tool_result.content)} chars"
        )
    else:
        agent.on_system_msg(
            f"[AUTO-COMPRESS] Summarization failed '{tool_result.name}' output: "
            f"{original_len} → {len(new_tool_result_content)} chars. Fallback to original output."
        )


def chunk_and_summarize_large_text(agent: LLMAgent, text: str, tool_name: str, task_goal: str) -> str:
    """
    Инкрементально собирает факты по каждому чанку и синтезирует их в единый связный отчет.
    """
    agent.on_system_msg(f"[CHUNK ANALYZER] Starting chunked analysis of {len(text)} chars for tool '{tool_name}'...")

    token_limit = agent.token_tracker.get_remaining()
    token_chunk_size = token_limit // Config.SUMMARIZATION_THRESHOLD_DIVIDER
    chunk_size = int(token_chunk_size * CHARS_PER_TOKEN)

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

    for idx, chunk in enumerate(chunks):
        current_num = idx + 1
        agent.on_system_msg(f"[CHUNK ANALYZER] Processing portion {current_num}/{total_chunks}...")

        history_str = "\n".join(findings_by_portion) if findings_by_portion else "No findings yet."

        step_agent = agent.make_sub_agent(
            tools_config=[],
            external_plugins={},
            safe_only=False,
            max_iter=1,
            temp=0.35,
            on_log=agent.on_system_msg,
        )

        specialist_instructions = (
            "You're an info extractor sub-agent. "
            "Your main job is to extract and preserve the most useful relevant "
            "to the goal info from portions of original text. "
            "Intelligently summarize what you read and CITE PORTION TEXT. "
            "Do NOT duplicate findings from previous portions."
        )

        prompt = (
            f"{specialist_instructions}\n\n"
            f"MAIN GOAL: {task_goal}\n"
            f"Instructions:\n"
            f"Return only a structured response in EXACTLY this format (no extra text):\n"
            f"FINDINGS: <your fresh new very detailed findings and citations from portion {current_num}>\n"
            f"DECISION: <one of: continue, stop_found, stop_useless>\n"
            f"REASON: <very brief explanation for your decision>\n\n"
            f"YOUR FINDINGS FROM PREVIOUS PORTIONS:\n{history_str}\n\n"
            f"--- PORTION ({current_num} / {total_chunks}) ---\n"
            f"{chunk}\n---\n\n"
            f"Consider that your findings will be read by another agent that can't read the original text unlike you. "
            f"So preserve many details except useless noise. Basically summarize the portions considering the goal."
        )

        step_agent.run(prompt)

        last_msg = step_agent._agent.history.get_last_message()
        if not last_msg or not hasattr(last_msg, 'content') or not last_msg.content:
            agent.on_system_msg(f"[CHUNK ANALYZER] Warning: Subagent returned empty at portion {current_num}. Skipping.")
            continue

        response_text = last_msg.content.strip()
        findings = ""
        decision = "continue"
        reason = ""

        for line in response_text.split("\n"):
            line_s = line.strip()
            if line_s.upper().startswith("FINDINGS:"):
                findings = line_s[len("FINDINGS:"):].strip()
            elif line_s.upper().startswith("DECISION:"):
                decision = line_s[len("DECISION:"):].strip().lower()
            elif line_s.upper().startswith("REASON:"):
                reason = line_s[len("REASON:"):].strip()

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
