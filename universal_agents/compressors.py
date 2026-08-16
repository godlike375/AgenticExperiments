from __future__ import annotations
from typing import Optional, TYPE_CHECKING

from universal_agents.config import Config, CHARS_PER_TOKEN, MIN_TOKENS_TO_SUMMARIZE
from universal_agents.constants import ENVIRONMENT_PREFIX, SUMMARY_MARKER
from universal_agents.models import UserMessage, ToolResult
from universal_agents.llm_client import LLMClient, TokenUsageTracker
from universal_agents.context_builder import prepare_messages_for_api

if TYPE_CHECKING:
    from universal_agents.agent import LLMAgent
    from universal_agents.models import Message


# ============================================================
# Автокомпрессия длинных результатов инструментов
# ============================================================


def is_summary_message(content: str) -> bool:
    """Проверяет, является ли контент ранее сгенерированным авто-саммари."""
    return bool(content) and (
        SUMMARY_MARKER in content
        or "[SUMMARY" in content
        or "auto-generated text" in content
    )


def _find_existing_summary(messages: list, end_id: int) -> Optional[dict]:
    """
    Ищет в обрезаемом диапазоне ранее сгенерированное авто-саммари.
    Возвращает {"index": int, "full_content": str, "body": str} или None.
    """
    for i in range(0, end_id):
        msg = messages[i]
        if isinstance(msg, UserMessage) and is_summary_message(msg.content):
            lines = msg.content.split("\n", 1)
            header, body = lines[0], lines[1] if len(lines) > 1 else ""
            return {"index": i, "full_content": msg.content, "body": body}
    return None


def _dense_summarize_message(agent: LLMAgent, content: str) -> Optional[str]:
    """
    Плотное, безлоосное саммари ОДНОГО сообщения. Используется в режиме
    per-message summarization: результат складывается в рабочую память агента
    (agent._per_msg_summaries) и НЕ попадает в контекст, а при сжатии диалога
    вставляется на место исходного сообщения.
    """
    content = (content or "").strip()
    if not content:
        return None
    prompt = (
        f"{ENVIRONMENT_PREFIX} Write a dense CONCISE version of the previous message ('{content[:20]}...').\n"
        f"Preserve critical concrete facts: file paths, identifiers, names, arguments, commands, values, numbers, errors, etc.\n"
        "Try to group something if possible.\n"
        f"Remove reasoning chains and redundant filler. Do NOT generalize identifiers.\n"
        f"Output ONLY the dense structured summary.\n"
    )
    history_msgs = [m.to_api_dict() for m in agent.history.get_all()]
    msgs = history_msgs + [{"role": "user", "content": prompt}]
    msg_obj, err, usage = LLMClient.call(msgs, temp=0.2, timeout=60, tools=(agent.tools if agent.tools else None))
    if err or not msg_obj or not msg_obj.content:
        return None
    return msg_obj.content.strip()


def _build_from_per_message_summaries(
    agent: LLMAgent, start_id: Optional[int], end_id: Optional[int]
) -> Optional[str]:
    """
    Собирает новый (более короткий) диалог из маленьких саммари, накопленных
    в рабочей памяти агента (agent._per_msg_summaries). Для сообщений без
    саммари (короткие выводы инструментов и т.п.) используется исходный контент.
    Возвращает None, если в сжимаемом диапазоне не нашлось ни одного саммари.
    """
    messages = agent.history.get_all()
    if start_id is None:
        start_id = Config.AFTER_SYSTEM_PROMPT
    if end_id is None:
        end_id = len(messages)

    end_id = min(end_id, len(messages))

    parts: list[str] = []
    used_summaries = 0
    for i in range(start_id, end_id):
        msg = messages[i]
        content = (msg.content or "").strip()
        if not content:
            continue
        summary = agent._per_msg_summaries.get(id(msg))
        if summary:
            parts.append(summary)
            used_summaries += 1
            continue
        if isinstance(msg, UserMessage):
            parts.append(f"USER: {content}")
        elif isinstance(msg, ToolResult):
            parts.append(f"TOOL({msg.name}): {content}")
        else:
            parts.append(f"AI: {content}")

    if used_summaries == 0 or not parts:
        return None
    return "\n\n".join(parts)


def summarize_dialogue(
    agent: LLMAgent,
    start_id: Optional[int] = None,
    end_id: Optional[int] = None,
) -> Optional[str]:
    """
    Суммаризация диалога через LLM.
    mode='per-message' (DISABLE_PER_MESSAGE_SUMMARIZATION=False) — новый диалог
        собирается из маленьких саммари, накопленных в рабочей памяти агента
        (по одному после каждого сообщения ассистента и длинного вывода
        инструмента). Ничего не запрашивается у LLM на лету — только склейка.
    mode='single' (DISABLE_PER_MESSAGE_SUMMARIZATION=True) — один вызов LLM со
        всей историей (черновик + отревьюенная версия).
    """
    messages = agent.history.get_all()

    if end_id is None:
        end_id = len(messages)

    # Структурированные сообщения истории для KV-cache reuse
    history_msgs = [msg.to_api_dict() for msg in messages]

    # Оригинальный системный промпт — должен быть 0-м сообщением в вызовах LLM,
    # чтобы префикс совпадал с реальным диалогом и переиспользовался KV-cache.
    system_msg = messages[0].to_api_dict() if messages else None

    # Ранее сгенерированное авто-саммари в обрезаемом диапазоне
    existing = _find_existing_summary(messages, end_id)

    if Config.DISABLE_PER_MESSAGE_SUMMARIZATION:
        # Сразу суммаризируем весь диалог целиком, не трогая отдельные сообщения
        summary = _single_phase_summarize(agent, history_msgs, system_msg, existing)
    else:
        # Новый диалог собирается из маленьких саммари рабочей памяти
        summary = (
            _build_from_per_message_summaries(agent, start_id, end_id)
            or _single_phase_summarize(agent, history_msgs, system_msg, existing)
        )

    return summary


def _single_phase_summarize(
    agent: LLMAgent,
    history_msgs: list[dict],
    system_msg: Optional[dict],
    existing: Optional[dict] = None,
) -> Optional[str]:
    """
    Однофазная суммаризация: черновик (draft) + review.
    Промпт суммаризации добавляется в конец structured истории.
    system_msg передаётся 0-м сообщением для переиспользования KV-cache.
    После черновика выполняется review-проход: прунинг устаревшего +
    добавление недостающих деталей.
    """
    summary = _draft_summary(agent, history_msgs, system_msg, existing)
    if not summary:
        return None
    if Config.AUTO_SUMMARY_REVIEW_PASS:
        summary = _review_summary(agent, history_msgs, summary, system_msg, existing) or summary
    return summary


def _draft_summary(
    agent: LLMAgent,
    history_msgs: list[dict],
    system_msg: Optional[dict],
    existing: Optional[dict] = None,
) -> Optional[str]:
    prompt = _build_draft_prompt(existing)
    msgs = history_msgs + [{"role": "user", "content": prompt}]
    msg_obj, err, usage = LLMClient.call(msgs, temp=0.1, timeout=60, tools=(agent.tools if agent.tools else None))
    if err or not msg_obj or not msg_obj.content:
        return None
    return msg_obj.content.strip()


def _review_summary(
    agent: LLMAgent,
    history_msgs: list[dict],
    draft: str,
    system_msg: Optional[dict],
    existing: Optional[dict] = None,
) -> Optional[str]:
    """
    Review-проход: модель отревьюивает черновик саммари на фоне реальной
    истории. Задача — СДЕЛАТЬ ПРУНИНГ: убрать устаревшие, дублирующиеся и
    ненужные детали (в первую очередь потерявшие актуальность), и ДОБАВИТЬ
    недостающие важные факты (chain-of-density).
    """
    existing_note = (
        "\nThere is an EXISTING auto-summary inside the history. Keep its still-relevant facts, "
        "fade out what is now outdated/unneeded."
        if existing and existing.get("body") else ""
    )
    review_prompt = (
        f"{ENVIRONMENT_PREFIX} Below is a DRAFT summary of the conversation above.\n"
        f"Review it against the actual conversation and produce the FINAL summary:\n"
        f"PRUNE: remove outdated, duplicated and no longer needed details — fade out first "
        f"what is already finished/superseded/irrelevant to the current task.\n"
        f"ADD: put back important facts missing from the draft (do not invent, only recover):\n"
        f"1. The ORIGINAL USER TASK and the MOST RECENT USER REQUEST (find the last user message in history).\n"
        f"2. Actual file/function/class/variable names, tool names and their arguments, exact paths, "
        f"commands and any concrete values/numbers/errors mentioned.\n"
        f"3. Key decisions, conclusions and results of tool execution.\n"
        f"4. Current state: what is DONE vs PENDING/BLOCKED, and what the next step is.\n"
        f"{existing_note}\n"
        f"Keep the summary dense, structured and complete. Do not add irrelevant noise. "
        f"Output ONLY the final summary.\n"
        f"\nDRAFT SUMMARY:\n{draft}"
    )
    msgs = history_msgs + [{"role": "user", "content": review_prompt}]
    msg_obj, err, usage = LLMClient.call(msgs, temp=0.1, timeout=60, tools=(agent.tools if agent.tools else None))
    if err or not msg_obj or not msg_obj.content:
        return None
    return msg_obj.content.strip()


def _build_draft_prompt(existing: Optional[dict] = None) -> str:
    """Строит промпт черновой суммаризации с акцентом на сохранность фактов."""
    if existing and existing.get("body"):
        summary_note = (
            f"\nThere is an EXISTING auto-summary of earlier part of the dialog inside the history. "
            f"Keep its still-important facts and MERGE with the newer messages. "
            f"Do not re-add facts that are no longer relevant."
        )
    else:
        summary_note = ""

    prompt = (
        f"{ENVIRONMENT_PREFIX} Based on the conversation above "
        f"write a very dense, detailed and lossless summary of the dialog.\n"
        f"Preserve ALL of the following:"
        f"1. The ORIGINAL USER TASK (findable at the start of the conversation) and the MOST RECENT "
        f"USER REQUEST (the last user message) — keep their intent precisely.\n"
        f"2. Every important concrete fact: file paths, function/class/variable names, tool names and "
        f"their arguments, exact commands, values, numbers, error messages.\n"
        f"3. Key decisions made and their reasons.\n"
        f"4. Results of tool executions: what was read, written, created, changed.\n"
        f"5. Current task state: what's DONE, what's PENDING/BLOCKED, what's the next step.\n"
        f"{summary_note}\n"
        f"Remove only reasoning chains and redundant filler. "
        f"Do NOT generalize or replace concrete identifiers with vague words. "
        f"Output ONLY the dense structured summary:"
        f"\n"
        f"SUMMARY:\n"
        f"TASK:\n"
        f"PROGRESS:\n"
        f"KEY FACTS:\n"
        f"DECISIONS:\n"
        f"STATE / NEXT STEPS:"
    )
    return prompt


def _draft_task_summary(
    agent: LLMAgent, history_msgs: list[dict], task_id: str, task_title: str
) -> Optional[str]:
    """Черновик саммари завершённой подзадачи по её сегменту истории."""
    prompt = (
        f"{ENVIRONMENT_PREFIX} Below is the execution trace of a COMPLETED subtask "
        f"'{task_title}' (id={task_id}).\n"
        f"Write a dense CONCISE summary of what was done and the result.\n"
        f"Preserve ALL important concrete facts: file paths, identifiers, names, arguments, commands, values, numbers, errors, etc.\n"
        "Try to group something if possible.\n"
        f"Key decisions and their reasons, and the results of tool executions.\n"
        f"State what is DONE vs PENDING/BLOCKED and what the next step is.\n"
        f"Remove only reasoning chains and redundant filler. Do NOT generalize identifiers.\n"
        f"Output ONLY the dense structured summary:\n"
        f"ACTIONS & DECISIONS MADE:\n"
        f"NEXT STEP:"
    )
    msgs = history_msgs + [{"role": "user", "content": prompt}]
    msg_obj, err, usage = LLMClient.call(msgs, temp=0.1, timeout=60, tools=(agent.tools if agent.tools else None))
    if err or not msg_obj or not msg_obj.content:
        return None
    return msg_obj.content.strip()


def _review_task_summary(
    agent: LLMAgent, history_msgs: list[dict], draft: str, task_id: str, task_title: str
) -> Optional[str]:
    """Review-проход: прунинг устаревшего + добавление пропущенного в черновике."""
    review_prompt = (
        f"{ENVIRONMENT_PREFIX} Below is a DRAFT summary of the completed subtask "
        f"'{task_title}' (id={task_id}) from the conversation above.\n"
        f"Review it against the actual conversation and produce the FINAL summary.\n"
        f"PRUNE: remove outdated, duplicated and no longer needed details.\n"
        f"ADD: put back important facts missing from the draft (do not invent, only recover): "
        f"concrete names, paths, tool names and args, values, errors, decisions, and "
        f"DONE vs PENDING/next step.\n"
        f"Keep the summary dense, structured and complete. Output ONLY the final summary.\n"
        f"\nDRAFT SUMMARY:\n{draft}"
    )
    msgs = history_msgs + [{"role": "user", "content": review_prompt}]
    msg_obj, err, usage = LLMClient.call(msgs, temp=0.1, timeout=60, tools=(agent.tools if agent.tools else None))
    if err or not msg_obj or not msg_obj.content:
        return None
    return msg_obj.content.strip()


def _summarize_batch(agent: LLMAgent, full_history_msgs: list[dict], target_content: str,
                     target_msg: Optional[dict] = None) -> Optional[str]:
    """
    Суммаризация одного длинного сообщения (до 2 попыток).
    full_history_msgs + prompt(с содержимым целевого сообщения) → summary.
    Если summary не короче оригинала после 2 попыток — возвращает None.
    """
    original_chars = len(target_content)

    # Ограничиваем, чтобы не превратить ошибку модели в прогресс
    cap = int(original_chars * 0.7)

    for attempt in range(2):
        # Включаем часть содержимого целевого сообщения прямо в промпт, чтобы
        # модель точно видела, что сжимать (не полагаемся на поиск по сниппету).
        sample = target_content[:cap]
        prompt = (
            f"{ENVIRONMENT_PREFIX} Summarize the single message below in a very dense detailed way. "
            f"Preserve critical details verbatim: file paths, names, arguments, values, errors. "
            f"Remove: reasoning chains / redundant info. "
            f"Output ONLY the dense detailed version of the message below.\n"
            f"\n--- TARGET MESSAGE (truncated to keep prompt short) ---\n"
            f"{sample}\n--- END ---"
        )

        msgs = full_history_msgs + [{"role": "user", "content": prompt}]
        msg_obj, err, usage = LLMClient.call(
            msgs, temp=0.2 if attempt == 0 else 0.45, tools=(agent.tools if agent.tools else None),
        )

        if err or not msg_obj or not msg_obj.content:
            if attempt == 1:
                return None
            continue

        summary = msg_obj.content.strip()
        if len(summary) < original_chars:
            return summary

    return None


def _batch_summarize(
    agent: LLMAgent, history_msgs: list[dict], full_history_msgs: list[dict],
    existing: Optional[dict] = None,
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

        # Ранее сгенерированное авто-саммари не сжимаем повторно
        if is_summary_message(content):
            parts.append(f"SUMMARY: {content}")
            continue

        if len(content) >= MIN_CHARS:
            batch = _summarize_batch(agent, full_history_msgs, content, msg_dict)
            if batch:
                parts.append(role + ': ' + batch)
                summarized += 1
                continue

        parts.append(role + ': ' + content)

    if summarized > 0 and parts:
        merged = "\n\n".join(parts)
        if existing and existing.get("body"):
            merged = f"SUMMARY (existing, pruned): {existing['body']}\n\n{merged}"
        return merged

    return None


def synthesize_task_goal(agent: LLMAgent, tool_name: str) -> str:
    """
    Анализирует всю историю диалога через LLM и формулирует точную цель
    для анализа вывода конкретного инструмента.
    """
    agent.on_system_msg(f"[GOAL SYNTHESIS] Analyzing conversation history to formulate goal for '{tool_name}'...")

    messages_base = prepare_messages_for_api(agent)[:-1]

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
        tools=(agent.tools if agent.tools else None)
    )

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
        # Содержимое результата сжато — модель потеряла фактический контент,
        # поэтому кэш хэша файла сбрасываем, чтобы следующий read перечитал файл.
        agent.file_states.prune()
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
