from __future__ import annotations
from typing import Optional, TYPE_CHECKING

from universal_agents.config import Config, CHARS_PER_TOKEN, MIN_TOKENS_TO_SUMMARIZE
from universal_agents.constants import ENVIRONMENT_PREFIX, ENVIRONMENT_PREFIX_END
from universal_agents.constants import (
    SUMMARY_PREFIX_USER,
    SUMMARY_PREFIX_AI,
    SUMMARY_PREFIX_TOOL_CALL,
    SUMMARY_PREFIX_TOOL_RESULT,
)
from universal_agents.models import UserMessage, AssistantMessage, ToolResult
from universal_agents.llm_client import TokenUsageTracker
from universal_agents.context_builder import prepare_messages_for_api

if TYPE_CHECKING:
    from universal_agents.agent import LLMAgent

SECTION_GUIDE = """
* 1.1 Ключевой контекст, сущности и связи.
* 1.2 Специфические термины/сокращения/договорённости в рамках проекта/задачи.
* 1.3 Главная задача, критерии выполнения, ожидаемый результат.
* 1.4 Ограничения и требования (строгие и нестрогие отдельно).
* 1.5 Предпочтения и дополнительные инструкции пользователя.

* 2.1 Уже выполненные шаги, их результаты, трудности (если возникли).
* 2.2 Текущий и следующие шаги, критерии выполнения, ожидаемый результат.
* 2.3 Ключевые действия и решения, неудачные попытки, отклонённые альтернативы. Причины перечисленного.

* 3.1 Ждущие задачи и зависимости, кто их блокирует и почему.

* 4.1 Что требет проверки/уточнения. 
* 4.2 Что подтверждено и как именно.
* 4.3 Наблюдения, выводы, гипотезы.

* 5.1 Мета-данные о задаче, проекте, пользователе.

* 6.1 Задействованные ресурсы/артефакты, незавершённые черновики. Для каждого: зачем нужен, где находится, мета-данные, что с ним сделали, что с ним осталось сделать.
* 6.2 Самые важные подробные цитаты/детали из задействованных ресурсов и черновиков для продолжения работы.

* 7.1 Подсказки для будущей версии себя (если есть).
* 7.2 Информация, не попавшая в предыдущие разделы, но возможно важная (если есть).

"""

NEW_TO_PREVIOUS = """
*   7.3 Новое относительно предыдущего саммари (обязательно).
"""

import re as _re

_REQUIRED_SECTION_IDS = tuple(sorted(
    set(_re.findall(r'\d+\.\d+', SECTION_GUIDE)),
    key=lambda s: tuple(map(int, s.split('.')))
))
_REQUIRED_SECTION_IDS_WITH_73 = _REQUIRED_SECTION_IDS + (
    tuple(s for s in _re.findall(r'\d+\.\d+', NEW_TO_PREVIOUS) if s not in _REQUIRED_SECTION_IDS)
)


# ============================================================
# Автокомпрессия длинных результатов инструментов
# ============================================================


def is_summary_message(msg) -> bool:
    """Проверяет, является ли сообщение ранее сгенерированным авто-саммари.

    Детекция исключительно по метаданным объекта `UserMessage.is_summary` —
   никаких текстовых маркеров внутри контента не используется."""
    return isinstance(msg, UserMessage) and bool(getattr(msg, "is_summary", False))


def _find_existing_summary(messages: list, end_id: int) -> Optional[dict]:
    """Ищет ранее сгенерированное авто-саммари в обрезаемом диапазоне; возвращает dict или None."""
    for i in range(0, end_id):
        msg = messages[i]
        if is_summary_message(msg):
            lines = msg.content.split("\n", 1)
            header, body = lines[0], lines[1] if len(lines) > 1 else ""
            return {"index": i, "full_content": msg.content, "body": body}
    return None


def _report_service_error(agent: LLMAgent, context: str, err, msg_obj=None) -> None:
    """Ошибки служебных LLM-вызовов не тонут — выводятся в system-канал."""
    if err:
        agent.on_system_msg(f"[llm-service] Error in {context}: {err}")
    elif not msg_obj or not getattr(msg_obj, "content", None):
        agent.on_system_msg(f"[llm-service] {context}: empty response from LLM")


def _dense_summarize_message(agent: LLMAgent, content: str) -> Optional[str]:
    """Плотное безпотерьное саммари одного сообщения (per-message режим); складывается в рабочую память."""
    content = (content or "").strip()
    if not content:
        return None
    prompt = (
        f"{ENVIRONMENT_PREFIX} Write a SHORT version of the previous message ('{content[:20]}...').\n"
        f"Preserve critical concrete things: reasons, decisions, actions taken, files, identifiers, names, arguments, commands, values, numbers, errors, etc.\n"
        "Try to group something if possible.\n"
        f"Remove reasoning chains and redundant filler. Do NOT generalize identifiers.\n"
        f"Output ONLY the dense structured summary.\n"
        f"{ENVIRONMENT_PREFIX_END}"
    )
        # Тот же формат, что у основного диалога (закэшированные заголовки), чтобы переиспользовался KV-кэш.
    history_msgs = prepare_messages_for_api(agent, normalize=False)
    msgs = history_msgs + [{"role": "user", "content": prompt}]
    msg_obj, err, usage = agent.service_llm_call(msgs, temp=Config.TEMP, timeout=Config.TIMEOUT)
    if err or not msg_obj or not msg_obj.content:
        _report_service_error(agent, "per-message summary", err, msg_obj)
        return None
    return msg_obj.content.strip()


def _build_from_per_message_summaries(
    agent: LLMAgent, start_id: Optional[int], end_id: Optional[int],
    truncate_result_ratio: float = 0.0, truncate_result_min_chars: int = 0,
) -> Optional[str]:
    """Собирает короткий диалог из накопленных per-message саммари (без саммари — исходный контент); None, если саммари нет."""
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
        # Чистый вызов инструмента (пустой текст) раньше терялся; сохраняем, чтобы саммари включало вызовы.
        if isinstance(msg, AssistantMessage) and msg.has_tool_calls() and not content:
            tcs = ", ".join(f"{tc.name}({tc.arguments})" for tc in msg.tool_calls)
            parts.append(f"{SUMMARY_PREFIX_TOOL_CALL} {tcs}")
            continue
        if not content:
            continue
        summary = agent.history.get_per_msg_summary(msg)
        if summary:
            parts.append(summary)
            used_summaries += 1
            continue
        if isinstance(msg, UserMessage):
            parts.append(f"{SUMMARY_PREFIX_USER} {content}")
        elif isinstance(msg, ToolResult):
            if truncate_result_ratio > 0 and len(content) > 0:
                # Относительная обрезка: доля оригинала, но не меньше truncate_result_min_chars.
                limit = max(int(len(content) * truncate_result_ratio), truncate_result_min_chars)
                if len(content) > limit:
                    content = content[:limit] + f"...[truncated to {limit} chars]"
            parts.append(f"{SUMMARY_PREFIX_TOOL_RESULT} {content}")
        else:
            parts.append(f"{SUMMARY_PREFIX_AI} {content}")

    # Без сохранённых саммари per-message сборка бессмысленна — отдаём None для отката на single-shot.
    if used_summaries == 0:
        return None

    return "\n\n".join(parts)


def _draft_task_summary(
    agent: LLMAgent, history_msgs: list[dict], task_id: str, task_title: str
) -> Optional[str]:
    """Черновик саммари завершённой подзадачи по её сегменту истории."""
    prompt = (
        f"{ENVIRONMENT_PREFIX} Below is the execution trace of a COMPLETED subtask "
        f"'{task_title}' (id={task_id}).\n"
        f"Write a very dense SHORT summary of what was done and the result.\n"
        f"Preserve ALL important concrete facts: file paths, identifiers, names, arguments, commands, values, numbers, errors, etc.\n"
        "Try to group something if possible.\n"
        f"Key decisions and their reasons, and the results of tool executions.\n"
        f"State what is DONE vs PENDING/BLOCKED and what the next step is.\n"
        f"Remove only reasoning chains and redundant filler. Do NOT generalize identifiers.\n"
        f"Output ONLY the dense structured summary:\n"
        f"ACTIONS & DECISIONS MADE:\n"
        f"NEXT STEP:"
        f"\n{ENVIRONMENT_PREFIX_END}"
    )
    msgs = history_msgs + [{"role": "user", "content": prompt}]
    msg_obj, err, usage = agent.service_llm_call(msgs, temp=Config.TEMP, timeout=Config.TIMEOUT)
    if err or not msg_obj or not msg_obj.content:
        _report_service_error(agent, "task segment draft", err, msg_obj)
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
        f"\n{ENVIRONMENT_PREFIX_END}"
    )
    msgs = history_msgs + [{"role": "user", "content": review_prompt}]
    msg_obj, err, usage = agent.service_llm_call(msgs, temp=Config.TEMP, timeout=Config.TIMEOUT)
    if err or not msg_obj or not msg_obj.content:
        _report_service_error(agent, "task segment review", err, msg_obj)
        return None
    return msg_obj.content.strip()


def synthesize_task_goal(agent: LLMAgent, tool_name: str) -> str:
    """Формулирует через LLM точную цель анализа вывода конкретного инструмента."""
    agent.on_system_msg(f"[GOAL SYNTHESIS] Analyzing conversation history to formulate goal for '{tool_name}'...")

    messages_base = prepare_messages_for_api(agent)[:-1]

    synthesis_prompt = (
        f"{ENVIRONMENT_PREFIX}\n"
        f"Based on the dialog create a very concise goal for a sub-agent who will summarize the output "
        f"of '{tool_name}' tool for you because it's too large to fit in your memory. "
        f"After that you will only read the summary so you need to list concrete things sub-agent "
        f"must pay attention to. Output ONLY very short instruction for sub-agent."
        f"\n{ENVIRONMENT_PREFIX_END}"
    )

    synthesis_messages = messages_base + [{"role": "user", "content": synthesis_prompt}]

    msg_obj, err, usage = agent.service_llm_call(
        synthesis_messages,
        temp=agent.temp,
        timeout=agent.timeout,
    )

    if err or not msg_obj or not msg_obj.content:
        _report_service_error(agent, "goal synthesis", err, msg_obj)
        agent.on_system_msg("[GOAL SYNTHESIS] Falling back to last user message.")
        for msg in reversed(agent.history.get_all()):
            if isinstance(msg, UserMessage):
                return msg.content
        return "Extract any useful info relevant to the general task."

    synthesized_goal = msg_obj.content.strip()
    agent.on_system_msg(f"[GOAL SYNTHESIS] Synthesized objective: \"{synthesized_goal}\"")
    return synthesized_goal


def auto_compress_tool_result(agent: LLMAgent, tool_result: ToolResult) -> None:
    """Автоматически сжимает длинный вывод инструмента перед добавлением в историю (порционный анализ)."""
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
        f"\n{ENVIRONMENT_PREFIX_END}"
    )

    if len(new_tool_result_content) < original_len * 0.95:
        tool_result.content = new_tool_result_content
        # Результат сжат — сбрасываем кэш хэша файла, чтобы следующий read перечитал.
        agent._on_history_changed()
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
    """Инкрементально собирает факты по чанкам и синтезирует их в единый отчёт."""
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
            denied_tools="*",
            max_iter=1,
            temp=Config.TEMP,
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


# ============================================================
# Транскрипция истории + single-shot session summary (7 разделов)
# ============================================================


def summarize_history_plain(
    agent: LLMAgent,
    history_msgs: list[dict],
    temp: float = 0.1,
    extra_instruction: str = None,
    include_new_to_previous: bool = False,
) -> Optional[str]:
    """Сборка session summary в режиме single-shot: подаётся полная история (префикс совпадает для KV-cache), инструкция — последним сообщением. Перезаписывает заметки с нуля. ``include_new_to_previous`` добавляет раздел 7.3 (новые находки относительно предыдущего саммари) — имеет смысл только при повторной компакции, на первой предыдущего саммари нет. Возвращает текст или None."""
    new_section = NEW_TO_PREVIOUS if include_new_to_previous else ""
    prompt = (
        f"{ENVIRONMENT_PREFIX} Fill these sections of the very detailed session summary because "
        f"you're out of memory and can't continue working without summarizing:\n```"
        f"{SECTION_GUIDE}{new_section}\n```\n"
        f"Answer only in the strict format. Do your best to avoid repeating already said things between sections.\n"
        f"If it's not the first summary then do your best to bring fresh things into this new detailed summary.\n"
        f"{extra_instruction if extra_instruction else ''}"
    )
    prompt += f"{ENVIRONMENT_PREFIX_END}"
    # history_msgs содержит system prompt первым — префикс совпадает, KV-cache переиспользуется.
    msgs = list(history_msgs) + [{"role": "user", "content": prompt}]
    msg_obj, err, usage = agent.service_llm_call(msgs, temp=temp, timeout=Config.TIMEOUT)
    if err or not msg_obj or not msg_obj.content:
        _report_service_error(agent, "dialog plain summary", err, msg_obj)
        return None
    return msg_obj.content.strip()
