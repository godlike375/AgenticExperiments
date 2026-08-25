from __future__ import annotations
from typing import TYPE_CHECKING

from universal_agents.tool import tool
from universal_agents.constants import ENVIRONMENT_PREFIX, ENVIRONMENT_PREFIX_END, err, ok
from universal_agents.config import Config
from universal_agents.models import UserMessage, AssistantMessage, ToolResult, SystemMessage
from universal_agents.compressors import summarize_dialogue

if TYPE_CHECKING:
    from universal_agents.agent import LLMAgent


@tool(description="Get short indexed current history with ids",
       short_description="show history")
def get_messages(agent: LLMAgent, chars_per_message: int = 30) -> str:
    history = agent.history
    if len(history) <= Config.AFTER_SYSTEM_PROMPT:
        return ok(" История пока пустая.")
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
        seq = getattr(msg, "seq", None)
        seq_label = f"seq={seq} " if seq is not None else ""
        lines.append(f"{i}. {seq_label}{role}: {content.strip()}")
    return f"{ENVIRONMENT_PREFIX} Your current history:\n" + "\n".join(lines) + f"\n{ENVIRONMENT_PREFIX_END}"


@tool(
    description="Edits a specific message in the history",
    short_description="edit history",
    requires_confirmation=True,
    id=("int", "ID of the message to edit"),
    old=("str", "Optional exact substr to replace. Empty str replaces whole text"),
    new=("str", "Text to insert in place of old"),
)
def edit_message(agent: LLMAgent, id: int, new: str, old: str = '') -> str:
    result = agent.history.edit_message(id, new, old)
    tm = getattr(agent, "tools_manager", None)
    if tm is not None:
        tm.flush_pending_unloads()
    return result


@tool(
    description="Deletes a range of messages from dialog history",
    short_description="delete history",
    requires_confirmation=True,
    start_id=("int", "Starting message ID to delete"),
    end_id=("int", "Optional ending message ID (-1 for last)"),
)
def delete_messages(agent: LLMAgent, start_id: int, end_id: int = -1) -> str:
    result = agent.history.delete_range(start_id, end_id)
    tm = getattr(agent, "tools_manager", None)
    if tm is not None:
        tm.flush_pending_unloads()
    return result


@tool(
    description="Summarizes a range of dialog messages into a single concise UserMessage. "
                "Use to free context tokens. Cannot summarize system prompt.",
    short_description="compress dialog",
    requires_confirmation=True,
    start_id=("int", "Start index of messages to summarize"),
    end_id=("int", "End index (inclusive). Use -1 for last message"),
)
def summarize_messages(agent: LLMAgent, start_id: int, end_id: int = -1) -> str:
    history = agent.history
    if end_id == -1 or end_id >= len(history):
        end_id = len(history) - 3

    safe_start = max(start_id, Config.AFTER_SYSTEM_PROMPT)
    safe_end = min(end_id, len(history) - 3)

    if safe_start > safe_end:
        return err(
            f" Cannot summarize: range [{start_id}:{safe_end}] "
            f"is invalid or overlaps with protected last 2 messages."
        )

    summary = summarize_dialogue(agent, start_id=safe_start, end_id=safe_end)
    if not summary:
        return err(" Summarization failed (empty response or error).")

    original_len = history.content_len(safe_start, safe_end)
    if len(summary) >= original_len:
        return err(
            f" Summarization produced text longer than original "
            f"({len(summary)} >= {original_len}). Nothing to compress."
        )

    # Оригиналы диапазона выселяем в архив (recall остаётся возможным),
    # затем заменяем их summary-сообщением. is_summary=True обязателен,
    # иначе последующие компакции не увидят этот блок как саммари.
    if Config.MEMORY_ARCHIVE_ENABLED and hasattr(agent, "archive"):
        agent.archive.append_messages(history.get_all()[safe_start:safe_end + 1])

    summary_content = ok(f" [SUMMARY of messages {safe_start}-{safe_end}]: {summary}\n")
    history.replace_range(
        safe_start, safe_end,
        [UserMessage(content=summary_content, is_summary=True)],
    )
    history.normalize()
    tm = getattr(agent, "tools_manager", None)
    if tm is not None:
        tm.flush_pending_unloads()

    freed = original_len - len(summary_content)
    return (
        f"{ENVIRONMENT_PREFIX} Successfully summarized "
        f"{safe_end - safe_start + 1} messages into 1. Freed ~{freed} chars."
        f"{ENVIRONMENT_PREFIX_END}"
    )

@tool(
    description="Delegates a task to a sub-agent that has access to the same tools and system prompt as you. "
                "The sub-agent inherits your conversation history for KV-cache reuse. "
                "It returns only its final result.",
    short_description="run task in sub-agent",
    task=("str", "Clear task description with all necessary context"),
    max_iter=("int", "Optional max tool calls for sub-agent"),
)
def delegate_to_subagent(agent: LLMAgent, task: str, max_iter: int = None) -> str:
    from universal_agents.sub_agent import run_subagent_once

    task_with_context = (
        "You are a sub-agent working on a specific subtask. "
        "Complete the task using tools if needed and provide a final answer. "
        "Do NOT ever ask clarifying questions — work with what you have.\n\n"
        f"Task:\n{task}"
    )
    return run_subagent_once(agent, task_with_context, temp=0.2, max_iter=max_iter)


@tool(
    description='load tool by its name / list loadable tools if no args passed or name="',
    short_description="load/list tools",
    name=("str", "Specific tool name to load"),
)
def load_tool(agent: LLMAgent, name: str = "") -> str:
    return agent.load_tool(name)


@tool(
    description="Marks a task from the plan as DONE. "
                "Call ONLY AFTER really performing actions with real tools you have. "
                "Do NOT mark a task done "
                "if you have not actually done the work. Must be called strictly "
                "in plan order (the system rejects out-of-order calls).",
    short_description="mark task done",
    id=("str", "Task id by the plan")
)
def have_done(agent: LLMAgent, id: str) -> str:
    from universal_agents.task_tracker import mark_task_done

    return mark_task_done(agent, id)


@tool(
    description="Plan in advance using make_plan to create a list of tasks/steps in execution order. "
                "Each entry is {\"id\", \"title\"}. Call this FIRST for any multi-step task, and call it again "
                "You can change the plan by creating a new one if something changed your mind.",
    short_description="make a plan",
    plan=("list", "List of {id, title} dicts, in execution order"),
)
def make_plan(agent: LLMAgent, plan: list) -> str:
    from universal_agents.task_tracker import set_plan

    return set_plan(agent, plan)
