from __future__ import annotations
from typing import TYPE_CHECKING

from universal_agents.tool import tool
from universal_agents.constants import ENVIRONMENT_PREFIX
from universal_agents.config import Config
from universal_agents.models import UserMessage, AssistantMessage, ToolResult, SystemMessage
from universal_agents.sub_agent import MAX_SUB_AGENT_DEPTH
from universal_agents.compressors import summarize_dialogue

if TYPE_CHECKING:
    from universal_agents.agent import LLMAgent


@tool(description="Get short indexed current history with ids",
       short_description="show history")
def get_messages(agent: LLMAgent, chars_per_message: int = 30) -> str:
    history = agent.history
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


@tool(
    description="Edits a specific message in the history",
    short_description="edit history",
    requires_confirmation=True,
    id=("int", "ID of the message to edit"),
    old=("str", "Optional exact substr to replace. Empty str replaces whole text"),
    new=("str", "Text to insert in place of old"),
)
def edit_message(agent: LLMAgent, id: int, new: str, old: str = '') -> str:
    return agent.history.edit_message(id, new, old)


@tool(
    description="Deletes a range of messages from dialog history",
    short_description="delete history",
    requires_confirmation=True,
    start_id=("int", "Starting message ID to delete"),
    end_id=("int", "Optional ending message ID (-1 for last)"),
)
def delete_messages(agent: LLMAgent, start_id: int, end_id: int = -1) -> str:
    return agent.history.delete_range(start_id, end_id)


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
        return (
            f"{ENVIRONMENT_PREFIX} Error Cannot summarize: range [{start_id}:{safe_end}] "
            f"is invalid or overlaps with protected last 2 messages."
        )

    summary = summarize_dialogue(agent, start_id=safe_start, end_id=safe_end)
    if not summary:
        return f"{ENVIRONMENT_PREFIX} Error Summarization failed (empty response or error)."

    original_len = history.content_len(safe_start, safe_end)
    if len(summary) >= original_len:
        return (
            f"{ENVIRONMENT_PREFIX} Error Summarization produced text longer than original "
            f"({len(summary)} >= {original_len}). Nothing to compress."
        )

    summary_content = f"{ENVIRONMENT_PREFIX} [SUMMARY of messages {safe_start}-{safe_end}]: {summary}"
    history.replace_range(safe_start, safe_end, [UserMessage(content=summary_content)])
    history.normalize()

    freed = original_len - len(summary_content)
    return (
        f"{ENVIRONMENT_PREFIX} Successfully summarized "
        f"{safe_end - safe_start + 1} messages into 1. Freed ~{freed} chars."
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
    depth = getattr(agent, '_depth', 0)
    if depth >= MAX_SUB_AGENT_DEPTH:
        return f"{ENVIRONMENT_PREFIX} Error Sub-agent depth limit ({MAX_SUB_AGENT_DEPTH}) reached. You can't delegate to sub-agent. Do it yourself."

    sub_plugins = {}
    for name, tool_info in agent._all_tools.items():
        sub_plugins[name] = tool_info["handler"]

    task_with_context = (
        "You are a sub-agent working on a specific subtask. "
        "Complete the task using tools if needed and provide a final answer. "
        "Do NOT ever ask clarifying questions — work with what you have.\n\n"
        f"Task:\n{task}"
    )

    sub = agent.make_sub_agent(
        tools_config=agent._tools_config,
        external_plugins=sub_plugins,
        safe_only=False,
        max_iter=max_iter,
        temp=0.2,
        on_log=agent.on_system_msg,
        depth=depth + 1,
    )

    agent.on_system_msg(f"[DELEGATE] Starting sub-agent for: {task}...")
    result = sub.run(task_with_context)
    agent.on_system_msg(f"[DELEGATE] Completed. Tokens spent by sub-agent: {sub.tokens_spent}")

    if not result.strip():
        return f"{ENVIRONMENT_PREFIX} Error Sub-agent returned empty result."
    return f"{ENVIRONMENT_PREFIX} Sub-agent result:\n{result}"


@tool(
    description='load tool by its name / list loadable tools if no args passed or name="',
    short_description="load/list tools",
    name=("str", "Optional specific tool name to load"),
)
def load_tools(agent: LLMAgent, name: str = "") -> str:
    if not name:
        return agent.list_available_tools()
    return agent.load_tools(name)


@tool(
    description="Disable a currently loaded tool by name. Cannot disable core tools like load_tools, unload_tool.",
    short_description="unload tool",
    name=("str", "Name of the tool to disable"),
)
def unload_tool(agent: LLMAgent, name: str) -> str:
    return agent.unload_tool(name)


@tool(
    description="Marks a task from the plan as DONE. "
                "Available ONLY AFTER a successful create_plan (it is dynamically loaded "
                "once the plan is set). Call ONLY AFTER really performing the task with real tools "
                "(read/search/edit_file/run_bash etc.). Do NOT mark a task done "
                "if you have not actually done the work. Must be called strictly "
                "in plan order (the system rejects out-of-order calls).",
    short_description="mark task done",
    id=("str", "Task id from the plan (e.g. 't1')")
)
def have_done(agent: LLMAgent, id: str) -> str:
    from universal_agents.task_tracker import mark_task_done

    return mark_task_done(agent, id)


@tool(
    description="Creates or revises the task plan: an ORDERED FLAT list of tasks to execute "
                "(like a bullet list). Each entry is {\"id\", \"title\"}. The list order IS the "
                "execution order. Execute each task for real with real tools, then mark it done "
                "with have_done. Call this FIRST for any multi-step task, and call it again "
                "to REVISE the plan (allows starting from a different step; execution then "
                "continues in the new plan's order).",
    short_description="create/revise task plan",
    plan=("list", "Ordered list of {id, title} dicts, in execution order"),
)
def create_plan(agent: LLMAgent, plan: list) -> str:
    from universal_agents.task_tracker import set_plan

    return set_plan(agent, plan)
