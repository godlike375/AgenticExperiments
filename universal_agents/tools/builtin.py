from __future__ import annotations
from typing import TYPE_CHECKING

from universal_agents.tool import tool, ENVIRONMENT_PREFIX
from universal_agents.config import Config
from universal_agents.models import UserMessage, AssistantMessage, ToolResult, SystemMessage
from universal_agents.sub_agent import SubAgent

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
    from universal_agents.compressors import summarize_text

    history = agent.history
    if end_id == -1 or end_id >= len(history):
        end_id = len(history) - 3

    safe_start = max(start_id, Config.AFTER_SYSTEM_PROMPT)
    safe_end = min(end_id, len(history) - 3)

    if safe_start > safe_end:
        return (
            f"{ENVIRONMENT_PREFIX} Cannot summarize: range [{start_id}:{end_id}] "
            f"is invalid or overlaps with protected last 2 messages."
        )

    lines = []
    for i in range(safe_start, safe_end + 1):
        msg = history[i]
        role = type(msg).__name__.replace("Message", "").upper()
        role = role if role != 'ASSISTANT' else 'AI'
        content = getattr(msg, 'content', str(msg))
        lines.append(f"{role}: {content}")

    raw_text = "\n".join(lines)
    summary = summarize_text(agent, raw_text)
    if not summary:
        return f"{ENVIRONMENT_PREFIX} Summarization failed (empty response or error)."

    summary_content = f"{ENVIRONMENT_PREFIX} [SUMMARY of messages {safe_start}-{safe_end}]: {summary}"
    del history._messages[safe_start:safe_end + 1]
    history._messages.insert(safe_start, UserMessage(content=summary_content))
    history.normalize()

    freed = len(raw_text) - len(summary_content)
    return (
        f"{ENVIRONMENT_PREFIX} Successfully summarized "
        f"{safe_end - safe_start + 1} messages into 1. Freed ~{freed} chars."
    )


@tool(
    description="Delegates a task to a limited sub-agent that has its own dialog history and just read-only tools unlike you. "
                "You can delegate to it, for example, 1 step of a multi-step task. "
                "Include necessary context for execution in task description. "
                "The tool returns only the final result of a task.",
    short_description="run task in sub-agent",
    task=("str", "Clear task description with all necessary context"),
    max_iter=("int", "Optional max tool calls for sub-agent"),
)
def delegate_to_subagent(agent: LLMAgent, task: str, max_iter: int = None) -> str:
    sub_plugins = {}
    for name, tool_info in agent._all_tools.items():
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
        max_context_tokens=agent.token_tracker.max_context_tokens // 3,
        tools_config={"exclude": [
            "edit_message", "delete_messages", "summarize_messages",
            "delegate_to_subagent", "run_bash"
        ]},
        external_plugins=sub_plugins,
        safe_only=True,
        max_iter=max_iter,
        temp=0.1,
        on_log=agent.on_system_msg,
    )

    agent.on_system_msg(f"[DELEGATE] Starting sub-agent for: {task[:100]}...")
    result = sub.run(task)
    agent.on_system_msg(f"[DELEGATE] Completed. Tokens spent by sub-agent: {sub.tokens_spent}")

    if not result.strip():
        return f"{ENVIRONMENT_PREFIX} Sub-agent returned empty result."
    return f"{ENVIRONMENT_PREFIX} Sub-agent result:\n{result}"


@tool(
    description="load tool by its name / list loadable tools if no args.",
    short_description="load/list tools",
    name=("str", "Optional tool name to load"),
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
