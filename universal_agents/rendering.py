"""Презентационный слой: рендеринг сообщений для консоли."""

from universal_agents.models import Message, SystemMessage, UserMessage, AssistantMessage, ToolResult


def render_message(msg: Message, label: str = "Agent") -> str:
    """Возвращает строку для вывода сообщения в консоль."""
    if isinstance(msg, SystemMessage):
        return ""
    if isinstance(msg, UserMessage):
        return f"👤 User: {msg.content}"
    if isinstance(msg, AssistantMessage):
        parts = []
        if msg.reasoning_content and not msg.streamed:
            parts.append(f"📝 {label} [reasoning]: {msg.reasoning_content}")
        if msg.content.strip() and not msg.streamed:
            parts.append(f"🤖 {label}: {msg.content}")
        for tc in msg.tool_calls:
            parts.append(f"🛠️ [{label} Tool Call: {tc.name}({tc.arguments})]")
        return "\n".join(parts)
    if isinstance(msg, ToolResult):
        prefix = "❌" if msg.is_error else "✅"
        display = str(msg.content)
        return f"{prefix} [{label} Result '{msg.name}']: {display}"
    return ""
