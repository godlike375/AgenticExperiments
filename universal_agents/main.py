import os

from universal_agents.agent import LLMAgent
from universal_agents.tool_registry import load_external_plugins
from universal_agents.ui import ConsoleUI, CLI
from universal_agents.constants import ENVIRONMENT_PREFIX
from universal_agents.project_root import find_project_root
from universal_agents.task_tracker import TASK_MARK_INSTRUCTIONS

if __name__ == "__main__":
    tools_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools")
    all_tools = load_external_plugins(tools_dir)
    startup_tools = {n: f for n, f in all_tools.items() if n in ("load_tools", "task_mark_done", "create_plan")}
    print(f"Loaded startup tools: {list(startup_tools.keys())}")
    print("Use load_tools to load tools dynamically.")

    project_root = find_project_root()
    root_line = (
        f"Current project root & working dir: '{project_root}'"
        if project_root
        else "Current project root & working dir: (not found - no .git upwards)"
    )

    sys_prompt = (
        "* You are tool-calling LLM-assistant.\n"
        "* You are in a special program environment to use tools.\n"
        f"* {root_line}\n"
        f"* '{ENVIRONMENT_PREFIX}' prefix means system says something.\n"
        "* Use 'load_tools' without args only 1 time.\n"
        "* Do NOT repeat identical tool calls with same arguments twice. You can call only 1 tool at 1 turn (message). "
        "So you must wait for tool results before making any next call. "
        "Говори только на русском."
        f"{TASK_MARK_INSTRUCTIONS}"
    )

    agent = LLMAgent(
        system_prompt=sys_prompt,
        tools_config=['load_tools', 'read', 'edit_file', 'cwd', 'search', 'get_messages', 'run_bash_host', 'run_powershell', 'task_mark_done', 'create_plan'],
        external_plugins=startup_tools,
        on_render=ConsoleUI.render_message,
        on_confirm=ConsoleUI.confirm_action,
        on_system_msg=ConsoleUI.system_msg,
        on_stream_chunk=ConsoleUI.stream_chunk,
        on_stream_start=ConsoleUI.start_stream,
        on_stream_end=ConsoleUI.end_stream,
        on_reasoning_chunk=ConsoleUI.stream_reasoning_chunk,
        on_reasoning_start=ConsoleUI.start_reasoning,
        on_reasoning_end=ConsoleUI.end_reasoning,
    )

    cli = CLI(agent)
    cli.run()
