from universal_agents.agent import LLMAgent
from universal_agents.tool_registry import load_external_plugins
from universal_agents.ui import ConsoleUI, CLI
from universal_agents.tool import ENVIRONMENT_PREFIX

if __name__ == "__main__":
    all_tools = load_external_plugins("tools")
    startup_tools = {n: f for n, f in all_tools.items() if n == "load_tools"}
    print(f"Loaded startup tools: {list(startup_tools.keys())}")
    print("Use load_tools to load tools dynamically.")

    sys_prompt = (
        "* You are Russian speaking tool-calling assistant.\n"
        "* You are in a special program environment to use tools.\n"
        f"* '{ENVIRONMENT_PREFIX}' prefix means system says something.\n"
        "* Use 'load_tools' without args only 1 time.\n"
        "* Do NOT repeat identical tool calls with same arguments twice. You can call only 1 tool at 1 turn (message). "
        "So you must wait for tool results before making any next call."
    )

    agent = LLMAgent(
        system_prompt=sys_prompt,
        tools_config=None,
        external_plugins=startup_tools,
        on_render=ConsoleUI.render_message,
        on_confirm=ConsoleUI.confirm_action,
        on_system_msg=ConsoleUI.system_msg,
        on_stream_chunk=ConsoleUI.stream_chunk,
        on_stream_start=ConsoleUI.start_stream,
        on_stream_end=ConsoleUI.end_stream,
    )

    cli = CLI(agent)
    cli.run()
