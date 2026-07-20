from universal_agents.agent import LLMAgent
from universal_agents.tool_registry import load_external_plugins
from universal_agents.ui import ConsoleUI, CLI
from universal_agents.tool import ENVIRONMENT_PREFIX

if __name__ == "__main__":
    all_tools = load_external_plugins("tools")
    startup_tools = {n: f for n, f in all_tools.items() if n == "load_tools"}
    print(f"Loaded startup tools: {list(startup_tools.keys())}")
    print("Use load_tools to load tools and tool_description dynamically.")

    sys_prompt = (
        "* You are Russian speaking tool-calling assistant.\n"
        "* You are in a special program environment to use tools.\n"
        f"* '{ENVIRONMENT_PREFIX}' prefix means system says something.\n"
        # "* If user asks you to do something requiring tools then call load_tools first.\n"
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
        max_context_tokens=50000
    )

    cli = CLI(agent)
    cli.run()
