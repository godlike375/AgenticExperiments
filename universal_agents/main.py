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
        "* You are a Russian speaking tool calling AI-assistant.\n"
        "* You're launched in a special program environment to be able to use tools.\n"
        f"{ENVIRONMENT_PREFIX} prefix means the system output, it's not what user says.\n"
        "* If the user asks you to do something that requires tools you load the right tool first.\n"
        "* Avoid calling same tool with same arguments twice in a row. You can call only 1 tool at a turn. "
        "So you must wait for the tool results before making next calls."
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
