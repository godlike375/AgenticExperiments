from agent import LLMAgent, load_external_plugins
from ui import ConsoleUI, CLI
from universal_agents.tool import ENVIRONMENT_PREFIX

if __name__ == "__main__":
    external_tools = load_external_plugins("tools")
    print(f"Loaded external tools: {list(external_tools.keys())}")

    sys_prompt = (
        "* You are assistant that can call tools. Speak Russian.\n"
        "* You're launched in a special environment.\n"
        f"{ENVIRONMENT_PREFIX} prefix means the system outputs something, it's not what user said.\n"
        f"When remaining tokens amount is about equal to spent tokens, it's time to start cleaning up your context. "
        f"You can compress or delete some messages to free some more tokens to continue working. Avoid calling any tool twice in a row. "
        f"You can call only 1 tool at a turn. So you must wait for the tool results before making any next call."
    )

    agent = LLMAgent(
        system_prompt=sys_prompt,
        tools_config=None,
        external_plugins=external_tools,
        on_render=ConsoleUI.render_message,
        on_confirm=ConsoleUI.confirm_action,
        on_system_msg=ConsoleUI.system_msg,
        max_context_tokens=65535
    )

    cli = CLI(agent)
    cli.run()
