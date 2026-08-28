import argparse
import os
from typing import Iterable

from universal_agents.agent import LLMAgent
from universal_agents.tool_registry import load_external_plugins
from universal_agents.tool_manager import ToolManager
from universal_agents.ui import ConsoleUI, CLI
from universal_agents.constants import ENVIRONMENT_PREFIX, ENVIRONMENT_PREFIX_END
from universal_agents.project_root import find_project_root, set_project_root
from universal_agents.config import Config
from universal_agents.task_tracker import TASK_MARK_INSTRUCTIONS

LOADABLE_TOOLS = [
    'run_bash_host'
]

PRELOADED_TOOLS = ("load_tool", "make_plan", "have_done", 'read', 'edit_file', 'cwd', 'search',
                   'run_powershell')


def build_allowed_tools(loadable: Iterable[str], preloaded: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(list(loadable) + list(preloaded)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Universal tool-calling LLM assistant."
    )
    parser.add_argument(
        "--project-root", "-r",
        default=None,
        help="Явно задать директорию корня проекта (перекрывает авто-поиск по .git). "
             "Можно также задать Config.PROJECT_ROOT в universal_agents/config.py.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Приоритет: CLI --project-root > Config.PROJECT_ROOT > авто-поиск.
    explicit_root = args.project_root or getattr(Config, "PROJECT_ROOT", None)
    if explicit_root:
        if not os.path.isdir(explicit_root):
            print(f"WARNING: --project-root '{explicit_root}' не существует, "
                  f"используется авто-поиск корня.")
        else:
            set_project_root(explicit_root)
            os.chdir(explicit_root)
            print(f"Рабочая директория установлена в корень проекта: {os.getcwd()}")

    tools_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools")
    all_tools = load_external_plugins(tools_dir)

    startup_tools = {
        n: f for n, f in all_tools.items()
        if n in PRELOADED_TOOLS
    }
    print(f"Loaded startup tools: {sorted(startup_tools.keys())}")
    print("Use load_tool to load additional tools dynamically.")

    project_root = find_project_root() or '(not found - no .git upwards)'
    root_line = f"Current project root: {project_root}, current working dir: {os.getcwd()}"

    # Предзагруженные инструменты всегда разрешены к использованию, даже если их
    # нет в LOADABLE_TOOLS (который задаёт ДОПОЛНИТЕЛЬНО доступные для загрузки).
    # Объединяем их в allow-список, чтобы ToolManager не отфильтровал стартовые
    # инструменты и чтобы load_tool мог подключать LOADABLE_TOOLS.
    allowed_tools = build_allowed_tools(LOADABLE_TOOLS, PRELOADED_TOOLS)

    # Список загружаемых (пока не активных) инструментов — встраиваем в системный
    # промпт, чтобы модель знала, что можно подключить через load_tool.
    tools_manager = ToolManager(tools_config=allowed_tools, external_plugins=startup_tools)
    available_tools_text = tools_manager.list_available()
    loaded_tools_text = tools_manager.list_loaded()

    sys_prompt = (
        f"{ENVIRONMENT_PREFIX} * You are tool-calling LLM-assistant.\n"
        "* You are launched in a custom environment to be able use tools.\n"
        f"* {root_line}\n"
        f"* '{ENVIRONMENT_PREFIX}' prefix means system says something.\n"
        f"* Already loaded tools you can use:\n{loaded_tools_text}\n"
        f"* Loadable tools (load with 'load_tool' + 'name' arg):\n{available_tools_text}\n"
        "* Do NOT repeat identical tool calls with same arguments twice. You can call only 1 tool at 1 turn (message). "
        "So you must wait for tool results before making any next call. "
        "Говори только на русском."
        f"{TASK_MARK_INSTRUCTIONS}"
        f"{ENVIRONMENT_PREFIX_END}"
    )

    agent = LLMAgent(
        system_prompt=sys_prompt,
        tools_config=allowed_tools,
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
        on_service_stream_start=ConsoleUI.start_service_stream,
        on_service_stream_chunk=ConsoleUI.service_stream_chunk,
        on_service_stream_end=ConsoleUI.end_service_stream,
    )

    cli = CLI(agent)
    cli.run()
