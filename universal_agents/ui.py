import json
import os
import shlex
from universal_agents.models import Message, SystemMessage, UserMessage, AssistantMessage, ToolResult
from universal_agents.rendering import render_message
from universal_agents.agent import LLMAgent

class ConsoleUI:
    _reasoning_started = False
    _answer_header_shown = False

    @staticmethod
    def render_message(msg: Message, label: str = "Agent"):
        if isinstance(msg, SystemMessage):
            return
        output = render_message(msg, label=label)
        if output:
            print(output)

    @staticmethod
    def system_msg(text: str):
        if text:
            print(f"\n⚙️ [System]: {text}")

    @staticmethod
    def confirm_action(name: str, args: dict) -> bool:
        print(f"\n[WARNING] Tool '{name}' modifies state")

        if args:
            formatted_args = json.dumps(args, indent=2, ensure_ascii=False)
            print(f"Arguments:\n{formatted_args}")
        else:
            print("Arguments: {} (None)")

        resp = input("Execute? (y/n): ").strip().lower()
        return resp == 'y'
    
    @staticmethod
    def start_stream():
        """Начало streaming вывода"""
        print('\n' + '=' * 15)
        print("🤖 Agent: ", end="", flush=True)
        ConsoleUI._reasoning_started = False
        ConsoleUI._answer_header_shown = False

    @staticmethod
    def stream_chunk(chunk: str):
        """Вывод чанка streaming"""
        if not ConsoleUI._answer_header_shown:
            if ConsoleUI._reasoning_started:
                print()
            print("💬 Answer: ", end="", flush=True)
            ConsoleUI._answer_header_shown = True
        print(chunk, end="", flush=True)

    @staticmethod
    def end_stream():
        """Завершение streaming вывода"""
        print('\n' + '=' * 15)

    @staticmethod
    def start_reasoning():
        """Начало streaming вывода reasoning"""
        print("\n📝 Reasoning: ", end="", flush=True)
        ConsoleUI._reasoning_started = True

    @staticmethod
    def stream_reasoning_chunk(chunk: str):
        """Вывод чанка reasoning"""
        print(chunk, end="", flush=True)

    @staticmethod
    def end_reasoning():
        """Завершение streaming вывода reasoning"""
        ConsoleUI._reasoning_started = False

class CLI:
    def __init__(self, agent: LLMAgent):
        self.agent = agent
        self.pending_prefill = None
        self.multiline = False
        self.commands = {
            "/regen": self.cmd_regen,
            "/prefill": self.cmd_prefill,
            "/save": self.cmd_save,
            "/load": self.cmd_load,
            "/consistent": self.cmd_consistent,
            "/multiline": self.cmd_multiline,
            "/trust": self.cmd_trust,
            "/untrust": self.cmd_untrust,
        }

    def cmd_regen(self, parts: list[str]):
        n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
        for _ in range(n - 1):
            self.agent.history.pop_until_user()
        user_msg = self.agent.history.pop_until_user()
        self.agent.file_states.prune()
        if not user_msg:
            ConsoleUI.system_msg("Cannot find a preceding user message to regenerate")
            return
        ConsoleUI.system_msg(f"Regenerating response for: '{user_msg}'")
        self.agent.chat(user_msg, prefill=self.pending_prefill)

    def cmd_prefill(self, parts: list[str]):
        if len(parts) > 1:
            self.pending_prefill = parts[1]
            ConsoleUI.system_msg(f"Next message will start with prefill: '{self.pending_prefill}'")
        else:
            self.pending_prefill = None
            ConsoleUI.system_msg("Prefill cleared")

    def cmd_save(self, parts: list[str]):
        filename = parts[1] if len(parts) > 1 else "default_history.json"
        try:
            tool_names = list(self.agent._all_tools.keys())
            self.agent.history.save(
                filename,
                loaded_tools=tool_names,
                file_states=self.agent.file_states.to_dict(),
                per_msg_summaries=self.agent._per_msg_summaries,
            )
            ConsoleUI.system_msg(f"History saved to '{filename}' (tools: {tool_names}, file_states: {len(self.agent.file_states)})")
        except Exception as e:
            ConsoleUI.system_msg(f"Error saving history: {e}")

    def cmd_load(self, parts: list[str]):
        filename = parts[1] if len(parts) > 1 else "default_history.json"
        if not os.path.exists(filename):
            ConsoleUI.system_msg(f"File '{filename}' not found")
            return
        try:
            loaded_tools, file_states, summaries = self.agent.history.load(filename)
            self.agent.file_states.from_dict(file_states)
            # Перепривязываем сохранённые саммари рабочей памяти к пересозданным
            # объектам сообщений (ключ id() меняется после загрузки).
            self.agent._per_msg_summaries = {}
            loaded_msgs = self.agent.history.get_all()
            for i, s in enumerate(summaries):
                if s and i < len(loaded_msgs):
                    self.agent._per_msg_summaries[id(loaded_msgs[i])] = s
            self.agent.rebuild_tool_usage()
            for name in loaded_tools:
                if name not in self.agent._all_tools:
                    self.agent.load_tools(name)
            ConsoleUI.system_msg(f"History loaded. Total messages: {len(self.agent.history)}. Tools restored: {loaded_tools}")
            print("\n" + "="*40 + "\n🔄 LOADED HISTORY:\n" + "="*40)
            for msg in self.agent.history.get_all():
                ConsoleUI.render_message(msg)
        except Exception as e:
            ConsoleUI.system_msg(f"Error loading history: {e}")

    def cmd_consistent(self, parts: list[str]):
        self.agent.self_consistency_mode = not self.agent.self_consistency_mode
        status = "ON" if self.agent.self_consistency_mode else "OFF"
        ConsoleUI.system_msg(f"Self-consistency mode turned {status}")

    def cmd_multiline(self, parts: list[str]):
        self.multiline = not self.multiline
        status = "ON" if self.multiline else "OFF"
        ConsoleUI.system_msg(f"Multiline input mode turned {status}. Type Ctrl+D to finish the input.")

    def cmd_trust(self, parts: list[str]):
        if len(parts) < 2:
            if self.agent.trusted_dirs:
                dirs = "\n".join(f"  {d}" for d in sorted(self.agent.trusted_dirs))
                ConsoleUI.system_msg(f"Trusted directories:\n{dirs}")
            else:
                ConsoleUI.system_msg("No trusted directories. Usage: /trust <directory>")
            return
        path = parts[1]
        if path == "--clear":
            self.agent.trusted_dirs.clear()
            ConsoleUI.system_msg("All trusted directories cleared")
            return
        result = self.agent.trust_dir(path)
        ConsoleUI.system_msg(result)

    def cmd_untrust(self, parts: list[str]):
        if len(parts) < 2:
            ConsoleUI.system_msg("Usage: /untrust <directory>")
            return
        result = self.agent.untrust_dir(parts[1])
        ConsoleUI.system_msg(result)

    def read_until_marker(self, marker="/mm"):
        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line.strip() == marker:
                break
            lines.append(line)
        return "\n".join(lines)

    def run(self):
        ConsoleUI.system_msg("Ready. Type 'exit' to quit")
        ConsoleUI.system_msg(f"Commands: {', '.join(self.commands.keys())}")
        while True:
            if self.multiline:
                print("\n👤 User: ")
            inp = self.read_until_marker() if self.multiline else input("\n👤 User: ").strip()
            if self.multiline:
                self.multiline = False
            if not inp: continue
            if inp.lower() in ("exit", "quit"): break
            if inp.startswith("/"):
                try: parts = shlex.split(inp)
                except ValueError as e:
                    ConsoleUI.system_msg(f"Error parsing command: {e}")
                    continue
                handler = self.commands.get(parts[0].lower())
                if handler: handler(parts)
                else: ConsoleUI.system_msg(f"Unknown command: {parts[0]}")
                continue
            self.agent.chat(inp, prefill=self.pending_prefill)
