import json
import os
import shlex
import sys
import threading
import time
import collections
from universal_agents.models import Message, SystemMessage, UserMessage, AssistantMessage, ToolResult
from universal_agents.rendering import render_message
from universal_agents.agent import LLMAgent
from universal_agents.exceptions import GenerationInterrupted
from universal_agents.project_root import set_project_root, get_project_root_override

class ConsoleUI:
    _reasoning_started = False
    _answer_header_shown = False
    _service_stream_shown = False
    # Функция ввода строки (переопределяется CLI на общую очередь ввода,
    # чтобы и подтверждения, и промпты читали stdin через единый поток).
    _input_fn = staticmethod(input)

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

        resp = ConsoleUI._input_fn("Execute? (y/n): ").strip().lower()
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

    @staticmethod
    def start_service_stream():
        """Начало стрима служебного вызова LLM. Отдельный канал со своей меткой, чтобы служебный текст не выглядел вторым ответом; метка печатается лениво при первом чанке."""
        ConsoleUI._service_stream_shown = False

    @staticmethod
    def service_stream_chunk(chunk: str):
        if not ConsoleUI._service_stream_shown:
            print("⚙️ [llm-service] ", end="", flush=True)
            ConsoleUI._service_stream_shown = True
        print(chunk, end="", flush=True)

    @staticmethod
    def end_service_stream():
        if ConsoleUI._service_stream_shown:
            print()

class CLI:
    def __init__(self, agent: LLMAgent):
        self.agent = agent
        self.pending_prefill = None
        self.multiline = False
        self._monitor_active = False
        self.commands = {
            "/regen": self.cmd_regen,
            "/list": self.cmd_list,
            "/clear": self.cmd_clear,
            "/prefill": self.cmd_prefill,
            "/save": self.cmd_save,
            "/load": self.cmd_load,
            "/consistent": self.cmd_consistent,
            "/multiline": self.cmd_multiline,
            "/trust": self.cmd_trust,
            "/untrust": self.cmd_untrust,
        }

    def cmd_regen(self, parts: list[str]):
        history = self.agent.history
        msgs = history.get_all()
        assistant_idxs = [i for i, m in enumerate(msgs) if isinstance(m, AssistantMessage)]
        if not assistant_idxs:
            ConsoleUI.system_msg("No assistant messages to regenerate")
            return

        target_idx = None
        if len(parts) > 1:
            arg = parts[1]
            if arg.startswith('@'):
                try:
                    seq = int(arg[1:])
                except ValueError:
                    ConsoleUI.system_msg("Invalid seq after '@'")
                    return
                for i in assistant_idxs:
                    if getattr(msgs[i], "seq", None) == seq:
                        target_idx = i
                        break
                if target_idx is None:
                    ConsoleUI.system_msg(f"No assistant message with seq {seq}")
                    return
            elif arg.isdigit():
                k = int(arg)
                if k < 1 or k > len(assistant_idxs):
                    ConsoleUI.system_msg(
                        f"Invalid back-count {k} (have {len(assistant_idxs)} assistant messages). "
                        f"Use /list to see seqs."
                    )
                    return
                target_idx = assistant_idxs[len(assistant_idxs) - k]
            else:
                ConsoleUI.system_msg("Usage: /regen [k|@seq]  (k=1 last, @seq by message seq)")
                return

        if target_idx is None:
            # Default: regenerate the most recent assistant response.
            user_msg = history.pop_until_user()
            self.agent._on_history_changed()
            if not user_msg:
                ConsoleUI.system_msg("Cannot find a preceding user message to regenerate")
                return
            ConsoleUI.system_msg(f"Regenerating latest response for: '{user_msg}'")
            self._call_chat(user_msg, prefill=self.pending_prefill)
            return

        # Regenerate from a previous assistant response: drop it and everything after,
        # then re-answer the user message that preceded it.
        user_idx = None
        for j in range(target_idx - 1, -1, -1):
            if isinstance(msgs[j], UserMessage):
                user_idx = j
                break
        if user_idx is None:
            ConsoleUI.system_msg("No preceding user message found")
            return
        history.delete_range(target_idx, len(msgs) - 1)
        user_msg = history.get_all()[user_idx].content
        self.agent._on_history_changed()
        seq = getattr(msgs[target_idx], "seq", None)
        ConsoleUI.system_msg(f"Regenerating from previous assistant turn (seq={seq}).")
        self._call_chat(user_msg, prefill=self.pending_prefill)

    def cmd_list(self, parts: list[str]):
        msgs = self.agent.history.get_all()
        print("\n" + "=" * 50 + "\n📜 HISTORY:")
        for i, m in enumerate(msgs):
            role = type(m).__name__.replace("Message", "")
            seq = getattr(m, "seq", None)
            content = (getattr(m, "content", "") or "").replace("\n", " ")
            preview = content[:90]
            extra = ""
            if hasattr(m, "tool_calls") and m.tool_calls:
                extra = " [" + ", ".join(tc.name for tc in m.tool_calls) + "]"
            print(f"{i:3d} | seq={seq} | {role:9s} | {preview}{extra}")
        print("=" * 50)

    def cmd_clear(self, parts: list[str]):
        self.agent.reset_dialog()
        ConsoleUI.system_msg(
            f"Dialog cleared. New autosave session: {os.path.basename(self.agent._autosave_path)}"
        )

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
            from universal_agents.task_tracker import plan_state_to_dict
            tool_names = list(self.agent._all_tools.keys())
            extras = {
                "archive": self.agent.archive.to_list(),
                "plan_state": plan_state_to_dict(self.agent),
                "pending_pins": list(getattr(self.agent, "_pending_pins", [])),
                "cwd": os.getcwd(),
                "project_root": get_project_root_override(),
            }
            self.agent.history.save(
                filename,
                loaded_tools=tool_names,
                file_states=self.agent.file_states.to_dict(),
                extras=extras,
            )
            ConsoleUI.system_msg(
                f"History saved to '{filename}' (tools: {tool_names}, "
                f"file_states: {len(self.agent.file_states)}, archive: {len(self.agent.archive)})"
            )
        except Exception as e:
            ConsoleUI.system_msg(f"Error saving history: {e}")

    def cmd_load(self, parts: list[str]):
        filename = parts[1] if len(parts) > 1 else "default_history.json"
        if not os.path.exists(filename):
            ConsoleUI.system_msg(f"File '{filename}' not found")
            return
        try:
            from universal_agents.archive import HistoryArchive
            from universal_agents.task_tracker import restore_plan_state
            loaded_tools, file_states, summaries = self.agent.history.load(filename)
            extras = self.agent.history.last_loaded_extras
            self.agent.file_states.from_dict(file_states)
            self.agent.archive = HistoryArchive.from_list(extras.get("archive") or [])
            restore_plan_state(self.agent, extras.get("plan_state"))
            self.agent._pending_pins = [str(x) for x in (extras.get("pending_pins") or [])]
            saved_root = extras.get("project_root")
            if saved_root:
                set_project_root(saved_root)
            saved_cwd = extras.get("cwd")
            if saved_cwd and os.path.isdir(saved_cwd):
                os.chdir(saved_cwd)
            self.agent.rebuild_tool_usage()
            for name in loaded_tools:
                if name not in self.agent._all_tools:
                    self.agent.load_tool(name)
            ConsoleUI.system_msg(
                f"History loaded. Total messages: {len(self.agent.history)}. "
                f"Tools restored: {loaded_tools}. Archive entries: {len(self.agent.archive)}. "
                f"CWD: {os.getcwd()}"
            )
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

    def _request_stop(self):
        """Единая точка остановки: ставит флаг и разблокирует возможный ожидающий ввод в воркере."""
        self.agent.request_stop()
        self._inject_line("\n")
        ConsoleUI.system_msg("⏹ Stop requested — finishing current step…")

    # --------------------------------------------------------
    # Единый ввод из stdin через очередь (работает в PyCharm и нативных консолях)
    # --------------------------------------------------------
    def _start_stdin_reader(self):
        self._line_queue: collections.deque = collections.deque()
        self._line_lock = threading.Lock()
        self._line_cond = threading.Condition(self._line_lock)
        self._stdin_eof = False
        self._reader_thread = threading.Thread(target=self._stdin_reader, daemon=True)
        self._reader_thread.start()
        # Подтверждения инструментов тоже читают через общую очередь ввода.
        ConsoleUI._input_fn = self._get_line

    def _stdin_reader(self):
        while True:
            try:
                line = sys.stdin.readline()
            except Exception:
                line = ''
            with self._line_lock:
                if line == '':
                    self._stdin_eof = True
                    self._line_cond.notify_all()
                    break
                self._line_queue.append(line)
                self._line_cond.notify_all()

    def _get_line(self) -> str:
        """Блокирующе читает следующую строку из общей очереди ввода (единственный читатель stdin)."""
        with self._line_cond:
            while not self._line_queue and not self._stdin_eof:
                self._line_cond.wait(timeout=0.2)
            if self._line_queue:
                return self._line_queue.popleft()
            return ''

    def _poll_stop(self) -> bool:
        """Неразрушающе проверяет очередь на стоп-команду (забирает только если это стоп)."""
        with self._line_lock:
            if self._line_queue:
                head = self._line_queue[0].strip().lower()
                if head in ('q', 'stop', 'exit-gen', '!q'):
                    self._line_queue.popleft()
                    return True
            return False

    def _inject_line(self, line: str):
        """Подкладывает строку в очередь (чтобы разблокировать ожидающий ввод в воркере при остановке)."""
        with self._line_lock:
            self._line_queue.append(line)
            self._line_cond.notify_all()

    def _call_chat(self, message: str, prefill=None):
        """Генерация идёт в отдельном потоке; основной поток опрашивает очередь ввода.
        Остановка (универсально, работает везде, включая консоль PyCharm): введите
        'q' на новой строке и нажмите Enter — прервётся и текущий вызов инструмента."""
        ConsoleUI.system_msg("💡 To stop generation, type 'q' and press Enter.")
        self._monitor_active = True
        self.agent.clear_stop()
        gen = threading.Thread(target=self._run_chat, args=(message,), kwargs={'prefill': prefill}, daemon=True)
        gen.start()
        stopped = False
        try:
            while gen.is_alive():
                if self.agent.stop_event.is_set():
                    stopped = True
                    break
                if self._poll_stop():
                    self._request_stop()
                    stopped = True
                    break
                time.sleep(0.05)
        finally:
            gen.join()
            self._monitor_active = False

    def _run_chat(self, message: str, prefill=None):
        try:
            self.agent.chat(message, prefill=prefill)
        except GenerationInterrupted:
            # Уже выведено в agent.chat; управление возвращается пользователю.
            pass
        except Exception as e:
            ConsoleUI.system_msg(f"[chat error] {e}")

    def read_until_marker(self, marker="/mm"):
        lines = []
        while True:
            line = self._get_line()
            if line == '':
                break
            if line.strip() == marker:
                break
            lines.append(line.rstrip('\n'))
        return "\n".join(lines)

    def run(self):
        ConsoleUI.system_msg("Ready. Type 'exit' to quit")
        ConsoleUI.system_msg(f"Commands: {', '.join(sorted(self.commands.keys()))}")
        self._start_stdin_reader()
        while True:
            try:
                if self.multiline:
                    print("\n👤 User: ")
                    inp = self.read_until_marker().strip()
                else:
                    inp = self._get_line().strip()
            except KeyboardInterrupt:
                break
            if self._stdin_eof:
                break
            if self.multiline:
                self.multiline = False
            if not inp:
                continue
            if inp.lower() in ("exit", "quit"):
                break
            if inp.startswith("/"):
                try:
                    parts = shlex.split(inp)
                except ValueError as e:
                    ConsoleUI.system_msg(f"Error parsing command: {e}")
                    continue
                handler = self.commands.get(parts[0].lower())
                if handler:
                    handler(parts)
                else:
                    ConsoleUI.system_msg(f"Unknown command: {parts[0]}")
                continue
            self._call_chat(inp, self.pending_prefill)
