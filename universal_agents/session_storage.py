from __future__ import annotations
import os
from datetime import datetime
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from universal_agents.agent import LLMAgent


class SessionStorage:
    """Управление сохранением истории сессии, ротацией автосохранений и сбросом диалога (§1.8)."""

    def __init__(
        self,
        agent: LLMAgent,
        autosave_dir: str = "autosaves",
        autosave_keep: int = 5,
        enabled: bool = True,
    ):
        self.agent = agent
        self.autosave_dir = autosave_dir
        self.autosave_keep = autosave_keep
        self.autosave_enabled = enabled
        self._autosave_path: Optional[str] = None
        self.reset_autosave_path()

    def reset_autosave_path(self) -> None:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._autosave_path = os.path.join(self.autosave_dir, f"autosave_{ts}.json")

    @property
    def autosave_path(self) -> Optional[str]:
        return self._autosave_path

    def build_save_extras(self) -> dict:
        from universal_agents.task_tracker import plan_state_to_dict
        from universal_agents.project_root import get_project_root_override

        return {
            "plan_state": plan_state_to_dict(self.agent),
            "cwd": os.getcwd(),
            "project_root": get_project_root_override(),
        }

    def save_history(self, path: str, extras: dict = None) -> None:
        self.agent.history.save(
            path,
            loaded_tools=list(self.agent._all_tools.keys()),
            extras=extras if extras is not None else self.build_save_extras(),
        )

    def rotate_autosaves(self) -> None:
        if self.autosave_keep <= 0:
            return
        try:
            if not os.path.isdir(self.autosave_dir):
                return
            files = [
                os.path.join(self.autosave_dir, f)
                for f in os.listdir(self.autosave_dir)
                if f.startswith("autosave_") and f.endswith(".json")
            ]
            files.sort()
            excess = files[:-self.autosave_keep]
            for old in excess:
                try:
                    os.remove(old)
                except OSError:
                    pass
        except OSError:
            pass

    def autosave(self) -> None:
        if not self.autosave_enabled:
            return
        try:
            os.makedirs(self.autosave_dir, exist_ok=True)
            path = self._autosave_path or os.path.join(
                self.autosave_dir, f"autosave_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
            )
            self.save_history(path)
            self.rotate_autosaves()
        except Exception as e:
            self.agent.on_system_msg(f"[Autosave] failed: {e}")

    def reset_dialog(self) -> None:
        from universal_agents.history import ChatHistory
        from universal_agents.llm_client import TokenUsageTracker, LoopDetector
        from universal_agents.config import Config

        system_prompt = self.agent.history[0].content if len(self.agent.history) else ""
        self.agent.history = ChatHistory(system_prompt)
        self.agent.task_plan = []
        self.agent.task_plan_map = {}
        self.agent._tool_usage.clear()
        self.agent.loop_detector = LoopDetector()
        self.agent.token_tracker = TokenUsageTracker(
            system_prompt,
            self.agent.token_tracker.max_context_tokens if self.agent.token_tracker else Config.MAX_CONTEXT_TOKENS,
        )
        self.agent._compacted_task_ids = set()
        self.agent._pending_operation = None
        self.agent._guard_nag_message = None
        self.reset_autosave_path()
        self.agent._on_history_changed()
