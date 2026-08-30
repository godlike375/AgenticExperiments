"""План-ориентированная декомпозиция и структурная компактизация истории: make_plan задаёт упорядоченный список задач, have_done помечает выполнение строго по порядку плана (подключается только после успешного make_plan)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from universal_agents.config import Config
from universal_agents.constants import ENVIRONMENT_PREFIX, ENVIRONMENT_PREFIX_END, err, ok
from universal_agents.models import AssistantMessage, UserMessage, ToolResult
from universal_agents.tool_parsing import parse_tool_args

if TYPE_CHECKING:
    from universal_agents.agent import LLMAgent

DONE_TOOL = "have_done"
PLAN_TOOL = "make_plan"


# ---------------------------------------------------------------------------
# План (плоский упорядоченный список)
# ---------------------------------------------------------------------------


def plan_leaf_sequence(plan_map: dict) -> list[str]:
    """Порядок выполнения задач — порядок списка плана."""
    return list(plan_map.keys())


def set_plan(agent: LLMAgent, plan_list: list) -> str:
    """Обработчик make_plan. Сохраняет плоский план на агенте и возвращает порядок."""
    if not isinstance(plan_list, list) or not plan_list:
        return err(": make_plan expects a non-empty list of {id, title} tasks.")
    meta: dict = {}
    for entry in plan_list:
        if not isinstance(entry, dict):
            return err(": each plan entry must be an object with 'id'.")
        tid = (str(entry.get("id", "")).strip())
        if not tid:
            return err(": each plan task needs a non-empty 'id'.")
        if tid in meta:
            return err(f": duplicate task id '{tid}' in plan.")
        meta[tid] = {"title": str(entry.get("title", "") or "").strip()}
    agent.task_plan = list(meta.keys())
    agent.task_plan_map = meta
    # Новый план — сбросить маркеры компактизации предыдущего, чтобы переиспользуемые id не считались выполненными.
    agent._compacted_task_ids = set()
    # Динамически подключаем have_done — доступен только после успешного make_plan.
    tm = getattr(agent, "tools_manager", None)
    if tm is not None:
        try:
            if "have_done" not in getattr(agent, "_all_tools", {}):
                tm.force_load("have_done")
        except Exception as e:
            agent.on_system_msg(f"[TASK PLAN] Failed to attach have_done tool: {e}")
    order = plan_leaf_sequence(meta)
    order_str = " -> ".join(order) or "(empty)"
    return (
        f"{ENVIRONMENT_PREFIX} Plan set ({len(meta)} tasks). Execution order: "
        f"{order_str}. Execute each task REALLY (with real tools: read/search/edit_file/run_bash "
        f"etc.) and only then mark it done with have_done, strictly in this order. "
        f"Do NOT mark a task done without actually doing it."
        f"{ENVIRONMENT_PREFIX_END}"
    )


# ---------------------------------------------------------------------------
# Информационные хендлеры (реальный контроль порядка — в _execute_tools)
# ---------------------------------------------------------------------------


def mark_task_done(agent: LLMAgent, task_id: str) -> str:
    """Обработчик have_done. Информационное подтверждение (не блокирует)."""
    tid = (task_id or "").strip()
    if not tid:
        return err(": have_done requires a non-empty 'id'.")
    plan_map = getattr(agent, "task_plan_map", None) or {}
    if plan_map and tid not in plan_map:
        return err(f": '{tid}' is not in the plan.")
    return (
        ok(f" Task '{tid}' marked done.")
    )


# ---------------------------------------------------------------------------
# Порядок выполнения (на основе плана)
# ---------------------------------------------------------------------------


def _parse_call_args(tc) -> dict:
    return parse_tool_args(getattr(tc, "arguments", None) or "{}")


def _last_plan_position(history: list) -> int:
    """Индекс последнего вызова make_plan (граница текущего плана), иначе -1."""
    pos = -1
    for i, msg in enumerate(history):
        if isinstance(msg, AssistantMessage) and msg.has_tool_calls():
            if any(tc.name == PLAN_TOOL for tc in msg.tool_calls):
                pos = i
    return pos


# Инструменты-маркеры, которые НЕ считаются реальной работой.
_META_TOOLS = {PLAN_TOOL, DONE_TOOL, "load_tool"}


def _last_real_work_position(history: list, after: int) -> int:
    """Индекс последнего вызова РЕАЛЬНОГО инструмента (не make_plan/have_done)
    после позиции `after`; иначе -1."""
    pos = -1
    for i, msg in enumerate(history):
        if i <= after or not isinstance(msg, AssistantMessage) or not msg.has_tool_calls():
            continue
        for tc in msg.tool_calls:
            if (tc.name or "") not in _META_TOOLS:
                pos = i
    return pos


def _last_done_position(history: list, after: int) -> int:
    """Индекс последнего УСПЕШНОГО вызова have_done после позиции `after`;
    иначе -1. Ошибочные/отклонённые вызовы игнорируются."""
    ok_results = _success_result_by_id(history)
    pos = -1
    for i, msg in enumerate(history):
        if i <= after or not isinstance(msg, AssistantMessage) or not msg.has_tool_calls():
            continue
        if any((tc.name or "") == DONE_TOOL and tc.id in ok_results for tc in msg.tool_calls):
            pos = i
    return pos


def _success_result_by_id(history: list) -> dict:
    """tool_call_id -> ToolResult, если результат УСПЕШЕН (не ошибка, не отказ)."""
    ok: dict = {}
    for msg in history:
        if not isinstance(msg, ToolResult):
            continue
        if not msg.is_error and not msg.is_user_denied:
            ok[msg.tool_call_id] = msg
    return ok


def _done_marker_positions(history: list, leaves: list, after: int = -1) -> dict:
    """id задачи -> индекс УСПЕШНОГО ToolResult её have_done (первое вхождение, маркеры после `after`; ошибочные/отклонённые вызовы игнорируются)."""
    result_idx: dict = {}
    for i, msg in enumerate(history):
        if isinstance(msg, ToolResult) and not msg.is_error and not msg.is_user_denied:
            result_idx[msg.tool_call_id] = i
    done: dict = {}
    for i, msg in enumerate(history):
        if i <= after or not isinstance(msg, AssistantMessage) or not msg.has_tool_calls():
            continue
        for tc in msg.tool_calls:
            if tc.name != DONE_TOOL:
                continue
            ridx = result_idx.get(tc.id)
            if ridx is None:
                continue  # вызов был отклонён/не выполнен — не считаем
            tid = ((_parse_call_args(tc)).get("id") or "").strip()
            if tid in leaves and tid not in done:
                done[tid] = ridx
    return done


def validate_task_mark_call(history_before: list, args: dict, plan_map: dict,
                            compacted: set[str]) -> Optional[str]:
    """Проверяет, что have_done вызван для следующего по плану шага; возвращает None или сообщение об ошибке (перегенерация)."""
    tid = (args.get("id") or "").strip()
    if not tid:
        return "have_done requires a non-empty 'id'."

    if not plan_map:
        return (
            "OUT-OF-ORDER: no task plan yet. Call make_plan with the ordered list of "
            "tasks FIRST, then mark them done with have_done in plan order."
        )
    leaves = plan_leaf_sequence(plan_map)
    if tid not in leaves:
        return f"CANNOT MARK '{tid}': not in the plan."

    # Учитываем только выполнение ТЕКУЩЕГО плана (маркеры после последнего make_plan), исключая утечку из предыдущего.
    plan_pos = _last_plan_position(history_before)
    done = set(compacted)
    done |= set(_done_marker_positions(history_before, leaves, after=plan_pos).keys())
    next_leaf = next((l for l in leaves if l not in done), None)
    if next_leaf is None:
        return (
            f"CANNOT MARK '{tid}': all planned tasks are already done. "
            f"Revise the plan with make_plan to add more tasks."
        )
    if tid != next_leaf:
        return (
            f"OUT-OF-ORDER: plan says the next task is '{next_leaf}', but you tried to mark '{tid}'. "
            f"Follow the plan order. To start from a different step, revise the plan with "
            f"make_plan(plan=[...])."
        )
    # Защита от «фиктивного» выполнения: запрещаем have_done без реального инструмента после последнего done/создания плана.
    segment_after = max(plan_pos, _last_done_position(history_before, plan_pos))
    if _last_real_work_position(history_before, segment_after) == -1:
        return (
            f"NO-WORK-DONE: you marked '{tid}' as done, but performed NO actual work since the "
            f"previous task (no read/search/edit_file/run_bash/run_powershell calls for this task). "
            f"Execute this task REALLY with real tools first, then call have_done. "
            f"Do NOT fabricate summaries or mark tasks done without doing the work."
        )
    return None


# ---------------------------------------------------------------------------
# Компактизация завершённых подзадач
# ---------------------------------------------------------------------------


def compact_completed_tasks(agent: LLMAgent) -> int:
    """Компактизирует завершённые задачи (по одной за проход, с самой поздней); возвращает число."""
    if not Config.TASK_COMPACTION_ENABLED:
        return 0
    compacted = 0
    for _ in range(Config.MAX_TASK_COMPACTION_ROUNDS):
        if _compact_one_group(agent) is None:
            break
        compacted += 1
    return compacted


def _earliest_done_leaf(history: list, leaves: list, compacted: set[str], plan_pos: int = -1) -> Optional[str]:
    done = _done_marker_positions(history, leaves, after=plan_pos)
    best = None
    for leaf in leaves:
        if leaf in done and leaf not in compacted:
            if best is None or done[leaf] < done[best]:
                best = leaf
    return best


def _leaf_block(history: list, leaf: str, leaves: list, plan_pos: int) -> tuple:
    """Возвращает (start, end) сегмента задачи — только реальная работа между сохраняемыми маркерами make_plan/have_done (сами маркеры остаются в истории)."""
    done = _done_marker_positions(history, leaves, after=plan_pos)
    done_idx = done[leaf]        # индекс результата have_done этой задачи

    # Индекс самого вызова have_done: ассистент-сообщение с DONE_TOOL,
    # чей tc.id совпадает с tool_call_id результата (обычно сразу перед ним).
    result_msg = history[done_idx]
    call_id = getattr(result_msg, 'tool_call_id', None)
    hd_call = done_idx - 1
    for j in range(done_idx - 1, plan_pos, -1):
        m = history[j]
        if (
            isinstance(m, AssistantMessage) and m.has_tool_calls()
            and any(tc.name == DONE_TOOL and tc.id == call_id for tc in m.tool_calls)
        ):
            hd_call = j
            break

    # Последняя сохраняемая граница перед работой: результат make_plan либо have_done предыдущей задачи.
    boundary = plan_pos + 1
    idx = leaves.index(leaf)
    for i in range(idx):
        prev = done.get(leaves[i])
        if prev is not None and prev > boundary:
            boundary = prev
    start = boundary + 1
    end = hd_call - 1
    return start, end


def _compact_one_group(agent: LLMAgent) -> Optional[str]:
    compacted: set[str] = agent._compacted_task_ids
    plan_map = getattr(agent, "task_plan_map", None) or {}
    if not plan_map:
        return None
    history = agent.history.get_all()
    leaves = plan_leaf_sequence(plan_map)
    if not leaves:
        return None

    # позиция последнего make_plan (текущий план) для границы сегмента
    plan_pos = _last_plan_position(history)

    leaf = _earliest_done_leaf(history, leaves, compacted, plan_pos=plan_pos)
    if leaf is None:
        return None

    done = _done_marker_positions(history, leaves, after=plan_pos)

    start, end = _leaf_block(history, leaf, leaves, plan_pos)
    if start > end:
        compacted.add(leaf)
        return None

    title = (plan_map[leaf].get("title") or leaf)
    summary = summarize_task_segment(agent, history, leaf, title)
    original_len = agent.history.content_len(start, end)
    if not summary or len(summary) >= original_len:
        compacted.add(leaf)
        agent.on_system_msg(
            f"[TASK COMPACTION] Skip subtask '{title}' [{leaf}]: summary not smaller "
            f"({len(summary or '')} >= {original_len})."
        )
        return None

    summary_msg = UserMessage(
        content=ok(f" ('{title}' [{leaf}] has been already marked as done so don't call have_done('{title}'):\n{summary} again. Just take next steps if anything remains undone.\n")
    )
    agent.history.replace_range(start, end, [summary_msg])
    # Обрезаем раздутый summary в have_done: детали уже в компактизационном summary, дублировать не нужно.
    done_idx = done.get(leaf)
    hd = history[done_idx] if done_idx is not None else None
    if hd is not None and not getattr(hd, 'is_error', False):
        if len(getattr(hd, 'content', '') or '') > Config.HAVE_DONE_TRIM_THRESHOLD:
            hd.content = ok(f" Task '{leaf}' marked done (summary compacted).")
    agent._on_history_changed()
    compacted.add(leaf)
    agent.on_system_msg(
        f"[TASK COMPACTION] Compressed subtask '{title}' [{leaf}] -> {len(summary)} chars "
        f"(segment {start}..{end}, {original_len} -> {len(summary)})"
    )
    return leaf


def summarize_task_segment(agent: LLMAgent, segment_msgs: list, task_id: str, task_title: str) -> Optional[str]:
    """Суммаризация сегмента истории через LLM: черновик + review."""
    from universal_agents.compressors import _review_task_summary, _draft_task_summary

    history_msgs = [m.to_api_dict() for m in segment_msgs]
    draft = _draft_task_summary(agent, history_msgs, task_id, task_title)
    if not draft:
        return None
    if Config.AUTO_SUMMARY_REVIEW_PASS:
        final = _review_task_summary(agent, history_msgs, draft, task_id, task_title)
        if final:
            return final
    return draft


# ---------------------------------------------------------------------------
# Сериализация состояния плана (переживает /save + /load)
# ---------------------------------------------------------------------------


def plan_state_to_dict(agent: "LLMAgent") -> dict:
    return {
        "task_plan": list(getattr(agent, "task_plan", None) or []),
        "task_plan_map": dict(getattr(agent, "task_plan_map", None) or {}),
        "compacted_task_ids": sorted(getattr(agent, "_compacted_task_ids", None) or set()),
    }


def restore_plan_state(agent: "LLMAgent", data: dict | None) -> None:
    if not isinstance(data, dict):
        return
    agent.task_plan = list(data.get("task_plan") or [])
    agent.task_plan_map = dict(data.get("task_plan_map") or {})
    agent._compacted_task_ids = set(data.get("compacted_task_ids") or [])

