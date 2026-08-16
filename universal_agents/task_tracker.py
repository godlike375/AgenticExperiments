"""План-ориентированная декомпозиция и структурная компактизация истории.

Модель:
   1) создаёт ПЛАН через `create_plan(plan=[...])` — плоский упорядоченный
     список задач {id, title} (как bullet list).
   2) выполняет задачи и помечает каждую выполненной через
     `have_done(id, summary)` — строго по порядку плана.

Система читает план из истории и заставляет `have_done` идти строго по
порядку списка. При изменении плана (повторный `create_plan`) «следующий шаг»
пересчитывается из нового плана, что позволяет начать с произвольного шага и
далее идти по порядку.

Инструмент `have_done` динамически подключается агентом только после УСПЕШНОГО
создания плана через `create_plan` (до этого он отсутствует в инструментарии).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from universal_agents.config import Config
from universal_agents.constants import ENVIRONMENT_PREFIX, SUMMARY_MARKER
from universal_agents.models import AssistantMessage, UserMessage, ToolResult
from universal_agents.tool_parsing import parse_tool_args

if TYPE_CHECKING:
    from universal_agents.agent import LLMAgent

DONE_TOOL = "have_done"
PLAN_TOOL = "create_plan"


# ---------------------------------------------------------------------------
# План (плоский упорядоченный список)
# ---------------------------------------------------------------------------


def plan_leaf_sequence(plan_map: dict) -> list[str]:
    """Порядок выполнения задач — порядок списка плана."""
    return list(plan_map.keys())


def set_plan(agent: LLMAgent, plan_list: list) -> str:
    """Обработчик create_plan. Сохраняет плоский план на агенте и возвращает порядок."""
    if not isinstance(plan_list, list) or not plan_list:
        return f"{ENVIRONMENT_PREFIX} Error: create_plan expects a non-empty list of {{id, title}} tasks."
    meta: dict = {}
    for entry in plan_list:
        if not isinstance(entry, dict):
            return f"{ENVIRONMENT_PREFIX} Error: each plan entry must be an object with 'id'."
        tid = (str(entry.get("id", "")).strip())
        if not tid:
            return f"{ENVIRONMENT_PREFIX} Error: each plan task needs a non-empty 'id'."
        if tid in meta:
            return f"{ENVIRONMENT_PREFIX} Error: duplicate task id '{tid}' in plan."
        meta[tid] = {"title": str(entry.get("title", "") or "").strip()}
    agent.task_plan = list(meta.keys())
    agent.task_plan_map = meta
    # Новый план — новый контекст выполнения: сбросить маркеры компактизации
    # предыдущего плана, чтобы переиспользуемые id не считались выполненными.
    agent._compacted_task_ids = set()
    # Динамически подключаем have_done: он доступен только после успешного
    # create_plan (до этого в инструментарии его нет).
    tm = getattr(agent, "tools_manager", None)
    if tm is not None:
        try:
            if "have_done" not in getattr(agent, "_all_tools", {}):
                tm.load("have_done")
        except Exception:
            pass
    order = plan_leaf_sequence(meta)
    order_str = " -> ".join(order) or "(empty)"
    return (
        f"{ENVIRONMENT_PREFIX} Plan set ({len(meta)} tasks). Execution order: "
        f"{order_str}. Execute each task REALLY (with real tools: read/search/edit_file/run_bash "
        f"etc.) and only then mark it done with have_done, strictly in this order. "
        f"Do NOT mark a task done without actually doing it."
    )


# ---------------------------------------------------------------------------
# Информационные хендлеры (реальный контроль порядка — в _execute_tools)
# ---------------------------------------------------------------------------


def mark_task_done(agent: LLMAgent, task_id: str, summary: str) -> str:
    """Обработчик have_done. Информационное подтверждение (не блокирует)."""
    tid = (task_id or "").strip()
    if not tid:
        return f"{ENVIRONMENT_PREFIX} Error: have_done requires a non-empty 'id'."
    plan_map = getattr(agent, "task_plan_map", None) or {}
    if plan_map and tid not in plan_map:
        return f"{ENVIRONMENT_PREFIX} Error: '{tid}' is not in the plan."
    return (
        f"{ENVIRONMENT_PREFIX} Task '{tid}' marked done. Summary recorded: {summary or '-'}"
    )


# ---------------------------------------------------------------------------
# Порядок выполнения (на основе плана)
# ---------------------------------------------------------------------------


def _parse_call_args(tc) -> dict:
    return parse_tool_args(getattr(tc, "arguments", None) or "{}")


def _last_plan_position(history: list) -> int:
    """Индекс последнего вызова create_plan (граница текущего плана), иначе -1."""
    pos = -1
    for i, msg in enumerate(history):
        if isinstance(msg, AssistantMessage) and msg.has_tool_calls():
            if any(tc.name == PLAN_TOOL for tc in msg.tool_calls):
                pos = i
    return pos


# Инструменты-маркеры, которые НЕ считаются реальной работой.
_META_TOOLS = {PLAN_TOOL, DONE_TOOL, "load_tools"}


def _last_real_work_position(history: list, after: int) -> int:
    """Индекс последнего вызова РЕАЛЬНОГО инструмента (не create_plan/have_done)
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
    """id задачи -> индекс УСПЕШНОГО ToolResult её have_done (первое вхождение).
    Учитываются только маркеры с индексом > after (после текущего плана).
    Ошибочные/отклонённые вызовы have_done НЕ считаются выполненными."""
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
    """Проверяет, что have_done вызывается для следующего по плану шага.

    Возвращает None, если допустимо, иначе сообщение об ошибке (ведёт к
    перегенерации ответа модели)."""
    tid = (args.get("id") or "").strip()
    if not tid:
        return "have_done requires a non-empty 'id'."

    if not plan_map:
        return (
            "OUT-OF-ORDER: no task plan yet. Call create_plan with the ordered list of "
            "tasks FIRST, then mark them done with have_done in plan order."
        )
    leaves = plan_leaf_sequence(plan_map)
    if tid not in leaves:
        return f"CANNOT MARK '{tid}': not in the plan."

    # Учитываем только выполнение ТЕКУЩЕГО плана: маркеры после последнего create_plan.
    # Это исключает утечку done-маркеров/компактизации предыдущего плана.
    plan_pos = _last_plan_position(history_before)
    done = set(compacted)
    done |= set(_done_marker_positions(history_before, leaves, after=plan_pos).keys())
    next_leaf = next((l for l in leaves if l not in done), None)
    if next_leaf is None:
        return (
            f"CANNOT MARK '{tid}': all planned tasks are already done. "
            f"Revise the plan with create_plan to add more tasks."
        )
    if tid != next_leaf:
        return (
            f"OUT-OF-ORDER: plan says the next task is '{next_leaf}', but you tried to mark '{tid}'. "
            f"Follow the plan order. To start from a different step, revise the plan with "
            f"create_plan(plan=[...])."
        )
    # Защита от «фиктивного» выполнения: нельзя пометить задачу выполненной, если
    # после последнего done-маркера (или создания плана) не было НИ ОДНОГО реального
    # инструмента (read/search/edit/run и т.п.) для ЭТОЙ задачи.
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
    """Компактизирует завершённые задачи (по одной за проход, начиная с самой
    поздней). Возвращает число компактизированных."""
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
    """Возвращает (start, end) компактизируемого сегмента задачи — только реальная
    работа между сохраняемыми маркерами.

    Маркеры create_plan и have_done (вызов + результат) в сегмент НЕ входят и
    остаются в истории (чтобы модель всегда видела порядок плана и факт завершения).
    start — сразу после последней сохраняемой границы (результата create_plan либо
    результата have_done предыдущей задачи); end — непосредственно перед вызовом
    have_done этой задачи (его результат на done[leaf])."""
    done = _done_marker_positions(history, leaves, after=plan_pos)
    done_idx = done[leaf]        # индекс результата have_done этой задачи
    hd_call = done_idx - 1       # вызов have_done (непосредственно перед результатом)

    # Последняя сохраняемая граница перед работой задачи: результат create_plan
    # (вызов на plan_pos) либо результат have_done последней размеченной предыдущей.
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

    # позиция последнего create_plan (текущий план) для границы сегмента
    plan_pos = _last_plan_position(history)

    leaf = _earliest_done_leaf(history, leaves, compacted, plan_pos=plan_pos)
    if leaf is None:
        return None

    done = _done_marker_positions(history, leaves, after=plan_pos)

    start, end = _leaf_block(history, leaf, leaves, plan_pos)
    if start > end:
        compacted.add(leaf)
        return None

    segment = history[start:end + 1]
    title = (plan_map[leaf].get("title") or leaf)
    summary = summarize_task_segment(agent, segment, leaf, title)
    original_len = agent.history.content_len(start, end)
    if not summary or len(summary) >= original_len:
        compacted.add(leaf)
        agent.on_system_msg(
            f"[TASK COMPACTION] Skip subtask '{title}' [{leaf}]: summary not smaller "
            f"({len(summary or '')} >= {original_len})."
        )
        return None

    summary_msg = UserMessage(
        content=f"{SUMMARY_MARKER} (subtask '{title}' [{leaf}] done):\n{summary}"
    )
    agent.history.replace_range(start, end, [summary_msg])
    # Обрезаем раздутый summary в результате have_done: детальные факты уже
    # сохранены в компактизационном summary, дублировать их в маркере не нужно.
    done_idx = done.get(leaf)
    hd = history[done_idx] if done_idx is not None else None
    if hd is not None and not getattr(hd, 'is_error', False):
        if len(getattr(hd, 'content', '') or '') > Config.HAVE_DONE_TRIM_THRESHOLD:
            hd.content = f"{ENVIRONMENT_PREFIX} Task '{leaf}' marked done (summary compacted)."
    agent.file_states.prune()
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
# Инструкции модели (встраиваются в системный промпт, см. main.py)
# ---------------------------------------------------------------------------

TASK_MARK_INSTRUCTIONS = (
    "\n"
    "* TASK DECOMPOSITION (для длинных многошаговых задач):\n"
    f"  - Сложную многошаговую просьбу разбивай на шаги-подзадачи.\n"
    "  - СНАЧАЛА вызови create_plan\n"
    "  - Затем выполняй задачи СТРОГО по одной в порядке плана. Для КАЖДОЙ задачи:\n"
    "      (1) реально выполни её РЕАЛЬНЫМИ инструментами (read, search, edit_file, run_bash, "
    "run_powershell и т.п.) — читай код, вноси правки, запускай команды;\n"
    '      (2) и только после реального выполнения вызови have_done(id="...", summary="краткий итог"), '
    "чтобы отметить её выполненной.\n"
    "  - ВАЖНО: have_done вызывай ТОЛЬКО ПОСЛЕ реального выполнения подзадачи. "
    "НЕ помечай задачу выполненной и НЕ выдумывай итог, если ты её ещё не выполнил(а). "
    "Саммари должно отражать то, что реально сделано (файлы/команды/результаты).\n"
    "  - Система заставит идти по плану: нельзя пометить выполненной задачу не по порядку. "
    "Чтобы изменить план или начать с другого шага — вызови create_plan заново (новая версия плана), "
    "далее иди по его порядку.\n"
    "  - Давай каждой задаче уникальный id (e.g. t1, t2, ...).\n"
    "  - Инструмент have_done становится доступным ТОЛЬКО после успешного create_plan; "
    "до создания плана его нет в списке инструментов.\n"
    "  - НЕ вызывай create_plan/have_done для простых ответов/вопросов без многошаговой работы — "
    "только при реальной декомпозиции.\n"
    "  - create_plan и have_done невидимы пользователю; они помогают системе компактизировать "
    "память после завершения подзадач.\n"
)
