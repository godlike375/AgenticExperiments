from __future__ import annotations

import datetime
import os
import re as _re
import fnmatch as _fnmatch
from typing import Optional

from universal_agents.config import Config
from universal_agents.constants import (
    ENVIRONMENT_PREFIX,
    ENVIRONMENT_PREFIX_END,
    err,
    ok,
)
from universal_agents.file_states import _content_hash
from universal_agents.tool import tool

# Лимиты поиска: не «виснуть» и не съесть гигабайты на больших деревьях/бинарниках.
SEARCH_MAX_FILE_SIZE = 1 * 1024 * 1024    # пропускать файлы крупнее 5 МБ
SEARCH_MAX_MATCH_FILES = Config.MAX_READ_LINES_PER_CALL  # макс. число файлов с совпадениями в ответе
SEARCH_MAX_OUTPUT_CHARS = Config.MAX_READ_CHARS_PER_CALL # макс. размер текста результата (до усечения)
SEARCH_BINARY_PROBE = 8192                # размер префикса для детекции бинарника


class FS:

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        if size_bytes < 1024: return f"{size_bytes}B"
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f}{unit}" if unit != 'B' else f"{size_bytes}{unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f}TB"

    @staticmethod
    def _count_hidden_size(root_path: str) -> int:
        total = 0
        try:
            for entry in os.scandir(root_path):
                try:
                    if entry.is_file(): total += entry.stat().st_size
                    elif entry.is_dir(): total += FS._count_hidden_size(entry.path)
                except PermissionError: continue
        except (PermissionError, FileNotFoundError): pass
        return total

    @staticmethod
    def _build_tree(root_path: str, depth: int = 0, density: int = 4) -> str:
        try: entries = list(os.scandir(root_path))
        except PermissionError: return f"{'  ' * depth}[Permission Denied]"
        except FileNotFoundError: return f"{'  ' * depth}[Path Not Found]"

        if depth > 0 and len(entries) > density:
            size = FS._format_size(FS._count_hidden_size(root_path))
            return f"{'  ' * depth}[{len(entries)} items TRUNCATED, {size}]"

        dirs = sorted([e for e in entries if e.is_dir()], key=lambda x: x.name.lower())
        files = sorted([e for e in entries if e.is_file()], key=lambda x: x.name.lower())

        lines = []
        for entry in dirs + files:
            mtime = datetime.datetime.fromtimestamp(entry.stat().st_mtime).strftime("%Y-%m-%d")
            prefix = f"{'  ' * depth}"
            if entry.is_dir():
                lines.append(f"{prefix}{entry.name}/ ({mtime})")
                if sub := FS._build_tree(entry.path, depth + 1, density):
                    lines.append(sub)
            else:
                lines.append(f"{prefix}{entry.name} ({FS._format_size(entry.stat().st_size)})")
        return "\n".join(lines)

@tool(description="Gets or changes current working dir",
      short_description="get/set working dir",
      path=("str", "Optional new working dir. Use '..' to go to the parent dir"))
def cwd(path: str = None):
    if path:
        try:
            os.chdir(path)
            return ok(f" Has set cwd to {path}")
        except Exception as e:
            raise RuntimeError(f"Error changing cwd: {e}")  # Было return, стало raise
    return os.getcwd()

@tool(
    description="Edits a file by line range: replaces the 1-based inclusive lines start_line..end_line with 'new'. Creates file with parent dirs if it doesn't exist",
    short_description="edit file text",
    requires_confirmation=True,
    safe_in_trusted=True,
    path=("str", "File path. Will be auto-created if missing"),
    #old=("str", "Exact text to replace. Supports \\n for multiline blocks. If '' or nothing passed then replaces whole content. For new files use '' to set initial content"),
    new=("str", "New text to replace the range with. Supports \\n. Be careful with indentation in the range."),
    #mode=("str", "'one' for 1 exclusive match, otherwise 'all' (default 'one')")
    start_line=("int", "1-based inclusive start line of the range to replace with 'new'. Default 1"),
    end_line=("int", "1-based inclusive end line of the range to replace with 'new'. If omitted: end of file when start_line omitted too, else a single line at start_line")
)
def edit_file(path: str, new: str, start_line: int = None, end_line: int = None):
    created_file = False
    if not os.path.isfile(path):
        # Создаём файл, если его нет
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write("")
            created_file = True
        except Exception as e:
            raise RuntimeError(f"Failed to create file: {e}")

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Режим по номерам строк: заменяем [start_line..end_line] (1-based, инклюзивно) на 'new'; '\n' старых строк отбрасываются, 'new' пишется как есть.
    lines = content.splitlines()
    start = max(1, start_line if start_line is not None else 1)
    if end_line is None:
        end = len(lines) if start_line is None else start
    else:
        end = end_line
    if end < 0:
        end = len(lines) + end
    end = max(start, min(end, len(lines)))

    replaced_lines = lines[start - 1:end]
    old_block = "\n".join(replaced_lines)

    head = lines[:start - 1]
    tail = lines[end:]

    new_clean = new.rstrip('\n')
    if head or tail:
        new_content = "\n".join(head + [new_clean] + tail)
    else:
        new_content = new_clean

    if new_content == content:
        return f"Nothing changed: lines {start}..{end} already equal to '{new[:20]}...'"

    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)

    result = [f"Replaced lines {start}-{end} with:"]
    for line in old_block.splitlines():
        result.append(f"   - {line}")
    result.append("---")
    for line in new_clean.splitlines():
        result.append(f"   + {line}")

    return "\n".join(result)


CHARS_PER_TOKEN = Config.CHARS_PER_TOKEN

MIN_TOKENS_TO_SUMMARIZE = Config.MIN_TOKENS_TO_SUMMARIZE
_SUMMARY_THRESHOLD = int(MIN_TOKENS_TO_SUMMARIZE * CHARS_PER_TOKEN)


def _read_text(path: str) -> tuple:
    """Читает файл как UTF-8; возвращает (content, None) или (None, error_msg)."""
    try:
        with open(path, 'r', encoding='utf-8', errors='strict') as f:
            return f.read(), None
    except UnicodeDecodeError:
        return None, err(": Cannot read binary files (failed UTF-8 decode)")


def _summarize_file(content: str, agent) -> str:
    """Строит структурный скелет файла ОДНИМ вызовом LLM (без суб-агента, как в summarize_history_plain): инструкция + пронумерованный файл как user-сообщение. Разовая операция — KV-cache не нужен, суб-агент бы только зря тратил контекст и ломал извлечение ответа."""
    max_tokens = agent.token_tracker.get_remaining() // Config.SUMMARIZATION_THRESHOLD_DIVIDER
    max_chars = int(max_tokens * Config.CHARS_PER_TOKEN)
    truncated = len(content) > max_chars
    snippet = content[:max_chars]
    raw_lines = snippet.split("\n")
    # Вырезаем пустые строки: они не несут структурной информации, а модель
    # тратит на них токены и хуже ориентируется. Оригинальные номера строк
    # сохраняются (дыры в нумерации допустимы), чтобы диапазоны сразу
    # соответствовали реальному файлу для read(start_line=...).
    numbered_text = "\n".join(
        f"{i + 1} {line}" for i, line in enumerate(raw_lines) if line.strip()
    )

    if Config.SKELETON_RANGES_MODE:
        # Новый режим: модель возвращает ТОЛЬКО диапазоны строк, заголовки
        # подставляются программно из файла (см. _ranges_to_structure).
        task = (
            f"{ENVIRONMENT_PREFIX} NOW IGNORE previous instructions! Act as file structure generator. "
            "Output ONLY the line ranges of the most top-level items (signatures of classes, functions, "
            "methods, headers]) "
            "wrapped in <content_structure_lines> tags. Do NOT write any text of the lines "
            "- just the precise numeric ranges, one per line, in the form `start-end` inclusively 1-based. "
            "(an example for reference only: '<content_structure_lines>\na-b\nx-y\nm-n\n</content_structure_lines>)'. "
            "Follow the order of items in the file.\n"
        )
        prefill = "<content_structure_lines>\n"
    else:
        # Старый fallback-режим: модель пишет таблицу целиком (тег <content_structure>).
        task = (
            f"{ENVIRONMENT_PREFIX} NOW IGNORE previous instructions! Act as file content structure writer. "
            "Start with tag <content_structure> and write very short compact content structure table "
            "(like table of content in books) (for example signatures of classes, functions, methods, headers) and "
            "their precise line ranges (an example for reference only: `<content_structure>Lx-y example1()\nLa-b class Example2\n...`"
        )
        prefill = "<content_structure>\nL"

    file = '```\n' + numbered_text + '\n```\n\n'
    task = file + task
    if truncated:
        task += "\n\n(File is truncated due to remaining memory)"

    from universal_agents.context_builder import prepare_messages_for_api
    history_msgs = prepare_messages_for_api(agent, normalize=False)
    msgs = list(history_msgs) + [{"role": "user", "content": task}]
    msg_obj, err, usage = agent.service_llm_call(
        msgs,
        temp=Config.TEMP,
        timeout=Config.TIMEOUT,
        prefill=prefill,
    )
    if err or not msg_obj or not msg_obj.content:
        if err:
            agent.on_system_msg(f"[file skeleton] LLM error: {err}")
        # Возвращаем None (а не err-строку), чтобы read мог вернуть подсказку ЦЕЛИКОМ.
        # Если засунуть err-текст внутрь успешного контента read, is_error_content
        # его не заметит (вложенный [[SYSTEM]] Error не детектится).
        return None
    result = msg_obj.content.strip()
    # Снимаем возможную служебную обёртку субагента, если модель её добавила.
    if result.startswith("<sub_agent>"):
        result = result[len("<sub_agent>"):]
    if result.endswith("</sub_agent>"):
        result = result[: -len("</sub_agent>")]
    result = result.strip()
    if not result:
        return None
    if Config.SKELETON_RANGES_MODE:
        return _ranges_to_structure(result, content)
    # Старый режим: модель уже вернула готовую таблицу структуры.
    return result


def _ranges_to_structure(ranges_text: str, content: str) -> str | None:
    """Превращает текст с диапазонами строк (например `1-7\\n9-28`) в структуру вида
    `L1-7 import abc\\nL9-28 class GenerationParams:`, подставляя реальный контент
    первой строки каждого диапазона из файла. Диапазоны — в оригинальной нумерации
    файла (пустые строки уже вырезаны на этапе подачи в LLM). Возвращает None,
    если не удалось извлечь ни одного валидного диапазона."""
    import re
    lines = content.splitlines()
    total = len(lines)
    out: list[str] = []
    for m in re.finditer(r"(\d+)\s*-\s*(\d+)", ranges_text):
        start = int(m.group(1))
        end = int(m.group(2))
        if start < 1 or end < start or start > total:
            continue
        end = min(end, total)
        # Заголовок берём с первой НЕпустой строки диапазона (пустые/пробельные
        # строки после strip считаются пустыми). Если start пустая — пробуем
        # start-1, затем start+1, затем ближайшую по обе стороны.
        repr_idx = _pick_nonempty_line(start, end, lines)
        head = lines[repr_idx - 1].strip()
        if len(head) > 120:
            head = head[:117] + "..."
        # Убираем лишние пробелы/табы внутри для компактности.
        head = re.sub(r"\s+", " ", head)
        out.append(f"L{start}-{end} {head}")
    if not out:
        return None
    return "\n".join(out)


def _is_meaningful_line(line: str) -> bool:
    """Строка годится для заголовка: непустая после strip() и содержит хотя бы 2
    буквы (пробелы/табуляция/разделители/``` и т.п. не считаются содержательными)."""
    s = line.strip()
    return bool(s) and sum(1 for c in s if c.isalpha()) >= 2


def _pick_nonempty_line(start: int, end: int, lines: list[str]) -> int:
    """Выбирает номер строки (1-based) для заголовка диапазона: первая
    содержательная строка (см. _is_meaningful_line). Порядок: start -> start-1
    -> start+1 -> ближайшая по обе стороны от start. Если все не содержательные
    — возвращает start (fallback)."""
    total = len(lines)
    if 1 <= start <= total and _is_meaningful_line(lines[start - 1]):
        return start
    if start - 1 >= 1 and _is_meaningful_line(lines[start - 2]):
        return start - 1
    if start + 1 <= total and _is_meaningful_line(lines[start]):
        return start + 1
    for d in range(1, total + 1):
        lo, hi = start - d, start + d
        if lo >= 1 and _is_meaningful_line(lines[lo - 1]):
            return lo
        if hi <= total and _is_meaningful_line(lines[hi - 1]):
            return hi
    return start


def _limit_read_chunk(lines: list[str]) -> tuple[list[str], bool]:
    """Порция чтения: ≤ MAX_READ_CHARS_PER_CALL символов и ≤ MAX_READ_LINES_PER_CALL строк; строка на лимите включается целиком, порция кончается концом строки. Возвращает (строки, обрезано_ли)."""
    out: list[str] = []
    total_chars = 0
    for line in lines[:Config.MAX_READ_LINES_PER_CALL]:
        out.append(line)
        total_chars += len(line) + 1  # +1 за перевод строки
        if total_chars >= Config.MAX_READ_CHARS_PER_CALL:
            break
    return out, len(out) < len(lines)


def _peripheral_indices(near: int, far: int, total: int, growth: float = 2.0) -> list[int]:
    """Периферийное (размытое) зрение: от границы `near` (сразу за фокусом) до края
    файла `far` (1 или total) выбирает строки с экспоненциально растущим шагом —
    плотно у границы фокуса, всё реже к краю файла. `growth` — во сколько раз растёт
    шаг каждое кольцо (меньше → плотнее). Всегда добавляет крайний якорь `far`.
    Возвращает отсортированный список номеров строк (1-based)."""
    if near < 1 or near > total or far < 1 or far > total or near == far:
        return []
    direction = 1 if far > near else -1
    out: list[int] = []
    step = 1.0
    cur = near
    while True:
        nxt = cur + direction * round(step)
        if direction > 0 and nxt > far:
            break
        if direction < 0 and nxt < far:
            break
        out.append(nxt)
        cur = nxt
        step *= growth  # экспоненциальный рост «размытия»
    if out and out[-1] != far:
        out.append(far)
    return sorted(out)


@tool(description="Reads a file or shows a directory tree. Small files are returned fully. "
                  "Large files WITHOUT start_line/end_line return a structural skeleton exactly once "
                  "To read actual code "
                  "from a large file, pass start_line/end_line (1-based, inclusive) — each call returns "
                  "a limited portion (~1000 chars); if truncated, continue with the suggested start_line.",
      short_description="read file / ls dir",
      path=("str", "Optional path to file/dir (default '.'). Use '..' to open parent dir"),
      start_line=("int", "Optional 1-based start line. Omit (with end_line) to get the one-time structural skeleton of a large file instead of raw content"),
      end_line=("int", "Optional 1-based inclusive end line (supports negative values like Python slices)"))
def read(agent: 'LLMAgent', path: str = '.', start_line: int = None, end_line: int = None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path not found: {path}")
    try:
        mtime = datetime.datetime.fromtimestamp(os.stat(path).st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        if os.path.isfile(path):
            if start_line is not None or end_line is not None:
                # Порционное чтение ДИАПАЗОНА с жёстким лимитом объёма (файл не выгружается в контекст за раз).
                raw, read_err = _read_text(path)
                if read_err:
                    return read_err
                lines = raw.splitlines()
                total = len(lines)
                start = max(1, start_line if start_line is not None else 1)
                end = total if end_line is None else (total + end_line if end_line < 0 else end_line)
                end = max(start, min(end, total))
                if start > total:
                    return err(f": start_line {start} is beyond the end of the file ({total} lines).")
                selected, truncated = _limit_read_chunk(lines[start - 1:end])
                actual_end = start + len(selected) - 1
                # «Центральное зрение»: запрошенный фокус-диапазон — чётко и целиком.
                focus_idx = range(start, actual_end + 1)
                # «Периферийное зрение»: вокруг фокуса до краёв файла, экспоненциально
                # реже — даёт контекст, не выгружая весь файл.
                peripheral: set[int] = set()
                focus_size = actual_end - start + 1
                peri_span = int(Config.PERIPHERAL_SIDE_FACTOR * focus_size)
                if start - 1 >= 1:
                    left_far = 1 if peri_span <= 0 else max(1, start - peri_span)
                    peripheral.update(_peripheral_indices(start - 1, left_far, total, Config.PERIPHERAL_GAP_GROWTH))
                if actual_end + 1 <= total:
                    right_far = total if peri_span <= 0 else min(total, actual_end + peri_span)
                    peripheral.update(_peripheral_indices(actual_end + 1, right_far, total, Config.PERIPHERAL_GAP_GROWTH))
                peripheral -= set(focus_idx)
                # Локальный контекст периферии: вокруг каждой выбранной строки
                # добавляем ±N соседей (без строк фокуса).
                if Config.PERIPHERAL_LINE_CONTEXT:
                    ctx = int(Config.PERIPHERAL_LINE_CONTEXT)
                    extra: set[int] = set()
                    for p in peripheral:
                        for d in range(-ctx, ctx + 1):
                            j = p + d
                            if 1 <= j <= total and not (start <= j <= actual_end):
                                extra.add(j)
                    peripheral |= extra
                # Собираем строки: фокус + периферия, упорядоченно по номеру,
                # чтобы сохранить пространственную картину файла. Периферийные
                # строки (маркер '~') обрезаются по длине (только контекст).
                included = sorted(set(focus_idx) | peripheral)
                max_peri = Config.PERIPHERAL_MAX_LINE_CHARS
                numbered = []
                for i in included:
                    is_focus = start <= i <= actual_end
                    line = lines[i - 1]
                    if not is_focus and max_peri and len(line) > max_peri:
                        line = line[:max_peri] + "..."
                    marker = "" if is_focus else "~"
                    numbered.append(f"{marker}{i} {line}")
                focus_note = (
                    f"Full focus lines {start}-{actual_end}/{total}. "
                    f"'~' marks sparse peripheral context lines (exponential vision)."
                )
                content = (
                    f"{ENVIRONMENT_PREFIX} File: {path}\nModified: {mtime}\n"
                    f"{focus_note}\n---\n"
                    + ("\n".join(numbered) if numbered else "")
                    + (f"\n{ENVIRONMENT_PREFIX} Output limit is ~{Config.MAX_READ_CHARS_PER_CALL} chars. "
                       f"Use start_line={actual_end + 1} to continue.{ENVIRONMENT_PREFIX_END}"
                       if truncated else "")
                    + f"\n{ENVIRONMENT_PREFIX_END}"
                )
                return content
            # Без диапазона
            raw, read_err = _read_text(path)
            if read_err:
                return read_err
            lines = raw.splitlines()
            total = len(lines)
            disk_hash = _content_hash(raw)
            header = f"{ENVIRONMENT_PREFIX} File: {path}\nModified: {mtime}\n"
            if len(raw) <= _SUMMARY_THRESHOLD:
                numbered = [f"{i+1} {line}" for i, line in enumerate(lines)]
                content = header + "Content:\n---\n" + ("\n".join(numbered) if numbered else "") + f"\n{ENVIRONMENT_PREFIX_END}"
                return _finish_read(agent, path, raw, content, disk_hash)
            # БОЛЬШОЙ файл: ровно один раз отдаём скелет; повторные целиком-файловые чтения без изменений запрещены.
            if agent.file_states.should_skip(path, disk_hash):
                return _reread_err(path)
            structure = ""
            if Config.BIG_FILE_SKELETON:
                structure = _summarize_file(raw, agent)
                if structure is None:
                    # Генерация скелета не удалась (например, сбой субагента) — НЕ возвращаем
                    # ошибку, но и не отдаём файл целиком (он может быть огромным). Вместо
                    # этого — лёгкая подсказка читать файл диапазонами start_line/end_line.
                    agent.on_system_msg(
                        f"[read] Skeleton generation for '{path}' failed; "
                        f"returning a hint to read it in portions instead of full content."
                    )
                    return (
                        f"{ENVIRONMENT_PREFIX} File: {path}\nModified: {mtime}\nTotal lines: {total}\n"
                        f"Structure skeleton is temporarily unavailable (sub-agent returned empty). "
                        f"Do NOT retry 'read' without range — read the file in portions using "
                        f"start_line/end_line, e.g. read('{path}', start_line=1). "
                        f"Each call returns ~{Config.MAX_READ_CHARS_PER_CALL} chars."
                        f"{ENVIRONMENT_PREFIX_END}"
                    )
            structure_block = (
                f"Content structure:\n---\n{structure}\n"
                if structure else ""
            )
            content = (
                header + f"Total lines: {total}\n"
                f"{structure_block}"
                f"\n{ENVIRONMENT_PREFIX_END}"
            )
            agent.file_states.record(path, disk_hash, _content_hash(content))
            agent._read_registrations.append(path)
            return content
        elif os.path.isdir(path):
            return f"{ENVIRONMENT_PREFIX} Directory Tree: {os.path.abspath(path)}\nModified: {mtime}\n\n{FS._build_tree(path)}{ENVIRONMENT_PREFIX_END}"
        raise RuntimeError("Unexpected file type")
    except Exception as e:
        raise PermissionError(f"Error accessing {path}: {e}")


def _reread_err(path: str) -> str:
    """Единая ошибка повторного чтения неизменённого файла (один текст, без дублирования в read и _finish_read)."""
    return err(
        f": re-reading file '{path}' is not allowed - it is unchanged "
        f"since the last read and its content/structure is already in context. "
        f"Use start_line/end_line to read specific portions "
        f"(~{Config.MAX_READ_CHARS_PER_CALL} chars at a time)."
    )


def _finish_read(agent: 'LLMAgent', path: str, raw: str, content: str, disk_hash: str) -> str:
    """Для маленьких файлов: пропуск неизменённого + регистрация чтения (привязка к _execute_tools). Контент уже собран в read — здесь только проверка/учёт."""
    if agent.file_states.should_skip(path, disk_hash):
        return _reread_err(path)
    agent.file_states.record(path, disk_hash, _content_hash(content))
    agent._read_registrations.append(path)
    return content


_REGEX_CHARS = set(r".*+?()[]|\\^$")

def _is_regex(pattern: str) -> bool:
    return any(c in _REGEX_CHARS for c in pattern)


@tool(
    description="Search for text or regex pattern across files. Use this INSTEAD of manually reading files to find something. Returns matches with context.",
    short_description="search in files",
    pattern=("str", "Text to find or regex pattern"),
    path=("str", "Directory or file to search in (default '.')"),
    include=("str", "File filter glob, e.g. '*.py', '*.ts'"),
    exclude=("str", "Glob patterns to skip, space-separated, e.g. '.git node_modules *.pyc'"),
)
def search(pattern: str, path: str = ".", include: str = None, exclude: str = None):
    use_regex = _is_regex(pattern)
    if use_regex:
        try:
            compiled = _re.compile(pattern)
        except _re.error as e:
            return err(f" Invalid regex: {e}")
        matcher = lambda text: compiled.search(text) is not None
    else:
        matcher = lambda text: pattern in text

    exclude_list = exclude.split() if exclude else []

    def _is_excluded(name: str) -> bool:
        return any(_fnmatch.fnmatch(name, pat) for pat in exclude_list)

    if os.path.isfile(path):
        files = [path]
    elif os.path.isdir(path):
        files = []
        for root, dirs, filenames in os.walk(path):
            dirs[:] = [d for d in dirs if not _is_excluded(d)]
            for fn in filenames:
                if _is_excluded(fn):
                    continue
                if include and not _fnmatch.fnmatch(fn, include):
                    continue
                files.append(os.path.join(root, fn))
    else:
        return err(f" Path not found: {path}")

    results = []
    total = 0
    skipped = 0
    file_count = 0
    truncated = False
    out_len = 0
    CONTEXT = 1  # lines before/after match

    for filepath in sorted(files):
        if file_count >= SEARCH_MAX_MATCH_FILES:
            truncated = True
            break

        try:
            size = os.path.getsize(filepath)
        except OSError:
            continue
        if size > SEARCH_MAX_FILE_SIZE:
            skipped += 1
            continue

        # Детекция бинарников по NUL-байту в префиксе (иначе — мусор-совпадения и раздутый вывод).
        try:
            with open(filepath, 'rb') as fb:
                probe = fb.read(SEARCH_BINARY_PROBE)
        except OSError:
            continue
        if b'\x00' in probe:
            skipped += 1
            continue

        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except Exception:
            continue

        match_indices = set()
        for i, line in enumerate(lines):
            if matcher(line):
                for j in range(max(0, i - CONTEXT), min(len(lines), i + CONTEXT + 1)):
                    match_indices.add(j)

        if not match_indices:
            continue

        file_count += 1
        total += sum(1 for i, line in enumerate(lines) if matcher(line))

        # Build grouped blocks
        sorted_idx = sorted(match_indices)
        blocks = []
        block = [sorted_idx[0]]
        for idx in sorted_idx[1:]:
            if idx == block[-1] + 1:
                block.append(idx)
            else:
                blocks.append(block)
                block = [idx]
        blocks.append(block)

        # Format blocks with line numbers
        formatted = []
        for block in blocks:
            for i in block:
                marker = "> " if matcher(lines[i]) else "  "
                formatted.append(f"{i + 1:4d} |{marker}{lines[i].rstrip()}")
            formatted.append("     ...")

        block_text = f"\n{filepath}:\n" + "\n".join(formatted)
        # Усечение «на лету», чтобы не копить гигантский вывод в память.
        if out_len + len(block_text) > SEARCH_MAX_OUTPUT_CHARS:
            remaining = SEARCH_MAX_OUTPUT_CHARS - out_len
            if remaining > 0:
                results.append(block_text[:remaining])
            truncated = True
            break
        results.append(block_text)
        out_len += len(block_text)

    if not results:
        msg = f" No matches found for '{pattern}'"
        if skipped:
            msg += f" (пропущено {skipped} бинарных/слишком больших файлов)"
        return err(msg)

    output = ENVIRONMENT_PREFIX + " " + f"Found {total} matches"
    if skipped:
        output += f" (пропущено {skipped} бинарных/больших файлов)"
    if truncated:
        output += (f" — результат усечён: показаны первые {file_count} файлов / "
                   f"≤{SEARCH_MAX_OUTPUT_CHARS} символов")
    output += ":\n" + "\n".join(results)
    return output + f"\n{ENVIRONMENT_PREFIX_END}"
