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

    # Режим по номерам строк: заменяем строки [start_line..end_line] (1-based,
    # инклюзивно) на 'new'. Конечные символы '\n' строк старого диапазона
    # отбрасываются; 'new' записывается как есть (без автоматического '\n').
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
    """Читает файл как UTF-8 текст. Возвращает (content, None) или (None, error_msg)."""
    try:
        with open(path, 'r', encoding='utf-8', errors='strict') as f:
            return f.read(), None
    except UnicodeDecodeError:
        return None, err(": Cannot read binary files (failed UTF-8 decode)")


def _summarize_file(path: str, content: str, agent) -> str:
    """Строит структурный скелет/саммари файла через изолированный субагент.

    Субагент наследует ПОЛНЫЙ набор схем родителя (KV-cache safe), но все
    инструменты запрещены (denied_tools='*'): попытка вызова вернёт ошибку."""
    sub = agent.make_sub_agent(
        denied_tools="*",
        max_iter=1,
        temp=0.2,
    )

    task = (
        f"{ENVIRONMENT_PREFIX} NOW IGNORE previous instructions! Act as file content structure summarizer. "
        "Answer starting with tag <content_structure> with a very short compact content structure summary (like in books) of top-level items (signatures of classes, functions, etc) and their EXACT "
        f"line numbered ranges (for example `L1-10 func()\\n` or `L8 class Some\\n`. NO COMMS!\n {ENVIRONMENT_PREFIX_END}"
    )

    max_tokens = agent.token_tracker.get_remaining() // Config.SUMMARIZATION_THRESHOLD_DIVIDER
    max_chars = int(max_tokens * Config.CHARS_PER_TOKEN)
    truncated = len(content) > max_chars
    snippet = content[:max_chars]
    start = 1
    selected = snippet.split("\n")
    numbered_text = "\n".join(f"{start + i} {line}" for i, line in enumerate(selected))

    task = '```\n' + numbered_text + "\n```" + task
    if truncated:
        task += "\n\n(File is truncated due to remaining memory)"

    result = sub.run(task, '<sub_agent><content_structure>L').strip()
    # Субагент ведёт свой изолированный трекер; его последний замер относим к
    # агенту, чтобы «Tokens spent» отражал самый свежий вызов (в т.ч. чтение файла).
    if sub._own_tracker.last_usage:
        agent.token_tracker.update_from_usage(sub._own_tracker.last_usage)
    if not result:
        return err(f" (Empty summary for {path}. It's probably an error, try one more time...)")
    return result


def _limit_read_chunk(lines: list[str]) -> tuple[list[str], bool]:
    """Порция чтения: не более MAX_READ_CHARS_PER_CALL символов и не более
    MAX_READ_LINES_PER_CALL строк. Строка, на которой превышен символьный
    лимит, включается целиком — порция всегда заканчивается концом строки.
    Возвращает (строки_порции, обрезано_ли)."""
    out: list[str] = []
    total_chars = 0
    for line in lines[:Config.MAX_READ_LINES_PER_CALL]:
        out.append(line)
        total_chars += len(line) + 1  # +1 за перевод строки
        if total_chars >= Config.MAX_READ_CHARS_PER_CALL:
            break
    return out, len(out) < len(lines)


@tool(description="Reads a file or shows a directory tree. Small files are returned fully. "
                  "Large files WITHOUT start_line/end_line return a structural skeleton exactly once "
                  "(repeated whole-file reads of an unchanged file are forbidden). To read actual code "
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
                # Порционное чтение ДИАПАЗОНА с жёстким лимитом объёма:
                # модель не должна выгружать большой файл в контекст за один раз.
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
                # Реальные номера строк файла, чтобы модель могла точно редактировать
                numbered = [f"{start + i} {line}" for i, line in enumerate(selected)]
                content = (
                    f"{ENVIRONMENT_PREFIX} File: {path}\nModified: {mtime}\n"
                    f"Lines {start}-{actual_end}/{total}:\n---\n"
                    + ("\n".join(numbered) if numbered else "")
                    + (f"\n{ENVIRONMENT_PREFIX} Output limited to ~{Config.MAX_READ_CHARS_PER_CALL} chars per read. "
                       f"Continue with start_line={actual_end + 1}.{ENVIRONMENT_PREFIX_END}"
                       if truncated else "")
                    + f"\n{ENVIRONMENT_PREFIX_END}"
                )
                return content
            # Без диапазона
            raw, read_err = _read_text(path)
            if read_err:
                return read_err
            lines = raw.splitlines()
            if len(raw) <= _SUMMARY_THRESHOLD:
                numbered = [f"{i+1} {line}" for i, line in enumerate(lines)]
                content = f"{ENVIRONMENT_PREFIX} File: {path}\nModified: {mtime}\nContent:\n---\n" + ("\n".join(numbered) if numbered else "") + f"\n{ENVIRONMENT_PREFIX_END}"
                return _read_or_skip(agent, path, raw, content)
            # БОЛЬШОЙ файл: ровно один раз отдаём структурный скелет;
            # повторные целиком-файловые чтения без изменений запрещены.
            total = len(lines)
            disk_hash = _content_hash(raw)
            if agent.file_states.should_skip(path, disk_hash):
                return err(
                    f": re-reading file '{path}' is not allowed - it is unchanged "
                    f"since the last read and its content/skeleton is already in context. "
                    f"Use start_line/end_line to read specific portions "
                    f"(~{Config.MAX_READ_CHARS_PER_CALL} chars at a time)."
                )
            skeleton = ""
            if Config.BIG_FILE_SKELETON:
                skeleton = _summarize_file(path, raw, agent).strip()
            skeleton_block = (
                f"Structural skeleton (read ONCE; repeated reads are forbidden):\n---\n{skeleton}\n"
                if skeleton else ""
            )
            content = (
                f"{ENVIRONMENT_PREFIX} File: {path}\nModified: {mtime}\nTotal lines: {total}\n"
                f"{skeleton_block}"
                f"To read the actual code use start_line/end_line (limited to ~{Config.MAX_READ_CHARS_PER_CALL} chars per call)."
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


def _read_or_skip(agent: 'LLMAgent', path: str, raw: str, content: str) -> str:
    """Возвращает контент чтения либо пропуск, если файл не менялся с прошлого чтения.

    Запоминает хэш содержимого и (через agent._read_registrations) указывает,
    какой read-результат нужно привязать к записи в _execute_tools.
    """
    disk_hash = _content_hash(raw)
    if agent.file_states.should_skip(path, disk_hash):
        return err(
            f": re-reading file '{path}' is not allowed - it is unchanged "
            f"since the last read in this session and its content is already in context. "
            f"Do NOT call 'read' again on unchanged files; use the content you already have."
        )
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
    CONTEXT = 1  # lines before/after match

    for filepath in sorted(files):
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

        results.append(f"\n{filepath}:\n" + "\n".join(formatted))

    if not results:
        return err(f" No matches found for '{pattern}'")

    output = ENVIRONMENT_PREFIX + " " + f"Found {total} matches:\n" + "\n".join(results)
    return output + f"\n{ENVIRONMENT_PREFIX_END}"
