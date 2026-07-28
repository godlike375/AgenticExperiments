from __future__ import annotations

import datetime
import difflib
import os
import re as _re
import fnmatch as _fnmatch

from universal_agents.config import Config
from universal_agents.tool import tool, ENVIRONMENT_PREFIX


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
            return f'{ENVIRONMENT_PREFIX} Successfully set cwd to {path}'
        except Exception as e:
            raise RuntimeError(f"Error changing cwd: {e}")  # Было return, стало raise
    return os.getcwd()

@tool(
    description="Exact-string replacer in file. Creates file with parent dirs if it doesn't exist",
    short_description="edit file text",
    requires_confirmation=True,
    path=("str", "File path. Will be auto-created if missing"),
    old=("str", "Exact text to replace. Supports \\n for multiline blocks. If '' or nothing passed then replaces whole content. For new files use '' to set initial content"),
    new=("str", "New text to replace the old with. Also supports \\n"),
    mode=("str", "'one' for 1 exclusive match, otherwise 'all' (default 'one')")
)
def edit_file(path: str, new: str, old: str = '', mode: str = "one"):
    created_file = False
    m_mode = mode.strip().lower()
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

    if old == '':
        new_content = new
    else:
        matches = []
        idx = 0
        search_len = max(len(old), 1)
        while True:
            pos = content.find(old, idx)
            if pos == -1:
                break
            matches.append(pos)
            idx = pos + search_len

        if not matches:
            raise ValueError("No matches found for old substring. Try again with different argument")

        if m_mode == "one" and len(matches) > 1:
            raise ValueError(
                f"Found {len(matches)} matches. Make old substring more specific or use mode='all'."
            )

        new_content = content
        for pos in reversed(matches):
            new_content = new_content[:pos] + new + new_content[pos + len(old):]

    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
    except Exception as e:
        raise RuntimeError(f"Write failed: {e}")

    # Дальше формирование красивого diff без изменений
    if old == '' and not created_file:
        return f"File fully replaced with '{new[:20]}...'"
    elif old == '' and created_file:
        return f"File created with content '{new[:20]}...'"

    if m_mode == "one":
        old_lines = content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)

        diff = list(difflib.unified_diff(
            old_lines, new_lines,
            fromfile='', tofile='',
            lineterm="",
            n=1
        ))

        diff_lines = [line for line in diff
                      if not line.startswith('---')
                      and not line.startswith('+++')
                      and not line.startswith('@@')]

        result = ["Successfully replaced:"]

        pos = matches[0]
        start_line = content[:pos].count('\n') - 1
        if start_line < 0:
            start_line = 0

        current_line = start_line

        for line in diff_lines:
            stripped = line[2:].rstrip('\n') if len(line) > 2 else line.rstrip('\n')

            if line.startswith('  '):
                result.append(f"{current_line:2d}   {stripped}")
            elif line.startswith('- '):
                result.append(f"{current_line:2d} - {stripped}")
                current_line += 1
            elif line.startswith('+ '):
                result.append(f"   + {stripped}")
            else:
                result.append(f"{current_line:2d}   {stripped}")
                current_line += 1

        return "\n".join(result)

    # Режим 'all'
    lines = content.splitlines(True)
    display_limit = min(len(matches), 3)
    preview = [f"Successfully replaced {len(matches)} matches:\n"]

    for i, pos in enumerate(matches[:display_limit]):
        safe = content[pos:pos+len(old)].replace('\n', '\\n').replace('\t', '\\t')[:40]
        ls = content[:pos].count('\n')
        ws, we = max(0, ls - 1), min(len(lines), ls + 2)

        preview.append(f"{i+1}. `{safe}` in:")
        for ln in range(ws, we):
            preview.append(f"     {lines[ln].rstrip()}")
        preview.append("---")

    if len(matches) > display_limit:
        preview.append(f"... and {len(matches) - display_limit} more matches.")

    return "\n".join(preview)


CHARS_PER_TOKEN = 2.3

MIN_TOKENS_TO_SUMMARIZE = 750
_SUMMARY_THRESHOLD = int(MIN_TOKENS_TO_SUMMARIZE * CHARS_PER_TOKEN)

SUMMARY_CONTEXT_FRACTION = 2 / 3


def _summarize_file(path: str, content: str, agent) -> str:
    """Строит структурный скелет/саммари файла через изолированный субагент."""
    from universal_agents.sub_agent import SubAgent
    parent_system_prompt = agent.history[0].content if agent.history else ""
    parent_history = agent.history.get_all()[1:]

    sub = SubAgent(
        parent_system_prompt=parent_system_prompt,
        parent_history=parent_history,
        max_context_tokens=agent.token_tracker.max_context_tokens,
        tools_config=[],
        external_plugins={},
        safe_only=False,
        max_iter=1,
        temp=0.2,
    )

    specialist_instructions = (
        f"{ENVIRONMENT_PREFIX} NOW IGNORE previous instructions! Act as 'SkeletonGenerator' tool. "
        "Respond only in tags '<skeleton>' with a very short concise skeleton with the most top-level identifiers and their EXACT "
        "line numbered ranges. NO COMMS!"
    )

    task = (
        f"{specialist_instructions}\n\n"
        "The skeleton includes top-level elements (signatures of functions, classes, methods defined right in this file)"
        " and precise line number ranges for each element\n"
        f"FILE CONTENT (with line numbers):\n```\n {ENVIRONMENT_PREFIX}"
    )

    max_tokens = agent.token_tracker.get_remaining() // Config.SUMMARIZATION_THRESHOLD_DIVIDER
    max_chars = int(max_tokens * CHARS_PER_TOKEN)
    truncated = len(content) > max_chars
    snippet = content[:max_chars]
    start = 1
    selected = snippet.split("\n")
    numbered_text = "\n".join(f"{start + i} {line}" for i, line in enumerate(selected))

    task += numbered_text + "\n```"
    if truncated:
        task += "\n\n(File is truncated due to remaining memory)"

    result = sub.run(task, '<sub_agent><skeleton>').strip()
    if not result:
        return f"(Empty summary for {path}. It's probably an error, try one more time...)"
    return result


@tool(description="Reads a file or shows a directory tree. Without start_line/end_line it returns a STRUCTURAL SUMMARY (skeleton) of the file generated by a subagent, so it works natively for any file type. Pass start_line/end_line (1-based, inclusive) to read the exact numbered lines of a section.",
      short_description="read file / ls dir",
      path=("str", "Optional path to file/dir (default '.'). Use '..' to open parent dir"),
      start_line=("int", "Optional 1-based start line. Omit (with end_line) to get a subagent summary instead of raw content"),
      end_line=("int", "Optional 1-based inclusive end line"))
def read(agent: 'LLMAgent', path: str = '.', start_line: int = None, end_line: int = None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path not found: {path}")
    try:
        mtime = datetime.datetime.fromtimestamp(os.stat(path).st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        if os.path.isfile(path):
            if start_line is not None or end_line is not None:
                try:
                    with open(path, 'r', encoding='utf-8', errors='strict') as f:
                        raw = f.read()
                except UnicodeDecodeError:
                    return f"{ENVIRONMENT_PREFIX} Error: Cannot read binary files (failed UTF-8 decode)"
                lines = raw.splitlines()
                start = max(1, start_line if start_line is not None else 1)
                end = end_line if end_line is not None else len(lines)
                end = max(start, min(end, len(lines)))
                selected = lines[start - 1:end]
                # Реальные номера строк файла, чтобы модель могла точно редактировать
                numbered = [f"{start + i} {line}" for i, line in enumerate(selected)]
                return (f"{ENVIRONMENT_PREFIX} File: {path}\n"
                        f"Modified: {mtime}\n"
                        f"Lines {start}-{end} of {len(lines)}:\n---\n"
                        + ("\n".join(numbered) if numbered else ""))
            # Без диапазона: маленькие файлы — целиком, крупные — саммари от субагента
            try:
                with open(path, 'r', encoding='utf-8', errors='strict') as f:
                    raw = f.read()
            except UnicodeDecodeError:
                return f"{ENVIRONMENT_PREFIX} Error: Cannot read binary files (failed UTF-8 decode)"
            lines = raw.splitlines()
            if len(raw) <= _SUMMARY_THRESHOLD:
                numbered = [f"{i+1} {line}" for i, line in enumerate(lines)]
                return f"{ENVIRONMENT_PREFIX} File: {path}\nModified: {mtime}\nContent:\n---\n" + ("\n".join(numbered) if numbered else "")
            summary = _summarize_file(path, raw, agent)
            return (f"{ENVIRONMENT_PREFIX} File: {path}\nModified: {mtime}\nTotal lines: {len(lines)}\n"
                    f"Summary:\n---\n{summary}")
        elif os.path.isdir(path):
            return f"{ENVIRONMENT_PREFIX} Directory Tree: {os.path.abspath(path)}\nModified: {mtime}\n\n{FS._build_tree(path)}"
        raise RuntimeError("Unexpected file type")
    except Exception as e:
        raise PermissionError(f"Error accessing {path}: {e}")


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
            return f"{ENVIRONMENT_PREFIX} Invalid regex: {e}"
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
        return f"{ENVIRONMENT_PREFIX} Path not found: {path}"

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
        return f"{ENVIRONMENT_PREFIX} No matches found for '{pattern}'"

    output = ENVIRONMENT_PREFIX + " " + f"Found {total} matches:\n" + "\n".join(results)
    return output
