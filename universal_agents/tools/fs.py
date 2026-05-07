import datetime
import os
import re
from collections import defaultdict

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


@tool(description="Gets file content or dir tree",
      path=("str", "Optional path to file/dir (default '.'). Use '..' to open parent dir"))
def read(path: str = '.'):
    if not os.path.exists(path): raise FileNotFoundError(f"Path not found: {path}")
    try:
        mtime = datetime.datetime.fromtimestamp(os.stat(path).st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        if os.path.isfile(path):
            try:
                with open(path, 'r', encoding='utf-8', errors='strict') as f:
                    return f"File: {path}\nModified: {mtime}\nContent:\n---\n{f.read()}"
            except UnicodeDecodeError:
                return "Error: Cannot read binary files (failed UTF-8 decode)"
        elif os.path.isdir(path):
            return f"Directory Tree: {os.path.abspath(path)}\nModified: {mtime}\n\n{FS._build_tree(path)}"
        raise RuntimeError("Unexpected file type")
    except Exception as e:
        raise PermissionError(f"Error accessing {path}: {e}")

@tool(description="Searches files by regex",
      pattern=("str", "Regex to search for"),
      path=("str", "Optional base dir (default '.')"))
def search_files(pattern: str, path: str = ""):
    if not os.path.isdir(path): raise FileNotFoundError(f"Base path not found: {path}")
    folders_map = defaultdict(list)
    for root, _, files in os.walk(path):
        for f in files:
            if re.search(pattern, f):
                rel = os.path.relpath(root, path)
                folders_map["" if rel == "" else rel].append(f)
    if not folders_map: return "No matches"

    lines = []
    for folder in sorted(folders_map.keys()):
        lines.append(f"{folder}/:")
        for fname in sorted(folders_map[folder]):
            lines.append(f"  - {fname}")
    return "\n".join(lines)

@tool(description="Gets or changes current working dir",
      path=("str", "Optional new working dir. Use '..' to go to the parent dir"))
def cwd(path: str = None):
    if path:
        try:
            os.chdir(path)
            return 'Successfully changed cwd'
        except Exception as e:
            return f"Error changing cwd: {e}"
    return os.getcwd()

# @tool(
#     description="Exact-string replacer in file",
#     requires_confirmation=True,
#     path=("str", "File path"),
#     old=("str", "Exact text to replace. Supports \\n for multiline blocks"),
#     new=("str", "New text to replace the old with. Also supports \\n"),
#     mode=("str", "Optional. 'one' for 1 exclusive match, otherwise 'all' (default 'one')")
# )
# def edit_file(path: str, old: str, new: str, mode: str = "one"):
#     if not os.path.isfile(path): raise FileNotFoundError(path)
#     with open(path, "r", encoding="utf-8") as f: content = f.read()
#
#     matches = []
#     idx = 0
#     search_len = max(len(old), 1)
#     while True:
#         pos = content.find(old, idx)
#         if pos == -1: break
#         matches.append(pos)
#         idx = pos + search_len
#
#     if not matches:
#         return f"No matches found for old substring."
#
#     m_mode = mode.strip().lower()
#
#     if m_mode == "one" and len(matches) > 1:
#         return f"Found {len(matches)} matches. Make old substring more specific for precise edits or use mode='all'."
#
#     new_content = content
#     for pos in reversed(matches):
#         new_content = new_content[:pos] + new + new_content[pos + len(old):]
#
#     try:
#         with open(path, "w", encoding="utf-8") as f: f.write(new_content)
#     except Exception as e:
#         return f"Write failed: {e}"
#
#     # Упрощённый вывод
#     if m_mode == "one":
#         return "Successfully replaced"
#
#     # Режим 'all' — показываем контекст для первых 3 совпадений
#     lines = content.splitlines(True)
#     display_limit = min(len(matches), 3)
#     preview = [f"Successfully replaced {len(matches)} match(es):\n"]
#
#     for i, pos in enumerate(matches[:display_limit]):
#         safe = content[pos:pos+len(old)].replace('\n', '\\n').replace('\t', '\\t')[:40]
#         ls = content[:pos].count('\n')
#         ws, we = max(0, ls - 1), min(len(lines), ls + 2)
#
#         preview.append(f"{i+1}. `{safe}` in:")
#         for ln in range(ws, we):
#             preview.append(f"     {lines[ln].rstrip()}")
#         preview.append("---")
#
#     if len(matches) > display_limit:
#         preview.append(f"... and {len(matches) - display_limit} more matches.")
#
#     return "\n".join(preview)


import os
import re

@tool(
    description="""Replace text in file in a range between [old_first_line; old_last_line] inclusively.
    If old_last_line is not provided, replaces only the old_first_line line.
    Both old_first_line & old_last_line must be only single lines (no '\\n' inside).
    """,
    requires_confirmation=True,
    path=("str", "File path"),
    old_first_line=(
            "str",
            "START line to replace. MUST be a single line (no '\\n' inside). "
    ),
    old_last_line=(
            "str",
            "Optional END line to replace. If provided, MUST be a single line without '\\n'. "
    ),
    new=("str", "New string to replace everything from old_first_line to old_last_line inclusively"),
    mode=("str", "'one' (default) or 'all'")
)
def edit_file(path: str, old_first_line: str, new: str, old_last_line: str = None, mode: str = "one"):
    if not os.path.isfile(path):
        return f"File not found: {path}"
    if not old_first_line:
        return "Error: old_first_line cannot be empty."
    if mode not in ("one", "all"):
        return f"Error: mode must be 'one' or 'all', got '{mode}'"

    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        return f"Read failed: {e}"

    # Распаковка строковых литералов (\n, \t и т.п.)
    def unescape(s: str) -> str:
        s = s.replace('\\r\\n', '\r\n')
        s = s.replace('\\n', '\n')
        s = s.replace('\\r', '\r')
        s = s.replace('\\t', '\t')
        return s

    old_start_raw = unescape(old_first_line)
    new_raw = unescape(new)

    # Если old_last_line не указан, ищем только одну строку
    is_single_line = not old_last_line
    if not is_single_line:
        old_end_raw = unescape(old_last_line)
    else:
        old_end_raw = old_start_raw  # Для однострочного режима границы совпадают

    # Гарантируем однострочность границ
    if '\n' in old_start_raw or '\r' in old_start_raw:
        return "Error: old_first_line must be a single line (no newlines)."
    if not is_single_line and ('\n' in old_end_raw or '\r' in old_end_raw):
        return "Error: old_last_line must be a single line (no newlines)."

    # Упрощённая компиляция: шаблон заведомо однострочный
    start_re = re.compile(re.escape(old_start_raw), re.IGNORECASE)
    end_re = re.compile(re.escape(old_end_raw), re.IGNORECASE) if not is_single_line else start_re

    matches = []
    search_start = 0

    while True:
        start_match = start_re.search(content, search_start)
        if not start_match:
            break
        start_pos = start_match.start()

        if is_single_line:
            # Для одной строки заменяем только её
            end_pos = start_match.end()
        else:
            # Для диапазона ищем конечную границу
            end_match = end_re.search(content, start_pos + len(old_start_raw))
            if not end_match:
                break
            end_pos = end_match.end()

        matches.append((start_pos, end_pos))
        search_start = end_pos

    if not matches:
        if is_single_line:
            return f"No matches found for: {old_start_raw[:80]!r}"
        else:
            return (f"No matches found.\n"
                    f"  START: {old_start_raw[:80]!r}\n"
                    f"  END:   {old_end_raw[:80]!r}")

    if mode == "one" and len(matches) > 1:
        report = [f"Found {len(matches)} matches. Use mode='all' or make boundaries more specific:\n"]

        for i, (s, e) in enumerate(matches[:3], 1):
            start_line = content[:s].count('\n')
            block_preview = content[s:e].splitlines()
            if is_single_line:
                report.append(f"{i}. At line {start_line+1}: {old_start_raw[:40]!r}")
            else:
                report.append(f"{i}. At line {start_line+1} (starts with {old_start_raw[:40]!r}):")
                for line in block_preview[:2]:
                    report.append(f"     {line[:80]}")
                if len(block_preview) > 2:
                    report.append(f"     ... ({len(block_preview)-2} more lines)")
                report.append("---")

        if len(matches) > 3:
            report.append(f"... and {len(matches)-3} more matches.")
        return "\n".join(report)

    new_content = content
    for start_pos, end_pos in reversed(matches):
        new_content = new_content[:start_pos] + new_raw + new_content[end_pos:]

    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
    except Exception as e:
        return f"Write failed: {e}"

    if mode == "one":
        if is_single_line:
            return f"Successfully replaced {old_start_raw[:40]!r} with {new_raw[:40]!r}"
        else:
            return f"Successfully replaced block from {old_start_raw[:40]!r} to {old_end_raw[:40]!r}"
    else:
        report = [f"Successfully replaced {len(matches)} match(es):\n"]

        for i, (s, e) in enumerate(matches[:3], 1):
            start_line = content[:s].count('\n')
            end_line = content[:e].count('\n')

            if is_single_line:
                report.append(f"{i}. Line {start_line+1}: {content[s:e].rstrip()!r} -> {new_raw[:40]!r}")
            else:
                report.append(f"{i}. Block from line {start_line+1} to {end_line+1}:")

                lines = content.splitlines(keepends=True)
                ctx_start = max(0, start_line - 2)
                ctx_end = min(len(lines), end_line + 2)
                for ln in range(ctx_start, ctx_end):
                    if ln == start_line:
                        report.append(f"  >>> {lines[ln].rstrip()}")
                    elif ln == end_line and end_line != start_line:
                        report.append(f"  <<< {lines[ln].rstrip()}")
                    else:
                        report.append(f"      {lines[ln].rstrip()}")
                report.append("---")

        if len(matches) > 3:
            report.append(f"... and {len(matches)-3} more blocks.")
        return "\n".join(report)