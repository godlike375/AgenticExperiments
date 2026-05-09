import datetime
import os
import re
from collections import defaultdict

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

@tool(description="Searches files by regex",
      pattern=("str", "Regex to search for"),
      path=("str", "Optional base dir (default '.')"))
def search_files(pattern: str, path: str = ""):
    if not os.path.isdir(path): raise FileNotFoundError(f"{ENVIRONMENT_PREFIX} Base path not found: {path}")
    folders_map = defaultdict(list)
    for root, _, files in os.walk(path):
        for f in files:
            if re.search(pattern, f):
                rel = os.path.relpath(root, path)
                folders_map["" if rel == "" else rel].append(f)
    if not folders_map: return f"{ENVIRONMENT_PREFIX} No matches"

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
            return f'{ENVIRONMENT_PREFIX} Successfully set cwd'
        except Exception as e:
            return f"{ENVIRONMENT_PREFIX} Error changing cwd: {e}"
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

@tool(description="Gets file content or dir tree (lines are numbered for precise editing)",
      path=("str", "Optional path to file/dir (default '.'). Use '..' to open parent dir"))
def read(path: str = '.'):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path not found: {path}")
    try:
        mtime = datetime.datetime.fromtimestamp(os.stat(path).st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        if os.path.isfile(path):
            try:
                with open(path, 'r', encoding='utf-8', errors='strict') as f:
                    raw = f.read()
                    lines = raw.splitlines()
                    numbered = [f"{i+1} {line}" for i, line in enumerate(lines)]
                    return f"{ENVIRONMENT_PREFIX} File: {path}\nModified: {mtime}\nContent:\n---\n" + ("\n".join(numbered) if numbered else "")
            except UnicodeDecodeError:
                return f"{ENVIRONMENT_PREFIX} Error: Cannot read binary files (failed UTF-8 decode)"
        elif os.path.isdir(path):
            return f"{ENVIRONMENT_PREFIX} Directory Tree: {os.path.abspath(path)}\nModified: {mtime}\n\n{FS._build_tree(path)}"
        raise RuntimeError("Unexpected file type")
    except Exception as e:
        raise PermissionError(f"Error accessing {path}: {e}")


@tool(
    description="""Replace a contiguous block of lines in a file by their line numbers.
    start_line & end_line refer to the LINE NUMBERS (1-based) shown by open().
    Range is INCLUSIVE. If end_line is omitted, only that single line is replaced.
    new_text: replacement text. Use '\\n' for new lines within the replacement.""",
    requires_confirmation=True,
    path=("str", "File path"),
    start_line=("str", "START line number (integer) from open() output."),
    end_line=("str", "Optional END line number (integer). Defaults to START."),
    new_text=("str", "New content to replace the specified range with.")
)
def edit_file(path: str, start_line: str, new_text: str, end_line: str = None):
    if not os.path.isfile(path):
        return f"File not found: {path}"

    # Парсинг номеров строк
    try:
        start_num = int(start_line)
        end_num = int(end_line) if end_line is not None else start_num
    except ValueError:
        return "Error: Line numbers must be valid integers."

    if start_num < 1 or end_num < 1:
        return "Error: Line numbers must be >= 1."
    if start_num > end_num:
        return "Error: Start line cannot be greater than end line."

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        file_lines = raw.split('\n') if raw != "" else []
    except Exception as e:
        return f"Read failed: {e}"

    # Проверка выхода за границы файла
    if start_num > len(file_lines) or end_num > len(file_lines):
        return f"Error: File has {len(file_lines)} lines. Requested range ({start_num}-{end_num}) exceeds file size."

    # Перевод в 0-based индексы
    idx_start = start_num - 1
    idx_end = end_num - 1

    # Разбиваем новый контент на строки для подстановки
    replacement_lines = new_text.split('\n')

    # Слайс-присваивание заменяет ровно указанный диапазон
    file_lines[idx_start : idx_end + 1] = replacement_lines

    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write('\n'.join(file_lines))
    except Exception as e:
        return f"Write failed: {e}"

    return f"{ENVIRONMENT_PREFIX} Successfully replaced lines {start_num}–{end_num} with {len(replacement_lines)} new line(s)."