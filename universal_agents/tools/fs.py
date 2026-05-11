import datetime
import difflib

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
      path=("str", "Optional new working dir. Use '..' to go to the parent dir"))
def cwd(path: str = None):
    if path:
        try:
            os.chdir(path)
            return f'{ENVIRONMENT_PREFIX} Successfully set cwd'
        except Exception as e:
            return f"{ENVIRONMENT_PREFIX} Error changing cwd: {e}"
    return os.getcwd()

@tool(
    description="Exact-string replacer in file",
    requires_confirmation=True,
    path=("str", "File path"),
    old=("str", "Exact text to replace. Supports \\n for multiline blocks. If '' passed then replaces whole content"),
    new=("str", "New text to replace the old with. Also supports \\n"),
    mode=("str", "'one' for 1 exclusive match, otherwise 'all' (default 'one')")
)
def edit_file(path: str, old: str, new: str, mode: str = "one"):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Специальная обработка: если old == '', заменяем весь файл
    if old == '':
        new_content = new
    else:
        matches = []
        idx = 0
        search_len = max(len(old), 1)
        while True:
            pos = content.find(old, idx)
            if pos == -1: break
            matches.append(pos)
            idx = pos + search_len

        if not matches:
            return f"No matches found for old substring."

        m_mode = mode.strip().lower()

        if m_mode == "one" and len(matches) > 1:
            return f"Found {len(matches)} matches. Make old substring more specific or use mode='all'."

        new_content = content
        for pos in reversed(matches):
            new_content = new_content[:pos] + new + new_content[pos + len(old):]

    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
    except Exception as e:
        return f"Write failed: {e}"

    if old == '':
        return f"File fully replaced with new content '{new[:20]}...'"

    if m_mode == "one":
        old_lines = content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)

        diff = list(difflib.unified_diff(
            old_lines, new_lines,
            fromfile='', tofile='',
            lineterm="",
            n=1
        ))

        # Убираем технические строки
        diff_lines = [line for line in diff
                      if not line.startswith('---')
                      and not line.startswith('+++')
                      and not line.startswith('@@')]

        result = ["Successfully replaced:"]

        # Находим номер первой строки в контексте
        pos = matches[0]
        start_line = content[:pos].count('\n') - 1  # -1 чтобы захватить предыдущую строку
        if start_line < 0:
            start_line = 0

        current_line = start_line

        for line in diff_lines:
            stripped = line[2:].rstrip('\n') if len(line) > 2 else line.rstrip('\n')

            if line.startswith('  '):   # контекст
                result.append(f"{current_line:2d}   {stripped}")
            elif line.startswith('- '): # удалено
                result.append(f"{current_line:2d} - {stripped}")
                current_line += 1
            elif line.startswith('+ '): # добавлено
                result.append(f"   + {stripped}")  # не увеличиваем номер, т.к. это новая строка
            else:
                result.append(f"{current_line:2d}   {stripped}")
                current_line += 1

        return "\n".join(result)

    # Режим 'all' — показываем контекст для первых 3 совпадений
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


import os

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
