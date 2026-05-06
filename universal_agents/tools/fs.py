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
def open(path: str = '.'):
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