from __future__ import annotations

import datetime
import difflib
import os
import re as _re
import fnmatch as _fnmatch
from typing import Optional

from universal_agents.config import Config
from universal_agents.constants import ENVIRONMENT_PREFIX
from universal_agents.file_states import _content_hash
from universal_agents.llm_client import LLMClient
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


CHARS_PER_TOKEN = Config.CHARS_PER_TOKEN

MIN_TOKENS_TO_SUMMARIZE = Config.MIN_TOKENS_TO_SUMMARIZE
_SUMMARY_THRESHOLD = int(MIN_TOKENS_TO_SUMMARIZE * CHARS_PER_TOKEN)


def _read_text(path: str) -> tuple:
    """Читает файл как UTF-8 текст. Возвращает (content, None) или (None, error_msg)."""
    try:
        with open(path, 'r', encoding='utf-8', errors='strict') as f:
            return f.read(), None
    except UnicodeDecodeError:
        return None, f"{ENVIRONMENT_PREFIX} Error: Cannot read binary files (failed UTF-8 decode)"


def _parse_important_lines(lines: list[str], ranges_text: str) -> list[int]:
    """Разбирает '[1, 3, 5:8, 12:14]' или '1-20, 35-50' в список 1-based индексов.

    Диапазоны 'a:b'/'a-b' трактуются инклюзивно (конечная строка тоже сохраняется).
    """
    inner = ranges_text or ""
    if '[' in inner and ']' in inner:
        inner = inner[inner.find('[') + 1:inner.rfind(']')]
    kept: set[int] = set()
    for token in inner.replace(';', ',').split(','):
        token = token.strip()
        if not token:
            continue
        sep = ':' if ':' in token else ('-' if '-' in token else None)
        if sep:
            try:
                a, b = token.split(sep, 1)
                kept.update(range(int(a.strip()), int(b.strip()) + 1))
            except ValueError:
                continue
        else:
            try:
                kept.add(int(token))
            except ValueError:
                continue
    return sorted(k for k in kept if 1 <= k <= len(lines))


def _interactive_file_extract(agent, path: str, content: str, mtime: str) -> Optional[str]:
    """Спрашивает LLM, какие строки файла наиболее полезны, и сохраняет только их.

    Модель отвечает одной строкой 'most_important_lines: [1, 3, 5:8, 12:14]' —
    выбранные строки вырезаются. К ним докидывается скелет от субагента
    (_summarize_file). None — если модель не дала пригодного ответа.
    """
    lines = content.splitlines()
    total = len(lines)

    max_tokens = agent.token_tracker.get_remaining() // Config.SUMMARIZATION_THRESHOLD_DIVIDER
    max_chars = int(max_tokens * Config.CHARS_PER_TOKEN)
    truncated = len(content) > max_chars
    snippet = content[:max_chars]
    numbered = "\n".join(f"{i + 1} {line}" for i, line in enumerate(snippet.split("\n")))

    prompt = (
        f"{ENVIRONMENT_PREFIX} The file '{path}' is {total} lines.\n"
        f"File content (line-numbered):\n{numbered}"
        "The file is too large to keep fully in memory. You need to "
        f"decide which lines are MOST useful for you now! Other ones will be REMOVED."
        f" right after your reply.\n"
        f"Reply just with a line in format:\n"
        f"`most_important_lines: [1, 3, 5:8, 12:14]` (just an example) \n"
        f"Do NOT call tools or write any other free text.\n"
    )
    if truncated:
        prompt += "\n\n(File is truncated due to remaining memory)"

    history_msgs = [m.to_api_dict() for m in agent.history.get_all()]
    msgs = history_msgs + [{"role": "user", "content": prompt}]
    # Tools передаём ОБЯЗАТЕЛЬНО: они вшиваются в системный промпт/префикс, и без
    # них не совпадает KV-cache префикс (вызов не переиспользует кэш родителя).
    reply = None
    for attempt in range(Config.ERROR_RECOVERY_RETRIES + 1):
        msg_obj, err, usage = LLMClient.call(
            msgs, temp=0.2, timeout=60, tools=(agent.tools if agent.tools else None)
        )
        if err:
            break
        if msg_obj and msg_obj.content and msg_obj.content.strip():
            reply = msg_obj.content.strip()
            break
        # Модель вместо текста выдала tool-call (например пустой ?()) — как в
        # основном цикле: инжектим запрет инструментов и просим перегенерировать.
        correction = (
            f"{ENVIRONMENT_PREFIX} You tried to call a tool, but tools are FORBIDDEN in this "
            f"extraction step. Answer in PLAIN TEXT only, in the form 'most_important_lines: [...]'."
        )
        agent.on_system_msg("[READ EXTRACT] Model returned a tool call instead of text; retrying with a tool ban...")
        msgs = msgs + [{"role": "user", "content": correction}]

    if not reply:
        return None

    lower = reply.lower()
    lines_text = None
    if "most_important_lines" in lower:
        _, _, lines_text = reply.partition("most_important_lines:")
    elif "lines:" in lower:
        _, _, lines_text = reply.partition("lines:")
    if lines_text is None:
        return None

    kept = _parse_important_lines(lines, lines_text)
    if not kept:
        return None

    kept_lines = [f"{i} {lines[i - 1]}" for i in kept]
    result = (f"{ENVIRONMENT_PREFIX} Most important file lines (for memory saving): ...\n"
              f"{ENVIRONMENT_PREFIX} File: {path}\nModified: {mtime}\nTotal lines: {total}\n"
              f"Kept lines ({len(kept)}/{total}):\n---\n"
              + "\n".join(kept_lines))

    # Скелет от субагента — отдельный, качественный, склеиваем с важными строками.
    if Config.INTERACTIVE_EXTRACT_WITH_SKELETON:
        skeleton = _summarize_file(path, content, agent).strip()
        if skeleton:
            result += f"\n\n--- File skeleton ---\n{skeleton}"

    agent.on_system_msg(f"[READ EXTRACT] Kept {len(kept)} of {total} lines from '{path}' (+ skeleton)")
    return result


def _summarize_file(path: str, content: str, agent) -> str:
    """Строит структурный скелет/саммари файла через изолированный субагент."""
    sub = agent.make_sub_agent(
        tools_config=[],
        external_plugins={},
        safe_only=False,
        max_iter=1,
        temp=0.2,
    )

    specialist_instructions = (
        f"{ENVIRONMENT_PREFIX} NOW IGNORE previous instructions! Act as SkeletonGenerator agent. "
        "Respond only in tags '<skeleton>' with a very short compact and concise skeleton with the most top-level identifiers and their EXACT "
        "line numbered ranges. NO COMMS!"
    )

    task = (
        f"{specialist_instructions}\n\n"
        "The skeleton includes top-level elements (signatures of functions, classes, methods defined right in this file)"
        " and precise line number ranges for each element\n"
        f"Now skeletonize it!"
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

    result = sub.run(task, '<sub_agent><skeleton>').strip()
    if not result:
        return f"Error (Empty summary for {path}. It's probably an error, try one more time...)"
    return result


@tool(description="Reads a file or shows a directory tree. Without start_line/end_line, small files are returned fully; large files go through interactive extraction (LLM keeps the most useful lines, optionally with a structural skeleton). Pass start_line/end_line (1-based, inclusive) to read the exact numbered lines of a section.",
      short_description="read file / ls dir",
      path=("str", "Optional path to file/dir (default '.'). Use '..' to open parent dir"),
      start_line=("int", "Optional 1-based start line. Omit (with end_line) to get a subagent summary instead of raw content"),
      end_line=("int", "Optional 1-based inclusive end line (supports negative values like Python slices)"))
def read(agent: 'LLMAgent', path: str = '.', start_line: int = None, end_line: int = None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path not found: {path}")
    try:
        mtime = datetime.datetime.fromtimestamp(os.stat(path).st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        if os.path.isfile(path):
            if start_line is not None or end_line is not None:
                raw, read_err = _read_text(path)
                if read_err:
                    return read_err
                lines = raw.splitlines()
                start = max(1, start_line if start_line is not None else 1)
                end = len(lines) if end_line is None else (len(lines) + end_line if end_line < 0 else end_line)
                end = max(start, min(end, len(lines)))
                selected = lines[start - 1:end]
                # Реальные номера строк файла, чтобы модель могла точно редактировать
                numbered = [f"{start + i} {line}" for i, line in enumerate(selected)]
                content = (f"{ENVIRONMENT_PREFIX} File: {path}\n"
                           f"Modified: {mtime}\n"
                           f"Lines {start}-{end} of {len(lines)}:\n---\n"
                           + ("\n".join(numbered) if numbered else ""))
                return _read_or_skip(agent, path, raw, content)
            # Без диапазона: маленькие файлы — целиком, крупные — интерактивная выемка
            raw, read_err = _read_text(path)
            if read_err:
                return read_err
            lines = raw.splitlines()
            if len(raw) <= _SUMMARY_THRESHOLD:
                numbered = [f"{i+1} {line}" for i, line in enumerate(lines)]
                content = f"{ENVIRONMENT_PREFIX} File: {path}\nModified: {mtime}\nContent:\n---\n" + ("\n".join(numbered) if numbered else "")
                return _read_or_skip(agent, path, raw, content)
            # Интерактивная выемка: LLM сама выбирает, что сохранить, остальное удаляется
            extracted = _interactive_file_extract(agent, path, raw, mtime)
            if extracted:
                return extracted
            # Фолбэк БЕЗ дополнительного LLM-вызова (чтобы не было двух
            # обращений к модели): отдаём усечённый префикс файла.
            max_tokens = agent.token_tracker.get_remaining() // Config.SUMMARIZATION_THRESHOLD_DIVIDER
            max_chars = int(max_tokens * Config.CHARS_PER_TOKEN)
            prefix_lines = raw[:max_chars].split("\n")
            numbered = [f"{i + 1} {line}" for i, line in enumerate(prefix_lines)]
            return (f"{ENVIRONMENT_PREFIX} File: {path}\nModified: {mtime}\nTotal lines: {len(lines)}\n"
                    f"Interactive extraction failed; showing truncated prefix. "
                    f"To see exact code pass start_line/end_line:\n---\n"
                    + "\n".join(numbered))
        elif os.path.isdir(path):
            return f"{ENVIRONMENT_PREFIX} Directory Tree: {os.path.abspath(path)}\nModified: {mtime}\n\n{FS._build_tree(path)}"
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
        return (f"{ENVIRONMENT_PREFIX} Error: re-reading file '{path}' is not allowed - it is unchanged "
                f"since the last read in this session and its content is already in context. "
                f"Do NOT call 'read' again on unchanged files; use the content you already have.")
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
            return f"{ENVIRONMENT_PREFIX} Error Invalid regex: {e}"
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
        return f"{ENVIRONMENT_PREFIX} Error Path not found: {path}"

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
        return f"{ENVIRONMENT_PREFIX} Error No matches found for '{pattern}'"

    output = ENVIRONMENT_PREFIX + " " + f"Found {total} matches:\n" + "\n".join(results)
    return output
