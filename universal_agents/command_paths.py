"""Извлечение файловых путей из командных строк (bash / PowerShell).

Алгоритм (в порядке улучшения точности):
1. Настоящий токенизатор с учётом одинарных/двойных кавычек и backtick-escape —
   а не примитивные регулярки.
2. Учёт команды: пропускаем имя команды и опции/флаги, не считаем их путями.
3. Раскрытие переменных окружения ($HOME, ${VAR}, %USERPROFILE%, $env:VAR).
4. Проверка существования: «голый» относительный токен без признаков пути
   считается путём только если существует на диске (снижает ложные срабатывания).

Нацелено на безопасность: лучше переспросить лишний раз, чем пропустить
внешний путь. Строковый анализ не является полной защитой (см. обвязку).
"""

import os
import re

# Разделители путей на Windows и *nix
_PATH_SEPS = ("\\", "/")

# Распространённые расширения, по которым можно опознать файл
_FILE_EXTS = {
    ".py", ".pyc", ".json", ".txt", ".md", ".log", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".env", ".csv", ".tsv", ".xml", ".html",
    ".js", ".ts", ".jsx", ".tsx", ".css", ".sh", ".bat", ".ps1", ".cmd",
    ".exe", ".dll", ".so", ".dylib", ".zip", ".tar", ".gz", ".7z", ".rar",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".pdf", ".sql", ".db", ".sqlite",
    ".lock", ".toml", ".yml", ".whl",
}

# Переменные окружения в разных синтаксисах: pattern -> имя переменной
_ENV_PATTERNS = (
    (re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}"), lambda m: m.group(1)),  # ${VAR}
    (re.compile(r"\$env:([A-Za-z_][A-Za-z0-9_]*)"), lambda m: m.group(1)),  # $env:VAR (PS)
    (re.compile(r"%([A-Za-z_][A-Za-z0-9_]*)%"), lambda m: m.group(1)),      # %VAR% (cmd)
)


def _expand_env(token: str) -> str:
    """Раскрывает переменные окружения ($HOME, ${VAR}, %VAR%, $env:VAR)."""
    result = token
    for pattern, getter in _ENV_PATTERNS:
        def _repl(m, _getter=getter):
            name = _getter(m)
            return os.environ.get(name, "")
        result = pattern.sub(_repl, result)
    # $HOME / $VAR как отдельный токен (bash): заменяем на значение
    result = re.sub(r"\$([A-Za-z_][A-Za-z0-9_]*)",
                    lambda m: os.environ.get(m.group(1), m.group(0)), result)
    return result


def _tokenize(command: str) -> list[str]:
    """Разбивает команду на токены, учитывая кавычки и backtick-escape (PowerShell).

    Возвращает токены с уже снятыми кавычками и применённым backtick-escape.
    Обратная косая черта считается литералом (важно для Windows-путей).
    """
    tokens: list[str] = []
    i = 0
    n = len(command)
    while i < n:
        c = command[i]
        if c in " \t\r\n":
            i += 1
            continue
        buf: list[str] = []
        quote = None
        while i < n:
            c = command[i]
            if quote == "'":
                if c == "'":
                    quote = None
                    i += 1
                    continue
                buf.append(c)
                i += 1
            elif quote == '"':
                if c == '"':
                    quote = None
                    i += 1
                    continue
                buf.append(c)
                i += 1
            elif c == "'":
                quote = "'"
                i += 1
            elif c == '"':
                quote = '"'
                i += 1
            elif c == "`":  # backtick escape (PowerShell)
                if i + 1 < n:
                    buf.append(command[i + 1])
                    i += 2
                else:
                    i += 1
            elif c in " \t\r\n":
                break
            else:
                buf.append(c)
                i += 1
        if buf:
            tokens.append("".join(buf))
    return tokens


def _is_option(token: str) -> bool:
    """True, если токен — опция/флаг (-r, --recursive, /x)."""
    if not token:
        return False
    if token.startswith("-"):
        return True
    # Windows-стиль /flag (но не /path — у / есть разделитель)
    if token.startswith("/") and "/" not in token[1:]:
        return True
    # PS: -Path=value
    if token.startswith("-") and "=" in token:
        return True
    return False


def _has_clear_path_marker(token: str) -> bool:
    """Признаки, однозначно говорящие о пути (без проверки существования)."""
    if any(sep in token for sep in _PATH_SEPS):
        return True
    if token.startswith(("./", "../", "~/", "~\\", ":\\", ":/", "/", "\\")):
        return True
    if re.match(r"^[A-Za-z]:", token):  # Windows drive: C:foo
        return True
    # файл по расширению
    if "." in os.path.basename(token):
        ext = "." + os.path.basename(token).rsplit(".", 1)[-1].lower()
        if ext in _FILE_EXTS:
            return True
    return False


def _looks_like_path(token: str, cwd: str) -> bool:
    """Решает, является ли токен путём, минимизируя ложные срабатывания."""
    if not token or _is_option(token):
        return False
    if _has_clear_path_marker(token):
        return True
    # «Голый» относительный токен (нет разделителей/расширения): только если существует
    resolved = os.path.normpath(os.path.join(cwd, token))
    return os.path.exists(resolved)


def _resolve(token: str, cwd: str) -> str | None:
    """Резолвит токен в абсолютный нормализованный путь, раскрывая переменные."""
    t = _expand_env(token).strip()
    if not t:
        return None
    # не даём пустым раскрытым переменным сломать путь
    if "=" in t and not os.path.isabs(t):
        return None
    expanded = os.path.expanduser(t)
    if os.path.isabs(expanded):
        return os.path.normpath(expanded)
    return os.path.normpath(os.path.join(cwd, expanded))


def extract_paths(command: str, cwd: str = None, shell: str = "auto") -> list[str]:
    """Извлекает из `command` список абсолютных путей (дедуплицированных).

    - Относительные пути резолвятся относительно `cwd` (по умолчанию — текущий каталог).
    - `shell` ("bash" | "powershell" | "auto") влияет на интерпретацию, но сейчас
      токенизация универсальна; параметр оставлен для расширяемости.
    """
    if not command:
        return []
    cwd = cwd or os.getcwd()
    tokens = _tokenize(command)
    found: list[str] = []
    seen: set[str] = set()

    for tok in tokens:
        if not _looks_like_path(tok, cwd):
            continue
        resolved = _resolve(tok, cwd)
        if not resolved:
            continue
        # пропускаем реконструкции типа пустого пути "="
        if resolved in (os.sep, ".") and resolved != os.sep:
            continue
        if resolved not in seen:
            seen.add(resolved)
            found.append(resolved)
    return found