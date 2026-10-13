
"""Извлечение файловых путей из командных строк (bash/PowerShell) для безопасности."""
import os
import re

_PATH_SEPS = ("\\", "/")

_FILE_EXTS = {
    ".py", ".pyc", ".json", ".txt", ".md", ".log", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".env", ".csv", ".tsv", ".xml", ".html",
    ".js", ".ts", ".jsx", ".tsx", ".css", ".sh", ".bat", ".ps1", ".cmd",
    ".exe", ".dll", ".so", ".dylib", ".zip", ".tar", ".gz", ".7z", ".rar",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".pdf", ".sql", ".db", ".sqlite",
    ".lock", ".toml", ".yml", ".whl",
}

_ENV_PATTERNS = (
    (re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}"), lambda m: m.group(1)),  # ${VAR}
    (re.compile(r"\$env:([A-Za-z_][A-Za-z0-9_]*)"), lambda m: m.group(1)),  # $env:VAR (PS)
    (re.compile(r"%([A-Za-z_][A-Za-z0-9_]*)%"), lambda m: m.group(1)),      # %VAR% (cmd)
)


def _expand_env(token: str) -> str:
    result = token
    for pattern, getter in _ENV_PATTERNS:
        def _repl(m, _getter=getter):
            name = _getter(m)
            return os.environ.get(name, "")
        result = pattern.sub(_repl, result)
    result = re.sub(r"\$([A-Za-z_][A-Za-z0-9_]*)",
                    lambda m: os.environ.get(m.group(1), m.group(0)), result)
    return result


def _tokenize(command: str) -> list[str]:
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
    if not token:
        return False
    if token.startswith("-"):
        return True
    if token.startswith("/") and "/" not in token[1:]:
        return True
    if token.startswith("-") and "=" in token:
        return True
    return False


def _has_clear_path_marker(token: str) -> bool:
    if not token:
        return False
    if len(token) <= 1:
        return False
    if all(c in _PATH_SEPS for c in token):
        return False
    if any(c in token for c in "=()|;&<>!"):
        return False
    if any(sep in token for sep in _PATH_SEPS):
        return True
    if token.startswith(("./", "../", "~/", "~\\", ":\\", ":/", "/", "\\")):
        return True
    if re.match(r"^[A-Za-z]:[\\/]", token):
        return True
    if "." in os.path.basename(token):
        ext = "." + os.path.basename(token).rsplit(".", 1)[-1].lower()
        if ext in _FILE_EXTS:
            return True
    return False


def _looks_like_path(token: str, cwd: str) -> bool:
    if not token or _is_option(token):
        return False
    if any(c in token for c in "=()|;&<>!"):
        return False
    if _has_clear_path_marker(token):
        return True
    resolved = os.path.normpath(os.path.join(cwd, token))
    if not os.path.exists(resolved):
        return False
    if os.path.normcase(resolved) == os.path.normcase(cwd):
        return False
    return True


def _resolve(token: str, cwd: str) -> str | None:
    t = _expand_env(token).strip()
    if not t:
        return None
    if "=" in t and not os.path.isabs(t):
        return None
    expanded = os.path.expanduser(t)
    if os.path.isabs(expanded):
        normed = os.path.normpath(expanded)
        if normed in (os.sep, os.altsep, os.path.splitdrive(normed)[0]):
            return None
        if normed.replace("/", "\\").strip("\\") == "":
            return None
        return normed
    return os.path.normpath(os.path.join(cwd, expanded))


def extract_paths(command: str, cwd: str = None) -> list[str]:
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
        if resolved in (os.sep, ".") and resolved != os.sep:
            continue
        if resolved not in seen:
            seen.add(resolved)
            found.append(resolved)
    return found