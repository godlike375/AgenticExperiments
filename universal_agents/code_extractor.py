import os
import sys
import fnmatch

def should_exclude(path, patterns):
    """Проверяет, соответствует ли путь (или имя файла/папки) хотя бы одной маске из списка."""
    # Разбиваем путь на части, чтобы проверять каждое имя компонента пути
    parts = path.replace(os.sep, '/').split('/')

    for part in parts:
        for pattern in patterns:
            if fnmatch.fnmatch(part, pattern):
                return True
    return False

def dump_project(root_path, exclude_patterns=None):
    if exclude_patterns is None:
        # Стандартный набор того, что обычно не нужно отправлять LLM
        exclude_patterns = [
            '__pycache__',
            '*.pyc',
            '.git',
            'venv',
            '.venv',
            'env',
            '.env',
            'node_modules',
            '.idea',
            '.vscode',
            'dist',
            'build',
            'code_extractor.py'
        ]

    if not os.path.isdir(root_path):
        print(f"Ошибка: Путь '{root_path}' не является директорией.", file=sys.stderr)
        sys.exit(1)

    # Проходим рекурсивно по всем файлам
    for dirpath, dirnames, filenames in os.walk(root_path):
        # Фильтруем директории "на лету", чтобы не спускаться в ненужные папки
        # Это критично для производительности (не читать venv целиком)
        dirnames[:] = [d for d in dirnames if not should_exclude(d, exclude_patterns)]

        # Сортируем файлы для предсказуемого порядка вывода
        sorted_filenames = sorted(filenames)

        for filename in sorted_filenames:
            # Пропускаем файлы, если они сами по себе подпадают под маску (например .pyc)
            if should_exclude(filename, exclude_patterns):
                continue

            if filename.endswith(".py"):
                full_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(full_path, root_path)

                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Вывод заголовка с путем и markdown блоки для удобства копирования
                    print(f"\n# --- Файл: {rel_path} ---")
                    print("```python")
                    print(content)
                    print("```")

                except UnicodeDecodeError:
                    print(f"# --- Ошибка кодировки в файле: {rel_path} ---", file=sys.stderr)
                except Exception as e:
                    print(f"# --- Ошибка чтения {rel_path}: {e} ---", file=sys.stderr)

if __name__ == "__main__":
    # Если аргумент передан, используем его, иначе берем текущую директорию '.'
    target = sys.argv[1] if len(sys.argv) > 1 else '.'
    dump_project(target)