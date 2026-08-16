class Config:
    API_URL = "http://192.168.50.196:1234/v1"
    MODEL_NAME = ""
    AFTER_SYSTEM_PROMPT = 1  # Index after which dialog starts (0 = system)
    BOOST_TEMP = 1.5
    ERROR_RECOVERY_TEMP = 1
    MAX_LOOP_RETRIES = 2  # попыток перегенерации при повторяющемся вызове/ответе
    ERROR_RECOVERY_RETRIES = 1  # попыток перегенерации после ошибки инструмента
    BROKEN_CALL_REGEN_RETRIES = 2  # попыток перегенерации при обнаружении сломанного вызова
    BROKEN_CALL_FIX_RETRIES = 2    # попыток «починить» вызов через промпт после неудачной регенерации
    DUPLICATE_CONTINUATION_TEMP = round(BOOST_TEMP / 4, 2)  # спокойная достройка после расхождения ⏤ BOOST/3 ≈ 0.67

    # Параметры генерации
    TEMP = 0.45
    TOP_P = 0.94
    FREQUENCY_PENALTY = 0.0
    PRESENCE_PENALTY = 0.0
    MAX_CONTEXT_TOKENS = 2700
    MAX_OUTPUT_TOKENS = min(32000, int(MAX_CONTEXT_TOKENS / 1.5))
    TIMEOUT = 1800
    MAX_ITER = 35
    SUMMARIZATION_THRESHOLD_DIVIDER = 2

    STREAM_ENABLED = True

    # Держать reasoning_content ассистента в истории/контексте (для экспериментов).
    # Если False (по умолчанию) — reasoning_content не попадает в API-контекст и в
    # сохранённую историю (экономия токенов/KV-кэша; reasoning не нужен downstream).
    KEEP_REASONING_CONTENT_IN_HISTORY = False

    USE_RESPONSES_API = False

    # Автоматическая суммаризация диалога
    AUTO_SUMMARY_THRESHOLD = 85  # процент контекста (55%)
    AUTO_SUMMARY_PRESERVE_LAST = 1  # сколько последних сообщений не трогать
    AUTO_SUMMARY_REVIEW_PASS = True  # отревьювить черновик саммари: подчистить устаревшее + добавить пропущенное

    # Структурная компактизация истории по завершённым подзадачам
    TASK_COMPACTION_ENABLED = True  # сжимать завершённые группы подзадач
    MAX_TASK_COMPACTION_ROUNDS = 20  # макс. число групп за один проход

    # Порог длины содержимого результата have_done (в символах): если больше —
    # при компактизации задачи обрезаем его до короткого стаба, т.к. детальные
    # факты уже сохранены в компактизационном summary.
    HAVE_DONE_TRIM_THRESHOLD = 250

    # Авто-доверие корня проекта: если в папке есть валидный .git, она доверена
    # для edit_file по умолчанию (без запроса подтверждения) — git поможет откатить.
    AUTO_TRUST_GIT_ROOT = True

    # Константы токенизации и суммаризации
    CHARS_PER_TOKEN = 2.3
    MIN_TOKENS_TO_SUMMARIZE = 150

    # Отключает скелетизацию больших файлов: read без start_line/end_line вернёт полный файл
    DISABLE_FILE_SKELETONIZATION = False

    # Отключает автоматическую суммаризацию большого вывода любых инструментов
    DISABLE_TOOL_AUTO_SUMMARIZATION = True

    # Режим per-message summarization (при False):
    #   • после каждого сообщения ассистента его плотное саммари складывается в
    #     рабочую память агента (НЕ в контекст);
    #   • длинные выводы инструментов (> MIN_TOKENS_TO_SUMMARIZE) тоже сразу
    #     суммаризируются в рабочую память;
    #   • когда контекст превышает AUTO_SUMMARY_THRESHOLD, из этих маленьких
    #     саммари собирается новый, более короткий диалог.
    DISABLE_PER_MESSAGE_SUMMARIZATION = False

    # Как исполнять bash на Windows (для run_bash_host):
    #   "wsl"     — через WSL (bash.exe / wsl.exe)
    #   "gitbash" — через Git Bash (C:\Program Files\Git\bin\bash.exe и т.п.)
    #   "auto"    — попытаться определить: git bash при наличии, иначе WSL (по умолчанию)
    #   "system"  — shutil.which("bash") без дополнительной логики
    BASH_BACKEND = "auto"

    # Явный путь к Git Bash для BASH_BACKEND="gitbash".
    # Если пусто — поиск по стандартным расположениям установки Git.
    GIT_BASH_PATH = ""


# Модульные алиасы часто используемых констант (атрибуты Config как имена модуля)
CHARS_PER_TOKEN = Config.CHARS_PER_TOKEN
MIN_TOKENS_TO_SUMMARIZE = Config.MIN_TOKENS_TO_SUMMARIZE
