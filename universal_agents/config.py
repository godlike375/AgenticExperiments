class Config:
    #API_URL = "http://192.168.50.196:1234/v1"
    API_URL = "http://localhost:1234/v1"
    MODEL_NAME = ""
    AFTER_SYSTEM_PROMPT = 1  # Index after which dialog starts (0 = system)
    BOOST_TEMP = 1.7
    ERROR_RECOVERY_TEMP = 1.2
    MAX_LOOP_RETRIES = 2  # попыток перегенерации при повторяющемся вызове/ответе
    # Порог Jaccard-схожести по множеству слов для признания текстового ответа повтором.
    DUPLICATE_SIMILARITY_THRESHOLD = 0.7
    ERROR_RECOVERY_RETRIES = 1  # попыток перегенерации после ошибки инструмента
    BROKEN_CALL_REGEN_RETRIES = 2  # попыток перегенерации при обнаружении сломанного вызова
    BROKEN_CALL_FIX_RETRIES = 2    # попыток «починить» вызов через промпт после неудачной регенерации
    DUPLICATE_CONTINUATION_TEMP = round(BOOST_TEMP / 4, 2)  # спокойная достройка после расхождения
    SUMMARY_DUPLICATE_TEMP = round(BOOST_TEMP / 2, 2)  # буст при тождественном повторе саммари (мягче полного BOOST_TEMP)

    # Параметры генерации
    TEMP = 0.4
    TOP_P = 0.94
    MAX_CONTEXT_TOKENS = 50000
    FREQUENCY_PENALTY = 0.05
    PRESENCE_PENALTY = 0.05
    MAX_OUTPUT_TOKENS = min(32000, int(MAX_CONTEXT_TOKENS / 1.5))
    TIMEOUT = 1800
    MAX_ITER = 150
    SUMMARIZATION_THRESHOLD_DIVIDER = 2

    STREAM_ENABLED = True

    KEEP_REASONING_CONTENT_IN_HISTORY = False

    USE_RESPONSES_API = False

    # Автоматическая суммаризация диалога
    AUTO_SUMMARY_THRESHOLD = 2  # процент контекста
    AUTO_SUMMARY_PRESERVE_LAST = 1  # сколько последних сообщений не трогать
    AUTO_SUMMARY_REVIEW_PASS = True  # отревьювить черновик саммари: подчистить устаревшее + добавить пропущенное
    # Попыток перегенерации саммари при неудаче; между ними температура чуть растёт, чтобы не повторять ту же ошибку.
    AUTO_SUMMARY_MAX_RETRIES = 5

    # Слабое сжатие (меньше этой доли) → поверх него ещё усекаются выводы инструментов.
    AUTO_SUMMARY_MIN_REDUCTION_RATIO = 0.25
    # Усечение выводов при слабом сжатии: оставляем эту долю оригинала, но не меньше TRUNCATE_TOOL_RESULT_CHARS (чтобы не резать слишком коротко).
    TRUNCATE_TOOL_RESULT_KEEP_RATIO = 0.2
    TRUNCATE_TOOL_RESULT_CHARS = 60

    # Структурная компактизация истории по завершённым подзадачам
    TASK_COMPACTION_ENABLED = True  # сжимать завершённые группы подзадач
    MAX_TASK_COMPACTION_ROUNDS = 20  # макс. число групп за один проход

    # Порог have_done (символов): при превышении при компактизации обрезаем до стаба (детали уже в summary).
    HAVE_DONE_TRIM_THRESHOLD = 250

    # Авто-доверие корня проекта: наличие .git → edit_file без подтверждения (git поможет откатить).
    AUTO_TRUST_GIT_ROOT = True

    # Константы токенизации и суммаризации
    CHARS_PER_TOKEN = 2.35
    MIN_TOKENS_TO_SUMMARIZE = 180

    # Чтение больших файлов: ровно один раз — структурный скелет, код модель
    # читает порциями через start_line/end_line; повторные целиком-файловые
    # чтения неизменённого файла запрещены (экономия контекста, см. REFACTORING_BASELINE §3.7).
    BIG_FILE_SKELETON = True

    # Единый лимит вывода ЛЮБОГО инструмента (символов за вызов). read/search уже
    # сами режут вывод до этого лимита с подсказками; остальные усекаются «хвостом»
    # в _execute_tools. MAX_READ_LINES_PER_CALL — доп. лимит строк порционного чтения (порция кончается концом строки).
    MAX_READ_CHARS_PER_CALL = 6000
    MAX_READ_LINES_PER_CALL = 120

    # Отключает авто-суммаризацию большого вывода любых инструментов.
    DISABLE_TOOL_AUTO_SUMMARIZATION = True

    # per-message summarization (гейт — PER_MSG_SUMMARIES_ENABLED): True — плотное саммари каждого сообщения копится в память (НЕ контекст),
    # длинные выводы (>MIN_TOKENS_TO_SUMMARIZE) тоже; при сжатии собирается короткий диалог. False — session summary одним single-shot вызовом.
    # Локальный оверрайд агента — disable_per_msg_summarization.

    # Bash на Windows (run_bash_host): "wsl" — через WSL, "gitbash" — Git Bash, "auto" — gitbash при наличии иначе WSL, "system" — shutil.which("bash").
    BASH_BACKEND = "auto"

    # Путь к Git Bash для BASH_BACKEND="gitbash"; пусто — поиск по стандартным местам.
    GIT_BASH_PATH = ""

    # Явный корень проекта перекрывает авто-поиск по .git; можно задать через --project-root. None = авто-поиск.
    PROJECT_ROOT = ""

    # ------------------------------------------------------------------
    # Авто-сохранение истории после каждого нового сообщения (защита от сбоев)
    # ------------------------------------------------------------------
    AUTOSAVE_ENABLED = True
    AUTOSAVE_DIR = "autosave"
    AUTOSAVE_KEEP = 50

    # ------------------------------------------------------------------
    # Память: session summary + архив (см. compressors.py, archive.py, MemoryMixin._auto_summarize_dialogue).
    # ------------------------------------------------------------------
    # Компакция: вытесняемый сегмент сворачивается ОДНИМ вызовом в session summary (UserMessage сразу после system prompt),
    PER_MSG_SUMMARIES_ENABLED = False
    # Таймаут компакции (сек): на большом сегменте длинный ответ не успевает за 120с.
    STATE_GEN_TIMEOUT = 240
    # Архив вытесненных оригиналов (recall); выключение ломает только ответы про давно удалённое из контекста.
    MEMORY_ARCHIVE_ENABLED = True
    # Лимиты recall-инструментов (символы).
    RECALL_SNIPPET_CHARS = 300
    RECALL_READ_MAX_CHARS = 4000
    RECALL_ENTRY_MAX_CHARS = 1500
    RECALL_MAX_ARG_CHARS = 500


# Модульные алиасы часто используемых констант (атрибуты Config как имена модуля)
CHARS_PER_TOKEN = Config.CHARS_PER_TOKEN
MIN_TOKENS_TO_SUMMARIZE = Config.MIN_TOKENS_TO_SUMMARIZE
