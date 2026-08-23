class Config:
    #API_URL = "http://192.168.50.196:1234/v1"
    API_URL = "http://localhost:1234/v1"
    MODEL_NAME = ""
    AFTER_SYSTEM_PROMPT = 1  # Index after which dialog starts (0 = system)
    BOOST_TEMP = 1.75
    ERROR_RECOVERY_TEMP = 1.25
    MAX_LOOP_RETRIES = 2  # попыток перегенерации при повторяющемся вызове/ответе
    # Порог Jaccard-схожести по множеству слов для признания текстового ответа повтором.
    DUPLICATE_SIMILARITY_THRESHOLD = 0.7
    ERROR_RECOVERY_RETRIES = 1  # попыток перегенерации после ошибки инструмента
    BROKEN_CALL_REGEN_RETRIES = 2  # попыток перегенерации при обнаружении сломанного вызова
    BROKEN_CALL_FIX_RETRIES = 2    # попыток «починить» вызов через промпт после неудачной регенерации
    DUPLICATE_CONTINUATION_TEMP = round(BOOST_TEMP / 4, 2)  # спокойная достройка после расхождения

    # Параметры генерации
    TEMP = 0.36
    TOP_P = 0.94
    FREQUENCY_PENALTY = 0.025
    PRESENCE_PENALTY = 0.025
    MAX_CONTEXT_TOKENS = 50000
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
    AUTO_SUMMARY_THRESHOLD = 60  # процент контекста
    AUTO_SUMMARY_PRESERVE_LAST = 1  # сколько последних сообщений не трогать
    AUTO_SUMMARY_REVIEW_PASS = True  # отревьювить черновик саммари: подчистить устаревшее + добавить пропущенное

    # Если авто-суммаризация сжала контекст менее чем на эту долю
    # (например 0.20 = 20%), сжатие считается слабым и поверх него дополнительно
    # усекаются результаты выполнения инструментов, чтобы ещё сэкономить контекст.
    AUTO_SUMMARY_MIN_REDUCTION_RATIO = 0.25
    # Относительная мера усечения выводов инструментов при слабом сжатии:
    # оставляем эту долю оригинала (0.5 = половину). Если от этой доли остаётся
    # меньше TRUNCATE_TOOL_RESULT_CHARS символов — используем абсолютный пол
    # TRUNCATE_TOOL_RESULT_CHARS (чтобы не резать слишком коротко).
    TRUNCATE_TOOL_RESULT_KEEP_RATIO = 0.2
    TRUNCATE_TOOL_RESULT_CHARS = 60

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
    CHARS_PER_TOKEN = 2.35
    MIN_TOKENS_TO_SUMMARIZE = 180

    # Чтение больших файлов: вместо выгрузки всего файла в контекст — ровно один
    # раз отдаём структурный скелет, а реальный код модель читает порциями через
    # start_line/end_line. Повторные целиком-файловые чтения неизменённого файла
    # запрещены (экономия контекста, см. REFACTORING_BASELINE §3.7).
    BIG_FILE_SKELETON = True

    # Лимит одного чтения диапазона из файла: не более N символов за вызов;
    # строка, на которой лимит превышен, включается целиком (порция всегда
    # заканчивается концом строки). Дополнительно — не более
    # MAX_READ_LINES_PER_CALL строк за вызов. Заставляет модель читать порциями,
    # как человек, а не выгружать файл целиком.
    MAX_READ_CHARS_PER_CALL = 2000
    MAX_READ_LINES_PER_CALL = 150

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
