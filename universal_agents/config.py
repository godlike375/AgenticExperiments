class Config:
    API_URL = "http://192.168.50.196:1234/v1"
    MODEL_NAME = ""
    AFTER_SYSTEM_PROMPT = 1  # Index after which dialog starts (0 = system)
    BOOST_TEMP = 2
    ERROR_RECOVERY_TEMP = 1
    MAX_LOOP_RETRIES = 2  # попыток перегенерации при повторяющемся вызове/ответе
    ERROR_RECOVERY_RETRIES = 2  # попыток перегенерации после ошибки инструмента
    BROKEN_CALL_REGEN_RETRIES = 2  # попыток перегенерации при обнаружении сломанного вызова
    BROKEN_CALL_FIX_RETRIES = 2    # попыток «починить» вызов через промпт после неудачной регенерации
    DUPLICATE_CONTINUATION_TEMP = round(BOOST_TEMP / 4, 2)  # спокойная достройка после расхождения ⏤ BOOST/3 ≈ 0.67

    # Параметры генерации
    TEMP = 0.38
    TOP_P = 0.94
    FREQUENCY_PENALTY = 0.05
    PRESENCE_PENALTY = 0.15
    MAX_CONTEXT_TOKENS = 3500
    MAX_OUTPUT_TOKENS = min(32000, int(MAX_CONTEXT_TOKENS / 1.5))
    TIMEOUT = 1800
    MAX_ITER = 35
    SUMMARIZATION_THRESHOLD_DIVIDER = 2

    STREAM_ENABLED = True

    USE_RESPONSES_API = False

    # Автоматическая суммаризация диалога
    AUTO_SUMMARY_THRESHOLD = 60  # процент контекста (55%)
    AUTO_SUMMARY_PRESERVE_LAST = 1  # сколько последних сообщений не трогать
    AUTO_SUMMARY_REVIEW_PASS = True  # отревьювить черновик саммари: подчистить устаревшее + добавить пропущенное

    # Константы токенизации и суммаризации
    CHARS_PER_TOKEN = 2.3
    MIN_TOKENS_TO_SUMMARIZE = 500

    # Отключает скелетизацию больших файлов: read без start_line/end_line вернёт полный файл
    DISABLE_FILE_SKELETONIZATION = True

    # Отключает автоматическую суммаризацию большого вывода любых инструментов
    DISABLE_TOOL_AUTO_SUMMARIZATION = True

    DISABLE_PER_MESSAGE_SUMMARIZATION = True


# Модульные алиасы часто используемых констант (атрибуты Config как имена модуля)
CHARS_PER_TOKEN = Config.CHARS_PER_TOKEN
MIN_TOKENS_TO_SUMMARIZE = Config.MIN_TOKENS_TO_SUMMARIZE
