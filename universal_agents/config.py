class Config:
    API_URL = "http://localhost:1234/v1"
    MODEL_NAME = ""
    AFTER_SYSTEM_PROMPT = 1  # Index after which dialog starts (0 = system)
    BOOST_TEMP = 2

    # Параметры генерации
    TEMP = 0.48
    TOP_P = 0.962
    FREQUENCY_PENALTY = 0
    PRESENCE_PENALTY = 0
    MAX_CONTEXT_TOKENS = 64000
    MAX_OUTPUT_TOKENS = min(32000, int(MAX_CONTEXT_TOKENS / 1.5))
    TIMEOUT = 1800
    MAX_ITER = 20
    SUMMARIZATION_THRESHOLD_DIVIDER = 2

    STREAM_ENABLED = True

    USE_RESPONSES_API = True

    # Автоматическая суммаризация диалога
    AUTO_SUMMARY_THRESHOLD = 55  # процент контекста (55%)
    AUTO_SUMMARY_PRESERVE_LAST = 1  # сколько последних сообщений не трогать

    # Константы токенизации и суммаризации
    CHARS_PER_TOKEN = 2.3
    MIN_TOKENS_TO_SUMMARIZE = 500
