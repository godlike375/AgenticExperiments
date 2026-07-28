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
    MAX_CONTEXT_TOKENS = 16000
    MAX_OUTPUT_TOKENS = min(32000, MAX_CONTEXT_TOKENS)
    TIMEOUT = 1800
    MAX_ITER = 20
    SUMMARIZATION_THRESHOLD_DIVIDER = 2

    STREAM_ENABLED = True

    USE_RESPONSES_API = True
