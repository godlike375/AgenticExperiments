class Config:
    API_URL = "http://localhost:1234/v1"
    MODEL_NAME = ""
    AFTER_SYSTEM_PROMPT = 1  # Index after which dialog starts (0 = system)
    BOOST_TEMP = 2

    # Параметры генерации
    TEMP = 0.575
    TOP_P = 0.962
    FREQUENCY_PENALTY = 0
    PRESENCE_PENALTY = 0
    MAX_TOKENS = 12000
    TIMEOUT = 1800
    MAX_ITER = 20
    MAX_CONTEXT_TOKENS = 66000
    
    # Streaming
    STREAM_ENABLED = True  # Включить streaming по умолчанию
    STREAM_CHUNK_DELAY = 0.0  # Задержка между чанками (для демонстрации)
