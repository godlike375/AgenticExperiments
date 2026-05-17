class Config:
    API_URL = 'http://192.168.50.221:1234/v1' # "http://localhost:1234/v1"
    MODEL_NAME = "local-model"
    AFTER_SYSTEM_PROMPT = 1  # Index after which dialog starts (0 = system)
    BOOST_TEMP = 0.8
