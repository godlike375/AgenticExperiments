class Config:
    API_URL = "http://localhost:1234/v1"
    MODEL_NAME = "local-model"
    AFTER_SYSTEM_PROMPT = 1  # Index after which dialog starts (0 = system)
    BOOST_TEMP = 0.8

    COMPRESS_THRESHOLD = 1000

    COMPRESSION_JUDGE_SYSTEM_PROMPT = (
        "You are a compression safety advisor. "
        "Decide whether a tool output can be safely summarized "
        "without losing information critical to the current task. "
        "Call the `decision` tool with your answer."
    )