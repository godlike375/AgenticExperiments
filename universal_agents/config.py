
"""Центральная конфигурация: единый dataclass, доступ через Config.X (без синглтона).
Производные константы вычисляются при определении класса."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import dotenv_values

from universal_agents.project_root import find_project_root

DEFAULT_API_URL = "http://localhost:1234/v1"
DEFAULT_API_TOKEN = ""
DEFAULT_MODEL_NAME = ""
ENV_KEYS = ("API_URL", "API_TOKEN", "MODEL_NAME")


def _default_env_file() -> str:
    root = find_project_root() or os.getcwd()
    return os.path.join(root, ".env")


def parse_env_file(path: str) -> dict[str, str]:
    if not os.path.isfile(path):
        return {}
    return {k: v for k, v in dotenv_values(path).items() if v is not None}


@dataclass
class Config:
    ENV_FILE: str = _default_env_file()
    USE_ENV: bool = False
    API_URL: str = DEFAULT_API_URL
    API_TOKEN: str = DEFAULT_API_TOKEN
    MODEL_NAME: str = DEFAULT_MODEL_NAME
    AFTER_SYSTEM_PROMPT: int = 1  # Index after which dialog starts (0 = system)
    BOOST_TEMP: float = 1.4
    ERROR_RECOVERY_TEMP: float = 1.2
    MAX_LOOP_RETRIES: int = 3  # попыток перегенерации при повторяющемся вызове/ответе
    DUPLICATE_SIMILARITY_THRESHOLD: float = 0.75
    DUPLICATE_TEXT_MIN_CHARS: int = 0
    DUPLICATE_REASONING_MIN_CHARS: int = 250  # короткие reasoning-блоки не считаются повтором
    DUPLICATE_RUN_LENGTH: int = 3             # одинаковый контент N раз подряд = зацикливание (X,X,X)
    DUPLICATE_REGENERATIONS_BEFORE_SYSTEM: int = 2  # регенераций без системного сообщения
    DUPLICATE_SYSTEM_WARNINGS: int = 2              # попыток после системного сообщения; далее управление пользователю
    ERROR_RECOVERY_RETRIES: int = 0  # попыток перегенерации после ошибки инструмента
    BROKEN_CALL_REGEN_RETRIES: int = 2  # попыток перегенерации при обнаружении сломанного вызова
    BROKEN_CALL_FIX_RETRIES: int = 2    # попыток «починить» вызов через промпт после неудачной регенерации
    NO_COMMENT_RETRIES: int = 2         # попыток перегенерации с prefill при вызове инструмента без пояснения; после исчерпания вызов исполняется как есть
    ANSWER_GUARD_MAX_RETRIES: int = 5   # срабатываний guard'а «ответь через respond_to_system»; после — сдача хода пользователю
    DUPLICATE_CONTINUATION_TEMP: float = round(BOOST_TEMP / 4, 2)  # спокойная достройка после расхождения
    SUMMARY_DUPLICATE_TEMP: float = round(BOOST_TEMP / 2, 2)  # буст при тождественном повторе саммари (мягче полного BOOST_TEMP)

    MAX_CONTEXT_TOKENS: int = 120000
    TEMP: float = 0.6
    TOP_P: float = 0.93
    FREQUENCY_PENALTY: float = 0.0
    PRESENCE_PENALTY: float = 0.0
    MIN_P: Optional[float] = 0.065
    MAX_OUTPUT_TOKENS: int = min(32000, int(MAX_CONTEXT_TOKENS / 1.5))
    TIMEOUT: int = 1800
    MAX_ITER: int = 250
    SUMMARIZATION_THRESHOLD_DIVIDER: int = 2

    STREAM_ENABLED: bool = True

    # Проверено на LM Studio + qwen3: смена reasoning_effort роняет кэш целиком
    # (сервер вставляет ~40 токенов в начало промпта, префикс расходится от позиции 0).
    KEEP_REASONING_CONTENT_IN_HISTORY: bool = True

    SIMULATED_REASONING_ENABLED: bool = False
    SIMULATED_REASONING_TAGS: tuple[str, ...] = ("thinking",)
    SIMULATED_REASONING_REPAIRS: int = 3
    SIMULATED_REASONING_TEMP: Optional[float] = 0.65
    SIMULATED_REASONING_TOP_P: Optional[float] = 0.93
    SIMULATED_REASONING_MIN_P: Optional[float] = 0.03
    SIMULATED_REASONING_FREQUENCY_PENALTY: Optional[float] = 0.65
    SIMULATED_REASONING_PRESENCE_PENALTY: Optional[float] = 0.15

    DEBUG_PREFIX_HASH_CHECK: bool = True

    DEBUG_LLM_CALLS: bool = False

    AUTO_SUMMARY_THRESHOLD: int = 80  # процент занятого контекста для начала авто-суммаризации
    AUTO_SUMMARY_PRESERVE_LAST: int = 1  # сколько последних сообщений не трогать
    AUTO_SUMMARY_REVIEW_PASS: bool = True  # отревьювить черновик саммари: подчистить устаревшее + добавить пропущенное
    AUTO_SUMMARY_MAX_RETRIES: int = 5

    TASK_COMPACTION_ENABLED: bool = True  # сжимать завершённые группы подзадач
    MAX_TASK_COMPACTION_ROUNDS: int = 20  # макс. число групп за один проход

    HAVE_DONE_TRIM_THRESHOLD: int = 250

    AUTO_TRUST_GIT_ROOT: bool = True

    CHARS_PER_TOKEN: float = 2.35
    MIN_TOKENS_TO_SUMMARIZE: int = 180

    BIG_FILE_SKELETON: bool = False

    SKELETON_RANGES_MODE: bool = False

    PERIPHERAL_GAP_GROWTH: float = 1.9
    PERIPHERAL_MAX_LINE_CHARS: int = 45
    PERIPHERAL_SIDE_FACTOR: float = 1.1
    PERIPHERAL_LINE_CONTEXT: int = 1

    MAX_READ_CHARS_PER_CALL: int = 4500
    MAX_READ_LINES_PER_CALL: int = 80

    DISABLE_TOOL_AUTO_SUMMARIZATION: bool = True

    TOOL_ISOLATION_ENABLED: bool = True
    TOOL_WORKER_POLL_INTERVAL: float = 0.05
    TOOL_WORKER_TIMEOUT: int = 3600
    TOOL_WORKER_KILL_GRACE: float = 1.0

    BASH_BACKEND: str = "auto"

    GIT_BASH_PATH: str = ""

    PROJECT_ROOT: str = r""

    AUTOSAVE_ENABLED: bool = True
    AUTOSAVE_DIR: str = "autosave"
    AUTOSAVE_KEEP: int = 25

    SAVE_IMAGES: bool = False

    SERVICE_TURN_KEEP_IN_HISTORY: bool = False
    SERVICE_TURN_MAX_ITER: int = 2

    PER_MSG_SUMMARIES_ENABLED: bool = False

    MEMORY_ARCHIVE_ENABLED: bool = True
    RECALL_SNIPPET_CHARS: int = 300
    RECALL_READ_MAX_CHARS: int = 4000
    RECALL_ENTRY_MAX_CHARS: int = 1500
    RECALL_MAX_ARG_CHARS: int = 500

    SCREENSHOT_MAX_WIDTH: int = 960
    SCREENSHOT_QUALITY: int = 60
    SCREENSHOT_DEBUG_PATH: str = ""
    MOUSE_MOVE_DURATION: float = 0.25
    PYAUTOGUI_PAUSE: float = 0.1


def _apply_env_file(cfg: type[Config]) -> None:
    if not cfg.USE_ENV:
        return
    env = parse_env_file(cfg.ENV_FILE)
    defaults = {"API_URL": DEFAULT_API_URL, "API_TOKEN": DEFAULT_API_TOKEN, "MODEL_NAME": DEFAULT_MODEL_NAME}
    for key in ENV_KEYS:
        value = env.get(key)
        if value and getattr(cfg, key) == defaults[key]:
            setattr(cfg, key, value)


_apply_env_file(Config)
