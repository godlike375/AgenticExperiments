"""Центральная конфигурация агента: единый dataclass с дефолтами.

Доступ — через классовые атрибуты (Config.X), как раньше со статическим классом:
модуль не создаёт синглтон-экземпляр, мутации в тестах работают на уровне класса.
Производные константы (DUPLICATE_CONTINUATION_TEMP, MAX_OUTPUT_TOKENS) вычисляются
при определении класса, поэтому всегда консистентны со своими источниками."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    API_URL: str = "http://192.168.50.196:1234/v1"
    MODEL_NAME: str = ""
    AFTER_SYSTEM_PROMPT: int = 1  # Index after which dialog starts (0 = system)
    BOOST_TEMP: float = 1.4
    ERROR_RECOVERY_TEMP: float = 1.2
    MAX_LOOP_RETRIES: int = 3  # попыток перегенерации при повторяющемся вызове/ответе
    # Порог биграммного Jaccard (с учётом порядка слов) для признания текстового ответа повтором.
    DUPLICATE_SIMILARITY_THRESHOLD: float = 0.75
    # Детекция повторов текста/reasoning-блока по ВСЕЙ истории (не только у предыдущего
    # ответа): точное совпадение хэша содержимого AssistantMessage с любым более ранним
    # сообщением (хоть 100 итераций назад) — зацикливание. Минимальная длина текста
    # (символов), считающегося повтором; 0 = без ограничения.
    DUPLICATE_TEXT_MIN_CHARS: int = 0
    # Сколько дублей подряд игнорировать ДО вставки NAG: первые попытки просто
    # отбрасывают ответ и бустят температуру; только после этого порога в контекст
    # добавляется предупреждающая инструкция (NAG) для перегенерации.
    DUPLICATE_NAG_THRESHOLD: int = 3
    ERROR_RECOVERY_RETRIES: int = 0  # попыток перегенерации после ошибки инструмента
    BROKEN_CALL_REGEN_RETRIES: int = 2  # попыток перегенерации при обнаружении сломанного вызова
    BROKEN_CALL_FIX_RETRIES: int = 2    # попыток «починить» вызов через промпт после неудачной регенерации
    NO_COMMENT_RETRIES: int = 2         # попыток перегенерации с prefill при вызове инструмента без пояснения; после исчерпания вызов исполняется как есть
    ANSWER_GUARD_MAX_RETRIES: int = 5   # срабатываний guard'а «ответь через respond_to_system»; после — сдача хода пользователю
    DUPLICATE_CONTINUATION_TEMP: float = round(BOOST_TEMP / 4, 2)  # спокойная достройка после расхождения
    SUMMARY_DUPLICATE_TEMP: float = round(BOOST_TEMP / 2, 2)  # буст при тождественном повторе саммари (мягче полного BOOST_TEMP)

    # Параметры генерации
    MAX_CONTEXT_TOKENS: int = 110000
    TEMP: float = 0.5
    TOP_P: float = 0.925
    FREQUENCY_PENALTY: float = 0.0
    PRESENCE_PENALTY: float = 0.0
    # min_p (минимальная вероятность токена относительно топ-1): None — не отправлять
    # вообще (сервер решит сам). Поддерживается llama.cpp и LM Studio.
    MIN_P: Optional[float] = 0.07
    MAX_OUTPUT_TOKENS: int = min(32000, int(MAX_CONTEXT_TOKENS / 1.5))
    TIMEOUT: int = 1800
    MAX_ITER: int = 250
    SUMMARIZATION_THRESHOLD_DIVIDER: int = 2

    STREAM_ENABLED: bool = True

    # Отправлять ли reasoning_content ассистента обратно в историю следующих запросов.
    # Проверено на LM Studio + qwen3: Единственное, что роняет кэш целиком — смена reasoning_effort:
    # сервер вставляет в начало промпта ~40 подпорных токенов, префикс расходится от позиции 0
    # (одноразовый пересчёт при каждом /think-тоггле, это ожидаемо).
    KEEP_REASONING_CONTENT_IN_HISTORY: bool = True

    SIMULATED_REASONING_ENABLED: bool = True
    # Тегированные секции ответа по порядку: ответ начинается с первой, а после её
    # закрытия идёт свободный текст. Меняешь здесь — меняется формат.
    SIMULATED_REASONING_TAGS: tuple[str, ...] = ("brief_thinking",)
    SIMULATED_REASONING_REPAIRS: int = 3
    SIMULATED_REASONING_TEMP: Optional[float] = 0.55
    SIMULATED_REASONING_TOP_P: Optional[float] = 0.92
    SIMULATED_REASONING_MIN_P: Optional[float] = 0.03
    SIMULATED_REASONING_FREQUENCY_PENALTY: Optional[float] = 0.65
    SIMULATED_REASONING_PRESENCE_PENALTY: Optional[float] = 0.15

    # Отладка KV-кэша: хэшировать каждое сообщение префикса и сравнивать с
    # предыдущей итерацией подготовки.
    DEBUG_PREFIX_HASH_CHECK: bool = True

    # Печатать в консоль вход/выход не-стримингового вызова LLM (отладка транспорта).
    DEBUG_LLM_CALLS: bool = False

    # Автоматическая суммаризация диалога
    AUTO_SUMMARY_THRESHOLD: int = 80  # процент занятого контекста для начала авто-суммаризации
    AUTO_SUMMARY_PRESERVE_LAST: int = 1  # сколько последних сообщений не трогать
    AUTO_SUMMARY_REVIEW_PASS: bool = True  # отревьювить черновик саммари: подчистить устаревшее + добавить пропущенное
    # Попыток перегенерации саммари при неудаче; между ними температура чуть растёт, чтобы не повторять ту же ошибку.
    AUTO_SUMMARY_MAX_RETRIES: int = 5

    # Структурная компактизация истории по завершённым подзадачам
    TASK_COMPACTION_ENABLED: bool = True  # сжимать завершённые группы подзадач
    MAX_TASK_COMPACTION_ROUNDS: int = 20  # макс. число групп за один проход

    # Порог have_done (символов): при превышении при компактизации обрезаем до стаба (детали уже в summary).
    HAVE_DONE_TRIM_THRESHOLD: int = 250

    # Авто-доверие корня проекта: наличие .git → редакторы файлов без подтверждения (git поможет откатить).
    AUTO_TRUST_GIT_ROOT: bool = True

    # Константы токенизации и суммаризации
    CHARS_PER_TOKEN: float = 2.35
    MIN_TOKENS_TO_SUMMARIZE: int = 180

    # Большие файлы: 1 раз — скелет, далее чтение порциями start_line/end_line.
    BIG_FILE_SKELETON: bool = False

    # Скелет файла: True — модель возвращает только диапазоны (заголовки
    # подставляем программно); False — старый режим (таблица целиком, может мусорить).
    SKELETON_RANGES_MODE: bool = False

    # Периферийное зрение read: шаг между строками растёт в ^PERIPHERAL_GAP_GROWTH
    # на каждом кольце от фокуса (меньше → плотнее).
    PERIPHERAL_GAP_GROWTH: float = 1.8
    # Периферийные строки обрезаются до N символов (фокус — без лимита). 0 = не резать.
    PERIPHERAL_MAX_LINE_CHARS: int = 45
    # Периферия в каждую сторону ≤ PERIPHERAL_SIDE_FACTOR × размер фокуса строк.
    # 0 = без ограничения (до краёв файла).
    PERIPHERAL_SIDE_FACTOR: float = 1.33
    # Вокруг каждой выбранной периферийной строки захватывается ещё ±N соседних
    # строк как локальный контекст (0 = выключено).
    PERIPHERAL_LINE_CONTEXT: int = 1

    # Лимит вывода любого инструмента (символов); read/search режут сами.
    # MAX_READ_LINES_PER_CALL — доп. лимит строк порционного чтения.
    MAX_READ_CHARS_PER_CALL: int = 4500
    MAX_READ_LINES_PER_CALL: int = 80

    # Отключает авто-суммаризацию большого вывода любых инструментов.
    DISABLE_TOOL_AUTO_SUMMARIZATION: bool = True

    # ------------------------------------------------------------------
    # Изоляция инструментов (docs/TOOL_ISOLATION_SPEC.md): внешние инструменты
    # ------------------------------------------------------------------
    # True — внешние инструменты идут через воркеры групп; False — исполняются в
    # процессе хоста (тесты и отладка; поведение как раньше, без изоляции).
    TOOL_ISOLATION_ENABLED: bool = True
    # Период опроса очереди воркера (сек): как часто хост проверяет результат и стоп.
    TOOL_WORKER_POLL_INTERVAL: float = 0.05
    # Крайний срок одного вызова инструмента в воркере (сек); по истечении воркер убивается.
    TOOL_WORKER_TIMEOUT: int = 3600
    # Сколько ждать завершения воркера после terminate, прежде чем идти дальше.
    TOOL_WORKER_KILL_GRACE: float = 1.0

    # Bash на Windows (run_bash_host): "wsl" — через WSL, "gitbash" — Git Bash, "auto" — gitbash при наличии иначе WSL, "system" — shutil.which("bash").
    BASH_BACKEND: str = "auto"

    # Путь к Git Bash для BASH_BACKEND="gitbash"; пусто — поиск по стандартным местам.
    GIT_BASH_PATH: str = ""

    # Явный корень проекта перекрывает авто-поиск по .git; можно задать через --project-root. None = авто-поиск.
    PROJECT_ROOT: str = r""

    # ------------------------------------------------------------------
    # Авто-сохранение истории после каждого нового сообщения (защита от сбоев)
    # ------------------------------------------------------------------
    AUTOSAVE_ENABLED: bool = True
    AUTOSAVE_DIR: str = "autosave"
    AUTOSAVE_KEEP: int = 25

    SAVE_IMAGES: bool = False

    SERVICE_TURN_KEEP_IN_HISTORY: bool = False
    # Лимит итераций служебного хода (перегенерации повторов/ошибок идут компактно).
    SERVICE_TURN_MAX_ITER: int = 2

    PER_MSG_SUMMARIES_ENABLED: bool = False

    # Архив вытесненных оригиналов (recall); выключение ломает только ответы про давно удалённое из контекста.
    MEMORY_ARCHIVE_ENABLED: bool = True
    # Лимиты recall-инструментов (символы).
    RECALL_SNIPPET_CHARS: int = 300
    RECALL_READ_MAX_CHARS: int = 4000
    RECALL_ENTRY_MAX_CHARS: int = 1500
    RECALL_MAX_ARG_CHARS: int = 500

    SCREENSHOT_MAX_WIDTH: int = 960
    SCREENSHOT_QUALITY: int = 60
    # Дубль последнего скриншота для отладки (как debug_latest.jpg в screen_agent); "" — не сохранять.
    SCREENSHOT_DEBUG_PATH: str = ""
    # Длительность анимации move (сек); 0 — мгновенный перенос.
    MOUSE_MOVE_DURATION: float = 0.25
    # Пауза после каждого pyautogui-действия (сек).
    PYAUTOGUI_PAUSE: float = 0.1
