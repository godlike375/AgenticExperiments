# AGENTS.md — Правила работы с репозиторием universal_agents

> Этот файл читают LLM-агенты (opencode, Cursor, Copilot) при работе с проектом.
> Следуй этим правилам — и код будет правильным с первого раза.

---

## 0. Цель проекта

Создать самую качественную обвязку (framework) вокруг LLM, которая выжмет максимум из любой модели через специальные трюки и хитрости:
- **KV-cache reuse** — стабильный префикс сообщений для максимального переиспользования кэша
- **Экономия контекстного окна** — умное сжатие, суммаризация, peripheral vision, архивация
- **Стабильность** — защита от зацикливания, восстановление после ошибок,.auto-repair истории

---

## 1. Железные инварианты (НАРУШИТЬ = СЛОМАТЬ)

### 1.1. Системный промпт всегда под индексом 0
Диалог начинается с индекса `Config.AFTER_SYSTEM_PROMPT` (=1). Никогда не удалять и не перемещать `SystemMessage`.

### 1.2. KV-cache reuse через стабильный префикс
Префикс сообщений (system prompt + инструменты + начало истории) должен оставаться байт-идентичным между вызовами LLM. Нарушение = каждый вызов пересылает весь контекст заново.

**Отсюда вытекает:**
- Инструменты всегда передаются в API даже в служебных вызовах (саммаризация и т.п.)
- Заголовки user-сообщений кэшируются (`UserMessage._cached_header`)
- Суб-агент наследует системный промпт и историю родителя
- При загрузке/выгрузке инструментов через `load_tool`/`unload_tool` схемы **не удаляются** из префикса — `unload_tool` только помечает, реальное удаление happens при `flush_pending_unloads()` на точках инвалидации кэша (сжатие/удаление/правка истории)

### 1.3. Один вызов инструмента за ход
Модель может вызвать только один инструмент за сообщение. Это ограничение API и промпта.

### 1.4. Единый формат ошибок инструментов
Результат начинается с `Error:` (через `constants.err()`) или помечен `is_user_denied`. Агент различает их по этим признакам.

### 1.5. Сброс кэша чтений при изменении истории
Любая операция над историей (удаление, сжатие, компактизация) = `self._on_history_changed()` → сброс `FileStateTracker` + `flush_pending_unloads()`.

### 1.6. Три файла с двумя представлениями
- `models.py`: `to_api_dict()` — для LLM; `to_persist_dict()` — для JSON (с `_skip_summarize`, `_is_error` и т.д.)
- Никогда не путать: `_api_dict` → в запрос к модели, `_persist_dict` → в файл/архив

---

## 2. Архитектура

### 2.1. Миксины
`LLMAgent` composing 7 миксинов:
- `ToolsMixin` — публичные методы управления инструментами (load/unload/trust)
- `MemoryMixin` — рабочая память, компакция диалога
- `HistoryMixin` — удаление/восстановление сообщений
- `StreamingMixin` — стриминг + divergence detection + watchdog
- `ResponseMixin` — сборка AssistantMessage, обработка ответа, NO COMMENT flow
- `ExecuteMixin` — выполнение ToolCall'ов, confirmation flow, dry_run
- `ConsistencyMixin` — self-consistency режим (несколько черновиков + синтез)

**Миксины обращаются к агенту через `self` неявно.** Нет интерфейса/протокола — это осознанный компромисс для простоты. При добавлении нового миксина полагайся на те же поля, что и остальные: `self.history`, `self.tools_manager`, `self.token_tracker`, `self.on_system_msg`, `self.on_render`, `self.stop_event`, `self._gen_params`, `self.loop_detector`, `self.file_states`, `self.archive`.

### 2.2. Инструменты
Инструменты — функции в `tools/`, декорированные `@tool`. Динамически загружаются из `tools/` директории.

**Создание нового инструмента:**
```python
from universal_agents.tool import tool

@tool(
    description="Что делает (видно LLM)",
    short_description="Короткое описание",
    # Флаги безопасности:
    # requires_confirmation=True — спрашивает подтверждение у пользователя
    # requires_model_confirmation=True — сухой прогон → превью → answer_to_system() → выполнение
    # path_safety=True — проверяет пути вне проекта
    # safe_in_trusted=True — в доверенной папке пропускает подтверждение
    param_name=("type", "Описание. 'Optional ...' если необязательный"),
)
def my_tool(agent: 'LLMAgent', param_name: str = "default") -> str:
    # agent передаётся первым если инструменту нужен доступ к агенту
    # Без agent: def my_tool(param_name: str) -> str:
    ...
```

**Типы параметров**: `str`, `int`, `float`, `bool`, `list`, `dict`

**Имя инструмента = `func.__name__`** (задаётся декоратором: `_tool_name = func.__name__`). Переименование функции автоматически переименовывает инструмент; никогда не дублировать имя строковым литералом в коде — вместо этого импортировать функцию и использовать `func.__name__` (например `_read_tool.__name__`).

**Метаданные** (фиксятся декоратором):
- `_is_tool = True` — маркер для загрузчика
- `_tool_name` — имя функции (`func.__name__`)
- `_requires_confirmation`, `_requires_model_confirmation`, `_path_safety`, `_safe_in_trusted`
- `_has_agent_param` — первый параметр `agent`?
- `_tool_schema` — JSON Schema для API

### 2.3. Конфигурация
`Config` в `config.py` — статический класс с константами. Все настройки с дефолтами.
При добавлении новой фичи: добавь константу в `Config` с понятным именем и дефолтом.

### 2.4. Потоки и прерывания
Агент работает в двух потоках: основной (UI) и генерации. Прерывание через `stop_event` (threading.Event). Watchdog-поток закрывает HTTP-соединение стрима.

**Три сценария прерывания:**
- **А/Б** (во время генерации текста): `request_stop()` → `stop_event` → генерация завершается → `inject_user_interrupt(text)`
- **В** (во время выполнения инструмента): `set_pending_interrupt(text)` → инструмент завершается → текст вставляется в конец ToolResult → новая генерация

---

## 3. Сжатие и экономия контекста

### 3.1. Три уровня сжатия
1. **Рабочая память** (per-message summaries) — плотные саммари каждого длинного сообщения, хранятся вне контекста. При сжатии всего диалога склеиваются.
2. **Автосуммаризация по порогу** — когда контекст забит >85%, старая часть заменяется session summary (UserMessage с `is_summary=True`).
3. **Сверхдлинные выводы инструментов** — `chunk_and_summarize_large_text` нарезает на чанки, каждый анализируется суб-агентом, результат склеивается.

### 3.2. Archive/Recall
При компактизации оригиналы сообщений уходят в `HistoryArchive`. Доступ через `recall_search(query)` и `recall_read(from_seq, to_seq)`.

### 3.3. Task compaction
Завершённые подзадачи (по `have_done`) сжимаются в одно summary-сообщение. Маркеры `make_plan`/`have_done` остаются видимыми.

---

## 4. Стиль кода

- Язык комментариев и docstrings: **русский** (как в текущем коде)
- Имена переменных/функций/классов: **английские**
- Строки: до ~120 символов (ограничение не жёсткое)
- Типизация: `from __future__ import annotations` в каждом файле, аннотации типов обязательны
- Импорты TYPE_CHECKING: `if TYPE_CHECKING: from universal_agents.agent import LLMAgent`
- Константы: `Config` (класс) или `constants.py` (для общих маркеров)
- Формат ошибок: `constants.err(msg)` и `constants.ok(msg)` — единый формат
- Системные сообщения агенту: `ENVIRONMENT_PREFIX` ... `ENVIRONMENT_PREFIX_END`

---

## 5. Тесты

- Фреймворк: `pytest` (файлы `tests/test_*.py`)
- Фейковый агент: общий `make_agent` fixture в `conftest.py` (не дублировать в каждом тесте)
- Моки: `unittest.mock` для `on_render`, `on_confirm`, `on_system_msg`
- Не тестировать имплементацию — тестировать поведение
- Покрытие: каждый новый инструмент/фича → тест

---

## 6. Чего НЕ делать

- **Не редактировать промпты и любой текст, который видит LLM** (системный промпт, `description`/`short_description`/`param`-описания инструментов, `ENVIRONMENT_PREFIX`-инструкции, наги, сообщения-ошибки для модели) без явного разрешения пользователя-разработчика. Формулировки промптов — зона ответственности человека: он экспериментирует и знает, что работает лучше. AI-агент может предложить изменение, но не применять его сам.
- **Не лезть в `self.history._messages` напрямую** из миксинов — использовать `self.history.get_all()`, `self.history.get_last_message()` и т.д.
- **Не мутировать историю без `_on_history_changed()`** — иначе кэш чтений и выгрузки инструментов не обновятся
- **Не забывать `self._needs_normalize = True`** после мутации истории (через `self.history.add()` и т.д. — уже вызывается автоматически)
- **Не добавлять `_response_id` в Streaming путь** — он только для Responses API
- **Не путать `to_api_dict()` и `to_persist_dict()`** — первый для LLM, второй для JSON
- **Не дублировать логику** между миксинами и standalone-функциями (compressors.py, task_tracker.py)
- **Не создавать новые глобальные синглтоны** — Config уже есть
- **Не менять порядок схем инструментов** — KV-кэш переиспользуется только при байт-идентичном массиве

---

## 7. Добавление новой фичи — чеклист

1. Добавить константу в `Config` (если нужна настройка)
2. Реализовать в существующем модуле или создать новый в `agent_mixins/` (если это поведение агента) / `tools/` (если это инструмент)
3. Если инструмент — декорировать `@tool`, описать параметры, выставить флаги безопасности
4. Если миксин — обращаться к агенту через те же `self.*` поля что и другие миксины
5. Добавить в `PRELOADED_TOOLS` (main.py) если нужен по умолчанию
6. Добавить тест в `tests/`
7. Обновить REFACTORING_BASELINE.md (описание поведения) и/или этот файл (если изменились правила)

---

## 8. Структура файлов

```
universal_agents/
├── agent.py              # LLMAgent — main class (composes mixins)
├── agent_mixins/         # Behavior modules (tools, memory, history, streaming, response, execute, consistency)
├── tools/                # Tool plugins (fs.py, builtin.py, host_shell.py, sandbox.py, memory.py)
├── config.py             # All configuration constants
├── constants.py          # Shared markers (ENVIRONMENT_PREFIX, err(), ok())
├── models.py             # Message types (SystemMessage, UserMessage, AssistantMessage, ToolResult, ToolCall)
├── history.py            # ChatHistory — message list with normalize/save/load
├── llm_client.py         # LLMClient — transport (chat completions, streaming)
├── context_builder.py    # prepare_messages_for_api — KV-cache-safe serialization
├── compressors.py        # Summarization, compression, chunk analysis
├── task_tracker.py       # Plan + have_done + compaction
├── tool_manager.py       # ToolManager — load/unload/trust/deny
├── tool_registry.py      # @tool decorator + dynamic plugin loader
├── tool_parsing.py       # Tool call parsing, broken call detection
├── file_states.py        # FileStateTracker — read dedup
├── archive.py            # HistoryArchive — recall after compaction
├── sub_agent.py          # SubAgent — isolated agent for delegation
├── generation.py         # GenerationParams dataclass
├── command_paths.py      # Path extraction from shell commands
├── project_root.py       # .git root finder
├── subprocess_utils.py   # Shell execution, truncation, interrupt
├── rendering.py          # Message rendering helpers
├── exceptions.py         # GenerationInterrupted
├── ui.py                 # ConsoleUI + CLI (main loop)
├── main.py               # Entry point
├── autosave/             # Auto-saved history snapshots
├── tests/                # pytest test suite
├── REFACTORING_BASELINE.md  # Behavioral invariants (source of truth for refactoring)
└── AGENTS.md             # This file — rules for LLM agents
```
