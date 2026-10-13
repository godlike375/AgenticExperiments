# Карта кодовой базы `universal_agents`

> Что это: фреймворк вокруг LLM с tool-calling, KV-cache-стабильным префиксом, сжатием
> контекста и защитой от зацикливания. Этот файл — навигатор: что где лежит, кто что
> делает и где это тестируется. AGENTS.md описывает правила/инварианты, этот файл — раскладку.

## Слои

```
tools/ (плагины: fs, builtin, host_shell, memory, sandbox, pc_control)
   │  @tool-декоратор (tool.py) → JSON Schema, флаги безопасности
   ▼
tool_registry / tool_manager — загрузка, allow/deny, trust_dir
   ▼
AGENT = agent.py LLMAgent = 7 миксинов (agent_mixins/) ──► history / models
   │   _run_turn_loop: вызов LLM → детект дублей/ошибок → исполнение инструментов
   ▼
context_builder.prepare_messages_for_api — KV-cache-безопасная сериализация
   ▼
llm_client.LLMClient.call — ВСЕГДА стриминг (StreamSession); InProcessStream — прямой httpx POST, отмена через shutdown на сокете
   ▼
ui.py CLI / main.py — точка входа, консоль, подтверждения, стоп
```

## Ядро агента

### `agent.py` — LLMAgent (главный класс) + TurnState

| Что | Где | Назначение |
|---|---|---|
| `TurnState` | agent.py | Состояние одного хода: prefill, счётчики ошибок (`record_tool_error`/`success`), broken-call ретраи, guard-ретраи подтверждения |
| `chat()` / `_run_turn_loop()` | agent.py | Главный цикл хода: вызвать LLM → обработать ответ → исполнить инструменты → повторить |
| `call_with_retries()` | agent.py | Эскалация при проблемах: температура → NAG → сдача хода (`_recover`) |
| `_detect_duplicate()` | agent.py | Детектор повторов (§1.9): хэш по всей истории + биграммный Jaccard; при live-pending отдаёт вердикт guard'у подтверждения |
| `service_llm_call()` | agent.py | Служебный ход против той же истории (саммаризация, компакция, драфты) |
| `_call_llm()` / `_call_with_streaming()` | agent.py + streaming_mixin | Стриминг, расхождение с prefill, watchdog |
| `_process_llm_response()` | response_mixin | Обработка ответа: NO COMMENT, подтверждение, скраб confirmation-мусора |
| `autosave` / `save_history` / `reset_dialog` | agent.py | Автосохранение и reset |
| `set_pending_operation`/`pop_pending_operation` | agent.py | Отложенная операция `requires_model_confirmation` |
| `inject_user_interrupt` | agent.py | Прерывание пользователя |
| `set_pending_interrupt`/`take_pending_interrupt` | agent.py | API для тестов (клик «стоп» на ходу) |

Тесты: `tests/test_agent.py` (+ `tests/agent_mixins/test_*`).

### `agent_mixins/` — семь миксинов (композиция в `agent.py`)

| Файл | Назначение |
|---|---|
| `tools_mixin.py` | Поверхность инструментов для агента: `load_tool`/`unload_tool` (помечает, не удаляет — KV), `trust_dir`, проверка внешних путей, авто-доверие git-root |
| `memory_mixin.py` | Сжатие: per-message саммари (`_summarize_assistant_message` и т.д.), компакция задач (`_compact_completed_tasks`), авто-суммаризация по порогу (`_auto_summarize_dialogue`) |
| `history_mixin.py` | Стирание ассистентских сообщений и failed-цепочек, сигнатура последней ошибки, `_scrub_confirmation_trail` (вычистка мусора подтверждения) |
| `streaming_mixin.py` | `_call_with_streaming`, callbacks, `_watch_diverged` (расхождение стрима с prefill) и продолжение после него |
| `response_mixin.py` | `_assemble_assistant_message`, `_append_tool_results`, NO COMMENT flow; `SimReasoningGroup` — симуляция reasoning (`<thinking>`-секция, §2.8) |
| `execute_mixin.py` | `_execute_tools`: исполнение вызовов, dry_run для `requires_model_confirmation`, подтверждение |
| `consistency_mixin.py` | Self-consistency: несколько драфтов + синтез (`/consistent`) |

Тесты: `tests/agent_mixins/test_<mixin>.py` (7 файлов).

## Модель и история

| Модуль | Назначение | Ключевые понятия | Тесты |
|---|---|---|---|
| `models.py` | Типы сообщений | `Message` (ABC), `SystemMessage` (всегда index 0), `UserMessage` (header-кэш `reset_header_cache`, `is_auxiliary`), `AssistantMessage`, `ToolResult` (`success`/`error`/`user_denied`), `ToolCall`; `to_api_dict` (для LLM) vs `to_persist_dict` (для JSON) | `tests/test_models.py` |
| `history.py` | `ChatHistory` | добавить/удалить/`edit_message`/`normalize`/`delete_range`/`replace_range`; per-message summary sidecar; `compress_old_messages`; save/load/`load_from_payload`; позиционная зависимость last-user header | `tests/test_history.py` |
| `context.py` | `AgentContext` protocol | Интерфейс миксинов (компромисс: без жёсткого протокола) | — |
| `context_builder.py` | `prepare_messages_for_api` | KV-cache-безопасная сериализация: шапки user-сообщений (волатильная цифра бюджета только у последнего user), `_check_prefix_hashes` ([PREFIX-HASH] debug) | `tests/test_context_builder.py` |

## Транспорт и генерация

| Модуль | Назначение | Ключевые понятия | Тесты |
|---|---|---|---|
| `llm_client.py` | Единый стриминговый транспорт | `LLMClient.call` всегда `StreamSession`; `LLMClient.open_stream` всегда возвращает `InProcessStream` (прямой httpx POST, SSE-парсинг руками, отмена через `shutdown` на сокете); `StreamSession.consume` (watchdog-поток для отмены); `StreamAccumulator` (prefill-эхо, маркеры); `TokenUsageTracker`; `LoopDetector` (повтор вызовов инструмента в ходу); `jaccard_similarity`, `text_hash`; stop-маркеры `stopped_at_marker` | `tests/test_llm_client.py`, `tests/test_in_process_stream.py` |
| `generation.py` | Параметры генерации | `GenerationParams` (+overrides), `StructuredOutputConfig` (`from_controller`/`from_prefill`), `apply_stop_markers` | `tests/test_generation.py` |
| `controllers.py` | Структурированный вывод | `XMLStructureController` — пофазовый вывод по схеме `<root><a/></root>`, автозакрытие тегов, `from_sibling_schema` (для симуляции reasoning) | `tests/test_controllers.py` |
| `tool_parsing.py` | Парсинг вызовов | `tc_name`/`tc_args`, `detect_broken_call`, `parse_tool_args`, `normalize_args` | `tests/test_tool_parsing.py` |
| `exceptions.py` | `GenerationInterrupted` | Исключение прерывания генерации | — |

## Инструменты

| Модуль | Назначение | Ключевые понятия | Тесты |
|---|---|---|---|
| `tool.py` | `@tool`-декоратор | Генерация JSON Schema из аннотаций, флаги `requires_confirmation`/`requires_model_confirmation`/`path_safety`/`safe_in_trusted`/`self_truncating`, `group`/`hidden` (§1.13), `ToolOutput` | `tests/test_tool.py` |
| `tool_registry.py` | Динамическая загрузка плагинов | `load_external_plugins` (+ кэш `_PLUGIN_CACHE`), `build_tool_dict` (выводит `group`/`is_external`/`hidden`) | `tests/test_tool_manager.py` (частично) |
| `tool_manager.py` | `ToolManager` | load/unload (пометить), deny, trust_dir, `flush_pending_unloads`, схемы для префикса; hidden-инструменты скрыты и не грузятся; мягкий контроль разметки групп | `tests/test_tool_manager.py` |
| `tool_session.py` | Состояние группы (contextvar) | `get_session`/`set_session`/`current_session_or_none` — сессия группы на время вызова | `tests/test_tool_isolation.py` |
| `tool_worker.py` | Универсальный воркер группы | `worker_main(group)` (spawn): грузит инструменты группы, применяет Config/корень/cwd, вызывает, возвращает результат и состояние | `tests/test_tool_isolation.py` |
| `group_worker.py` | Изоляция инструментов | `GroupManager` (сессии+воркеры+метрики), `ProcessGroupWorker` (ленивый старт, kill по стопу/таймауту), `make_worker_resolve` (применение обёрнутой правки) | `tests/test_tool_isolation.py` |
| `plugin_proxy.py` | Прокси-агент плагинов | `plugin_proxy.call(method, …)` — межпроцессный доступ плагина к allow-list методов агента | `tests/test_tool_isolation.py` |
| `tools/builtin.py` | Встроенные инструменты | `get_messages`, `edit_message`, `delete_messages`, `summarize_messages`, `delegate_to_subagent`, `load_tool`, `unload_tool`, `make_plan`, `have_done`, **`respond_to_system`** (инструмент подтверждения guard'а) | `tests/test_tool.py` (частично), `tests/agent_mixins/test_execute_mixin.py` (guard) |
| `tools/fs.py` | Файловая система | `read` (периферийное зрение, саммари, dedup через `file_states`), `search`, `line_range_edit`/`match_replace_edit` (dry_run), `cwd` | `tests/tools/test_fs.py` |
| `tools/host_shell.py` | Shell | `run_powershell` (legacy `fallback_encoding="oem"`) / `run_bash_host` (git-bash) | `tests/tools/test_host_shell.py` |
| `tools/sandbox.py` | Docker-песочница | `start_sandbox`/`run_bash`/`run_python`/`stop_sandbox` — `UnifiedDockerAgent` | — |
| `tools/memory.py` | Архив | `recall_search`/`recall_read` — доступ к `HistoryArchive` | (через memory/archive) |
| `tools/pc_control.py` | Управление ПК | `screenshot`, `mouse_move/click`, `scroll`, `type_text_on_keyboard`, `press_key` | `tests/tools/test_pc_control.py` (исключён из обычного прогона) |
| `screen_state.py` | Состояние PC-контроля | `last_scale` — шкала координат для pc_control | — |

PRELOADED_TOOLS (main.py): `load_tool`, `read`, `line_range_edit`, `match_replace_edit`, `cwd`, `search`, `run_powershell`, `respond_to_system`. Остальное модель грузит сама `load_tool`.

## Сжатие, память, задачи

| Модуль | Назначение | Ключевые понятия | Тесты |
|---|---|---|---|
| `compressors.py` | Пайплайн сжатия | `compact_segment`, `summarize_history_plain` (+review pass), `chunk_and_summarize_large_text`, `auto_compress_tool_result`, `_dense_summarize_message` (per-message) | `tests/agent_mixins/test_memory_mixin.py`, `tests/test_service_turn.py` |
| `task_tracker.py` | План/выполнение | `set_plan`, `mark_task_done`, `validate_task_mark_call`, `compact_completed_tasks` (маркеры `make_plan`/`have_done` остаются) | `tests/test_task_tracker.py` |
| `archive.py` | `HistoryArchive` | Хранит удалённые при компакции сообщения; `search`, `read_span` | (в составе test_memory_mixin/history) |
| `file_states.py` | Dedup чтений | `FileStateTracker`: помнит прочитанные файлы (диск+контент-хэши), `should_skip`, связь с tool call | `tests/tools/test_fs.py` (dedup), test_memory_mixin |
| `subprocess_utils.py` | Выполнение shell | `run_capture` (таймаут, прерывание, `_decode_bytes`+fallback), `truncate_middle`, kill tree/job | `tests/test_subprocess_utils.py` |
| `command_paths.py` | Пути из команд | `extract_paths` — для `path_safety` со shell-команд | `tests/test_path_safety.py` |
| `project_root.py` | Корень проекта | `find_project_root` (по `.git`), `is_within`, `external_paths` | `tests/test_path_safety.py` |
| `sub_agent.py` | Суб-агент | `SubAgent` — наследует системный промпт/history/инструменты родителя, свой трекер токенов; `run_subagent_once` | `tests/test_tool_manager.py` (частично) |

## UI, вход, конфиг

| Модуль | Назначение | Ключевые понятия | Тесты |
|---|---|---|---|
| `ui.py` | Консоль и CLI | `ConsoleUI` (рендер, колбэки стрима), `CLI` (команды `/regen`, `/think`, `/stop`, `/trust`, `/compact_history`, `/consistent` …); `_call_chat` (поток генерации сджойнен, без detach) | `tests/test_ui.py` |
| `main.py` | Точка входа | `build_allowed_tools`, `parse_args`, запуск CLI | — |
| `rendering.py` | Рендер сообщений | `render_message` | `tests/test_rendering.py` |
| `code_extractor.py` | Дамп проекта | `dump_project` — скинуть код проекта одним файлом (для контекста LLM) | — |
| `config.py` | Все константы | `Config` (dataclass): температура/эскалация (`BOOST_TEMP`…), KV-безопасность (`KEEP_REASONING…`), сжатие (`AUTO_SUMMARY_*`, `TASK_COMPACTION_ENABLED`, `MAX_READ_CHARS_PER_CALL`), стриминг/прерывание (`STREAM_ENABLED`), симуляция reasoning (`SIMULATED_REASONING_*`), PC-контроль, autosave | везде через `monkeypatch`/`patch.object(Config, …)` |
| `constants.py` | Маркеры | `ENVIRONMENT_PREFIX`/`ENVIRONMENT_PREFIX_END`, `INTERRUPT_HEADER`, `err()`/`ok()` | — |

## Сквозные механики: какие файлы трогают

| Механика | Файлы |
|---|---|
| KV-cache/стабильный префикс (§1.2) | `models.py` (header-кэш), `context_builder.py`, `history.py` (нормализация, ремувы), `agent_mixins/tools_mixin.py` (unload не удаляет), `agent.py` (`_ensure_guard_nag`, скраб не трогает шапки) |
| Компакция/сжатие (§3) | `compressors.py`, `task_tracker.py`, `archive.py`, `history.py` (`compress_old_messages`), `agent_mixins/memory_mixin.py` |
| Guard подтверждения (§1.11) | `tools/builtin.py` (`respond_to_system`), `agent.py` (`_detect_duplicate`, `_CONFIRMATION_GUARD_VERDICT`, `_run_turn_loop`, `set_pending_operation`), `agent_mixins/history_mixin.py` (`_scrub_confirmation_trail`), `agent_mixins/execute_mixin.py` (dry_run) |
| Детекция повторов/ошибок (§1.9-1.10) | `agent.py` (`_detect_duplicate`, `TurnState`, `_recover`), `llm_client.py` (`jaccard_similarity`, `text_hash`, `LoopDetector`), `agent_mixins/history_mixin.py` (`_last_failed_tool_signature`) |
| Стриминг/прерывание (§1.12) | `llm_client.py` (`LLMClient.call`, `StreamSession`), `agent_mixins/streaming_mixin.py`, `ui.py` (`_call_chat`, `gen.join()`) |
| Симуляция reasoning (§2.8) | `agent_mixins/response_mixin.py` (`SimReasoningGroup`), `controllers.py` (`from_sibling_schema`), `config.py` (`SIMULATED_REASONING_*`) |
| Чтение файлов (§3.4, dedup) | `tools/fs.py`, `file_states.py`, `compressors.py` (`chunk_and_summarize_large_text`) |

## Цепочки вызовов (главные пути)

> Деревья по реальному коду (проверено на текущей ветке). `→` — вызов, отступ — вложенность.
> Показаны основные ветки; редкие исключения опущены.

### 1. Точка входа: CLI

```
main.py: main()
 └─ ConsoleUI(agent), CLI(agent) ───────────────► ui.py
     └─ cli.run()                       [главный цикл]
         ├─ /команды (regen, clear, prefill, save, load, consistent,
         │   think/think_on/think_off, compact_history, trust/untrust…)
         │   ├─ /stop  → self._request_stop() → agent.request_stop()
         │   └─ /compact_history → agent._auto_summarize_dialogue(force=True) + _report_compaction
         └─ обычное сообщение → self._call_chat(message, prefill)
             ├─ запуск потока-демона: gen ← threading.Thread(_run_chat, args=(message,))
             │    └─ _run_chat: inject_user_interrupt (если interrupt) | agent.chat(message, prefill)
             │         ├─ поки mgmt → вход в цикл _run_turn_loop (см. №2)
             │         └─ вывод идёт через on_render / on_stream_* колбэки
             ├─ монитор ввода, пока gen.is_alive():
             │    ├─ _answer_next_confirm()  — если ждёт подтверждение инструмента (y/n)
             │    ├─ 'q'/'stop' → _request_stop(), выход
             │    ├─ текст при живом LLM:
             │    │    ├─ в момент выполнения инструмента → agent.set_pending_interrupt(line)   [сценарий В]
             │    │    └─ иначе → _request_stop(), redirect=line                                [А/Б]
             │    └─ (поток завершился сам) → redirect = agent.take_pending_interrupt()
             ├─ gen.join()   [без ограничения: отсоединения нет]
             └─ loop: redirect есть → новый _call_chat(redirect); иначе выход в run()
```

### 2. Ход пользователя: `agent.chat()` / `inject_user_interrupt()`

```
LLMAgent.chat(message, max_iter, prefill, structured_output)
 ├─ self_consistency_mode? → return _chat_self_consistent(message, prefill)
 ├─ _prepare_turn(message)
 │    ├─ pop_pending_tool_calls() при висящем assistant-вызове / пустая заглушка assistant
 │    ├─ history.add(UserMessage(message))          [индекс 1+, KV-безопасно]
 │    ├─ per-msg: _maybe_summarize_user_message(user_msg)
 │    ├─ autosave()                                [§1.8, если autosave=True]
 │    └─ clear_stop()
 └─ return _run_turn_loop(max_iter, prefill, structured_output)

LLMAgent.inject_user_interrupt(user_text)      [сценарии А/Б/В, из ui]
 ├─ _append_interrupt_to_tool_result(user_text)?  → если последний — ToolResult:
 │    дописывает шапку прерывания в его content + autosave; ход продолжается
 └─ иначе _prepare_turn(INTERRUPT_HEADER + user_text)
 └─ return _run_turn_loop(max_iter)
```

### 3. Главный цикл хода: `_run_turn_loop()`  ⭐

```
_run_turn_loop(max_iter, prefill, structured_output, service_mode=False)
 ├─ state = TurnState(prefill, structured_output)      [prefill, счётчики ошибок/ретраев]
 ├─ SimReasoningGroup(...) — если симуляция <thinking> активна (§2.8)
 │    (отключена при thinking, service_mode, чужом structured_output, prefill, субагенте)
 ├─ for _ in range(max_iter):
 │    ├─ if pending_interrupt / stop_event → autosave + return ""   [выход]
 │    ├─ step_prefill = state.step_prefill(); sim_group.prefill()
 │    ├─ effective_markers = маркеры структурированного вывода / (</tag>,) дума-фазы / ()
 │    ├─ all_messages = prepare_messages_for_api(self, debug_hash_check)
 │    │     └─ [KV-cache: шапки user-сообщений ложатся статично, мен. только у последнего user]
 │    ├─ message_obj, api_error, stopped_at_marker = call_with_retries(all_messages, …)
 │    │     └─ for attempt in max_generation_attempts:
 │    │          ├─ _call_llm(messages, prefill, gen_params, watch_prefix, stop_check, …)
 │    │          │    ├─ streaming_enabled? → _call_with_streaming (gr; mixin)
  │    │          │    │    └─ LLMClient.call → StreamSession(...).consume(...)   [stop: watchdog → close → shutdown на сокете]
 │    │          │    └─ иначе → LLMClient.call + apply_stop_markers (постфактум для моков)
 │    │          ├─ err → api_error_occurred=True; break
 │    │          ├─ duplicate = _detect_duplicate(message_obj)
 │    │          │    ├─ (CONFIRMATION_GUARD, …) → duplicate=None   [его решает guard, §1.11]
 │    │          │    ├─ ('answer'|'reasoning'|'tool_call', …) → буст температуры;
 │    │          │    │    NAG после DUPLICATE_NAG_THRESHOLD; continue
 │    │          │    └─ None → break
 │    │          └─ исчерпаны попытки → message_obj=None → _recover('duplicate_loop')
 │    ├─ GenerationInterrupted → autosave + return ""
 │    ├─ api_error / !message_obj → _recover('api_error'|'duplicate_loop') + return ""
 │    ├─ контроллер: controller.advance(content) → вердикт фазы / prefill следующей
 │    ├─ sim-группа: absorb → close_section (хвост в живой вывод) → фаза ответа →
 │    │    _sim_message / _sim_verify_phase; sim_open=False
 │    ├─ result_text, tool_error, broken_call, rerun_prefill = _process_llm_response(…)
 │    │    └─ см. №4
 │    ├─ rerun_prefill? → state.set_prefill(_NO_COMMENT_PREFILL); continue   [NO COMMENT]
 │    ├─ guard: pending + текст без вызова?
 │    │    ├─ _erase_last_assistant()
 │    │    ├─ can_retry_answer_guard()? → notice (attempt N/5) + _ensure_guard_nag
 │    │    │    + temp=ERROR_RECOVERY_TEMP + continue
 │    │    └─ иначе → _recover('answer_guard') + return ""
 │    ├─ broken_call → _erase_last_assistant + _recover('broken_call'|'broken_fix') / сдача
 │    ├─ tool_error → _erase_last_failed_tool_call + _recover('tool_error') / сдача
 │    ├─ компакция (не service_mode):
 │    │    ├─ _compact_completed_tasks() > 0 → compacted=True
 │    │    └─ при пороге контекста → _auto_summarize_dialogue() (одна за ход)
 │    ├─ state.record_tool_error(sig)/success; max_errors_reached → _recover('error_limit')
 │    ├─ autosave(); stop_event → return result_text
 │    ├─ контроллер: complete/фаза/next_prefill → return document | continue
 │    └─ нет tool_calls и нет ошибки → return result_text        [ход завершён]
 └─ (последний цикл) → return ""
```

### 4. Обработка ответа LLM: `_process_llm_response()`  ⭐

```
response_mixin._process_llm_response(message_obj, no_comment_retry_left, …)
 ├─ service_mode (allow_tools=False): tool_calls отбрасываются;
 │    есть текст → остаётся; текста нет → rerun_prefill / tool_error
 ├─ _build_assistant_msg(message_obj, clean_content)            [модель → AssistantMessage]
 ├─ tool_calls: выбирается ОДИН валидный вызов (остальные отбрасываются, [MULTIPLE TOOLS])
 ├─ голый вызов без текста (reasoning=off):
 │    все попытки есть → rerun_prefill = _NO_COMMENT_PREFILL (не добавляем в историю)
 │    исчерпаны → исполняется как есть
 ├─ пустой ответ без вызовов → ('', True, …)  → перегенерация с бустом
 ├─ pending_before = текущее pending ДО _execute_tools
 ├─ _append_assistant(assistant_msg) → в историю
 ├─ при pending и вызов ≠ respond_to_system → _mark_confirmation_junk(assistant_msg)
 ├─ нет tool_calls:
 │    detect_broken_call(clean_content, known_tools) → broken_call=True
 │    иначе → return (текст, False, False, None)
 ├─ tool_results = _execute_tools(tool_calls)                    [см. №5]
 │    ├─ respond_to_system + pending + ошибка → junk (упавшая пара)
 │    └─ чужой инструмент при pending → junk (результат)
 ├─ _append_tool_results(tool_results)
 ├─ respond_to_system успешно → _scrub_confirmation_trail()      [вычистить наг и мусор]
 └─ return (текст, tool_error, False, None)
```

### 5. Исполнение инструментов: `_execute_tools()`  ⭐

```
execute_mixin._execute_tools(tool_calls)
 ├─ set_interrupt_event(stop_event); _in_tool_execution=True     [для убийства subprocess]
 ├─ для каждого tc:
 │    ├─ tools_manager.is_denied → ToolResult.error (forbidden у субагента)
 │    ├─ is_pending_unload → ошибка «unloaded, load_tool сначала»
 │    ├─ LoopDetector: дубликат в пакете/ходу → ENVIRONMENT-варнинг
 │    ├─ неизвестный инструмент → ошибка
 │    ├─ name == have_done → validate_task_mark_call (порядок задач)
 │    ├─ requires_confirmation / path_safety:
 │    │    ├─ _check_external_paths (вне project root → предупреждение)
 │    │    └─ !on_confirm(name, args) → ToolResult.user_denied
 │    ├─ requires_model_confirmation:                    [двухфазное подтверждение]
 │    │    ├─ handler(self, **{**args, dry_run:True}) → (preview, resolve, ask)
 │    │    ├─ set_pending_operation({resolve, tool_name})
 │    │    └─ ToolResult = preview + ask → ждём respond_to_system
 │    ├─ respond_to_system ──выполнение подтверждения:
 │    │    ├─ pop_pending_operation() → resolve()
 │    │    └─ (после цикла) _scrub_confirmation_trail()
 │    └─ обычный вызов: handler(self, **args) 
 │         ├─ ToolOutput → .text + .images (side-field)
 │         ├─ усечение > MAX_READ_CHARS_PER_CALL (если не self_truncating)
 │         ├─ _tool_usage[n]++, file_states.mark_tool_call (для read)
 │         ├─ auto_compress_tool_result (chunk_and_summarize_large_text) — если не skip
 │         └─ autosave() [после КАЖДОГО инструмента]
 │         ├─ Exception → ToolResult.error + sys.msg
 ├─ clear_interrupt_event(); _in_tool_execution=False
 └─ return results
```

### 6. Служебный вызов: `service_llm_call()` → `_run_service_turn()`

```
service_llm_call(msgs, temp, timeout, tools, prefill, params, structured_output)
 ├─ извлекает промпт из последнего user в msgs (без системной шапки)
 ├─ совпадает с последним user в истории? → append_prompt=False   [consistency-драфт]
 ├─ _run_service_turn(prompt, append_prompt, keep_in_history, temp, …)
 │    ├─ start_len = len(history)
 │    ├─ swap стрим-каналов: on_*stream* → on_service_stream_*
 │    ├─ _auto_summarize_suppressed=True; _in_service_turn=True
 │    ├─ _gen_params = dc_replace(temp/timeout…)                 [свои параметры хода]
 │    └─ return _run_turn_loop(service_mode=True)
 │         ├─ промпт в историю как UserMessage (KV-cache-безопасно)
 │         ├─ allow_tools=False: tool_calls отбрасываются/перегенерация (§2.4)
 │         ├─ авто-сейв/компакция/авто-суммаризация отключены
 │         └─ итог: _run_service_turn откатывает ход (remove_at + normalize
 │              + _on_history_changed), если не keep_in_history
 └─ return SimpleNamespace(content=result, tool_calls=[], …), None, None
```

### 7. Восстановление после ошибок: `_recover(kind)`

```
agent._recover(kind, retries_left, erased_count)     [единая точка эскалации, §2.6]
 ├─ kinds: 'duplicate_loop' | 'api_error' | 'answer_guard' | 'error_limit' |
 │         'broken_call' | 'broken_fix' | 'broken_giveup' | 'tool_error'
 ├─ печатает sys.msg («LIMIT REACHED…», «BROKEN CALL…», …)
 ├─ буст температуры/параметры хода
 └─ ход либо продолжает (continue в цикле), либо сдаётся пользователю (return "")
```

### 8. Самые частые входы в сжатие контекста

```
_read / search (fs.py) ────► FileStateTracker (dedup: disk+content hash → should_skip)
_at порога контекста ─────► memory_mixin._auto_summarize_dialogue() ──► compressors.compact_segment
                                  → summarize_history_plain (контроллер XML) → history.compress_old_messages
                                  → HistoryArchive.append_messages (оригиналы в архив)
_task_tracker.set_plan/mark_task_done ──► compact_completed_tasks(have_done-листья) → summarize_task_segment
_delegate_to_subagent (builtin) ────────► SubAgent.run ──► subagent._chat_self_consistent/chat (только текст)
```

| Тест-файл | Покрывает |
|---|---|
| `tests/test_agent.py` | цикл хода, детект дублей, префикс, guard-вердикты, стоп |
| `tests/test_llm_client.py` | транспорт (всегда стриминг), stop-маркеры, usage |
| `tests/test_history.py` | история, header-инвалидация, сжатие |
| `tests/test_context_builder.py` | KV-безопасная сериализация, шапки |
| `tests/test_ui.py` | CLI, ввод/подтверждения, компакция |
| `tests/agent_mixins/test_*.py` | каждый миксин (7 файлов) |
| `tests/tools/test_*.py` | fs, host_shell, pc_control (вне основного прогона) |
| `tests/test_models.py`, `test_tool.py`, `test_tool_parsing.py`, `test_tool_manager.py`, `test_generation.py`, `test_controllers.py`, `test_task_tracker.py`, `test_subprocess_utils.py`, `test_rendering.py`, `test_service_turn.py`, `test_path_safety.py` | по одному модулю каждый |

Прогон: `python -m pytest tests -q --ignore=tests/tools/test_pc_control.py` (сейчас 466 passed, 11 subtests).
Общие фикстуры/фабрики (в т.ч. `make_agent`, `sim_reasoning_off`) — в `tests/conftest.py`.

## Куда смотреть при правке

- **Поведение агента** → `agent.py` + нужный миксин; перед правкой читай §1.1-1.12 AGENTS.md.
- **Новый инструмент** → файл в `tools/` + `@tool` + флаги; добавить в `PRELOADED_TOOLS` (main.py) если нужен по умолчанию; тест в `tests/tools/`.
- **Промпты/тексты, видимые LLM** → правка только с разрешения (§7 AGENTS.md).
- **Сломались KV / байтовые контракты префикса** → `context_builder.py`, `models.py` (header-кэш), `history.py`; не удалять контракты-тесты.