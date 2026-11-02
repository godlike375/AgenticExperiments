# Карта кодовой базы `universal_agents`

> Что это: фреймворк вокруг LLM с tool-calling, KV-cache-стабильным префиксом, сжатием
> контекста и защитой от зацикливания. Этот файл — навигатор: что где лежит, кто что
> делает и где это тестируется. AGENTS.md описывает правила/инварианты, этот файл — раскладку.

## Слои

```
tools/ (плагины: fs, builtin, host_shell, sandbox, pc_control)
   │  @tool-декоратор (tool.py) → JSON Schema, флаги безопасности
   ▼
tool_registry / tool_manager — загрузка, allow/deny, trust_dir
   ▼
tool_worker.ToolWorker — ЕДИНЫЙ изолированный воркер-процесс для всех инструментов
   ▼
AGENT = agent.py LLMAgent (чистый фасад без миксинов)
   ├── TurnExecutor (turn_executor.py) — цикл ходов, стриминг, ретраи, сборка ответов
   ├── ToolPipeline (tool_pipeline.py) — конвейер инструментов: безопасность путей, подтверждения, вызов
   ├── RecoveryPolicy (recovery_policy.py) — детекция дублей текста, аргументов инструментов и ошибок
   └── SessionStorage (session_storage.py) — автосохранения, ротация, сброс диалога
   ▼
context_builder.prepare_messages_for_api — KV-cache-безопасная сериализация
   ▼
llm_client.LLMClient.call — ВСЕГДА стриминг (StreamSession); InProcessStream — прямой httpx POST, отмена через shutdown на сокете
   ▼
ui.py CLI / main.py — точка входа, консоль, подтверждения, стоп
```

## Ядро агента

### `agent.py` — LLMAgent (главный класс-фасад)

| Что | Где | Назначение |
|---|---|---|
| `LLMAgent` | agent.py | Компактный фасад (<400 строк), координирующий исполнитель ходов, политики восстановления и хранилище сессий |
| `TurnExecutor` | turn_executor.py | Главный цикл хода `run_turn_loop`, `call_with_retries`, `service_llm_call`, `inject_user_interrupt` |
| `TurnState` | turn_executor.py | Состояние одного хода: prefill, счётчики ошибок (`record_tool_error`/`success`), broken-call ретраи, guard-ретраи подтверждения |
| `RecoveryPolicy` | recovery_policy.py | Детектор повторов (§1.9), предупреждения дублей, наги подтверждения, эскалация восстановления `recover` |
| `SessionStorage` | session_storage.py | Автосохранение (§1.8), ротация файлов `autosave_*.json`, сериализация экстра-данных, сброс диалога `reset_dialog` |
| `set_pending_operation`/`pop_pending_operation` | agent.py | Отложенная операция подтверждения модели (`requires_model_confirmation`) |
| `make_sub_agent` | agent.py | Порождение суб-агента с наследованием контекста и инструментов (§1.2, §2.2) |

Тесты: `tests/test_chat_e2e.py`, `tests/test_resilience_and_loops.py`.

### Сервисы и модули координации (вместо миксинов)

| Файл | Назначение |
|---|---|
| `tool_pipeline.py` | `ToolPipeline`: линейный конвейер вызовов инструментов (разворачивание dispatch, проверка прав, path safety, dry-run подтверждения, вызов воркера, сжатие) |
| `callbacks.py` | `AgentCallbacks`: датакласс функций обратного вызова для стриминга, UI-рендера, подтверждений и логов |
| `simulated_reasoning.py` | `SimReasoningGroup`: двухфазная симуляция рассуждений (`<thinking>`-секция, §2.8), автозакрытие тегов |

Тесты: `tests/test_chat_e2e.py`, `tests/test_compaction_and_memory.py`, `tests/test_tools_and_safety_e2e.py`.

## Модель и история

| Модуль | Назначение | Ключевые понятия | Тесты |
|---|---|---|---|
| `models.py` | Типы сообщений | `Message` (ABC), `SystemMessage` (всегда index 0), `UserMessage` (header-кэш `reset_header_cache`, `is_auxiliary`), `AssistantMessage`, `ToolResult` (`success`/`error`/`user_denied`), `ToolCall`; `to_api_dict` (для LLM) vs `to_persist_dict` (для JSON) | `tests/test_chat_e2e.py` |
| `history.py` | `ChatHistory` | добавить/удалить/`edit_message`/`normalize`/`delete_range`/`replace_range`; per-message summary sidecar; `compress_old_messages`; save/load/`load_from_payload`; позиционная зависимость last-user header | `tests/test_compaction_and_memory.py` |
| `context.py` | `AgentContext` protocol | Интерфейс контекста агента для инструментов | — |
| `context_builder.py` | `prepare_messages_for_api` | KV-cache-безопасная сериализация: шапки user-сообщений (волатильная цифра бюджета только у последнего user), `_check_prefix_hashes` ([PREFIX-HASH] debug) | `tests/test_compaction_and_memory.py` |

## Транспорт и генерация

| Модуль | Назначение | Ключевые понятия | Тесты |
|---|---|---|---|
| `llm_client.py` | Единый стриминговый транспорт | `LLMClient.call` всегда `StreamSession`; `LLMClient.open_stream` всегда возвращает `InProcessStream` (прямой httpx POST, SSE-парсинг руками, отмена через `shutdown` на сокете); `StreamSession.consume` (watchdog-поток для отмены); `StreamAccumulator` (prefill-эхо, маркеры); `TokenUsageTracker`; `LoopDetector` (повтор вызовов инструмента в ходу); `jaccard_similarity`, `text_hash`; stop-маркеры `stopped_at_marker` | `tests/test_streaming_and_ui.py` |
| `generation.py` | Параметры генерации | `GenerationParams` (+overrides), `StructuredOutputConfig` (`from_controller`/`from_prefill`), `apply_stop_markers` | `tests/test_chat_e2e.py` |
| `controllers.py` | Структурированный вывод | `XMLStructureController` — пофазовый вывод по схеме `<root><a/></root>`, автозакрытие тегов, `from_sibling_schema` (для симуляции reasoning) | `tests/test_chat_e2e.py` |
| `tool_parsing.py` | Парсинг вызовов | `tc_name`/`tc_args`, `detect_broken_call`, `parse_tool_args`, `normalize_args` | `tests/test_resilience_and_loops.py` |
| `exceptions.py` | `GenerationInterrupted` | Исключение прерывания генерации | — |

## Инструменты и изоляция

| Модуль | Назначение | Ключевые понятия | Тесты |
|---|---|---|---|
| `tool.py` | `@tool`-декоратор | Генерация JSON Schema из аннотаций, флаги `requires_confirmation`/`requires_model_confirmation`/`path_safety`/`safe_in_trusted`/`self_truncating`, `group`/`hidden` (§1.13), `ToolOutput` | `tests/test_tools_and_safety_e2e.py` |
| `tool_registry.py` | Динамическая загрузка плагинов | `load_external_plugins` (+ кэш `_PLUGIN_CACHE`), `build_tool_dict` (выводит `group`/`is_external`/`hidden`) | `tests/test_chat_e2e.py` (частично) |
| `tool_manager.py` | `ToolManager` | load/unload (пометить), deny, trust_dir, `flush_pending_unloads`, схемы для префикса; hidden-инструменты скрыты и не грузятся | `tests/test_chat_e2e.py` |
| `tool_worker.py` | Единый воркер инструментов | `ToolWorker` (один дочерний процесс для всех инструментов, мгновенный kill по стопу/таймауту, синхронизация CWD) | `tests/test_tools_and_safety_e2e.py` |
| `tools/builtin.py` | Встроенные инструменты | `get_messages`, `edit_message`, `delete_messages`, `delegate_to_subagent`, `load_tool`, `unload_tool`, `make_plan`, `have_done`, `apply_edit` (подтверждение правок), `respond_to_system` (совместимость), `complete_milestone` (вехи) | `tests/test_tools_and_safety_e2e.py`, `tests/test_tools_and_safety_e2e.py` |
| `tools/fs.py` | Файловая система | `read` (периферийное зрение, саммари), `search`, `line_range_edit`/`match_replace_edit` (diff preview с номерами строк целевого файла `L  13: +code`, подтверждение через `apply_edit`), `cwd` | `tests/test_tools_and_safety_e2e.py` |
| `tools/host_shell.py` | Shell | `run_powershell` (legacy `fallback_encoding="oem"`) / `run_bash_host` (git-bash) | `tests/test_tools_and_safety_e2e.py` |
| `tools/sandbox.py` | Docker-песочница | `start_sandbox`/`run_bash`/`run_python`/`stop_sandbox` — `UnifiedDockerAgent` | — |
| `tools/pc_control.py` | Управление ПК | `screenshot`, `mouse_move/click`, `scroll`, `type_text_on_keyboard`, `press_key` | `tests/test_tools_and_safety_e2e.py` |
| `screen_state.py` | Состояние PC-контроля | `last_scale` — шкала координат для pc_control | — |

PRELOADED_TOOLS (main.py): `load_tool`, `read`, `line_range_edit`, `match_replace_edit`, `cwd`, `search`, `run_powershell`, `run_bash_host`, `respond_to_system`, `apply_edit`, `complete_milestone`. Остальное модель грузит сама `load_tool`.

## Сжатие, память, задачи

| Модуль | Назначение | Ключевые понятия | Тесты |
|---|---|---|---|
| `compressors.py` | Пайплайн сжатия | `compact_segment`, `summarize_history_plain` (+review pass), `chunk_and_summarize_large_text`, `auto_compress_tool_result`, `_dense_summarize_message` (per-message) | `tests/test_compaction_and_memory.py`, `tests/test_subagent_and_service.py` |
| `task_tracker.py` | План / вехи / выполнение | `set_plan`, `mark_task_done`, `validate_task_mark_call`, `complete_milestone` (сжатие истории по этапам работы), `compact_completed_tasks` | `tests/test_compaction_and_memory.py` |
| `subprocess_utils.py` | Выполнение shell | `run_capture` (таймаут, прерывание, `_decode_bytes`+fallback), `truncate_middle`, kill tree/job | `tests/test_tools_and_safety_e2e.py` |
| `command_paths.py` | Пути из команд | `extract_paths` — для `path_safety` со shell-команд | `tests/test_tools_and_safety_e2e.py` |
| `project_root.py` | Корень проекта | `find_project_root` (по `.git`), `is_within`, `external_paths` | `tests/test_tools_and_safety_e2e.py` |
| `sub_agent.py` | Суб-агент | `SubAgent` — наследует системный промпт/history/инструменты родителя, свой трекер токенов; `run_subagent_once` | `tests/test_chat_e2e.py` (частично) |

## UI, вход, конфиг

| Модуль | Назначение | Ключевые понятия | Тесты |
|---|---|---|---|
| `ui.py` | Консоль и CLI | `ConsoleUI` (рендер, колбэки стрима), `CLI` (команды `/regen`, `/think_on`, `/stop`, `/trust`, `/compact_history`, `/consistent`, `/milestone` …); мгновенный стоп и инъекция текста | `tests/test_streaming_and_ui.py` |
| `main.py` | Точка входа | `build_allowed_tools`, `parse_args`, запуск CLI | — |
| `rendering.py` | Рендер сообщений | `render_message` | `tests/test_streaming_and_ui.py` |
| `code_extractor.py` | Дамп проекта | `dump_project` — скинуть код проекта одним файлом (для контекста LLM) | — |
| `config.py` | Все константы | `Config` (dataclass): температура/эскалация (`BOOST_TEMP`…), KV-безопасность (`KEEP_REASONING…`), сжатие (`AUTO_SUMMARY_*`, `TASK_COMPACTION_ENABLED`, `MAX_READ_CHARS_PER_CALL`), стриминг/прерывание (`STREAM_ENABLED`), симуляция reasoning (`SIMULATED_REASONING_*`), PC-контроль, autosave | везде через `monkeypatch`/`patch.object(Config, …)` |
| `constants.py` | Маркеры | `ENVIRONMENT_PREFIX`/`ENVIRONMENT_PREFIX_END`, `INTERRUPT_HEADER`, `err()`/`ok()` | — |

## Сквозные механики: какие файлы трогают

| Механика | Файлы |
|---|---|
| KV-cache/стабильный префикс (§1.2) | `models.py` (header-кэш), `context_builder.py`, `history.py` (нормализация, ремувы), `tool_manager.py` (unload не удаляет), `recovery_policy.py` (`ensure_guard_nag`) |
| Компакция/сжатие/вехи (§3) | `compressors.py`, `task_tracker.py` (`complete_milestone`), `history.py` (`compress_old_messages`), `compressors.py` (SummaryService) |
| Guard подтверждения (§1.11) | `tools/builtin.py` (`apply_edit`, `respond_to_system`), `recovery_policy.py` (`detect_duplicate`, `_CONFIRMATION_GUARD_VERDICT`), `turn_executor.py`, `tool_pipeline.py` (dry_run) |
| Детекция повторов/ошибок (§1.9-1.10) | `recovery_policy.py` (`detect_duplicate`, `recover`), `turn_executor.py` (`TurnState`), `llm_client.py` (`jaccard_similarity`, `text_hash`, `LoopDetector`), `recovery_policy.py` (`last_failed_tool_signature`) |
| Изоляция инструментов (§1.13) | `tool_worker.py` (ToolWorker), `tool_pipeline.py`, `subprocess_utils.py` |
| Стриминг/прерывание (§1.12) | `llm_client.py` (`LLMClient.call`, `StreamSession`), `turn_executor.py`, `ui.py` (`_call_chat`, `gen.join()`), `tool_worker.py` (`kill()`) |
| Симуляция reasoning (§2.8) | `simulated_reasoning.py` (`SimReasoningGroup`), `controllers.py` (`from_sibling_schema`), `config.py` (`SIMULATED_REASONING_*`) |
| Чтение и правка файлов (§3.4) | `tools/fs.py` (`read`, `line_range_edit`, `match_replace_edit`), `compressors.py` (`chunk_and_summarize_large_text`) |

## Цепочки вызовов (главные пути)

### 1. Точка входа: CLI

```
main.py: main()
 └─ ConsoleUI(agent), CLI(agent) ───────────────► ui.py
     └─ cli.run()                       [главный цикл]
         ├─ /команды (regen, clear, prefill, save, load, consistent,
         │   think_on, compact_history, milestone, trust/untrust…)
         │   ├─ /stop  → self._request_stop() → agent.request_stop() → tool_worker.kill()
         │   ├─ /milestone → complete_milestone(agent, stage, summary)
         │   └─ /compact_history → agent._auto_summarize_dialogue(force=True) + _report_compaction
         └─ обычное сообщение → self._call_chat(message, prefill)
             ├─ запуск потока-демона: gen ← threading.Thread(_run_chat, args=(message,))
             │    └─ _run_chat: inject_user_interrupt (если interrupt) | agent.chat(message, prefill)
             │         ├─ вход в цикл TurnExecutor.run_turn_loop
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
 ├─ TurnExecutor.prepare_turn(message)
 │    ├─ pop_pending_tool_calls() при висящем assistant-вызове / пустая заглушка assistant
 │    ├─ history.add(UserMessage(message))          [индекс 1+, KV-безопасно]
 │    ├─ per-msg: _maybe_summarize_user_message(user_msg)
 │    ├─ SessionStorage.autosave()                  [§1.8, если autosave=True]
 │    └─ clear_stop()
 └─ TurnExecutor.run_turn_loop(max_iter, prefill, structured_output)
      ├─ state = TurnState(prefill, structured_output)      [prefill, счётчики ошибок/ретраев]
      ├─ SimReasoningGroup(Config.SIMULATED_REASONING_REPAIRS)  [если turn_reasoning=='none']
      └─ for _ in range(max_iter):
           ├─ pending_interrupt / stop_event? → finish_turn("")
           ├─ history.repair_tail()
           ├─ all_messages = prepare_messages_for_api(agent)
           ├─ call_with_retries(all_messages, step_prefill, markers) ──► LLMClient.call / _call_with_streaming
           │    └─ loop до 3 попыток при дублях:
           │         ├─ RecoveryPolicy.detect_duplicate(msg)
           │         │    ├─ повтор вызова того же инструмента с теми же аргументами
           │         │    ├─ повтор текста / reasoning (consecutive_repeats >= required)
           │         │    └─ Jaccard сходство ответа >= threshold
           │         ├─ дубль найден → boost BOOST_TEMP + duplicate_system_warning
           │         └─ исчерпаны попытки → give up к пользователю
           ├─ _process_llm_response(message_obj)
           │    ├─ NO COMMENT проверка: вызов инструмента без текста/рассуждений? → rerun с prefill
           │    ├─ history.add(assistant_msg) + on_render
           │    ├─ ExecuteMixin._execute_tools(tool_calls)
           │    │    └─ для каждого ToolCall:
           │    │         ├─ requires_model_confirmation (edit_file / match_replace)?
           │    │         │    ├─ ToolWorker.call(name, args, dry_run=True)
           │    │         │    ├─ set_pending_operation({"resolve": resolve, ...})
           │    │         │    └─ ToolResult = preview + ask → ждём apply_edit / respond_to_system
           │    │         ├─ внешний инструмент?
           │    │         │    └─ ToolWorker.call(name, args, dry_run=False) в изолированном дочернем процессе
           │    │         └─ ToolResult.success / error / user_denied
           │    └─ history.extend(tool_results)
           ├─ pending confirmation и модель ответила текстом вместо apply_edit?
           │    ├─ erase_last_assistant()
           │    ├─ RecoveryPolicy.ensure_guard_nag() (You can't continue...)
           │    └─ retry answer guard
           ├─ broken_call / tool_error? → RecoveryPolicy.recover()
           ├─ complete_milestone вызван? → сжатие промежуточных сообщений этапа
           ├─ порог контекста превышен? → _auto_summarize_dialogue()
           └─ нет tool_calls? → finish_turn(result_text)
```
