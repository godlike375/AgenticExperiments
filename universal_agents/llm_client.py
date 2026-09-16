import hashlib
import threading
from typing import Optional, Callable
from types import SimpleNamespace
from openai import OpenAI
from universal_agents.config import Config, CHARS_PER_TOKEN
from universal_agents.generation import GenerationParams, apply_stop_markers
from universal_agents.tool_parsing import normalize_args, build_tool_calls
from universal_agents.tools.builtin import make_plan as _make_plan_tool
from universal_agents.exceptions import GenerationInterrupted


def jaccard_similarity(a: str, b: str) -> float:
    """Доля пересечения множеств слов (Jaccard) двух текстов: 1.0 при равенстве, 0.0 без общих слов."""
    set_a = set(a.split())
    set_b = set(b.split())
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def text_hash(text: str) -> str:
    """SHA-256 хэш очищенного (по краям) текста. Единая «память повторов»: если хэш
    нового ответа ассистента совпал с хэшем сообщения из истории (текста или
    reasoning-блока) — это повтор, независимо от того, сколько итераций назад он был."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def build_usage_dict(prompt_tokens: int, completion_tokens: int, total_tokens: Optional[int] = None) -> dict:
    """Собирает usage-словарь в едином формате для агента."""
    if total_tokens is None:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def apply_prefill(content: Optional[str], prefill: Optional[str]) -> str:
    """Добавляет prefill в начало содержимого, если его там ещё нет."""
    if prefill is None:
        return content or ""
    content = content or ""
    if content.startswith(prefill):
        return content
    return prefill + content


class TokenUsageTracker:
    def __init__(self, system_prompt: str, max_context_tokens: int = 8192):
        self.max_context_tokens = max_context_tokens
        self.last_usage = None
        self.system_prompt = system_prompt

    def update_from_usage(self, usage: dict):
        self.last_usage = usage

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Грубая оценка токенов: символы / CHARS_PER_TOKEN"""
        return int(len(text) / CHARS_PER_TOKEN)

    def get_total_context_tokens(self, first_system_message: str = "", last_user_content: str = "") -> int:
        known = self.estimate_tokens(first_system_message)
        if self.last_usage:
            known = self.last_usage.get("prompt_tokens", 0)
        if last_user_content:
            known += self.estimate_tokens(last_user_content)
        return known

    def get_remaining(self, last_user_content: str = ""):
        total = self.get_total_context_tokens(self.system_prompt, last_user_content)
        remaining = self.max_context_tokens - total
        return remaining

    def format_user_token_info(self) -> str:
        """Информация о токенах для пользователя: «Context size» из total_tokens последнего вызова, «Remaining» — из prompt_tokens; неизвестное заранее оценивается через CHARS_PER_TOKEN."""
        if not self.last_usage:
            return ""
        spent = self.last_usage.get("total_tokens", 0)
        remaining = self.max_context_tokens - (self.last_usage.get("prompt_tokens", 0) or 0)
        return f"Context size: {spent} (Remaining: {remaining})"

class LoopDetector:
    def __init__(self):
        self.threshold = 1

    @staticmethod
    def normalize_args(args_str: str) -> str:
        """Канонизирует аргументы для сравнения на дубликаты (см. tool_parsing.normalize_args)."""
        return normalize_args(args_str)

    def check_duplicate_in_turn(self, tool_name: str, arguments: str, messages: list) -> bool:
        """Проверяет повторный вызов того же инструмента с теми же аргументами после последнего сообщения пользователя. Ошибочные/отклонённые вызовы дубликатами не считаются."""
        from universal_agents.models import UserMessage, AssistantMessage, ToolResult
        norm_args = self.normalize_args(arguments)
        failed_call_ids = set()

        # Идем с конца истории сообщений
        for msg in reversed(messages):
            # Дошли до сообщения пользователя — текущий ход начался здесь; всё до него не считаем.
            if isinstance(msg, UserMessage):
                break

            if isinstance(msg, ToolResult):
                if msg.is_error or msg.is_user_denied:
                    failed_call_ids.add(msg.tool_call_id)
                continue

            if isinstance(msg, AssistantMessage):
                # make_plan особый случай: повтор с теми же аргументами = зацикливание, с другими = ревизия плана (вызовы после него не повтор).
                found_plan = False
                for tc in msg.tool_calls:
                    if getattr(tc, "name", "") == _make_plan_tool.__name__:
                        found_plan = True
                        if self.normalize_args(tc.arguments) == norm_args:
                            return True
                if found_plan:
                    break
                for tc in msg.tool_calls:
                    if tc.name == tool_name:
                        # вызов, упавший с ошибкой/отклонением, не является дубликатом
                        if tc.id in failed_call_ids:
                            continue
                        if self.normalize_args(tc.arguments) == norm_args:
                            return True
                # Проверяем только самый последний AssistantMessage: если между
                # двумя одинаковыми вызовами был другой инструмент (прогресс),
                # это не зацикливание.
                break
        return False

class StreamAccumulator:
    """Собирает из чанков стрима финальный ответ (текст, reasoning, tool calls, usage); единая логика для диалога и служебных вызовов."""

    def __init__(self, prefill=None, on_stream_chunk=None,
                 on_reasoning_start=None, on_reasoning_chunk=None):
        self.content = ""
        self.reasoning = ""
        self.tool_calls_data: dict = {}
        self.usage = None
        self.reasoning_started = False
        self._prefill_pending = prefill
        self.on_stream_chunk = on_stream_chunk
        self.on_reasoning_start = on_reasoning_start
        self.on_reasoning_chunk = on_reasoning_chunk

    def process(self, chunk) -> str:
        """Применяет чанк (usage, reasoning, текст, tool calls); возвращает text-delta."""
        usage = getattr(chunk, 'usage', None)
        if usage:
            self.usage = build_usage_dict(
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens,
            )
        if not getattr(chunk, 'choices', None):
            return ""
        delta = chunk.choices[0].delta

        rc = getattr(delta, 'reasoning_content', None)
        if rc:
            self.reasoning += rc
            if not self.reasoning_started:
                self.reasoning_started = True
                if self.on_reasoning_start:
                    self.on_reasoning_start()
            if self.on_reasoning_chunk:
                self.on_reasoning_chunk(rc)

        added = ""
        if delta.content:
            if self._prefill_pending:
                if self.on_stream_chunk:
                    self.on_stream_chunk(self._prefill_pending)
                self._prefill_pending = None
            added = delta.content
            self.content += added
            if self.on_stream_chunk:
                self.on_stream_chunk(added)

        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in self.tool_calls_data:
                    self.tool_calls_data[idx] = {
                        "id": tc.id or "",
                        "type": "function",
                        "function": {
                            "name": tc.function.name if tc.function and tc.function.name else "",
                            "arguments": tc.function.arguments if tc.function and tc.function.arguments else ""
                        }
                    }
                else:
                    if tc.id:
                        self.tool_calls_data[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            self.tool_calls_data[idx]["function"]["name"] += tc.function.name
                        if tc.function.arguments:
                            self.tool_calls_data[idx]["function"]["arguments"] += tc.function.arguments
        return added

    def build_message(self, prefill=None):
        return SimpleNamespace(
            content=apply_prefill(self.content, prefill),
            tool_calls=build_tool_calls(self.tool_calls_data) or None,
            reasoning_content=self.reasoning or None,
        )


class StreamSession:
    """Единый блокирующий потребитель стрима. Создаёт raw-соединение через
    LLMClient.stream и пожирает его до конца/остановки, наполняя StreamAccumulator.
    Watchdog и чанковый цикл — в одном месте (единая точка для обоих потребителей:
    LLMClient.call и StreamingMixin._call_with_streaming)."""

    def __init__(self, messages, temp=None, timeout=None, tools=None, prefill=None,
                 top_p=None, frequency_penalty=None, presence_penalty=None,
                 max_tokens=None, params=None, reasoning_effort="none",
                 on_stream_chunk=None, on_reasoning_start=None, on_reasoning_chunk=None):
        self.acc = StreamAccumulator(
            prefill=prefill,
            on_stream_chunk=on_stream_chunk,
            on_reasoning_start=on_reasoning_start,
            on_reasoning_chunk=on_reasoning_chunk,
        )
        self._raw = LLMClient.stream(
            messages,
            temp=temp, timeout=timeout, tools=tools, prefill=prefill,
            top_p=top_p, frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty, max_tokens=max_tokens,
            params=params,
            reasoning_effort=reasoning_effort,
        )
        self.stopped_at_marker = False

    def consume(self, stop_check=None, stop_on_chunk=None, on_stream_start=None,
                stop_markers: tuple = ()):
        """Блокирующе потребляет стрим.

        stop_check — пользовательская остановка (watchdog закрывает соединение, ловит
        ошибку; в цикле — закрывает соединение и break).
        stop_on_chunk — внутренний критерий (напр. расхождение), вызывается ПОСЛЕ
        каждого обработанного чанка; True прерывает потребление (stream НЕ закрывается
        намеренно — watcher закроет через _watch_done).
        on_stream_start — вызывается ДО первого чанка (единый порядок; старый
        StreamingMixin запускал UI-отрисовку позже, но это безопасная унификация).

        stop_markers — стоп-маркеры структурированного вывода. После каждого чанка
        проверяется накопленный контент: если какой-то маркер появился, контент
        обрезается до этого вхождения включительно (см. apply_stop_markers), стрим
        закрывается и self.stopped_at_marker становится True. «Хвост» после маркера
        (лишний анализ модели) отбрасывается; следующая фаза генерации запускается
        вызывающим через новый prefill.

        Возвращает (error, stopped):
        - error непуст → сбой соединения или создания (stop_on_chunk не достигнут).
        - stopped=True → прерывание по stop_check, stop_on_chunk или stop_markers (error пуст).

        Колбэки жизненного цикла после потребления (on_stream_end, on_reasoning_end) —
        контракт вызывающего (единый для обоих потребителей).

        Watchdog стартует ДО первого чанка: первый next() может блокироваться на время
        префилла длинного контекста/сетевой задержки, и без него запрос остановки
        пользователя ('q') некому обработать. Стриму с ошибкой создания (error-generator)
        ранний watcher не мешает: тот не управляет реальным HTTP-соединением, а при
        запрошенной остановке подавление его ошибки и не нужно.
        """
        error = ""
        stopped = False
        _watch_done = threading.Event()
        try:
            # Стоп уже запрошен до начала потребления — не начинаем блокирующий цикл.
            if stop_check and stop_check():
                return "stopped", True
            if on_stream_start:
                on_stream_start()
            # Watchdog: закрывает соединение при остановке пользователя. Стартует ДО
            # первого чанка, чтобы прервать долгий префилл/сетевую задержку первого
            # next(). Закрытие стрима подавит и genuine-ошибку, и error-generator — но
            # оба раза это происходит только когда пользователь сам запросил остановку.
            if stop_check is not None:
                def _watcher():
                    while not _watch_done.is_set():
                        if stop_check():
                            LLMClient.close_stream(self._raw)
                            break
                        _watch_done.wait(0.05)
                threading.Thread(target=_watcher, daemon=True).start()
            first = next(self._raw, None)
            if first is None:
                return "empty stream", False
            if isinstance(first, dict) and "error" in first:
                return f"stream creation failed: {first['error']}", False
            self.acc.process(first)
            # Проверка стоп-маркера сразу после первого чанка: если маркер уже появился
            # (единственный чанк), обрезаем и выходим — иначе check только в цикле ниже.
            if stop_markers:
                content, marker_hit = apply_stop_markers(self.acc.content, stop_markers)
                if marker_hit:
                    self.acc.content = content
                    LLMClient.close_stream(self._raw)
                    self.stopped_at_marker = True
                    stopped = True
            if not stopped:
                for chunk in self._raw:
                    self.acc.process(chunk)
                    if stop_check and stop_check():
                        LLMClient.close_stream(self._raw)
                        stopped = True
                        break
                    if stop_on_chunk and stop_on_chunk():
                        stopped = True
                        break
                    if stop_markers:
                        content, marker_hit = apply_stop_markers(self.acc.content, stop_markers)
                        if marker_hit:
                            self.acc.content = content
                            LLMClient.close_stream(self._raw)
                            self.stopped_at_marker = True
                            stopped = True
                            break
        except Exception as e:
            if isinstance(e, GenerationInterrupted):
                raise
            error = str(e)
        finally:
            _watch_done.set()
        return error, stopped


class LLMClient:
    _client = None

    @classmethod
    def get_client(cls) -> OpenAI:
        if cls._client is None:
            cls._client = OpenAI(api_key="lm-studio", base_url=Config.API_URL)
        return cls._client

    @classmethod
    def close_stream(cls, stream) -> None:
        """Закрывает конкретное соединение стрима. Watchdog'и закрывают ИМЕННО свой
        стрим — чтобы отмена одного запроса не рубила чужой параллельный стрим."""
        if stream is None:
            return
        try:
            if hasattr(stream, "close"):
                stream.close()
            elif hasattr(stream, "_response") and hasattr(stream._response, "close"):
                stream._response.close()
            elif hasattr(stream, "response") and hasattr(stream.response, "close"):
                stream.response.close()
        except Exception:
            pass

    @staticmethod
    def call(
        messages: list[dict],
        temp: float = None,
        timeout: int = None,
        tools: list[dict] = None,
        prefill: str = None,
        top_p: float = None,
        frequency_penalty: float = None,
        presence_penalty: float = None,
        max_tokens: int = None,
        params: GenerationParams = None,
        callbacks: Optional[dict] = None,
        stop_check: Optional[Callable[[], bool]] = None,
        reasoning_effort: str = "none",
        stop_markers: tuple = (),
    ):
        """Единая точка обращения к LLM. При заданных стриминговых колбэках и STREAM_ENABLED идёт через стриминг; иначе — обычный chat.completions вызов."""
        temp, timeout, top_p, frequency_penalty, presence_penalty, max_tokens = LLMClient._resolve_params(
            params, temp, timeout, top_p, frequency_penalty, presence_penalty, max_tokens
        )

        messages_to_send = LLMClient._prepare_messages(messages, prefill)

        cb = callbacks or {}
        want_stream = (
            Config.STREAM_ENABLED
            and bool(cb.get("on_stream_chunk") or cb.get("on_reasoning_chunk"))
        )
        if want_stream:
            session = StreamSession(
                messages_to_send, temp=temp, timeout=timeout, tools=tools,
                prefill=prefill, top_p=top_p, frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty, max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                on_stream_chunk=cb.get("on_stream_chunk"),
                on_reasoning_start=cb.get("on_reasoning_start"),
                on_reasoning_chunk=cb.get("on_reasoning_chunk"),
            )
            error, stopped = session.consume(
                stop_check=stop_check,
                on_stream_start=cb.get("on_stream_start"),
                stop_markers=stop_markers,
            )
            end_cb = cb.get("on_stream_end")
            reasoning_end_cb = cb.get("on_reasoning_end")
            if error:
                if end_cb:
                    end_cb()
                if session.acc.reasoning_started and reasoning_end_cb:
                    reasoning_end_cb()
                if stop_check and stop_check():
                    return None, "stopped: stream unavailable and stop requested", None
                return None, error, None
            if end_cb:
                end_cb()
            if session.acc.reasoning_started and reasoning_end_cb:
                reasoning_end_cb()
            return session.acc.build_message(prefill), None, session.acc.usage

        result = None
        if result is None:
            # Не-стриминговый путь: не начинаем блокирующий вызов, если пользователь
            # уже запросил остановку (такой вызов прервать невозможно).
            if stop_check and stop_check():
                return None, "stopped", None
            result = LLMClient._call_chat_completions(
                messages_to_send, temp, timeout, tools, prefill, top_p,
                frequency_penalty, presence_penalty, max_tokens,
                reasoning_effort=reasoning_effort,
            )

        if stop_markers and result and result[0] is not None and getattr(result[0], "content", None):
            # Пост-фактум обрезка по маркеру для не-стримингового пути: чанков нет,
            # поэтому останавливать нечего — просто отбрасываем «хвост» после маркера.
            content, _ = apply_stop_markers(result[0].content, stop_markers)
            result[0].content = content

        LLMClient._debug_log(messages_to_send, result)
        return result

    @staticmethod
    def _prepare_messages(messages: list[dict], prefill: Optional[str]) -> list[dict]:
        messages_to_send = list(messages)
        if prefill:
            messages_to_send.append({"role": "assistant", "content": prefill})
        return messages_to_send

    @staticmethod
    def _debug_log(messages_to_send, result):
        """Выводит в лог экрана содержимое служебного вызова LLM (для отладки)."""
        try:
            print("\n" + "=" * 30)
            print("📤 LLM DEBUG CALL INPUT:")

            print(messages_to_send[-1])

            print("📤 LLM DEBUG CALL OUTPUT:")

            msg, err, usage = result
            if err:
                print(f"  ⚠️ ERROR: {err}")
            else:
                if msg is not None and getattr(msg, "content", None):
                    print(msg.content)
                elif msg is not None and getattr(msg, "tool_calls", None):
                    for tc in msg.tool_calls:
                        print(f"  🔨 {getattr(tc, 'name', '?')}({getattr(tc, 'arguments', '')})")
                else:
                    print("  (no content / empty)")
                if usage:
                    print(f"  ⏱ usage: {usage}")
            print("=" * 30)
        except Exception:
            pass

    @staticmethod
    def _resolve_params(params, temp, timeout, top_p, frequency_penalty, presence_penalty, max_tokens):
        if params is not None:
            p = params.resolved()
            temp = p.temp if temp is None else temp
            timeout = p.timeout if timeout is None else timeout
            top_p = p.top_p if top_p is None else top_p
            frequency_penalty = p.frequency_penalty if frequency_penalty is None else frequency_penalty
            presence_penalty = p.presence_penalty if presence_penalty is None else presence_penalty
            max_tokens = p.max_tokens if max_tokens is None else max_tokens
        return temp, timeout, top_p, frequency_penalty, presence_penalty, max_tokens

    @staticmethod
    def _chat_kwargs(temp, timeout, tools, top_p, frequency_penalty, presence_penalty, max_tokens,
                     reasoning_effort="none") -> dict:
        """Общий конструктор параметров chat.completions (для обычного вызова и стрима)."""
        return {
            "model": Config.MODEL_NAME,
            "temperature": temp if temp is not None else Config.TEMP,
            "max_tokens": max_tokens if max_tokens is not None else Config.MAX_OUTPUT_TOKENS,
            "tools": tools,
            "parallel_tool_calls": False,
            "timeout": timeout if timeout is not None else Config.TIMEOUT,
            "reasoning_effort": reasoning_effort,
            "frequency_penalty": frequency_penalty if frequency_penalty is not None else Config.FREQUENCY_PENALTY,
            "presence_penalty": presence_penalty if presence_penalty is not None else Config.PRESENCE_PENALTY,
            "top_p": top_p if top_p is not None else Config.TOP_P,
        }

    @staticmethod
    def _call_chat_completions(messages_to_send, temp, timeout, tools, prefill, top_p,
                               frequency_penalty, presence_penalty, max_tokens,
                               reasoning_effort="none"):
        try:
            response = LLMClient.get_client().chat.completions.create(
                messages=messages_to_send,
                **LLMClient._chat_kwargs(temp, timeout, tools, top_p,
                                         frequency_penalty, presence_penalty, max_tokens,
                                         reasoning_effort=reasoning_effort),
            )
            msg = response.choices[0].message
            msg.content = apply_prefill(msg.content, prefill)

            usage = None
            if hasattr(response, 'usage') and response.usage:
                usage = build_usage_dict(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                    response.usage.total_tokens,
                )
            return msg, None, usage
        except Exception as e:
            return None, str(e), None

    @staticmethod
    def stream(
        messages: list[dict],
        temp: float = None,
        timeout: int = None,
        tools: list[dict] = None,
        prefill: str = None,
        top_p: float = None,
        frequency_penalty: float = None,
        presence_penalty: float = None,
        max_tokens: int = None,
        params: GenerationParams = None,
        reasoning_effort: str = "none",
    ):
        """Streaming version of call() - returns generator of chunks.
        """
        temp, timeout, top_p, frequency_penalty, presence_penalty, max_tokens = LLMClient._resolve_params(
            params, temp, timeout, top_p, frequency_penalty, presence_penalty, max_tokens
        )

        messages_to_send = LLMClient._prepare_messages(messages, prefill)

        try:
            stream = LLMClient.get_client().chat.completions.create(
                messages=messages_to_send,
                **LLMClient._chat_kwargs(temp, timeout, tools, top_p,
                                         frequency_penalty, presence_penalty, max_tokens,
                                         reasoning_effort=reasoning_effort),
                stream=True,
                stream_options={"include_usage": True},
            )
            return stream
        except Exception as e:
            # Для streaming ошибки возвращаем генератор с ошибкой
            def error_generator(err=str(e)):
                yield {"error": err}
            return error_generator()
