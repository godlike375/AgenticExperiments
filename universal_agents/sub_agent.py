import sys
from dataclasses import asdict
from typing import Optional, Callable, Union, Iterable
from universal_agents.models import AssistantMessage
from universal_agents.rendering import render_message
from universal_agents.llm_client import TokenUsageTracker
from universal_agents.config import Config
from universal_agents.generation import GenerationParams

MAX_SUB_AGENT_DEPTH = 1


def run_subagent_once(agent, prompt: str, temp: float = 0.2, max_iter: int = None) -> str:
    """Запускает субагента для задачи; проверка глубины, полный набор схем родителя (KV-cache safe) и наследование прочитанных файлов (read-skip)."""
    from universal_agents.constants import err, ok

    depth = getattr(agent, '_depth', 0)
    if depth >= MAX_SUB_AGENT_DEPTH:
        return err(
            f" Sub-agent depth limit ({MAX_SUB_AGENT_DEPTH}) reached. "
            f"You can't delegate to sub-agent. Do it yourself."
        )

    sub = agent.make_sub_agent(
        safe_only=False,
        max_iter=max_iter,
        temp=temp,
        on_log=agent.on_system_msg,
        depth=depth + 1,
    )

    agent.on_system_msg(f"[DELEGATE] Starting sub-agent for: {prompt}...")
    result = sub.run(prompt)
    agent.on_system_msg(f"[DELEGATE] Completed. Context size in sub-agent: {sub.tokens_spent}")

    # read-skip: наследуем прочитанные субагентом файлы родителем.
    parent_regs = getattr(agent, '_read_registrations', None)
    if parent_regs is not None:
        for p in sub._agent._read_registrations:
            if p not in parent_regs:
                parent_regs.append(p)

    if not result.strip():
        return err(" Sub-agent returned empty result.")
    return ok(f" Sub-agent result:\n{result}")



class SubAgent:
    """Субагент на базе LLMAgent: наследует системный промпт, историю и полный набор схем родителя (KV-cache safe, denied_tools => ошибка «forbidden»); изолированный трекер токенов; рекурсия блокируется depth check."""

    def __init__(
            self,
            system_prompt: str = 'You are a sub-agent. Always start your answers with "<sub-agent>" tag.',
            max_context_tokens: int = None,
            parent_tools: Optional[dict] = None,
            denied_tools: Union[str, Iterable[str], None] = None,
            max_iter: int = None,
            temp: float = None,
            on_log: Callable[[str], None] = lambda x: None,
            top_p: float = None,
            frequency_penalty: float = None,
            presence_penalty: float = None,
            max_tokens: int = None,
            timeout: int = None,
            parent_history: Optional[list] = None,
            parent_system_prompt: Optional[str] = None,
            depth: int = 0,
            on_stream_chunk: Optional[Callable[[str], None]] = None,
            on_stream_start: Optional[Callable[[], None]] = None,
            on_stream_end: Optional[Callable[[], None]] = None,
            on_reasoning_chunk: Optional[Callable[[str], None]] = None,
            on_reasoning_start: Optional[Callable[[], None]] = None,
            on_reasoning_end: Optional[Callable[[], None]] = None,
            disable_per_msg_summarization: bool = False,
    ):
        self._max_iter = max_iter if max_iter is not None else Config.MAX_ITER
        self._on_log = on_log
        self._depth = depth

        # Если указан системный промпт родителя — используем его для разделения KV-кеша
        effective_system_prompt = parent_system_prompt if parent_system_prompt is not None else system_prompt

        # Полный контекст: без деления бюджета — сервер сам управляет limits
        effective_max_context_tokens = max_context_tokens if max_context_tokens is not None else Config.MAX_CONTEXT_TOKENS
        self._own_tracker = TokenUsageTracker(effective_system_prompt, effective_max_context_tokens)

        def _render_subagent(msg):
            output = render_message(msg, label="[🤖sub-agent]")
            if output:
                on_log(output)

        # Streaming для субагента (по умолчанию — вывод в stdout)
        if on_stream_chunk is None:
            def _sub_stream_start():
                sys.stdout.write("[🤖sub-agent] ")
                sys.stdout.flush()
            def _sub_stream_chunk(chunk):
                sys.stdout.write(chunk)
                sys.stdout.flush()
            def _sub_stream_end():
                sys.stdout.write("\n")
                sys.stdout.flush()
            on_stream_start = _sub_stream_start
            on_stream_chunk = _sub_stream_chunk
            on_stream_end = _sub_stream_end

        if on_reasoning_chunk is None:
            def _sub_reasoning_start():
                sys.stdout.write("[🤖sub-agent] 📝[reasoning] ")
                sys.stdout.flush()
            def _sub_reasoning_chunk(chunk):
                sys.stdout.write(chunk)
                sys.stdout.flush()
            def _sub_reasoning_end():
                sys.stdout.write("\n")
                sys.stdout.flush()
            on_reasoning_start = _sub_reasoning_start
            on_reasoning_chunk = _sub_reasoning_chunk
            on_reasoning_end = _sub_reasoning_end

        params = GenerationParams.from_overrides(
            temp=temp,
            timeout=timeout if timeout is not None else 60,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            max_tokens=max_tokens,
        )

        self._agent = self._build_llm_agent(
            system_prompt=effective_system_prompt,
            parent_tools=parent_tools,
            denied_tools=denied_tools,
            max_context_tokens=effective_max_context_tokens,
            params=params,
            on_render=_render_subagent,
            on_stream_chunk=on_stream_chunk,
            on_stream_start=on_stream_start,
            on_stream_end=on_stream_end,
            on_reasoning_chunk=on_reasoning_chunk,
            on_reasoning_start=on_reasoning_start,
            on_reasoning_end=on_reasoning_end,
            disable_per_msg_summarization=disable_per_msg_summarization,
        )

        # Клонируем историю родителя как префикс для KV-cache reuse
        if parent_history:
            self._agent.history.extend(parent_history)

    def _build_llm_agent(
        self,
        *,
        system_prompt: str,
        parent_tools: Optional[dict],
        denied_tools,
        max_context_tokens: int,
        params: GenerationParams,
        on_render: Callable,
        on_stream_chunk: Callable[[str], None],
        on_stream_start: Callable[[], None],
        on_stream_end: Callable[[], None],
        on_reasoning_chunk: Callable[[str], None],
        on_reasoning_start: Callable[[], None],
        on_reasoning_end: Callable[[], None],
        disable_per_msg_summarization: bool,
    ):
        """Фабрика сборки LLMAgent для субагента: трекер передаётся в конструктор; tools_config=None + external_plugins=parent_tools дают родительский набор схем (префикс совпадает)."""
        from universal_agents.agent import LLMAgent

        agent = LLMAgent(
            system_prompt=system_prompt,
            **asdict(params),
            tools_config=None,
            external_plugins=parent_tools,
            denied_tools=denied_tools,
            on_render=on_render,
            on_confirm=lambda n, a: True,
            on_system_msg=self._on_log,
            max_context_tokens=max_context_tokens,
            max_generation_attempts=1,
            token_tracker=self._own_tracker,
            streaming_enabled=True,
            on_stream_chunk=on_stream_chunk,
            on_stream_start=on_stream_start,
            on_stream_end=on_stream_end,
            on_reasoning_chunk=on_reasoning_chunk,
            on_reasoning_start=on_reasoning_start,
            on_reasoning_end=on_reasoning_end,
            disable_per_msg_summarization=disable_per_msg_summarization,
        )
        agent._depth = self._depth
        agent._is_subagent = True
        return agent

    def run(self, task: str, prefill: str = None) -> str:
        """Выполняет задачу и возвращает финальный ответ субагента.

        Ответ берётся напрямую из возвращаемого значения ``chat`` (оно уже
        содержит финальный текст ассистента, включая prefill) — это надёжнее,
        чем извлечение из истории: история субагента клонирует родительский
        префикс, и при срабатывании внутри ``chat`` auto-summary (компакции)
        сообщения перестраиваются/сдвигаются, из-за чего поиск по истории
        (даже по всей ``history[:]``) может не найти ответ. Поиск по истории
        оставлен только как страховка, если ``chat`` вернул пустоту."""
        result = self._agent.chat(task, max_iter=self._max_iter, prefill=prefill)
        if result and result.strip():
            return result
        msgs = self._agent.history.get_all()
        for msg in reversed(msgs):
            if isinstance(msg, AssistantMessage) and msg.content:
                return msg.content
        return ""

    @property
    def tokens_spent(self) -> int:
        if self._own_tracker.last_usage:
            return self._own_tracker.last_usage.get("total_tokens", 0)
        return 0
