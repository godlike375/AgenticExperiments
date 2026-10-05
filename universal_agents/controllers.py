"""Контроллеры структурированного вывода: абстрактный интерфейс + XML-контроллер.

Контроллер — активный компонент, который во время генерации следит за потоком
тегов модели и сверяет его с ожидаемой структурой (схемой). При расхождении он
стирает ошибочный фрагмент из вывода и подсказывает модели следующий нужный тег
через prefill. В отличие от статичной цепочки фаз (next_prefills), контроллер
принимает решения по фактическому содержанию ответа на каждом шаге.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol


class ControllerVerdict:
    """Результат обработки одной фазы генерации контроллером."""

    __slots__ = ("corrected_content", "next_prefill", "complete")

    def __init__(
        self,
        corrected_content: str,
        next_prefill: Optional[str] = None,
        complete: bool = False,
    ) -> None:
        # Контент фазы после стирания ошибочного тега (пойдёт в историю).
        self.corrected_content = corrected_content
        # Следующий нужный тег как prefill следующей фазы (None = модель продолжает сама).
        self.next_prefill = next_prefill
        # True, когда вся схема корректно завершена (закрыт корень).
        self.complete = complete


class PhaseController(Protocol):
    """Интерфейс контроллера структурированного вывода.

    Агентский цикл вызывает advance() после каждой фазы генерации (когда стрим
    остановился по стоп-маркеру) и по его вердикту правит историю и продолжает
    генерацию либо завершает ход.
    """

    def initial_prefill(self) -> Optional[str]:
        """Первый тег/структура, который должен быть задан как prefill."""
        ...

    def markers(self) -> tuple[str, ...]:
        """Стоп-маркеры текущей фазы (закрывающие теги схемы)."""
        ...

    def advance(self, content: str) -> ControllerVerdict:
        """Обрабатывает контент очередной фазы: возвращает исправленный контент,
        следующий prefill и признак завершения."""
        ...

    def document(self) -> str:
        """Собранный итоговый документ (конкатенация корректных фаз)."""
        ...


# ──────────────────────────────────────────────────────────────────────────
# XML-контроллер
# ──────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class XNode:
    """Узел XML-схемы: имя тега и вложенные узлы (без атрибутов, без повторов)."""

    tag: str
    children: tuple["XNode", ...] = ()


@dataclass
class XMLStructureController:
    """Контроллер генерации XML-подобной структуры по схеме.

    Схема — дерево вложенных тегов без атрибутов (XNode). Контроллер разворачивает
    её в последовательность ожидаемых токенов (открывающие и закрывающие теги в
    порядке обхода DFS) и шагает по ним, сравнивая с тем, что пишет модель.

    Логика:
      - initial_prefill() возвращает первый тег (<root> и т.п.).
      - markers() — закрывающие теги всех тегов схемы: любой неожиданный закрывающий
        останавливает генерацию, и контроллер проверяет, тот ли тег закрыт.
      - advance(content): разбирает контент фазы на теги и сверяет с ожиданием.
        Совпадение → шагаем дальше. Несовпадение → «стираем» ошибочный токен из
        контента (обрезаем до начала токена) и возвращаем ожидаемый токен как
        следующий prefill, чтобы модель продолжила с правильного места.
      - Когда обход дошёл до конца (корень закрыт корректно) — complete=True,
        document() возвращает собранный документ.

    Текст между тегами (контент элементов) на сравнение не влияет — он принимается
    как есть. При нескольких несоответствиях в одной фазе стирается всё начиная с
    первого ошибочного токена (модель перегенерирует хвост с правильного места).

    Две формы схемы: from_schema_text — один корень с вложенными детьми;
    from_sibling_schema — сестринские корневые секции подряд, без обёртки (тогда
    root указывает на первую секцию, а токены строятся по всем siblings).
    """

    # Корень схемы (None у сестринской схемы — см. siblings).
    root: Optional[XNode] = None
    # Произвольный текст сразу после открывающего тега корня (вставляется в prefill,
    # чтобы направить модель: например "<content_structure>\nL" — сигнал начать
    # содержимое с номеров строк). Этот текст пишет модель как часть фазы, на
    # сравнение с токенами он не влияет.
    prefill_suffix: str = ""
    # Схема из нескольких сестринских корневых секций ('<a/><b/>' → <a>…</a> сразу
    # затем <b>…</b>, без обёртки-корня). Задаётся только через from_sibling_schema.
    siblings: tuple[XNode, ...] = ()
    # Плоская последовательность ожидаемых токенов в порядке DFS.
    _tokens: tuple[str, ...] = field(default=(), init=False, repr=False)
    # Текущая позиция в последовательности.
    _cursor: int = field(default=0, init=False, repr=False)
    # Накопленный корректный документ.
    _document: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        nodes = self.siblings if self.siblings else ((self.root,) if self.root is not None else ())
        tokens: list[str] = []
        for node in nodes:
            tokens.extend(XMLStructureController._flatten(node))
        if not tokens:
            raise ValueError("XML structure must contain at least the root tag")
        self._tokens = tuple(tokens)
        if self.root is None:
            self.root = nodes[0]

    @staticmethod
    def _flatten(node: XNode) -> tuple[str, ...]:
        tokens = [f"<{node.tag}>"]
        for child in node.children:
            tokens.extend(XMLStructureController._flatten(child))
        tokens.append(f"</{node.tag}>")
        return tuple(tokens)

    @classmethod
    def from_schema_text(
        cls, text: str, prefill_suffix: str = ""
    ) -> "XMLStructureController":
        """Собирает контроллер из текстовой схемы с вложенными тегами без атрибутов.

        Пример:
            <root>
              <a>
                <b/>
              </a>
              <c/>
            </root>
        Поддерживаются самозакрывающиеся теги (<b/>) как листовые элементы и
        обычные пары <x>...</x>. Текст между тегами игнорируется.
        prefill_suffix — текст, добавляемый в initial_prefill() после открывающего
        тега корня (см. описание поля).
        """
        root = cls._parse_tree(text)
        return cls(root=root, prefill_suffix=prefill_suffix)

    @classmethod
    def from_sibling_schema(
        cls, text: str, prefill_suffix: str = ""
    ) -> "XMLStructureController":
        """Собирает контроллер из схемы с СЕСТРИНСКИМИ корневыми секциями (без обёртки).

        Пример:
            <alpha/><beta/>
        Означает: сначала заполняется <alpha>…</alpha>, затем <beta>…</beta> — токены идут
        подряд, response начинается с первого тега. Отличие от from_schema_text (там всё
        вложено в один корень, и <root> попадает в начало ответа). Симуляция reasoning
        (§ SIMULATED_REASONING_*) этот API больше не использует: там одна секция и один
        вызов LLM без стоп-маркеров.
        """
        roots = cls._parse_roots(text)
        if len(roots) == 1:
            return cls(root=roots[0], prefill_suffix=prefill_suffix)
        return cls(siblings=tuple(roots), prefill_suffix=prefill_suffix)

    @staticmethod
    def _tokenize(text: str) -> list[tuple[str, str]]:
        """Возвращает список (kind, name): kind в {'open', 'close'}."""
        tokens: list[tuple[str, str]] = []
        i = 0
        n = len(text)
        while i < n:
            if text[i] == "<":
                j = text.find(">", i)
                if j < 0:
                    break
                raw = text[i + 1:j].strip()
                if raw:
                    if raw.startswith("/"):
                        name = raw[1:].strip()
                        if name:
                            tokens.append(("close", name))
                    elif raw.endswith("/"):
                        name = raw[:-1].strip()
                        if name:
                            tokens.append(("open", name))
                            tokens.append(("close", name))
                    else:
                        name = raw.strip()
                        if name and all(ch.isalnum() or ch in "._-:" for ch in name):
                            tokens.append(("open", name))
                i = j + 1
            else:
                i += 1
        return tokens

    @classmethod
    def _parse_roots(cls, text: str) -> list[XNode]:
        """Разбирает текстовую схему в список корневых узлов (обычно ровно один)."""
        tokens = cls._tokenize(text)
        stack: list[tuple[str, list[XNode]]] = []
        roots: list[XNode] = []
        for kind, name in tokens:
            if kind == "open":
                stack.append((name, []))
            elif kind == "close":
                if not stack:
                    raise ValueError(f"Unexpected closing tag '</{name}>'")
                tag, children = stack.pop()
                if tag != name:
                    raise ValueError(f"Mismatched closing tag '</{name}>' for '<{tag}>'")
                node = XNode(tag=tag, children=tuple(children))
                if stack:
                    stack[-1][1].append(node)
                else:
                    roots.append(node)
        if stack:
            raise ValueError(f"Unclosed tag(s): {[t for t, _ in stack]}")
        if not roots:
            raise ValueError("Empty schema")
        return roots

    @classmethod
    def _parse_tree(cls, text: str) -> XNode:
        """Разбирает схему с ровно одним корнем (несколько корней — ошибка)."""
        roots = cls._parse_roots(text)
        if len(roots) > 1:
            raise ValueError("Multiple root tags are not supported")
        return roots[0]

    # ── интерфейс PhaseController ─────────────────────────────────────────

    def initial_prefill(self) -> Optional[str]:
        # Первый opening-токен схемы: у вложенной схемы это <root>, у сестринской —
        # тег первой секции (обёртки нет, ответ начинается с него).
        return f"{self._tokens[0]}{self.prefill_suffix}"

    def markers(self) -> tuple[str, ...]:
        return tuple(f"</{name}>" for name in self._tag_names())

    def _tag_names(self) -> list[str]:
        return [tok.split("<")[1].rsplit(">", 1)[0] for tok in self._tokens if tok.startswith("<") and not tok.startswith("</")]

    def advance(self, content: str) -> ControllerVerdict:
        # Токены контента фазы с позициями в исходной строке.
        positions: list[tuple[tuple[str, str], int]] = []
        i = 0
        n = len(content)
        while i < n:
            if content[i] == "<":
                j = content.find(">", i)
                if j < 0:
                    break
                raw = content[i + 1:j].strip()
                if raw:
                    if raw.startswith("/"):
                        name = raw[1:].strip()
                        if name:
                            positions.append((("close", name), i))
                    elif raw.endswith("/"):
                        name = raw[:-1].strip()
                        if name:
                            positions.append((("open", name), i))
                            positions.append((("close", name), i))
                    else:
                        name = raw.strip()
                        if name and all(ch.isalnum() or ch in "._-:" for ch in name):
                            positions.append((("open", name), i))
                i = j + 1
            else:
                i += 1

        cursor = self._cursor
        tokens = self._tokens
        for idx, ((kind, name), pos) in enumerate(positions):
            if cursor >= len(tokens):
                # Модель пишет больше, чем нужно: стираем лишнее и завершаем документ.
                corrected = content[:pos]
                self._document += corrected
                return ControllerVerdict(corrected, None, True)
            expected = tokens[cursor]
            expected_kind, expected_name = XMLStructureController._parse_tok(expected)
            if kind == expected_kind and name == expected_name:
                cursor += 1
                continue
            # Несовпадение: стираем всё, начиная с ошибочного токена, и подсказываем
            # ожидаемый токен как следующий prefill. Согласованный префикс до ошибки
            # фиксируется в курсоре, чтобы следующая фаза продолжила с правильного места.
            corrected = content[:pos]
            self._cursor = cursor
            self._document += corrected
            return ControllerVerdict(corrected, expected, False)

        # Всё совпало до конца контента фазы. Если курсор в конце (вся схема пройдена),
        # обрезаем content до конца последнего совпавшего тега (убираем хвостовый текст).
        if cursor >= len(tokens) and positions:
            last_tag_start = positions[-1][1]
            end = content.find(">", last_tag_start)
            trimmed = content[: end + 1] if end >= 0 else content
            self._document += trimmed
            self._cursor = cursor
            return ControllerVerdict(trimmed, None, True)
        self._document += content
        self._cursor = cursor
        if cursor >= len(tokens):
            return ControllerVerdict(content, None, True)

        # Пустой контент (модель ничего не написала) не даёт повода ни закрывать,
        # ни подсказывать следующий тег — оставляем прежний путь (empty → None).
        if not content.strip():
            return ControllerVerdict(content, None, False)

        # Автозакрытие при досрочном завершении ответа: оставшиеся токены
        # с текущей позиции — это либо закрывающие теги (дописываем их), либо
        #Opening-теги (ставим next_prefill и передаём управление обратно в цикл).
        remaining = self._tokens[cursor:]
        auto_closed: list[str] = []
        closed_count = 0
        for tok in remaining:
            if tok.startswith("</"):
                auto_closed.append(tok)
                closed_count += 1
            else:
                break

        if auto_closed:
            suffix = "".join(auto_closed)
            self._document += suffix
            new_cursor = cursor + closed_count
            self._cursor = new_cursor
            if new_cursor >= len(tokens):
                return ControllerVerdict(content + suffix, None, True)
            # Следующий токен — opening: модель заполнит его на следующей фазе.
            return ControllerVerdict(content + suffix, self._tokens[new_cursor], False)

        # Следующий токен — opening (незаполненный ребёнок): возвращаем prefill.
        return ControllerVerdict(content, self._tokens[cursor], False)

    @staticmethod
    def _parse_tok(tok: str) -> tuple[str, str]:
        if tok.startswith("</"):
            return ("close", tok[2:-1])
        return ("open", tok[1:-1])

    def document(self) -> str:
        return self._document

    def expect_tokens(self, text: str) -> None:
        """Сдвигает курсор по тегам, которые вписаны за модель (вызывающим).

        Текст не проверяется и не авто-закрывается: его писали мы, а модель его ещё не
        «произносила», поэтому advance() для него не годится (он трактует отсутствие
        продолжения как досрочное завершение). Используется фазовой симуляцией
        reasoning: закрывающий тег секции дописывается, когда модель написала вместо
        него мусор. При рассинхроне курсор не двигается — разбираться будет advance()."""
        for kind, name in self._tokenize(text):
            if self._cursor >= len(self._tokens):
                break
            expected_kind, expected_name = XMLStructureController._parse_tok(self._tokens[self._cursor])
            if kind != expected_kind or name != expected_name:
                break
            self._cursor += 1
        self._document += text
