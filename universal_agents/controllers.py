
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol


class ControllerVerdict:

    __slots__ = ("corrected_content", "next_prefill", "complete")

    def __init__(
        self,
        corrected_content: str,
        next_prefill: Optional[str] = None,
        complete: bool = False,
    ) -> None:
        self.corrected_content = corrected_content
        self.next_prefill = next_prefill
        self.complete = complete


class PhaseController(Protocol):

    def initial_prefill(self) -> Optional[str]:
        ...

    def markers(self) -> tuple[str, ...]:
        ...

    def advance(self, content: str) -> ControllerVerdict:
        ...

    def document(self) -> str:
        ...



@dataclass(frozen=True)
class XNode:

    tag: str
    children: tuple["XNode", ...] = ()


@dataclass
class XMLStructureController:
    """Контроллер XML-подобной структуры: сверяет поток тегов со схемой, стирает ошибки, автозакрытие."""

    root: Optional[XNode] = None
    prefill_suffix: str = ""
    siblings: tuple[XNode, ...] = ()
    _tokens: tuple[str, ...] = field(default=(), init=False, repr=False)
    _cursor: int = field(default=0, init=False, repr=False)
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
        root = cls._parse_tree(text)
        return cls(root=root, prefill_suffix=prefill_suffix)

    @classmethod
    def from_sibling_schema(
        cls, text: str, prefill_suffix: str = ""
    ) -> "XMLStructureController":
        """Сестринские корневые секции без обёртки: <alpha/><beta/> → сначала alpha, затем beta."""
        roots = cls._parse_roots(text)
        if len(roots) == 1:
            return cls(root=roots[0], prefill_suffix=prefill_suffix)
        return cls(siblings=tuple(roots), prefill_suffix=prefill_suffix)

    @staticmethod
    def _tokenize(text: str) -> list[tuple[str, str]]:
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
        roots = cls._parse_roots(text)
        if len(roots) > 1:
            raise ValueError("Multiple root tags are not supported")
        return roots[0]


    def initial_prefill(self) -> Optional[str]:
        return f"{self._tokens[0]}{self.prefill_suffix}"

    def markers(self) -> tuple[str, ...]:
        return tuple(f"</{name}>" for name in self._tag_names())

    def _tag_names(self) -> list[str]:
        return [tok.split("<")[1].rsplit(">", 1)[0] for tok in self._tokens if tok.startswith("<") and not tok.startswith("</")]

    def advance(self, content: str) -> ControllerVerdict:
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
                corrected = content[:pos]
                self._document += corrected
                return ControllerVerdict(corrected, None, True)
            expected = tokens[cursor]
            expected_kind, expected_name = XMLStructureController._parse_tok(expected)
            if kind == expected_kind and name == expected_name:
                cursor += 1
                continue
            corrected = content[:pos]
            self._cursor = cursor
            self._document += corrected
            return ControllerVerdict(corrected, expected, False)

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

        if not content.strip():
            return ControllerVerdict(content, None, False)

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
            return ControllerVerdict(content + suffix, self._tokens[new_cursor], False)

        return ControllerVerdict(content, self._tokens[cursor], False)

    @staticmethod
    def _parse_tok(tok: str) -> tuple[str, str]:
        if tok.startswith("</"):
            return ("close", tok[2:-1])
        return ("open", tok[1:-1])

    def document(self) -> str:
        return self._document

    def expect_tokens(self, text: str) -> None:
        for kind, name in self._tokenize(text):
            if self._cursor >= len(self._tokens):
                break
            expected_kind, expected_name = XMLStructureController._parse_tok(self._tokens[self._cursor])
            if kind != expected_kind or name != expected_name:
                break
            self._cursor += 1
        self._document += text
