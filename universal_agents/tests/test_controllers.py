"""Тесты контроллеров: XML-парсер, advance(), интеграция с agent.chat()."""

import unittest
from types import SimpleNamespace
from unittest import mock

from universal_agents.agent import LLMAgent
from universal_agents.controllers import (
    ControllerVerdict,
    PhaseController,
    XMLStructureController,
    XNode,
)
from universal_agents.generation import StructuredOutputConfig
from universal_agents.models import AssistantMessage


# ── Вспомогательные ────────────────────────────────────────────────

def _ctrl(schema: str, max_phases: int = 5) -> StructuredOutputConfig:
    """Фабрика конфига с XML-контроллером из текстовой схемы."""
    return StructuredOutputConfig.from_controller(
        XMLStructureController.from_schema_text(schema),
        max_phases=max_phases,
    )


# ── Парсер ─────────────────────────────────────────────────────────

class TestXMLParser(unittest.TestCase):
    def test_flat_self_closing(self):
        ctrl = XMLStructureController.from_schema_text("<root><a/><b/></root>")
        self.assertEqual(ctrl.root.tag, "root")
        self.assertEqual([c.tag for c in ctrl.root.children], ["a", "b"])
        self.assertEqual(ctrl.initial_prefill(), "<root>")
        self.assertEqual(ctrl._tokens, ("<root>", "<a>", "</a>", "<b>", "</b>", "</root>"))

    def test_nested_structure(self):
        ctrl = XMLStructureController.from_schema_text("<root><a><b/></a></root>")
        self.assertEqual(ctrl.root.children[0].tag, "a")
        self.assertEqual(ctrl.root.children[0].children[0].tag, "b")
        self.assertEqual(ctrl._tokens, ("<root>", "<a>", "<b>", "</b>", "</a>", "</root>"))

    def test_markers_cover_all_tags(self):
        ctrl = XMLStructureController.from_schema_text("<root><a/></root>")
        markers = ctrl.markers()
        self.assertIn("</root>", markers)
        self.assertIn("</a>", markers)

    def test_mismatched_close_raises(self):
        with self.assertRaises(ValueError, msg="Mismatched closing tag"):
            XMLStructureController.from_schema_text("<root><a></root>")

    def test_unclosed_tag_raises(self):
        with self.assertRaises(ValueError, msg="Unclosed tag"):
            XMLStructureController.from_schema_text("<root><a>")

    def test_multiple_roots_raises(self):
        with self.assertRaises(ValueError, msg="Multiple root tags"):
            XMLStructureController.from_schema_text("<a/><b/>")

    def test_empty_schema_raises(self):
        with self.assertRaises(ValueError, msg="Empty schema"):
            XMLStructureController.from_schema_text("")

    def test_only_whitespace_raises(self):
        with self.assertRaises(ValueError, msg="Empty schema"):
            XMLStructureController.from_schema_text("  \n  ")


# ── advance() ──────────────────────────────────────────────────────

class TestAdvance(unittest.TestCase):
    """Unit-тесты advance(): правильные/неправильные теги, partial match, complete."""

    def test_correct_full_document_single_phase(self):
        ctrl = XMLStructureController.from_schema_text("<root><a/></root>")
        # Фаза 1: <root><a>hello</a> (prefill + модель). После </a> остаётся только
        # закрывающий </root> — контроллер дописывает его сам (автозакрытие).
        v = ctrl.advance("<root><a>hello</a>")
        self.assertTrue(v.complete)
        self.assertIsNone(v.next_prefill)
        self.assertEqual(ctrl.document(), "<root><a>hello</a></root>")

    def test_wrong_close_erases_and_suggests_prefill(self):
        ctrl = XMLStructureController.from_schema_text("<root><a/></root>")
        v = ctrl.advance("<root></root>")
        # Корень сразу закрыт: ожидаем <a>
        self.assertFalse(v.complete)
        self.assertEqual(v.next_prefill, "<a>")
        self.assertEqual(v.corrected_content, "<root>")
        # Документ накопил согласованный префикс.
        self.assertEqual(ctrl.document(), "<root>")

    def test_correct_partial_advances_cursor(self):
        ctrl = XMLStructureController.from_schema_text("<root><a/><b/></root>")
        v = ctrl.advance("<root><a>text</a>")
        self.assertFalse(v.complete)
        # Следующий токен — opening <b>: ставим prefill, модель заполнит его дальше.
        self.assertEqual(v.next_prefill, "<b>")
        self.assertEqual(ctrl.document(), "<root><a>text</a>")

    def test_early_close_on_child_with_multiple_children(self):
        """Модель закрыла корень сразу, когда ожидается <a>."""
        ctrl = XMLStructureController.from_schema_text("<root><a/><b/></root>")
        v = ctrl.advance("<root></root>")
        self.assertEqual(v.next_prefill, "<a>")
        self.assertEqual(v.corrected_content, "<root>")

    def test_wrong_open_child_tag(self):
        """Модель открыла неверный дочерний тег <z> вместо ожидаемого <a>."""
        ctrl = XMLStructureController.from_schema_text("<root><a/></root>")
        v = ctrl.advance("<root><z>x</z>")
        # позиция <z> mismatches <a>
        self.assertFalse(v.complete)
        self.assertEqual(v.next_prefill, "<a>")
        self.assertEqual(v.corrected_content, "<root>")

    def test_empty_content_no_change(self):
        ctrl = XMLStructureController.from_schema_text("<root><a/></root>")
        v = ctrl.advance("")
        self.assertFalse(v.complete)
        self.assertIsNone(v.next_prefill)
        self.assertEqual(v.corrected_content, "")
        self.assertEqual(ctrl.document(), "")

    def test_self_closing_tag_matches_both_tokens(self):
        ctrl = XMLStructureController.from_schema_text("<root><b/></root>")
        v = ctrl.advance("<root><b/>")
        # <b/> should consume both <b> and </b> at the same position; остаётся только
        # </root> — контроллер закрывает его сам (автозакрытие).
        self.assertTrue(v.complete)
        self.assertEqual(v.corrected_content, "<root><b/></root>")
        self.assertEqual(ctrl.document(), "<root><b/></root>")

    def test_extra_tokens_after_complete(self):
        """Корень уже закрыт, но модель продолжает писать — лишнее стирается."""
        ctrl = XMLStructureController.from_schema_text("<root/>")
        v = ctrl.advance("<root></root>trailing junk")
        self.assertTrue(v.complete)
        self.assertEqual(ctrl.document(), "<root></root>")

    def test_multiple_mismatches_earliest_wins(self):
        """Два неверных тега в одной фазе — стираем до первого."""
        ctrl = XMLStructureController.from_schema_text("<root><a/></root>")
        v = ctrl.advance("<root><z>x</z><b>nope</b></root>")
        self.assertEqual(v.next_prefill, "<a>")
        # Стирает всё начиная с <z> (позиция 6).
        self.assertEqual(v.corrected_content, "<root>")

    def test_document_accumulates_across_phases(self):
        ctrl = XMLStructureController.from_schema_text("<root><a/><b/></root>")
        ctrl.advance("<root><a>one</a>")
        self.assertEqual(ctrl.document(), "<root><a>one</a>")
        ctrl.advance("<b>two</b>")
        self.assertEqual(ctrl.document(), "<root><a>one</a><b>two</b></root>")

    def test_text_between_tags_not_confused_with_tags(self):
        ctrl = XMLStructureController.from_schema_text("<root><a/></root>")
        # Пробелы, переносы, текст — не разбираются как токены. Все токены пройдены,
        # остаётся только </root> → автозакрытие.
        v = ctrl.advance("<root>\n  <a>data</a>\n")
        self.assertTrue(v.complete)
        self.assertEqual(ctrl.document(), "<root>\n  <a>data</a>\n</root>")

    # ── Автозакрытие при досрочном завершении ответа ──

    def test_prefill_suffix_in_initial_prefill(self):
        """prefill_suffix добавляется после открывающего тега корня в initial_prefill()."""
        ctrl = XMLStructureController.from_schema_text("<content_structure/>", prefill_suffix="\nL")
        self.assertEqual(ctrl.initial_prefill(), "<content_structure>\nL")
        # Токены и маркеры не зависят от суффикса.
        self.assertEqual(ctrl._tokens, ("<content_structure>", "</content_structure>"))
        self.assertEqual(ctrl.markers(), ("</content_structure>",))

    def test_prefill_suffix_survives_corrected_content_single_phase(self):
        """Схема <root/> с суффиксом: модель вернула <root>\\nL42 foo — суффикс-хвост не токен,
        автозакрытие дописывает закрывающий тег."""
        ctrl = XMLStructureController.from_schema_text("<root/>", prefill_suffix="\nL")
        v = ctrl.advance("<root>\nL42 foo")
        self.assertTrue(v.complete)
        self.assertEqual(v.corrected_content, "<root>\nL42 foo</root>")
        self.assertEqual(ctrl.document(), "<root>\nL42 foo</root>")

    def test_autoclose_single_closing_tag(self):
        """Схема <root/>: модель написала <root> и остановилась — контроллер дописывает </root>."""
        ctrl = XMLStructureController.from_schema_text("<root/>")
        v = ctrl.advance("<root>")
        self.assertTrue(v.complete)
        self.assertEqual(v.corrected_content, "<root></root>")
        self.assertEqual(ctrl.document(), "<root></root>")

    def test_autoclose_single_closing_with_text(self):
        """Схема <root/>: модель написала <root>L1-4 imports и остановилась."""
        ctrl = XMLStructureController.from_schema_text("<root/>")
        v = ctrl.advance("<root>L1-4 imports setup code")
        self.assertTrue(v.complete)
        self.assertEqual(v.corrected_content, "<root>L1-4 imports setup code</root>")
        self.assertEqual(ctrl.document(), "<root>L1-4 imports setup code</root>")

    def test_autoclose_nested_closing_tags(self):
        """Схема <root><a/></root>: модель закрыла </a>, остановилась на </root>."""
        ctrl = XMLStructureController.from_schema_text("<root><a/></root>")
        v = ctrl.advance("<root><a>text</a>")
        self.assertTrue(v.complete)
        self.assertEqual(v.corrected_content, "<root><a>text</a></root>")
        self.assertEqual(ctrl.document(), "<root><a>text</a></root>")

    def test_opening_tag_sets_prefill(self):
        """Схема <root><a/></root>: модель написала только <root> — следующий токен <a> (opening)."""
        ctrl = XMLStructureController.from_schema_text("<root><a/></root>")
        v = ctrl.advance("<root>")
        self.assertFalse(v.complete)
        self.assertEqual(v.next_prefill, "<a>")
        self.assertEqual(v.corrected_content, "<root>")
        self.assertEqual(ctrl.document(), "<root>")

    def test_autoclose_in_multi_child_then_prefill_next(self):
        """Схема <root><a/><b/></root>: написали <a>, остановились — автозакрываем нет (след. <b>),
        а ставим prefill."""
        ctrl = XMLStructureController.from_schema_text("<root><a/><b/></root>")
        v = ctrl.advance("<root><a>text</a>")
        self.assertFalse(v.complete)
        self.assertEqual(v.next_prefill, "<b>")
        self.assertEqual(v.corrected_content, "<root><a>text</a>")

    def test_multichild_autoclose_final(self):
        """Схема <root><a/><b/></root>: оба ребёнка написаны, остановились — автозакрываем </root>."""
        ctrl = XMLStructureController.from_schema_text("<root><a/><b/></root>")
        v = ctrl.advance("<root><a>one</a><b>two</b>")
        self.assertTrue(v.complete)
        self.assertEqual(v.corrected_content, "<root><a>one</a><b>two</b></root>")

    def test_autoclose_nested_all_remaining_closings(self):
        """Схема <root><a><b/></a></root>: модель дошла до </a>, остановилась — дописываем </a></root>."""
        ctrl = XMLStructureController.from_schema_text("<root><a><b/></a></root>")
        v = ctrl.advance("<root><a>text<b/>")
        self.assertTrue(v.complete)
        self.assertEqual(v.corrected_content, "<root><a>text<b/></a></root>")

    def test_prefill_only_no_autoclose(self):
        """Схема <root><a/></root>: модель вернула пустой контент (только prefill <root>).
        Следующий токен <a> (opening) — ставим prefill, не автозакрываем."""
        ctrl = XMLStructureController.from_schema_text("<root><a/></root>")
        v = ctrl.advance("<root>")
        self.assertFalse(v.complete)
        self.assertEqual(v.next_prefill, "<a>")


# ── Интеграция с agent.chat() ─────────────────────────────────────

class TestAgentController(unittest.TestCase):
    """E2E тесты контроллерного режима через agent.chat()."""

    def _agent(self):
        return LLMAgent(
            system_prompt="sys",
            disable_per_msg_summarization=True,
            autosave_enabled=False,
        )

    def _mock_call(self, phases):
        """Фабрика fake_call, которая отдаёт фазы по очереди.

        Каждый элемент phases — AssistantMessage.content (str). Фейк применяет
        prefill + обрезку по маркеру, как реальный транспорт (LLMClient.call),
        чтобы advance() получил корректный контент фазы.
        """
        it = iter(phases)

        def fake_call(messages, prefill=None, stop_markers=(), **kwargs):
            content = next(it)
            # Имитируем apply_prefill + apply_stop_markers транспорта:
            if prefill and not content.startswith(prefill):
                content = prefill + content
            if stop_markers:
                best_idx = len(content)
                for m in stop_markers:
                    idx = content.find(m)
                    if 0 <= idx < best_idx:
                        best_idx = idx
                        best_marker = m
                else:
                    best_marker = None
                if best_marker is not None:
                    content = content[: best_idx + len(best_marker)]
            return AssistantMessage(content=content), None, None

        return fake_call

    def test_happy_path_two_phases(self):
        """Схема <root><a/></root>: phase1 пишет <a>, phase2 закрывает </root>."""
        ctrl_cfg = _ctrl("<root><a/></root>")
        with mock.patch(
            "universal_agents.agent.LLMClient.call",
            side_effect=self._mock_call([
                "<a>hello</a>",   # phase 1 (prefill <root> добавится фейком)
                "</root>",         # phase 2 — закрытие корня
            ]),
        ):
            result = self._agent().chat("go", structured_output=ctrl_cfg)

        self.assertEqual(result, "<root><a>hello</a></root>")

    def test_wrong_close_repair(self):
        """Модель закрыла корень сразу → контрлер стирает, подсказывает <a>, модель генерирует заново."""
        ctrl_cfg = _ctrl("<root><a/></root>")
        with mock.patch(
            "universal_agents.agent.LLMClient.call",
            side_effect=self._mock_call([
                "</root>",        # phase 1: неверное закрытие сразу
                "<a>fixed</a>",   # phase 2: модель исправилась
                "</root>",         # phase 3: закрытие корня
            ]),
        ):
            result = self._agent().chat("go", structured_output=ctrl_cfg)

        self.assertEqual(result, "<root><a>fixed</a></root>")

    def test_max_phases_stops(self):
        """Если max_phases=2 и модель не закрывает корень — документ возвращается как есть."""
        ctrl_cfg = _ctrl("<root><a/></root>", max_phases=2)
        # Фаза 1: правильный partial (cursor=3, ожидается </root>). Фаза 2: неверный контент
        # (модель открыла <a> вместо закрытия </root>) → empty erase → max_phases стоп.
        with mock.patch(
            "universal_agents.agent.LLMClient.call",
            side_effect=self._mock_call([
                "<a>x</a>",
                "<a>y</a>",       # модель повторяет вместо закрытия
            ]),
        ):
            result = self._agent().chat("go", structured_output=ctrl_cfg)

        # Документ содержит накопленный partial из фазы 1 (фаза 2 стёрта).
        self.assertIn("<a>x</a>", result)
        self.assertIn("<root>", result)

    def test_empty_correction_retries_with_prefill(self):
        """Если corrected_content пуст после стирания — модель вернула только неверный тег,
        повтор с prefill. </root> → пустой erase + prefill <a> → <a>ok</a> (автозакроет </root>)."""
        ctrl_cfg = _ctrl("<root><a/></root>")
        call_count = [0]

        def counting_call(messages, prefill=None, stop_markers=(), **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                content = "</root>"
            else:
                content = "<a>ok</a>"
            if prefill and not content.startswith(prefill):
                content = prefill + content
            if stop_markers:
                best_idx = len(content)
                best_marker = None
                for m in stop_markers:
                    idx = content.find(m)
                    if 0 <= idx < best_idx:
                        best_idx = idx
                        best_marker = m
                if best_marker is not None:
                    content = content[: best_idx + len(best_marker)]
            return AssistantMessage(content=content), None, None

        with mock.patch("universal_agents.agent.LLMClient.call", side_effect=counting_call):
            result = self._agent().chat("go", structured_output=ctrl_cfg)

        self.assertEqual(result, "<root><a>ok</a></root>")
        self.assertEqual(call_count[0], 2)  # третья фаза </root> автозакрыта контроллером

    def test_nested_structure_handled(self):
        """Более сложная схема с вложенными детьми: transport обрезает по первому маркеру,
        поэтому phases = [<a><b>x</b>>, </a>]; финальный </root> автозакрывается."""
        ctrl_cfg = _ctrl("<root><a><b/></a></root>")
        with mock.patch(
            "universal_agents.agent.LLMClient.call",
            side_effect=self._mock_call([
                "<a><b>x</b></a>",  # phase 1: transport обрезает на </b> → <root><a><b>x</b>
                "</a>",              # phase 2: </a> → курсор на </root>, автозакрытие
            ]),
        ):
            result = self._agent().chat("go", structured_output=ctrl_cfg)

        self.assertEqual(result, "<root><a><b>x</b></a></root>")

    def test_controller_stops_when_document_complete(self):
        """Документ возвращается через controller.document() при complete=True."""
        ctrl_cfg = _ctrl("<root/>")
        with mock.patch(
            "universal_agents.agent.LLMClient.call",
            side_effect=self._mock_call(["</root>"]),
        ):
            result = self._agent().chat("go", structured_output=ctrl_cfg)

        self.assertEqual(result, "<root></root>")

    def test_no_controller_still_works(self):
        """Без контроллера (static structured output) — регрессия."""
        agent = LLMAgent(system_prompt="sys", disable_per_msg_summarization=True, autosave_enabled=False)
        phase1 = AssistantMessage(content="<structure>\ndata\n</structure>\nanalysis")

        def fake_call(messages, prefill=None, **kwargs):
            return phase1, None, None

        with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake_call):
            result = agent.chat(
                "show",
                structured_output=StructuredOutputConfig.from_prefill("<structure>\n"),
            )
        self.assertEqual(result, "<structure>\ndata\n</structure>")

    # ── Автозакрытие при досрочном завершении ответа (E2E) ──

    def test_autoclose_on_premature_end(self):
        """Модель написала корень и досрочно остановилась (не закрыла тег) — контроллер
        сам дописывает закрывающие теги и документ возвращается как валидный."""
        ctrl_cfg = _ctrl("<content_structure/>")
        with mock.patch(
            "universal_agents.agent.LLMClient.call",
            side_effect=self._mock_call([
                "L1-4 imports\nL8-91 UnifiedDockerAgent class",  # prefill <content_structure> добавится фейком, закрывающий тег НЕ написан
            ]),
        ):
            result = self._agent().chat("go", structured_output=ctrl_cfg)

        self.assertEqual(result, "<content_structure>L1-4 imports\nL8-91 UnifiedDockerAgent class</content_structure>")

    def test_autoclose_then_prefill_continuation(self):
        """Модель закрыла <a>, остановилась на <b> (opening) — контроллер ставит prefill <b>,
        следующая фаза заполняет <b>, финальный </root> дописывается сам."""
        ctrl_cfg = _ctrl("<root><a/><b/></root>")
        with mock.patch(
            "universal_agents.agent.LLMClient.call",
            side_effect=self._mock_call([
                "<a>one</a>",      # phase 1: полное <a>, но стоп на <b> (transport обрезает) → cursor на <b>
                "<b>two</b>",       # phase 2: prefill <b>, модель пишет <b>two</b>, стоп на </root>
            ]),
        ):
            result = self._agent().chat("go", structured_output=ctrl_cfg)

        # После phase2 контроллер видит cursor на </root> → автозакрывает.
        self.assertEqual(result, "<root><a>one</a><b>two</b></root>")


if __name__ == "__main__":
    unittest.main()
