
import unittest
from unittest import mock

from universal_agents.agent import LLMAgent
from universal_agents.controllers import (
    XMLStructureController,
)
from universal_agents.generation import StructuredOutputConfig
from universal_agents.models import AssistantMessage



def _ctrl(schema: str, max_phases: int = 5) -> StructuredOutputConfig:
    return StructuredOutputConfig.from_controller(
        XMLStructureController.from_schema_text(schema),
        max_phases=max_phases,
    )



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



class TestAdvance(unittest.TestCase):

    def test_correct_full_document_single_phase(self):
        ctrl = XMLStructureController.from_schema_text("<root><a/></root>")
        v = ctrl.advance("<root><a>hello</a>")
        self.assertTrue(v.complete)
        self.assertIsNone(v.next_prefill)
        self.assertEqual(ctrl.document(), "<root><a>hello</a></root>")

    def test_wrong_close_erases_and_suggests_prefill(self):
        ctrl = XMLStructureController.from_schema_text("<root><a/></root>")
        v = ctrl.advance("<root></root>")
        self.assertFalse(v.complete)
        self.assertEqual(v.next_prefill, "<a>")
        self.assertEqual(v.corrected_content, "<root>")
        self.assertEqual(ctrl.document(), "<root>")

    def test_correct_partial_advances_cursor(self):
        ctrl = XMLStructureController.from_schema_text("<root><a/><b/></root>")
        v = ctrl.advance("<root><a>text</a>")
        self.assertFalse(v.complete)
        self.assertEqual(v.next_prefill, "<b>")
        self.assertEqual(ctrl.document(), "<root><a>text</a>")

    def test_early_close_on_child_with_multiple_children(self):
        ctrl = XMLStructureController.from_schema_text("<root><a/><b/></root>")
        v = ctrl.advance("<root></root>")
        self.assertEqual(v.next_prefill, "<a>")
        self.assertEqual(v.corrected_content, "<root>")

    def test_wrong_open_child_tag(self):
        ctrl = XMLStructureController.from_schema_text("<root><a/></root>")
        v = ctrl.advance("<root><z>x</z>")
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
        self.assertTrue(v.complete)
        self.assertEqual(v.corrected_content, "<root><b/></root>")
        self.assertEqual(ctrl.document(), "<root><b/></root>")

    def test_extra_tokens_after_complete(self):
        ctrl = XMLStructureController.from_schema_text("<root/>")
        v = ctrl.advance("<root></root>trailing junk")
        self.assertTrue(v.complete)
        self.assertEqual(ctrl.document(), "<root></root>")

    def test_multiple_mismatches_earliest_wins(self):
        ctrl = XMLStructureController.from_schema_text("<root><a/></root>")
        v = ctrl.advance("<root><z>x</z><b>nope</b></root>")
        self.assertEqual(v.next_prefill, "<a>")
        self.assertEqual(v.corrected_content, "<root>")

    def test_document_accumulates_across_phases(self):
        ctrl = XMLStructureController.from_schema_text("<root><a/><b/></root>")
        ctrl.advance("<root><a>one</a>")
        self.assertEqual(ctrl.document(), "<root><a>one</a>")
        ctrl.advance("<b>two</b>")
        self.assertEqual(ctrl.document(), "<root><a>one</a><b>two</b></root>")

    def test_text_between_tags_not_confused_with_tags(self):
        ctrl = XMLStructureController.from_schema_text("<root><a/></root>")
        v = ctrl.advance("<root>\n  <a>data</a>\n")
        self.assertTrue(v.complete)
        self.assertEqual(ctrl.document(), "<root>\n  <a>data</a>\n</root>")


    def test_prefill_suffix_in_initial_prefill(self):
        ctrl = XMLStructureController.from_schema_text("<content_structure/>", prefill_suffix="\nL")
        self.assertEqual(ctrl.initial_prefill(), "<content_structure>\nL")
        self.assertEqual(ctrl._tokens, ("<content_structure>", "</content_structure>"))
        self.assertEqual(ctrl.markers(), ("</content_structure>",))

    def test_prefill_suffix_survives_corrected_content_single_phase(self):
        ctrl = XMLStructureController.from_schema_text("<root/>", prefill_suffix="\nL")
        v = ctrl.advance("<root>\nL42 foo")
        self.assertTrue(v.complete)
        self.assertEqual(v.corrected_content, "<root>\nL42 foo</root>")
        self.assertEqual(ctrl.document(), "<root>\nL42 foo</root>")

    def test_autoclose_single_closing_tag(self):
        ctrl = XMLStructureController.from_schema_text("<root/>")
        v = ctrl.advance("<root>")
        self.assertTrue(v.complete)
        self.assertEqual(v.corrected_content, "<root></root>")
        self.assertEqual(ctrl.document(), "<root></root>")

    def test_autoclose_single_closing_with_text(self):
        ctrl = XMLStructureController.from_schema_text("<root/>")
        v = ctrl.advance("<root>L1-4 imports setup code")
        self.assertTrue(v.complete)
        self.assertEqual(v.corrected_content, "<root>L1-4 imports setup code</root>")
        self.assertEqual(ctrl.document(), "<root>L1-4 imports setup code</root>")

    def test_autoclose_nested_closing_tags(self):
        ctrl = XMLStructureController.from_schema_text("<root><a/></root>")
        v = ctrl.advance("<root><a>text</a>")
        self.assertTrue(v.complete)
        self.assertEqual(v.corrected_content, "<root><a>text</a></root>")
        self.assertEqual(ctrl.document(), "<root><a>text</a></root>")

    def test_opening_tag_sets_prefill(self):
        ctrl = XMLStructureController.from_schema_text("<root><a/></root>")
        v = ctrl.advance("<root>")
        self.assertFalse(v.complete)
        self.assertEqual(v.next_prefill, "<a>")
        self.assertEqual(v.corrected_content, "<root>")
        self.assertEqual(ctrl.document(), "<root>")

    def test_autoclose_in_multi_child_then_prefill_next(self):
        ctrl = XMLStructureController.from_schema_text("<root><a/><b/></root>")
        v = ctrl.advance("<root><a>text</a>")
        self.assertFalse(v.complete)
        self.assertEqual(v.next_prefill, "<b>")
        self.assertEqual(v.corrected_content, "<root><a>text</a>")

    def test_multichild_autoclose_final(self):
        ctrl = XMLStructureController.from_schema_text("<root><a/><b/></root>")
        v = ctrl.advance("<root><a>one</a><b>two</b>")
        self.assertTrue(v.complete)
        self.assertEqual(v.corrected_content, "<root><a>one</a><b>two</b></root>")

    def test_autoclose_nested_all_remaining_closings(self):
        ctrl = XMLStructureController.from_schema_text("<root><a><b/></a></root>")
        v = ctrl.advance("<root><a>text<b/>")
        self.assertTrue(v.complete)
        self.assertEqual(v.corrected_content, "<root><a>text<b/></a></root>")

    def test_prefill_only_no_autoclose(self):
        ctrl = XMLStructureController.from_schema_text("<root><a/></root>")
        v = ctrl.advance("<root>")
        self.assertFalse(v.complete)
        self.assertEqual(v.next_prefill, "<a>")

    def test_expect_tokens_consumes_known_tokens_without_autoclose(self):
        ctrl = XMLStructureController.from_sibling_schema("<a/><b/>")
        v = ctrl.advance("<a>мысли</wrong>")
        self.assertFalse(v.complete)
        self.assertEqual(v.corrected_content, "<a>мысли")
        self.assertEqual(v.next_prefill, "</a>")
        ctrl.expect_tokens(v.next_prefill)
        v2 = ctrl.advance("<b>ответ</b>")
        self.assertTrue(v2.complete)
        self.assertEqual(v2.corrected_content, "<b>ответ</b>")

    def test_expect_tokens_ignores_mismatch(self):
        ctrl = XMLStructureController.from_sibling_schema("<a/><b/>")
        ctrl.expect_tokens("</a>")
        v = ctrl.advance("<a>текст</a><b>ответ</b>")
        self.assertTrue(v.complete)
        self.assertEqual(v.corrected_content, "<a>текст</a><b>ответ</b>")



class TestAgentController(unittest.TestCase):

    def _agent(self):
        return LLMAgent(
            system_prompt="sys",
            disable_per_msg_summarization=True,
            autosave_enabled=False,
        )

    def _mock_call(self, phases):
        it = iter(phases)

        def fake_call(messages, prefill=None, stop_markers=(), **kwargs):
            content = next(it)
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
        ctrl_cfg = _ctrl("<root><a/></root>", max_phases=2)
        with mock.patch(
            "universal_agents.agent.LLMClient.call",
            side_effect=self._mock_call([
                "<a>x</a>",
                "<a>y</a>",       # модель повторяет вместо закрытия
            ]),
        ):
            result = self._agent().chat("go", structured_output=ctrl_cfg)

        self.assertIn("<a>x</a>", result)
        self.assertIn("<root>", result)

    def test_empty_correction_retries_with_prefill(self):
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
        ctrl_cfg = _ctrl("<root/>")
        with mock.patch(
            "universal_agents.agent.LLMClient.call",
            side_effect=self._mock_call(["</root>"]),
        ):
            result = self._agent().chat("go", structured_output=ctrl_cfg)

        self.assertEqual(result, "<root></root>")

    def test_no_controller_still_works(self):
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


    def test_autoclose_on_premature_end(self):
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
        ctrl_cfg = _ctrl("<root><a/><b/></root>")
        with mock.patch(
            "universal_agents.agent.LLMClient.call",
            side_effect=self._mock_call([
                "<a>one</a>",      # phase 1: полное <a>, но стоп на <b> (transport обрезает) → cursor на <b>
                "<b>two</b>",       # phase 2: prefill <b>, модель пишет <b>two</b>, стоп на </root>
            ]),
        ):
            result = self._agent().chat("go", structured_output=ctrl_cfg)

        self.assertEqual(result, "<root><a>one</a><b>two</b></root>")


if __name__ == "__main__":
    unittest.main()
