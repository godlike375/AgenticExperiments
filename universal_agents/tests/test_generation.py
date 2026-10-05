import unittest
from unittest import mock

from universal_agents.agent import LLMAgent
from universal_agents.config import Config
from universal_agents.generation import (
    GenerationParams,
    StructuredOutputConfig,
    apply_stop_markers,
    extract_opening_tag,
)
from universal_agents.models import AssistantMessage


class TestGenerationParams(unittest.TestCase):
    def test_resolved_uses_config_defaults(self):
        params = GenerationParams()
        resolved = params.resolved()
        self.assertEqual(resolved.temp, Config.TEMP)
        self.assertEqual(resolved.timeout, Config.TIMEOUT)
        self.assertEqual(resolved.top_p, Config.TOP_P)
        self.assertEqual(resolved.max_tokens, Config.MAX_OUTPUT_TOKENS)

    def test_resolved_keeps_explicit_values(self):
        params = GenerationParams(temp=0.7, max_tokens=100)
        resolved = params.resolved()
        self.assertEqual(resolved.temp, 0.7)
        self.assertEqual(resolved.max_tokens, 100)
        self.assertEqual(resolved.timeout, Config.TIMEOUT)

    def test_with_temp_overrides_only_temp(self):
        params = GenerationParams(temp=0.2)
        boosted = params.with_temp(2.0)
        self.assertEqual(boosted.temp, 2.0)
        self.assertNotEqual(params.temp, boosted.temp)

    def test_resolved_does_not_mutate_original(self):
        params = GenerationParams(temp=None)
        params.resolved()
        self.assertIsNone(params.temp)

    def test_resolved_min_p_defaults_to_none(self):
        with mock.patch.object(Config, "MIN_P", None):
            self.assertIsNone(GenerationParams().resolved().min_p)

    def test_resolved_keeps_explicit_min_p(self):
        self.assertEqual(GenerationParams(min_p=0.1).resolved().min_p, 0.1)


class TestExtractOpeningTag(unittest.TestCase):
    def test_simple_tag(self):
        self.assertEqual(extract_opening_tag("<content_structure>\nL"), "content_structure")

    def test_tag_with_attributes(self):
        self.assertEqual(extract_opening_tag('<response type="xml">\n'), "response")

    def test_tag_with_newline_and_body(self):
        self.assertEqual(extract_opening_tag("<details>\nsome data"), "details")

    def test_no_angle_bracket(self):
        self.assertIsNone(extract_opening_tag("plain text"))

    def test_closing_tag(self):
        self.assertIsNone(extract_opening_tag("</tag>"))

    def test_self_closing_tag(self):
        self.assertIsNone(extract_opening_tag("<br/>"))

    def test_self_closing_with_space(self):
        self.assertIsNone(extract_opening_tag("<tag />"))

    def test_empty_string(self):
        self.assertIsNone(extract_opening_tag(""))

    def test_none(self):
        self.assertIsNone(extract_opening_tag(None))

    def test_nested_tags_returns_outer(self):
        self.assertEqual(extract_opening_tag("<outer><inner>\n"), "outer")

    def test_invalid_name_chars_in_tag(self):
        self.assertIsNone(extract_opening_tag("<my!tag>\n"))

    def test_valid_tag_with_invalid_attribute(self):
        # tag name 'my' is valid; attribute 'tag!' has invalid chars but that's fine
        self.assertEqual(extract_opening_tag("<my tag!>\n"), "my")

    def test_tag_with_colon(self):
        self.assertEqual(extract_opening_tag("<ns:tag>\n"), "ns:tag")


class TestStructuredOutputConfig(unittest.TestCase):
    def test_from_prefill_extracts_tag(self):
        cfg = StructuredOutputConfig.from_prefill("<content_structure>\nL")
        self.assertEqual(cfg.stop_markers, ("</content_structure>",))

    def test_from_prefill_with_next_prefill(self):
        cfg = StructuredOutputConfig.from_prefill("<a>\n", next_prefill="<b>\n")
        self.assertEqual(cfg.stop_markers, ("</a>",))
        self.assertEqual(cfg.next_prefills, ("<b>\n",))

    def test_from_prefill_no_tag(self):
        cfg = StructuredOutputConfig.from_prefill("plain text", next_prefill="<x>\n")
        self.assertEqual(cfg.stop_markers, ())
        self.assertEqual(cfg.next_prefills, ("<x>\n",))

    def test_effective_markers_explicit(self):
        cfg = StructuredOutputConfig(stop_markers=("END",))
        self.assertEqual(cfg.effective_markers("<anything>"), ("END",))

    def test_effective_markers_auto_from_prefill(self):
        cfg = StructuredOutputConfig()
        self.assertEqual(cfg.effective_markers("<root>\n"), ("</root>",))

    def test_effective_markers_none_without_prefill(self):
        cfg = StructuredOutputConfig()
        self.assertEqual(cfg.effective_markers(), ())


class TestApplyStopMarkers(unittest.TestCase):
    def test_no_marker(self):
        content, hit = apply_stop_markers("hello world", ())
        self.assertEqual(content, "hello world")
        self.assertFalse(hit)

    def test_marker_found(self):
        content, hit = apply_stop_markers(
            "<structure>a-b\n</structure>\nanalysis text", ("</structure>",)
        )
        self.assertEqual(content, "<structure>a-b\n</structure>")
        self.assertTrue(hit)

    def test_marker_not_in_content(self):
        content, hit = apply_stop_markers("no tags here", ("</end>",))
        self.assertEqual(content, "no tags here")
        self.assertFalse(hit)

    def test_multiple_markers_first_wins(self):
        content, hit = apply_stop_markers(
            "aaa</first>bbb</second>", ("</second>", "</first>")
        )
        # </second> appears at index 10, </first> at 3; first marker in CONTENT is </first>
        self.assertEqual(content, "aaa</first>")
        self.assertTrue(hit)

    def test_empty_content(self):
        content, hit = apply_stop_markers("", ("</end>",))
        self.assertEqual(content, "")
        self.assertFalse(hit)


class TestAgentStructuredOutput(unittest.TestCase):
    def test_two_phase_structured_output(self):
        """Фаза 1: <structure>...</structure> обрезается по маркеру, затем запускается
        фаза 2 с другим prefill. Цикл завершается когда next_prefills исчерпан."""
        agent = LLMAgent(system_prompt="sys", disable_per_msg_summarization=True, autosave_enabled=False)

        phase1 = AssistantMessage(content="<structure>\nL1-5 foo\nL7-10 bar\n</structure>\nanalysis text")
        phase2 = AssistantMessage(content="<details>\nitem1\nitem2\n</details>")

        responses = [(phase1, None, None), (phase2, None, None)]
        call_idx = [0]

        def fake_call(messages, prefill=None, **kwargs):
            idx = call_idx[0]
            call_idx[0] += 1
            return responses[idx]

        with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake_call):
            result = agent.chat(
                "Give me the structure",
                structured_output=StructuredOutputConfig.from_prefill(
                    "<structure>\n", next_prefill="<details>\n", max_phases=5
                ),
            )

        self.assertIn("item1", result)
        self.assertIn("<details>", result)
        self.assertEqual(call_idx[0], 2)

    def test_single_phase_just_truncates(self):
        """Одна фаза без next_prefills: модель обрезается по маркеру, цикл завершается."""
        agent = LLMAgent(system_prompt="sys", disable_per_msg_summarization=True, autosave_enabled=False)

        phase1 = AssistantMessage(content="<data>\nrow1\nrow2\n</data>\nextra analysis")
        responses = [(phase1, None, None)]

        def fake_call(messages, prefill=None, **kwargs):
            return responses.pop(0)

        with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake_call):
            result = agent.chat(
                "Show data",
                structured_output=StructuredOutputConfig.from_prefill("<data>\n"),
            )

        self.assertEqual(result, "<data>\nrow1\nrow2\n</data>")

    def test_max_phases_limited(self):
        """Если max_phases=1, вторая фаза не запускается (next_prefills[0] игнорируется)."""
        agent = LLMAgent(system_prompt="sys", disable_per_msg_summarization=True, autosave_enabled=False)

        phase1 = AssistantMessage(content="<a>\n</a>\nanalysis")
        responses = [(phase1, None, None)]

        def fake_call(messages, prefill=None, **kwargs):
            return responses.pop(0) if responses else (AssistantMessage(content="ignored"), None, None)

        with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake_call):
            result = agent.chat(
                "go",
                structured_output=StructuredOutputConfig.from_prefill(
                    "<a>\n", next_prefill="<b>\n", max_phases=1
                ),
            )

        # Phase 1 was the only response; phase 2 was NOT triggered
        self.assertEqual(result, "<a>\n</a>")
        self.assertEqual(len(responses), 0)

    def test_broken_call_skipped_when_marker_hit(self):
        """Если маркер сработал, broken_call detection НЕ срабатывает на XML-контенте."""
        agent = LLMAgent(
            system_prompt="sys",
            disable_per_msg_summarization=True,
            autosave_enabled=False,
        )

        # Content contains a tool-like reference (would normally trigger broken_call)
        phase1 = AssistantMessage(content="<output>\nresult of read(file)\n</output>")
        responses = [(phase1, None, None)]

        def fake_call(messages, prefill=None, **kwargs):
            return responses.pop(0)

        with mock.patch("universal_agents.agent.LLMClient.call", side_effect=fake_call):
            result = agent.chat(
                "output",
                structured_output=StructuredOutputConfig.from_prefill("<output>\n"),
            )

        self.assertIn("<output>", result)
        self.assertIn("</output>", result)


if __name__ == "__main__":
    unittest.main()
