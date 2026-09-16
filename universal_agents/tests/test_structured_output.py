"""Тесты механизма структурированного вывода: стоп-маркеры, auto-detect, multi-phase."""

import unittest
from types import SimpleNamespace
from unittest import mock

from universal_agents.agent import LLMAgent
from universal_agents.generation import (
    StructuredOutputConfig,
    extract_opening_tag,
    apply_stop_markers,
)
from universal_agents.llm_client import LLMClient, StreamSession
from universal_agents.models import AssistantMessage


def _chunk(delta=None, usage=None):
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)], usage=usage)


# ── extract_opening_tag ──────────────────────────────────────────────

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


# ── StructuredOutputConfig ───────────────────────────────────────────

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


# ── apply_stop_markers ───────────────────────────────────────────────

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


# ── StreamSession: stop_markers ──────────────────────────────────────

class TestStreamSessionStopMarkers(unittest.TestCase):
    def _chunk(self, delta=None):
        return SimpleNamespace(choices=[SimpleNamespace(delta=delta)], usage=None)

    def test_marker_truncates_content_and_closes(self):
        """Стоп-маркер обрезает контент, закрывает стрим и ставит флаг."""
        def stream():
            yield self._chunk(SimpleNamespace(content="L1-5 foo\n", tool_calls=None, reasoning_content=None))
            yield self._chunk(SimpleNamespace(content="</structure> more stuff", tool_calls=None, reasoning_content=None))
            yield self._chunk(SimpleNamespace(content=" never seen", tool_calls=None, reasoning_content=None))

        with mock.patch("universal_agents.llm_client.LLMClient.stream", return_value=stream()), \
             mock.patch("universal_agents.llm_client.LLMClient.close_stream"):
            session = StreamSession(
                [{"role": "user", "content": "hi"}],
                on_stream_chunk=lambda _: None,
            )
            error, stopped = session.consume(stop_markers=("</structure>",))

        self.assertFalse(error)
        self.assertTrue(stopped)
        self.assertTrue(session.stopped_at_marker)
        self.assertEqual(session.acc.content, "L1-5 foo\n</structure>")

    def test_no_marker_leaves_content_untouched(self):
        def stream():
            yield self._chunk(SimpleNamespace(content="no marker", tool_calls=None, reasoning_content=None))

        with mock.patch("universal_agents.llm_client.LLMClient.stream", return_value=stream()):
            session = StreamSession(
                [{"role": "user", "content": "hi"}],
                on_stream_chunk=lambda _: None,
            )
            error, stopped = session.consume(stop_markers=("</end>",))

        self.assertFalse(stopped)
        self.assertFalse(session.stopped_at_marker)
        self.assertEqual(session.acc.content, "no marker")

    def test_marker_in_first_chunk(self):
        def stream():
            yield self._chunk(SimpleNamespace(content="<tag>\n</tag>", tool_calls=None, reasoning_content=None))

        with mock.patch("universal_agents.llm_client.LLMClient.stream", return_value=stream()), \
             mock.patch("universal_agents.llm_client.LLMClient.close_stream"):
            session = StreamSession(
                [{"role": "user", "content": "hi"}],
                on_stream_chunk=lambda _: None,
            )
            error, stopped = session.consume(stop_markers=("</tag>",))

        self.assertTrue(stopped)
        self.assertTrue(session.stopped_at_marker)
        self.assertEqual(session.acc.content, "<tag>\n</tag>")

    def test_stop_check_takes_priority_over_marker(self):
        """Если stop_check сработал раньше маркера — это stop_check, а не маркер."""
        processed = [0]

        def stream():
            for c in ["a", "b", "c"]:
                processed[0] += 1
                yield self._chunk(SimpleNamespace(content=c, tool_calls=None, reasoning_content=None))

        with mock.patch("universal_agents.llm_client.LLMClient.stream", return_value=stream()), \
             mock.patch("universal_agents.llm_client.LLMClient.close_stream"):
            session = StreamSession(
                [{"role": "user", "content": "hi"}],
                on_stream_chunk=lambda _: None,
            )
            error, stopped = session.consume(
                stop_check=lambda: processed[0] >= 2,
                stop_markers=("b",),
            )

        self.assertTrue(stopped)
        self.assertFalse(session.stopped_at_marker)


# ── LLMClient.call: non-streaming with stop_markers ─────────────────

class TestCallWithStopMarkers(unittest.TestCase):
    def _fake_response(self, content):
        msg = SimpleNamespace(content=content, tool_calls=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=msg)],
            usage=None,
        )

    def test_nonstream_truncates_at_marker(self):
        fake_client = mock.Mock()
        fake_client.chat.completions.create.return_value = self._fake_response(
            "<tag>\ndata\n</tag>\nanalysis"
        )
        with mock.patch("universal_agents.llm_client.LLMClient.get_client", return_value=fake_client):
            result, err, usage = LLMClient.call(
                [{"role": "user", "content": "hi"}],
                stop_markers=("</tag>",),
            )
        self.assertIsNone(err)
        self.assertEqual(result.content, "<tag>\ndata\n</tag>")

    def test_nonstream_no_marker_keeps_content(self):
        fake_client = mock.Mock()
        fake_client.chat.completions.create.return_value = self._fake_response("no tags")
        with mock.patch("universal_agents.llm_client.LLMClient.get_client", return_value=fake_client):
            result, err, usage = LLMClient.call(
                [{"role": "user", "content": "hi"}],
                stop_markers=("</end>",),
            )
        self.assertEqual(result.content, "no tags")


# ── Agent: multi-phase structured output ─────────────────────────────

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

    def test_no_structured_output_works_normally(self):
        """Без structured_output — всё работает как раньше."""
        agent = LLMAgent(system_prompt="sys")
        fake = AssistantMessage(content="normal reply")
        with mock.patch("universal_agents.agent.LLMClient.call", return_value=(fake, None, None)):
            result = agent.chat("hi")
        self.assertEqual(result, "normal reply")

    def test_auto_detect_marker_from_prefill(self):
        """from_prefill автоматически создаёт маркер из opening-тега."""
        cfg = StructuredOutputConfig.from_prefill("<content_structure>\nL", next_prefill="<next>\n")
        self.assertEqual(cfg.effective_markers("<content_structure>\nL"), ("</content_structure>",))
        self.assertEqual(cfg.next_prefills, ("<next>\n",))

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


# ── service_llm_call with structured_output ──────────────────────────

class TestServiceLlmCallStructuredOutput(unittest.TestCase):
    def test_truncates_content_at_marker(self):
        agent = LLMAgent(system_prompt="sys", disable_per_msg_summarization=True, autosave_enabled=False)
        fake = AssistantMessage(content="<tag>\ndata\n</tag>\ntrailing text")
        with mock.patch("universal_agents.agent.LLMClient.call", return_value=(fake, None, None)) as m:
            msg_obj, err, usage = agent.service_llm_call(
                [{"role": "user", "content": "hi"}],
                structured_output=StructuredOutputConfig.from_prefill("<tag>\n"),
            )
        self.assertEqual(msg_obj.content, "<tag>\ndata\n</tag>")
        # stop_markers should have been passed to LLMClient.call
        call_kwargs = m.call_args.kwargs
        self.assertEqual(call_kwargs.get("stop_markers"), ("</tag>",))


if __name__ == "__main__":
    unittest.main()
