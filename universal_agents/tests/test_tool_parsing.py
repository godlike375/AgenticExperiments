import unittest
from types import SimpleNamespace

from universal_agents.constants import ENVIRONMENT_PREFIX
from universal_agents.tool_parsing import (
    tc_name,
    tc_args,
    is_error_content,
    detect_broken_call,
    parse_tool_args,
    try_parse_tool_args,
    args_are_valid,
    normalize_args,
)
from universal_agents.tools.fs import line_range_edit as _line_range_edit_tool
from universal_agents.tools.fs import read as _read_tool
from universal_agents.tools.fs import search as _search_tool
from universal_agents.tools.host_shell import run_powershell as _run_powershell_tool


class TestNameArgs(unittest.TestCase):
    def test_tc_name_openai_format(self):
        tc = SimpleNamespace(function=SimpleNamespace(name=_read_tool.__name__, arguments="{}"))
        self.assertEqual(tc_name(tc), _read_tool.__name__)
        self.assertEqual(tc_args(tc), "{}")

    def test_tc_name_responses_format(self):
        tc = SimpleNamespace(name=_search_tool.__name__, arguments='{"q": "x"}')
        self.assertEqual(tc_name(tc), _search_tool.__name__)
        self.assertEqual(tc_args(tc), '{"q": "x"}')

    def test_tc_name_missing_returns_empty(self):
        self.assertEqual(tc_name(SimpleNamespace()), "")
        self.assertEqual(tc_args(SimpleNamespace()), "")


class TestIsErrorContent(unittest.TestCase):
    def test_error_prefix(self):
        self.assertTrue(is_error_content("Error: boom"))
        self.assertTrue(is_error_content(f"{ENVIRONMENT_PREFIX} Error: boom"))
        self.assertTrue(is_error_content(f"{ENVIRONMENT_PREFIX} {ENVIRONMENT_PREFIX} Error: x"))

    def test_normal_content(self):
        self.assertFalse(is_error_content("all good"))
        self.assertFalse(is_error_content(""))


class TestDetectBrokenCall(unittest.TestCase):
    def test_strong_tag(self):
        self.assertTrue(detect_broken_call("Please call <tool_call>read</tool_call>", {_read_tool.__name__}))
        self.assertTrue(detect_broken_call("<tool>read</tool>", set()))

    def test_requires_xml_tag(self):
        self.assertFalse(detect_broken_call("The function cwd() returns the dir.", {"cwd"}))
        self.assertFalse(detect_broken_call("", set()))

    def test_xml_tag_with_tool_sign(self):
        self.assertTrue(detect_broken_call("<custom_tag> read({}) </custom_tag>", {_read_tool.__name__}))
        self.assertFalse(detect_broken_call("<custom_tag> nothing here </custom_tag>", {_read_tool.__name__}))

    def test_quoted_tag_with_prose_tool_mention_is_clean(self):
        text = "Префикс <SYSTEM>, затем read(path) и search(pattern, path) в обзоре."
        self.assertFalse(detect_broken_call(text, {_read_tool.__name__, _search_tool.__name__}))

    def test_standalone_call_without_tag_is_broken(self):
        self.assertTrue(detect_broken_call('read({"path": "a.txt"})', {_read_tool.__name__}))


class TestDetectBrokenCallWithoutGate(unittest.TestCase):

    NAMES = {_read_tool.__name__, _line_range_edit_tool.__name__, _run_powershell_tool.__name__}

    def test_prose_mention_is_not_broken_call(self):
        for text in (
            "read() выполнен: показана структура каталога.",
            "Вызываю read() для просмотра содержимого текущей директории:",
            "Сделаю read(), потом напишу итог.",
            "Инструменты: read(path), write(path).",
            "The function cwd() returns the dir.",
        ):
            with self.subTest(text=text):
                self.assertFalse(detect_broken_call(text, self.NAMES, require_tag=False))

    def test_standalone_call_is_broken_call(self):
        self.assertTrue(detect_broken_call('read({"path": "a.txt"})', self.NAMES, require_tag=False))
        self.assertTrue(detect_broken_call("read({}).", self.NAMES, require_tag=False))

    def test_tool_call_closing_tag_is_broken_call(self):
        content = '<invoke_function>{"function_name":"read","args":{}}</function></tool_call>'
        self.assertTrue(detect_broken_call(content, self.NAMES, require_tag=False))

    def test_strong_tag_inside_is_broken_call(self):
        self.assertTrue(detect_broken_call("<tool>read</tool>", self.NAMES, require_tag=False))

    def test_empty_content_is_clean(self):
        self.assertFalse(detect_broken_call("", self.NAMES, require_tag=False))


class TestParseArgs(unittest.TestCase):
    def test_parse_valid(self):
        self.assertEqual(parse_tool_args('{"a": 1}'), {"a": 1})

    def test_parse_empty(self):
        self.assertEqual(parse_tool_args(""), {})
        self.assertEqual(parse_tool_args("{}"), {})
        self.assertEqual(parse_tool_args("null"), {})

    def test_parse_invalid(self):
        self.assertEqual(parse_tool_args("not json"), {})
        self.assertEqual(parse_tool_args("[1,2]"), {})

    def test_try_parse_invalid_is_none(self):
        self.assertIsNone(try_parse_tool_args("not json"))
        self.assertIsNone(try_parse_tool_args(""))
        self.assertEqual(try_parse_tool_args('{"a": 1}'), {"a": 1})

    def test_args_are_valid(self):
        self.assertTrue(args_are_valid("{}"))
        self.assertTrue(args_are_valid(""))
        self.assertTrue(args_are_valid('{"a": 1}'))
        self.assertFalse(args_are_valid("not json"))


class TestNormalizeArgs(unittest.TestCase):
    def test_normalize_empty(self):
        self.assertEqual(normalize_args(""), "")
        self.assertEqual(normalize_args("{}"), "")

    def test_normalize_sorts_keys_and_compacts(self):
        self.assertEqual(normalize_args('{"b": 2, "a": 1}'), '{"a":1,"b":2}')

    def test_normalize_invalid_returns_stripped(self):
        self.assertEqual(normalize_args("  raw text  "), "raw text")


if __name__ == "__main__":
    unittest.main()
