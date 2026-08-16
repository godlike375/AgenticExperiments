import unittest
from types import SimpleNamespace

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


class TestNameArgs(unittest.TestCase):
    def test_tc_name_openai_format(self):
        tc = SimpleNamespace(function=SimpleNamespace(name="read", arguments="{}"))
        self.assertEqual(tc_name(tc), "read")
        self.assertEqual(tc_args(tc), "{}")

    def test_tc_name_responses_format(self):
        tc = SimpleNamespace(name="search", arguments='{"q": "x"}')
        self.assertEqual(tc_name(tc), "search")
        self.assertEqual(tc_args(tc), '{"q": "x"}')

    def test_tc_name_missing_returns_empty(self):
        self.assertEqual(tc_name(SimpleNamespace()), "")
        self.assertEqual(tc_args(SimpleNamespace()), "")


class TestIsErrorContent(unittest.TestCase):
    def test_error_prefix(self):
        self.assertTrue(is_error_content("Error: boom"))
        self.assertTrue(is_error_content("[[SYS ENV]] Error: boom"))
        self.assertTrue(is_error_content("[[SYS ENV]] [[SYS ENV]] Error: x"))

    def test_normal_content(self):
        self.assertFalse(is_error_content("all good"))
        self.assertFalse(is_error_content(""))


class TestDetectBrokenCall(unittest.TestCase):
    def test_strong_tag(self):
        self.assertTrue(detect_broken_call("Please call <tool_call>read</tool_call>", {"read"}))
        self.assertTrue(detect_broken_call("<tool>read</tool>", set()))

    def test_requires_xml_tag(self):
        self.assertFalse(detect_broken_call("The function cwd() returns the dir.", {"cwd"}))
        self.assertFalse(detect_broken_call("", set()))

    def test_xml_tag_with_tool_sign(self):
        self.assertTrue(detect_broken_call("<custom_tag> read({}) </custom_tag>", {"read"}))
        self.assertFalse(detect_broken_call("<custom_tag> nothing here </custom_tag>", {"read"}))


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