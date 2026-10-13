import unittest

from universal_agents.subprocess_utils import _decode_bytes


class TestDecodeBytes(unittest.TestCase):
    def test_valid_utf8_uses_primary_encoding(self):
        data = "привет".encode("utf-8")
        self.assertEqual(_decode_bytes(data, "utf-8", "cp866"), "привет")

    def test_cp866_bytes_fall_back(self):
        data = "привет".encode("cp866")
        self.assertEqual(_decode_bytes(data, "utf-8", "cp866"), "привет")

    def test_without_fallback_invalid_bytes_are_replaced(self):
        data = "привет".encode("cp866")
        out = _decode_bytes(data, "utf-8", None)
        self.assertIn("\ufffd", out)


if __name__ == "__main__":
    unittest.main()
