import unittest

from universal_agents.generation import GenerationParams
from universal_agents.config import Config


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


if __name__ == "__main__":
    unittest.main()
