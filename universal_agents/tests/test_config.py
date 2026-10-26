from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import dataclass

from universal_agents import config as config_module
from universal_agents.config import (
    DEFAULT_API_TOKEN,
    DEFAULT_API_URL,
    DEFAULT_MODEL_NAME,
    Config,
    _apply_env_file,
    parse_env_file,
)


class TestParseEnvFile(unittest.TestCase):
    def test_parses_keys_quotes_and_comments(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, ".env")
            with open(path, "w", encoding="utf-8") as f:
                f.write('# comment\nAPI_URL="http://x/v1"\nexport MODEL_NAME=m1\n\nBAD LINE\nAPI_TOKEN=tok\n')
            values = parse_env_file(path)
        self.assertEqual(values, {"API_URL": "http://x/v1", "MODEL_NAME": "m1", "API_TOKEN": "tok"})

    def test_missing_file_returns_empty(self):
        self.assertEqual(parse_env_file(os.path.join(tempfile.gettempdir(), "no_such_env_file_x")), {})


@dataclass
class _Cfg:
    ENV_FILE: str
    USE_ENV: bool = True
    API_URL: str = DEFAULT_API_URL
    API_TOKEN: str = DEFAULT_API_TOKEN
    MODEL_NAME: str = DEFAULT_MODEL_NAME


class TestApplyEnvFile(unittest.TestCase):
    def _env(self, text: str) -> str:
        fd, path = tempfile.mkstemp(suffix=".env")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        self.addCleanup(os.remove, path)
        return path

    def test_env_fills_fields_left_at_default(self):
        cfg = _Cfg(ENV_FILE=self._env("API_URL=http://env/v1\nMODEL_NAME=env-model\nAPI_TOKEN=env-tok\n"))
        _apply_env_file(cfg)
        self.assertEqual((cfg.API_URL, cfg.MODEL_NAME, cfg.API_TOKEN), ("http://env/v1", "env-model", "env-tok"))

    def test_explicit_config_value_wins_over_env(self):
        cfg = _Cfg(ENV_FILE=self._env("API_URL=http://env/v1\nMODEL_NAME=env-model\n"), API_URL="http://cfg/v1")
        _apply_env_file(cfg)
        self.assertEqual(cfg.API_URL, "http://cfg/v1")
        self.assertEqual(cfg.MODEL_NAME, "env-model")

    def test_missing_env_keeps_defaults(self):
        cfg = _Cfg(ENV_FILE=os.path.join(tempfile.gettempdir(), "no_such_env_file_y"))
        _apply_env_file(cfg)
        self.assertEqual(cfg.API_URL, DEFAULT_API_URL)

    def test_use_env_false_ignores_env_file(self):
        cfg = _Cfg(ENV_FILE=self._env("API_URL=http://env/v1\nMODEL_NAME=env-model\n"), USE_ENV=False)
        _apply_env_file(cfg)
        self.assertEqual((cfg.API_URL, cfg.MODEL_NAME), (DEFAULT_API_URL, DEFAULT_MODEL_NAME))

    def test_use_env_true_reads_env_file(self):
        cfg = _Cfg(ENV_FILE=self._env("API_URL=http://env/v1\n"), USE_ENV=True)
        _apply_env_file(cfg)
        self.assertEqual(cfg.API_URL, "http://env/v1")


class TestConfigDefaults(unittest.TestCase):
    def test_default_api_url_is_localhost(self):
        self.assertEqual(config_module.DEFAULT_API_URL, "http://localhost:1234/v1")

    def test_env_file_default_points_to_project_root(self):
        from universal_agents.project_root import find_project_root

        root = find_project_root() or os.getcwd()
        self.assertEqual(os.path.dirname(Config.ENV_FILE), root)
        self.assertEqual(os.path.basename(Config.ENV_FILE), ".env")


if __name__ == "__main__":
    unittest.main()
