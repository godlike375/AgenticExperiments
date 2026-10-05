"""Тесты доверия путям и доверенных каталогов (tools_mixin)."""

import os
import tempfile
import unittest
from unittest import mock

from universal_agents.agent import LLMAgent


class TestTrustedDirs(unittest.TestCase):
    def test_auto_trust_git_root(self):
        with tempfile.TemporaryDirectory() as repo:
            os.makedirs(os.path.join(repo, ".git"))
            open(os.path.join(repo, ".git", "HEAD"), "w").close()
            with mock.patch("universal_agents.agent_mixins.tools_mixin.find_project_root", return_value=repo):
                agent = LLMAgent(system_prompt="sys")
            self.assertIn(os.path.abspath(repo), agent.trusted_dirs)
            # файлы внутри корня считаются доверенными
            self.assertTrue(agent.is_path_trusted(os.path.join(repo, "src", "index.html")))

    def test_auto_trust_skipped_when_no_git(self):
        with tempfile.TemporaryDirectory():
            with mock.patch("universal_agents.agent_mixins.tools_mixin.find_project_root", return_value=None):
                agent = LLMAgent(system_prompt="sys")
            self.assertEqual(agent.trusted_dirs, set())


if __name__ == "__main__":
    unittest.main()
