import os
import tempfile
import unittest
from unittest import mock

from universal_agents.config import Config
from universal_agents.tools.fs import is_protected_path


class TestIsProtectedPath(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self._tmp, ignore_errors=True))
        patcher = mock.patch.object(Config, "API_URL", "https://api.example.com/v1")
        patcher.start()
        self.addCleanup(patcher.stop)

    def _touch(self, rel: str) -> str:
        path = os.path.join(self._tmp, *rel.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").close()
        return path

    def test_name_and_extension_rules(self):
        for name in (".env", ".env.local", "id_rsa", "kubeconfig", "prod.pem", "app.yaml", "data.json",
                     "my-secret.yaml", "sealed-secrets", "credentials.json"):
            with self.subTest(name=name):
                self.assertTrue(is_protected_path(self._touch(name)))

    def test_ordinary_files_not_protected(self):
        self.assertFalse(is_protected_path(self._touch("main.py")))
        self.assertFalse(is_protected_path(self._touch("README.md")))

    def test_directories_are_not_protected(self):
        d = os.path.join(self._tmp, "config")
        os.makedirs(d)
        self.assertFalse(is_protected_path(d))

    def test_home_paths_matched_by_full_path(self):
        home = os.path.join(self._tmp, "home")
        with mock.patch("os.path.expanduser", side_effect=lambda p: p.replace("~", home, 1)):
            aws = os.path.join(home, ".aws", "credentials")
            os.makedirs(os.path.dirname(aws))
            open(aws, "w").close()
            self.assertTrue(is_protected_path("~/.aws/credentials"))

    def test_gitignore_pattern_protects_file(self):
        root = os.path.join(self._tmp, "proj")
        os.makedirs(os.path.join(root, ".git"))
        open(os.path.join(root, ".git", "HEAD"), "w").close()
        with open(os.path.join(root, ".gitignore"), "w", encoding="utf-8") as f:
            f.write("# comment\nbuild/\nnotes.txt\n")
        note = os.path.join(root, "notes.txt")
        open(note, "w").close()
        built = os.path.join(root, "build", "out.txt")
        os.makedirs(os.path.dirname(built))
        open(built, "w").close()
        with mock.patch("universal_agents.tools.fs.find_project_root", return_value=root):
            self.assertTrue(is_protected_path(note))
            self.assertTrue(is_protected_path(built))

    def test_gitignore_negation_unprotects(self):
        root = os.path.join(self._tmp, "proj")
        os.makedirs(root)
        with open(os.path.join(root, ".gitignore"), "w", encoding="utf-8") as f:
            f.write("*.log\n!keep.log\n")
        keep = os.path.join(root, "keep.log")
        open(keep, "w").close()
        with mock.patch("universal_agents.tools.fs.find_project_root", return_value=root):
            self.assertFalse(is_protected_path(keep))


class TestLocalApiDisablesProtection(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self._tmp, ignore_errors=True))
        self._env = os.path.join(self._tmp, ".env")
        open(self._env, "w").close()

    def _check(self, url: str) -> bool:
        with mock.patch("universal_agents.tools.fs.Config") as cfg:
            cfg.API_URL = url
            cfg.LOCAL_API_DISABLES_SECRET_PROTECTION = True
            return is_protected_path(self._env)

    def test_local_urls_disable_protection(self):
        for url in ("http://localhost:1234/v1", "http://127.0.0.1:1234/v1",
                    "http://192.168.50.196:1234/v1", "http://10.0.0.5/v1"):
            with self.subTest(url=url):
                self.assertFalse(self._check(url))

    def test_remote_url_keeps_protection(self):
        self.assertTrue(self._check("https://api.openai.com/v1"))

    def test_flag_off_keeps_protection_for_local(self):
        with mock.patch("universal_agents.tools.fs.Config") as cfg:
            cfg.API_URL = "http://localhost:1234/v1"
            cfg.LOCAL_API_DISABLES_SECRET_PROTECTION = False
            self.assertTrue(is_protected_path(self._env))

    def test_switching_url_takes_effect_immediately(self):
        self.assertFalse(self._check("http://localhost:1234/v1"))
        self.assertTrue(self._check("https://api.openai.com/v1"))


if __name__ == "__main__":
    unittest.main()
