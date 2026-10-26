import os
import tempfile
import unittest
from unittest import mock

from universal_agents.tools.fs import is_protected_path


class TestIsProtectedPath(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self._tmp, ignore_errors=True))

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


if __name__ == "__main__":
    unittest.main()
