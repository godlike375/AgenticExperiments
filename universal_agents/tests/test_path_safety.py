import os
import sys
import unittest
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from universal_agents.project_root import find_project_root, is_within, external_paths
from universal_agents.command_paths import extract_paths
from universal_agents.tool import tool


class TestProjectRoot(unittest.TestCase):
    def _make_repo(self, with_head=True):
        root = tempfile.mkdtemp()
        git_dir = os.path.join(root, ".git")
        os.makedirs(git_dir)
        if with_head:
            with open(os.path.join(git_dir, "HEAD"), "w") as f:
                f.write("ref: refs/heads/main\n")
        os.makedirs(os.path.join(root, "sub", "deep"))
        return root

    def test_finds_git_root(self):
        root = self._make_repo()
        deep = os.path.join(root, "sub", "deep")
        self.assertEqual(find_project_root(deep), os.path.normpath(root))

    def test_skips_empty_git_dir(self):
        # Пустой каталог .git не считается репозиторием (как и для самого git)
        root = self._make_repo(with_head=False)
        parent = tempfile.mkdtemp()
        os.rename(root, os.path.join(parent, "repo"))
        # в parent нет .git, поэтому корень не находится
        self.assertIsNone(find_project_root(os.path.join(parent, "repo")))

    def test_gitdir_pointer_file(self):
        # .git как файл-указатель gitdir: ... (субмодули/worktree)
        parent = tempfile.mkdtemp()
        repo = os.path.join(parent, "repo")
        os.makedirs(repo)
        with open(os.path.join(repo, ".git"), "w") as f:
            f.write("gitdir: /elsewhere/.git\n")
        self.assertEqual(find_project_root(repo), os.path.normpath(repo))

    def test_no_git_returns_none(self):
        tmp = tempfile.mkdtemp()
        self.assertIsNone(find_project_root(tmp))

    def test_is_within(self):
        root = self._make_repo()
        inside = os.path.join(root, "sub")
        outside = os.path.join(os.path.dirname(root), "elsewhere")
        self.assertTrue(is_within(inside, root))
        self.assertTrue(is_within(root, root))
        self.assertFalse(is_within(outside, root))

    def test_external_paths(self):
        root = self._make_repo()
        inside = os.path.join(root, "a.txt")
        outside = os.path.join(os.path.dirname(root), "b.txt")
        self.assertEqual(external_paths([inside], root), [])
        self.assertEqual(external_paths([outside], root), [outside])


class TestCommandPaths(unittest.TestCase):
    def _make_workspace(self):
        ws = tempfile.mkdtemp()
        sub = os.path.join(ws, "proj")
        os.makedirs(os.path.join(sub, ".git"))
        os.makedirs(os.path.join(sub, "dir"))
        return ws, sub

    def test_absolute_external(self):
        ws, _ = self._make_workspace()
        external = os.path.join(ws, "outside", "file.txt")
        found = extract_paths(f"Remove-Item {external}", ws)
        self.assertIn(os.path.normpath(external), found)

    def test_relative_resolves(self):
        ws, sub = self._make_workspace()
        found = extract_paths("Remove-Item ./dir/file.txt", sub)
        self.assertIn(os.path.normpath(os.path.join(sub, "dir", "file.txt")), found)

    def test_parent_dir(self):
        ws, sub = self._make_workspace()
        target = os.path.join(ws, "outside.txt")
        found = extract_paths("Remove-Item ../outside.txt", sub)
        self.assertIn(os.path.normpath(target), found)

    def test_no_path_flags(self):
        self.assertEqual(extract_paths("git status", "."), [])
        self.assertEqual(extract_paths("echo hello", "."), [])
        self.assertEqual(extract_paths("pip install requests", "."), [])
        self.assertEqual(extract_paths("ls -la", "."), [])
        self.assertEqual(extract_paths("Get-Process -Name notepad", "."), [])

    def test_quoted_space_path(self):
        ws, _ = self._make_workspace()
        target = os.path.join(ws, "Some App", "x.txt")
        found = extract_paths(f"Remove-Item '{target}'", ws)
        self.assertIn(os.path.normpath(target), found)

    def test_env_vars(self):
        ws, _ = self._make_workspace()
        os.environ["UA_TESTDIR"] = ws
        found = extract_paths("Remove-Item %UA_TESTDIR%/x.txt", ws)
        self.assertIn(os.path.normpath(os.path.join(ws, "x.txt")), found)
        found2 = extract_paths("Remove-Item ${UA_TESTDIR}/y.txt", ws)
        self.assertIn(os.path.normpath(os.path.join(ws, "y.txt")), found2)
        found3 = extract_paths("Remove-Item $env:UA_TESTDIR\\z.txt", ws)
        self.assertIn(os.path.normpath(os.path.join(ws, "z.txt")), found3)

    def test_flags_not_paths(self):
        ws, _ = self._make_workspace()
        # опции не должны давать пути
        self.assertEqual(extract_paths(f"Remove-Item -Path {ws}\\x.txt -Force", ws),
                         [os.path.normpath(os.path.join(ws, "x.txt"))])


class TestToolPathSafety(unittest.TestCase):
    def test_path_safety_flag(self):
        @tool(description="t", path_safety=True)
        def my_tool():
            return "ok"
        self.assertTrue(my_tool._path_safety)


if __name__ == "__main__":
    unittest.main()
