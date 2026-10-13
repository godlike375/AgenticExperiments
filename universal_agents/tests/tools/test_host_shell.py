import unittest
from unittest import mock

from universal_agents.tools.host_shell import run_powershell


class TestRunPowershellEncoding(unittest.TestCase):
    def _run(self, which_map, command="Write-Output 'тест'"):
        captured = {}

        def fake_which(name):
            return which_map.get(name)

        def fake_run_capture(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return "ok"

        with mock.patch("universal_agents.tools.host_shell.shutil.which", side_effect=fake_which), \
                mock.patch("universal_agents.tools.host_shell.run_capture", side_effect=fake_run_capture):
            result = run_powershell(command)
        return result, captured

    def test_pwsh_command_is_not_prefixed(self):
        """PowerShell 7+ отдаёт UTF-8 в pipe — команду нельзя подмешивать префиксом,
        иначе меняется её семантика."""
        _, captured = self._run({"pwsh": r"C:\pwsh.exe"}, command="Get-Process")
        self.assertEqual(captured["args"][0][-1], "Get-Process")
        self.assertNotIn("fallback_encoding", captured["kwargs"])

    def test_legacy_powershell_uses_oem_fallback(self):
        """PowerShell 5.1 в pipe пишет OEM-кодировку (cp866 на ru-RU). Команду не трогаем,
        а просим run_capture декодировать с фолбэком на OEM."""
        _, captured = self._run({"powershell": r"C:\powershell.exe"}, command="Write-Output 'тест'")
        self.assertEqual(captured["args"][0][-1], "Write-Output 'тест'")
        self.assertEqual(captured["kwargs"].get("fallback_encoding"), "oem")

    def test_missing_executable_returns_error(self):
        result, captured = self._run({})
        self.assertIn("not found", result)
        self.assertNotIn("args", captured)


if __name__ == "__main__":
    unittest.main()
