import unittest

from universal_agents.archive import HistoryArchive
from universal_agents.history import ChatHistory
from universal_agents.models import (
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolCall,
    ToolResult,
)
from universal_agents.tools.fs import line_range_edit as _line_range_edit_tool
from universal_agents.tools.fs import read as _read_tool

B64 = "aGVsbG8taW1hZ2U="


def history_with_dialog():
    h = ChatHistory("sys")
    h.add(UserMessage("u1"))
    h.add(AssistantMessage(content="a1", tool_calls=[ToolCall(id="t1", name=_read_tool.__name__, arguments="{}")]))
    h.add(ToolResult.success("t1", _read_tool.__name__, "data"))
    h.add(AssistantMessage(content="a2"))
    h.add(UserMessage("u2"))
    h.add(AssistantMessage(content="a3"))
    return h


class TestChatHistory(unittest.TestCase):
    def test_initial_state(self):
        h = ChatHistory("sys")
        self.assertEqual(len(h), 1)
        self.assertIsInstance(h[0], SystemMessage)
        self.assertIsNone(h.get_last_message())

    def test_get_all_api(self):
        h = history_with_dialog()
        api = h.get_all_api()
        self.assertEqual(api[0], {"role": "system", "content": "sys"})
        self.assertEqual(api[1]["role"], "user")
        self.assertEqual(api[3]["role"], "tool")

    def test_remove_at(self):
        h = history_with_dialog()
        h.remove_at({3})  # tool result
        roles = [m.to_api_dict()["role"] for m in h]
        self.assertNotIn("tool", roles)
        self.assertEqual(len(h), 6)

    def test_replace_range(self):
        h = history_with_dialog()
        h.replace_range(1, 2, [UserMessage("summary")])
        self.assertEqual(len(h), 6)
        self.assertEqual(h[1].content, "summary")

    def test_normalize_keeps_sequence(self):
        h = history_with_dialog()
        h.normalize()
        roles = [m.to_api_dict()["role"] for m in h]
        self.assertEqual(roles, ["system", "user", "assistant", "tool", "assistant", "user", "assistant"])

    def test_normalize_merges_consecutive_same_type(self):
        h = ChatHistory("sys")
        h.add(UserMessage("u1"))
        h.add(UserMessage("u2"))
        h.add(AssistantMessage(content="a1"))
        h.add(AssistantMessage(content="a2"))
        h.normalize()
        roles = [m.to_api_dict()["role"] for m in h]
        self.assertEqual(roles, ["system", "user", "assistant"])
        self.assertEqual(h[1].content, "u1\n\nu2")

    def test_normalize_drops_orphan_tool_result(self):
        h = ChatHistory("sys")
        h.add(UserMessage("u1"))
        h.add(ToolResult.success("t1", _read_tool.__name__, "data"))
        h.normalize()
        roles = [m.to_api_dict()["role"] for m in h]
        self.assertEqual(roles, ["system", "user"])

    def test_delete_range_resyncs_new_last_user_header(self):
        """Удаление последнего вопроса возвращает «последность» к предыдущему — его шапку
        пересобираем, чтобы бюджет был актуален (прежняя цифра устарела/отсутствует).
        Пересборка бесплатна: удаление инвалидирует префикс от своей точки."""
        h = history_with_dialog()  # 0 sys, 1 u1, 2 a1, 3 tool, 4 a2, 5 u2, 6 a3
        h[1]._cached_header = "cached-u1"
        h.delete_range(5, 6)  # удаляем u2 и a3 → последним остаётся u1
        self.assertIsNone(h[1]._cached_header)

    def test_delete_range_before_last_user_resyncs_header(self):
        """Удаление сообщений ПЕРЕД вопросом меняет его префикс и токен-бюджет — шапку
        пересобираем (KV от точки удаления и так мёртв)."""
        h = history_with_dialog()
        h[2]._cached_header = "cached-u2"  # u2 после удаления станет индексом 2
        h.delete_range(2, 4)  # удаляем a1+tool+a2 (перед u2)
        self.assertIsNone(h[2]._cached_header)

    def test_remove_at_after_last_user_keeps_its_header(self):
        """Удаление строго ПОСЛЕ вопроса (напр. хвостовая заглушка или наг guard'а) шапку
        вопроса не трогает: его префикс цел, сдвиг волатильной цифры сбросил бы KV всего
        хода (§1.2)."""
        h = history_with_dialog()
        h[5]._cached_header = "cached-u2"
        h.remove_at({6})  # a3 — после u2
        self.assertEqual(h[5]._cached_header, "cached-u2")

    def test_add_preserves_previous_last_user_header(self):
        """Добавление нового user-сообщения не трогает кэш старого последнего: собственные
        байта старого не менялись, а остаток бюджета в его шапке — снимок на момент
        постановки вопроса (осознанный платёж за KV-стабильность)."""
        h = history_with_dialog()
        h[5]._cached_header = "cached-u2"
        h.add(UserMessage("u3"))
        self.assertEqual(h[5]._cached_header, "cached-u2")

    def test_remove_at_resyncs_surviving_user_header(self):
        h = history_with_dialog()
        h[1]._cached_header = "cached-u1"
        h.remove_at({5, 6})  # последнее user u2 и его ответ удалены → u1 снова текущий
        self.assertIsNone(h[1]._cached_header)

    def test_edit_message_resets_modified_user_header(self):
        """Обратная сторона правила: у сообщения, которое МЫ ИЗМЕНИЛИ, кэш сбрасывается —
        его KV и так мёртв, пересборка шапки ничего не стоит (§1.2)."""
        h = history_with_dialog()
        h[1]._cached_header = "cached-u1"
        h[5]._cached_header = "cached-u2"
        h.edit_message(5, "u2-исправлено", old_text="u2")
        self.assertIsNone(h[5]._cached_header)
        self.assertEqual(h[1]._cached_header, "cached-u1")

    def test_compress_old_messages(self):
        h = history_with_dialog()
        h[5]._cached_header = "cached-u2"
        h.compress_old_messages("long summary", preserve_last=2)
        self.assertEqual(h[0].to_api_dict()["role"], "system")
        self.assertTrue("summary" in h[1].content)
        self.assertEqual(h[-1].content, "a3")
        self.assertIsNone(h[-2]._cached_header)

    def test_save_load_roundtrip(self, tmpdir=None):
        import tempfile, os
        h = history_with_dialog()
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "hist.json")
            h.save(path, loaded_tools=[_read_tool.__name__, _line_range_edit_tool.__name__], file_states={"a.py": {"disk_hash": "x", "content_hash": "y", "tool_call_id": "t1"}})
            h2 = ChatHistory("sys")
            tools, file_states, summaries = h2.load(path)
            self.assertEqual(tools, [_read_tool.__name__, _line_range_edit_tool.__name__])
            self.assertEqual(file_states["a.py"]["disk_hash"], "x")
            self.assertEqual(summaries, [])
            self.assertEqual(len(h2), len(h))


class TestNormalizeMergeImages(unittest.TestCase):
    """normalize(): склейка соседних user-сообщений не теряет картинки."""

    def _merged_users(self, first: UserMessage, second: UserMessage) -> list[UserMessage]:
        h = ChatHistory("sys")
        h.add(first)
        h.add(second)
        h.normalize()
        return [m for m in h.get_all() if isinstance(m, UserMessage)]

    def test_image_of_first_message_survives(self):
        users = self._merged_users(UserMessage("до", images=[B64]), UserMessage("после"))
        self.assertEqual(len(users), 1)
        self.assertIn("до", users[0].content)
        self.assertIn("после", users[0].content)
        self.assertEqual(users[0].images, [B64])

    def test_image_of_second_message_survives(self):
        users = self._merged_users(UserMessage("до"), UserMessage("после", images=[B64]))
        self.assertEqual(len(users), 1)
        self.assertEqual(users[0].images, [B64])

    def test_both_images_concatenated(self):
        users = self._merged_users(
            UserMessage("до", images=[B64]), UserMessage("после", images=["eA=="])
        )
        self.assertEqual(users[0].images, [B64, "eA=="])


def _archive_dialog() -> list:
    h = ChatHistory("sys")
    h.add(UserMessage("please fix the login bug in src/auth.py"))
    h.add(AssistantMessage(
        content="Investigating the auth flow.",
        tool_calls=[ToolCall(id="t1", name=_read_tool.__name__, arguments='{"path": "src/auth.py"}')],
    ))
    h.add(ToolResult.success("t1", _read_tool.__name__, "def login(): raise ValueError('bad token')"))
    h.add(AssistantMessage(content="Found it: token expiry check inverted. Fixed and added test."))
    h.add(UserMessage("why did you touch the middleware?"))
    return h.get_all()[1:]


class TestArchiveBasics(unittest.TestCase):
    def test_append_and_len(self):
        a = HistoryArchive()
        msgs = _archive_dialog()
        self.assertEqual(a.append_messages(msgs), len(msgs))
        self.assertEqual(len(a), 5)

    def test_search_finds_content(self):
        a = HistoryArchive()
        a.append_messages(_archive_dialog())
        out = a.search("login bug")
        self.assertIn("seq=", out)
        self.assertIn("auth.py", out)

    def test_search_by_tool_call_arguments(self):
        a = HistoryArchive()
        a.append_messages(_archive_dialog())
        out = a.search("src/auth.py")
        self.assertIn("ASSISTANT", out)  # вызов инструмента виден в контенте ассистента

    def test_role_filter(self):
        a = HistoryArchive()
        a.append_messages(_archive_dialog())
        only_user = a.search("middleware", role="user")
        self.assertIn("USER", only_user)
        none_assistant = a.search("middleware", role="assistant")
        self.assertIn("No matches", none_assistant)

    def test_no_matches_message(self):
        a = HistoryArchive()
        a.append_messages(_archive_dialog())
        self.assertIn("No matches", a.search("quantum entanglement"))

    def test_empty_query(self):
        self.assertIn("Empty", HistoryArchive().search(""))


class TestReadSpan(unittest.TestCase):
    def setUp(self):
        self.archive = HistoryArchive()
        self.msgs = _archive_dialog()
        self.archive.append_messages(self.msgs)

    def test_read_range_inclusive(self):
        out = self.archive.read_span(2, 4)
        self.assertIn("#2", out)
        self.assertIn("#3", out)
        self.assertIn("#4", out)
        self.assertNotIn("#5 USER", out.split("#5")[1] if "#5" in out else "")

    def test_read_missing_range_hint(self):
        out = self.archive.read_span(100, 200)
        self.assertIn("Archived range covers seq 1..5", out.replace("\n", " "))

    def test_max_chars_truncation(self):
        out = self.archive.read_span(2, 6, max_chars=120)
        self.assertLessEqual(len(out), 400)


if __name__ == "__main__":
    unittest.main()
