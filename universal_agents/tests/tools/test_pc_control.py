ef """Тесты PC-инструментов (tools/pc_control.py): координаты, схемы, allow-list.

GUI замокан (ImageGrab / pyautogui / pyperclip) — тесты не трогают реальный экран.
"""

from __future__ import annotations

import base64
import ctypes
import inspect
import io
import sys
import unittest
from unittest.mock import patch

from PIL import Image

from universal_agents import screen_state
from universal_agents.constants import ENVIRONMENT_PREFIX
from universal_agents.tool import ToolOutput
from universal_agents.tools import pc_control

_IS_WIN32 = sys.platform == "win32"
PC_TOOL_NAMES = ["screenshot", "mouse_move", "mouse_click", "scroll", "type_text", "press_key"]


def _fake_screen(w: int = 1920, h: int = 1080) -> Image.Image:
    return Image.new("RGB", (w, h), (30, 30, 30))


class PCControlTestCase(unittest.TestCase):
    """Общий state (screen_state.last_scale) сохраняется между тестами."""

    def setUp(self):
        prev = screen_state.last_scale
        self.addCleanup(lambda: setattr(screen_state, "last_scale", prev))
        screen_state.last_scale = None


class TestTakeScreenshot(PCControlTestCase):
    def test_resizes_and_stores_scale(self):
        with patch.object(pc_control.ImageGrab, "grab", return_value=_fake_screen()), \
                patch.object(pc_control.pyautogui, "position", return_value=(960, 540)):
            b64, ow, oh, iw, ih, scale = pc_control.take_screenshot()
        self.assertEqual((ow, oh), (1920, 1080))
        self.assertEqual((iw, ih), (960, 540))
        self.assertAlmostEqual(scale, 0.5)
        self.assertEqual(screen_state.last_scale, 0.5)
        img = Image.open(io.BytesIO(base64.b64decode(b64)))
        self.assertEqual(img.size, (960, 540))

    def test_small_screen_keeps_original_size(self):
        with patch.object(pc_control.ImageGrab, "grab", return_value=_fake_screen(800, 600)), \
                patch.object(pc_control.pyautogui, "position", return_value=(10, 20)):
            _, ow, oh, iw, ih, scale = pc_control.take_screenshot()
        self.assertEqual((ow, oh, iw, ih), (800, 600, 800, 600))
        self.assertEqual(scale, 1.0)

    def test_draw_grid_marks_grid_lines(self):
        img = Image.new("RGB", (400, 300), (0, 0, 0))
        pc_control.draw_grid(img, cursor_x=200, cursor_y=150)
        line_pixels = [img.getpixel((x, 50)) for x in (99, 100, 101)]
        self.assertIn(pc_control.COLORS["grid"], line_pixels)


class TestScaleConversion(PCControlTestCase):
    def test_fallback_scale_from_screen_size(self):
        with patch.object(pc_control.pyautogui, "size", return_value=(1920, 1080)):
            self.assertAlmostEqual(pc_control._current_scale(), 0.5)

    def test_fallback_scale_when_screen_small(self):
        with patch.object(pc_control.pyautogui, "size", return_value=(800, 600)):
            self.assertEqual(pc_control._current_scale(), 1.0)

    def test_stored_scale_wins(self):
        screen_state.last_scale = 0.375
        self.assertEqual(pc_control._current_scale(), 0.375)

    def test_zero_stored_scale_falls_back(self):
        screen_state.last_scale = 0
        with patch.object(pc_control.pyautogui, "size", return_value=(1920, 1080)):
            self.assertAlmostEqual(pc_control._current_scale(), 0.5)

    def test_to_screen_divides_by_scale(self):
        screen_state.last_scale = 0.5
        self.assertEqual(pc_control._to_screen(480, 270), (960, 540))
        self.assertEqual(pc_control._to_screen(0, 0), (0, 0))


class TestMouseTools(PCControlTestCase):
    def setUp(self):
        super().setUp()
        screen_state.last_scale = 0.5

    def test_mouse_move_converts_coordinates(self):
        with patch.object(pc_control.pyautogui, "moveTo") as move:
            res = pc_control.mouse_move(100, 200)
        move.assert_called_once_with(200, 400, duration=pc_control.Config.MOUSE_MOVE_DURATION)
        self.assertTrue(res.startswith("Курсор перемещён"))

    def test_mouse_click_left_default(self):
        with patch.object(pc_control.pyautogui, "click") as click, \
                patch.object(pc_control.pyautogui, "rightClick") as right, \
                patch.object(pc_control.pyautogui, "doubleClick") as dbl:
            res = pc_control.mouse_click(100, 200)
        click.assert_called_once_with(200, 400)
        right.assert_not_called()
        dbl.assert_not_called()
        self.assertTrue(res.startswith("left click"))

    def test_mouse_click_right(self):
        with patch.object(pc_control.pyautogui, "rightClick") as right:
            res = pc_control.mouse_click(10, 20, button="right")
        right.assert_called_once_with(20, 40)
        self.assertTrue(res.startswith("right click"))

    def test_mouse_click_double(self):
        with patch.object(pc_control.pyautogui, "doubleClick") as dbl:
            res = pc_control.mouse_click(10, 20, double=True)
        dbl.assert_called_once_with(20, 40, button="left")
        self.assertTrue(res.startswith("double left"))

    def test_mouse_click_invalid_button_is_error(self):
        with patch.object(pc_control.pyautogui, "click") as click:
            res = pc_control.mouse_click(10, 20, button="x")
        click.assert_not_called()
        self.assertTrue(res.startswith(f"{ENVIRONMENT_PREFIX} Error"))
        self.assertIn("left/right/middle", res)

    def test_scroll_at_point_converts(self):
        with patch.object(pc_control.pyautogui, "scroll") as scroll:
            res = pc_control.scroll(3, 100, 200)
        scroll.assert_called_once_with(3, 200, 400)
        self.assertIn("вверх", res)

    def test_scroll_under_cursor(self):
        with patch.object(pc_control.pyautogui, "scroll") as scroll:
            res = pc_control.scroll(-3)
        scroll.assert_called_once_with(-3, None, None)
        self.assertIn("вниз", res)
        self.assertIn("под курсором", res)

    def test_scroll_partial_coords_is_error(self):
        with patch.object(pc_control.pyautogui, "scroll") as scroll:
            res = pc_control.scroll(3, x=10)
        scroll.assert_not_called()
        self.assertTrue(res.startswith(f"{ENVIRONMENT_PREFIX} Error"))


class TestKeyResolution(unittest.TestCase):
    """Раскладка не должна ломать раскладку клавиш: буквы — scancode, именованные — VK."""

    def test_letters_resolve_to_scancodes(self):
        self.assertEqual(pc_control._resolve_key("v"), ("scan", 0x2F, 0))
        self.assertEqual(pc_control._resolve_key("A"), ("scan", 0x1E, 0))
        self.assertEqual(pc_control._resolve_key("5"), ("scan", 0x06, 0))

    def test_modifiers_resolve_to_vk(self):
        self.assertEqual(pc_control._resolve_key("ctrl"), ("vk", 0x11, 0))
        self.assertEqual(pc_control._resolve_key("Shift"), ("vk", 0x10, 0))
        self.assertEqual(pc_control._resolve_key("win"), ("vk", 0x5B, 0))

    def test_named_keys_use_vk(self):
        self.assertEqual(pc_control._resolve_key("enter"), ("vk", 0x0D, 0))
        self.assertEqual(pc_control._resolve_key("esc"), ("vk", 0x1B, 0))
        self.assertEqual(pc_control._resolve_key("f5"), ("vk", 0x74, 0))

    def test_extended_keys_flagged(self):
        for name in ("up", "down", "left", "right", "home", "end", "delete", "insert"):
            mode, _code, flags = pc_control._resolve_key(name)
            self.assertEqual(mode, "vk", name)
            self.assertEqual(flags, pc_control._KEYEVENTF_EXTENDEDKEY, name)

    def test_unknown_key_is_none(self):
        self.assertIsNone(pc_control._resolve_key("qwerty123"))
        self.assertIsNone(pc_control._resolve_key(""))


class TestKeyboardTools(unittest.TestCase):
    def _sent(self, result_patcher):
        """Список (mode, code, flags, up) реально отправленных событий."""
        calls = []

        def fake(mode, code, flags, up):
            calls.append((mode, code, flags, up))
            return result_patcher

        return calls, fake

    def test_press_key_single_taps_vk(self):
        calls, fake = self._sent(True)
        with patch.object(pc_control, "_send_key_event", side_effect=fake), \
                patch.object(pc_control.time, "sleep"):
            res = pc_control.press_key("Enter")
        self.assertEqual(calls, [("vk", 0x0D, 0, False), ("vk", 0x0D, 0, True)])
        self.assertEqual(res, "Нажато: enter.")

    def test_press_key_combination_order(self):
        """Модификатор вниз → клавиша вниз → клавиша вверх → модификатор вверх."""
        calls, fake = self._sent(True)
        with patch.object(pc_control, "_send_key_event", side_effect=fake), \
                patch.object(pc_control.time, "sleep"):
            res = pc_control.press_key("ctrl+v")
        self.assertEqual(
            calls,
            [("vk", 0x11, 0, False), ("scan", 0x2F, 0, False),
             ("scan", 0x2F, 0, True), ("vk", 0x11, 0, True)],
        )
        self.assertEqual(res, "Нажато: ctrl+v.")

    def test_press_key_paste_uses_new_layer(self):
        """Ctrl+V: модификатор по VK, буква — по scancode (без pyautogui-маппинга)."""
        with patch.object(pc_control, "_send_key_event", return_value=True) as send, \
                patch.object(pc_control.time, "sleep"):
            pc_control.press_key("ctrl+v")
        self.assertEqual(
            [(c.args[0], c.args[1]) for c in send.call_args_list],
            [("vk", 0x11), ("scan", 0x2F), ("scan", 0x2F), ("vk", 0x11)],
        )

    def test_press_key_empty_is_error(self):
        with patch.object(pc_control, "_send_key_event") as send:
            res = pc_control.press_key(" + ")
        send.assert_not_called()
        self.assertTrue(res.startswith(f"{ENVIRONMENT_PREFIX} Error"))

    def test_press_key_unknown_key_is_error(self):
        with patch.object(pc_control, "_send_key_event") as send:
            res = pc_control.press_key("ctrl+qwerty")
        send.assert_not_called()
        self.assertTrue(res.startswith(f"{ENVIRONMENT_PREFIX} Error"))

    def test_press_key_reports_rejected_input(self):
        with patch.object(pc_control, "_send_key_event", return_value=False), \
                patch.object(pc_control.time, "sleep"):
            res = pc_control.press_key("enter")
        self.assertTrue(res.startswith(f"{ENVIRONMENT_PREFIX} Error"))
        self.assertIn("SendInput", res)

    def test_type_text_pastes_and_enters(self):
        calls, fake = self._sent(True)
        with patch.object(pc_control.pyperclip, "copy") as copy, \
                patch.object(pc_control, "_send_key_event", side_effect=fake), \
                patch.object(pc_control.time, "sleep"):
            res = pc_control.type_text_on_keyboard("Привет мир", press_enter=True)
        copy.assert_called_once_with("Привет мир")
        self.assertEqual(
            calls,
            [("vk", 0x11, 0, False), ("scan", 0x2F, 0, False),
             ("scan", 0x2F, 0, True), ("vk", 0x11, 0, True),
             ("vk", 0x0D, 0, False), ("vk", 0x0D, 0, True)],
        )
        self.assertIn("10 символов", res)
        self.assertIn("Enter", res)

    def test_type_text_empty_is_error(self):
        with patch.object(pc_control.pyperclip, "copy") as copy:
            res = pc_control.type_text_on_keyboard("")
        copy.assert_not_called()
        self.assertTrue(res.startswith(f"{ENVIRONMENT_PREFIX} Error"))

    def test_type_text_does_not_touch_focus(self):
        """type_text не управляет фокусом: только буфер + Ctrl+V в активное окно."""
        src = inspect.getsource(pc_control.type_text_on_keyboard)
        for forbidden in ("SetForegroundWindow", "activate", "SwitchToThisWindow",
                          "AttachThreadInput", "pygetwindow"):
            self.assertNotIn(forbidden, src)

    def test_keyboard_never_uses_pyautogui(self):
        """pyautogui на не-US раскладке шлёт мусорные VK — ввод только через SendInput."""
        for fn in (pc_control.type_text_on_keyboard, pc_control.press_key):
            self.assertNotIn("pyautogui", inspect.getsource(fn))

    def test_send_key_event_retries_then_reports_failure(self):
        if not _IS_WIN32:
            self.skipTest("Windows-only")
        with patch.object(pc_control, "_user32") as user32, \
                patch.object(pc_control.time, "sleep"):
            user32.SendInput.return_value = 0
            self.assertFalse(pc_control._send_key_event("vk", 0x0D, 0, up=False))
            self.assertEqual(user32.SendInput.call_count, pc_control._SEND_RETRIES)
            user32.SendInput.return_value = 1
            self.assertTrue(pc_control._send_key_event("vk", 0x0D, 0, up=False))

    def test_send_key_event_builds_scancode_event(self):
        if not _IS_WIN32:
            self.skipTest("Windows-only")
        seen = {}

        def capture(count, pointer, size):
            event = ctypes.cast(pointer, ctypes.POINTER(pc_control._INPUT)).contents
            seen["wVk"] = event.union.ki.wVk
            seen["wScan"] = event.union.ki.wScan
            seen["flags"] = event.union.ki.dwFlags
            seen["size"] = size
            return 1

        with patch.object(pc_control, "_user32") as user32:
            user32.SendInput.side_effect = capture
            self.assertTrue(pc_control._send_key_event("scan", 0x2F, 0, up=True))
        self.assertEqual(seen["wVk"], 0)
        self.assertEqual(seen["wScan"], 0x2F)
        self.assertEqual(seen["flags"], pc_control._KEYEVENTF_SCANCODE | pc_control._KEYEVENTF_KEYUP)


class TestScreenshotToolOutput(PCControlTestCase):
    def test_returns_tool_output_with_image(self):
        fake_b64 = base64.b64encode(b"jpeg-bytes").decode()
        with patch.object(pc_control, "take_screenshot",
                          return_value=(fake_b64, 2560, 1440, 960, 540, 0.375)), \
                patch.object(pc_control.pyautogui, "position", return_value=(100, 200)):
            out = pc_control.screenshot()
        self.assertIsInstance(out, ToolOutput)
        self.assertEqual(out.images, [fake_b64])
        self.assertIn("scale=0.375", out.text)
        self.assertIn("960x540", out.text)


class TestSchemas(unittest.TestCase):
    def test_all_tools_have_schema_and_short_description(self):
        for name in PC_TOOL_NAMES:
            fn = getattr(pc_control, name)
            self.assertTrue(getattr(fn, "_is_tool", False), name)
            schema = fn._tool_schema
            self.assertEqual(schema["function"]["name"], name)
            self.assertTrue(getattr(fn, "_short_description", ""), name)

    def test_required_and_optional_params(self):
        params = pc_control.mouse_click._tool_schema["function"]["parameters"]
        self.assertEqual(params["required"], ["x", "y"])
        for opt in ("button", "double"):
            self.assertTrue(params["properties"][opt]["description"].lower().startswith("optional"), opt)

    def test_type_text_required_and_optional_params(self):
        params = pc_control.type_text_on_keyboard._tool_schema["function"]["parameters"]
        self.assertEqual(params["required"], ["text"])
        self.assertEqual(set(params["properties"]), {"text", "press_enter"})
        self.assertTrue(params["properties"]["press_enter"]["description"].lower().startswith("optional"))

    def test_screenshot_takes_no_params(self):
        params = pc_control.screenshot._tool_schema["function"]["parameters"]
        self.assertEqual(params.get("properties", {}), {})


class TestAllowListIntegration(unittest.TestCase):
    def test_all_six_tools_allowed(self):
        from universal_agents.main import LOADABLE_TOOLS, PRELOADED_TOOLS, build_allowed_tools
        allowed = build_allowed_tools(LOADABLE_TOOLS, PRELOADED_TOOLS)
        for name in PC_TOOL_NAMES:
            self.assertIn(name, allowed, name)

    def test_load_via_tool_manager(self):
        from universal_agents.main import LOADABLE_TOOLS, PRELOADED_TOOLS, build_allowed_tools
        from universal_agents.tool_manager import ToolManager
        tm = ToolManager(tools_config=build_allowed_tools(LOADABLE_TOOLS, PRELOADED_TOOLS))
        for name in PC_TOOL_NAMES:
            res = tm.load(name)
            self.assertTrue(res.endswith("loaded."), f"{name}: {res}")
            self.assertIn(name, tm.tools_map)


if __name__ == "__main__":
    unittest.main()
