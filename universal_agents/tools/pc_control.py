"""PC-control инструменты: скриншот с сеткой 4x4, мышь, колесо, клавиатура.

Перенос screen_agent.py на рельсы universal_agents (@tool + ToolOutput с картинкой).
Координаты в аргументах mouse/scroll-инструментов — В ПИКСЕЛЯХ ИЗОБРАЖЕНИЯ
последнего скриншота (ровно те, что подписаны на сетке); конвертация image→screen
идёт по масштабу последнего скриншота (screen_state.last_scale).

FAILSAFE: перемести мышь в левый верхний угол экрана — pyautogui бросит исключение,
его перехватит ExecuteMixin и вернёт модели ошибку.
"""

from __future__ import annotations

import base64
import ctypes
import io
import sys
import time

import pyautogui
import pyperclip
from PIL import Image, ImageDraw, ImageFont, ImageGrab

from universal_agents import screen_state
from universal_agents.config import Config
from universal_agents.constants import err
from universal_agents.tool import tool, ToolOutput

pyautogui.FAILSAFE = True
pyautogui.PAUSE = Config.PYAUTOGUI_PAUSE

QUADRANT_LABELS = ["A", "B", "C", "D"]
COLORS = {
    "grid": (255, 80, 80),
    "text_bg": (0, 0, 0),
    "text_fg": (255, 255, 255),
    "coord": (0, 255, 255),
    "quadrant_label": (255, 255, 0),
    "axis": (255, 200, 0),
}


# ─── Скриншот с сеткой (перенос из screen_agent.py) ─────────────────────────

def take_screenshot() -> tuple[str, int, int, int, int, float]:
    """Делает скриншот с наложенной сеткой.
    Возвращает (base64_jpeg, original_w, original_h, img_w, img_h, scale)."""
    cursor_x, cursor_y = pyautogui.position()
    img = ImageGrab.grab()
    w, h = img.size
    max_width = Config.SCREENSHOT_MAX_WIDTH
    scale = 1.0
    if w > max_width:
        scale = max_width / w
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    draw_grid(img, round(cursor_x * scale), round(cursor_y * scale))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=Config.SCREENSHOT_QUALITY)
    if Config.SCREENSHOT_DEBUG_PATH:
        try:
            img.save(Config.SCREENSHOT_DEBUG_PATH, format="JPEG")
        except Exception as e:
            print(f"[screenshot] debug save failed: {e}")
    b64 = base64.b64encode(buf.getvalue()).decode()
    img_w, img_h = img.size
    # Масштаб фиксируется здесь же: любой вызовующий (не только @tool-обёртка)
    # обновляет координатное пространство для mouse-инструментов.
    screen_state.last_scale = scale
    return b64, w, h, img_w, img_h, scale


def _get_font(size=14):
    for s in (size, size - 2, size - 4):
        try:
            return ImageFont.truetype("arial.ttf", s)
        except (IOError, OSError):
            continue
    try:
        return ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", size)
    except (IOError, OSError):
        return ImageFont.load_default()


def draw_grid(img: Image.Image, cursor_x=None, cursor_y=None) -> None:
    """Рисует сетку 4x4 с подписями координат, квадрантов и курсором."""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    font = _get_font(11)
    font_small = _get_font(10)
    lw = max(2, min(w, h) // 300)

    # ── Линии сетки ──
    for i in range(1, 4):
        x = int(w * i / 4)
        draw.line([(x, 0), (x, h)], fill=COLORS["grid"], width=lw)
        y = int(h * i / 4)
        draw.line([(0, y), (w, y)], fill=COLORS["grid"], width=lw)

    # ── Подписи квадрантов (A1, A2, ..., D4) ──
    for row in range(4):
        for col in range(4):
            x1 = int(w * col / 4)
            y1 = int(h * row / 4)
            cx = x1 + int(w / 8)
            cy = y1 + int(h / 8)
            label = f"{QUADRANT_LABELS[row]}{col + 1}"
            bbox = draw.textbbox((0, 0), label, font=font_small)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.rectangle(
                [cx - tw // 2 - 3, cy - th // 2 - 2, cx + tw // 2 + 3, cy + th // 2 + 2],
                fill=(0, 0, 0, 180),
            )
            draw.text((cx - tw // 2, cy - th // 2), label, fill=COLORS["quadrant_label"], font=font_small)

    # ── Координаты углов каждого квадранта ──
    for row in range(4):
        for col in range(4):
            x1 = int(w * col / 4) + 2
            y1 = int(h * row / 4) + 2
            text = f"({x1},{y1})"
            bbox = draw.textbbox((0, 0), text, font=font_small)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.rectangle(
                [x1 - 1, y1 - 1, x1 + tw + 2, y1 + th + 2],
                fill=(0, 0, 0, 160),
            )
            draw.text((x1 + 1, y1), text, fill=COLORS["coord"], font=font_small)

    # ── Курсор мыши ──
    if cursor_x is not None and cursor_y is not None:
        cr = max(5, min(w, h) // 80)
        draw.ellipse(
            [cursor_x - cr, cursor_y - cr, cursor_x + cr, cursor_y + cr],
            outline=(0, 255, 0),
            width=max(2, cr // 2),
        )
        draw.ellipse(
            [cursor_x - 2, cursor_y - 2, cursor_x + 2, cursor_y + 2],
            fill=(0, 255, 0),
        )
        label = f"cursor ({cursor_x},{cursor_y})"
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        lx = cursor_x + cr + 4
        ly = cursor_y - th - 4
        if lx + tw > w:
            lx = cursor_x - tw - cr - 4
        if ly < 0:
            ly = cursor_y + cr + 4
        draw.rectangle(
            [lx - 2, ly - 2, lx + tw + 2, ly + th + 2],
            fill=(0, 40, 0, 200),
        )
        draw.text((lx, ly), label, fill=(0, 255, 0), font=font)


# ─── Координаты: изображение → экран ─────────────────────────────────────────

def _current_scale() -> float:
    """Масштаб перевода координат изображения в экранные.

    Приоритет — у масштаба последнего реального скриншота (он знает точную ширину
    захвата, включая мониторы); до первого скриншота считается тем же фингулем,
    что и take_screenshot."""
    s = screen_state.last_scale
    if s is not None and s > 0:
        return s
    w, _ = pyautogui.size()
    return Config.SCREENSHOT_MAX_WIDTH / w if w > Config.SCREENSHOT_MAX_WIDTH else 1.0


def _to_screen(x: int, y: int) -> tuple[int, int]:
    """Координаты изображения (пиксели сетки) → координаты экрана."""
    scale = _current_scale()
    return round(x / scale), round(y / scale)


# ─── Клавиатура: SendInput напрямую, минуя pyautogui ────────────────────────
#
# pyautogui на Windows раскладывает буквы через VkKeyScan текущей раскладки: на русской
# клавиатуре 'a' и 'v' дают -1, и вместо клавиш в очередь ввода уходят мусорные
# VK-коды (255/18/17/16) — Ctrl+V не срабатывает ни в чём. Поэтому события ввода
# отправляем сами:
#   • модификаторы и именованные клавиши — по VK: комбинации (ctrl+v, alt+tab) в любой
#     раскладке остаются одними и теми же, потому что акселератор приложения смотрит на VK;
#   • печатные символы — по scancode физической клавиши (US set-1), чтобы не зависеть
#     от VkKeyScan (но сам символ, как и у живого человека, печатается по текущей
#     раскладке — поэтому текст tool'а идёт через буфер обмена).
# Раскладку и фокус не меняем: ввод уходит в то окно, которое сейчас в фокусе.

_IS_WINDOWS = sys.platform == "win32"

_INPUT_KEYBOARD = 1
_KEYEVENTF_EXTENDEDKEY = 0x0001
_KEYEVENTF_KEYUP = 0x0002
_KEYEVENTF_SCANCODE = 0x0008

_KEY_EVENT_DELAY = 0.02     # пауза между нажатием и отпусканием — приложению нужно время среагировать
_SEND_RETRIES = 3           # SendInput периодически возвращает 0 (фильтры/оверлеи) — повторяем
_SEND_RETRY_DELAY = 0.08

_MODIFIER_VKS = {
    "ctrl": 0x11, "control": 0x11,
    "alt": 0x12,
    "shift": 0x10,
    "win": 0x5B, "super": 0x5B, "cmd": 0x5B, "command": 0x5B,
}

# именованные клавиши → (VK, нужна ли EXTENDEDKEY)
_NAMED_KEYS = {
    "enter": (0x0D, False), "return": (0x0D, False),
    "esc": (0x1B, False), "escape": (0x1B, False),
    "tab": (0x09, False),
    "backspace": (0x08, False), "bksp": (0x08, False),
    "space": (0x20, False),
    "capslock": (0x14, False), "caps": (0x14, False),
    "pause": (0x13, False),
    "delete": (0x2E, True), "del": (0x2E, True),
    "insert": (0x2D, True), "ins": (0x2D, True),
    "home": (0x24, True), "end": (0x23, True),
    "pageup": (0x21, True), "pgup": (0x21, True),
    "pagedown": (0x22, True), "pgdn": (0x22, True),
    "up": (0x26, True), "down": (0x28, True), "left": (0x25, True), "right": (0x27, True),
    "numlock": (0x90, True), "scrolllock": (0x91, True),
    "printscreen": (0x2C, True), "prtsc": (0x2C, True),
    "menu": (0x5D, True), "apps": (0x5D, True),
}
for _f in range(1, 25):
    _NAMED_KEYS[f"f{_f}"] = (0x6F + _f, False)
del _f

# печатные символы → scancode клавиши в US set-1
_SCAN_CODES = {
    "a": 0x1E, "b": 0x30, "c": 0x2E, "d": 0x20, "e": 0x12, "f": 0x21, "g": 0x22, "h": 0x23,
    "i": 0x17, "j": 0x24, "k": 0x25, "l": 0x26, "m": 0x32, "n": 0x31, "o": 0x18, "p": 0x19,
    "q": 0x10, "r": 0x13, "s": 0x1F, "t": 0x14, "u": 0x16, "v": 0x2F, "w": 0x11, "x": 0x2D,
    "y": 0x15, "z": 0x2C,
    "1": 0x02, "2": 0x03, "3": 0x04, "4": 0x05, "5": 0x06,
    "6": 0x07, "7": 0x08, "8": 0x09, "9": 0x0A, "0": 0x0B,
    "-": 0x0C, "=": 0x0D, "[": 0x1A, "]": 0x1B, ";": 0x27, "'": 0x28, "`": 0x29,
    "\\": 0x2B, ",": 0x33, ".": 0x34, "/": 0x35,
}

if _IS_WINDOWS:
    _ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong

    class _KEYBDINPUT(ctypes.Structure):
        _fields_ = [("wVk", ctypes.c_ushort), ("wScan", ctypes.c_ushort),
                    ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong),
                    ("dwExtraInfo", _ULONG_PTR)]

    class _INPUTUNION(ctypes.Union):
        _fields_ = [("ki", _KEYBDINPUT), ("pad", ctypes.c_byte * 24)]

    class _INPUT(ctypes.Structure):
        _fields_ = [("type", ctypes.c_ulong), ("union", _INPUTUNION)]

    _user32 = ctypes.windll.user32
    _user32.SendInput.argtypes = [ctypes.c_uint, ctypes.POINTER(_INPUT), ctypes.c_int]
    _user32.SendInput.restype = ctypes.c_uint


def _resolve_key(name: str) -> tuple[str, int, int] | None:
    """Имя клавиши → ('vk'|'scan', код, флаги). None — клавиша неизвестна."""
    n = (name or "").strip().lower()
    if not n:
        return None
    if n in _MODIFIER_VKS:
        return "vk", _MODIFIER_VKS[n], 0
    if n in _NAMED_KEYS:
        vk, extended = _NAMED_KEYS[n]
        return "vk", vk, _KEYEVENTF_EXTENDEDKEY if extended else 0
    if n in _SCAN_CODES:
        return "scan", _SCAN_CODES[n], 0
    return None


def _send_key_event(mode: str, code: int, flags: int, up: bool) -> bool:
    """Одно событие клавиатуры. False — система отклонила ввод (SendInput вернул 0)."""
    if not _IS_WINDOWS:
        return False
    fl = flags | (_KEYEVENTF_KEYUP if up else 0)
    by_scan = mode == "scan"
    if by_scan:
        fl |= _KEYEVENTF_SCANCODE
    event = _INPUT(type=_INPUT_KEYBOARD)
    event.union.ki = _KEYBDINPUT(
        wVk=0 if by_scan else code,
        wScan=code if by_scan else 0,
        dwFlags=fl, time=0, dwExtraInfo=0,
    )
    size = ctypes.sizeof(_INPUT)
    for _ in range(_SEND_RETRIES):
        if _user32.SendInput(1, ctypes.byref(event), size):
            return True
        time.sleep(_SEND_RETRY_DELAY)
    return False


def _send_combo(parts: list[str]) -> str | None:
    """Нажимает комбинацию клавиш. None — успех, иначе текст ошибки для модели."""
    resolved: list[tuple[str, tuple[str, int, int]]] = []
    for part in parts:
        key = _resolve_key(part)
        if key is None:
            return err(f": неизвестная клавиша '{part}'")
        resolved.append((part, key))
    for part, (mode, code, flags) in resolved:
        if not _send_key_event(mode, code, flags, up=False):
            return err(f": система отклонила ввод с клавиатуры (SendInput) на '{part}'")
        time.sleep(_KEY_EVENT_DELAY)
    for part, (mode, code, flags) in reversed(resolved):
        if not _send_key_event(mode, code, flags, up=True):
            return err(f": система отклонила ввод с клавиатуры (SendInput) при отпускании '{part}'")
    return None


# ─── Инструменты ─────────────────────────────────────────────────────────────

@tool(
    description=(
        "Сделать скриншот экрана с сеткой 4x4: подписи квадрантов (A1..D4), координаты "
        "углов каждого квадранта, зелёный кружок — текущий курсор мыши. Возвращает текстовое "
        "описание И изображение. Все координаты в подписях сетки = аргументы mouse/scroll-инструментов."
    ),
    short_description="screenshot with coordinate grid",
)
def screenshot() -> ToolOutput:
    b64, orig_w, orig_h, img_w, img_h, scale = take_screenshot()
    cx, cy = pyautogui.position()
    text = (
        f"Экран: {orig_w}x{orig_h} | Изображение: {img_w}x{img_h} (scale={scale:.3f}) | "
        f"Курсор: экран ({cx}, {cy}) = изображение ({round(cx * scale)}, {round(cy * scale)}). "
        "Координаты в подписях сетки — это аргументы mouse_move/mouse_click/scroll."
    )
    return ToolOutput(text=text, images=[b64])


@tool(
    description=(
        "Переместить курсор мыши БЕЗ нажатия. Координаты — в пикселях изображения "
        "последнего скриншота (как подписано на сетке)."
    ),
    short_description="move mouse cursor",
    x=("int", "X-координата на изображении скриншота"),
    y=("int", "Y-координата на изображении скриншота"),
)
def mouse_move(x: int, y: int) -> str:
    sx, sy = _to_screen(x, y)
    pyautogui.moveTo(sx, sy, duration=Config.MOUSE_MOVE_DURATION)
    return f"Курсор перемещён: экран ({sx}, {sy}) = изображение ({x}, {y})."


@tool(
    description=(
        "Клик мышью в точке. Координаты — в пикселях изображения последнего скриншота "
        "(как подписано на сетке)."
    ),
    short_description="click mouse (left/right/middle, optional double)",
    x=("int", "X-координата на изображении скриншота"),
    y=("int", "Y-координата на изображении скриншота"),
    button=("str", "optional: 'left' (по умолчанию), 'right' или 'middle'"),
    double=("bool", "optional: true — двойной клик"),
)
def mouse_click(x: int, y: int, button: str = "left", double: bool = False) -> str:
    btn = (button or "left").strip().lower()
    if btn not in ("left", "right", "middle"):
        return err(f": unknown button '{button}' — use left/right/middle")
    sx, sy = _to_screen(x, y)
    if double:
        pyautogui.doubleClick(sx, sy, button=btn)
        kind = f"double {btn}"
    else:
        if btn == "right":
            pyautogui.rightClick(sx, sy)
        elif btn == "middle":
            pyautogui.middleClick(sx, sy)
        else:
            pyautogui.click(sx, sy)
        kind = btn
    return f"{kind} click: экран ({sx}, {sy}) = изображение ({x}, {y})."


@tool(
    description="Прокрутить колесо мыши. Координаты — в пикселях изображения последнего скриншота.",
    short_description="scroll mouse wheel",
    amount=("int", "Сколько юнитов: >0 — вверх, <0 — вниз"),
    x=("int", "optional: X-координата на изображении (иначе — под текущим курсором)"),
    y=("int", "optional: Y-координата на изображении (иначе — под текущим курсором)"),
)
def scroll(amount: int, x: int = None, y: int = None) -> str:
    if (x is None) != (y is None):
        return err(": укажите x и y вместе, либо не указывайте вовсе (прокрутка под курсором)")
    if x is not None:
        x, y = _to_screen(x, y)
    pyautogui.scroll(amount, x, y)
    direction = "вверх" if amount > 0 else "вниз"
    where = f" в точке ({x}, {y})" if x is not None else " под курсором"
    return f"Прокручено на {abs(amount)} ({direction}){where}."


@tool(
    description=(
        "Напечатать строку текста в активное окно (кириллица поддерживается): текст "
        "временно кладётся в буфер обмена, затем нажимается Ctrl+V. Работает в полях ввода, "
        "чатах, редакторах."
    ),
    short_description="type text into active window",
    text=("str", "Текст для ввода"),
    press_enter=("bool", "optional: нажать Enter после ввода текста"),
)
def type_text(text: str, press_enter: bool = False) -> str:
    if not text:
        return err(": пустой текст")
    pyperclip.copy(text)
    failure = _send_combo(["ctrl", "v"])
    if failure:
        return failure
    time.sleep(0.15)
    if press_enter:
        failure = _send_combo(["enter"])
        if failure:
            return failure
    suffix = " + Enter" if press_enter else ""
    return f"Вставлено {len(text)} символов{suffix}."


@tool(
    description=(
        "Нажать клавишу или комбинацию клавиш. Примеры: enter, esc, tab, backspace, delete, "
        "f5; комбинации через '+': ctrl+v, ctrl+c, alt+tab, win+d, shift+tab."
    ),
    short_description="press key or combination",
    keys=("str", "Клавиша или комбинация через '+', напр.: enter, ctrl+v, alt+tab"),
)
def press_key(keys: str) -> str:
    parts = [p.strip().lower() for p in (keys or "").split("+") if p.strip()]
    if not parts:
        return err(": пустая клавиша")
    failure = _send_combo(parts)
    if failure:
        return failure
    return f"Нажато: {'+'.join(parts)}."
