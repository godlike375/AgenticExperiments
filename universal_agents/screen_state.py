from __future__ import annotations

last_scale: float | None = None


def get_last_scale() -> float | None:
    return last_scale


def set_last_scale(value: float | None) -> None:
    global last_scale
    last_scale = value
