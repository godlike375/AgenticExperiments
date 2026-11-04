from __future__ import annotations

class GenerationInterrupted(Exception):
    """Поднимается, когда пользователь запросил остановку (например, убит запущенный инструмент)."""
