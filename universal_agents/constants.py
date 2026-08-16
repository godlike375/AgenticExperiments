"""Общие константы фреймворка."""

# Маркер, которым системные сообщения помечаются в истории и промптах
ENVIRONMENT_PREFIX = '[[SYSTEM]]'

# Инструменты, которые нельзя отключить
CORE_TOOLS = ("load_tools", "unload_tool")

# Маркер авто-саммари (для распознавания и обновления)
SUMMARY_MARKER = f"{ENVIRONMENT_PREFIX} [AUTO-SUMMARY]"
