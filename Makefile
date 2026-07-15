.PHONY: setup hardware test lint

setup:
	uv sync

hardware:
	uv run python tools/hardware_report.py

test:
	uv run pytest

lint:
	uv run ruff check .
