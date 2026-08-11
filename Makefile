.PHONY: setup test lint frontend-build check assets run

setup:
	uv sync --extra backend
	npm --prefix frontend ci

test:
	uv run pytest -q
	npm --prefix frontend test -- --run

lint:
	uv run ruff check .
	uv run ruff format --check .

frontend-build:
	npm --prefix frontend run build

check: lint test frontend-build

assets:
	uv run python -m scripts.check_assets

run:
	uv run uvicorn backend.main:app --reload
