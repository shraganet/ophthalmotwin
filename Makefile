PKG := ophthalmotwin_gp

.PHONY: help install dev lint format type test docker clean

help:
	@echo "targets: install dev lint format type test docker clean"

install:
	pip install .

dev:
	pip install -e ".[dev]"

lint:
	ruff check .
	isort --check-only .
	black --check .

format:
	isort .
	black .

type:
	mypy $(PKG)

test:
	pytest

docker:
	docker build -t $(PKG):0.1.0 .

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache **/__pycache__ *.egg-info runs
