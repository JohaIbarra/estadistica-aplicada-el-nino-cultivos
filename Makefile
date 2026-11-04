.PHONY: install data all test lint app

install:
	pip install -e ".[dev]"

all:            ## Reproduce todo: descarga, panel, inferencia, pronóstico e informes
	agroclima all

test:
	pytest

lint:
	ruff check src app tests && ruff format --check src app tests

app:
	streamlit run app/streamlit_app.py
