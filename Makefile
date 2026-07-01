.PHONY: install test eval eval-online web run clean

VENV ?= .venv
PY := $(VENV)/bin/python

install:
	python3.13 -m venv $(VENV)
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -e ".[dev]"
	@echo "Installed. Copy .env.example to crucible/.env and add GOOGLE_API_KEY (or set CRUCIBLE_OFFLINE=1)."

test:
	$(PY) -m pytest -q

# Offline scored report — no API key, no quota used.
eval:
	$(PY) -m crucible.eval.runner

# Real Gemini run over the dataset (needs GOOGLE_API_KEY; paces requests).
eval-online:
	$(PY) -m crucible.eval.runner --online

# ADK developer web UI with full trace visualization at http://localhost:8000
web:
	$(VENV)/bin/adk web

# One-off question through the pipeline (offline by default).
run:
	$(PY) -m crucible.eval.runner -q "$(Q)" --no-write

clean:
	rm -rf reports .pytest_cache **/__pycache__
