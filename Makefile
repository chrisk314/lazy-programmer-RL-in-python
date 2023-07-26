SHELL := /bin/bash
VENV := .venv
BIN := $(VENV)/bin
PYTHON_VERSION ?= 3.10
PY := python$(PYTHON_VERSION)
SRC := ./
TESTS := ./tests
# Windows compatibility
ifeq ($(OS), Windows_NT)
    BIN := $(VENV)/Scripts
    PY := python
endif

.PHONY: all
all: lint test

.PHONY: clean
clean:
	rm -rf $(VENV)
	find $(SRC) -type f -name *.pyc -delete
	find $(SRC) -type d -name __pycache__ -delete

$(VENV):
	$(PY) -m venv $(VENV)
	$(BIN)/$(PY) -m pip install --upgrade pip setuptools poetry
	$(BIN)/$(PY) -m poetry config virtualenvs.in-project true
	$(BIN)/$(PY) -m poetry config virtualenvs.prompt venv
	$(BIN)/$(PY) -m poetry source add --priority=explicit pytorch-gpu-src https://download.pytorch.org/whl/cu118
	mkdir -p $(VENV)/.stamps
	@touch $@

$(VENV)/.stamps/init: $(VENV) pyproject.toml
	$(BIN)/$(PY) -m poetry install
	@touch $@

.PHONY: init
init: $(VENV)/.stamps/init

.PHONY: lint
lint: init
	$(BIN)/$(PY) -m isort --check-only .
	$(BIN)/$(PY) -m black --check .
	$(BIN)/$(PY) -m flake8 $(SRC)/ --docstring-convention google
	$(BIN)/$(PY) -m flake8 $(TESTS)/ --docstring-convention google
	$(BIN)/$(PY) -m mypy $(SRC)/ --explicit-package-bases
	$(BIN)/$(PY) -m mypy $(TESTS)/

.PHONY: test
test: init
	$(BIN)/$(PY) -m pytest -rs $(TESTS)/ --ignore=$(TESTS)/smoke

.PHONY: build
build: $(VENV)
	$(BIN)/$(PY) -m poetry version $$(git describe --tags --abbrev=0)
	$(BIN)/$(PY) -m poetry build
