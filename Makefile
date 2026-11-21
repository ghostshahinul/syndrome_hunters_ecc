# One-command workflows. Override the interpreter with: make bench PY=python3
ifeq ($(OS),Windows_NT)
PY ?= .venv/Scripts/python.exe
else
PY ?= .venv/bin/python
endif

DISTANCES ?= 3 5 7
PS        ?= 0.001 0.003 0.005 0.007 0.01
SHOTS     ?= 100000
DECODERS  ?= pymatching union_find experimental

.PHONY: install data bench plots test all

install:
	$(PY) -m pip install -e ".[dev]"

data:
	$(PY) -m src.data --distances $(DISTANCES) --ps $(PS) --shots $(SHOTS)

bench:
	$(PY) -m src.evaluate --decoders $(DECODERS) --distances $(DISTANCES) --ps $(PS) --shots $(SHOTS)

plots:
	$(PY) -m src.plots

test:
	$(PY) -m pytest -q

all: test bench plots
