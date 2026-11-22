# One-command workflows. Override the interpreter with: make bench PY=python3
ifeq ($(OS),Windows_NT)
PY ?= .venv/Scripts/python.exe
else
PY ?= .venv/bin/python
endif

DISTANCES ?= 3 5 7
PS        ?= 0.001 0.003 0.005 0.007 0.01
SHOTS     ?= 50000
DECODERS  ?= fault_join pymatching pymatching_correlated union_find
SEED      ?= 20261121
BASIS     ?= z
PLOT_FILES ?=
GRID      = --distances $(DISTANCES) --ps $(PS) --shots $(SHOTS) --seed $(SEED) --basis $(BASIS)

.PHONY: install data bench plots test all

install:
	$(PY) -m pip install -e ".[dev]"

data:
	$(PY) -m src.data $(GRID)

bench:
	$(PY) -m src.evaluate --decoders $(DECODERS) $(GRID)

plots:
	$(PY) -m src.plots $(PLOT_FILES)

test:
	$(PY) -m pytest -q

all: test
	$(MAKE) bench
	$(MAKE) plots
