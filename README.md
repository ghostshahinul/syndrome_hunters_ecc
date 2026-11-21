# syndrome_hunters_ecc

Hackathon project: fast decoders for the rotated surface code, benchmarked against
PyMatching (minimum-weight perfect matching) under circuit-level depolarizing noise.

## Problem statement

TODO: one paragraph on the goal (e.g. match MWPM accuracy at lower latency, or beat it
with a novel idea) and how we measure success.

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## How to run

| Task | Make | Plain command |
|---|---|---|
| Run tests | `make test` | `python -m pytest -q` |
| Sample datasets to `data/` | `make data` | `python -m src.data --distances 3 5 7 --ps 0.001 0.005` |
| Benchmark decoders to `results/` | `make bench` | `python -m src.evaluate --decoders pymatching --distances 3 5 7` |
| Plot newest results to `figures/` | `make plots` | `python -m src.plots` |

Make variables: `DISTANCES`, `PS`, `SHOTS`, `DECODERS`, e.g. `make bench SHOTS=10000`.

## Layout

```
src/
  circuits.py        Stim surface-code circuits (d, rounds, noise p) and DEMs
  data.py            sample detection events + observable flips; save/load .npz
  decoders/
    base.py          the Decoder interface
    baseline.py      PyMatching wrapper (reference)
    union_find.py    our union-find implementation
    experimental.py  our novel idea
  evaluate.py        logical error rate, latency, memory -> results/*.csv
  plots.py           threshold curves, latency scaling -> figures/*.png
native/              Rust/C++ hot loop, only if needed
tests/               contract, hand-checked cases, agreement with baseline
notebooks/           exploration only
results/             benchmark outputs (timestamped, never overwritten)
figures/             generated plots for the presentation
```

## Adding a decoder

1. Subclass `src.decoders.base.Decoder` and implement `decode_batch(dets) -> obs`.
2. Register it in `DECODERS` in `src/decoders/__init__.py`.
3. Run `make test` to check it automatically against the contract, the known cases and the baseline.

## Results

TODO: threshold estimate, LER at d = 3/5/7, µs/shot vs PyMatching, figures.
