# Integration branch: every team decoder on the same shots

Branch `integration/all-decoders`, created from `main` at `2b77b27`. `main` is untouched.

## What was brought in

| Source branch | What | How |
|---|---|---|
| `aRAYofsunshine07-patch-1` | MWPM pros/cons notes (`notebooks/MWPM.txt`) | plain merge, no conflicts |
| `shahi_2` (commit `0351fe0`) | FaultJoin decoder + its tests | selected files moved into `src/decoders/faultjoin/`, registered as `fault_join` |
| `shahi_3` | NVIDIA Ising hybrid | **not integrated**: it only runs on NVIDIA's own circuit (see below) |

## FaultJoin: conflicts and how each was resolved

FaultJoin's modules keep their `shahi_2` contents; only import paths changed.

| File on `shahi_2` | Conflict with `main` | Resolution |
|---|---|---|
| `src/decoders/faults.py`, `join.py`, `condition.py` | new files | moved to `src/decoders/faultjoin/` |
| `src/decoders/experimental.py` (holds `FaultJoinDecoder`) | `main` deleted this path | moved to `src/decoders/faultjoin/decoder.py`; the `ExperimentalDecoder` alias is dropped |
| `src/decoders/union_find.py` (shahi_2's own UF kernel, used by FaultJoin) | `main` deleted this path | moved to `src/decoders/faultjoin/union_find.py` (private; its `UnionFindDecoder` is not registered) |
| `src/decoders/base.py` (adds `Decoder._check_single`) | shared interface | kept `main`'s `base.py`; `_check_single` is defined on `FaultJoinDecoder` instead |
| `src/decoders/baseline.py` (adds `CorrelatedMatchingDecoder`) | shared file | kept `main`'s; correlated PyMatching is already a reference in `main`'s operating point |
| `src/decoders/__init__.py` (its own registry and `experimental` alias) | shared file | kept `main`'s registry and added `FaultJoinDecoder` |
| `src/evaluate.py`, `src/plots.py`, `src/data.py`, `src/circuits.py` | shared files | kept `main`'s |
| `README.md`, `Makefile`, `pyproject.toml`, `requirements.txt`, `native/README.md` | shared files | kept `main`'s |
| `src/decode.py` (blind b8 entry point) | needs shahi_2's registry defaults | not brought in |
| `tests/test_fault_join.py` | new | brought in; imports point at the new package (11 tests, all pass) |
| `tests/test_pipeline.py` | needs shahi_2's `evaluate.py`, `data.py`, `decode.py` | not brought in |
| `tests/conftest.py`, `test_known_cases.py`, `test_vs_baseline.py` | shared tests | kept `main`'s. FaultJoin now runs through all of them: contract tests, every d=3 single fault decoded exactly (allowance 0), and LER vs PyMatching |
| `figures/*.png`, `results/bench_*.csv` | shahi_2's own results | not brought in |

Also on this branch (tooling only, no decoder changes): FaultJoin's figure style, a dark slide
version of the operating-point figure, a markdown table writer,
`scripts/experiments/compare_all.py`, and an operating-point warm-up of 2,000 untimed shots instead of 16. The warm-up fix is
needed because FaultJoin compiles its UF stage lazily and at d=3 none of the first 16 shots reached it.

## Comparison: operating-point run 60

Circuit-level p = 0.5%, rotated Z memory, rounds = d, 5 batches × 50,000 = 250,000 shots per point
(seeds 0–4, the same shots for every decoder), every decoder single-thread, AC power, "Best
performance", numba cache cleared, clean commit `bacbec4`. Source:
`results/operating_point/operating_point_run60.csv`. The figures `operating_point_run60.png` (standard)
and `operating_point_run60_dark.png` (slides) plot batch time per shot (mean over the 5 batches)
against logical failures, with the 95% Wilson interval on failures.

| Decoder | d = 3 | d = 5 | d = 7 | d = 9 |
|---|---|---|---|---|
| PyMatching | 1.656% [1.607, 1.707]; 0.60 µs | 1.383% [1.338, 1.429]; 4.43 µs | 1.035% [0.996, 1.075]; 14.37 µs | 0.670% [0.638, 0.702]; 37.15 µs |
| Correlated PyMatching | 1.625% [1.576, 1.675]; 1.62 µs | 1.076% [1.036, 1.117]; 12.21 µs | 0.609% [0.579, 0.640]; 41.04 µs | 0.299% [0.278, 0.321]; 105.60 µs |
| UnionFind3D | 1.667% [1.617, 1.718]; 1.10 µs | 1.523% [1.476, 1.572]; 4.68 µs | 1.222% [1.180, 1.266]; 13.36 µs | 0.846% [0.811, 0.883]; 32.64 µs |
| UF3D correlated | 1.608% [1.559, 1.658]; 1.48 µs | 1.234% [1.192, 1.278]; 8.32 µs | 0.859% [0.823, 0.896]; 22.70 µs | 0.511% [0.484, 0.540]; 55.54 µs |
| Hierarchical | 1.703% [1.653, 1.755]; 0.85 µs | 1.236% [1.193, 1.280]; 9.07 µs | 0.859% [0.823, 0.896]; 24.95 µs | 0.511% [0.484, 0.540]; 59.58 µs |
| FaultJoin | 1.596% [1.548, 1.646]; 0.57 µs | 1.282% [1.239, 1.327]; 8.71 µs | 0.842% [0.807, 0.878]; 29.03 µs | 0.477% [0.451, 0.505]; 70.04 µs |

- **Our five decoders reproduce run 40's failure counts exactly** (same seeds and shots), and so does
  FaultJoin between runs 56, 57 and 60.
- **FaultJoin vs plain PyMatching:**
  - FaultJoin is more accurate at d = 5/7/9, with non-overlapping CIs.
  - At d=3 the CIs overlap.
- **FaultJoin vs Hierarchical:**
  - At d=3 FaultJoin is more accurate (1.596% vs 1.703%, CIs separate).
  - At d = 5, 7 and 9 the CIs overlap (d=9: 0.477% [0.451, 0.505] vs 0.511% [0.484, 0.540]).
- **Correlated PyMatching** is the most accurate at d ≥ 5.
- **Time ÷ plain PyMatching, same run:**

  | d | FaultJoin | Hierarchical | Correlated PyMatching |
  |---|---|---|---|
  | 3 | 0.94× | 1.41× | 2.68× |
  | 5 | 1.97× | 2.04× | 2.75× |
  | 7 | 2.02× | 1.74× | 2.86× |
  | 9 | 1.89× | 1.60× | 2.84× |

- **The laptop ran slower today than in run 40** (PyMatching d=9: 37.2 vs 19.1 µs), so compare times
  only within one run.
- **Runs 56 and 57** (same shots) are kept for the record:
  - run 56's FaultJoin d=3 time included FaultJoin's first-call JIT; fixed by the longer warm-up;
  - runs 56 and 57 also drew fastest-to-slowest batch whiskers on the time axis, which the current
    figures no longer do.

## NVIDIA Ising hybrid (`shahi_3`): not comparable, not run here

- **It cannot decode our shots.** The decoder is built from an `NvidiaExperiment` (NVIDIA's
  `MemoryCircuit` with its XV stabilizer maps and 25-channel noise model), not from a DEM.
  `docs/ising.md` on that branch records that NVIDIA's detector order, observable and DEM differ from
  ours at every (basis, d) setting, so feeding it our detector rows "would assign different meanings
  to its columns".
- **Running it on its own circuit needs things this branch does not install:**
  - a clone of NVIDIA's Ising-Decoding repo in `external/`;
  - the Fast R9 checkpoint from Hugging Face (SHA-256 pinned);
  - PyTorch and about eight more packages (`requirements-ising-constraints.txt`, tested on Python 3.12; this venv is 3.14).

  None are present here. The laptop has an RTX 4050, so CUDA itself would be available. Installing was left
  for a decision, per the no-unrequested-installs rule.
- **The branch's own recorded numbers** (`results/ising/hybrid-final/bench_20261122T001433622270Z.csv`
  on `shahi_3`) were measured on **NVIDIA's circuit**, on another machine (GTX 1650 Ti GPU), with
  50,000 shots. They are **a different circuit and not comparable** with the table above. At p = 0.5%:

  | Decoder (NVIDIA circuit) | d = 3 | d = 5 | d = 7 | d = 9 |
  |---|---|---|---|---|
  | Ising + PyMatching | 1.990% | 1.738% | 1.540% | 1.136% |
  | Ising + correlated PyMatching | 2.014% | 1.730% | 1.528% | 1.102% |
  | Ising, boundary-restored + correlated | 1.996% | 1.734% | 1.498% | 1.090% |
  | PyMatching | 2.074% | 2.164% | 1.826% | 1.188% |
  | Correlated PyMatching | 1.910% | 1.488% | 0.960% | 0.530% |
  | FaultJoin | 1.750% | 1.820% | 1.248% | 0.816% |

  On that circuit, the hybrids take 105–1,954 µs per shot on the GPU, against 0.9–83 µs for PyMatching on the CPU.
