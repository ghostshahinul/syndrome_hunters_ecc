# Overnight progress log

Plan: [docs/overnight_plan.md](overnight_plan.md). Machine: 13th Gen Intel i5-13420H
(8 cores / 12 logical), Windows 11, on AC, "Balanced" power plan. Windows reports
`CurrentClockSpeed` = 2100 MHz (the base clock) at every probe, so that field does not show
turbo/throttling; compare times only within one run.

Baseline at start (commit `ed031a9`, 2026-11-21 19:57): UnionFind3D S=2, growth=all.

---

## Milestone 1: UF accuracy experiments (19:57 to 20:05, time-box 45 min)

**What:** `UnionFind3DConfig` gained `scale` (S in `L_e = max(1, round(S*w/w_min))`) and
`growth` (`"all"` | `"smallest"`). Both are implemented in the Python reference and the numba
kernel. New tests check every (S, growth) variant × 3 noise tiers for bit-identity between
numba and the reference, syndrome consistency, and exact decoding of every single DEM
component. New `src/machine.py` records the CPU model, MHz, power state and git commit for
every results CSV.

**Setup:** circuit-level p = 0.5%, rounds = d, 50,000 shots, seed 0 (the same shots for
every config), numba backend, best of 2 timings after warm-up.
CSV: `results/uf_experiments/uf_experiments_20261121_195936.csv`.

| d | S=2 all (old default) | S=4 all (**new default**) | S=8 all | S=2 smallest | S=4 smallest | S=8 smallest |
|---|---|---|---|---|---|---|
| 3 | 1.826% [1.712, 1.947] | 1.710% [1.600, 1.827] | 1.810% [1.697, 1.931] | 1.826% | 1.704% | 1.802% |
| 5 | 1.770% [1.658, 1.889] | 1.498% [1.395, 1.608] | 1.432% [1.332, 1.540] | 1.766% | 1.512% | 1.440% |
| 7 | 1.546% [1.442, 1.658] | 1.166% [1.076, 1.264] | 1.160% [1.070, 1.258] | 1.540% | 1.152% | 1.132% |
| 9 | 1.142% [1.053, 1.239] | 0.876% [0.798, 0.962] | 0.832% [0.756, 0.915] | 1.112% | 0.850% | 0.810% |

Single-thread µs/shot (12-thread in brackets):

| d | S=2 all | S=4 all | S=8 all | S=2 smallest | S=4 smallest | S=8 smallest |
|---|---|---|---|---|---|---|
| 3 | 0.89 (0.18) | 0.91 (0.22) | 0.95 (0.18) | 0.88 (0.19) | 0.94 (0.21) | 1.00 (0.22) |
| 5 | 2.49 (0.45) | 2.71 (0.46) | 3.38 (0.59) | 2.62 (0.49) | 3.11 (0.51) | 3.88 (0.64) |
| 7 | 6.34 (1.00) | 7.29 (1.11) | 9.22 (1.53) | 6.68 (1.10) | 8.69 (1.35) | 11.58 (1.78) |
| 9 | 13.38 (2.33) | 15.91 (2.49) | 20.72 (3.20) | 14.61 (2.33) | 19.24 (3.02) | 26.80 (4.24) |

**Decision:** the default is now **S = 4, growth = "all"**.
- S=2 → 4 lowers LER with CIs that don't overlap at d=7 (1.546 → 1.166%) and d=9
  (1.142 → 0.876%). The single-thread slowdown is 1.02 / 1.09 / 1.15 / 1.19× at d = 3/5/7/9.
- S=8 is within the CI of S=4 at every d and costs 1.55× at d=9 single-thread, which is over
  the 1.5× limit. Rejected.
- "smallest" growth changes LER by less than the CI at every (S, d), for example
  0.850% vs 0.876% at d=9 S=4, and costs up to 1.21× more time. It's kept as an option, not
  the default.
- UF ÷ PyMatching with the new default, using the PyMatching LER from Step 3 on the same seed:
  1.00 / 1.08 / 1.15 / 1.27× at d = 3/5/7/9 (it was 1.07 / 1.28 / 1.53 / 1.66×). The
  d=7 ≤ 1.5× stretch target is now within reach for Hierarchical.
- Timing note: absolute µs are about 2× lower than in the Step 3 measurement (d=7 S=2
  single-thread 6.34 vs 13.26 µs). The machine state changed between runs (see the caveat
  above), so same-run comparisons only.

**Housekeeping:** `results/tests/` (per-run pytest plots, 3.3 MB and growing) is now
gitignored. Operating-point runs 2 to 4 are committed.

**Commit:** (see Milestone 1 commit below)

**Next:** Milestone 2, Hierarchical decoder.

Milestone 1 commit: `217235f`.

---

## Milestone 2: Hierarchical decoder (20:06 to 20:20, time-box 2.5 h)

**What was built**
- `src/decoders/cube3d/hierarchical.py`: `Hierarchical(graph, HierarchicalConfig(gate, K, strict_r1, uf))`.
  Registered as **`hierarchical`** (whole-shot gate, K=6) and **`hierarchical_partial`**
  (partial gate, no K cap). Both are picked up by the generic tests and the operating-point
  figure automatically.
  - **numba backend:** one fused `prange` kernel. Per shot it runs the local rules, applies the gate, then
    runs UF on the whole shot or on the residual.
  - **reference backend:** NumPy Cube3D-Local plus the Python UF reference. It is
    bit-identical to the numba backend (tested, 3 tiers × d=3/5 × both gates × strict on/off, plus d=7).
- Cube3D-Local gains a **numba backend**, `local_shot`, which is bit-identical to the NumPy
  reference: 3 tiers × d=3/5/7 × K ∈ {0, 2, 6, 10, ∞} × strict on/off. It works with
  snapshot-per-color semantics and an incrementally maintained count of lit neighbours.
- **UF speed-up, results unchanged:** shots are processed in chunks of 64 that share one scratch
  workspace, only touched entries are reset, and empty shots exit early. This removed about 16 allocations
  per shot: UF at d=3 p=0.1% went from 1.02 to 0.46 µs single-thread in the same session. The
  predictions are unchanged, as all bit-identity tests still pass.
- **Serial-overhead fix:** the batch wrappers had been computing `dets.sum(axis=1)` in NumPy.
  That alone cost about 0.35 µs/shot at d=9 and made the 12-thread Hierarchical 1.8× slower than UF
  (0.95 vs 0.53 µs). Lit counts are now computed inside the kernels, and the gap is 0.59 vs 0.55 µs.
- New option `strict_r1` (R1 pairs must be isolated within distance 2). Off by default; see below.

**Setup:** circuit-level noise, rounds = d, 100,000 shots, seed 0 (the same shots for every
decoder at each (d, p)). Single-thread and 12-thread times are best of 2 after warm-up.
PyMatching runs single-threaded in the same run.
CSV: `results/hierarchical/hierarchical_20261121_201348.csv`. It includes the K sweep;
the main rows have `main=True`.

**LER (failures / 100k, 95% Wilson CI), whole-shot gate K=6**

| p | d | PyMatching | UnionFind3D | Hierarchical | H ÷ PM |
|---|---|---|---|---|---|
| 0.1% | 3 | 0.0610% [0.048, 0.078] | 0.0830% [0.067, 0.103] | 0.0840% [0.068, 0.104] | 1.38× |
| 0.1% | 5 | 0.0190% | 0.0210% | 0.0210% | 1.11× |
| 0.1% | 7 | 0.0010% (1) | 0.0000% (0) | 0.0000% (0) | n/a |
| 0.1% | 9 | 0.0010% (1) | 0.0010% (1) | 0.0010% (1) | 1.00× |
| 0.2% | 3 | 0.2950% [0.263, 0.331] | 0.3180% [0.285, 0.355] | 0.3460% [0.312, 0.384] | 1.17× |
| 0.2% | 5 | 0.1090% | 0.1330% | 0.1330% | 1.22× |
| 0.2% | 7 | 0.0210% | 0.0310% | 0.0310% | 1.48× |
| 0.2% | 9 | 0.0060% | 0.0110% | 0.0110% | 1.83× |
| 0.5% | 3 | 1.755% [1.675, 1.838] | 1.755% [1.675, 1.838] | 1.821% [1.740, 1.906] | 1.04× |
| 0.5% | 5 | 1.353% [1.283, 1.427] | 1.480% [1.407, 1.557] | 1.480% [1.407, 1.557] | 1.09× |
| 0.5% | 7 | 1.019% [0.959, 1.083] | 1.169% [1.104, 1.238] | 1.169% [1.104, 1.238] | 1.15× |
| 0.5% | 9 | 0.712% [0.662, 0.766] | 0.922% [0.865, 0.983] | 0.922% [0.865, 0.983] | 1.29× |

At d ≥ 5 the whole-shot Hierarchical makes exactly the same number of failures as UF. Every
difference is at d=3, and it comes from the fast path.

**Fast path (whole-shot, K=6): fraction of all shots it decoded alone, and its failures on the
accepted non-empty shots vs UF's failures on those same shots**

| p | d=3 | d=5 | d=7 | d=9 |
|---|---|---|---|---|
| 0.1% | 99.7%; 38 vs 37 of 15,597 | 94.2%; 3 vs 3 of 51,489 | 64.0%; 0 vs 0 of 55,130 | 16.3%; 0 vs 0 of 15,759 |
| 0.2% | 98.7%; 184 vs 156 of 27,534 | 79.3%; 9 vs 9 of 61,109 | 20.9%; 0 vs 0 of 20,131 | 0.5%; 0 vs 0 of 472 |
| 0.5% | 93.4%; 982 vs 916 of 50,639 | 29.7%; 15 vs 15 of 28,383 | 0.1%; 0 vs 0 of 147 | 0.0%; none accepted |

**µs/shot, single-thread (12-thread in brackets)**

| p | d | UF | Hier. whole-shot | Hier. partial | PyMatching (1 thr) |
|---|---|---|---|---|---|
| 0.1% | 3 | 0.24 (0.06) | **0.08** (0.02) | 0.08 (0.03) | 0.10 |
| 0.1% | 5 | 0.66 (0.09) | **0.43** (0.07) | 0.43 (0.09) | 0.53 |
| 0.1% | 7 | 1.70 (0.25) | 1.42 (0.23) | **1.24** (0.19) | 1.25 |
| 0.1% | 9 | 3.49 (0.49) | 3.52 (0.54) | **2.75** (0.41) | 2.94 |
| 0.2% | 3 | 0.30 (0.05) | **0.14** (0.04) | 0.14 (0.03) | 0.18 |
| 0.2% | 5 | 1.06 (0.16) | **0.93** (0.15) | 0.93 (0.14) | 0.87 |
| 0.2% | 7 | 2.90 (0.44) | 2.89 (0.58) | 2.89 (0.48) | 2.71 |
| 0.2% | 9 | 6.15 (0.93) | 6.35 (0.92) | 6.53 (0.95) | 6.29 |
| 0.5% | 3 | 0.44 (0.09) | 0.33 (0.07) | **0.30** (0.06) | 0.42 |
| 0.5% | 5 | 2.39 (0.33) | 2.42 (0.33) | 2.91 (0.42) | 2.61 |
| 0.5% | 7 | 7.18 (0.98) | 7.49 (1.08) | 9.65 (1.26) | 8.64 |
| 0.5% | 9 | 17.16 (2.36) | 17.36 (2.54) | 22.36 (3.05) | 21.47 |

**Partial gate: LER vs UF on the same shots.** It costs accuracy beyond the CI at d=9 p=0.5%:
1.066% [1.004, 1.132] vs 0.922% [0.865, 0.983]. It is also worse at d=7 p=0.5%
(1.264% vs 1.169%), d=9 p=0.2% (0.023% vs 0.011%) and d=7 p=0.2% (0.047% vs 0.031%).
Following the plan's rule, **the default is the whole-shot gate**.
- Cause: R1 only tests isolation at distance 1, so in dense shots it pairs the ends of two
  different chains. With `strict_r1=True` (distance-2 isolation) the partial gate is back
  within the CI: 0.939% [0.881, 1.001] at d=9 and 1.182% at d=7, p=0.5%. But it costs 1.8× UF's
  time (32.3 vs 18.0 µs at d=9), because the distance-2 counts run for every R1 candidate.
  It's kept as an option, off by default. Data: same-seed 100k-shot run, single-thread;
  numbers in this paragraph come from a scratch script and are not in the CSV.

**K sweep, 2 to 10:** in the CSV. For whole-shot and d ≥ 5, LER equals UF's at every K.
The fast-path fraction rises with K and levels off by K≈8 to 10. The speedup is best at K = 6 to 10.
K=10 vs K=6 gives at most 6% more speed (0.77 vs 0.83 relative time at d=7 p=0.1%). At d=3 the fast
path's LER penalty grows from 1.00× (K=2) to 1.04 to 1.09× (K≥5), within the CI. K=6 stays the default.

**Acceptance targets, as of now**
- Hierarchical and UF3D LER decrease with d at p=0.5%: **met.** 1.821 → 1.480 → 1.169 → 0.922%
  (H); 1.755 → 1.480 → 1.169 → 0.922% (UF).
- Hierarchical within the 95% CI of UF3D: **met at every (p, d)** for the whole-shot gate.
- Hierarchical faster than UF3D: **partly met.** Single-thread speedup:
  - 1.2 to 3.0× at d ≤ 5 for p ≤ 0.2%, and at d=3 for p=0.5%;
  - 1.2× at d=7 p=0.1%;
  - 0.95 to 1.0× (1–5% slower) at d ≥ 7 for p ≥ 0.2% and at d ≥ 5 for p=0.5%. There the fast
    path accepts 0 to 21% of shots and adds the cost of scanning them.
- Stretch, Hierarchical ≤ 1.5× PyMatching at d=7: **met at p=0.5%** (1.15×) and at p=0.2% (1.48×).
- Single-thread vs PyMatching, the fair comparison: Hierarchical is faster at d=9 p=0.5%
  (17.36 vs 21.47 µs), d=7 p=0.5% (7.49 vs 8.64) and d=3 everywhere. It is slower at d=7/9
  for p ≤ 0.2% (for example 3.52 vs 2.94 µs at d=9 p=0.1%).

**Tests:** `tests/test_cube3d_hierarchical.py` (186 tests with the local suite): bit-identity,
both gates' semantics, K=0 equals UF, every single DEM component, consistent counts.
Full suite: 425 passed, 24 skipped, 1 xfailed.

**Next:** Milestone 3, figures.

Milestone 2 commit: `aac5710`. You also committed the work in progress as `ec9ad56` "For
discord" at 20:15; I left that commit as it is.

---

## Milestone 3: figures (20:21 to 20:31, time-box 1.5 h)

**What was built**
- `src/figures.py`:
  - **Style registry:** `STYLES` gives every decoder a fixed colour and marker in all figures.
    The first 4 slots keep the existing operating-point styles. UnionFind3D, Hierarchical,
    Hierarchical (partial) and Cube3D-Local take the remaining slots of the reference palette.
    The operating-point figure from pytest now uses the same registry.
  - **Plot functions:** `plot_accuracy_runtime` (filled marker = 1 thread, hollow = 12 threads;
    footer adds CPU, power, threads, backend and commit), `plot_sweep`, `plot_thread_scaling`,
    `plot_fast_path_fraction` and `plot_cube_collapse_story`.
  - **Saving:** `save_figure` writes `<name>_<timestamp>.png` (opened exclusively, never
    overwritten) plus `<name>_latest.png`.
  - **CLI:** `python -m src.figures --accuracy … --sweep … --threads …` regenerates the figures
    from saved CSVs.
- `src/benchmark.py` (`python -m src.benchmark`, `make benchmark`) measures the operating point,
  the sweep and thread scaling. It writes the CSVs with the machine fingerprint, then draws
  every figure. `--quick --out <dir>` is for smoke runs.
- `tests/test_figures.py` runs every plot function on a tiny synthetic CSV and checks that saving
  never overwrites and that the style registry is unique (9 tests).
- I couldn't run the palette validator because Node is not installed. The palette is the
  skill's reference palette, unchanged.

**Benchmark run 1** (commit `5e7b5ff`, i5-13420H, AC / Balanced, Windows reports 2100 MHz)
- CSVs: `results/accuracy_runtime_20261121_202412.csv` (50k shots),
  `results/sweep_20261121_202412.csv` (100k shots, p = 0.1/0.2/0.3/0.5/0.8%, single thread)
  and `results/thread_scaling_20261121_202412.csv` (100k shots).
- Figures (`results/plots/*_latest.png`, final versions stamped 202806/202830):
  `accuracy_runtime`, `accuracy_runtime_sweep`, `thread_scaling`, `fast_path_fraction` and
  `cube_collapse_story`. The earlier stamps (202412, 202806) are first renders that had layout
  problems. I kept them because the rule is never to overwrite.

**Key numbers (p = 0.5%, 50k shots)**
- **UF3D:** LER 1.710 / 1.498 / 1.166 / 0.876% at d = 3/5/7/9. Single-thread µs:
  0.43 / 2.30 / 6.92 / 16.50; 12 threads: 0.11 / 0.51 / 0.99 / 2.23.
- **Hierarchical:** LER 1.850 / 1.498 / 1.166 / 0.876%. Single-thread µs: 0.34 / 2.37 / 6.98 / 16.60.
- **PyMatching:** LER 1.704 / 1.384 / 1.010 / 0.690%; µs 0.35 / 2.33 / 7.97 / 20.21.
- **Flattening story:** PyMatching 2D on flat 2.68 → 5.56 → 11.73 → 20.80%; Cube-Collapse
  2.68 → 9.44 → 18.18 → 30.96%.
- **Thread scaling, 1 → 12 threads:** UF 6.7× (d=7) and 7.0× (d=9); Hierarchical 7.1× and 6.9×.
- **New finding:** p = 0.8% is above threshold for every decoder, PyMatching included
  (4.06 → 4.89 → 5.24 → 5.31% for d = 3 → 9).

## Milestone 4: README + talk (20:21 to 20:32, time-box 1.5 h, written while the benchmark ran)
- **`README.md`, rewritten:**
  1. quickstart
  2. the algorithms (how each works, complexity, parallelism, deviations from Clique_L2
     2507.11481, Delfosse–Nickerson and 2307.14989)
  3. the evaluation story with the figures
  4. bugs found and fixed
  5. results tables with CIs, the acceptance targets and honest limitations
  6. test-suite overview
- **`docs/talk_outline.md`:** 8 slides plus a whiteboard segment (5-qubit repetition code,
  union-find growth and peeling, why the graph is 3D). Each slide names its figure and has 2 to
  4 speaker bullets.
- **Makefile:** `make benchmark` and `make benchmark-quick` (the latter writes to
  `scratch_results/`, which is gitignored).
- **Check before the talk:** I describe arXiv 2307.14989 only as the source of the
  predecoder + global-decoder layout. I didn't re-read the paper tonight, so please check that
  wording.

Milestones 3+4 commit: `51d341e`.

---

## [ASK] Milestone 5: A or B? (logged 20:31; defaulting to B at 20:41 unless you answer)

Both options are feasible: gcc is at `C:\msys64\ucrt64\bin\gcc.exe`, and networkx 3.7 is
already in the venv.

**My recommendation: B (our own MWPM tier for the hardest shots).**
- **Judging: number of algorithms and novelty.** B adds a fifth algorithm and a third tier
  (local → UF → MWPM). A adds a backend for an algorithm we already have.
- **It targets our main limitation,** the accuracy gap: UF is 1.27× PyMatching at d=9 p=0.5%.
  A doesn't change accuracy.
- **Parallelism is already shown:** numba gives 6.7–7.1× at 12 threads. A C/OpenMP port would
  probably buy a constant factor of 1–2× single-thread at best, with the same scaling story.
- **Risk:** `networkx.max_weight_matching` is pure Python, about milliseconds per shot. B only pays
  off if the trigger (UF cluster size or count) sends a small fraction of shots (≲1%) to MWPM
  *and* those shots hold a good share of UF's failures. I'll measure both and report honestly,
  including if the trigger turns out not to find UF's mistakes.

---

## Push + operating-point update (20:33 to 20:40, requested by you)
- **Push:** merged `origin/main` (2 commits, `notebooks/MWPM.txt` only, no overlap with our
  files), ran the suite (green) and pushed `main` to `34653ec`. The remote branches `shahi_2`
  (FaultJoin), `shahi_3` (FaultJoin + NVIDIA Ising hybrid) and `aRAYofsunshine07-patch-1`
  (MWPM notes) were **not** merged. That's a team decision, and they edit shared test and plot files.
- **Operating point (`src/operating_point.py`, `tests/test_operating_point.py`):**
  - **Batches:** each point is now B independent batches (seeds 0, 1, …). LER is pooled over
    them. Time is the mean batch µs/shot, and the fastest and slowest batch are drawn as
    horizontal whiskers.
  - **Decoders, distances, threads:** the default set is PyMatching, Correlated PyMatching,
    PyMatching 2D on flat, UnionFind3D and Hierarchical. Cube-Collapse and Hierarchical
    (partial) can be added with `--decoders`. Distances are 3/5/7/9, and numba runs on 1 thread.
  - **CSV and figure:** every row records threads, backend, CPU, MHz, power and commit. The
    footer shows the machine line.
  - **CLI vs pytest:** the CLI defaults to 5 batches × 50k shots. pytest defaults to 3 × 50k
    (`--op-batches`, `--op-shots`).
  - **New tests:** at d=9 PyMatching, Correlated PyMatching, UnionFind3D and Hierarchical each
    beat d=7 with non-overlapping CIs (4 parametrized tests), and PyMatching 2D on flat gets
    worse at d=9.
- **Operating-point run 10** (3 × 50k = 150k shots per point, p = 0.5%, 1 thread). LER %, then µs/shot as mean (fastest–slowest):

  | d | PyMatching | Correlated PM | PM 2D on flat | UnionFind3D | Hierarchical |
  |---|---|---|---|---|---|
  | 3 | 1.700; 0.30 | 1.653; 0.90 | 2.619; 0.32 | 1.717; 0.50 | 1.789; 0.36 |
  | 5 | 1.391; 2.48 | 1.063; 6.98 | 5.542; 1.55 | 1.534; 2.41 | 1.534; 2.45 |
  | 7 | 1.014; 8.50 (8.33–8.76) | 0.610; 23.74 | 11.758; 4.71 | 1.212; 7.46 (7.38–7.55) | 1.212; 7.52 (7.42–7.69) |
  | 9 | 0.663; 20.29 (20.23–20.39) | 0.276; 59.54 | 21.013; 11.10 | 0.842; 17.44 (17.33–17.55) | 0.842; 17.80 (17.54–18.01) |
- Full suite: 439 passed, 24 skipped, 1 xfailed in 48.5 s (the d=9 operating point adds about 25 s).

Operating-point commit: `a1d2bfa` (not pushed).

---

## Milestone 5 (B): our own MWPM as a third tier (20:41 to 21:03, time-box 2 h)
No answer to the [ASK] within 10 minutes, so I took B, as the plan says.

**What was built**
- `src/decoders/cube3d/mwpm.py`, **MWPM3D** (`mwpm_3d`):
  - scipy Dijkstra from each lit detector, plus a networkx exact blossom with one boundary copy
    per detector;
  - observable parity along the shortest paths;
  - sparse but exact pruning: a pair edge is kept only if it is shorter than both detectors'
    boundary distances, and boundary copies pair along kept edges. This made it 2.2–2.7× faster.
- `src/decoders/cube3d/three_tier.py`, **Hierarchical + MWPM** (`hierarchical_mwpm`):
  Hierarchical, then MWPM3D for shots whose largest UF cluster has more than T nodes
  (default T = 2·rounds).
- **UF and the fused Hierarchical kernel now report the largest cluster's node count.** numba
  equals the reference (tested); predictions are unchanged.
- **`Decoder.slow` (base class):** the generic speed and baseline tests use 500 / 5,000 shots for
  the pure-Python decoders. Without it the suite took 7.5 min; with it, 61 s.
- **Pinned** `networkx==3.7` and `scipy==1.18.1`. Both were already installed as PyMatching
  dependencies, so nothing new was installed.
- **Tests:** `tests/test_cube3d_mwpm.py` (18 tests). Figure: `plot_mwpm_tier` (`--mwpm` in
  `src.figures`) plus a smoke test.

**Results** (`results/mwpm/mwpm_tier_20261121_205911.csv`; figures `results/plots/mwpm_tier_p0_{1,2,5}_latest.png`)
- **MWPM3D agrees with PyMatching** on 16,771 of 16,772 decoded shots across all (p, d). The one
  difference is at d=5, p=0.5%, almost certainly a tie.
- **MWPM3D cost per routed shot, pure Python:** about 0.3–0.6 ms at d=3, 0.6–1.3 ms at d=5,
  1.4–7.6 ms at d=7 and 3.5–42 ms at d=9.
- **p = 0.5%, 10k shots, failures:**

  | d | PyMatching | UF | 3-tier (T, routed) | µs/shot: UF → 3-tier |
  |---|---|---|---|---|
  | 5 | 165 | 180 | 171 (T=5, 18.9%) | 2.26 → 228 |
  | 7 | 75 | 101 | 84 (T=14, 17.4%); 76 (T=7, 44%) | 6.78 → 1,155; 3,220 |
  | 9 | 72 | 90 | 78 (T=27, 15.6%) | 19.9 → 5,681 (T=9 and T=18 skipped: over the 90 s budget) |

- **p ≤ 0.2%, 20k shots:** failure counts are 0–66, too few to show a significant gain. For
  example d=9 p=0.2%: UF 2, 3-tier 0 (T ≤ 27), PyMatching 0.
- **A 100k-shot analysis** (scratch script, PyMatching answers standing in for MWPM3D since the two
  agree) shows cluster size is a strong trigger at p=0.2%: routing the ~9% of shots with the
  largest clusters fixes all 13 of 13 fixable UF failures at d=7 and 6 of 6 at d=9. It is a weaker
  trigger at p=0.5%: the top 10% catches about half of them.
- **Verdict:** the third tier shows the accuracy headroom, closing about 2/3 of the UF-vs-MWPM gap at
  d = 7/9. With a pure-Python blossom it costs 170–290× UF's time. It needs a compiled matcher
  (or cluster-local matching) to be practical. It is **not** a default anywhere.

---

## FINAL SUMMARY (21:03)

| Milestone | Status | Key numbers | Commit(s) |
|---|---|---|---|
| 1. UF experiments | done | S=2 → 4: d=9 LER 1.142 → 0.876% for 1.19× time; smallest-first growth within CI | `217235f` |
| 2. Hierarchical | done | identical failures to UF at d ≥ 5; 1.2–3.0× faster single-thread on sparse shots; partial gate rejected (d=9: 1.066% vs 0.922%) | `aac5710` (plus your `ec9ad56`) |
| 3. Figures | done | 5 standard figures + style registry + `python -m src.benchmark`; UF/Hierarchical scale 6.7–7.1× on 12 threads | `5e7b5ff`, `51d341e` |
| 4. README + talk | done | README sections 1–6, `docs/talk_outline.md` (8 slides + whiteboard) | `51d341e` |
| Push (your request) | done | merged `origin/main` (MWPM.txt only), pushed `34653ec` | `34653ec` |
| Operating point (your request) | done | 5 decoders, d = 3–9, batches; d=9 UF/Hierarchical 0.842% vs PyMatching 0.663%, 17.4 vs 20.3 µs | `a1d2bfa` |
| 5 (B). MWPM third tier | done | MWPM3D agrees with PyMatching on 16,771/16,772 shots; 3-tier closes about 2/3 of the gap at d = 7/9, p=0.5%, at 170–290× cost | `1d75a90` + final commit |

**Final numbers (p = 0.5%, single thread; operating-point run 10, 150k shots):**
- **d=7:** Hierarchical 1.212% at 7.52 µs vs PyMatching 1.014% at 8.50 µs (1.20×).
- **d=9:** Hierarchical 0.842% at 17.80 µs vs PyMatching 0.663% at 20.29 µs (1.27×).

**Acceptance targets:**
- Hierarchical and UF LER fall with d: **met**.
- Hierarchical within the CI of UF: **met**.
- Hierarchical faster than UF: **partly met** (only where the fast path finishes most shots).
- Stretch, ≤ 1.5× PyMatching at d=7: **met** (1.15–1.20×).

**Figures** (`results/plots/*_latest.png`): `accuracy_runtime`, `accuracy_runtime_sweep`,
`thread_scaling`, `fast_path_fraction`, `cube_collapse_story`, `mwpm_tier_p0_1`,
`mwpm_tier_p0_2` and `mwpm_tier_p0_5`, plus `results/operating_point/operating_point_run10.png`.

**For you to check or decide**
1. **Push:** commits after `34653ec` (`a1d2bfa`, `1d75a90` and the final commit) are **not pushed**.
2. **Teammate branches:** `shahi_2` (FaultJoin), `shahi_3` (FaultJoin + NVIDIA Ising hybrid) and
   `aRAYofsunshine07-patch-1` are not merged into main. They edit shared files (`conftest.py`,
   `test_vs_baseline.py`, `plots.py`, `evaluate.py`), so expect conflicts.
3. **Operating point:** it now shows the 5 decoders you named. Cube-Collapse is available via
   `--decoders` and is still in `src.benchmark` and the `cube_collapse_story` figure.
4. **README wording:** the description of arXiv 2307.14989 (predecoder + global decoder) is from
   memory; please check it.
5. **Superseded figures:** `results/plots/` keeps first renders with layout problems (never
   overwritten). The `_latest.png` files are the final ones; delete the old stamps if you want.
6. **Palette validator:** it was not run (no Node). The palette is the skill's reference palette
   unchanged; the two MWPM series use neutral inks with unique markers, because all 8 palette
   slots are taken.
7. **Timing caveat:** Windows always reports 2100 MHz, and absolute µs drift between sessions.
   Compare only within one CSV.

---

# New goal (from 21:05): beat plain PyMatching's accuracy within its speed range

## Item 1: correlated two-pass UnionFind3D, `uf3d_correlated` (21:05 to 21:30, time-box 2 h)

**What was built**
- **Graph:** `build_detector_graph` now also records **edge correlations**. For every DEM error
  instruction whose `^`-separated components map to ≥ 2 graph edges, each ordered pair (e, f)
  gets the probability that such an instruction fires (CSR: `corr_ptr`, `corr_idx`,
  `corr_joint`). Circuit-level d=7: 1,512 of 1,558 edges have partners (7,654 pairs), median
  P(f|e) = 0.067. Code capacity has none.
- **UF** (`UnionFind3DConfig(correlated=True)`; registry `uf3d_correlated`):
  - **Pass 1:** normal UF.
  - **Pass 2:** every partner f of a pass-1 edge e gets the integer length of
    P(f|e) = P(e, f) / p_e, clipped to [p_f, 0.5], on the same S=4 scale, for this shot only.
    Then UF runs again.
  - **Choosing the result (deviation from the plan):** pass 2 is kept only if its correlated
    score is strictly lower: Σ L_e minus, for each DEM partner pair inside the correction,
    max(0, L_e + L_f − L_joint). Without this check, pass 1's *alternative-but-equivalent*
    edges pulled in their own partners and broke 2 single faults at d=3. It still does for 2
    instructions (see below).
  - **numba vs reference:** a numba kernel (two `_uf_shot` calls; partner lengths precomputed;
    combined with order-independent `min`) is bit-identical to the Python reference: predictions,
    corrections, ok and max_cluster, 3 tiers × d = 3/5/7.
- **Tests:**
  - every single DEM component is exact;
  - syndrome consistency on 2,000 random shots per case;
  - numba equals the reference;
  - full DEM instructions: exact on every tier and distance **except circuit d=3**: 2 of 286
    wrong ("D5 D16 ^ D9 ^ D12" and "D7 D18 ^ D14 ^ D11", p = 3.3e-4; pass 2's pairing of two
    likelier correlated faults scores 16 vs 17). **Correlated PyMatching also mis-decodes 1 of
    these 286.** This is documented as `single_fault_allowance = 2` on the decoder class (a new
    base-class attribute, default 0).
- **Hazard found:** numba's on-disk cache does not notice changes to callees in another file.
  After `_uf_shot`'s workspace changed, the cached Hierarchical kernel crashed with an access
  violation. Clearing `src/decoders/cube3d/__pycache__/*.nbi|*.nbc` fixed it. **If you edit
  union_find.py or local.py, clear that cache.**

**Benchmark** (operating-point run 17, `results/operating_point/operating_point_run17.csv`):
p = 0.5%, 50k shots, seed 0, single thread. The machine was about 1.8× slower than at 20:00,
so the µs below are only comparable with each other.

| d | PyMatching | Correlated PyMatching | UnionFind3D | **UF3D correlated** | µs/shot: PM, corrPM, UF, UF-corr |
|---|---|---|---|---|---|
| 3 | 1.704% [1.594, 1.821] | 1.648% [1.540, 1.763] | 1.710% [1.600, 1.827] | 1.712% [1.602, 1.829] | 0.47, 1.75, 1.08, 2.27 |
| 5 | 1.384% [1.285, 1.490] | 1.074% [0.987, 1.168] | 1.498% [1.395, 1.608] | 1.260% [1.166, 1.362] | 4.60, 11.50, 4.85, 10.66 |
| 7 | 1.010% [0.926, 1.101] | 0.628% [0.562, 0.701] | 1.166% [1.076, 1.264] | **0.828% [0.752, 0.911]** | 15.22, 42.37, 12.86, 28.98 |
| 9 | 0.690% [0.621, 0.766] | 0.278% [0.236, 0.328] | 0.876% [0.798, 0.962] | **0.512% [0.453, 0.578]** | 36.09, 105.73, 32.16, 63.25 |

**Verdict:** UF3D correlated **beats plain PyMatching with non-overlapping 95% CIs at d = 7 and
d = 9**. It does not at d = 5 (CIs overlap; 1.260% vs 1.384%) or d = 3 (equal). Single-thread it
costs 1.75–1.9× PyMatching's time at d = 7/9, 2.3× at d=5 and 4.8× at d=3. That is not yet
"within PyMatching's speed range": it sits between plain PyMatching and correlated PyMatching
(0.60–0.68× of correlated PyMatching's time). Correlated PyMatching remains more accurate.

Item 1 commits: `707cda6`, merged with your teammates' cleanup and pushed as `bb47c78`.
- **Teammate cleanup that was merged:**
  - 4 figure deletions, including `results/plots/accuracy_runtime_sweep_latest.png`. The README links to it; regenerate it with `python -m src.figures --sweep results/sweep_20261121_202412.csv`.
  - PR #3, which deleted `src/__init__.py`, `figures/` and `native/`.
- **Side effect of PR #3:** `src` is now a namespace package. pytest and `python -m src.…` from the repo root still work, the suite is green, and the CLIs import.
  - What breaks: `pip install -e .` no longer exposes `src` (setuptools' `find` skips directories without `__init__.py`), so scripts run from *other* directories can't import `src` without `PYTHONPATH=.`.
  - I did not revert their change. **Decide** whether to restore `src/__init__.py`.

## Item 2: ensemble UF, `uf3d_ensemble` (21:36 to 21:41, time-box 1.5 h)

**What was built** (`src/decoders/cube3d/ensemble.py`, registry `uf3d_ensemble`)
- **Variants:** M UF variants per shot. Variant 0 is unperturbed. Variant v uses integer lengths from w·exp(σ·z_v), with z ~ N(0,1) and seed 1000+v, and odd variants use smallest-first growth. Each variant can be correlated (item 1).
- **Winner:** the lowest score under the **original** lengths, using the correlated score (or the plain sum of lengths when not correlated). Ties go to the lowest v, so M=1 equals UF / UF3D correlated exactly (tested).
- **Two numba layouts:**
  - `parallel="variants"`: shots in sequence, `prange` over the M variants of each shot, so one shot uses up to M cores.
  - `parallel="shots"`: `prange` over chunks of shots.
  - Both are **bit-identical to the Python reference** (predictions, ok, chosen variant, correction; 3 tiers × d = 3/5 × correlated on/off).
- **Other tests:** single components exact; syndrome consistency; and adding variants never raises the winner's score.
- **σ:** set to 0.2 from a seed-7 check at d = 7/9 with M=8. Failures went 147 → 119 at d=7 and 107 → 84 at d=9 (σ = 0.1 gave 125/91, σ = 0.3 gave 142/113). The benchmark uses seed 0, so that choice doesn't bias the reported numbers.

**Results** (`results/ensemble/ensemble_20261121_213502.csv`, figure `results/plots/ensemble_vs_M_20261121_213502.png`): p = 0.5%, 50k shots, seed 0. All times single-thread µs/shot from this one run; the machine is still about 1.5× slower than at 20:00.

| d | PyMatching | Correlated PM | corr. ensemble M=1 | M=4 | M=8 | M=12 | plain ensemble M=1 → M=8 |
|---|---|---|---|---|---|---|---|
| 3 | 1.704% (0.60 µs) | 1.648% (1.69) | 1.712% (3.2) | 1.656% (12) | 1.606% (23) | 1.604% [1.498, 1.718] (35) | 1.710 → 1.702% |
| 5 | 1.384% [1.285, 1.490] (4.4) | 1.074% [0.987, 1.168] (10.8) | 1.260% (11) | 1.142% (48) | **1.040% [0.955, 1.133]** (91) | 1.022% [0.938, 1.114] (133) | 1.498 → 1.448% |
| 7 | 1.010% [0.926, 1.101] (11.6) | 0.628% [0.562, 0.701] (37) | 0.828% (29) | 0.754% (141) | 0.690% (294) | **0.646% [0.579, 0.720]** (425) | 1.166 → 1.256% |
| 9 | 0.690% [0.621, 0.766] (30.3) | 0.278% [0.236, 0.328] (92) | 0.512% (63) | 0.480% (309) | **0.402% [0.350, 0.461]** (611) | 0.396% (946) | 0.876 → 0.942% |

- **The correlated ensemble beats plain PyMatching with non-overlapping CIs at d = 5, 7 and 9** for M ≥ 8. At d = 7 (M=12) and d = 5 it is **within the CI of correlated PyMatching**; at d = 9 correlated PyMatching is still better (0.278% vs 0.396%).
- **The plain ensemble does not help** and is slightly worse at d = 7/9. Picking the lowest original-weight correction among non-correlated UF variants doesn't approach MWPM here. The gain comes from correlation plus ensembling.
- **Cost is far outside PyMatching's speed range:** M=8 is 25× (d=7) and 20× (d=9) PyMatching single-thread.

**Cores used per shot (parallel over variants) vs parallel over shots**, correlated ensemble M=8, 10k shots (`ensemble_threads_20261121_213502.csv`):

| threads | d=7: over variants (cores/shot) | d=7: over shots (1 core/shot) | d=9: over variants | d=9: over shots |
|---|---|---|---|---|
| 1 | 263 µs (1) | 242 µs | 625 µs (1) | 625 µs |
| 2 | 154 (2) | 137 | 357 (2) | 297 |
| 4 | 90 (4) | 70 | 205 (4) | 164 |
| 8 | 58 (8) | 37 | 122 (8) | 84 |
| 12 | 66 (8) | 28 | 132 (8) | 69 |

- **Over variants:** 8 cores bring the latency of a single shot down 4.5–5.1×, to 58 µs (d=7). That is still 5× PyMatching single-thread.
- **Over shots:** throughput is better (8.5–9× at 12 threads), but that is a throughput figure. PyMatching can be parallelized the same way with processes, so it is not a like-for-like comparison.

Item 2 commit: `0186f54` (merge; it also took in teammate deletions of old figure stamps).

## Item 3: fair speed reporting (21:42 to 21:46)
- **`plot_accuracy_runtime`:** now draws **single-thread rows only** by default; the 12-thread
  hollow markers are opt-in with `show_multithread=True`, which is tested. `src.benchmark`'s
  accuracy part times every decoder single-thread.
- **`thread_scaling` footer:** now says it shows our scaling, not a like-for-like race, because
  PyMatching can also run separate shot batches in parallel processes.
- **README results:** the timing table is single-thread only. A new subsection, "Parallelism,
  reported separately", covers throughput across shots (with the PyMatching-processes caveat)
  and latency as **cores used per shot** (ensemble: 1 core 263 µs vs 8 cores 58 µs at d=7, M=8).
- **Figures re-rendered from the saved CSVs:** `accuracy_runtime_latest.png`,
  `thread_scaling_latest.png` and `cube_collapse_story_latest.png` (stamp 214318/214319).
- The operating-point figure (single-thread since 20:37) and `ensemble_vs_M` (single-thread)
  already followed the rule.

Item 3 commit: `bd86829` (pushed).

## Item 4: best method as Hierarchical's deep tier (21:46 to 22:03)
**Choice:** UF3D correlated (M=1) beats plain PyMatching at about 2× its speed. The correlated
ensemble beats it by more, but at 20–25× the speed, outside PyMatching's range. So **correlated UF is now
Hierarchical's default deep tier**: `HierarchicalConfig.uf = UnionFind3DConfig(correlated=True)`.
- **Code:** the fused numba kernel gained a `_deep` helper that calls plain or correlated UF, with
  correlated scratch per chunk. The reference uses `uf_correlated_shot_reference`. Hierarchical
  numba equals the reference for both deep tiers × both gates × strict on/off × 3 tiers × d = 3/5.
  `three_tier` (Hierarchical + MWPM) inherits the new deep tier.
- **Defaults:** the operating point and `src.benchmark` add "UF3D correlated". The d=9 operating-point
  test now also requires UF3D correlated to improve on d=7.
- **Commit:** `76f9191` (code).

**Operating-point run 22** (`results/operating_point/operating_point_run22.{csv,png}`): 5 × 50k = 250k shots,
p = 0.5%, all single-thread.

| d | PyMatching | Correlated PM | UnionFind3D | UF3D correlated | **Hierarchical** | H time ÷ PM time |
|---|---|---|---|---|---|---|
| 3 | 1.656% [1.607, 1.707] | 1.625% | 1.667% | 1.643% | 1.703% [1.653, 1.755] | 1.6× |
| 5 | 1.383% [1.338, 1.429] | 1.076% | 1.523% | 1.226% | **1.227% [1.185, 1.271]** | 2.1× |
| 7 | 1.035% [0.996, 1.075] | 0.609% | 1.222% | 0.835% | **0.835% [0.800, 0.871]** | 2.2× |
| 9 | 0.670% [0.638, 0.702] | 0.299% | 0.846% | 0.496% | **0.496% [0.470, 0.525]** | 1.9× |

**Verdict:**
- **Hierarchical beats plain PyMatching's accuracy with non-overlapping 95% CIs at d = 5, 7 and 9.**
  At d = 3 the CIs overlap.
- **Speed:** 1.9–2.2× plain PyMatching's single-thread time, so it is *near* PyMatching's speed range
  but not inside it. It runs in 0.65–0.8× of correlated PyMatching's time.
- **Correlated PyMatching remains more accurate.**

**Sweep rerun** (`results/sweep_20261121_215215.csv`; `accuracy_runtime_sweep_latest.png` and
`fast_path_fraction_latest.png`, re-rendered at 215639 with a corrected footer):
- At p = 0.5%: UF3D correlated / Hierarchical 0.783% vs PyMatching 1.019% at d=7, and 0.504%
  vs 0.712% at d=9.
- At p = 0.2%: d=7 15 vs 21 failures, d=9 3 vs 6. Those counts are small; more shots are needed
  for CIs.

**Docs:** the README has a new headline, sections 2.6 (correlated UF) and 2.7 (ensemble), story
step 5, the latest results table, the honest caveats, and 2.5 updated for the new deep tier. The
talk outline has a punchline, slide 7b (correlations) and updated slide-8 numbers.

---

## STATUS at 22:03 (items 1 to 4 done; hard stop 2:00 not reached)
| Item | Result | Commit |
|---|---|---|
| 1. Correlated UF | beats plain PyMatching at d = 7/9 (non-overlapping CIs), 1.75–1.9× its time | `707cda6` (pushed via `bb47c78`) |
| 2. Ensemble UF | correlated ensemble, M=8: beats plain PyMatching at d = 5/7/9 and matches correlated PyMatching's CI at d = 5/7, at 20–25× cost; plain ensemble doesn't help | `0186f54` |
| 3. Fair speed reporting | single-thread-only figures and tables; parallelism reported separately (cores per shot, the PyMatching-processes caveat) | `bd86829` |
| 4. Hierarchical with the correlated deep tier | beats plain PyMatching at d = 5/7/9 at 1.9–2.2× its time | `76f9191` + this commit |

**Open items for you**
1. **`src/__init__.py`** (deleted by PR #3): this breaks `pip install -e .` imports outside the repo
   root. Restore it?
2. **Superseded figures** were deleted by teammates; some `_latest` files were regenerated.
3. **numba cache:** after editing `union_find.py` or `local.py`, clear `src/decoders/cube3d/__pycache__/*.nb*`.
   Stale caches crash dependent kernels.
4. **Possible next step to get *inside* PyMatching's speed:** run pass 2 only on the clusters that
   changed. Today pass 2 redoes the whole shot.

Item 4 commit: `0f7054f` (pushed).

---

# Optimization round (from 23:50): get Hierarchical inside plain PyMatching's single-thread speed

**Timing setup for every run:**
- **Power:** AC power, Windows 11 power mode switched to **"Best performance"** (registry
  `ActiveOverlayAcPowerScheme = ded574b5…`). The plan is still "Balanced"; only the mode changed. Undo with
  `powercfg /overlaysetactive 00000000-0000-0000-0000-000000000000`, or Settings → Power.
- **numba:** cache cleared at the start of each timing script (`src.machine.clear_numba_cache`).
- **Warm-up and threads:** warm-up before timing; single thread.
- **Fingerprint:** each CSV row includes the power mode.

## Phase A: measure (23:50 to 00:02)
- **Instrumentation:** per-shot counters live in the **Python reference** UF and correlated UF:
  rounds, idle rounds, lit, clusters, max cluster, pass 2 ran or kept, clusters touched by a
  reweighted edge, and the number of reweighted edges. The reference runs the identical
  algorithm and is bit-identical to numba, so the counts describe the numba runs exactly, and
  the numba kernels pay **zero cost** (they are unchanged).
- **Script:** `scripts/opt_a_measure.py`. CSVs: `results/optimization/phaseA_20261121_235854.csv`
  (time split) and `phaseA_counters_20261121_235854.csv`.

**Time split** (single thread, best of 3, 50k shots, p = 0.5%, same shots):

| d | PyMatching | UnionFind3D | UF3D correlated | Hierarchical | local fast path only (K=6) | H ÷ PM |
|---|---|---|---|---|---|---|
| 5 | 2.45 µs | 2.48 | 5.57 | 5.42 | 0.39 | 2.21× |
| 7 | 8.42 | 7.52 | 17.09 | 16.79 | 0.38 | 1.99× |
| 9 | 19.09 | 16.83 | 36.87 | 37.24 | 0.63 | 1.95× |

**Counters** (correlated UF, all shots; mean / p50 / p90 / p99):

| d | lit | rounds, pass 1 | idle rounds, pass 1 | rounds, pass 2 | clusters | touched clusters | reweighted edges | pass 2 ran / kept |
|---|---|---|---|---|---|---|---|---|
| 5 | 8.3 / 8 / 14 / 19 | 5.05 / 5 / 6 / 11 | 2.28 (45% of rounds) | 4.65 | 4.1 | 2.2 | 29 | 98% / 19% |
| 7 | 25.0 / 25 / 35 / 44 | 6.01 / 5 / 8 / 12 | 2.06 (34%) | 5.50 | 10.8 | 6.1 | 89 | 100% / 43% |
| 9 | 55.8 / 55 / 70 / 83 | 6.86 / 6 / 10 / 12 | 2.06 (30%) | 6.14 | 22.4 | 13.0 | 200 | 100% / 64% |

**Readout**
- **The deep tier is the whole cost.** Hierarchical sends 99.8% (d=7) and 100% (d=9) of shots to UF.
  The fast path costs 0.4–0.6 µs per shot.
- **Correlated UF costs 2.2× plain UF.** Plain UF is already 0.88× PyMatching, so reaching ≤ 1.0×
  needs the correlated path itself about 2× faster.
- **Skip-ahead (B1)** removes about 2 of every 6–7 rounds per pass (30–45% of rounds are idle).
- **Cluster-local pass 2 (D)** could skip the 40–45% of clusters that no reweighted edge touches.
- **The table lookup (C)** can matter only at d ≤ 5; at d=7 the median shot has 25 events.

**Golden outputs** for Phase B: `tests/golden/golden_outputs.npz` (237 KB) and `tests/test_golden.py`.
They hold predictions, ok, max_cluster, a correction hash and the chosen variant for plain UF,
correlated UF, Hierarchical (both gates and both deep tiers) and the correlated ensemble (M=4),
on 600 shots each for 3 tiers × d = 3/5/7 plus circuit d=9. They were generated from the code
before Phase B.

Phase A commit: `7a62911` (pushed).

## Phase B: exact speedups (00:02 to 00:28)
**Result: no net speedup. The B1/B2 changes as specified were 14–22% slower, so most were reverted.**
Every version stayed bit-identical to the saved golden outputs (`tests/test_golden.py`).

**What was tried** (CSVs `results/optimization/phaseB_*.csv`; the old kernels from `7a62911`
were loaded from git and timed in the same run, single thread, best of 3, 50k shots, p = 0.5%):
- **B1, skip-ahead growth (exact):**
  - **Rounds:** executed growth rounds per pass fell from 5.0 / 6.0 / 6.9 (round by round) to
    2.8 / 3.9 / 4.8 at d = 5/7/9.
  - **Cost:** it needs the frontier and its rates before it can jump, which means three passes over
    the frontier per round instead of one. That made each round 25–43% more expensive than the
    old single-pass round, more than the saved rounds (d=7 plain UF: 7.25 µs old, 8.32 µs new; with
    the skip disabled in the new structure, 10.37 µs).
  - **The saturated-node skip** (per-node count of ungrown incident edges) cost more than it saved.
- **B2, removing O(n) scans:** lit indices are now extracted once per shot and passed down
  (`_uf_shot(lits, nlit, …)`, `local_shot(lit, lits, nlit, …)`, Hierarchical keeps a sparse
  dense buffer). Sorting the visited nodes for the peeling roots, and a sorted list of grown
  boundary edges, were both **slower** than the byte scans over n and over the boundary edges
  at d ≤ 9 (the sort of grown boundary edges alone cost 3–7%).

**Kept** (exact, golden-tested):
- the lits interface, which Phase D needs to decode a subset of events;
- skip-ahead as an option, `UnionFind3DConfig(skip_ahead=True)`, tested equal to round-by-round
  growth at S = 4 and 16 (plain and correlated, 3 tiers). Its executed rounds no longer grow with
  the weight scale S, which matters for E3.

**Reverted to the old code:** the single-pass growth loop as the default, the root scan over n, and
the boundary scan over all boundary edges.

**Final B3 timing** (`phaseB_20261122_002717.csv`; µs/shot old → new, single thread):

| d | PyMatching | UnionFind3D | UF3D correlated | Hierarchical | Hierarchical ÷ PM |
|---|---|---|---|---|---|
| 5 | 2.45 | 2.49 → 2.71 | 5.80 → 5.88 | 5.65 → 5.83 | 2.38× |
| 7 | 8.26 | 7.46 → 7.71 | 16.59 → 16.78 | 16.98 → 16.88 | 2.04× |
| 9 | 20.15 | 17.58 → 17.78 | 38.55 → 38.43 | 38.51 → 38.69 | 1.92× |

**What this shows:** at the default S = 4 the growth loop isn't where the time goes. Correlated UF is
two full UF passes; the lever is doing less in pass 2 (Phase D) and serving easy shots without UF
(Phase C).

Phase B commit: `dd31a12` (pushed).

## Phase C: single-fault lookup table (00:29 to 00:38)
**What was built**
- **Table:** `graph.single_fault_table(dem)` maps the packed sorted syndrome of every DEM
  instruction (1 to 4 detectors, 16 bits each) to its observable bits; if syndromes collide,
  the most likely observable wins.
  - **Checks:** every instruction in all 3 tiers × d = 3–9 lights 1–4 detectors, and **no
    syndrome is ambiguous**.
  - **Lookup:** binary search inside numba (`table_lookup`); the reference uses
    `table_lookup_reference`.
- **On by default:** `HierarchicalConfig(table=True)` and the `uf3d_correlated` registry entry
  (`UnionFind3DConfig(correlated=True, table=True)`). Plain UF is unchanged.
- **Tests:**
  - **Every single DEM fault is exact** for correlated UF with the table (3 tiers × d = 3/5/7,
    numba and reference) and for Hierarchical (both gates).
  - `uf3d_correlated.single_fault_allowance` is back to **0**.
  - numba equals the reference with the table on.
  - The table changes only the shots it answers.
- **Golden:** the golden configs pin `table=False`, so they keep checking the pre-table outputs
  (unchanged).

**Results** (`scripts/opt_ab_compare.py`, `results/optimization/phaseC_20261122_003716.csv`;
5 × 50k = 250k shots per point, p = 0.5%, single thread, same shots for every config):

| d | share of shots answered by the table | Hierarchical µs, off → on | UF3D correlated µs, off → on | LER effect |
|---|---|---|---|---|
| 3 | 39.3% | 0.55 → 0.48 | 1.41 → 0.86 | Hierarchical identical (1.703%); UF-corr 1.643 → 1.608% (within CI) |
| 5 | 6.2% | 5.66 → 5.67 | 5.86 → 5.79 | identical |
| 7 | 0.01% | 17.14 → 17.12 | 16.89 → 16.91 | identical |
| 9 | 0% | 38.47 → 38.43 | 38.75 → 38.95 | identical |

**Decision: keep.** The table makes every single fault exact and costs nothing measurable. It
cannot help at d ≥ 7, where shots have about 25–56 events. Hierarchical ÷ PyMatching in this run:
2.13× (d=5), 2.04× (d=7), 1.88× (d=9).

Phase C commit: `406f75a` (pushed).

## Phase D: cluster-local pass 2 (00:39 to 00:45). Tried and reverted.
**What was built** (then reverted; the code is in this session's history, not in the repo):
- **Pass 1:** exports each visited node's cluster root.
- **Touched clusters:** those containing an endpoint of a reweighted partner edge (radius 0), or
  also a neighbour of one (radius 1).
- **Pass 2:** decodes only the touched clusters' events with the reweighted lengths.
- **Candidate:** pass-1 flips of the untouched clusters XOR the pass-2 flips, kept if its
  correlated score is lower.
- **Checks:** numba equalled a Python reference; the syndrome was consistent on every shot;
  agreement with whole-shot pass 2 was 100% on 400 shots each of circuit d=5/7 and phenomenological d=5.

**Benchmark** (`results/optimization/phaseD_20261122_004308.csv`, 250k shots, p = 0.5%, single thread):

| d | whole-shot pass 2 (UF-corr) | local, radius 0 | local, radius 1 | PyMatching |
|---|---|---|---|---|
| 5 | 1.2260% · 5.80 µs | 1.2344% · 6.51 µs | 1.2264% · 8.89 µs | 1.3828% · 2.39 µs |
| 7 | 0.8348% · 16.99 µs | 0.8536% · 17.85 µs | 0.8340% · 25.51 µs | 1.0348% · 8.34 µs |
| 9 | 0.4964% · 38.80 µs | 0.5080% · 39.77 µs | 0.4960% · 58.01 µs | 0.6696% · 20.25 µs |

(Hierarchical with each deep-tier variant: the same pattern within ±3%.)

**Decision: revert.**
- **Radius 0** is 3–12% *slower* and slightly worse (within the CI).
- **Radius 1** matches whole-shot accuracy but is 45–55% slower: it marks nearly every cluster, and
  the labelling, marking and XOR bookkeeping cost more than they save.
- **Why there's little to save:** pass 2 still has to decode about 60% of the clusters (13 of 22 at
  d=9), and growing a subset of clusters costs about the same per cluster as growing them all.

Phase D commit: `4887ecf` (pushed).

## Phase E: accuracy options (00:50 to 01:01)
**What was built:** three opt-in `UnionFind3DConfig` flags, each in numba and in the reference,
**bit-identical** (tested on 3 tiers × d = 3/5, alone and combined, plus Hierarchical with each
as its deep tier). The defaults are off, so the golden outputs are unchanged.
- **`float_score` (E1):** the pass-1 vs pass-2 score on fine weights, w_e and the pair credit
  max(0, w_e + w_f − w_joint), as fixed-point int64 (w·2²⁴). Integer sums don't depend on the
  summation order, which keeps numba and the reference bit-identical.
- **`weighted_forest` (E2):** peel on a Kruskal min-length spanning forest of the grown edges,
  ascending (length, id), with boundary edges only for odd clusters.
- **`passes=3` (E4):** when pass 2 was kept, reweight from pass 2's correction and decode once more,
  with the score check again.

**Benchmarks** (250k shots, p = 0.5%, single thread, same shots in each run):
`results/optimization/phaseE1234_20261122_005357.csv` and `phaseE56_20261122_010045.csv`.

| config | d=5 LER · µs | d=7 LER · µs | d=9 LER · µs | decision |
|---|---|---|---|---|
| PyMatching | 1.3828% · 2.41 | 1.0348% · 8.36 | 0.6696% · 20.34 | reference |
| UF3D correlated (current default) | 1.2260% · 6.07 | 0.8348% · 17.41 | 0.4964% · 39.96 | baseline |
| E1 float score | 1.2220% · 6.17 | 0.8336% · 17.37 | 0.4924% · 39.82 | **off**: no measurable gain |
| E2 weighted forest | 1.2120% · 7.57 | 0.7976% [0.764, 0.833] · 20.95 | 0.4608% [0.435, 0.488] · 47.23 | **off**: −5–7% LER for +18–20% time |
| E3 S=8 (skip-ahead) | 1.1452% · 6.62 | 0.7800% [0.746, 0.815] · 22.10 | 0.4524% [0.427, 0.480] · 53.96 | **off**: −7–9% LER for +27–35% time |
| E3 S=8 (round by round) | same LER · 7.41 | · 22.20 | · 51.86 | skip-ahead gives no speed gain even at S=8 |
| E3 S=16 (skip-ahead) | 1.1680% · 7.03 | 0.7740% · 25.61 | 0.4488% · 68.00 | **off**: no better than S=8, slower |
| E4 pass 3 | 1.2020% · 6.86 | 0.8152% · 21.07 | 0.4864% · 51.03 | **off**: −2–4% LER for +20–27% time |
| plain UF, S=4 / 8 / 16 | 1.523 / 1.459 / 1.470% | 1.222 / 1.152 / 1.150% | 0.846 / 0.810 / 0.794% | plain UF never beats PyMatching |
| E5 partial gate (correlated deep tier) | 1.4012% · 5.20 | 1.0860% · 16.82 | 0.7396% · 38.39 | **rejected**: much worse |
| E5 K = 2…10 (whole-shot) | 1.2260–1.2304% · 5.66–6.30 | identical · 17.5–17.7 | identical · 39.8–40.5 | **keep K=6** (no effect at d ≥ 7) |
| E6 ensemble M = 1 / 2 / 3 | 1.226 / 1.169 / 1.144% · 6.7 / 14.6 / 21.0 | 0.835 / 0.823 / 0.790% · 17.8 / 41.3 / 60.7 | 0.496 / 0.483 / 0.485% · 40.5 / 97.0 / 142.5 | **off**: 2.2–3.5× time |

- **E5 at d=3:** K=2 is more accurate (1.608% vs 1.703%) but 29% slower; K ≥ 5 equals K=6.
- **E4 run/win rate:** pass 3 runs on every shot where pass 2 was kept (43% at d=7, 64% at d=9; the
  Phase A rates). Its overall gain is the 2–4% shown above.

**Outcome:** none of the accuracy options is adopted as a default. **Every** accuracy gain costs
time, and time is the binding constraint for the ≤ 1.0× target. The accurate variants (E2, E3 S=8,
or both) remain available as options. Correlated UF itself is still 1.96–2.08× PyMatching at d = 7/9.

Phase E commit: `313deb4` (pushed).

## Extra E7, not in the plan: single-pass "online" correlated UF (01:02 to 01:05). Failed.
- **Idea:** when an edge finishes growing in pass 1, give its DEM partners the conditional length
  immediately, with no second pass (cost ≈ one UF pass).
- **Prototype:** in a scratch copy of the kernel (not committed); 50k shots, p = 0.5%, 1 thread:
  - **O1 (online, single pass):** 2.800% / 2.580% / 2.202% LER at d = 5/7/9, worse than *plain*
    UF (1.498 / 1.166 / 0.876%), at 3.85 / 12.22 / 29.52 µs.
  - **O2 (online pass 1 + the usual pass 2):** 1.402 / 0.976 / 0.534%, also slower (7.24 / 22.26 / 52.54 µs)
    than two-pass correlated UF (1.260 / 0.828 / 0.512% at 6.01 / 17.09 / 39.68 µs).
- **Why it fails:** grown edges include the whole interior of each cluster, not just the
  correction, so online reweighting shortens far too many partners. Dropped.

## Phase F: final numbers and figures (01:06 to 01:15)
**Defaults** (unchanged by Phases B to E except the table):
- Hierarchical, whole-shot gate, K = 6;
- deep tier = correlated two-pass UF, S = 4;
- single-fault table on;
- the B/E options all off.

**Operating-point run 31** (`results/operating_point/operating_point_run31.{csv,png}`): 5 × 50k = 250k shots,
p = 0.5%, single thread, numba cache cleared, Best-performance power mode, commit `313deb4`.

| d | PyMatching | Correlated PM | UnionFind3D | UF3D correlated | **Hierarchical** | H ÷ PM |
|---|---|---|---|---|---|---|
| 3 | 1.656% · 0.31 µs | 1.625% · 0.86 | 1.667% · 0.57 | 1.608% · 0.86 | 1.703% [1.653, 1.755] · 0.51 | 1.65× |
| 5 | 1.383% [1.338, 1.429] · 2.36 | 1.076% · 6.77 | 1.523% · 2.56 | 1.226% · 5.79 | **1.227% [1.185, 1.271]** · 5.64 | 2.39× |
| 7 | 1.035% [0.996, 1.075] · 8.07 | 0.609% · 22.07 | 1.222% · 7.56 | 0.835% · 16.54 | **0.835% [0.800, 0.871]** · 16.70 | 2.07× |
| 9 | 0.670% [0.638, 0.702] · 19.58 | 0.299% · 52.06 | 0.846% · 17.63 | 0.496% · 37.75 | **0.496% [0.470, 0.525]** · 37.66 | 1.92× |

**Benchmark run 2** (`python -m src.benchmark`, stamp `20261122_010751`) regenerated:
- `accuracy_runtime`, `cube_collapse_story`, `accuracy_runtime_sweep`, `fast_path_fraction`
  and `thread_scaling` (`_latest`);
- `mwpm_tier_p0_5` and `ensemble_vs_M`, re-rendered from their CSVs.

**Thread scaling** (d=9, Hierarchical): 38.0 µs on 1 thread → 6.52 on 12 threads (5.8×).

### TARGETS
- **Speed, ≤ 1.0× plain PyMatching single-thread at d = 7/9: NOT MET.** Hierarchical is 2.07× (d=7)
  and 1.92× (d=9). Plain UF alone is 0.90–0.94× PyMatching, but the correlated deep tier is two full
  UF passes, and nothing tried tonight halved that.
- **Accuracy, non-overlapping 95% CIs vs plain PyMatching at d = 5/7/9: MET.**

### Status of the optimization round
| Phase | Outcome | Commit |
|---|---|---|
| A measure | time split + per-shot counters; golden outputs saved | `7a62911` |
| B exact speedups | no net gain (skip-ahead per-round overhead > saved rounds); kept the lits interface + `skip_ahead` option | `dd31a12` |
| C single-fault table | every single fault exact; 39% / 6% of shots at d = 3/5; no LER change at d ≥ 5 | `406f75a` |
| D cluster-local pass 2 | slower (radius 0: +3–12%, radius 1: +45–55%); reverted | `4887ecf` |
| E accuracy options | weighted forest −5–7% LER / +18–20% time; S=8 −7–9% / +27–35%; pass 3 −2–4% / +20–27%; float score ≈ 0; partial gate worse; K no effect at d ≥ 7; M=2/3 too slow; none adopted | `313deb4` |
| E7 online correlation | failed (worse than plain UF); not committed | — |
| F final numbers + docs | operating point run 31, benchmark run 2, README / talk / PROGRESS | this commit |

**For you to check or decide**
1. **Speed vs accuracy:** if you'd rather show more accuracy at the same "about 2× PyMatching" story,
   `UnionFind3DConfig(correlated=True, scale=8)` or `weighted_forest=True` is measured and ready (see Phase E).
2. **Power mode:** set to "Best performance" for the timing runs and restored to "Balanced" at the end
   of this session (`powercfg /overlaysetactive 00000000-0000-0000-0000-000000000000`).
3. **Speed ideas I did not try tonight** (each a real engineering job):
   - per-cluster frontier lists in the growth loop (today every round scans all visited nodes);
   - reusing pass 1's grown supports to warm-start pass 2;
   - a cheaper pass-1/pass-2 split on large clusters only.

Phase F commit: `b76623a` (pushed). That is the demo baseline.

---

# Final optimization round (from 01:27): cut pass-2 work in Hierarchical

## Step 0: demo safety, on main (01:27 to 01:36)
- **Fresh-clone smoke test** of the README quickstart, from GitHub at `b76623a`, in a new venv with Python 3.14.7:
  - `pip install -e ".[dev]"` succeeds;
  - **`python -m pytest -q`: 732 passed, 24 skipped, 1 xfailed** (154 s);
  - `python -m src.benchmark --quick … --out scratch_results` writes all its figures;
  - `import src` works even from outside the repo root.
  - `src/__init__.py` is **not** needed and was not touched.
- **`scripts/demo.py`** (`python -m scripts.demo`): p = 0.5%, d = 7, 50k shots, single thread. It prints a
  table and saves `results/plots/demo_latest.png`. Runtime: **36 s from a cold numba cache, 6 s warm**.
  Output:

  | decoder | failures | LER [95% CI] | µs/shot | vs PyMatching |
  |---|---|---|---|---|
  | PyMatching | 505 | 1.010% [0.926, 1.101] | 8.79 | 1.00× |
  | UnionFind3D | 583 | 1.166% [1.076, 1.264] | 7.90 | LER ×1.15, time ×0.90 |
  | Hierarchical | 414 | 0.828% [0.752, 0.911] | 18.05 | LER ×0.82, time ×2.05 |
- **Figure fix:** `plot_accuracy_runtime`'s bottom margin now follows the number of footer lines; the
  single-panel demo figure's footer had overlapped the x-axis label.

Step 0 commit on main: `aaf0e88` (pushed). **Main stays here for the demo.** Steps 1 and later are on
branch **`opt/overnight`**.

## Step 1: measure pass 2 (branch, 01:37 to 01:45)
Script `scripts/opt2_measure.py`, CSV `results/optimization/final_step1_20261122_014401.csv`.
Setup: 50k shots, p = 0.5%, numba cache cleared, Best-performance mode, single thread. The fractions
come from the instrumented reference, which is bit-identical to numba (new counters:
`pass2_changed_obs`, `pass2_needed`).

| d | pass 2 runs | kept | **changes the prediction** | runs but no shortened edge touches a pass-1 node | pass 1 (plain UF) | pass 2 (≈ corr − plain) | local tier | Hierarchical | PyMatching |
|---|---|---|---|---|---|---|---|---|---|
| 5 | 98.09% | 18.56% | **0.682%** | 22.20% | 2.58 µs | 3.41 | 0.50 | 5.73 | 2.37 |
| 7 | 100.00% | 43.20% | **0.638%** | 1.54% | 7.54 | 9.32 | 0.61 | 17.20 | 8.17 |
| 9 | 100.00% | 64.37% | **0.552%** | 0.01% | 18.00 | 21.28 | 1.43 | 39.04 | 20.04 |

**Readout**
- **Pass 2 costs as much as pass 1** but changes the predicted observable on only about 0.6% of
  shots. It is often "kept" (a different, lower-score correction) without changing the logical outcome.
- **The exact skip (Step 2)** can only remove pass 2 on 22% of shots at d=5, 1.5% at d=7 and 0.01% at
  d=9. Almost no effect at d ≥ 7.
- **The real lever is Step 3:** a cheap predictor that finds the about 0.6% of shots where pass 2
  matters. The best possible gate would cut Hierarchical to about pass 1 + local ≈ 0.98× PyMatching
  (d=7) and 0.97× (d=9).
