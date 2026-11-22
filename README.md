# syndrome_hunters_ecc

Fast, parallel decoders for the rotated surface code, written from scratch and benchmarked
against PyMatching (minimum-weight perfect matching) under circuit-level noise in Stim.
MidQWest Northrop Grumman decoder challenge.

**What's here:** four decoders of our own, all on Stim's detector error model:

| Decoder | Registry name | One line |
|---|---|---|
| Cube-Collapse | `cube_collapse` | flatten the time axis, then local Clique rules + greedy completion (2D) |
| Cube3D-Local | (fast path, not standalone) | mutual-isolation rules R1–R3 on the 3D space-time graph |
| UnionFind3D | `union_find_3d` | weighted Delfosse–Nickerson union-find + peeling on the 3D graph |
| Hierarchical | `hierarchical`, `hierarchical_partial` | Cube3D-Local first, UnionFind3D only when needed |

PyMatching (plain and correlated) appears only as a reference baseline. It is never used
inside our decoders.

---

## 1. Quickstart

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"            # numpy, stim, pymatching, numba, matplotlib, pytest
```

| Task | Command |
|---|---|
| Run the test suite (writes test plots to `results/tests/<run>/`) | `python -m pytest -q` |
| Full benchmark + every standard figure (about 3 min on the laptop) | `python -m src.benchmark` or `make benchmark` |
| Quick smoke benchmark into a scratch folder | `python -m src.benchmark --quick --out scratch_results` |
| One part only | `python -m src.benchmark --only accuracy` (or `sweep`, `threads`) |
| Regenerate figures from saved CSVs | `python -m src.figures --accuracy results/accuracy_runtime_<ts>.csv --sweep results/sweep_<ts>.csv --threads results/thread_scaling_<ts>.csv` |
| Milestone experiments | `python -m scripts.m1_uf_experiments`, `python -m scripts.m2_hierarchical` |
| Operating point (5 decoders, d = 3–9, 5 batches × 50k shots, numba on 1 thread) | `python -m src.operating_point` |

Each benchmark writes CSVs to `results/` and figures to `results/plots/` as
`<name>_<timestamp>.png` plus `<name>_latest.png`. Nothing is ever overwritten. Every CSV
row records the CPU model, clock, power state, thread count, backend and git commit.

---

## 2. Algorithms

All of our decoders share one input: Stim's `detector_error_model(decompose_errors=True)` for a
rotated memory experiment (d rounds plus the final data-readout layer).

### 2.1 Shared 3D detector graph (`src/decoders/cube3d/graph.py`)
- **Nodes and edges:** nodes are detectors (x, y, t). Each two-detector DEM component becomes
  an edge; each one-detector component becomes an edge to a virtual boundary node.
- **Edge types and weights:** edges are typed space, time, diagonal (hook errors, Δt = 1) or
  boundary. Parallel components merge with an XOR of probabilities, and the weight is
  `w = log((1-p)/p)`.
- **Size at d=7, p = 0.5%:** 336 detectors and 1,558 edges (490 space, 288 time, 636 diagonal,
  144 boundary), maximum degree 12.
- **Checks:** 0 parallel edges with conflicting observables and 0 undetectable components
  (both tested).

### 2.2 Cube-Collapse (`src/decoders/cube_collapse/`): kept as a documented baseline
- **How it works:**
  1. Rebuild each check's raw value over time.
  2. Remove one-round blips with a 3-point majority, and keep the **last** filtered value. This
     gives a flat 2D syndrome.
  3. On the 2D check graph, run the Clique rules: S1 (lit node with exactly one lit neighbour),
     S2 (unlit centre with exactly two lit neighbours) and S3 (lone node next to the boundary).
  4. A growing-radius greedy pass matches whatever is left along precomputed shortest paths.
     Every shot is resolved, and `syndrome(correction) == flat` is asserted.
- **Complexity:** O(T·n) for the filter. The rules are O(n·deg) per shot and colour class. The
  greedy pass is O(k² log k) for k leftover events.
- **Parallelism:** fully over shots. Inside a shot, all rules of one conflict-graph colour run
  at once (centres at least 3 apart).
- **Deviations from Clique_L2 (Jia, Veerapaneni & Ravi 2025, arXiv 2507.11481):**
  - generalized from the square lattice to any check graph;
  - the paper's 3-lit-neighbour case is left to the greedy pass;
  - the filter uses the end state, not a whole-window majority;
  - leftovers go to a growing-radius greedy completion instead of being offloaded to a global decoder.
- **Finding:** flattening throws away the information that tells a measurement error from a
  data error. Even exact matching on the flattened syndrome gets worse with d (section 3).

### 2.3 Cube3D-Local, the fast path (`src/decoders/cube3d/local.py`)
- **How it works:** three rules on the 3D graph, applied to every edge type in the order
  R1 → R2 → R3:
  - **R1:** u's only lit neighbour is v, and v's only lit neighbour is u. Flip the
    most likely u–v edge.
  - **R2:** an unlit c has exactly two lit neighbours a and b, and each is the only lit node
    within distance 2 of the other. Flip a–c and c–b.
  - **R3:** a lit u with a boundary edge and no lit node within distance 2. Flip it to the
    boundary.
- **Scheduling:** two rule applications conflict when their closed neighbourhoods meet. The
  conflict graph is greedy-coloured (9 / 17 / 20 / 21 colours at d = 3/5/7/9, circuit-level).
  Each colour is evaluated against the state before that colour and then applied.
- **Gate:** the fast path accepts a shot only when it has ≤ K events (K=6) and the rules clear
  all of them.
- **Complexity:** O(k · deg · |ball₂|) per shot for k events, independent of n except for
  one O(n) scan.
- **Parallelism:** over shots (numba `prange`), and over the centres of one colour.
- **Backends:** NumPy reference (vectorized over shots and centres) and a numba kernel, which
  are **bit-identical** (tested).
  - The numba kernel keeps an incremental count of lit neighbours to prune candidates.
  - Option `strict_r1=True` requires distance-2 isolation for R1 too (see 2.5).

### 2.4 UnionFind3D, the deep tier (`src/decoders/cube3d/union_find.py`)
- **Growth:**
  - Integer edge lengths `L_e = max(1, round(S·w_e / w_min))`, with S = 4 by default.
  - Each lit detector starts as an odd cluster.
  - Every round, each odd cluster that doesn't touch the boundary adds one half-step to every
    incident edge it hasn't fully grown. All increments land before any merge.
  - Fully grown edges merge clusters (union by rank, path compression, parity XOR).
  - A cluster freezes once it is even or touches the boundary.
- **Peeling:** on a canonical spanning forest of the grown edges, built by BFS from the
  boundary first, then from nodes in ascending id. Only **odd** clusters may use their
  boundary edges. Leaves are peeled in reverse BFS order, and the prediction is the XOR of the
  peeled edges' observable bits.
- **Complexity:** almost linear in the cluster size, O(k·α(k)·deg·L_max) per shot. With
  per-chunk scratch buffers and sparse reset, an empty shot costs O(n) for the scan only.
- **Parallelism:** over shots, in chunks of 64 per thread.
- **Determinism:** the result doesn't depend on root choice or the iteration order inside a
  round, so the Python reference and the numba kernel are **bit-identical** (tested on every
  tier and distance, and on every S/growth variant).
- **Deviations from Delfosse–Nickerson (arXiv 1709.06218) and weighted UF:**
  - All odd clusters grow each round. The paper's smallest-cluster-first rule is available as
    `growth="smallest"`, but in Milestone 1 it was within the CI of "all" at every d and cost
    up to 1.21× the time.
  - Weights are quantized to small integers (S=4: lengths 4 to 16).
  - We add the odd-clusters-only boundary rule in peeling (section 4).

### 2.5 Hierarchical (`src/decoders/cube3d/hierarchical.py`)
- **How it works:** a predecoder plus a global decoder, the layout surveyed in arXiv 2307.14989.
  Per shot:
  - lit = 0: no flip;
  - lit > K: UnionFind3D;
  - otherwise run the local rules. If every event is cleared, the fast-path answer stands.
    If not:
    - `gate="whole_shot"` (default): discard the local work and run UF on the whole shot;
    - `gate="partial"`: run UF on the **residual**, and XOR the two predictions.
- **Parallelism:** one fused numba kernel (local rules, gate and UF), parallel over shots. A
  NumPy/Python reference is bit-identical (tested).
- **Deviation:** the plan allowed either gate. We measured both, and the partial gate costs
  accuracy (section 5), so whole-shot is the default.

---

## 3. The evaluation story

All runs: circuit-level noise (all four Stim noise parameters), rotated Z memory, rounds = d,
the same sampled shots (seed 0) for every decoder at each (d, p).

1. **Flattening fails** (`results/plots/cube_collapse_story_latest.png`). At p = 0.5%,
   PyMatching on Cube-Collapse's flattened syndrome gets *worse* with d, while full 3D
   matching gets better. The loss is in the flattening, not in our local rules, so the rest of
   the project works in 3D.
2. **3D local rules** (Cube3D-Local) never mis-decode an isolated fault: every single DEM
   instruction is decoded correctly whenever the fast path accepts it. But they can only
   finish sparse shots on their own (`fast_path_fraction_latest.png`).
3. **Our own UF** (UnionFind3D) decodes everything, and its failure rate falls with d at every
   p ≤ 0.5% (`accuracy_runtime_sweep_latest.png`). It is 1.00–1.27× PyMatching at p = 0.5%.
4. **Hierarchical**: the fast path takes the easy shots, and UF takes the rest.
   - **Accuracy:** exactly UF's failures at d ≥ 5, within the CI at d=3.
   - **Speed:** 1.2–3.0× faster single-thread than UF on sparse shots.
   - **Scaling:** both scale over threads (`thread_scaling_latest.png`).

The figures in `results/plots/` (`*_latest.png`) come from benchmark run 1
(`python -m src.benchmark`, commit `5e7b5ff`). Section 5 has the numbers.

---

## 4. Bugs found and fixed

- **UF boundary peeling (d=3).** With weighted lengths, a cheap boundary edge (L=2 at S=2)
  finished growing in the **same round** as the internal edge (L=3) that made the cluster even.
  Peeling then routed through the boundary and flipped the logical on *every* single internal
  fault at d=3. Fix, in both implementations: only odd clusters may use their boundary edges
  in the peeling forest. Test: every DEM component and every full DEM instruction decodes
  exactly (9 tier × distance cases).
- **Serial bottleneck outside the kernel.** `dets.sum(axis=1)` in NumPy cost about 0.35 µs/shot at
  d=9 and made the 12-thread Hierarchical 1.8× slower than UF (0.95 vs 0.53 µs). Lit counts
  are now computed inside the parallel kernels (0.59 vs 0.55 µs).
- **Per-shot allocation in numba.** About 16 array allocations per shot dominated sparse
  shots. Per-chunk workspaces with sparse reset took UF at d=3, p=0.1% from 1.02 to 0.46 µs/shot,
  with bit-identical results.
- **Partial gate accuracy.** R1's distance-1 isolation pairs the ends of two different chains in
  dense shots. d=9, p=0.5%: 1.066% vs UF's 0.922%. Distance-2 isolation (`strict_r1`) restores
  0.939%, but at 1.8× UF's time. Whole-shot stays the default.
- **Cube-Collapse isolated-pair variant** (Cube-Collapse phase). Making S1 stricter made things
  worse (1,120 vs 349 weight-3 failures at d=7), so it was reverted and documented.
- **Benchmark artefacts.** The laptop's speed drifted about 2.6× between sessions (same code, all
  decoders). Every CSV now records CPU, clock, power state and commit, and figures compare
  times only within one run.

---

## 5. Results

### Headline numbers (benchmark run 1, commit `5e7b5ff`)

Circuit-level p = 0.5%, rounds = d, 50,000 shots per point (seed 0, the same shots for every
decoder). Source: `results/accuracy_runtime_20261121_202412.csv`, figure
`results/plots/accuracy_runtime_latest.png`.

**Logical failure rate, % [95% Wilson CI]**

| Decoder | d = 3 | d = 5 | d = 7 | d = 9 |
|---|---|---|---|---|
| PyMatching (reference) | 1.704 [1.594, 1.821] | 1.384 [1.285, 1.490] | 1.010 [0.926, 1.102] | 0.690 [0.621, 0.766] |
| Correlated PyMatching (reference) | 1.648 [1.540, 1.763] | 1.074 [0.987, 1.168] | 0.628 [0.562, 0.701] | 0.278 [0.236, 0.328] |
| **UnionFind3D** | 1.710 [1.600, 1.827] | 1.498 [1.395, 1.608] | 1.166 [1.076, 1.264] | 0.876 [0.798, 0.962] |
| **Hierarchical** | 1.850 [1.736, 1.972] | 1.498 [1.395, 1.608] | 1.166 [1.076, 1.264] | 0.876 [0.798, 0.962] |
| Hierarchical (partial) | 1.858 [1.743, 1.980] | 1.498 [1.395, 1.608] | 1.246 [1.152, 1.347] | 0.994 [0.911, 1.085] |
| Cube-Collapse | 2.676 [2.538, 2.821] | 9.436 [9.183, 9.695] | 18.18 [17.85, 18.52] | 30.96 [30.56, 31.37] |
| PyMatching 2D on flat | 2.676 [2.538, 2.821] | 5.556 [5.359, 5.760] | 11.73 [11.45, 12.02] | 20.80 [20.45, 21.16] |

**Batch time, µs per shot: single thread (12 numba threads in brackets)**

| Decoder | d = 3 | d = 5 | d = 7 | d = 9 |
|---|---|---|---|---|
| PyMatching | 0.35 | 2.33 | 7.97 | 20.21 |
| Correlated PyMatching | 0.97 | 6.51 | 22.08 | 54.39 |
| **UnionFind3D** | 0.43 (0.11) | 2.30 (0.51) | 6.92 (0.99) | 16.50 (2.23) |
| **Hierarchical** | 0.34 (0.05) | 2.37 (0.35) | 6.98 (0.96) | 16.60 (2.26) |
| Hierarchical (partial) | 0.30 (0.04) | 2.84 (0.37) | 9.19 (1.26) | 22.20 (2.80) |
| Cube-Collapse | 0.55 | 6.39 | 23.63 | 44.08 |
| PyMatching 2D on flat | 0.31 | 1.46 | 4.07 | 10.08 |

PyMatching runs on one thread. **Single thread vs single thread is the fair comparison.**
The bracketed numbers show how our decoders scale. All times come from one run on one laptop
(i5-13420H, 12 logical CPUs, AC power, "Balanced" plan) and are not comparable across runs.

**Sweep** (100,000 shots per point, single thread; `results/sweep_20261121_202412.csv`).
Hierarchical / PyMatching LER at p = 0.1 / 0.2 / 0.3 / 0.5%:

| d | p = 0.1% | p = 0.2% | p = 0.3% | p = 0.5% |
|---|---|---|---|---|
| 3 | 84 vs 61 failures (1.38×) | 1.17× | 1.09× | 1.04× |
| 5 | 21 vs 19 (1.11×) | 1.22× | 1.09× | 1.09× |
| 7 | 0 vs 1 | 31 vs 21 (1.48×) | 1.23× | 1.15× |
| 9 | 1 vs 1 | 11 vs 6 (1.83×) | 1.15× | 1.29× |

**Thread scaling** (d = 7 / 9, p = 0.5%, 100,000 shots; `results/thread_scaling_20261121_202412.csv`):
- UnionFind3D: 6.96 → 1.04 µs (6.7×) at d=7 and 16.54 → 2.36 µs (7.0×) at d=9, from 1 to 12 threads.
- Hierarchical: 7.00 → 0.99 µs (7.1×) and 16.57 → 2.40 µs (6.9×).
- Single-thread PyMatching: 8.53 and 19.53 µs.

### Acceptance targets

| Target | Status | Numbers (p = 0.5%) |
|---|---|---|
| Hierarchical and UF3D LER decrease with d | **met** | H: 1.850 → 1.498 → 1.166 → 0.876%; UF: 1.710 → 1.498 → 1.166 → 0.876% |
| Hierarchical within the 95% CI of UF3D | **met** at every d and p measured | identical failure counts at d ≥ 5; d=3: 1.850% [1.736, 1.972] vs 1.710% [1.600, 1.827] |
| Hierarchical faster than UF3D | **partly met** | single thread: 1.2–3.0× faster where the fast path takes most shots (d ≤ 5 at p ≤ 0.2%, d=3 at p=0.5%, d=7 at p=0.1%); 0.95–1.0× elsewhere (fast path takes 0–30% of shots) |
| Stretch: Hierarchical ≤ 1.5× PyMatching at d=7 | **met** | 1.166 / 1.010 = 1.15× |

### Honest limitations
- **Union-find is not minimum-weight matching.** The gap to PyMatching grows with d: 1.27× at
  d=9, p=0.5%, and 1.83× at d=9, p=0.2% (11 vs 6 failures, so the ratio is noisy). Correlated
  PyMatching is better still (0.278% at d=9).
- **The fast path pays off only on sparse shots.** At d ≥ 7 and p = 0.5% it accepts ≤ 0.2% of
  shots, and Hierarchical is UF plus a scan. At d=3 the fast path makes 4–9% more failures than
  UF on the same shots (within the CI).
- **The partial gate is not worth it:** it is less accurate at d ≥ 7 for p = 0.5% and slower.
- **p = 0.8% is above threshold for every decoder here,** PyMatching included: its LER is
  4.06 → 4.89 → 5.24 → 5.31% for d = 3 → 9, and UF behaves the same way.
- **Timings come from one laptop.** Its absolute speed drifted up to 2.6× between sessions,
  and the sweep figure uses a single timing per point.
- **Cube-Collapse is kept as a documented negative result.** Its LER rises with d (2.7 → 31%).


---

## 6. Test suite

`python -m pytest -q` runs 464 tests in about 50 s on the laptop: 439 pass, 24 are skipped (stub decoders) and 1 is an expected failure (Cube-Collapse vs PyMatching at circuit-level d=5). Plots of the recorded metrics go to
`results/tests/<run>/` (`--no-plots` skips them).

| File | What it checks |
|---|---|
| `test_decoder_contract.py` | every registered decoder: registration, output shape and dtype, single shot == batch, deterministic and doesn't mutate its input, empty batch, rejects the wrong width |
| `test_known_cases.py` | d=3: every single DEM fault decodes to its own observable flip |
| `test_vs_baseline.py` | LER(ours) ≤ 1.5 × LER(PyMatching) + 3σ at d = 3/5 (Cube-Collapse at circuit-level d=5 is a documented xfail) |
| `test_cube3d_graph.py` | 3D graph vs Stim: counts match the DEM, every DEM component is an edge, edge types from coordinates, neighbour tables, caching |
| `test_cube3d_local.py` | fast path: every single component accepted and correct, full DEM instructions never accepted wrongly, measurement-error pair, hook error, R2/R3 fixtures, gate, accepted shots syndrome-consistent |
| `test_cube3d_uf.py` | UF: every component and DEM instruction exact, syndrome(correction) == events on random shots, numba == reference, every S/growth variant |
| `test_cube3d_hierarchical.py` | local numba == reference (all K, strict on/off), Hierarchical numba == reference (both gates), gate semantics, K=0 equals UF |
| `test_cc_*.py` | Cube-Collapse stages: filter series, graph vs code-capacity DEM, Clique fixtures, completion, exhaustive low-weight errors |
| `test_figures.py` | each figure function on a tiny synthetic CSV; never-overwrite; style registry unique |
| `test_operating_point.py` | PyMatching, Correlated PyMatching, PyMatching 2D on flat, UnionFind3D and Hierarchical at p = 0.5%, d = 3/5/7/9, 3 batches × 50k shots (`--op-batches`, `--op-shots`). Writes `results/operating_point/operating_point_run<N>.png`. At d=9, every 3D decoder beats d=7 with non-overlapping CIs, and 2D-on-flat gets worse |
| `test_performance.py` | Stim stage timings and decoder µs/shot vs distance (recorded, loose bounds) |

---

## Layout

```
src/
  circuits.py              Stim circuits for 3 noise tiers, DEMs
  machine.py               CPU / clock / power / git fingerprint for results CSVs
  benchmark.py             standard benchmark -> CSVs + figures
  figures.py               style registry + standard figures
  operating_point.py       accuracy vs batch time (used by pytest)
  decoders/
    base.py                Decoder interface; __init__.py holds the registry
    baseline.py            PyMatching wrapper (reference only)
    cube_collapse/         Cube-Collapse (2D) baseline
    cube3d/graph.py        shared 3D detector graph
    cube3d/local.py        Cube3D-Local fast path (NumPy + numba)
    cube3d/union_find.py   UnionFind3D (Python reference + numba)
    cube3d/hierarchical.py Hierarchical (fused numba kernel + reference)
scripts/                   milestone experiments (m1, m2)
docs/                      specs, overnight plan, PROGRESS.md, talk_outline.md
results/                   CSVs and figures (timestamped, never overwritten)
```
