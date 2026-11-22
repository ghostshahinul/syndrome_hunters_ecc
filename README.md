# syndrome_hunters_ecc

Fast, parallel decoders for the rotated surface code, written from scratch and benchmarked
against PyMatching (minimum-weight perfect matching) under circuit-level noise in Stim.
MidQWest Northrop Grumman decoder challenge.

**Headline:** our **Hierarchical** decoder (local fast path, then **correlated two-pass
union-find** whose second pass runs only on shots with a large first-pass cluster) beats plain
PyMatching's accuracy with non-overlapping 95% CIs at d = 5, 7 and 9 under circuit-level noise at
p = 0.5% (operating-point run 40). At d=9, for example: 0.511% [0.484, 0.540] vs 0.670% [0.638, 0.702], 250k shots. Single
thread vs single thread it costs **1.58× (d=7) and 1.54× (d=9)** PyMatching's time: closer, but
**not yet inside PyMatching's speed**. Correlated PyMatching is still more accurate (0.299% at d=9, run 40)
and 2.7× PyMatching's time.

**What's here:** eight decoders of our own, all on Stim's detector error model:

| Decoder | Registry name | One line |
|---|---|---|
| Cube-Collapse | `cube_collapse` | flatten the time axis, then local Clique rules + greedy completion (2D) |
| Cube3D-Local | (fast path, not standalone) | mutual-isolation rules R1–R3 on the 3D space-time graph |
| UnionFind3D | `union_find_3d` | weighted Delfosse–Nickerson union-find + peeling on the 3D graph |
| UF3D correlated | `uf3d_correlated` | two-pass UF: pass 2 makes the DEM partners of pass-1 edges cheaper (Y / hook correlations) |
| UF3D ensemble | `uf3d_ensemble` | M perturbed (correlated) UF variants per shot; the cheapest correction wins; parallel over variants |
| Hierarchical | `hierarchical`, `hierarchical_partial` | Cube3D-Local first, UF3D correlated only when needed |
| MWPM3D | `mwpm_3d` | our own exact minimum-weight perfect matching (Dijkstra + blossom), pure Python |
| Hierarchical + MWPM | `hierarchical_mwpm` | three tiers: MWPM3D re-decodes shots where UF built a large cluster |

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
| **One-minute demo:** PyMatching vs UnionFind3D vs Hierarchical at p = 0.5%, d = 7 (table + `results/plots/demo_latest.png`) | `python -m scripts.demo` |
| Run the test suite (writes test plots to `results/tests/<run>/`) | `python -m pytest -q` |
| Full benchmark + every standard figure (747 s ≈ 12.5 min measured on the i5-13420H laptop, AC power, Balanced mode) | `python -m src.benchmark` or `make benchmark` |
| Quick smoke benchmark into a scratch folder | `python -m src.benchmark --quick --out scratch_results` |
| One part only | `python -m src.benchmark --only accuracy` (or `sweep`, `threads`) |
| Regenerate figures from saved CSVs | `python -m src.figures --accuracy results/accuracy_runtime_<ts>.csv --sweep results/sweep_<ts>.csv --threads results/thread_scaling_<ts>.csv` |
| Milestone experiments | `python -m scripts.m1_uf_experiments`, `python -m scripts.m2_hierarchical`, `python -m scripts.m5_mwpm` |
| Operating point (5 decoders, d = 3–9, 5 batches × 50k shots, numba on 1 thread) | `python -m src.operating_point` |
| Error-rate sweep, p = 0.2–0.7% (same decoders, 10 batches × 50k shots); draws our style, a journal-style per-round figure, and an operating-point run at p = 0.5% | `python -m src.error_rate_sweep` |

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
  - lit > K: the deep tier. Since item 4 this is **UF3D correlated** (section 2.6);
    `HierarchicalConfig(uf=UnionFind3DConfig())` gives plain UF, the version measured in
    the run-1 tables;
  - otherwise run the local rules. If every event is cleared, the fast-path answer stands.
    If not:
    - `gate="whole_shot"` (default): discard the local work and run UF on the whole shot;
    - `gate="partial"`: run UF on the **residual**, and XOR the two predictions.
- **Parallelism:** one fused numba kernel (local rules, gate and UF), parallel over shots. A
  NumPy/Python reference is bit-identical (tested).
- **Deviation:** the plan allowed either gate. We measured both, and the partial gate costs
  accuracy (section 5), so whole-shot is the default.

### 2.6 UF3D correlated, two passes (`union_find.py`, `UnionFind3DConfig(correlated=True)`)
- **Correlation table:** the graph records which edge pairs come from one DEM error
  instruction, i.e. components separated by `^`: the X and Z parts of a Y fault, or the two
  halves of a hook error. Each pair gets its joint probability. At circuit-level d=7, 1,512 of
  1,558 edges have partners.
- **Pass 1:** plain UF.
- **Pass 2:** every partner f of a pass-1 edge e gets the length of P(f|e) = P(e, f)/p_e,
  clipped to [p_f, 0.5], for this shot only. Then UF runs again.
- **Choosing the result:** pass 2's correction is kept only if its *correlated score* is
  strictly lower. The score is Σ L_e minus, for every DEM partner pair inside the correction,
  max(0, L_e + L_f − L_joint). Without this check, a pass-1 edge that was equivalent but
  *different* pulled in its own partners and broke single faults.
- **Cost and exactness:** about 2× UF's work, since UF runs twice; all lengths are integers;
  numba is bit-identical to the Python reference.
- **Single faults:** the two passes alone mis-decode 2 of the 286 circuit-level d=3
  instructions (two three-component hook faults, p = 3.3e-4; correlated PyMatching misses 1).
  The registered decoder (`uf3d_correlated`, `table=True`) first looks up syndromes with
  1–4 events that are exactly one DEM fault's, so it decodes all 286 exactly
  (`single_fault_allowance = 0`).
- **Compared with correlated PyMatching:** that works with full matching and its own
  reweighting rule; ours is a cheaper heuristic on UF clusters.

### 2.7 UF3D ensemble (`src/decoders/cube3d/ensemble.py`)
- **How it works:**
  - M variants per shot. Variant 0 is unperturbed; variant v uses weights w·exp(0.2·z_v), with
    a fixed seed per variant, and odd variants grow smallest-first.
  - Each variant runs (correlated) UF.
  - The correction with the lowest score under the **original** weights wins.
- **Parallelism:** numba `prange` over variants (one shot on up to M cores) or over shots.
  Both are bit-identical to the reference.
- **Results:**
  - It improves accuracy steadily with M when correlated, and matches correlated
    PyMatching's CI at d = 5/7.
  - It costs 20–25× PyMatching single-thread at M=8, so it is an accuracy demonstration, not
    a speed contender.
  - The plain (non-correlated) ensemble did **not** help.

### 2.8 MWPM3D and the three-tier decoder (`src/decoders/cube3d/mwpm.py`, `three_tier.py`)
- **MWPM3D, our own exact matching:**
  - scipy Dijkstra from every lit detector over the 3D graph plus the boundary;
  - a matching graph with one boundary copy per detector, solved by `networkx.max_weight_matching`
    (exact blossom) on negated lengths;
  - the prediction is the XOR of the observable bits along each matched shortest path.
- **Sparse but exact:** a detector pair edge is kept only when it is shorter than sending both
  detectors to the boundary, and boundary copies pair only along kept edges.
- **Checked against PyMatching:** it agrees on **16,771 of 16,772** shots (the one difference is a
  degenerate tie). It is exact, but the blossom is pure Python: 0.3–1.2 ms per shot at d = 3/5
  and 7–42 ms at d = 7/9, p = 0.5%.
- **Three tiers:** Hierarchical decodes every shot and reports the node count of UF's largest
  final cluster. Shots above a threshold T (default 2·rounds) are re-decoded by MWPM3D.
- **Complexity:** O(k³) blossom on k lit detectors plus k Dijkstra runs. Shots are
  independent, so the routed shots could also run in parallel; they don't yet.

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
5. **Correlations**: decoding each sublattice independently throws away the fact that one Y or hook
   fault lights edges in both. Correlated two-pass UF puts that information back, and as
   Hierarchical's deep tier it **beats plain PyMatching** at d = 5/7/9
   (`results/operating_point/operating_point_run22.png`). Accuracy in the sweep is in
   `accuracy_runtime_sweep_latest.png`.
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

### Latest: Hierarchical with correlated UF vs PyMatching (operating-point run 40, commit `5660331`)
p = 0.5%, circuit-level, rounds = d, 5 batches × 50,000 = 250,000 shots per point (seeds 0–4,
the same for every decoder). **Every decoder single-thread** (Windows "Best performance" power
mode, numba cache cleared and warmed up); time is the mean batch µs/shot. Source:
`results/operating_point/operating_point_run40.csv` and `operating_point_run40.png`.

| d | PyMatching | Correlated PyMatching | UnionFind3D | UF3D correlated | **Hierarchical** | Hierarchical ÷ PM time |
|---|---|---|---|---|---|---|
| 3 | 1.656% [1.607, 1.707]; 0.28 µs | 1.625%; 0.85 µs | 1.667%; 0.56 µs | 1.608%; 0.81 µs | 1.703% [1.653, 1.755]; 0.45 µs | 1.57× |
| 5 | 1.383% [1.338, 1.429]; 2.31 µs | 1.076% [1.036, 1.117]; 6.51 µs | 1.523%; 2.47 µs | 1.234%; 4.53 µs | **1.236% [1.193, 1.280]**; 4.66 µs | 2.01× |
| 7 | 1.035% [0.996, 1.075]; 8.05 µs | 0.609% [0.579, 0.640]; 21.60 µs | 1.222%; 7.36 µs | 0.859%; 12.40 µs | **0.859% [0.823, 0.896]**; 12.70 µs | **1.58×** |
| 9 | 0.670% [0.638, 0.702]; 19.14 µs | 0.299% [0.278, 0.321]; 51.28 µs | 0.846%; 17.10 µs | 0.511%; 29.51 µs | **0.511% [0.484, 0.540]**; 29.55 µs | **1.54×** |

- **Accuracy target met:** Hierarchical beats plain PyMatching with non-overlapping 95% CIs at
  d = 5, 7 and 9. At d=3 the CIs overlap (1.703% vs 1.656%).
- **Speed target (≤ 1.0× plain PyMatching single-thread at d = 7/9) not met:** 1.58× and 1.54×.
  That is down from 2.07× / 1.92× before the final optimization round, and Hierarchical is now
  0.58–0.59× of correlated PyMatching's time.
- **What made it faster** (branch `opt/overnight`, merged; details in `docs/PROGRESS.md`):
  - Pass 2 changes the predicted observable on only about 0.6% of shots, but it ran on 100% of
    them. It now runs only when pass 1's largest cluster has more than ⌊rounds²/6.5⌋ nodes (1/3/7/12 at
    d = 3/5/7/9), and is skipped exactly when no shortened edge touches a pass-1 node.
  - This costs at most +0.024% absolute LER (d=7), inside the 95% CI of always running pass 2.
  - The gate was fitted at p = 0.5%.
- **What didn't** (measured, reverted): skip-ahead growth, sorted scans, cluster-local pass 2,
  frontier lists, and online single-pass correlation. Growth is 70–73% of UF's time and is
  dominated by the support increments themselves.
- **Accurate options** (off by default, measured at 250k shots):
  - **`weighted_forest`:** −5–7% LER for +18–20% time;
  - **`scale=8`:** −7–9% LER for +27–35% time;
  - **the correlated ensemble with M=8–12:** within correlated PyMatching's CI at d = 5/7, but at 20–25× the time.
- **Single-fault table:** every single DEM fault decodes exactly. 39% of d=3 shots and 6% of d=5
  shots are answered by table lookup.
- **Multi-core** (reported separately, not a race): see `thread_scaling_latest.png`. PyMatching
  can be parallelized across processes the same way.

### Error-rate sweep run 1: p = 0.2–0.7%, plugged in (operating-point run 44 is its p = 0.5% slice)
Same five decoders, d = 3–9, rounds = d, 10 batches × 50,000 = 500,000 shots per point, every
decoder single-thread, AC power + "Best performance". Source:
`results/error_rate_sweep/sweep_run1.csv`. Figures: `sweep_run1_ours.png` (our style, LER per
shot) and `sweep_run1_paper.png` (journal style, LER per round ε = [1 − (1 − 2P_L)^(1/d)]/2).

Hierarchical vs plain PyMatching, logical failures per shot, % [95% CI]:

| p | d = 5 | d = 7 | d = 9 |
|---|---|---|---|
| 0.2% | 0.086 [0.078, 0.094] vs 0.101 [0.093, 0.110] | 0.023 [0.019, 0.028] vs 0.031 [0.026, 0.036] | 0.005 [0.003, 0.007] vs 0.007 [0.005, 0.010] |
| 0.3% | 0.278 [0.264, 0.293] vs 0.330 [0.314, 0.346] | 0.110 [0.102, 0.120] vs 0.137 [0.127, 0.148] | 0.036 [0.031, 0.042] vs 0.053 [0.047, 0.060] |
| 0.5% | 1.235 [1.205, 1.266] vs 1.401 [1.369, 1.434] | 0.839 [0.814, 0.865] vs 0.995 [0.967, 1.022] | 0.518 [0.498, 0.538] vs 0.671 [0.649, 0.694] |
| 0.7% | 3.149 [3.101, 3.198] vs 3.451 [3.401, 3.502] | 2.957 [2.910, 3.004] vs 3.331 [3.281, 3.381] | 2.708 [2.663, 2.753] vs 3.124 [3.076, 3.172] |

- Hierarchical has the lower LER at every p for d = 5/7/9. The CIs are separate for p ≥ 0.3%
  and overlap at p = 0.2%, where there are only 25–429 failures per point.
- Correlated PyMatching stays the most accurate for d ≥ 5 at every p.
- At d = 3, p ≤ 0.3%, correlated PyMatching is *worse* than plain PyMatching (0.344% vs 0.304% at 0.2%),
  and Hierarchical's fast path costs a little (0.333% vs UF3D correlated 0.302%).
- Time at p = 0.5% (run 44): Hierarchical ÷ PyMatching = 1.99× (d=5), 1.58× (d=7), 1.54× (d=9),
  matching run 40.
- Slide version: `sweep_run1_slides.png` shows d = 5/7/9, four decoders, large type, with Hierarchical drawn last and thicker.
  Replot every figure from the saved CSVs (no decoding) with
  `python -m src.error_rate_sweep --replot 1 --op-run 44`. The replot also prints the table below.

**Error suppression per +2 distance**, Λ = P_L(d) / P_L(d+2) from the per-shot failure rates
(higher is better; Λ > 1 means the larger code fails less):

| p | Decoder | d=3 → 5 | d=5 → 7 | d=7 → 9 |
|---|---|---|---|---|
| 0.2% | PyMatching | 3.01 | 3.29 | 4.28 |
| 0.2% | Correlated PyMatching | 4.72 | 4.74 | 5.13 |
| 0.2% | UnionFind3D | 2.68 | 3.43 | 3.02 |
| 0.2% | Hierarchical | 3.88 | 3.70 | 4.64 |
| 0.3% | PyMatching | 1.98 | 2.41 | 2.59 |
| 0.3% | Correlated PyMatching | 2.82 | 3.36 | 3.58 |
| 0.3% | UnionFind3D | 1.80 | 2.20 | 2.27 |
| 0.3% | Hierarchical | 2.49 | 2.52 | 3.07 |
| 0.4% | PyMatching | 1.48 | 1.83 | 1.79 |
| 0.4% | Correlated PyMatching | 1.90 | 2.48 | 2.34 |
| 0.4% | UnionFind3D | 1.39 | 1.63 | 1.70 |
| 0.4% | Hierarchical | 1.77 | 1.92 | 1.98 |
| 0.5% | PyMatching | 1.18 | 1.41 | 1.48 |
| 0.5% | Correlated PyMatching | 1.49 | 1.81 | 1.97 |
| 0.5% | UnionFind3D | 1.08 | 1.29 | 1.40 |
| 0.5% | Hierarchical | 1.37 | 1.47 | 1.62 |
| 0.6% | PyMatching | 1.02 | 1.18 | 1.24 |
| 0.6% | Correlated PyMatching | 1.25 | 1.49 | 1.55 |
| 0.6% | UnionFind3D | 0.94 | 1.12 | 1.16 |
| 0.6% | Hierarchical | 1.16 | 1.26 | 1.31 |
| 0.7% | PyMatching | 0.91 | 1.04 | 1.07 |
| 0.7% | Correlated PyMatching | 1.09 | 1.22 | 1.28 |
| 0.7% | UnionFind3D | 0.91 | 0.98 | 1.01 |
| 0.7% | Hierarchical | 1.01 | 1.06 | 1.09 |

- **p = 0.2% has few failures:** at d = 9 only 15–57 per decoder, at d = 7 77–172, out of 500,000
  shots. Λ(7 → 9) there has a 95% uncertainty of roughly ±30–55% (≈ 1.96·√(1/k₉ + 1/k₇)), so the
  ordering within the 0.2% rows is not meaningful (e.g. UnionFind3D's 3.43 → 3.02).
- Hierarchical's Λ is above plain PyMatching's in all 18 (p, d) cells. Correlated PyMatching's is the highest at every
  point.
- At p = 0.7% Λ is close to 1 for every decoder, and below 1 for d = 3 → 5 (PyMatching and
  UnionFind3D 0.91; UnionFind3D also at 0.6%, 0.94). Near p = 0.7%, a larger distance no longer
  reduces the failure rate.

### Earlier: benchmark run 1, before correlations (commit `5e7b5ff`)


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

**Batch time, µs per shot, every decoder single-thread** (the fair comparison: PyMatching is
single-threaded per decode call, and so is everything in this table)

| Decoder | d = 3 | d = 5 | d = 7 | d = 9 |
|---|---|---|---|---|
| PyMatching | 0.35 | 2.33 | 7.97 | 20.21 |
| Correlated PyMatching | 0.97 | 6.51 | 22.08 | 54.39 |
| **UnionFind3D** | 0.43 | 2.30 | 6.92 | 16.50 |
| **Hierarchical** | 0.34 | 2.37 | 6.98 | 16.60 |
| Hierarchical (partial) | 0.30 | 2.84 | 9.19 | 22.20 |
| Cube-Collapse | 0.55 | 6.39 | 23.63 | 44.08 |
| PyMatching 2D on flat | 0.31 | 1.46 | 4.07 | 10.08 |

All times come from one run on one laptop (i5-13420H, 12 logical CPUs, AC power, "Balanced"
plan) and are not comparable across runs. Multi-core numbers are kept separate (next section).

### Parallelism, reported separately
Two different things, and neither belongs in a race against single-thread PyMatching:
- **Throughput (parallel over shots).** Shots are independent, so numba `prange` splits a
  batch across threads, with 1 core per shot. UnionFind3D / Hierarchical go from 6.96 / 7.00 µs to
  1.04 / 0.99 µs per shot at d=7 on 12 threads (6.7× / 7.1×) (`thread_scaling_latest.png`).
  **PyMatching can do the same** by running separate shot batches in parallel processes, so
  this shows how our kernels scale, not that we are faster.
- **Latency (cores used per shot).** The ensemble decoder can split *one* shot's M variants
  across cores (`parallel="variants"`). With M=8 at d=7 one shot takes 263 µs on 1 core and
  58 µs on 8 cores (cores used per shot = 8). Plain PyMatching has no per-shot parallelism.

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

### Acceptance targets (benchmark run 1, commit `5e7b5ff`: plain-UF deep tier, 50k shots)

In sweep run 1 (500k shots, correlated deep tier), Hierarchical at d=3 is *outside* its deep tier's
95% CI for p ≥ 0.3% (see Honest limitations). At d = 5–9 the two differ by at most 0.0022
percentage points.

| Target | Status | Numbers (p = 0.5%) |
|---|---|---|
| Hierarchical and UF3D LER decrease with d | **met** | H: 1.850 → 1.498 → 1.166 → 0.876%; UF: 1.710 → 1.498 → 1.166 → 0.876% |
| Hierarchical within the 95% CI of UF3D | **met** at every d and p measured | identical failure counts at d ≥ 5; d=3: 1.850% [1.736, 1.972] vs 1.710% [1.600, 1.827] |
| Hierarchical faster than UF3D | **partly met** | single thread: 1.2–3.0× faster where the fast path takes most shots (d ≤ 5 at p ≤ 0.2%, d=3 at p=0.5%, d=7 at p=0.1%); 0.95–1.0× elsewhere (fast path takes 0–30% of shots) |
| Stretch: Hierarchical ≤ 1.5× PyMatching at d=7 | **met** | 1.166 / 1.010 = 1.15× |

### Third tier: MWPM3D on UF's largest clusters (Milestone 5)
Source: `results/mwpm/mwpm_tier_20261121_205911.csv`, `results/plots/mwpm_tier_p0_5_latest.png`.
Circuit-level p = 0.5%, 10,000 shots, single thread. Time = Hierarchical batch time plus the
MWPM time of the routed shots, per shot.

| d | PyMatching | UnionFind3D | Hierarchical + MWPM (T, share routed) | µs/shot: UF → three tiers |
|---|---|---|---|---|
| 5 | 165 failures | 180 | 171 (T=5, 18.9%) | 2.26 → 228 |
| 7 | 75 | 101 | 84 (T=14, 17.4%); 76 (T=7, 44%) | 6.78 → 1,155; 3,220 |
| 9 | 72 | 90 | 78 (T=27, 15.6%) | 19.9 → 5,681 |

- **The trigger finds UF's mistakes.** At d = 7/9 it removes about two-thirds of the UF-vs-PyMatching
  gap while routing about 16% of shots.
- **The cost is not acceptable:** 170–290× UF's time. To be practical the tier needs a compiled
  matcher (a C/C++ blossom) or a cluster-local match instead of a whole-shot one.
- **At p ≤ 0.2%** there were too few failures in 20,000 shots (0–66 per point) to measure a gain.

### Honest limitations
- **"Beats PyMatching" means plain PyMatching only.** Correlated PyMatching is still clearly
  more accurate (0.609% / 0.299% at d = 7/9 vs Hierarchical's 0.859% / 0.511%, run 40), and
  Hierarchical's speed is 1.54–2.01× plain PyMatching single-thread at d = 5–9 (run 40), not
  equal to it. The ensemble closes most of the accuracy gap to correlated PyMatching, but only at
  20–25× the cost.
- **The pass-2 gate threshold floor(rounds²/6.5) was fitted at p = 0.5% only;** at p = 0.2%
  there are too few failures to validate its accuracy.
- **Union-find is not minimum-weight matching.** Plain UnionFind3D's gap to PyMatching grows with
  d: 1.10× / 1.18× / 1.26× at d = 5/7/9, p = 0.5% (run 40), and 1.58× at d=9, p=0.2% (57 vs 36
  failures, sweep run 1, so that ratio is noisy).
- **The fast path pays off only on sparse shots.** At d ≥ 7 and p = 0.5% it accepts ≤ 0.2% of
  shots (benchmark run 1), and Hierarchical is UF plus a scan. At d=3 the fast path costs
  accuracy: Hierarchical fails 4.5–10% more often than its own deep tier (UF3D correlated) on
  the same shots, with separate 95% CIs for p ≥ 0.3% (sweep run 1, 500k shots).
- **The partial gate is not worth it:** it is less accurate at d ≥ 7 for p = 0.5% and slower
  (benchmark run 1).
- **p = 0.8% is above threshold for every decoder here,** PyMatching included: its LER is
  4.06 → 4.89 → 5.24 → 5.31% for d = 3 → 9, and UF behaves the same way (benchmark sweep run 1,
  `results/sweep_20261121_202412.csv`).
- **Timings come from one laptop.** Its absolute speed drifted up to 2.6× between sessions,
  and the sweep figure uses a single timing per point.
- **Cube-Collapse is kept as a documented negative result.** Its LER rises with d (2.7 → 31%).
- **MWPM3D is exact but slow** (pure-Python blossom), so the three-tier decoder is a
  demonstration of the accuracy headroom, not a fast decoder.


---

## 6. Test suite

`python -m pytest -q` runs 803 tests: 802 pass, 0 are skipped and 1 is an expected failure (Cube-Collapse vs PyMatching at circuit-level d=5). On the laptop it takes 313 s right after the numba cache is cleared (JIT compiles everything) and 103–208 s on later runs.
The suite has two skip conditions, and neither fires on a full checkout:
`conftest.build_or_skip` skips a registered decoder that raises `NotImplementedError`, which none do any more (the 24 skips
in earlier counts were the `union_find` / `experimental` stubs, since deleted), and `test_golden.py` skips if
`tests/golden/golden_outputs.npz` is missing. Plots of the recorded metrics go to
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
| `test_cube3d_mwpm.py` | MWPM3D decodes every single DEM component exactly, agrees with PyMatching (≤ 1% of shots may differ by ties), three-tier routing semantics (T = ∞ equals Hierarchical, T = 0 sends every UF shot to MWPM) |
| `test_figures.py` | each figure function on a tiny synthetic CSV; never-overwrite; style registry unique |
| `test_operating_point.py` | PyMatching, Correlated PyMatching, UnionFind3D, UF3D correlated and Hierarchical at p = 0.5%, d = 3/5/7/9, 3 batches × 50k shots (`--op-batches`, `--op-shots`). Writes `results/operating_point/operating_point_run<N>.png`. At d=9, every 3D decoder beats d=7 with non-overlapping CIs |
| `test_performance.py` | Stim stage timings and decoder µs/shot vs distance (recorded, loose bounds) |

---

## Other team explorations (separate branches)

These were developed in parallel by teammates. They are **not merged into `main`** and are **not
included in any benchmark, table or figure in this README**. The descriptions follow each
branch's own README and docstrings.

- **FaultJoin decoder**, branch [`shahi_2`](https://github.com/ghostshahinul/syndrome_hunters_ecc/tree/shahi_2).
  - On sparse syndromes, it lets whole DEM faults vote for a logical class: an exact vote over explanations of
    zero, one or two faults.
  - When the vote finds no explanation, it conditions union–find growth costs on the clicks of correlated partner
    detectors.
- **NVIDIA Ising hybrid decoder**, branch [`shahi_3`](https://github.com/ghostshahinul/syndrome_hunters_ecc/tree/shahi_3)
  (it also contains the FaultJoin commit).
  - NVIDIA's pretrained Fast R9 neural predecoder predicts local corrections and a logical frame.
  - Ordinary or correlated PyMatching then decodes the residual syndrome, and the two logical predictions are XORed.
  - It runs on NVIDIA's own Stim circuit, whose detectors and DEM differ from the circuits used here.
- **MWPM pros/cons notes**, branch
  [`aRAYofsunshine07-patch-1`](https://github.com/ghostshahinul/syndrome_hunters_ecc/tree/aRAYofsunshine07-patch-1).
  It adds notes on minimum-weight perfect matching's strengths and weaknesses to `notebooks/MWPM.txt`, with a link
  to fusion-blossom.

---

## Layout

```
src/
  circuits.py              Stim circuits for 3 noise tiers, DEMs
  machine.py               CPU / clock / power / git fingerprint for results CSVs
  benchmark.py             standard benchmark -> CSVs + figures
  figures.py               style registry + standard figures
  operating_point.py       accuracy vs batch time (used by pytest)
  error_rate_sweep.py      LER vs p sweep: our-style, journal-style and slide figures, replot, Λ table
  decoders/
    base.py                Decoder interface; __init__.py holds the registry
    baseline.py            PyMatching wrapper (reference only)
    cube_collapse/         Cube-Collapse (2D) baseline
    cube3d/graph.py        shared 3D detector graph
    cube3d/local.py        Cube3D-Local fast path (NumPy + numba)
    cube3d/union_find.py   UnionFind3D (Python reference + numba)
    cube3d/hierarchical.py Hierarchical (fused numba kernel + reference)
    cube3d/ensemble.py     UF3D ensemble (M perturbed correlated-UF variants, parallel over shots or variants)
    cube3d/mwpm.py         MWPM3D (Dijkstra + networkx blossom)
    cube3d/three_tier.py   Hierarchical + MWPM (three tiers)
scripts/
  demo.py                  one-minute demo (python -m scripts.demo)
  m1_uf_experiments.py, m2_hierarchical.py, m5_mwpm.py, item2_ensemble.py   milestone experiments
  experiments/             one-off optimization tools (run from the repo root as python -m scripts.experiments.<name>)
    opt_a_measure.py       Phase A: baseline timings and pass/cluster statistics before optimizing
    opt_b_profile.py       Phase B: times current kernels against an older git revision's, in one process
    opt_ab_compare.py      Phases C–E: compare decoder configurations on the same shots in one run
    opt2_measure.py        final round, Step 1: how often and how expensively correlated UF's pass 2 matters
    final_compare.py       final round, Step 5: Hierarchical on a branch vs on main, timed in one run
    make_golden.py         writes tests/golden/golden_outputs.npz (bit-identity reference for test_golden.py)
docs/                      specs, overnight plan, PROGRESS.md, talk_outline.md
results/                   CSVs and figures (timestamped, never overwritten)
```
