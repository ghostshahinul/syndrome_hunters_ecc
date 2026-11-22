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
