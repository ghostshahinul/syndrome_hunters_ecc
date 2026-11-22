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
