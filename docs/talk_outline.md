# Talk outline: 10 minutes, 8 slides + one whiteboard segment

Team: syndrome_hunters_ecc · MidQWest Northrop Grumman decoder challenge.
Figures live in `results/plots/` (`<name>_latest.png`). Regenerate them all with
`python -m src.benchmark`. Times below are from one laptop run, so compare them only with
each other.

| # | Slide | Time | Figure |
|---|---|---|---|
| 1 | The problem: decode fast enough, and keep getting better as the code grows | 0:45 | none (title + one-line goal) |
| 2 | Whiteboard: a 5-qubit repetition code and union-find growth | 2:00 | whiteboard |
| 3 | First idea, Cube-Collapse, and why flattening time fails | 1:15 | `cube_collapse_story_latest.png` |
| 4 | Go 3D: one shared space-time detector graph | 0:45 | `accuracy_runtime_latest.png` (only the d=7 panel, with PyMatching 3D vs 2D on flat) |
| 5 | Our own UnionFind3D: weighted growth, peeling, and a bug we caught | 1:30 | `accuracy_runtime_sweep_latest.png` |
| 6 | Hierarchical: the local fast path first, UF only when needed | 1:30 | `fast_path_fraction_latest.png` |
| 7 | Parallel by construction: every shot is independent | 1:00 | `thread_scaling_latest.png` |
| 8 | Results, honest limits, what's next | 1:15 | `accuracy_runtime_latest.png` (all panels) + results table |

---

## 1. The problem (0:45)
- A surface-code decoder reads the detection events from d rounds of stabilizer measurements and
  has to say whether the logical qubit flipped, faster than the hardware produces new rounds.
- We're scored on accuracy (logical failure rate) and speed (µs per shot). The real test
  is whether the failure rate **falls as d grows**.
- What we built: 4 decoders of our own (Cube-Collapse, Cube3D-Local, UnionFind3D,
  Hierarchical). PyMatching is used only as the yardstick.

## 2. Whiteboard: repetition code and union-find (2:00)
Draw 5 data qubits in a line, `D1 D2 D3 D4 D5`, with 4 parity checks between neighbours,
`C12 C23 C34 C45`. Both ends of the line act as the boundary.
- **Syndrome.** Flip D3. Checks C23 and C34 light up. A chain of flips lights up only its two ends.
  Flip D1 instead: only C12 lights up, because D1's other side is the boundary.
- **Union-find growth.** Each lit check starts its own cluster. Every round, each odd cluster
  grows half an edge in every direction. After one round, C23 and C34 meet on D3's edge, the
  merged cluster has 2 defects (even parity), and it stops. With C12 alone, the cluster grows
  until it reaches the boundary, and it stops there.
- **Peeling.** Build a spanning tree inside each cluster and remove leaves one at a time: when the
  leaf is lit, flip its tree edge and toggle its parent. The flipped edges are the correction,
  here D3 (or D1 to the boundary).
- **Time.** Add a second copy of the line to represent the next round. A measurement error
  then lights up the same check in two consecutive rounds, which is a vertical edge. That is why the
  real graph is 3D (x, y, t). Union-find works unchanged on it.

## 3. Cube-Collapse and why flattening fails (1:15)
- Our first idea: squash the time axis. A majority filter over each check's history gives a 2D
  syndrome, and fast local "Clique" rules from arXiv 2507.11481 plus a greedy pass to the
  boundary resolve it. It is fully parallel and resolves every shot.
- The figure shows the failure rate going **up** with d at p = 0.5%. Even exact matching
  (PyMatching) on the same flattened syndrome goes up: 2.68 → 5.56 → 11.73% at d = 3/5/7.
  Full 3D matching goes down: 1.70 → 1.38 → 1.01%.
- So the loss comes from flattening, not from our rules. Measurement errors and data errors
  look the same once time is gone. That finding sent us to 3D.

## 4. One shared 3D graph (0:45)
- Built straight from Stim's detector error model. Edges are space, time, diagonal
  (hook errors) and boundary. At d=7: 336 detectors and 1,558 edges, with every edge
  weighted log((1-p)/p).
- Every new decoder (local rules, union-find, hierarchical) runs on this one graph, and tests
  check it against Stim, for example by decoding every single DEM fault exactly.

## 5. UnionFind3D (1:30)
- Our own Delfosse–Nickerson union-find. Integer edge lengths `round(4·w / w_min)` give weighted
  growth with integer arithmetic, and the result is peeled on a canonical spanning forest.
- There's a NumPy/Python reference and a numba kernel, and they are **bit-identical**
  (tested on every tier and distance).
- **Bug we found:** at d=3 every single internal fault flipped the logical. A cheap boundary
  edge finished growing in the same round as the internal edge, and peeling went through the
  boundary. Fix: only odd clusters may peel through the boundary. Every DEM fault now decodes
  exactly.
- Tuning: weight resolution S=2 → 4 cut the d=9 failure rate from 1.14% to 0.88% for 1.19× time.
  Smallest-cluster-first growth made no measurable difference.
- On the sweep figure UF falls with d for every p ≤ 0.5%. p = 0.8% is above threshold for
  every decoder, PyMatching included. At p = 0.5%, UF's failure rate is 1.00–1.27× PyMatching's.

## 6. Hierarchical (1:30)
- Fast path (Cube3D-Local) on the 3D graph. Three rules clear mutually isolated pairs, pairs
  two steps apart, and isolated nodes next to the boundary. They're scheduled by graph coloring,
  so rules of one color never interact.
- Whole-shot gate: if the shot has ≤ K=6 events and the rules clear all of them, we're done.
  Otherwise UF decodes the whole shot. At d ≥ 5 it makes **exactly the same failures** as UF,
  and it is within the CI at d=3.
- The figure shows the fast path deciding 94–99.7% of shots at d ≤ 5, p ≤ 0.2%, and 0% at
  d=9, p=0.5%. Single-thread speed-up over UF: 3.0× (d=3, p=0.1%), 1.5× (d=5), and about 1.0×
  where the fast path rarely applies.
- Negative result: a "partial" gate (rules strip the isolated pairs, UF decodes what's left) costs
  accuracy at d=9 p=0.5% (1.07% vs 0.92%). Distance-2 isolation fixes the accuracy but is
  1.8× slower, so whole-shot is the default.

## 7. Parallel by construction (1:00)
- Shots are independent, so one fused numba `prange` kernel decodes chunks of 64 shots per
  thread, reusing scratch memory and resetting only what each shot touched.
- The figure shows µs/shot vs threads at d = 7 and 9, with single-thread PyMatching as the
  dashed line. The fair comparison is single-thread vs single-thread. Threads show how well
  the decoder scales.
- The local rules are parallel inside a shot as well: one color class at a time, all
  centers at once.

## 8. Results and limits (1:15)
- Show the full operating-point figure and the README results table.
  - Accuracy: Hierarchical ≈ UF, 1.15× PyMatching at d=7, p=0.5% (target was ≤ 1.5×).
  - Speed: single-thread, faster than PyMatching at d = 7 and 9 for p = 0.5%, slower for p ≤ 0.2%.
- Limits:
  - Union-find is not MWPM; the gap grows with d (1.29× at d=9).
  - The fast path only pays off when shots are sparse.
  - All timings come from one laptop.
- Next: a C/OpenMP backend, or our own MWPM tier for the hardest shots.
