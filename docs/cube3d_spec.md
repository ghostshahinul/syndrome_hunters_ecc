New direction based on our evaluation (results/plots/accuracy_runtime_latest.png, circuit-level, p = 0.5%, rounds = d, 50k shots):
- PyMatching 3D: 1.7% / 1.4% / 1.0% at d = 3 / 5 / 7 (improves with d, as it should)
- PyMatching 2D on the flattened syndrome: 2.6% / 5.5% / 11.7% (gets WORSE with d)
- Cube-Collapse: 2.6% / 9.4% / 18.1%, and the slowest at d = 7 (~23 µs/shot vs ~9 µs PyMatching)
Conclusion: collapsing time into 2D is lossy under circuit-level noise; even an optimal matcher fails on the flattened input. About two thirds of our loss is from flattening and one third from the greedy local stages. We keep the cube but drop the collapse.

Save this message to docs/cube3d_spec.md. Keep Cube-Collapse intact as its own decoder (it's a documented baseline and a finding for the talk). Use pymatching ONLY as a reference baseline, never inside our decoders. Do the steps in order; stop and report numbers after each.

STEP 1: 3D detector graph (shared by all new decoders)
- Build from circuit.detector_error_model(decompose_errors=True): nodes = ALL detectors (every time layer), plus one boundary node. Each decomposed component with 2 detectors is an edge; with 1 detector it's a boundary edge.
- Each edge stores: endpoints, observable mask, probability, weight w = log((1-p)/p), and a type label (space / time / space-time diagonal / boundary) from detector coordinates.
- Precompute padded neighbor arrays (n_det, max_deg) with -1, edge ids, and per-node boundary edge id. Cache per (d, basis, noise model, p).
- Tests: node/edge counts vs the DEM; every single fault (each DEM component) is represented.

STEP 2: Cube3D-Local (the fast path): local rules on detection events in 3D
- Input: detection events directly (no raw values, no temporal filter).
- Early exit: shots with zero detection events -> predict no flip.
- Mutual-isolation rules on the 3D graph (any edge type, so measurement errors and hook errors are handled by the same rule):
  - R1: lit node u whose ONLY lit neighbor is v, and v's ONLY lit neighbor is u -> flip edge (u, v) and clear both. If several parallel edges connect u and v, pick the highest-probability one.
  - R2: unlit node c with exactly 2 lit neighbors a, b, where neither a nor b has any other lit node within graph distance 2 except each other -> flip edges (a, c) and (c, b) and clear a, b.
  - R3: lit node u with a boundary edge and no other lit node within graph distance 2 -> flip the boundary edge.
- Parallel schedule: two rule applications conflict if their closed neighborhoods intersect. Greedy-color the conflict graph and process one color at a time, vectorized over shots and nodes. Report the number of colors.
- Prediction = XOR of the observable masks of flipped edges (keep a precomputed per-edge observable bit; no path building).
- Gate: Cube3D-Local ACCEPTS a shot only if the rules clear every lit node AND the total number of lit nodes is <= K (default 6, configurable). Otherwise the WHOLE shot is routed to the deep tier.
- Tests: exhaustive single-fault test (every DEM component's syndrome is cleared and predicts the right observable) for all noise tiers and d = 3, 5, 7; syndrome consistency on accepted shots; a fixture for a measurement-error pair (same ancilla, consecutive rounds) and for a hook error.

STEP 3: UnionFind3D (the deep tier): our own Delfosse-Nickerson Union-Find on the same 3D graph
- weighted cluster growth (weights rounded to small integers, half-edge increments)
- disjoint sets with path compression + union by rank
- clusters freeze at even parity or when they touch the boundary
- peeling decoder on a spanning forest of each cluster
- prediction = XOR of the observable masks of the peeled edges
- Tests: exhaustive single-fault test; syndrome consistency on random shots.
- numba @njit(parallel=True), prange over shots; bit-identical to the reference implementation (tested).

STEP 4: Hierarchical decoder
- Hierarchical(fast=Cube3D-Local, deep=UnionFind3D, K) following our Decoder base class.
- Report: the fraction of shots accepted by the fast path, and the fast path's error rate on accepted shots vs UnionFind3D on those same shots (the gate must not hurt accuracy).
- Sweep K = 2..10 and report the accuracy vs speed trade-off.

STEP 5: Speed
- Profile (cProfile, d = 7, 50k shots) and report the top 10 functions.
- numba for the fast path and UF; optionally C99 + OpenMP via ctypes if a C compiler already exists (don't install toolchains without asking).
- All backends bit-identical to the reference (tested). Report speedup vs thread count.

STEP 6: Keep our comparison figure as a standard output of every benchmark run
- plot_accuracy_runtime(...) is called automatically at the end of every benchmark run. Keep the exact current layout:
  - one panel per distance (3, 5, 7, plus 9 when available)
  - x = batch time per shot (µs), y = logical failures (%)
  - 95% Wilson interval whiskers
  - subtitle with basis, rounds rule, shots, p, noise model and run number
  - footer text unchanged
- One style registry (color + marker per decoder, the same in every figure). Keep the current styles for Cube-Collapse, PyMatching, Correlated PyMatching and PyMatching 2D on flat. Add distinct styles for Cube3D-Local, UnionFind3D and Hierarchical.
- Add CPU model, thread count and backend to the footer.
- Write CSV results/accuracy_runtime_<timestamp>.csv (decoder, d, p, rounds, shots, failures, LER, CI low/high, µs/shot, threads, backend, fast-path fraction, git commit) and PNGs accuracy_runtime_<timestamp>.png + accuracy_runtime_latest.png. Never overwrite old results.
- Companion figure accuracy_runtime_sweep.png: LER vs p (log-log), one line per decoder, one panel per distance, p in 0.001-0.008 circuit-level.
- Pytest smoke test: the plot function runs on a tiny synthetic CSV and creates the PNG.

ACCEPTANCE TARGETS (report honestly whether each is met):
- Hierarchical and UnionFind3D LER DECREASE with d at p = 0.5% circuit-level.
- Hierarchical LER is within the 95% CI of UnionFind3D alone.
- Hierarchical is faster than UnionFind3D alone, with fast-path fraction reported per d and p.
- Stretch goal: Hierarchical within 1.5x of PyMatching's LER at d = 7.

STEP 7: README
Document the evaluation story with the figures: flattening (and why it fails, using the PyMatching-2D-on-flat evidence), then Cube3D-Local + UnionFind3D, then Hierarchical. Include a results table and reproduction commands.

Report with numbers, not adjectives. If anything conflicts with the existing code or Stim's detector layout, stop, explain, and propose a fix.
