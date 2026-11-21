Implement the "Cube-Collapse" decoder in this repo. This message is the full spec: first save a copy of it to docs/cube_collapse_spec.md, then read CLAUDE.md and the existing code (Decoder base class, Stim helpers, metrics, tests). Reuse what exists and adapt the paths below to our layout. Give me a 5-line summary of what exists and your plan, then start Phase 1. Context: MidQWest Northrop Grumman decoder challenge, judged on presentation, novelty, parallelizability, code quality and number of algorithms. We're mid-hackathon, so working beats perfect.

HARD REQUIREMENT: the decoder must resolve EVERY syndrome. No "unresolved" flags, no hand-off to an external/global decoder. For every shot, the correction's syndrome must equal the flattened syndrome.

## Setup
Stim rotated surface-code memory experiment ("surface_code:rotated_memory_{basis}"): d rounds plus Stim's final data-readout detector layer. Only use the checks relevant to the basis (for memory_z: the sites that have detectors at t=0).

## Stage 0: syndrome cube
- Map detectors -> (round t, site) via circuit.get_detector_coordinates() (x, y, t).
- Convert detection events to RAW ancilla values with a cumulative XOR over time. Shape (shots, T, n_sites), uint8. The final data-readout layer is the last slice (config include_final_layer=True).

## Stage 1: temporal filter (vectorized over shots and sites)
1. Remove single-round blips with a 3-point majority over time on the raw series. Pad the left edge with 0 (known initial state); pad the right edge by replicating the last value.
2. flat = filtered value at the LAST round. Do NOT use a whole-window majority vote; it misses data errors late in the window. Keep whole-window majority only as a baseline for comparison.
3. Diagnostics per ancilla: hold = number of final rounds holding the final value; transitions = number of changes in the filtered series. Confidence class: sure (transitions <= 1 and hold >= late_k, default 2), late (hold < late_k), messy (transitions >= 2). Confidence only affects processing order in Stage 4 and appears in diagnostics; it never causes a node to be skipped.

## Stage 2: 2D check graph
- From circuit.detector_error_model(decompose_errors=True), keep space-like error components among the selected sites (both detectors in the same bulk time layer) plus boundary components (one detector). Key nodes by (x, y).
- Each edge stores: endpoints (u, v, or BOUNDARY), observable mask, probability, and weight w = log((1-p)/p).
- Test: this graph must equal the one from a rounds=1 code-capacity circuit with the same d and basis.
- Precompute: padded neighbor arrays (n_sites, max_deg) with -1; edge ids; the boundary edge per node (-1 if none); all-pairs shortest paths (Dijkstra on w; deterministic tie-breaks) giving distance and path edge list for every node pair and every node->boundary. n_sites is small, so precomputing everything is fine. Assert every node has a finite path to the boundary.

## Stage 3: Clique local pass
Graph version of Clique_L1 + L2 (Jia, Veerapaneni & Ravi 2025, arXiv 2507.11481), applied to flat in this exact order:
- S1 (L1): a lit node with exactly 1 lit neighbor -> flip the edge between them, clear both. (The paper also decodes 3 lit neighbors locally; we leave that case for Stage 4. Document it.)
- S2 (L2): an unlit center with exactly 2 lit neighbors -> flip both center edges, clear both neighbors. An unlit degree-4 center with all 4 neighbors lit -> flip all 4.
- S3 (boundary): a lit node with 0 lit neighbors and a boundary edge -> flip that boundary edge.
Parallel schedule: two cliques conflict if their closed neighborhoods (center + neighbors) intersect. Greedy-color the conflict graph; within each stage, process colors one after another and all cliques of one color at once (vectorized over shots and cliques). Report the number of colors; don't hardcode 4.

## Stage 4: growing-radius greedy completion (always resolves)
For every shot with lit nodes remaining after Stage 3:
- for radius r = 1, 2, 3, ... (up to the max node-to-boundary distance):
  - candidates = all pairs of still-lit nodes with path distance <= r, plus (lit node, BOUNDARY) with boundary distance <= r
  - sort by (distance, lower-confidence nodes last, node ids)
  - greedily accept candidates whose endpoints are both still lit: flip the precomputed shortest-path edges, clear the endpoints
  - stop when no lit nodes remain
This must terminate with zero lit nodes for every input (guaranteed by boundary reachability). Shots are independent, so this stage runs in parallel over shots.

## Output
Predicted observable flips (XOR of the observable masks of all flipped edges), the full correction, and per-stage stats (lit nodes cleared by S1/S2/S3/S4, fraction of shots needing S4, max radius used). Match our Decoder base class interface.

## Implementation rules
- Python >= 3.10, numpy, stim. Use pymatching ONLY for reference baselines and validation, never inside the pipeline.
- Write a pure-NumPy/Python reference first; it is the source of truth. Use type hints, docstrings, and a frozen dataclass config (late_k, include_final_layer, backend).
- Phase 3 speed: profile first, then accelerate the hot loops (filter, Clique color passes, Stage 4 greedy) with numba (@njit(parallel=True), prange over shots). Optionally add a C99 backend in csrc/ with OpenMP over shots, loaded via ctypes and built by a small script. Check for an existing C compiler first; if there isn't one, skip C and tell me; don't install system toolchains without asking.
- All backends must give bit-identical outputs to the reference (tested).
- Make small commits after each phase with clear messages. Don't modify other decoders except to share utilities.

## Required tests (pytest)
- Filter: hand-built series for flat, early/mid step, single blip, late step, double blip, messy, and a final-round blip with and without the final layer.
- Graph: node/edge counts and observable masks match the code-capacity DEM for d = 3, 5, 7; finite boundary distance for every node.
- Clique fixtures: single bulk error, boundary error, corner error, length-2 chains through an unlit center (horizontal/vertical/diagonal analogues), two separated errors, errors resolved in different stages, overlapping cliques.
- Stage 4 fixtures: chains of length 3 and 4, a lone lit node far from others (goes to the boundary), dense clusters.
- Property (assert, every shot, every noise tier): syndrome(correction) == flat.
- Exhaustive: every single error (each 2D edge) for d = 3, 5, 7 predicts the correct observable (assert). All weight-2 errors at d = 5: report correct/wrong counts and compare with PyMatching 2D on the same syndromes; investigate any failures.
- Backends: identical outputs on 10k random shots.

## Benchmark script (scripts/bench_cube_collapse.py)
- Noise tiers: code-capacity; phenomenological (data noise + measurement flips, d rounds); circuit-level (all four Stim noise parameters). d in {3, 5, 7, 9}. p sweeps: code-capacity 0.01-0.12, phenomenological 0.005-0.04, circuit-level 0.001-0.012.
- Metrics:
  - logical error rate with Wilson 95% CIs (enough shots for ~100 failures where feasible)
  - per-stage share of lit nodes cleared and fraction of shots needing Stage 4
  - filter accuracy: flat computed without the final layer vs. the data-derived parity from the final layer
  - throughput (ns/shot) and latency (mean / p99)
  - number of colors
- Baselines on the same samples (same seed): PyMatching full 3D; PyMatching 2D on flat (isolates our spatial stages vs. matching on the same filtered input); whole-window majority vote -> our stages (shows whether the filter change matters).
- Write CSVs to results/ and plots to results/plots/: LER vs p per d, stage share vs p, speedup vs thread count per backend.

## Phases (stop and report numbers after each)
1. Cube + filter + graph + Clique + Stage 4 (reference implementation), tests green on code-capacity noise.
2. Phenomenological and circuit-level runs + benchmark script + plots.
3. Profiling + numba backend (+ C if a compiler is available) + equivalence tests + speedup plot.
4. README section: pipeline overview, stage table, deviations from the Clique_L2 paper (graph generalization, 3-lit-neighbor case moved to Stage 4, end-state filter instead of whole-window majority, growing-radius completion instead of offloading), results table, and reproduction steps.

Out of scope for now: sliding across multiple windows (but keep the API window-ready: decode accepts any T).

Report with numbers, not adjectives. If anything here doesn't match Stim's actual detector layout, stop, explain, and propose a fix rather than forcing it.
