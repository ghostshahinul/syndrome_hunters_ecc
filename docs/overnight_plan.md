# Overnight plan (verbatim from the team, 2026-11-21)

We're working through the night on the MidQWest Northrop Grumman decoder challenge. Demos are Sunday. Judging: presentation quality, novelty, parallelizability, code quality, number of implemented algorithms. Save this message to docs/overnight_plan.md, then work through the milestones below IN ORDER.

CURRENT STATE: UnionFind3D is done (commit ed031a9), with LER falling with d (circuit-level p = 0.5%: 1.83 / 1.77 / 1.55 / 1.14% at d = 3/5/7/9; 1.07-1.66x PyMatching). numba 12-thread is 6-10x faster than 1-thread, and 1-thread beats PyMatching at d = 7/9. Suite green: 226 passed, 24 skipped, 1 xfail.

WORKING RULES (apply to every milestone)
- I may be away for hours. Do NOT wait for me except at the decision points marked [ASK]. When a choice is ambiguous, take the safer/simpler option, note it in the progress log, and continue.
- Keep docs/PROGRESS.md updated after every milestone: timestamp, what was done, key numbers, commit hash, what's next, any decisions or problems. This is how I catch up.
- The full test suite must be green before every commit. Commit after each milestone (and mid-milestone at safe points) with clear messages. Never push; I'll push myself.
- Keep the reference and numba implementations bit-identical (tested) for anything you change.
- Use pymatching only as a reference baseline, never inside our decoders.
- Timing hygiene: warm up numba before timing, record CPU model, frequency and power state, threads and backend in every results CSV, and only compare times within one run. Report single-thread and 12-thread times separately. Single-thread vs PyMatching is the fair comparison.
- Never overwrite old results. Commit results CSVs/PNGs (gitignore anything large).
- If something takes more than 1.5x its time-box, stop that item, log why, and move on to the next milestone.
- Report with numbers, not adjectives.

MILESTONE 1: Quick UF accuracy experiments (time-box 45 min)
At p = 0.5% circuit-level, d = 3/5/7/9, same seed, report before/after LER with CIs and the speed impact:
 a. Weight resolution: integer edge length = round(S * w / w_min), for S in {2 (current), 4, 8}.
 b. Growth order: grow only the smallest odd cluster(s) per round (Delfosse-Nickerson rule) vs growing all odd clusters.
Keep whatever improves LER without more than 1.5x slowdown, and make it the default. Log the results either way.

MILESTONE 2: Hierarchical decoder (Step 4; time-box 2.5 h)
- Hierarchical(fast=Cube3D-Local, deep=UnionFind3D) with two gates: whole-shot (route the entire shot to UF unless the fast path clears it with lit nodes <= K) and partial (the fast path fixes isolated pairs, UF decodes the residual).
- Run p in {0.1%, 0.2%, 0.5%}, d = 3/5/7/9, same seed. Report per gate:
  - fast-path acceptance fraction
  - Hierarchical LER vs UF alone on the same shots (95% CIs)
  - the fast path's error rate on accepted shots vs UF on those same shots
  - 1-thread and 12-thread µs/shot for Hierarchical vs UF alone
  - the K = 2..10 sweep
- If the partial gate costs accuracy beyond the CI, say so and default to whole-shot.
- Registry name "hierarchical"; it must flow into the generic tests and figures automatically.

MILESTONE 3: Figures (Step 6; time-box 1.5 h)
- plot_accuracy_runtime(...) runs automatically after every benchmark. Keep the current layout:
  - one panel per distance; x = batch µs/shot, y = logical failure %
  - 95% Wilson whiskers
  - subtitle with basis, rounds rule, shots, p, noise model and run number
  - the same footer, plus CPU, threads and backend
- One style registry: fixed color + marker per decoder across all figures. Keep existing styles; give proper display names and styles to UnionFind3D, Cube3D-Local, Hierarchical and Cube-Collapse.
- Companion figures:
  - accuracy_runtime_sweep.png: LER vs p (log-log), one line per decoder, one panel per distance, circuit-level p in 0.001-0.008
  - thread_scaling.png: µs/shot vs threads (1, 2, 4, 8, 12) for UF and Hierarchical at d = 7 and 9, with PyMatching single-thread as a horizontal reference line
  - fast_path_fraction.png: acceptance fraction vs p, one line per d
  - cube_collapse_story.png: LER vs d for Cube-Collapse, PyMatching 2D on flat and PyMatching 3D at p = 0.5% (this shows why flattening fails)
- Write figures to results/plots/ as both <name>_<timestamp>.png and <name>_latest.png.
- Pytest smoke test: each plot function runs on a tiny synthetic CSV.

MILESTONE 4: README + presentation material (time-box 1.5 h)
README sections:
 1. Overview and quickstart (install, run tests, run benchmark, regenerate figures)
 2. Algorithms implemented: Cube-Collapse, Cube3D-Local, UnionFind3D, Hierarchical. For each: how it works in plain language, complexity, parallelism, and deviations from the papers (Clique_L2 arXiv 2507.11481; Delfosse-Nickerson UF; review arXiv 2307.14989)
 3. The evaluation story with the figures: flattening fails (the PyMatching-2D-on-flat evidence) -> 3D local rules -> our own UF -> hierarchical
 4. Bugs found and fixed (including the UF boundary-peeling bug at d = 3)
 5. Results table (latest numbers with CIs) and honest limitations
 6. Test suite overview (what the metrics tests check)
Also write docs/talk_outline.md: a 10-minute talk outline, slide by slide (about 8 slides), with the figure to use on each slide and 2-3 speaker bullets per slide, including a whiteboard segment explaining a 5-qubit repetition code and UF cluster growth.

MILESTONE 5 (optional, only if time remains; time-box 2 h): pick ONE
 A. C99 + OpenMP backend for UF via ctypes (gcc should be at C:\msys64\ucrt64\bin; if not found, skip), bit-identical to the reference, added to thread_scaling.png.
 B. Our own MWPM as a third tier for the hardest shots: Dijkstra on the 3D graph between lit detectors + networkx.max_weight_matching, used only when the UF cluster count or size exceeds a threshold. Report LER/speed vs UF alone.
[ASK] Before starting Milestone 5, log your recommendation for A or B, then default to B if I haven't answered within 10 minutes of finishing Milestone 4.

FINAL: Update docs/PROGRESS.md with a summary table of all milestones, the final numbers, every commit hash, the list of figures, and anything I need to check or decide when I'm back. Make sure the suite is green and everything is committed (not pushed).
