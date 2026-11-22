# Faster correlated PyMatching for rotated surface-code memory

This branch compares the original PyMatching wheel with packed I/O, a separately compiled upstream control, and an experimental C++ prediction path for correlated Sparse Blossom. All correlated paths use the same DEM priors and two-pass rule. Each worker keeps a matching graph for the lifetime of the circuit.

## Benchmark result

**The experimental 12-worker decoder had the highest aggregate throughput on the measured machine.** It decoded the 1.25 million-shot grid in **21.81 s**, or **57,313 shots/s**, versus **101.08 s** and **12,367 shots/s** for the original correlated wheel. Every correlated strategy returned the original logical prediction on every shot. The 12-worker strategy had the fastest batch time at 20 of 25 points. Four workers led at four points and six workers led at one. Measurements used a Windows 11 laptop with an Intel Core i7-10750H, six physical cores and twelve logical processors.

| Speed rank | Implementation | Grid failures / 1,250,000 | Grid µs/shot | Speedup vs original | d=11 µs/shot¹ | d=11 peak MiB¹ |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Experimental, 12 workers | 28,760 | 17.45 | **4.63×** | **40.92** | 266.2 |
| 2 | Experimental, 6 workers | 28,760 | 22.31 | 3.62× | 52.17 | 211.6 |
| 3 | Experimental, 4 workers | 28,760 | 28.04 | 2.88× | 106.50 | 192.9 |
| 4 | Unmodified native core, 4 workers | 28,760 | 29.60 | 2.73× | 71.00 | 193.0 |
| 5 | Ordinary PyMatching | 37,231 | 29.67 | 2.73× | 72.92 | 208.1 |
| 6 | Experimental, 1 worker | 28,760 | 65.04 | 1.24× | 212.90 | 165.3 |
| 7 | Unmodified native core, 1 worker | 28,760 | 68.43 | 1.18× | 171.26 | 165.1 |
| 8 | Packed correlated wheel | 28,760 | 78.43 | 1.03× | 184.79 | **157.3** |
| 9 | Original correlated wheel | 28,760 | 80.86 | 1.00× | 193.81 | 217.3 |

¹ Distance 11 at 0.5% physical noise. Speedups and grid times use summed median pass times over all 25 points. At equal worker counts, the experimental path used **5.0% less grid time at one worker** and **5.3% less at four workers** than the same-compiler native control. At the single distance-11 point in the table, the four-worker control and the single-worker control each beat the corresponding fast core. [Rankings, including accuracy and scalability ranks](20261122T133916020788Z/report/ranking.csv) · [All 225 operating-point measurements and confidence intervals](20261122T133916020788Z/report/all_points.csv)

**Throughput, latency and memory have different winners.** At distance 11 and 0.5% noise, every correlated strategy recorded **76 failures / 50,000**, or **0.152%** with a 95% Wilson interval of **0.121%–0.190%**. Ordinary MWPM recorded **216**. The 12-worker batch decoder was 4.74× faster here, but its individual-call p50/p95 were 328/588 µs, compared with 211/321 µs for the original wheel. Its graph setup took 0.84 s versus 0.11 s and its process peaked at 266 versus 217 MiB. Packing reduced the detector input from 62.94 to 7.87 MiB at this point. The packed stock wheel used the least peak memory in this comparison.

**Timing replication.** The initial one-worker upstream control at distance 7 and 0.5% noise took 49.70, 63.01 and 65.43 s for 50,000 shots. After observing that anomaly, we used a full-point replication rule for measurements over five times slower than the same strategy at a higher distance with fixed noise or at higher noise with fixed distance. This was the only triggered point. All nine strategies were remeasured on the same sample hash. The repeated control median was 1.77 s. The ranking uses the full repeated point, including its new reference time. The raw files retain both measurements. All 15 points shared with the archived `shahi_2` data matched their archived sample hashes. [Initial raw timings](20261122T133916020788Z/raw.csv) · [Additional workers and matched control](20261122T133916020788Z/scaling_20261122T141230641556Z) · [Whole-point replication](20261122T133916020788Z/replication/d7_p0.005)

| Measured quantity | Figure |
|---|---|
| Logical error vs physical noise, with 95% intervals | [Logical error rates](20261122T133916020788Z/report/A_logical_error_rate.png) |
| Batch time vs distance | [Microseconds per shot](20261122T133916020788Z/report/B_time.png) |
| Throughput vs distance | [Shots per second](20261122T133916020788Z/report/C_throughput.png) |
| Speedup vs distance | [Relative to original correlated PyMatching](20261122T133916020788Z/report/D_speedup.png) |
| Memory vs distance | [Peak process memory](20261122T133916020788Z/report/E_memory.png) |

[Run configuration, seed, dependency versions and native build hashes](20261122T133916020788Z/config.json) · [Separate Stim sampling and DEM timings](20261122T133916020788Z/sampling.csv)

## What the original decoder does

The baseline in `shahi_3` inherits the wrapper from `shahi_2`. It sets the class attribute `correlations=True`, calls `Matching.from_detector_error_model(dem, enable_correlations=True)` once, and calls `matching.decode_batch(dets, enable_correlations=True).astype(bool)` for each batch. The evaluator loops over 4,096-shot batches. A separate Python loop calls `decode` only for single-shot latency samples. Stim originally returned dense Boolean detectors and observables. The original correlated graph, both original decoder classes, and all historical benchmark records are retained. [Original decoder](../../src/decoders/baseline.py) · [Original evaluator](../../src/evaluate.py)

The installed wheel is PyMatching **2.4.0**. Its public batch signature exposes `enable_correlations`, `bit_packed_shots`, and `bit_packed_predictions`. Both correlation flags must be enabled, one when loading the graph and one at decode time. The pinned pybind batch implementation runs a C++ loop, constructs a weight output even when Python discards it, and retains the Python GIL during that call. [Pinned Python wrapper](https://github.com/oscarhiggott/PyMatching/blob/6f63b2b9474ba0fa7e511fe52bffdce858a06984/src/pymatching/matching.py) · [Pinned C++ binding](https://github.com/oscarhiggott/PyMatching/blob/6f63b2b9474ba0fa7e511fe52bffdce858a06984/src/pymatching/sparse_blossom/driver/user_graph.pybind.cc)

## Profile before optimization

Each entry is the median milliseconds for 4,096 shots, three repetitions, physical noise 0.5% and rounds equal to distance. Sampling includes sampler compilation. Graph construction measures the original Python constructor. Warm-up includes lazy creation of the native search/matching state and 256 first decodes. The preprocessing column times the wrapper's shape/dtype check. The PyMatching call includes its binding's conversions and detector scan. The last column includes Boolean output conversion and scoring.

| Distance | Stim sampling | DEM | Graph constructor | Warm-up | Preprocessing | Batch decode | Prediction and scoring |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0.342 | 0.881 | 1.230 | 0.636 | 0.003 | 4.975 | 0.035 |
| 7 | 6.076 | 11.202 | 21.833 | 15.500 | 0.008 | 137.470 | 0.118 |
| 11 | 23.143 | 28.176 | 71.267 | 77.244 | 0.008 | 666.453 | 0.054 |

The native decode call dominated this profile. Graph construction and lazy initialization also matter when a circuit is used for few shots. Replacing the Python batch loop with another dispatcher could only address a small fraction of the measured time. [Raw stage timings](profile.csv)

## What changed inside Sparse Blossom

1. **Finding correlated edges.** Upstream compiles DEM joint probabilities into per-edge lists of implied weight updates. The first pass performs matching and recovers the edges on the selected correction, including cancellation of edges traversed twice. It then visits the lists belonging to those edges. The experimental code uses this same candidate selection, edge order, and path recovery. [Graph construction](https://github.com/oscarhiggott/PyMatching/blob/6f63b2b9474ba0fa7e511fe52bffdce858a06984/src/pymatching/sparse_blossom/driver/user_graph.cc)
2. **Updating only affected weights.** Upstream already changes targeted edge weights and records individual changed locations for reverse-order restoration. The experiment preserves this behavior on the matching graph. Repeated changes to the same location must be undone in reverse order. [Matching graph](https://github.com/oscarhiggott/PyMatching/blob/6f63b2b9474ba0fa7e511fe52bffdce858a06984/src/pymatching/sparse_blossom/flooder/graph.h)
3. **Removing a redundant update.** The original mutates both the matching graph and the separate shortest-path search graph. For up to 64 observables, the second pass accumulates the logical XOR mask directly from the matching flooder. The search graph is read during first-pass edge recovery, before the updates, and its weights are next needed on the following shot, after restoration. Omitting its intervening update/undo leaves every weight read by either pass unchanged. For more than 64 observables, upstream reconstructs paths in the second pass and those updated weights matter. The experimental backend calls the full original correlated routine in that case. [Upstream two-pass driver](https://github.com/oscarhiggott/PyMatching/blob/6f63b2b9474ba0fa7e511fe52bffdce858a06984/src/pymatching/sparse_blossom/driver/mwpm_decoding.cc) · [Experimental kernel](../../native/fast_cp/kernel.cc)
4. **Reusing work.** Each native workspace reuses the first-pass edge vector and detector-index scratch storage between shots. The fused packed kernel enumerates set bits in ascending order, writes logical predictions directly, and omits the unused solution-weight output array. The fast mode can cache the actual correlated answer to the empty syndrome on its first occurrence. It retains the full second matching solve for other shots because altered weights can move the optimum beyond the updated edges. The solver preserves upstream tie ordering and uses the full, unpruned matching graph.
5. **Parallel throughput.** The standalone C ABI releases the Python GIL through `ctypes.CDLL`. A persistent thread pool divides each input batch into contiguous disjoint ranges and returns predictions in original order. Each concurrent worker owns its mutable matching workspace. Inputs, output buffers and Python runtime are shared. Small batches use one workspace directly. Extra workspaces increase setup time and native memory. The separately compiled upstream control uses the same compiler and packed transport and invokes the original correlated solver on every shot. [Python API and routing](../../src/decoders/fast_correlated.py)

The stage-instrumented C++ control reproduced the wheel's predictions on 8,192 shots per distance. At distance 11, first-pass matching and path recovery took 48.9% of measured native time, the second solve took 35.1%, and matching-graph and search-graph updates took 7.3% each. Restoration and counter collection took 1.4%. The two graphs each recorded about 719 weight-location writes per shot. At distance 7, the search-graph updates used 6.9% and averaged 168 writes per shot. These measurements identify the specific work the fast path removes and the two solves that still dominate. The profiler includes per-stage clock overhead and runs separately from the headline timings. [Native stage measurements](20261122T133916020788Z/native_stages.csv)

In a separate component comparison, input packing, output packing, empty-syndrome caching and the core change were varied independently. With caching disabled, switching from the control to the fast native core reduced median batch time from 32.47 to 29.21 µs/shot at distance 7 and from 150.94 to 138.61 at distance 11. Neither sample of 8,192 shots contained an empty syndrome. At distance 3, 3,470 syndromes were empty, and enabling the cache on the fast core reduced time from 1.26 to 1.13 µs/shot. Every configuration reproduced the original predictions. [All component timing repetitions](20261122T133916020788Z/ablation.csv)

## Reproduce

Use a Python 3.12 virtual environment and install the project with `python -m pip install -e ".[dev,fast-cp-bench]"`. The wheel's optimized packed path needs no source build. The native backends require Git and a C++20 compiler. The recorded Windows build used portable LLVM-MinGW 20241217, Clang 19.1.6. Unpack the compiler under `external/` or give the build script another compiler path. It verifies both upstream Git revisions and writes libraries under `external/fast_cp_build/`. The Python installation and upstream source checkouts are separate from the build output. The run manifest records source, kernel and binary hashes. [Compiler release](https://github.com/mstorsjo/llvm-mingw/releases/tag/20241217) · [Build script](../../native/fast_cp/build.py)

```powershell
python native/fast_cp/build.py --cxx external/llvm-mingw-20241217-ucrt-x86_64/bin/clang++.exe
python native/fast_cp/build.py --cxx external/llvm-mingw-20241217-ucrt-x86_64/bin/clang++.exe --detail
python -m pytest -q tests/test_fast_correlated.py
$env:OPENBLAS_NUM_THREADS = '1'
python -m src.fast_cp.bench
```

The benchmark prints its new run directory. Substitute that path for `RUN` in the following commands. Reports require an unused output directory. The profile commands also require a new output filename.

```sh
python -m src.fast_cp.scaling RUN --workers 6 12
python -m src.fast_cp.recheck RUN
python -m src.fast_cp.report RUN
python -m src.fast_cp.ablation --out RUN/ablation.csv
python -m src.fast_cp.profile --out RUN/reference_stages.csv
python -m src.fast_cp.detail --out RUN/native_stages.csv
```

For blind inference from Stim b8 data, use `python -m src.fast_cp.decode --dem circuit.dem --dets dets.b8 --out predictions.b8 --backend native --workers 4`. Choose `--backend packed` for the stock wheel or `--backend native-control` for the separately compiled upstream kernel. `--no-cache-zero` disables empty-syndrome caching in the native backends. In Python, `PackedCorrelated.predict` configures input and output packing independently. Both new classes also implement the repository's dense Boolean `decode_batch` contract.

The full repository test run passed 75 tests. Eight inherited optional Ising tests skipped because the environment lacked PyTorch or their expected checkpoint. The correlated tests covered independent packing flags, deliberately correlation-sensitive predictions, repeated calls, multiple workers, b8 streaming, and the fallback above 64 observables.

## Benchmark protocol

The grid uses Stim rotated Z memory, rounds equal to distance, the same per-channel noise for Clifford, reset, measurement and before-round data errors, distances 3/5/7/9/11, and physical noise 0.1%, 0.3%, 0.5%, 0.7% and 1.0%. It retains the repository seed 20261121 and 50,000 shots per point. Each strategy decodes the same 1.25 million experiments. The benchmark independently regenerates dense and little-endian packed shots, compares every detector and observable bit, and stores the canonical repository checksum.

Timing excludes simulation, setup, warm-up and scoring. Each grid point runs each strategy in a fresh process, rotates the order, warms on a batch and takes the median of three timed passes. Single-shot p50/p95 include public-call dispatch on 256 evenly spaced shots. The packed APIs use one packed row for this latency measurement. Native multi-worker strategies take the serial path for a one-shot call. The raw CSV keeps all pass times. Per-point summaries contain total decode seconds, microseconds/shot, throughput, failures, 95% Wilson binomial intervals and exact disagreement counts. Predictions are compared row by row, including every logical bit. Scaling measurements repeat the same inputs and check the original prediction hash.

Peak resident memory includes the process runtime, source-format input, native workspaces and the output collection buffer. It is the OS process high-water mark before single-shot latency tests and scoring. Summaries also report input bytes, post-warm-up resident memory, growth over the pre-input high-water mark, and setup time. Ranking sums the median decode seconds across the equal-shot grid. Failure-rate ranks describe that grid's mix of noise and distances. The scalability rank uses measured batch time at distance 11 and 0.5% noise.
