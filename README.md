# Syndrome Hunters · FaultJoin

**FaultJoin is our handwritten surface-code decoder.** Given the detectors that fired in a memory experiment, it predicts whether the stored logical qubit flipped. It uses whole circuit faults to vote on sparse syndromes and uses evidence from correlated detector clicks to decode denser syndromes. Stim generates the circuits and Monte Carlo shots. PyMatching supplies two independent benchmarks.

At distance 3 and 0.5% physical noise, FaultJoin made **799 errors in 50,000 shots**, with a **4.25 µs median single-shot call**. Standard PyMatching made 860 errors with a 13.1 µs median call. At distance 7, FaultJoin also made fewer errors than standard matching, but took longer. [Benchmark data](results/bench_20261121T205718478691Z.csv)

![Logical error rate versus batch decoding time at distances 3, 5 and 7](figures/accuracy_runtime.png)

## Measured results

Each cell gives **logical failure rate / batch time per shot** for rotated Z memory at 0.5% physical noise, with one round of syndrome extraction per unit of code distance.

| Distance | FaultJoin | Standard PyMatching | Correlated PyMatching | Handwritten union–find |
|---:|---:|---:|---:|---:|
| 3 | **1.598% / 0.636 µs** | 1.720% / 0.729 µs | 1.646% / 1.717 µs | 1.834% / 1.156 µs |
| 5 | **1.316% / 6.944 µs** | 1.384% / 5.141 µs | 1.042% / 10.549 µs | 1.556% / 11.533 µs |
| 7 | **0.892% / 23.049 µs** | 1.030% / 13.282 µs | 0.600% / 35.519 µs | 1.304% / 43.801 µs |

At distance 7, FaultJoin correctly decoded 262 shots that standard matching missed. Standard matching correctly decoded 193 that FaultJoin missed. Correlated PyMatching achieved the lowest error rate at distances 5 and 7 at this noise level. At distance 7 its median single call took 47.0 µs, compared with 31.9 µs for FaultJoin and 24.35 µs for standard PyMatching. [Runtime and p95 latency figure](figures/runtime_scaling.png)

The Z sweep used 50,000 shots at each of 15 combinations of distance and physical noise. For FaultJoin, distance 7 had fewer failures than distance 5 at 0.7% noise and more at 1.0%. This brackets their finite-distance crossing. An asymptotic threshold would require larger distances and a finer sweep. A separate X-memory run at 0.5% noise used 20,000 shots per distance. At distance 7 it gave 195 FaultJoin failures, 275 standard matching failures and 137 correlated matching failures. [Noise-sweep figure with 95% intervals](figures/logical_error_rates.png) · [X-memory results](results/bench_20261121T205500869314Z.csv)

## How FaultJoin works

1. **Read whole faults.** Each Stim detector error model instruction gives a fault probability, the detectors it toggles and the logical bits it flips. A separator suggests graph edges for the same physical event. FaultJoin retains both the complete event and those edges. [Stim detector error model specification](https://github.com/quantumlib/Stim/blob/main/doc/file_format_dem_detector_error_model.md)
2. **Vote on a sparse syndrome.** Let the odds of fault `i` be `rᵢ = pᵢ / (1 − pᵢ)`. A matching single fault contributes `rᵢ` to its logical class. A pair whose detector sets XOR to the observed syndrome contributes `rᵢ rⱼ` to its combined class. The class with the greatest total wins. This is the exact vote restricted to zero, one or two effective faults. Small models tabulate these votes once. On larger models the decoder chooses a fired detector with few incident faults. Exactly one fault in a matching pair contains that detector, so an indexed lookup finds the other fault and counts each pair once. [Vote implementation](src/decoders/join.py)
3. **Condition edge costs when the vote has no explanation.** For each projected edge and a disjoint partner from the same physical fault, compute the probability that the edge toggled given that the partner's detectors fired. This uses the independent DEM probabilities and the identity `E[(−1)^parity] = ∏(1 − 2pᵢ)`. During decoding, the strongest visible partner discount sets the edge's growth cost. Choosing one discount and quantizing the costs are approximations. Detectors outside logical-carrying components can still provide evidence. [Conditional probability calculation](src/decoders/condition.py)
4. **Complete the correction globally.** Odd clusters grow simultaneously and merge when an edge fills. XOR labels on the union–find forest give the predicted logical flip when every cluster is even or reaches a boundary. The plain union–find comparator runs the same handwritten kernel with fixed edge costs. Both kernels run on the CPU with Numba. [Cluster implementation](src/decoders/union_find.py) · [Delfosse and Nickerson's union–find decoder](https://arxiv.org/abs/1709.06218)

For a small example, suppose a single fault explains a click at detector 0 with logical flip odds of 0.00402. Two different fault pairs can explain that same click without a logical flip, each with odds of 0.00277. Their combined weight of 0.00554 wins the vote. The tests verify this decision by enumerating the underlying Bernoulli events, including the cancelled detector in each pair. [Exact-enumeration tests](tests/test_fault_join.py)

The project contribution is the sparse logical vote combined with partner-conditioned cluster growth. Correlation-aware decoding has established precedents, including belief-matching and belief-find. [Higgott et al.](https://arxiv.org/abs/2203.04948) · [PyMatching and its correlated decoder](https://github.com/oscarhiggott/PyMatching)

## Run it

The reported runs used Python 3.12.10. On Windows, from the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest -q
```

On Linux or macOS, create the environment with `python3.12 -m venv .venv` and activate it with `source .venv/bin/activate`. The same installation and test commands apply.

To reproduce the published grid and figures:

```sh
python -m src.evaluate --basis z --distances 3 5 7 --ps 0.001 0.003 0.005 0.007 0.01 --shots 50000 --seed 20261121
python -m src.plots
```

To reproduce the X-memory check:

```sh
python -m src.evaluate --basis x --distances 3 5 7 --ps 0.005 --shots 20000 --seed 20261122
```

`make test`, `make data`, `make bench SHOTS=50000` and `make plots` are also available. `python -m src.data` writes bit-packed sample archives. Pass the same sampling options to `src.evaluate` with `--data-dir data` to score those saved shots. Each benchmark writes a new timestamped CSV and a JSON manifest. To plot a particular run, pass its CSV path to `python -m src.plots`. Use `--out-dir` to keep a separate figure set.

## Judge interface

Construct one decoder per circuit or noise model and reuse it across shots.

```python
from src.circuits import surface_code_circuit, detector_error_model
from src.decoders.experimental import FaultJoinDecoder

circuit = surface_code_circuit(distance=5, p=0.005, rounds=5, basis="z")
decoder = FaultJoinDecoder(detector_error_model(circuit))
dets, truth = circuit.compile_detector_sampler(seed=12).sample(1000, separate_observables=True)
predicted = decoder.decode_batch(dets)   # bool array (shots, logical observables)
error_rate = (predicted != truth).any(axis=1).mean()
```

For blind evaluation, supply a decomposed Stim `.dem` file and detector-only shots in Stim's little-endian `b8` format. The command writes one packed row of predicted logical flips per shot. The decoder also accepts a single unpacked syndrome through `decode(syndrome)`. [Decoder entry point](src/decoders/experimental.py)

```sh
python -m src.decode --dem code.dem --dets heldout.b8 --out predictions.b8
```

## Measurement details

Stim applies the same physical probability to post-Clifford depolarization, post-reset flips, pre-measurement flips and per-round data depolarization. A shot fails if any predicted logical observable differs from Stim's sampled truth. All four decoders receive identical detector shots at each point. The CSV records a sample checksum, raw failure counts, 95% Wilson intervals, paired disagreement counts and an exact two-sided binomial test on disagreements against standard matching.

Timing used an Intel Core i7-10750H on Windows 11. Batch time is the median of three warmed serial passes with up to 4,096 shots per call. Single-call p50 and p95 use 256 shots. Setup and warm-up have separate CSV columns. Memory is the Python-traced peak during a separate warm batch and can omit native allocations. Stim seeds should be compared on the same version, sampling call shape and CPU instruction set. [Z-run environment and settings](results/bench_20261121T205718478691Z.json) · [X-run environment and settings](results/bench_20261121T205500869314Z.json)

Two earlier runs used independent seeds and measured single calls through the original batch adapter. Their results are archived for comparison. [Earlier Z sweep](results/bench_20261121T205035315898Z.csv) · [Earlier X check](results/bench_20261121T204823713052Z.csv)
