# Syndrome Hunters · Free-energy matching

Free-energy matching predicts the logical flip of a rotated surface-code memory experiment from its detector clicks. It starts from correlated PyMatching. On the few shots where the two logical outcomes are nearly tied, it re-decides by comparing the total probability of each outcome under the full fault model, not just the single cheapest correction. Everything it needs is built from the supplied Stim detector error model (DEM), and the same settings are used at every distance, basis, round count and noise level.

## Results at 0.5% physical noise

Each cell gives the **logical failure rate / batch time in microseconds per shot**. All decoders saw the identical rotated Z-memory shots: 100,000 at each distance, with as many measurement rounds as the distance. "Correlated, regrouped" is stage 1 of free-energy matching on its own (see below). It is included to show what the re-decision step adds.

| Distance | Free-energy matching | PyMatching | Correlated PyMatching | Correlated, regrouped | Weighted union–find |
|---:|---:|---:|---:|---:|---:|
| 3 | **1.448%** / 56.2 | 1.721% / 2.3 | 1.691% / 6.1 | 1.527% / 6.3 | 1.807% / 4.9 |
| 5 | **0.864%** / 198.5 | 1.399% / 16.1 | 1.050% / 46.3 | 0.965% / 47.3 | 1.563% / 49.4 |
| 7 | **0.461%** / 559.0 | 1.003% / 54.7 | 0.595% / 148.6 | 0.505% / 157.0 | 1.221% / 174.3 |
| 9 | **0.248%** / 1225.2 | 0.700% / 133.3 | 0.320% / 356.4 | 0.281% / 385.5 | 0.883% / 444.8 |

Against correlated PyMatching, free-energy matching had **14.4%, 17.7%, 22.5% and 22.5%** fewer failures at distances 3, 5, 7 and 9. On the shots where the two decoders disagreed, two-sided exact paired tests gave p = 8.1 × 10⁻²², 2.4 × 10⁻¹⁴, 3.9 × 10⁻¹⁰ and 2.6 × 10⁻⁶. The relative gain grows with distance rather than shrinking. The price is time: batch decoding took 3.4× as long as correlated matching at distance 9, 3.8× at distance 7 and 4.3× at distance 5. [Confirmatory CSV and run manifest](results/confirm)

![Logical failure rate against batch time at distances 3, 5, 7 and 9](figures/accuracy_runtime.png)

These timings were measured on a laptop (Intel i7-10750H, one thread) that was throttling during the run. Correlated PyMatching took about 5× longer per shot than in this repository's previous run on the same shots. Compare decoders within one table, not across runs.

## Across distances, noise levels and bases

Three held-out sweeps cover **1.29 million memory experiments at 25 operating points**. They use the same seeds as the previous FaultJoin benchmarks. All tuning used a separate seed (555). Every decoder saw identical samples at each point.

| Held-out sweep | Points | Fewest failures, outright | Tied for fewest | Fewer failures than correlated PyMatching |
|---|---:|---:|---:|---:|
| Z memory, 0.5% noise, 100,000 shots per point | 4 | 4 | 0 | 4 of 4 |
| X memory, five rounds, 0.5% noise, 50,000 per point | 5 | 5 | 0 | 5 of 5 |
| Z memory, 0.1%, 0.3%, 0.7%, 1.0% noise, 40,000 per point | 16 | 12 | 3 | 13 of 14 points with any failures (one 1–1 tie) |

![Free-energy matching failure-rate differences and batch-time ratios against the best observed baseline across the Z-memory noise grid](figures/baseline_comparison.png)

- **Noise grid.** At 0.7% noise, failures fell by 16.1%, 20.6% and 22.9% relative to correlated PyMatching at distances 5, 7 and 9 (all p < 10⁻¹²). At 1% noise the gain is smaller: 14.6%, 12.3% and 9.6%. The three ties are points where every decoder made zero or one error. The one point where another decoder did better is distance 9 at 0.3% noise: the regrouped comparator made 3 errors to our 5 (p = 0.5), while correlated PyMatching made 7. [Noise sweep](results/noise) · [Failure-rate figure](figures/logical_error_rates.png)
- **X memory.** Failures fell by 17.4%, 17.3%, 17.9%, 23.5% and 13.2% at distances 4, 5, 6, 7 and 9. The distance-9 difference, 92 against 106, is not significant on its own (p = 0.10). [X-memory results](results/x_memory)
- **Cost across the grid.** At distances 5–9 and noise of 0.3% or more, batch time was 4.1–7.4× correlated PyMatching; the high end is at 1% noise, where about a third of shots need the re-decision step. At 0.1% noise it was 4–11×, and at distance 3 it was 7–17×, because matching itself costs only a few microseconds there. The noise-grid figure compares time against plain PyMatching, the fastest baseline. [Batch and single-shot timing](figures/runtime_scaling.png)
- **Compared with FaultJoin.** The confirmatory samples match those of the previous FaultJoin run (identical SHA-256 per point). FaultJoin made 1,434, 932, 507 and 318 errors at distances 3, 5, 7 and 9. Free-energy matching made 1,448, 864, 461 and 248. At distance 3, FaultJoin's exhaustive table is exactly optimal and stays slightly ahead. Free-energy matching gets within 1% of it with no exponential table. At distance 9, where FaultJoin tied correlated matching, free-energy matching has 22% fewer failures. FaultJoin's gain at larger distances came almost entirely from the regrouped, reweighted model it passed to correlated matching: at distance 7 that model alone matches FaultJoin. That model is now stage 1 here, listed separately as "Correlated, regrouped". [FaultJoin source](https://github.com/ghostshahinul/syndrome_hunters_ecc/tree/8faaa37c29926dbc2320dcef0cc473baf6b236b9)

## How the decoder works

The optimal decoder picks the logical class with the larger total probability, Z₀ or Z₁, summed over every set of faults that explains the syndrome. Matching instead reports the class of the single cheapest correction. The two rules disagree only when the cheapest corrections of the two classes are close in weight, so free-energy matching spends extra work only there.

1. **Correlated matching on a regrouped model.** Stim's graphlike decomposition sometimes splits one fault into several edges inside the same detector sector. When their XOR is a single edge, the parts are merged, so only genuine cross-sector correlations, such as Y errors, remain hyperedges. Fault odds are also multiplied by 1/2 per component when building matching weights; Stim always samples the original circuit. Every shot is decoded with correlated PyMatching on this model. [Model transforms](src/decoders/model.py)
2. **Measure the logical gap.** The boundary of the sector that carries the observable is replaced by two nodes. A collects the boundary edges that flip the observable, B the rest. If the observable also sits on bulk edges, a set of detectors is first folded into it so that only boundary edges carry it. The fold is found by two-colouring the bulk graph. Setting the clicks of A and B then forces matching to return the best correction in a chosen logical class. Plain matching factorizes over sectors, so this runs on that one sector. A shot is "hard" when the opposite class costs less than 4 nats more than the best correction, or when plain and correlated matching disagree. Hard shots were 2.6–8% of shots at 0.5% noise, and they contain almost all failures.
3. **Free-energy vote on hard shots.** For each class, correlated matching returns its best correction in that class. Each correction is scored by its free energy, F = E − H:
   - **E** is the negative log-likelihood ratio of the most probable fault set behind the correction. Hyperedge faults can explain several of its edges at once for less.
   - **H** counts the nearby corrections. The triangles and squares of the matching graph are local stabilizer moves. A first-order cluster expansion around a correction S gives log Z(S) ≈ −E(S) + Σₘ [log(1 + e^(−ΔEₘ)) − log(1 + e^(−E(m)))], summed over short cycles m touching S. The subtracted term removes vacuum fluctuations shared by every correction.
   - Each correction is first moved downhill by single cycle moves, so every ΔEₘ is non-negative. A diagonal space-time chain has many equal-weight routes and earns about log 2 per corner, which matching ignores.

   The class with the larger summed e^(−F) wins. [Scoring kernel](src/decoders/free_energy.py) · [Decoder](src/decoders/free_energy_matching.py)

The cycle and hyperedge tables are built once per DEM. Setup took 1.6 s at distance 5 and 13.5 s at distance 9. Scoring a correction touches only the cycles next to it, so per-shot work grows with the size of the correction rather than with the size of the code. No step enumerates syndromes, and the relative gain held through distance 11 in tuning runs (46 against 32 failures in 30,000 shots, too few to be conclusive). Ablations on the tuning seed showed that squares and the downhill step are essential. Triangles alone, or skipping the descent, removed about half the gain. A second matcher with different weights added nothing.

Limits:
- The class vote needs one logical observable carried by one detector sector, which covers standard memory experiments. Other models fall back to correlated matching on the regrouped model. Faults with probability above 1/2 are complemented and handled as a known reference flip.
- The 1/2 odds scale and the 4-nat gate are tuning parameters, chosen on the tuning seed.
- Accuracy and runtime at larger distances are empirical. Related work: belief-matching and ensembles of matching decoders also improve on matching by approximating class probabilities. [Higgott et al.](https://arxiv.org/abs/2203.04948)

## Run the decoder and reproduce the benchmarks

Use Python 3.12 in a virtual environment. On Windows, run `py -3.12 -m venv .venv` then `.\.venv\Scripts\Activate.ps1`. On Linux or macOS, use `python3.12 -m venv .venv` then `source .venv/bin/activate`.

```sh
python -m pip install -e ".[dev]"
python -m pytest -q
```

To decode a file of bit-packed detector shots (Stim b8 format) given a graphlike decomposed Stim DEM, run the following. It writes one logical prediction per shot into a new file.

```sh
python -m src.decode --dem circuit.dem --dets shots.b8 --out predicted.b8
```

In Python, construct the decoder once per circuit and reuse it across batches.

```python
from src.circuits import detector_error_model, surface_code_circuit
from src.decoders.free_energy_matching import FreeEnergyMatchingDecoder

circuit = surface_code_circuit(distance=7, p=0.005, rounds=7, basis="z")
decoder = FreeEnergyMatchingDecoder(detector_error_model(circuit))
dets, truth = circuit.compile_detector_sampler(seed=123).sample(
    4096, separate_observables=True
)
predicted = decoder.decode_batch(dets)
logical_error_rate = (predicted != truth).any(axis=1).mean()
```

These commands reproduce the three evaluation designs. For comparable CPU timing, set `OPENBLAS_NUM_THREADS=1` before running Python (`$env:OPENBLAS_NUM_THREADS = '1'` in PowerShell). The two sweeps above used one timing pass (`--repeats 1`) and the confirmatory run used the default three.

```sh
python -m src.evaluate --distances 3 5 7 9 --ps .005 --shots 100000 --seed 9073101 --latency-shots 128 --out-dir results/confirm
python -m src.evaluate --distances 3 5 7 9 --ps .001 .003 .007 .01 --shots 40000 --seed 9073102 --latency-shots 64 --repeats 1 --out-dir results/noise
python -m src.evaluate --basis x --rounds 5 --distances 4 5 6 7 9 --ps .005 --shots 50000 --seed 9073103 --latency-shots 64 --repeats 1 --out-dir results/x_memory
python -m src.plots
```

The evaluator rotates decoder order between grid points. It reports the median of the warmed passes with batches of 4,096 shots, plus single-call p50 and p95, setup and warm-up time, Wilson intervals, and two-sided exact paired tests for failures. The `vs_candidate_p` column compares each decoder with free-energy matching on the same shots. The reported p-values are per comparison. Batch numbers exclude simulation, setup and scoring. Single calls go through the public `decode` API. They include the full pipeline for one shot and are much slower than batch decoding: 1.45 ms median at distance 9. Memory readings in the CSV are Python-traced peaks for one warm batch and can miss native allocations.

`make data` archives bit-packed samples. Give `src.evaluate` the same sampling arguments and `--data-dir data` to reuse them. `make bench`, `make test` and `make plots` are also available. Each run writes a timestamped CSV and a JSON manifest with its seed, device environment and package versions. The CSV records a SHA-256 of the detector and observable arrays for each point. Stim's seeded sample stream can vary with version and machine SIMD capabilities. [Stim sampler documentation](https://github.com/quantumlib/Stim/blob/main/doc/python_api_reference_vDev.md#stimcircuitcompile_detector_sampler)
