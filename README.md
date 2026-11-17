# quantum-local-vol

Code, stored results and the technical supplement for

> K. Hoshisashi, "Quantum Calibration of Local-Volatility Surfaces from Sparse Option Quotes," in
> *Proc. IEEE Symposium Series on Computational Intelligence (SSCI)*, Gold Coast, Australia,
> Feb. 2027.

The repository holds a hybrid calibration loop in which finite-shot counts of residual circuits are
the only input of a classical search, a comparison of classical and finite-LCHS propagation, and
the stored runs from which the tables and figures of the paper are generated. All targets are
synthetic. Quantum execution is logical gate simulation with finite sampling on Qiskit Aer; nothing
runs on a device.

## Quick start

Use Python 3.11 or 3.12 on a CPU, from the root of the repository.

```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python scripts/verify_artifact.py          # a few seconds
python scripts/regenerate_figures.py       # about 15 s
python scripts/reproduce.py --profile smoke   # under a minute
python scripts/reproduce.py --profile paper   # the experiments of the paper, about 45 min
```

`verify_artifact.py` compares the stored results with the values printed in Tables I to III of the
paper and recomputes the butterfly prices of Fig. 4. It exits with an error if a value differs at
the printed precision. `regenerate_figures.py` audits the archived benchmark again and rewrites the
tables and figures under `paper/` from the stored runs, without a calibration; it also rewrites the
timing fields of `consistency_example/consistency_validation.json`. `reproduce.py` runs the cells of
the notebook and writes to `build/<profile>/`, and `compare_with_stored.py build/paper` then compares
that rerun with the stored results.

## Layout

```
surface_calibration_experiments.ipynb   self-contained notebook, for Colab or a local Jupyter
src/                      modules of the notebook, embedded in it verbatim
  surface_model.py          pricing model of the measured experiment and its classical references
  quantum_experiments.py    residual circuits, measured objective, pattern search, resource sweep
  surface_example.py        smile and skew examples
  consistency_example.py    fixed-coefficient comparison
  audit_results.py          audit of the archived benchmark
  submission_figures.py     Figs. 1 to 4, rows of Tables II and III, numbers quoted in the text
  experiment_figures.py     quadrature sweep and further figures of the measured experiment
  legacy_figures.py         implied-volatility inversion, figures of the archived benchmark
scripts/                  command-line entry points
new_experiment/           stored run of the measured calibration
surface_examples/         stored smile and skew examples
consistency_example/      stored fixed-coefficient comparison
paper/                    what the paper prints or cites
  supplement.pdf            technical supplement, with its TeX source next to it
  tables/                   rows and numbers printed in the paper and the supplement
  figures/                  figures of the paper and the supplement
  audit/                    audit of the archived benchmark
  evidence/                 archived benchmark: raw quotes, 568 checkpoints, results, source
optional/                 notebook and source modules of the archived benchmark
validation/               record of the checks run on this version
```

The top-level folders `src`, `paper`, `new_experiment`, `surface_examples`, `consistency_example`
and `optional` are those that the notebook writes into its output folder. The manuscript itself is
not part of the repository.

## What produces what

| Paper | Stored input | Generator |
|---|---|---|
| Fig. 1, multi-expiry calibration | a schematic, without data | `architecture` in `src/submission_figures.py` |
| Fig. 2, smile and skew | `surface_examples/surface_predictions.csv`, `*_quotes.json` | `src/surface_example.py` |
| Table I, held-out price RMSE at `N = 64` | `paper/audit/summary_verified.csv`, from the 567 runs in `paper/evidence/` | `src/audit_results.py`, also `paper/tables/all_cases.tex` |
| Table II, small-circuit calibration | `new_experiment/calibration_results.csv` | `src/quantum_experiments.py`, rows in `paper/tables/measured_rows.tex` |
| Fig. 3, independent residual measurements | `new_experiment/calibration_results.csv` | `precision` in `src/submission_figures.py` |
| Fig. 4, fixed-coefficient stress case | `consistency_example/consistency_predictions.csv` and the checkpoint `out_bp0_d101_n7_lchs-beta_s0` | `src/consistency_example.py` |
| Table III, logical resources | `new_experiment/resource_sweep.csv` | `resource_sweep` in `src/quantum_experiments.py`, rows in `paper/tables/resource_rows.tex` |
| Numbers quoted in the text | `paper/tables/new_run_macros.tex`, `surface_macros.tex`, `consistency_macros.tex` | `src/submission_figures.py` |
| Paired comparisons and results at `N = 128` | `paper/supplement.pdf`, `paper/audit/` | `scripts/build_supplement.py` |

`new_experiment/` also holds the raw measurement log, the history of every search, the transpiled
residual circuits in QPY format with their resource counts, the configuration and the provenance of
the run. The figures that the scripts write to `paper/figures/` come with a JSON file that names
their sources.

## Running the experiments

Open `surface_calibration_experiments.ipynb` in Google Colab, choose a CPU runtime and run all
cells. No uploads or credentials are needed: the notebook embeds the modules of `src/`, the TeX of
the supplement and the archived benchmark. `scripts/reproduce.py` runs the same cells from a
terminal, for the profiles `smoke` and `paper`.

| Profile | LCHS terms, cutoff | Shots per maturity and candidate | Seeds | Sweeps |
|---|---|---|---|---|
| `smoke` | 4, 4 | 4,096 | 101 | 1 |
| `pilot` | 8, 12 | 65,536 and 262,144 | 101, 202 | 1 |
| `paper` | 8, 12 | 262,144 and 4,194,304 | 101, 202, 303 | 2 |
| `precision` | 32, 32 | 1,048,576 and 16,777,216 | 101, 202, 303 | 2 |

`paper` reruns the six measured calibrations, the two surface examples, the circuit checks, the
resource sweep and the comparison of Fig. 4, and compiles the supplement. `smoke` checks the
execution path with smaller experiments, and its numbers are not those of the paper. `precision` is
an additional search that can take several hours and is not an experiment of the paper.

The notebook audits the archived benchmark of Table I and does not rerun it.
`optional/full_legacy_experiment.ipynb` reruns that benchmark, with the Andreasen–Huge and PINN
comparators. It needs JAX and QuantLib, and its `paper` profile can take days on a CPU. For this
release only its `smoke` profile was run, which takes five minutes and about 6 GB of memory.

## Not in this repository

- The manuscript. The published version is the one in IEEE Xplore.
- Runs on quantum hardware. None was made.
- Market data. None is used; the archived benchmark contains only an empty template for it.

## Conventions the numbers depend on

- **Units.** Prices are forward-normalised, `c = C/(D F)` at strike `k = K/F`. One basis point (bp)
  of price error is `1e-4` of `c`.
- **Model.** The local variance on a maturity interval is `0.01 + sum_j xi_j phi_j(log k)` with
  `0 <= xi_j <= 1`. Log-strike nodes are uniform on `[-1.2, 1.2]`, with a fitted central difference
  for the first derivative.
- **Measured calibration (Tables II and III, Fig. 3).** Four interior nodes, two basis functions
  with centres `-0.55` and `0.55` and width `0.85`, maturities `0.5` and `1`, quotes at
  `k = 0.80, 0.95, 1.10, 1.30` and `0.85, 1.00, 1.15, 1.35`. The target is exact propagation on the
  same grid at the coefficients `(0.75, 0.25)` and `(0.25, 0.75)`, without noise.
- **Surface examples (Fig. 2).** Three basis functions `0.08`, `0.1 (x + 0.45)^2` and
  `0.1 (x - 0.45)^2`, maturities `0.25, 0.5, 0.75, 1`, nine quotes at
  `k = 0.70, 0.78, 0.87, 0.95, 1.00, 1.06, 1.15, 1.28, 1.40`. Reference prices use 1,024 nodes,
  calibration and evaluation 128. LCHS uses 8,192 terms and cutoff 96.
- **Archived benchmark (Table I).** Four raised-cosine basis functions with centres
  `-0.6, -0.2, 0.2, 0.6` and width `0.45`, `N = 32, 64, 128`, maturities `182/365` and `1`, seven
  and five quotes. Baseline targets come from the calibration model, alternative targets from the
  local variance `0.01 + 0.055 exp(-((x + 0.1)/0.3)^2) + 0.035/(1 + exp(4x))` on both intervals,
  both on a reference grid of 2,048 nodes. Noisy quotes carry uniform perturbations bounded by
  5 bp, with the data seeds 101, 202 and 303 and five optimiser starts.
- **Held-out errors.** Equally spaced strikes on `[0.7, 1.4]` at every quoted maturity: 141 for the
  surface examples, 181 for the benchmark and 81 for the measured calibration, without those that
  coincide with a calibration strike. The reference is the noise-free price of the generating model.
- **LCHS.** Midpoint nodes on `[-K_c, K_c]` and weights normalised to sum to one. `beta` is the
  improved kernel with `beta = 0.8`, `simple` the original kernel. The shift is the sum, over the
  constant generator and the basis generators, of the negative part of the smallest eigenvalue of
  their symmetric parts.
- **Search.** Bounded pattern search from all coefficients at `0.5` with step `0.25`, over four
  coordinate directions and two within-interval tilt directions. The incumbent is measured again
  for every direction, and ties keep the earlier candidate.
- **Sampling.** Aer finite-shot counts of the decomposed circuits. The sampler seed of a call is the
  base seed plus 1009 times the index of the call.
- **Intervals of Fig. 3.** Clopper–Pearson intervals of a fresh evaluation after the search, with
  tail probabilities `0.05/(2S)` for `S` maturities.
- **Resources (Table III).** Counts of `u` and CNOT gates after decomposition, with all-to-all
  connectivity and without error correction. The sweep uses cutoff 4 up to eight terms and 32
  above; the calibration with eight terms uses cutoff 12.

## Provenance

The measured calibration, the surface examples and the fixed-coefficient comparison were run in
September 2026 with the package versions of `requirements.txt`; `new_experiment/run_provenance.json`
records the run. The archived benchmark was run in September 2026 on a Colab CPU, with version
`2026-09-05-r2` of its notebook and the package versions listed in `paper/evidence/manifest.json`.
It has 567 completed runs and one Andreasen–Huge run that failed on its input, which is kept in
`paper/evidence/failures.json`. `paper/evidence/` is the output of that run as it was archived,
without its README and without the manuscript draft that the run wrote as `main.tex`, whose hash
remains in `paper/evidence/manifest.json`. Neither belongs to the paper.

The repository was prepared for the public release in November 2026 from the package that
accompanied the submission. The modules under `src/` and the stored runs are unchanged. The
manuscript and the files of the review period were removed, the notebook was rebuilt without them
and with its text in English, the notebook under `optional/` no longer embeds that draft and its
Japanese text was translated, the supplement names its author and this repository and writes the
shift and the quadrature nodes as the published paper does, and `scripts/verify_artifact.py` now
checks the tables of the paper. `validation/` records the checks run on this version, including a
rerun of the `paper` profile. The preparation was done with the assistance of generative AI (Claude,
Anthropic).

## Licence and citation

The code is released under the MIT licence; `THIRD_PARTY_NOTICES.md` lists what comes from
elsewhere. To cite the repository, please cite the paper.

```bibtex
@inproceedings{hoshisashi2027quantum,
  author    = {Hoshisashi, Kentaro},
  title     = {Quantum Calibration of Local-Volatility Surfaces from Sparse Option Quotes},
  booktitle = {Proc. IEEE Symposium Series on Computational Intelligence (SSCI)},
  address   = {Gold Coast, Australia},
  year      = {2027},
  note      = {To appear}
}
```
