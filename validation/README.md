# Checks run on this version

Run on 17 November 2026 with Python 3.12.3 and the package versions of `requirements.txt`, on two
CPU cores. The verification and the smoke profile were also run with Python 3.11.18.

| Check | Command | Result | Record |
|---|---|---|---|
| Stored results against Tables I to III and Fig. 4 of the paper | `python scripts/verify_artifact.py` | passed | `verify_artifact.txt` |
| Audit of the archived benchmark, tables and figures | `python scripts/regenerate_figures.py` | 567 completed runs and 1 failed run accounted for; the eight audit files and the nine table files come out identical to the stored ones | `regenerate_figures.txt` |
| Supplement | `python scripts/build_supplement.py` | 11 pages | |
| Every cell of the notebook, small settings | `python scripts/reproduce.py --profile smoke` | 10 of 10 code cells passed | `smoke_execution.json` |
| Experiments of the paper | `python scripts/reproduce.py --profile paper` | 10 of 10 code cells passed, in 30 minutes on one core | `paper_execution.json` |
| Rerun against the stored results | `python scripts/compare_with_stored.py build/paper` | 37 files expected, none missing or with differences | `paper_rerun_comparison.json` |
| Archive embedded in the notebook | file-by-file comparison with the archive of the review version | 668 files, each with the same content as in that archive; three of its files are left out | `embedded_archive.json` |
| Notebook of the archived benchmark, small settings | the code cells of `optional/full_legacy_experiment.ipynb` with `PROFILE = "smoke"` | 17 of 17 code cells passed, in 5 minutes | `legacy_smoke_execution.json` |

The rerun of the `paper` profile reproduces the six measured calibrations with every count of the
measurement log, the resource sweep, the surface examples, the fixed-coefficient comparison, the
tables and the audit. The comparison leaves out timing fields and reports three notes on layout.
They come from the logging of the delivered source, which is one revision later than the one the
stored run was made with (`new_experiment/run_provenance.json`): the rerun writes two verification
totals into `checkpoint.json`, it logs each independent final check directly after its search
where the stored `measurement_log.json` has the twelve of them at the end, and it writes the
columns of `resource_sweep.csv` in another order.

The notebook of the archived benchmark was run with Python 3.13.18, jax 0.11.1, qiskit 2.5.2 and
QuantLib 1.43, the versions its first cell installs, and numpy 2.5.5, scipy 1.18.1, pandas 3.1.1 and
matplotlib 3.11.2. Its circuit check used about 5.5 GB of memory. The run was made with the
notebook as it is in the repository, after the manuscript draft was taken out of it and its text
translated.

The records of the three notebook runs hold the SHA-256 of the notebook they ran, and the runs of
the main notebook were made after its last rebuild. The archive embedded in the notebook leaves out
`figures/heldout_comparison.pdf`, which the audit cell regenerates, and the README and the
manuscript draft of the archived run, which are not part of the repository.

Not covered: the `paper` profile of `optional/full_legacy_experiment.ipynb`, the profiles `pilot`
and `precision` of the main notebook, and Colab. Both notebooks were run from a terminal, cell by
cell.
