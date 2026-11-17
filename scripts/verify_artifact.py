"""Fast checks of the stored results against the values printed in the paper.

No calibration, circuit or hardware run is performed.
"""
from pathlib import Path
import hashlib, json
import numpy as np, pandas as pd
from pypdf import PdfReader
import nbformat

ROOT = Path(__file__).resolve().parents[1]
paper = ROOT / 'paper'


def close(value, printed, digits):
    """True when `value` rounds to `printed` at the printed number of digits."""
    return abs(value - printed) <= 0.5 * 10 ** (-digits) + 1e-9


nb = nbformat.read(ROOT / 'surface_calibration_experiments.ipynb', as_version=4)
nbformat.validate(nb)
assert sum(c.cell_type == 'code' for c in nb.cells) == 10
pdf = paper / 'supplement.pdf'
assert pdf.read_bytes().rstrip().endswith(b'%%EOF') and len(PdfReader(pdf, strict=True).pages) > 0

# Table I: held-out price RMSE (bp) at N = 64, against the audit of the archived benchmark.
TABLE_I = {'expm': (1.08, 1.91, 6.42, 6.58), 'fd': (1.08, 1.92, 6.44, 6.60),
           'lchs-beta': (1.07, 1.92, 6.45, 6.60), 'lchs-simple': (20.80, 20.80, 9.28, 9.33),
           'AH': (0.35, 2.73, 1.83, 3.48), 'PINN-fixed': (1.30, 1.73, 1.17, 2.12),
           'PINN-Wam': (1.35, 1.78, 1.14, 2.06)}
summary = pd.read_csv(paper / 'audit/summary_verified.csv')
stats = json.loads((paper / 'audit/audit_summary.json').read_text())
assert stats['completed_runs'] == 567 and stats['failed_runs'] == 1
for method, printed in TABLE_I.items():
    for (truth, noise), value in zip([('in', 0), ('in', 5), ('out', 0), ('out', 5)], printed):
        row = summary[(summary.n == 6) & (summary.method == method) & (summary.truth == truth)
                      & (summary.noise_bp == noise)].iloc[0]
        assert close(1e4 * row.heldout, value, 2), (method, truth, noise, 1e4 * row.heldout, value)

# Table II: held-out price RMSE (bp) and coefficient RMSE of the small-circuit calibration.
TABLE_II = {('exact', 0): (0.00, 0.000), ('fd', 0): (0.03, 0.000), ('lchs', 0): (8.67, 0.125),
            ('quantum_counts', 262144): (18.08, 0.252), ('quantum_counts', 4194304): (10.06, 0.186)}
runs = pd.read_csv(ROOT / 'new_experiment/calibration_results.csv')
assert (runs.method == 'quantum_counts').sum() == 6
for (method, shots), (held, coefficient) in TABLE_II.items():
    rows = runs[(runs.method == method) & (runs.shots == shots)]
    assert len(rows) == (3 if method == 'quantum_counts' else 1)
    assert close(rows.heldout_rmse_bp.mean(), held, 2), (method, shots, rows.heldout_rmse_bp.mean())
    assert close(rows.coefficient_rmse.mean(), coefficient, 3), (method, shots, rows.coefficient_rmse.mean())

# Table III: qubits and CNOT gates of the residual circuits with one and two maturity intervals.
TABLE_III = [(4, 4, 6, 2260, 8, 4322), (4, 8, 7, 18432, 10, 36666), (4, 16, 8, 153438, 12, 306678),
             (4, 32, 9, 1256996, 14, 2513794), (8, 4, 7, 10090, 9, 19200), (8, 8, 8, 77590, 11, 154200)]
sweep = pd.read_csv(ROOT / 'new_experiment/resource_sweep.csv')
assert len(sweep) == 12
for N, J, q1, cx1, q2, cx2 in TABLE_III:
    for expiries, qubits, cx in [(1, q1, cx1), (2, q2, cx2)]:
        row = sweep[(sweep.N == N) & (sweep.J == J) & (sweep.expiries == expiries)]
        assert len(row) == 1 and int(row.qubits.iloc[0]) == qubits and int(row.cx.iloc[0]) == cx, (N, J, expiries)

# Fig. 4: butterfly prices recomputed from the stored prices with the nonuniform-strike weights.
c = pd.read_csv(ROOT / 'consistency_example/consistency_predictions.csv')
failures = {}
for method, z in c.groupby('method'):
    k = z.k.to_numpy(); p = z.price.to_numpy(); w = np.diff(k)[1:] / (k[2:] - k[:-2])
    b = w * p[:-2] + (1 - w) * p[2:] - p[1:-1]
    np.testing.assert_allclose(b * 1e4, z.butterfly_bp.to_numpy()[1:-1], atol=3e-11, rtol=0)
    np.testing.assert_allclose((z.price - z.exact_price) * 1e4, z.error_bp, atol=3e-11, rtol=0)
    failures[method] = int(np.sum(b[z.complete_butterfly_in_band.to_numpy()[1:-1]] < -1e-8))
assert failures == {'exact': 0, 'fd': 0, 'lchs_2048': 5, 'lchs_4096': 0}

# The archived sources are those the benchmark ran with.
manifest = json.loads((paper / 'evidence/manifest.json').read_text())
for name, digest in manifest['source_sha256'].items():
    for copy in [paper / 'evidence/source' / name, ROOT / 'optional/legacy_source' / name]:
        assert hashlib.sha256(copy.read_bytes()).hexdigest() == digest, copy

print('Stored results agree with the paper: Table I (28 values), Table II (10 values), Table III (24 values),')
print('Fig. 4 butterfly prices, 567 completed and 1 failed benchmark runs, 6 measured runs, 12 circuits.')
print('No calibration or hardware run was performed by this check.')
