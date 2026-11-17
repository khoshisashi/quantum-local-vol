"""Regenerate paper graphics from stored runs; no calibration is repeated."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import submission_figures,audit_results,consistency_example
paper=ROOT/'paper'
audit=audit_results.run(paper/'evidence',paper)
assert audit['completed_runs']==567 and audit['failed_runs']==1
submission_figures.generate(ROOT/'surface_examples',ROOT/'new_experiment',paper)
comparison=consistency_example.run(paper/'evidence',ROOT/'consistency_example')
submission_figures.consistency(ROOT/'consistency_example',paper)
print('Generated Figures 1–4, numeric macros and audited tables.')
print('New calculation: fixed-coefficient forward comparison only.')
