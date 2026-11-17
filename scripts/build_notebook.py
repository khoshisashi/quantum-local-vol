"""Build the self-contained notebook from src/, paper/ and optional/.

The notebook embeds the source modules, the TeX of the supplement and the
archived benchmark, so that it runs on Colab without uploads.
"""
from pathlib import Path
import base64, hashlib, io, zipfile, zlib
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'paper'
ARCHIVED_FIGURES = ['solver_comparison', 'precision_basis', 'fd_identifiability', 'ablation',
                    'domain_slices', 'pinn_losses', 'qhd_diagnostics']


def archive():
    """Deterministic ZIP of the archived benchmark; it is data and is never imported."""
    files = [p for folder in ['evidence', 'audit'] for p in sorted((PAPER / folder).rglob('*')) if p.is_file()]
    files += [PAPER / 'figures' / (name + '.pdf') for name in sorted(ARCHIVED_FIGURES)]
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as z:
        for p in files:
            info = zipfile.ZipInfo(p.relative_to(PAPER).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            z.writestr(info, p.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return stream.getvalue()


payload = archive()
payload_hash = hashlib.sha256(payload).hexdigest()
assets = {name: (PAPER / name).read_text() for name in
          ['supplement.tex', 'measured_supplement.tex', 'supplement_references.tex']}
assets['requirements.txt'] = (ROOT / 'requirements.txt').read_text()
sources = {p.name: p.read_text() for p in sorted((ROOT / 'src').glob('*.py'))}
legacy64 = base64.b64encode(zlib.compress((ROOT / 'optional/full_legacy_experiment.ipynb').read_bytes(), 9)).decode()

nb = nbf.v4.new_notebook(metadata={
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.11'}, 'colab': {'provenance': []},
    'quantum_local_vol': {'version': '1.0.0', 'default_profile': 'paper',
                          'archived_benchmark_zip_sha256': payload_hash}})
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s))


def code(s, tag):
    if tag == 'materialise':
        s = '# @title Embedded sources and archived benchmark (expand to inspect)\n' + s
    c = nbf.v4.new_code_cell(s)
    c.metadata['tags'] = [tag]
    if tag == 'materialise':
        c.metadata.update({'jupyter': {'source_hidden': True}, 'cellView': 'form'})
    cells.append(c)


md('''# Quantum calibration of local-volatility surfaces from sparse option quotes

**Run all cells on a CPU runtime (Google Colab or a local Jupyter). No input files have to be uploaded.**

This notebook accompanies the paper of the same title (IEEE SSCI 2027, CIFEr). It builds the quantum circuits, runs the **calibration experiment** in which the coefficients are updated from finite-shot measurement counts, and writes the figures, tables, CSV files, measurement logs, executable circuits and the technical supplement.

### What the experiments test

1. **Financial task**: estimate the local volatility of the pricing equation from prices at several strikes and maturities.
2. **Role of the quantum circuit**: propagate prices for candidate coefficients and turn the difference from the observed prices into a measurement probability. A classical search updates the coefficients from the measured values.
3. **How to read the comparisons**: agreement with classical prices of the same model, the difference from surfaces of other models, measurement precision and circuit cost.

The smile and skew examples compare exact propagation, finite differences and LCHS at common, classically calibrated coefficients. The calibration from quantum measurements is tested on a separate grid of four nodes.
Implied volatility (IV) restates a price in the market's volatility quotation; local volatility is the coefficient of the model that produces the prices. The figures keep the two apart.
One basis point (bp) of price error is 0.0001 of the normalised price.

### Scope of execution

| Experiment | In this notebook | What it shows |
|---|---|---|
| Measured multi-expiry calibration | Rerun by default: 4 price nodes, 4 coefficients, 2 maturities | Logical circuits, measured counts and the behaviour of the calibration loop |
| Four-maturity smile and skew | Calibrated and computed by default: 128 nodes, 3 coefficients per interval | Financial shapes and the forward approximations at the same coefficients |
| Financial comparison on larger grids | Stored results are audited and redrawn | Classical solvers, matrix LCHS, AH and PINN |
| Fixed-coefficient consistency (Fig. 4) | Four methods re-evaluated at stored coefficients | Price error and butterfly prices as the number of terms grows |
| Full recomputation of the large comparison | A separate notebook is written at the end | Can be rerun independently when needed |

`paper` runs two shot budgets with three seeds each. Allow tens of minutes or more on a CPU.
`smoke` checks the execution path and does not reproduce the numbers of the paper.
`precision` is an additional search with more quadrature terms. It can take **several hours** and is not an experiment of the paper.
Nothing is sent to hardware: execution is logical gate simulation on Qiskit Aer with finite sampling.
''')
code('''# Run settings. "paper" is the configuration of the paper.
PROFILE = "paper"  # "smoke", "pilot", "paper", "precision"
BUILD_PDF = True
INSTALL_DEPENDENCIES = True
DOWNLOAD_ZIP = False  # Set True and rerun the last cell to download the archive in Colab.
import os, sys, subprocess, importlib.metadata, pathlib
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
OUTPUT_ROOT = "/content/quantum_surface_run" if pathlib.Path("/content").exists() else "quantum_surface_run"
# Dependency versions of the validated environment. Use a CPU runtime.
packages = ["numpy==2.3.5", "scipy==1.17.0", "pandas==2.2.3",
            "matplotlib==3.10.8", "qiskit==2.5.2", "qiskit-aer==0.17.2", "pypdf==5.9.0"]
if INSTALL_DEPENDENCIES:
    subprocess.check_call([sys.executable,"-m","pip","install","--quiet",*packages])
assert PROFILE in {"smoke","pilot","paper","precision"}
print("Profile:",PROFILE,"| Output:",OUTPUT_ROOT)
''', 'parameters')
md('''## 1. Prepare the sources and the stored data

The next cell holds the source modules and the TeX of the supplement. The Python files under `src/` can be read and edited as they are.
The compressed data belong to the archived financial comparison only. Their hash and extraction paths are checked, and the Python code archived with them is not executed.
''')
code('SOURCES = ' + repr(sources) + '\nASSETS = ' + repr(assets) + '\nEVIDENCE_SHA256 = ' + repr(payload_hash)
     + '\nEVIDENCE_BASE64 = ' + repr(base64.b64encode(payload).decode())
     + '\nLEGACY_NOTEBOOK_ZLIB64 = ' + repr(legacy64) + '''
import base64, hashlib, io, json, zipfile, shutil, zlib
from pathlib import Path
root=Path(OUTPUT_ROOT).resolve();root.mkdir(parents=True,exist_ok=True)
src=root/"src";src.mkdir(exist_ok=True)
paper=root/"paper";paper.mkdir(exist_ok=True)
run=root/"new_experiment";run.mkdir(exist_ok=True)
for name,content in SOURCES.items(): (src/name).write_text(content,encoding="utf-8")
for name,content in ASSETS.items():
    path=paper/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(content,encoding="utf-8")
blob=base64.b64decode(EVIDENCE_BASE64)
assert hashlib.sha256(blob).hexdigest()==EVIDENCE_SHA256
with zipfile.ZipFile(io.BytesIO(blob)) as z:
    for info in z.infolist():
        path=(paper/info.filename).resolve()
        if not path.is_relative_to(paper): raise ValueError("Archive path escapes output directory")
        if not info.is_dir():
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(z.read(info))
(root/"optional").mkdir(exist_ok=True)
(root/"optional/full_legacy_experiment.ipynb").write_bytes(zlib.decompress(base64.b64decode(LEGACY_NOTEBOOK_ZLIB64)))
sys.path.insert(0,str(src))
from IPython.display import display, Image, FileLink
print("Source modules:",list(SOURCES));print("Archived benchmark:",EVIDENCE_SHA256)
''', 'materialise')
md('''## 2. Set up the pricing model and check the circuits independently

Four quotes are placed at each of two maturities (six months and one year). The generating coefficients are known but are not passed to the measured search.
The circuit of the later maturity also passes through the propagation up to the earlier one.

Only here is the circuit statevector compared with an independent matrix calculation. **No statevector is used to decide the coefficients of the calibration.**
''')
code('''import numpy as np, pandas as pd
from surface_model import grid, data_for, save_json
from quantum_experiments import circuit_checks, run_calibration_experiment, resource_sweep
problem=data_for(grid())
checks=circuit_checks(problem,terms=4,cutoff=4.,out=run)
display(checks[["expiry","probability","expected","absolute_error"]])
assert checks.absolute_error.max()<2e-10
''', 'circuit_check')
md('''## 3. Estimate the surface coefficients from measurements

For every candidate the circuit is decomposed into gates, and the number of times the ancillas are found in the success state is counted. Only the loss obtained by scaling these counts with known factors is passed to the classical bounded pattern search.
The same search order is used for exact propagation, finite differences and the matrix of the same finite LCHS. The unit of comparison is an independent sampler seed.

A checkpoint is saved after every run. The measurements used by the search and the final independent check are kept apart in the records.
''')
code('''results=run_calibration_experiment(run,PROFILE)
display(results[["method","shots","seed","evaluations","quote_rmse_bp","heldout_rmse_bp","coefficient_rmse"]])
status=json.loads((run/"status.json").read_text())
assert status["completed"] and status["no_statevector_access_in_optimiser"]
''', 'new_calibration')
md('''## 4. Circuit resources and approximation accuracy

The numbers of qubits and CNOT gates, the depth and the classical synthesis time of the decomposed circuits are saved.
CNOT counts are per circuit execution. The totals multiplied by the number of shots are in the calibration CSV.
The resource sweep is a diagnostic of gate counts, so its cutoff differs from that of the calibration in some cases, and the CSV records both.

The additional quadrature sweep is a **matrix calculation**. The number of terms and the cutoff are varied independently to show how the choice of approximation affects the price accuracy.
''')
code('''resources=resource_sweep(run,PROFILE)
from experiment_figures import quadrature_diagnostic
quadrature=quadrature_diagnostic(run)
display(resources[["N","J","cutoff","expiries","qubits","cx","depth"]])
display(quadrature[["N","J","cutoff","heldout_rmse_bp"]].head(10))
''', 'resources')
md('''## 5. Smile, skew and the diagram of the calibration

From nine quotes at each of four maturities, the coefficients of three positive quadratic basis functions are calibrated interval by interval. Two sets of generating coefficients are stated, a smile and a skew.
Reference prices use 1,024 nodes; calibration and evaluation use 128. With the **same coefficients**, calibrated by exact propagation, the surface is evaluated by exact propagation, finite differences and LCHS.
The example compares the representation of the surface and the forward approximations. It is not an independent calibration from quantum measurements.
LCHS uses 8,192 terms and cutoff 96; `smoke` uses a lighter setting.

Each row is the smile or the skew: the surface over the four maturities with the input quotes on the left, the one-year slice on the right. The axes are k, T and IV (%). IV is not smoothed, and invalid prices are not clipped.
''')
code('''import surface_example, submission_figures
examples=root/"surface_examples"
example_accuracy=surface_example.run(examples,"paper" if PROFILE=="paper" else "smoke")
example_checks=surface_example.validate_reference(examples,"paper" if PROFILE=="paper" else "smoke")
figure_values=submission_figures.generate(examples,run,paper)
display(Image(filename=str(paper/"figures/calibration_architecture.png")))
display(Image(filename=str(paper/"figures/financial_surfaces.png")))
display(example_accuracy[["case","method","heldout_rmse_bp","quote_rmse_bp"]])
''', 'surface_examples')
md('''## 6. Measurement precision and calibration results

The figure shows how finely the final price residual can be resolved as the number of shots grows.
Points and 95% intervals are values **measured independently after the search**, at the selected coefficients. Crosses are the values of the finite-LCHS matrix at the same coefficients.
All declared seeds are shown, not the best one. A count of zero does not establish that the true residual is zero.
The intervals are neither confidence intervals of the coefficients nor a guarantee of global optimality.
''')
code('''display(Image(filename=str(paper/"figures/measurement_precision.png")))
measured=results[results.method=="quantum_counts"].copy()
display(results[["method","shots","seed","heldout_rmse_bp","coefficient_rmse"]])
display(measured[["shots","seed","independent_final_measured_loss","independent_final_mse_lower","independent_final_mse_upper","total_shots","total_cx_executions"]])
''', 'measured_precision')
md('''## 7. Audit the archived comparison with classical methods

The 567 completed runs and the one failed run of the archived benchmark are audited again. The calibrations of that benchmark are not rerun here.
LCHS, exact propagation and finite differences share one coefficient model; AH and PINN use different representations, and the two groups should not be confused.
The no-arbitrage violations of finite LCHS in the archived benchmark are reported as they are.

Fig. 4 compares, at the same coefficients, the first maturity of the clean outside-model case with N=128 and optimiser start 0.
This is the stress case with the largest butterfly violation inside the quoted band among the stored results, not an average case.
LCHS is recomputed with 2,048 and 4,096 terms (cutoff 48, beta 0.8) and compared with exact propagation and finite differences.
The top panel shows the difference from the exact price at the same coefficients, the bottom panel the butterfly price of three adjacent strikes. A negative value is a convexity violation.
This is neither a new calibration nor a new circuit measurement. The improvement inside the quoted band and the violations that remain outside it are reported separately.
''')
code('''import audit_results
audit=audit_results.run(paper/"evidence",paper)
assert audit["completed_runs"]==567 and audit["failed_runs"]==1
summary=pd.read_csv(paper/"audit/summary_verified.csv")
a=summary[(summary.n==6)&(summary.truth=="out")&(summary.noise_bp==5)].copy()
a["heldout_price_bp"]=a.heldout*1e4;a["dataset_sd_bp"]=a.heldout_sd*1e4
display(a[["method","heldout_price_bp","dataset_sd_bp","opt_s","datasets","runs"]])
import consistency_example
consistency_dir=root/"consistency_example"
consistency_checks=consistency_example.run(paper/"evidence",consistency_dir)
submission_figures.consistency(consistency_dir,paper)
display(Image(filename=str(paper/"figures/financial_consistency.png")))
display(pd.DataFrame(consistency_checks["metrics"]).T)
finance={"surface_examples":figure_values,"validation":example_checks,
         "fixed_coefficient_consistency":consistency_checks,
         "scope":"new common-coefficient quadratic examples; retained independently calibrated benchmark"}
''', 'financial_audit')
md('''## 8. Collect the supplement, tables, figures and results

The technical supplement is compiled only when the six runs of the `paper` profile have completed.
Other profiles write their figures and logs but no PDF, so that their numbers are not taken for those of the paper.
On Colab without a TeX installation this cell installs the packages it needs.

The ZIP archive holds the figures and tables, the audited archived data, the new measurement logs, the runtime environment, the sources, the circuits in QPY format and the supplement.
`optional/full_legacy_experiment.ipynb` reruns the large archived experiment in full, with the classical solvers, AH and PINN (a long run).
''')
code('''from datetime import datetime, timezone
import platform
paper_ready=(PROFILE=="paper" and status["measured_runs"]==6 and json.loads((examples/"configuration.json").read_text())["profile"]=="paper")
if not paper_ready:
    (paper/"supplement.pdf").unlink(missing_ok=True)
    print("Non-paper profile: figures and logs generated; the supplement is not compiled.")
elif BUILD_PDF:
    if shutil.which("pdflatex") is None and Path("/content").exists():
        subprocess.check_call(["apt-get","-qq","update"])
        subprocess.check_call(["apt-get","-qq","install","-y","texlive-latex-base","texlive-latex-extra","texlive-fonts-recommended"])
    if shutil.which("pdflatex"):
        from pypdf import PdfReader
        for suffix in ["aux","log","out","toc"]:
            (paper/("supplement."+suffix)).unlink(missing_ok=True)
        for _ in range(3):
            log=subprocess.run(["pdflatex","-interaction=nonstopmode","-halt-on-error","supplement.tex"],cwd=paper,text=True,capture_output=True)
            (paper/"supplement_compile.txt").write_text(log.stdout+log.stderr)
            if log.returncode:raise RuntimeError("TeX compilation failed; see supplement_compile.txt")
        generated_pdf=paper/"supplement.pdf"
        assert generated_pdf.read_bytes().rstrip().endswith(b"%%EOF"), "Incomplete PDF"
        pages=len(PdfReader(generated_pdf,strict=True).pages)
        assert pages>0, "Empty PDF"
        print("Compiled the technical supplement:",pages,"pages.")
    else:
        print("TeX files generated. Install LaTeX or compile paper/supplement.tex elsewhere.")
save_json(root/"reproduction_manifest.json",dict(profile=PROFILE,paper_ready=paper_ready,
    generated_utc=datetime.now(timezone.utc).isoformat(),python=platform.python_version(),
    original_archive_sha256="6684a5cf912b9703b9180395bb1df48e832fcc5944610421d279a211952b4fca",
    historical_evidence_sha256=EVIDENCE_SHA256,
    source_sha256={k:hashlib.sha256(v.encode()).hexdigest() for k,v in SOURCES.items()},
    new_experiment_executed=True,new_surface_examples_executed=True,
    fixed_coefficient_consistency_reexecuted=True,
    historical_benchmark_reexecuted=False,hardware_execution=False,
    financial_audit=audit,finance_figure_checks=finance))
archive=root.parent/("quantum_surface_"+PROFILE+"_outputs.zip")
with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
    for p in sorted(root.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and p.suffix not in {".aux",".log",".out",".toc"}:
            z.write(p,p.relative_to(root))
print("Results:",archive)
display(FileLink(str(archive)))
if paper_ready and (paper/"supplement.pdf").exists():display(FileLink(str(paper/"supplement.pdf")))
''', 'export')
code('''if DOWNLOAD_ZIP:
    from google.colab import files
    files.download(str(archive))
''', 'download')
for i, c in enumerate(cells):
    c['id'] = f'cell-{i:02d}'
nb.cells = cells
nbf.validate(nb)
path = ROOT / 'surface_calibration_experiments.ipynb'
nbf.write(nb, path)
print(path.name, len(cells), 'cells;', sum(c.cell_type == 'code' for c in cells), 'code cells;', path.stat().st_size, 'bytes')
print('archived benchmark ZIP sha256:', payload_hash)
