"""Run the actual Colab cells from the command line, recording each cell."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import argparse,time,json,hashlib,traceback
import nbformat
from IPython.utils.capture import capture_output
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--profile',choices=['smoke','paper'],default='smoke')
p.add_argument('--output',type=Path,help='Optional output directory; defaults to build/PROFILE')
a=p.parse_args();out=(a.output or ROOT/'build'/a.profile).resolve();out.mkdir(parents=True,exist_ok=True)
notebook=ROOT/'surface_calibration_experiments.ipynb';nb=nbformat.read(notebook,as_version=4);nbformat.validate(nb)
ns={'__name__':'__main__'};rows=[]
for i,c in enumerate(nb.cells):
    if c.cell_type!='code':continue
    tag=c.metadata.tags[0];source=c.source
    if tag=='parameters':
        source=source.replace('PROFILE = "paper"','PROFILE = '+repr(a.profile))
        source+='\nOUTPUT_ROOT = '+repr(str(out))+'\nDOWNLOAD_ZIP = False\n'
    start=time.perf_counter();print('Running',tag,flush=True)
    try:
        with capture_output() as log:exec(compile(source,'notebook:'+tag,'exec'),ns)
        (out/(f'cell_{i:02d}_{tag}.log')).write_text(log.stdout+log.stderr)
        rows.append(dict(cell=i,tag=tag,passed=True,seconds=time.perf_counter()-start))
    except Exception:
        rows.append(dict(cell=i,tag=tag,passed=False,traceback=traceback.format_exc()));break
result=dict(profile=a.profile,notebook_sha256=hashlib.sha256(notebook.read_bytes()).hexdigest(),
    code_cells=sum(c.cell_type=='code' for c in nb.cells),executed=rows,
    all_passed=len(rows)==sum(c.cell_type=='code' for c in nb.cells) and all(r['passed'] for r in rows))
(out/'execution_validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
if not result['all_passed']:raise SystemExit(1)
