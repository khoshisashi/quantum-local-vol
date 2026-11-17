"""Compile the technical supplement from the stored tables and figures."""
from pathlib import Path
import shutil, subprocess
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
paper = ROOT / 'paper'
if not shutil.which('pdflatex'):
    raise SystemExit('Install a LaTeX distribution providing pdflatex first.')
for suffix in ['aux', 'out', 'toc', 'log']:
    (paper / ('supplement.' + suffix)).unlink(missing_ok=True)
for _ in range(3):
    p = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'supplement.tex'],
                       cwd=paper, capture_output=True, text=True)
    (paper / 'supplement_compile.txt').write_text(p.stdout + p.stderr)
    if p.returncode:
        raise SystemExit('Compilation failed; see paper/supplement_compile.txt')
path = paper / 'supplement.pdf'
assert path.read_bytes().rstrip().endswith(b'%%EOF')
pages = len(PdfReader(path, strict=True).pages)
assert pages > 0
print(path.relative_to(ROOT), pages, 'pages')
