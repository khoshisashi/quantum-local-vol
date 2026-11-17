"""Compare a rerun of the notebook with the stored results.

usage: python scripts/compare_with_stored.py [build/paper] [--json report.json]

Timing fields, runtime descriptions, version labels and file hashes of the
runtime are left out. Numbers are compared with a relative tolerance of 1e-9.
A stored file that the rerun lacks counts as a difference. A different order
of columns or records and fields that only the rerun writes are reported as
notes. Records in another order are matched on the fields that every stored
record has, with numbers to ten decimal places.
"""
from pathlib import Path
import argparse, json, math, sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = ['new_experiment', 'surface_examples', 'consistency_example', 'paper/tables', 'paper/audit']
NOT_RERUN = {'new_experiment/run_provenance.json'}  # written once for the stored run, not by the notebook
SKIP_KEYS = ('seconds', 'solver_s', 'generated_utc', 'python', 'platform', 'packages', 'source_sha256', 'version')
RTOL, ATOL = 1e-9, 1e-12


def skipped(name):
    return any(word in str(name).lower() for word in SKIP_KEYS)


def canonical(value, keys=None):
    """Form in which records can be matched whatever their order: numbers to ten decimal places."""
    if isinstance(value, dict):
        return {k: canonical(v) for k, v in sorted(value.items()) if not skipped(k) and (keys is None or k in keys)}
    if isinstance(value, list):
        return [canonical(v) for v in value]
    if isinstance(value, float):
        return round(value, 10)
    return value


def differ(a, b, path, out, notes):
    """Collect the places where the stored value `a` and the rerun value `b` differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if skipped(key):
                continue
            if key not in a:
                notes.append(f'{path}/{key}: written by the rerun only')
            elif key not in b:
                out.append(f'{path}/{key}: missing in the rerun')
            else:
                differ(a[key], b[key], f'{path}/{key}', out, notes)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f'{path}: lengths {len(a)} and {len(b)}')
            return
        found = []
        for i, (x, y) in enumerate(zip(a, b)):
            differ(x, y, f'{path}[{i}]', found, [])
        if found and all(isinstance(x, dict) for x in a + b):
            required = {k for k in set.intersection(*(set(x) for x in a)) if not skipped(k)}
            partial = sorted(k for k in set().union(*(set(x) for x in a)) - required if not skipped(k))
            if all(required <= set(y) for y in b) and \
                    sorted(json.dumps(canonical(x, required), sort_keys=True) for x in a) == \
                    sorted(json.dumps(canonical(y, required), sort_keys=True) for y in b):
                notes.append(f'{path}: the same {len(a)} records in another order, matched on the fields of every stored record'
                             + (f' (only in some stored records: {", ".join(partial)})' if partial else ''))
                return
        out.extend(found)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if not (math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL) or (a != a and b != b)):
            out.append(f'{path}: {a} and {b}')
    elif a != b:
        out.append(f'{path}: {str(a)[:60]} and {str(b)[:60]}')


def compare(stored, rerun):
    out, notes = [], []
    if stored.suffix == '.csv':
        a, b = pd.read_csv(stored), pd.read_csv(rerun)
        if set(a.columns) != set(b.columns) or len(a) != len(b):
            return [f'shape or columns: {a.shape} and {b.shape}'], notes
        if list(a.columns) != list(b.columns):
            notes.append('the same columns in another order')
        for column in a.columns:
            if not skipped(column):
                differ(json.loads(a[column].to_json(orient='values')), json.loads(b[column].to_json(orient='values')),
                       column, out, notes)
    elif stored.suffix == '.json':
        differ(json.loads(stored.read_text()), json.loads(rerun.read_text()), '', out, notes)
    elif stored.suffix == '.tex':
        if stored.read_text() != rerun.read_text():
            out.append('text differs')
    return out, notes


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('rerun', nargs='?', default='build/paper', type=Path)
parser.add_argument('--json', type=Path, help='write the report to this file')
args = parser.parse_args()
rerun_root = args.rerun if args.rerun.is_absolute() else ROOT / args.rerun
if not rerun_root.is_dir():
    sys.exit(f'{rerun_root} is not a folder; run scripts/reproduce.py first')
report, failed = {}, 0
for folder in FOLDERS:
    for stored in sorted((ROOT / folder).glob('*')):
        name = f'{folder}/{stored.name}'
        if not stored.is_file() or stored.suffix not in {'.csv', '.json', '.tex'} or name in NOT_RERUN:
            continue
        rerun = rerun_root / folder / stored.name
        if not rerun.exists():
            report[name] = dict(result='missing in the rerun')
            failed += 1
            print(f'MISSING   {name}')
            continue
        differences, notes = compare(stored, rerun)
        report[name] = dict(result='different', differences=differences[:20]) if differences else \
            dict(result='same', **({'notes': notes[:20]} if notes else {}))
        failed += bool(differences)
        detail = f'  ({len(differences)} places, first: {differences[0]})' if differences else \
            (f'  (note: {"; ".join(notes[:3])})' if notes else '')
        print(f'{"DIFFERENT" if differences else "same     "} {name}{detail}')
print(f'{len(report)} files expected, {failed} missing or with differences.')
if args.json:
    args.json.write_text(json.dumps(dict(rerun=str(args.rerun), files_expected=len(report), files_missing_or_different=failed,
                                         relative_tolerance=RTOL, files=report), indent=2) + '\n')
sys.exit(1 if failed else 0)
