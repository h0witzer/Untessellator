#!/usr/bin/env python3
"""
rebuild_patches.py - ml/solid_rebuild.py (topo) over every part of some patch folders: one process per
part (a broken input can overflow OCCT's stack), several at once, resumable.

    python rebuild_patches.py --mesh D:/abc-data/fill_eval --truth D:/abc-data/fit_truth \\
        --patches D:/abc-data/lab/s13/patches_fill_v2stack --variants base-all,retype-ring1 \\
        --meshers native,gmsh --workers 8 --out rebuild.json

A patch folder is test/surfaceScore.mjs --patches-out: <variant>/<stem>.fid (a segment per facet)
and <variant>/<stem>.json (each segment's surface). Only fully analytic parts are rebuilt, as in
solid_rebuild.py's default (--part-set freeform / all for the others; --freeform to give surfaceless
groups a fitted B-spline; --fallback for facets where the analytic build fails, counted apart).
RECREATED = closed, BRepCheck valid, and volume within 1% of the STEP's (of the mesh's where no STEP
volume is known: the Onshape set).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mesh', type=Path, required=True)
    ap.add_argument('--truth', type=Path, required=True)
    ap.add_argument('--patches', type=Path, required=True, help='folder of variant folders')
    ap.add_argument('--variants', required=True)
    ap.add_argument('--meshers', default='native,gmsh')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--timeout', type=int, default=900)
    ap.add_argument('--limit', type=int, default=0, help='at most this many parts')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--repair-loop', action='store_true', help='solid_rebuild.py --repair-loop')
    ap.add_argument('--rebuild-dir', default='rebuild', help='per-variant result folder (cached: reused if present)')
    ap.add_argument('--freeform', action='store_true', help='solid_rebuild.py --freeform (B-splines for surfaceless groups)')
    ap.add_argument('--part-set', choices=['analytic', 'freeform', 'all'], default='analytic',
                    help='fully analytic parts (default), parts with a non-analytic face, or all')
    ap.add_argument('--fallback', action='store_true',
                    help='solid_rebuild.py --fallback (facets where the analytic build fails; tiers reported)')
    ap.add_argument('--stems', type=Path, default=None, help='only these stems (a file, comma or line separated)')
    args = ap.parse_args()

    manifest = json.loads((args.mesh / 'manifest.json').read_text())
    only = set(args.stems.read_text().replace(',', ' ').split()) if args.stems else None
    jobs, parts = [], 0
    for entry in manifest['parts']:
        part = entry['part']
        srf_path = args.truth / f'{part}.srf'
        if not srf_path.exists():
            continue
        srf = np.fromfile(srf_path, np.float64).reshape(-1, 12)
        analytic = bool(np.all((srf[:, 0] == srf[:, 1]) & (srf[:, 1] <= 4)))
        if args.part_set != 'all' and analytic != (args.part_set == 'analytic'):
            continue
        added = False
        for mesher in args.meshers.split(','):
            v = entry['variants'].get(mesher)
            if not v or (only is not None and v['stem'] not in only):
                continue
            for variant in args.variants.split(','):
                if (args.patches / variant / f"{v['stem']}.json").exists():
                    jobs.append((variant, part, mesher, v['stem']))
                    added = True
        parts += added
        if args.limit and parts >= args.limit:
            break
    print(f'{len(jobs)} rebuilds over {parts} {args.part_set} parts', flush=True)

    def run(job):
        variant, part, mesher, stem = job
        d = args.patches / variant
        out = d / args.rebuild_dir / f'{stem}.json'
        error = None
        if not out.exists():
            cmd = [sys.executable, str(HERE / 'solid_rebuild.py'), '--mesh', str(args.mesh), '--truth', str(args.truth),
                   '--fits', str(d), '--fid-dir', str(d), '--meshers', mesher, '--method', 'topo', '--parts', part,
                   '--json-out', str(out)] + (['--repair-loop'] if args.repair_loop else []) + \
                  (['--freeform'] if args.freeform else []) +                   (['--fallback'] if args.fallback else [])
            # A Store-installed interpreter sometimes refuses to start under load ("Unable to create
            # process"): that is no verdict on the rebuild, so it is retried.
            for attempt in range(4):
                try:
                    r = subprocess.run(cmd, timeout=args.timeout, capture_output=True, cwd=HERE, text=True,
                                       errors='replace')
                    error = f'exit {r.returncode}: {r.stderr.strip()[-160:]}' if not out.exists() else None
                    if error and 'Unable to create process' not in error and r.returncode not in (126, 127):
                        break
                except subprocess.TimeoutExpired:
                    error = 'timed out'
                    break
                except OSError as exc:
                    error = f'could not start: {exc}'
                if out.exists():
                    break
                time.sleep(5)
        if out.exists():
            rows = json.loads(out.read_text())
        else:
            rows = [{'part': part, 'mesher': mesher, 'method': 'topo', 'closed': False, 'valid': False,
                     'error': error or 'no output'}]
        for r in rows:
            r['variant'] = variant
        return rows

    rows = []
    with ThreadPoolExecutor(args.workers) as ex:
        for i, rs in enumerate(ex.map(run, jobs)):
            rows.extend(rs)
            if i % 20 == 19:
                print(f'  {i + 1}/{len(jobs)}', flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, default=str))

    def recreated(r):
        err = r.get('true_vol_err') if r.get('true_vol_err') is not None else r.get('vol_err')
        return bool(r.get('closed') and r.get('valid') and err is not None and err < 0.01)

    def tier(r):
        return (r.get('info') or {}).get('tier', 'analytic' if recreated(r) else 'none')

    # With --fallback a recreated solid may hold facets: "analytic" counts only the fully analytic ones.
    print('\nvariant           mesher   parts  closed  valid  recreated  analytic  hybrid  planar  faceted')
    for variant in args.variants.split(','):
        for mesher in args.meshers.split(','):
            rs = [r for r in rows if r['variant'] == variant and r['mesher'] == mesher]
            if not rs:
                continue
            by = {t: sum(1 for r in rs if recreated(r) and tier(r) == t) for t in ('analytic', 'hybrid', 'planar', 'faceted')}
            print(f'  {variant:15} {mesher:7} {len(rs):5}  {sum(bool(r.get("closed")) for r in rs):6}'
                  f'  {sum(bool(r.get("valid")) for r in rs):5}  {sum(recreated(r) for r in rs):9}'
                  f'  {by["analytic"]:8}  {by["hybrid"]:6}  {by["planar"]:6}  {by["faceted"]:7}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
