#!/usr/bin/env python3
"""
export_abc_obj_eval.py - an evaluation set of REAL-EXPORTER meshes: ABC's own Onshape tessellations.

    python export_abc_obj_eval.py --obj D:/abc-data/obj --out D:/abc-data/onshape_eval --n 300

Every mesh the models have been scored on so far came from our own tessellators (OCCT at three
densities, OCCT after a NURBS conversion, gmsh, a game-art bake). A user's STL comes from their CAD
system's exporter. ABC publishes the meshes Onshape itself exported (the `obj` format), and its
feature files - which we already have for every model - label each surface with the OBJ triangles
it covers (`face_indices`). So those meshes are real-exporter tessellations with exact labels.

Output is test/fillEval.mjs's format (one variant, `onshape`), so test/autoSeed.mjs and
test/growEval.mjs score it unchanged:

    <stem>.pos  float32, 9 per triangle     <stem>.fid  int32 face (surface index in the feature file)
    <stem>.typ  uint8 stored surface type (TYPE_NAMES order), plus manifest.json
    <truth-out>/<part>.srf  the true surface of every face id, in fit_truth's layout (--truth-only
                            rewrites these for an existing export)

HELD OUT: a model that appears anywhere in the bakes (every mesher, either split), fill_eval or the
edge-training set is excluded, so no model here reached any trained component (the seed models, the
refiner and both edge models). Types are ABC's STORED types
(not the canonical relabelling, which exists only for baked models), and a model whose labels do not
cover every triangle exactly once is skipped.

FREEFORM (--freeform): only models with at least one non-analytic face (Revolution, Extrusion,
BSpline, Other), the set the freeform fitter is graded on; the rebuild scorecards so far used only
fully analytic parts. Both modes also write

    <truth-out>/<part>.ffs.json  {surface index: exact definition} of every non-analytic face
                                 (abc_freeform.read_freeform: poles, knots, weights, swept curves)

    python export_abc_obj_eval.py --obj D:/abc-data/obj --freeform --n 150 \\
        --out D:/abc-data/freeform_eval --truth-out D:/abc-data/freeform_truth
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from abc_freeform import read_freeform  # noqa: E402
from relabel_sidecar import val_split  # noqa: E402

TYPE_NAMES = ['Plane', 'Cylinder', 'Cone', 'Sphere', 'Torus', 'Revolution', 'Extrusion', 'BSpline', 'Other']


def read_obj(path: Path):
    verts, tris = [], []
    with open(path, 'r', encoding='utf8', errors='replace') as f:
        for line in f:
            if line.startswith('v '):
                p = line.split()
                verts.append((float(p[1]), float(p[2]), float(p[3])))
            elif line.startswith('f '):
                idx = [int(tok.split('/')[0]) for tok in line.split()[1:]]
                if len(idx) != 3:
                    return None, None
                tris.append(idx)
    return np.asarray(verts, np.float64), np.asarray(tris, np.int64) - 1


LIST_KEYS = ('face_indices', 'location', 'apex', 'z_axis')
SCALAR_KEYS = ('radius', 'angle', 'max_radius', 'min_radius')


def read_surfaces(path: Path):
    """Per surface of an ABC feature file: type, face_indices ('faces') and the primitive's parameters
    (location, apex, z_axis, radius, angle, max_radius, min_radius), by a line scan. The lists are YAML
    flow sequences that wrap onto continuation lines, so a list is collected up to its closing bracket."""
    out, cur, inside, pending = [], None, False, None

    def finish(key, text):
        body = text.split('[', 1)[1].split(']', 1)[0].strip()
        dtype = np.int64 if key == 'face_indices' else np.float64
        cur['faces' if key == 'face_indices' else key] = (np.fromstring(body, sep=',', dtype=dtype) if body
                                                          else np.zeros(0, dtype))

    with open(path, 'r', encoding='utf8', errors='replace') as f:
        for line in f:
            if pending is not None:
                pending[1].append(line)
                if ']' in line:
                    finish(pending[0], ''.join(pending[1]))
                    pending = None
                continue
            if not line.startswith(' ') and not line.startswith('-'):
                inside = line.startswith('surfaces:')
                continue
            if not inside:
                continue
            if line.startswith('- '):
                if cur:
                    out.append(cur)
                cur = {'type': None, 'faces': None}
                line = '  ' + line[2:]
            m = re.match(r'  (\w+):\s*(.*)$', line)
            if not m or cur is None:
                continue
            key, value = m.group(1), m.group(2).strip()
            if key == 'type':
                cur['type'] = value
            elif key in LIST_KEYS and value.startswith('['):
                if ']' in value:
                    finish(key, value)
                else:
                    pending = (key, [value])
            elif key in SCALAR_KEYS and value:
                cur[key] = float(value)
    if cur:
        out.append(cur)
    return out


def srf_rows(surfaces):
    """fit_truth's .srf layout (export_fit_truth.py): 12 float64 per surface - stored kind, canonical
    kind (the same here: ABC's stored types), then surface_targets.py's params P(3) A(3) s6 s7."""
    rows = np.zeros((len(surfaces), 12), np.float64)
    for i, s in enumerate(surfaces):
        k = TYPE_NAMES.index(s['type']) if s['type'] in TYPE_NAMES else 8
        rows[i, 0] = rows[i, 1] = k
        loc = s.get('location', np.zeros(3))
        axis = s.get('z_axis', np.zeros(3))
        if k == 0:
            rows[i, 2:5], rows[i, 5:8] = loc, axis
        elif k == 1:
            rows[i, 2:5], rows[i, 5:8], rows[i, 8] = loc, axis, s.get('radius', 0.0)
        elif k == 2:
            rows[i, 2:5], rows[i, 5:8], rows[i, 8] = s.get('apex', loc), axis, s.get('angle', 0.0)
        elif k == 3:
            rows[i, 2:5], rows[i, 8] = loc, s.get('radius', 0.0)
        elif k == 4:
            rows[i, 2:5], rows[i, 5:8] = loc, axis
            rows[i, 8], rows[i, 9] = s.get('max_radius', 0.0), s.get('min_radius', 0.0)
    return rows


def truth_floor(rows, pos, fid):
    """Largest distance of each analytic face's mesh vertices from its true surface, / part diagonal:
    ~0 when the parameters and the mesh share a frame (Onshape's vertices lie on the surface)."""
    V = pos.reshape(-1, 3).astype(np.float64)
    diag = np.linalg.norm(V.max(0) - V.min(0))
    out = {}
    for f in np.unique(fid):
        k = int(rows[f, 0])
        if k > 4:
            continue
        X = V.reshape(-1, 3, 3)[fid == f].reshape(-1, 3)
        P, A = rows[f, 2:5], rows[f, 5:8] / max(np.linalg.norm(rows[f, 5:8]), 1e-300)
        v = X - P
        if k == 0:
            d = v @ A
        elif k == 3:
            d = np.linalg.norm(v, axis=1) - rows[f, 8]
        else:
            z = v @ A
            rho = np.linalg.norm(v - np.outer(z, A), axis=1)
            if k == 1:
                d = rho - rows[f, 8]
            elif k == 2:
                d = rho * np.cos(rows[f, 8]) - np.abs(z) * np.sin(rows[f, 8])
            else:
                d = np.hypot(rho - rows[f, 8], z) - rows[f, 9]
        out[int(f)] = (TYPE_NAMES[k], float(np.abs(d).max() / diag))
    return out


def write_freeform(feat: Path, rows, out: Path) -> None:
    """<part>.ffs.json: the exact definitions of the part's non-analytic faces (none: not written)."""
    if (rows[:, 0] > 4).any():
        out.write_text(json.dumps({str(k): v for k, v in read_freeform(feat).items()}))


def write_truth_only(args) -> int:
    """True surfaces for the models already exported to --out, with a check that each face's mesh
    vertices lie on its surface (the parameters and the OBJ share a frame)."""
    manifest = json.loads((args.out / 'manifest.json').read_text())
    floors = {}
    for entry in manifest['parts']:
        mid, suffix = entry['part'].split('_')
        feats = sorted((args.feat / mid).glob(f'*_features_{suffix}.yml'))
        rows = srf_rows(read_surfaces(feats[0]))
        rows.tofile(args.truth_out / f"{entry['part']}.srf")
        write_freeform(feats[0], rows, args.truth_out / f"{entry['part']}.ffs.json")
        stem = entry['variants']['onshape']['stem']
        pos = np.fromfile(args.out / f'{stem}.pos', np.float32)
        fid = np.fromfile(args.out / f'{stem}.fid', np.int32)
        for kind, fl in truth_floor(rows, pos, fid).values():
            floors.setdefault(kind, []).append(fl)
    for kind, fl in sorted(floors.items()):
        fl = np.asarray(fl)
        print(f'  {kind:9} faces {len(fl):5}  floor p50 {np.median(fl):.1e}  p99 {np.quantile(fl, 0.99):.1e}'
              f'  max {fl.max():.1e}  over 1e-4: {(fl > 1e-4).sum()}')
    print(f"true surfaces for {len(manifest['parts'])} models -> {args.truth_out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--obj', type=Path, required=True, help='extracted ABC obj chunk(s): <id>/<id>_<hash>_trimesh_NNN.obj')
    ap.add_argument('--feat', type=Path, default=Path('D:/abc-data/feat'))
    ap.add_argument('--bake', type=Path, default=Path('D:/abc-data/bake'))
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--max-tris', type=int, default=100_000)
    ap.add_argument('--min-tris', type=int, default=200)
    ap.add_argument('--seed', type=int, default=5)
    ap.add_argument('--truth-out', type=Path, default=Path('D:/abc-data/onshape_truth'),
                    help="true surfaces per model, <part>.srf in fit_truth's layout, for test/surfaceScore.mjs")
    ap.add_argument('--truth-only', action='store_true', help='(re)write the truth for the models already in --out')
    ap.add_argument('--freeform', action='store_true', help='only models with a non-analytic face')
    args = ap.parse_args()
    args.truth_out.mkdir(parents=True, exist_ok=True)
    if args.truth_only:
        return write_truth_only(args)

    _, train = val_split(args.bake)
    trained = {f.stem.split('_')[0] for f in train}
    # Stricter than the training split: no model that appears in ANY bake (every mesher, either
    # split) or in the evaluation and edge-training sets, since the edge models trained on
    # fill_eval parts and the mesher-diverse runs on other bakes' splits.
    seen = set(trained)
    for root in [*args.bake.parent.glob('bake*'), args.bake.parent / 'fill_eval', args.bake.parent / 'edge_train_set']:
        if root.is_dir():
            seen.update(f.name.split('_')[0] for f in root.rglob('*') if f.is_file())
    models = sorted(d for d in args.obj.iterdir() if d.is_dir())
    held = [d for d in models if d.name not in seen]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(held)
    print(f'{len(models):,} models in the OBJ chunk(s); {len(models) - len(held):,} appear in a bake or eval set; '
          f'{len(held):,} held out')

    args.out.mkdir(parents=True, exist_ok=True)
    manifest, skipped = [], {'no feature file': 0, 'unreadable': 0, 'size': 0, 'labels': 0, 'no analytic face': 0,
                             'no freeform face': 0}
    for d in held:
        if len(manifest) >= args.n:
            break
        obj = sorted(d.glob('*_trimesh_*.obj'))
        if not obj:
            skipped['unreadable'] += 1
            continue
        obj = obj[0]
        suffix = obj.stem.split('_trimesh_')[-1]
        feats = sorted((args.feat / d.name).glob(f'*_features_{suffix}.yml'))
        if not feats:
            skipped['no feature file'] += 1
            continue
        if args.freeform and not re.search(r'type: (Revolution|Extrusion|BSpline|Other)\s*$',
                                           feats[0].read_text(errors='replace').split('\nsurfaces:', 1)[-1], re.M):
            skipped['no freeform face'] += 1          # checked before the OBJ is read: most models have none
            continue
        verts, tris = read_obj(obj)
        if verts is None or len(tris) == 0:
            skipped['unreadable'] += 1
            continue
        if not (args.min_tris <= len(tris) <= args.max_tris):
            skipped['size'] += 1
            continue
        surfaces = read_surfaces(feats[0])
        fid = np.full(len(tris), -1, np.int64)
        typ = np.full(len(tris), 8, np.uint8)
        ok = True
        for i, s in enumerate(surfaces):
            f = s['faces']
            if f is None or (f.size and (f.min() < 0 or f.max() >= len(tris))) or (fid[f] >= 0).any():
                ok = False
                break
            fid[f] = i
            typ[f] = TYPE_NAMES.index(s['type']) if s['type'] in TYPE_NAMES else 8
        if not ok or (fid < 0).any():
            skipped['labels'] += 1
            continue
        if not args.freeform and not np.isin(typ, [0, 1, 2, 3, 4]).any():
            skipped['no analytic face'] += 1
            continue
        if args.freeform and not (typ > 4).any():
            skipped['no freeform face'] += 1
            continue
        rows = srf_rows(surfaces)
        rows.tofile(args.truth_out / f'{d.name}_{suffix}.srf')
        write_freeform(feats[0], rows, args.truth_out / f'{d.name}_{suffix}.ffs.json')
        stem = f'{d.name}_{suffix}_onshape'
        verts[tris].astype(np.float32).reshape(-1).tofile(args.out / f'{stem}.pos')
        fid.astype(np.int32).tofile(args.out / f'{stem}.fid')
        typ.tofile(args.out / f'{stem}.typ')
        manifest.append({'part': f'{d.name}_{suffix}', 'variants': {'onshape': {'stem': stem, 'tris': int(len(tris))}}})
    (args.out / 'manifest.json').write_text(json.dumps({'variants': ['onshape'], 'density': 'onshape',
                                                        'parts': manifest}, indent=1))
    print(f'exported {len(manifest)} held-out models -> {args.out}; skipped {skipped}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
