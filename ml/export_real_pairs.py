#!/usr/bin/env python3
"""
export_real_pairs.py - STEP + STL exports of the same part (a user's CAD system, e.g. Onshape) as a
labelled evaluation set in fillEval's format, with the true surface of every face.

    python export_real_pairs.py --dir ~/Downloads --names "Star,Fencepost,Cake Pan" \\
        --out D:/abc-data/twisty_eval --truth-out D:/abc-data/twisty_truth

Each <name>.step (or .stp) is paired with every <name>*.stl beside it (an exporter's density
settings); the variants are named by triangle count, most first: fine, medium, coarse (d3, d4, ...
past three). Written, like export_abc_obj_eval.py:

    <out>/<part>_<variant>.pos/.fid/.typ     triangles, STEP face per triangle, face type per triangle
    <out>/manifest.json
    <truth-out>/<part>.srf                   analytic faces' parameters (fit_truth layout)
    <truth-out>/<part>.ffs.json              every other face as an exact B-spline definition in ABC's
                                             layout (abc_freeform.truth_surface reads it): B-spline faces
                                             as stored, other kinds converted by OCCT over the face's own
                                             parameter range; "stored_type" keeps the STEP's kind

LABELS: an STL carries no face ids. Each triangle takes the face of the nearest sample of a fine
OCCT tessellation of the STEP (its nodes and triangle centroids), found on the GPU when there is one.
An exporter's triangles never cross faces and their vertices lie on the surface, so the nearest sample
is on the triangle's own face except within a sample spacing of an edge; the reported label distance
(p99, max) shows whether the tessellation was fine enough.

UNITS: the mesh is scaled to the STEP's units by the ratio of their bounding boxes, snapped to a
standard factor (mm, cm, m, inch); a pair whose boxes then disagree is skipped (as test/realPairs.mjs).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
from pathlib import Path

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepTools import BRepTools
from OCP.Bnd import Bnd_Box
from OCP.GeomAbs import (GeomAbs_BezierSurface, GeomAbs_BSplineSurface, GeomAbs_Cone, GeomAbs_Cylinder,
                         GeomAbs_Plane, GeomAbs_Sphere, GeomAbs_SurfaceOfExtrusion, GeomAbs_SurfaceOfRevolution,
                         GeomAbs_Torus)
from OCP.GeomConvert import GeomConvert
from OCP.Geom import Geom_BSplineSurface, Geom_RectangularTrimmedSurface
from OCP.STEPControl import STEPControl_Reader
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS

TYPE_NAMES = ['Plane', 'Cylinder', 'Cone', 'Sphere', 'Torus', 'Revolution', 'Extrusion', 'BSpline', 'Other']
KIND_OF = {GeomAbs_Plane: 0, GeomAbs_Cylinder: 1, GeomAbs_Cone: 2, GeomAbs_Sphere: 3, GeomAbs_Torus: 4,
           GeomAbs_SurfaceOfRevolution: 5, GeomAbs_SurfaceOfExtrusion: 6, GeomAbs_BSplineSurface: 7,
           GeomAbs_BezierSurface: 7}
UNIT_FACTORS = (1.0, 10.0, 25.4, 100.0, 1000.0, 0.1, 0.01, 0.001, 1 / 25.4, 0.0254)
DENSITY = ('fine', 'medium', 'coarse')
LABEL_BAND = 3                 # rings round a label boundary that are checked against the surfaces
LABEL_PASSES = 20              # band checks, each round the boundary as the last one moved it
LABEL_TOL = 1e-4              # x diag: a triangle whose vertices miss its face surface by more is re-labelled


def read_stl(path: Path):
    d = path.read_bytes()
    n = struct.unpack('<I', d[80:84])[0] if len(d) >= 84 else 0
    if len(d) == 84 + 50 * n:
        rec = np.frombuffer(d, dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]), count=n, offset=84)
        return rec['v'].astype(np.float64)
    pts = [tuple(map(float, m.groups())) for m in
           re.finditer(rb'vertex\s+(\S+)\s+(\S+)\s+(\S+)', d)]
    return np.asarray(pts, np.float64).reshape(-1, 3, 3)


def read_step(path: Path):
    r = STEPControl_Reader()
    if r.ReadFile(str(path)) != 1:
        raise RuntimeError(f'cannot read {path}')
    r.TransferRoots()
    shape = r.OneShape()
    faces = []
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        faces.append(TopoDS.Face_s(ex.Current()))
        ex.Next()
    return shape, faces


def bbox(shape):
    b = Bnd_Box()
    BRepBndLib.Add_s(shape, b)
    x0, y0, z0, x1, y1, z1 = b.Get()
    return np.array([x0, y0, z0]), np.array([x1, y1, z1])


def samples(shape, faces, deflection):
    """Nodes and triangle centroids of a fine tessellation, with each one's face index."""
    BRepMesh_IncrementalMesh(shape, deflection, False, 0.1, True)
    P, L = [], []
    for i, f in enumerate(faces):
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(f, loc)
        if tri is None:
            continue
        tr = loc.Transformation()
        nodes = np.array([[q.X(), q.Y(), q.Z()] for q in
                          (tri.Node(k).Transformed(tr) for k in range(1, tri.NbNodes() + 1))])
        tris = np.array([tri.Triangle(k).Get() for k in range(1, tri.NbTriangles() + 1)]) - 1
        pts = np.concatenate([nodes, nodes[tris].mean(1)])
        P.append(pts)
        L.append(np.full(len(pts), i))
    return np.concatenate(P), np.concatenate(L)


def nearest(Q, P, k=8, chunk=2048):
    """Indices of the k nearest rows of P for every row of Q, nearest first (GPU when available)."""
    try:
        import torch
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        Pt = torch.as_tensor(P, dtype=torch.float64, device=dev)
        idx = []
        for s in range(0, len(Q), chunk):
            d = torch.cdist(torch.as_tensor(Q[s:s + chunk], dtype=torch.float64, device=dev), Pt)
            idx.append(torch.topk(d, min(k, len(P)), dim=1, largest=False).indices.cpu().numpy())
        return np.concatenate(idx)
    except ImportError:
        idx = np.empty((len(Q), min(k, len(P))), np.int64)
        pp = (P * P).sum(1)
        for s in range(0, len(Q), 256):
            d = pp[None, :] - 2 * Q[s:s + 256] @ P.T
            part = np.argpartition(d, min(k, len(P)) - 1, axis=1)[:, :k]
            idx[s:s + 256] = np.take_along_axis(part, np.argsort(np.take_along_axis(d, part, 1), 1), 1)
        return idx


def surface_distance(rows, ffs, i, X):
    """Distance of points X to face i's (untrimmed) true surface."""
    import abc_freeform
    import solid_rebuild as sr
    k = int(rows[i, 0])
    if k <= 4:
        kind = sr.KIND[k]
        return np.abs(sr.dist_normal(kind, sr.truth_params(kind, rows[i, 2:], X.mean(0)), X)[0])
    s = abc_freeform.truth_surface(ffs[str(i)])
    _, _, Q = s.closest(X)
    return np.linalg.norm(Q - X, axis=1)


def label(X, P, L, rows, ffs):
    """Each triangle's face: the face of its centroid's nearest tessellation sample, and where the 8
    nearest samples disagree, the candidate whose true surface its three vertices lie closest to (an
    exporter's vertices are ON their face's surface; a centroid near an edge can be nearer the other
    face's samples)."""
    cand = L[nearest(X.mean(1), P)]
    fid = cand[:, 0].copy()
    amb = np.nonzero((cand != cand[:, :1]).any(1))[0]
    if len(amb):
        best = np.full(len(amb), np.inf)
        for f in np.unique(cand[amb]):
            has = (cand[amb] == f).any(1)
            sel = amb[has]
            d = surface_distance(rows, ffs, int(f), X[sel].reshape(-1, 3)).reshape(-1, 3).max(1)
            better = d < best[has]
            upd = np.nonzero(has)[0][better]
            best[upd] = d[better]
            fid[amb[upd]] = f
    # Every label checked against its own surface; a triangle whose vertices miss it (by more than a
    # real exporter's floor: Cake Pan's vertices sit 2e-5 x diag off its STEP surfaces) goes to the face
    # that fits it best among ALL faces (real parts have few). On the twisted ornament all 8 nearest
    # samples of 37 of one face's 612 triangles were a neighbour's, 6e-3 x diag off it.
    # Only triangles within LABEL_BAND rings of a label boundary are checked: a wrong label is a band
    # of a neighbour's triangles along the edge, and a full check projects millions of vertices.
    # Repeated until nothing changes: on a fine mesh the wrong band is wider than LABEL_BAND rings
    # (the ornament's 683k-triangle export), and each pass moves the boundary into it.
    diag = float(np.linalg.norm(X.reshape(-1, 3).max(0) - X.reshape(-1, 3).min(0)))
    checked, relabelled = np.zeros(len(X), bool), 0
    for _ in range(LABEL_PASSES):
        near = boundary_band(X, fid, LABEL_BAND) & ~checked
        off = np.zeros(len(X), bool)
        for f in np.unique(fid[near]):
            sel = np.nonzero(near & (fid == f))[0]
            off[sel] = surface_distance(rows, ffs, int(f), X[sel].reshape(-1, 3)).reshape(-1, 3).max(1) > LABEL_TOL * diag
        checked |= near & ~off
        bad = np.nonzero(off)[0]
        if not len(bad):
            break
        D = np.stack([surface_distance(rows, ffs, f, X[bad].reshape(-1, 3)).reshape(-1, 3).max(1) for f in range(len(rows))])
        fid[bad] = np.argmin(D, 0)
        checked[bad] = True
        relabelled += len(bad)
    return fid.astype(np.int32), len(amb) + relabelled


def boundary_band(X, fid, rings):
    """Triangles within `rings` edge-steps of a triangle with another label."""
    V, inv = np.unique(np.round(X.reshape(-1, 3), 9), axis=0, return_inverse=True)
    T = inv.reshape(-1, 3)
    e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    e.sort(1)
    t = np.tile(np.arange(len(T)), 3)
    order = np.lexsort((e[:, 1], e[:, 0]))
    e, t = e[order], t[order]
    same = (e[1:] == e[:-1]).all(1)
    a, b = t[:-1][same], t[1:][same]                   # triangle pairs across a shared edge
    band = np.zeros(len(T), bool)
    diff = fid[a] != fid[b]
    band[a[diff]] = band[b[diff]] = True
    for _ in range(rings - 1):
        grow = band.copy()
        grow[a[band[b]]] = True
        grow[b[band[a]]] = True
        band = grow
    return band


def floors(X, fid, rows, ffs, diag, cap=5000, seed=0):
    """Largest distance of each face's triangle vertices from its true surface, x diag."""
    rng = np.random.default_rng(seed)
    out = {}
    for f in np.unique(fid):
        V = np.unique(X[fid == f].reshape(-1, 3), axis=0)
        if len(V) > cap:
            V = V[rng.choice(len(V), cap, replace=False)]
        out[int(f)] = float(surface_distance(rows, ffs, int(f), V).max() / diag)
    return out


def srf_row(face):
    """fit_truth's 12 floats for one face: stored kind, canonical kind, then the primitive's
    parameters as export_abc_obj_eval.srf_rows lays them out."""
    ad = BRepAdaptor_Surface(face)
    k = KIND_OF.get(ad.GetType(), 8)
    row = np.zeros(12)
    row[0] = row[1] = k
    xyz = lambda p: [p.X(), p.Y(), p.Z()]
    if k == 0:
        ax = ad.Plane().Position()
        row[2:5], row[5:8] = xyz(ax.Location()), xyz(ax.Direction())
    elif k == 1:
        c = ad.Cylinder()
        row[2:5], row[5:8], row[8] = xyz(c.Location()), xyz(c.Axis().Direction()), c.Radius()
    elif k == 2:
        c = ad.Cone()
        row[2:5], row[5:8], row[8] = xyz(c.Apex()), xyz(c.Axis().Direction()), c.SemiAngle()
    elif k == 3:
        s = ad.Sphere()
        row[2:5], row[8] = xyz(s.Location()), s.Radius()
    elif k == 4:
        t = ad.Torus()
        row[2:5], row[5:8] = xyz(t.Location()), xyz(t.Axis().Direction())
        row[8], row[9] = t.MajorRadius(), t.MinorRadius()
    return row


def bspline_def(face):
    """The face's surface as an exact B-spline in ABC's layout (flat knots, non-periodic), over the
    face's own parameter range for surfaces that are not B-splines already."""
    ad = BRepAdaptor_Surface(face)
    kind = TYPE_NAMES[KIND_OF.get(ad.GetType(), 8)]
    if ad.GetType() == GeomAbs_BSplineSurface:
        bs = ad.BSpline()                                 # with the face's location applied
    else:
        loc = TopLoc_Location()
        surf = BRep_Tool.Surface_s(face, loc)
        u0, u1, v0, v1 = BRepTools.UVBounds_s(face)
        bs = GeomConvert.SurfaceToBSplineSurface_s(Geom_RectangularTrimmedSurface(surf, u0, u1, v0, v1))
        if not loc.IsIdentity():
            bs.Transform(loc.Transformation())
    bs = bs.Copy()                                        # OCP returns the derived type
    if bs.IsUPeriodic():
        bs.SetUNotPeriodic()
    if bs.IsVPeriodic():
        bs.SetVNotPeriodic()
    nu, nv = bs.NbUPoles(), bs.NbVPoles()
    poles = [[[bs.Pole(i, j).X(), bs.Pole(i, j).Y(), bs.Pole(i, j).Z()] for j in range(1, nv + 1)] for i in range(1, nu + 1)]
    weights = [[bs.Weight(i, j) for j in range(1, nv + 1)] for i in range(1, nu + 1)]
    tu = bs.UKnotSequence()
    tv = bs.VKnotSequence()
    rational = bs.IsURational() or bs.IsVRational()
    return {'type': 'BSpline', 'stored_type': kind, 'u_degree': bs.UDegree(), 'v_degree': bs.VDegree(),
            'u_knots': [tu.Value(i) for i in range(tu.Lower(), tu.Upper() + 1)],
            'v_knots': [tv.Value(i) for i in range(tv.Lower(), tv.Upper() + 1)],
            'poles': poles, 'weights': weights, 'u_rational': rational, 'v_rational': rational,
            'u_closed': bs.IsUClosed(), 'v_closed': bs.IsVClosed()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', type=Path, required=True)
    ap.add_argument('--names', required=True, help='comma list of part names (<name>.step + <name>*.stl)')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--truth-out', type=Path, required=True)
    ap.add_argument('--deflection', type=float, default=2e-4, help='x the part diagonal, for the labelling tessellation')
    ap.add_argument('--merge', action='store_true', help="keep the manifest's other parts (re-export some names only)")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    args.truth_out.mkdir(parents=True, exist_ok=True)
    manifest = []
    if args.merge and (args.out / 'manifest.json').exists():
        redo = {re.sub(r'[^0-9a-z]+', '_', n.strip().lower()).strip('_') for n in args.names.split(',')}
        manifest = [e for e in json.loads((args.out / 'manifest.json').read_text())['parts'] if e['part'] not in redo]
    for name in [n.strip() for n in args.names.split(',') if n.strip()]:
        step = next((args.dir / f'{name}{ext}' for ext in ('.step', '.stp', '.STEP', '.STP')
                     if (args.dir / f'{name}{ext}').exists()), None)
        meshes = [p for p in args.dir.glob(f'{name}*.stl') if re.fullmatch(re.escape(name) + r'( \(\d+\))?\.stl', p.name)]
        if step is None or not meshes:
            print(f'{name}: no STEP / STL pair, skipped')
            continue
        part = re.sub(r'[^0-9a-z]+', '_', name.lower()).strip('_')
        shape, faces = read_step(step)
        lo, hi = bbox(shape)                              # before meshing: from the control nets, a
        P, L = samples(shape, faces, args.deflection * float(np.linalg.norm(hi - lo)))
        lo, hi = P.min(0), P.max(0)                       # B-spline part's box is too big (Cake Pan 2.7%)
        diag = float(np.linalg.norm(hi - lo))
        rows = np.stack([srf_row(f) for f in faces])
        rows.tofile(args.truth_out / f'{part}.srf')
        ffs = {str(i): bspline_def(f) for i, f in enumerate(faces) if rows[i, 0] > 4}
        (args.truth_out / f'{part}.ffs.json').write_text(json.dumps(ffs))
        tris = [(read_stl(m), m) for m in meshes]
        tris.sort(key=lambda z: -len(z[0]))
        variants = {}
        for k, (X, m) in enumerate(tris):
            # units: snap the bounding-box ratio to a standard factor, then check the boxes agree
            mlo, mhi = X.reshape(-1, 3).min(0), X.reshape(-1, 3).max(0)
            ratio = float(np.linalg.norm(hi - lo) / max(np.linalg.norm(mhi - mlo), 1e-300))
            f = min(UNIT_FACTORS, key=lambda u: abs(math.log(ratio / u)))
            X = X * f
            if np.abs(X.reshape(-1, 3).min(0) - lo).max() > 0.01 * diag or np.abs(X.reshape(-1, 3).max(0) - hi).max() > 0.01 * diag:
                print(f'  {m.name}: boxes disagree after x{f:g} (ratio {ratio:.4f}), skipped')
                continue
            fid, n_amb = label(X, P, L, rows, ffs)
            variant = DENSITY[k] if k < len(DENSITY) else f'd{k}'
            stem = f'{part}_{variant}'
            X.astype(np.float32).reshape(-1).tofile(args.out / f'{stem}.pos')
            fid.tofile(args.out / f'{stem}.fid')
            rows[fid, 0].astype(np.uint8).tofile(args.out / f'{stem}.typ')
            fl = floors(X, fid, rows, ffs, diag)
            variants[variant] = {'stem': stem, 'tris': int(len(X)), 'source': m.name, 'scale': f,
                                 'floor_max': max(fl.values())}
            worst = sorted(fl.items(), key=lambda z: -z[1])[:3]
            print(f'  {stem:32} {len(X):7} tris from {m.name!r} x{f:g}; faces labelled {len(np.unique(fid))}/{len(faces)}'
                  f' ({n_amb} decided by surface distance); vertex-to-truth floor max {max(fl.values()):.1e} x diag,'
                  f' worst faces {[(a, f"{b:.1e}") for a, b in worst]}')
        kinds = {TYPE_NAMES[int(k)]: int((rows[:, 0] == k).sum()) for k in np.unique(rows[:, 0])}
        print(f'{name}: {len(faces)} faces {kinds}, {len(ffs)} freeform definitions')
        if variants:
            manifest.append({'part': part, 'variants': variants})
    (args.out / 'manifest.json').write_text(json.dumps({'variants': list(DENSITY), 'density': 'real-export',
                                                        'parts': manifest}, indent=1))
    print(f'{len(manifest)} parts -> {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
