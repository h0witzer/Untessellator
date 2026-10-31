#!/usr/bin/env python3
"""
step_match.py - does an exported STEP reproduce the original CAD part, face for face?

    python step_match.py <ours.step> <truth.step> [--json out.json] [--samples 200]

A part MATCHES when all of these hold:

    ours has as many solids as the original, and is valid with no free edges
    ours has exactly as many faces as the original, and of every surface type exactly as many
    the faces pair up one-to-one, each with a face of the same type
    every pair is within --tol of each other, both ways, measured on the TRIMMED faces
    the volumes differ by no more than a --tol thick skin over the original's whole surface

--tol defaults to 1e-3 mm because that is what the originals declare: ABC's Onshape STEP files carry
UNCERTAINTY_MEASURE_WITH_UNIT(1e-6) in metres. So a match is within the original file's own modelling
uncertainty everywhere.

Pairing: faces of one type are paired greedily by centroid distance plus area difference, both scaled to
the part. The pair distance is two-sided. Points are sampled on face A (its triangulation's nodes, which
include the boundary), and each point's distance to the trimmed face B is measured
(BRepExtrema_DistShapeShape). Then the same is done from B to A, and the larger maximum is kept. So a
right surface with a wrong trim fails, and so does a right trim on a slightly wrong surface.

It also reports, per pair, the change in the size parameters (radius, cone half-angle, torus radii) and
in the axis or normal direction. It also compares edge and vertex counts, but they don't gate the match.

Both files are read in millimetres (OCCT's default unit), whatever unit they were written in.
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.GeomAbs import (GeomAbs_BSplineSurface, GeomAbs_Cone, GeomAbs_Cylinder, GeomAbs_Plane, GeomAbs_Sphere,
                         GeomAbs_SurfaceOfExtrusion, GeomAbs_SurfaceOfRevolution, GeomAbs_Torus)
from OCP.GeomAdaptor import GeomAdaptor_Surface
from OCP.GProp import GProp_GProps
from OCP.gp import gp_Pnt
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_Reader
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID, TopAbs_VERTEX
from OCP.TopExp import TopExp
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS
from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape, TopTools_IndexedMapOfShape

VERSION = 2          # bump when a change can alter a verdict, so cached reports are redone

KIND = {GeomAbs_Plane: 'plane', GeomAbs_Cylinder: 'cylinder', GeomAbs_Cone: 'cone', GeomAbs_Sphere: 'sphere',
        GeomAbs_Torus: 'torus', GeomAbs_BSplineSurface: 'bspline', GeomAbs_SurfaceOfRevolution: 'revolution',
        GeomAbs_SurfaceOfExtrusion: 'extrusion'}


def read_step(path):
    r = STEPControl_Reader()
    if r.ReadFile(str(path)) != IFSelect_RetDone:
        return None
    r.TransferRoots()
    return r.OneShape()


def shape_map(shape, kind):
    m = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, kind, m)
    return [m.FindKey(i) for i in range(1, m.Extent() + 1)]


def free_edges(shape):
    """Edges with one face (seam edges count their face twice, degenerate edges are skipped)."""
    m = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, m)
    n = 0
    for i in range(1, m.Extent() + 1):
        if m.FindFromIndex(i).Extent() == 1 and not BRep_Tool.Degenerated_s(TopoDS.Edge_s(m.FindKey(i))):
            n += 1
    return n


def surface_params(face):
    """Kind, unit axis or normal (None for a sphere) and size parameters of the face's surface."""
    s = GeomAdaptor_Surface(BRep_Tool.Surface_s(face))
    k = KIND.get(s.GetType(), 'other')
    d = lambda ax: np.array([ax.Direction().X(), ax.Direction().Y(), ax.Direction().Z()])
    if k == 'plane':
        return k, d(s.Plane().Axis()), {}
    if k == 'cylinder':
        c = s.Cylinder()
        return k, d(c.Axis()), {'radius': c.Radius()}
    if k == 'cone':
        c = s.Cone()
        return k, d(c.Axis()), {'half_angle_deg': math.degrees(abs(c.SemiAngle()))}
    if k == 'sphere':
        return k, None, {'radius': s.Sphere().Radius()}
    if k == 'torus':
        t = s.Torus()
        return k, d(t.Axis()), {'major_radius': t.MajorRadius(), 'minor_radius': t.MinorRadius()}
    return k, None, {}


def face_samples(face, cap, rng):
    """The face's own vertices (its corners, where trims disagree most), then up to `cap` nodes of its
    triangulation, which include the boundary."""
    corners = [BRep_Tool.Pnt_s(TopoDS.Vertex_s(v)) for v in shape_map(face, TopAbs_VERTEX)]
    corners = np.array([[p.X(), p.Y(), p.Z()] for p in corners]).reshape(-1, 3)
    loc = TopLoc_Location()
    tri = BRep_Tool.Triangulation_s(face, loc)
    if tri is None:
        return corners
    tr = loc.Transformation()
    pts = np.array([[p.X(), p.Y(), p.Z()] for p in
                    (tri.Node(i).Transformed(tr) for i in range(1, tri.NbNodes() + 1))])
    if len(pts) > cap:
        pts = pts[rng.choice(len(pts), cap, replace=False)]
    return np.concatenate([corners, pts])


def describe(shape, diag_hint=None, cap=200, seed=0):
    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    x0, y0, z0, x1, y1, z1 = box.Get()
    diag = math.dist((x0, y0, z0), (x1, y1, z1))
    BRepMesh_IncrementalMesh(shape, 2e-3 * (diag_hint or diag), False, 0.2, True)
    rng = np.random.default_rng(seed)
    faces = []
    for f in shape_map(shape, TopAbs_FACE):
        f = TopoDS.Face_s(f)
        g = GProp_GProps()
        BRepGProp.SurfaceProperties_s(f, g)
        c = g.CentreOfMass()
        kind, axis, size = surface_params(f)
        faces.append({'face': f, 'kind': kind, 'axis': axis, 'size': size, 'area': g.Mass(),
                      'centroid': np.array([c.X(), c.Y(), c.Z()]), 'samples': face_samples(f, cap, rng)})
    g = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, g)
    return {'shape': shape, 'diag': diag, 'faces': faces, 'volume': abs(g.Mass()),
            'solids': len(shape_map(shape, TopAbs_SOLID)), 'edges': len(shape_map(shape, TopAbs_EDGE)),
            'vertices': len(shape_map(shape, TopAbs_VERTEX)), 'valid': bool(BRepCheck_Analyzer(shape).IsValid()),
            'free_edges': free_edges(shape)}


def pair_faces(a, b, diag):
    """Greedy one-to-one pairing within each surface type, cheapest first. Returns [(i, j)]."""
    pairs = []
    for kind in sorted({f['kind'] for f in a}):
        ia = [i for i, f in enumerate(a) if f['kind'] == kind]
        ib = [j for j, f in enumerate(b) if f['kind'] == kind]
        if not ia or not ib:
            continue
        ca = np.array([a[i]['centroid'] for i in ia])
        cb = np.array([b[j]['centroid'] for j in ib])
        aa = np.array([a[i]['area'] for i in ia])
        ab = np.array([b[j]['area'] for j in ib])
        cost = (np.linalg.norm(ca[:, None] - cb[None], axis=2) / diag
                + np.abs(aa[:, None] - ab[None]) / np.maximum(np.maximum(aa[:, None], ab[None]), 1e-30))
        used_a, used_b = set(), set()
        for flat in np.argsort(cost, axis=None):
            r, c = divmod(int(flat), len(ib))
            if r in used_a or c in used_b:
                continue
            used_a.add(r)
            used_b.add(c)
            pairs.append((ia[r], ib[c]))
            if len(used_a) == len(ia) or len(used_b) == len(ib):
                break
    return pairs


def one_way(points, face):
    worst = 0.0
    for p in points:
        v = BRepBuilderAPI_MakeVertex(gp_Pnt(*map(float, p))).Vertex()
        d = BRepExtrema_DistShapeShape(v, face)
        if not d.IsDone():
            return math.inf
        worst = max(worst, d.Value())
    return worst


def size_delta(fa, fb):
    out = {}
    for key, va in fa['size'].items():
        vb = fb['size'].get(key)
        if vb is not None:
            out[key] = vb - va
    if fa['axis'] is not None and fb['axis'] is not None:
        c = abs(float(np.dot(fa['axis'], fb['axis'])))
        out['axis_deg'] = math.degrees(math.acos(min(1.0, c)))
    return out


def match(ours_path, truth_path, tol=1e-3, cap=200):
    truth_shape, ours_shape = read_step(truth_path), read_step(ours_path)
    if truth_shape is None:
        return {'match': False, 'reason': 'original unreadable'}
    if ours_shape is None:
        return {'match': False, 'reason': 'export unreadable'}
    t = describe(truth_shape, cap=cap)
    o = describe(ours_shape, diag_hint=t['diag'], cap=cap)
    count = lambda d: {k: sum(f['kind'] == k for f in d['faces']) for k in sorted({f['kind'] for f in d['faces']})}
    rep = {
        'diag_mm': t['diag'],
        'truth': {'faces': len(t['faces']), 'types': count(t), 'edges': t['edges'], 'vertices': t['vertices'],
                  'solids': t['solids'], 'volume_mm3': t['volume']},
        'ours': {'faces': len(o['faces']), 'types': count(o), 'edges': o['edges'], 'vertices': o['vertices'],
                 'solids': o['solids'], 'valid': o['valid'], 'free_edges': o['free_edges'], 'volume_mm3': o['volume']},
        'volume_rel_err': abs(o['volume'] - t['volume']) / t['volume'] if t['volume'] else math.inf,
    }
    vol_tol = sum(f['area'] for f in t['faces']) * tol / t['volume'] if t['volume'] else 0.0
    rep['topology_identical'] = (rep['truth']['faces'], rep['truth']['edges'], rep['truth']['vertices']) == \
                                (rep['ours']['faces'], rep['ours']['edges'], rep['ours']['vertices'])
    reasons = []
    if not t['solids']:
        reasons.append('original has no solid')
    elif o['solids'] != t['solids'] or not o['valid'] or o['free_edges']:
        reasons.append(f"export is not {t['solids']} valid closed solid{'s' if t['solids'] > 1 else ''}")
    if rep['truth']['types'] != rep['ours']['types']:
        reasons.append('face types differ')
    if not reasons:
        pairs = pair_faces(t['faces'], o['faces'], t['diag'])
        worst, per_face, size_err = 0.0, [], {}
        for i, j in pairs:
            ft, fo = t['faces'][i], o['faces'][j]
            d = max(one_way(ft['samples'], fo['face']), one_way(fo['samples'], ft['face']))
            worst = max(worst, d)
            delta = size_delta(ft, fo)
            for key, v in delta.items():
                size_err[key] = max(size_err.get(key, 0.0), abs(v))
            per_face.append({'truth_face': i, 'our_face': j, 'kind': ft['kind'], 'area_mm2': ft['area'],
                             'dev_mm': d, 'truth': ft['size'], 'delta': delta})
            if d > tol:
                break                                       # one face over is enough to fail
        rep['pairs'] = len(pairs)
        rep['max_dev_mm'] = worst
        rep['max_param_err'] = size_err
        rep['faces'] = per_face
        if len(pairs) != len(t['faces']):
            reasons.append('faces do not pair one-to-one')
        elif worst > tol:
            reasons.append(f'a face is {worst:.3g} mm off')
        if rep['volume_rel_err'] > vol_tol:
            reasons.append(f"volume differs by {rep['volume_rel_err']:.3g}")
    rep['tol_mm'], rep['vol_tol'], rep['version'] = tol, vol_tol, VERSION
    rep['match'] = not reasons
    rep['reason'] = '; '.join(reasons)
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('ours')
    ap.add_argument('truth')
    ap.add_argument('--json', help='write the full report here')
    ap.add_argument('--tol', type=float, default=1e-3, help='largest face-to-face distance in mm')
    ap.add_argument('--samples', type=int, default=200, help='sample points per face, each way')
    args = ap.parse_args(argv)
    rep = match(args.ours, args.truth, args.tol, args.samples)
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(rep, f, indent=1)
    t, o = rep.get('truth', {}), rep.get('ours', {})
    print(f"{'MATCH' if rep['match'] else 'no match'}  faces {o.get('faces')}/{t.get('faces')}  "
          f"max dev {rep.get('max_dev_mm', float('nan')):.3g} mm  volume {rep.get('volume_rel_err', float('nan')):.2g}"
          f"{'  ' + rep['reason'] if rep['reason'] else ''}")
    return 0 if rep['match'] else 1


if __name__ == '__main__':
    sys.exit(main())
