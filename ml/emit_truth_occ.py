#!/usr/bin/env python3
"""
emit_truth_occ.py - rebuild the STEP faces js/brepBuilder.js would export, with OCCT, and score
where they land against the TRUE surfaces.

Reads the rows test/emitTruth.mjs writes (labelled true faces, their fits through the export
adapter, and brepBuilder's computeAnalyticalBoundaries extents). Each curved face is built the way
brepBuilder._buildLargePatch builds it - same surface constructor, same UV arguments - and sampled
on a grid over its UV bounds. Faces are built from brepBuilder.patchBounds (the row's bound.patch),
with the same frame, surface constructor and UV range brepBuilder passes to OCCT:

    dev     max distance of face samples from the true surface / part diagonal
    excess  share of face samples farther than 2 median edges from the true patch's TRIANGLES
    axIoU   cylinders and cones: emitted vs true axial range, along the TRUE axis

Planes are rebuilt twice from the row: the OLD face (the single loop brepBuilder._orderBoundaryLoop
walked, filled) and the NEW one (planeBoundary: outer wire plus holes). Both are meshed by OCCT and
scored by area: `off` is the share of emitted area farther than the tolerance from the true patch,
`miss` the share of the true patch's area farther than that from the emitted face.

Usage (OCCT venv):
    D:/abc-data/venv/Scripts/python.exe emit_truth_occ.py rows.json
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.TopLoc import TopLoc_Location
from OCP.BRepTools import BRepTools
from OCP.gp import gp_Ax3, gp_Cone, gp_Cylinder, gp_Dir, gp_Pln, gp_Pnt, gp_Sphere, gp_Torus

GRID = 32


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def truth_distance(kind, s, X, centre):
    X = np.asarray(X, float)
    P = np.array(s[0:3])
    a = unit(s[3:6]) if kind != "sphere" else None
    v = X - P
    if kind == "plane":
        return np.abs(v @ a)
    if kind == "sphere":
        return np.abs(np.linalg.norm(v, axis=1) - s[6])
    z = v @ a
    rho = np.sqrt(np.maximum(0, (v * v).sum(1) - z * z))
    if kind == "cylinder":
        return np.abs(rho - s[6])
    if kind == "torus":
        return np.abs(np.hypot(rho - s[6], z) - s[7])
    if (np.asarray(centre) - P) @ a < 0:
        a = -a
    return np.abs(rho * math.cos(s[6]) - (v @ a) * math.sin(s[6]))


def build_face(row):
    """The face js/brepBuilder.js builds for this row (from its patchBounds), or None where it builds none."""
    kind, p, b = row["kind"], row["fit"], (row["bound"] or {}).get("patch")
    if not b or kind not in ("cylinder", "cone", "sphere", "torus"):
        return None
    ax3 = gp_Ax3(gp_Pnt(*b["origin"]), gp_Dir(*unit(b["axis"])), gp_Dir(*unit(b["xDir"])))
    if kind == "cylinder":
        surf = gp_Cylinder(ax3, p["radius"])
    elif kind == "cone":
        surf = gp_Cone(ax3, p["halfAngle"], 0.0)
    elif kind == "sphere":
        surf = gp_Sphere(ax3, p["radius"])
    else:
        surf = gp_Torus(ax3, p["majorRadius"], p["minorRadius"])
    mk = BRepBuilderAPI_MakeFace(surf, b["umin"], b["umax"], b["vmin"], b["vmax"])
    return mk.Face() if mk.IsDone() else None


def point_triangle_distance(X, T, chunk=256):
    """Min distance from each point X (n,3) to the triangles T (m,9): exact closest-point test."""
    A, B, C = T[:, 0:3], T[:, 3:6], T[:, 6:9]
    AB, AC = B - A, C - A
    N = np.cross(AB, AC)
    NN = (N * N).sum(1)
    out = np.empty(len(X))
    for s in range(0, len(X), chunk):
        P = X[s:s + chunk, None, :]                               # (k,1,3)
        AP = P - A
        # barycentrics of the projection onto each triangle's plane
        inv = np.where(NN > 0, 1.0 / np.where(NN > 0, NN, 1.0), 0.0)
        v = (np.cross(AP, AC) * N).sum(-1) * inv
        w = (np.cross(AB, AP) * N).sum(-1) * inv
        inside = (v >= 0) & (w >= 0) & (v + w <= 1) & (NN > 0)
        plane = np.abs((AP * N).sum(-1)) * np.sqrt(inv)
        def seg(P0, D):
            t = np.clip(((P - P0) * D).sum(-1) / np.maximum((D * D).sum(-1), 1e-300), 0, 1)
            return np.linalg.norm(P - (P0 + t[..., None] * D), axis=-1)
        edge = np.minimum(np.minimum(seg(A, AB), seg(A, AC)), seg(B, C - B))
        out[s:s + chunk] = np.where(inside, plane, edge).min(1)
    return out


def sample(face):
    ad = BRepAdaptor_Surface(face)
    u0, u1, v0, v1 = BRepTools.UVBounds_s(face)
    pts = []
    for u in np.linspace(u0, u1, GRID):
        for v in np.linspace(v0, v1, GRID):
            p = ad.Value(float(u), float(v))
            pts.append((p.X(), p.Y(), p.Z()))
    return np.array(pts)


def polygon_wire(pts):
    mp = BRepBuilderAPI_MakePolygon()
    for p in pts:
        mp.Add(gp_Pnt(*p))
    mp.Close()
    return mp.Wire() if mp.IsDone() else None


def plane_face(row, old):
    p = row["fit"]
    n = unit(p["normal"])
    o = np.asarray(p["origin"], float)
    snap = lambda q: list(np.asarray(q, float) - ((np.asarray(q, float) - o) @ n) * n)
    if old:
        if not row.get("oldLoop"):
            return None
        outer, holes = [snap(q) for q in row["oldLoop"]], []
    else:
        b = row["bound"] or {}
        if not b.get("outer"):
            return None
        outer, holes = b["outer"], b.get("holes") or []
    w = polygon_wire(outer)
    if w is None:
        return None
    mk = BRepBuilderAPI_MakeFace(gp_Pln(gp_Pnt(*o), gp_Dir(*n)), w, True)
    for h in holes:
        hw = polygon_wire(h)
        if hw is not None:
            mk.Add(hw)
    return mk.Face() if mk.IsDone() else None


def face_mesh(face, deflection):
    """OCCT's own triangulation of a face: (m,9) triangles."""
    BRepMesh_IncrementalMesh(face, deflection)
    loc = TopLoc_Location()
    tri = BRep_Tool.Triangulation_s(face, loc)
    if tri is None:
        return None
    nodes = np.array([[tri.Node(i).X(), tri.Node(i).Y(), tri.Node(i).Z()] for i in range(1, tri.NbNodes() + 1)])
    idx = np.array([tri.Triangle(i).Get() for i in range(1, tri.NbTriangles() + 1)]) - 1
    return nodes[idx].reshape(-1, 9)


def area_scores(E, T, tol):
    """off: share of emitted area away from the true patch; miss: share of true area not emitted."""
    def cen_area(X):
        A, B, C = X[:, 0:3], X[:, 3:6], X[:, 6:9]
        return (A + B + C) / 3, 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)
    ce, ae = cen_area(E)
    ct, at = cen_area(T)
    off = float(ae[point_triangle_distance(ce, T) > tol].sum() / max(ae.sum(), 1e-300))
    miss = float(at[point_triangle_distance(ct, E) > tol].sum() / max(at.sum(), 1e-300))
    return off, miss


def main(argv):
    rows = json.load(open(argv[1]))
    stats = defaultdict(lambda: defaultdict(list))
    for row in rows:
        kind = row["kind"]
        if row["floor"] >= 1e-4:
            continue
        if kind == "plane":
            if not row["emitted"]:
                continue
            T = np.asarray(row["tris"], float)
            for label in ("old", "new"):
                st = stats[(row["mesher"], "plane " + label)]
                st["n"].append(1)
                face = plane_face(row, old=(label == "old"))
                E = face_mesh(face, row["tol"]) if face is not None else None
                if E is None or not len(E):
                    st["missing"].append(1)
                    continue
                off, miss = area_scores(E, T, row["tol"])
                st["off"].append(off)
                st["miss"].append(miss)
            continue
        key = (row["mesher"], kind)
        st = stats[key]
        st["n"].append(1)
        face = build_face(row) if row["emitted"] else None
        if face is None:
            st["missing"].append(1)
            continue
        X = sample(face)
        V = np.asarray(row["verts"], float)
        st["dev"].append(float(truth_distance(kind, row["truth"], X, row["centre"]).max()) / row["diag"])
        near = point_triangle_distance(X, np.asarray(row["tris"], float))
        st["excess"].append(float((near > row["tol"]).mean()))
        if kind in ("cylinder", "cone"):
            a = unit(row["truth"][3:6])
            e, t = X @ a, V @ a
            inter = max(0.0, min(e.max(), t.max()) - max(e.min(), t.min()))
            uni = max(e.max(), t.max()) - min(e.min(), t.min())
            st["axIoU"].append(inter / uni if uni > 0 else 1.0)

    q = lambda a, p: float(np.quantile(a, p)) if len(a) else float("nan")
    print("\nOCCT faces as js/brepBuilder.js builds them, scored against the true surfaces")
    print("dev = max face-sample distance from true surface / diagonal; excess = share of samples off the true patch;")
    print("axIoU = emitted vs true axial range along the true axis\n")
    for (mesher, kind), st in sorted(stats.items()):
        n = len(st["n"])
        built = n - len(st["missing"])
        line = f"  {mesher:6} {kind:9} n {n:5}  built {100 * built / n:5.1f}%"
        if built and st["off"]:
            off, miss = np.array(st["off"]), np.array(st["miss"])
            line += (f"  off-patch area p50 {100 * q(off, .5):5.1f}% p90 {100 * q(off, .9):5.1f}% (>5%: {100 * np.mean(off > .05):4.1f}% of faces)"
                     f"  missed area p50 {100 * q(miss, .5):5.1f}% p90 {100 * q(miss, .9):5.1f}% (>5%: {100 * np.mean(miss > .05):4.1f}%)")
        elif built:
            dev = st["dev"]
            line += (f"  dev p50 {q(dev, .5):8.1e} ok<4 {100 * np.mean(np.array(dev) < 1e-4):5.1f}%"
                     f"  excess p50 {100 * q(st['excess'], .5):5.1f}% p90 {100 * q(st['excess'], .9):5.1f}%")
            if st["axIoU"]:
                line += f"  axIoU p50 {q(st['axIoU'], .5):.2f} p10 {q(st['axIoU'], .1):.2f}"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
