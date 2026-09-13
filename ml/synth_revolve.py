#!/usr/bin/env python3
"""
synth_revolve.py - synthesise GENUINE surfaces of revolution, embedded in real geometry.

Why this exists. Most of what ABC labels Revolution is a cylinder, cone, sphere or torus
written out as a revolved B-spline (see canonical.py). Once those are relabeled, genuinely
freeform revolves are a thin class - a revolved spline or conic profile - and too thin to
learn from. So we make more.

What makes a synthetic set USEFUL rather than merely large is that it looks like the
places real revolves occur, not like a catalogue of clean closed ones:

  - PARTIAL revolves, so the surface of revolution ends at planar caps and its patches
    meet hard edges rather than wrapping seamlessly back on themselves;
  - revolves FUSED with and CUT by other solids, so they are trimmed by booleans and the
    tessellation carries the artefacts of those intersections - sliver triangles, dense
    bands along the trim curve, tangent junctions;
  - several non-primitive solids combined, so a patch may border another revolve.

Every generated face is checked with canonical.canonical_type. A synthetic "revolution"
that is secretly a cylinder would reintroduce exactly the label noise this is meant to
fix, so profiles are constructed to be genuinely curved and verified after the fact.
"""

from __future__ import annotations

import math
import random
import struct
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace,
                                BRepBuilderAPI_MakeWire, BRepBuilderAPI_Transform)
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeRevol
from OCP.Geom import Geom_Ellipse, Geom_TrimmedCurve
from OCP.GeomAbs import GeomAbs_Shape
from OCP.GeomAPI import GeomAPI_PointsToBSpline
from OCP.gp import gp_Ax1, gp_Ax2, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec
from OCP.TColgp import TColgp_Array1OfPnt
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from canonical import canonical_type                     # noqa: E402
from occ_tessellate import DENSITY_LADDER, bbox_diagonal, face_type_name, tessellate  # noqa: E402

Z_AXIS = gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))


def _edge(p, q):
    return BRepBuilderAPI_MakeEdge(p, q).Edge()


def spline_profile(rng: random.Random, height: float = 2.0):
    """A revolved B-spline: radius varies smoothly with height and never touches the axis.

    Bumps are kept to at least 12% of the base radius, so the profile is visibly curved
    rather than a line in disguise.
    """
    base = rng.uniform(0.5, 1.0)
    bumps = [(rng.uniform(0.12, 0.35) * base, rng.uniform(0.6, 2.4), rng.uniform(0, math.tau))
             for _ in range(rng.randint(2, 4))]
    n = 14
    arr = TColgp_Array1OfPnt(1, n)
    for i in range(n):
        z = height * i / (n - 1)
        r = base + sum(a * math.sin(f * math.pi * z / height + ph) for a, f, ph in bumps)
        arr.SetValue(i + 1, gp_Pnt(max(r, 0.2), 0.0, z))
    curve = GeomAPI_PointsToBSpline(arr, 3, 8, GeomAbs_Shape.GeomAbs_C2, 1e-7).Curve()
    p0 = curve.Value(curve.FirstParameter())
    p1 = curve.Value(curve.LastParameter())
    a0, a1 = gp_Pnt(0, 0, p0.Z()), gp_Pnt(0, 0, p1.Z())
    w = BRepBuilderAPI_MakeWire()
    w.Add(BRepBuilderAPI_MakeEdge(curve).Edge())
    w.Add(_edge(p1, a1))
    w.Add(_edge(a1, a0))
    w.Add(_edge(a0, p0))
    return BRepBuilderAPI_MakeFace(w.Wire(), True).Face()


def conic_profile(rng: random.Random):
    """Half of a non-circular ellipse, revolved: a spheroid, which no primitive fits."""
    a = rng.uniform(0.6, 1.2)
    b = a * rng.choice([rng.uniform(0.45, 0.8), rng.uniform(1.25, 1.9)])
    major, minor = max(a, b), min(a, b)
    zc = rng.uniform(0.8, 1.4)
    if a >= b:     # major axis radial: points (major*cos t, 0, zc + minor*sin t)
        ax2 = gp_Ax2(gp_Pnt(0, 0, zc), gp_Dir(0, -1, 0), gp_Dir(1, 0, 0))
        t0, t1 = -math.pi / 2, math.pi / 2
    else:          # major axis along Z: points (-minor*sin t, 0, zc + major*cos t)
        ax2 = gp_Ax2(gp_Pnt(0, 0, zc), gp_Dir(0, -1, 0), gp_Dir(0, 0, 1))
        t0, t1 = math.pi, 2 * math.pi
    arc = Geom_TrimmedCurve(Geom_Ellipse(ax2, major, minor), t0, t1)
    p0 = arc.Value(arc.FirstParameter())
    p1 = arc.Value(arc.LastParameter())
    w = BRepBuilderAPI_MakeWire()
    w.Add(BRepBuilderAPI_MakeEdge(arc).Edge())
    w.Add(_edge(p1, p0))
    return BRepBuilderAPI_MakeFace(w.Wire(), True).Face()


def revolve(profile, angle: float):
    return BRepPrimAPI_MakeRevol(profile, Z_AXIS, angle).Shape()


def moved(shape, dx=0.0, dy=0.0, dz=0.0):
    t = gp_Trsf()
    t.SetTranslation(gp_Vec(dx, dy, dz))
    return BRepBuilderAPI_Transform(shape, t, True).Shape()


def generate(kind: str, seed: int):
    rng = random.Random(seed)
    prof = conic_profile(rng) if kind.startswith("conic") else spline_profile(rng)
    if kind in ("spline_full", "conic_full"):
        return revolve(prof, math.tau)
    if kind in ("spline_partial", "conic_partial"):
        return revolve(prof, math.radians(rng.uniform(60, 300)))
    if kind == "spline_fused_box":
        body = revolve(prof, math.tau)
        s = rng.uniform(0.6, 1.1)
        box = BRepPrimAPI_MakeBox(gp_Pnt(rng.uniform(0.1, 0.6), -s / 2, rng.uniform(0.2, 1.0)),
                                  s * 1.4, s, s).Shape()
        return BRepAlgoAPI_Fuse(body, box).Shape()
    if kind == "spline_cut_hole":
        body = revolve(prof, math.tau)
        hole = BRepPrimAPI_MakeCylinder(
            gp_Ax2(gp_Pnt(-3, 0, rng.uniform(0.5, 1.5)), gp_Dir(1, 0, 0)),
            rng.uniform(0.12, 0.3), 6.0).Shape()
        return BRepAlgoAPI_Cut(body, hole).Shape()
    if kind == "two_revolves":
        a = revolve(prof, math.tau)
        b = moved(revolve(spline_profile(rng), math.tau), dx=rng.uniform(0.7, 1.3))
        return BRepAlgoAPI_Fuse(a, b).Shape()
    if kind == "partial_cut":
        body = revolve(prof, math.radians(rng.uniform(120, 300)))
        cutter = BRepPrimAPI_MakeBox(gp_Pnt(-2, -2, rng.uniform(1.2, 1.7)), 4, 4, 2).Shape()
        return BRepAlgoAPI_Cut(body, cutter).Shape()
    raise ValueError(kind)


KINDS = ("spline_full", "spline_partial", "conic_full", "conic_partial",
         "spline_fused_box", "spline_cut_hole", "two_revolves", "partial_cut")


def audit(shape):
    """Stored vs canonical type of every face."""
    stored = Counter()
    rev_genuine = rev_total = 0
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        f = TopoDS.Face_s(ex.Current())
        ex.Next()
        st = face_type_name(f)
        stored[st] += 1
        if st == "Revolution":
            rev_total += 1
            rev_genuine += canonical_type(f, 1e-6, st)[0] == "Revolution"
    return stored, rev_total, rev_genuine


def write_stl(path, positions):
    tris = positions.reshape(-1, 3, 3).astype(np.float64)
    with open(path, "wb") as fh:
        fh.write(b"synth_revolve preview".ljust(80, b"\0"))
        fh.write(struct.pack("<I", len(tris)))
        for t in tris:
            n = np.cross(t[1] - t[0], t[2] - t[0])
            L = np.linalg.norm(n)
            n = n / L if L > 0 else n
            fh.write(struct.pack("<12fH", *n, *t[0], *t[1], *t[2], 0))


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "D:/abc-data/synth_preview")
    out.mkdir(parents=True, exist_ok=True)
    per = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    print(f"\n  {'kind':<18} {'seed':>4}  {'faces':>5}  {'rev genuine':>11}  {'tris':>8}  stored types")
    tot_rev = tot_gen = fails = 0
    for kind in KINDS:
        for seed in range(per):
            try:
                shape = generate(kind, 1000 + seed)
                if shape is None or shape.IsNull():
                    raise RuntimeError("null shape")
                stored, rt, rg = audit(shape)
                _, lin, ang = DENSITY_LADDER[1]
                t = tessellate(shape, lin, ang, bbox_diagonal(shape))
                if len(t) == 0:
                    raise RuntimeError("empty tessellation")
                if seed == 0:
                    write_stl(out / f"{kind}.stl", t.positions)
                tot_rev += rt
                tot_gen += rg
                print(f"  {kind:<18} {seed:>4}  {sum(stored.values()):>5}  {rg:>4}/{rt:<6}  "
                      f"{len(t):>8,}  {dict(stored)}")
            except Exception as exc:
                fails += 1
                print(f"  {kind:<18} {seed:>4}  FAILED  {type(exc).__name__}: {exc}"[:150])
    print(f"\n  surface-of-revolution faces that are genuine, not a hidden primitive: {tot_gen}/{tot_rev}")
    print(f"  generation failures: {fails}/{len(KINDS) * per}")
    print(f"  previews (medium density, seed 0): {out}")
