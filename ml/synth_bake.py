#!/usr/bin/env python3
"""
synth_bake.py - generate genuine-revolution parts and write them in bake format.

Output is interchangeable with stage1_bake.py's: one .npz per model under <out>/<prefix>/,
holding raw triangles, face id and face type at each density - except that face_type here
is the CANONICAL type from canonical.py, not the stored one. patches.BakedPatches can read
either directory unchanged.

Profiles, all genuinely curved so no revolved face is a primitive in disguise:

  spline       revolved B-spline, radius varying smoothly with height
  ellipse      half of a non-circular ellipse -> oblate or prolate spheroid
  hyperbola    hyperbola arc -> waisted hyperboloid of one sheet
  parabola     parabola arc  -> flared paraboloidal shell

Contexts, so the tessellation carries the artefacts real revolves have:

  full / partial revolve, fused with a box, cut by a cross hole, fused with a second
  revolve on a parallel axis, fused with a second revolve on a TILTED axis (so the
  intersection curve is not a circle), and a partial revolve cut by a plane.

Each part is rotated randomly before tessellation. patches.py re-orients every patch to
its seed normal anyway, but boolean trims and tessellator behaviour both depend on
orientation relative to the world axes, so axis-aligned parts would be a narrow sample.

BUSY CONTEXTS, and why they were added. The first version made parts dominated by their
revolve, and its neighbourhood statistics were measurably wrong. Fraction of a 512-facet
patch sharing the seed's face, for Revolution seeds:

                          mean    p10    p50    p90
    real ABC (canonical)  0.524   0.08   0.48   1.00
    synthetic, first cut  0.840   0.71   0.85   1.00

Real revolves often sit as a SMALL face among much larger neighbours - a knob on a plate,
the exposed end of a mostly buried shaft - and the synthetic set had no such cases. A
model trained on it could learn "Revolution seed => nearly everything belongs", which real
parts punish. The boss / buried / cluster contexts exist to put that low tail back.

They did not, at first, and the reason is worth keeping. The statistic is a FACET fraction,
not an area fraction, and tessellation deflection is relative to the bounding box: a big
plate face becomes a couple of triangles while a curved revolve gets many. Measured on
the first busy parts, a small knob on a plate held 87.4% of the triangles and 3.6% of the
area. Adding planes cannot lower the positive rate. Adding CURVED neighbours can - fillets
on the boolean junctions, holes, cylindrical bosses - because each contributes facets in
proportion to its curvature, the way real parts' fillets and features do.

Every revolution face is audited with canonical.py, and the summary reports how many
came out genuine. A part whose revolution face turns out to be a primitive is written
with that primitive's label, so a bad profile degrades gracefully instead of poisoning
the class.

Usage:
    python synth_bake.py --n 4000 --out D:/abc-data/synth_bake --jobs 16
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

KINDS = (
    "spline:full", "spline:partial", "spline:box", "spline:hole", "spline:pair",
    "spline:tilted_pair", "spline:partial_cut",
    "ellipse:full", "ellipse:partial", "ellipse:hole", "ellipse:box",
    "hyperbola:full", "hyperbola:partial", "hyperbola:box", "hyperbola:tilted_pair",
    "parabola:full", "parabola:partial", "parabola:hole",
    # busy contexts: a small revolve among much larger neighbours
    "spline:boss", "spline:buried", "spline:cluster",
    "ellipse:boss", "ellipse:buried", "ellipse:cluster",
    "hyperbola:boss", "hyperbola:buried", "hyperbola:cluster",
    "parabola:boss", "parabola:buried", "parabola:cluster",
    # dense curved surroundings: the case that actually moves the facet statistics
    "spline:busy_plate", "ellipse:busy_plate", "hyperbola:busy_plate", "parabola:busy_plate",
    "spline:busy_plate", "ellipse:busy_plate",
    # a very small revolve: tessellation deflection follows the PART's size, so a small
    # revolve on a large part gets only tens of triangles, the real low tail
    "spline:tiny", "ellipse:tiny", "hyperbola:tiny", "parabola:tiny",
    # turned parts: a stepped profile of lines with one or two short curved segments, so the
    # genuine revolve is a NARROW BAND among cylinders, cones and flat shoulders - the shape
    # of most real Revolution faces, and the only scene that reaches the low tail (a real
    # revolve face at p10 carries about 12 triangles; a tiny whole revolve still gets 100+,
    # because angular deflection sets a minimum segment count around any circle)
    "spline:lathe", "spline:lathe_partial", "spline:lathe_holes",
)

# Scene weights, fitted to real parts rather than chosen by taste. With every kind drawn
# equally, Revolution seeds saw a mean positive rate of 0.705 (p10 0.32) against real
# ABC's 0.496 (p10 0.09, 539 seeds over the whole validation split), because isolated
# revolves dominate their own neighbourhood. Simple scenes are kept - real parts reach
# p90 0.98 too - but drawn rarely; buried and tiny revolves carry the low tail.
SIMPLE_CONTEXTS = {"full", "partial", "box", "hole", "pair", "tilted_pair", "partial_cut"}
# Turned parts are about 60% of draws. They are the only scene whose Revolution statistics
# match real parts on their own (mean 0.494 vs 0.504), and predicting the full mix from the
# measured per-scene rates showed the share matters less than expected: mean 0.541 at a 16%
# lathe share, 0.516 at 60%, 0.509 at 75%, with p10 near 0.21 and p90 near 0.83 throughout.
# 60% rather than 75% keeps the other scenes' variety (tilted intersections, busy plates,
# buried shafts) in the set, at a cost of 0.007 in the mean.
CONTEXT_WEIGHT = {"buried": 2.5, "tiny": 2.5, "lathe": 24.0, "lathe_holes": 24.0, "lathe_partial": 12.0}


def kind_weight(kind: str) -> float:
    ctx = kind.split(":")[1]
    return 0.3 if ctx in SIMPLE_CONTEXTS else CONTEXT_WEIGHT.get(ctx, 1.0)


def kind_for(idx: int, seed: int, only: str | None = None) -> str:
    """Deterministic weighted draw, so a part is reproducible from (idx, seed, only) alone.
    `only` restricts the draw to kinds containing that substring, for smoke-testing one scene."""
    kinds = [k for k in KINDS if only is None or only in k] or list(KINDS)
    return random.Random(seed * 7919 + idx).choices(kinds, weights=[kind_weight(k) for k in kinds])[0]


def _profiles():
    from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace,
                                    BRepBuilderAPI_MakeWire)
    from OCP.Geom import Geom_Hyperbola, Geom_Parabola, Geom_TrimmedCurve
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
    from synth_revolve import conic_profile, spline_profile

    def close_to_axis(curve):
        """Profile curve plus the two radial lines and the axis segment that close it."""
        p0 = curve.Value(curve.FirstParameter())
        p1 = curve.Value(curve.LastParameter())
        a0, a1 = gp_Pnt(0, 0, p0.Z()), gp_Pnt(0, 0, p1.Z())
        w = BRepBuilderAPI_MakeWire()
        w.Add(BRepBuilderAPI_MakeEdge(curve).Edge())
        w.Add(BRepBuilderAPI_MakeEdge(p1, a1).Edge())
        w.Add(BRepBuilderAPI_MakeEdge(a1, a0).Edge())
        w.Add(BRepBuilderAPI_MakeEdge(a0, p0).Edge())
        return BRepBuilderAPI_MakeFace(w.Wire(), True).Face()

    def hyperbola(rng):
        # x = a cosh t, z = zc + b sinh t : a waist of radius a, never touching the axis.
        a = rng.uniform(0.35, 0.8)
        b = rng.uniform(0.4, 1.2)
        zc = rng.uniform(0.9, 1.3)
        ax2 = gp_Ax2(gp_Pnt(0, 0, zc), gp_Dir(0, -1, 0), gp_Dir(1, 0, 0))
        t = rng.uniform(0.8, 1.4)
        return close_to_axis(Geom_TrimmedCurve(Geom_Hyperbola(ax2, max(a, 1e-3), b), -t, t))

    def parabola(rng):
        # x = x0 + s^2 / (4 f), z = zc + s : flares away from a minimum radius x0.
        x0 = rng.uniform(0.3, 0.7)
        f = rng.uniform(0.25, 0.8)
        zc = rng.uniform(0.9, 1.3)
        ax2 = gp_Ax2(gp_Pnt(x0, 0, zc), gp_Dir(0, -1, 0), gp_Dir(1, 0, 0))
        h = rng.uniform(0.6, 1.0)
        return close_to_axis(Geom_TrimmedCurve(Geom_Parabola(ax2, f), -h, h))

    return {"spline": spline_profile, "ellipse": conic_profile,
            "hyperbola": hyperbola, "parabola": parabola}


def _zmin(shape):
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    return box.Get()[2]


def _scaled(shape, factor):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Pnt, gp_Trsf
    t = gp_Trsf()
    t.SetScale(gp_Pnt(0, 0, 0), factor)
    return BRepBuilderAPI_Transform(shape, t, True).Shape()


def _fillet_some(shape, rng, lo, hi, tries=4):
    """Fillet a random subset of edges, backing off when OCCT refuses.

    Filleting arbitrary edges fails often - seams of revolution, edges too short for the
    radius - so this tries progressively fewer edges and returns the input unchanged if
    nothing succeeds, rather than losing the part.
    """
    from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    edges = []
    ex = TopExp_Explorer(shape, TopAbs_EDGE)
    while ex.More():
        edges.append(TopoDS.Edge_s(ex.Current()))
        ex.Next()
    if not edges:
        return shape
    k = min(len(edges), rng.randint(3, 10))
    for _ in range(tries):
        pick = rng.sample(edges, k)
        try:
            mk = BRepFilletAPI_MakeFillet(shape)
            r = rng.uniform(lo, hi)
            for e in pick:
                mk.Add(r, e)
            mk.Build()
            if mk.IsDone():
                out = mk.Shape()
                if out is not None and not out.IsNull():
                    return out
        except Exception:
            pass
        k = max(1, k // 2)
    return shape


def _lathe_profile(rng):
    """Closed XZ profile of a turned part: many line segments, rounded shoulders, and one or
    two SHORT, gently curved segments.

    Shaped by measurement, not by eye. Real Revolution seeds whose patch is mostly other
    faces (positive rate < 0.2) sit on small faces (median 84 triangles) in complex parts
    (median 100 faces), surrounded by Torus 30%, Plane 21%, Cylinder 19%, BSpline 15%. The
    first lathe version had few segments and a large bulge; its revolve faces averaged 1,695
    triangles against 96 per cylinder face - a strongly curved profile is subdivided along
    its length, a straight one is not - and it moved the statistic the wrong way (mean 0.77).

    Shoulders are rounded with exact tangent arcs in the profile, which revolve into true
    tori like real fillets, without BRepFilletAPI's failures and hangs.

    NEAR-TORI, and why they carry the low tail. Measured on the real validation split, 36%
    of genuine Revolution faces come within 1e-3 (residual / extent) of a primitive, nearly
    always a torus: fillets approximated as a revolved B-spline. Those faces are small
    (median 36-311 triangles) and hold the low positive rates (seeds under 0.2 sit on faces
    with median residual 7.6e-4), while clearly curved revolves (residual ~4e-2) hold the
    high ones. So half the rounded shoulders are written as an interpolating B-spline
    through a slightly perturbed arc, off by 1e-4 to 1e-2 of the local radius: a genuine
    revolve at the 1e-6 label threshold, tessellated like the fillet it approximates. Curved segments are
    exact interpolating B-splines with a non-circular bulge; anything that nonetheless
    comes out a primitive is caught by the canonical audit.
    """
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire
    from OCP.GC import GC_MakeArcOfCircle
    from OCP.GeomAPI import GeomAPI_Interpolate
    from OCP.TColgp import TColgp_HArray1OfPnt
    from OCP.gp import gp_Pnt, gp_Vec

    knots, z = [], 0.0
    r = rng.uniform(0.3, 1.0)
    knots.append((r, z))
    for _ in range(rng.randint(10, 18)):
        if rng.random() < 0.35:                             # shoulder: radial step, flat annulus
            r = min(1.1, max(0.15, r + rng.choice((-1, 1)) * rng.uniform(0.08, 0.3)))
        else:                                               # axial run: cylinder or cone
            z += rng.uniform(0.08, 0.45)
            if rng.random() < 0.4:
                r = min(1.1, max(0.15, r + rng.uniform(-0.2, 0.2)))
        if abs(r - knots[-1][0]) > 1e-9 or abs(z - knots[-1][1]) > 1e-9:
            knots.append((r, z))
    segs = len(knots) - 1
    axial = [i for i in range(segs) if 1e-9 < knots[i + 1][1] - knots[i][1] < 0.3]
    axial = axial or [i for i in range(segs) if knots[i + 1][1] > knots[i][1]]
    curved = set(rng.sample(axial, min(len(axial), rng.randint(0, 1)))) if axial else set()

    # rounded shoulders: a tangent arc at a knot joining two straight segments
    rounds = {}
    for k in range(1, segs):
        if k - 1 in curved or k in curved or rng.random() > 0.75:
            continue
        (ra, za), (rc, zc), (rb, zb) = knots[k - 1], knots[k], knots[k + 1]
        la, lb = math.hypot(rc - ra, zc - za), math.hypot(rb - rc, zb - zc)
        u = ((rc - ra) / la, (zc - za) / la)
        w = ((rb - rc) / lb, (zb - zc) / lb)
        cos_t = max(-1.0, min(1.0, u[0] * w[0] + u[1] * w[1]))
        if cos_t > 0.999 or cos_t < -0.95:
            continue
        d = rng.uniform(0.02, 0.08) * math.tan(math.acos(cos_t) / 2)
        if d >= 0.45 * la or d >= 0.45 * lb:
            continue
        # near-arc: deviation as a fraction of the local radius, log-uniform 1e-4 .. 1e-2
        near = 10 ** rng.uniform(-4, -2) * rc if rng.random() < 0.5 else 0.0
        rounds[k] = ((rc - u[0] * d, zc - u[1] * d), (rc + w[0] * d, zc + w[1] * d), u, near, (rc, zc))

    pnt = lambda q: gp_Pnt(q[0], 0.0, q[1])
    w_ = BRepBuilderAPI_MakeWire()
    for i in range(segs):
        a = rounds[i][1] if i in rounds else knots[i]
        b = rounds[i + 1][0] if i + 1 in rounds else knots[i + 1]
        if i in curved:
            (r0, z0), (r1, z1) = knots[i], knots[i + 1]
            amp = rng.choice((-1, 1)) * rng.uniform(0.02, 0.08)
            amp = max(-0.5 * min(r0, r1), min(0.5 * min(r0, r1), amp))
            arr = TColgp_HArray1OfPnt(1, 6)
            for j in range(6):
                t = j / 5
                rr = r0 + (r1 - r0) * t + amp * math.sin(math.pi * t) * (1 + 0.4 * math.sin(3 * math.pi * t))
                arr.SetValue(j + 1, gp_Pnt(rr, 0.0, z0 + (z1 - z0) * t))
            itp = GeomAPI_Interpolate(arr, False, 1e-9)
            itp.Perform()
            w_.Add(BRepBuilderAPI_MakeEdge(itp.Curve()).Edge())
        else:
            w_.Add(BRepBuilderAPI_MakeEdge(pnt(a), pnt(b)).Edge())
        if i + 1 in rounds:
            p1, p2, u, near, corner = rounds[i + 1]
            arc = GC_MakeArcOfCircle(pnt(p1), gp_Vec(u[0], 0.0, u[1]), pnt(p2)).Value()
            if near <= 0.0:
                w_.Add(BRepBuilderAPI_MakeEdge(arc).Edge())
            else:
                # push the arc's interior towards the corner-to-chord bisector; the ends stay put
                mr, mz = (p1[0] + p2[0]) / 2 - corner[0], (p1[1] + p2[1]) / 2 - corner[1]
                ml = math.hypot(mr, mz) or 1.0
                vr, vz = mr / ml, mz / ml
                n, u0, u1 = 7, arc.FirstParameter(), arc.LastParameter()
                arr = TColgp_HArray1OfPnt(1, n)
                for j in range(n):
                    t = j / (n - 1)
                    q = arc.Value(u0 + (u1 - u0) * t)
                    bump = near * math.sin(math.pi * t) * (1 + 0.5 * math.sin(2 * math.pi * t))
                    arr.SetValue(j + 1, gp_Pnt(q.X() + vr * bump, 0.0, q.Z() + vz * bump))
                arr.SetValue(1, pnt(p1))
                arr.SetValue(n, pnt(p2))
                itp = GeomAPI_Interpolate(arr, False, 1e-9)
                itp.Perform()
                w_.Add(BRepBuilderAPI_MakeEdge(itp.Curve()).Edge())
    (rl, zl), (rf, zf) = knots[-1], knots[0]
    w_.Add(BRepBuilderAPI_MakeEdge(gp_Pnt(rl, 0, zl), gp_Pnt(0, 0, zl)).Edge())
    w_.Add(BRepBuilderAPI_MakeEdge(gp_Pnt(0, 0, zl), gp_Pnt(0, 0, zf)).Edge())
    w_.Add(BRepBuilderAPI_MakeEdge(gp_Pnt(0, 0, zf), gp_Pnt(rf, 0, zf)).Edge())
    return BRepBuilderAPI_MakeFace(w_.Wire(), True).Face()


def build(kind: str, seed: int):
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax1, gp_Ax2, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec
    from synth_revolve import moved, revolve

    rng = random.Random(seed)
    prof_name, ctx = kind.split(":")
    make = _profiles()
    prof = make[prof_name](rng)

    if ctx == "full":
        shape = revolve(prof, math.tau)
    elif ctx == "partial":
        shape = revolve(prof, math.radians(rng.uniform(60, 300)))
    elif ctx == "box":
        s = rng.uniform(0.5, 1.0)
        box = BRepPrimAPI_MakeBox(gp_Pnt(rng.uniform(0.1, 0.6), -s / 2, rng.uniform(0.3, 1.2)),
                                  s * 1.4, s, s).Shape()
        shape = BRepAlgoAPI_Fuse(revolve(prof, math.tau), box).Shape()
    elif ctx == "hole":
        hole = BRepPrimAPI_MakeCylinder(
            gp_Ax2(gp_Pnt(-3, 0, rng.uniform(0.7, 1.4)), gp_Dir(1, 0, 0)),
            rng.uniform(0.1, 0.25), 6.0).Shape()
        shape = BRepAlgoAPI_Cut(revolve(prof, math.tau), hole).Shape()
    elif ctx == "pair":
        other = moved(revolve(make["spline"](rng), math.tau), dx=rng.uniform(0.7, 1.2))
        shape = BRepAlgoAPI_Fuse(revolve(prof, math.tau), other).Shape()
    elif ctx == "tilted_pair":
        other = revolve(make[rng.choice(["spline", "ellipse"])](rng), math.tau)
        t = gp_Trsf()
        t.SetRotation(gp_Ax1(gp_Pnt(0, 0, 1.0), gp_Dir(0, 1, 0)), math.radians(rng.uniform(25, 70)))
        other = BRepBuilderAPI_Transform(other, t, True).Shape()
        other = moved(other, dx=rng.uniform(0.4, 0.9))
        shape = BRepAlgoAPI_Fuse(revolve(prof, math.tau), other).Shape()
    elif ctx == "boss":
        # small revolve standing on a large plate, base sunk slightly into it
        w_ = rng.uniform(4.0, 6.0)
        th = rng.uniform(0.4, 0.8)
        plate = BRepPrimAPI_MakeBox(gp_Pnt(-w_ / 2, -w_ / 2, -th), w_, w_, th).Shape()
        knob = _scaled(revolve(prof, math.tau), rng.uniform(0.3, 0.6))
        knob = moved(knob, dx=rng.uniform(-w_ / 4, w_ / 4), dy=rng.uniform(-w_ / 4, w_ / 4),
                     dz=-_zmin(knob) - 0.05)
        shape = BRepAlgoAPI_Fuse(plate, knob).Shape()
        shape = _fillet_some(shape, rng, 0.03, 0.12)
    elif ctx == "buried":
        # most of the revolve swallowed by a wide block; only its upper end stays exposed
        body = revolve(prof, math.tau)
        w_ = rng.uniform(4.0, 6.0)
        top = _zmin(body) + rng.uniform(0.6, 0.85) * 2.0
        block = BRepPrimAPI_MakeBox(gp_Pnt(-w_ / 2, -w_ / 2, _zmin(body) - 0.5), w_, w_,
                                    top - _zmin(body) + 0.5).Shape()
        shape = BRepAlgoAPI_Fuse(block, body).Shape()
        shape = _fillet_some(shape, rng, 0.03, 0.12)
    elif ctx == "cluster":
        # a plate carrying several small revolves and a few through-holes
        w_ = rng.uniform(5.0, 8.0)
        th = rng.uniform(0.4, 0.8)
        shape = BRepPrimAPI_MakeBox(gp_Pnt(-w_ / 2, -w_ / 2, -th), w_, w_, th).Shape()
        for _ in range(rng.randint(2, 4)):
            knob = _scaled(revolve(make[rng.choice(list(make))](rng), math.tau), rng.uniform(0.25, 0.5))
            knob = moved(knob, dx=rng.uniform(-w_ / 3, w_ / 3), dy=rng.uniform(-w_ / 3, w_ / 3),
                         dz=-_zmin(knob) - 0.05)
            shape = BRepAlgoAPI_Fuse(shape, knob).Shape()
        for _ in range(rng.randint(1, 3)):
            hole = BRepPrimAPI_MakeCylinder(
                gp_Ax2(gp_Pnt(rng.uniform(-w_ / 3, w_ / 3), rng.uniform(-w_ / 3, w_ / 3), -th - 1),
                       gp_Dir(0, 0, 1)), rng.uniform(0.12, 0.35), th + 2).Shape()
            shape = BRepAlgoAPI_Cut(shape, hole).Shape()
        shape = _fillet_some(shape, rng, 0.03, 0.12)
    elif ctx == "busy_plate":
        # a plate densely populated with CURVED features around the revolve: holes,
        # cylindrical bosses, other small revolves, and fillets on the junctions
        w_ = rng.uniform(5.0, 8.0)
        th = rng.uniform(0.5, 0.9)
        shape = BRepPrimAPI_MakeBox(gp_Pnt(-w_ / 2, -w_ / 2, -th), w_, w_, th).Shape()
        main = _scaled(revolve(prof, math.tau), rng.uniform(0.35, 0.6))
        main = moved(main, dx=rng.uniform(-0.5, 0.5), dy=rng.uniform(-0.5, 0.5), dz=-_zmin(main) - 0.05)
        shape = BRepAlgoAPI_Fuse(shape, main).Shape()
        for _ in range(rng.randint(2, 5)):
            r_ = rng.uniform(0.12, 0.4)
            boss = BRepPrimAPI_MakeCylinder(
                gp_Ax2(gp_Pnt(rng.uniform(-w_ / 2.5, w_ / 2.5), rng.uniform(-w_ / 2.5, w_ / 2.5), -0.05),
                       gp_Dir(0, 0, 1)), r_, rng.uniform(0.3, 1.2)).Shape()
            shape = BRepAlgoAPI_Fuse(shape, boss).Shape()
        for _ in range(rng.randint(0, 2)):
            knob = _scaled(revolve(make[rng.choice(list(make))](rng), math.tau), rng.uniform(0.2, 0.4))
            knob = moved(knob, dx=rng.uniform(-w_ / 3, w_ / 3), dy=rng.uniform(-w_ / 3, w_ / 3),
                         dz=-_zmin(knob) - 0.05)
            shape = BRepAlgoAPI_Fuse(shape, knob).Shape()
        for _ in range(rng.randint(3, 7)):
            hole = BRepPrimAPI_MakeCylinder(
                gp_Ax2(gp_Pnt(rng.uniform(-w_ / 2.3, w_ / 2.3), rng.uniform(-w_ / 2.3, w_ / 2.3), -th - 1),
                       gp_Dir(0, 0, 1)), rng.uniform(0.08, 0.3), th + 2.5).Shape()
            shape = BRepAlgoAPI_Cut(shape, hole).Shape()
        shape = _fillet_some(shape, rng, 0.03, 0.12)
    elif ctx == "tiny":
        # a small revolve on a part many times its size, among holes and fillets
        w_ = rng.uniform(5.0, 9.0)
        th = rng.uniform(0.5, 1.0)
        shape = BRepPrimAPI_MakeBox(gp_Pnt(-w_ / 2, -w_ / 2, -th), w_, w_, th).Shape()
        for _ in range(rng.randint(1, 3)):
            knob = _scaled(revolve(prof if _ == 0 else make[rng.choice(list(make))](rng), math.tau),
                           rng.uniform(0.07, 0.2))
            knob = moved(knob, dx=rng.uniform(-w_ / 3, w_ / 3), dy=rng.uniform(-w_ / 3, w_ / 3),
                         dz=-_zmin(knob) - 0.02)
            shape = BRepAlgoAPI_Fuse(shape, knob).Shape()
        for _ in range(rng.randint(2, 6)):
            hole = BRepPrimAPI_MakeCylinder(
                gp_Ax2(gp_Pnt(rng.uniform(-w_ / 2.3, w_ / 2.3), rng.uniform(-w_ / 2.3, w_ / 2.3), -th - 1),
                       gp_Dir(0, 0, 1)), rng.uniform(0.08, 0.4), th + 2.5).Shape()
            shape = BRepAlgoAPI_Cut(shape, hole).Shape()
        shape = _fillet_some(shape, rng, 0.03, 0.15)
    elif ctx in ("lathe", "lathe_partial", "lathe_holes"):
        angle = math.radians(rng.uniform(90, 300)) if ctx == "lathe_partial" else math.tau
        shape = revolve(_lathe_profile(rng), angle)
        if ctx == "lathe_holes":
            for _ in range(rng.randint(1, 3)):
                t = gp_Trsf()
                t.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), rng.uniform(0, math.tau))
                hole = BRepPrimAPI_MakeCylinder(
                    gp_Ax2(gp_Pnt(-3, 0, rng.uniform(0.2, 2.5)), gp_Dir(1, 0, 0)),
                    rng.uniform(0.05, 0.18), 6.0).Shape()
                shape = BRepAlgoAPI_Cut(shape, BRepBuilderAPI_Transform(hole, t, True).Shape()).Shape()
    elif ctx == "partial_cut":
        body = revolve(prof, math.radians(rng.uniform(120, 300)))
        cutter = BRepPrimAPI_MakeBox(gp_Pnt(-2, -2, rng.uniform(1.2, 1.7)), 4, 4, 2).Shape()
        shape = BRepAlgoAPI_Cut(body, cutter).Shape()
    else:
        raise ValueError(kind)

    # Random orientation: booleans and the tessellator both behave differently off-axis.
    t = gp_Trsf()
    axis = np.random.default_rng(seed).normal(size=3)
    axis /= np.linalg.norm(axis)
    t.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(*axis)), rng.uniform(0, math.tau))
    return BRepBuilderAPI_Transform(shape, t, True).Shape()


def one(task):
    idx, seed, base_seed, out_root, *rest = task
    only = rest[0] if rest else None
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from canonical import canonical_type
    from occ_tessellate import DENSITY_LADDER, TYPE_INDEX, bbox_diagonal, face_type_name, tessellate

    kind = kind_for(idx, base_seed, only)
    rec = {"kind": kind, "status": "ok", "stored": Counter(), "canon": Counter(),
           "rev": 0, "rev_genuine": 0, "tris": 0}
    try:
        shape = build(kind, seed)
        if shape is None or shape.IsNull():
            rec["status"] = "null"
            return rec
        canon = []
        ex = TopExp_Explorer(shape, TopAbs_FACE)
        while ex.More():
            f = TopoDS.Face_s(ex.Current())
            ex.Next()
            st = face_type_name(f)
            c = canonical_type(f, 1e-6, st)[0]
            rec["stored"][st] += 1
            rec["canon"][c] += 1
            if st == "Revolution":
                rec["rev"] += 1
                rec["rev_genuine"] += c == "Revolution"
            canon.append(TYPE_INDEX[c])
        canon = np.array(canon, np.uint8)

        diag = bbox_diagonal(shape)
        arrays = {}
        for name, lin, ang in DENSITY_LADDER:
            t = tessellate(shape, lin, ang, diag)
            if len(t) < 32 or len(t) > 400_000:
                continue
            arrays[f"{name}_positions"] = t.positions
            arrays[f"{name}_face_id"] = t.face_id
            arrays[f"{name}_face_type"] = canon[t.face_id]
            rec["tris"] += len(t)
        if not arrays:
            rec["status"] = "no_usable_density"
            return rec
        mid = f"s{idx:07d}"
        out = Path(out_root) / mid[:5] / f"{mid}_0.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, model_id=mid, part="0", diag=np.float64(diag), kind=kind, **arrays)
    except Exception as exc:
        rec["status"] = f"error:{type(exc).__name__}"
    return rec


def run_parts(tasks, jobs: int, timeout: float, max_attempts: int = 2):
    """Yield one record per task, surviving parts that hang or kill their worker.

    OCCT booleans and fillets can loop forever on unlucky input - the first weighted smoke
    bake finished 359 of 360 parts and then sat on the last one indefinitely - and a
    ProcessPoolExecutor cannot cancel a running call. So at most `jobs` tasks are in flight
    (submit time is then close to start time), and when one overruns `timeout` the whole
    pool is killed and rebuilt: the overrunning part is recorded as a timeout, and the
    innocent parts that were running beside it are resubmitted. A worker that dies outright
    breaks the pool the same way; those parts are retried, and recorded as crashes only if
    they break it again.
    """
    from collections import deque
    from concurrent.futures import FIRST_COMPLETED, wait
    from concurrent.futures.process import BrokenProcessPool

    def failed(task, status):
        return {"kind": kind_for(task[0], task[2], task[4] if len(task) > 4 else None), "status": status}

    pending = deque((t, 0) for t in tasks)
    while pending:
        pool = ProcessPoolExecutor(max_workers=jobs)
        live, restart = {}, False
        try:
            while (pending or live) and not restart:
                while pending and len(live) < jobs:
                    task, attempts = pending.popleft()
                    live[pool.submit(one, task)] = (task, attempts, time.time())
                done, _ = wait(live, timeout=2.0, return_when=FIRST_COMPLETED)
                for f in done:
                    task, attempts, _ = live.pop(f)
                    try:
                        yield f.result()
                    except BrokenProcessPool:
                        restart = True
                        if attempts + 1 >= max_attempts:
                            yield failed(task, "crash")
                        else:
                            pending.append((task, attempts + 1))
                    except Exception as exc:
                        yield failed(task, f"error:{type(exc).__name__}")
                now = time.time()
                for f in [f for f, (_, _, t0) in live.items() if now - t0 > timeout]:
                    task, _, _ = live.pop(f)
                    yield failed(task, "timeout")
                    restart = True
            if restart:
                pending.extendleft((task, attempts) for task, attempts, _ in live.values())
        finally:
            if restart:
                for proc in list((getattr(pool, "_processes", None) or {}).values()):
                    try:
                        proc.kill()
                    except Exception:
                        pass
            pool.shutdown(wait=not restart, cancel_futures=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("D:/abc-data/synth_bake"))
    ap.add_argument("--seed", type=int, default=20260913)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--only", default=None,
                    help="restrict to kinds containing this substring (smoke-testing one scene)")
    ap.add_argument("--timeout", type=float, default=180.0,
                    help="seconds before a part is abandoned (a typical part takes about 1-2 s)")
    args = ap.parse_args(argv)

    tasks = [(i, args.seed + i, args.seed, str(args.out), args.only) for i in range(args.n)]
    t0 = time.time()
    status, stored, canon, per_kind = Counter(), Counter(), Counter(), Counter()
    rev = rev_gen = tris = 0
    for i, rec in enumerate(run_parts(tasks, args.jobs, args.timeout), 1):
        status[rec["status"]] += 1
        per_kind[(rec["kind"], rec["status"] == "ok")] += 1
        if rec["status"] == "ok":
            stored.update(rec["stored"])
            canon.update(rec["canon"])
            rev += rec["rev"]
            rev_gen += rec["rev_genuine"]
            tris += rec["tris"]
        if i % 200 == 0:
            print(f"  {i:,}/{args.n:,}  {i / (time.time() - t0):.1f}/s", flush=True)

    print(f"\n{args.n:,} parts in {(time.time() - t0) / 60:.1f} min -> {args.out}")
    print("status: " + ", ".join(f"{k} {v:,}" for k, v in status.most_common()))
    print(f"revolution faces genuine: {rev_gen:,}/{rev:,}")
    print(f"triangles across all densities: {tris:,}")
    print("canonical face types written: " + ", ".join(f"{k} {v:,}" for k, v in canon.most_common()))
    bad = [(k, n) for (k, ok), n in per_kind.items() if not ok]
    if bad:
        print("failures by kind: " + ", ".join(f"{k} {n}" for k, n in sorted(bad)))


if __name__ == "__main__":
    raise SystemExit(main())
