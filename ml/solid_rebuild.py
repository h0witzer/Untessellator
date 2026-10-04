#!/usr/bin/env python3
"""
solid_rebuild.py - turn fitted surfaces on labelled mesh patches into closed, valid B-rep solids.

The step this isolates: patches and their types are given (a perfect classifier / painting
session), each patch has a fitted analytic surface, and the job is a VALID, CLOSED solid - open only
where the input mesh itself is open.

METHODS

  sew   what js/brepBuilder.js does: an independent bounded face per patch, BRepBuilderAPI_Sewing at
        0.5% of the part diagonal, MakeSolid. 0 of 31 fully analytic ABC parts closed, on any mesher,
        with true or fitted surfaces.

  topo  topology from the mesh, geometry from the surfaces (the method to use):
          mesh     triangles paired across every edge (radially where two touching solids share an
                   edge), winding made consistent through the pairs and outward per body
          corner   a mesh vertex where >= 3 patches of one body meet (or a patch boundary meets an open
                   mesh boundary); positioned on ALL its patches' surfaces at once
          edge     each run of boundary between two patches, corner to corner; ONE curve through the
                   boundary vertices projected onto BOTH surfaces, shared by the two faces (no
                   surface-surface intersection, so tangent fillets are well defined); points where
                   the projected run doubles back are dropped first (unfold_chain)
          face     the patch's surface - U-reversed where its natural normal points into the solid,
                   judged over the patch's triangles - bounded by its edges in mesh winding
          repair   ShapeFix_Shape over the shell, then natural-bound wires it adds to closed surfaces
                   are removed (a torus face otherwise becomes the whole torus minus the patch)
        Every inner edge belongs to exactly two faces by construction, so the shell closes unless the
        mesh is open. Each trap met on the way is recorded where it is handled.

  --repair-loop  topo judged by the kernel (build_repaired): while the part is not a closed, valid
        solid of the mesh's volume, find the failing patches (no face, BRepCheck-invalid, edges used
        the same way by two faces, a face far off its patch's area, a free edge), try ONE move -
        absorb the group into the neighbours it lies on, or its next surface type from
        test/fitCandidates.mjs - rebuild, keep it only if the part got better. Inference-time signals
        only; no STEP truth.
        Also here, off by default because they did not pay: constrained_refit (Mesh2Brep's same-point
        and tangency constraints, solved jointly) and exact intersection edges (build_topo exact=).

  --fallback  a solid every time, honestly labelled (facet_fallback): when the part is not recreated,
        the patches still failing (and surfaceless ones) become facets - a planar face per triangle -
        beside the analytic faces; then every non-plane patch; then the whole mesh as a polyhedral
        solid. info.tier: analytic / hybrid / planar / faceted, or none.

  --freeform  a fitted B-spline ("bspline" kind) for every group with no analytic surface (add_freeform,
        freeform_fit.py): one bicubic per group over the group plus a margin, trimmed by the group's
        edges like any other face; periodic in one direction for rings and tubes; a group no single map
        covers is split by normal-cone clustering into pieces with new labels; with --freeform-tangent
        also G1 to analytic neighbours where the free fit already agrees (FREEFORM_TANGENT: better
        surfaces, worse knitting). Before it a freeform face was a hole: 0 of the 25 Onshape-set parts
        with freeform faces rebuilt (true segments). Graded face by face against ABC's true B-splines
        by freeform_eval.py.

MEASURED (31 fully analytic validation parts, one process per part, "recreated" = closed, BRepCheck
valid, STEP volume within 1%, every face area within 2% of the STEP face):

                      true surfaces   fitted surfaces
    native   topo         29/31           23/31
    gmsh     topo         28/31           22/31
    both     sew           0/31            0/31

AUTOMATIC SEGMENTATIONS (s13 tm-ring1 patches, the auto-painter's groups and surfaces, against the
same meshes' true groups with fitted surfaces; recreated = closed, valid, and within 1% of the STEP
volume - of the MESH volume on the Onshape set, which has no STEP volumes; 2026-09-27):

                                      Onshape auto   Onshape true   fill_eval auto   fill_eval true
    topo before unfold_chain              196            254              14                36
    topo                                  211            254              15                36
    topo --repair-loop                    247            260              21                36
    ... + refit and exact moves           246            261              20                36
    topo --repair-loop, etm-ring1 *       249             -               17                 -
    ... + --fallback, any tier **         269             -               32                 -
    (of)                                  275            275              45                45

  *  groups typed by paintCore.typeFinalGroup (2026-10-02, docs/ml-experiment-scope.md)
  ** analytic 249, hybrid 6, planar 5, faceted 9 on Onshape; 17 / 13 / 2 / 0 on fill_eval (a faceted
     solid has the mesh's volume, which on fill_eval is not always within 1% of the STEP's)

Not recreated with TRUE surfaces: 00001644_1 (6 bodies; its mesh pairs into 341 pieces and OCCT
overflows its stack), 00003852_1 (mesh volumes disagree with the STEP by 40%+ - input data), and
00072333_2 on gmsh (5 touching bodies, radial pairing leaves unpaired edges). The extra fitted-surface
failures are almost all small-patch fit ambiguities - 6-facet cylinders fitted perpendicular to the
truth, 11-facet cones 3.5 deg off - which no construction can recover.

INPUT  the fill_eval meshes (.pos/.fid), test/exportPatchFits.mjs output (fitted surfaces), and for
       --surfaces truth the fit_truth sidecars; D:/abc-data/solid_truth_volumes.json for STEP volumes
       and face areas.
       --fid-dir takes the patches from another labelling of the same meshes (test/surfaceScore.mjs
       --patches-out: an automatic segmentation and its surfaces, --fits pointing at the same folder).
       Its patches are not the STEP's faces, so the per-face area check is skipped; the volume is not.

Usage (OCCT venv; run parts in separate processes - one broken mesh can overflow OCCT's stack):
    python solid_rebuild.py --fits D:/abc-data/patch_fits --meshers native,gmsh --method topo,sew --parts <id>
    ... --method topo --repair-loop        (needs node on PATH for test/fitCandidates.mjs)
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from OCP.BRep import BRep_Builder
from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon,
                                BRepBuilderAPI_MakeVertex, BRepBuilderAPI_Sewing)
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.BRepGProp import BRepGProp
from OCP.Geom import (Geom_ConicalSurface, Geom_CylindricalSurface, Geom_Plane, Geom_SphericalSurface,
                      Geom_ToroidalSurface)
from OCP.GeomAPI import GeomAPI_Interpolate
from OCP.GProp import GProp_GProps
from OCP.gp import gp_Ax3, gp_Cone, gp_Cylinder, gp_Dir, gp_Pln, gp_Pnt, gp_Sphere, gp_Torus
from OCP.ShapeFix import ShapeFix_Face, ShapeFix_Shape, ShapeFix_Shell
from OCP.TColgp import TColgp_HArray1OfPnt
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_IN, TopAbs_REVERSED, TopAbs_SHELL, TopAbs_SOLID, TopAbs_WIRE
from OCP.BRep import BRep_Tool
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Face, TopoDS_Shell, TopoDS_Solid, TopoDS_Wire
from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape

import freeform_fit
import nurbs

KIND = ("plane", "cylinder", "cone", "sphere", "torus")

# True STEP solid volumes (computed from the ABC STEP files), the reference a rebuilt solid is
# judged against. The mesh volume is not: a tessellation loses a percent or two on coarse meshes.
try:
    TRUE_VOLUMES = json.loads(Path("D:/abc-data/solid_truth_volumes.json").read_text())
except Exception:
    TRUE_VOLUMES = {}


def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


# ── surfaces ─────────────────────────────────────────────────────────────────

def truth_params(kind, s, centre):
    """fit_truth sidecar row (surface_targets layout) -> gaussMap param format."""
    P, A = list(s[0:3]), list(unit(s[3:6])) if kind != "sphere" else None
    if kind == "plane":
        return {"origin": P, "normal": A}
    if kind == "cylinder":
        return {"axisPoint": P, "axis": A, "radius": s[6]}
    if kind == "sphere":
        return {"center": P, "radius": s[6]}
    if kind == "torus":
        return {"center": P, "axis": A, "majorRadius": s[6], "minorRadius": s[7]}
    a = np.asarray(A)
    if (np.asarray(centre) - np.asarray(P)) @ a < 0:
        a = -a
    return {"apex": P, "axis": list(a), "halfAngle": s[6]}


def dist_normal(kind, p, X):
    """Signed distance and unit gradient of points X (n,3) to a surface. Cones are double cones.
    A bspline's is the distance along its natural normal (Su x Sv) at the closest point."""
    X = np.atleast_2d(np.asarray(X, float))
    if kind == "bspline":
        return nurbs.surface_of(p).signed_distance(X)
    if kind == "plane":
        n = unit(p["normal"])
        return (X - p["origin"]) @ n, np.broadcast_to(n, X.shape).copy()
    if kind == "sphere":
        v = X - p["center"]
        r = np.linalg.norm(v, axis=1)
        return r - p["radius"], v / np.maximum(r, 1e-300)[:, None]
    a = unit(p["axis"])
    o = np.asarray(p["axisPoint"] if kind == "cylinder" else p["apex"] if kind == "cone" else p["center"], float)
    v = X - o
    z = v @ a
    rad = v - np.outer(z, a)
    rho = np.linalg.norm(rad, axis=1)
    u = rad / np.maximum(rho, 1e-300)[:, None]
    if kind == "cylinder":
        return rho - p["radius"], u
    if kind == "cone":
        ca, sa = math.cos(p["halfAngle"]), math.sin(p["halfAngle"])
        s = np.where(z >= 0, 1.0, -1.0)
        return rho * ca - np.abs(z) * sa, u * ca - (s * sa)[:, None] * a
    w = v - p["majorRadius"] * u
    wl = np.linalg.norm(w, axis=1)
    return wl - p["minorRadius"], w / np.maximum(wl, 1e-300)[:, None]


def project_onto(surfs, x, scale, iters=30):
    """Nearest point to x lying on every surface in surfs (Gauss-Newton, min-norm steps).

    Tangent surfaces make the normals nearly parallel; the damped min-norm step then simply moves
    x onto the shared tangency instead of blowing up."""
    x = np.asarray(x, float).copy()
    for _ in range(iters):
        d = np.array([dist_normal(k, p, x)[0][0] for k, p in surfs])
        if np.max(np.abs(d)) < 1e-12 * scale:
            break
        J = np.array([dist_normal(k, p, x)[1][0] for k, p in surfs])
        step = J.T @ np.linalg.solve(J @ J.T + 1e-10 * np.eye(len(surfs)), d)
        x -= step
        if np.linalg.norm(step) < 1e-13 * scale:
            break
    return x


def seam_away(axis, mean_dir):
    """An X direction perpendicular to axis pointing AWAY from the patch, so the periodic seam
    (u = 0) sits on the far side of the surface from the face."""
    a = unit(axis)
    m = np.asarray(mean_dir, float) - (np.asarray(mean_dir, float) @ a) * a
    if np.linalg.norm(m) < 1e-9:
        ref = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        m = ref - (ref @ a) * a
    return -unit(m)


TAU = 2 * math.pi

# Off: trimming tori and spheres to their patch ranges was measured WORSE on the 31 analytic parts
# (native closed 28 -> 22; torus faces went from ~7x to ~0.08x their STEP area), which points at the
# torus boundary loops being oriented or parameterised wrongly, not at the surface domain.
TRIM_CLOSED_SURFACES = False

# How the raw shell is repaired (pcurves, seams, degenerated apex edges):
#   faces  ShapeFix_Face on every face, all sharing ONE ShapeBuild_ReShape context that is applied to
#          the whole shell at the end, so an edge one face replaces is replaced in its neighbour too
#   shape  one ShapeFix_Shape over the shell
# Measured: a torus face whose boundary crosses the tube seam repairs to exactly its STEP area on its
# own (00032949_19 face 7: 1.000x, 00062231_2 face 12: 1.000x), while ShapeFix_Shape over the shell
# turned the same faces into ~7x their area.
REPAIR_MODE = "shape"
#   inplace (default) ShapeFix_Edge.FixAddPCurve writes each edge's curve on each of its faces onto
#          the EXISTING shared edge, then per face only FixMissingSeam / FixPeriodicDegenerated. No
#          context, no wire rebuild. "wires" (ShapeFix_Wire.FixEdgeCurves) repaired regions correctly
#          but returned COPIES of shared edges: 00158924_0 went from 52 edges to 60, all 10 free edges
#          new copies; it also crashed natively on 00157920_30.
#   wires  (default) no ShapeBuild_ReShape at all: per face, ShapeFix_Wire.FixEdgeCurves adds every
#          edge's curve on that face (the edge is shared, so both faces update the same edge), then
#          only FixMissingSeam and FixPeriodicDegenerated, whose new edges belong to that face alone.
#          Every captured raw face repairs to its exact STEP area this way (torus 48.85 = STEP,
#          cylinder 1016 = STEP); the 6-24x torus faces appeared only when a face was rebuilt through
#          the context afterwards to pick up its neighbours' edge replacements.


def arc_of_triangles(tri_angles):
    """Smallest arc (start, span, full) containing the union of each triangle's own arc.

    From triangles rather than vertex angles: a patch's opening is an arc no triangle spans, while a
    closed ring leaves no gap at all (js/brepBuilder.js arcOfTriangles)."""
    spans = []
    for t in tri_angles:
        a = sorted(x % TAU for x in t)
        gaps = [a[1] - a[0], a[2] - a[1], a[0] + TAU - a[2]]
        gi = int(np.argmax(gaps))
        st = a[(gi + 1) % 3]
        sp = TAU - gaps[gi]
        if st + sp <= TAU:
            spans.append((st, st + sp))
        else:
            spans += [(st, TAU), (0.0, st + sp - TAU)]
    if not spans:
        return 0.0, TAU, True
    spans.sort()
    merged = [list(spans[0])]
    for st, en in spans[1:]:
        if st <= merged[-1][1] + 1e-12:
            merged[-1][1] = max(merged[-1][1], en)
        else:
            merged.append([st, en])
    best_len, best_end = 0.0, 0.0
    for i, (st, en) in enumerate(merged):
        nxt = merged[i + 1][0] if i + 1 < len(merged) else merged[0][0] + TAU
        if nxt - en > best_len:
            best_len, best_end = nxt - en, nxt
    if best_len <= 1e-9:
        return 0.0, TAU, True
    return best_end % TAU, TAU - best_len, False


def trimmed_to_patch(kind, surf, p, tri_pts):
    """A torus or sphere restricted to the patch's own parameter ranges.

    Both are closed in two directions (a sphere through its poles), so a single boundary loop on
    them separates TWO regions, and the repair pass kept choosing the far one: every oversized face
    on the 31 analytic parts was a torus bounded by one loop, rebuilt at 6x to 24x its patch area.
    Trimming the surface to the patch's ranges (plus a margin) leaves only the region the triangles
    cover. A direction the patch wraps completely stays periodic."""
    from OCP.Geom import Geom_RectangularTrimmedSurface
    ax = surf.Position()
    o = np.array([ax.Location().X(), ax.Location().Y(), ax.Location().Z()])
    z = np.array([ax.Direction().X(), ax.Direction().Y(), ax.Direction().Z()])
    x = np.array([ax.XDirection().X(), ax.XDirection().Y(), ax.XDirection().Z()])
    y = np.array([ax.YDirection().X(), ax.YDirection().Y(), ax.YDirection().Z()])
    d = tri_pts - o                                      # (m, 3, 3)
    dz, dx, dy = d @ z, d @ x, d @ y
    u = np.arctan2(dy, dx)
    rho = np.hypot(dx, dy)
    if kind == "torus":
        v = np.arctan2(dz, rho - p["majorRadius"])
        u0, us, ufull = arc_of_triangles(u)
        v0, vs, vfull = arc_of_triangles(v)
    else:
        lat = np.arctan2(dz, rho)
        u0, us, ufull = arc_of_triangles(u)
        v0, vs, vfull = float(lat.min()), float(lat.max() - lat.min()), False
    mu, mv = 0.02 * us + 1e-3, 0.02 * vs + 1e-3
    utrim = not ufull and us + 2 * mu < TAU
    vtrim = kind == "sphere" or (not vfull and vs + 2 * mv < TAU)
    U1, U2 = u0 - mu, u0 + us + mu
    V1, V2 = v0 - mv, v0 + vs + mv
    if kind == "sphere":
        V1, V2 = max(-math.pi / 2, V1), min(math.pi / 2, V2)
    if utrim and vtrim:
        return Geom_RectangularTrimmedSurface(surf, U1, U2, V1, V2)
    if utrim:
        return Geom_RectangularTrimmedSurface(surf, U1, U2, True)
    if vtrim:
        return Geom_RectangularTrimmedSurface(surf, V1, V2, False)
    return surf


def geom_surface(kind, p, patch_pts):
    if kind == "bspline":
        return nurbs.to_occ(p)
    c = patch_pts.mean(0)
    if kind == "plane":
        # Not re-anchored like the cylinder: moving the plane origin beside the patch was measured on
        # all 31 parts, did not fix 00104382_0 (gmsh, fitted: 4 planes UnorientableShape) and broke the
        # same part with STEP planes. Left as fitted.
        return Geom_Plane(gp_Pln(gp_Pnt(*p["origin"]), gp_Dir(*unit(p["normal"]))))
    if kind == "sphere":
        o = np.asarray(p["center"], float)
        mean = unit((patch_pts - o).mean(0))
        ref = np.array([1.0, 0, 0]) if abs(mean[0]) < 0.9 else np.array([0, 1.0, 0])
        axis = unit(np.cross(mean, ref))                  # poles away from the patch
        x = seam_away(axis, mean)
        return Geom_SphericalSurface(gp_Ax3(gp_Pnt(*o), gp_Dir(*axis), gp_Dir(*x)), p["radius"])
    a = unit(p["axis"])
    o = np.asarray(p["axisPoint"] if kind == "cylinder" else p["apex"] if kind == "cone" else p["center"], float)
    x = seam_away(a, c - o)
    if kind == "cone":
        if ((patch_pts - o) @ a).mean() < 0:
            a = -a
            x = seam_away(a, c - o)
        ax = gp_Ax3(gp_Pnt(*o), gp_Dir(*a), gp_Dir(*x))
        return Geom_ConicalSurface(ax, p["halfAngle"], 0.0)
    if kind == "cylinder":
        # Frame origin on the axis beside the patch. Any point of the axis describes the same
        # cylinder, but the face built on it did not come out the same: 00038169_0 face 6 rebuilt at
        # 1.000x its STEP area from the STEP axis point and 0.320x from a fitted one 8.7 units further
        # along an identical axis. Anchoring the frame at the patch makes equivalent fits equivalent.
        o = o + float((c - o) @ a) * a
        return Geom_CylindricalSurface(gp_Ax3(gp_Pnt(*o), gp_Dir(*a), gp_Dir(*x)), p["radius"])
    ax = gp_Ax3(gp_Pnt(*o), gp_Dir(*a), gp_Dir(*x))
    return Geom_ToroidalSurface(ax, p["majorRadius"], p["minorRadius"])


# Chain points that double back are dropped before an edge is interpolated through them. When an
# automatic segment's boundary runs off the true edge (0.2-2.5% of the diagonal on the Onshape set),
# its vertices project onto the two surfaces' intersection OUT OF ORDER and the spline folds: 85% of
# the closed-but-invalid Onshape rebuilds had such a chain (5% of the valid ones, 1% with true
# segments), and the fold is BRepCheck's SelfIntersectingWire + UnorientableShape. Measured on 275
# Onshape parts (s13 tm-ring1): recreated 196 -> 211, none lost, the 8 OCCT exceptions gone; no change
# on true segments (Onshape 254, fill_eval 30) or fill_eval tm-ring1 (10). A 90 deg limit gained 212
# but lost 14 parts with true segments: coarse rings of 4-6 vertices turn by 60-90 deg at every point.
UNFOLD_MAX_TURN = 135.0


def unfold_chain(pts, closed, info, max_turn=None):
    """Drop interior points where the polyline turns by more than max_turn degrees, worst first.
    The ends of an open chain are corners and always stay; a closed chain keeps at least 3 points."""
    max_turn = UNFOLD_MAX_TURN if max_turn is None else max_turn
    pts = [np.asarray(q, float) for q in pts]
    removed = 0
    while len(pts) > (3 if closed else 2):
        n = len(pts)
        worst, wi = max_turn, None
        for i in (range(n) if closed else range(1, n - 1)):
            u, v = pts[i] - pts[i - 1], pts[(i + 1) % n] - pts[i]
            nu, nv = np.linalg.norm(u), np.linalg.norm(v)
            if nu == 0 or nv == 0:
                continue
            t = math.degrees(math.acos(max(-1.0, min(1.0, float(u @ v) / (nu * nv)))))
            if t > worst:
                worst, wi = t, i
        if wi is None:
            break
        del pts[wi]
        removed += 1
    if removed:
        info["unfold_points_removed"] += removed
        info["unfolded_chains"] += 1
    return pts


# ── mesh ─────────────────────────────────────────────────────────────────────

class Mesh:
    def __init__(self, pos, fid):
        tri = pos.reshape(-1, 3, 3).astype(np.float64)
        keys = {}
        idx = np.empty((len(tri), 3), np.int64)
        verts = []
        for t in range(len(tri)):
            for k in range(3):
                key = tuple(pos[t * 9 + k * 3: t * 9 + k * 3 + 3])
                i = keys.get(key)
                if i is None:
                    i = keys[key] = len(verts)
                    verts.append(tri[t, k])
                idx[t, k] = i
        self.V = np.array(verts)
        good = (idx[:, 0] != idx[:, 1]) & (idx[:, 1] != idx[:, 2]) & (idx[:, 0] != idx[:, 2])
        self.F = fid[good].astype(np.int64)
        self.src = np.nonzero(good)[0]             # each kept triangle's index in the input soup
        T0 = idx[good]
        self.partner = self._pair(T0)
        self.T = self._orient(T0)
        self.partner = self._pair(self.T)          # again, with the winding now consistent
        self.body = self._bodies()
        a, b, c = self.V[self.T[:, 0]], self.V[self.T[:, 1]], self.V[self.T[:, 2]]
        self.N = np.cross(b - a, c - a)
        self.diag = float(np.linalg.norm(self.V.max(0) - self.V.min(0)))

    def _pair(self, T):
        # (triangle, undirected edge) -> the triangle across that edge IN THE SAME SOLID.
        #
        # An edge used by two triangles pairs them. An edge used by more - two solids meshed where
        # they touch, 148 such edges on 00005430_2 - is resolved radially: around the edge, a triangle
        # traversing it backwards has material on its counter-clockwise side and one traversing it
        # forwards on its clockwise side, so going counter-clockwise each backward triangle pairs
        # with the next forward one. Coincident triangles tie on angle and the forward one sorts
        # first, so a zero-thickness gap between touching bodies is never paired as material.
        # Checked against STEP solid counts: 00005430_2 2/2, 00072333_2 5/5, 00157920_30 2/2.
        V = self.V
        edge_tris = defaultdict(list)
        for t, tri in enumerate(T):
            for k in range(3):
                u, v = int(tri[k]), int(tri[(k + 1) % 3])
                edge_tris[(min(u, v), max(u, v))].append((t, 1 if u < v else -1))
        partner = {}
        for key, lst in edge_tris.items():
            if len(lst) == 2:
                (t0, _), (t1, _) = lst
                partner[(t0, key)] = t1
                partner[(t1, key)] = t0
                continue
            if len(lst) < 2:
                continue
            ea, eb = key
            e = unit(V[eb] - V[ea])
            ref = np.array([1.0, 0, 0]) if abs(e[0]) < 0.9 else np.array([0, 1.0, 0])
            b1 = unit(np.cross(e, ref))
            b2 = np.cross(e, b1)
            items = []
            for t, d in lst:
                w = [x for x in T[t] if x != ea and x != eb][0]
                r = V[w] - V[ea]
                items.append((float(np.arctan2(r @ b2, r @ b1)), 0 if d == 1 else 1, t, d))
            items.sort()
            n = len(items)
            for i in range(n):
                if items[i][3] != -1:
                    continue
                nxt = items[(i + 1) % n]
                if nxt[3] == 1:
                    partner[(items[i][2], key)] = nxt[2]
                    partner[(nxt[2], key)] = items[i][2]
        return partner

    def _bodies(self):
        from collections import deque
        T = self.T
        body = -np.ones(len(T), np.int64)
        nb = 0
        for s0 in range(len(T)):
            if body[s0] >= 0:
                continue
            body[s0] = nb
            q = deque([s0])
            while q:
                t = q.popleft()
                tri = T[t]
                for k in range(3):
                    u, v = int(tri[k]), int(tri[(k + 1) % 3])
                    n = self.partner.get((t, (min(u, v), max(u, v))))
                    if n is not None and body[n] < 0:
                        body[n] = nb
                        q.append(n)
            nb += 1
        return body

    def _orient(self, T):
        # Consistent winding across paired triangles, then outward (positive signed volume) per body.
        # The loop walk and each face's outward side both read triangle winding.
        from collections import deque
        T = T.copy()
        seen = np.zeros(len(T), bool)
        V = self.V
        for s0 in range(len(T)):
            if seen[s0]:
                continue
            comp = [s0]
            seen[s0] = True
            q = deque([s0])
            while q:
                t = q.popleft()
                i, j, k = T[t]
                for u, v in ((i, j), (j, k), (k, i)):
                    n = self.partner.get((t, (min(u, v), max(u, v))))
                    if n is None or seen[n]:
                        continue
                    a_, b_, c_ = T[n]
                    if (u, v) in ((a_, b_), (b_, c_), (c_, a_)):   # same direction: neighbour is flipped
                        T[n] = T[n][::-1]
                    seen[n] = True
                    comp.append(n)
                    q.append(n)
            c = np.array(comp)
            pa, pb, pc = V[T[c, 0]], V[T[c, 1]], V[T[c, 2]]
            if np.einsum("ij,ij->i", pa, np.cross(pb, pc)).sum() < 0:
                T[c] = T[c][:, ::-1]
        return T

    def volume(self):
        a, b, c = self.V[self.T[:, 0]], self.V[self.T[:, 1]], self.V[self.T[:, 2]]
        return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


# ── topology method ──────────────────────────────────────────────────────────

# ── exact edges ──────────────────────────────────────────────────────────────
#
# An edge between two surfaces that cross at a real angle can be their exact intersection curve
# instead of a spline through projected boundary vertices: a line, circle, ellipse or conic from
# GeomAPI_IntSS (OCCT solves quadric pairs analytically), trimmed at the two corner vertices. The mesh
# run only picks the branch and the arc. Measured on the 275 Onshape parts with the surfaces AS
# FITTED: 196 -> 192 - the corners' surfaces often miss one common point (4.3% of corners in the
# invalid parts), so the exact curve cannot reach the vertex, and exact and interpolated edges on one
# face cross near the corners, and on sliver faces the near-coincident corners give faces up to 1e9
# times their patch area. After constrained_refit, as a repair-loop move, it was kept on 40 of 129
# failing parts without changing the recreated count. Off by default.
EXACT_MIN_DIHEDRAL = 30.0      # degrees; nearer tangent than this the pair keeps the projected run


def exact_edge(ga, gb, sa, sb, pts, v0, v1, x0, x1, closed, scale, info):
    """The edge from corner x0 to corner x1 along the exact intersection of surfaces a and b, or None
    (near tangent, no intersection, a corner off the curve, an ambiguous branch)."""
    from OCP.GeomAPI import GeomAPI_IntSS, GeomAPI_ProjectPointOnCurve
    mid = pts[len(pts) // 2]
    n1 = dist_normal(sa[0], sa[1], mid)[1][0]
    n2 = dist_normal(sb[0], sb[1], mid)[1][0]
    if math.degrees(math.acos(min(1.0, abs(float(n1 @ n2))))) < EXACT_MIN_DIHEDRAL:
        info["exact_skip_tangent"] += 1
        return None
    inter = GeomAPI_IntSS(ga, gb, 1e-9 * scale)
    if not inter.IsDone() or inter.NbLines() == 0:
        info["exact_no_intersection"] += 1
        return None
    probe = pts[::max(1, len(pts) // 12)]
    best = None
    for i in range(1, inter.NbLines() + 1):
        c = inter.Line(i)
        ds = []
        for q in probe:
            pr = GeomAPI_ProjectPointOnCurve(gp_Pnt(*q), c)
            ds.append(pr.LowerDistance() if pr.NbPoints() else 1e300)
        m = float(np.median(ds))
        if best is None or m < best[0]:
            best = (m, c)
    if best[0] > 1e-5 * scale:
        info["exact_branch_far"] += 1
        return None
    c = best[1]
    if c.DynamicType().Name() == "Geom_TrimmedCurve":
        c = c.BasisCurve()              # quadric intersections come back as trimmed conics; arcs may wrap

    def par(q):
        pr = GeomAPI_ProjectPointOnCurve(gp_Pnt(*q), c)
        if not pr.NbPoints():
            return None, 1e300
        return pr.LowerDistanceParameter(), pr.LowerDistance()

    t0, d0 = par(x0)
    t1, d1 = par(x1)
    if t0 is None or t1 is None or max(d0, d1) > 1e-6 * scale:
        info["exact_corner_off_curve"] += 1
        return None
    ts = [t for t in (par(q)[0] for q in pts[1:-1]) if t is not None]
    bb = BRep_Builder()
    for v, d in ((v0, d0), (v1, d1)):
        if d > 1e-7:                    # a corner a hair off the curve is covered by its tolerance
            bb.UpdateVertex(v, max(BRep_Tool.Tolerance_s(v), 1.01 * d + 1e-7))
    if c.IsPeriodic():
        P = c.Period()
        if closed:
            inc = np.diff([t0] + ts + [t0])
            fwd = ((inc + P / 2) % P - P / 2).sum() >= 0
            a_, b_ = t0, t0 + P
        else:
            span = (t1 - t0) % P
            fwd = np.mean([((t - t0) % P) < span for t in ts] or [True]) >= 0.5
            a_, b_ = (t0, t0 + span) if fwd else (t1, t1 + (P - span))
        e = (BRepBuilderAPI_MakeEdge(c, v0, v1, a_, b_) if fwd else
             BRepBuilderAPI_MakeEdge(c.Reversed(), v0, v1, c.ReversedParameter(b_), c.ReversedParameter(a_)))
    else:
        if closed:
            info["exact_closed_open_curve"] += 1
            return None
        lo, hi = min(t0, t1), max(t0, t1)
        if ts and np.mean([lo - 1e-9 <= t <= hi + 1e-9 for t in ts]) < 0.5:
            info["exact_chain_outside_span"] += 1
            return None
        e = (BRepBuilderAPI_MakeEdge(c, v0, v1, t0, t1) if t0 < t1 else
             BRepBuilderAPI_MakeEdge(c.Reversed(), v0, v1, c.ReversedParameter(t0), c.ReversedParameter(t1)))
    if not e.IsDone():
        info["exact_makeedge_failed"] += 1
        return None
    info["exact_edges"] += 1
    return e


def mesh_topology(mesh, force_tris=None):
    """The B-rep's combinatorics from the labelled mesh alone: corners (vertex, body) where >= 3
    patches meet, and chains (a, b, path) - each run of boundary between patches a and b, corner to
    corner (b = -1 along an open mesh boundary). Surfaces play no part. force_tris: triangles every
    vertex of which is a corner too (facet faces: each of their edges is one chain)."""
    T, F = mesh.T, mesh.F
    info = defaultdict(int)
    # Boundary half-edges from the triangle pairs (not a directed-edge dictionary, which two touching
    # solids overwrite), and every vertex taken PER BODY: two solids meeting at a point get their own
    # vertices, projected only onto their own surfaces.
    partner, tbody = mesh.partner, mesh.body
    fb = {}
    faces_at = defaultdict(set)
    bnd = {}                                          # (a, u, v) -> b, in a's winding
    open_at = set()
    for t, tri in enumerate(T):
        a = int(F[t])
        bt = int(tbody[t])
        fb[a] = bt
        for k in range(3):
            u, v = int(tri[k]), int(tri[(k + 1) % 3])
            faces_at[(u, bt)].add(a)
            n = partner.get((t, (min(u, v), max(u, v))))
            b = int(F[n]) if n is not None else -1
            if b != a:
                bnd[(a, u, v)] = b
                if b == -1:
                    open_at.update(((u, bt), (v, bt)))
    corners = {key for key, fs in faces_at.items() if len(fs) >= 3}
    corners |= {key for key in open_at if len(faces_at[key]) >= 2}
    if force_tris is not None:
        corners |= {(int(v), int(tbody[t])) for t in force_tris for v in T[t]}

    # chains: runs of boundary edges between one ordered pair, traversed along the lower face's
    # winding (open runs along their only face), cut at corners
    succ = defaultdict(list)
    for (a, u, v), b in bnd.items():
        if b == -1 or a < b:
            succ[(a, b, u)].append(v)
    used = set()
    chains = []

    def walk(a, b, start):
        path = [start]
        cur = start
        while True:
            nxt = [v for v in succ.get((a, b, cur), []) if (a, b, cur, v) not in used]
            if not nxt:
                return path
            v = nxt[0]
            used.add((a, b, cur, v))
            path.append(v)
            cur = v
            if (v, fb[a]) in corners:
                return path

    for (a, b, u) in list(succ.keys()):
        if (u, fb[a]) in corners:
            for v in succ[(a, b, u)]:
                if (a, b, u, v) not in used:
                    used.add((a, b, u, v))
                    path = [u, v] if (v, fb[a]) in corners else [u] + walk(a, b, v)
                    chains.append((a, b, path))
    for (a, b, u), vs in list(succ.items()):
        for v in vs:
            if (a, b, u, v) in used:
                continue
            used.add((a, b, u, v))
            path = [u] + walk(a, b, v)
            if path[-1] != path[0]:
                info["broken_loop"] += 1
            corners.add((path[0], fb[a]))             # a closed run gets a corner of its own
            chains.append((a, b, path))

    return {"fb": fb, "faces_at": faces_at, "corners": corners, "chains": chains, "info": info}


def build_topo(mesh, surfaces, _faces=None, exact=False, facet_labels=None):
    """surfaces: face id -> (kind, params). Returns (shape, info). _faces, a dict, receives
    "face_of": patch -> its final face (or the pieces the repair split it into). exact: edges between
    surfaces crossing at EXACT_MIN_DIHEDRAL or more are their exact intersection (exact_edge).
    facet_labels: labels of ONE triangle each (facet_fallback), built as a plane through its three
    corners with straight edges; corners stay where the analytic surfaces meeting there put them."""
    V, T, F, scale = mesh.V, mesh.T, mesh.F, mesh.diag
    info = defaultdict(int)
    facet_labels = facet_labels or set()

    topo = mesh_topology(mesh, np.nonzero(np.isin(F, list(facet_labels)))[0] if facet_labels else None)
    fb, faces_at, corners, chains = topo["fb"], topo["faces_at"], topo["corners"], topo["chains"]
    for k, v in topo["info"].items():
        info[k] += v
    patch_pts = {f: V[np.unique(T[F == f])] for f in np.unique(F)}
    surf_of = lambda f: surfaces.get(f)

    # corners
    vertex, corner_pos = {}, {}
    for c in corners:
        ss = [surf_of(f) for f in faces_at[c] if surf_of(f)]
        x = project_onto(ss, V[c[0]], scale) if ss else V[c[0]]
        corner_pos[c] = x
        vertex[c] = BRepBuilderAPI_MakeVertex(gp_Pnt(*x)).Vertex()

    # edges
    edges = []                                        # (a, b, start corner, end corner, TopoDS_Edge)
    for a, b, path in chains:
        ss = [s for s in (surf_of(a), surf_of(b) if b >= 0 else None) if s]
        # Interior points on BOTH surfaces; the ends ARE the corner vertices (projected onto every
        # surface meeting there), or MakeEdge rejects the vertex as off the curve.
        bd = fb[a]
        pts = [corner_pos[(i, bd)] if (i, bd) in corner_pos else (project_onto(ss, V[i], scale) if ss else V[i]) for i in path]
        closed = path[0] == path[-1]
        both_planes = len(ss) == 2 and ss[0][0] == "plane" and ss[1][0] == "plane"
        # A facet's edges are its own straight sides (every facet vertex is a corner, so the chain is
        # one mesh edge): a chord where it meets a curved face, which the repair covers by tolerance.
        straight = both_planes or a in facet_labels or b in facet_labels
        try:
            e = None
            if exact and len(ss) == 2 and not straight and "bspline" not in (ss[0][0], ss[1][0]):
                try:
                    e = exact_edge(geom_surface(*ss[0], patch_pts[a]), geom_surface(*ss[1], patch_pts[b]),
                                   ss[0], ss[1], [np.asarray(q) for q in pts], vertex[(path[0], bd)],
                                   vertex[(path[-1], bd)], pts[0], pts[-1], closed, scale, info)
                except Exception:
                    info["exact_exception"] += 1
            if e is not None:
                pass
            elif straight and not closed:
                e = BRepBuilderAPI_MakeEdge(vertex[(path[0], bd)], vertex[(path[-1], bd)])
            else:
                body = pts[:-1] if closed else pts
                clean = [body[0]]
                for q in body[1:]:
                    if np.linalg.norm(q - clean[-1]) > 1e-9 * scale:
                        clean.append(q)
                if not closed:
                    clean[-1] = pts[-1]
                clean = unfold_chain(clean, closed, info)
                n = len(clean)
                if n < 2 or (closed and n < 3):
                    info["edge_too_short"] += 1
                    continue
                arr = TColgp_HArray1OfPnt(1, n)
                for i, q in enumerate(clean, 1):
                    arr.SetValue(i, gp_Pnt(*q))
                it = GeomAPI_Interpolate(arr, closed, 1e-9 * scale)
                it.Perform()
                if not it.IsDone():
                    info["interp_failed"] += 1
                    continue
                crv = it.Curve()
                e = BRepBuilderAPI_MakeEdge(crv, vertex[(path[0], bd)], vertex[(path[-1], bd)],
                                            crv.FirstParameter(), crv.LastParameter())
            if not e.IsDone():
                info["edge_failed"] += 1
                continue
            edges.append((a, b, (path[0], bd), (path[-1], bd), e.Edge()))
        except Exception:
            info["edge_exception"] += 1

    # faces
    by_face = defaultdict(list)                       # face -> [(start, end, oriented edge)]
    for a, b, s, t, e in edges:
        by_face[a].append((s, t, e))
        if b >= 0:
            by_face[b].append((t, s, e.Reversed()))
    builder = BRep_Builder()
    shell = TopoDS_Shell()
    builder.MakeShell(shell)
    face_of = {}                                      # patch -> FORWARD face (repair works on these)
    flip_of = {}                                      # patch -> reversed in the shell to point outward
    for f, pts in patch_pts.items():
        sp = surf_of(f)
        facet = f in facet_labels
        if not sp and not facet:
            info["face_no_surface"] += 1
            continue
        if facet:
            # The plane through the facet's three corners, normal by the mesh winding: outward.
            t = int(np.nonzero(F == f)[0][0])
            q = [np.asarray(corner_pos[(int(v), fb[f])]) for v in T[t]]
            nrm = np.cross(q[1] - q[0], q[2] - q[0])
            if not by_face.get(f) or np.linalg.norm(nrm) <= 1e-12 * scale * scale:
                info["facet_degenerate"] += 1
                continue
            kind, p = "facet", None
            surf = Geom_Plane(gp_Pnt(*q[0]), gp_Dir(*unit(nrm)))
        else:
            kind, p = sp
            surf = geom_surface(kind, p, pts)
        base_surf = surf
        if TRIM_CLOSED_SURFACES and kind in ("torus", "sphere") and by_face.get(f):
            try:
                surf = trimmed_to_patch(kind, surf, p, V[T[F == f]])
                info["trimmed_faces"] += 1
            except Exception:
                info["trim_failed"] += 1
        if not by_face.get(f):
            # A patch with no boundary is a whole closed surface (a ball, a ring): its face is the
            # surface's natural bounds.
            if kind in ("sphere", "torus"):
                cen_t = V[T[F == f]].mean(1)
                _, g = dist_normal(kind, p, cen_t)
                inward = float(np.einsum("ij,ij->i", g, mesh.N[F == f]).sum()) < 0
                nat = BRepBuilderAPI_MakeFace(base_surf.UReversed() if inward else base_surf, 1e-7)
                if nat.IsDone():
                    fc = nat.Face()
                    flip_of[f] = False
                    builder.Add(shell, fc)
                    face_of[f] = fc
                    info["faces"] += 1
                    info["natural_faces"] += 1
                    continue
            info["face_no_edges"] += 1
            continue
        # loops in mesh winding: follow end corner -> an unused edge starting there
        todo = list(by_face[f])
        wires = []
        while todo:
            s, t, e = todo.pop(0)
            loop = [e]
            cur = t
            while cur != s:
                k = next((i for i, (s2, _, _) in enumerate(todo) if s2 == cur), None)
                if k is None:
                    break
                s2, t2, e2 = todo.pop(k)
                loop.append(e2)
                cur = t2
            if cur != s:
                info["open_loop"] += 1
            w = TopoDS_Wire()
            builder.MakeWire(w)
            for e2 in loop:
                builder.Add(w, e2)
            wires.append(w)
        # mesh winding is counter-clockwise about the OUTWARD normal; OCCT wants it counter-clockwise
        # about the surface normal, so a surface facing inward gets its wires reversed and the face
        # itself reversed afterwards.
        # Does the surface's natural normal point out of the solid? Summed over the patch's own
        # triangles: the analytic gradient at each triangle centroid (these fitted surfaces' natural
        # normals all follow the gradient) against the triangle's outward normal. A single test at
        # the patch's vertex centroid, projected onto the surface, was wrong for strongly curved
        # patches - the centroid of a half-tube torus patch sits near the tube's centre line, where
        # the normal is arbitrary - and left 00062231_2's 180 deg fillet facing into the solid
        # (19% volume error with every face area correct).
        if facet:
            outward = True
        else:
            tri_c = V[T[F == f]].mean(1)
            _, g_tri = dist_normal(kind, p, tri_c)
            outward = float(np.einsum("ij,ij->i", g_tri, mesh.N[F == f]).sum()) >= 0
        if kind == "cone":
            ax_c = np.asarray(p["axis"], float)
            apex_c = np.asarray(p["apex"], float)
            if ((pts - apex_c) @ ax_c).mean() < 0:
                pass   # geom_surface flips the cone axis to open toward the patch; the gradient is axis-sign free
        nat = None
        # A surface whose natural normal points INTO the solid is replaced by its U-reversed copy, so
        # that every face is FORWARD with the mesh-order wires. Reversing the FACE instead made the
        # repair compose that orientation twice, and on a torus (closed in both directions) that
        # selects the complement region: 00032949_19's torus faces rebuilt at 12x their STEP area.
        if not outward:
            surf = surf.UReversed()
            info["reversed_surfaces"] += 1
        face = TopoDS_Face()
        builder.MakeFace(face, surf, 1e-7)
        for w in wires:
            builder.Add(face, w)
        builder.Add(shell, face)
        face_of[f] = face
        flip_of[f] = False
        info["faces"] += 1
        if facet:
            info["facet_faces"] += 1
    # ONE repair pass over the whole shell. ShapeFix_Face per face replaces edges with copies (adding
    # pcurves, seams) and the neighbouring face keeps the original: 00082567_1 went from 4 edges, each
    # on two faces, to 12 edges with 10 on one face. ShapeFix_Shape records every replacement in one
    # context and applies it to all faces, so shared edges stay shared.
    if REPAIR_MODE == "inplace":
        from OCP.ShapeFix import ShapeFix_Edge
        fe = ShapeFix_Edge()
        for f, fc in face_of.items():
            ex = TopExp_Explorer(fc, TopAbs_EDGE)
            while ex.More():
                e = TopoDS.Edge_s(ex.Current())
                ex.Next()
                try:
                    if not BRep_Tool.Degenerated_s(e):
                        fe.FixAddPCurve(e, fc, False, 1e-7)
                except Exception:
                    info["pcurve_failed"] += 1
        result = TopoDS_Shell()
        builder.MakeShell(result)
        for f, fc in list(face_of.items()):
            out = fc
            try:
                sf = ShapeFix_Face(fc)
                sf.FixMissingSeam()
                sf.FixPeriodicDegenerated()
                out = TopoDS.Face_s(sf.Face())
            except Exception:
                info["repair_failed"] += 1
            builder.Add(result, TopoDS.Face_s(out.Reversed()) if flip_of.get(f) else out)
            face_of[f] = out
        ctx = None
    elif REPAIR_MODE == "wires":
        from OCP.ShapeFix import ShapeFix_Wire
        result = TopoDS_Shell()
        builder.MakeShell(result)
        for f, fc in list(face_of.items()):
            try:
                out = TopoDS_Face()
                builder.MakeFace(out, BRep_Tool.Surface_s(fc), 1e-7)
                wx = TopExp_Explorer(fc, TopAbs_WIRE)
                while wx.More():
                    fw = ShapeFix_Wire(TopoDS.Wire_s(wx.Current()), fc, 1e-7)
                    fw.FixEdgeCurves()
                    builder.Add(out, fw.Wire())
                    wx.Next()
                sf = ShapeFix_Face(out)
                sf.FixMissingSeam()
                sf.FixPeriodicDegenerated()
                out = TopoDS.Face_s(sf.Face())
            except Exception:
                info["repair_failed"] += 1
                out = fc
            builder.Add(result, TopoDS.Face_s(out.Reversed()) if flip_of.get(f) else out)
            face_of[f] = out
        ctx = None
    elif REPAIR_MODE == "pre":
        # Give the repair nothing to replace: every edge gets its curve on each of its faces in
        # place, then edge and curves are made to agree in parameter (SameParameter, in place),
        # and only then does ShapeFix_Face run.
        from OCP.ShapeFix import ShapeFix_Edge
        from OCP.BRepLib import BRepLib
        from OCP.ShapeBuild import ShapeBuild_ReShape
        from OCP.TopTools import TopTools_IndexedMapOfShape
        raw_edges = TopTools_IndexedMapOfShape()
        TopExp.MapShapes_s(shell, TopAbs_EDGE, raw_edges)
        fe = ShapeFix_Edge()
        for fc in face_of.values():
            ex = TopExp_Explorer(fc, TopAbs_EDGE)
            while ex.More():
                e = TopoDS.Edge_s(ex.Current())
                ex.Next()
                if not BRep_Tool.Degenerated_s(e):
                    try:
                        fe.FixAddPCurve(e, fc, False, 1e-7)
                    except Exception:
                        info["pcurve_failed"] += 1
        try:
            BRepLib.SameParameter_s(shell, 1e-7, False)
        except Exception:
            info["sameparam_failed"] += 1
        ctx = ShapeBuild_ReShape()
        fixed = {}
        for f, fc in face_of.items():
            sf = ShapeFix_Face(TopoDS.Face_s(ctx.Apply(fc)))
            sf.SetContext(ctx)
            sf.Perform()
            fixed[f] = sf.Face()
        result = TopoDS_Shell()
        builder.MakeShell(result)
        for f, ff in fixed.items():
            final_face = TopoDS.Face_s(ctx.Apply(ff))
            builder.Add(result, final_face)
            face_of[f] = final_face
        fin = TopTools_IndexedMapOfShape()
        TopExp.MapShapes_s(result, TopAbs_EDGE, fin)
        info["replaced_edges"] = sum(1 for i in range(1, fin.Extent() + 1) if not raw_edges.Contains(fin.FindKey(i)))
    elif REPAIR_MODE in ("faces", "faces-rev", "pcurve"):
        from OCP.ShapeBuild import ShapeBuild_ReShape
        ctx = ShapeBuild_ReShape()
        order = list(face_of.values())
        if REPAIR_MODE == "faces-rev":
            order.reverse()
        if REPAIR_MODE == "pcurve":
            # Every edge gets its curve on each of its faces IN PLACE (the shared TShape is updated,
            # not replaced) before any face repair runs, so no later replacement can hand a face an
            # edge that lacks its curve on that face.
            from OCP.ShapeFix import ShapeFix_Edge
            fe = ShapeFix_Edge()
            for fc in order:
                ex = TopExp_Explorer(fc, TopAbs_EDGE)
                while ex.More():
                    e = TopoDS.Edge_s(ex.Current())
                    ex.Next()
                    try:
                        fe.FixAddPCurve(e, fc, False, 1e-7)
                    except Exception:
                        info["pcurve_failed"] += 1
        # Take each repaired face from the tool itself, and use the shared context ONLY for the edge
        # replacements. Looking a face up through the context composes its orientation twice: for a
        # face reversed to point outward on a surface closed in both directions, that returns the
        # COMPLEMENT region - 00032949_19's torus face repaired to 48.85 (its STEP area) and came back
        # from the context as 602.4, the whole torus (651.2) less the face. The same lookup inside
        # ShapeFix_Shape is why every repair mode gave identical 6-24x torus faces.
        #
        # So every face is repaired and looked up FORWARD, and turned outward only here, once.
        fixed = {}
        items = list(face_of.items())
        if REPAIR_MODE == "faces-rev":
            items.reverse()
        for f, fc in items:
            sf = ShapeFix_Face(TopoDS.Face_s(ctx.Apply(fc)))
            sf.SetContext(ctx)
            sf.Perform()
            fixed[f] = sf.Face()
        result = TopoDS_Shell()
        builder.MakeShell(result)
        for f, ff in fixed.items():
            final_face = TopoDS.Face_s(ctx.Apply(ff))
            builder.Add(result, TopoDS.Face_s(final_face.Reversed()) if flip_of.get(f) else final_face)
            face_of[f] = final_face
        from OCP.TopTools import TopTools_IndexedMapOfShape
        raw_edges, fin = TopTools_IndexedMapOfShape(), TopTools_IndexedMapOfShape()
        TopExp.MapShapes_s(shell, TopAbs_EDGE, raw_edges)
        TopExp.MapShapes_s(result, TopAbs_EDGE, fin)
        info["replaced_edges"] = sum(1 for i in range(1, fin.Extent() + 1) if not raw_edges.Contains(fin.FindKey(i)))
    else:
        fix = ShapeFix_Shape(shell)
        fix.Perform()
        result = fix.Shape()
        ctx = fix.Context()
        from OCP.TopAbs import TopAbs_FACE as _FACE
        remapped = {}
        for f, fc in face_of.items():
            r = ctx.Apply(fc)
            if r.IsNull():
                info["face_removed_by_repair"] += 1
                continue
            if r.ShapeType() == _FACE:
                remapped[f] = TopoDS.Face_s(r)
            else:
                # Split by the repair: keep the whole result, so the area check measures every piece.
                ex_r = TopExp_Explorer(r, _FACE)
                if ex_r.More():
                    remapped[f] = r
                    info["face_split_by_repair"] += 1
        face_of = remapped
    # DROP NATURAL-BOUND WIRES the repair added. ShapeFix_Face's natural-bound fix gives a face on a
    # closed surface its full parameter rectangle as an outer wire whenever it judges the real loop to
    # be a hole: the torus face of 00032949_19 came back with its 4 true edges plus a 360 x 360 deg
    # rectangle, i.e. the whole torus minus the patch (602.4 = 651.2 - 48.85). The OCP bindings only
    # expose that mode as a getter, so the wire is removed afterwards: on a face that has real boundary
    # edges, a wire with no edge shared by another face is not part of the patch's boundary. (Seam and
    # apex edges are unshared too, but they sit inside wires that also hold shared edges.)
    if REPAIR_MODE != "shape" or True:
        m = TopTools_IndexedDataMapOfShapeListOfShape()
        TopExp.MapShapesAndAncestors_s(result, TopAbs_EDGE, TopAbs_FACE, m)
        def shared(e, face):
            # Another DISTINCT face must use the edge: a seam edge lists its own face twice, and the
            # natural-bound rectangle's vertical sides are exactly such a seam.
            i = m.FindIndex(e)
            if i <= 0:
                return False
            it = m.FindFromIndex(i)
            for anc in it:
                if not anc.IsSame(face):
                    return True
            return False
        cleaned = TopoDS_Shell()
        builder.MakeShell(cleaned)
        ex = TopExp_Explorer(result, TopAbs_FACE)
        replaced = {}
        while ex.More():
            fc = TopoDS.Face_s(ex.Current())
            ex.Next()
            wires_all, keep = [], []
            wx = TopExp_Explorer(fc, TopAbs_WIRE)
            while wx.More():
                w = wx.Current()
                wx.Next()
                wires_all.append(w)
                ee = TopExp_Explorer(w, TopAbs_EDGE)
                has_shared = False
                while ee.More():
                    if shared(ee.Current(), fc):
                        has_shared = True
                        break
                    ee.Next()
                if has_shared:
                    keep.append(w)
            if keep and len(keep) < len(wires_all):
                nf = TopoDS_Face()
                builder.MakeFace(nf, BRep_Tool.Surface_s(fc), BRep_Tool.Tolerance_s(fc))
                for w in keep:
                    builder.Add(nf, w)
                nf = TopoDS.Face_s(nf.Oriented(fc.Orientation()))
                builder.Add(cleaned, nf)
                replaced[fc.TShape()] = nf
                info["natural_bound_wires_removed"] += len(wires_all) - len(keep)
            else:
                builder.Add(cleaned, fc)
        if replaced:
            for f, fc in list(face_of.items()):
                for sh in [x for x in [replaced.get(fc.TShape())] if x is not None]:
                    face_of[f] = sh
            result = cleaned
    # Area of every patch's final face (after the repair's replacements) against the patch's mesh
    # area: a face that took the wrong region of its surface shows up here even when the shell is
    # closed and valid.
    a_, b_, c_ = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    tri_area = 0.5 * np.linalg.norm(np.cross(b_ - a_, c_ - a_), axis=1)
    ratios = {}
    for f, fc in face_of.items():
        try:
            g = GProp_GProps()
            BRepGProp.SurfaceProperties_s(fc, g)
            ratios[int(f)] = abs(g.Mass()) / max(float(tri_area[F == f].sum()), 1e-300)
        except Exception:
            ratios[int(f)] = float("nan")
    info["_area_ratio"] = ratios
    if _faces is not None:
        _faces["face_of"] = face_of
    return result, info


# ── constrained refit ────────────────────────────────────────────────────────
#
# Mesh2Brep (Shen, Zhao, Yan, Wang, TVCG 2025) builds its B-rep the way build_topo does - topology
# from patch adjacency, geometry from the surfaces - and makes exact intersections work by first
# refitting the surfaces jointly under the constraints the topology implies. Two of its
# "intersection-aware" constraints are the ones our failures need:
#   same point   every surface of a corner passes through one point (an auxiliary point, solved with
#                them). Fitted independently they often do not: 4.3% of corners in the invalid
#                Onshape rebuilds miss by more than 1e-6 x the diagonal (p99 3.7e-3), 0.2% with true
#                segments - and an exact edge then cannot reach its vertex.
#   tangency     along a chain whose surfaces meet within REFIT_TANGENT_DEG, sample points lie on both
#                surfaces with parallel normals: a fillet is tangent to its neighbours, not a hair off
#                them (no intersection) or a hair into them (two).
# Only the surfaces in such constraints move; each stays close to its own group's vertices (the data
# term), and a cluster whose refit drags any surface away from its data is dropped. Solved by
# Levenberg-Marquardt on a penalty that stiffens (numpy only; Mesh2Brep uses Ipopt).
#
# MEASURED as a build_repaired move on the 134 parts the loop runs on (four sets): 451 inconsistent
# corners and 826 unsettled tangent chains found, 225 clusters refit (706 surfaces), 78 dropped for
# drifting off their data; the move was kept on 29 parts, and the recreated count went 552 -> 551.
# The corners that stay inconsistent belong to wrong groups or types - a solver cannot make four
# surfaces meet when one of them is the wrong surface - which the loop's absorb / retype moves fix.

REFIT_TANGENT_DEG = 5.0        # a chain whose surfaces meet within this angle is a tangent edge
REFIT_CORNER_TOL = 1e-6        # x diagonal: a corner whose surfaces miss one point by more is inconsistent
REFIT_SAMPLES = 40             # data points per surface
REFIT_TANGENT_SAMPLES = 3      # constraint points per tangent chain
REFIT_TANGENT_OK_DEG = 0.05    # a tangent pair already this close (and touching) needs no constraint
REFIT_MAX_SURFACES = 60        # per cluster; a larger cluster is left as fitted
REFIT_MAX_DRIFT = 1e-4         # x diagonal: a surface may lose at most this much RMS fit (or triple it)


def _to_vec(kind, p):
    if kind == "plane":
        n = unit(p["normal"])
        return np.r_[n, float(np.asarray(p["origin"], float) @ n)]
    if kind == "sphere":
        return np.r_[np.asarray(p["center"], float), p["radius"]]
    if kind == "cylinder":
        return np.r_[unit(p["axis"]), np.asarray(p["axisPoint"], float), p["radius"]]
    if kind == "cone":
        return np.r_[np.asarray(p["apex"], float), unit(p["axis"]), p["halfAngle"]]
    return np.r_[np.asarray(p["center"], float), unit(p["axis"]), p["majorRadius"], p["minorRadius"]]


def _from_vec(kind, th):
    th = np.asarray(th, float)
    if kind == "plane":
        n = unit(th[:3])
        return {"origin": list(n * th[3]), "normal": list(n)}
    if kind == "sphere":
        return {"center": list(th[:3]), "radius": float(th[3])}
    if kind == "cylinder":
        return {"axis": list(unit(th[:3])), "axisPoint": list(th[3:6]), "radius": float(th[6])}
    if kind == "cone":
        return {"apex": list(th[:3]), "axis": list(unit(th[3:6])), "halfAngle": float(th[6])}
    return {"center": list(th[:3]), "axis": list(unit(th[3:6])), "majorRadius": float(th[6]),
            "minorRadius": float(th[7])}


_DIR = {"plane": slice(0, 3), "cylinder": slice(0, 3), "cone": slice(3, 6), "torus": slice(3, 6)}
_POS = {"plane": [3], "sphere": [0, 1, 2, 3], "cylinder": [3, 4, 5, 6], "cone": [0, 1, 2],
        "torus": [0, 1, 2, 6, 7]}


def _rms(kind, p, X):
    return float(np.sqrt(np.mean(dist_normal(kind, p, X)[0] ** 2)))


def constrained_refit(mesh, surfaces, topo=None, info=None, deadline=None):
    """surfaces refit under same-point (inconsistent corners) and tangency (near-tangent chains)
    constraints. Returns (new dict, info); surfaces outside every accepted cluster are unchanged."""
    V, scale = mesh.V, mesh.diag
    topo = topo or mesh_topology(mesh)
    info = info if info is not None else defaultdict(int)
    faces_at, corners, chains = topo["faces_at"], topo["corners"], topo["chains"]
    surfaces = dict(surfaces)

    corner_gs, bad_corner = {}, set()
    for c in corners:
        gs = sorted(g for g in faces_at[c] if g in surfaces)
        if len(gs) < 3:
            continue
        corner_gs[c] = gs
        x = project_onto([surfaces[g] for g in gs], V[c[0]], scale)
        if max(abs(float(dist_normal(*surfaces[g], x)[0][0])) for g in gs) > REFIT_CORNER_TOL * scale:
            bad_corner.add(c)
    tangent = []
    for a, b, path in chains:
        if b < 0 or a not in surfaces or b not in surfaces:
            continue
        if surfaces[a][0] == "plane" and surfaces[b][0] == "plane":
            continue                                    # tangent planes are coplanar: no edge at all
        inner = path[1:-1] if len(path) > 2 else path
        pts = [project_onto([surfaces[a], surfaces[b]], V[i], scale) for i in inner]
        mid = pts[len(pts) // 2]
        na, nb = dist_normal(*surfaces[a], mid)[1][0], dist_normal(*surfaces[b], mid)[1][0]
        ang = math.degrees(math.acos(min(1.0, abs(float(na @ nb)))))
        gap = max(abs(float(dist_normal(*surfaces[g], mid)[0][0])) for g in (a, b)) / scale
        if ang < REFIT_TANGENT_DEG and (ang > REFIT_TANGENT_OK_DEG or gap > REFIT_CORNER_TOL):
            pick = sorted(set(np.linspace(0, len(pts) - 1, min(REFIT_TANGENT_SAMPLES, len(pts))).round().astype(int)))
            tangent.append((a, b, [pts[i] for i in pick]))
    info["refit_bad_corners"] += len(bad_corner)
    info["refit_tangent_chains"] += len(tangent)

    # clusters of surfaces tied together by a violated corner or a tangent chain
    parent = {}

    def find(g):
        while parent.setdefault(g, g) != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    for c in bad_corner:
        for g in corner_gs[c][1:]:
            parent[find(corner_gs[c][0])] = find(g)
    for a, b, _ in tangent:
        parent[find(a)] = find(b)
    clusters = defaultdict(set)
    for g in list(parent):
        clusters[find(g)].add(g)

    boundary = set()
    for a, b, path in chains:
        boundary.update(path)
    for cl in clusters.values():
        if len(cl) > REFIT_MAX_SURFACES:
            info["refit_cluster_too_big"] += 1
            continue
        if any(surfaces[g][0] == "bspline" for g in cl if g in surfaces):
            info["refit_cluster_freeform"] += 1         # no parameter vector for a B-spline (yet)
            continue
        if deadline is not None and time.time() > deadline:
            info["refit_out_of_time"] += 1
            break
        new = _refit_cluster(mesh, surfaces, sorted(cl), corner_gs, tangent, boundary, info, deadline)
        if new:
            surfaces.update(new)
            info["refit_clusters_kept"] += 1
            info["refit_surfaces_moved"] += len(new)
        else:
            info["refit_clusters_dropped"] += 1
    return surfaces, info


def _refit_cluster(mesh, surfaces, cl, corner_gs, tangent, boundary, info, deadline=None):
    """One cluster's surfaces refit together; None when any of them drifts off its data.

    Every residual term depends on a few variable blocks (its surface, or a constraint point and the
    surfaces through it), so the finite-difference Jacobian recomputes only the terms of the block
    being perturbed: one refit of a 40-surface cluster went from 540 s to seconds."""
    V, T, F, scale = mesh.V, mesh.T, mesh.F, mesh.diag
    rng = np.random.default_rng(0)
    kinds = {g: surfaces[g][0] for g in cl}
    th0 = {g: _to_vec(*surfaces[g]) for g in cl}
    data = {}
    for g in cl:
        vs = np.unique(T[F == g])
        inner = np.array([v for v in vs if v not in boundary])
        use = inner if len(inner) >= 6 else vs      # boundary vertices of a loose segment are off-edge
        if len(use) > REFIT_SAMPLES:
            use = rng.choice(use, REFIT_SAMPLES, replace=False)
        data[g] = V[use]
    rms0 = {g: _rms(kinds[g], surfaces[g][1], data[g]) for g in cl}
    inset = set(cl)
    cons = []                                          # (surfaces, start point, tangent?)
    for c, gs in corner_gs.items():
        if inset.intersection(gs):
            cons.append((gs, project_onto([surfaces[g] for g in gs], V[c[0]], scale), False))
    for a, b, pts in tangent:
        if a in inset or b in inset:
            for q in pts:
                cons.append(([a, b], np.asarray(q, float), True))

    # variable blocks: one per surface, one per constraint point
    blocks, n = [], 0
    off = {}
    for g in cl:
        off[("s", g)] = (n, n + len(th0[g]))
        n += len(th0[g])
    for i in range(len(cons)):
        off[("q", i)] = (n, n + 3)
        n += 3
    x = np.concatenate([th0[g] for g in cl] + [q for _, q, _ in cons])
    anchor = {g: th0[g][3:6].copy() for g in cl if kinds[g] == "cylinder"}
    mu = [1.0]

    def surf(g, x):
        if g in inset:
            a, b = off[("s", g)]
            return kinds[g], _from_vec(kinds[g], x[a:b])
        return surfaces[g]

    def data_term(g):
        def fn(x):
            k, p = surf(g, x)
            a, b = off[("s", g)]
            th = x[a:b]
            r = [dist_normal(k, p, data[g])[0] / (scale * math.sqrt(len(data[g])))]
            if k in _DIR:
                r.append([float(th[_DIR[k]] @ th[_DIR[k]] - 1.0)])
            if k == "cylinder":                         # the axis point may not slide along the axis
                r.append([float((th[3:6] - anchor[g]) @ unit(th[:3])) / scale])
            return np.concatenate([np.atleast_1d(np.asarray(v, float)) for v in r])
        return fn, {("s", g)}

    def con_term(i):
        gs, q0, tan = cons[i]

        def fn(x):
            a, b = off[("q", i)]
            q = x[a:b]
            r, ns = [], []
            for g in gs:
                d, nrm = dist_normal(*surf(g, x), q)
                r.append(mu[0] * d / scale)
                ns.append(nrm[0])
            if tan:
                r.append(mu[0] * np.cross(ns[0], ns[1]))
            r.append(1e-3 * (q - q0) / scale)          # a point may slide along its curve, not wander
            return np.concatenate([np.atleast_1d(np.asarray(v, float)) for v in r])
        return fn, {("q", i)} | {("s", g) for g in gs if g in inset}

    terms = [data_term(g) for g in cl] + [con_term(i) for i in range(len(cons))]
    rows, m = [], 0
    for fn, _ in terms:
        k = len(fn(x))
        rows.append((m, m + k))
        m += k
    terms_of = defaultdict(list)
    for ti, (_, deps) in enumerate(terms):
        for bk in deps:
            terms_of[bk].append(ti)

    def residual(x):
        return np.concatenate([fn(x) for fn, _ in terms])

    step = np.full(n, 1e-7)
    for g in cl:
        a, _ = off[("s", g)]
        for j in _POS[kinds[g]]:
            step[a + j] = 1e-7 * scale
    for i in range(len(cons)):
        a, b = off[("q", i)]
        step[a:b] = 1e-7 * scale

    def jacobian(x, r):
        J = np.zeros((m, n))
        for bk, (a, b) in off.items():
            tis = terms_of[bk]
            for j in range(a, b):
                xp = x.copy()
                xp[j] += step[j]
                for ti in tis:
                    r0, r1 = rows[ti]
                    J[r0:r1, j] = (terms[ti][0](xp) - r[r0:r1]) / step[j]
        return J

    for mu[0] in (1e1, 1e3, 1e5):
        lam = 1e-3
        r = residual(x)
        cost = r @ r
        for _ in range(30):
            if deadline is not None and time.time() > deadline:
                info["refit_out_of_time"] += 1
                return None
            J = jacobian(x, r)
            A, gvec = J.T @ J, J.T @ r
            improved, rel = False, 0.0
            for _ in range(8):
                try:
                    dx = -np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), gvec)
                except np.linalg.LinAlgError:
                    lam *= 10
                    continue
                r2 = residual(x + dx)
                c2 = r2 @ r2
                if c2 < cost:
                    rel = (cost - c2) / max(cost, 1e-300)
                    x, r, cost, lam, improved = x + dx, r2, c2, max(lam / 3, 1e-9), True
                    break
                lam *= 10
            if not improved or rel < 1e-10:
                break
    new = {g: surf(g, x) for g in cl}
    for g in cl:
        after = _rms(kinds[g], new[g][1], data[g])
        if after > max(3 * rms0[g], rms0[g] + REFIT_MAX_DRIFT * scale):
            info["refit_drift_reject"] += 1
            return None
    miss = 0.0
    for i, (gs, _, _) in enumerate(cons):
        a, b = off[("q", i)]
        for g in gs:
            miss = max(miss, abs(float(dist_normal(*(new[g] if g in new else surfaces[g]), x[a:b])[0][0])) / scale)
    info["refit_worst_miss_e9"] = max(info["refit_worst_miss_e9"], int(miss * 1e9))
    return new


# ── repair loop ──────────────────────────────────────────────────────────────
#
# build_topo takes the segmentation and its surfaces as given. build_repaired uses the kernel as the
# judge instead: build, find the patches whose faces fail (no face at all, a face BRepCheck rejects, a
# free edge), change ONE thing about one of them, rebuild, and keep the change only if the part got
# better. The changes, in the order tried:
#   absorb   hand the group's facets to the neighbouring groups whose surfaces they lie on. For slivers
#            the painter left untyped (170 of the 223 surfaceless groups on the Onshape tm-ring1 set are
#            single triangles), and as a last resort for any failing group.
#   retype   the group's next surface from test/fitCandidates.mjs (gaussMap.fitRegionAs for every
#            type): among fits within LOOP_TIE_RATIO of the best, the simplest type first (an 89 deg
#            "cone" that is a plane), then the rest by residual. A surfaceless group gets its first.
# "Better" uses only what inference has - no STEP truth: a closed, valid solid within 1% of the MESH's
# volume beats a closed valid one, which beats fewer failing patches, fewer free edges, a smaller
# volume error.

SIMPLER = {"plane": 0, "sphere": 1, "cylinder": 2, "cone": 3, "torus": 4}
LOOP_SMALL_GROUP = 3          # facets; a failing group this small is absorbed, never refit
LOOP_TIE_RATIO = 1.5          # fits within this factor of the best residual count as tied
LOOP_RETYPES = 2              # alternatives tried per group before absorbing it
LOOP_MAX_BUILDS = 40
LOOP_BUDGET_S = 300.0
LOOP_FIT_TIMEOUT_S = 120        # one fitter call (all of a round's groups)
# A running build cannot be interrupted, and one bad trial - a surface that does not fit its group, a
# large group absorbed into a neighbour it does not lie on - sends ShapeFix_Shape grinding for minutes
# (fill_eval 00182468_37: 3 s builds, then a trial past 15 minutes). So only plausible moves are tried:
LOOP_MAX_RESIDUAL_RATIO = 10.0  # a retype candidate fits within this factor of the best candidate
LOOP_ABSORB_TOL = 2e-3          # x diagonal: a larger group's facets move only onto a surface this near


def node_candidates(pos_path, script=None):
    """fit_candidates for build_repaired: test/fitCandidates.mjs over the mesh's .pos soup. Takes
    {group: [triangle indices in the soup]}, returns {group: [{type, params, residual}, ...]}."""
    script = script or str(Path(__file__).resolve().parent.parent / "test" / "fitCandidates.mjs")

    def fit(groups):
        req = json.dumps({"pos": str(pos_path), "groups": {str(g): [int(t) for t in ts] for g, ts in groups.items()}})
        try:
            r = subprocess.run(["node", script], input=req, capture_output=True, text=True, timeout=LOOP_FIT_TIMEOUT_S)
        except (OSError, subprocess.TimeoutExpired):
            return {}
        if r.returncode != 0:
            return {}
        return {int(g): c for g, c in json.loads(r.stdout).items()}
    return fit


def plane_candidate(mesh, labels, g):
    """Least-squares plane through a group's vertices: the fallback surface without the node fitter."""
    P = mesh.V[np.unique(mesh.T[labels == g])]
    c = P.mean(0)
    n = np.linalg.eigh(np.cov((P - c).T) if len(P) > 2 else np.eye(3))[1][:, 0]
    if len(P) == 3:
        n = unit(np.cross(P[1] - P[0], P[2] - P[0]))
    return {"type": "plane", "params": {"origin": list(c), "normal": list(unit(n))},
            "residual": float(np.abs((P - c) @ n).max())}


LOOP_AREA_RATIO = 1.25       # a face this far from its patch's mesh area took the wrong region


def patch_problems(result, face_of, labels, ratios=None):
    """Patches whose face is
      missing   not built at all
      invalid   rejected by BRepCheck (the face, a wire or an edge of it)
      orient    using a shared edge in the same direction as the face across it. BRepCheck reports
                that only on the shell IN THE SOLID (BadOrientationOfSubshape), never per face: a
                cone retyped to a plane coplanar with its neighbour flipped four faces of 00000954_000
                with every face "valid"
      area      off its patch's mesh area by more than LOOP_AREA_RATIO either way: the complement
                region of a closed surface, a closed valid solid 1e10 times the mesh's volume
      open      bounded by a free edge"""
    owner = {}
    for f, sh in face_of.items():
        ex = TopExp_Explorer(sh, TopAbs_FACE)
        while ex.More():
            owner[hash(ex.Current())] = f
            ex.Next()
    # Faces are checked as they sit in the result: face_of can still name a face the repair replaced,
    # and the analyzer throws (NoSuchObject) on a shape it did not see.
    ana = BRepCheck_Analyzer(result)
    invalid, present = set(), set()
    ex = TopExp_Explorer(result, TopAbs_FACE)
    while ex.More():
        fc = ex.Current()
        ex.Next()
        f = owner.get(hash(fc))
        if f is None:
            continue
        present.add(f)
        if f in invalid:
            continue
        if not ana.IsValid(fc):
            invalid.add(f)
            continue
        for typ in (TopAbs_WIRE, TopAbs_EDGE):
            sub = TopExp_Explorer(fc, typ)
            while sub.More():
                if not ana.IsValid(sub.Current()):
                    invalid.add(f)
                    break
                sub.Next()
            if f in invalid:
                break
    opened = set()
    use = defaultdict(list)                             # edge -> [(patch, traversed reversed)]
    ex = TopExp_Explorer(result, TopAbs_FACE)
    while ex.More():
        fc = ex.Current()
        ex.Next()
        f = owner.get(hash(fc))
        flipped = fc.Orientation() == TopAbs_REVERSED
        sub = TopExp_Explorer(fc, TopAbs_EDGE)
        while sub.More():
            e = TopoDS.Edge_s(sub.Current())
            sub.Next()
            if BRep_Tool.Degenerated_s(e) or BRep_Tool.IsClosed_s(e, TopoDS.Face_s(fc)):
                continue                                # apex / seam: one face by nature
            use[hash(e)].append((f, (e.Orientation() == TopAbs_REVERSED) != flipped))
    orient = set()
    for lst in use.values():
        if len(lst) == 1 and lst[0][0] is not None:
            opened.add(lst[0][0])
        elif len(lst) == 2 and lst[0][1] == lst[1][1]:
            orient.update(f for f, _ in lst if f is not None)
    area = set()
    for f, r in (ratios or {}).items():
        if not (1 / LOOP_AREA_RATIO <= r <= LOOP_AREA_RATIO):     # NaN counts as off
            area.add(f)
    missing = set(int(g) for g in np.unique(labels)) - set(int(f) for f in present)
    orient -= invalid
    return {"missing": missing, "invalid": invalid, "orient": orient, "area": area - invalid - orient,
            "open": opened - invalid - orient}


def absorb_group(mesh, labels, surfaces, g, tol=None):
    """Hand group g's facets to neighbouring groups, working in from its border: each facet goes to
    the adjacent group (with a surface) nearest all three of its vertices - when tol is given, only if
    they are all within tol of it. Returns the facets moved."""
    T, V = mesh.T, mesh.V
    left = set(np.nonzero(labels == g)[0].tolist())
    moved = 0
    while left:
        assign = {}
        for t in left:
            best = None
            for k in range(3):
                u, v = int(T[t][k]), int(T[t][(k + 1) % 3])
                n = mesh.partner.get((t, (min(u, v), max(u, v))))
                if n is None:
                    continue
                q = int(labels[n])
                if q == g or q not in surfaces:
                    continue
                kind, p = surfaces[q]
                d = float(np.abs(dist_normal(kind, p, V[T[t]])[0]).max())
                if best is None or d < best[0]:
                    best = (d, q)
            if best and (tol is None or best[0] <= tol):
                assign[t] = best[1]
        if not assign:
            break
        for t, q in assign.items():
            labels[t] = q
            left.discard(t)
            moved += 1
    return moved


def alternatives(cands, current, scale):
    """Candidate surfaces other than the current type: the simplest first among those tied with the
    best residual, then the others by residual."""
    cands = [c for c in cands if c.get("params") and (current is None or c["type"] != current[0])]
    if not cands:
        return []
    best = min(c["residual"] for c in cands)
    tied = [c for c in cands if c["residual"] <= LOOP_TIE_RATIO * best + 1e-7 * scale]
    rest = sorted((c for c in cands if c not in tied and c["residual"] <= LOOP_MAX_RESIDUAL_RATIO * best + 1e-7 * scale),
                  key=lambda c: c["residual"])
    return sorted(tied, key=lambda c: SIMPLER.get(c["type"], 9)) + rest


def grade(shape, mesh, problems):
    sc = score(shape, mesh)
    solid = bool(sc.get("closed") and sc.get("valid"))
    ve = sc.get("vol_err")
    ok = solid and ve is not None and ve < 0.01
    nprob = sum(len(v) for v in problems.values())
    return (ok, solid, -nprob, -(sc.get("free_edges") or 0), -(ve if ve is not None else 1e9)), sc


def build_repaired(mesh, surfaces, fit_candidates=None, max_builds=LOOP_MAX_BUILDS, budget=LOOP_BUDGET_S,
                   refit=False, exact=False, on_improve=None, _state=None):
    """build_topo, then repair moves while the part is not a closed, valid solid of the mesh's volume.
    fit_candidates: {group: [soup triangle indices]} -> {group: [candidate fits]}, e.g. node_candidates;
    without it a failing group can only be absorbed or given a least-squares plane. refit / exact
    allow the two whole-part moves (constrained_refit, then exact intersection edges); measured on the
    134 parts the loop runs on, they were kept 29 and 40 times but changed the recreated count by -1
    (552 -> 551 over the four sets), so both are off by default. on_improve(shape,
    score) is called with every new best, so a caller that has to stop the loop keeps the best so far.
    _state, a dict, receives the best build's labels, surfaces, problems and whether it is a recreated
    solid, for facet_fallback."""
    t0 = time.time()
    log = defaultdict(int)
    work = copy.copy(mesh)
    slowest = [0.0]

    def build(labels, surfs, ex, first=False):
        t = time.time()
        work.F = labels
        faces = {}
        try:
            shape, info = build_topo(work, surfs, _faces=faces, exact=ex)
            probs = patch_problems(shape, faces["face_of"], labels, info.get("_area_ratio"))
            g, sc = grade(shape, work, probs)
        except Exception:
            # A move that makes OCCT throw is a rejected move, not a lost part: 11 of the 275 Onshape
            # parts ended as exceptions (NCollection_Array1::Resize in a trial) with a closed base.
            if first:
                raise
            log["trial_exception"] += 1
            return None
        finally:
            slowest[0] = max(slowest[0], time.time() - t)
        return [g, shape, info, probs, labels, surfs, ex, sc]

    def time_left():
        # a build is only started when the slowest one so far still fits in the budget
        return budget - (time.time() - t0) > slowest[0]

    best = build(mesh.F.copy(), dict(surfaces), False, first=True)
    builds = 1
    if on_improve:
        on_improve(best[1], best[7])
    log["problems_before"] = sum(len(v) for v in best[3].values())
    cands, tried = {}, set()

    def facets(lab, g):
        return int((lab == g).sum())

    def attempt(lab, sf, ex, name):
        nonlocal best, builds
        trial = build(lab, sf, ex)
        builds += 1
        log[f"{name}_tried"] += 1
        if trial is not None and trial[0] > best[0]:
            best = trial
            log[f"{name}_kept"] += 1
            if on_improve:
                on_improve(best[1], best[7])
            return True
        return False

    # Untyped slivers first, all at once: a patch with no surface has no face, so moving its facets
    # to a neighbour cannot lose anything.
    if not best[0][0]:
        lab, sf = best[4].copy(), dict(best[5])
        slivers = [g for g in best[3]["missing"] if g not in sf and facets(lab, g) <= LOOP_SMALL_GROUP]
        if slivers and sum(absorb_group(mesh, lab, sf, g) for g in slivers) and time_left():
            trial = build(lab, sf, False)
            builds += 1
            log["sliver_absorb_tried"] += 1
            if trial is not None and trial[0] >= best[0]:
                best = trial
                log["slivers_absorbed"] += len(slivers)
                if on_improve:
                    on_improve(best[1], best[7])
        tried.update(("absorb", g) for g in slivers)

    # Whole-part moves: surfaces made consistent at corners and tangent edges, then exact edges.
    if refit and not best[0][0] and time_left():
        work.F = best[4]
        sf, rinfo = constrained_refit(work, best[5], deadline=t0 + budget / 3)
        for k, v in rinfo.items():
            log[k] += v
        if rinfo.get("refit_surfaces_moved") and time_left():
            attempt(best[4], sf, best[6], "refit")
    if exact and not best[0][0] and time_left():
        attempt(best[4], best[5], True, "exact")

    while builds < max_builds and time_left() and not best[0][0]:
        _, _, _, probs, lab0, sf0, ex0, _ = best
        order = [g for k in ("missing", "invalid", "orient", "area", "open") for g in sorted(probs[k])]
        if not order:
            break
        need = [g for g in order if g not in cands and facets(lab0, g) > LOOP_SMALL_GROUP]
        if need:
            got = fit_candidates({g: mesh.src[lab0 == g].tolist() for g in need}) if fit_candidates else {}
            for g in need:
                cands[g] = got.get(g) or [plane_candidate(mesh, lab0, g)]
        move = None
        for g in order:
            if facets(lab0, g) > LOOP_SMALL_GROUP:
                for i, c in enumerate(alternatives(cands.get(g, []), sf0.get(g), mesh.diag)[:LOOP_RETYPES]):
                    if ("retype", g, i) not in tried:
                        move = ("retype", g, i, c)
                        break
            if move is None and ("absorb", g) not in tried:
                move = ("absorb", g)
            if move:
                break
        if move is None:
            break
        tried.add(move[:3] if move[0] == "retype" else move)
        lab, sf = lab0.copy(), dict(sf0)
        if move[0] == "retype":
            sf[move[1]] = (move[3]["type"], move[3]["params"])
        elif not absorb_group(mesh, lab, sf, move[1],
                              None if facets(lab0, move[1]) <= LOOP_SMALL_GROUP else LOOP_ABSORB_TOL * mesh.diag):
            continue
        elif not (lab == move[1]).any():
            sf.pop(move[1], None)                      # fully absorbed; stranded facets keep it
        attempt(lab, sf, ex0, move[0])
    _, shape, info, probs, _, _, ex, _ = best
    if _state is not None:
        _state.update(labels=best[4], surfaces=best[5], problems=probs, ok=bool(best[0][0]))
    for k, v in log.items():
        info[f"repair_{k}" if not k.startswith("refit") else k] = v
    info["repair_builds"] = builds
    info["repair_exact"] = int(ex)
    info["repair_problems_after"] = sum(len(v) for v in probs.values())
    for k, v in probs.items():
        if v:
            info[f"repair_left_{k}"] = len(v)
    info["repair_secs"] = round(time.time() - t0, 1)
    return shape, info


# ── facet fallback ───────────────────────────────────────────────────────────
#
# next.BREP.io's stl2step always hands back a solid: a region whose boundary it cannot build exactly is
# demoted to facets (a planar face per triangle) and kept beside the analytic faces, and a body nothing
# else closes is exported as the repaired mesh. Its segmentation is weaker than ours off OCCT meshes,
# yet on the 45 fill_eval meshes it returned a closed, valid solid of the mesh's volume 36 times where
# the knitter returned one 15 times (2026-10-02; D:/abc-data/lab/competitor_brepio). facet_fallback is
# that safety net. Each tier runs only when the ones before did not recreate the part (closed, valid,
# volume within 1% of the mesh):
#   analytic  the knitter as is (build_topo, or build_repaired with --repair-loop)
#   hybrid    the patches the best build still has problems with, and every surfaceless patch, as
#             facets; the rest stay analytic. Up to FALLBACK_ROUNDS builds, each adding the analytic
#             patches that newly fail.
#   planar    only the plane patches analytic, every other patch facets
#   faceted   the mesh as a polyhedral solid (build_polyhedral): the input, exactly, as a B-rep
# What came out is reported, never hidden: info["tier"] (or "none" when no tier recreated the part),
# the facet count, and their share of the area. A curved face meeting a facet gets the facet's straight
# side as its edge, a chord the repair covers with edge tolerance.
FALLBACK_ROUNDS = 3
FALLBACK_MAX_FACETS = 6000      # a facet tier with more goes straight to build_polyhedral (no ShapeFix)
FALLBACK_KINDS = ("missing", "invalid", "orient", "area", "open")


def build_polyhedral(mesh):
    """The mesh as a solid: a planar face per triangle, one straight edge per paired mesh edge, a vertex
    per (mesh vertex, body), a shell per body. No repair pass: planes need no stored pcurves."""
    from OCP.TopoDS import TopoDS_Compound
    V, T = mesh.V, mesh.T
    builder = BRep_Builder()
    shells = {}
    info = defaultdict(int)
    verts, pending = {}, {}

    def vert(v, bd):
        if (v, bd) not in verts:
            verts[(v, bd)] = BRepBuilderAPI_MakeVertex(gp_Pnt(*V[v])).Vertex()
        return verts[(v, bd)]

    for t, tri in enumerate(T):
        q = V[tri]
        nrm = np.cross(q[1] - q[0], q[2] - q[0])
        if np.linalg.norm(nrm) <= 1e-12 * mesh.diag * mesh.diag:
            info["facet_degenerate"] += 1
            continue
        bd = int(mesh.body[t])
        wire = TopoDS_Wire()
        builder.MakeWire(wire)
        for k in range(3):
            u, v = int(tri[k]), int(tri[(k + 1) % 3])
            key = (min(u, v), max(u, v))
            e = pending.pop((t, key), None)
            if e is None:
                mk = BRepBuilderAPI_MakeEdge(vert(u, bd), vert(v, bd))
                if not mk.IsDone():
                    info["edge_failed"] += 1
                    continue
                e = mk.Edge()
                n = mesh.partner.get((t, key))
                if n is not None:
                    pending[(n, key)] = e.Reversed()   # the partner runs it the other way (consistent winding)
            builder.Add(wire, e)
        face = TopoDS_Face()
        builder.MakeFace(face, Geom_Plane(gp_Pnt(*q[0]), gp_Dir(*unit(nrm))), 1e-7)
        builder.Add(face, wire)
        if bd not in shells:
            shells[bd] = TopoDS_Shell()
            builder.MakeShell(shells[bd])
        builder.Add(shells[bd], face)
        info["facet_faces"] += 1
    comp = TopoDS_Compound()
    builder.MakeCompound(comp)
    for sh in shells.values():
        builder.Add(comp, sh)
    return comp, info


def build_with_facets(mesh, labels, surfaces, demote, exact=False):
    """build_topo with every triangle of the patches in `demote` a facet face of its own. Returns
    (grade, shape, info, problems, first facet label)."""
    lab = np.asarray(labels).copy()
    tris = np.nonzero(np.isin(lab, list(demote)))[0]
    base = int(lab.max()) + 1
    lab[tris] = base + np.arange(len(tris))
    work = copy.copy(mesh)
    work.F = lab
    faces = {}
    shape, info = build_topo(work, {g: s for g, s in surfaces.items() if g not in demote}, _faces=faces,
                             exact=exact, facet_labels=set(range(base, base + len(tris))))
    probs = patch_problems(shape, faces["face_of"], lab, info.get("_area_ratio"))
    g, _ = grade(shape, work, probs)
    return g, shape, info, probs, base


def topo_state(mesh, surfaces, exact=False):
    """build_topo once, with what facet_fallback needs to know about it (build_repaired's _state)."""
    faces = {}
    shape, info = build_topo(mesh, surfaces, _faces=faces, exact=exact)
    probs = patch_problems(shape, faces["face_of"], mesh.F, info.get("_area_ratio"))
    g, _ = grade(shape, mesh, probs)
    return shape, info, {"labels": mesh.F, "surfaces": surfaces, "problems": probs, "ok": bool(g[0])}


def facet_fallback(mesh, shape, info, state):
    """The first tier (see above) that recreates the part, else the best-graded of them all. `shape` and
    `info` are the analytic build's, `state` its labels, surfaces, problems and verdict. Fills
    info["tier"], info["facet_tris"], info["facet_area"]."""
    t0 = time.time()
    a_, b_, c_ = mesh.V[mesh.T[:, 0]], mesh.V[mesh.T[:, 1]], mesh.V[mesh.T[:, 2]]
    tri_area = 0.5 * np.linalg.norm(np.cross(b_ - a_, c_ - a_), axis=1)
    labels, surfaces = np.asarray(state["labels"]), state["surfaces"]
    present = {int(g) for g in np.unique(labels)}

    def facet_share(demote):
        m = np.isin(labels, list(demote))
        return int(m.sum()), float(tri_area[m].sum() / max(tri_area.sum(), 1e-300))

    if state["ok"]:
        info.update(tier="analytic", facet_tris=0, facet_area=0.0)
        return shape
    best = (grade(shape, mesh, state["problems"])[0], shape, "analytic", (0, 0.0))
    tried = set()

    def attempt(tier, demote):
        nonlocal best
        key = frozenset(demote)
        if not demote or key in tried:
            return None
        tried.add(key)
        n, share = facet_share(demote)
        info[f"fallback_{tier}_tried"] += 1
        if n > FALLBACK_MAX_FACETS:
            info["fallback_too_many_facets"] += 1
            return None
        try:
            g, sh, _, probs, base = build_with_facets(mesh, labels, surfaces, demote)
        except Exception:
            info["fallback_exception"] += 1
            return None
        if g > best[0]:
            best = (g, sh, tier, (n, share))
        return g, probs, base

    demote = {int(g) for k in FALLBACK_KINDS for g in state["problems"].get(k, ())} | \
             {g for g in present if g not in surfaces}
    demote &= present
    for _ in range(FALLBACK_ROUNDS):
        r = attempt("hybrid", demote)
        if r is None or r[0][0]:
            break
        g, probs, base = r
        new = {int(x) for k in FALLBACK_KINDS for x in probs[k] if int(x) < base} - demote
        if not new:
            break
        demote |= new
    if not best[0][0]:
        attempt("planar", {g for g in present if (surfaces.get(g) or ("",))[0] != "plane"})
    if not best[0][0]:
        info["fallback_faceted_tried"] += 1
        try:
            sh, pinfo = build_polyhedral(mesh)
            g, _ = grade(sh, mesh, {})
            if g > best[0]:
                best = (g, sh, "faceted", (len(mesh.T), 1.0))
        except Exception:
            info["fallback_exception"] += 1
    g, shape, tier, (n, share) = best
    info.update(tier=tier if g[0] else "none", best_tier=tier, facet_tris=n, facet_area=round(share, 4),
                fallback_secs=round(time.time() - t0, 1))
    return shape


# ── freeform groups ──────────────────────────────────────────────────────────
#
# A group with no analytic surface (a fillet Onshape made as a B-spline, a spline extrusion, a group
# the painter left untyped) has had no face at all: the knitter left a hole there. add_freeform gives
# each such group a B-spline fitted to its own triangles (freeform_fit.fit_group: one untrimmed
# bicubic over the group plus a margin; the face is trimmed from it by the group's edges like any
# other). Groups of FREEFORM_MIN_FACETS or fewer are left alone, as the repair loop absorbs slivers.
FREEFORM_MIN_FACETS = LOOP_SMALL_GROUP + 1
# G1 where a neighbour meets a freeform group tangentially. At the mesh level a coarse fillet and a
# sharp edge look alike (one triangle across a 90 deg fillet sits 45 deg off each neighbour), so the
# mesh dihedral across the boundary edge only picks CANDIDATES: below this, the neighbour's surface
# normal at the shared vertices is offered to the fitter, which imposes it only where its own free fit
# agrees (freeform_fit.TANGENT_AGREE_DEG) - B-spline faces meeting planes at 45 deg sharp edges exist.
FREEFORM_TANGENT_DEG = 60.0
# ... and OFF by default in the rebuild. Tangency makes the fitted surfaces recover the truth better
# (freeform_eval found4 +52 -33 faces) but a B-spline made tangent to its neighbour TOUCHES it instead
# of crossing it, and where it hovers a hair off, project_onto puts the edge between the two surfaces
# and on neither: 5 of the 25 Onshape freeform parts (true segments) rebuilt only without it - four
# BRepCheck-invalid, one at 1.0% volume. The fix belongs to the knitter (a tangent run's edge taken on
# the neighbour, and the B-spline made to pass through it). --freeform-tangent turns it on.
FREEFORM_TANGENT = False


def tangent_normals(mesh, g, surfaces):
    """{vertex: unit normal} along group g's boundary where a neighbouring group with a surface meets
    it at a mesh dihedral below FREEFORM_TANGENT_DEG: the neighbour's normal at the vertex, turned to
    g's outward side."""
    T, F, V, P = mesh.T, mesh.F, mesh.V, mesh.partner
    cos_max = math.cos(math.radians(FREEFORM_TANGENT_DEG))
    acc = defaultdict(list)
    for t in np.nonzero(F == g)[0]:
        tri = T[t]
        nt = unit(mesh.N[t])
        for k in range(3):
            a, b = int(tri[k]), int(tri[(k + 1) % 3])
            nb = P.get((int(t), (min(a, b), max(a, b))))
            if nb is None:
                continue
            h = int(F[nb])
            if h == g or h not in surfaces or float(unit(mesh.N[nb]) @ nt) < cos_max:
                continue
            kind, p = surfaces[h]
            _, grad = dist_normal(kind, p, V[[a, b]])
            for vtx, n in zip((a, b), grad):
                acc[vtx].append(n if n @ nt >= 0 else -n)
    return {v: unit(np.mean(ns, 0)) for v, ns in acc.items()}


def add_freeform(mesh, surfaces, min_facets=None, info=None, tangent=None):
    """surfaces plus a "bspline" surface for every surfaceless group of min_facets or more facets,
    tangent to its analytic neighbours where they meet it smoothly (tangent_normals). A group no
    single parameterisation covers is split (freeform_fit.fit_pieces): its pieces after the first
    get NEW labels, written into mesh.F."""
    min_facets = FREEFORM_MIN_FACETS if min_facets is None else min_facets
    tangent = FREEFORM_TANGENT if tangent is None else tangent
    out = dict(surfaces)
    info = info if info is not None else defaultdict(int)
    next_label = int(mesh.F.max()) + 1
    for g in np.unique(mesh.F):
        g = int(g)
        if g in out:
            continue
        sel = np.nonzero(mesh.F == g)[0]
        if len(sel) < min_facets:
            continue
        normals = None
        if tangent:
            tn = tangent_normals(mesh, g, surfaces)
            if tn:
                keys = np.array(sorted(tn), np.int64)
                normals = (keys, np.array([tn[k] for k in keys]))
        try:
            pieces, failed = freeform_fit.fit_pieces(mesh.V, mesh.T[sel], scale=mesh.diag, normals=normals)
        except Exception:
            pieces, failed = [], [(np.arange(len(sel)), {"fail": "exception"})]
        for _, fi in failed:
            info[f"freeform_fail_{fi.get('fail', '?')}"] += 1
        if len(pieces) > 1 or failed:
            info["freeform_split_groups"] += 1
        for k, (tri, params, fi) in enumerate(pieces):
            label = g
            if k > 0:
                label = next_label
                next_label += 1
                mesh.F[sel[tri]] = label
            out[label] = ("bspline", params)
            info["freeform_fitted"] += 1
            info[f"freeform_param_{fi['param']}"] += 1
        if failed and pieces:
            # the unfitted rest keeps a label of its own, apart from the fitted first piece
            for tri, _ in failed:
                mesh.F[sel[tri]] = next_label
                next_label += 1
    return out, info


# ── sewing baseline (js/brepBuilder.js) ──────────────────────────────────────

def build_sew(mesh, surfaces):
    info = defaultdict(int)
    sew = BRepBuilderAPI_Sewing(max(1e-6, 5e-3 * mesh.diag))
    V, T, F = mesh.V, mesh.T, mesh.F
    for f in np.unique(F):
        sp = surfaces.get(f)
        if not sp:
            info["face_no_surface"] += 1
            continue
        kind, p = sp
        tris = T[F == f]
        pts = V[np.unique(tris)]
        try:
            if kind == "plane":
                # all boundary loops, largest as outer (brepBuilder.planeBoundary)
                cnt = defaultdict(int)
                for i, j, k in tris:
                    for u, v in ((i, j), (j, k), (k, i)):
                        cnt[(min(u, v), max(u, v))] += 1
                nxt = defaultdict(list)
                for i, j, k in tris:
                    for u, v in ((i, j), (j, k), (k, i)):
                        if cnt[(min(u, v), max(u, v))] == 1:
                            nxt[u].append(v)
                seen, loops = set(), []
                for u0 in list(nxt):
                    for v0 in nxt[u0]:
                        if (u0, v0) in seen:
                            continue
                        loop, u, v = [u0], u0, v0
                        while (u, v) not in seen:
                            seen.add((u, v))
                            loop.append(v)
                            cand = [w for w in nxt[v] if (v, w) not in seen]
                            if not cand:
                                break
                            u, v = v, cand[0]
                        loops.append(loop)
                n = unit(p["normal"])
                o = np.asarray(p["origin"], float)
                snap = lambda q: q - ((q - o) @ n) * n
                polys = []
                for loop in loops:
                    q = [snap(V[i]) for i in loop]
                    mp = BRepBuilderAPI_MakePolygon()
                    for x in q:
                        mp.Add(gp_Pnt(*x))
                    mp.Close()
                    ref = np.array([0, 0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0, 0])
                    xa = unit(np.cross(n, ref)); ya = np.cross(n, xa)
                    uv = np.array([[(x - o) @ xa, (x - o) @ ya] for x in q])
                    area = 0.5 * np.sum(uv[:, 0] * np.roll(uv[:, 1], -1) - np.roll(uv[:, 0], -1) * uv[:, 1])
                    polys.append((abs(area), mp.Wire()))
                polys.sort(key=lambda z: -z[0])
                mk = BRepBuilderAPI_MakeFace(gp_Pln(gp_Pnt(*o), gp_Dir(*n)), polys[0][1], True)
                for _, w in polys[1:]:
                    mk.Add(w)
            else:
                surf = geom_surface(kind, p, pts)
                # UV bounds of the patch vertices on the surface (brepBuilder.patchBounds, simplified)
                from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
                us, vs = [], []
                for x in pts[:: max(1, len(pts) // 200)]:
                    pr = GeomAPI_ProjectPointOnSurf(gp_Pnt(*x), surf)
                    u, v = pr.LowerDistanceParameters()
                    us.append(u); vs.append(v)
                us, vs = np.array(us), np.array(vs)
                if kind in ("cylinder", "cone", "sphere", "torus"):
                    # seam is placed away from the patch, so u is continuous across it
                    pass
                mk = BRepBuilderAPI_MakeFace(surf, us.min(), us.max(), vs.min(), vs.max(), 1e-7)
            if mk.IsDone():
                sew.Add(mk.Face())
                info["faces"] += 1
            else:
                info["face_failed"] += 1
        except Exception:
            info["face_exception"] += 1
    sew.Perform()
    return sew.SewedShape(), info


# ── scoring ──────────────────────────────────────────────────────────────────

def score(shape, mesh):
    out = {}
    m = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, m)
    from OCP.BRep import BRep_Tool
    free = bad = 0
    for i in range(1, m.Extent() + 1):
        lst = m.FindFromIndex(i)
        n = lst.Size()
        e = TopoDS.Edge_s(m.FindKey(i))
        if BRep_Tool.Degenerated_s(e):
            continue                                   # apex / pole: one face by nature
        if n == 1 and BRep_Tool.IsClosed_s(e, TopoDS.Face_s(lst.First())):
            continue                                   # seam: the same face on both sides
        if n == 1:
            free += 1
        elif n > 2:
            bad += 1
    out["edges"], out["free_edges"], out["nonmanifold_edges"] = m.Extent(), free, bad

    shells = []
    ex = TopExp_Explorer(shape, TopAbs_SHELL)
    while ex.More():
        shells.append(TopoDS.Shell_s(ex.Current()))
        ex.Next()
    out["shells"] = len(shells)
    out["closed"] = free == 0 and bad == 0 and len(shells) >= 1
    out["valid"] = False
    out["vol_err"] = None
    # One solid per shell: multi-body parts (two separate bodies in one STEP) are legitimate.
    try:
        builder = BRep_Builder()
        from OCP.TopoDS import TopoDS_Compound
        comp = TopoDS_Compound()
        builder.MakeCompound(comp)
        valid = True
        total = 0.0
        for sh in shells:
            fix = ShapeFix_Shell(sh)
            fix.Perform()
            shell = TopoDS.Shell_s(fix.Shell())
            solid = TopoDS_Solid()
            builder.MakeSolid(solid)
            builder.Add(solid, shell)
            cls = BRepClass3d_SolidClassifier(solid)
            cls.PerformInfinitePoint(1e-7)
            if cls.State() == TopAbs_IN:
                solid = TopoDS.Solid_s(solid.Reversed())
            valid = valid and bool(BRepCheck_Analyzer(solid).IsValid())
            g = GProp_GProps()
            BRepGProp.VolumeProperties_s(solid, g)
            total += g.Mass()
            builder.Add(comp, solid)
        out["valid"] = valid and len(shells) >= 1
        out["volume"] = total
        mv = mesh.volume()
        out["vol_err"] = abs(total - mv) / abs(mv) if mv else None
        if TRUE_VOLUMES.get(mesh.part):
            tv = TRUE_VOLUMES[mesh.part]["volume"]
            out["true_vol_err"] = abs(total - tv) / abs(tv)
            out["mesh_vs_true"] = abs(mv - tv) / abs(tv)
        out["solid"] = comp
    except Exception as exc:
        out["error"] = type(exc).__name__
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fits", type=Path, default=Path("D:/abc-data/patch_fits"))
    ap.add_argument("--mesh", type=Path, default=Path("D:/abc-data/fill_eval"))
    ap.add_argument("--truth", type=Path, default=Path("D:/abc-data/fit_truth"))
    ap.add_argument("--meshers", default="native,gmsh")
    ap.add_argument("--method", default="topo,sew")
    ap.add_argument("--surfaces", choices=["fit", "truth"], default="fit")
    ap.add_argument("--parts", default="", help="comma list; default all fully analytic parts")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--step-out", type=Path, default=None)
    ap.add_argument("--repair", choices=["pre", "inplace", "wires", "faces", "faces-rev", "pcurve", "shape"], default=REPAIR_MODE)
    ap.add_argument("--fid-dir", type=Path, default=None, help="patch labels <stem>.fid from here instead of --mesh")
    ap.add_argument("--json-out", type=Path, default=None, help="write the result rows here")
    ap.add_argument("--repair-loop", action="store_true",
                    help="topo through build_repaired (kernel-judged repair moves, test/fitCandidates.mjs)")
    ap.add_argument("--refit", action="store_true", help="with --repair-loop: also try constrained_refit")
    ap.add_argument("--exact-edges", action="store_true",
                    help="exact intersection edges (with --repair-loop: as a move; without: always)")
    ap.add_argument("--freeform", action="store_true",
                    help="a fitted B-spline for every surfaceless group (add_freeform)")
    ap.add_argument("--fallback", action="store_true",
                    help="topo: when the part is not recreated, demote failing patches to facets, then all but "
                         "planes, then the whole mesh (facet_fallback; info.tier says which)")
    ap.add_argument("--freeform-tangent", action="store_true",
                    help="with --freeform: G1 to analytic neighbours (better surfaces, worse knitting; FREEFORM_TANGENT)")
    ap.add_argument("--part-set", choices=["analytic", "freeform", "all"], default="analytic",
                    help="without --parts: fully analytic parts (default), parts with a non-analytic face, or all")
    args = ap.parse_args(argv)
    globals()["REPAIR_MODE"] = args.repair

    manifest = json.loads((args.mesh / "manifest.json").read_text())
    rows = []
    for entry in manifest["parts"]:
        part = entry["part"]
        if args.parts and part not in args.parts.split(","):
            continue
        srf = np.fromfile(args.truth / f"{part}.srf", np.float64).reshape(-1, 12)
        analytic = bool(np.all((srf[:, 0] == srf[:, 1]) & (srf[:, 1] <= 4)))
        if not args.parts and args.part_set != "all" and analytic != (args.part_set == "analytic"):
            continue
        for mesher in args.meshers.split(","):
            v = entry["variants"].get(mesher)
            if not v:
                continue
            fits_path = args.fits / f"{v['stem']}.json"
            if args.surfaces == "fit" and not fits_path.exists():
                continue
            pos = np.fromfile(args.mesh / f"{v['stem']}.pos", np.float32)
            fid = np.fromfile((args.fid_dir or args.mesh) / f"{v['stem']}.fid", np.int32)
            mesh = Mesh(pos, fid)
            mesh.part = part
            if args.surfaces == "fit":
                fits = json.loads(fits_path.read_text())
                surfaces = {f["id"]: (f["type"], f["params"]) for f in fits["faces"] if f["params"]}
            else:
                surfaces = {}
                for f in np.unique(mesh.F):
                    if srf[f, 1] <= 4:
                        kind = KIND[int(srf[f, 1])]
                        cen = mesh.V[np.unique(mesh.T[mesh.F == f])].mean(0)
                        surfaces[int(f)] = (kind, truth_params(kind, srf[f, 2:], cen))
            ff_info = {}
            if args.freeform:
                t_ff = time.time()
                surfaces, ff_info = add_freeform(mesh, surfaces, tangent=args.freeform_tangent)
                ff_info["freeform_secs"] = round(time.time() - t_ff, 2)
            for method in args.method.split(","):
                t0 = time.time()
                try:
                    if method == "topo" and args.repair_loop:
                        state = {}
                        shape, info = build_repaired(mesh, surfaces, node_candidates(args.mesh / f"{v['stem']}.pos"),
                                                     refit=args.refit, exact=args.exact_edges, _state=state)
                        if args.fallback:
                            shape = facet_fallback(mesh, shape, info, state)
                    elif method == "topo" and args.fallback:
                        shape, info, state = topo_state(mesh, surfaces, exact=args.exact_edges)
                        shape = facet_fallback(mesh, shape, info, state)
                    elif method == "topo":
                        shape, info = build_topo(mesh, surfaces, exact=args.exact_edges)
                    else:
                        shape, info = build_sew(mesh, surfaces)
                    sc = score(shape, mesh)
                except Exception as exc:
                    info, sc = {"exception": repr(exc)[:120]}, {"closed": False, "valid": False}
                    if method == "topo" and args.fallback:
                        # The analytic build threw (OCCT's NCollection_Array1::Resize on 2 of 275 Onshape
                        # parts): the mesh itself is the only tier left.
                        try:
                            shape, _ = build_polyhedral(mesh)
                            sc = score(shape, mesh)
                            ok = bool(sc.get("closed") and sc.get("valid") and sc.get("vol_err") is not None
                                      and sc["vol_err"] < 0.01)
                            info.update(tier="faceted" if ok else "none", best_tier="faceted",
                                        facet_tris=int(len(mesh.T)), facet_area=1.0)
                        except Exception as exc2:
                            info["fallback_exception"] = repr(exc2)[:120]
                ratios = info.pop("_area_ratio", None) if hasattr(info, "pop") else None
                true_areas = None if args.fid_dir else (TRUE_VOLUMES.get(part) or {}).get("face_areas")
                if ratios and true_areas:
                    # ratios are against the mesh patch area; re-base them on the true face area,
                    # since a coarse tessellation under-reports curved area by several percent
                    a_, b_, c_ = mesh.V[mesh.T[:, 0]], mesh.V[mesh.T[:, 1]], mesh.V[mesh.T[:, 2]]
                    tri_area = 0.5 * np.linalg.norm(np.cross(b_ - a_, c_ - a_), axis=1)
                    ratios = {f: r * float(tri_area[mesh.F == f].sum()) / true_areas[f] for f, r in ratios.items()}
                if ratios and not args.fid_dir:
                    lr = np.abs(np.log(np.clip(np.array(list(ratios.values()), float), 1e-12, None)))
                    sc["bad_area_faces"] = int(np.sum(~(lr < math.log(1.02 if true_areas else 1.05))))
                    sc["worst_area"] = sorted(((round(v, 3), k) for k, v in ratios.items()),
                                              key=lambda z: -abs(math.log(max(z[0], 1e-12))))[:3]
                row = {"part": part, "mesher": mesher, "method": method, "faces": int(len(np.unique(mesh.F))),
                       "secs": round(time.time() - t0, 2), **{k: val for k, val in sc.items() if k != "solid"},
                       "info": {**dict(ff_info), **dict(info)}}
                rows.append(row)
                if args.step_out and sc.get("solid") is not None:
                    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
                    args.step_out.mkdir(parents=True, exist_ok=True)
                    w = STEPControl_Writer()
                    w.Transfer(sc["solid"], STEPControl_AsIs)
                    w.Write(str(args.step_out / f"{v['stem']}_{method}_{args.surfaces}.step"))
                print(f"  {part:14} {mesher:6} {method:4} faces {row['faces']:4} tier {row['info'].get('tier', '-')} closed {str(row.get('closed')):5} "
                      f"valid {str(row.get('valid')):5} free {row.get('free_edges')} nonman {row.get('nonmanifold_edges')} "
                      f"shells {row.get('shells')} badArea {row.get('bad_area_faces')} {row.get('worst_area')} volErr(mesh) {row.get('vol_err') if row.get('vol_err') is None else round(row['vol_err'], 5)} "
                      f"volErr(STEP) {row.get('true_vol_err') if row.get('true_vol_err') is None else round(row['true_vol_err'], 5)} "
                      f"{row['secs']}s {row['info']}", flush=True)
            if args.limit and len({(r['part'], r['mesher']) for r in rows}) >= args.limit:
                break
        if args.limit and len({(r['part'], r['mesher']) for r in rows}) >= args.limit:
            break

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, default=str))
    print("\nsummary")
    for mesher in args.meshers.split(","):
        for method in args.method.split(","):
            rs = [r for r in rows if r["mesher"] == mesher and r["method"] == method]
            if not rs:
                continue
            ve = [r["true_vol_err"] for r in rs if r.get("true_vol_err") is not None and r.get("valid") and r.get("closed")]
            print(f"  {mesher:6} {method:4} parts {len(rs):3}  closed {sum(bool(r.get('closed')) for r in rs):3}"
                  f"  valid solid {sum(bool(r.get('valid')) for r in rs):3}"
                  f"  closed & valid & STEP volErr<1%: {sum(1 for r in rs if r.get('closed') and r.get('valid') and (r.get('true_vol_err') or 1) < 0.01):3}"
                  f"  median STEP volErr (closed & valid) {np.median(ve) if ve else float('nan'):.2e}"
                  f"  closed & valid & every face area within 2% of STEP: {sum(1 for r in rs if r.get('closed') and r.get('valid') and r.get('bad_area_faces') == 0):3}")
            tiers = defaultdict(int)
            for r in rs:
                tiers[r["info"].get("tier", "-")] += 1
            if set(tiers) != {"-"}:
                print("         tiers: " + "  ".join(f"{k} {v}" for k, v in sorted(tiers.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
