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
                   surface-surface intersection, so tangent fillets are well defined)
          face     the patch's surface - U-reversed where its natural normal points into the solid,
                   judged over the patch's triangles - bounded by its edges in mesh winding
          repair   ShapeFix_Shape over the shell, then natural-bound wires it adds to closed surfaces
                   are removed (a torus face otherwise becomes the whole torus minus the patch)
        Every inner edge belongs to exactly two faces by construction, so the shell closes unless the
        mesh is open. Each trap met on the way is recorded where it is handled.

MEASURED (31 fully analytic validation parts, one process per part, "recreated" = closed, BRepCheck
valid, STEP volume within 1%, every face area within 2% of the STEP face):

                      true surfaces   fitted surfaces
    native   topo         29/31           23/31
    gmsh     topo         28/31           22/31
    both     sew           0/31            0/31

Not recreated with TRUE surfaces: 00001644_1 (6 bodies; its mesh pairs into 341 pieces and OCCT
overflows its stack), 00003852_1 (mesh volumes disagree with the STEP by 40%+ - input data), and
00072333_2 on gmsh (5 touching bodies, radial pairing leaves unpaired edges). The extra fitted-surface
failures are almost all small-patch fit ambiguities - 6-facet cylinders fitted perpendicular to the
truth, 11-facet cones 3.5 deg off - which no construction can recover.

INPUT  the fill_eval meshes (.pos/.fid), test/exportPatchFits.mjs output (fitted surfaces), and for
       --surfaces truth the fit_truth sidecars; D:/abc-data/solid_truth_volumes.json for STEP volumes
       and face areas.

Usage (OCCT venv; run parts in separate processes - one broken mesh can overflow OCCT's stack):
    python solid_rebuild.py --fits D:/abc-data/patch_fits --meshers native,gmsh --method topo,sew --parts <id>
"""

from __future__ import annotations

import argparse
import json
import math
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
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_IN, TopAbs_SHELL, TopAbs_SOLID, TopAbs_WIRE
from OCP.BRep import BRep_Tool
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Face, TopoDS_Shell, TopoDS_Solid, TopoDS_Wire
from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape

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
    """Signed distance and unit gradient of points X (n,3) to a surface. Cones are double cones."""
    X = np.atleast_2d(np.asarray(X, float))
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

def build_topo(mesh, surfaces):
    """surfaces: face id -> (kind, params). Returns (shape, info)."""
    V, T, F, scale = mesh.V, mesh.T, mesh.F, mesh.diag
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
        try:
            if both_planes and not closed:
                e = BRepBuilderAPI_MakeEdge(vertex[(path[0], bd)], vertex[(path[-1], bd)])
            else:
                body = pts[:-1] if closed else pts
                clean = [body[0]]
                for q in body[1:]:
                    if np.linalg.norm(q - clean[-1]) > 1e-9 * scale:
                        clean.append(q)
                if not closed:
                    clean[-1] = pts[-1]
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
        if not sp:
            info["face_no_surface"] += 1
            continue
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
    return result, info


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
    args = ap.parse_args(argv)
    globals()["REPAIR_MODE"] = args.repair

    manifest = json.loads((args.mesh / "manifest.json").read_text())
    rows = []
    for entry in manifest["parts"]:
        part = entry["part"]
        if args.parts and part not in args.parts.split(","):
            continue
        srf = np.fromfile(args.truth / f"{part}.srf", np.float64).reshape(-1, 12)
        if not args.parts and not np.all((srf[:, 0] == srf[:, 1]) & (srf[:, 1] <= 4)):
            continue                                   # fully analytic parts only
        for mesher in args.meshers.split(","):
            v = entry["variants"].get(mesher)
            if not v:
                continue
            fits_path = args.fits / f"{v['stem']}.json"
            if args.surfaces == "fit" and not fits_path.exists():
                continue
            pos = np.fromfile(args.mesh / f"{v['stem']}.pos", np.float32)
            fid = np.fromfile(args.mesh / f"{v['stem']}.fid", np.int32)
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
            for method in args.method.split(","):
                t0 = time.time()
                try:
                    shape, info = (build_topo if method == "topo" else build_sew)(mesh, surfaces)
                    sc = score(shape, mesh)
                except Exception as exc:
                    info, sc = {"exception": repr(exc)[:120]}, {"closed": False, "valid": False}
                ratios = info.pop("_area_ratio", None) if hasattr(info, "pop") else None
                true_areas = (TRUE_VOLUMES.get(part) or {}).get("face_areas")
                if ratios and true_areas:
                    # ratios are against the mesh patch area; re-base them on the true face area,
                    # since a coarse tessellation under-reports curved area by several percent
                    a_, b_, c_ = mesh.V[mesh.T[:, 0]], mesh.V[mesh.T[:, 1]], mesh.V[mesh.T[:, 2]]
                    tri_area = 0.5 * np.linalg.norm(np.cross(b_ - a_, c_ - a_), axis=1)
                    ratios = {f: r * float(tri_area[mesh.F == f].sum()) / true_areas[f] for f, r in ratios.items()}
                if ratios:
                    lr = np.abs(np.log(np.clip(np.array(list(ratios.values()), float), 1e-12, None)))
                    sc["bad_area_faces"] = int(np.sum(~(lr < math.log(1.02 if true_areas else 1.05))))
                    sc["worst_area"] = sorted(((round(v, 3), k) for k, v in ratios.items()),
                                              key=lambda z: -abs(math.log(max(z[0], 1e-12))))[:3]
                row = {"part": part, "mesher": mesher, "method": method, "faces": int(len(np.unique(mesh.F))),
                       "secs": round(time.time() - t0, 2), **{k: val for k, val in sc.items() if k != "solid"},
                       "info": dict(info)}
                rows.append(row)
                if args.step_out and sc.get("solid") is not None:
                    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
                    args.step_out.mkdir(parents=True, exist_ok=True)
                    w = STEPControl_Writer()
                    w.Transfer(sc["solid"], STEPControl_AsIs)
                    w.Write(str(args.step_out / f"{v['stem']}_{method}_{args.surfaces}.step"))
                print(f"  {part:14} {mesher:6} {method:4} faces {row['faces']:4} closed {str(row.get('closed')):5} "
                      f"valid {str(row.get('valid')):5} free {row.get('free_edges')} nonman {row.get('nonmanifold_edges')} "
                      f"shells {row.get('shells')} badArea {row.get('bad_area_faces')} {row.get('worst_area')} volErr(mesh) {row.get('vol_err') if row.get('vol_err') is None else round(row['vol_err'], 5)} "
                      f"volErr(STEP) {row.get('true_vol_err') if row.get('true_vol_err') is None else round(row['true_vol_err'], 5)} "
                      f"{row['secs']}s {row['info']}", flush=True)
            if args.limit and len({(r['part'], r['mesher']) for r in rows}) >= args.limit:
                break
        if args.limit and len({(r['part'], r['mesher']) for r in rows}) >= args.limit:
            break

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
