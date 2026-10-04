#!/usr/bin/env python3
"""
surface_targets.py - the surface hypothesis the seeded-fill model predicts, and its inverse.

WHY THIS EXISTS

paintFrom grows a fill by testing facets against a FITTED surface, and the fitters fail on
exactly the small regions a click provides: 12-facet torus seeds come back degenerate 46% of
the time, and small cone seeds fit poorly. A surface predicted from the whole patch can gate
growth until enough facets exist for an exact fit. Nothing predicted here reaches the output:
the analytic fitters still fit whatever facets end up selected.

WHAT IS PREDICTED

Five numbers, in the patch's canonical frame (seed centroid at the origin, seed facet normal
along +Z, lengths divided by the patch scale). No position and no unbounded radius is
regressed directly; everything else follows from these plus the seed facet (reconstruct()).

  axis (3)  the surface axis, sign-free (cylinder, cone, torus)
  kn        signed normal curvature, scale / d. d is the distance from the seed along its facet
            normal to: the axis (cylinder), the centre (sphere), the axis via the normal line
            (cone, d = rho / cos(half-angle)), the tube centre (torus, d = r). Positive when
            that point lies BEHIND the facet normal. Zero for a plane.
  ks        torus only: side * scale / R. side is +1 when the seed lies on the outer half of the
            tube, which fixes which way the spine centre sits from the tube centre.

A cone's half-angle is not a separate output: |n . axis| = sin(half-angle) holds exactly on
the surface, so it follows from the axis in the frame where n is +Z.

kn and ks are stored as asinh(value). Fillets make scale / r large and flat faces make it
vanish; asinh keeps the sign and compresses the tail.

SIDECAR LAYOUT (surface_sidecar.py), float64 (n_faces, 10), by STORED kind:

  Plane     origin(3) normal(3)
  Cylinder  axis point(3) axis(3) R
  Cone      apex(3) axis(3) half-angle
  Sphere    centre(3) - - - R
  Torus     centre(3) axis(3) R r
"""

from __future__ import annotations

import math

import numpy as np

from occ_tessellate import TYPE_NAMES

PLANE, CYL, CONE, SPHERE, TORUS = (TYPE_NAMES.index(t) for t in
                                   ("Plane", "Cylinder", "Cone", "Sphere", "Torus"))
ANALYTIC = (PLANE, CYL, CONE, SPHERE, TORUS)
N_SURFACE = 5          # axis(3), asinh(kn), asinh(ks)
KN_CLIP = 1e4

_NONE = (np.zeros(3), 0.0, 0.0, 0.0, 0.0, 0.0)


def _sgn(x: float) -> float:
    return 1.0 if x >= 0 else -1.0


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-300 else None


def targets(kind: int, p: np.ndarray, q: np.ndarray, n: np.ndarray, R: np.ndarray, scale: float):
    """Targets for a seed with centroid q and unit facet normal n (world), frame rotation R.

    Returns (axis3, has_axis, asinh_kn, has_kn, asinh_ks, has_ks).
    """
    if kind == PLANE:
        return np.zeros(3), 0.0, 0.0, 1.0, 0.0, 0.0
    if kind == SPHERE:
        u = _unit(q - p[0:3])
        if u is None or p[6] <= 0:
            return _NONE
        kn = _sgn(float(n @ u)) * scale / p[6]
        return np.zeros(3), 0.0, math.asinh(kn), 1.0, 0.0, 0.0
    a = _unit(p[3:6])
    if a is None:
        return _NONE
    ax = R @ a
    if kind == CYL:
        v = q - p[0:3]
        u = _unit(v - (v @ a) * a)
        if u is None or p[6] <= 0:
            return ax, 1.0, 0.0, 0.0, 0.0, 0.0
        kn = _sgn(float(n @ u)) * scale / p[6]
        return ax, 1.0, math.asinh(kn), 1.0, 0.0, 0.0
    if kind == CONE:
        w = q - p[0:3]
        radial = w - (w @ a) * a
        rho = float(np.linalg.norm(radial))
        if rho < 1e-300:
            return ax, 1.0, 0.0, 0.0, 0.0, 0.0
        kn = _sgn(float(n @ (radial / rho))) * scale * math.cos(p[6]) / rho
        return ax, 1.0, math.asinh(max(-KN_CLIP, min(KN_CLIP, kn))), 1.0, 0.0, 0.0
    if kind == TORUS:
        w = q - p[0:3]
        u = _unit(w - (w @ a) * a)
        if u is None or p[6] <= 0 or p[7] <= 0:
            return ax, 1.0, 0.0, 0.0, 0.0, 0.0
        t = p[0:3] + p[6] * u                        # tube centre nearest the seed
        sn = _sgn(float(n @ (q - t)))
        kn = sn * scale / p[7]
        nperp = n - (n @ a) * a
        side = _sgn(sn * float(nperp @ u))
        ks = side * scale / p[6]
        return ax, 1.0, math.asinh(max(-KN_CLIP, min(KN_CLIP, kn))), 1.0, math.asinh(ks), 1.0
    return _NONE


def facet_targets(kinds: np.ndarray, params: np.ndarray, face_ids: np.ndarray, cen: np.ndarray,
                  nrm: np.ndarray, R: np.ndarray, scale: float):
    """targets() for EVERY facet of a patch: each facet against its OWN face's surface, evaluated
    at that facet's centroid and normal, in the patch frame. Vectorised per face; agrees with
    targets() facet by facet (test_surface_targets.py --facets). Feeds the training-only per-facet
    surface head (model.SeedFillNet facet_surface).

    Returns (targets (n, 5) float32, masks (n, 3) float32: has axis, has kn, has ks).
    """
    n = len(face_ids)
    tgt = np.zeros((n, N_SURFACE), np.float32)
    msk = np.zeros((n, 3), np.float32)
    nrm = np.asarray(nrm, np.float64)
    nrm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-30)
    cen = np.asarray(cen, np.float64)
    sgn = lambda x: np.where(x >= 0, 1.0, -1.0)
    for f in np.unique(face_ids):
        if f >= len(kinds):
            continue
        sel = np.nonzero(face_ids == f)[0]
        kind, p = int(kinds[f]), params[f]
        Q, Nn = cen[sel], nrm[sel]
        if kind == PLANE:
            msk[sel, 1] = 1.0
            continue
        if kind == SPHERE:
            if p[6] <= 0:
                continue
            u = Q - p[0:3]
            ok = np.linalg.norm(u, axis=1) > 1e-300
            kn = sgn((Nn * u).sum(1)) * scale / p[6]
            tgt[sel[ok], 3] = np.arcsinh(kn[ok])
            msk[sel[ok], 1] = 1.0
            continue
        if kind not in (CYL, CONE, TORUS):
            continue
        a = _unit(p[3:6])
        if a is None:
            continue
        tgt[sel, 0:3] = R @ a
        msk[sel, 0] = 1.0
        w = Q - p[0:3]
        perp = w - np.outer(w @ a, a)
        pn = np.linalg.norm(perp, axis=1)
        safe = np.maximum(pn, 1e-300)[:, None]
        if kind == CYL:
            ok = (pn > 1e-300) & (p[6] > 0)
            kn = sgn((Nn * (perp / safe)).sum(1)) * scale / p[6] if p[6] > 0 else np.zeros(len(sel))
            tgt[sel[ok], 3] = np.arcsinh(kn[ok])
            msk[sel[ok], 1] = 1.0
        elif kind == CONE:
            ok = pn > 1e-300
            kn = sgn((Nn * (perp / safe)).sum(1)) * scale * math.cos(p[6]) / np.maximum(pn, 1e-300)
            tgt[sel[ok], 3] = np.arcsinh(np.clip(kn[ok], -KN_CLIP, KN_CLIP))
            msk[sel[ok], 1] = 1.0
        else:  # TORUS
            if p[6] <= 0 or p[7] <= 0:
                continue
            ok = pn > 1e-300
            u = perp / safe
            t = p[0:3] + p[6] * u
            sn = sgn((Nn * (Q - t)).sum(1))
            kn = sn * scale / p[7]
            nperp = Nn - np.outer(Nn @ a, a)
            side = sgn(sn * (nperp * u).sum(1))
            ks = side * scale / p[6]
            tgt[sel[ok], 3] = np.arcsinh(np.clip(kn[ok], -KN_CLIP, KN_CLIP))
            tgt[sel[ok], 4] = np.arcsinh(ks[ok])
            msk[sel[ok], 1] = 1.0
            msk[sel[ok], 2] = 1.0
    return tgt, msk


def reconstruct(kind, axis, asinh_kn: float, asinh_ks: float = 0.0):
    """The surface implied by (axis, kn, ks) in the patch frame: seed at the origin, facet normal
    +Z, unit length = scale. Returns (kind, params) or None when the numbers do not define one."""
    if kind is None:
        return None
    ez = np.array([0.0, 0.0, 1.0])
    kn = math.sinh(float(asinh_kn))
    if kind == PLANE or abs(kn) < 1e-6:
        return PLANE, {"origin": np.zeros(3), "normal": ez}
    s = 1.0 if kn > 0 else -1.0
    d = 1.0 / abs(kn)
    if kind == SPHERE:
        return SPHERE, {"centre": -s * d * ez, "R": d}
    a = _unit(axis)
    if a is None:
        return None
    if kind == CYL:
        return CYL, {"point": -s * d * ez, "axis": a, "R": d}
    if kind == CONE:
        sina = min(abs(float(a @ ez)), 0.999999)
        if sina < 1e-6:
            return CYL, {"point": -s * d * ez, "axis": a, "R": d}
        a_open = -_sgn(s * float(ez @ a)) * a          # radius grows along a_open
        apex = -s * d * ez - (d / sina) * a_open
        return CONE, {"apex": apex, "axis": a_open, "half": math.asin(sina)}
    if kind == TORUS:
        ks = math.sinh(float(asinh_ks))
        if abs(ks) < 1e-9:
            return None
        up = _unit(ez - float(ez @ a) * a)
        if up is None:
            return None
        u = (1.0 if ks > 0 else -1.0) * s * up
        t = -s * d * ez
        big_r = 1.0 / abs(ks)
        return TORUS, {"centre": t - big_r * u, "axis": a, "R": big_r, "r": d}
    return None


def to_frame(kind: int, p: np.ndarray, q: np.ndarray, R: np.ndarray, scale: float):
    """A sidecar surface (world) expressed in a patch frame, in reconstruct()'s format."""
    pt = lambda x: R @ (np.asarray(x, float) - q) / scale
    if kind == PLANE:
        return PLANE, {"origin": pt(p[0:3]), "normal": R @ _unit(p[3:6])}
    if kind == SPHERE:
        return SPHERE, {"centre": pt(p[0:3]), "R": p[6] / scale}
    a = _unit(p[3:6])
    if kind == CYL:
        return CYL, {"point": pt(p[0:3]), "axis": R @ a, "R": p[6] / scale}
    if kind == CONE:
        if (q - p[0:3]) @ a < 0:                      # orient toward the nappe the seed is on
            a = -a
        return CONE, {"apex": pt(p[0:3]), "axis": R @ a, "half": p[6]}
    if kind == TORUS:
        return TORUS, {"centre": pt(p[0:3]), "axis": R @ a, "R": p[6] / scale, "r": p[7] / scale}
    return None


def distance(kind: int, prm: dict, X: np.ndarray):
    """Signed distance of points X (n, 3) to a surface, and the unit surface normal there.
    Same conventions as js/regionGrow.js surfaceEval."""
    X = np.asarray(X, float)
    if kind == PLANE:
        nr = _unit(prm["normal"])
        return (X - prm["origin"]) @ nr, np.broadcast_to(nr, X.shape)
    if kind == SPHERE:
        v = X - prm["centre"]
        r = np.linalg.norm(v, axis=1)
        return r - prm["R"], v / np.maximum(r, 1e-300)[:, None]
    a = _unit(prm["axis"])
    if kind == CYL:
        v = X - prm["point"]
        rad = v - np.outer(v @ a, a)
        rho = np.linalg.norm(rad, axis=1)
        return rho - prm["R"], rad / np.maximum(rho, 1e-300)[:, None]
    if kind == CONE:
        w = X - prm["apex"]
        z = w @ a
        rad = w - np.outer(z, a)
        rho = np.linalg.norm(rad, axis=1)
        u = rad / np.maximum(rho, 1e-300)[:, None]
        ca, sa = math.cos(prm["half"]), math.sin(prm["half"])
        nrm = u * ca - a[None, :] * sa
        return rho * ca - z * sa, nrm / np.maximum(np.linalg.norm(nrm, axis=1), 1e-300)[:, None]
    if kind == TORUS:
        w = X - prm["centre"]
        z = w @ a
        rad = w - np.outer(z, a)
        rho = np.linalg.norm(rad, axis=1)
        u = rad / np.maximum(rho, 1e-300)[:, None]
        v = X - (prm["centre"] + prm["R"] * u)
        vn = np.linalg.norm(v, axis=1)
        return vn - prm["r"], v / np.maximum(vn, 1e-300)[:, None]
    raise ValueError(kind)
