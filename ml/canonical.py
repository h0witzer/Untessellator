#!/usr/bin/env python3
"""
canonical.py - what surface a STEP face really IS, as opposed to how it was written.

ABC's labels come from the B-rep's surface TYPE, which records how the modeller built a
face and how the kernel chose to store it - not what shape it is. Measured on 3,867
SurfaceOfRevolution faces from 500 models: the curve-type check says every profile is a
B-spline, but 2,146 of those are degree-1 (straight lines written as splines), and
tested geometrically 74.8% of all Revolution faces are exactly a cylinder, cone, sphere
or torus. The trained model called those faces Cylinder and Sphere and was scored wrong
for giving the geometrically correct answer.

So the label a model should learn is the SIMPLEST PRIMITIVE THAT THE SURFACE EXACTLY IS,
and only falls back to the stored type when no primitive fits.

METHOD

Sample the face's actual parametric surface - points and normals on a grid over its UV
bounds - and test primitives from simplest to most complex, accepting the first whose
residual is below tolerance relative to the face's extent. Sampling the analytic surface
rather than a tessellation matters: an exact primitive comes out at round-off level, so
"exact" is a clean decision rather than a threshold argued over sagitta.

Every fit here uses the same relations the converter's fitters settled on after each
was found to fail in its obvious form (see js/gaussMap.js): a cone's half-angle from
|n . axis| and its apex from tangent-plane concurrency, a torus from the tube-centre
relation q = p - r n.

Order matters and is simplest-first, because the families nest: a plane is the limit of
a sphere, a cylinder and a cone; a cylinder is the limit of a cone and of a torus. Taking
the first acceptable fit keeps a flat face from being reported as a vast sphere.
"""

from __future__ import annotations

import math

import numpy as np
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepTools import BRepTools
from OCP.GeomAbs import GeomAbs_SurfaceType
from OCP.gp import gp_Pnt, gp_Vec

from occ_tessellate import TYPE_INDEX, face_type_name

_GA = GeomAbs_SurfaceType
PRIMITIVE_TYPES = {_GA.GeomAbs_Plane, _GA.GeomAbs_Cylinder, _GA.GeomAbs_Cone,
                   _GA.GeomAbs_Sphere, _GA.GeomAbs_Torus}


def sample_face(face, n: int = 16):
    """Points and unit normals on an n x n grid over the face's UV bounds."""
    ad = BRepAdaptor_Surface(face)
    umin, umax, vmin, vmax = BRepTools.UVBounds_s(face)
    if not all(math.isfinite(x) for x in (umin, umax, vmin, vmax)):
        return None, None
    if umax <= umin or vmax <= vmin:
        return None, None
    P, N = [], []
    p, du, dv = gp_Pnt(), gp_Vec(), gp_Vec()
    for u in np.linspace(umin, umax, n):
        for v in np.linspace(vmin, vmax, n):
            try:
                ad.D1(float(u), float(v), p, du, dv)
            except Exception:
                continue
            nx = du.Y() * dv.Z() - du.Z() * dv.Y()
            ny = du.Z() * dv.X() - du.X() * dv.Z()
            nz = du.X() * dv.Y() - du.Y() * dv.X()
            L = math.sqrt(nx * nx + ny * ny + nz * nz)
            if not L > 1e-300:
                continue                                  # singular point (pole, apex)
            P.append((p.X(), p.Y(), p.Z()))
            N.append((nx / L, ny / L, nz / L))
    if len(P) < 12:
        return None, None
    return np.array(P), np.array(N)


def _extent(P):
    c = P.mean(0)
    return float(np.max(np.linalg.norm(P - c, axis=1))) or 1.0


def fit_plane(P, N):
    c = P - P.mean(0)
    w, v = np.linalg.eigh(c.T @ c / len(P))
    return math.sqrt(max(w[0], 0.0)), {"normal": v[:, 0]}


def fit_sphere(P, N):
    A = np.column_stack([2 * P, np.ones(len(P))])
    b = (P * P).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:3]
    r2 = sol[3] + c @ c
    if not r2 > 0:
        return math.inf, None
    r = math.sqrt(r2)
    return float(np.sqrt(np.mean((np.linalg.norm(P - c, axis=1) - r) ** 2))), {"r": r}


def _circle2d(X, Y):
    A = np.column_stack([2 * X, 2 * Y, np.ones_like(X)])
    sol, *_ = np.linalg.lstsq(A, X * X + Y * Y, rcond=None)
    cx, cy = sol[0], sol[1]
    r2 = sol[2] + cx * cx + cy * cy
    if not r2 > 0:
        return math.inf, None
    r = math.sqrt(r2)
    return float(np.sqrt(np.mean((np.hypot(X - cx, Y - cy) - r) ** 2))), r


def _frame(a):
    a = a / np.linalg.norm(a)
    t = np.eye(3)[int(np.argmin(np.abs(a)))]
    e1 = np.cross(a, t); e1 /= np.linalg.norm(e1)
    return a, e1, np.cross(a, e1)


def fit_cylinder(P, N):
    # Normals of a cylinder are perpendicular to its axis, so the axis is the direction
    # least represented in the UNCENTRED second moment of the normals.
    w, v = np.linalg.eigh(N.T @ N / len(N))
    a, e1, e2 = _frame(v[:, 0])
    res, r = _circle2d(P @ e1, P @ e2)
    return res, {"r": r}


def fit_cone(P, N):
    # |n . axis| is constant on a cone: the axis is a least-varying direction of the
    # CENTRED normal covariance. On a narrow arc two eigenvalues are nearly equal and
    # the smallest is not reliably the axis - measured, 2.8% of genuine cones fitted to
    # a relative residual of 0.41 - so every eigenvector is tried and scored.
    mu = N.mean(0)
    w, v = np.linalg.eigh((N - mu).T @ (N - mu) / len(N))
    M = N.T @ N
    rhs = N.T @ (N * P).sum(1)
    apex, *_ = np.linalg.lstsq(M, rhs, rcond=None)        # tangent-plane concurrency
    best, info = math.inf, None
    for k in range(3):
        a = v[:, k]
        s_ = float(np.mean(np.abs(N @ a)))
        if not (1e-6 < s_ < 1 - 1e-9):
            continue
        half = math.asin(s_)
        d = P - apex
        rho = np.linalg.norm(d - np.outer(d @ a, a), axis=1)
        for sg in (1.0, -1.0):
            e = rho * math.cos(half) - sg * (d @ a) * math.sin(half)
            val = float(np.sqrt(np.mean(e * e)))
            if val < best:
                best, info = val, {"half": half}
    return best, info


def _torus_res(x, P):
    c = x[:3]
    a = np.array([math.sin(x[3]) * math.cos(x[4]), math.sin(x[3]) * math.sin(x[4]), math.cos(x[3])])
    d = P - c
    z = d @ a
    rho = np.linalg.norm(d - np.outer(z, a), axis=1)
    return np.sqrt((rho - x[5]) ** 2 + z ** 2) - x[6]


def fit_torus(P, N):
    """Tube-centre initialisation, then Levenberg-Marquardt on all seven parameters.

    The tube-centre relation alone reached only ~1e-3 on genuine tori (p99 9.0e-4,
    max 2.2e-3), which overlaps where approximated and freeform surfaces sit and so
    cannot support an exactness decision. It is an excellent starting point, though,
    and from there a local least-squares refinement converges to round-off.

    Spindle tori (r >= R) are accepted: they are legitimate STEP tori, and fillets
    around small holes produce them. Rejecting them caused most outright failures.
    """
    ext = _extent(P)

    def planar(r):
        Q = P - r * N
        c = Q - Q.mean(0)
        w, v = np.linalg.eigh(c.T @ c / len(Q))
        return math.sqrt(max(w[0], 0.0)), Q, v[:, 0]

    inits = []
    for sg in (1.0, -1.0):
        rs = np.linspace(0.01 * ext, 3.0 * ext, 80)
        vals = [planar(sg * r)[0] for r in rs]
        for i in np.argsort(vals)[:2]:
            r = float(rs[i])
            _, Q, ax = planar(sg * r)
            a, e1, e2 = _frame(ax)
            X, Y = Q @ e1, Q @ e2
            A = np.column_stack([2 * X, 2 * Y, np.ones_like(X)])
            sol, *_ = np.linalg.lstsq(A, X * X + Y * Y, rcond=None)
            R2 = sol[2] + sol[0] ** 2 + sol[1] ** 2
            if not R2 > 0:
                continue
            centre = sol[0] * e1 + sol[1] * e2 + (Q @ a).mean() * a
            if a[2] < 0:
                a = -a
            th = math.acos(max(-1.0, min(1.0, a[2])))
            ph = math.atan2(a[1], a[0])
            inits.append(np.array([*centre, th, ph, math.sqrt(R2), r]))
    if not inits:
        return math.inf, None

    best = (math.inf, None)
    n = len(P)
    for x in inits:
        lam = 1e-3
        f = _torus_res(x, P)
        cost = f @ f
        for _ in range(80):
            J = np.empty((n, 7))
            for j in range(7):
                h = (1e-7 * ext) if j in (0, 1, 2, 5, 6) else 1e-7
                xp = x.copy(); xp[j] += h
                J[:, j] = (_torus_res(xp, P) - f) / h
            Ah = J.T @ J
            g = J.T @ f
            try:
                dx = np.linalg.solve(Ah + lam * np.diag(np.diag(Ah) + 1e-30), -g)
            except np.linalg.LinAlgError:
                break
            xn = x + dx
            fn = _torus_res(xn, P)
            cn = fn @ fn
            if cn < cost:
                x, f, cost = xn, fn, cn
                lam = max(lam * 0.3, 1e-12)
                if np.linalg.norm(dx) < 1e-13 * ext:
                    break
            else:
                lam *= 10
                if lam > 1e12:
                    break
        if not (x[5] > 0 and x[6] > 0):
            continue
        val = math.sqrt(cost / n)
        if val < best[0]:
            best = (val, {"R": float(x[5]), "r": float(x[6])})
    return best


ORDER = (("Plane", fit_plane), ("Sphere", fit_sphere), ("Cylinder", fit_cylinder),
         ("Cone", fit_cone), ("Torus", fit_torus))


def canonical_type(face, tol: float = 1e-6, stored: str | None = None):
    """(name, relative residual). The stored type when no primitive fits within tol.

    Faces already stored as an analytic primitive are returned unchanged without
    sampling - the relabel is only ever about faces written as something more general.
    """
    stored = stored or face_type_name(face)
    try:
        st = BRepAdaptor_Surface(face).GetType()
    except Exception:
        return stored, None
    if st in PRIMITIVE_TYPES:
        return stored, 0.0
    P, N = sample_face(face)
    if P is None:
        return stored, None
    ext = _extent(P)
    for name, fn in ORDER:
        try:
            res, _ = fn(P, N)
        except Exception:
            continue
        if res / ext < tol:
            return name, res / ext
    return stored, None


def canonical_index(face, tol: float = 1e-6) -> int:
    name, _ = canonical_type(face, tol)
    return TYPE_INDEX[name]
