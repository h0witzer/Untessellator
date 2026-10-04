#!/usr/bin/env python3
"""
nurbs.py - tensor-product (rational) B-spline surfaces in numpy: basis functions and their
derivatives, evaluation, and closest-point projection.

Shared by the freeform fitter (freeform_fit.py), the true ABC surfaces it is graded against
(abc_freeform.py) and the knitter (solid_rebuild.py's "bspline" kind). numpy only, like the rest of
the rebuild path.

Knot vectors are FLAT (every knot repeated by its multiplicity), length n + p + 1 for n control
points of degree p, and need not be clamped: ABC stores closed surfaces with unclamped vectors such as
[-0.5, 0, 0, 0, 0.5, ...]. The valid domain of a direction is [t[p], t[n]].
"""

from __future__ import annotations

import math

import numpy as np

# A single-point query this close to the previous one (x the control net's size) starts Newton where
# that one ended instead of at the nearest grid sample.
WARM_RADIUS = 0.02
GRID_PER_SIZE = 150            # projection start samples per control-net size, along the longest row
GRID_MAX = 250_000


def span_of(t, p, x):
    """Knot span index s (t[s] <= x < t[s+1], non-empty) of each x, clamped to the domain; the
    nonzero basis functions there are s - p .. s."""
    t = np.asarray(t, float)
    n = len(t) - p - 1
    x = np.clip(np.atleast_1d(np.asarray(x, float)), t[p], t[n])
    span = np.minimum(np.searchsorted(t, x, side="right") - 1, n - 1)
    # the right end, and any x on a repeated knot at the end, fall back into the last real span
    while True:
        bad = t[span + 1] <= t[span]
        if not bad.any():
            break
        span = np.where(bad, span - 1, span)
    return span


def nz_basis(t, p, x, nder=0):
    """The p + 1 nonzero basis functions at each x and their derivatives (NURBS Book A2.3,
    vectorised over x): (first index (m,), [(m, p+1) for orders 0..nder]); function first + j."""
    t = np.asarray(t, float)
    n = len(t) - p - 1
    x = np.clip(np.atleast_1d(np.asarray(x, float)), t[p], t[n])
    s = span_of(t, p, x)
    m = len(x)
    ndu = np.zeros((m, p + 1, p + 1))
    ndu[:, 0, 0] = 1.0
    left = np.zeros((m, p + 1))
    right = np.zeros((m, p + 1))
    for j in range(1, p + 1):
        left[:, j] = x - t[s + 1 - j]
        right[:, j] = t[s + j] - x
        saved = np.zeros(m)
        for r in range(j):
            ndu[:, j, r] = right[:, r + 1] + left[:, j - r]
            temp = ndu[:, r, j - 1] / ndu[:, j, r]
            ndu[:, r, j] = saved + right[:, r + 1] * temp
            saved = left[:, j - r] * temp
        ndu[:, j, j] = saved
    ders = np.zeros((nder + 1, m, p + 1))
    ders[0] = ndu[:, :, p]
    for r in range(p + 1):
        a = np.zeros((2, m, p + 1))
        a[0, :, 0] = 1.0
        s1, s2 = 0, 1
        for k in range(1, nder + 1):
            d = np.zeros(m)
            rk, pk = r - k, p - k
            if r >= k:
                a[s2, :, 0] = a[s1, :, 0] / ndu[:, pk + 1, rk]
                d = a[s2, :, 0] * ndu[:, rk, pk]
            j1 = 1 if rk >= -1 else -rk
            j2 = k - 1 if r - 1 <= pk else p - r
            for j in range(j1, j2 + 1):
                a[s2, :, j] = (a[s1, :, j] - a[s1, :, j - 1]) / ndu[:, pk + 1, rk + j]
                d = d + a[s2, :, j] * ndu[:, rk + j, pk]
            if r <= pk:
                a[s2, :, k] = -a[s1, :, k - 1] / ndu[:, pk + 1, r]
                d = d + a[s2, :, k] * ndu[:, r, pk]
            ders[k, :, r] = d
            s1, s2 = s2, s1
    f = p
    for k in range(1, nder + 1):
        ders[k] *= f
        f *= p - k
    return s - p, list(ders)


def basis(t, p, x, nder=0):
    """Dense basis matrices [N, N', ..., N^(nder)], each (len(x), n), of degree p on flat knots t.
    x is clamped to the domain [t[p], t[n]]; its right end belongs to the last non-empty span."""
    t = np.asarray(t, float)
    x = np.atleast_1d(np.asarray(x, float))
    n = len(t) - p - 1
    x = np.clip(x, t[p], t[n])
    m = len(x)
    span = span_of(t, p, x)
    B = np.zeros((m, len(t) - 1))
    B[np.arange(m), span] = 1.0
    # D[k][r]: r-th derivative of the degree-k basis, shape (m, len(t) - k - 1)
    D = [[B]]
    for k in range(1, p + 1):
        nb = len(t) - k - 1
        d1 = t[k:k + nb] - t[:nb]
        d2 = t[k + 1:k + 1 + nb] - t[1:1 + nb]
        i1 = np.where(d1 > 0, 1.0 / np.where(d1 > 0, d1, 1.0), 0.0)
        i2 = np.where(d2 > 0, 1.0 / np.where(d2 > 0, d2, 1.0), 0.0)
        prev = D[k - 1]
        cur = [(x[:, None] - t[:nb]) * i1 * prev[0][:, :nb] + (t[k + 1:k + 1 + nb] - x[:, None]) * i2 * prev[0][:, 1:nb + 1]]
        for r in range(1, min(nder, k) + 1):
            cur.append(k * (i1 * prev[r - 1][:, :nb] - i2 * prev[r - 1][:, 1:nb + 1]))
        D.append(cur)
    out = D[p]
    while len(out) < nder + 1:
        out.append(np.zeros((m, n)))
    return out


class Surface:
    """A (rational) tensor-product B-spline surface.

    poles (nu, nv, 3), weights (nu, nv) or None, flat knots tu, tv, degrees pu, pv."""

    def __init__(self, poles, tu, tv, pu, pv, weights=None):
        self.P = np.asarray(poles, float)
        self.tu, self.tv = np.asarray(tu, float), np.asarray(tv, float)
        self.pu, self.pv = int(pu), int(pv)
        self.W = None if weights is None else np.asarray(weights, float)
        if self.W is not None and np.allclose(self.W, self.W.flat[0]):
            self.W = None                                  # uniform weights: polynomial
        nu, nv = self.P.shape[:2]
        assert len(self.tu) == nu + self.pu + 1 and len(self.tv) == nv + self.pv + 1, "knots / poles mismatch"
        self.dom = (self.tu[self.pu], self.tu[nu], self.tv[self.pv], self.tv[nv])
        self._grid = None
        self._last = None
        self._size = None
        self._closed = None
        self._Pw = None

    # ── evaluation ──
    def eval(self, u, v, nder=0):
        """Points (and with nder >= 1 the first, with 2 the second partials): returns a dict with
        S, and Su Sv, Suu Suv Svv as requested, each (m, 3)."""
        fu, Bu = nz_basis(self.tu, self.pu, u, nder)
        fv, Bv = nz_basis(self.tv, self.pv, v, nder)
        iu = fu[:, None] + np.arange(self.pu + 1)
        iv = fv[:, None] + np.arange(self.pv + 1)
        if self.W is None:
            blk = self.P[iu[:, :, None], iv[:, None, :]]           # (m, pu+1, pv+1, 3)

            def comb(a, b):
                return np.einsum("mi,mj,mijc->mc", Bu[a], Bv[b], blk)
            out = {"S": comb(0, 0)}
            if nder >= 1:
                out["Su"], out["Sv"] = comb(1, 0), comb(0, 1)
            if nder >= 2:
                out["Suu"], out["Suv"], out["Svv"] = comb(2, 0), comb(1, 1), comb(0, 2)
            return out
        if self._Pw is None:
            self._Pw = np.concatenate([self.P * self.W[..., None], self.W[..., None]], axis=2)
        blk = self._Pw[iu[:, :, None], iv[:, None, :]]

        def comb(a, b):
            return np.einsum("mi,mj,mijc->mc", Bu[a], Bv[b], blk)
        A = comb(0, 0)
        w = A[:, 3:4]
        S = A[:, :3] / w
        out = {"S": S}
        if nder >= 1:
            Au, Av = comb(1, 0), comb(0, 1)
            Su = (Au[:, :3] - Au[:, 3:4] * S) / w
            Sv = (Av[:, :3] - Av[:, 3:4] * S) / w
            out["Su"], out["Sv"] = Su, Sv
        if nder >= 2:
            Auu, Auv, Avv = comb(2, 0), comb(1, 1), comb(0, 2)
            out["Suu"] = (Auu[:, :3] - 2 * Au[:, 3:4] * Su - Auu[:, 3:4] * S) / w
            out["Svv"] = (Avv[:, :3] - 2 * Av[:, 3:4] * Sv - Avv[:, 3:4] * S) / w
            out["Suv"] = (Auv[:, :3] - Au[:, 3:4] * Sv - Av[:, 3:4] * Su - Auv[:, 3:4] * S) / w
        return out

    def normal(self, u, v):
        d = self.eval(u, v, 1)
        n = np.cross(d["Su"], d["Sv"])
        return d["S"], n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)

    # ── projection ──
    def grid(self):
        """Dense samples for projection starts, spaced by LENGTH, not by knot span: about
        GRID_PER_SIZE samples per control-net size along the longest control-polygon row / column
        (and 8 per span, 24 at least). A long closed profile folding into a narrow slot put a
        truth extrusion's nearest span-spaced sample on the slot's far wall (0.7% of the diagonal
        off; exact from a closer start)."""
        if self._grid is None:
            u0, u1, v0, v1 = self.dom
            h = self.size() / GRID_PER_SIZE
            Lu = np.linalg.norm(np.diff(self.P, axis=0), axis=2).sum(0).max() if self.P.shape[0] > 1 else 0.0
            Lv = np.linalg.norm(np.diff(self.P, axis=1), axis=2).sum(1).max() if self.P.shape[1] > 1 else 0.0
            nu = max(24, 8 * (len(np.unique(self.tu)) - 1), int(math.ceil(Lu / h)))
            nv = max(24, 8 * (len(np.unique(self.tv)) - 1), int(math.ceil(Lv / h)))
            nu, nv = min(nu, 1500), min(nv, 1500)
            while nu * nv > GRID_MAX:
                nu, nv = max(24, int(nu * 0.8)), max(24, int(nv * 0.8))
            us, vs = np.linspace(u0, u1, nu), np.linspace(v0, v1, nv)
            U, V = np.meshgrid(us, vs, indexing="ij")
            U, V = U.ravel(), V.ravel()
            self._grid = (U, V, self.eval(U, V)["S"])
        return self._grid

    def closest(self, X, uv0=None, iters=25):
        """Closest points on the surface (within its domain) to X (m, 3): returns (u, v, S).
        Starts from uv0 if given, else from the nearest grid sample, then Newton on |S - X|^2 with
        a Gauss-Newton fallback where the Hessian is not positive definite."""
        X = np.atleast_2d(np.asarray(X, float))
        u0, u1, v0, v1 = self.dom
        if uv0 is None and len(X) == 1 and self._last is not None:
            # The knitter projects one point at a time, each a small step from the last (Gauss-Newton
            # onto several surfaces at once): start where the previous query ended.
            lx, lu, lv = self._last
            if np.linalg.norm(X[0] - lx) < WARM_RADIUS * self.size():
                uv0 = (np.array([lu]), np.array([lv]))
        if uv0 is None:
            U, V, G = self.grid()
            idx = np.empty(len(X), np.int64)
            gg = (G * G).sum(1)
            for s in range(0, len(X), 2048):
                x = X[s:s + 2048]
                d = gg[None, :] - 2 * x @ G.T
                idx[s:s + 2048] = np.argmin(d, axis=1)
            u, v = U[idx].copy(), V[idx].copy()
        else:
            u, v = np.array(uv0[0], float), np.array(uv0[1], float)
        cu, cv = self.closed()

        def fold(a, lo, hi, closed):
            # a closed direction wraps across its seam; an open one stops at its end
            return lo + np.mod(a - lo, hi - lo) if closed else np.clip(a, lo, hi)
        prev = None
        for _ in range(iters):
            d = self.eval(u, v, 2)
            r = d["S"] - X
            dist = (r * r).sum(1)
            if prev is not None:
                # backtrack where the step made things worse (at most a few halvings, all at once)
                worse = dist > prev[2] * (1 + 1e-12) + 1e-300
                h = 1.0
                while worse.any() and h > 1 / 16:
                    h /= 2
                    u = np.where(worse, fold(prev[0] + h * (u - prev[0]), u0, u1, cu), u)
                    v = np.where(worse, fold(prev[1] + h * (v - prev[1]), v0, v1, cv), v)
                    d = self.eval(u, v, 2)
                    r = d["S"] - X
                    dist = (r * r).sum(1)
                    worse = dist > prev[2] * (1 + 1e-12) + 1e-300
                if worse.any():
                    u, v = np.where(worse, prev[0], u), np.where(worse, prev[1], v)
                    d = self.eval(u, v, 2)
                    r = d["S"] - X
                    dist = (r * r).sum(1)
            prev = (u.copy(), v.copy(), dist)
            gu, gv = (r * d["Su"]).sum(1), (r * d["Sv"]).sum(1)
            a = (d["Su"] * d["Su"]).sum(1)
            b = (d["Su"] * d["Sv"]).sum(1)
            c = (d["Sv"] * d["Sv"]).sum(1)
            h11 = a + (r * d["Suu"]).sum(1)
            h12 = b + (r * d["Suv"]).sum(1)
            h22 = c + (r * d["Svv"]).sum(1)
            det = h11 * h22 - h12 * h12
            pd = (h11 > 0) & (det > 1e-12 * np.maximum(a * c, 1e-300))
            # Gauss-Newton where the full Hessian is indefinite
            h11 = np.where(pd, h11, a)
            h12 = np.where(pd, h12, b)
            h22 = np.where(pd, h22, c)
            det = h11 * h22 - h12 * h12
            det = np.where(np.abs(det) > 1e-300, det, 1e-300)
            du = -(h22 * gu - h12 * gv) / det
            dv = -(h11 * gv - h12 * gu) / det
            # no single step longer than half the domain
            du = np.clip(du, -(u1 - u0) / 2, (u1 - u0) / 2)
            dv = np.clip(dv, -(v1 - v0) / 2, (v1 - v0) / 2)
            nu_, nv_ = fold(u + du, u0, u1, cu), fold(v + dv, v0, v1, cv)
            step = np.abs(du) / max(u1 - u0, 1e-300) + np.abs(dv) / max(v1 - v0, 1e-300)
            u, v = nu_, nv_
            if step.max() < 1e-12:
                break
        if len(X) == 1:
            self._last = (X[0].copy(), float(u[0]), float(v[0]))
        return u, v, self.eval(u, v)["S"]

    def closed(self):
        """(closed in u, closed in v), judged geometrically: the two ends of a direction coincide
        (ABC stores periodic surfaces with unclamped knots, so the flags are not in the knots)."""
        if self._closed is None:
            u0, u1, v0, v1 = self.dom
            s = np.linspace(0, 1, 7)
            tol = 1e-9 * self.size()
            a = self.eval(np.full(7, u0), v0 + s * (v1 - v0))["S"] - self.eval(np.full(7, u1), v0 + s * (v1 - v0))["S"]
            b = self.eval(u0 + s * (u1 - u0), np.full(7, v0))["S"] - self.eval(u0 + s * (u1 - u0), np.full(7, v1))["S"]
            self._closed = (bool(np.abs(a).max() < tol), bool(np.abs(b).max() < tol))
        return self._closed

    def size(self):
        if self._size is None:
            P = self.P.reshape(-1, 3)
            self._size = float(np.linalg.norm(P.max(0) - P.min(0))) or 1.0
        return self._size

    def signed_distance(self, X):
        """Signed distance of X along the natural normal Su x Sv at the closest point, and that
        normal: solid_rebuild.dist_normal's contract."""
        X = np.atleast_2d(np.asarray(X, float))
        u, v, _ = self.closest(X)
        S, n = self.normal(u, v)
        return ((X - S) * n).sum(1), n

    # ── (de)serialisation ──
    def to_params(self):
        return {"poles": self.P.tolist(), "tu": self.tu.tolist(), "tv": self.tv.tolist(), "pu": self.pu, "pv": self.pv,
                "weights": None if self.W is None else self.W.tolist()}

    @staticmethod
    def from_params(p):
        return Surface(p["poles"], p["tu"], p["tv"], p["pu"], p["pv"], p.get("weights"))


def surface_of(p):
    """The Surface for a params dict, built once and cached in the dict (under a private key that
    serialisers skip)."""
    s = p.get("_surface")
    if s is None:
        s = Surface.from_params(p)
        p["_surface"] = s
    return s


def to_occ(p):
    """Geom_BSplineSurface for a params dict (flat knots -> distinct knots + multiplicities).

    params with "u_periodic" (freeform_fit's rings and tubes) hold the unclamped form - the first
    pu columns of poles repeated at the end, knots running pu past each end of the domain - which
    becomes an OCCT PERIODIC surface: the distinct poles only, the domain's knots with multiplicity 1.
    OCCT then treats the face like a cylinder's (the repair adds its seam)."""
    P = np.asarray(p["poles"], float)
    if p.get("u_periodic"):
        pu = int(p["pu"])
        n = P.shape[0] - pu
        tu = np.asarray(p["tu"], float)[pu:pu + n + 1]
        q = dict(p, poles=P[:n].tolist(), tu=list(tu), u_periodic=False,
                 weights=None if p.get("weights") is None else np.asarray(p["weights"])[:n].tolist())
        return _occ_surface(q, periodic_u=True)
    return _occ_surface(p)


def _occ_surface(p, periodic_u=False):
    from OCP.Geom import Geom_BSplineSurface
    from OCP.TColgp import TColgp_Array2OfPnt
    from OCP.TColStd import TColStd_Array1OfInteger, TColStd_Array1OfReal, TColStd_Array2OfReal
    from OCP.gp import gp_Pnt
    P = np.asarray(p["poles"], float)
    nu, nv = P.shape[:2]
    poles = TColgp_Array2OfPnt(1, nu, 1, nv)
    for i in range(nu):
        for j in range(nv):
            poles.SetValue(i + 1, j + 1, gp_Pnt(*P[i, j]))

    def knots(t):
        k, m = np.unique(np.round(np.asarray(t, float), 14), return_counts=True)
        K, M = TColStd_Array1OfReal(1, len(k)), TColStd_Array1OfInteger(1, len(k))
        for i, (a, b) in enumerate(zip(k, m), 1):
            K.SetValue(i, float(a))
            M.SetValue(i, int(b))
        return K, M
    ku, mu = knots(p["tu"])
    kv, mv = knots(p["tv"])
    if p.get("weights") is not None:
        W = np.asarray(p["weights"], float)
        w = TColStd_Array2OfReal(1, nu, 1, nv)
        for i in range(nu):
            for j in range(nv):
                w.SetValue(i + 1, j + 1, float(W[i, j]))
        return Geom_BSplineSurface(poles, w, ku, kv, mu, mv, int(p["pu"]), int(p["pv"]), periodic_u, False)
    return Geom_BSplineSurface(poles, ku, kv, mu, mv, int(p["pu"]), int(p["pv"]), periodic_u, False)
