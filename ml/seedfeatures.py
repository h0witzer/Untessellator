#!/usr/bin/env python3
"""
seedfeatures.py - the extended (v6) per-facet inputs of the seed-fill model.

A deliberate duplicate of js/seedFeatures.js, compared elementwise by test/netparity.mjs. Loops
that accumulate sums run in the same order as the JS so the two agree to rounding. See the JS
file for what each column is and the probe screen that chose them.
"""

from __future__ import annotations

import math

import numpy as np

EXT_FEATURE_NAMES = (
    "bottleneck", "local_crease", "nev0", "nev1", "nev2",
    "plane_d", "plane_a", "cyl_d", "cyl_a", "cone_d", "cone_a", "sphere_d", "sphere_a",
)
N_EXT = len(EXT_FEATURE_NAMES)
FIT_RING = 32
FIT_CREASE = 30.0
NORMAL_RADIUS = 0.35
SENTINEL_D = 10.0


def adjacency_csr_angles(pairs: np.ndarray, angles: np.ndarray, ntri: int):
    """meshcore.adjacency_csr plus the dihedral angle of every entry, in the same order."""
    if pairs.shape[0] == 0:
        return np.zeros(ntri + 1, np.int64), np.zeros(0, np.int32), np.zeros(0, np.float64)
    src = np.concatenate([pairs[:, 0], pairs[:, 1]])
    dst = np.concatenate([pairs[:, 1], pairs[:, 0]])
    ang = np.concatenate([angles, angles])
    order = np.argsort(src, kind="stable")
    indptr = np.zeros(ntri + 1, np.int64)
    np.cumsum(np.bincount(src, minlength=ntri), out=indptr[1:])
    return indptr, dst[order].astype(np.int32), ang[order].astype(np.float64)


def eigen3(m):
    """Port of meshCore.eigen3: Jacobi rotations; eigenvalues ascending with their vectors."""
    a = [[m[0], m[1], m[2]], [m[1], m[3], m[4]], [m[2], m[4], m[5]]]
    V = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    for _ in range(100):
        off, p, q = 0.0, 0, 1
        for i in range(3):
            for j in range(i + 1, 3):
                if abs(a[i][j]) > off:
                    off, p, q = abs(a[i][j]), i, j
        if off < 1e-15:
            break
        theta = (a[q][q] - a[p][p]) / (2 * a[p][q])
        t = (1.0 if theta >= 0 else -1.0) / (abs(theta) + math.sqrt(1 + theta * theta))
        c = 1 / math.sqrt(1 + t * t)
        s = t * c
        apq = a[p][q]
        a[p][p] -= t * apq
        a[q][q] += t * apq
        a[p][q] = a[q][p] = 0.0
        for r in range(3):
            if r == p or r == q:
                continue
            apr, aqr = a[p][r], a[q][r]
            a[p][r] = a[r][p] = c * apr - s * aqr
            a[q][r] = a[r][q] = s * apr + c * aqr
        for r in range(3):
            vp, vq = V[r][p], V[r][q]
            V[r][p] = c * vp - s * vq
            V[r][q] = s * vp + c * vq
    order = sorted(range(3), key=lambda i: a[i][i])
    return [a[i][i] for i in order], [[V[0][i], V[1][i], V[2][i]] for i in order]


def solve_n(A, b):
    """Port of seedFeatures.solveN."""
    n = len(b)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    big = 0.0
    for row in A:
        for v in row:
            big = max(big, abs(v))
    if not big > 0:
        return None
    for col in range(n):
        piv = col
        for r in range(col + 1, n):
            if abs(M[r][col]) > abs(M[piv][col]):
                piv = r
        if piv != col:
            M[col], M[piv] = M[piv], M[col]
        if abs(M[col][col]) < 1e-12 * big:
            return None
        for r in range(col + 1, n):
            f = M[r][col] / M[col][col]
            for c in range(col, n + 1):
                M[r][c] -= f * M[col][c]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        v = M[i][n]
        for j in range(i + 1, n):
            v -= M[i][j] * x[j]
        x[i] = v / M[i][i]
    return x


def eig3sym_values(a00, a01, a02, a11, a12, a22):
    """Vectorised seedFeatures.eig3symValues: ascending eigenvalues of symmetric 3x3 matrices."""
    p1 = a01 * a01 + a02 * a02 + a12 * a12
    q = (a00 + a11 + a22) / 3
    p2 = (a00 - q) ** 2 + (a11 - q) ** 2 + (a22 - q) ** 2 + 2 * p1
    small = p2 < 1e-30
    p = np.sqrt(np.where(small, 6.0, p2) / 6)
    b00, b11, b22 = (a00 - q) / p, (a11 - q) / p, (a22 - q) / p
    b01, b02, b12 = a01 / p, a02 / p, a12 / p
    r = (b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b12 * b02) + b02 * (b01 * b12 - b11 * b02)) / 2
    phi = np.arccos(np.clip(r, -1.0, 1.0)) / 3
    e1 = q + 2 * p * np.cos(phi)
    e3 = q + 2 * p * np.cos(phi + (2 * np.pi) / 3)
    e2 = 3 * q - e1 - e3
    return (np.where(small, q, e3), np.where(small, q, e2), np.where(small, q, e1))


def _perpendicular_basis(n):
    a = [abs(n[0]), abs(n[1]), abs(n[2])]
    k = 0
    if a[1] < a[k]:
        k = 1
    if a[2] < a[k]:
        k = 2
    w = [0.0, 0.0, 0.0]
    w[k] = 1.0
    t1 = [n[1] * w[2] - n[2] * w[1], n[2] * w[0] - n[0] * w[2], n[0] * w[1] - n[1] * w[0]]
    l1 = math.hypot(t1[0], t1[1], t1[2]) or 1e-30
    t1 = [t1[0] / l1, t1[1] / l1, t1[2] / l1]
    t2 = [n[1] * t1[2] - n[2] * t1[1], n[2] * t1[0] - n[0] * t1[2], n[0] * t1[1] - n[1] * t1[0]]
    return t1, t2


def crease_ring(seed, indptr, indices, eangles, want, deg, excluded=None):
    """Port of seedFeatures.creaseRing."""
    seed = int(seed)
    seen = {seed}
    ids = [seed]
    frontier = [seed]
    while len(ids) < want and frontier:
        nxt = []
        for f in frontier:
            lo, hi = int(indptr[f]), int(indptr[f + 1])
            for nb, ang in zip(indices[lo:hi].tolist(), eangles[lo:hi].tolist()):
                if nb in seen or ang >= deg or (excluded is not None and excluded[nb]):
                    continue
                seen.add(nb)
                nxt.append(nb)
        nxt.sort()
        for f in nxt:
            if len(ids) >= want:
                break
            ids.append(f)
        frontier = nxt
    return ids


def patch_bottleneck(ids, indptr, indices, eangles):
    """Port of seedFeatures.patchBottleneck."""
    n = len(ids)
    index = {int(t): i for i, t in enumerate(ids)}
    edges = []
    for i, t in enumerate(ids):
        lo, hi = int(indptr[t]), int(indptr[t + 1])
        for nb, ang in zip(indices[lo:hi].tolist(), eangles[lo:hi].tolist()):
            j = index.get(nb)
            if j is not None and j > i:
                edges.append((ang, i, j))
    edges.sort()
    parent = list(range(n))
    members = [[i] for i in range(n)]
    out = np.full(n, np.inf)
    out[0] = 0.0

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for w, i, j in edges:
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        rs = find(0)
        if ri == rs:
            out[members[rj]] = w
        elif rj == rs:
            out[members[ri]] = w
        if len(members[ri]) < len(members[rj]):
            ri, rj = rj, ri
        parent[rj] = ri
        members[ri].extend(members[rj])
        members[rj] = []
    return out


def fit_primitives(ring, centroids, normals):
    """Port of seedFeatures.fitPrimitives."""
    out = {"plane": None, "sphere": None, "cylinder": None, "cone": None}
    N = len(ring)
    if N < 3:
        return out
    C = [[float(centroids[t, 0]), float(centroids[t, 1]), float(centroids[t, 2])] for t in ring]
    Nn = [[float(normals[t, 0]), float(normals[t, 1]), float(normals[t, 2])] for t in ring]
    m = [0.0, 0.0, 0.0]
    sn = [0.0, 0.0, 0.0]
    for c, n in zip(C, Nn):
        for a in range(3):
            m[a] += c[a]
            sn[a] += n[a]
    for a in range(3):
        m[a] /= N
    ss = 0.0
    for c in C:
        ss += (c[0] - m[0]) ** 2 + (c[1] - m[1]) ** 2 + (c[2] - m[2]) ** 2
    s = math.sqrt(ss / N)
    if not s > 0:
        return out
    P = [[(c[0] - m[0]) / s, (c[1] - m[1]) / s, (c[2] - m[2]) / s] for c in C]

    sl = math.hypot(sn[0], sn[1], sn[2])
    if sl > 1e-12:
        out["plane"] = (m, [sn[0] / sl, sn[1] / sl, sn[2] / sl])

    A = [[0.0] * 4 for _ in range(4)]
    b = [0.0] * 4
    for p in P:
        row = [2 * p[0], 2 * p[1], 2 * p[2], 1.0]
        rhs = p[0] * p[0] + p[1] * p[1] + p[2] * p[2]
        for i in range(4):
            b[i] += row[i] * rhs
            for j in range(4):
                A[i][j] += row[i] * row[j]
    x = solve_n(A, b)
    if x:
        r2 = x[3] + x[0] * x[0] + x[1] * x[1] + x[2] * x[2]
        if r2 > 0:
            out["sphere"] = ([m[0] + s * x[0], m[1] + s * x[1], m[2] + s * x[2]], s * math.sqrt(r2))

    n00 = n01 = n02 = n11 = n12 = n22 = 0.0
    for n in Nn:
        n00 += n[0] * n[0]
        n01 += n[0] * n[1]
        n02 += n[0] * n[2]
        n11 += n[1] * n[1]
        n12 += n[1] * n[2]
        n22 += n[2] * n[2]

    axis = eigen3([n00, n01, n02, n11, n12, n22])[1][0]
    u, v = _perpendicular_basis(axis)
    A = [[0.0] * 3 for _ in range(3)]
    b = [0.0] * 3
    for p in P:
        xx = p[0] * u[0] + p[1] * u[1] + p[2] * u[2]
        yy = p[0] * v[0] + p[1] * v[1] + p[2] * v[2]
        row = [2 * xx, 2 * yy, 1.0]
        rhs = xx * xx + yy * yy
        for i in range(3):
            b[i] += row[i] * rhs
            for j in range(3):
                A[i][j] += row[i] * row[j]
    x = solve_n(A, b)
    if x:
        r2 = x[2] + x[0] * x[0] + x[1] * x[1]
        if r2 > 0:
            out["cylinder"] = (axis, [m[a] + s * (x[0] * u[a] + x[1] * v[a]) for a in range(3)], s * math.sqrt(r2))

    mn = [sn[0] / N, sn[1] / N, sn[2] / N]
    cov = [n00 / N - mn[0] * mn[0], n01 / N - mn[0] * mn[1], n02 / N - mn[0] * mn[2],
           n11 / N - mn[1] * mn[1], n12 / N - mn[1] * mn[2], n22 / N - mn[2] * mn[2]]
    axis = eigen3(cov)[1][0]
    sdot = 0.0
    for n in Nn:
        sdot += n[0] * axis[0] + n[1] * axis[1] + n[2] * axis[2]
    sdot /= N
    if sdot < 0:
        axis = [-axis[0], -axis[1], -axis[2]]
        sdot = -sdot
    sin = min(1.0, sdot)
    b = [0.0, 0.0, 0.0]
    for i in range(N):
        n = Nn[i]
        np_ = n[0] * P[i][0] + n[1] * P[i][1] + n[2] * P[i][2]
        for a in range(3):
            b[a] += n[a] * np_
    x = solve_n([[n00, n01, n02], [n01, n11, n12], [n02, n12, n22]], b)
    if x:
        out["cone"] = (axis, [m[a] + s * x[a] for a in range(3)], sin, math.sqrt(max(0.0, 1 - sin * sin)))
    return out


def _columns(dist, cos, scale):
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.where(np.isfinite(dist), np.minimum(SENTINEL_D, dist / scale), SENTINEL_D)
        a = np.where(np.isfinite(cos), 1 - np.minimum(1.0, cos), 1.0)
    return d, a


def evaluate_fits(fits, q, nt, scale):
    """(n, 8) columns: distance / scale and 1-|cos| for plane, cylinder, cone, sphere."""
    n = q.shape[0]
    out = np.empty((n, 8))
    nan = np.full(n, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        if fits["plane"] is not None:
            o, nn = (np.array(v) for v in fits["plane"])
            d = np.abs((q - o) @ nn)
            c = np.abs(nt @ nn)
        else:
            d = c = nan
        out[:, 0], out[:, 1] = _columns(d, c, scale)

        if fits["cylinder"] is not None:
            ax, ce, r = np.array(fits["cylinder"][0]), np.array(fits["cylinder"][1]), fits["cylinder"][2]
            v = q - ce
            al = v @ ax
            w = v - al[:, None] * ax
            y = np.sqrt((w * w).sum(1))
            ok = y > 1e-30
            d = np.where(ok, np.abs(y - r), np.nan)
            c = np.where(ok, np.abs((nt * w).sum(1)) / np.where(ok, y, 1.0), np.nan)
        else:
            d = c = nan
        out[:, 2], out[:, 3] = _columns(d, c, scale)

        if fits["cone"] is not None:
            ax, apex, sn, cs = np.array(fits["cone"][0]), np.array(fits["cone"][1]), fits["cone"][2], fits["cone"][3]
            v = q - apex
            al = v @ ax
            w = v - al[:, None] * ax
            y = np.sqrt((w * w).sum(1))
            ok = y > 1e-30
            sg = np.where(al < 0, -1.0, 1.0)
            nrm = cs * w / np.where(ok, y, 1.0)[:, None] - sn * sg[:, None] * ax
            d = np.where(ok, np.abs(y * cs - np.abs(al) * sn), np.nan)
            c = np.where(ok, np.abs((nt * nrm).sum(1)), np.nan)
        else:
            d = c = nan
        out[:, 4], out[:, 5] = _columns(d, c, scale)

        if fits["sphere"] is not None:
            ce, r = np.array(fits["sphere"][0]), fits["sphere"][1]
            w = q - ce
            y = np.sqrt((w * w).sum(1))
            ok = y > 1e-30
            d = np.where(ok, np.abs(y - r), np.nan)
            c = np.where(ok, np.abs((nt * w).sum(1)) / np.where(ok, y, 1.0), np.nan)
        else:
            d = c = nan
        out[:, 6], out[:, 7] = _columns(d, c, scale)
    return out


def ext_features(ids, seed, facets, indptr, indices, eangles, d_scaled, n_frame, scale, excluded=None):
    """(n, 13) float32 extended columns for one patch; see js/seedFeatures.writeExtendedFeatures."""
    ids = np.asarray(ids)
    n = ids.size
    out = np.zeros((n, N_EXT), np.float64)
    out[:, 0] = np.minimum(180.0, patch_bottleneck(ids.tolist(), indptr, indices, eangles)) / 90

    starts, ends = indptr[ids], indptr[ids + 1]
    counts = (ends - starts).astype(np.int64)
    local = np.zeros(n)
    if counts.sum():
        rows = np.repeat(np.arange(n), counts)
        idx = np.concatenate([np.arange(a, b) for a, b in zip(starts, ends)])
        np.maximum.at(local, rows, eangles[idx])
    out[:, 1] = local / 90

    D = np.asarray(d_scaled, np.float64)
    Nf = np.asarray(n_frame, np.float64)
    d2 = ((D[None, :, :] - D[:, None, :]) ** 2).sum(-1)
    mask = (d2 <= NORMAL_RADIUS * NORMAL_RADIUS).astype(np.float64)
    w = mask.sum(1)
    a, b, g = Nf[:, 0], Nf[:, 1], Nf[:, 2]
    m0, m1, m2 = mask @ a / w, mask @ b / w, mask @ g / w
    ev = eig3sym_values(mask @ (a * a) / w - m0 * m0, mask @ (a * b) / w - m0 * m1, mask @ (a * g) / w - m0 * m2,
                        mask @ (b * b) / w - m1 * m1, mask @ (b * g) / w - m1 * m2, mask @ (g * g) / w - m2 * m2)
    out[:, 2] = np.maximum(0.0, ev[0])
    out[:, 3] = np.maximum(0.0, ev[1])
    out[:, 4] = np.maximum(0.0, ev[2])

    ring = crease_ring(seed, indptr, indices, eangles, FIT_RING, FIT_CREASE, excluded)
    fits = fit_primitives(ring, facets["centroids"], facets["normals"])
    out[:, 5:13] = evaluate_fits(fits, np.asarray(facets["centroids"][ids], np.float64),
                                 np.asarray(facets["normals"][ids], np.float64), scale)
    return out.astype(np.float32)
