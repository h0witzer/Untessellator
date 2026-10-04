#!/usr/bin/env python3
"""
freeform_fit.py - one untrimmed bicubic B-spline surface per mesh group, for groups no analytic
surface fits (the "bspline" kind of solid_rebuild.py).

This is the classic reverse-engineering recipe the literature converged on for a single face (each
CAD face IS one trimmed tensor-product surface), not a patch network:

  parameterise  the group's vertices get (u, v): an orthogonal projection onto the plane of the
                mean normal when no triangle folds over in it (a fillet's normals span ~90 deg),
                otherwise a least-squares conformal map (LSCM, free boundary, solved by CG). UV is
                in length units and rotated so the box around it is tight.
  domain        that box plus a margin, so the knitter's boundary curves - which lie a hair outside
                the group's own vertices - are on the surface and the face is trimmed from it.
  fit           least squares on the vertices (a CAD tessellation's vertices lie ON the surface;
                triangle interiors are chords), plus a thin-plate term that decides the shape where
                there is no data (the margin, sparse fillets) - it continues the surface linearly.
  correct       each vertex re-projected onto the fitted surface and the fit repeated ("moving
                parameterisation", Giannelli et al. 2025): the first parameterisation is only a
                start.
  refine        knots inserted mid-span in every u / v span whose vertices miss by more than the
                tolerance, until they all fit or the net is at its cap (Weiss et al. 2002).

numpy only. fit_group(V, T) -> (params | None, info); params are nurbs.Surface.to_params().
fit_pieces(V, T) splits a group no parameterisation covers (rings, tubes, strongly curved disks)
into pieces that each get one.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

import nurbs

DEGREE = 3
MARGIN = 0.10                  # domain margin, x the UV extent, each side
PLANE_MIN_COS = 0.15           # every triangle's normal within ~81 deg of the projection plane's
TOL = 1e-5                     # x part diagonal: vertex error the refinement aims for
LAMBDA = 1e-6                  # thin-plate weight, relative to the data term (trace ratio)
CORRECT_ITERS = 3              # parameter corrections per refinement round
REFINE_ROUNDS = 8
MAX_SPANS = 80                 # per direction
MAX_POINTS = 6000              # data points used to fit (boundary vertices always kept)
TANGENT_WEIGHT = 1.0           # a tangency row weighs like a distance of (angle x median edge length) x this
TANGENT_MAX_CP = 900           # control points; above this the (coupled, 3n x 3n) tangency solve is skipped
TANGENT_AGREE_DEG = 15.0       # a neighbour normal is imposed only where the free fit is this close to it
TANGENT_ERR_RATIO = 2.0        # the tangent refit may raise the max vertex error at most this much
PERIODIC_MIN_SPANS = 6         # spans round a periodic (ring / tube) direction to start with
DOF_RATIO = 3.0                # refinement never passes this many control points per data point ...
MAX_CP = 3000                  # ... or this many in all (the dense solve's cost)
HOLDOUT_FRAC = 0.1             # interior vertices held out to choose the refinement level
HOLDOUT_MIN = 8                # fewer held-out points than this: no hold-out, refine on the vertex error
VAL_PATIENCE = 1               # rounds without a better held-out error before refinement stops
VSA_ITERS = 8                  # Lloyd iterations of the normal-cone split
SPLIT_LOPSIDED = 0.9           # a normal split leaving one piece this big falls back to a plane cut
REFINE_MIN_POINTS = 4          # a span is split only if it holds this many data points


# ── parameterisation ─────────────────────────────────────────────────────────

def _frame(n):
    n = n / np.linalg.norm(n)
    ref = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(n, ref)
    x /= np.linalg.norm(x)
    return x, np.cross(n, x)


def boundary_loops(T):
    """Boundary edges (used by one triangle, in winding order) chained into loops."""
    cnt = defaultdict(int)
    for a, b, c in T:
        for u, v in ((a, b), (b, c), (c, a)):
            cnt[(min(u, v), max(u, v))] += 1
    nxt = {}
    for a, b, c in T:
        for u, v in ((a, b), (b, c), (c, a)):
            if cnt[(min(u, v), max(u, v))] == 1:
                nxt[u] = v
    loops, seen = [], set()
    for s in list(nxt):
        if s in seen:
            continue
        loop, u = [], s
        while u not in seen and u in nxt:
            seen.add(u)
            loop.append(u)
            u = nxt[u]
        loops.append(loop)
    return loops


def _components(T, n):
    par = np.arange(n)

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for a, b, c in T:
        ra, rb, rc = find(a), find(b), find(c)
        par[rb] = ra
        par[find(rc)] = ra
    return len({find(x) for x in np.unique(T)})


def plane_param(V, T):
    """(uv, normal) by projection onto the plane of the area-weighted mean normal, or None when a
    triangle would fold over (its normal PLANE_MIN_COS or less along the plane's)."""
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    N = np.cross(b - a, c - a)
    n = N.sum(0)
    if np.linalg.norm(n) < 1e-300:
        return None
    n /= np.linalg.norm(n)
    area = np.linalg.norm(N, axis=1)
    cos = (N @ n) / np.maximum(area, 1e-300)
    if (cos[area > 1e-12 * area.max()] <= PLANE_MIN_COS).any():
        return None
    x, y = _frame(n)
    return np.stack([V @ x, V @ y], 1), n


def annulus_param(V, T):
    """(uv, period) for a ring or tube (connected, Euler characteristic 0, two boundary loops): a
    periodic conformal map in which u goes once round, or None.

    This is how the CAD truth has these faces - one B-spline closed in one direction (ABC stores 4% of
    its B-splines u-closed, 2% v-closed) - so they are not split: the ring is cut along ONE seam, the
    shortest path between its two boundary loops, and LSCM solved with the triangles on the seam's
    right carrying their seam vertices shifted by the period (the first loop's length). The fitted
    surface is then periodic in u, and the knitter's repair adds its seam edge as it does a cylinder's."""
    import heapq
    nV = len(V)
    cnt = defaultdict(int)
    for t in T:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            cnt[(min(a, b), max(a, b))] += 1
    if nV - len(cnt) + len(T) != 0 or _components(T, nV) != 1:
        return None
    par = {}

    def find(x):
        while par.setdefault(x, x) != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    bed = [e for e, c in cnt.items() if c == 1]
    for a, b in bed:
        par[find(a)] = find(b)
    comps = defaultdict(set)
    for a, b in bed:
        comps[find(a)].update((a, b))
    if len(comps) != 2:
        return None
    A, B = comps.values()
    period = sum(float(np.linalg.norm(V[a] - V[b])) for a, b in bed if a in A)
    adj = defaultdict(list)
    for a, b in cnt:
        l = float(np.linalg.norm(V[a] - V[b]))
        adj[a].append((b, l))
        adj[b].append((a, l))
    dist = {a: 0.0 for a in A}
    prev = {}
    heap = [(0.0, a) for a in A]
    end = None
    while heap:
        d, x = heapq.heappop(heap)
        if d > dist.get(x, math.inf):
            continue
        if x in B:
            end = x
            break
        for y, l in adj[x]:
            if y in A:
                continue
            if d + l < dist.get(y, math.inf):
                dist[y] = d + l
                prev[y] = x
                heapq.heappush(heap, (d + l, y))
    if end is None:
        return None
    path = [end]
    while path[-1] in prev:
        path.append(prev[path[-1]])
    path.reverse()                                      # path[0] on loop A, path[-1] on loop B
    seam = {(min(a, b), max(a, b)) for a, b in zip(path[:-1], path[1:])}
    tri_of = {}
    inc = defaultdict(list)
    for i, t in enumerate(T):
        for k in range(3):
            tri_of[(int(t[k]), int(t[(k + 1) % 3]))] = i
            inc[int(t[k])].append(i)
    shift = set()
    for i, x in enumerate(path):
        # the triangle right of the seam at x: it holds the seam edge reversed
        start = tri_of.get((path[i + 1], x)) if i + 1 < len(path) else tri_of.get((x, path[i - 1]))
        if start is None:
            return None
        fan, seen, stack = set(inc[x]), {start}, [start]
        while stack:
            t = stack.pop()
            for w in T[t]:
                w = int(w)
                if w == x or (min(x, w), max(x, w)) in seam:
                    continue
                for o in (tri_of.get((x, w)), tri_of.get((w, x))):
                    if o is not None and o != t and o in fan and o not in seen:
                        seen.add(o)
                        stack.append(o)
        for t in seen:
            shift.add((t, int(np.nonzero(T[t] == x)[0][0])))
    uv = _lscm_solve(V, T, {int(path[0]): (0.0, 0.0)}, shift, period)
    if uv is None:
        return None
    return uv, period


def lscm(V, T):
    """Least-squares conformal map (Levy et al. 2002) of a disk with two far-apart boundary vertices
    pinned; None when the group is not a disk or the map folds."""
    nV = len(V)
    # A disk: connected, Euler characteristic 1. (Not "one boundary loop": 96 of the failed fits
    # on the freeform set were disks whose boundary passes through a vertex twice, which a
    # successor-map loop walk splits into many pieces.)
    cnt = defaultdict(int)
    for t in T:
        for u, v in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            cnt[(min(u, v), max(u, v))] += 1
    if nV - len(cnt) + len(T) != 1 or _components(T, nV) != 1:
        return None
    b = np.unique([q for e, c in cnt.items() if c == 1 for q in e])
    if len(b) < 2:
        return None
    Pb = V[b]
    i0 = int(np.argmax(np.linalg.norm(Pb - Pb.mean(0), axis=1)))
    i1 = int(np.argmax(np.linalg.norm(Pb - Pb[i0], axis=1)))
    pin = {int(b[i0]): (0.0, 0.0), int(b[i1]): (float(np.linalg.norm(Pb[i1] - Pb[i0])), 0.0)}
    return _lscm_solve(V, T, pin)


def _lscm_solve(V, T, pin, shift=(), period=0.0, iters=20000):
    """The LSCM energy minimised with the pinned vertices fixed, by Jacobi-preconditioned CG on the
    normal equations (matrix-free). shift: (triangle, corner) pairs whose vertex sits at u + period
    in that triangle (the right side of a ring's seam). Returns vertex uv, mirrored in v if need be
    so the map keeps orientation, or None if any triangle folds."""
    nV = len(V)
    # per triangle: local 2D coordinates, the opposite-edge vectors, and the two Cauchy-Riemann rows
    p0, p1, p2 = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    e1 = p1 - p0
    l1 = np.linalg.norm(e1, axis=1)
    ex = e1 / np.maximum(l1, 1e-300)[:, None]
    nn = np.cross(e1, p2 - p0)
    A2 = np.linalg.norm(nn, axis=1)                     # twice the area
    ey = np.cross(nn / np.maximum(A2, 1e-300)[:, None], ex)
    xs = np.stack([np.zeros(len(T)), l1, ((p2 - p0) * ex).sum(1)], 1)
    ys = np.stack([np.zeros(len(T)), np.zeros(len(T)), ((p2 - p0) * ey).sum(1)], 1)
    good = A2 > 1e-14 * (A2.max() if len(A2) else 1)
    s = 1.0 / np.sqrt(np.maximum(A2, 1e-300) * 2.0)     # 1/(2 sqrt(A)) with A = A2/2
    sh = np.zeros((len(T), 3))
    for t, a_ in shift:
        sh[t, a_] = period
    rows, cols, vals = [], [], []
    const = np.zeros(2 * len(T))
    for a_ in range(3):
        k, l = (a_ + 1) % 3, (a_ + 2) % 3
        eax, eay = xs[:, l] - xs[:, k], ys[:, l] - ys[:, k]      # edge opposite vertex a_
        # r1 = u_x - v_y : u coeff -eay, v coeff -eax ; r2 = u_y + v_x : u coeff eax, v coeff -eay
        t = np.nonzero(good)[0]
        vi = T[t, a_]
        for r, cu, cv in ((0, -eay[t], -eax[t]), (1, eax[t], -eay[t])):
            rows += [2 * t + r, 2 * t + r]
            cols += [2 * vi, 2 * vi + 1]
            vals += [cu * s[t], cv * s[t]]
            const[2 * t + r] += cu * s[t] * sh[t, a_]          # a shifted corner's u is u + period
    rows, cols, vals = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
    nr = 2 * len(T)
    fixed = np.zeros(2 * nV)
    isfree = np.ones(2 * nV, bool)
    for vtx, (pu, pv) in pin.items():
        fixed[2 * vtx], fixed[2 * vtx + 1] = pu, pv
        isfree[2 * vtx] = isfree[2 * vtx + 1] = False
    # b = A_fixed x_fixed + the shifts' constants ; unknowns: the free columns
    fx = ~isfree[cols]
    rhs = -np.bincount(rows[fx], vals[fx] * fixed[cols[fx]], minlength=nr) - const
    keep = ~fx
    r_, c_, v_ = rows[keep], cols[keep], vals[keep]
    fidx = -np.ones(2 * nV, np.int64)
    fidx[isfree] = np.arange(isfree.sum())
    c_ = fidx[c_]
    nf = int(isfree.sum())

    def Av(x):
        return np.bincount(r_, v_ * x[c_], minlength=nr)

    def Atv(y):
        return np.bincount(c_, v_ * y[r_], minlength=nf)
    diag = np.bincount(c_, v_ * v_, minlength=nf)
    Minv = 1.0 / np.maximum(diag, 1e-300)
    # initial guess: the plane projection (a good start for mildly curved disks; zero for a ring)
    init = np.zeros(2 * nV)
    if not len(shift):
        x0, y0 = _frame(np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]).sum(0) + 1e-300)
        init[0::2], init[1::2] = V @ x0, V @ y0
    x = init[isfree]
    r = Atv(rhs - Av(x))
    z = Minv * r
    p = z.copy()
    rz = r @ z
    r0 = math.sqrt(max(r @ r, 1e-300))
    for _ in range(iters):
        Ap = Atv(Av(p))
        alpha = rz / max(p @ Ap, 1e-300)
        x += alpha * p
        r -= alpha * Ap
        if math.sqrt(r @ r) < 1e-10 * r0:
            break
        z = Minv * r
        rz2 = r @ z
        p = z + (rz2 / max(rz, 1e-300)) * p
        rz = rz2
    full = fixed.copy()
    full[isfree] = x
    uv = full.reshape(-1, 2)
    # orientation-preserving without folds: every triangle's UV area positive
    q0, q1, q2 = (uv[T[:, a]] + np.stack([sh[:, a], np.zeros(len(T))], 1) for a in range(3))
    sa = (q1[:, 0] - q0[:, 0]) * (q2[:, 1] - q0[:, 1]) - (q1[:, 1] - q0[:, 1]) * (q2[:, 0] - q0[:, 0])
    if (sa[good] <= 0).mean() > 0.5:
        uv[:, 1] = -uv[:, 1]                             # mirrored: flip v
        sa = -sa
    if (sa[good] <= 0).any():
        return None
    return uv


# ── splitting what cannot be parameterised ───────────────────────────────────
#
# One surface per group needs a group that maps to the plane: a disk that projects without folding or
# whose conformal map does not fold, or a ring / tube (annulus_param). What is left - disks whose
# conformal map folds (their boundary pinches, or they wrap too far), higher genus - fit_pieces cuts
# into pieces and fits each: a small patch network, only where one surface cannot cover the face. The
# knitter joins the pieces like any other groups (their shared run becomes an edge between two
# B-spline faces).
#
# The cut is by NORMAL-CONE clustering, not position: variational shape approximation (Cohen-Steiner,
# Alliez, Desbrun 2004) with its L2,1 normal metric, as D-charts (Julius et al. 2005) do for
# parameterisation. Two proxies grow regions whose normals cluster round them, so the cut falls where
# the normal field turns and each piece is near a height field over its proxy plane - which is what
# plane_param needs. A plane through the centroid (the first version) cut wherever the centroid
# happened to be.

def _tri_components(T):
    """Connected components of triangles sharing an edge: list of triangle index arrays."""
    owner = defaultdict(list)
    for i, t in enumerate(T):
        for u, v in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            owner[(min(u, v), max(u, v))].append(i)
    par = np.arange(len(T))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for lst in owner.values():
        for j in lst[1:]:
            par[find(j)] = find(lst[0])
    roots = np.array([find(i) for i in range(len(T))])
    return [np.nonzero(roots == r)[0] for r in np.unique(roots)]


def _euler(T):
    E = {(min(u, v), max(u, v)) for t in T for u, v in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))}
    return len(np.unique(T)) - len(E) + len(T)


def _split(V, T, iters=VSA_ITERS):
    """Edge-connected pieces (triangle index arrays) of T from 2-proxy normal-cone clustering (VSA,
    L2,1 metric): seeds at the two most different normals, regions flooded from the seeds by the
    area-weighted distance of each triangle's normal to its region's proxy, proxies re-estimated,
    repeated."""
    import heapq
    Nr = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    A = 0.5 * np.linalg.norm(Nr, axis=1) + 1e-300
    N = Nr / (2 * A[:, None])
    owner = defaultdict(list)
    for i, t in enumerate(T):
        for u, v in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            owner[(min(u, v), max(u, v))].append(i)
    nbrs = [[] for _ in range(len(T))]
    for lst in owner.values():
        for a in lst:
            nbrs[a] += [b for b in lst if b != a]
    m = (N * A[:, None]).sum(0)
    m /= max(np.linalg.norm(m), 1e-300)
    s1 = int(np.argmin(N @ m))
    s2 = int(np.argmin(N @ N[s1]))
    if s1 == s2:
        return [np.arange(len(T))]
    # proxies by area-weighted 2-means on the normal sphere, started at the two most different
    # normals; the flood seeds are then each proxy's best-fitting triangle. Seeding the flood AT the
    # extreme normals boxed one region into a small feature (a 3866 + 4 triangle split).
    proxies = np.array([N[s1], N[s2]])
    for _ in range(10):
        lab = np.argmax(N @ proxies.T, axis=1)
        for k in range(2):
            p = (N[lab == k] * A[lab == k, None]).sum(0)
            if np.linalg.norm(p) > 0:
                proxies[k] = p / np.linalg.norm(p)
    seeds = [int(np.argmax(N @ proxies[k])) for k in range(2)]
    if seeds[0] == seeds[1]:
        return [np.arange(len(T))]
    label = None
    for _ in range(iters):
        label = -np.ones(len(T), np.int64)
        heap = []
        for k, s in enumerate(seeds):
            label[s] = k
        for k, s in enumerate(seeds):
            for nb in nbrs[s]:
                heapq.heappush(heap, (A[nb] * float(((N[nb] - proxies[k]) ** 2).sum()), nb, k))
        while heap:
            _, t, k = heapq.heappop(heap)
            if label[t] >= 0:
                continue
            label[t] = k
            for nb in nbrs[t]:
                if label[nb] < 0:
                    heapq.heappush(heap, (A[nb] * float(((N[nb] - proxies[k]) ** 2).sum()), nb, k))
        new = []
        for k in range(2):
            sel = np.nonzero(label == k)[0]
            p = (N[sel] * A[sel, None]).sum(0)
            proxies[k] = p / max(np.linalg.norm(p), 1e-300)
            new.append(int(sel[np.argmin(((N[sel] - proxies[k]) ** 2).sum(1))]))
        if new == seeds:
            break
        seeds = new
    pieces = []
    for k in (0, 1, -1):                               # -1: triangles no seed reached (disconnected)
        sub = np.nonzero(label == k)[0]
        if len(sub):
            pieces += [sub[p] for p in _tri_components(T[sub])]
    if max(len(p) for p in pieces) >= SPLIT_LOPSIDED * len(T):
        return _split_plane(V, T)
    return pieces


def _split_plane(V, T):
    """The fallback cut when the normals do not separate: a plane through the area-weighted centroid,
    across the first principal direction of the triangle centroids whose pieces are all disks."""
    C = V[T].mean(1)
    A = 0.5 * np.linalg.norm(np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]), axis=1) + 1e-300
    c = (C * A[:, None]).sum(0) / A.sum()
    w, R = np.linalg.eigh(np.cov((C - c).T, aweights=A) + 1e-300 * np.eye(3))
    best = None
    for d in R[:, ::-1].T:
        s = (C - c) @ d
        side = s >= np.median(s)
        if side.all() or not side.any():
            continue
        pieces = []
        for mask in (side, ~side):
            sub = np.nonzero(mask)[0]
            pieces += [sub[p] for p in _tri_components(T[sub])]
        if all(_euler(T[p]) == 1 for p in pieces):
            return pieces
        if best is None:
            best = pieces
    return best or [np.arange(len(T))]


def fit_pieces(V, T, scale=None, normals=None, max_depth=4, **kw):
    """fit_group, splitting the group (fit_pieces' comment above) where it cannot be parameterised.
    Returns [(triangle indices into T, params, info)] covering T, less any piece still failing at
    max_depth, and the list of failures [(triangle indices, info)]."""
    out, failed = [], []
    todo = [(np.arange(len(T)), 0)]
    nmap = dict(zip(np.asarray(normals[0]).tolist(), np.asarray(normals[1]))) if normals is not None else {}
    while todo:
        tri, depth = todo.pop()
        sub = T[tri]
        vid, inv = np.unique(sub, return_inverse=True)
        Tl = inv.reshape(-1, 3)
        nl = None
        if nmap:
            keep = [i for i, q in enumerate(vid) if int(q) in nmap]
            if keep:
                nl = (np.array(keep, np.int64), np.array([nmap[int(vid[i])] for i in keep]))
        params, info = fit_group(V[vid], Tl, scale=scale, normals=nl, **kw)
        if params is not None:
            info["depth"] = depth
            out.append((tri, params, info))
        elif info.get("fail") == "param" and depth < max_depth and len(tri) >= 2:
            for p in _split(V, sub):
                if len(p) < len(tri):
                    todo.append((tri[p], depth + 1))
                else:
                    failed.append((tri, info))
        else:
            failed.append((tri, info))
    return out, failed


# ── fitting ──────────────────────────────────────────────────────────────────

def _nz(t, p, x, nder):
    """Nonzero basis values: (first index (m,), [(m, p+1) per derivative order])."""
    return nurbs.nz_basis(t, p, x, nder)


def _clamped(lo, hi, spans, p=DEGREE):
    inner = list(np.linspace(lo, hi, spans + 1)[1:-1])
    return [lo] * (p + 1) + inner + [hi] * (p + 1)


def _periodic(period, n, p=DEGREE):
    """Uniform knots of a periodic direction with n distinct control columns on [0, period]."""
    h = period / n
    return [(i - p) * h for i in range(n + 2 * p + 1)]


def _ncols(t, wrap):
    return wrap or len(t) - DEGREE - 1


def _cols(fu, fv, nv, wrap):
    """Tensor-product column of each nonzero; a periodic u (wrap = its distinct columns) folds the
    last DEGREE columns onto the first."""
    k = DEGREE + 1
    cu = fu[:, None] + np.arange(k)
    if wrap:
        cu = cu % wrap
    return (cu[:, :, None] * nv + (fv[:, None] + np.arange(k))[:, None, :]).reshape(len(fu), -1)


def _surface(P, tu, tv, wrap=0):
    if wrap:
        P = np.concatenate([P, P[:DEGREE]], axis=0)
    return nurbs.Surface(P, tu, tv, DEGREE, DEGREE)


def _assemble(tu, tv, u, v, X, w=None, wrap=0):
    """A^T A and A^T X for points (u, v) -> X, A the tensor-product basis rows."""
    nu, nv = _ncols(tu, wrap), len(tv) - DEGREE - 1
    fu, (bu,) = _nz(tu, DEGREE, u, 0)
    fv, (bv,) = _nz(tv, DEGREE, v, 0)
    cols = _cols(fu, fv, nv, wrap)
    vals = (bu[:, :, None] * bv[:, None, :]).reshape(len(u), -1)
    if w is not None:
        vals = vals * np.sqrt(w)[:, None]
        X = X * np.sqrt(w)[:, None]
    n = nu * nv
    AtA = np.bincount((cols[:, :, None] * n + cols[:, None, :]).ravel(),
                      (vals[:, :, None] * vals[:, None, :]).ravel(), minlength=n * n).reshape(n, n)
    AtX = np.stack([np.bincount(cols.ravel(), (vals * X[:, None, c]).ravel(), minlength=n) for c in range(3)], 1)
    return AtA, AtX


def _thin_plate(tu, tv, wrap=0):
    """Thin-plate energy matrix: integral of |S_uu|^2 + 2|S_uv|^2 + |S_vv|^2 over the domain, by
    4x4 Gauss points per knot cell."""
    g, gw = np.polynomial.legendre.leggauss(4)
    tu, tv = np.asarray(tu), np.asarray(tv)
    ku = np.unique(tu[DEGREE:len(tu) - DEGREE])        # the domain's knots only (a periodic vector
    kv = np.unique(tv[DEGREE:len(tv) - DEGREE])        # extends past it)
    us, uw = [], []
    for a, b in zip(ku[:-1], ku[1:]):
        us.append((a + b) / 2 + (b - a) / 2 * g)
        uw.append((b - a) / 2 * gw)
    vs, vw = [], []
    for a, b in zip(kv[:-1], kv[1:]):
        vs.append((a + b) / 2 + (b - a) / 2 * g)
        vw.append((b - a) / 2 * gw)
    us, uw, vs, vw = map(np.concatenate, (us, uw, vs, vw))
    U, Vv = np.meshgrid(us, vs, indexing="ij")
    W = np.outer(uw, vw).ravel()
    U, Vv = U.ravel(), Vv.ravel()
    nu, nv = _ncols(tu, wrap), len(tv) - DEGREE - 1
    fu, bu = _nz(tu, DEGREE, U, 2)
    fv, bv = _nz(tv, DEGREE, Vv, 2)
    cols = _cols(fu, fv, nv, wrap)
    n = nu * nv
    R = np.zeros(n * n)
    for (a, b), c in (((2, 0), 1.0), ((1, 1), 2.0), ((0, 2), 1.0)):
        vals = (bu[a][:, :, None] * bv[b][:, None, :]).reshape(len(U), -1) * np.sqrt(W * c)[:, None]
        R += np.bincount((cols[:, :, None] * n + cols[:, None, :]).ravel(),
                         (vals[:, :, None] * vals[:, None, :]).ravel(), minlength=n * n)
    return R.reshape(n, n)


def thin_plate_weight(tu, tv, u, v, X, lam, wrap=0):
    """The absolute thin-plate weight: lam x trace(A^T A) / trace(R) on the given (initial) net.

    Fixed once per fit. The energy is a property of the surface, not of the basis, so its weight must
    not follow the net: re-normalised every refinement round it fell as the spans shrank (trace(R)
    grows much faster than the data term), and a refined net wiggled freely between rows of vertices
    - a synthetic tube 1e-5 on its vertices and 20-30% of its radius off between them."""
    AtA, _ = _assemble(tu, tv, u, v, X, wrap=wrap)
    return lam * np.trace(AtA) / max(np.trace(_thin_plate(tu, tv, wrap)), 1e-300)


def _solve(tu, tv, u, v, X, lam_abs, cons=None, wrap=0):
    """Control points minimising the data + lam_abs x thin-plate energy, and with cons = (uc, vc, N, w)
    also w * (S_u . n)^2 + w * (S_v . n)^2 at each constrained parameter (tangency to the normal n
    there, which couples x, y and z: one 3n x 3n solve). wrap: u periodic with that many columns."""
    nu, nv = _ncols(tu, wrap), len(tv) - DEGREE - 1
    AtA, AtX = _assemble(tu, tv, u, v, X, wrap=wrap)
    M = AtA + lam_abs * _thin_plate(tu, tv, wrap)
    M[np.diag_indices_from(M)] += 1e-12 * np.trace(AtA) / len(M)
    if cons is None or len(cons[0]) == 0:
        return np.linalg.solve(M, AtX).reshape(nu, nv, 3)
    uc, vc, N, w = cons
    n = nu * nv
    fu, bu = _nz(tu, DEGREE, uc, 1)
    fv, bv = _nz(tv, DEGREE, vc, 1)
    cols = _cols(fu, fv, nv, wrap)
    M3 = np.kron(M, np.eye(3))
    for a, b in ((1, 0), (0, 1)):
        vals = (bu[a][:, :, None] * bv[b][:, None, :]).reshape(len(uc), -1) * np.sqrt(w)[:, None]
        # row entries: control point cols[q] times coordinate c, value vals[q] * N[c]
        c3 = (cols[:, :, None] * 3 + np.arange(3)).reshape(len(uc), -1)
        v3 = (vals[:, :, None] * N[:, None, :]).reshape(len(uc), -1)
        M3 += np.bincount((c3[:, :, None] * (3 * n) + c3[:, None, :]).ravel(),
                          (v3[:, :, None] * v3[:, None, :]).ravel(), minlength=9 * n * n).reshape(3 * n, 3 * n)
    return np.linalg.solve(M3, AtX.reshape(-1)).reshape(nu, nv, 3)


def fit_group(V, T, scale=None, tol=TOL, lam=LAMBDA, seed=0, normals=None):
    """Fit a bicubic B-spline to a triangle group. V (n, 3) its vertices, T (m, 3) its triangles
    (indices into V, consistent winding). scale: the part diagonal (for the tolerance; default the
    group's own). normals: (vertex indices, unit normals (k, 3)) the surface must be tangent to -
    where a neighbour meets the group tangentially (solid_rebuild.tangent_normals). Returns
    (params | None, info)."""
    V = np.asarray(V, float)
    T = np.asarray(T, np.int64)
    info = {"n_vertices": len(V), "n_triangles": len(T)}
    if len(T) == 0 or len(V) < 3:
        info["fail"] = "empty"
        return None, info
    scale = scale or float(np.linalg.norm(V.max(0) - V.min(0)))
    period = 0.0
    pp = plane_param(V, T)
    if pp is not None:
        uv = pp[0]
        info["param"] = "plane"
    else:
        uv = lscm(V, T)
        info["param"] = "lscm"
        if uv is None:
            ap = annulus_param(V, T)
            if ap is None:
                info["fail"] = "param"
                return None, info
            uv, period = ap
            info["param"] = "periodic"
    if period:
        # u goes once round: the domain is the period itself, no margin; v as below
        uv = uv.copy()
        uv[:, 0] = np.mod(uv[:, 0], period)
        lo, hi = np.array([0.0, uv[:, 1].min()]), np.array([period, uv[:, 1].max()])
        ext = np.maximum(hi - lo, 1e-9 * scale)
        m = MARGIN * ext[1] + 1e-6 * scale
        lo[1], hi[1] = lo[1] - m, hi[1] + m
    else:
        # tight box: principal axes of the UV points, u the long one
        c = uv.mean(0)
        w_, R_ = np.linalg.eigh(np.cov((uv - c).T) + 1e-300 * np.eye(2))
        uv = (uv - c) @ R_[:, ::-1]
        lo, hi = uv.min(0), uv.max(0)
        ext = np.maximum(hi - lo, 1e-9 * scale)
        m = MARGIN * ext + 1e-6 * scale
        lo, hi = lo - m, hi + m
    # data: all vertices, or the boundary plus a random interior sample
    idx = np.arange(len(V))
    if len(V) > MAX_POINTS:
        bnd = np.unique(np.concatenate([np.array(l) for l in boundary_loops(T)] or [np.zeros(0, np.int64)]))
        if normals is not None:
            bnd = np.union1d(bnd, np.asarray(normals[0], np.int64))
        rest = np.setdiff1d(idx, bnd)
        rng = np.random.default_rng(seed)
        idx = np.concatenate([bnd, rng.choice(rest, max(0, MAX_POINTS - len(bnd)), replace=False)])
    X, u, v = V[idx], uv[idx, 0].copy(), uv[idx, 1].copy()
    # tangency rows ride on the data points of their vertices (their parameters move with them)
    ci = cn = None
    if normals is not None and len(normals[0]):
        where = {int(q): k for k, q in enumerate(idx)}
        ci = np.array([where[int(q)] for q in normals[0]], np.int64)
        cn = np.asarray(normals[1], float)
        e = np.linalg.norm(V[T[:, [1, 2, 0]]] - V[T], axis=2)
        cw = np.full(len(ci), (TANGENT_WEIGHT * float(np.median(e))) ** 2)
        info["tangent_vertices"] = int(len(ci))
    # initial net: one span across, spans along in proportion to the aspect ratio (a periodic
    # direction needs PERIODIC_MIN_SPANS to close smoothly)
    su = int(np.clip(round(ext[0] / max(ext[1], 1e-300)), PERIODIC_MIN_SPANS if period else 1, MAX_SPANS))
    sv = 1
    wrap = su if period else 0
    tu = _periodic(period, su) if period else _clamped(lo[0], hi[0], su)
    tv = _clamped(lo[1], hi[1], sv)
    abs_tol = tol * scale
    lam_abs = thin_plate_weight(tu, tv, u, v, X, lam, wrap)
    # HOLD-OUT VALIDATION picks the refinement level. Vertex error alone cannot see a net too free
    # for its data: it fits every vertex and wiggles between them (a synthetic tube 1e-5 on its
    # vertices, 30% of its radius off between them; a split piece 1e-3 / 9e-2). HOLDOUT_FRAC of the
    # interior vertices are left out of the least squares, re-projected every round, and the net with
    # the lowest held-out error is kept - refinement stops once it stops improving. The kept net is
    # then fitted to every vertex. Boundary and tangency vertices always stay in (the knitter's
    # edges lie on them).
    hold = np.zeros(len(X), bool)
    bset = set(np.concatenate([np.array(l) for l in boundary_loops(T)] or [np.zeros(0, np.int64)]).tolist())
    cand = [k for k, q in enumerate(idx) if int(q) not in bset]
    if ci is not None:
        cand = sorted(set(cand) - set(ci.tolist()))
    n_hold = int(HOLDOUT_FRAC * len(X))
    if n_hold >= HOLDOUT_MIN and len(cand) > 2 * n_hold:
        hold[np.random.default_rng(seed).choice(cand, n_hold, replace=False)] = True
    F = ~hold
    best, stale = None, 0
    for rnd in range(REFINE_ROUNDS + 1):
        for _ in range(CORRECT_ITERS):
            P = _solve(tu, tv, u[F], v[F], X[F], lam_abs, wrap=wrap)
            S = _surface(P, tu, tv, wrap)
            u, v, Q = S.closest(X, uv0=(u, v), iters=8)
        err = np.linalg.norm(Q - X, axis=1)
        val = float(err[hold].max()) if hold.any() else float(err.max())
        if best is None or val < best[0]:
            best, stale = (val, list(tu), list(tv), wrap, u.copy(), v.copy(), rnd), 0
        else:
            stale += 1
        if err[F].max() <= abs_tol or rnd == REFINE_ROUNDS or stale >= VAL_PATIENCE:
            break
        # insert a knot mid-span in every span (either direction) holding a point over tolerance and
        # REFINE_MIN_POINTS points in all; a periodic u is refined uniformly (its spans doubled).
        # DOF_RATIO and MAX_CP only bound the cost; the hold-out decides.
        over = (err > abs_tol) & F
        kv = np.unique(tv)
        vs = np.searchsorted(kv, v, side="right") - 1
        add_v = [(kv[i] + kv[i + 1]) / 2 for i in set(vs[over].tolist())
                 if 0 <= i < len(kv) - 1 and (vs == i).sum() >= REFINE_MIN_POINTS]
        if len(kv) - 1 + len(add_v) > MAX_SPANS:
            add_v = []
        nv_new = len(tv) - DEGREE - 1 + len(add_v)
        cap = min(DOF_RATIO * len(X), MAX_CP)
        if wrap:
            grow_u = 2 * wrap <= MAX_SPANS and 2 * wrap * nv_new <= cap
            if not grow_u and (not add_v or wrap * nv_new > cap):
                break
            if grow_u:
                wrap *= 2
                tu = _periodic(period, wrap)
        else:
            ku = np.unique(tu)
            us = np.searchsorted(ku, u, side="right") - 1
            add_u = [(ku[i] + ku[i + 1]) / 2 for i in set(us[over].tolist())
                     if 0 <= i < len(ku) - 1 and (us == i).sum() >= REFINE_MIN_POINTS]
            if len(ku) - 1 + len(add_u) > MAX_SPANS:
                add_u = []
            if not add_u and not add_v:
                break
            if (len(tu) - DEGREE - 1 + len(add_u)) * nv_new > cap:
                break
            tu = sorted(tu + add_u)
        tv = sorted(tv + add_v)
    # the best net by the hold-out, fitted to every vertex
    _, tu, tv, wrap, u, v, rnd = best
    for _ in range(CORRECT_ITERS):
        P = _solve(tu, tv, u, v, X, lam_abs, wrap=wrap)
        S = _surface(P, tu, tv, wrap)
        u, v, Q = S.closest(X, uv0=(u, v), iters=8)
    err = np.linalg.norm(Q - X, axis=1)
    info["holdout"] = int(hold.sum())
    info["holdout_err"] = float(best[0] / scale)
    # Tangency, checked against the data: a neighbour's normal is imposed only where the free fit
    # already agrees with it (TANGENT_AGREE_DEG), and the tangent refit - same knots - is kept only if
    # the vertices still fit. A mesh dihedral cannot tell a coarse fillet from a sharp edge: imposed
    # blindly below 60 deg it pulled B-spline faces meeting planes at 45 deg edges off their data
    # (smoke set found3 97.8% -> 42.7%).
    if ci is not None and _ncols(tu, wrap) * (len(tv) - DEGREE - 1) <= TANGENT_MAX_CP:
        _, n0 = S.normal(u[ci], v[ci])
        agree = np.abs((n0 * cn).sum(1)) >= math.cos(math.radians(TANGENT_AGREE_DEG))
        info["tangent_agree"] = int(agree.sum())
        if agree.any():
            u1, v1 = u.copy(), v.copy()
            for _ in range(CORRECT_ITERS):
                P1 = _solve(tu, tv, u1, v1, X, lam_abs, (u1[ci[agree]], v1[ci[agree]], cn[agree], cw[agree]), wrap=wrap)
                S1 = _surface(P1, tu, tv, wrap)
                u1, v1, Q1 = S1.closest(X, uv0=(u1, v1), iters=8)
            err1 = np.linalg.norm(Q1 - X, axis=1)
            if err1.max() <= max(abs_tol, TANGENT_ERR_RATIO * err.max()):
                S, u, v, err = S1, u1, v1, err1
                info["tangent_used"] = int(agree.sum())
    info.update(spans=(_ncols(tu, wrap) - (0 if wrap else DEGREE), len(np.unique(tv)) - 1), rounds=rnd,
                max_err=float(err.max() / scale), rms_err=float(np.sqrt((err ** 2).mean()) / scale))
    params = S.to_params()
    if wrap:
        params["u_periodic"] = True
    return params, info
