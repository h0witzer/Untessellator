/**
 * freeformFit.js — one untrimmed bicubic B-spline surface per mesh group, for groups no analytic surface
 * fits. A port of the core of ml/freeform_fit.py (measured there on 2105 ABC freeform faces: found3
 * 87%; with true segments it rebuilt 102 of 150 freeform parts):
 *
 *   parameterise  orthogonal projection onto the mean-normal plane when no triangle folds over in it,
 *                 else a least-squares conformal map (LSCM, by conjugate gradients)
 *   domain        the tight UV box plus a 10% margin, so the knitter's boundary curves lie on the surface
 *   fit           least squares on the vertices plus a thin-plate term (fixed weight) for where there is
 *                 no data; vertices re-projected and the fit repeated (parameter correction)
 *   refine        knots inserted mid-span where vertices miss the tolerance; a 10% interior hold-out
 *                 picks the refinement level, so the net cannot wiggle between vertices
 *
 * Rings and tubes (no disk map) get a periodic conformal map and a surface periodic round them; groups
 * no single map covers are split by normal cones (fitPieces). Not ported: tangency to neighbours (off
 * by default in Python too). The normal equations are solved banded (a tensor-product
 * basis couples only nearby control points), so a 3000-pole net costs n x bandwidth^2, not n^3.
 */

import { nzBasis, Surface } from './nurbs.js';

export const FIT = {
  DEGREE: 3, MARGIN: 0.10, PLANE_MIN_COS: 0.15, TOL: 1e-5, LAMBDA: 1e-6, CORRECT_ITERS: 3, REFINE_ROUNDS: 8,
  MAX_SPANS: 80, MAX_POINTS: 6000, DOF_RATIO: 3, MAX_CP: 3000, HOLDOUT_FRAC: 0.1, HOLDOUT_MIN: 8,
  VAL_PATIENCE: 1, REFINE_MIN_POINTS: 4, PERIODIC_MIN_SPANS: 6,
};
const P_ = 3;

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm = (a) => Math.hypot(a[0], a[1], a[2]);

function frame(n) {
  const l = norm(n);
  n = [n[0] / l, n[1] / l, n[2] / l];
  const ref = Math.abs(n[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  let x = cross(n, ref);
  const lx = norm(x);
  x = [x[0] / lx, x[1] / lx, x[2] / lx];
  return [x, cross(n, x)];
}

function edgeCounts(T) {
  const cnt = new Map();
  for (const [a, b, c] of T) for (const [u, v] of [[a, b], [b, c], [c, a]]) {
    const k = u < v ? `${u}_${v}` : `${v}_${u}`;
    cnt.set(k, (cnt.get(k) ?? 0) + 1);
  }
  return cnt;
}

function components(T, n) {
  const par = Int32Array.from({ length: n }, (_, i) => i);
  const find = (x) => { while (par[x] !== x) { par[x] = par[par[x]]; x = par[x]; } return x; };
  for (const [a, b, c] of T) { const ra = find(a); par[find(b)] = ra; par[find(c)] = ra; }
  const roots = new Set();
  for (const [a, b, c] of T) { roots.add(find(a)); roots.add(find(b)); roots.add(find(c)); }
  return roots.size;
}

/** UV by projection onto the plane of the area-weighted mean normal, or null when a triangle folds over. */
export function planeParam(V, T) {
  const N = T.map(([a, b, c]) => cross(sub(V[b], V[a]), sub(V[c], V[a])));
  const n = N.reduce((s, x) => [s[0] + x[0], s[1] + x[1], s[2] + x[2]], [0, 0, 0]);
  const ln = norm(n);
  if (ln < 1e-300) return null;
  const nu = [n[0] / ln, n[1] / ln, n[2] / ln];
  const areas = N.map(norm);
  const amax = Math.max(...areas);
  for (let i = 0; i < N.length; i++) if (areas[i] > 1e-12 * amax && dot(N[i], nu) / areas[i] <= FIT.PLANE_MIN_COS) return null;
  const [x, y] = frame(nu);
  return V.map((p) => [dot(p, x), dot(p, y)]);
}

/** Least-squares conformal map of a disk, two far-apart boundary vertices pinned; null if not a disk or it folds. */
export function lscm(V, T, iters = 20000) {
  const nV = V.length;
  const cnt = edgeCounts(T);
  if (nV - cnt.size + T.length !== 1 || components(T, nV) !== 1) return null;
  const bset = new Set();
  for (const [k, c] of cnt) if (c === 1) for (const q of k.split('_').map(Number)) bset.add(q);
  const b = [...bset];
  if (b.length < 2) return null;
  const mean = b.reduce((s, q) => [s[0] + V[q][0] / b.length, s[1] + V[q][1] / b.length, s[2] + V[q][2] / b.length], [0, 0, 0]);
  let i0 = b[0];
  for (const q of b) if (norm(sub(V[q], mean)) > norm(sub(V[i0], mean))) i0 = q;
  let i1 = b[0];
  for (const q of b) if (norm(sub(V[q], V[i0])) > norm(sub(V[i1], V[i0]))) i1 = q;
  const pin = new Map([[i0, [0, 0]], [i1, [norm(sub(V[i1], V[i0])), 0]]]);
  return lscmSolve(V, T, pin, null, 0, iters);
}

/**
 * A ring or tube (connected, Euler characteristic 0, two boundary loops): {uv, period}, a periodic
 * conformal map in which u goes once round, or null (ml/freeform_fit.py annulus_param). The ring is cut
 * along ONE seam, the shortest path between its two boundary loops, and LSCM is solved with the
 * triangles right of the seam carrying their seam vertices shifted by the period (the first loop's length).
 */
export function annulusParam(V, T) {
  const nV = V.length;
  const cnt = edgeCounts(T);
  if (nV - cnt.size + T.length !== 0 || components(T, nV) !== 1) return null;
  const bedges = [...cnt].filter(([, c]) => c === 1).map(([k]) => k.split('_').map(Number));
  const par = new Map();
  const find = (x) => { if (!par.has(x)) par.set(x, x); while (par.get(x) !== x) { par.set(x, par.get(par.get(x))); x = par.get(x); } return x; };
  for (const [a, b] of bedges) par.set(find(a), find(b));
  const comps = new Map();
  for (const [a, b] of bedges) { const r = find(a); if (!comps.has(r)) comps.set(r, new Set()); comps.get(r).add(a).add(b); }
  if (comps.size !== 2) return null;
  const [A, B] = [...comps.values()];
  let period = 0;
  for (const [a, b] of bedges) if (A.has(a)) period += norm(sub(V[a], V[b]));
  const adj = new Map();
  for (const k of cnt.keys()) {
    const [a, b] = k.split('_').map(Number);
    const l = norm(sub(V[a], V[b]));
    (adj.get(a) ?? adj.set(a, []).get(a)).push([b, l]);
    (adj.get(b) ?? adj.set(b, []).get(b)).push([a, l]);
  }
  // Dijkstra from loop A to loop B (a binary heap)
  const dist = new Map([...A].map((a) => [a, 0]));
  const prev = new Map();
  const heap = [...A].map((a) => [0, a]);
  const push = (e) => { heap.push(e); let i = heap.length - 1; while (i > 0) { const p = (i - 1) >> 1; if (heap[p][0] <= heap[i][0]) break; [heap[p], heap[i]] = [heap[i], heap[p]]; i = p; } };
  const pop = () => { const top = heap[0], last = heap.pop(); if (heap.length) { heap[0] = last; let i = 0; for (;;) { const l = 2 * i + 1, r = l + 1; let m = i; if (l < heap.length && heap[l][0] < heap[m][0]) m = l; if (r < heap.length && heap[r][0] < heap[m][0]) m = r; if (m === i) break; [heap[m], heap[i]] = [heap[i], heap[m]]; i = m; } } return top; };
  for (let i = (heap.length >> 1) - 1; i >= 0; i--) { /* already all zero: a valid heap */ }
  let end = null;
  while (heap.length) {
    const [d, x] = pop();
    if (d > (dist.get(x) ?? Infinity)) continue;
    if (B.has(x)) { end = x; break; }
    for (const [y, l] of adj.get(x) ?? []) {
      if (A.has(y)) continue;
      if (d + l < (dist.get(y) ?? Infinity)) { dist.set(y, d + l); prev.set(y, x); push([d + l, y]); }
    }
  }
  if (end === null) return null;
  const path = [end];
  while (prev.has(path[path.length - 1])) path.push(prev.get(path[path.length - 1]));
  path.reverse();                                       // path[0] on loop A, the last on loop B
  const seam = new Set(path.slice(1).map((b, i) => { const a = path[i]; return a < b ? `${a}_${b}` : `${b}_${a}`; }));
  const triOf = new Map(), inc = new Map();
  T.forEach((t, i) => { for (let k = 0; k < 3; k++) { triOf.set(`${t[k]}>${t[(k + 1) % 3]}`, i); (inc.get(t[k]) ?? inc.set(t[k], []).get(t[k])).push(i); } });
  const shift = new Set();                              // "triangle,corner"
  for (let i = 0; i < path.length; i++) {
    const x = path[i];
    // the triangle right of the seam at x: it holds the seam edge reversed
    const start = i + 1 < path.length ? triOf.get(`${path[i + 1]}>${x}`) : triOf.get(`${x}>${path[i - 1]}`);
    if (start === undefined) return null;
    const fan = new Set(inc.get(x)), seen = new Set([start]), stack = [start];
    while (stack.length) {
      const t = stack.pop();
      for (const w of T[t]) {
        if (w === x || seam.has(x < w ? `${x}_${w}` : `${w}_${x}`)) continue;
        for (const o of [triOf.get(`${x}>${w}`), triOf.get(`${w}>${x}`)]) {
          if (o !== undefined && o !== t && fan.has(o) && !seen.has(o)) { seen.add(o); stack.push(o); }
        }
      }
    }
    for (const t of seen) shift.add(`${t},${T[t].indexOf(x)}`);
  }
  const uv = lscmSolve(V, T, new Map([[path[0], [0, 0]]]), shift, period, 20000);
  return uv ? { uv, period } : null;
}

/**
 * The LSCM energy minimised with the pinned vertices fixed, by Jacobi-preconditioned CG on the normal
 * equations. shift: "triangle,corner" keys whose vertex sits at u + period in that triangle (right of
 * a ring's seam). Returns vertex uv, mirrored in v if need be, or null if any triangle folds.
 */
function lscmSolve(V, T, pin, shift, period, iters) {
  const nV = V.length;
  // per triangle: local 2D frame, its two Cauchy-Riemann rows over the corner (u, v) unknowns
  const rows = [], cols = [], vals = [], konst = [];
  let nr = 0;
  const good = [];
  const shifted = (ti, k) => (shift && shift.has(`${ti},${k}`) ? period : 0);
  const A2s = T.map(([a, bb, c]) => norm(cross(sub(V[bb], V[a]), sub(V[c], V[a]))));
  const A2max = Math.max(...A2s, 0);
  T.forEach(([a, bb, c], ti) => {
    const p0 = V[a], p1 = V[bb], p2 = V[c];
    const e1 = sub(p1, p0), l1 = norm(e1);
    const ex = [e1[0] / (l1 || 1e-300), e1[1] / (l1 || 1e-300), e1[2] / (l1 || 1e-300)];
    const nn = cross(e1, sub(p2, p0));
    const A2 = A2s[ti];
    if (!(A2 > 1e-14 * A2max)) return;
    good.push(ti);
    const ey = cross([nn[0] / A2, nn[1] / A2, nn[2] / A2], ex);
    const xs = [0, l1, dot(sub(p2, p0), ex)], ys = [0, 0, dot(sub(p2, p0), ey)];
    const s = 1 / Math.sqrt(A2 * 2);
    const tri = [a, bb, c];
    for (let k = 0; k < 3; k++) {
      const kk = (k + 1) % 3, l = (k + 2) % 3;
      const eax = xs[l] - xs[kk], eay = ys[l] - ys[kk];
      const vi = tri[k];
      rows.push(nr, nr); cols.push(2 * vi, 2 * vi + 1); vals.push(-eay * s, -eax * s);
      rows.push(nr + 1, nr + 1); cols.push(2 * vi, 2 * vi + 1); vals.push(eax * s, -eay * s);
      const sh = shifted(ti, k);                        // a shifted corner's u is u + period
      if (sh) konst.push([nr, -eay * s * sh], [nr + 1, eax * s * sh]);
    }
    nr += 2;
  });
  const fixed = new Float64Array(2 * nV), isfree = new Uint8Array(2 * nV).fill(1);
  for (const [q, [pu, pv]] of pin) { fixed[2 * q] = pu; fixed[2 * q + 1] = pv; isfree[2 * q] = isfree[2 * q + 1] = 0; }
  const fidx = new Int32Array(2 * nV).fill(-1);
  let nf = 0;
  for (let i = 0; i < 2 * nV; i++) if (isfree[i]) fidx[i] = nf++;
  const rhs = new Float64Array(nr);
  for (const [row, c] of konst) rhs[row] -= c;
  const R = [], C = [], W = [];
  for (let k = 0; k < rows.length; k++) {
    if (isfree[cols[k]]) { R.push(rows[k]); C.push(fidx[cols[k]]); W.push(vals[k]); }
    else rhs[rows[k]] -= vals[k] * fixed[cols[k]];
  }
  const Av = (x) => { const y = new Float64Array(nr); for (let k = 0; k < R.length; k++) y[R[k]] += W[k] * x[C[k]]; return y; };
  const Atv = (y) => { const x = new Float64Array(nf); for (let k = 0; k < R.length; k++) x[C[k]] += W[k] * y[R[k]]; return x; };
  const diag = new Float64Array(nf);
  for (let k = 0; k < R.length; k++) diag[C[k]] += W[k] * W[k];
  // start from the plane projection
  const nsum = T.reduce((s, [a, bb, c]) => { const n = cross(sub(V[bb], V[a]), sub(V[c], V[a])); return [s[0] + n[0], s[1] + n[1], s[2] + n[2]]; }, [1e-300, 0, 0]);
  const [x0, y0] = frame(nsum);
  const x = new Float64Array(nf);                    // a ring starts from zero
  if (!shift) for (let q = 0; q < nV; q++) {
    if (isfree[2 * q]) x[fidx[2 * q]] = dot(V[q], x0);
    if (isfree[2 * q + 1]) x[fidx[2 * q + 1]] = dot(V[q], y0);
  }
  const Ax = Av(x);
  const res = new Float64Array(nr);
  for (let i = 0; i < nr; i++) res[i] = rhs[i] - Ax[i];
  const r = Atv(res);
  const z = r.map((v, i) => v / Math.max(diag[i], 1e-300));
  const p = Float64Array.from(z);
  let rz = r.reduce((s, v, i) => s + v * z[i], 0);
  const r0 = Math.sqrt(Math.max(r.reduce((s, v) => s + v * v, 0), 1e-300));
  for (let it = 0; it < iters; it++) {
    const Ap = Atv(Av(p));
    let pAp = 0;
    for (let i = 0; i < nf; i++) pAp += p[i] * Ap[i];
    const alpha = rz / Math.max(pAp, 1e-300);
    let rr = 0;
    for (let i = 0; i < nf; i++) { x[i] += alpha * p[i]; r[i] -= alpha * Ap[i]; rr += r[i] * r[i]; }
    if (Math.sqrt(rr) < 1e-10 * r0) break;
    let rz2 = 0;
    for (let i = 0; i < nf; i++) { z[i] = r[i] / Math.max(diag[i], 1e-300); rz2 += r[i] * z[i]; }
    const beta = rz2 / Math.max(rz, 1e-300);
    for (let i = 0; i < nf; i++) p[i] = z[i] + beta * p[i];
    rz = rz2;
  }
  const uv = V.map((_, q) => [isfree[2 * q] ? x[fidx[2 * q]] : fixed[2 * q], isfree[2 * q + 1] ? x[fidx[2 * q + 1]] : fixed[2 * q + 1]]);
  const signedArea = (ti) => {
    const [a, bb, c] = T[ti];
    const ua = uv[a][0] + shifted(ti, 0), ub = uv[bb][0] + shifted(ti, 1), uc = uv[c][0] + shifted(ti, 2);
    return (ub - ua) * (uv[c][1] - uv[a][1]) - (uv[bb][1] - uv[a][1]) * (uc - ua);
  };
  let neg = 0;
  for (const ti of good) if (signedArea(ti) <= 0) neg++;
  if (neg > 0.5 * good.length) for (const q of uv) q[1] = -q[1];
  for (const ti of good) if (signedArea(ti) <= 0) return null;
  return uv;
}

// ── the least-squares fit ────────────────────────────────────────────────────

function clamped(lo, hi, spans) {
  const t = [lo, lo, lo, lo];
  for (let i = 1; i < spans; i++) t.push(lo + ((hi - lo) * i) / spans);
  t.push(hi, hi, hi, hi);
  return t;
}

/** Symmetric banded matrix (half-bandwidth bw), upper band stored per row. */
class Band {
  constructor(n, bw) { this.n = n; this.bw = bw; this.a = new Float64Array(n * (bw + 1)); }
  add(i, j, v) { if (j < i) [i, j] = [j, i]; this.a[i * (this.bw + 1) + (j - i)] += v; }
  get(i, j) { if (j < i) [i, j] = [j, i]; return j - i > this.bw ? 0 : this.a[i * (this.bw + 1) + (j - i)]; }
  /** Solve A X = B (B: n x 3, flat) by banded Cholesky; returns null if not positive definite. */
  solve(B) {
    const { n, bw } = this;
    const L = new Float64Array(n * (bw + 1));       // L[i][i - j] for j in [i - bw, i]
    const Lget = (i, j) => (i - j > bw || j > i ? 0 : L[i * (bw + 1) + (i - j)]);
    for (let j = 0; j < n; j++) {
      let s = this.get(j, j);
      for (let k = Math.max(0, j - bw); k < j; k++) { const l = Lget(j, k); s -= l * l; }
      if (!(s > 0)) return null;
      const d = Math.sqrt(s);
      L[j * (bw + 1)] = d;
      for (let i = j + 1; i <= Math.min(n - 1, j + bw); i++) {
        let t = this.get(i, j);
        for (let k = Math.max(0, i - bw); k < j; k++) t -= Lget(i, k) * Lget(j, k);
        L[i * (bw + 1) + (i - j)] = t / d;
      }
    }
    const X = Float64Array.from(B);
    for (let c = 0; c < 3; c++) {
      for (let i = 0; i < n; i++) {
        let t = X[i * 3 + c];
        for (let k = Math.max(0, i - bw); k < i; k++) t -= Lget(i, k) * X[k * 3 + c];
        X[i * 3 + c] = t / L[i * (bw + 1)];
      }
      for (let i = n - 1; i >= 0; i--) {
        let t = X[i * 3 + c];
        for (let k = i + 1; k <= Math.min(n - 1, i + bw); k++) t -= Lget(k, i) * X[k * 3 + c];
        X[i * 3 + c] = t / L[i * (bw + 1)];
      }
    }
    return X;
  }
}

const GAUSS = [[-0.8611363115940526, 0.3478548451374538], [-0.3399810435848563, 0.6521451548625461],
               [0.3399810435848563, 0.6521451548625461], [0.8611363115940526, 0.3478548451374538]];

function distinct(t) { const out = []; for (const x of t) if (!out.length || x > out[out.length - 1]) out.push(x); return out; }

/** Uniform knots of a periodic direction with n distinct control columns on [0, period]. */
function periodicKnots(period, n) {
  const h = period / n;
  return Array.from({ length: n + 2 * P_ + 1 }, (_, i) => (i - P_) * h);
}

/** Data term into M (band) and rhs: A^T A, A^T X for points (u, v) -> X. Returns trace(A^T A). wrap: v periodic. */
function assemble(M, rhs, tu, tv, U, Vv, X, nv, wrap = 0) {
  let tr = 0;
  for (let k = 0; k < U.length; k++) {
    const bu = nzBasis(tu, P_, U[k], 0), bv = nzBasis(tv, P_, Vv[k], 0);
    const cols = [], vals = [];
    for (let i = 0; i <= P_; i++) for (let j = 0; j <= P_; j++) {
      const w = bu.ders[0][i] * bv.ders[0][j];
      if (!w) continue;
      cols.push((bu.first + i) * nv + (wrap ? (bv.first + j) % wrap : bv.first + j)); vals.push(w);
    }
    for (let a = 0; a < cols.length; a++) {
      for (let b = a; b < cols.length; b++) M.add(cols[a], cols[b], vals[a] * vals[b]);
      for (let c = 0; c < 3; c++) rhs[cols[a] * 3 + c] += vals[a] * X[k][c];
      tr += vals[a] * vals[a];
    }
  }
  return tr;
}

/** Thin-plate energy (|S_uu|^2 + 2|S_uv|^2 + |S_vv|^2 over the domain) into M with weight lam. Returns its trace. */
function thinPlate(M, tu, tv, nv, lam, wrap = 0) {
  const ku = distinct(tu.slice(P_, tu.length - P_)), kv = distinct(tv.slice(P_, tv.length - P_));
  let tr = 0;
  for (let a = 0; a + 1 < ku.length; a++) for (const [gu, wu] of GAUSS) {
    const u = (ku[a] + ku[a + 1]) / 2 + ((ku[a + 1] - ku[a]) / 2) * gu, wU = ((ku[a + 1] - ku[a]) / 2) * wu;
    const bu = nzBasis(tu, P_, u, 2);
    for (let b = 0; b + 1 < kv.length; b++) for (const [gv, wv] of GAUSS) {
      const v = (kv[b] + kv[b + 1]) / 2 + ((kv[b + 1] - kv[b]) / 2) * gv, wV = ((kv[b + 1] - kv[b]) / 2) * wv;
      const bv = nzBasis(tv, P_, v, 2);
      for (const [du, dvv, c] of [[2, 0, 1], [1, 1, 2], [0, 2, 1]]) {
        const cols = [], vals = [];
        const sw = Math.sqrt(wU * wV * c);
        for (let i = 0; i <= P_; i++) for (let j = 0; j <= P_; j++) {
          const w = bu.ders[du][i] * bv.ders[dvv][j] * sw;
          if (!w) continue;
          cols.push((bu.first + i) * nv + (wrap ? (bv.first + j) % wrap : bv.first + j)); vals.push(w);
        }
        for (let x = 0; x < cols.length; x++) {
          tr += vals[x] * vals[x];
          if (lam) for (let y = x; y < cols.length; y++) M.add(cols[x], cols[y], lam * vals[x] * vals[y]);
        }
      }
    }
  }
  return tr;
}

const bandOf = (nv, wrap) => (wrap ? (P_ + 1) * wrap : P_ * nv + P_);

function solveNet(tu, tv, U, Vv, X, lamAbs, wrap = 0) {
  const nu = tu.length - P_ - 1, nv = wrap || tv.length - P_ - 1, n = nu * nv;
  const M = new Band(n, bandOf(nv, wrap));
  const rhs = new Float64Array(n * 3);
  const tr = assemble(M, rhs, tu, tv, U, Vv, X, nv, wrap);
  thinPlate(M, tu, tv, nv, lamAbs, wrap);
  const ridge = (1e-12 * tr) / n;
  for (let i = 0; i < n; i++) M.add(i, i, ridge);
  const P = M.solve(rhs);
  if (!P) return null;
  if (!wrap) return new Surface(P, nu, nv, tu, tv, P_, P_);
  // the unwrapped form: the first P_ columns repeated at the end, over the periodic knot vector
  const W = nv + P_, Q = new Float64Array(nu * W * 3);
  for (let i = 0; i < nu; i++) for (let j = 0; j < W; j++) for (let c = 0; c < 3; c++) Q[(i * W + j) * 3 + c] = P[(i * nv + (j % nv)) * 3 + c];
  const S = new Surface(Q, nu, W, tu, tv, P_, P_);
  S.wrap = nv;
  return S;
}

function lcg(seed) { let s = seed >>> 0 || 1; return () => (s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296; }

/**
 * Fit a bicubic B-spline to a triangle group. V: [[x, y, z]] its vertices, T: [[a, b, c]] its triangles
 * (indices into V). scale: the part diagonal, for the tolerance (default the group's own). Returns
 * {params (nurbs.Surface.toParams) | null, info: {param, spans, rounds, max_err, rms_err (x scale), fail?}}.
 */
export function fitGroup(V, T, { scale = null, tol = FIT.TOL, lam = FIT.LAMBDA, seed = 0 } = {}) {
  const info = { n_vertices: V.length, n_triangles: T.length };
  if (!T.length || V.length < 3) return { params: null, info: { ...info, fail: 'empty' } };
  if (!scale) {
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (const p of V) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], p[k]); hi[k] = Math.max(hi[k], p[k]); }
    scale = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
  }
  let uv = planeParam(V, T), period = 0;
  info.param = 'plane';
  if (!uv) {
    uv = lscm(V, T);
    info.param = 'lscm';
    if (!uv) {
      const ap = annulusParam(V, T);
      if (!ap) return { params: null, info: { ...info, fail: 'param' } };
      ({ uv, period } = ap);
      info.param = 'periodic';
    }
  }
  const lo = [Infinity, Infinity], hi = [-Infinity, -Infinity];
  if (period) {
    // round the ring is v here (the banded solve wraps v; the params are transposed to Python's
    // u-periodic form at the end): its domain is the period itself, no margin
    uv = uv.map((q) => [q[1], ((q[0] % period) + period) % period]);
    for (const q of uv) { lo[0] = Math.min(lo[0], q[0]); hi[0] = Math.max(hi[0], q[0]); }
    lo[1] = 0; hi[1] = period;
  } else {
    // tight box: principal axes of the UV points, u the long one
    const c = uv.reduce((s, q) => [s[0] + q[0] / uv.length, s[1] + q[1] / uv.length], [0, 0]);
    let sxx = 0, sxy = 0, syy = 0;
    for (const q of uv) { const dx = q[0] - c[0], dy = q[1] - c[1]; sxx += dx * dx; sxy += dx * dy; syy += dy * dy; }
    const ang = 0.5 * Math.atan2(2 * sxy, sxx - syy);          // the major axis
    const ca = Math.cos(ang), sa = Math.sin(ang);
    uv = uv.map((q) => [(q[0] - c[0]) * ca + (q[1] - c[1]) * sa, -(q[0] - c[0]) * sa + (q[1] - c[1]) * ca]);
    for (const q of uv) for (let k = 0; k < 2; k++) { lo[k] = Math.min(lo[k], q[k]); hi[k] = Math.max(hi[k], q[k]); }
  }
  const ext = [Math.max(hi[0] - lo[0], 1e-9 * scale), Math.max(hi[1] - lo[1], 1e-9 * scale)];
  for (let k = 0; k < (period ? 1 : 2); k++) { const m = FIT.MARGIN * ext[k] + 1e-6 * scale; lo[k] -= m; hi[k] += m; }
  // data: all vertices, or the boundary plus a sample of the interior
  const cnt = edgeCounts(T);
  const bset = new Set();
  for (const [k, n] of cnt) if (n === 1) for (const q of k.split('_').map(Number)) bset.add(q);
  let idx = V.map((_, i) => i);
  const rand = lcg(seed + 1);
  if (V.length > FIT.MAX_POINTS) {
    const rest = idx.filter((i) => !bset.has(i));
    for (let i = rest.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [rest[i], rest[j]] = [rest[j], rest[i]]; }
    idx = [...bset, ...rest.slice(0, Math.max(0, FIT.MAX_POINTS - bset.size))];
  }
  const X = idx.map((i) => V[i]);
  let U = Float64Array.from(idx, (i) => uv[i][0]), Vv = Float64Array.from(idx, (i) => uv[i][1]);
  // initial net: one span across, spans along in proportion to the aspect ratio (round a ring, at
  // least PERIODIC_MIN_SPANS so it closes smoothly; wrap = its distinct columns)
  let tu, tv, wrap = 0;
  if (period) {
    wrap = Math.min(FIT.MAX_SPANS, Math.max(FIT.PERIODIC_MIN_SPANS, Math.round(ext[1] / Math.max(ext[0], 1e-300))));
    tu = clamped(lo[0], hi[0], 1);
    tv = periodicKnots(period, wrap);
  } else {
    const su = Math.min(FIT.MAX_SPANS, Math.max(1, Math.round(ext[0] / Math.max(ext[1], 1e-300))));
    tu = clamped(lo[0], hi[0], su);
    tv = clamped(lo[1], hi[1], 1);
  }
  const absTol = tol * scale;
  // the thin-plate weight, fixed once: lam x trace(A^T A) / trace(R) on the initial net
  {
    const nv = wrap || tv.length - P_ - 1, n = (tu.length - P_ - 1) * nv;
    const M = new Band(n, bandOf(nv, wrap));
    const trA = assemble(M, new Float64Array(n * 3), tu, tv, U, Vv, X, nv, wrap);
    const trR = thinPlate(M, tu, tv, nv, 0, wrap);
    info.lam_abs = (lam * trA) / Math.max(trR, 1e-300);
  }
  const lamAbs = info.lam_abs;
  // hold-out: 10% of the interior vertices left out, to choose the refinement level
  const hold = new Uint8Array(X.length);
  const cand = idx.map((q, k) => (bset.has(q) ? -1 : k)).filter((k) => k >= 0);
  const nHold = Math.floor(FIT.HOLDOUT_FRAC * X.length);
  if (nHold >= FIT.HOLDOUT_MIN && cand.length > 2 * nHold) {
    for (let i = cand.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [cand[i], cand[j]] = [cand[j], cand[i]]; }
    for (const k of cand.slice(0, nHold)) hold[k] = 1;
  }
  const F = [...hold.keys()].filter((k) => !hold[k]);
  const err = new Float64Array(X.length);
  const project = (S) => {
    for (let k = 0; k < X.length; k++) {
      const r = S.closest(X[k], [U[k], Vv[k]], 8);
      U[k] = r.u; Vv[k] = r.v;
      err[k] = norm(sub(r.S, X[k]));
    }
  };
  let best = null, stale = 0, S = null;
  for (let rnd = 0; rnd <= FIT.REFINE_ROUNDS; rnd++) {
    for (let it = 0; it < FIT.CORRECT_ITERS; it++) {
      const s2 = solveNet(tu, tv, F.map((k) => U[k]), F.map((k) => Vv[k]), F.map((k) => X[k]), lamAbs, wrap);
      if (!s2) break;
      S = s2;
      project(S);
    }
    if (!S) return { params: null, info: { ...info, fail: 'solve' } };
    let val = 0, errF = 0;
    for (let k = 0; k < X.length; k++) { if (hold[k]) val = Math.max(val, err[k]); else errF = Math.max(errF, err[k]); }
    if (!hold.some(Boolean)) val = errF;
    if (!best || val < best.val) { best = { val, tu: tu.slice(), tv: tv.slice(), wrap, U: Float64Array.from(U), V: Float64Array.from(Vv), rnd }; stale = 0; } else stale++;
    if (errF <= absTol || rnd === FIT.REFINE_ROUNDS || stale >= FIT.VAL_PATIENCE) break;
    // a knot mid-span in every span holding a vertex over tolerance and REFINE_MIN_POINTS in all;
    // round a ring the spans are doubled instead
    const over = (k) => err[k] > absTol && !hold[k];
    const kvD = distinct(tv), kuD = distinct(tu);
    const spanIdx = (kd, x) => { let i = 0; while (i + 1 < kd.length - 1 && x >= kd[i + 1]) i++; return i; };
    const addV = [], addU = [];
    const countV = new Map(), countU = new Map(), overV = new Set(), overU = new Set();
    for (let k = 0; k < X.length; k++) {
      const iv = spanIdx(kvD, Vv[k]), iu = spanIdx(kuD, U[k]);
      countV.set(iv, (countV.get(iv) ?? 0) + 1); countU.set(iu, (countU.get(iu) ?? 0) + 1);
      if (over(k)) { overV.add(iv); overU.add(iu); }
    }
    if (!wrap) for (const i of overV) if ((countV.get(i) ?? 0) >= FIT.REFINE_MIN_POINTS) addV.push((kvD[i] + kvD[i + 1]) / 2);
    for (const i of overU) if ((countU.get(i) ?? 0) >= FIT.REFINE_MIN_POINTS) addU.push((kuD[i] + kuD[i + 1]) / 2);
    if (kvD.length - 1 + addV.length > FIT.MAX_SPANS) addV.length = 0;
    if (kuD.length - 1 + addU.length > FIT.MAX_SPANS) addU.length = 0;
    const cap = Math.min(FIT.DOF_RATIO * X.length, FIT.MAX_CP);
    const nuNew = tu.length - P_ - 1 + addU.length;
    if (wrap) {
      const grow = 2 * wrap <= FIT.MAX_SPANS && 2 * wrap * nuNew <= cap;
      if (!grow && (!addU.length || wrap * nuNew > cap)) break;
      if (grow) { wrap *= 2; tv = periodicKnots(period, wrap); }
    } else {
      if (!addU.length && !addV.length) break;
      if (nuNew * (tv.length - P_ - 1 + addV.length) > cap) break;
      tv = [...tv, ...addV].sort((a, b) => a - b);
    }
    tu = [...tu, ...addU].sort((a, b) => a - b);
  }
  // the best net by the hold-out, fitted to every vertex
  ({ tu, tv, wrap } = best);
  U = best.U; Vv = best.V;
  for (let it = 0; it < FIT.CORRECT_ITERS; it++) {
    const s2 = solveNet(tu, tv, U, Vv, X, lamAbs, wrap);
    if (!s2) break;
    S = s2;
    project(S);
  }
  let maxErr = 0, sq = 0;
  for (const e of err) { maxErr = Math.max(maxErr, e); sq += e * e; }
  Object.assign(info, { spans: [distinct(tu).length - 1, wrap || distinct(tv).length - 1], rounds: best.rnd,
    holdout: hold.reduce((s, h) => s + h, 0), max_err: maxErr / scale, rms_err: Math.sqrt(sq / err.length) / scale });
  return { params: wrap ? uPeriodic(S) : S.toParams(), info };
}

/** A v-wrapped net as Python's params: transposed so u goes round, flagged u_periodic (unclamped form). */
function uPeriodic(S) {
  const p = S.toParams();
  const poles = Array.from({ length: S.nv }, (_, j) => p.poles.map((row) => row[j]));
  return { poles, tu: p.tv, tv: p.tu, pu: p.pv, pv: p.pu, weights: null, u_periodic: true };
}

// ── splitting what cannot be parameterised ───────────────────────────────────
//
// One surface needs a group that maps to the plane: a disk that projects without folding, a disk whose
// conformal map does not fold, or a ring / tube. What is left (disks whose map folds because their
// boundary pinches or they wrap too far, higher genus) fitPieces cuts in two and fits each piece, again
// if need be: a small patch network only where one surface cannot cover the face. The cut is by
// normal-cone clustering (variational shape approximation, Cohen-Steiner, Alliez & Desbrun 2004, L2,1
// metric), so it falls where the normal field turns and each piece is near a height field over its
// proxy plane. ml/freeform_fit.py fit_pieces.

const VSA_ITERS = 8, SPLIT_LOPSIDED = 0.9;

const edgeKey = (u, v) => (u < v ? `${u}_${v}` : `${v}_${u}`);

/** Edge-connected components of the triangles `sel` (indices into T): arrays of indices into T. */
function triComponents(T, sel) {
  const par = new Map(sel.map((t) => [t, t]));
  const find = (x) => { while (par.get(x) !== x) { par.set(x, par.get(par.get(x))); x = par.get(x); } return x; };
  const owner = new Map();
  for (const t of sel) for (let k = 0; k < 3; k++) {
    const key = edgeKey(T[t][k], T[t][(k + 1) % 3]);
    if (owner.has(key)) par.set(find(t), find(owner.get(key))); else owner.set(key, t);
  }
  const comps = new Map();
  for (const t of sel) { const r = find(t); (comps.get(r) ?? comps.set(r, []).get(r)).push(t); }
  return [...comps.values()];
}

function euler(T, sel) {
  const vs = new Set(), es = new Set();
  for (const t of sel) for (let k = 0; k < 3; k++) { vs.add(T[t][k]); es.add(edgeKey(T[t][k], T[t][(k + 1) % 3])); }
  return vs.size - es.size + sel.length;
}

function heapQueue() {
  const h = [];
  return {
    get size() { return h.length; },
    push(e) { h.push(e); let i = h.length - 1; while (i > 0) { const p = (i - 1) >> 1; if (h[p][0] <= h[i][0]) break; [h[p], h[i]] = [h[i], h[p]]; i = p; } },
    pop() {
      const top = h[0], last = h.pop();
      if (h.length) {
        h[0] = last;
        for (let i = 0; ;) {
          const l = 2 * i + 1, r = l + 1;
          let m = i;
          if (l < h.length && h[l][0] < h[m][0]) m = l;
          if (r < h.length && h[r][0] < h[m][0]) m = r;
          if (m === i) break;
          [h[m], h[i]] = [h[i], h[m]]; i = m;
        }
      }
      return top;
    },
  };
}

/** Two-proxy normal-cone clustering of the triangles `sel` into edge-connected pieces (Python's _split). */
function splitNormals(V, T, sel) {
  const n = sel.length;
  const N = [], A = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const [a, b, c] = T[sel[i]];
    const nr = cross(sub(V[b], V[a]), sub(V[c], V[a]));
    A[i] = 0.5 * norm(nr) + 1e-300;
    N.push([nr[0] / (2 * A[i]), nr[1] / (2 * A[i]), nr[2] / (2 * A[i])]);
  }
  const owner = new Map();
  for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) {
    const key = edgeKey(T[sel[i]][k], T[sel[i]][(k + 1) % 3]);
    (owner.get(key) ?? owner.set(key, []).get(key)).push(i);
  }
  const nbrs = Array.from({ length: n }, () => []);
  for (const lst of owner.values()) for (const a of lst) for (const b of lst) if (b !== a) nbrs[a].push(b);
  const unit = (p) => { const l = norm(p); return l > 0 ? [p[0] / l, p[1] / l, p[2] / l] : null; };
  const m = [0, 0, 0];
  for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) m[k] += N[i][k] * A[i];
  const mu = unit(m) ?? [0, 0, 0];
  const argMax = (f) => { let k = 0, best = -Infinity; for (let i = 0; i < n; i++) { const v = f(i); if (v > best) { best = v; k = i; } } return k; };
  const s1 = argMax((i) => -dot(N[i], mu));
  const s2 = argMax((i) => -dot(N[i], N[s1]));
  if (s1 === s2) return [sel.slice()];
  // proxies by area-weighted 2-means on the normal sphere, started at the two most different normals;
  // the flood seeds are then each proxy's best-fitting triangle (seeding the flood AT the extreme
  // normals boxed one region into a small feature)
  const proxies = [N[s1].slice(), N[s2].slice()];
  for (let r = 0; r < 10; r++) {
    const acc = [[0, 0, 0], [0, 0, 0]];
    for (let i = 0; i < n; i++) {
      const k = dot(N[i], proxies[0]) >= dot(N[i], proxies[1]) ? 0 : 1;
      for (let c = 0; c < 3; c++) acc[k][c] += N[i][c] * A[i];
    }
    for (let k = 0; k < 2; k++) proxies[k] = unit(acc[k]) ?? proxies[k];
  }
  let seeds = [0, 1].map((k) => argMax((i) => dot(N[i], proxies[k])));
  if (seeds[0] === seeds[1]) return [sel.slice()];
  const d2 = (i, k) => (N[i][0] - proxies[k][0]) ** 2 + (N[i][1] - proxies[k][1]) ** 2 + (N[i][2] - proxies[k][2]) ** 2;
  let label = null;
  for (let it = 0; it < VSA_ITERS; it++) {
    label = new Int32Array(n).fill(-1);
    const q = heapQueue();
    seeds.forEach((s, k) => { label[s] = k; });
    seeds.forEach((s, k) => { for (const nb of nbrs[s]) q.push([A[nb] * d2(nb, k), nb, k]); });
    while (q.size) {
      const [, t, k] = q.pop();
      if (label[t] >= 0) continue;
      label[t] = k;
      for (const nb of nbrs[t]) if (label[nb] < 0) q.push([A[nb] * d2(nb, k), nb, k]);
    }
    const next = [];
    for (let k = 0; k < 2; k++) {
      const p = [0, 0, 0];
      for (let i = 0; i < n; i++) if (label[i] === k) for (let c = 0; c < 3; c++) p[c] += N[i][c] * A[i];
      const lp = Math.max(norm(p), 1e-300);
      proxies[k] = [p[0] / lp, p[1] / lp, p[2] / lp];
      let bi = seeds[k], bd = Infinity;
      for (let i = 0; i < n; i++) if (label[i] === k && d2(i, k) < bd) { bd = d2(i, k); bi = i; }
      next.push(bi);
    }
    if (next[0] === seeds[0] && next[1] === seeds[1]) break;
    seeds = next;
  }
  const pieces = [];
  for (const k of [0, 1, -1]) {                         // -1: triangles no seed reached (disconnected)
    const part = [];
    for (let i = 0; i < n; i++) if (label[i] === k) part.push(sel[i]);
    if (part.length) pieces.push(...triComponents(T, part));
  }
  if (Math.max(...pieces.map((p) => p.length)) >= SPLIT_LOPSIDED * n) return splitPlane(V, T, sel);
  return pieces;
}

/** The fallback cut when the normals do not separate: a plane through the area-weighted centroid, across the first principal direction whose pieces are all disks. */
function splitPlane(V, T, sel) {
  const n = sel.length;
  const C = [], A = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const [a, b, c] = T[sel[i]];
    C.push([0, 1, 2].map((k) => (V[a][k] + V[b][k] + V[c][k]) / 3));
    A[i] = 0.5 * norm(cross(sub(V[b], V[a]), sub(V[c], V[a]))) + 1e-300;
  }
  const At = A.reduce((s, x) => s + x, 0);
  const c0 = [0, 1, 2].map((k) => C.reduce((s, q, i) => s + q[k] * A[i], 0) / At);
  const cov = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < n; i++) {
    const d = sub(C[i], c0);
    for (let r = 0; r < 3; r++) for (let s = 0; s < 3; s++) cov[r][s] += (A[i] * d[r] * d[s]) / At;
  }
  let best = null;
  for (const d of principalAxes(cov)) {
    const s = C.map((q) => dot(sub(q, c0), d));
    const sorted = [...s].sort((a, b) => a - b);
    const med = n % 2 ? sorted[(n - 1) / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
    const side = s.map((x) => x >= med);
    if (side.every(Boolean) || !side.some(Boolean)) continue;
    const pieces = [];
    for (const want of [true, false]) pieces.push(...triComponents(T, sel.filter((_, i) => side[i] === want)));
    if (pieces.every((p) => euler(T, p) === 1)) return pieces;
    if (!best) best = pieces;
  }
  return best ?? [sel.slice()];
}

/** Eigenvectors of a symmetric 3x3 matrix, largest eigenvalue first (cyclic Jacobi). */
function principalAxes(M) {
  const a = M.map((r) => r.slice());
  const v = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let sweep = 0; sweep < 50; sweep++) {
    if (a[0][1] ** 2 + a[0][2] ** 2 + a[1][2] ** 2 < 1e-30 * (a[0][0] ** 2 + a[1][1] ** 2 + a[2][2] ** 2 + 1e-300)) break;
    for (let p = 0; p < 3; p++) for (let q = p + 1; q < 3; q++) {
      if (Math.abs(a[p][q]) < 1e-300) continue;
      const th = (a[q][q] - a[p][p]) / (2 * a[p][q]);
      const t = (th >= 0 ? 1 : -1) / (Math.abs(th) + Math.sqrt(th * th + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < 3; k++) { const x = a[k][p], y = a[k][q]; a[k][p] = c * x - s * y; a[k][q] = s * x + c * y; }
      for (let k = 0; k < 3; k++) { const x = a[p][k], y = a[q][k]; a[p][k] = c * x - s * y; a[q][k] = s * x + c * y; }
      for (let k = 0; k < 3; k++) { const x = v[k][p], y = v[k][q]; v[k][p] = c * x - s * y; v[k][q] = s * x + c * y; }
    }
  }
  return [0, 1, 2].sort((i, j) => a[j][j] - a[i][i]).map((i) => [v[0][i], v[1][i], v[2][i]]);
}

/**
 * fitGroup, splitting the group where it cannot be parameterised (above). V, T, options as fitGroup's.
 * Returns {pieces: [{tris (indices into T), params, info}], failed: [{tris, info}]}: the pieces cover T
 * less any piece still failing at maxDepth.
 */
export function fitPieces(V, T, { maxDepth = 4, ...kw } = {}) {
  const pieces = [], failed = [];
  const todo = [[T.map((_, i) => i), 0]];
  while (todo.length) {
    const [tri, depth] = todo.pop();
    const local = new Map(), Vl = [];
    const Tl = tri.map((t) => T[t].map((q) => { if (!local.has(q)) { local.set(q, Vl.length); Vl.push(V[q]); } return local.get(q); }));
    const { params, info } = fitGroup(Vl, Tl, kw);
    if (params) { info.depth = depth; pieces.push({ tris: tri, params, info }); }
    else if (info.fail === 'param' && depth < maxDepth && tri.length >= 2) {
      for (const p of splitNormals(V, T, tri)) {
        if (p.length < tri.length) todo.push([p, depth + 1]);
        else failed.push({ tris: tri, info });
      }
    } else failed.push({ tris: tri, info });
  }
  return { pieces, failed };
}
