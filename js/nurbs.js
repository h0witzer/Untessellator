/**
 * nurbs.js — polynomial tensor-product B-spline surfaces: basis functions and their derivatives,
 * evaluation, closest-point projection. A port of ml/nurbs.py (the parts the freeform fitter and the
 * knitter need; rational surfaces are not fitted, so not supported here).
 *
 * Knot vectors are FLAT (each knot repeated by its multiplicity), length n + p + 1 for n control points
 * of degree p; the domain of a direction is [t[p], t[n]].
 */

/** Knot span s (t[s] <= x < t[s+1], non-empty), x clamped to the domain. */
export function spanOf(t, p, x) {
  const n = t.length - p - 1;
  if (x < t[p]) x = t[p];
  if (x > t[n]) x = t[n];
  let lo = p, hi = n;                     // binary search for the last s with t[s] <= x, s < n
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (t[mid] <= x) lo = mid; else hi = mid;
  }
  let s = lo;
  while (s > p && t[s + 1] <= t[s]) s--;
  return s;
}

/** The p + 1 nonzero basis functions at x and their derivatives up to nder (NURBS Book A2.3). */
export function nzBasis(t, p, x, nder = 0) {
  const n = t.length - p - 1;
  if (x < t[p]) x = t[p];
  if (x > t[n]) x = t[n];
  const s = spanOf(t, p, x);
  const ndu = Array.from({ length: p + 1 }, () => new Float64Array(p + 1));
  const left = new Float64Array(p + 1), right = new Float64Array(p + 1);
  ndu[0][0] = 1;
  for (let j = 1; j <= p; j++) {
    left[j] = x - t[s + 1 - j];
    right[j] = t[s + j] - x;
    let saved = 0;
    for (let r = 0; r < j; r++) {
      ndu[j][r] = right[r + 1] + left[j - r];
      const tmp = ndu[r][j - 1] / ndu[j][r];
      ndu[r][j] = saved + right[r + 1] * tmp;
      saved = left[j - r] * tmp;
    }
    ndu[j][j] = saved;
  }
  const ders = Array.from({ length: nder + 1 }, () => new Float64Array(p + 1));
  for (let j = 0; j <= p; j++) ders[0][j] = ndu[j][p];
  const a = [new Float64Array(p + 1), new Float64Array(p + 1)];
  for (let r = 0; r <= p; r++) {
    let s1 = 0, s2 = 1;
    a[0].fill(0); a[1].fill(0);
    a[0][0] = 1;
    for (let k = 1; k <= nder; k++) {
      let d = 0;
      const rk = r - k, pk = p - k;
      if (r >= k) { a[s2][0] = a[s1][0] / ndu[pk + 1][rk]; d = a[s2][0] * ndu[rk][pk]; }
      const j1 = rk >= -1 ? 1 : -rk;
      const j2 = r - 1 <= pk ? k - 1 : p - r;
      for (let j = j1; j <= j2; j++) { a[s2][j] = (a[s1][j] - a[s1][j - 1]) / ndu[pk + 1][rk + j]; d += a[s2][j] * ndu[rk + j][pk]; }
      if (r <= pk) { a[s2][k] = -a[s1][k - 1] / ndu[pk + 1][r]; d += a[s2][k] * ndu[r][pk]; }
      ders[k][r] = d;
      [s1, s2] = [s2, s1];
    }
  }
  let f = p;
  for (let k = 1; k <= nder; k++) { for (let j = 0; j <= p; j++) ders[k][j] *= f; f *= p - k; }
  return { first: s - p, ders };
}

const GRID_PER_SIZE = 150;
const GRID_MAX = 250000;

export class Surface {
  /**
   * @param P   Float64Array, nu * nv * 3 poles, row-major (pole (i, j) at (i * nv + j) * 3)
   * @param tu  flat knots in u (length nu + pu + 1), tv likewise
   */
  constructor(P, nu, nv, tu, tv, pu = 3, pv = 3) {
    Object.assign(this, { P, nu, nv, tu: Float64Array.from(tu), tv: Float64Array.from(tv), pu, pv });
    if (tu.length !== nu + pu + 1 || tv.length !== nv + pv + 1) throw new Error('knots / poles mismatch');
    this.dom = [this.tu[pu], this.tu[nu], this.tv[pv], this.tv[nv]];
    this._grid = null;
    this._size = null;
    this._closed = null;
    this._last = null;
  }

  static fromParams(p) {
    const nu = p.poles.length, nv = p.poles[0].length;
    const P = new Float64Array(nu * nv * 3);
    for (let i = 0; i < nu; i++) for (let j = 0; j < nv; j++) for (let c = 0; c < 3; c++) P[(i * nv + j) * 3 + c] = p.poles[i][j][c];
    return new Surface(P, nu, nv, p.tu, p.tv, p.pu ?? 3, p.pv ?? 3);
  }

  toParams() {
    const poles = [];
    for (let i = 0; i < this.nu; i++) {
      const row = [];
      for (let j = 0; j < this.nv; j++) { const o = (i * this.nv + j) * 3; row.push([this.P[o], this.P[o + 1], this.P[o + 2]]); }
      poles.push(row);
    }
    return { poles, tu: [...this.tu], tv: [...this.tv], pu: this.pu, pv: this.pv, weights: null };
  }

  /** {S, Su?, Sv?, Suu?, Suv?, Svv?} at (u, v), each [x, y, z]. */
  eval(u, v, nder = 0) {
    const bu = nzBasis(this.tu, this.pu, u, nder), bv = nzBasis(this.tv, this.pv, v, nder);
    const { P, nv, pu, pv } = this;
    const comb = (a, b) => {
      const out = [0, 0, 0];
      const Bu = bu.ders[a], Bv = bv.ders[b];
      for (let i = 0; i <= pu; i++) {
        const row = (bu.first + i) * nv;
        for (let j = 0; j <= pv; j++) {
          const w = Bu[i] * Bv[j];
          if (!w) continue;
          const o = (row + bv.first + j) * 3;
          out[0] += w * P[o]; out[1] += w * P[o + 1]; out[2] += w * P[o + 2];
        }
      }
      return out;
    };
    const out = { S: comb(0, 0) };
    if (nder >= 1) { out.Su = comb(1, 0); out.Sv = comb(0, 1); }
    if (nder >= 2) { out.Suu = comb(2, 0); out.Suv = comb(1, 1); out.Svv = comb(0, 2); }
    return out;
  }

  /** Point and unit natural normal (Su x Sv) at (u, v). */
  normal(u, v) {
    const d = this.eval(u, v, 1);
    const n = [d.Su[1] * d.Sv[2] - d.Su[2] * d.Sv[1], d.Su[2] * d.Sv[0] - d.Su[0] * d.Sv[2], d.Su[0] * d.Sv[1] - d.Su[1] * d.Sv[0]];
    const l = Math.hypot(n[0], n[1], n[2]) || 1e-300;
    return [d.S, [n[0] / l, n[1] / l, n[2] / l]];
  }

  size() {
    if (this._size === null) {
      const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
      for (let i = 0; i < this.P.length; i += 3) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], this.P[i + k]); hi[k] = Math.max(hi[k], this.P[i + k]); }
      this._size = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) || 1;
    }
    return this._size;
  }

  /** (closed in u, closed in v), judged geometrically. */
  closed() {
    if (this._closed === null) {
      const [u0, u1, v0, v1] = this.dom;
      const tol = 1e-9 * this.size();
      let cu = true, cv = true;
      for (let k = 0; k < 7; k++) {
        const s = k / 6;
        const a = this.eval(u0, v0 + s * (v1 - v0)).S, b = this.eval(u1, v0 + s * (v1 - v0)).S;
        const c = this.eval(u0 + s * (u1 - u0), v0).S, d = this.eval(u0 + s * (u1 - u0), v1).S;
        if (Math.max(...[0, 1, 2].map((i) => Math.abs(a[i] - b[i]))) >= tol) cu = false;
        if (Math.max(...[0, 1, 2].map((i) => Math.abs(c[i] - d[i]))) >= tol) cv = false;
      }
      this._closed = [cu, cv];
    }
    return this._closed;
  }

  /** Dense samples for projection starts, spaced by control-polygon length (nurbs.py Surface.grid). */
  grid() {
    if (!this._grid) {
      const [u0, u1, v0, v1] = this.dom;
      const { P, nu, nv } = this;
      const h = this.size() / GRID_PER_SIZE;
      let Lu = 0, Lv = 0;
      for (let j = 0; j < nv; j++) { let l = 0; for (let i = 1; i < nu; i++) { const a = ((i - 1) * nv + j) * 3, b = (i * nv + j) * 3; l += Math.hypot(P[b] - P[a], P[b + 1] - P[a + 1], P[b + 2] - P[a + 2]); } Lu = Math.max(Lu, l); }
      for (let i = 0; i < nu; i++) { let l = 0; for (let j = 1; j < nv; j++) { const a = (i * nv + j - 1) * 3, b = (i * nv + j) * 3; l += Math.hypot(P[b] - P[a], P[b + 1] - P[a + 1], P[b + 2] - P[a + 2]); } Lv = Math.max(Lv, l); }
      const spans = (t) => new Set(t).size - 1;
      let gu = Math.min(1500, Math.max(24, 8 * spans(this.tu), Math.ceil(Lu / h)));
      let gv = Math.min(1500, Math.max(24, 8 * spans(this.tv), Math.ceil(Lv / h)));
      while (gu * gv > GRID_MAX) { gu = Math.max(24, Math.floor(gu * 0.8)); gv = Math.max(24, Math.floor(gv * 0.8)); }
      const U = new Float64Array(gu * gv), V = new Float64Array(gu * gv), G = new Float64Array(gu * gv * 3);
      for (let i = 0; i < gu; i++) for (let j = 0; j < gv; j++) {
        const k = i * gv + j;
        U[k] = u0 + ((u1 - u0) * i) / (gu - 1);
        V[k] = v0 + ((v1 - v0) * j) / (gv - 1);
        const S = this.eval(U[k], V[k]).S;
        G[k * 3] = S[0]; G[k * 3 + 1] = S[1]; G[k * 3 + 2] = S[2];
      }
      this._grid = { U, V, G };
    }
    return this._grid;
  }

  /**
   * Closest point on the surface (within its domain) to x: {u, v, S}. Starts at uv0 when given, else
   * where the previous query ended when x is near it, else at the nearest grid sample; then Newton on
   * |S - x|^2 with Gauss-Newton where the Hessian is not positive definite, with backtracking.
   */
  closest(x, uv0 = null, iters = 25) {
    const [u0, u1, v0, v1] = this.dom;
    let u, v;
    if (uv0) [u, v] = uv0;
    else if (this._last && Math.hypot(x[0] - this._last[0][0], x[1] - this._last[0][1], x[2] - this._last[0][2]) < 0.02 * this.size()) {
      [, u, v] = this._last;
    } else {
      const { U, V, G } = this.grid();
      let best = Infinity, k0 = 0;
      for (let k = 0; k < U.length; k++) {
        const dx = G[k * 3] - x[0], dy = G[k * 3 + 1] - x[1], dz = G[k * 3 + 2] - x[2];
        const d = dx * dx + dy * dy + dz * dz;
        if (d < best) { best = d; k0 = k; }
      }
      u = U[k0]; v = V[k0];
    }
    const [cu, cv] = this.closed();
    const fold = (a, lo, hi, c) => (c ? lo + (((a - lo) % (hi - lo)) + (hi - lo)) % (hi - lo) : Math.min(hi, Math.max(lo, a)));
    const dist2 = (S) => (S[0] - x[0]) ** 2 + (S[1] - x[1]) ** 2 + (S[2] - x[2]) ** 2;
    let d = this.eval(u, v, 2);
    let cur = dist2(d.S);
    for (let it = 0; it < iters; it++) {
      const r = [d.S[0] - x[0], d.S[1] - x[1], d.S[2] - x[2]];
      const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
      const gu = dot(r, d.Su), gv = dot(r, d.Sv);
      const a = dot(d.Su, d.Su), b = dot(d.Su, d.Sv), c = dot(d.Sv, d.Sv);
      let h11 = a + dot(r, d.Suu), h12 = b + dot(r, d.Suv), h22 = c + dot(r, d.Svv);
      let det = h11 * h22 - h12 * h12;
      if (!(h11 > 0 && det > 1e-12 * Math.max(a * c, 1e-300))) { h11 = a; h12 = b; h22 = c; det = h11 * h22 - h12 * h12; }
      if (Math.abs(det) <= 1e-300) det = 1e-300;
      let du = -(h22 * gu - h12 * gv) / det, dv = -(h11 * gv - h12 * gu) / det;
      du = Math.max(-(u1 - u0) / 2, Math.min((u1 - u0) / 2, du));
      dv = Math.max(-(v1 - v0) / 2, Math.min((v1 - v0) / 2, dv));
      let h = 1, nu = fold(u + du, u0, u1, cu), nv = fold(v + dv, v0, v1, cv);
      let nd = this.eval(nu, nv, 2), next = dist2(nd.S);
      while (next > cur * (1 + 1e-12) + 1e-300 && h > 1 / 16) {
        h /= 2;
        nu = fold(u + h * du, u0, u1, cu); nv = fold(v + h * dv, v0, v1, cv);
        nd = this.eval(nu, nv, 2); next = dist2(nd.S);
      }
      if (next > cur * (1 + 1e-12) + 1e-300) break;
      const step = Math.abs(nu - u) / Math.max(u1 - u0, 1e-300) + Math.abs(nv - v) / Math.max(v1 - v0, 1e-300);
      u = nu; v = nv; d = nd; cur = next;
      if (step < 1e-12) break;
    }
    if (!uv0) this._last = [[x[0], x[1], x[2]], u, v];
    return { u, v, S: d.S };
  }

  /** Signed distance of x along the natural normal at its closest point, and that normal (distNormal's contract). */
  signedDistance(x) {
    const { u, v } = this.closest(x);
    const [S, n] = this.normal(u, v);
    return [(x[0] - S[0]) * n[0] + (x[1] - S[1]) * n[1] + (x[2] - S[2]) * n[2], n];
  }
}

const cache = new WeakMap();
/** The Surface for a params object, built once. */
export function surfaceOf(p) {
  if (!cache.has(p)) cache.set(p, Surface.fromParams(p));
  return cache.get(p);
}
