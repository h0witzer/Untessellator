/**
 * knit/geometry.js — the knitter's surface math, no OCCT: distances to the fitted surfaces, the
 * nearest point on several surfaces at once, chain unfolding, and the cubic B-spline through a chain.
 *
 * A port of ml/solid_rebuild.py (dist_normal, project_onto, seam_away, unfold_chain), which is where
 * every rule here was measured. Surfaces are gaussMap's param format: plane {origin, normal},
 * cylinder {axisPoint, axis, radius}, cone {apex, axis, halfAngle}, sphere {center, radius},
 * torus {center, axis, majorRadius, minorRadius}.
 */

import { surfaceOf } from '../nurbs.js';

export const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
export const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const norm = (a) => Math.hypot(a[0], a[1], a[2]);
export function unit(a) {
  const n = norm(a);
  return n > 0 ? [a[0] / n, a[1] / n, a[2] / n] : [a[0], a[1], a[2]];
}

/**
 * Signed distance and unit gradient of point x to a surface. Cones are double cones. A bspline's
 * (js/freeformFit.js params) is the distance along its natural normal at the closest point.
 */
export function distNormal(kind, p, x) {
  if (kind === 'bspline') return surfaceOf(p).signedDistance(x);
  if (kind === 'plane') {
    const n = unit(p.normal);
    return [dot(sub(x, p.origin), n), n];
  }
  if (kind === 'sphere') {
    const v = sub(x, p.center);
    const r = norm(v);
    return [r - p.radius, scale(v, 1 / Math.max(r, 1e-300))];
  }
  const a = unit(p.axis);
  const o = kind === 'cylinder' ? p.axisPoint : kind === 'cone' ? p.apex : p.center;
  const v = sub(x, o);
  const z = dot(v, a);
  const rad = sub(v, scale(a, z));
  const rho = norm(rad);
  const u = scale(rad, 1 / Math.max(rho, 1e-300));
  if (kind === 'cylinder') return [rho - p.radius, u];
  if (kind === 'cone') {
    const ca = Math.cos(p.halfAngle), sa = Math.sin(p.halfAngle);
    const s = z >= 0 ? 1 : -1;
    return [rho * ca - Math.abs(z) * sa, sub(scale(u, ca), scale(a, s * sa))];
  }
  const w = sub(v, scale(u, p.majorRadius));
  const wl = norm(w);
  return [wl - p.minorRadius, scale(w, 1 / Math.max(wl, 1e-300))];
}

/** Solve the small dense system A x = b (Gaussian elimination, partial pivoting). */
function solveSmall(A, b) {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    [M[c], M[piv]] = [M[piv], M[c]];
    const d = M[c][c];
    if (d === 0) continue;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = M[r][c] / d;
      if (f) for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  return M.map((row, i) => (row[i] ? row[n] / row[i] : 0));
}

/**
 * Nearest point to x lying on every surface in `surfs` ([kind, params] pairs): Gauss-Newton with
 * min-norm steps. Tangent surfaces make the normals nearly parallel; the damped step then moves x onto
 * the shared tangency instead of blowing up.
 */
export function projectOnto(surfs, x0, scaleLen, iters = 30) {
  let x = [x0[0], x0[1], x0[2]];
  const m = surfs.length;
  for (let it = 0; it < iters; it++) {
    const dn = surfs.map(([k, p]) => distNormal(k, p, x));
    if (Math.max(...dn.map(([d]) => Math.abs(d))) < 1e-12 * scaleLen) break;
    const J = dn.map(([, n]) => n);
    const JJt = J.map((ri, i) => J.map((rj, j) => dot(ri, rj) + (i === j ? 1e-10 : 0)));
    const y = solveSmall(JJt, dn.map(([d]) => d));
    const step = [0, 0, 0];
    for (let i = 0; i < m; i++) for (let k = 0; k < 3; k++) step[k] += J[i][k] * y[i];
    x = sub(x, step);
    if (norm(step) < 1e-13 * scaleLen) break;
  }
  return x;
}

/**
 * An X direction perpendicular to `axis` pointing AWAY from the patch, so the periodic seam (u = 0)
 * sits on the far side of the surface from the face.
 */
export function seamAway(axis, meanDir) {
  const a = unit(axis);
  let m = sub(meanDir, scale(a, dot(meanDir, a)));
  if (norm(m) < 1e-9) {
    const ref = Math.abs(a[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
    m = sub(ref, scale(a, dot(ref, a)));
  }
  return scale(unit(m), -1);
}

/**
 * Chain points that double back are dropped before an edge is interpolated through them: when an
 * automatic segment's boundary runs off the true edge, its vertices project onto the two surfaces'
 * intersection out of order and the spline folds (BRepCheck's SelfIntersectingWire). Interior points
 * where the polyline turns by more than `maxTurn` degrees go, worst first; an open chain's ends are
 * corners and stay, and so does a closed chain's first point (the corner its edge starts and ends at).
 */
export const UNFOLD_MAX_TURN = 135;

export function unfoldChain(pts, closed, info, maxTurn = UNFOLD_MAX_TURN) {
  pts = pts.slice();
  let removed = 0;
  while (pts.length > (closed ? 3 : 2)) {
    const n = pts.length;
    let worst = maxTurn, wi = -1;
    for (let i = 1; i < (closed ? n : n - 1); i++) {
      const u = sub(pts[i], pts[i - 1]), v = sub(pts[(i + 1) % n], pts[i]);
      const nu = norm(u), nv = norm(v);
      if (nu === 0 || nv === 0) continue;
      const t = (Math.acos(Math.max(-1, Math.min(1, dot(u, v) / (nu * nv)))) * 180) / Math.PI;
      if (t > worst) { worst = t; wi = i; }
    }
    if (wi < 0) break;
    pts.splice(wi, 1);
    removed++;
  }
  if (removed) {
    info.unfold_points_removed = (info.unfold_points_removed ?? 0) + removed;
    info.unfolded_chains = (info.unfolded_chains ?? 0) + 1;
  }
  return pts;
}

/**
 * Derivative at `at` of the Lagrange polynomial through (ts[i], Q[i]): OCCT's BuildTangents /
 * BuildPeriodicTangent end tangents (PLib::EvalLagrange).
 */
function lagrangeDerivative(ts, Q, at) {
  const out = [0, 0, 0];
  const n = ts.length;
  for (let j = 0; j < n; j++) {
    let denom = 1;
    for (let m = 0; m < n; m++) if (m !== j) denom *= ts[j] - ts[m];
    let s = 0;
    for (let k = 0; k < n; k++) {
      if (k === j) continue;
      let prod = 1;
      for (let m = 0; m < n; m++) if (m !== j && m !== k) prod *= at - ts[m];
      s += prod;
    }
    for (let c = 0; c < 3; c++) out[c] += (Q[j][c] * s) / denom;
  }
  return out;
}

/**
 * The B-spline through `pts` in order, exactly as GeomAPI_Interpolate builds it (whose point-array
 * argument has no binding in opencascade.js). The scheme matters: averaged knots instead turned a valid
 * torus face of 00032949_19 invalid in the Python knitter too. Measured against OCCT 7.9's own curves to
 * 1e-15 on random chains:
 *   open, 4+ points   cubic, knots at the chord-length parameters (end multiplicity 4), through every
 *                     point, end tangents = derivative of the cubic Lagrange polynomial through the first
 *                     / last 4 points
 *   open, 3 / 2       the quadratic / line through them
 *   closed, 3+        periodic in OCCT (C1 at the seam, tangent from the first 4 points, 3 when only 3);
 *                     here the same curve as a clamped one over the closing parameter range: through the
 *                     points, back to the first, with that tangent at both ends
 * Returns {poles, knots, mults, degree}; the parameter runs from 0 to the chain's length.
 */
export function interpolateChain(pts, closed) {
  const n = pts.length;
  const chord = (Q) => {
    const t = [0];
    for (let i = 1; i < Q.length; i++) t.push(t[i - 1] + norm(sub(Q[i], Q[i - 1])));
    return t;
  };
  if (!closed && n === 2) return { poles: [pts[0], pts[1]], knots: [0, norm(sub(pts[1], pts[0]))], mults: [2, 2], degree: 1 };
  if (!closed && n === 3) {
    const t = chord(pts);
    const U = [t[0], t[0], t[0], t[2], t[2], t[2]];
    const A = t.map((u) => basisFuns(findSpan(2, 2, u, U), u, 2, U));    // a quadratic Bezier: 3 x 3
    const sol = [0, 1, 2].map((c) => solveSmall(A.map((r) => [...r]), pts.map((q) => q[c])));
    return { poles: [0, 1, 2].map((i) => [sol[0][i], sol[1][i], sol[2][i]]), knots: [t[0], t[2]], mults: [3, 3], degree: 2 };
  }
  const Q = closed ? [...pts, pts[0]] : pts;
  const t = chord(Q);
  const m = Q.length;
  const k0 = Math.min(4, n);
  const T0 = lagrangeDerivative(t.slice(0, k0), pts.slice(0, k0), t[0]);
  const T1 = closed ? T0 : lagrangeDerivative(t.slice(m - 4), Q.slice(m - 4), t[m - 1]);
  // clamped cubic, knots at the parameters: m + 2 poles
  const U = [t[0], t[0], t[0], t[0], ...t.slice(1, m - 1), t[m - 1], t[m - 1], t[m - 1], t[m - 1]];
  const np = m + 2;
  const rows = [], rhs = [];
  const row = (u, order) => {
    const span = findSpan(np - 1, 3, u, U);
    const ders = dersBasisFuns(span, u, 3, order, U)[order];
    const r = new Float64Array(np);
    for (let j = 0; j <= 3; j++) r[span - 3 + j] = ders[j];
    return r;
  };
  rows.push(row(t[0], 0)); rhs.push(Q[0]);
  rows.push(row(t[0], 1)); rhs.push(T0);
  for (let k = 1; k < m - 1; k++) { rows.push(row(t[k], 0)); rhs.push(Q[k]); }
  rows.push(row(t[m - 1], 1)); rhs.push(T1);
  rows.push(row(t[m - 1], 0)); rhs.push(Q[m - 1]);
  const P = solveBanded(rows, rhs, 3);
  P[0] = [...Q[0]];
  P[np - 1] = [...Q[m - 1]];
  const knots = [t[0], ...t.slice(1, m - 1), t[m - 1]];
  const mults = [4, ...t.slice(1, m - 1).map(() => 1), 4];
  return { poles: P, knots, mults, degree: 3 };
}

/** Banded A x = b (rows of length n, nonzeros within `bw` of the diagonal); partial pivoting in the band. */
function solveBanded(rows, rhs, bw) {
  const n = rows.length;
  const A = rows.map((r) => Float64Array.from(r));
  const B = rhs.map((v) => [v[0], v[1], v[2]]);
  for (let c = 0; c < n; c++) {
    const hi = Math.min(n, c + 2 * bw + 2);
    let piv = c;
    for (let r = c + 1; r < hi; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r;
    if (piv !== c) { [A[c], A[piv]] = [A[piv], A[c]]; [B[c], B[piv]] = [B[piv], B[c]]; }
    const d = A[c][c];
    if (!d) continue;
    for (let r = c + 1; r < hi; r++) {
      const f = A[r][c] / d;
      if (!f) continue;
      for (let k = c; k < Math.min(n, c + 3 * bw + 3); k++) A[r][k] -= f * A[c][k];
      for (let k = 0; k < 3; k++) B[r][k] -= f * B[c][k];
    }
  }
  for (let r = n - 1; r >= 0; r--) {
    for (let k = r + 1; k < Math.min(n, r + 3 * bw + 3); k++) {
      if (A[r][k]) for (let m = 0; m < 3; m++) B[r][m] -= A[r][k] * B[k][m];
    }
    for (let m = 0; m < 3; m++) B[r][m] /= A[r][r];
  }
  return B;
}

/** Basis functions and their derivatives up to `nd` at u (Piegl & Tiller A2.3). */
function dersBasisFuns(i, u, p, nd, U) {
  const ndu = Array.from({ length: p + 1 }, () => new Float64Array(p + 1));
  const left = new Float64Array(p + 1), right = new Float64Array(p + 1);
  ndu[0][0] = 1;
  for (let j = 1; j <= p; j++) {
    left[j] = u - U[i + 1 - j];
    right[j] = U[i + j] - u;
    let saved = 0;
    for (let r = 0; r < j; r++) {
      ndu[j][r] = right[r + 1] + left[j - r];
      const tmp = ndu[r][j - 1] / ndu[j][r];
      ndu[r][j] = saved + right[r + 1] * tmp;
      saved = left[j - r] * tmp;
    }
    ndu[j][j] = saved;
  }
  const ders = Array.from({ length: nd + 1 }, () => new Float64Array(p + 1));
  for (let j = 0; j <= p; j++) ders[0][j] = ndu[j][p];
  const a = [new Float64Array(p + 1), new Float64Array(p + 1)];
  for (let r = 0; r <= p; r++) {
    let s1 = 0, s2 = 1;
    a[0][0] = 1;
    for (let k = 1; k <= nd; k++) {
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
  for (let k = 1; k <= nd; k++) { for (let j = 0; j <= p; j++) ders[k][j] *= f; f *= p - k; }
  return ders;
}

/** A B-spline's point at u (de Boor), for tests. */
export function evalBSpline({ poles, knots, mults, degree }, u) {
  const U = [];
  knots.forEach((k, i) => { for (let j = 0; j < mults[i]; j++) U.push(k); });
  const span = findSpan(poles.length - 1, degree, u, U);
  const N = basisFuns(span, u, degree, U);
  const out = [0, 0, 0];
  for (let j = 0; j <= degree; j++) for (let c = 0; c < 3; c++) out[c] += N[j] * poles[span - degree + j][c];
  return out;
}

function findSpan(nn, p, u, U) {
  if (u >= U[nn + 1]) return nn;
  if (u <= U[p]) return p;
  let low = p, high = nn + 1, mid = (low + high) >> 1;
  while (u < U[mid] || u >= U[mid + 1]) {
    if (u < U[mid]) high = mid; else low = mid;
    mid = (low + high) >> 1;
  }
  return mid;
}

function basisFuns(i, u, p, U) {
  const N = [1], left = [0], right = [0];
  for (let j = 1; j <= p; j++) {
    left[j] = u - U[i + 1 - j];
    right[j] = U[i + j] - u;
    let saved = 0;
    for (let r = 0; r < j; r++) {
      const tmp = N[r] / (right[r + 1] + left[j - r]);
      N[r] = saved + right[r + 1] * tmp;
      saved = left[j - r] * tmp;
    }
    N[j] = saved;
  }
  return N;
}


// ── exact edge curves ────────────────────────────────────────────────────────────────────────────────
//
// Where two surfaces are exactly tangent (knit/axes.js makes them so), the edge between their faces is
// their contact line or circle - but projecting points onto both surfaces there is degenerate (parallel
// normals), and the projected edge drifted 0.01-0.3 degrees round a cylinder off the line where it
// touches its plane. So a tangent pair's edge points go onto the analytic contact locus:
//   coaxial surfaces of revolution   tangent in their (rho, z) profile: cylinder, cone and a plane square
//                                     to the axis are lines there, torus and sphere circles; the contact
//                                     point turned about the axis is the contact circle
//   plane along a cylinder's axis     the line where the plane touches it
//   plane tangent to a cone           the ruling through the apex
//   parallel cylinders                the line where they touch

const TANGENT_TOL = 1e-7;                 // x scale: tangency the shared-axes step makes exact

const lineDist = (o, d, x) => { const v = sub(x, o); return norm(sub(v, scale(d, dot(v, d)))); };

/** A surface of revolution about (o, d) as its profile curve in (rho, z): {line: [p, q] point, direction} or {circle}. */
function profileOf(kind, p, o, d) {
  const zOf = (x) => dot(sub(x, o), d);
  if (kind === 'cylinder') return { line: [[p.radius, 0], [0, 1]] };
  if (kind === 'plane') return { line: [[0, zOf(p.origin)], [1, 0]] };
  if (kind === 'cone') {
    const za = zOf(p.apex);
    return { lines: [[[0, za], [Math.sin(p.halfAngle), Math.cos(p.halfAngle)]], [[0, za], [Math.sin(p.halfAngle), -Math.cos(p.halfAngle)]]] };
  }
  if (kind === 'torus') return { circle: [[p.majorRadius, zOf(p.center)], p.minorRadius] };
  if (kind === 'sphere') return { circle: [[0, zOf(p.center)], p.radius] };
  return null;
}

/** Is (kind, p) a surface of revolution about the line (o, d)? */
function revolvedAbout(kind, p, o, d, tol) {
  if (kind === 'plane') return Math.abs(Math.abs(dot(unit(p.normal), d)) - 1) <= 1e-9;
  if (kind === 'sphere') return lineDist(o, d, p.center) <= tol;
  if (kind === 'cylinder' || kind === 'cone' || kind === 'torus') {
    const q = kind === 'cylinder' ? p.axisPoint : kind === 'cone' ? p.apex : p.center;
    return Math.abs(Math.abs(dot(unit(p.axis), d)) - 1) <= 1e-9 && lineDist(o, d, q) <= tol;
  }
  return false;
}

/** The contact point of two tangent profile curves, or null. */
function profileContact(A, B, tol) {
  const foot = ([p, u], c) => { const t = (c[0] - p[0]) * u[0] + (c[1] - p[1]) * u[1]; return [p[0] + t * u[0], p[1] + t * u[1]]; };
  if (A.line && B.circle || A.circle && B.line) {
    const L = A.line ?? B.line, [c, r] = A.circle ?? B.circle;
    const f = foot(L, c);
    return Math.abs(Math.hypot(f[0] - c[0], f[1] - c[1]) - r) <= tol ? f : null;
  }
  if (A.circle && B.circle) {
    const [c1, r1] = A.circle, [c2, r2] = B.circle, dd = Math.hypot(c2[0] - c1[0], c2[1] - c1[1]);
    if (!(dd > 0)) return null;
    const u = [(c2[0] - c1[0]) / dd, (c2[1] - c1[1]) / dd];
    if (Math.abs(dd - (r1 + r2)) <= tol) return [c1[0] + r1 * u[0], c1[1] + r1 * u[1]];
    if (Math.abs(dd - Math.abs(r1 - r2)) <= tol) { const s = r1 > r2 ? 1 : -1; return [c1[0] + s * r1 * u[0], c1[1] + s * r1 * u[1]]; }
  }
  return null;                                 // two lines are tangent only if they coincide
}

/**
 * The contact locus of two exactly tangent surfaces: {line: {o, d}} or {circle: {c, n, r}}, or null when
 * they are not tangent (or not a case above).
 */
export function tangentLocus([k1, p1], [k2, p2], scaleLen) {
  const tol = TANGENT_TOL * scaleLen;
  // coaxial surfaces of revolution
  for (const [ka, pa] of [[k1, p1], [k2, p2]]) {
    if (ka === 'plane' || ka === 'sphere') continue;
    const d = unit(pa.axis), o = ka === 'cylinder' ? pa.axisPoint : ka === 'cone' ? pa.apex : pa.center;
    if (!revolvedAbout(k1, p1, o, d, tol) || !revolvedAbout(k2, p2, o, d, tol)) continue;
    const A = profileOf(k1, p1, o, d), B = profileOf(k2, p2, o, d);
    // a cone's two nappes are both candidates: the contact must lie on one of them
    const alts = (X) => (X?.lines ? X.lines.map((l) => ({ line: l })) : X ? [X] : []);
    let f = null;
    for (const a of alts(A)) for (const b of alts(B)) if (!f) { const c = profileContact(a, b, tol); if (c && c[0] > tol) f = c; }
    if (!f || !(f[0] > tol)) return null;
    return { circle: { c: add(o, scale(d, f[1])), n: d, r: f[0] } };
  }
  const [pl, other] = k1 === 'plane' ? [p1, [k2, p2]] : k2 === 'plane' ? [p2, [k1, p1]] : [null, null];
  if (pl && other[0] === 'cylinder') {
    const n = unit(pl.normal), q = other[1], a = unit(q.axis);
    const off = dot(sub(q.axisPoint, pl.origin), n);
    if (Math.abs(dot(n, a)) > 1e-9 || Math.abs(Math.abs(off) - q.radius) > tol) return null;
    return { line: { o: sub(q.axisPoint, scale(n, off)), d: a } };
  }
  if (pl && other[0] === 'cone') {
    const n = unit(pl.normal), q = other[1], a = unit(q.axis);
    if (Math.abs(dot(sub(q.apex, pl.origin), n)) > tol || Math.abs(Math.abs(dot(n, a)) - Math.sin(q.halfAngle)) > 1e-9) return null;
    const r = sub(a, scale(n, dot(a, n)));
    return norm(r) > 1e-9 ? { line: { o: q.apex, d: unit(r) } } : null;
  }
  if (k1 === 'cylinder' && k2 === 'cylinder') {
    const a = unit(p1.axis);
    if (Math.abs(Math.abs(dot(a, unit(p2.axis))) - 1) > 1e-9) return null;
    const v = sub(p2.axisPoint, p1.axisPoint), w = sub(v, scale(a, dot(v, a))), dd = norm(w);
    if (!(dd > 0)) return null;
    const u = scale(w, 1 / dd);
    let s = 0;
    if (Math.abs(dd - (p1.radius + p2.radius)) <= tol) s = 1;
    else if (Math.abs(dd - Math.abs(p1.radius - p2.radius)) <= tol) s = p1.radius > p2.radius ? 1 : -1;
    else return null;
    return { line: { o: add(p1.axisPoint, scale(u, s * p1.radius)), d: a } };
  }
  return null;
}

/** x projected onto a locus. */
export function ontoLocus(L, x) {
  if (L.line) { const { o, d } = L.line; return add(o, scale(d, dot(sub(x, o), d))); }
  const { c, n, r } = L.circle, v = sub(x, c), w = sub(v, scale(n, dot(v, n))), l = norm(w);
  if (!(l > 0)) return null;
  return add(c, scale(w, r / l));
}

/** The point of a locus nearest x that best lies on the other surfaces `rest` (1D Newton along the locus). */
export function cornerOnLocus(L, rest, x0, scaleLen) {
  let x = ontoLocus(L, x0);
  if (!x || !rest.length) return x;
  const at = (t) => {
    if (L.line) return add(L.line.o, scale(L.line.d, t));
    const { c, n, r } = L.circle, e1 = unit(sub(x0, add(c, scale(n, dot(sub(x0, c), n))))), e2 = cross(n, e1);
    return add(c, add(scale(e1, r * Math.cos(t)), scale(e2, r * Math.sin(t))));
  };
  const f = (t) => rest.reduce((s, [k, p]) => s + distNormal(k, p, at(t))[0] ** 2, 0);
  let t = L.line ? dot(sub(x, L.line.o), L.line.d) : 0;
  const h0 = L.line ? 1e-6 * scaleLen : 1e-6;
  for (let it = 0; it < 50; it++) {
    const fm = f(t - h0), f0 = f(t), fp = f(t + h0);
    const g = (fp - fm) / (2 * h0), H = (fp - 2 * f0 + fm) / (h0 * h0);
    const step = H > 0 ? -g / H : -Math.sign(g) * h0;
    t += step;
    if (Math.abs(step) < 1e-13 * (L.line ? scaleLen : 1)) break;
  }
  return at(t);
}

/**
 * Points on one line or circle (within tol): {line} or {circle: {c, n, r, x}} with x the direction of
 * the first point from the centre; null otherwise.
 */
export function exactCurve(pts, tol) {
  if (pts.length < 2) return null;
  const a = pts[0], b = pts[pts.length - 1];
  const ab = sub(b, a), L = norm(ab);
  if (L > 0) {
    const d = scale(ab, 1 / L);
    if (pts.every((q) => lineDist(a, d, q) <= tol)) return { line: { o: a, d } };
  }
  if (pts.length < 3) return null;
  // circle through the first, a middle and the last point (for a closed chain: first, a third, two thirds)
  const m = pts[Math.floor(pts.length / 3)], m2 = pts[Math.floor((2 * pts.length) / 3)];
  const [p, q, r] = norm(sub(b, a)) > tol ? [a, pts[pts.length >> 1], b] : [a, m, m2];
  const u = sub(q, p), v = sub(r, p), n0 = cross(u, v), nn = dot(n0, n0);
  if (!(nn > 0)) return null;
  const c = add(p, scale(add(scale(cross(v, n0), dot(u, u)), scale(cross(n0, u), dot(v, v))), 1 / (2 * nn)));
  const n = unit(n0), rad = norm(sub(p, c));
  for (const x of pts) {
    const w = sub(x, c);
    if (Math.abs(dot(w, n)) > tol || Math.abs(norm(w) - rad) > tol) return null;
  }
  return { circle: { c, n, r: rad, x: unit(sub(a, c)) } };
}
