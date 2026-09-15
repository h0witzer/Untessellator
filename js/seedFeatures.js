/**
 * seedFeatures.js — the extended (v6) per-facet inputs of the seed-fill model.
 *
 * Mirrored line for line by ml/seedfeatures.py and compared elementwise by test/netparity.mjs.
 * Chosen by a probe screen (test/featureScreen.mjs, ml/feature_screen.py) on 80 ABC parts:
 *
 *   bottleneck    the largest dihedral angle on the gentlest path from the seed through the
 *                 patch (/ 90). Membership AUC +0.04 native, +0.10 gmsh, +0.10 game.
 *   local_crease  the largest dihedral on the facet's own edges (/ 90).
 *   nev0..2       eigenvalues of the covariance of normals within 0.35 patch radii, ascending:
 *                 plane 0/0/0, cylinder 0/0/+, sphere 0/+/+. Helped game-art membership.
 *   {plane,cyl,cone,sphere}_{d,a}
 *                 distance / patch scale and 1-|cos| to each primitive fitted to a
 *                 crease-bounded 32-facet ring at the seed. Type accuracy +11 to +17 points.
 *                 The fits here are closed-form (mean normal, Kasa circle and sphere, Gauss-map
 *                 axis) so both languages compute them identically; they are INPUTS to the
 *                 model, never output geometry, which stays with js/gaussMap.js.
 *
 * Invalid fits (too few facets, singular systems) give distance 10 and angle term 1.
 */

import { eigen3 } from './meshCore.js';

export const EXT_FEATURE_NAMES = [
  'bottleneck', 'local_crease', 'nev0', 'nev1', 'nev2',
  'plane_d', 'plane_a', 'cyl_d', 'cyl_a', 'cone_d', 'cone_a', 'sphere_d', 'sphere_a',
];
export const N_EXT = EXT_FEATURE_NAMES.length;
export const FIT_RING = 32;
export const FIT_CREASE = 30;
export const NORMAL_RADIUS = 0.35;

/** Gaussian elimination with partial pivoting; null when a pivot is below 1e-12 of the largest entry. */
export function solveN(A, b) {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  let big = 0;
  for (const row of A) for (const v of row) big = Math.max(big, Math.abs(v));
  if (!(big > 0)) return null;
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    if (piv !== col) { const tmp = M[col]; M[col] = M[piv]; M[piv] = tmp; }
    if (Math.abs(M[col][col]) < 1e-12 * big) return null;
    for (let r = col + 1; r < n; r++) {
      const f = M[r][col] / M[col][col];
      for (let c = col; c <= n; c++) M[r][c] -= f * M[col][c];
    }
  }
  const x = new Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let v = M[i][n];
    for (let j = i + 1; j < n; j++) v -= M[i][j] * x[j];
    x[i] = v / M[i][i];
  }
  return x;
}

/** Eigenvalues of a symmetric 3x3 matrix, ascending (trigonometric method). */
export function eig3symValues(a00, a01, a02, a11, a12, a22) {
  const p1 = a01 * a01 + a02 * a02 + a12 * a12;
  const q = (a00 + a11 + a22) / 3;
  const p2 = (a00 - q) ** 2 + (a11 - q) ** 2 + (a22 - q) ** 2 + 2 * p1;
  if (p2 < 1e-30) return [q, q, q];
  const p = Math.sqrt(p2 / 6);
  const b00 = (a00 - q) / p, b11 = (a11 - q) / p, b22 = (a22 - q) / p;
  const b01 = a01 / p, b02 = a02 / p, b12 = a12 / p;
  const r = (b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b12 * b02) + b02 * (b01 * b12 - b11 * b02)) / 2;
  const phi = Math.acos(Math.min(1, Math.max(-1, r))) / 3;
  const e1 = q + 2 * p * Math.cos(phi);
  const e3 = q + 2 * p * Math.cos(phi + (2 * Math.PI) / 3);
  return [e3, 3 * q - e1 - e3, e1];
}

/** Tangent axes perpendicular to `n`: seedNet's frameFromNormal (least-aligned world axis, first minimum). */
function perpendicularBasis(n) {
  const a = [Math.abs(n[0]), Math.abs(n[1]), Math.abs(n[2])];
  let k = 0;
  if (a[1] < a[k]) k = 1;
  if (a[2] < a[k]) k = 2;
  const w = [0, 0, 0];
  w[k] = 1;
  let t1 = [n[1] * w[2] - n[2] * w[1], n[2] * w[0] - n[0] * w[2], n[0] * w[1] - n[1] * w[0]];
  const l1 = Math.hypot(t1[0], t1[1], t1[2]) || 1e-30;
  t1 = [t1[0] / l1, t1[1] / l1, t1[2] / l1];
  const t2 = [n[1] * t1[2] - n[2] * t1[1], n[2] * t1[0] - n[0] * t1[2], n[0] * t1[1] - n[1] * t1[0]];
  return [t1, t2];
}

/**
 * Layered BFS from `seed` across edges below `deg` degrees, skipping excluded facets. Each ring's
 * new facets are sorted by index before truncation at `want`, as ml/meshcore.k_ring does.
 */
export function creaseRing(seed, adjacency, want, deg, excluded = null) {
  const seen = new Set([seed]);
  const ids = [seed];
  let frontier = [seed];
  while (ids.length < want && frontier.length) {
    const next = [];
    for (const f of frontier) {
      for (const e of adjacency.get(f) || []) {
        const nb = e.neighbor;
        if (seen.has(nb) || e.angle >= deg || (excluded && excluded[nb])) continue;
        seen.add(nb);
        next.push(nb);
      }
    }
    next.sort((a, b) => a - b);
    for (const f of next) {
      if (ids.length >= want) break;
      ids.push(f);
    }
    frontier = next;
  }
  return ids;
}

/** Largest dihedral on the gentlest path from patch index 0, by Kruskal over the patch's own edges. */
export function patchBottleneck(ids, adjacency) {
  const n = ids.length;
  const index = new Map(ids.map((t, i) => [t, i]));
  const edges = [];
  for (let i = 0; i < n; i++) {
    for (const e of adjacency.get(ids[i]) || []) {
      const j = index.get(e.neighbor);
      if (j !== undefined && j > i) edges.push([e.angle, i, j]);
    }
  }
  edges.sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]);
  const parent = Int32Array.from({ length: n }, (_, i) => i);
  const members = Array.from({ length: n }, (_, i) => [i]);
  const out = new Float64Array(n).fill(Infinity);
  out[0] = 0;
  const find = (i) => {
    while (parent[i] !== i) { parent[i] = parent[parent[i]]; i = parent[i]; }
    return i;
  };
  for (const [w, i, j] of edges) {
    let ri = find(i), rj = find(j);
    if (ri === rj) continue;
    const rs = find(0);
    if (ri === rs) for (const m of members[rj]) out[m] = w;
    else if (rj === rs) for (const m of members[ri]) out[m] = w;
    if (members[ri].length < members[rj].length) { const tmp = ri; ri = rj; rj = tmp; }
    parent[rj] = ri;
    for (const m of members[rj]) members[ri].push(m);
    members[rj] = [];
  }
  return out;
}

/** Closed-form plane, sphere, cylinder and cone fitted to the facet centroids and normals of `ring`. */
export function fitPrimitives(ring, facets) {
  const none = { plane: null, sphere: null, cylinder: null, cone: null };
  const N = ring.length;
  if (N < 3) return none;
  const C = facets.centroids;
  const Nr = facets.normals;
  const m = [0, 0, 0];
  const sn = [0, 0, 0];
  for (const t of ring) {
    for (let a = 0; a < 3; a++) { m[a] += C[t * 3 + a]; sn[a] += Nr[t * 3 + a]; }
  }
  for (let a = 0; a < 3; a++) m[a] /= N;
  let ss = 0;
  for (const t of ring) ss += (C[t * 3] - m[0]) ** 2 + (C[t * 3 + 1] - m[1]) ** 2 + (C[t * 3 + 2] - m[2]) ** 2;
  const s = Math.sqrt(ss / N);
  if (!(s > 0)) return none;
  const P = ring.map((t) => [(C[t * 3] - m[0]) / s, (C[t * 3 + 1] - m[1]) / s, (C[t * 3 + 2] - m[2]) / s]);
  const Nn = ring.map((t) => [Nr[t * 3], Nr[t * 3 + 1], Nr[t * 3 + 2]]);
  const out = { ...none };

  const sl = Math.hypot(sn[0], sn[1], sn[2]);
  if (sl > 1e-12) out.plane = { origin: m, normal: [sn[0] / sl, sn[1] / sl, sn[2] / sl] };

  // Sphere, Kasa: |p|^2 = 2 c.p + k, r^2 = k + |c|^2, in centred and scaled coordinates.
  {
    const A = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
    const b = [0, 0, 0, 0];
    for (const p of P) {
      const row = [2 * p[0], 2 * p[1], 2 * p[2], 1];
      const rhs = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
      for (let i = 0; i < 4; i++) {
        b[i] += row[i] * rhs;
        for (let j = 0; j < 4; j++) A[i][j] += row[i] * row[j];
      }
    }
    const x = solveN(A, b);
    if (x) {
      const r2 = x[3] + x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
      if (r2 > 0) out.sphere = { center: [m[0] + s * x[0], m[1] + s * x[1], m[2] + s * x[2]], radius: s * Math.sqrt(r2) };
    }
  }

  // Normal second moment, shared by the cylinder axis and the cone apex.
  let n00 = 0, n01 = 0, n02 = 0, n11 = 0, n12 = 0, n22 = 0;
  for (const n of Nn) {
    n00 += n[0] * n[0]; n01 += n[0] * n[1]; n02 += n[0] * n[2];
    n11 += n[1] * n[1]; n12 += n[1] * n[2]; n22 += n[2] * n[2];
  }

  // Cylinder: the axis is the direction the normals avoid; then a Kasa circle across it.
  {
    const axis = eigen3([n00, n01, n02, n11, n12, n22]).vectors[0];
    const [u, v] = perpendicularBasis(axis);
    const A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    const b = [0, 0, 0];
    for (const p of P) {
      const x = p[0] * u[0] + p[1] * u[1] + p[2] * u[2];
      const y = p[0] * v[0] + p[1] * v[1] + p[2] * v[2];
      const row = [2 * x, 2 * y, 1];
      const rhs = x * x + y * y;
      for (let i = 0; i < 3; i++) {
        b[i] += row[i] * rhs;
        for (let j = 0; j < 3; j++) A[i][j] += row[i] * row[j];
      }
    }
    const x = solveN(A, b);
    if (x) {
      const r2 = x[2] + x[0] * x[0] + x[1] * x[1];
      if (r2 > 0) {
        out.cylinder = {
          axis,
          center: [0, 1, 2].map((a) => m[a] + s * (x[0] * u[a] + x[1] * v[a])),
          radius: s * Math.sqrt(r2),
        };
      }
    }
  }

  // Cone: the normals keep a constant angle to the axis (smallest centred eigenvector), and every
  // tangent plane passes through the apex: (sum n n^T) apex = sum n (n.p).
  {
    const mn = [sn[0] / N, sn[1] / N, sn[2] / N];
    const cov = [n00 / N - mn[0] * mn[0], n01 / N - mn[0] * mn[1], n02 / N - mn[0] * mn[2],
                 n11 / N - mn[1] * mn[1], n12 / N - mn[1] * mn[2], n22 / N - mn[2] * mn[2]];
    let axis = eigen3(cov).vectors[0];
    let sdot = 0;
    for (const n of Nn) sdot += n[0] * axis[0] + n[1] * axis[1] + n[2] * axis[2];
    sdot /= N;
    if (sdot < 0) { axis = [-axis[0], -axis[1], -axis[2]]; sdot = -sdot; }
    const sin = Math.min(1, sdot);
    const b = [0, 0, 0];
    for (let i = 0; i < N; i++) {
      const n = Nn[i];
      const np = n[0] * P[i][0] + n[1] * P[i][1] + n[2] * P[i][2];
      for (let a = 0; a < 3; a++) b[a] += n[a] * np;
    }
    const x = solveN([[n00, n01, n02], [n01, n11, n12], [n02, n12, n22]], b);
    if (x) {
      out.cone = { axis, apex: [0, 1, 2].map((a) => m[a] + s * x[a]), sin, cos: Math.sqrt(Math.max(0, 1 - sin * sin)) };
    }
  }
  return out;
}

const SENTINEL_D = 10;

/** Distance and |cos| of facet (q, nt) to each fitted primitive; [d, a] pairs with the sentinels for invalid ones. */
function evaluateFits(fits, q, nt, scale, into, o) {
  const put = (k, d, cos) => {
    into[o + 2 * k] = Number.isFinite(d) ? Math.min(SENTINEL_D, d / scale) : SENTINEL_D;
    into[o + 2 * k + 1] = Number.isFinite(cos) ? 1 - Math.min(1, cos) : 1;
  };
  const { plane, cylinder, cone, sphere } = fits;
  if (plane) {
    const d = (q[0] - plane.origin[0]) * plane.normal[0] + (q[1] - plane.origin[1]) * plane.normal[1] + (q[2] - plane.origin[2]) * plane.normal[2];
    put(0, Math.abs(d), Math.abs(nt[0] * plane.normal[0] + nt[1] * plane.normal[1] + nt[2] * plane.normal[2]));
  } else put(0, NaN, NaN);
  if (cylinder) {
    const v = [q[0] - cylinder.center[0], q[1] - cylinder.center[1], q[2] - cylinder.center[2]];
    const al = v[0] * cylinder.axis[0] + v[1] * cylinder.axis[1] + v[2] * cylinder.axis[2];
    const w = [v[0] - al * cylinder.axis[0], v[1] - al * cylinder.axis[1], v[2] - al * cylinder.axis[2]];
    const y = Math.hypot(w[0], w[1], w[2]);
    if (y > 1e-30) put(1, Math.abs(y - cylinder.radius), Math.abs(nt[0] * w[0] + nt[1] * w[1] + nt[2] * w[2]) / y);
    else put(1, NaN, NaN);
  } else put(1, NaN, NaN);
  if (cone) {
    const v = [q[0] - cone.apex[0], q[1] - cone.apex[1], q[2] - cone.apex[2]];
    const al = v[0] * cone.axis[0] + v[1] * cone.axis[1] + v[2] * cone.axis[2];
    const w = [v[0] - al * cone.axis[0], v[1] - al * cone.axis[1], v[2] - al * cone.axis[2]];
    const y = Math.hypot(w[0], w[1], w[2]);
    if (y > 1e-30) {
      const sg = al < 0 ? -1 : 1;
      const nrm = [0, 1, 2].map((a) => (cone.cos * w[a]) / y - cone.sin * sg * cone.axis[a]);
      put(2, Math.abs(y * cone.cos - Math.abs(al) * cone.sin), Math.abs(nt[0] * nrm[0] + nt[1] * nrm[1] + nt[2] * nrm[2]));
    } else put(2, NaN, NaN);
  } else put(2, NaN, NaN);
  if (sphere) {
    const w = [q[0] - sphere.center[0], q[1] - sphere.center[1], q[2] - sphere.center[2]];
    const y = Math.hypot(w[0], w[1], w[2]);
    if (y > 1e-30) put(3, Math.abs(y - sphere.radius), Math.abs(nt[0] * w[0] + nt[1] * w[1] + nt[2] * w[2]) / y);
    else put(3, NaN, NaN);
  } else put(3, NaN, NaN);
}

/**
 * Write the 13 extended columns for patch `ids` into `out` at column `col0` of each row of width
 * `width`. `dScaled` and `nFrame` are the patch's canonical positions (already / scale) and normals
 * as float64 triples; `scale` is the patch radius in world units.
 */
export function writeExtendedFeatures(out, width, col0, ids, seed, facets, adjacency, dScaled, nFrame, scale, excluded = null) {
  const n = ids.length;
  const bottleneck = patchBottleneck(ids, adjacency);
  const fits = fitPrimitives(creaseRing(seed, adjacency, FIT_RING, FIT_CREASE, excluded), facets);
  const r2 = NORMAL_RADIUS * NORMAL_RADIUS;
  const tmp = new Float64Array(8);
  for (let i = 0; i < n; i++) {
    const t = ids[i];
    const o = i * width + col0;
    out[o] = Math.min(180, bottleneck[i]) / 90;
    let local = 0;
    for (const e of adjacency.get(t) || []) if (e.angle > local) local = e.angle;
    out[o + 1] = local / 90;

    let w = 0, m0 = 0, m1 = 0, m2 = 0, s00 = 0, s01 = 0, s02 = 0, s11 = 0, s12 = 0, s22 = 0;
    const px = dScaled[i * 3], py = dScaled[i * 3 + 1], pz = dScaled[i * 3 + 2];
    for (let j = 0; j < n; j++) {
      const dx = dScaled[j * 3] - px, dy = dScaled[j * 3 + 1] - py, dz = dScaled[j * 3 + 2] - pz;
      if (dx * dx + dy * dy + dz * dz > r2) continue;
      const a = nFrame[j * 3], b = nFrame[j * 3 + 1], g = nFrame[j * 3 + 2];
      w++; m0 += a; m1 += b; m2 += g;
      s00 += a * a; s01 += a * b; s02 += a * g; s11 += b * b; s12 += b * g; s22 += g * g;
    }
    m0 /= w; m1 /= w; m2 /= w;
    const ev = eig3symValues(s00 / w - m0 * m0, s01 / w - m0 * m1, s02 / w - m0 * m2,
                             s11 / w - m1 * m1, s12 / w - m1 * m2, s22 / w - m2 * m2);
    out[o + 2] = Math.max(0, ev[0]);
    out[o + 3] = Math.max(0, ev[1]);
    out[o + 4] = Math.max(0, ev[2]);

    const q = [facets.centroids[t * 3], facets.centroids[t * 3 + 1], facets.centroids[t * 3 + 2]];
    const nt = [facets.normals[t * 3], facets.normals[t * 3 + 1], facets.normals[t * 3 + 2]];
    evaluateFits(fits, q, nt, scale, tmp, 0);
    for (let k = 0; k < 8; k++) out[o + 5 + k] = tmp[k];
  }
}
