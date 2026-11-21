/**
 * fitCage.js — fit a quad cage so that its NURBS surface (js/subd/fastNurbs.js) lies on a mesh.
 *
 * Every output pole is a fixed linear combination of the cage's vertices (subdLinearMap, subdivision included),
 * so the surface point at (face, u, v) is a fixed row times the cage. The fit alternates footpoints (each mesh
 * point's closest point on the current surface, warm-started from the last round) and one sparse linear
 * least-squares solve for the cage. A Laplacian term on the cage's displacement, 1e-6 of the data's own weight,
 * only settles cage vertices that no data point reaches.
 */
import { subdLinearMap, applyLinearMap, bezierKnots } from './fastNurbs.js';
import { envelope } from './envelopeCholesky.js';

const BIN = [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1], [1, 4, 6, 4, 1], [1, 5, 10, 10, 5, 1], [1, 6, 15, 20, 15, 6, 1]];
const TP = new Float64Array(8), SP = new Float64Array(8);

/** Bernstein basis of degree p at t, and its first and second derivatives, into b0, b1, b2. */
function bern(p, t, b0, b1, b2) {
  const s = 1 - t;
  TP[0] = 1; SP[0] = 1;
  for (let k = 1; k <= p; k++) { TP[k] = TP[k - 1] * t; SP[k] = SP[k - 1] * s; }
  const c1 = (i) => (i < 0 || i > p - 1 ? 0 : BIN[p - 1][i] * TP[i] * SP[p - 1 - i]);
  const c2 = (i) => (i < 0 || i > p - 2 ? 0 : BIN[p - 2][i] * TP[i] * SP[p - 2 - i]);
  for (let i = 0; i <= p; i++) {
    b0[i] = BIN[p][i] * TP[i] * SP[p - i];
    if (b1) b1[i] = p * (c1(i - 1) - c1(i));
    if (b2) b2[i] = p * (p - 1) * (c2(i - 2) - 2 * c2(i - 1) + c2(i));
  }
}

const BU = [new Float64Array(7), new Float64Array(7), new Float64Array(7)];
const BV = [new Float64Array(7), new Float64Array(7), new Float64Array(7)];
const D = new Float64Array(18);                     // S, Su, Sv, Suu, Suv, Svv

/** Point and derivatives up to order 2 of a Bezier patch (n x n poles P, 3 each) at (u, v), into D. */
function evalAt(n, P, u, v) {
  const p = n - 1;
  bern(p, u, BU[0], BU[1], BU[2]); bern(p, v, BV[0], BV[1], BV[2]);
  D.fill(0);
  for (let i = 0; i < n; i++) {
    let r0x = 0, r0y = 0, r0z = 0, r1x = 0, r1y = 0, r1z = 0, r2x = 0, r2y = 0, r2z = 0;
    for (let j = 0; j < n; j++) {
      const o = (i * n + j) * 3, x = P[o], y = P[o + 1], z = P[o + 2], a = BV[0][j], b = BV[1][j], c = BV[2][j];
      r0x += a * x; r0y += a * y; r0z += a * z;
      r1x += b * x; r1y += b * y; r1z += b * z;
      r2x += c * x; r2y += c * y; r2z += c * z;
    }
    const a = BU[0][i], b = BU[1][i], c = BU[2][i];
    D[0] += a * r0x; D[1] += a * r0y; D[2] += a * r0z;
    D[3] += b * r0x; D[4] += b * r0y; D[5] += b * r0z;
    D[6] += a * r1x; D[7] += a * r1y; D[8] += a * r1z;
    D[9] += c * r0x; D[10] += c * r0y; D[11] += c * r0z;
    D[12] += b * r1x; D[13] += b * r1y; D[14] += b * r1z;
    D[15] += a * r2x; D[16] += a * r2y; D[17] += a * r2z;
  }
}

/** Closest point on a face to m by Newton from (u, v), clamped to the face. */
function project(n, P, m, u, v) {
  for (let it = 0; it < 8; it++) {
    evalAt(n, P, u, v);
    const rx = D[0] - m[0], ry = D[1] - m[1], rz = D[2] - m[2];
    const gu = D[3] * rx + D[4] * ry + D[5] * rz, gv = D[6] * rx + D[7] * ry + D[8] * rz;
    const huu = D[3] * D[3] + D[4] * D[4] + D[5] * D[5] + D[9] * rx + D[10] * ry + D[11] * rz;
    const huv = D[3] * D[6] + D[4] * D[7] + D[5] * D[8] + D[12] * rx + D[13] * ry + D[14] * rz;
    const hvv = D[6] * D[6] + D[7] * D[7] + D[8] * D[8] + D[15] * rx + D[16] * ry + D[17] * rz;
    const det = huu * hvv - huv * huv;
    if (!(det > 0)) break;
    const du = (hvv * gu - huv * gv) / det, dv = (huu * gv - huv * gu) / det;
    u = Math.min(1, Math.max(0, u - du)); v = Math.min(1, Math.max(0, v - dv));
    if (Math.abs(du) + Math.abs(dv) < 1e-10) break;
  }
  evalAt(n, P, u, v);
  const d2 = (D[0] - m[0]) ** 2 + (D[1] - m[1]) ** 2 + (D[2] - m[2]) ** 2;
  return { u, v, d2, edge: u <= 0 || u >= 1 || v <= 0 || v >= 1 };
}

/** Nearest-sample index over a grid of samples per face: nearest(m) -> [face, u, v]. Exact nearest sample. */
function sampleIndex(poles, sizes, grid) {
  const ns = poles.length * grid * grid, S = new Float64Array(ns * 3), at = new Int32Array(ns);
  const basis = new Map();
  for (const n of new Set(sizes)) {
    const rows = Array.from({ length: grid }, (_, a) => { const b = new Float64Array(n); bern(n - 1, a / (grid - 1), b); return b; });
    basis.set(n, rows);
  }
  let k = 0;
  poles.forEach((P, f) => {
    const n = sizes[f], B = basis.get(n);
    for (let a = 0; a < grid; a++) for (let b = 0; b < grid; b++, k++) {
      let x = 0, y = 0, z = 0;
      for (let i = 0; i < n; i++) {
        const bi = B[a][i];
        if (!bi) continue;
        for (let j = 0; j < n; j++) {
          const w = bi * B[b][j];
          if (!w) continue;
          const o = (i * n + j) * 3;
          x += w * P[o]; y += w * P[o + 1]; z += w * P[o + 2];
        }
      }
      S[k * 3] = x; S[k * 3 + 1] = y; S[k * 3 + 2] = z; at[k] = f;
    }
  });
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < ns; i++) for (let c = 0; c < 3; c++) { lo[c] = Math.min(lo[c], S[i * 3 + c]); hi[c] = Math.max(hi[c], S[i * 3 + c]); }
  const diag = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]), cell = 3 * diag / Math.sqrt(ns);
  const nx = Math.floor((hi[0] - lo[0]) / cell) + 3, ny = Math.floor((hi[1] - lo[1]) / cell) + 3;
  const cellOf = (x, c) => Math.floor((x - lo[c]) / cell) + 1;
  const key = (ix, iy, iz) => ix + nx * (iy + ny * iz);
  const hash = new Map();
  for (let i = 0; i < ns; i++) {
    const kk = key(cellOf(S[i * 3], 0), cellOf(S[i * 3 + 1], 1), cellOf(S[i * 3 + 2], 2));
    const l = hash.get(kk);
    if (l) l.push(i); else hash.set(kk, [i]);
  }
  const nearest = (m) => {
    const cx = cellOf(m[0], 0), cy = cellOf(m[1], 1), cz = cellOf(m[2], 2);
    let best = -1, bd = Infinity;
    for (let r = 0; r < 400; r++) {
      for (let dx = -r; dx <= r; dx++) for (let dy = -r; dy <= r; dy++) for (let dz = -r; dz <= r; dz++) {
        if (Math.max(Math.abs(dx), Math.abs(dy), Math.abs(dz)) !== r) continue;   // this shell only
        const l = hash.get(key(cx + dx, cy + dy, cz + dz));
        if (!l) continue;
        for (const i of l) {
          const d = (S[i * 3] - m[0]) ** 2 + (S[i * 3 + 1] - m[1]) ** 2 + (S[i * 3 + 2] - m[2]) ** 2;
          if (d < bd) { bd = d; best = i; }
        }
      }
      if (best >= 0 && bd <= (r * cell) ** 2) break;  // nothing outside the searched block is closer
    }
    const f = at[best], g = best - f * grid * grid;
    return [f, Math.floor(g / grid) / (grid - 1), (g % grid) / (grid - 1)];
  };
  return { nearest, diag };
}

/**
 * Closest points: foot[i] = { fi, u, v, d2 } for points[i]. Warm-started from foot[i] when it is set (last
 * round's); the nearest sample's face is tried as well when that ends on a face edge or there is none.
 */
function footpoints(poles, sizes, points, foot, index) {
  let worst = 0, sum = 0, searched = 0;
  points.forEach((m, i) => {
    let best = null;
    if (foot[i]) {
      const f = foot[i].fi, r = project(sizes[f], poles[f], m, foot[i].u, foot[i].v);
      best = { fi: f, ...r };
    }
    if (!best || best.edge) {
      searched++;
      const [f, u, v] = index.nearest(m), r = project(sizes[f], poles[f], m, u, v);
      if (!best || r.d2 < best.d2) best = { fi: f, ...r };
    }
    foot[i] = best;
    worst = Math.max(worst, best.d2); sum += best.d2;
  });
  return { worst, sum, searched };
}

/** Distance from each point to a patch surface (faces: fastNurbs / fitCage output), and the surface's diagonal. */
export function surfaceDistances(faces, points, { grid = 7 } = {}) {
  const poles = faces.map((f) => Float64Array.from(f.poles)), sizes = faces.map((f) => f.uCount);
  const foot = new Array(points.length);
  const index = sampleIndex(poles, sizes, grid);
  footpoints(poles, sizes, points, foot, index);
  return { distances: Float64Array.from(foot, (f) => Math.sqrt(f.d2)), diag: index.diag };
}

/**
 * Fit. points: [[x, y, z]...] on the mesh. Returns { vertices (fitted cage), faces (NURBS), cageFaces (the
 * NURBS faces' corners), stats }.
 */
export function fitCage(cage, points, { rounds = 5, grid = 7 } = {}) {
  const t0 = performance.now();
  const N = cage.vertices.length;
  const map = subdLinearMap(N, cage.faces);
  const sizes = map.faces.map((f) => f.uCount);
  // the normal equations' pattern: each face's support, and each cage vertex's Laplacian stencil
  const adj = Array.from({ length: N }, () => new Set());
  const clique = (ids) => { for (const a of ids) for (const b of ids) if (a !== b) adj[a].add(b); };
  for (const f of map.faces) clique(f.ids);
  const neighbours = Array.from({ length: N }, () => new Set());
  for (const q of cage.faces) for (let k = 0; k < 4; k++) { neighbours[q[k]].add(q[(k + 1) % 4]); neighbours[q[(k + 1) % 4]].add(q[k]); }
  neighbours.forEach((nb, j) => clique([j, ...nb]));
  const A = envelope(N, adj);
  const ms = { map: performance.now() - t0, samples: 0, foot: 0, assemble: 0, solve: 0 };
  let tick = performance.now();
  const lap = (k) => { const t = performance.now(); ms[k] += t - tick; tick = t; };

  let X = cage.vertices.map((p) => [...p]);
  const X0 = cage.vertices.map((p) => [...p]);
  const history = [];
  const foot = new Array(points.length);
  let poles = null;
  for (let round = 0; round <= rounds; round++) {
    tick = performance.now();
    poles = applyLinearMap(map, X);
    const index = sampleIndex(poles, sizes, grid);
    lap('samples');
    const { worst, sum, searched } = footpoints(poles, sizes, points, foot, index);
    lap('foot');
    history.push({ round, max: Math.sqrt(worst) / index.diag, rms: Math.sqrt(sum / points.length) / index.diag, searched });
    if (round === rounds) break;
    // normal equations for the cage, coordinate by coordinate (the rows are the same for x, y, z): per face,
    // G = sum over its points of b b^T (b: the pole basis at the footpoint) and R = sum b m^T, then W^T G W, W^T R
    const byFace = map.faces.map(() => []);
    foot.forEach((f, i) => byFace[f.fi].push(i));
    A.clear();
    const rhs = [new Float64Array(N), new Float64Array(N), new Float64Array(N)];
    let trace = 0;
    const bu = new Float64Array(7), bv = new Float64Array(7);
    byFace.forEach((list, fi) => {
      if (!list.length) return;
      const { uCount: n, ids, W } = map.faces[fi], np = n * n, nl = ids.length;
      const G = new Float64Array(np * np), R = new Float64Array(np * 3), b = new Float64Array(np);
      for (const i of list) {
        bern(n - 1, foot[i].u, bu); bern(n - 1, foot[i].v, bv);
        for (let a = 0; a < n; a++) for (let c = 0; c < n; c++) b[a * n + c] = bu[a] * bv[c];
        const m = points[i];
        for (let p = 0; p < np; p++) {
          const bp = b[p];
          if (!bp) continue;
          for (let q = 0; q <= p; q++) G[p * np + q] += bp * b[q];
          R[p * 3] += bp * m[0]; R[p * 3 + 1] += bp * m[1]; R[p * 3 + 2] += bp * m[2];
        }
      }
      for (let p = 0; p < np; p++) for (let q = 0; q < p; q++) G[q * np + p] = G[p * np + q];
      // GW = G W (np x nl), then local = W^T GW
      const GW = new Float64Array(np * nl);
      for (let p = 0; p < np; p++) for (let q = 0; q < np; q++) {
        const g = G[p * np + q];
        if (!g) continue;
        for (let l = 0; l < nl; l++) GW[p * nl + l] += g * W[q * nl + l];
      }
      for (let l = 0; l < nl; l++) {
        for (let l2 = 0; l2 <= l; l2++) {
          let s = 0;
          for (let p = 0; p < np; p++) s += W[p * nl + l] * GW[p * nl + l2];
          if (s) A.add(ids[l], ids[l2], s);
          if (l2 === l) trace += s;
        }
        let x = 0, y = 0, z = 0;
        for (let p = 0; p < np; p++) { const w = W[p * nl + l]; x += w * R[p * 3]; y += w * R[p * 3 + 1]; z += w * R[p * 3 + 2]; }
        rhs[0][ids[l]] += x; rhs[1][ids[l]] += y; rhs[2][ids[l]] += z;
      }
    });
    const mu = 1e-6 * trace / N;                     // settles only what the data does not reach
    for (let j = 0; j < N; j++) {
      const nb = [...neighbours[j]], w = 1 / nb.length;
      const stencil = [[j, 1], ...nb.map((k) => [k, -w])];
      for (let a = 0; a < stencil.length; a++) {
        const [ia, wa] = stencil[a];
        for (let c = 0; c <= a; c++) A.add(ia, stencil[c][0], mu * wa * stencil[c][1]);
        for (const [ic, wc] of stencil) for (let k = 0; k < 3; k++) rhs[k][ia] += mu * wa * wc * X0[ic][k];
      }
    }
    lap('assemble');
    A.factor();
    const xs = rhs.map((r) => A.solve(r));
    X = X.map((_, j) => [xs[0][j], xs[1][j], xs[2][j]]);
    lap('solve');
  }
  const faces = map.faces.map(({ uCount, vCount }, f) => ({ uKnots: bezierKnots(uCount - 1), vKnots: bezierKnots(vCount - 1), uCount, vCount, poles: Array.from(poles[f]), weights: null }));
  return { vertices: X, faces, cageFaces: map.cageFaces,
    stats: { history, subdivisions: map.subdivisions, envelope: A.size, ms: { ...ms, total: performance.now() - t0 } } };
}
