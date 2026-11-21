/**
 * fitCage.js — fit a quad cage so that its NURBS surface (js/subd/fastNurbs.js) lies on a mesh.
 *
 * Every output pole is a fixed linear combination of the cage's vertices (subdivision included), so the surface
 * point at (face, u, v) is a fixed row times the cage. The fit alternates footpoints (each mesh point's closest
 * point on the current surface) and one linear least-squares solve for the cage. A Laplacian term on the cage's
 * displacement, 1e-6 of the data's own weight, only settles cage vertices that no data point reaches.
 */
import { fastSubdToNurbs } from './fastNurbs.js';
import { bernstein } from './kpCap.js';

/** The sparse linear map: for output face f and pole p, [[cage vertex, weight], ...]. */
export function linearMap(vertices, faces) {
  const n = vertices.length, map = [];
  for (let k = 0; k < n; k += 3) {
    const V = vertices.map(() => [0, 0, 0]);
    for (let c = 0; c < 3 && k + c < n; c++) V[k + c][c] = 1;
    const out = fastSubdToNurbs(V, faces).faces;
    out.forEach((f, fi) => {
      if (!map[fi]) map[fi] = { uCount: f.uCount, poles: Array.from({ length: f.uCount * f.vCount }, () => []) };
      for (let p = 0; p < f.uCount * f.vCount; p++) for (let c = 0; c < 3 && k + c < n; c++) {
        const w = f.poles[p * 3 + c];
        if (Math.abs(w) > 1e-13) map[fi].poles[p].push([k + c, w]);
      }
    });
  }
  return map;
}

function evalPatch(f, u, v, nd = 2) {
  const p = f.uCount - 1, Bu = bernstein(p, u, nd), Bv = bernstein(p, v, nd), out = {};
  for (const [a, b] of [[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]]) {
    if (a + b > nd) continue;
    const x = [0, 0, 0];
    for (let i = 0; i <= p; i++) for (let j = 0; j <= p; j++) {
      const w = Bu[a][i] * Bv[b][j];
      if (w) { const o = (i * f.uCount + j) * 3; x[0] += w * f.poles[o]; x[1] += w * f.poles[o + 1]; x[2] += w * f.poles[o + 2]; }
    }
    out[`${a}${b}`] = x;
  }
  return out;
}

/** Closest point on face f to m, Newton from (u, v), clamped to the face. */
function project(f, m, u, v) {
  for (let it = 0; it < 6; it++) {
    const d = evalPatch(f, u, v), r = [d['00'][0] - m[0], d['00'][1] - m[1], d['00'][2] - m[2]];
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const gu = dot(d['10'], r), gv = dot(d['01'], r);
    const huu = dot(d['10'], d['10']) + dot(d['20'], r), huv = dot(d['10'], d['01']) + dot(d['11'], r), hvv = dot(d['01'], d['01']) + dot(d['02'], r);
    const det = huu * hvv - huv * huv;
    if (!(det > 0)) break;
    const du = (hvv * gu - huv * gv) / det, dv = (huu * gv - huv * gu) / det;
    u = Math.min(1, Math.max(0, u - du)); v = Math.min(1, Math.max(0, v - dv));
    if (Math.abs(du) + Math.abs(dv) < 1e-10) break;
  }
  const p = evalPatch(f, u, v, 0)['00'];
  return { u, v, dist: Math.hypot(p[0] - m[0], p[1] - m[1], p[2] - m[2]) };
}

function cholesky(A, n) {
  const L = Float64Array.from(A);
  for (let j = 0; j < n; j++) {
    let d = L[j * n + j];
    for (let k = 0; k < j; k++) d -= L[j * n + k] ** 2;
    const l = Math.sqrt(Math.max(d, 1e-300));
    L[j * n + j] = l;
    for (let i = j + 1; i < n; i++) {
      let s = L[i * n + j];
      for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
      L[i * n + j] = s / l;
    }
  }
  return (b) => {
    const x = Float64Array.from(b);
    for (let i = 0; i < n; i++) { let s = x[i]; for (let k = 0; k < i; k++) s -= L[i * n + k] * x[k]; x[i] = s / L[i * n + i]; }
    for (let i = n - 1; i >= 0; i--) { let s = x[i]; for (let k = i + 1; k < n; k++) s -= L[k * n + i] * x[k]; x[i] = s / L[i * n + i]; }
    return x;
  };
}

/**
 * Fit. points: [[x, y, z]...] on the mesh. Returns { vertices (fitted cage), faces (NURBS), stats }.
 */
export function fitCage(cage, points, { rounds = 5, grid = 7 } = {}) {
  const t0 = performance.now();
  const N = cage.vertices.length;
  const map = linearMap(cage.vertices, cage.faces);
  const tMap = performance.now() - t0;
  const neighbours = Array.from({ length: N }, () => new Set());
  for (const q of cage.faces) for (let k = 0; k < 4; k++) { neighbours[q[k]].add(q[(k + 1) % 4]); neighbours[q[(k + 1) % 4]].add(q[k]); }
  let X = cage.vertices.map((p) => [...p]);
  const X0 = cage.vertices.map((p) => [...p]);
  const history = [];
  let nurbs = null;
  for (let round = 0; round <= rounds; round++) {
    nurbs = fastSubdToNurbs(X, cage.faces);
    const faces = nurbs.faces;
    // footpoints: nearest of a grid of samples per face (spatial hash), then Newton on that face
    const samples = [];
    faces.forEach((f, fi) => { for (let a = 0; a < grid; a++) for (let b = 0; b < grid; b++) {
      const u = a / (grid - 1), v = b / (grid - 1); samples.push([fi, u, v, evalPatch(f, u, v, 0)['00']]);
    } });
    let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (const s of samples) for (let c = 0; c < 3; c++) { lo[c] = Math.min(lo[c], s[3][c]); hi[c] = Math.max(hi[c], s[3][c]); }
    const diag = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]), cell = diag / 40;
    const hash = new Map(), key = (p) => p.map((x, c) => Math.floor((x - lo[c]) / cell)).join(',');
    samples.forEach((s, i) => { const k = key(s[3]); if (!hash.has(k)) hash.set(k, []); hash.get(k).push(i); });
    const foot = points.map((m) => {
      const c0 = m.map((x, c) => Math.floor((x - lo[c]) / cell));
      let best = -1, bd = Infinity;
      for (let r = 1; best < 0 || r <= 2; r++) {
        for (let dx = -r; dx <= r; dx++) for (let dy = -r; dy <= r; dy++) for (let dz = -r; dz <= r; dz++) {
          for (const i of hash.get(`${c0[0] + dx},${c0[1] + dy},${c0[2] + dz}`) ?? []) {
            const s = samples[i][3], d = (s[0] - m[0]) ** 2 + (s[1] - m[1]) ** 2 + (s[2] - m[2]) ** 2;
            if (d < bd) { bd = d; best = i; }
          }
        }
        if (r > 40) break;
      }
      const [fi, u, v] = samples[best];
      return { fi, ...project(faces[fi], m, u, v) };
    });
    let worst = 0, sum = 0;
    for (const f of foot) { worst = Math.max(worst, f.dist); sum += f.dist * f.dist; }
    history.push({ round, max: worst / diag, rms: Math.sqrt(sum / foot.length) / diag });
    if (round === rounds) break;
    // normal equations for the cage, coordinate by coordinate (the rows are the same for x, y, z)
    const A = new Float64Array(N * N), b = [new Float64Array(N), new Float64Array(N), new Float64Array(N)];
    let trace = 0;
    points.forEach((m, i) => {
      const { fi, u, v } = foot[i], f = map[fi], p = f.uCount - 1, Bu = bernstein(p, u, 0)[0], Bv = bernstein(p, v, 0)[0];
      const row = new Map();
      for (let a = 0; a <= p; a++) for (let c = 0; c <= p; c++) {
        const w = Bu[a] * Bv[c];
        if (Math.abs(w) < 1e-15) continue;
        for (const [j, x] of f.poles[a * (p + 1) + c]) row.set(j, (row.get(j) ?? 0) + w * x);
      }
      const ents = [...row];
      for (const [j, wj] of ents) {
        for (const [k, wk] of ents) A[j * N + k] += wj * wk;
        for (let c = 0; c < 3; c++) b[c][j] += wj * m[c];
        trace += wj * wj;
      }
    });
    const mu = 1e-6 * trace / N;                     // settles only what the data does not reach
    for (let j = 0; j < N; j++) {
      const nb = [...neighbours[j]], w = 1 / nb.length;
      const lap = new Map([[j, 1]]);
      for (const k of nb) lap.set(k, (lap.get(k) ?? 0) - w);
      for (const [a, wa] of lap) for (const [c2, wc] of lap) A[a * N + c2] += mu * wa * wc;
      for (const [a, wa] of lap) for (const [c2, wc] of lap) for (let c = 0; c < 3; c++) b[c][a] += mu * wa * wc * X0[c2][c];
    }
    const solve = cholesky(A, N);
    const xs = b.map((bb) => solve(bb));
    X = X.map((_, j) => [xs[0][j], xs[1][j], xs[2][j]]);
  }
  return { vertices: X, faces: nurbs.faces, cageFaces: nurbs.cageFaces, stats: { history, ms: { map: tMap, total: performance.now() - t0 } } };
}
