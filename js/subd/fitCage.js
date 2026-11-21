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
import { bern, evalAt, D } from './bezier.js';

const SELF = [1 / 6, 1 / 2, 5 / 6];                 // a face's own points for the two-sided fit: 3 x 3 (midpoint rule)

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
  const nz = Math.floor((hi[2] - lo[2]) / cell) + 3;
  const nearest = (m) => {
    const cx = cellOf(m[0], 0), cy = cellOf(m[1], 1), cz = cellOf(m[2], 2);
    let best = -1, bd = Infinity;
    const look = (ix, iy, iz) => {
      if (ix < 0 || iy < 0 || iz < 0 || ix >= nx || iy >= ny || iz >= nz) return;
      const l = hash.get(key(ix, iy, iz));
      if (l) for (const i of l) {
        const d = (S[i * 3] - m[0]) ** 2 + (S[i * 3 + 1] - m[1]) ** 2 + (S[i * 3 + 2] - m[2]) ** 2;
        if (d < bd) { bd = d; best = i; }
      }
    };
    // shells of cells round the point's cell until nothing outside them can be closer; past the grid, all is seen
    const reach = Math.max(cx, nx - 1 - cx, cy, ny - 1 - cy, cz, nz - 1 - cz);
    for (let r = 0; r <= reach; r++) {
      for (let dx = -r; dx <= r; dx++) for (let dy = -r; dy <= r; dy++) {
        if (Math.abs(dx) === r || Math.abs(dy) === r) for (let dz = -r; dz <= r; dz++) look(cx + dx, cy + dy, cz + dz);
        else { look(cx + dx, cy + dy, cz - r); look(cx + dx, cy + dy, cz + r); }
      }
      if (best >= 0 && bd <= (r * cell) ** 2) break;  // nothing outside the searched block is closer
    }
    const f = at[best], g = best - f * grid * grid;
    return [f, Math.floor(g / grid) / (grid - 1), (g % grid) / (grid - 1)];
  };
  return { nearest, diag };
}

/** sampleIndex per group of faces: nearest(m, group) searches only that group's faces. */
function groupedIndex(poles, sizes, grid, faceGroup) {
  const lists = new Map();
  faceGroup.forEach((g, f) => { if (!lists.has(g)) lists.set(g, []); lists.get(g).push(f); });
  const sub = new Map();
  let diag2 = 0;
  for (const [g, fs] of lists) {
    const idx = sampleIndex(fs.map((f) => poles[f]), fs.map((f) => sizes[f]), grid);
    sub.set(g, { fs, idx });
    diag2 = Math.max(diag2, idx.diag);
  }
  // the whole surface's diagonal, for the fit's relative error
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const P of poles) for (let i = 0; i < P.length; i += 3) for (let c = 0; c < 3; c++) { lo[c] = Math.min(lo[c], P[i + c]); hi[c] = Math.max(hi[c], P[i + c]); }
  return {
    nearest: (m, g) => { const s = sub.get(g); const [f, u, v] = s.idx.nearest(m); return [s.fs[f], u, v]; },
    diag: Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) || diag2,
  };
}

/**
 * Closest points: foot[i] = { fi, u, v, d2 } for points[i]. Warm-started from foot[i] when it is set (last
 * round's); the nearest sample's face is tried as well when that ends on a face edge or there is none.
 */
function footpoints(poles, sizes, points, foot, index, weights = null, pointGroup = null) {
  let worst = 0, sum = 0, searched = 0;
  points.forEach((m, i) => {
    let best = null;
    if (foot[i]) {
      const f = foot[i].fi, r = project(sizes[f], poles[f], m, foot[i].u, foot[i].v);
      best = { fi: f, ...r };
    }
    if (!best || best.edge) {
      searched++;
      const [f, u, v] = index.nearest(m, pointGroup ? pointGroup[i] : undefined), r = project(sizes[f], poles[f], m, u, v);
      if (!best || r.d2 < best.d2) best = { fi: f, ...r };
    }
    foot[i] = best;
    worst = Math.max(worst, best.d2); sum += (weights ? weights[i] : 1) * best.d2;
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
 * Fit. points: [[x, y, z]...] on the mesh; weights (optional, one per point): the area each point stands for, so
 * the fit minimizes the integral of squared distance over the mesh (unweighted: over its vertices). Returns { vertices (fitted cage), faces (NURBS), loops (each
 * NURBS face's boundary: [vertex, u, v] round its square), stats }.
 */
export function fitCage(cage, points, options = {}) {
  const t0 = performance.now();
  return fitMap(subdLinearMap(cage.vertices.length, cage.faces), cage.vertices, cage.faces, points, options, t0);
}

/**
 * The fit on a given linear map (subdLinearMap, or groupLinearMap for a cage in groups). pointGroup (optional, one
 * per point, with map.faceGroup): each point's closest point is sought on its own group's faces only, so a point
 * near a border never pulls the face across it.
 *
 * project (optional, (group, point) -> the closest point on the input mesh's part for that group) makes the fit
 * two-sided: besides the input's points pulling the surface (the integral of squared distance over the input),
 * each face's own points are pulled to the input (the same integral over the surface, 3 x 3 points per face, each
 * carrying its share of the face's area). A face that the input's points miss (a small one at a sharp tip) is
 * then still held to the mesh, and a flap off it costs what it should.
 */
export function fitMap(map, vertices, cageFaces, points, { rounds = 5, grid = 7, weights = null, pointGroup = null, project = null } = {}, t0 = performance.now()) {
  const N = vertices.length;
  const sizes = map.faces.map((f) => f.uCount);
  // the normal equations' pattern: each face's support, and each cage vertex's Laplacian stencil
  const adj = Array.from({ length: N }, () => new Set());
  const clique = (ids) => { for (const a of ids) for (const b of ids) if (a !== b) adj[a].add(b); };
  for (const f of map.faces) clique(f.ids);
  const neighbours = Array.from({ length: N }, () => new Set());
  for (const q of cageFaces) for (let k = 0; k < 4; k++) { neighbours[q[k]].add(q[(k + 1) % 4]); neighbours[q[(k + 1) % 4]].add(q[k]); }
  neighbours.forEach((nb, j) => clique([j, ...nb]));
  const A = envelope(N, adj);
  const ms = { map: performance.now() - t0, samples: 0, foot: 0, assemble: 0, solve: 0 };
  let tick = performance.now();
  const lap = (k) => { const t = performance.now(); ms[k] += t - tick; tick = t; };

  let X = vertices.map((p) => [...p]);
  const X0 = vertices.map((p) => [...p]);
  const history = [];
  const foot = new Array(points.length);
  const totalWeight = weights ? weights.reduce((s, w) => s + w, 0) : points.length;
  if (project && !weights) throw new Error('fitMap: the two-sided fit needs area weights on the points');
  let poles = null;
  for (let round = 0; round <= rounds; round++) {
    tick = performance.now();
    poles = applyLinearMap(map, X);
    const index = pointGroup ? groupedIndex(poles, sizes, grid, map.faceGroup) : sampleIndex(poles, sizes, grid);
    lap('samples');
    const { worst, sum, searched } = footpoints(poles, sizes, points, foot, index, weights, pointGroup);
    lap('foot');
    history.push({ round, max: Math.sqrt(worst) / index.diag, rms: Math.sqrt(sum / totalWeight) / index.diag, searched });
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
      if (!list.length && !project) return;
      const { uCount: n, ids, W } = map.faces[fi], np = n * n, nl = ids.length;
      const G = new Float64Array(np * np), R = new Float64Array(np * 3), b = new Float64Array(np);
      // the input's points on this face, then (two-sided) the face's own points pulled to the input
      const rows = list.map((i) => [foot[i].u, foot[i].v, points[i], weights ? weights[i] : 1]);
      if (project) {
        for (const u of SELF) for (const v of SELF) {
          const d = evalAt(n, poles[fi], u, v), at = [d[0], d[1], d[2]];
          const area = Math.hypot(d[4] * d[8] - d[5] * d[7], d[5] * d[6] - d[3] * d[8], d[3] * d[7] - d[4] * d[6]) / 9;
          rows.push([u, v, project(map.faceGroup ? map.faceGroup[fi] : 0, at), area]);
        }
      }
      for (const [fu, fv, m, w] of rows) {
        bern(n - 1, fu, bu); bern(n - 1, fv, bv);
        for (let a = 0; a < n; a++) for (let c = 0; c < n; c++) b[a * n + c] = bu[a] * bv[c];
        for (let p = 0; p < np; p++) {
          const bp = w * b[p];
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
  return { vertices: X, faces, loops: map.loops,
    stats: { history, subdivisions: map.subdivisions, envelope: A.size, ms: { ...ms, total: performance.now() - t0 } } };
}
