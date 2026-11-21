/**
 * fastNurbs.js — a Catmull-Clark cage as NURBS faces in closed form: regular faces are their bicubic patches
 * (B-spline to Bezier conversion of the 4 x 4 neighbourhood), and the faces around each star point are the bi-6 G2
 * cap of js/subd/kpCap.js, a precomputed matrix per valence applied to the data along the cap's outer sides. No
 * iteration: the cost is a few small matrix products per face.
 *
 * The cap needs one layer of regular points round its star point (no face with two star corners); a cage whose
 * star points are closer than that is subdivided (Catmull-Clark) until they are not.
 *
 * Everything is linear in the cage: subdLinearMap gives each output pole as fixed weights on cage vertices.
 *
 * Output faces are Bezier patches as clamped B-splines: { uKnots, vKnots, uCount, vCount, poles } with poles
 * row-major, u from corner 0 to corner 1 of the face and v from corner 0 to corner 3.
 */
import { cageTopology, stepStencils } from './catmullClark.js';
import { capOperator, applyCap, bernstein } from './kpCap.js';

const K3 = [0, 0, 0, 0, 1, 1, 1, 1];
const K6 = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1];
const B2BB = [[1, 4, 1, 0], [0, 4, 2, 0], [0, 2, 4, 0], [0, 1, 4, 1]].map((r) => r.map((x) => x / 6));

/** Inverse of the 7 x 7 Bernstein collocation matrix at t = 0, 1/6, ..., 1. */
const BINV6 = (() => {
  const n = 7, A = Array.from({ length: n }, (_, a) => [...bernstein(6, a / 6, 0)[0], ...Array.from({ length: n }, (__, j) => (a === j ? 1 : 0))]);
  for (let c = 0; c < n; c++) {
    let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    [A[c], A[p]] = [A[p], A[c]];
    const d = A[c][c]; for (let j = 0; j < 2 * n; j++) A[c][j] /= d;
    for (let r = 0; r < n; r++) if (r !== c) { const f = A[r][c]; for (let j = 0; j < 2 * n; j++) A[r][j] -= f * A[c][j]; }
  }
  return A.map((row) => row.slice(n));
})();

function topology(nv, F) {
  const nextToPrev = Array.from({ length: nv }, () => new Map()), facesAt = Array.from({ length: nv }, () => []);
  const corner = new Map();                         // "c,a,b" -> fourth vertex of the face through c with neighbours a, b
  const sideOwner = new Map();                      // "a,b" (directed, as in a loop) -> [face, side]
  F.forEach((q, f) => {
    for (let k = 0; k < 4; k++) {
      const v = q[k], next = q[(k + 1) % 4], prev = q[(k + 3) % 4];
      nextToPrev[v].set(next, prev);
      facesAt[v].push([f, k]);
      corner.set(`${v},${next},${prev}`, q[(k + 2) % 4]);
      sideOwner.set(`${v},${next}`, [f, k]);
    }
  });
  const ring = (v) => {
    const m = nextToPrev[v], start = m.keys().next().value, r = [start];
    for (let x = m.get(start); x !== start; x = m.get(x)) r.push(x);
    return r;
  };
  const valence = facesAt.map((l) => l.length);
  return { ring, valence, facesAt, corner, sideOwner };
}

/**
 * The paper's condition: every star point is surrounded by one layer of regular points, i.e. no face has two star
 * corners. The faces across a star face's outer sides may be other stars' faces: the cap's data there is the
 * B-spline's along the shared side (sideGrid), so two caps meeting on a side both meet that jet, and each other, G2.
 */
function isolated(F, T) {
  return F.every((q) => q.filter((v) => T.valence[v] !== 4).length <= 1);
}

/**
 * Subdivide until isolated. Returns the final faces and topology, the step count, and (withRows) each final
 * vertex as weights on cage vertices: rows[i] = [[cage vertex, weight], ...].
 */
function refine(nv, cageFaces, maxSubdivisions, withRows) {
  let F = cageFaces.map((q) => [...q]), T = topology(nv, F), steps = 0, rows = null;
  const stencils = [];
  while (!isolated(F, T)) {
    if (steps === maxSubdivisions) throw new Error('SUBD_STARS_TOO_CLOSE: star points stay too close after subdividing');
    const s = stepStencils(nv, Int32Array.from(F.flat()), new Float64Array(F.length * 8));
    stencils.push(s.rows);
    nv = s.rows.length;
    F = Array.from({ length: s.faces.length / 4 }, (_, f) => [s.faces[f * 4], s.faces[f * 4 + 1], s.faces[f * 4 + 2], s.faces[f * 4 + 3]]);
    T = topology(nv, F);
    steps++;
  }
  if (withRows && steps) {
    rows = stencils[0];
    for (let l = 1; l < stencils.length; l++) {
      const prev = rows;
      rows = stencils[l].map((row) => {
        const w = new Map();
        for (const [j, a] of row) for (const [c, b] of prev[j]) w.set(c, (w.get(c) ?? 0) + a * b);
        return [...w];
      });
    }
  }
  return { F, T, steps, stencils, rows };
}

/** Vertex ids of a regular face's 4 x 4 control-point neighbourhood, G[i * 4 + j], i along u and j along v. */
function regularGrid(q, T) {
  const opp = (c, x) => { const r = T.ring(c); return r[(r.indexOf(x) + 2) % 4]; };
  const G = new Array(16);
  const [c0, c1, c2, c3] = q;
  G[5] = c0; G[9] = c1; G[10] = c2; G[6] = c3;
  const fill = (c, a, b, pa, pb, pd) => {          // c's neighbours a (along one grid line) and b (the other)
    const oa = opp(c, a), ob = opp(c, b);
    G[pa] = oa; G[pb] = ob; G[pd] = T.corner.get(`${c},${oa},${ob}`) ?? T.corner.get(`${c},${ob},${oa}`);
  };
  fill(c0, c1, c3, 1, 4, 0);
  fill(c1, c0, c2, 13, 8, 12);
  fill(c2, c1, c3, 11, 14, 15);
  fill(c3, c0, c2, 7, 2, 3);
  return G;
}

/**
 * Vertex ids round side k of face q, G[i * 3 + j] for i = 0..3 along the side and j = 0..2 across it (j = 0 the
 * row outside the face), in the face's frame turned so that side k is side 0. The B-spline's three Bezier rows next
 * to the side need only these: the control points round the side's two ends (both regular), so the face's far
 * corners may be star points.
 */
function sideGrid(q, k, T) {
  const [c0, c1, c2, c3] = [0, 1, 2, 3].map((j) => q[(k + j) % 4]);
  const opp = (c, x) => { const r = T.ring(c); return r[(r.indexOf(x) + 2) % 4]; };
  const four = (c, a, b) => T.corner.get(`${c},${a},${b}`) ?? T.corner.get(`${c},${b},${a}`);
  const G = new Array(12);
  G[4] = c0; G[7] = c1; G[8] = c2; G[5] = c3;
  G[1] = opp(c0, c1); G[3] = opp(c0, c3); G[0] = four(c0, G[1], G[3]); G[2] = four(c0, G[1], c3);
  G[10] = opp(c1, c0); G[6] = opp(c1, c2); G[9] = four(c1, G[10], G[6]); G[11] = four(c1, G[10], c2);
  return G;
}

/** Bezier net (4 x 4 x dim) from control points V[G[i * cols + j]] with B-spline to Bezier along both directions; rows b < cols only. */
function bezierNet(V, G, cols, dim) {
  const net = new Float64Array(16 * dim);
  for (let a = 0; a < 4; a++) for (let b = 0; b < cols; b++) {
    const o = (a * 4 + b) * dim;
    for (let i = 0; i < 4; i++) {
      const wa = B2BB[a][i];
      if (!wa) continue;
      for (let j = 0; j < cols; j++) {
        const w = wa * B2BB[b][j];
        if (!w) continue;
        const p = V[G[i * cols + j]];
        for (let c = 0; c < dim; c++) net[o + c] += w * p[c];
      }
    }
  }
  return net;
}

/** Position, along-side and two cross derivatives of a bicubic net on its side 0 at parameter s. */
function sideJets(net, s, dim) {
  const Bu = bernstein(3, s, 1), Bv = bernstein(3, 0, 2);
  const ev = (a, b) => {
    const x = new Float64Array(dim);
    for (let i = 0; i < 4; i++) for (let j = 0; j < 3; j++) {
      const w = Bu[a][i] * Bv[b][j];
      if (!w) continue;
      const o = (i * 4 + j) * dim;
      for (let c = 0; c < dim; c++) x[c] += w * net[o + c];
    }
    return x;
  };
  return { p: ev(0, 0), s: ev(1, 0), t: ev(0, 1), tt: ev(0, 2) };
}

/**
 * The bi-6 cap of star point v: for each face round v, [face, poles (49 x dim, face frame)]. V[i] is vertex i as a
 * vector of length dim (positions, or columns of a basis for the linear map).
 */
function capFaces(V, F, T, v, dim, variant = {}) {
  const n = T.valence[v];
  const nSign = variant.nSign ?? 1, eSign = variant.eSign ?? 1;
  const op = capOperator(n, { flipSpoke: !!variant.flipSpoke });
  const K = op.constants;
  // faces round v counter-clockwise, each as [face, position of v in its loop]: F_k = [v, w_k, x_k, w_(k+1)]
  const around = new Map(T.facesAt[v].map(([f, k]) => [F[f][(k + 1) % 4], [f, k]]));
  const order = [T.facesAt[v][0]];
  while (order.length < n) {
    const [f, k] = order[order.length - 1];
    order.push(around.get(F[f][(k + 3) % 4]));
  }
  const data = new Float64Array(op.ndata * dim);
  // the star point: Catmull-Clark limit point
  for (let c = 0; c < dim; c++) data[c] = K.limit.centre * V[v][c];
  for (const [f, k] of order) {
    const w = V[F[f][(k + 1) % 4]], x = V[F[f][(k + 2) % 4]];
    for (let c = 0; c < dim; c++) data[c] += K.limit.edge * w[c] + K.limit.diagonal * x[c];
  }
  order.forEach(([f, k], kk) => {
    const q = F[f], w0 = q[(k + 1) % 4], x = q[(k + 2) % 4], w1 = q[(k + 3) % 4];
    // Bezier rows 0..2 of the outer side a -> b of F_k (loop order) at positions 0..6, from the B-spline across it
    const rowsFor = (a, b, tOf) => {
      const [g, kg] = T.sideOwner.get(`${b},${a}`);
      const net = bezierNet(V, sideGrid(F[g], kg, T), 3, dim);
      const vals = [new Float64Array(7 * dim), new Float64Array(7 * dim), new Float64Array(7 * dim)];
      for (let j = 0; j <= 6; j++) {
        const t = tOf(j / 6);                       // t: 0 at the far corner, 1 at the spoke's end
        // the neighbour's side runs b -> a; its parameter is 0 at b
        const sg = a === x ? 1 - t : t;             // U1: a = w0, b = x  -> s_g = t;  V1: a = x, b = w1 -> s_g = 1 - t
        const J = sideJets(net, sg, dim);
        const dsdt = a === x ? -1 : 1, dB = K.dBorder(t), eB = K.eBorder(t);
        for (let c = 0; c < dim; c++) {
          const bn = -J.t[c], R0 = J.p[c], R1 = R0 + bn / 6;
          vals[0][j * dim + c] = R0; vals[1][j * dim + c] = R1;
          vals[2][j * dim + c] = 2 * R1 - R0 + (J.tt[c] + eSign * eB * dsdt * J.s[c] + nSign * dB * bn) / 30;
        }
      }
      // values at t = 0, 1/6, ..., 1 to Bezier coefficients
      return vals.map((val) => {
        const out = new Float64Array(7 * dim);
        for (let i = 0; i < 7; i++) for (let a2 = 0; a2 < 7; a2++) {
          const w = BINV6[i][a2];
          if (Math.abs(w) < 1e-300) continue;
          for (let c = 0; c < dim; c++) out[i * dim + c] += w * val[a2 * dim + c];
        }
        return out;
      });
    };
    const u1 = rowsFor(w0, x, (vprime) => 1 - vprime);   // side u = 1, position j = v'
    const v1 = rowsFor(x, w1, (u) => 1 - u);             // side v = 1, position i = u
    for (let r = 0; r < 3; r++) for (let j = 0; j <= 6; j++) data.set(u1[r].subarray(j * dim, (j + 1) * dim), (1 + 33 * kk + r * 7 + j) * dim);
    for (let r = 0; r < 3; r++) for (let i = 0; i <= 3; i++) data.set(v1[r].subarray(i * dim, (i + 1) * dim), (1 + 33 * kk + 21 + r * 4 + i) * dim);
  });
  const patches = applyCap(op, data, dim);
  return order.map(([f, k], kk) => {
    // cap frame: u along v -> loop[k+1], v' along v -> loop[k+3]; face frame: U = corner0 -> corner1, V = corner0 -> corner3
    const P = patches[kk], net = new Float64Array(49 * dim);
    for (let i = 0; i <= 6; i++) for (let j = 0; j <= 6; j++) {
      const [I, J] = k === 0 ? [i, j] : k === 1 ? [6 - j, i] : k === 2 ? [6 - i, 6 - j] : [j, 6 - i];
      net.set(P.subarray((i * 7 + j) * dim, (i * 7 + j + 1) * dim), (I * 7 + J) * dim);
    }
    return [f, net];
  });
}

export function fastSubdToNurbs(vertices, cageFaces, options = {}) {
  const t0 = performance.now();
  cageTopology(vertices.length, cageFaces);        // refuses open, non-manifold or non-quad cages
  const { F, T, steps, stencils } = refine(vertices.length, cageFaces, options.maxSubdivisions ?? 2, false);
  let V = vertices;
  for (const rows of stencils) {
    const P = V;
    V = rows.map((row) => {
      const x = [0, 0, 0];
      for (const [j, w] of row) { x[0] += w * P[j][0]; x[1] += w * P[j][1]; x[2] += w * P[j][2]; }
      return x;
    });
  }
  const t1 = performance.now();
  const out = new Array(F.length);
  let stars = 0;
  F.forEach((q, f) => { if (q.every((v) => T.valence[v] === 4)) out[f] = { uKnots: K3, vKnots: K3, uCount: 4, vCount: 4, poles: bezierNet(V, regularGrid(q, T), 4, 3) }; });
  for (let v = 0; v < V.length; v++) {
    if (T.valence[v] === 4) continue;
    stars++;
    for (const [f, poles] of capFaces(V, F, T, v, 3, options.variant)) out[f] = { uKnots: K6, vKnots: K6, uCount: 7, vCount: 7, poles };
  }
  const t2 = performance.now();
  return { faces: out.map((o) => ({ ...o, poles: Array.from(o.poles), weights: null })), vertices: V, cageFaces: F,
    stats: { subdivisions: steps, faces: F.length, stars, ms: { setup: t1 - t0, build: t2 - t1, total: t2 - t0 } } };
}

/**
 * The conversion as a linear map. faces[f] (same order as fastSubdToNurbs) = { uCount, vCount, ids, W } with
 * pole p = sum over a of W[p * ids.length + a] * cage vertex ids[a]; cageFaces: the output faces' corners (the
 * subdivided cage when star points were too close). Built directly: the regular faces' fixed 16 x 16 weights, each
 * cap evaluated once on a basis of its star's 2-ring, subdivision composed in.
 */
export function subdLinearMap(nv, cageFaces, options = {}) {
  cageTopology(nv, cageFaces);
  const { F, T, rows, steps } = refine(nv, cageFaces, options.maxSubdivisions ?? 2, true);
  const compose = (fineIds, M, np) => {
    if (!rows) return { ids: Int32Array.from(fineIds), W: M };
    const col = new Map();
    for (const i of fineIds) for (const [c] of rows[i]) if (!col.has(c)) col.set(c, col.size);
    const nc = col.size, nf = fineIds.length, W = new Float64Array(np * nc);
    for (let a = 0; a < nf; a++) {
      const r = rows[fineIds[a]].map(([c, w]) => [col.get(c), w]);
      for (let p = 0; p < np; p++) {
        const m = M[p * nf + a];
        if (m) for (const [cc, w] of r) W[p * nc + cc] += m * w;
      }
    }
    return { ids: Int32Array.from(col.keys()), W };
  };
  const out = new Array(F.length);
  // regular faces: pole (a, b) = sum B2BB[a][i] B2BB[b][j] G[i][j]
  const R16 = new Float64Array(256);
  for (let a = 0; a < 4; a++) for (let b = 0; b < 4; b++) for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) R16[(a * 4 + b) * 16 + i * 4 + j] = B2BB[a][i] * B2BB[b][j];
  F.forEach((q, f) => {
    if (!q.every((v) => T.valence[v] === 4)) return;
    const G = regularGrid(q, T), uniq = [...new Set(G)];
    let M = R16;
    if (uniq.length !== 16) {                       // a vertex twice in the neighbourhood (tiny cages): sum its columns
      M = new Float64Array(16 * uniq.length);
      const at = new Map(uniq.map((v, a) => [v, a]));
      for (let p = 0; p < 16; p++) for (let g = 0; g < 16; g++) M[p * uniq.length + at.get(G[g])] += R16[p * 16 + g];
    }
    out[f] = { uCount: 4, vCount: 4, ...compose(uniq, M, 16) };
  });
  // caps: run once with V = the identity on the star's 2-ring
  const Vloc = new Array(T.valence.length);
  for (let v = 0; v < T.valence.length; v++) {
    if (T.valence[v] === 4) continue;
    const local = new Set();
    for (const [f] of T.facesAt[v]) for (const w of F[f]) for (const [g] of T.facesAt[w]) for (const x of F[g]) local.add(x);
    const L = [...local], dim = L.length;
    L.forEach((x, a) => { const e = new Float64Array(dim); e[a] = 1; Vloc[x] = e; });
    for (const [f, poles] of capFaces(Vloc, F, T, v, dim, options.variant)) out[f] = { uCount: 7, vCount: 7, ...compose(L, poles, 49) };
    for (const x of L) Vloc[x] = undefined;
  }
  return { faces: out, cageFaces: F, subdivisions: steps };
}

/** Poles of every face for cage positions X ([[x, y, z]...]) through the map: Float64Array per face, 3 per pole. */
export function applyLinearMap(map, X) {
  return map.faces.map(({ uCount, vCount, ids, W }) => {
    const nl = ids.length, np = uCount * vCount, P = new Float64Array(np * 3);
    for (let l = 0; l < nl; l++) {
      const x = X[ids[l]];
      for (let p = 0; p < np; p++) {
        const w = W[p * nl + l];
        if (w) { P[p * 3] += w * x[0]; P[p * 3 + 1] += w * x[1]; P[p * 3 + 2] += w * x[2]; }
      }
    }
    return P;
  });
}

/** Clamped Bezier knot vector of degree p. */
export const bezierKnots = (p) => [...Array(p + 1).fill(0), ...Array(p + 1).fill(1)];
