/**
 * fastNurbs.js — a Catmull-Clark cage as NURBS faces in closed form: regular faces are their bicubic patches
 * (B-spline to Bezier conversion of the 4 x 4 neighbourhood), and the faces around each star point are the bi-6 G2
 * cap of js/subd/kpCap.js, a precomputed matrix per valence applied to the data along the cap's outer sides. No
 * iteration: the cost is a few small matrix products per face.
 *
 * The cap needs one layer of regular points round its star point (no face with two star corners). Where star points
 * are closer, one Catmull-Clark step separates them and only the faces round them are output refined (plan()).
 *
 * Everything is linear in the cage: subdLinearMap gives each output pole as fixed weights on cage vertices.
 *
 * Output faces are Bezier patches as clamped B-splines: { uKnots, vKnots, uCount, vCount, poles } with poles
 * row-major, u from corner 0 to corner 1 of the face and v from corner 0 to corner 3, each with its boundary loop.
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

const UV = [[0, 0], [1, 0], [1, 1], [0, 1]];

/**
 * What to output, face by face. A cage whose stars are isolated is output as it is. Otherwise one Catmull-Clark
 * step isolates every star (each new face has one original vertex), but only the faces round a star that shares a
 * face with another star (a dipole star) are output refined, as their four children: elsewhere the limit surface is
 * the coarse B-spline, so coarse bicubics and coarse caps stay, and their sides next to refined faces carry the same
 * jet as the children (each child's three rows next to the side come from B-spline stencils of the side's regular
 * ends), so they meet G2 at a T-junction.
 * Returns { levels: [{F, T}] (coarse, and fine when refined), stencils (fine vertex -> cage weights) or null,
 * outputs: [{level, face, star (cap) or -1 (bicubic), loop: [[vertex, u, v], ...]}], subdivisions }. Loop vertex ids
 * are the finest level's (a cage vertex keeps its id; a side's midpoint is that edge's new point).
 */
function plan(nv, cageFaces) {
  const F0 = cageFaces.map((q) => [...q]), T0 = topology(nv, F0);
  const loopOf = (q) => q.map((v, k) => [v, ...UV[k]]);
  const starOf = (q, T) => { const s = q.filter((v) => T.valence[v] !== 4); return s.length ? s[0] : -1; };
  if (isolated(F0, T0)) {
    return { levels: [{ F: F0, T: T0 }], stencils: null, subdivisions: 0,
      outputs: F0.map((q, f) => ({ level: 0, face: f, star: starOf(q, T0), loop: loopOf(q) })) };
  }
  const s = stepStencils(nv, Int32Array.from(F0.flat()), new Float64Array(F0.length * 8));
  const F1 = Array.from({ length: s.faces.length / 4 }, (_, f) => [s.faces[f * 4], s.faces[f * 4 + 1], s.faces[f * 4 + 2], s.faces[f * 4 + 3]]);
  const T1 = topology(s.rows.length, F1);
  const dipole = new Set();
  F0.forEach((q) => { const st = q.filter((v) => T0.valence[v] !== 4); if (st.length > 1) st.forEach((v) => dipole.add(v)); });
  const refined = F0.map((q) => q.some((v) => dipole.has(v)));
  const outputs = [];
  F0.forEach((q, f) => {
    if (refined[f]) {
      for (let k = 0; k < 4; k++) outputs.push({ level: 1, face: 4 * f + k, star: starOf(F1[4 * f + k], T1), loop: loopOf(F1[4 * f + k]) });
      return;
    }
    const loop = [];
    for (let k = 0; k < 4; k++) {
      loop.push([q[k], ...UV[k]]);
      const [g] = T0.sideOwner.get(`${q[(k + 1) % 4]},${q[k]}`);
      if (refined[g]) loop.push([F1[4 * f + k][1], (UV[k][0] + UV[(k + 1) % 4][0]) / 2, (UV[k][1] + UV[(k + 1) % 4][1]) / 2]);
    }
    outputs.push({ level: 0, face: f, star: starOf(q, T0), loop });
  });
  return { levels: [{ F: F0, T: T0 }, { F: F1, T: T1 }], stencils: s.rows, subdivisions: 1, outputs };
}

/** For each (level, star) of the plan's caps, the cap's faces computed once: Map "level,star" -> Map face -> poles. */
function capsOf(pl, run) {
  const caps = new Map();
  for (const o of pl.outputs) {
    if (o.star < 0) continue;
    const key = `${o.level},${o.star}`;
    if (!caps.has(key)) caps.set(key, new Map(run(o.level, o.star)));
  }
  return caps;
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

/**
 * The conversion: { faces (Bezier patches, the plan's order), loops ([[vertex, u, v], ...] per face: the face's
 * boundary, corners and any side midpoints where a refined neighbour meets it), stats }.
 */
export function fastSubdToNurbs(vertices, cageFaces, options = {}) {
  const t0 = performance.now();
  cageTopology(vertices.length, cageFaces);        // refuses open, non-manifold or non-quad cages
  const pl = plan(vertices.length, cageFaces);
  const V = [vertices];
  if (pl.stencils) V.push(pl.stencils.map((row) => {
    const x = [0, 0, 0];
    for (const [j, w] of row) { x[0] += w * vertices[j][0]; x[1] += w * vertices[j][1]; x[2] += w * vertices[j][2]; }
    return x;
  }));
  const t1 = performance.now();
  const caps = capsOf(pl, (level, star) => capFaces(V[level], pl.levels[level].F, pl.levels[level].T, star, 3, options.variant));
  const faces = pl.outputs.map(({ level, face, star }) => {
    const { F, T } = pl.levels[level];
    const poles = star < 0 ? bezierNet(V[level], regularGrid(F[face], T), 4, 3) : caps.get(`${level},${star}`).get(face);
    const p = star < 0 ? 3 : 6;
    return { uKnots: p === 3 ? K3 : K6, vKnots: p === 3 ? K3 : K6, uCount: p + 1, vCount: p + 1, poles: Array.from(poles), weights: null };
  });
  const t2 = performance.now();
  return { faces, loops: pl.outputs.map((o) => o.loop),
    stats: { subdivisions: pl.subdivisions, faces: faces.length, stars: caps.size, ms: { setup: t1 - t0, build: t2 - t1, total: t2 - t0 } } };
}

/**
 * The conversion as a linear map: faces[f] (fastSubdToNurbs' order) = { uCount, vCount, ids, W } with pole p = sum
 * over a of W[p * ids.length + a] * cage vertex ids[a]; loops as fastSubdToNurbs gives them. Built directly: the
 * regular faces' fixed 16 x 16 weights, each cap evaluated once on a basis of its star's 2-ring, refined faces
 * composed with the subdivision stencils.
 */
export function subdLinearMap(nv, cageFaces, options = {}) {
  cageTopology(nv, cageFaces);
  const pl = plan(nv, cageFaces);
  const compose = (level, fineIds, M, np) => {
    if (level === 0) return { ids: Int32Array.from(fineIds), W: M };
    const rows = pl.stencils, col = new Map();
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
  // regular faces: pole (a, b) = sum B2BB[a][i] B2BB[b][j] G[i][j]
  const R16 = new Float64Array(256);
  for (let a = 0; a < 4; a++) for (let b = 0; b < 4; b++) for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) R16[(a * 4 + b) * 16 + i * 4 + j] = B2BB[a][i] * B2BB[b][j];
  // caps: run once with V = the identity on the star's 2-ring (at the cap's level)
  const Vloc = [new Array(nv), pl.stencils ? new Array(pl.stencils.length) : null];
  const locals = new Map();
  const caps = capsOf(pl, (level, v) => {
    const { F, T } = pl.levels[level], local = new Set();
    for (const [f] of T.facesAt[v]) for (const w of F[f]) for (const [g] of T.facesAt[w]) for (const x of F[g]) local.add(x);
    const L = [...local], dim = L.length, Vl = Vloc[level];
    L.forEach((x, a) => { const e = new Float64Array(dim); e[a] = 1; Vl[x] = e; });
    const out = capFaces(Vl, F, T, v, dim, options.variant);
    for (const x of L) Vl[x] = undefined;
    locals.set(`${level},${v}`, L);
    return out;
  });
  const faces = pl.outputs.map(({ level, face, star }) => {
    const { F, T } = pl.levels[level];
    if (star >= 0) return { uCount: 7, vCount: 7, ...compose(level, locals.get(`${level},${star}`), caps.get(`${level},${star}`).get(face), 49) };
    const G = regularGrid(F[face], T), uniq = [...new Set(G)];
    let M = R16;
    if (uniq.length !== 16) {                       // a vertex twice in the neighbourhood (tiny cages): sum its columns
      M = new Float64Array(16 * uniq.length);
      const at = new Map(uniq.map((v, a) => [v, a]));
      for (let p = 0; p < 16; p++) for (let g = 0; g < 16; g++) M[p * uniq.length + at.get(G[g])] += R16[p * 16 + g];
    }
    return { uCount: 4, vCount: 4, ...compose(level, uniq, M, 16) };
  });
  return { faces, loops: pl.outputs.map((o) => o.loop), subdivisions: pl.subdivisions };
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
