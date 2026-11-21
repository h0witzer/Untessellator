/**
 * fastNurbs.js — a Catmull-Clark cage as NURBS faces in closed form: regular faces are their bicubic patches
 * (B-spline to Bezier conversion of the 4 x 4 neighbourhood), and the faces around each star point are the bi-6 G2
 * cap of js/subd/kpCap.js, a precomputed matrix per valence applied to the data along the cap's outer sides. No
 * iteration: the cost is a few small matrix products per face.
 *
 * The cap needs its star point's faces and the faces across their outer sides to be otherwise regular; a cage
 * whose star points are closer than that is subdivided once (Catmull-Clark) until they are not.
 *
 * Output faces are Bezier patches as clamped B-splines: { uKnots, vKnots, uCount, vCount, poles } with poles
 * row-major, u from corner 0 to corner 1 of the face and v from corner 0 to corner 3.
 */
import { cageTopology, subdivide } from './catmullClark.js';
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

function topology(V, F) {
  const nv = V.length, nextToPrev = Array.from({ length: nv }, () => new Map()), facesAt = Array.from({ length: nv }, () => []);
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

/** The faces' neighbourhoods are fine for the cap: every star face has one star corner, every face across its outer sides is regular. */
function isolated(F, T) {
  for (let v = 0; v < T.valence.length; v++) {
    if (T.valence[v] === 4) continue;
    for (const [f, k] of T.facesAt[v]) {
      const q = F[f];
      for (let j = 1; j < 4; j++) if (T.valence[q[(k + j) % 4]] !== 4) return false;
      for (const s of [1, 2]) {                    // the outer sides q[k+1] -> q[k+2] and q[k+2] -> q[k+3]
        const a = q[(k + s) % 4], b = q[(k + s + 1) % 4], g = T.sideOwner.get(`${b},${a}`);
        if (!g || F[g[0]].some((x) => T.valence[x] !== 4)) return false;
      }
    }
  }
  return true;
}

/** Bicubic Bezier net (4 x 4, row-major u then v) of a regular face from its 4 x 4 control-point neighbourhood. */
function regularPatch(V, q, T) {
  const opp = (c, x) => { const r = T.ring(c); return r[(r.indexOf(x) + 2) % 4]; };
  const G = Array.from({ length: 4 }, () => new Array(4));
  const [c0, c1, c2, c3] = q;
  G[1][1] = c0; G[2][1] = c1; G[2][2] = c2; G[1][2] = c3;
  const fill = (c, a, b, pa, pb, pd) => {          // c's neighbours a (along one grid line) and b (the other)
    const oa = opp(c, a), ob = opp(c, b);
    G[pa[0]][pa[1]] = oa; G[pb[0]][pb[1]] = ob; G[pd[0]][pd[1]] = T.corner.get(`${c},${oa},${ob}`) ?? T.corner.get(`${c},${ob},${oa}`);
  };
  fill(c0, c1, c3, [0, 1], [1, 0], [0, 0]);
  fill(c1, c0, c2, [3, 1], [2, 0], [3, 0]);
  fill(c2, c1, c3, [2, 3], [3, 2], [3, 3]);
  fill(c3, c0, c2, [1, 3], [0, 2], [0, 3]);
  const net = new Float64Array(48);
  for (let a = 0; a < 4; a++) for (let b = 0; b < 4; b++) for (let c = 0; c < 3; c++) {
    let s = 0;
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) s += B2BB[a][i] * B2BB[b][j] * V[G[i][j]][c];
    net[(a * 4 + b) * 3 + c] = s;
  }
  return net;
}

/** Position, along-side and two cross derivatives of a bicubic net on its side k at parameter s (s along the loop). */
function sideJets(net, k, s) {
  const uv = [[s, 0], [1, s], [1 - s, 1], [0, 1 - s]][k];
  // [u-order, v-order, sign] for d/ds and d/dt (t into the face)
  const D = [
    { s: [1, 0, 1], t: [0, 1, 1], tt: [0, 2, 1] }, { s: [0, 1, 1], t: [1, 0, -1], tt: [2, 0, 1] },
    { s: [1, 0, -1], t: [0, 1, -1], tt: [0, 2, 1] }, { s: [0, 1, -1], t: [1, 0, 1], tt: [2, 0, 1] },
  ][k];
  const Bu = bernstein(3, uv[0], 2), Bv = bernstein(3, uv[1], 2);
  const ev = ([a, b, sign]) => {
    const x = [0, 0, 0];
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
      const w = sign * Bu[a][i] * Bv[b][j];
      for (let c = 0; c < 3; c++) x[c] += w * net[(i * 4 + j) * 3 + c];
    }
    return x;
  };
  return { p: ev([0, 0, 1]), s: ev(D.s), t: ev(D.t), tt: ev(D.tt) };
}

export function fastSubdToNurbs(vertices, cageFaces, options = {}) {
  const t0 = performance.now();
  let V = vertices.map((p) => [...p]), F = cageFaces.map((q) => [...q]);
  cageTopology(V.length, F);                       // refuses open, non-manifold or non-quad cages
  let T = topology(V, F), steps = 0;
  while (!isolated(F, T)) {
    if (steps === (options.maxSubdivisions ?? 2)) throw new Error('SUBD_STARS_TOO_CLOSE: star points stay too close after subdividing');
    const s = subdivide(V, F, 1);
    V = Array.from({ length: s.V.length / 3 }, (_, i) => [s.V[i * 3], s.V[i * 3 + 1], s.V[i * 3 + 2]]);
    F = Array.from({ length: s.F.length / 4 }, (_, f) => [s.F[f * 4], s.F[f * 4 + 1], s.F[f * 4 + 2], s.F[f * 4 + 3]]);
    T = topology(V, F);
    steps++;
  }
  const t1 = performance.now();
  const out = new Array(F.length);
  const regular = new Map();
  const patchOf = (f) => { if (!regular.has(f)) regular.set(f, regularPatch(V, F[f], T)); return regular.get(f); };
  F.forEach((q, f) => { if (q.every((v) => T.valence[v] === 4)) out[f] = { uKnots: K3, vKnots: K3, uCount: 4, vCount: 4, poles: patchOf(f) }; });
  const variant = options.variant ?? {};
  const nSign = variant.nSign ?? 1, eSign = variant.eSign ?? 1;
  let stars = 0;
  for (let v = 0; v < V.length; v++) {
    const n = T.valence[v];
    if (n === 4) continue;
    stars++;
    const op = capOperator(n, { flipSpoke: !!variant.flipSpoke });
    const K = op.constants;
    // faces round v counter-clockwise, each as [face, position of v in its loop]: F_k = [v, w_k, x_k, w_(k+1)]
    const around = new Map(T.facesAt[v].map(([f, k]) => [F[f][(k + 1) % 4], [f, k]]));
    const first = T.facesAt[v][0], order = [first];
    while (order.length < n) {
      const [f, k] = order[order.length - 1];
      order.push(around.get(F[f][(k + 3) % 4]));
    }
    const data = new Float64Array(op.ndata * 3);
    // the star point: Catmull-Clark limit point
    const lim = [0, 0, 0];
    for (let c = 0; c < 3; c++) lim[c] = K.limit.centre * V[v][c];
    for (const [f, k] of order) {
      const w = F[f][(k + 1) % 4], x = F[f][(k + 2) % 4];
      for (let c = 0; c < 3; c++) lim[c] += K.limit.edge * V[w][c] + K.limit.diagonal * V[x][c];
    }
    data.set(lim, 0);
    order.forEach(([f, k], kk) => {
      const q = F[f], w0 = q[(k + 1) % 4], x = q[(k + 2) % 4], w1 = q[(k + 3) % 4];
      const rowsFor = (a, b, tOf) => {            // the outer side a -> b of F_k (loop order), neighbour across: b -> a
        const [g, kg] = T.sideOwner.get(`${b},${a}`);
        const net = patchOf(g), rows = [[], [], []];
        for (let j = 0; j <= 6; j++) {
          const pos = j / 6, t = tOf(pos);         // t: 0 at the far corner, 1 at the spoke's end
          // the neighbour's side runs b -> a; its parameter is 0 at b
          const sg = a === x ? 1 - t : t;           // U1: a = w0, b = x  -> s_g = t;  V1: a = x, b = w1 -> s_g = 1 - t
          const J = sideJets(net, kg, sg);
          const dsdt = a === x ? -1 : 1;
          const bn = J.t.map((y) => -y), bnn = J.tt, bt = J.s.map((y) => dsdt * y);
          const R0 = J.p, R1 = R0.map((y, c) => y + bn[c] / 6);
          const dB = K.dBorder(t), eB = K.eBorder(t);
          const R2 = R1.map((y, c) => 2 * y - R0[c] + (bnn[c] + eSign * eB * bt[c] + nSign * dB * bn[c]) / 30);
          rows[0].push(R0); rows[1].push(R1); rows[2].push(R2);
        }
        // Bezier coefficients along the side, at positions 0..6
        return rows.map((vals) => Array.from({ length: 7 }, (_, i) => [0, 1, 2].map((c) => vals.reduce((s, val, a2) => s + BINV6[i][a2] * val[c], 0))));
      };
      const u1 = rowsFor(w0, x, (vprime) => 1 - vprime);   // side u = 1, position j = v'
      const v1 = rowsFor(x, w1, (u) => 1 - u);             // side v = 1, position i = u
      for (let r = 0; r < 3; r++) for (let j = 0; j <= 6; j++) data.set(u1[r][j], (1 + 33 * kk + r * 7 + j) * 3);
      for (let r = 0; r < 3; r++) for (let i = 0; i <= 3; i++) data.set(v1[r][i], (1 + 33 * kk + 21 + r * 4 + i) * 3);
    });
    const patches = applyCap(op, data);
    order.forEach(([f, k], kk) => {
      // cap frame: u along v -> loop[k+1], v' along v -> loop[k+3]; face frame: U = corner0 -> corner1, V = corner0 -> corner3
      const P = patches[kk], net = new Float64Array(49 * 3);
      for (let i = 0; i <= 6; i++) for (let j = 0; j <= 6; j++) {
        const [I, J] = k === 0 ? [i, j] : k === 1 ? [6 - j, i] : k === 2 ? [6 - i, 6 - j] : [j, 6 - i];
        net.set(P.subarray((i * 7 + j) * 3, (i * 7 + j) * 3 + 3), (I * 7 + J) * 3);
      }
      out[f] = { uKnots: K6, vKnots: K6, uCount: 7, vCount: 7, poles: net };
    });
  }
  const t2 = performance.now();
  return { faces: out.map((o) => ({ ...o, poles: Array.from(o.poles), weights: null })), vertices: V, cageFaces: F,
    stats: { subdivisions: steps, faces: F.length, stars, ms: { setup: t1 - t0, build: t2 - t1, total: t2 - t0 } } };
}
