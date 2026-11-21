/**
 * toNurbs.js — a Catmull-Clark cage as NURBS faces, one per cage face, G2 across every edge.
 *
 *   regular faces    one bicubic span, fitted to exact limit samples: the subdivision surface itself
 *   star faces       a face with a star-point corner (3, 5 or more faces meet): 2 x 2 polynomial pieces of
 *                    degree 6, C1 across the middle lines (knots 0^7 .5^5 1^7, 12 x 12 control points), fitted
 *                    to the target surface, then given the smallest smooth correction that makes every edge
 *                    between two star faces, and each star face's middle lines, join G2
 *
 * Held while solving: each star point's control point (its limit position), and along an edge to a regular
 * face the first three rows, which are set from that face's own position, first and second cross derivatives
 * (parametric C2 there, so G2 holds exactly). Free: everything else in a star face.
 *
 * Conditions along an edge between star faces f and g, at points of each edge half, with s the edge parameter
 * running away from the half's star point (unit per face side), t into f and tau into g:
 *   G0   the shared boundary rows are equal (so the faces share the boundary curve and its parameter)
 *   G1   f_t + g_tau = beta f_s
 *   G2   (f_s x f_t) . (g_tautau - f_tt + 2 beta f_st - beta^2 f_ss) = 0
 * with beta = 2 cos(2 pi / n) (1 - 2 s)^2 on a half that starts at a star point of n faces, 0 elsewhere. That is
 * g(s, tau) = f(s + beta tau, -tau) to second order, normal component only (tangential second-order terms are
 * free); beta vanishes with its slope at the edge's middle, where each face's pieces join only C1.
 *
 * The objective (defaults, measured 2026-11-21 on five cages against NURBS Proto's converter):
 *   target   TARGET 'natural': the cage subdivided twice, read as a bicubic B-spline, star corners at their
 *            limit points (js/subd/naturalNet.js). It is the limit surface except in the one level-2 cell at
 *            each star corner, where Catmull-Clark's own curvature misbehaves; correcting the exact limit
 *            ('limit') imports that flaw (star-point evenness 71 against 12 on the cube).
 *   metric   METRIC 'smooth': the correction's size plus its bending, integral |d|^2 + ELL^4 integral
 *            |d_uu|^2 + 2 |d_uv|^2 + |d_vv|^2, ELL = how far a correction spreads in face sides (NURBS Proto's
 *            discrete weight 3 corresponds to about 0.11). Measured alternatives: 'samples' (closest surface:
 *            ripples at star points), 'bending' (no size term: flattens corners), 'nproto' (its move metric).
 * Solved by stabilised SQP (multiplier estimates carried between steps, exact L1 merit line search) on the
 * Schur complement of the conditions, one block per edge or face, factored by js/subd/blockCholesky.js.
 * Measurement code for all of this lives outside the repo (bake-off harness).
 */
import { nzBasis } from '../nurbs.js';
import { cageTopology, limitGrids } from './catmullClark.js';
import { factorBlocks } from './blockCholesky.js';
import { naturalNets } from './naturalNet.js';

export const SUBD = {
  LEVELS: 4,          // limit samples per face side: 2^LEVELS + 1
  G1_SAMPLES: 8,      // per edge half: the G1 residual is a polynomial of degree 7 there
  G2_SAMPLES: 20,     // per edge half: the G2 residual is a polynomial of degree 19 there
  MAX_STEPS: 24,
  TARGET: 'natural',  // 'natural': the level-2 natural net; 'limit': the exact limit surface
  METRIC: 'smooth',   // 'smooth' | 'samples' (+ FAIR_MU variation) | 'bending' | 'nproto' (see the header)
  ANCHORED: Infinity, // steps that pull towards the fit; later steps make the least move (NURBS Proto: 6)
  ELL: 0.11,          // METRIC 'smooth': correction length in face sides (NURBS Proto's weight 3 is about 0.11)
  PIN_STAR: 1,        // 1: each star point's control point held at its limit position
  FAIR_MU: 0,         // weight of the linearised curvature variation against the distance to the samples
  LINEAR_G2: 0,       // 1: G2 with the reparameterisation's tangential terms fixed from the fit (linear)
  SECOND_ORDER: 0,    // GMRES iterations per step on the second-order system (measured worse so far: off)
  ACCEPT: 'residual', // accept a step that lowers the largest condition residual ('merit': L1 merit, slower)
  STAB: 1e-8,         // stabilisation: diagonal shift on each condition, relative to its Schur diagonal
  TOL: 2e-6,          // largest scaled residual left (the checker reads G2 within about 0.001 % there)
  RIDGE: 0,           // extra relative ridge in the block factorisation (STAB already regularises)
  DROP: 1e-11,        // relative pivot below which a condition is taken as implied by earlier ones
  INNER_SAMPLES: 20,  // per half of a star face's middle line: its G2 residual is a polynomial of degree 17
};

const K3 = [0, 0, 0, 0, 1, 1, 1, 1];
const K6 = [0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1, 1, 1, 1];

// Side k of a face runs corner k -> k + 1; s along it, t into the face. [u-order, v-order, sign] per derivative.
const SIDE = [
  { uv: (s) => [s, 0], d: { x: [0, 0, 1], s: [1, 0, 1], t: [0, 1, 1], ss: [2, 0, 1], st: [1, 1, 1], tt: [0, 2, 1] } },
  { uv: (s) => [1, s], d: { x: [0, 0, 1], s: [0, 1, 1], t: [1, 0, -1], ss: [0, 2, 1], st: [1, 1, -1], tt: [2, 0, 1] } },
  { uv: (s) => [1 - s, 1], d: { x: [0, 0, 1], s: [1, 0, -1], t: [0, 1, -1], ss: [2, 0, 1], st: [1, 1, 1], tt: [0, 2, 1] } },
  { uv: (s) => [0, 1 - s], d: { x: [0, 0, 1], s: [0, 1, -1], t: [1, 0, 1], ss: [0, 2, 1], st: [1, 1, -1], tt: [2, 0, 1] } },
];
const NAMES = ['x', 's', 't', 'ss', 'st', 'tt'];

/** Control point index of row tIdx (into the face) at position sIdx along side k. */
function cvOfSide(k, sIdx, tIdx, nc) {
  if (k === 0) return sIdx * nc + tIdx;
  if (k === 1) return (nc - 1 - tIdx) * nc + sIdx;
  if (k === 2) return (nc - 1 - sIdx) * nc + (nc - 1 - tIdx);
  return tIdx * nc + (nc - 1 - sIdx);
}

/** Linear functionals (control point indices and weights) for x and its s/t derivatives at s on side k. */
function functionals(face, k, s) {
  const [u, v] = SIDE[k].uv(s);
  const Bu = nzBasis(face.K, face.p, u, 2), Bv = nzBasis(face.K, face.p, v, 2);
  const n = face.p + 1;
  const idx = new Int32Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) idx[i * n + j] = (Bu.first + i) * face.nc + Bv.first + j;
  const out = { idx };
  for (const name of NAMES) {
    const [a, b, sign] = SIDE[k].d[name];
    const w = new Float64Array(n * n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) w[i * n + j] = sign * Bu.ders[a][i] * Bv.ders[b][j];
    out[name] = w;
  }
  return out;
}

function apply(net, idx, w) {
  let x = 0, y = 0, z = 0;
  for (let k = 0; k < idx.length; k++) {
    const o = idx[k] * 3, c = w[k];
    x += c * net[o]; y += c * net[o + 1]; z += c * net[o + 2];
  }
  return [x, y, z];
}

const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm = (a) => Math.hypot(a[0], a[1], a[2]);

/** Dense symmetric positive definite inverse (Cholesky), n x n row-major. */
function spdInverse(M, n) {
  const L = Float64Array.from(M);
  for (let j = 0; j < n; j++) {
    let d = L[j * n + j];
    for (let k = 0; k < j; k++) d -= L[j * n + k] ** 2;
    const ljj = Math.sqrt(d);
    L[j * n + j] = ljj;
    for (let i = j + 1; i < n; i++) {
      let s = L[i * n + j];
      for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
      L[i * n + j] = s / ljj;
    }
  }
  const inv = new Float64Array(n * n);
  const col = new Float64Array(n);
  for (let c = 0; c < n; c++) {
    col.fill(0); col[c] = 1;
    for (let i = 0; i < n; i++) { let s = col[i]; for (let k = 0; k < i; k++) s -= L[i * n + k] * col[k]; col[i] = s / L[i * n + i]; }
    for (let i = n - 1; i >= 0; i--) { let s = col[i]; for (let k = i + 1; k < n; k++) s -= L[k * n + i] * col[k]; col[i] = s / L[i * n + i]; }
    for (let i = 0; i < n; i++) inv[i * n + c] = col[i];
  }
  return inv;
}

/** (B^T B)^-1 B^T for samples at a / (N - 1): nc x N, row-major. */
function pseudoInverse(K, p, nc, params) {
  const N = params.length;
  const B = params.map((x) => {
    const row = new Float64Array(nc), r = nzBasis(K, p, x, 0);
    for (let i = 0; i <= p; i++) row[r.first + i] = r.ders[0][i];
    return row;
  });
  const G = new Float64Array(nc * nc);
  for (const row of B) for (let i = 0; i < nc; i++) for (let j = 0; j < nc; j++) G[i * nc + j] += row[i] * row[j];
  const Gi = spdInverse(G, nc);
  const P = new Float64Array(nc * N);
  for (let i = 0; i < nc; i++) for (let a = 0; a < N; a++) {
    let s = 0;
    for (let j = 0; j < nc; j++) s += Gi[i * nc + j] * B[a][j];
    P[i * N + a] = s;
  }
  return P;
}

/** Separable least-squares net from an N x N sample grid. */
function fitNet(grid, N, P, nc) {
  const T = new Float64Array(nc * N * 3);
  for (let i = 0; i < nc; i++) for (let b = 0; b < N; b++) for (let c = 0; c < 3; c++) {
    let s = 0;
    for (let a = 0; a < N; a++) s += P[i * N + a] * grid[(a * N + b) * 3 + c];
    T[(i * N + b) * 3 + c] = s;
  }
  const net = new Float64Array(nc * nc * 3);
  for (let i = 0; i < nc; i++) for (let j = 0; j < nc; j++) for (let c = 0; c < 3; c++) {
    let s = 0;
    for (let b = 0; b < N; b++) s += T[(i * N + b) * 3 + c] * P[j * N + b];
    net[(i * nc + j) * 3 + c] = s;
  }
  return net;
}

/**
 * The objective's Hessian: sum over the N x N limit samples of |S(u, v) - L(u, v)|^2 is, per coordinate,
 * (x - fit)^T H (x - fit) + const with H = G (x) G, G = B^T B the 1D Gram matrix of the basis at the sample
 * parameters. Returns H (nc^2 x nc^2).
 */
function sampleHessian(K, p, nc, params) {
  const G = new Float64Array(nc * nc);
  for (const x of params) {
    const r = nzBasis(K, p, x, 0);
    for (let i = 0; i <= p; i++) for (let j = 0; j <= p; j++) G[(r.first + i) * nc + r.first + j] += r.ders[0][i] * r.ders[0][j];
  }
  const n = nc * nc, H = new Float64Array(n * n);
  for (let a = 0; a < nc; a++) for (let b = 0; b < nc; b++) for (let c = 0; c < nc; c++) for (let d = 0; d < nc; d++) {
    H[(a * nc + b) * n + c * nc + d] = G[a * nc + c] * G[b * nc + d];
  }
  return H;
}

/**
 * Linearised curvature variation (Moreton and Sequin's minimum-variation idea, linearised): the integral over
 * the face of |S_uuu|^2 + 3 |S_uuv|^2 + 3 |S_uvv|^2 + |S_vvv|^2, per coordinate x^T Q x with
 * Q = sum_a C(3, a) M_a (x) M_(3-a), M_a the 1D Gram matrix of the a-th derivatives (Gauss-Legendre, exact).
 */
function variationHessian(K, p, nc) {
  const gx = [-0.9491079123427585, -0.7415311855993945, -0.4058451513773972, 0, 0.4058451513773972, 0.7415311855993945, 0.9491079123427585];
  const gw = [0.1294849661688697, 0.2797053914892766, 0.3818300505051189, 0.4179591836734694, 0.3818300505051189, 0.2797053914892766, 0.1294849661688697];
  const spans = [...new Set(K)].sort((x, y) => x - y);
  const M = [0, 1, 2, 3].map(() => new Float64Array(nc * nc));
  for (let k = 0; k + 1 < spans.length; k++) {
    const lo = spans[k], hi = spans[k + 1], h = (hi - lo) / 2;
    gx.forEach((x, q) => {
      const r = nzBasis(K, p, lo + h * (x + 1), 3);
      for (let a = 0; a <= 3; a++) for (let i = 0; i <= p; i++) for (let j = 0; j <= p; j++) {
        M[a][(r.first + i) * nc + r.first + j] += gw[q] * h * r.ders[a][i] * r.ders[a][j];
      }
    });
  }
  const n = nc * nc, Q = new Float64Array(n * n), binom = [1, 3, 3, 1];
  for (let a = 0; a <= 3; a++) {
    const Mu = M[a], Mv = M[3 - a];
    for (let i = 0; i < nc; i++) for (let j = 0; j < nc; j++) for (let k = 0; k < nc; k++) for (let l = 0; l < nc; l++) {
      Q[(i * nc + j) * n + k * nc + l] += binom[a] * Mu[i * nc + k] * Mv[j * nc + l];
    }
  }
  return Q;
}

/**
 * The move metric of NURBS Proto's method (specification 3.10), for our 12 x 12 net: I + w x (squared second
 * differences along every row and column) on the face's 13 x 13 grid of Bezier piece control points, pulled back
 * through the knot insertion that turns the C1 net into its pieces.
 */
function pieceGridMetric(K, p, nc, w) {
  const m = 2 * p + 1;
  const E = Array.from({ length: m }, () => new Float64Array(nc));
  const binom = (n, k) => { let r = 1; for (let i = 1; i <= k; i++) r = (r * (n - k + i)) / i; return r; };
  const T = Array.from({ length: p + 1 }, (_, i) => i / p);
  const Bern = new Float64Array((p + 1) * (p + 1));
  T.forEach((t, a) => { for (let i = 0; i <= p; i++) Bern[a * (p + 1) + i] = binom(p, i) * t ** i * (1 - t) ** (p - i); });
  const Binv = (() => {                              // Gauss-Jordan inverse of the Bernstein collocation matrix
    const n = p + 1, A = Array.from({ length: n }, (_, i) => [...Bern.slice(i * n, i * n + n), ...Array.from({ length: n }, (__, j) => (i === j ? 1 : 0))]);
    for (let c = 0; c < n; c++) {
      let piv = c; for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r;
      [A[c], A[piv]] = [A[piv], A[c]];
      const d = A[c][c]; for (let j = 0; j < 2 * n; j++) A[c][j] /= d;
      for (let r = 0; r < n; r++) if (r !== c) { const f = A[r][c]; for (let j = 0; j < 2 * n; j++) A[r][j] -= f * A[c][j]; }
    }
    return A.map((row) => row.slice(n));
  })();
  for (let q = 0; q < 2; q++) {
    const vals = T.map((t) => {
      const x = Math.min((q + t) / 2, q === 0 ? 0.5 - 1e-15 : 1), r = nzBasis(K, p, x, 0), row = new Float64Array(nc);
      for (let i = 0; i <= p; i++) row[r.first + i] = r.ders[0][i];
      return row;
    });
    for (let i = 0; i <= p; i++) for (let k = 0; k < nc; k++) {
      let s = 0;
      for (let a = 0; a <= p; a++) s += Binv[i][a] * vals[a][k];
      E[q * p + i][k] = s;
    }
  }
  const n = m * m, MB = new Float64Array(n * n);
  for (let i = 0; i < n; i++) MB[i * n + i] = 1;
  const stencil = (a, b, c) => { const ids = [a, b, c], ws = [1, -2, 1]; for (let x = 0; x < 3; x++) for (let y = 0; y < 3; y++) MB[ids[x] * n + ids[y]] += w * ws[x] * ws[y]; };
  for (let i = 0; i < m; i++) for (let j = 1; j + 1 < m; j++) {
    stencil(i * m + j - 1, i * m + j, i * m + j + 1);
    stencil((j - 1) * m + i, j * m + i, (j + 1) * m + i);
  }
  const N2 = nc * nc, G = new Float64Array(n * N2);   // piece grid = (E x E) net
  for (let r = 0; r < m; r++) for (let c = 0; c < m; c++) for (let k = 0; k < nc; k++) for (let l = 0; l < nc; l++) G[(r * m + c) * N2 + k * nc + l] = E[r][k] * E[c][l];
  const MG = new Float64Array(n * N2);
  for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) { const x = MB[a * n + b]; if (x) for (let k = 0; k < N2; k++) MG[a * N2 + k] += x * G[b * N2 + k]; }
  const M = new Float64Array(N2 * N2);
  for (let a = 0; a < n; a++) for (let k = 0; k < N2; k++) { const g = G[a * N2 + k]; if (g) for (let l = 0; l < N2; l++) M[k * N2 + l] += g * MG[a * N2 + l]; }
  return M;
}

/**
 * Bending (thin-plate) energy of a field over the face, per coordinate: the integral of
 * |d_uu|^2 + 2 |d_uv|^2 + |d_vv|^2 is d^T T d with T = M2 (x) M0 + 2 M1 (x) M1 + M0 (x) M2, M_a the 1D Gram
 * matrix of a-th derivatives (Gauss-Legendre per span, exact). Zero exactly on affine fields.
 */
function bendingHessian(K, p, nc) {
  const gx = [-0.9491079123427585, -0.7415311855993945, -0.4058451513773972, 0, 0.4058451513773972, 0.7415311855993945, 0.9491079123427585];
  const gw = [0.1294849661688697, 0.2797053914892766, 0.3818300505051189, 0.4179591836734694, 0.3818300505051189, 0.2797053914892766, 0.1294849661688697];
  const spans = [...new Set(K)].sort((x, y) => x - y);
  const M = [0, 1, 2].map(() => new Float64Array(nc * nc));
  for (let k = 0; k + 1 < spans.length; k++) {
    const lo = spans[k], hi = spans[k + 1], h = (hi - lo) / 2;
    gx.forEach((x, q) => {
      const r = nzBasis(K, p, lo + h * (x + 1), 2);
      for (let a = 0; a <= 2; a++) for (let i = 0; i <= p; i++) for (let j = 0; j <= p; j++) {
        M[a][(r.first + i) * nc + r.first + j] += gw[q] * h * r.ders[a][i] * r.ders[a][j];
      }
    });
  }
  const n = nc * nc, T = new Float64Array(n * n), terms = [[2, 0, 1], [1, 1, 2], [0, 2, 1]];
  const S0 = new Float64Array(n * n);                   // the size term: integral of |d|^2, M0 (x) M0
  for (let i = 0; i < nc; i++) for (let j = 0; j < nc; j++) for (let k = 0; k < nc; k++) for (let l = 0; l < nc; l++) {
    S0[(i * nc + j) * n + k * nc + l] = M[0][i * nc + k] * M[0][j * nc + l];
  }
  for (const [a, b, w] of terms) {
    const Mu = M[a], Mv = M[b];
    for (let i = 0; i < nc; i++) for (let j = 0; j < nc; j++) for (let k = 0; k < nc; k++) for (let l = 0; l < nc; l++) {
      T[(i * nc + j) * n + k * nc + l] += w * Mu[i * nc + k] * Mv[j * nc + l];
    }
  }
  return { T, S0 };
}

/** H restricted to the free control points, and its inverse. */
function restricted(H, n, free) {
  const m = free.length;
  const A = new Float64Array(m * m);
  for (let a = 0; a < m; a++) for (let b = 0; b < m; b++) A[a * m + b] = H[free[a] * n + free[b]];
  return { A, Ainv: spdInverse(A, m) };
}

function chebyshev(m, lo, hi) {
  return Array.from({ length: m }, (_, k) => lo + (hi - lo) * (1 - Math.cos(Math.PI * (k + 0.5) / m)) / 2);
}

/**
 * Convert a closed quad cage. Returns { faces: [{ uKnots, vKnots, uCount, vCount, poles }], stats }; poles are
 * row-major, u then v, u running corner 0 -> 1 of the cage face and v corner 0 -> 3.
 */
export function subdToNurbs(vertices, cageFaces, options = {}) {
  const opt = { ...SUBD, ...options };
  const t0 = performance.now();
  const nv = vertices.length;
  const { edges, valence } = cageTopology(nv, cageFaces);
  const isStar = (v) => valence[v] !== 4;
  const { grids, N } = limitGrids(vertices, cageFaces, opt.LEVELS);
  const t1 = performance.now();
  const params = Array.from({ length: N }, (_, a) => a / (N - 1));
  const P3 = pseudoInverse(K3, 3, 4, params), P6 = pseudoInverse(K6, 6, 12, params);
  if (opt.TARGET === 'natural') {                 // star faces fitted to the level-2 natural net instead
    const nn = naturalNets(vertices, cageFaces);
    cageFaces.forEach((loop, f) => {
      if (!loop.some(isStar)) return;
      const g = grids[f];
      for (let a = 0; a < N; a++) for (let b = 0; b < N; b++) g.set(nn.sample(f, params[a], params[b]), (a * N + b) * 3);
    });
  }

  const faces = cageFaces.map((loop, f) => {
    const star = loop.some(isStar);
    const face = star ? { p: 6, nc: 12, K: K6 } : { p: 3, nc: 4, K: K3 };
    face.loop = loop;
    face.star = star;
    face.net = fitNet(grids[f], N, star ? P6 : P3, face.nc);
    face.fixed = new Uint8Array(face.nc * face.nc);
    return face;
  });
  const rawFits = faces.map((f) => Float64Array.from(f.net));   // unconstrained least squares
  const neighbour = new Map();          // "face,side" -> { face, side }
  for (const e of edges.values()) {
    const [x, y] = e.uses;
    neighbour.set(`${x.face},${x.side}`, y);
    neighbour.set(`${y.face},${y.side}`, x);
  }

  // fixed control points of star faces
  const corners = [[0, 0], [11, 0], [11, 11], [0, 11]];
  const sParams = chebyshev(31, 0, 1);
  const P1 = pseudoInverse(K6, 6, 12, sParams);
  faces.forEach((face, fi) => {
    if (!face.star) return;
    face.loop.forEach((v, k) => {
      if (!isStar(v) || !opt.PIN_STAR) return;
      const [i, j] = corners[k], gi = k === 0 ? [0, 0] : k === 1 ? [N - 1, 0] : k === 2 ? [N - 1, N - 1] : [0, N - 1];
      face.net.set(grids[fi].subarray((gi[0] * N + gi[1]) * 3, (gi[0] * N + gi[1]) * 3 + 3), (i * 12 + j) * 3);
      face.fixed[i * 12 + j] = 1;
    });
    for (let k = 0; k < 4; k++) {
      const nb = neighbour.get(`${fi},${k}`);
      const g = faces[nb.face];
      if (g.star) continue;
      const h = 0.5, p = 6;                                   // first piece's width, degree
      const rows = [[], [], []];
      for (const s of sParams) {
        const F = functionals(g, nb.side, 1 - s);
        const G = apply(g.net, F.idx, F.x), Gt = apply(g.net, F.idx, F.t), Gtt = apply(g.net, F.idx, F.tt);
        const R0 = G, R1 = R0.map((x, c) => x - Gt[c] * h / p);
        const R2 = R1.map((x, c) => 2 * x - R0[c] + Gtt[c] * h * h / (p * (p - 1)));
        rows[0].push(R0); rows[1].push(R1); rows[2].push(R2);
      }
      for (let tIdx = 0; tIdx < 3; tIdx++) {
        for (let sIdx = 0; sIdx < 12; sIdx++) {
          const cv = cvOfSide(k, sIdx, tIdx, 12);
          for (let c = 0; c < 3; c++) {
            let x = 0;
            for (let a = 0; a < sParams.length; a++) x += P1[sIdx * sParams.length + a] * rows[tIdx][a][c];
            face.net[cv * 3 + c] = x;
          }
          face.fixed[cv] = 1;
        }
      }
    }
  });
  // Free variables per star face, the objective restricted to them, and its minimiser with the fixed control
  // points held: target_F = fit_F - A_FF^-1 H_FX (x_X - fit_X). The solve starts there.
  const H6 = sampleHessian(K6, 6, 12, params);
  const Q6 = opt.FAIR_MU > 0 ? variationHessian(K6, 6, 12) : null;
  // 'bending': the objective is the bending energy of the displacement from the fit, with a 1e-8 relative
  // diagonal so a face held in fewer than three places stays invertible (it moves no result measurably)
  const A6 = opt.METRIC === 'nproto' ? pieceGridMetric(K6, 6, 12, 3)
    : opt.METRIC === 'bending' ? (() => {
      const { T } = bendingHessian(K6, 6, 12);
      let mean = 0; for (let i = 0; i < 144; i++) mean += T[i * 144 + i] / 144;
      for (let i = 0; i < 144; i++) T[i * 144 + i] += 1e-8 * mean;
      return T;
    })()
    // 'smooth': size plus bending of the displacement, integral |d|^2 + ELL^4 integral |bending of d|^2, ELL in
    // face-side units: how far a correction spreads
    : opt.METRIC === 'smooth' ? (() => {
      const { T, S0 } = bendingHessian(K6, 6, 12), l4 = opt.ELL ** 4;
      return S0.map((x, i) => x + l4 * T[i]);
    })()
    : Q6 ? H6.map((h, i) => h + opt.FAIR_MU * Q6[i]) : H6;
  const Hbase = opt.METRIC === 'bending' || opt.METRIC === 'smooth' ? A6 : H6;
  const metricCache = new Map();
  const targets = faces.map((f) => f.net);
  faces.forEach((face, fi) => {
    if (!face.star) return;
    face.free = [];
    face.local = new Int32Array(144).fill(-1);
    for (let cv = 0; cv < 144; cv++) if (!face.fixed[cv]) { face.local[cv] = face.free.length; face.free.push(cv); }
    const key = String(face.fixed.join(''));
    if (!metricCache.has(key)) metricCache.set(key, restricted(A6, 144, face.free));
    Object.assign(face, metricCache.get(key));
    const m = face.free.length, raw = rawFits[fi], net = face.net;
    const fixedList = [];
    for (let cv = 0; cv < 144; cv++) if (face.fixed[cv]) fixedList.push(cv);
    if (opt.METRIC !== 'nproto') for (let c = 0; c < 3; c++) {
      // the minimiser over the free points of (x - fit)^T H (x - fit) + mu x^T Q x, fixed points held:
      // (H_FF + mu Q_FF) x_F = H_FF fit_F - H_FX (x_X - fit_X) - mu Q_FX x_X
      const g = new Float64Array(m);
      for (let a = 0; a < m; a++) {
        const r = face.free[a] * 144;
        let s = 0;
        for (let b = 0; b < m; b++) s += Hbase[r + face.free[b]] * raw[face.free[b] * 3 + c];
        for (const cv of fixedList) {
          s -= Hbase[r + cv] * (net[cv * 3 + c] - raw[cv * 3 + c]);
          if (Q6 && opt.METRIC !== 'bending') s -= opt.FAIR_MU * Q6[r + cv] * net[cv * 3 + c];
        }
        g[a] = s;
      }
      for (let a = 0; a < m; a++) {
        let s = 0;
        for (let b = 0; b < m; b++) s += face.Ainv[a * m + b] * g[b];
        net[face.free[a] * 3 + c] = s;
      }
    }
    targets[fi] = Float64Array.from(net);
  });
  const fitNets = targets;

  // condition samples on every edge between two star faces
  const cedges = [];
  for (const e of edges.values()) {
    const [x, y] = e.uses;
    if (!faces[x.face].star || !faces[y.face].star) continue;
    const f = faces[x.face], g = faces[y.face];
    const a = f.loop[x.side], b = f.loop[(x.side + 1) % 4];
    const mid = functionals(f, x.side, 0.5);
    const L = norm(apply(fitNets[x.face], mid.idx, mid.s)) || 1;
    const halves = [];
    for (const [lo, hi, end, sigma] of [[0, 0.5, a, 1], [0.5, 1, b, -1]]) {
      const c = isStar(end) ? 2 * Math.cos(2 * Math.PI / valence[end]) : 0;
      const pts = (m) => chebyshev(m, lo, hi).map((s) => {
        const away = sigma > 0 ? s : 1 - s;
        return { s, beta: c * (1 - 2 * away) ** 2, sigma, F: functionals(f, x.side, s), G: functionals(g, y.side, 1 - s) };
      });
      halves.push({ g1: pts(opt.G1_SAMPLES), g2: pts(opt.G2_SAMPLES) });
    }
    cedges.push({ fi: x.face, gi: y.face, kf: x.side, kg: y.side, L, halves });
  }

  // G2 across each star face's middle lines, where its pieces join only C1: n . (x_tt right - x_tt left) = 0,
  // t across the line (beta is 0 inside a face). Each half of a line is one polynomial piece.
  const inner = [];
  faces.forEach((face, fi) => {
    if (!face.star) return;
    const samples = [];
    for (const dir of ['u', 'v']) {
      for (const w of [...chebyshev(opt.INNER_SAMPLES, 0, 0.5), ...chebyshev(opt.INNER_SAMPLES, 0.5, 1)]) {
        const R = nzBasis(face.K, face.p, 0.5, 2), Lb = nzBasis(face.K, face.p, 0.5 - 1e-12, 2);
        const B = nzBasis(face.K, face.p, w, 2);
        const n = face.p + 1;
        const idxR = new Int32Array(n * n), idxL = new Int32Array(n * n);
        const wa = new Float64Array(n * n), wb = new Float64Array(n * n), wqR = new Float64Array(n * n), wqL = new Float64Array(n * n);
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
          const k = i * n + j;
          if (dir === 'u') {        // the line u = 0.5: along v, across u
            idxR[k] = (R.first + i) * face.nc + B.first + j;
            idxL[k] = (Lb.first + i) * face.nc + B.first + j;
            wa[k] = R.ders[0][i] * B.ders[1][j];
            wb[k] = R.ders[1][i] * B.ders[0][j];
            wqR[k] = R.ders[2][i] * B.ders[0][j];
            wqL[k] = -Lb.ders[2][i] * B.ders[0][j];
          } else {                  // the line v = 0.5: along u, across v
            idxR[k] = (B.first + i) * face.nc + R.first + j;
            idxL[k] = (B.first + i) * face.nc + Lb.first + j;
            wa[k] = B.ders[1][i] * R.ders[0][j];
            wb[k] = B.ders[0][i] * R.ders[1][j];
            wqR[k] = B.ders[0][i] * R.ders[2][j];
            wqL[k] = -B.ders[0][i] * Lb.ders[2][j];
          }
        }
        samples.push({ s: w, idxR, idxL, wa, wb, wqR, wqL });
      }
    }
    const mid = samples[opt.INNER_SAMPLES];
    const L = norm(apply(fitNets[fi], mid.idxR, mid.wa)) || 1;
    inner.push({ fi, L, samples });
  });
  const nBlocks = cedges.length + inner.length;

  /** gam, del with q ~ gam a + del b (least squares, i.e. the tangential part of q), at every G2 sample. */
  function estimateReparam() {
    const solve2 = (a, b, q) => {
      const aa = dot(a, a), ab = dot(a, b), bb = dot(b, b), det = aa * bb - ab * ab;
      return [(bb * dot(a, q) - ab * dot(b, q)) / det, (aa * dot(b, q) - ab * dot(a, q)) / det];
    };
    for (const ce of cedges) {
      const f = faces[ce.fi], g = faces[ce.gi];
      for (const half of ce.halves) for (const pt of half.g2) {
        const { beta, sigma, F, G } = pt;
        const a = apply(f.net, F.idx, F.s).map((x) => sigma * x), b = apply(f.net, F.idx, F.t);
        const Xss = apply(f.net, F.idx, F.ss), Xst = apply(f.net, F.idx, F.st), Xtt = apply(f.net, F.idx, F.tt);
        const Ytt = apply(g.net, G.idx, G.tt);
        const q = [0, 1, 2].map((c) => Ytt[c] - Xtt[c] + 2 * beta * sigma * Xst[c] - beta * beta * Xss[c]);
        [pt.gam, pt.del] = solve2(a, b, q);
      }
    }
    for (const ib of inner) {
      const face = faces[ib.fi];
      for (const sm of ib.samples) {
        const a = apply(face.net, sm.idxR, sm.wa), b = apply(face.net, sm.idxR, sm.wb);
        const qR = apply(face.net, sm.idxR, sm.wqR), qL = apply(face.net, sm.idxL, sm.wqL);
        [sm.gam, sm.del] = solve2(a, b, [qR[0] + qL[0], qR[1] + qL[1], qR[2] + qL[2]]);
      }
    }
  }
  if (opt.LINEAR_G2) estimateReparam();

  /** Rows at the current nets: { edge, value (scaled), kind, terms: [{ face, vars, w }] }. */
  function conditions(withG2 = true) {
    const rows = [];
    let dropped = 0;
    const push = (edge, kind, value, terms, at, sec) => {
      const live = terms.filter((t) => t.vars.length);
      if (!live.length) { dropped = Math.max(dropped, Math.abs(value)); return; }
      rows.push({ edge, kind, value, terms: live, s: at, sec });
    };
    cedges.forEach((ce, ei) => {
      const f = faces[ce.fi], g = faces[ce.gi];
      for (let sIdx = 0; sIdx < 12; sIdx++) {
        const cf = cvOfSide(ce.kf, sIdx, 0, 12), cg = cvOfSide(ce.kg, 11 - sIdx, 0, 12);
        for (let c = 0; c < 3; c++) {
          const terms = [];
          if (f.local[cf] >= 0) terms.push({ face: ce.fi, vars: [f.local[cf] * 3 + c], w: [1 / ce.L] });
          if (g.local[cg] >= 0) terms.push({ face: ce.gi, vars: [g.local[cg] * 3 + c], w: [-1 / ce.L] });
          push(ei, 'g0', (f.net[cf * 3 + c] - g.net[cg * 3 + c]) / ce.L, terms, sIdx / 11);
        }
      }
      for (const half of ce.halves) {
        for (const { s, beta, sigma, F, G } of half.g1) {
          const Xs = apply(f.net, F.idx, F.s), Xt = apply(f.net, F.idx, F.t), Yt = apply(g.net, G.idx, G.t);
          for (let c = 0; c < 3; c++) {
            const tf = { face: ce.fi, vars: [], w: [] }, tg = { face: ce.gi, vars: [], w: [] };
            for (let k = 0; k < F.idx.length; k++) {
              const l = f.local[F.idx[k]];
              const w = F.t[k] - beta * sigma * F.s[k];
              if (l >= 0 && w !== 0) { tf.vars.push(l * 3 + c); tf.w.push(w / ce.L); }
            }
            for (let k = 0; k < G.idx.length; k++) {
              const l = g.local[G.idx[k]];
              if (l >= 0 && G.t[k] !== 0) { tg.vars.push(l * 3 + c); tg.w.push(G.t[k] / ce.L); }
            }
            push(ei, 'g1', (Xt[c] + Yt[c] - beta * sigma * Xs[c]) / ce.L, [tf, tg], s);
          }
        }
        const L3 = ce.L ** 3;
        if (withG2) for (const pt of half.g2) {
          const { s, beta, sigma, F, G } = pt;
          const a = apply(f.net, F.idx, F.s).map((x) => sigma * x), b = apply(f.net, F.idx, F.t);
          const Xss = apply(f.net, F.idx, F.ss), Xst = apply(f.net, F.idx, F.st), Xtt = apply(f.net, F.idx, F.tt);
          const Ytt = apply(g.net, G.idx, G.tt);
          const q = [0, 1, 2].map((c) => Ytt[c] - Xtt[c] + 2 * beta * sigma * Xst[c] - beta * beta * Xss[c]);
          if (opt.LINEAR_G2) {
            // q = gam a + del b with the reparameterisation's tangential terms fixed: linear, 3 rows
            for (let c = 0; c < 3; c++) {
              const tf = { face: ce.fi, vars: [], w: [] }, tg = { face: ce.gi, vars: [], w: [] };
              for (let k = 0; k < F.idx.length; k++) {
                const l = f.local[F.idx[k]];
                if (l < 0) continue;
                const w = -F.tt[k] + 2 * beta * sigma * F.st[k] - beta * beta * F.ss[k] - pt.gam * sigma * F.s[k] - pt.del * F.t[k];
                if (w !== 0) { tf.vars.push(l * 3 + c); tf.w.push(w / ce.L); }
              }
              for (let k = 0; k < G.idx.length; k++) {
                const l = g.local[G.idx[k]];
                if (l >= 0 && G.tt[k] !== 0) { tg.vars.push(l * 3 + c); tg.w.push(G.tt[k] / ce.L); }
              }
              push(ei, 'g2', (q[c] - pt.gam * a[c] - pt.del * b[c]) / ce.L, [tf, tg], s);
            }
            continue;
          }
          const bq = cross(b, q), qa = cross(q, a), ab = cross(a, b);
          const tf = { face: ce.fi, vars: [], w: [] }, tg = { face: ce.gi, vars: [], w: [] };
          for (let k = 0; k < F.idx.length; k++) {
            const l = f.local[F.idx[k]];
            if (l < 0) continue;
            const cq = -F.tt[k] + 2 * beta * sigma * F.st[k] - beta * beta * F.ss[k];
            for (let c = 0; c < 3; c++) {
              const w = sigma * F.s[k] * bq[c] + F.t[k] * qa[c] + cq * ab[c];
              if (w !== 0) { tf.vars.push(l * 3 + c); tf.w.push(w / L3); }
            }
          }
          for (let k = 0; k < G.idx.length; k++) {
            const l = g.local[G.idx[k]];
            if (l < 0) continue;
            for (let c = 0; c < 3; c++) {
              const w = G.tt[k] * ab[c];
              if (w !== 0) { tg.vars.push(l * 3 + c); tg.w.push(w / L3); }
            }
          }
          push(ei, 'g2', dot(ab, q) / L3, [tf, tg], s, { edge: true, fi: ce.fi, gi: ce.gi, F, G, sigma, beta, a, b, q, scale: 1 / L3 });
        }
      }
    });
    if (withG2) inner.forEach((ib, ii) => {
      const face = faces[ib.fi], L3 = ib.L ** 3;
      for (const sm of ib.samples) {
        const a = apply(face.net, sm.idxR, sm.wa), b = apply(face.net, sm.idxR, sm.wb);
        const qR = apply(face.net, sm.idxR, sm.wqR), qL = apply(face.net, sm.idxL, sm.wqL);
        const q = [qR[0] + qL[0], qR[1] + qL[1], qR[2] + qL[2]];
        if (opt.LINEAR_G2) {
          for (let c = 0; c < 3; c++) {
            const t = { face: ib.fi, vars: [], w: [] };
            for (let k = 0; k < sm.idxR.length; k++) {
              const l = face.local[sm.idxR[k]];
              const w = sm.wqR[k] - sm.gam * sm.wa[k] - sm.del * sm.wb[k];
              if (l >= 0 && w !== 0) { t.vars.push(l * 3 + c); t.w.push(w / ib.L); }
            }
            for (let k = 0; k < sm.idxL.length; k++) {
              const l = face.local[sm.idxL[k]];
              if (l >= 0 && sm.wqL[k] !== 0) { t.vars.push(l * 3 + c); t.w.push(sm.wqL[k] / ib.L); }
            }
            push(cedges.length + ii, 'g2in', (q[c] - sm.gam * a[c] - sm.del * b[c]) / ib.L, [t], sm.s);
          }
          continue;
        }
        const bq = cross(b, q), qa = cross(q, a), ab = cross(a, b);
        const t = { face: ib.fi, vars: [], w: [] };
        for (let k = 0; k < sm.idxR.length; k++) {
          const l = face.local[sm.idxR[k]];
          if (l < 0) continue;
          for (let c = 0; c < 3; c++) {
            const w = sm.wa[k] * bq[c] + sm.wb[k] * qa[c] + sm.wqR[k] * ab[c];
            if (w !== 0) { t.vars.push(l * 3 + c); t.w.push(w / L3); }
          }
        }
        for (let k = 0; k < sm.idxL.length; k++) {
          const l = face.local[sm.idxL[k]];
          if (l < 0 || sm.wqL[k] === 0) continue;
          for (let c = 0; c < 3; c++) { t.vars.push(l * 3 + c); t.w.push(sm.wqL[k] * ab[c] / L3); }
        }
        push(cedges.length + ii, 'g2in', dot(ab, q) / L3, [t], sm.s, { edge: false, fi: ib.fi, sm, a, b, q, scale: 1 / L3 });
      }
    });
    return { rows, dropped };
  }

  // block graph: condition blocks (edges, and faces' middle lines) sharing a star face
  const faceEdges = new Map();
  const touch = (fi, bi) => { if (!faceEdges.has(fi)) faceEdges.set(fi, []); faceEdges.get(fi).push(bi); };
  cedges.forEach((ce, ei) => { touch(ce.fi, ei); touch(ce.gi, ei); });
  inner.forEach((ib, ii) => touch(ib.fi, cedges.length + ii));
  const adj = Array.from({ length: nBlocks }, () => new Set());
  for (const list of faceEdges.values()) for (const a of list) for (const b of list) if (a !== b) adj[a].add(b);

  const t2 = performance.now();
  let steps = 0, dropped = 0, worst = {};
  const history = [];
  const ms = { conditions: 0, assemble: 0, factor: 0 };
  const evaluate = (withG2) => {
    const t = performance.now();
    const c = conditions(withG2);
    ms.conditions += performance.now() - t;
    c.residual = c.rows.reduce((m, r) => Math.max(m, Math.abs(r.value)), 0);
    return c;
  };
  /**
   * sum_r lam_r (Hessian of condition r) applied to a step, for the G2 rows (G0, G1 are linear). A condition
   * a . (b x q), with a, b, q linear in the control points, has directional gradient terms
   * da: db x q + b x dq,  db: q x da + dq x a,  dq: da x b + a x db.
   */
  function hessianTimes(rows, lam, step) {
    const out = new Map();
    const vec = (fi) => { if (!out.has(fi)) out.set(fi, new Float64Array(faces[fi].free.length * 3)); return out.get(fi); };
    const gather = (fi, idx, w) => {
      const face = faces[fi], dl = step.get(fi), x = [0, 0, 0];
      if (!dl) return x;
      for (let k = 0; k < idx.length; k++) {
        const l = face.local[idx[k]];
        if (l < 0 || w[k] === 0) continue;
        x[0] += w[k] * dl[l * 3]; x[1] += w[k] * dl[l * 3 + 1]; x[2] += w[k] * dl[l * 3 + 2];
      }
      return x;
    };
    const scatter = (fi, idx, w, g, coef) => {
      const face = faces[fi], o = vec(fi);
      for (let k = 0; k < idx.length; k++) {
        const l = face.local[idx[k]];
        if (l < 0 || w[k] === 0) continue;
        for (let c = 0; c < 3; c++) o[l * 3 + c] += coef * w[k] * g[c];
      }
    };
    rows.forEach((r, ri) => {
      const sec = r.sec, lm = lam[ri];
      if (!sec || !lm) return;
      const { a, b, q } = sec, coef = lm * sec.scale;
      let da, db, dq;
      if (sec.edge) {
        const { F, G, sigma, beta } = sec;
        if (!sec.wq) sec.wq = F.tt.map((x, k) => -x + 2 * beta * sigma * F.st[k] - beta * beta * F.ss[k]);
        if (!sec.wa) sec.wa = F.s.map((x) => sigma * x);
        da = gather(sec.fi, F.idx, sec.wa); db = gather(sec.fi, F.idx, F.t);
        const dqf = gather(sec.fi, F.idx, sec.wq), dqg = gather(sec.gi, G.idx, G.tt);
        dq = [dqf[0] + dqg[0], dqf[1] + dqg[1], dqf[2] + dqg[2]];
        const ga = cross(db, q).map((x, c) => x + cross(b, dq)[c]);
        const gb = cross(q, da).map((x, c) => x + cross(dq, a)[c]);
        const gq = cross(da, b).map((x, c) => x + cross(a, db)[c]);
        scatter(sec.fi, F.idx, sec.wa, ga, coef); scatter(sec.fi, F.idx, F.t, gb, coef);
        scatter(sec.fi, F.idx, sec.wq, gq, coef); scatter(sec.gi, G.idx, G.tt, gq, coef);
      } else {
        const { sm } = sec;
        da = gather(sec.fi, sm.idxR, sm.wa); db = gather(sec.fi, sm.idxR, sm.wb);
        const q1 = gather(sec.fi, sm.idxR, sm.wqR), q2 = gather(sec.fi, sm.idxL, sm.wqL);
        dq = [q1[0] + q2[0], q1[1] + q2[1], q1[2] + q2[2]];
        const ga = cross(db, q).map((x, c) => x + cross(b, dq)[c]);
        const gb = cross(q, da).map((x, c) => x + cross(dq, a)[c]);
        const gq = cross(da, b).map((x, c) => x + cross(a, db)[c]);
        scatter(sec.fi, sm.idxR, sm.wa, ga, coef); scatter(sec.fi, sm.idxR, sm.wb, gb, coef);
        scatter(sec.fi, sm.idxR, sm.wqR, gq, coef); scatter(sec.fi, sm.idxL, sm.wqL, gq, coef);
      }
    });
    return out;
  }

  // The objective: squared distance to the limit samples over the star faces, (x - target)^T A (x - target).
  const objective = () => {
    let E = 0;
    faces.forEach((face, fi) => {
      if (!face.star) return;
      const m = face.free.length, e = new Float64Array(m);
      for (let c = 0; c < 3; c++) {
        for (let l = 0; l < m; l++) e[l] = face.net[face.free[l] * 3 + c] - targets[fi][face.free[l] * 3 + c];
        for (let i = 0; i < m; i++) { let t = 0; for (let j = 0; j < m; j++) t += face.A[i * m + j] * e[j]; E += e[i] * t; }
      }
    });
    return E;
  };
  const l1 = (c) => c.rows.reduce((t, r) => t + Math.abs(r.value), 0);
  // Stabilised SQP (multiplier estimates carried from step to step, so a condition the small diagonal shift
  // lets slip is pulled back at the next step and the fixed point is exactly feasible); steps are accepted by an
  // exact L1 merit function, objective + penalty x sum |condition|, the penalty above twice the largest multiplier.
  let cond = evaluate(true), lamBar = new Float64Array(cond.rows.length), penalty = 0;
  const alphas = [];
  let newtonFallbacks = 0;
  for (; ; steps++) {
    const rows = cond.rows, residual = cond.residual;
    dropped = cond.dropped;
    worst = {};
    for (const r of rows) { const w = worst[r.kind]; if (!w || Math.abs(r.value) > w.value) worst[r.kind] = { value: Math.abs(r.value), s: +r.s.toFixed(3), block: r.edge }; }
    history.push(residual);
    if (residual < opt.TOL || steps === opt.MAX_STEPS) break;
    const ta = performance.now();
    const sizes = new Array(nBlocks).fill(0);
    for (const r of rows) r.at = sizes[r.edge]++;
    const onFace = new Map();
    rows.forEach((r, ri) => r.terms.forEach((t, ti) => {
      if (!onFace.has(t.face)) onFace.set(t.face, []);
      onFace.get(t.face).push([ri, ti]);
    }));
    const blocks = new Map();
    const block = (ea, eb) => {
      const key = ea + ',' + eb;
      let B = blocks.get(key);
      if (!B) { B = new Float64Array(sizes[ea] * sizes[eb]); blocks.set(key, B); }
      return B;
    };
    const rhs = sizes.map((m) => new Float64Array(m));
    const W = new Map();
    for (const [fi, list] of onFace) {
      const face = faces[fi], m = face.free.length, Ainv = face.Ainv;
      const d = new Float64Array(m * 3);
      if (steps < opt.ANCHORED) for (let l = 0; l < m; l++) for (let c = 0; c < 3; c++) d[l * 3 + c] = targets[fi][face.free[l] * 3 + c] - face.net[face.free[l] * 3 + c];
      const Ws = [];
      for (const [ri, ti] of list) {
        const t = rows[ri].terms[ti];
        const w = new Float64Array(m * 3);
        for (let k = 0; k < t.vars.length; k++) {
          const l = (t.vars[k] / 3) | 0, c = t.vars[k] % 3, x = t.w[k];
          for (let i = 0; i < m; i++) w[i * 3 + c] += x * Ainv[i * m + l];
        }
        Ws.push(w);
        let jd = 0;
        for (let k = 0; k < t.vars.length; k++) jd += t.w[k] * d[t.vars[k]];
        rhs[rows[ri].edge][rows[ri].at] += jd;
      }
      W.set(fi, { list, Ws, d });
      for (let a = 0; a < list.length; a++) {
        const ra = rows[list[a][0]];
        for (let b = 0; b < list.length; b++) {
          const rb = rows[list[b][0]];
          if (ra.edge < rb.edge) continue;
          const tb = rb.terms[list[b][1]], wa = Ws[a];
          let s = 0;
          for (let k = 0; k < tb.vars.length; k++) s += tb.w[k] * wa[tb.vars[k]];
          block(ra.edge, rb.edge)[ra.at * sizes[rb.edge] + rb.at] += s;
        }
      }
    }
    for (let e = 0; e < nBlocks; e++) block(e, e);
    rows.forEach((r, ri) => {
      const D = block(r.edge, r.edge), k = r.at * sizes[r.edge] + r.at;
      const shift = opt.STAB * D[k];
      r.shift = shift;
      D[k] += shift;
      rhs[r.edge][r.at] += r.value + shift * lamBar[ri];
    });
    const tf = performance.now();
    ms.assemble += tf - ta;
    const solve = factorBlocks(sizes, blocks, adj, opt.RIDGE, opt.DROP);
    const nu = solve(rhs);
    ms.factor += performance.now() - tf;
    const deltas = new Map();
    for (const [fi, { list, Ws, d }] of W) {
      const delta = Float64Array.from(d);
      list.forEach(([ri], a) => {
        const l = nu[rows[ri].edge][rows[ri].at], w = Ws[a];
        for (let i = 0; i < delta.length; i++) delta[i] -= l * w[i];
      });
      deltas.set(fi, delta);
    }
    // Second order (Newton-Krylov): GMRES on the true step system
    //   [A + sum lam Hessian(c), J^T; J, -D] [step; nu] = [A d; -c - D lamBar]
    // right-preconditioned by the factorisation above (the same system without the curvature term) and started
    // from the Gauss-Newton step. lamBar: the multipliers of the previous step.
    let gnDeltas = null, gnNu = null;
    if (opt.SECOND_ORDER && steps > 0) {
      const faceIds = [...W.keys()];
      const offs = new Map();
      let nx = 0;
      for (const fi of faceIds) { offs.set(fi, nx); nx += faces[fi].free.length * 3; }
      const n = nx + rows.length;
      const split = (z) => {
        const dm = new Map();
        for (const fi of faceIds) dm.set(fi, z.subarray(offs.get(fi), offs.get(fi) + faces[fi].free.length * 3));
        return dm;
      };
      const applyA = (face, vec, o, out, oo) => {
        const m = face.free.length;
        for (let c = 0; c < 3; c++) for (let i = 0; i < m; i++) {
          let x = 0;
          for (let j = 0; j < m; j++) x += face.A[i * m + j] * vec[o + j * 3 + c];
          out[oo + i * 3 + c] = x;
        }
      };
      const applyK = (z) => {                        // the true operator
        const out = new Float64Array(n), dm = split(z), H = hessianTimes(rows, lamBar, dm);
        for (const fi of faceIds) {
          const o = offs.get(fi), h = H.get(fi);
          applyA(faces[fi], z, o, out, o);
          if (h) for (let i = 0; i < h.length; i++) out[o + i] += h[i];
        }
        rows.forEach((r, ri) => {
          const l = z[nx + ri];
          let jd = 0;
          for (const t of r.terms) {
            const o = offs.get(t.face);
            for (let k = 0; k < t.vars.length; k++) { jd += t.w[k] * z[o + t.vars[k]]; out[o + t.vars[k]] += t.w[k] * l; }
          }
          out[nx + ri] = jd - r.shift * l;
        });
        return out;
      };
      const applyPinv = (v) => {                     // the factorised operator, inverted through the Schur complement
        const U = new Map();
        for (const fi of faceIds) {
          const face = faces[fi], m = face.free.length, o = offs.get(fi), u = new Float64Array(m * 3);
          for (let c = 0; c < 3; c++) for (let i = 0; i < m; i++) {
            let x = 0;
            for (let j = 0; j < m; j++) x += face.Ainv[i * m + j] * v[o + j * 3 + c];
            u[i * 3 + c] = x;
          }
          U.set(fi, u);
        }
        const r2 = sizes.map((m) => new Float64Array(m));
        rows.forEach((r, ri) => {
          let x = -v[nx + ri];
          for (const t of r.terms) { const u = U.get(t.face); for (let k = 0; k < t.vars.length; k++) x += t.w[k] * u[t.vars[k]]; }
          r2[r.edge][r.at] = x;
        });
        const lam2 = solve(r2), out = new Float64Array(n);
        for (const [fi, { list, Ws }] of W) {
          const u = U.get(fi), o = offs.get(fi);
          for (let i = 0; i < u.length; i++) out[o + i] = u[i];
          list.forEach(([ri], q) => {
            const l = lam2[rows[ri].edge][rows[ri].at], w = Ws[q];
            for (let i = 0; i < w.length; i++) out[o + i] -= l * w[i];
          });
        }
        rows.forEach((r, ri) => { out[nx + ri] = lam2[r.edge][r.at]; });
        return out;
      };
      const bvec = new Float64Array(n);
      for (const [fi, { d }] of W) applyA(faces[fi], d, 0, bvec, offs.get(fi));
      rows.forEach((r, ri) => { bvec[nx + ri] = -r.value - r.shift * lamBar[ri]; });
      const z0 = new Float64Array(n);                // the Gauss-Newton step
      for (const fi of faceIds) z0.set(deltas.get(fi), offs.get(fi));
      rows.forEach((r, ri) => { z0[nx + ri] = nu[r.edge][r.at]; });
      const Kz0 = applyK(z0), r0 = bvec.map((x, i) => x - Kz0[i]);
      const dotv = (x, y) => { let t = 0; for (let i = 0; i < x.length; i++) t += x[i] * y[i]; return t; };
      const beta0 = Math.sqrt(dotv(r0, r0));
      if (beta0 > 0) {
        const V = [r0.map((x) => x / beta0)], Hh = [], cs = [], sn = [], g = [beta0];
        let k = 0;
        while (k < opt.SECOND_ORDER) {
          const w = applyK(applyPinv(V[k]));
          const h = new Float64Array(k + 2);
          for (let i = 0; i <= k; i++) { h[i] = dotv(w, V[i]); for (let t = 0; t < n; t++) w[t] -= h[i] * V[i][t]; }
          h[k + 1] = Math.sqrt(dotv(w, w));
          for (let i = 0; i < k; i++) { const t = cs[i] * h[i] + sn[i] * h[i + 1]; h[i + 1] = -sn[i] * h[i] + cs[i] * h[i + 1]; h[i] = t; }
          const rr = Math.hypot(h[k], h[k + 1]);
          const hk1 = h[k + 1];
          cs.push(h[k] / rr); sn.push(hk1 / rr);
          h[k] = rr; h[k + 1] = 0;
          g.push(-sn[k] * g[k]); g[k] = cs[k] * g[k];
          Hh.push(h);
          k++;
          if (Math.abs(g[k]) < 1e-8 * beta0 || hk1 === 0) break;
          V.push(w.map((x) => x / hk1));
        }
        const y = new Float64Array(k);
        for (let i = k - 1; i >= 0; i--) { let t = g[i]; for (let j = i + 1; j < k; j++) t -= Hh[j][i] * y[j]; y[i] = t / Hh[i][i]; }
        const corr = new Float64Array(n);
        for (let i = 0; i < k; i++) for (let t = 0; t < n; t++) corr[t] += y[i] * V[i][t];
        const dz = applyPinv(corr);
        gnDeltas = new Map([...deltas].map(([fi, d]) => [fi, Float64Array.from(d)]));
        gnNu = nu.map((x) => Float64Array.from(x));
        for (const fi of faceIds) { const d = deltas.get(fi), o = offs.get(fi); for (let i = 0; i < d.length; i++) d[i] += dz[o + i]; }
        rows.forEach((r, ri) => { nu[r.edge][r.at] += dz[nx + ri]; });
      }
    }
    const moves = [];
    for (const [fi, delta] of deltas) moves.push([faces[fi], delta, Float64Array.from(faces[fi].net)]);
    const newLam = new Float64Array(rows.length);
    let biggest = 0;
    rows.forEach((r, ri) => { newLam[ri] = nu[r.edge][r.at]; biggest = Math.max(biggest, Math.abs(newLam[ri])); });
    penalty = Math.max(penalty, 2 * biggest);
    const merit0 = objective() + penalty * l1(cond);
    const tryMoves = (lastTry) => {
      let alpha = 1, next;
      for (let tries = 0; tries <= 6; tries++) {
        for (const [face, delta, before] of moves) {
          face.net.set(before);
          face.free.forEach((cv, l) => { for (let c = 0; c < 3; c++) face.net[cv * 3 + c] += alpha * delta[l * 3 + c]; });
        }
        next = evaluate(true);
        if (opt.ACCEPT === "residual" ? next.residual < cond.residual : objective() + penalty * l1(next) < merit0) return { alpha, next, ok: true };
        if (tries === 6 && lastTry) return { alpha, next, ok: false };
        alpha /= 2;
      }
      return { alpha, next, ok: false };
    };
    let { alpha, next, ok } = tryMoves(!gnDeltas);
    if (!ok && gnDeltas) {                            // the Newton step failed: take the Gauss-Newton step
      newtonFallbacks++;
      for (const mv of moves) mv[1] = gnDeltas.get(faces.indexOf(mv[0]));
      ({ alpha, next } = tryMoves(true));
      rows.forEach((r, ri) => { newLam[ri] = gnNu[r.edge][r.at]; });
    }
    alphas.push(alpha);
    lamBar = newLam;
    cond = next;
  }
  const t3 = performance.now();

  const out = faces.map((face) => ({
    uKnots: face.K, vKnots: face.K, uCount: face.nc, vCount: face.nc, poles: Array.from(face.net), weights: null,
  }));
  return {
    faces: out,
    stats: {
      faces: faces.length, starFaces: faces.filter((f) => f.star).length, edgesSolved: cedges.length,
      steps, residual: history[history.length - 1], dropped, history, worst, alphas, newtonFallbacks,
      maxDev: (() => {
        let worst = 0;
        faces.forEach((face, fi) => {
          if (!face.star) return;
          for (let a = 0; a < N; a++) {
            const Bu = nzBasis(K6, 6, params[a]);
            for (let b = 0; b < N; b++) {
              const Bv = nzBasis(K6, 6, params[b]);
              let x = 0, y = 0, z = 0;
              for (let i = 0; i <= 6; i++) for (let j = 0; j <= 6; j++) {
                const w = Bu.ders[0][i] * Bv.ders[0][j], o = ((Bu.first + i) * 12 + Bv.first + j) * 3;
                x += w * face.net[o]; y += w * face.net[o + 1]; z += w * face.net[o + 2];
              }
              const o = (a * N + b) * 3, gr = grids[fi];
              worst = Math.max(worst, Math.hypot(x - gr[o], y - gr[o + 1], z - gr[o + 2]));
            }
          }
        });
        return worst;
      })(),
      ms: { limit: t1 - t0, fit: t2 - t1, solve: t3 - t2, total: t3 - t0, ...ms },
    },
  };
}
