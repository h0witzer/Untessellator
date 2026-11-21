/**
 * kpCap.js — G2 caps of one bi-6 patch per face around a star point, after Karciauskas and Peters, "Minimal bi-6
 * G2 completion of bicubic spline surfaces", CAGD 41 (2016) 10-22.
 *
 * A star point of valence n whose surrounding faces are regular gets n bi-6 patches (7 x 7 Bezier coefficients),
 * one per face around it, joining each other G2 along the spokes and the regular bicubic neighbours G2 along the
 * outer sides. The paper fixes the reparameterisation, which makes every constraint linear:
 *   spokes (u from the star point, v into each patch):
 *     f~_v + f_v - b f_u = 0,   f~_vv - f_vv + 2 b f_uv - b^2 f_uu - e f_u - d f_v = 0,
 *     b = 2c (1-u)^2,  d = delta (1-u)^2 / (1 - u + tau u),  delta = 24 c (tau - 1),  e = b (b' - d/2),  c = cos(2pi/n)
 *   outer sides: the neighbours' data b reparameterised by beta, (b o beta)_nn = b_nn + e_b b_t + d_b b_n with the
 *     cubic d_b = (0, 0, d2, 2c/3 + d2) and quartic e_b = (0, 0, d2/2, (4c(1-tau) + tau d2)/(4 tau), 0) in Bernstein
 *     form, t from the face's far corner to the spoke's end, n into the patch;
 *   the star point at the Catmull-Clark limit point; tau and d2 per valence as published;
 *   the remaining coefficients minimise F5, the integral of the fifth derivatives squared (binomially weighted).
 * Everything is linear in the data, coordinate by coordinate, so the cap is a fixed matrix per valence applied to
 * the outer data: solved here numerically once per valence (not transcribed from the paper's appendix), with the
 * constraints' consistency measured, and cached.
 */

const DEG = 6;

function binom(n, k) { let r = 1; for (let i = 1; i <= k; i++) r = (r * (n - k + i)) / i; return r; }

/** Bernstein basis of degree p at t and its derivatives up to order nd: ders[k][i]. */
export function bernstein(p, t, nd = 2) {
  const out = [];
  for (let k = 0; k <= nd; k++) {
    const row = new Float64Array(p + 1);
    if (k <= p) {
      const lower = new Float64Array(p - k + 1);
      for (let i = 0; i <= p - k; i++) lower[i] = binom(p - k, i) * t ** i * (1 - t) ** (p - k - i);
      // d^k/dt^k B^p_i = p!/(p-k)! sum_j (-1)^(k-j) C(k, j) B^(p-k)_(i-j)
      let fac = 1; for (let q = 0; q < k; q++) fac *= p - q;
      for (let i = 0; i <= p; i++) {
        let s = 0;
        for (let j = 0; j <= k; j++) { const m = i - j; if (m >= 0 && m <= p - k) s += (k - j) % 2 ? -binom(k, j) * lower[m] : binom(k, j) * lower[m]; }
        row[i] = fac * s;
      }
    }
    out.push(row);
  }
  return out;
}

/** Per-valence constants of the construction (paper sections 3.1-3.6). */
export function valenceConstants(n) {
  const c = Math.cos(2 * Math.PI / n);
  const tau = n === 3 ? 4 / 3 : ({ 5: 0.87, 6: 0.85, 7: 0.84, 8: 0.83, 9: 0.82, 10: 0.81 })[n] ?? 0.8;
  const d2 = n === 3 ? 0.1 : (1 - (3 + 2 * c) * tau + (3 + c) * tau * tau) / (2 * tau * tau);
  const delta = 24 * c * (tau - 1);
  const dB = [0, 0, d2, 2 * c / 3 + d2];
  const eB = [0, 0, d2 / 2, (4 * c * (1 - tau) + tau * d2) / (4 * tau), 0];
  const bern = (coef, t) => coef.reduce((s, x, i) => s + x * binom(coef.length - 1, i) * t ** i * (1 - t) ** (coef.length - 1 - i), 0);
  return {
    c, tau, d2, delta,
    b: (u) => 2 * c * (1 - u) ** 2,
    db: (u) => -4 * c * (1 - u),
    dTimesDen: (u) => delta * (1 - u) ** 2,                  // d (1 - u + tau u)
    den: (u) => 1 - u + tau * u,
    dBorder: (t) => bern(dB, t),
    eBorder: (t) => bern(eB, t),
    // limit-point weights: centre, each edge neighbour, each face-diagonal neighbour
    limit: { centre: 1 - n * (1 / (n * (n + 5)) + (n === 3 ? 1 / 96 : 0)) - n * 4 / (n * (n + 5)),
      edge: 4 / (n * (n + 5)), diagonal: 1 / (n * (n + 5)) + (n === 3 ? 1 / 96 : 0) },
  };
}

/** Symmetric eigen-decomposition by cyclic Jacobi rotations: { values, vectors (columns, row-major n x n) }. */
function jacobiEigen(A0, n) {
  const A = Float64Array.from(A0), V = new Float64Array(n * n);
  for (let i = 0; i < n; i++) V[i * n + i] = 1;
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p * n + q] ** 2;
    if (off < 1e-30) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      const apq = A[p * n + q];
      if (Math.abs(apq) < 1e-300) continue;
      const theta = (A[q * n + q] - A[p * n + p]) / (2 * apq);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const cs = 1 / Math.sqrt(t * t + 1), sn = t * cs;
      for (let k = 0; k < n; k++) {
        const akp = A[k * n + p], akq = A[k * n + q];
        A[k * n + p] = cs * akp - sn * akq; A[k * n + q] = sn * akp + cs * akq;
      }
      for (let k = 0; k < n; k++) {
        const apk = A[p * n + k], aqk = A[q * n + k];
        A[p * n + k] = cs * apk - sn * aqk; A[q * n + k] = sn * apk + cs * aqk;
      }
      for (let k = 0; k < n; k++) {
        const vkp = V[k * n + p], vkq = V[k * n + q];
        V[k * n + p] = cs * vkp - sn * vkq; V[k * n + q] = sn * vkp + cs * vkq;
      }
    }
  }
  return { values: Array.from({ length: n }, (_, i) => A[i * n + i]), vectors: V };
}

/** F5 for one bi-6 patch: x^T Q x with Q = sum_{i+j=5} C(5,i) M_i (x) M_j, M_a the Gram of a-th derivatives. */
function f5Hessian() {
  const gx = [-0.9491079123427585, -0.7415311855993945, -0.4058451513773972, 0, 0.4058451513773972, 0.7415311855993945, 0.9491079123427585];
  const gw = [0.1294849661688697, 0.2797053914892766, 0.3818300505051189, 0.4179591836734694, 0.3818300505051189, 0.2797053914892766, 0.1294849661688697];
  const m = DEG + 1, M = Array.from({ length: 6 }, () => new Float64Array(m * m));
  gx.forEach((x, q) => {
    const B = bernstein(DEG, (x + 1) / 2, 5);
    for (let a = 0; a <= 5; a++) for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) M[a][i * m + j] += 0.5 * gw[q] * B[a][i] * B[a][j];
  });
  const n = m * m, Q = new Float64Array(n * n);
  for (let a = 0; a <= 5; a++) {
    const w = binom(5, a), Mu = M[a], Mv = M[5 - a];
    for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) for (let k = 0; k < m; k++) for (let l = 0; l < m; l++) {
      Q[(i * m + j) * n + k * m + l] += w * Mu[i * m + k] * Mv[j * m + l];
    }
  }
  return Q;
}

/**
 * Where each coefficient P^k[i][j] of a valence-n cap comes from. Patch k: u along spoke k (P[i][0]), v along spoke
 * k+1 (P[0][j]), far corner P[6][6]. Data per patch (33 values): rows r = 0, 1, 2 from the outer side u = 1 at all
 * positions j (21), then rows r from the side v = 1 at positions i = 0..3 (12); data 0 is the star point.
 */
function sources(n) {
  const nvar = 12 * n, ndata = 33 * n + 1;
  const src = (k, i, j) => {
    k = ((k % n) + n) % n;
    if (i === 0 && j === 0) return { data: 0 };
    if (i >= 4) return { data: 1 + 33 * k + (6 - i) * 7 + j };
    if (j >= 4) return { data: 1 + 33 * k + 21 + (6 - j) * 4 + i };
    if (j === 0) return { variable: 3 * k + (i - 1) };              // spoke k, interior points 1..3
    if (i === 0) return { variable: 3 * ((k + 1) % n) + (j - 1) };  // spoke k + 1
    return { variable: 3 * n + 9 * k + (i - 1) * 3 + (j - 1) };
  };
  return { nvar, ndata, src };
}

const cache = new Map();

/**
 * The cap operator for valence n: { K (nvar x ndata), nvar, ndata, src, residual } with variables = K data. The
 * residual is how far the constraints are from being satisfiable for arbitrary data (0 when consistent).
 */
export function capOperator(n, variant = {}) {
  const key = `${n}:${JSON.stringify(variant)}`;
  if (cache.has(key)) return cache.get(key);
  const V = valenceConstants(n), { nvar, ndata, src } = sources(n);
  const rows = [];                                   // { a: Map var -> w, r: Map data -> w } meaning sum a x = sum r d
  const flipSpoke = !!variant.flipSpoke;
  const addTerm = (row, k, i, j, w) => {
    if (w === 0) return;
    const s = src(k, i, j);
    if (s.variable !== undefined) row.a.set(s.variable, (row.a.get(s.variable) ?? 0) + w);
    else row.r.set(s.data, (row.r.get(s.data) ?? 0) - w);   // moved to the right-hand side
  };
  const cheb = (m) => Array.from({ length: m }, (_, q) => (1 - Math.cos(Math.PI * (q + 0.5) / m)) / 2);
  for (let k = 0; k < n; k++) {
    // spoke k: patch k (s = u, t = v) and patch k-1 (s = v, t = u); f~ = patch k, f = patch k-1 (or flipped)
    const P = [{ k, along: 'u' }, { k: k - 1, along: 'v' }];
    const [ft, f] = flipSpoke ? [P[1], P[0]] : P;
    const deriv = (pt, s, ds, dt) => {               // weights of d^ds/ds d^dt/dt of patch pt at (s, 0)
      const Bs = bernstein(DEG, s, 2), B0 = bernstein(DEG, 0, 2), terms = [];
      for (let a = 0; a <= DEG; a++) for (let b = 0; b <= DEG; b++) {
        const w = Bs[ds][a] * B0[dt][b];
        if (w !== 0) terms.push(pt.along === 'u' ? [pt.k, a, b, w] : [pt.k, b, a, w]);
      }
      return terms;
    };
    for (const s of cheb(9)) {                        // G1: f~_t + f_t - b f_s = 0
      const row = { a: new Map(), r: new Map(), kind: 'g1' };
      for (const [kk, i, j, w] of deriv(ft, s, 0, 1)) addTerm(row, kk, i, j, w);
      for (const [kk, i, j, w] of deriv(f, s, 0, 1)) addTerm(row, kk, i, j, w);
      for (const [kk, i, j, w] of deriv(f, s, 1, 0)) addTerm(row, kk, i, j, -V.b(s) * w);
      rows.push(row);
    }
    for (const s of cheb(12)) {                       // G2 times (1 - s + tau s)
      const den = V.den(s), b = V.b(s), dd = V.dTimesDen(s);
      const eTimesDen = b * V.db(s) * den - b * dd / 2;
      const row = { a: new Map(), r: new Map(), kind: 'g2' };
      for (const [kk, i, j, w] of deriv(ft, s, 0, 2)) addTerm(row, kk, i, j, den * w);
      for (const [kk, i, j, w] of deriv(f, s, 0, 2)) addTerm(row, kk, i, j, -den * w);
      for (const [kk, i, j, w] of deriv(f, s, 1, 1)) addTerm(row, kk, i, j, 2 * b * den * w);
      for (const [kk, i, j, w] of deriv(f, s, 2, 0)) addTerm(row, kk, i, j, -b * b * den * w);
      for (const [kk, i, j, w] of deriv(f, s, 1, 0)) addTerm(row, kk, i, j, -eTimesDen * w);
      for (const [kk, i, j, w] of deriv(f, s, 0, 1)) addTerm(row, kk, i, j, -dd * w);
      rows.push(row);
    }
  }
  // C y = R d, solved by least squares through the eigen-decomposition of C^T C
  const m = rows.length;
  const C = new Float64Array(m * nvar), R = new Float64Array(m * ndata);
  rows.forEach((row, q) => {
    for (const [v, w] of row.a) C[q * nvar + v] = w;
    for (const [d, w] of row.r) R[q * ndata + d] = w;
  });
  // row scaling to unit norm
  for (let q = 0; q < m; q++) {
    let s = 0; for (let v = 0; v < nvar; v++) s += C[q * nvar + v] ** 2;
    s = Math.sqrt(s) || 1;
    for (let v = 0; v < nvar; v++) C[q * nvar + v] /= s;
    for (let d = 0; d < ndata; d++) R[q * ndata + d] /= s;
  }
  const CtC = new Float64Array(nvar * nvar);
  for (let q = 0; q < m; q++) for (let a = 0; a < nvar; a++) { const x = C[q * nvar + a]; if (x) for (let b = 0; b < nvar; b++) CtC[a * nvar + b] += x * C[q * nvar + b]; }
  const { values, vectors } = jacobiEigen(CtC, nvar);
  const lmax = Math.max(...values), range = [], nulls = [];
  values.forEach((l, i) => (l > 1e-11 * lmax ? range : nulls).push(i));
  // particular solution operator y_p = C^+ R = V_r L^-1 V_r^T C^T R
  const CtR = new Float64Array(nvar * ndata);
  for (let q = 0; q < m; q++) for (let a = 0; a < nvar; a++) { const x = C[q * nvar + a]; if (x) for (let d = 0; d < ndata; d++) CtR[a * ndata + d] += x * R[q * ndata + d]; }
  const Yp = new Float64Array(nvar * ndata);
  for (const e of range) {
    const proj = new Float64Array(ndata);
    for (let a = 0; a < nvar; a++) { const va = vectors[a * nvar + e]; if (va) for (let d = 0; d < ndata; d++) proj[d] += va * CtR[a * ndata + d]; }
    for (let a = 0; a < nvar; a++) { const va = vectors[a * nvar + e] / values[e]; if (va) for (let d = 0; d < ndata; d++) Yp[a * ndata + d] += va * proj[d]; }
  }
  // consistency: || C Yp - R || relative to || R ||
  let res = 0, rn = 0;
  for (let q = 0; q < m; q++) for (let d = 0; d < ndata; d++) {
    let s = 0; for (let a = 0; a < nvar; a++) s += C[q * nvar + a] * Yp[a * ndata + d];
    res = Math.max(res, Math.abs(s - R[q * ndata + d])); rn = Math.max(rn, Math.abs(R[q * ndata + d]));
  }
  // F5 over all patches: x_k = E_k y + D_k d; minimise over the null space y = Yp d + Z w
  const Q = f5Hessian(), np = 49;
  const Hy = new Float64Array(nvar * nvar), Gd = new Float64Array(nvar * ndata);
  for (let k = 0; k < n; k++) {
    const E = new Float64Array(np * nvar), D = new Float64Array(np * ndata);
    for (let i = 0; i <= DEG; i++) for (let j = 0; j <= DEG; j++) {
      const s = src(k, i, j), c = i * 7 + j;
      if (s.variable !== undefined) E[c * nvar + s.variable] = 1; else D[c * ndata + s.data] = 1;
    }
    const QE = new Float64Array(np * nvar), QD = new Float64Array(np * ndata);
    for (let a = 0; a < np; a++) for (let b = 0; b < np; b++) {
      const q = Q[a * np + b]; if (!q) continue;
      for (let v = 0; v < nvar; v++) QE[a * nvar + v] += q * E[b * nvar + v];
      for (let d = 0; d < ndata; d++) QD[a * ndata + d] += q * D[b * ndata + d];
    }
    for (let a = 0; a < np; a++) for (let v = 0; v < nvar; v++) {
      const e = E[a * nvar + v]; if (!e) continue;
      for (let w = 0; w < nvar; w++) Hy[v * nvar + w] += e * QE[a * nvar + w];
      for (let d = 0; d < ndata; d++) Gd[v * ndata + d] += e * QD[a * ndata + d];
    }
  }
  const z = nulls.length;
  const Z = new Float64Array(nvar * z);
  nulls.forEach((e, c) => { for (let a = 0; a < nvar; a++) Z[a * z + c] = vectors[a * nvar + e]; });
  // w = -(Z^T H Z)^-1 Z^T (H Yp + G) d
  const HZ = new Float64Array(nvar * z);
  for (let a = 0; a < nvar; a++) for (let b = 0; b < nvar; b++) { const h = Hy[a * nvar + b]; if (h) for (let c = 0; c < z; c++) HZ[a * z + c] += h * Z[b * z + c]; }
  const ZHZ = new Float64Array(z * z);
  for (let a = 0; a < nvar; a++) for (let c1 = 0; c1 < z; c1++) { const za = Z[a * z + c1]; if (za) for (let c2 = 0; c2 < z; c2++) ZHZ[c1 * z + c2] += za * HZ[a * z + c2]; }
  const T = new Float64Array(nvar * ndata);                  // H Yp + G
  for (let a = 0; a < nvar; a++) for (let b = 0; b < nvar; b++) { const h = Hy[a * nvar + b]; if (h) for (let d = 0; d < ndata; d++) T[a * ndata + d] += h * Yp[b * ndata + d]; }
  for (let i = 0; i < nvar * ndata; i++) T[i] += Gd[i];
  const ZtT = new Float64Array(z * ndata);
  for (let a = 0; a < nvar; a++) for (let c = 0; c < z; c++) { const za = Z[a * z + c]; if (za) for (let d = 0; d < ndata; d++) ZtT[c * ndata + d] += za * T[a * ndata + d]; }
  const eig = jacobiEigen(ZHZ, z), W = new Float64Array(z * ndata);
  const emax = Math.max(...eig.values);
  eig.values.forEach((l, e) => {
    if (l <= 1e-12 * emax) return;
    const proj = new Float64Array(ndata);
    for (let c = 0; c < z; c++) { const v = eig.vectors[c * z + e]; if (v) for (let d = 0; d < ndata; d++) proj[d] += v * ZtT[c * ndata + d]; }
    for (let c = 0; c < z; c++) { const v = eig.vectors[c * z + e] / l; if (v) for (let d = 0; d < ndata; d++) W[c * ndata + d] -= v * proj[d]; }
  });
  const K = Float64Array.from(Yp);
  for (let a = 0; a < nvar; a++) for (let c = 0; c < z; c++) { const za = Z[a * z + c]; if (za) for (let d = 0; d < ndata; d++) K[a * ndata + d] += za * W[c * ndata + d]; }
  const op = { n, K, nvar, ndata, src, residual: rn ? res / rn : res, rank: range.length, nullity: z, constants: V };
  cache.set(key, op);
  return op;
}

/** The n patches (each 49 coefficients, row-major P[i][j], per coordinate c) from the data, by the operator. */
export function applyCap(op, data /* Float64Array ndata x 3 */) {
  const { K, nvar, ndata, src, n } = op;
  const y = new Float64Array(nvar * 3);
  for (let v = 0; v < nvar; v++) {
    let x = 0, yy = 0, z = 0;
    const o = v * ndata;
    for (let d = 0; d < ndata; d++) { const k = K[o + d]; x += k * data[d * 3]; yy += k * data[d * 3 + 1]; z += k * data[d * 3 + 2]; }
    y[v * 3] = x; y[v * 3 + 1] = yy; y[v * 3 + 2] = z;
  }
  const patches = [];
  for (let k = 0; k < n; k++) {
    const P = new Float64Array(49 * 3);
    for (let i = 0; i <= DEG; i++) for (let j = 0; j <= DEG; j++) {
      const s = src(k, i, j), from = s.variable !== undefined ? y.subarray(s.variable * 3, s.variable * 3 + 3) : data.subarray(s.data * 3, s.data * 3 + 3);
      P.set(from, (i * 7 + j) * 3);
    }
    patches.push(P);
  }
  return patches;
}
