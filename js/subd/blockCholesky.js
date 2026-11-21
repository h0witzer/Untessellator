/**
 * blockCholesky.js — Cholesky factorisation of a symmetric positive (semi)definite matrix stored as dense
 * blocks on a sparse block graph, with a greedy minimum-degree ordering of the blocks.
 *
 * Used for the Schur complement of the continuity solve (js/subd/toNurbs.js): one block per cage edge, two
 * blocks coupled when their edges share a face. A small relative ridge on the diagonal keeps redundant
 * constraints (the same condition reached from two edge halves) from stopping the factorisation.
 */

/** Order block nodes by greedy minimum degree; returns { order, adj } with adj the filled graph (Sets). */
function minimumDegree(adj0) {
  const n = adj0.length;
  const adj = adj0.map((s) => new Set(s));
  const filled = adj0.map((s) => new Set(s));
  const done = new Uint8Array(n);
  const order = [];
  for (let step = 0; step < n; step++) {
    let best = -1, bestDeg = Infinity;
    for (let i = 0; i < n; i++) if (!done[i] && adj[i].size < bestDeg) { best = i; bestDeg = adj[i].size; }
    done[best] = 1;
    order.push(best);
    const nb = [...adj[best]];
    for (const a of nb) {
      adj[a].delete(best);
      for (const b of nb) if (a !== b) { adj[a].add(b); filled[a].add(b); }
    }
  }
  return { order, filled };
}

/**
 * In-place lower Cholesky of a dense m x m row-major block. A pivot that falls to `drop` of the block's
 * largest diagonal belongs to a condition the earlier ones already imply: its row and column are dropped
 * (multiplier zero), which is exact for consistent redundant conditions. Returns the dropped mask.
 */
function cholesky(A, m, ridge, drop) {
  let scale = 0;
  for (let i = 0; i < m; i++) scale = Math.max(scale, Math.abs(A[i * m + i]));
  const eps = ridge * (scale || 1), tiny = drop * (scale || 1);
  const dropped = new Uint8Array(m);
  for (let j = 0; j < m; j++) {
    let d = A[j * m + j] + eps;
    for (let k = 0; k < j; k++) d -= A[j * m + k] * A[j * m + k];
    if (!(d > tiny)) {
      dropped[j] = 1;
      A[j * m + j] = 1;
      for (let i = j + 1; i < m; i++) A[i * m + j] = 0;
      for (let k = 0; k < j; k++) A[j * m + k] = 0;
    } else {
      const ljj = Math.sqrt(d);
      A[j * m + j] = ljj;
      for (let i = j + 1; i < m; i++) {
        let s = A[i * m + j];
        for (let k = 0; k < j; k++) s -= A[i * m + k] * A[j * m + k];
        A[i * m + j] = s / ljj;
      }
    }
    for (let i = 0; i < j; i++) A[i * m + j] = 0;
  }
  return dropped;
}

/** B <- B L^-T for B (r x m), L lower (m x m); dropped columns become zero. */
function solveRight(B, r, L, m, dropped) {
  for (let i = 0; i < r; i++) {
    for (let j = 0; j < m; j++) {
      if (dropped[j]) { B[i * m + j] = 0; continue; }
      let s = B[i * m + j];
      for (let k = 0; k < j; k++) s -= B[i * m + k] * L[j * m + k];
      B[i * m + j] = s / L[j * m + j];
    }
  }
}

/** C (r x c) -= A (r x m) B^T (c x m). */
function subtractABt(C, A, B, r, c, m) {
  for (let i = 0; i < r; i++) {
    for (let j = 0; j < c; j++) {
      let s = 0;
      const ai = i * m, bj = j * m;
      for (let k = 0; k < m; k++) s += A[ai + k] * B[bj + k];
      C[i * c + j] -= s;
    }
  }
}

/**
 * Factor and solve. sizes[i]: rows of block i; blocks: Map "i,j" (i >= j, original indices) -> Float64Array
 * (sizes[i] x sizes[j], row-major) holding the lower half; adj: Array of Sets (the block graph).
 * Returns solve(rhs: Array of Float64Array per block) -> Array of Float64Array.
 */
export function factorBlocks(sizes, blocks, adj, ridge = 1e-14, drop = 1e-11) {
  const n = sizes.length;
  const { order, filled } = minimumDegree(adj);
  const pos = new Int32Array(n);
  order.forEach((b, k) => { pos[b] = k; });
  const L = new Map();                              // key "p,q" in elimination positions, p >= q
  const get = (p, q) => L.get(p * n + q);
  for (const [key, M] of blocks) {
    const [i, j] = key.split(',').map(Number);
    let p = pos[i], q = pos[j];
    let A = M, ri = sizes[i], cj = sizes[j];
    if (p < q) {                                    // store transposed so p >= q
      const T = new Float64Array(ri * cj);
      for (let a = 0; a < ri; a++) for (let b = 0; b < cj; b++) T[b * ri + a] = M[a * cj + b];
      A = T; [p, q] = [q, p];
    } else A = Float64Array.from(M);
    L.set(p * n + q, A);
  }
  const size = (p) => sizes[order[p]];
  const below = order.map((b) => [...filled[b]].map((x) => pos[x]).filter((x) => x > pos[b]).sort((a, c) => a - c));
  const droppedAt = new Array(n);
  for (let k = 0; k < n; k++) {
    const mk = size(k);
    let Lkk = get(k, k);
    if (!Lkk) { Lkk = new Float64Array(mk * mk); L.set(k * n + k, Lkk); }
    droppedAt[k] = cholesky(Lkk, mk, ridge, drop);
    const nb = below[k];
    for (const i of nb) {
      let Lik = get(i, k);
      if (!Lik) { Lik = new Float64Array(size(i) * mk); L.set(i * n + k, Lik); }
      solveRight(Lik, size(i), Lkk, mk, droppedAt[k]);
    }
    for (let a = 0; a < nb.length; a++) {
      const i = nb[a], Lik = get(i, k);
      for (let b = 0; b <= a; b++) {
        const j = nb[b], Ljk = get(j, k);
        let C = get(i, j);
        if (!C) { C = new Float64Array(size(i) * size(j)); L.set(i * n + j, C); }
        subtractABt(C, Lik, Ljk, size(i), size(j), mk);
      }
    }
  }
  const lowerOf = order.map(() => []);              // for each column q: rows p > q with blocks
  for (const key of L.keys()) {
    const p = Math.floor(key / n), q = key % n;
    if (p > q) lowerOf[q].push(p);
  }
  return function solve(rhs) {
    const y = order.map((b) => Float64Array.from(rhs[b]));
    for (let q = 0; q < n; q++) {                   // forward: L y = b
      const mq = size(q), Lqq = get(q, q), yq = y[q], dq = droppedAt[q];
      for (let i = 0; i < mq; i++) {
        if (dq[i]) { yq[i] = 0; continue; }
        let s = yq[i];
        for (let k = 0; k < i; k++) s -= Lqq[i * mq + k] * yq[k];
        yq[i] = s / Lqq[i * mq + i];
      }
      for (const p of lowerOf[q]) {
        const Lpq = get(p, q), mp = size(p), yp = y[p];
        for (let i = 0; i < mp; i++) {
          let s = 0;
          for (let k = 0; k < mq; k++) s += Lpq[i * mq + k] * yq[k];
          yp[i] -= s;
        }
      }
    }
    for (let q = n - 1; q >= 0; q--) {              // backward: L^T x = y
      const mq = size(q), Lqq = get(q, q), yq = y[q];
      for (const p of lowerOf[q]) {
        const Lpq = get(p, q), mp = size(p), yp = y[p];
        for (let k = 0; k < mq; k++) {
          let s = 0;
          for (let i = 0; i < mp; i++) s += Lpq[i * mq + k] * yp[i];
          yq[k] -= s;
        }
      }
      const dq = droppedAt[q];
      for (let i = mq - 1; i >= 0; i--) {
        if (dq[i]) { yq[i] = 0; continue; }
        let s = yq[i];
        for (let k = i + 1; k < mq; k++) s -= Lqq[k * mq + i] * yq[k];
        yq[i] = s / Lqq[i * mq + i];
      }
    }
    const out = new Array(n);
    order.forEach((b, p) => { out[b] = y[p]; });
    return out;
  };
}
