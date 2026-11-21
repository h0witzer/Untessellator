/**
 * envelopeCholesky.js — Cholesky of a sparse symmetric positive definite matrix in envelope (skyline) storage,
 * rows in reverse Cuthill-McKee order. The pattern is fixed once (envelope(n, adjacency)); each numeric
 * factorisation then costs about n b^2 / 2 for a bandwidth b, against n^3 / 3 dense.
 */

/** Reverse Cuthill-McKee order of the graph adj (Array of iterables), from a pseudo-peripheral start per component. */
function reverseCuthillMcKee(adj) {
  const n = adj.length, deg = adj.map((a) => a.length), seen = new Uint8Array(n), order = [];
  const bfs = (s, mark) => {                          // levels from s; returns the last level's vertices
    const lvl = new Int32Array(n).fill(-1); lvl[s] = 0;
    let front = [s], last = front;
    while (front.length) {
      last = front;
      const next = [];
      for (const v of front) for (const w of adj[v]) if (lvl[w] < 0) { lvl[w] = lvl[v] + 1; next.push(w); }
      front = next;
    }
    return { last, depth: Math.max(...lvl) };
  };
  for (let s0 = 0; s0 < n; s0++) {
    if (seen[s0]) continue;
    let s = s0, { last, depth } = bfs(s);
    for (let it = 0; it < 4; it++) {                  // George-Liu: move to a far vertex of least degree
      const t = last.reduce((a, b) => (deg[b] < deg[a] ? b : a));
      const r = bfs(t);
      if (r.depth <= depth) break;
      s = t; last = r.last; depth = r.depth;
    }
    const queue = [s]; seen[s] = 1;
    for (let h = 0; h < queue.length; h++) {
      const v = queue[h];
      order.push(v);
      const nb = adj[v].filter((w) => !seen[w]).sort((a, b) => deg[a] - deg[b]);
      for (const w of nb) { seen[w] = 1; queue.push(w); }
    }
  }
  return order.reverse();
}

/**
 * The envelope for a pattern: adj[i] lists j != i with A[i][j] possibly nonzero (symmetric). Returns an object
 * with add(i, j, x) (original indices, either triangle), factor(), solve(b) and clear().
 */
export function envelope(n, adj) {
  const order = reverseCuthillMcKee(adj.map((a) => [...a]));
  const pos = new Int32Array(n);
  order.forEach((v, k) => { pos[v] = k; });
  const first = new Int32Array(n), offset = new Int32Array(n + 1);
  for (let v = 0; v < n; v++) {
    const i = pos[v];
    let f = i;
    for (const w of adj[v]) f = Math.min(f, pos[w]);
    first[i] = f;
  }
  for (let i = 0; i < n; i++) offset[i + 1] = offset[i] + (i - first[i] + 1);
  const E = new Float64Array(offset[n]);
  const at = (i, j) => offset[i] + (j - first[i]);     // permuted i >= j
  return {
    size: offset[n],
    clear() { E.fill(0); },
    add(a, b, x) {
      let i = pos[a], j = pos[b];
      if (i < j) [i, j] = [j, i];
      if (j < first[i]) throw new Error(`envelope: (${a}, ${b}) is outside the pattern`);
      E[at(i, j)] += x;
    },
    factor() {
      for (let i = 0; i < n; i++) {
        const fi = first[i], oi = offset[i] - fi;
        for (let j = fi; j < i; j++) {
          const fj = first[j], oj = offset[j] - fj;
          let s = E[oi + j];
          for (let k = Math.max(fi, fj); k < j; k++) s -= E[oi + k] * E[oj + k];
          E[oi + j] = s / E[oj + j];
        }
        let d = E[oi + i];
        for (let k = fi; k < i; k++) d -= E[oi + k] * E[oi + k];
        E[oi + i] = Math.sqrt(Math.max(d, 1e-300));
      }
    },
    solve(b) {
      const y = new Float64Array(n);
      for (let v = 0; v < n; v++) y[pos[v]] = b[v];
      for (let i = 0; i < n; i++) {
        const fi = first[i], oi = offset[i] - fi;
        let s = y[i];
        for (let k = fi; k < i; k++) s -= E[oi + k] * y[k];
        y[i] = s / E[oi + i];
      }
      for (let i = n - 1; i >= 0; i--) {
        const fi = first[i], oi = offset[i] - fi;
        y[i] /= E[oi + i];
        const yi = y[i];
        for (let k = fi; k < i; k++) y[k] -= E[oi + k] * yi;
      }
      const x = new Float64Array(n);
      for (let v = 0; v < n; v++) x[v] = y[pos[v]];
      return x;
    },
  };
}
