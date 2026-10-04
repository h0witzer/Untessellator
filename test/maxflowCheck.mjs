/**
 * maxflowCheck.mjs — js/maxflow.js against a plain Edmonds-Karp solver on random graphs.
 *
 * Usage:  node test/maxflowCheck.mjs [trials]
 *
 * Checks, per graph: the two flows agree, and the cut js/maxflow.js reports (onSource) has exactly
 * that capacity, so the labelling it hands a fill is a minimum cut and not just a flow value.
 */

import { MaxFlow } from '../js/maxflow.js';

const trials = Number(process.argv[2] ?? 400);
let seed = 7;
const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;

function edmondsKarp(n, s, t, edges) {
  const cap = Array.from({ length: n }, () => new Map());
  const add = (u, v, c) => cap[u].set(v, (cap[u].get(v) ?? 0) + c);
  for (const [u, v, c] of edges) { add(u, v, c); if (!cap[v].has(u)) cap[v].set(u, 0); }
  let flow = 0;
  for (;;) {
    const prev = new Int32Array(n).fill(-1);
    prev[s] = s;
    const q = [s];
    while (q.length && prev[t] < 0) {
      const u = q.shift();
      for (const [v, c] of cap[u]) if (c > 1e-12 && prev[v] < 0) { prev[v] = u; q.push(v); }
    }
    if (prev[t] < 0) return flow;
    let b = Infinity;
    for (let v = t; v !== s; v = prev[v]) b = Math.min(b, cap[prev[v]].get(v));
    for (let v = t; v !== s; v = prev[v]) {
      cap[prev[v]].set(v, cap[prev[v]].get(v) - b);
      cap[v].set(prev[v], (cap[v].get(prev[v]) ?? 0) + b);
    }
    flow += b;
  }
}

let worst = 0;
for (let k = 0; k < trials; k++) {
  const n = 2 + Math.floor(rnd() * 40);
  const g = new MaxFlow(n, 4);
  const S = n, T = n + 1;
  const edges = [];
  const tw = [];
  for (let i = 0; i < n; i++) {
    const a = rnd() < 0.4 ? rnd() * 5 : 0;
    const b = rnd() < 0.4 ? rnd() * 5 : 0;
    // several calls per node exercise the signed residual bookkeeping
    g.addTweights(i, a, b);
    g.addTweights(i, a * 0.5, b * 0.25);
    tw.push([i, 1.5 * a, 1.25 * b]);
    if (1.5 * a > 0) edges.push([S, i, 1.5 * a]);
    if (1.25 * b > 0) edges.push([i, T, 1.25 * b]);
  }
  const m = Math.floor(rnd() * n * 3);
  const pairs = [];
  for (let e = 0; e < m; e++) {
    const i = Math.floor(rnd() * n), j = Math.floor(rnd() * n);
    if (i === j) continue;
    const c = rnd() < 0.8 ? rnd() * 3 : 0, r = rnd() < 0.8 ? rnd() * 3 : 0;
    g.addEdge(i, j, c, r);
    pairs.push([i, j, c, r]);
    if (c > 0) edges.push([i, j, c]);
    if (r > 0) edges.push([j, i, r]);
  }
  const f = g.maxflow();
  const ref = edmondsKarp(n + 2, S, T, edges);
  // capacity of the reported cut: terminal links cut plus arcs from source side to sink side
  let cut = 0;
  for (const [i, a, b] of tw) cut += g.onSource(i) ? b : a;
  for (const [i, j, c, r] of pairs) {
    if (g.onSource(i) && !g.onSource(j)) cut += c;
    if (g.onSource(j) && !g.onSource(i)) cut += r;
  }
  const err = Math.max(Math.abs(f - ref), Math.abs(cut - ref)) / Math.max(1, ref);
  worst = Math.max(worst, err);
  if (err > 1e-9) {
    console.error(`FAIL trial ${k}: n ${n}, maxflow ${f}, reference ${ref}, reported cut ${cut}`);
    process.exit(1);
  }
}
console.log(`PASS: ${trials} random graphs, worst relative error ${worst.toExponential(2)}`);
