/**
 * isotropic.js — incremental isotropic remeshing (Botsch & Kobbelt 2004) of a closed triangle mesh to edge length
 * L: split edges longer than 4/3 L, collapse edges shorter than 4/5 L, flip edges towards valence 6, relax each
 * vertex tangentially and put it back on the input surface (closest point). QuadriFlow's field follows the input's
 * triangles; CAD exports' slivers (a 254 mm edge beside 0.7 mm ones) leave it nothing to follow, so the cage is
 * made from this mesh instead. The fit still uses the input's own vertices.
 */

/** Closest point on triangle abc to p (Ericson, Real-Time Collision Detection 5.1.5). */
function closestOnTriangle(p, a, b, c) {
  const ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]], ap = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
  const dot = (x, y) => x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
  const at = (s, t) => [a[0] + s * ab[0] + t * ac[0], a[1] + s * ab[1] + t * ac[1], a[2] + s * ab[2] + t * ac[2]];
  const d1 = dot(ab, ap), d2 = dot(ac, ap);
  if (d1 <= 0 && d2 <= 0) return [...a];
  const bp = [p[0] - b[0], p[1] - b[1], p[2] - b[2]], d3 = dot(ab, bp), d4 = dot(ac, bp);
  if (d3 >= 0 && d4 <= d3) return [...b];
  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) return at(d1 / (d1 - d3), 0);
  const cp = [p[0] - c[0], p[1] - c[1], p[2] - c[2]], d5 = dot(ab, cp), d6 = dot(ac, cp);
  if (d6 >= 0 && d5 <= d6) return [...c];
  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) return at(0, d2 / (d2 - d6));
  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) { const w = (d4 - d3) / ((d4 - d3) + (d5 - d6)); return [b[0] + w * (c[0] - b[0]), b[1] + w * (c[1] - b[1]), b[2] + w * (c[2] - b[2])]; }
  const den = 1 / (va + vb + vc);
  return at(vb * den, vc * den);
}

/** Closest point on a triangle mesh: a uniform grid of its triangles, searched in shells of cells. */
function projector(V, T, cell) {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const p of V) for (let c = 0; c < 3; c++) { lo[c] = Math.min(lo[c], p[c]); hi[c] = Math.max(hi[c], p[c]); }
  const n = [0, 1, 2].map((c) => Math.floor((hi[c] - lo[c]) / cell) + 1);
  const at = (x, c) => Math.min(n[c] - 1, Math.max(0, Math.floor((x - lo[c]) / cell)));
  const grid = new Map(), key = (i, j, k) => i + n[0] * (j + n[1] * k);
  T.forEach((t, f) => {
    const a = V[t[0]], b = V[t[1]], c = V[t[2]];
    const i0 = [0, 1, 2].map((d) => at(Math.min(a[d], b[d], c[d]), d)), i1 = [0, 1, 2].map((d) => at(Math.max(a[d], b[d], c[d]), d));
    for (let i = i0[0]; i <= i1[0]; i++) for (let j = i0[1]; j <= i1[1]; j++) for (let k = i0[2]; k <= i1[2]; k++) {
      const kk = key(i, j, k), l = grid.get(kk);
      if (l) l.push(f); else grid.set(kk, [f]);
    }
  });
  const reach = Math.max(...n);
  return (p) => {
    const c0 = [0, 1, 2].map((d) => at(p[d], d));
    let best = null, bd = Infinity;
    const seen = new Set();
    for (let r = 0; r <= reach; r++) {
      for (let i = c0[0] - r; i <= c0[0] + r; i++) for (let j = c0[1] - r; j <= c0[1] + r; j++) for (let k = c0[2] - r; k <= c0[2] + r; k++) {
        if (Math.max(Math.abs(i - c0[0]), Math.abs(j - c0[1]), Math.abs(k - c0[2])) !== r) continue;
        if (i < 0 || j < 0 || k < 0 || i >= n[0] || j >= n[1] || k >= n[2]) continue;
        for (const f of grid.get(key(i, j, k)) ?? []) {
          if (seen.has(f)) continue;
          seen.add(f);
          const t = T[f], q = closestOnTriangle(p, V[t[0]], V[t[1]], V[t[2]]);
          const d = (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2 + (q[2] - p[2]) ** 2;
          if (d < bd) { bd = d; best = q; }
        }
      }
      // a point outside the grid's box is clamped into it, so the searched block reaches r cells past its cell
      if (best && Math.sqrt(bd) <= r * cell) break;
    }
    return best ?? [...p];
  };
}

/** The mesh remeshed to edge length L: { vertices, faces }. vertices: [[x, y, z]...]; triangles: [[a, b, c]...]. */
export function isotropicRemesh(vertices, triangles, L, { iterations = 5 } = {}) {
  const V = vertices.map((p) => [...p]), F = triangles.map((t) => [...t]);
  const vf = V.map(() => new Set());
  F.forEach((t, f) => t.forEach((v) => vf[v].add(f)));
  let mean = 0;
  for (const t of F) for (let k = 0; k < 3; k++) mean += Math.hypot(...[0, 1, 2].map((c) => V[t[(k + 1) % 3]][c] - V[t[k]][c]));
  mean /= 3 * F.length;
  const project = projector(vertices, triangles, Math.max(L / 2, mean));
  const hi2 = (4 / 3 * L) ** 2, lo2 = (4 / 5 * L) ** 2;
  const d2 = (p, q) => (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2;
  const facesOf = (a, b) => { const out = []; for (const f of vf[a]) if (vf[b].has(f)) out.push(f); return out; };
  const nbrs = (v) => { const s = new Set(); for (const f of vf[v]) for (const w of F[f]) if (w !== v) s.add(w); return s; };
  const normal = (a, b, c) => {
    const u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], w = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    return [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]];
  };
  let area = 0;
  for (const t of F) area += Math.hypot(...normal(V[t[0]], V[t[1]], V[t[2]])) / 2;

  // each edge once: a closed mesh runs it a -> b in one face and b -> a in the other; take the a < b one
  const edges = () => {
    const out = [];
    for (const t of F) if (t) for (let k = 0; k < 3; k++) { const a = t[k], b = t[(k + 1) % 3]; if (a < b) out.push([a, b]); }
    return out;
  };
  const split = (a, b) => {
    const fs = facesOf(a, b);
    if (fs.length !== 2) return -1;
    const opp = fs.map((f) => F[f].find((x) => x !== a && x !== b));
    if (opp[0] === opp[1]) return -1;                          // two faces folded onto each other
    const m = V.length;
    V.push([(V[a][0] + V[b][0]) / 2, (V[a][1] + V[b][1]) / 2, (V[a][2] + V[b][2]) / 2]);
    vf.push(new Set());
    for (const f of fs) {
      const t = F[f], i = t.indexOf(a);
      const [p, q] = t[(i + 1) % 3] === b ? [a, b] : [b, a];
      const c = t.find((x) => x !== a && x !== b), g = F.length;
      F[f] = [p, m, c];
      F.push([m, q, c]);
      vf[q].delete(f); vf[q].add(g); vf[c].add(g); vf[m].add(f); vf[m].add(g);
    }
    return m;
  };
  // longest edge first (a max-heap of squared lengths): the edge split is then the longest of both its triangles,
  // Rivara's bisection, which keeps the triangles' shape bounded; splitting every long edge at once instead fans a
  // sliver's far corner into ever more slivers
  const splitLong = () => {
    const key = [], ea = [], eb = [];
    const up = (i) => { while (i > 0) { const j = (i - 1) >> 1; if (key[j] >= key[i]) break; swap(i, j); i = j; } };
    const down = (i) => {
      for (;;) {
        const l = 2 * i + 1, r = l + 1;
        let m = i;
        if (l < key.length && key[l] > key[m]) m = l;
        if (r < key.length && key[r] > key[m]) m = r;
        if (m === i) return;
        swap(i, m); i = m;
      }
    };
    const swap = (i, j) => { [key[i], key[j]] = [key[j], key[i]]; [ea[i], ea[j]] = [ea[j], ea[i]]; [eb[i], eb[j]] = [eb[j], eb[i]]; };
    const push = (a, b) => { const l = d2(V[a], V[b]); if (l > hi2) { key.push(l); ea.push(a); eb.push(b); up(key.length - 1); } };
    for (const [a, b] of edges()) push(a, b);
    // a sound mesh at edge L has about 2.3 area / L^2 triangles, which no more than that many splits reach; more means
    // an edge that cannot be split (a boundary, non-manifold or folded edge) is the longest of its triangle, and
    // bisecting the triangle's other edges would never end
    let budget = 4 * area / (L * L);
    while (key.length && budget-- > 0) {
      const a = ea[0], b = eb[0];
      const last = key.length - 1;
      swap(0, last); key.pop(); ea.pop(); eb.pop(); down(0);
      const m = split(a, b);                                   // -1 when a -> b is gone (split already)
      if (m >= 0) for (const w of nbrs(m)) push(m, w);
    }
  };
  const collapse = (a, b) => {
    const fs = facesOf(a, b);
    if (fs.length !== 2) return false;
    const nb = nbrs(b), common = [...nbrs(a)].filter((x) => nb.has(x));
    if (common.length !== 2) return false;                     // the link condition: stays a manifold
    const p = [(V[a][0] + V[b][0]) / 2, (V[a][1] + V[b][1]) / 2, (V[a][2] + V[b][2]) / 2];
    for (const v of [a, b]) for (const w of nbrs(v)) if (w !== a && w !== b && d2(p, V[w]) > hi2) return false;
    for (const v of [a, b]) for (const f of vf[v]) {             // no face may turn over
      if (fs.includes(f)) continue;
      const t = F[f], old = normal(V[t[0]], V[t[1]], V[t[2]]);
      const moved = t.map((x) => (x === a || x === b ? p : V[x])), now = normal(moved[0], moved[1], moved[2]);
      if (old[0] * now[0] + old[1] * now[1] + old[2] * now[2] <= 0) return false;
    }
    V[a] = p;
    for (const f of [...vf[b]]) {
      if (fs.includes(f)) { for (const x of F[f]) vf[x]?.delete(f); F[f] = null; continue; }
      F[f] = F[f].map((x) => (x === b ? a : x));
      vf[a].add(f);
    }
    vf[b] = null;
    return true;
  };
  const flip = (a, b) => {
    const fs = facesOf(a, b);
    if (fs.length !== 2) return;
    // T1 runs a -> b and has c; T2 runs b -> a and has d
    let [t1, t2] = fs;
    if (F[t1][(F[t1].indexOf(a) + 1) % 3] !== b) [t1, t2] = [t2, t1];
    const c = F[t1].find((x) => x !== a && x !== b), d = F[t2].find((x) => x !== a && x !== b);
    if (c === d || nbrs(c).has(d)) return;
    const val = (v) => vf[v].size, dev = (x) => (x - 6) ** 2;
    const before = dev(val(a)) + dev(val(b)) + dev(val(c)) + dev(val(d));
    const after = dev(val(a) - 1) + dev(val(b) - 1) + dev(val(c) + 1) + dev(val(d) + 1);
    if (after >= before || val(a) <= 3 || val(b) <= 3) return;
    // only where the two triangles are near coplanar: a flip across a crease would cut it
    const n1 = normal(V[a], V[b], V[c]), n2 = normal(V[b], V[a], V[d]);
    if (n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2] < 0.9 * Math.hypot(...n1) * Math.hypot(...n2)) return;
    F[t1] = [a, d, c]; F[t2] = [b, c, d];
    vf[a].delete(t2); vf[b].delete(t1); vf[c].add(t2); vf[d].add(t1);
  };
  const relax = () => {
    const N = V.map(() => [0, 0, 0]);
    F.forEach((t) => {
      if (!t) return;
      const n = normal(V[t[0]], V[t[1]], V[t[2]]);
      for (const v of t) for (let c = 0; c < 3; c++) N[v][c] += n[c];
    });
    const next = V.map((p, v) => {
      if (!vf[v] || !vf[v].size) return p;
      const nb = [...nbrs(v)], q = [0, 0, 0];
      for (const w of nb) for (let c = 0; c < 3; c++) q[c] += V[w][c] / nb.length;
      const n = N[v], l = Math.hypot(...n) || 1, u = n.map((x) => x / l);
      const d = [q[0] - p[0], q[1] - p[1], q[2] - p[2]], s = d[0] * u[0] + d[1] * u[1] + d[2] * u[2];
      return project([p[0] + d[0] - s * u[0], p[1] + d[1] - s * u[1], p[2] + d[2] - s * u[2]]);
    });
    next.forEach((p, v) => { V[v] = p; });
  };
  for (let it = 0; it < iterations; it++) {
    splitLong();
    // collapse until nothing more does: a dense input comes down to L within the first iteration
    for (let pass = 0; pass < 20; pass++) {
      let n = 0;
      for (const [a, b] of edges()) if (vf[a] && vf[b] && d2(V[a], V[b]) < lo2 && collapse(a, b)) n++;
      if (!n) break;
    }
    for (const [a, b] of edges()) if (vf[a] && vf[b]) flip(a, b);
    relax();
  }
  const id = new Int32Array(V.length).fill(-1), out = [];
  const faces = [];
  for (const t of F) if (t) faces.push(t.map((v) => { if (id[v] < 0) { id[v] = out.length; out.push(V[v]); } return id[v]; }));
  return { vertices: out, faces };
}
