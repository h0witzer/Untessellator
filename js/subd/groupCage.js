/**
 * groupCage.js — a quad cage for a mesh painted in groups (a part's faces), laid out from the groups' borders.
 *
 * Borders are the mesh edges between two groups; corners are the vertices where three or more groups meet, and a
 * border runs corner to corner as a chain. Each group must be a disk with k >= 3 corners: it is split from its
 * centre into k quad grids, one per corner, each n x n (the midpoint split of a k-sided face), so every chain gets
 * 2n cage edges and the centre is a star point of valence k (none when k = 4). Chain points are spaced by arc
 * length and shared by the two groups beside the chain; spokes run from each chain's midpoint to the centre, and
 * each grid's inside is the Coons patch of its four sides; points off the border are put on the group's triangles
 * (closest point). The fit (fitCage.js) then moves every point.
 *
 * Returns { vertices: [[x, y, z]...], groups: [{ label, faces: [[a, b, c, d]...], triangles: [t...] }], n } or
 * { error }. Quads are wound like the mesh's triangles (outward when they are).
 */
import { projector } from './isotropic.js';

export function groupCage(vertices, triangles, labels, { quads = 600 } = {}) {
  const nv = vertices.length;
  const half = new Map();                            // "a,b" -> triangle owning the directed edge a -> b
  triangles.forEach((t, f) => { for (let k = 0; k < 3; k++) half.set(`${t[k]},${t[(k + 1) % 3]}`, f); });
  const labelsAt = Array.from({ length: nv }, () => new Set());
  triangles.forEach((t, f) => t.forEach((v) => labelsAt[v].add(labels[f])));
  const isCorner = (v) => labelsAt[v].size >= 3;
  // each group's triangles, and its border half-edges (group on the left) as next-pointers
  const byLabel = new Map();
  triangles.forEach((t, f) => { if (!byLabel.has(labels[f])) byLabel.set(labels[f], []); byLabel.get(labels[f]).push(f); });
  const groups = [];
  for (const [label, tris] of byLabel) {
    const next = new Map();
    for (const f of tris) {
      const t = triangles[f];
      for (let k = 0; k < 3; k++) {
        const a = t[k], b = t[(k + 1) % 3], g = half.get(`${b},${a}`);
        if (g === undefined) return { error: `the mesh is open at edge ${a}-${b}` };
        if (labels[g] === label) continue;
        if (next.has(a)) return { error: `group ${label}: its border touches itself at vertex ${a}` };
        next.set(a, b);
      }
    }
    if (!next.size) return { error: `group ${label} has no border (a whole closed body)` };
    // one border loop: a disk
    const start = next.keys().next().value, loop = [start];
    for (let v = next.get(start); v !== start; v = next.get(v)) loop.push(v);
    if (loop.length !== next.size) return { error: `group ${label} has ${next.size === loop.length ? 1 : 'several'} border loops (a hole or a handle): not a disk` };
    const ci = loop.map((v, i) => (isCorner(v) ? i : -1)).filter((i) => i >= 0);
    if (ci.length < 3) return { error: `group ${label} has ${ci.length} corners (3 or more are needed for now)` };
    const rot = loop.slice(ci[0]).concat(loop.slice(0, ci[0]));
    const cornersAt = ci.map((i) => i - ci[0]);
    const chains = cornersAt.map((s, j) => rot.slice(s, (cornersAt[j + 1] ?? rot.length) + 1).concat(j + 1 === cornersAt.length ? [rot[0]] : []));
    groups.push({ label, tris, chains, k: chains.length });
  }
  const totalK = groups.reduce((s, g) => s + g.k, 0);
  const n = Math.max(2, Math.round(Math.sqrt(quads / totalK)));

  const X = [];
  const point = (p) => { X.push(p); return X.length - 1; };
  const cornerId = new Map();
  const cornerOf = (v) => { if (!cornerId.has(v)) cornerId.set(v, point([...vertices[v]])); return cornerId.get(v); };
  // chains: 2n segments by arc length, shared by the two groups (keyed by their two end corners and next vertex)
  const chainIds = new Map();
  const sampleChain = (chain) => {
    const len = [0];
    for (let i = 1; i < chain.length; i++) len.push(len[i - 1] + Math.hypot(...[0, 1, 2].map((c) => vertices[chain[i]][c] - vertices[chain[i - 1]][c])));
    const L = len[len.length - 1], ids = [cornerOf(chain[0])];
    let seg = 1;
    for (let j = 1; j < 2 * n; j++) {
      const s = L * j / (2 * n);
      while (seg < chain.length - 1 && len[seg] < s) seg++;
      const a = vertices[chain[seg - 1]], b = vertices[chain[seg]], w = (s - len[seg - 1]) / ((len[seg] - len[seg - 1]) || 1);
      ids.push(point([0, 1, 2].map((c) => a[c] + w * (b[c] - a[c]))));
    }
    ids.push(cornerOf(chain[chain.length - 1]));
    return ids;
  };
  const chainOf = (chain) => {
    const fwd = chain.join(','), back = [...chain].reverse().join(',');
    if (chainIds.has(fwd)) return chainIds.get(fwd);
    if (chainIds.has(back)) return [...chainIds.get(back)].reverse();
    const ids = sampleChain(chain);
    chainIds.set(fwd, ids);
    return ids;
  };

  const out = [];
  for (const g of groups) {
    const T = g.tris.map((f) => triangles[f]);
    let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity], area = 0;
    const centroid = [0, 0, 0];
    for (const t of T) {
      const [a, b, c] = t.map((v) => vertices[v]);
      for (const p of [a, b, c]) for (let d = 0; d < 3; d++) { lo[d] = Math.min(lo[d], p[d]); hi[d] = Math.max(hi[d], p[d]); }
      const u = [0, 1, 2].map((d) => b[d] - a[d]), w = [0, 1, 2].map((d) => c[d] - a[d]);
      const ar = Math.hypot(u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]) / 2;
      area += ar;
      for (let d = 0; d < 3; d++) centroid[d] += ar * (a[d] + b[d] + c[d]) / 3;
    }
    const onGroup = projector(vertices, T, Math.max(Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) / 16, 1e-9));
    const k = g.k, C = g.chains.map(chainOf);         // C[i]: chain i in this group's direction, corner i -> corner i+1
    const centre = point(onGroup(centroid.map((x) => x / area)));
    const spokes = C.map((ids) => {
      const a = X[ids[n]], b = X[centre], s = [ids[n]];
      for (let j = 1; j < n; j++) s.push(point(onGroup([0, 1, 2].map((d) => a[d] + (b[d] - a[d]) * j / n))));
      s.push(centre);
      return s;
    });
    const inner = Array.from({ length: k }, () => new Int32Array((n - 1) * (n - 1)));
    const id = (i, s, t) => {
      if (t === 0) return C[i][s];
      if (s === 0) return C[(i + k - 1) % k][2 * n - t];
      if (s === n) return spokes[i][t];
      if (t === n) return spokes[(i + k - 1) % k][s];
      return inner[i][(s - 1) * (n - 1) + t - 1];
    };
    for (let i = 0; i < k; i++) {
      const P = (s, t) => X[id(i, s, t)];
      for (let s = 1; s < n; s++) for (let t = 1; t < n; t++) {
        const u = s / n, v = t / n, p = [0, 0, 0];
        for (let d = 0; d < 3; d++) {
          p[d] = (1 - v) * P(s, 0)[d] + v * P(s, n)[d] + (1 - u) * P(0, t)[d] + u * P(n, t)[d]
            - ((1 - u) * (1 - v) * P(0, 0)[d] + u * (1 - v) * P(n, 0)[d] + (1 - u) * v * P(0, n)[d] + u * v * P(n, n)[d]);
        }
        inner[i][(s - 1) * (n - 1) + t - 1] = point(onGroup(p));
      }
    }
    const faces = [];
    for (let i = 0; i < k; i++) for (let s = 0; s < n; s++) for (let t = 0; t < n; t++) faces.push([id(i, s, t), id(i, s + 1, t), id(i, s + 1, t + 1), id(i, s, t + 1)]);
    out.push({ label: g.label, faces, triangles: g.tris, k });
  }
  return { vertices: X, groups: out, n };
}
