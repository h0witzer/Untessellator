/**
 * meshRepair.js — mesh hygiene before painting: the defects that stop a solid, fixed where the fix is
 * unambiguous, and counted. No three.js, no OCCT.
 *
 * Painting links two triangles only across an edge with exactly two owners (meshCore.buildAdjacency), and the
 * knitter closes a shell only where every edge has a partner, so one bad triangle walls a fill off and one gap
 * opens the solid. Measured on the eval sets (test/exportRisk.mjs): every Onshape part that exported no solid at
 * all had duplicate triangles or non-manifold edges, and the duplicates had also shattered its painting into
 * single triangles. In order:
 *
 *   weld        vertices within WELD x the diagonal become one (exporters write a seam's two sides a float
 *               apart); positions are snapped to the kept vertex, so the painter's grid weld and the knitter's
 *               exact one agree afterwards
 *   collapsed   triangles with two corners on one vertex go
 *   duplicates  a triangle repeated with the same winding is kept once; a pair with OPPOSITE windings is a
 *               wall between two bodies that touch (zero thickness, inside the part) and both go, which joins
 *               the bodies into one solid (option walls: 'keep' leaves them)
 *   needles     triangles with an altitude under WELD x the diagonal go; the crack each leaves (and any other
 *               T-junction) is zipped by splitting the triangle across the long edge at the stray vertex. On a
 *               curved face the stray vertex is ON the surface and the long edge a chord, so it lies off the edge
 *               by the chord's sag: an open edge takes any vertex of its own loop within ZIP_REL x its length
 *               (sag / chord is about an eighth of the angle the chord turns: 0.1 covers chords of 45 degrees).
 *               Only open loops are zipped, so a closed mesh's thin triangles are never touched
 *   holes       a boundary loop that is still open after zipping is capped by a fan from its centroid (the cap
 *               triangles carry src -1: they are not in the input, and the painter decides what they are)
 *   winding     made consistent across every manifold edge, then outward per piece (positive volume)
 *
 * Returns the repaired soup, src (the input triangle each output triangle came from; -1 for caps), and counts.
 */

export const REPAIR = { WELD: 1e-6, ZIP_REL: 0.1, LOOP_MAX: 4000, ZIP_ROUNDS: 4 };

export function repairMesh(positions, opts = {}) {
  const cfg = { ...REPAIR, walls: 'remove', caps: true, ...opts };
  const nIn = Math.floor(positions.length / 9);
  const report = { triangles_in: nIn, welded: 0, collapsed: 0, dup_same: 0, dup_walls: 0, needles: 0, zipped: 0,
    holes_capped: 0, cap_triangles: 0, holes_left: 0, flipped: 0, nonmanifold_edges: 0 };

  // ── weld: grid hash, merging into the first vertex within tol (neighbouring cells searched) ──
  let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i + 2 < positions.length; i += 3) for (let k = 0; k < 3; k++) {
    lo[k] = Math.min(lo[k], positions[i + k]); hi[k] = Math.max(hi[k], positions[i + k]);
  }
  const diag = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) || 1;
  const tol = cfg.WELD * diag;
  const cell = tol * 2;
  const grid = new Map();
  const V = [];
  const exact = new Map();
  const vid = (x, y, z) => {
    const ek = x + ',' + y + ',' + z;
    const e = exact.get(ek);
    if (e !== undefined) return e;
    const cx = Math.floor(x / cell), cy = Math.floor(y / cell), cz = Math.floor(z / cell);
    for (let i = -1; i <= 1; i++) for (let j = -1; j <= 1; j++) for (let k = -1; k <= 1; k++) {
      const l = grid.get(`${cx + i},${cy + j},${cz + k}`);
      if (!l) continue;
      for (const v of l) {
        const p = V[v];
        if (Math.abs(p[0] - x) <= tol && Math.abs(p[1] - y) <= tol && Math.abs(p[2] - z) <= tol) {
          report.welded++; exact.set(ek, v); return v;
        }
      }
    }
    const v = V.length;
    V.push([x, y, z]);
    const gk = `${cx},${cy},${cz}`;
    (grid.get(gk) ?? grid.set(gk, []).get(gk)).push(v);
    exact.set(ek, v);
    return v;
  };
  let T = [], src = [];
  for (let t = 0; t < nIn; t++) {
    const o = t * 9;
    const tri = [vid(positions[o], positions[o + 1], positions[o + 2]), vid(positions[o + 3], positions[o + 4], positions[o + 5]),
      vid(positions[o + 6], positions[o + 7], positions[o + 8])];
    if (tri[0] === tri[1] || tri[1] === tri[2] || tri[0] === tri[2]) { report.collapsed++; continue; }
    T.push(tri); src.push(t);
  }

  // ── duplicates ──
  {
    const byKey = new Map();
    T.forEach((tri, i) => { const k = [...tri].sort((a, b) => a - b).join(','); (byKey.get(k) ?? byKey.set(k, []).get(k)).push(i); });
    const drop = new Uint8Array(T.length);
    const cyc = (tri) => { const m = tri.indexOf(Math.min(...tri)); return `${tri[m]},${tri[(m + 1) % 3]},${tri[(m + 2) % 3]}`; };
    for (const ids of byKey.values()) {
      if (ids.length < 2) continue;
      const fwd = cyc(T[ids[0]]);
      const same = ids.filter((i) => cyc(T[i]) === fwd), other = ids.filter((i) => cyc(T[i]) !== fwd);
      // pairs of opposite windings are walls; what is left over of the larger side is one triangle
      const walls = Math.min(same.length, other.length);
      if (cfg.walls === 'remove') {
        for (let w = 0; w < walls; w++) { drop[same[w]] = 1; drop[other[w]] = 1; }
        report.dup_walls += walls;
        const rest = same.length > other.length ? same.slice(walls) : other.slice(walls);
        for (let r = 1; r < rest.length; r++) { drop[rest[r]] = 1; report.dup_same++; }
      } else {
        for (const side of [same, other]) for (let r = 1; r < side.length; r++) { drop[side[r]] = 1; report.dup_same++; }
      }
    }
    T = T.filter((_, i) => !drop[i]); src = src.filter((_, i) => !drop[i]);
  }

  // ── needles ──
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const crossN = (a, b) => Math.hypot(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]);
  const len = (a) => Math.hypot(a[0], a[1], a[2]);
  {
    const keep = T.map(([a, b, c]) => {
      const e = [len(sub(V[b], V[a])), len(sub(V[c], V[b])), len(sub(V[a], V[c]))];
      const alt = crossN(sub(V[b], V[a]), sub(V[c], V[a])) / Math.max(...e, 1e-300);
      return alt > tol;
    });
    report.needles = keep.filter((k) => !k).length;
    T = T.filter((_, i) => keep[i]); src = src.filter((_, i) => keep[i]);
  }

  // ── zip T-junctions: within each zero-width loop, split the edge a stray vertex lies on ──
  const segParam = (p, a, b) => {
    const ab = sub(b, a), l2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
    if (l2 <= 0) return null;
    const ap = sub(p, a);
    const t = (ap[0] * ab[0] + ap[1] * ab[1] + ap[2] * ab[2]) / l2;
    if (t <= 0.02 || t >= 0.98) return null;
    const d = len([ap[0] - t * ab[0], ap[1] - t * ab[1], ap[2] - t * ab[2]]);
    return d <= Math.max(tol, cfg.ZIP_REL * Math.sqrt(l2)) ? { t, rel: d / Math.sqrt(l2) } : null;
  };
  for (let round = 0; round < cfg.ZIP_ROUNDS; round++) {
    const { open } = edgeTable(T, V.length);
    if (!open.length) break;
    const loops = boundaryLoops(open);
    const splits = new Map();                    // triangle index -> [{u, v, t, x}] for its open edge (u, v)
    let n = 0;
    for (const L of loops) {
      if (L.length > cfg.LOOP_MAX) continue;
      const verts = new Set();
      for (const e of L) { verts.add(e.u); verts.add(e.v); }
      // each stray vertex goes to the ONE edge it lies closest to: two cracks meeting at a corner each come
      // within ZIP_REL of the other's edge too
      const best = new Map();
      for (const e of L) for (const x of verts) {
        if (x === e.u || x === e.v) continue;
        const m = segParam(V[x], V[e.u], V[e.v]);
        if (m && !(best.get(x)?.rel <= m.rel)) best.set(x, { e, t: m.t, rel: m.rel });
      }
      for (const [x, { e, t }] of best) {
        (splits.get(e.tri) ?? splits.set(e.tri, []).get(e.tri)).push({ u: e.u, v: e.v, t, x });
        n++;
      }
    }
    if (!n) break;
    const T2 = [], src2 = [];
    T.forEach((tri, i) => {
      const sp = splits.get(i);
      if (!sp) { T2.push(tri); src2.push(src[i]); return; }
      // one edge per triangle per round (the rest wait for the next round): split it into a fan from the apex
      const { u, v } = sp[0];
      const pts = sp.filter((s) => s.u === u && s.v === v).sort((a, b) => a.t - b.t);
      const k = [0, 1, 2].find((j) => (tri[j] === u && tri[(j + 1) % 3] === v) || (tri[j] === v && tri[(j + 1) % 3] === u));
      const a = tri[k], b = tri[(k + 1) % 3], apex = tri[(k + 2) % 3];
      const chain = [a, ...(a === u ? pts : [...pts].reverse()).map((s) => s.x), b];
      for (let j = 0; j + 1 < chain.length; j++) { T2.push([chain[j], chain[j + 1], apex]); src2.push(src[i]); }
      report.zipped += pts.length;
    });
    T = T2; src = src2;
  }

  // ── cap the holes left ──
  {
    const { open } = edgeTable(T, V.length);
    for (const L of boundaryLoops(open)) {
      const cycles = cfg.caps ? simpleCycles(L) : null;
      if (!cycles) { report.holes_left++; continue; }
      for (const cycle of cycles) {
        const c = [0, 0, 0];
        for (const e of cycle) for (let k = 0; k < 3; k++) c[k] += V[e.u][k] / cycle.length;
        const ci = V.length;
        V.push(c);
        // each boundary edge is traversed u -> v by its triangle; the cap runs it v -> u
        for (const e of cycle) { T.push([e.v, e.u, ci]); src.push(-1); report.cap_triangles++; }
        report.holes_capped++;
      }
    }
  }

  // ── winding ──
  report.flipped = orient(T, V);
  const { multi } = edgeTable(T, V.length);
  report.nonmanifold_edges = multi;

  const out = new Float64Array(T.length * 9);
  T.forEach((tri, i) => { for (let k = 0; k < 3; k++) for (let j = 0; j < 3; j++) out[i * 9 + k * 3 + j] = V[tri[k]][j]; });
  report.triangles_out = T.length;
  return { positions: out, src: Int32Array.from(src), report };
}

/** Undirected edges: the open ones (one triangle; with that triangle and its direction) and the count of 3+. */
function edgeTable(T, nV) {
  const edges = new Map();
  T.forEach((tri, i) => {
    for (let k = 0; k < 3; k++) {
      const u = tri[k], v = tri[(k + 1) % 3];
      const key = Math.min(u, v) * nV + Math.max(u, v);
      const e = edges.get(key);
      if (e) { e.n++; } else edges.set(key, { u, v, tri: i, n: 1 });
    }
  });
  const open = [];
  let multi = 0;
  for (const e of edges.values()) { if (e.n === 1) open.push(e); else if (e.n > 2) multi++; }
  return { open, multi };
}

/** Open edges joined at shared vertices. */
function boundaryLoops(open) {
  const parent = new Map();
  const find = (x) => { while (parent.get(x) !== x) { parent.set(x, parent.get(parent.get(x))); x = parent.get(x); } return x; };
  for (const e of open) for (const x of [e.u, e.v]) if (!parent.has(x)) parent.set(x, x);
  for (const e of open) { const a = find(e.u), b = find(e.v); if (a !== b) parent.set(a, b); }
  const loops = new Map();
  for (const e of open) { const r = find(e.u); (loops.get(r) ?? loops.set(r, []).get(r)).push(e); }
  return [...loops.values()];
}

/**
 * A boundary loop's edges as directed cycles (u -> v -> ...). A loop through one vertex twice (two holes that
 * touch at a corner: a figure 8) is split there into simple cycles. Null when the edges do not form cycles
 * (an edge's direction disagrees with its neighbours': a flipped triangle on the rim).
 */
function simpleCycles(L) {
  const out = new Map();                         // vertex -> its unused outgoing edges
  for (const e of L) (out.get(e.u) ?? out.set(e.u, []).get(e.u)).push(e);
  const cycles = [];
  let left = L.length;
  while (left > 0) {
    const start = [...out.values()].find((l) => l.length)[0];
    // walk until a vertex repeats; the part from its first visit is a simple cycle
    const path = [], at = new Map();
    let e = start;
    for (;;) {
      const l = out.get(e.u);
      l.splice(l.indexOf(e), 1);
      left--;
      at.set(e.u, path.length);
      path.push(e);
      if (at.has(e.v)) {
        const cyc = path.splice(at.get(e.v));
        cycles.push(cyc);
        // what is left of the path goes back to be walked again
        for (const x of path) { out.get(x.u).push(x); left++; }
        break;
      }
      const nxt = out.get(e.v);
      if (!nxt || !nxt.length) return null;
      e = nxt[0];
    }
  }
  return cycles;
}

/** Consistent winding over manifold edges, then outward per piece. Returns how many triangles were reversed. */
function orient(T, V) {
  const nV = V.length;
  const byEdge = new Map();
  T.forEach((tri, i) => {
    for (let k = 0; k < 3; k++) {
      const u = tri[k], v = tri[(k + 1) % 3];
      const key = Math.min(u, v) * nV + Math.max(u, v);
      (byEdge.get(key) ?? byEdge.set(key, []).get(key)).push(i);
    }
  });
  const seen = new Uint8Array(T.length);
  const flip = new Uint8Array(T.length);
  let flipped = 0;
  const dir = (tri, u, v) => { for (let k = 0; k < 3; k++) if (tri[k] === u && tri[(k + 1) % 3] === v) return 1; return -1; };
  for (let s = 0; s < T.length; s++) {
    if (seen[s]) continue;
    seen[s] = 1;
    const comp = [s], queue = [s];
    for (let qi = 0; qi < queue.length; qi++) {
      const t = queue[qi];
      const tri = T[t];
      for (let k = 0; k < 3; k++) {
        const u = tri[k], v = tri[(k + 1) % 3];
        const owners = byEdge.get(Math.min(u, v) * nV + Math.max(u, v));
        if (owners.length !== 2) continue;
        const n = owners[0] === t ? owners[1] : owners[0];
        if (seen[n]) continue;
        // a consistent neighbour runs the shared edge the other way
        if (dir(T[n], u, v) === 1) { T[n].reverse(); flip[n] ^= 1; }
        seen[n] = 1; comp.push(n); queue.push(n);
      }
    }
    let vol = 0;
    for (const t of comp) {
      const [a, b, c] = T[t].map((i) => V[i]);
      vol += a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
    }
    if (vol < 0) for (const t of comp) { T[t].reverse(); flip[t] ^= 1; }
  }
  for (let t = 0; t < T.length; t++) flipped += flip[t];
  return flipped;
}
