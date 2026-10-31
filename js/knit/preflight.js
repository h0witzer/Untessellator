/**
 * knit/preflight.js — what the export will run into, measured before it runs. No OCCT.
 *
 * The knitter (knit/index.js) takes minutes on a large part and only then says the solid came out faceted, or
 * not at all. Most of what decides that is visible in the painted mesh beforehand:
 *
 *   mesh     open edges, grouped into boundary loops. A loop of zero width is a CRACK (a T-junction or an
 *            unwelded seam: the two sides lie on each other, and zipping them closes it); a loop with width is a
 *            HOLE, which needs a cap. Edges of 3+ triangles, triangles of zero area (collinear corners) and
 *            collapsed ones (two corners on one vertex), duplicate triangles, and edges two triangles traverse
 *            the same way (a flipped triangle).
 *   groups   per painted group with a surface:
 *              dev     the farthest of its vertices from its own surface, x the part diagonal. On a CAD
 *                      tessellation every vertex lies on its face, so a correct group is at float precision
 *                      (~1e-7) and anything well above that holds a vertex of another surface.
 *              seam    the knitter projects each run of boundary between two groups onto BOTH surfaces. Where a
 *                      boundary vertex is far from either, that edge is built off the mesh (it kinks, folds or
 *                      will not close): the largest such gap on the group's boundaries, x the diagonal.
 *              arc     for cylinders, cones and tori, the degrees of turn about the axis the group covers. A
 *                      narrow strip fits a cone or a torus about as well as a cylinder.
 *              shard   a group of a few triangles whose every boundary edge is smooth: a piece of a curved
 *                      face painted on its own (one triangle fits a plane exactly, so dev and seam are 0).
 *            and the groups with no surface (they go out as B-splines or facets).
 *
 * Nothing here is a verdict: thresholds are in PREFLIGHT and are measured by test/exportRisk.mjs against the
 * true faces and the knitter's outcome.
 */

import { distNormal, dot, sub, cross, unit, norm } from './geometry.js';

export const PREFLIGHT = {
  ZERO_ALTITUDE: 1e-9,    // x diagonal: a triangle this thin is a line
  CRACK_WIDTH: 1e-6,      // x diagonal: a boundary loop no wider than this is a crack, not a hole
  LOOP_MAX: 4000,         // boundary loops longer than this are not measured for width (counted as holes)
  DEV: 1e-5,              // x diagonal: a group's vertex this far off its own surface
  SEAM: 1e-4,             // x diagonal: a boundary vertex this far off one of its two surfaces
  ARC: 45,                // degrees: an axial group narrower than this
  SHARD_TRIS: 3,          // a group this small...
  SMOOTH_DEG: 20,         // ...whose every boundary turns less than this is a shard of a larger surface
};

const AXIAL = new Set(['cylinder', 'cone', 'torus']);

/**
 * @param {ArrayLike<number>} positions  triangle soup, 9 per triangle
 * @param {ArrayLike<number>} labels     group per triangle (-1 unpainted); optional
 * @param {Map<number, [string, object]>} surfaces  group -> [type, params]; optional
 */
export function preflight(positions, labels = null, surfaces = new Map(), opts = {}) {
  const cfg = { ...PREFLIGHT, ...opts };
  const nT = Math.floor(positions.length / 9);
  // the knitter's weld: exact coordinates
  const keys = new Map(), V = [];
  const idx = new Int32Array(nT * 3);
  for (let t = 0; t < nT; t++) for (let k = 0; k < 3; k++) {
    const o = t * 9 + k * 3;
    const key = positions[o] + ',' + positions[o + 1] + ',' + positions[o + 2];
    let i = keys.get(key);
    if (i === undefined) { i = V.length; keys.set(key, i); V.push([positions[o], positions[o + 1], positions[o + 2]]); }
    idx[t * 3 + k] = i;
  }
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const p of V) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], p[k]); hi[k] = Math.max(hi[k], p[k]); }
  const diag = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) || 1;

  // ── mesh health ──
  const mesh = { triangles: nT, vertices: V.length, collapsed: 0, zeroArea: 0, duplicates: 0, openEdges: 0,
    nonmanifoldEdges: 0, flippedEdges: 0, loops: 0, cracks: 0, holes: 0, holeEdges: 0, crackEdges: 0 };
  const live = new Uint8Array(nT);
  const seenTri = new Set();
  for (let t = 0; t < nT; t++) {
    const a = idx[t * 3], b = idx[t * 3 + 1], c = idx[t * 3 + 2];
    if (a === b || b === c || a === c) { mesh.collapsed++; continue; }
    const tk = [a, b, c].sort((x, y) => x - y).join(',');
    if (seenTri.has(tk)) mesh.duplicates++;
    seenTri.add(tk);
    const e1 = sub(V[b], V[a]), e2 = sub(V[c], V[a]), e3 = sub(V[c], V[b]);
    const longest = Math.max(norm(e1), norm(e2), norm(e3));
    if (norm(cross(e1, e2)) / Math.max(longest, 1e-300) <= cfg.ZERO_ALTITUDE * diag) mesh.zeroArea++;
    live[t] = 1;
  }
  const nV = V.length;
  const edges = new Map();                       // undirected key -> {u, v, fwd, back, tris}
  for (let t = 0; t < nT; t++) {
    if (!live[t]) continue;
    for (let k = 0; k < 3; k++) {
      const u = idx[t * 3 + k], v = idx[t * 3 + (k + 1) % 3];
      const key = Math.min(u, v) * nV + Math.max(u, v);
      let e = edges.get(key);
      if (!e) edges.set(key, (e = { u: Math.min(u, v), v: Math.max(u, v), fwd: 0, back: 0, tris: [] }));
      if (u < v) e.fwd++; else e.back++;
      e.tris.push(t);
    }
  }
  const open = [];
  for (const e of edges.values()) {
    const m = e.tris.length;
    if (m === 1) open.push(e);
    else if (m > 2) mesh.nonmanifoldEdges++;
    else if (e.fwd === 2 || e.back === 2) mesh.flippedEdges++;
  }
  mesh.openEdges = open.length;
  // boundary loops: open edges joined at shared vertices
  const parent = new Map();
  const find = (x) => { while (parent.get(x) !== x) { parent.set(x, parent.get(parent.get(x))); x = parent.get(x); } return x; };
  for (const e of open) for (const x of [e.u, e.v]) if (!parent.has(x)) parent.set(x, x);
  for (const e of open) { const a = find(e.u), b = find(e.v); if (a !== b) parent.set(a, b); }
  const loops = new Map();
  for (const e of open) { const r = find(e.u); (loops.get(r) ?? loops.set(r, []).get(r)).push(e); }
  mesh.loops = loops.size;
  const crackTol = cfg.CRACK_WIDTH * diag;
  const segDist = (p, a, b) => {
    const ab = sub(b, a), t = Math.max(0, Math.min(1, dot(sub(p, a), ab) / Math.max(dot(ab, ab), 1e-300)));
    return norm(sub(p, [a[0] + t * ab[0], a[1] + t * ab[1], a[2] + t * ab[2]]));
  };
  // open-boundary vertices on a grid, to find a vertex's twin across an unwelded seam
  const cellOf = (p) => `${Math.floor(p[0] / (2 * crackTol))},${Math.floor(p[1] / (2 * crackTol))},${Math.floor(p[2] / (2 * crackTol))}`;
  const bgrid = new Map();
  const incident = new Map();                    // open-boundary vertex -> its open edges
  for (const e of open) for (const x of [e.u, e.v]) {
    if (!incident.has(x)) { incident.set(x, []); const c = cellOf(V[x]); (bgrid.get(c) ?? bgrid.set(c, []).get(c)).push(x); }
    incident.get(x).push(e);
  }
  const hasTwin = (x) => {
    const p = V[x];
    const [cx, cy, cz] = cellOf(p).split(',').map(Number);
    for (let i = -1; i <= 1; i++) for (let j = -1; j <= 1; j++) for (let k = -1; k <= 1; k++) {
      for (const y of bgrid.get(`${cx + i},${cy + j},${cz + k}`) ?? []) if (y !== x && norm(sub(V[y], p)) <= crackTol) return true;
    }
    return false;
  };
  // a crack's end: its two open edges leave in the same direction (the two sides of a T-junction)
  const isFold = (x) => {
    const es = incident.get(x);
    if (es.length !== 2) return false;
    const d = es.map((e) => unit(sub(V[e.u === x ? e.v : e.u], V[x])));
    return dot(d[0], d[1]) >= 1 - 1e-6;
  };
  for (const L of loops.values()) {
    // ZERO WIDTH: every vertex is a crack's end, lies on another edge of the loop, or has a twin within tol
    let crack = L.length <= cfg.LOOP_MAX && L.length >= 2;
    if (crack) {
      const verts = new Set();
      for (const e of L) { verts.add(e.u); verts.add(e.v); }
      outer: for (const x of verts) {
        if (isFold(x) || hasTwin(x)) continue;
        for (const e of L) if (e.u !== x && e.v !== x && segDist(V[x], V[e.u], V[e.v]) <= crackTol) continue outer;
        crack = false; break;
      }
    }
    if (crack) { mesh.cracks++; mesh.crackEdges += L.length; } else { mesh.holes++; mesh.holeEdges += L.length; }
  }
  mesh.closed = mesh.openEdges === 0 && mesh.nonmanifoldEdges === 0;

  // ── groups ──
  const groups = new Map();
  if (labels) {
    const byG = new Map();
    for (let t = 0; t < nT; t++) if (live[t]) (byG.get(labels[t]) ?? byG.set(labels[t], []).get(labels[t])).push(t);
    const dist = new Map();                      // `${g}|${v}` -> |distance| of vertex v from g's surface
    const dOf = (g, v) => {
      const s = surfaces.get(g);
      if (!s || !s[1]) return NaN;
      const k = g * nV + v;
      let d = dist.get(k);
      if (d === undefined) { try { d = Math.abs(distNormal(s[0], s[1], V[v])[0]); } catch { d = Infinity; } dist.set(k, d); }
      return d;
    };
    // a group's boundary vertices: on an edge it shares with another label, or an open edge
    const onBoundary = new Set();                // g * nV + v
    for (const e of edges.values()) {
      const ls = new Set(e.tris.map((t) => labels[t]));
      if (ls.size < 2 && e.tris.length > 1) continue;
      for (const g of ls) { onBoundary.add(g * nV + e.u); onBoundary.add(g * nV + e.v); }
    }
    for (const [g, tris] of byG) {
      if (g < 0) continue;
      const s = surfaces.get(g);
      const row = { tris: tris.length, type: s ? s[0] : null, dev: null, devMedian: null, seam: 0, seamVerts: 0,
        seamOff: 0, arc: null, bEdges: 0, smoothEdges: 0, neighbours: new Set() };
      if (s && s[1]) {
        const vs = new Set();
        for (const t of tris) for (let k = 0; k < 3; k++) vs.add(idx[t * 3 + k]);
        // dev over the INTERIOR vertices (a surface swallowed inside the group); the boundary ones are the seam
        const ds = [...vs].filter((v) => !onBoundary.has(g * nV + v)).map((v) => dOf(g, v)).sort((a, b) => a - b);
        row.dev = ds.length ? ds[ds.length - 1] / diag : 0;
        row.devMedian = ds.length ? ds[ds.length >> 1] / diag : 0;
        if (AXIAL.has(s[0])) row.arc = arcOf(s[0], s[1], [...vs].map((v) => V[v]));
      }
      groups.set(g, row);
    }
    // seams: every edge between two different groups, both ends on both surfaces
    const seamSeen = new Set();
    const normalOf = (t) => unit(cross(sub(V[idx[t * 3 + 1]], V[idx[t * 3]]), sub(V[idx[t * 3 + 2]], V[idx[t * 3]])));
    const cosSmooth = Math.cos((cfg.SMOOTH_DEG * Math.PI) / 180);
    for (const e of edges.values()) {
      if (e.tris.length !== 2) continue;
      const a = labels[e.tris[0]], b = labels[e.tris[1]];
      if (a === b || a < 0 || b < 0) continue;
      groups.get(a)?.neighbours.add(b);
      groups.get(b)?.neighbours.add(a);
      // winding may be inconsistent across a flipped triangle: the turn is the smaller of the two readings
      const smooth = Math.abs(dot(normalOf(e.tris[0]), normalOf(e.tris[1]))) >= cosSmooth;
      for (const g of [a, b]) { const r = groups.get(g); if (r) { r.bEdges++; if (smooth) r.smoothEdges++; } }
      for (const x of [e.u, e.v]) {
        const key = `${Math.min(a, b)}|${Math.max(a, b)}|${x}`;
        if (seamSeen.has(key)) continue;
        seamSeen.add(key);
        const da = dOf(a, x), db = dOf(b, x);
        if (Number.isNaN(da) || Number.isNaN(db)) continue;   // a side with no surface: a B-spline or facets
        for (const [g, d] of [[a, da], [b, db]]) {
          const r = groups.get(g);
          r.seamVerts++;
          // the gap is charged to the side the vertex is off: that surface does not reach the edge
          if (d / diag > cfg.SEAM) r.seamOff++;
          r.seam = Math.max(r.seam, d / diag);
        }
      }
    }
    for (const r of groups.values()) {
      r.neighbours = r.neighbours.size;
      r.flags = [];
      if (!r.type) r.flags.push(r.tris > 4 ? 'no surface (B-spline)' : 'no surface (facets)');
      if (r.dev !== null && r.dev > cfg.DEV) r.flags.push('off its surface');
      if (r.seamOff > 0) r.flags.push('seam gap');
      if (r.arc !== null && r.arc < cfg.ARC) r.flags.push('narrow arc');
      if (r.tris <= cfg.SHARD_TRIS && r.bEdges > 0 && r.smoothEdges === r.bEdges) r.flags.push('shard');
    }
  }
  const part = summarize(mesh, groups);
  return { diag, mesh, groups, part };
}

/** Degrees of turn about the axis that the points cover: 360 less the largest empty gap. */
function arcOf(type, p, pts) {
  const a = unit(p.axis);
  const o = type === 'cylinder' ? p.axisPoint : type === 'cone' ? p.apex : p.center;
  const ref = Math.abs(a[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  const e1 = unit(cross(a, ref)), e2 = cross(a, e1);
  const ang = [];
  for (const x of pts) {
    const v = sub(x, o);
    const r = sub(v, [a[0] * dot(v, a), a[1] * dot(v, a), a[2] * dot(v, a)]);
    if (norm(r) < 1e-300) continue;
    ang.push(Math.atan2(dot(r, e2), dot(r, e1)));
  }
  if (ang.length < 2) return 0;
  ang.sort((x, y) => x - y);
  let gap = ang[0] + 2 * Math.PI - ang[ang.length - 1];
  for (let i = 1; i < ang.length; i++) gap = Math.max(gap, ang[i] - ang[i - 1]);
  return 360 - (gap * 180) / Math.PI;
}

function summarize(mesh, groups) {
  const flagged = [...groups].filter(([, r]) => r.flags.length);
  const count = (f) => flagged.filter(([, r]) => r.flags.includes(f)).length;
  const worst = (k) => Math.max(0, ...[...groups.values()].map((r) => r[k] ?? 0));
  return {
    closed: mesh.closed,
    groups: groups.size,
    flagged: flagged.length,
    offSurface: count('off its surface'),
    seamGap: count('seam gap'),
    narrow: count('narrow arc'),
    shards: count('shard'),
    noSurface: flagged.filter(([, r]) => !r.type).length,
    worstDev: worst('dev'),
    worstSeam: worst('seam'),
  };
}
