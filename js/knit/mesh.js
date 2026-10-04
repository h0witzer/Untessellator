/**
 * knit/mesh.js — the labelled mesh the knitter builds from, and its B-rep combinatorics. No OCCT.
 *
 * A port of ml/solid_rebuild.py's Mesh and mesh_topology. Topology comes from the mesh, geometry from
 * the fitted surfaces: every run of boundary between two patches becomes ONE edge both faces share,
 * so the shell closes by construction unless the mesh is open. That is what the old sewing builder
 * (js/brepBuilder.js) lacked: it built every face on its own and sewed them within 0.5% of the
 * diagonal, and closed 0 of 31 fully analytic parts.
 */

import { cross, sub, unit, dot } from './geometry.js';

export const BODY_STRIDE = 2 ** 20;           // corner key = vertex * BODY_STRIDE + body
const PK = 2 ** 26;                           // partner key = triangle * PK + edge id

export class KnitMesh {
  /**
   * @param {ArrayLike<number>} positions  flat triangle soup, 9 per triangle
   * @param {ArrayLike<number>} labels     patch label per soup triangle (-1: unpainted)
   */
  constructor(positions, labels) {
    const nSoup = Math.floor(positions.length / 9);
    const keys = new Map();
    const V = [];
    const idx = new Int32Array(nSoup * 3);
    for (let t = 0; t < nSoup; t++) {
      for (let k = 0; k < 3; k++) {
        const o = t * 9 + k * 3;
        const key = positions[o] + ',' + positions[o + 1] + ',' + positions[o + 2];
        let i = keys.get(key);
        if (i === undefined) {
          i = V.length;
          keys.set(key, i);
          V.push([positions[o], positions[o + 1], positions[o + 2]]);
        }
        idx[t * 3 + k] = i;
      }
    }
    this.V = V;
    const T0 = [], F = [], src = [];
    for (let t = 0; t < nSoup; t++) {
      const a = idx[t * 3], b = idx[t * 3 + 1], c = idx[t * 3 + 2];
      if (a === b || b === c || a === c) continue;
      T0.push([a, b, c]);
      F.push(labels[t]);
      src.push(t);
    }
    this.F = Int32Array.from(F);
    this.src = Int32Array.from(src);           // each kept triangle's index in the input soup
    this.nV = V.length;
    this.edgeId = new Map();
    this.partner = this._pair(T0);
    this.T = this._orient(T0);
    this.partner = this._pair(this.T);         // again, with the winding now consistent
    this.body = this._bodies();
    this.N = this.T.map(([a, b, c]) => cross(sub(V[b], V[a]), sub(V[c], V[a])));
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (const p of V) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], p[k]); hi[k] = Math.max(hi[k], p[k]); }
    this.diag = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
    this.area = this.N.map((n) => 0.5 * Math.hypot(n[0], n[1], n[2]));
  }

  /** Dense id of the undirected edge (u, v). */
  eid(u, v) {
    const key = Math.min(u, v) * this.nV + Math.max(u, v);
    let id = this.edgeId.get(key);
    if (id === undefined) { id = this.edgeId.size; this.edgeId.set(key, id); }
    return id;
  }

  /** The triangle across undirected edge (u, v) from triangle t, in the same solid; -1 if none. */
  across(t, u, v) {
    const n = this.partner.get(t * PK + this.eid(u, v));
    return n === undefined ? -1 : n;
  }

  _pair(T) {
    // (triangle, undirected edge) -> the triangle across that edge IN THE SAME SOLID. An edge used by
    // two triangles pairs them. One used by more (two solids meshed where they touch) is resolved
    // radially: around the edge, a triangle traversing it backwards has material on its
    // counter-clockwise side and one traversing it forwards on its clockwise side, so going
    // counter-clockwise each backward triangle pairs with the next forward one. Coincident triangles
    // tie on angle and the forward one sorts first, so a zero-thickness gap is never paired.
    const V = this.V;
    const edgeTris = new Map();
    for (let t = 0; t < T.length; t++) {
      for (let k = 0; k < 3; k++) {
        const u = T[t][k], v = T[t][(k + 1) % 3];
        const id = this.eid(u, v);
        let l = edgeTris.get(id);
        if (!l) edgeTris.set(id, (l = { u: Math.min(u, v), v: Math.max(u, v), tris: [] }));
        l.tris.push([t, u < v ? 1 : -1]);
      }
    }
    const partner = new Map();
    const key = (t, id) => t * PK + id;
    for (const [id, { u: ea, v: eb, tris }] of edgeTris) {
      if (tris.length === 2) {
        partner.set(key(tris[0][0], id), tris[1][0]);
        partner.set(key(tris[1][0], id), tris[0][0]);
        continue;
      }
      if (tris.length < 2) continue;
      const e = unit(sub(V[eb], V[ea]));
      const ref = Math.abs(e[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
      const b1 = unit(cross(e, ref));
      const b2 = cross(e, b1);
      const items = tris.map(([t, d]) => {
        const w = T[t].find((x) => x !== ea && x !== eb);
        const r = sub(V[w], V[ea]);
        return [Math.atan2(dot(r, b2), dot(r, b1)), d === 1 ? 0 : 1, t, d];
      });
      items.sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]);
      const n = items.length;
      for (let i = 0; i < n; i++) {
        if (items[i][3] !== -1) continue;
        const nxt = items[(i + 1) % n];
        if (nxt[3] === 1) {
          partner.set(key(items[i][2], id), nxt[2]);
          partner.set(key(nxt[2], id), items[i][2]);
        }
      }
    }
    return partner;
  }

  _orient(T0) {
    // Consistent winding across paired triangles, then outward (positive signed volume) per piece.
    const T = T0.map((t) => t.slice());
    const seen = new Uint8Array(T.length);
    const V = this.V;
    for (let s0 = 0; s0 < T.length; s0++) {
      if (seen[s0]) continue;
      const comp = [s0];
      seen[s0] = 1;
      const queue = [s0];
      for (let qi = 0; qi < queue.length; qi++) {
        const t = queue[qi];
        const [i, j, k] = T[t];
        for (const [u, v] of [[i, j], [j, k], [k, i]]) {
          const n = this.partner.get(t * PK + this.eid(u, v));
          if (n === undefined || seen[n]) continue;
          const [a, b, c] = T[n];
          if ((u === a && v === b) || (u === b && v === c) || (u === c && v === a)) T[n].reverse();
          seen[n] = 1;
          comp.push(n);
          queue.push(n);
        }
      }
      let vol = 0;
      for (const t of comp) {
        const [a, b, c] = T[t];
        vol += dot(V[a], cross(V[b], V[c]));
      }
      if (vol < 0) for (const t of comp) T[t].reverse();
    }
    return T;
  }

  _bodies() {
    const T = this.T;
    const body = new Int32Array(T.length).fill(-1);
    let nb = 0;
    for (let s0 = 0; s0 < T.length; s0++) {
      if (body[s0] >= 0) continue;
      body[s0] = nb;
      const queue = [s0];
      for (let qi = 0; qi < queue.length; qi++) {
        const t = queue[qi];
        for (let k = 0; k < 3; k++) {
          const n = this.across(t, T[t][k], T[t][(k + 1) % 3]);
          if (n >= 0 && body[n] < 0) { body[n] = nb; queue.push(n); }
        }
      }
      nb++;
    }
    this.nBodies = nb;
    return body;
  }

  volume() {
    const V = this.V;
    let s = 0;
    for (const [a, b, c] of this.T) s += dot(V[a], cross(V[b], V[c]));
    return s / 6;
  }

  /** A copy sharing everything but the labels. */
  withLabels(F) {
    const m = Object.create(KnitMesh.prototype);
    Object.assign(m, this);
    m.F = F;
    return m;
  }
}

/**
 * The B-rep's combinatorics from the labelled mesh alone: corners (vertex, body) where >= 3 patches
 * meet (or a patch boundary meets an open mesh boundary), and chains {a, b, path} - each run of
 * boundary between patches a and b, corner to corner (b = -1 along an open boundary). forceTris:
 * triangles every vertex of which is a corner too (facet faces: each of their edges is one chain).
 */
export function meshTopology(mesh, forceTris = null) {
  const { T, F, body: tbody } = mesh;
  const info = {};
  const fb = new Map();
  const facesAt = new Map();
  const bnd = new Map();                       // "a|u|v" -> b, in a's winding
  const openAt = new Set();
  const ck = (v, b) => v * BODY_STRIDE + b;
  for (let t = 0; t < T.length; t++) {
    const a = F[t], bt = tbody[t];
    fb.set(a, bt);
    for (let k = 0; k < 3; k++) {
      const u = T[t][k], v = T[t][(k + 1) % 3];
      const c = ck(u, bt);
      let s = facesAt.get(c);
      if (!s) facesAt.set(c, (s = new Set()));
      s.add(a);
      const n = mesh.across(t, u, v);
      const b = n >= 0 ? F[n] : -1;
      if (b !== a) {
        bnd.set(`${a}|${u}|${v}`, b);
        if (b === -1) { openAt.add(ck(u, bt)); openAt.add(ck(v, bt)); }
      }
    }
  }
  const corners = new Set();
  for (const [key, fs] of facesAt) if (fs.size >= 3) corners.add(key);
  for (const key of openAt) if (facesAt.get(key).size >= 2) corners.add(key);
  if (forceTris) for (const t of forceTris) for (const v of T[t]) corners.add(ck(v, tbody[t]));

  // chains: runs of boundary edges between one ordered pair, traversed along the lower face's winding
  // (open runs along their only face), cut at corners
  const succ = new Map();
  for (const [key, b] of bnd) {
    const [a, u, v] = key.split('|').map(Number);
    if (b === -1 || a < b) {
      const sk = `${a}|${b}|${u}`;
      let l = succ.get(sk);
      if (!l) succ.set(sk, (l = []));
      l.push(v);
    }
  }
  const used = new Set();
  const chains = [];
  const walk = (a, b, start) => {
    const path = [start];
    let cur = start;
    for (;;) {
      const nxt = (succ.get(`${a}|${b}|${cur}`) ?? []).filter((v) => !used.has(`${a}|${b}|${cur}|${v}`));
      if (!nxt.length) return path;
      const v = nxt[0];
      used.add(`${a}|${b}|${cur}|${v}`);
      path.push(v);
      cur = v;
      if (corners.has(ck(v, fb.get(a)))) return path;
    }
  };
  for (const [sk, vs] of succ) {
    const [a, b, u] = sk.split('|').map(Number);
    if (!corners.has(ck(u, fb.get(a)))) continue;
    for (const v of vs) {
      if (used.has(`${a}|${b}|${u}|${v}`)) continue;
      used.add(`${a}|${b}|${u}|${v}`);
      const path = corners.has(ck(v, fb.get(a))) ? [u, v] : [u, ...walk(a, b, v)];
      chains.push({ a, b, path });
    }
  }
  for (const [sk, vs] of succ) {
    const [a, b, u] = sk.split('|').map(Number);
    for (const v of vs) {
      if (used.has(`${a}|${b}|${u}|${v}`)) continue;
      used.add(`${a}|${b}|${u}|${v}`);
      const path = [u, ...walk(a, b, v)];
      if (path[path.length - 1] !== path[0]) info.broken_loop = (info.broken_loop ?? 0) + 1;
      corners.add(ck(path[0], fb.get(a)));       // a closed run gets a corner of its own
      chains.push({ a, b, path });
    }
  }
  return { fb, facesAt, corners, chains, info };
}
