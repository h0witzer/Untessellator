/**
 * meshFixWorker.js — the mesh fixes in Fix bad areas, worked out off the main thread as a mesh opens and after each
 * edit, so the card can list them and Apply is at once. In order:
 *
 *   pieces  small loose pieces go (the triangles the studio marks)
 *   holes   js/holeFill.js: fins and walls stuck to the surface go, then every hole is capped (on the survey's 45 game
 *           models, closed meshes 33 -> 43, after the import repair)
 *   join    pieces that cut through each other become one solid, by a boolean union in Manifold (manifold-3d 3.5.4,
 *           Apache-2.0, 532 KB of WebAssembly from the CDN): 38 of the survey's 51 meshes with overlapping pieces
 *           joined whole, median 6 ms; OCCT's fuse of the same pieces took 0.7 s to 20 s on 400 to 10,700 triangles
 *
 * In:  {id, positions: Float32Array (9 per triangle), small: Uint8Array (per triangle) | null, steps: {pieces, holes, join}}
 * Out: {id, positions, src (each output triangle's input triangle; -1 for a cap), pieces: {removed, triangles},
 *       holes: holeFill's report, join: {pieces, joined, refused, overlap, bodies}} or {id, error}
 * A piece goes into the union with each triangle's index as its faceID, which Manifold carries to every triangle it
 * makes of it; so src maps the result back through all three, and the studio's groups come across.
 */

import { fillHoles } from '../holeFill.js';

let wasm = null;
const manifold = () => (wasm ??= import('https://cdn.jsdelivr.net/npm/manifold-3d@3.5.4/manifold.js')
  .then(({ default: Module }) => Module()).then((m) => { m.setup(); return m; }));

/** The soup's pieces (triangles joined through shared edges, corners welded exactly): lists of triangles. */
function pieces(P) {
  const nT = P.length / 9;
  const ids = new Map(), idx = new Int32Array(nT * 3);
  for (let c = 0; c < nT * 3; c++) {
    const k = `${P[c * 3]},${P[c * 3 + 1]},${P[c * 3 + 2]}`;
    let i = ids.get(k);
    if (i === undefined) { i = ids.size; ids.set(k, i); }
    idx[c] = i;
  }
  const nV = ids.size;
  const parent = Int32Array.from({ length: nT }, (_, i) => i);
  const find = (i) => { while (parent[i] !== i) { parent[i] = parent[parent[i]]; i = parent[i]; } return i; };
  const firstAt = new Map();
  for (let t = 0; t < nT; t++) {
    for (let k = 0; k < 3; k++) {
      const a = idx[t * 3 + k], b = idx[t * 3 + (k + 1) % 3], key = Math.min(a, b) * nV + Math.max(a, b);
      const f = firstAt.get(key);
      if (f === undefined) firstAt.set(key, t); else parent[find(t)] = find(f);
    }
  }
  const groups = new Map();
  for (let t = 0; t < nT; t++) { const r = find(t); (groups.get(r) ?? groups.set(r, []).get(r)).push(t); }
  return { list: [...groups.values()], idx };
}

/** Manifold's union of the soup's pieces: {positions, src (into P), report}, or null with fewer than two closed pieces. */
async function join(P) {
  const { list, idx } = pieces(P);
  const report = { pieces: list.length, joined: 0, refused: 0, overlap: false, bodies: list.length };
  if (list.length < 2) return { report };
  const { Manifold, Mesh } = await manifold();
  const made = [], accepted = [], refusedTris = [];
  let volume = 0;
  try {
    for (const tris of list) {
      const local = new Map(), verts = [], triVerts = new Uint32Array(tris.length * 3);
      tris.forEach((t, j) => {
        for (let k = 0; k < 3; k++) {
          const g = idx[t * 3 + k];
          let l = local.get(g);
          if (l === undefined) { l = verts.length / 3; local.set(g, l); verts.push(P[t * 9 + k * 3], P[t * 9 + k * 3 + 1], P[t * 9 + k * 3 + 2]); }
          triVerts[j * 3 + k] = l;
        }
      });
      try {
        const mesh = new Mesh({ numProp: 3, vertProperties: Float32Array.from(verts), triVerts, faceID: Uint32Array.from(tris) });
        mesh.merge();
        const m = new Manifold(mesh);
        made.push(m);
        if (String(m.status()) !== 'NoError' || m.isEmpty()) { refusedTris.push(...tris); continue; }
        accepted.push(m);
        volume += m.volume();
      } catch {
        refusedTris.push(...tris);           // not a clean closed surface (open, folded or pinched): left as it is
      }
    }
    report.joined = accepted.length;
    report.refused = list.length - accepted.length;
    if (accepted.length < 2) return { report };
    const union = Manifold.union(accepted);
    made.push(union);
    report.overlap = volume - union.volume() > 1e-6 * volume;
    const bodies = union.decompose();
    report.bodies = bodies.length + report.refused;
    for (const b of bodies) b.delete();
    if (!report.overlap) return { report };    // nothing cuts anything: the mesh stays as it is
    const out = union.getMesh();
    const nOut = out.triVerts.length / 3;
    const positions = new Float32Array((nOut + refusedTris.length) * 9);
    const src = new Int32Array(nOut + refusedTris.length);
    for (let t = 0; t < nOut; t++) {
      for (let k = 0; k < 3; k++) {
        const v = out.triVerts[t * 3 + k] * out.numProp;
        positions.set([out.vertProperties[v], out.vertProperties[v + 1], out.vertProperties[v + 2]], t * 9 + k * 3);
      }
      src[t] = out.faceID[t];
    }
    refusedTris.forEach((t, j) => { positions.set(P.subarray(t * 9, t * 9 + 9), (nOut + j) * 9); src[nOut + j] = t; });
    return { positions, src, report };
  } finally {
    for (const m of made) m.delete();
  }
}

self.onmessage = async ({ data }) => {
  const { id, positions, small, steps = { pieces: true, holes: true, join: true } } = data;
  try {
    let P = positions;
    let src = Int32Array.from({ length: P.length / 9 }, (_, t) => t);
    const after = (next, map) => { P = next; src = Int32Array.from(map, (s) => (s < 0 ? -1 : src[s])); };
    const result = { id, pieces: { removed: 0, triangles: 0 }, holes: null, join: null };
    // small loose pieces
    if (steps.pieces && small) {
      const keep = [];
      for (let t = 0; t < small.length; t++) if (!small[t]) keep.push(t);
      if (keep.length < small.length) {
        const next = new Float32Array(keep.length * 9);
        keep.forEach((t, i) => next.set(P.subarray(t * 9, t * 9 + 9), i * 9));
        result.pieces = { removed: data.smallPieces ?? 0, triangles: small.length - keep.length };
        after(next, keep);
      }
    }
    // holes, fins and walls
    if (steps.holes) {
      const r = fillHoles(P);
      result.holes = r.report;
      if (r.report.filled || r.report.removed_triangles) after(Float32Array.from(r.positions), r.src);
    }
    // pieces that cut through each other
    if (steps.join) {
      const r = await join(P);
      result.join = r.report;
      if (r.positions) after(r.positions, r.src);
    }
    result.positions = P;
    result.src = src;
    self.postMessage(result);
  } catch (error) {
    self.postMessage({ id, error: String(error?.message ?? error) });
  }
};
