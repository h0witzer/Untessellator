/**
 * retopo.js — a quad cage for a triangle mesh, by QuadriFlow (Huang, Zhou, Niessner, Shewchuk, Guibas 2018;
 * BSD-3-Clause, built to WebAssembly by tools/quadriflow/build.sh, licenses in tools/quadriflow/NOTICE). It comes
 * close to the quad count asked for and places few star points, which keeps the NURBS caps few and apart.
 * Single-threaded; the module (about 0.6 MB) loads on first use.
 */
import { parseQuadCage } from './objCage.js';

let loading = null;
export function loadQuadriFlow() {
  if (!loading) loading = import('../vendor/quadriflow/quadriflow.mjs').then((m) => m.default({ print: () => {}, printErr: () => {} }));
  return loading;
}

/**
 * Remesh. vertices: [[x, y, z]...]; triangles: [[a, b, c]...] (0-based). faces: the quad count to aim for.
 * Returns { vertices, faces } (a pure quad cage) or { error }.
 */
export async function quadRemesh(vertices, triangles, { faces = 300, seed = 0 } = {}) {
  const M = await loadQuadriFlow();
  // only the vertices the triangles use, renumbered (QuadriFlow's hierarchy has no place for loose points)
  const used = new Int32Array(vertices.length).fill(-1);
  let nv = 0;
  for (const t of triangles) for (const v of t) if (used[v] < 0) used[v] = nv++;
  const pV = M._malloc(nv * 24), pT = M._malloc(triangles.length * 12);
  const V = new Float64Array(M.HEAPF64.buffer, pV, nv * 3), T = new Int32Array(M.HEAP32.buffer, pT, triangles.length * 3);
  vertices.forEach((p, i) => { const j = used[i]; if (j >= 0) { V[j * 3] = p[0]; V[j * 3 + 1] = p[1]; V[j * 3 + 2] = p[2]; } });
  triangles.forEach((t, i) => { T[i * 3] = used[t[0]]; T[i * 3 + 1] = used[t[1]]; T[i * 3 + 2] = used[t[2]]; });
  const status = M.ccall('qf_remesh', 'number', ['number', 'number', 'number', 'number', 'number', 'number', 'string'],
    [pV, nv, pT, triangles.length, faces, seed, '/out.obj']);
  M._free(pV); M._free(pT);
  if (status !== 0) return { error: `QuadriFlow failed (status ${status})` };
  const cage = parseQuadCage(M.FS.readFile('/out.obj', { encoding: 'utf8' }));
  M.FS.unlink('/out.obj');
  return cage.error ? cage : dissolveValence2(cage);
}

/**
 * A vertex on just two quads, [v, a, x, b] and [v, b, y, a], is a fold the NURBS can't use: dropping v leaves the
 * one quad [a, x, b, y]. Repeats until none is left; unused vertices are dropped and the rest renumbered.
 */
export function dissolveValence2({ vertices, faces }) {
  let F = faces.map((q) => [...q]);
  for (let changed = true; changed;) {
    changed = false;
    const at = new Map();
    F.forEach((q, f) => q.forEach((v) => { if (!at.has(v)) at.set(v, []); at.get(v).push(f); }));
    for (const [v, list] of at) {
      if (list.length !== 2) continue;
      const [f1, f2] = list, q1 = F[f1], q2 = F[f2];
      const r1 = q1.indexOf(v), r2 = q2.indexOf(v);
      const [a, x, b] = [q1[(r1 + 1) % 4], q1[(r1 + 2) % 4], q1[(r1 + 3) % 4]];
      const y = q2[(r2 + 2) % 4];
      if (q2[(r2 + 1) % 4] !== b || q2[(r2 + 3) % 4] !== a) continue;   // not the two-quad fold
      F = F.filter((_, f) => f !== f1 && f !== f2).concat([[a, x, b, y]]);
      changed = true;
      break;
    }
  }
  const used = new Map();
  const out = F.map((q) => q.map((v) => { if (!used.has(v)) used.set(v, used.size); return used.get(v); }));
  const V = new Array(used.size);
  for (const [v, i] of used) V[i] = vertices[v];
  return { vertices: V, faces: out };
}
