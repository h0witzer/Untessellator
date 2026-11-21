/**
 * retopo.js — a quad cage for a triangle mesh, by Instant Meshes (Jakob, Tarini, Panozzo, Sorkine-Hornung 2015;
 * BSD-3-Clause, built to WebAssembly by tools/instant-meshes/build.sh, licences in tools/instant-meshes/NOTICE).
 * Single-threaded; the module (about 0.5 MB) loads on first use.
 */
import { parseQuadCage } from './objCage.js';

let loading = null;
export function loadInstantMeshes() {
  if (!loading) loading = import('../vendor/instant-meshes/instant-meshes.mjs').then((m) => m.default({ print: () => {}, printErr: () => {} }));
  return loading;
}

/**
 * Remesh. vertices: [[x, y, z]...]; triangles: [[a, b, c]...] (0-based). faces: the target quad count.
 * Returns { vertices, faces } (a pure quad cage, Instant Meshes' extrinsic 4-RoSy/4-PoSy mode) or { error }.
 */
export async function quadRemesh(vertices, triangles, { faces = 300, creaseDegrees = -1 } = {}) {
  const M = await loadInstantMeshes();
  const lines = [];
  for (const v of vertices) lines.push(`v ${v[0]} ${v[1]} ${v[2]}`);
  for (const t of triangles) lines.push(`f ${t[0] + 1} ${t[1] + 1} ${t[2] + 1}`);
  M.FS.writeFile('/in.obj', lines.join('\n'));
  const status = M.ccall('im_remesh', 'number', ['string', 'string', 'number', 'number', 'number'], ['/in.obj', '/out.obj', faces, 1, creaseDegrees]);
  if (status !== 0) return { error: `Instant Meshes failed (status ${status})` };
  const cage = parseQuadCage(M.FS.readFile('/out.obj', { encoding: 'utf8' }));
  M.FS.unlink('/in.obj'); M.FS.unlink('/out.obj');
  return cage.error ? cage : dissolveValence2(cage);
}

/**
 * Instant Meshes can leave a vertex on just two quads, [v, a, x, b] and [v, b, y, a]: dropping v leaves the one quad
 * [a, x, b, y]. Repeats until none is left; unused vertices are dropped and the rest renumbered.
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
