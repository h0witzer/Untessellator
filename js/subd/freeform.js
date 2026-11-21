/**
 * freeform.js — a whole freeform mesh (sculpt, SubD or T-spline tessellation, scan) as a NURBS solid per body, no
 * kernel: QuadriFlow quad cage (retopo.js), cage fitted so its G2 surface lies on the mesh (fitCage.js), bi-3
 * patches with bi-6 caps at star points (fastNurbs.js), STEP written directly (patchStep.js), preview from the
 * patches (preview.js). Bodies are the studio's (js/bodies.js): each closed one becomes a solid; open bodies and
 * bodies with voids are refused with a reason.
 *
 * In:  positions (9 per triangle, the studio's export soup); bodies (a body id per triangle) and names ({id: name})
 *      optional; quads: the quad count to aim for per body (default from the body's vertex count, at most 1000:
 *      about four mesh vertices per cage vertex, so the data determines the cage).
 * Out: {step, preview, report: {bodies: [{name, triangles, quads, faces, stars, subdivisions, rms, max, error?}],
 *      ms}} with rms and max the fitted surface's distance to the body's mesh vertices, in model units.
 */
import { findBodies } from '../bodies.js';
import { quadRemesh } from './retopo.js';
import { fitCage } from './fitCage.js';
import { patchStep } from './patchStep.js';
import { patchPreview } from './preview.js';

/** The triangle soup welded on exact coordinates: { V (3 per vertex), idx (3 per triangle) }. */
function weld(positions) {
  const ids = new Map(), V = [], idx = new Int32Array(positions.length / 3);
  for (let i = 0; i < idx.length; i++) {
    const x = positions[i * 3], y = positions[i * 3 + 1], z = positions[i * 3 + 2], key = `${x},${y},${z}`;
    let v = ids.get(key);
    if (v === undefined) { v = V.length / 3; ids.set(key, v); V.push(x, y, z); }
    idx[i] = v;
  }
  return { V: Float64Array.from(V), idx };
}

export async function freeformSolid(positions, { bodies = null, names = null, part = 'Part', quads = null, onStatus = () => {} } = {}) {
  const t0 = performance.now();
  const { V, idx } = weld(positions);
  const found = findBodies(idx, V);
  const bodyOf = bodies ?? found.bodyOf;
  const groups = new Map();
  for (let t = 0; t < bodyOf.length; t++) {
    if (!groups.has(bodyOf[t])) groups.set(bodyOf[t], []);
    groups.get(bodyOf[t]).push(t);
  }
  const solids = [], report = [];
  for (const [id, tris] of groups) {
    const name = names?.[id] ?? `Body ${solids.length + report.filter((r) => r.error).length + 1}`;
    const row = { name, triangles: tris.length };
    report.push(row);
    const shells = new Set(tris.map((t) => found.shellOf[t]));
    if ([...shells].some((s) => !found.shells[s].closed)) { row.error = 'open: the freeform solid needs a closed mesh'; continue; }
    if (shells.size > 1) { row.error = 'has a cavity or a second shell: not supported by the freeform solid yet'; continue; }
    // the body's triangles, wound outward, on its own vertices
    const local = new Map(), vertices = [], faces = [];
    const at = (v) => { let i = local.get(v); if (i === undefined) { i = vertices.length; local.set(v, i); vertices.push([V[v * 3], V[v * 3 + 1], V[v * 3 + 2]]); } return i; };
    for (const t of tris) {
      const [a, b, c] = [idx[t * 3], idx[t * 3 + 1], idx[t * 3 + 2]];
      faces.push(found.flip[t] ? [at(a), at(c), at(b)] : [at(a), at(b), at(c)]);
    }
    const target = quads ?? Math.max(100, Math.min(1000, Math.round(vertices.length / 4)));
    try {
      onStatus(`${name}: quad cage…`);
      const ta = performance.now();
      const cage = await quadRemesh(vertices, faces, { faces: target });
      if (cage.error) { row.error = cage.error; continue; }
      onStatus(`${name}: fitting ${cage.faces.length} quads…`);
      const tb = performance.now();
      const fit = fitCage(cage, vertices, { rounds: 4 });
      const tc = performance.now();
      const h = fit.stats.history[fit.stats.history.length - 1], diag = fitDiag(vertices);
      Object.assign(row, { quads: cage.faces.length, faces: fit.faces.length, stars: starCount(cage.faces),
        subdivisions: fit.stats.subdivisions, rms: h.rms * diag, max: h.max * diag, ms: { remesh: tb - ta, fit: tc - tb } });
      solids.push({ name, faces: fit.faces, loops: fit.loops });
    } catch (err) {
      row.error = String(err?.message ?? err);
    }
  }
  if (!solids.length) return { step: null, preview: null, report: { bodies: report, ms: { total: performance.now() - t0 } } };
  onStatus('Writing STEP…');
  const td = performance.now();
  const step = patchStep(solids, { name: part });
  const te = performance.now();
  const preview = patchPreview(solids);
  const tf = performance.now();
  return { step, preview, report: { bodies: report, ms: { step: te - td, preview: tf - te, total: tf - t0 } } };
}

function starCount(faces) {
  const deg = new Map();
  for (const q of faces) for (const v of q) deg.set(v, (deg.get(v) ?? 0) + 1);
  let n = 0;
  for (const d of deg.values()) if (d !== 4) n++;
  return n;
}

function fitDiag(P) {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const p of P) for (let c = 0; c < 3; c++) { lo[c] = Math.min(lo[c], p[c]); hi[c] = Math.max(hi[c], p[c]); }
  return Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
}
