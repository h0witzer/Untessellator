/**
 * faceExport.js — the face path: the studio's export payload (a painted mesh and its groups' surfaces) as a STEP
 * solid of trimmed faces, written here, with no geometry kernel. One STEP face per painted group (faceSolid.js
 * faceSolidSplit): an analytic group on its own surface exactly as the studio fitted it, any other group a fitted
 * B-spline (split only where no single surface carries it). The mesh gives the topology, so every edge is shared
 * by its two faces by construction; nothing is sewn or healed.
 *
 * In:  { positions (9 per triangle), labels (group per triangle, < 0 unpainted), surfaces (Map group -> [type,
 *        params]) }, { part, names? }
 * Out: { step, report } or { error, report }: report as the knitter's (js/knit/index.js knitSolid) where the studio
 *      reads it (tier 'analytic': every face on a surface, no facets), with info.path 'faces' and what it did.
 *
 * Unpainted triangles: each connected run a group of its own, fitted like a freeform one. A body with a void, or a
 * mesh with a hole in it, is not this path's (the knitter's facet tiers and repair take those).
 */
import { findBodies } from '../bodies.js';
import { faceSolidSplit } from './faceSolid.js';
import { faceStep } from './faceStep.js';
import { ANALYTIC } from './analytic.js';

export function faceExport({ positions, labels, surfaces }, { part = 'Part' } = {}) {
  const t0 = performance.now();
  const nT = labels.length;
  // weld on exact coordinates (the payload is a triangle soup)
  const key = new Map(), V = [], idx = new Int32Array(nT * 3);
  for (let i = 0; i < nT * 3; i++) {
    const x = positions[i * 3], y = positions[i * 3 + 1], z = positions[i * 3 + 2], k = `${x},${y},${z}`;
    let v = key.get(k);
    if (v === undefined) { v = V.length; key.set(k, v); V.push([x, y, z]); }
    idx[i] = v;
  }
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const p of V) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], p[k]); hi[k] = Math.max(hi[k], p[k]); }
  const diag = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
  const found = findBodies(idx, Float64Array.from(V.flat()));
  const surfs = surfaces instanceof Map ? surfaces : new Map(Object.entries(surfaces ?? {}).map(([k, v]) => [Number(k), v]));
  const given = new Map([...surfs].filter(([, s]) => s && (ANALYTIC.has(s[0]) || s[0] === 'bspline')));
  const info = { path: 'faces', bodies: found.bodies.length, splits: [], analytic_faces: 0, fitted_faces: 0 };
  const retyped = [];
  const report = (extra) => ({
    tier: 'none', bestTier: 'none', facetTris: 0, facetArea: 0, closed: false, valid: false, vol_err: null, free_edges: null,
    nonmanifold_edges: null, shells: null, groups: new Set(labels).size, meshPieces: found.bodies.length, triangles: nT,
    secs: Math.round((performance.now() - t0) / 100) / 10, info,
    faces: { problems: {}, moves: [], absorbed: [], retyped: [], final: [], demoted: [], pieces: [] }, ...extra,
  });
  // unpainted triangles: a group per connected run, labelled past the painted ones
  const lab = Int32Array.from(labels);
  let next = Math.max(0, ...lab) + 1;
  {
    const nbr = new Map(), runOf = new Int32Array(nT).fill(-1);
    for (let t = 0; t < nT; t++) if (lab[t] < 0) for (let k = 0; k < 3; k++) {
      const a = idx[t * 3 + k], b = idx[t * 3 + (k + 1) % 3], e = a < b ? `${a}_${b}` : `${b}_${a}`;
      (nbr.get(e) ?? nbr.set(e, []).get(e)).push(t);
    }
    for (let t = 0; t < nT; t++) {
      if (lab[t] >= 0 || runOf[t] >= 0) continue;
      const g = next++, stack = [t];
      runOf[t] = g;
      while (stack.length) {
        const u = stack.pop();
        lab[u] = g;
        for (let k = 0; k < 3; k++) {
          const a = idx[u * 3 + k], b = idx[u * 3 + (k + 1) % 3];
          for (const w of nbr.get(a < b ? `${a}_${b}` : `${b}_${a}`) ?? []) if (runOf[w] < 0) { runOf[w] = g; stack.push(w); }
        }
      }
    }
  }
  const solids = [], final = [], rows = [], loose = [];
  let bulge = 0;
  let edgeTol = 0;
  for (let b = 0; b < found.bodies.length; b++) {
    const body = found.bodies[b];
    if (!body.closed) return { error: 'the mesh has holes (open edges)', report: report({}) };
    if (body.shells.length > 1) return { error: 'a body with a void', report: report({}) };
    const local = new Map(), Vb = [], Tb = [], Lb = [];
    const at = (v) => { let i = local.get(v); if (i === undefined) { i = Vb.length; local.set(v, i); Vb.push(V[v]); } return i; };
    for (let t = 0; t < nT; t++) {
      if (found.bodyOf[t] !== b) continue;
      const [a, c, d] = [idx[t * 3], idx[t * 3 + 1], idx[t * 3 + 2]];
      Tb.push(found.flip[t] ? [at(a), at(d), at(c)] : [at(a), at(c), at(d)]);
      Lb.push(lab[t]);
    }
    let r;
    try { r = faceSolidSplit(Vb, Tb, Lb, { scale: diag, surfaces: given }); } catch (e) { r = { error: `threw: ${String(e?.message ?? e).slice(0, 200)}` }; }
    if (r.error) return { error: r.error, report: report({}) };
    for (const sp of r.splits ?? []) info.splits.push(sp);
    for (const [g, was] of r.retyped ?? []) retyped.push([g, was, 'bspline']);
    for (const f of r.faces) {
      if (f.params?.analytic) info.analytic_faces++; else info.fitted_faces++;
      final.push([f.origin ?? f.label, f.params?.analytic ? f.params.analytic.type : 'bspline']);
    }
    edgeTol = Math.max(edgeTol, r.report.edgeTol);
    loose.push(...r.report.looseEdges);
    bulge += r.report.bulgeVolume;
    solids.push({ faces: r.faces, edges: r.edges, points: r.points, name: `Body ${b + 1}`, tolerance: r.tolerance });
    rows.push({ id: b, name: `Body ${b + 1}`, tier: 'analytic', bestTier: 'analytic', facetTris: 0, facetArea: 0, triangles: Tb.length });
  }
  info.edge_tol = edgeTol;
  info.bulge_volume = bulge;
  info.loose_edges = loose.sort((a, b) => b.tol - a.tol).slice(0, 10);
  const tol = Math.max(...solids.map((s) => s.tolerance));
  const step = faceStep(solids, { name: part, tol });
  info.step_faces = solids.reduce((s, x) => s + x.faces.length, 0);
  final.sort((x, y) => x[0] - y[0]);
  return {
    step, solids,
    report: report({ tier: 'analytic', bestTier: 'analytic', closed: true, valid: true, free_edges: 0, nonmanifold_edges: 0, shells: solids.length, bodies: rows,
      faces: { problems: {}, moves: [], absorbed: [], retyped, final, demoted: [], pieces: [] } }),
  };
}
