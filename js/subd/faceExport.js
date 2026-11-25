/**
 * faceExport.js — the face path: the studio's export payload (a painted mesh and its groups' surfaces) as a STEP
 * solid of trimmed faces, written here, with no geometry kernel. One STEP face per painted group (faceSolid.js
 * faceSolidSplit): an analytic group on its own surface, as the studio fitted it and the knitter's shared-axes step
 * then makes exact (shared axes and planes, tangencies, equal sizes: js/knit/axes.js), any other group a fitted
 * B-spline (split only where no single surface carries it). The mesh gives the topology, so every edge is shared
 * by its two faces by construction; nothing is sewn or healed.
 *
 * In:  { positions (9 per triangle), labels (group per triangle, < 0 unpainted), surfaces (Map group -> [type,
 *        params]), bodies? (the studio's body id per triangle) }, { part, names? (by body id), split? }
 * Out: { step, report, faceKeys } or { error, report }: faceKeys, per face in the file's order, { b: its body, tris:
 * its triangles in that body, g: the same as the payload's triangle indices, label: its painted group } for split, and
 * for an analytic face also { analytic: true }
 * (knit/facePath.js fits that group freely instead); report as the knitter's (js/knit/index.js knitSolid) where the studio
 *      reads it (tier 'analytic': every face on a surface, no facets), with info.path 'faces' and what it did.
 *
 * Unpainted triangles: each connected run a group of its own, fitted like a freeform one. A body with a void, or a
 * mesh with a hole in it, is not this path's (the knitter's facet tiers and repair take those).
 */
import { findBodies } from '../bodies.js';
import { faceSolidSplit } from './faceSolid.js';
import { faceStep } from './faceStep.js';
import { ANALYTIC } from './analytic.js';
import { KnitMesh } from '../knit/mesh.js';
import { shareAxes } from '../knit/axes.js';

export function faceExport({ positions, labels, surfaces, bodies = null }, { part = 'Part', split = null, names = null } = {}) {
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
  // the analytic surfaces made exact together, as the knitter does before building: an edge between two of them is
  // then their exact intersection (faceSolid.js), which raw fits a hair apart do not have
  if ([...given.values()].some((s) => ANALYTIC.has(s[0]))) {
    const ta = performance.now(), r = shareAxes(new KnitMesh(positions, labels), given);
    for (const [g, s] of r.surfaces) if (s) given.set(g, s);
    Object.assign(info, r.info, { axes_ms: Math.round(performance.now() - ta) });
  }
  const retyped = [];
  const report = (extra) => ({
    tier: 'none', bestTier: 'none', facetTris: 0, facetArea: 0, closed: false, valid: false, vol_err: null, free_edges: null,
    nonmanifold_edges: null, shells: null, groups: new Set(labels).size, meshPieces: found.bodies.length, triangles: nT,
    secs: Math.round((performance.now() - t0) / 100) / 10, info,
    faces: { problems: {}, moves: [], absorbed: [], retyped: [], final: [], demoted: [], pieces: [] }, ...extra,
  });
  // unpainted triangles: a group per connected run, labelled past the painted ones
  const lab = Int32Array.from(labels);
  // (a loop, not Math.max(...lab): a browser takes only so many arguments, and a part has more triangles)
  let next = 1;
  for (const g of lab) if (g + 1 > next) next = g + 1;
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
  // The solids as the knitter writes them (js/knit/index.js knitSolid): each mesh body under the studio's body it
  // belongs to (most of its triangles), in body order, with that body's name (a body in several pieces numbered "Name
  // 1", "Name 2"); with no names, Body 1..n. The file then names its bodies the same whichever path built it.
  const appOf = found.bodies.map((_, b) => (bodies ? new Map() : b));
  if (bodies) for (let t = 0; t < nT; t++) { const m = appOf[found.bodyOf[t]]; m.set(bodies[t], (m.get(bodies[t]) ?? 0) + 1); }
  const app = appOf.map((m) => (m instanceof Map ? [...m].sort((x, y) => y[1] - x[1] || x[0] - y[0])[0][0] : m));
  const ids = [...new Set(app)].sort((x, y) => x - y);
  const order = found.bodies.map((_, b) => b).sort((x, y) => app[x] - app[y] || x - y);
  const baseName = (id) => names?.[id] ?? `Body ${ids.indexOf(id) + 1}`;
  const pieces = new Map(), piece = new Map();
  for (const b of order) { pieces.set(app[b], (pieces.get(app[b]) ?? 0) + 1); piece.set(b, pieces.get(app[b])); }
  const nameOf = (b) => (pieces.get(app[b]) > 1 ? `${baseName(app[b])} ${piece.get(b)}` : baseName(app[b]));
  const solids = [], final = [], rows = [], loose = [], faceKeys = [], gaps = {}, exactN = {};
  let bulge = 0;
  let edgeTol = 0;
  for (const b of order) {
    const body = found.bodies[b];
    if (!body.closed) return { error: 'the mesh has holes (open edges)', report: report({}) };
    if (body.shells.length > 1) return { error: 'a body with a void', report: report({}) };
    const local = new Map(), Vb = [], Tb = [], Lb = [], Gb = [];
    const at = (v) => { let i = local.get(v); if (i === undefined) { i = Vb.length; local.set(v, i); Vb.push(V[v]); } return i; };
    for (let t = 0; t < nT; t++) {
      if (found.bodyOf[t] !== b) continue;
      const [a, c, d] = [idx[t * 3], idx[t * 3 + 1], idx[t * 3 + 2]];
      Tb.push(found.flip[t] ? [at(a), at(d), at(c)] : [at(a), at(c), at(d)]);
      Lb.push(lab[t]);
      Gb.push(t);
    }
    let r;
    // split: faces to cut first, [{ b, tris }] (the kernel's verdicts on an earlier try, knit/facePath.js)
    const cut = (split ?? []).filter((x) => x.b === b).map((x) => x.tris);
    try { r = faceSolidSplit(Vb, Tb, Lb, { scale: diag, surfaces: given, split: cut }); } catch (e) { r = { error: `threw: ${String(e?.message ?? e).slice(0, 200)}` }; }
    if (r.error) return { error: r.error, report: report({}) };
    for (const sp of r.splits ?? []) info.splits.push(sp);
    for (const [g, was] of r.retyped ?? []) retyped.push([g, was, 'bspline']);
    for (const f of r.faces) {
      const g = f.tris.map((i) => Gb[i]);
      faceKeys.push(f.params?.analytic ? { b, tris: f.tris, g, label: f.origin ?? f.label, analytic: true } : { b, tris: f.tris, g, label: f.origin ?? f.label });
      if (f.params?.analytic) info.analytic_faces++; else info.fitted_faces++;
      final.push([f.origin ?? f.label, f.params?.analytic ? f.params.analytic.type : 'bspline']);
    }
    edgeTol = Math.max(edgeTol, r.report.edgeTol);
    loose.push(...r.report.looseEdges);
    for (const [k, g] of Object.entries(r.report.edgeGaps ?? {})) { for (const x of g.gaps) (gaps[k] ??= []).push(x); exactN[k] = (exactN[k] ?? 0) + (g.exact ?? 0); }
    bulge += r.report.bulgeVolume;
    solids.push({ faces: r.faces, edges: r.edges, points: r.points, name: nameOf(b), tolerance: r.tolerance });
    const row = rows.find((x) => x.id === app[b]);
    if (row) row.triangles += Tb.length;
    else rows.push({ id: app[b], name: baseName(app[b]), tier: 'analytic', bestTier: 'analytic', facetTris: 0, facetArea: 0, triangles: Tb.length });
  }
  info.edge_tol = edgeTol;
  // the edges by what meets there (faceSolid's edgeGaps): count, median and worst gap, x the diagonal
  info.edge_gaps = Object.fromEntries(Object.entries(gaps).map(([k, xs]) => { xs.sort((a, b) => a - b); return [k, { n: xs.length, exact: exactN[k] ?? 0, median: xs[xs.length >> 1], max: xs[xs.length - 1] }]; }));
  info.bulge_volume = bulge;
  info.loose_edges = loose.sort((a, b) => b.tol - a.tol).slice(0, 10);
  const tol = Math.max(...solids.map((s) => s.tolerance));
  const step = faceStep(solids, { name: part, tol });
  info.step_faces = solids.reduce((s, x) => s + x.faces.length, 0);
  final.sort((x, y) => x[0] - y[0]);
  return {
    step, solids, faceKeys,
    report: report({ tier: 'analytic', bestTier: 'analytic', closed: true, valid: true, free_edges: 0, nonmanifold_edges: 0, shells: solids.length, bodies: rows,
      faces: { problems: {}, moves: [], absorbed: [], retyped, final, demoted: [], pieces: [] } }),
  };
}
