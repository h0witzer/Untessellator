/**
 * knit/index.js — a STEP solid from painted groups: the topo knitter (ml/solid_rebuild.py's method,
 * ported), the facet fallback, with an honest account of what came out, and the kernel-judged repair loop
 * as a step of its own (repairGroups), so the export builds the groups and types it is given.
 *
 *   const oc = await loadOcct();
 *   const { solid, report } = knitSolid(oc, { positions, labels, surfaces });
 *   const step = writeStep(oc, solid);
 *
 * positions  flat triangle soup (9 per triangle), the coordinates the STEP is written in
 * labels     group id per triangle (-1 unpainted: built as facets by the fallback, never dropped)
 * surfaces   Map group id -> [type, params] (gaussMap's param format), for groups that have one
 */

import { KnitMesh } from './mesh.js';
import { buildRepaired, facetFallback, progressiveRecovery, LOOP, now } from './repair.js';
import { buildPolyhedral, grade, tidy, better, BUILD } from './build.js';
import { readBrep } from './occt.js';
import { buildFacets } from '../meshCore.js';
import { attachCurvature } from '../curvature.js';
import { fitRegionAs } from '../gaussMap.js';
import { fitPieces } from '../freeformFit.js';
import { shareAxes } from './axes.js';

export { loadOcct, writeStep, brepText, readBrep, readStep } from './occt.js';
export { BUILD } from './build.js';

const TYPES = ['plane', 'cylinder', 'cone', 'sphere', 'torus'];

/** The repair loop's alternative surfaces: every analytic type fitted to each group, best first. */
export function fitCandidatesFor(positions) {
  let facets = null;
  return (groups) => {
    if (!facets) {
      const P = Float64Array.from(positions);
      facets = buildFacets(P);
      attachCurvature(P, facets);
    }
    const out = {};
    for (const [g, tris] of Object.entries(groups)) {
      const found = [];
      for (const type of TYPES) {
        let fit;
        try { fit = fitRegionAs(type, new Set(tris), facets); } catch { continue; }
        if (!fit?.params || !Number.isFinite(fit.residual)) continue;
        found.push({ type, params: fit.params, residual: fit.residual });
      }
      found.sort((a, b) => a.residual - b.residual);
      out[g] = found;
    }
    return out;
  };
}

/** Groups of more than this many triangles with no surface get a fitted B-spline (solid_rebuild.py --freeform). */
export const FREEFORM_MIN_FACETS = 4;

/**
 * A B-spline (js/freeformFit.js) for every group with no surface and more than FREEFORM_MIN_FACETS triangles.
 * Python measured this as solid_rebuild --freeform: freeform parts rebuilt 0 -> 102 of 150 with true groups.
 * A group no single map covers is split (fitPieces): its pieces after the first get NEW labels, written
 * into mesh.F; a piece still failing keeps a label of its own and goes out as facets. Adds to `surfaces`;
 * returns counts.
 */
export function addFreeform(mesh, surfaces, onProgress = null, { split = true, firstLabel = 0 } = {}) {
  const byLabel = new Map();
  let next = firstLabel;                         // new labels from here up (knitSolid's bodies: unique across them)
  for (let t = 0; t < mesh.T.length; t++) {
    const g = mesh.F[t];
    next = Math.max(next, g + 1);
    if (g < 0 || surfaces.has(g)) continue;
    (byLabel.get(g) ?? byLabel.set(g, []).get(g)).push(t);
  }
  const out = { freeform_groups: 0, freeform_fitted: 0, freeform_failed: 0, freeform_split_groups: 0, freeform_periodic: 0 };
  for (const [g, tris] of byLabel) {
    if (tris.length <= FREEFORM_MIN_FACETS) continue;
    out.freeform_groups++;
    onProgress?.({ freeform: out.freeform_groups, of: byLabel.size });
    const local = new Map(), V = [], T = [];
    for (const t of tris) T.push(mesh.T[t].map((v) => { if (!local.has(v)) { local.set(v, V.length); V.push(mesh.V[v]); } return local.get(v); }));
    let r;
    try { r = fitPieces(V, T, { scale: mesh.diag, maxDepth: split ? 4 : 0 }); } catch { r = { pieces: [], failed: [{ tris: T.map((_, i) => i) }] }; }
    out.freeform_failed += r.failed.length;
    if (r.pieces.length > 1 || r.failed.length) out.freeform_split_groups++;
    r.pieces.forEach(({ tris: sub, params, info }, k) => {
      let label = g;
      if (k > 0) { label = next++; for (const i of sub) mesh.F[tris[i]] = label; }
      surfaces.set(label, ['bspline', params]);
      out.freeform_fitted++;
      if (info.param === 'periodic') out.freeform_periodic++;
    });
    // the unfitted rest keeps a label of its own, apart from the fitted pieces
    if (r.pieces.length) for (const { tris: sub } of r.failed) { const label = next++; for (const i of sub) mesh.F[tris[i]] = label; }
  }
  return out;
}

/**
 * Knit a solid. Returns {solid (compound of solids, to write), report}: report.tier is analytic (every
 * face on a fitted surface), hybrid / planar (some patches as facets: report.facetTris triangles,
 * report.facetArea of the area), faceted (the mesh itself), or none (no closed, valid solid of the
 * mesh's volume; the best attempt is still returned, and report says how it fails). freeform: groups with
 * no surface get a fitted B-spline first (addFreeform). A build that is not fully analytic is mended where it
 * fails (repair.js progressiveRecovery: other edge methods there, then facets, each place rebuilt alone while it
 * is tried), then the planar and faceted tiers; progressive false: the whole part rebuilt plain and from the
 * mesh's points, then the facet tiers, as before 2026-11-15. No group changes type or gives facets to another:
 * that is repairGroups, which the studio runs in view. `deadline` (s): past it no retry, trial or facet tier
 * starts (a running OCCT call cannot be stopped, so callers that must bound the time also need a hard stop,
 * as the studio's worker timeout). mode 'faceted': the mesh as a polyhedral solid only.
 */
export function knitSolid(oc, { positions, labels, surfaces, bodies = null, names = null, oriented = false }, options = {}) {
  const ids = bodies ? [...new Set(bodies)].sort((a, b) => a - b) : [];
  if (ids.length <= 1) {
    const r = knitOne(oc, { positions, labels, surfaces, oriented }, options);
    const id = ids[0] ?? 0, name = names?.[id] ?? null;
    r.report.bodies = [{ id, name: name ?? 'Body 1', ...bodyRow(r.report) }];
    // a name is written only when the file says what its bodies are called (3MF/OBJ objects, the Bodies list)
    return { ...r, names: name !== null ? solidNames(oc, r.solid, [name]) : null };
  }
  return knitBodies(oc, { positions, labels, surfaces, bodies, names, oriented, ids }, options);
}

/** One body (or the whole mesh, as before bodies): knitSolid's build, recovery and facet tiers. */
function knitOne(oc, { positions, labels, surfaces, oriented = false }, { fallback = true, onProgress = null, deadline = Infinity,
  mode = 'knit', freeform = true, split = true, axes = true, axesOptions = {}, tidy: tidyUp = true, retryPlain = true,
  progressive = true, firstLabel = 0 } = {}) {
  const t0 = performance.now();
  const end = now() + deadline;                  // past it: no plain retry or facet tiers
  progressive &&= fallback;                      // with no facets, the whole-part retries as before (test/reportReplay.mjs)
  const mesh = new KnitMesh(positions, labels, { oriented });
  let surfs = surfaces instanceof Map ? surfaces : new Map(Object.entries(surfaces).map(([k, v]) => [Number(k), v]));
  const info = {};
  let shape = null, state = null, out, pieces = null;
  if (mode === 'faceted') {
    // The mesh as a solid and nothing else: the studio's answer when the analytic build ran out of time.
    const { shape: sh } = buildPolyhedral(oc, mesh);
    const { g, sc } = grade(oc, sh, mesh, {});
    out = { shape: sh, sc, tier: g[0] ? 'faceted' : 'none', bestTier: 'faceted', facetTris: mesh.T.length, facetArea: 1 };
    return finish(out);
  }
  const unsplit = Int32Array.from(mesh.F);
  if (freeform) Object.assign(info, addFreeform(mesh, surfs, onProgress, { split, firstLabel }));
  pieces = piecesOf(unsplit, mesh.F);
  const plainSurfs = new Map(surfs);              // the fits as they came, for the plain retry
  // shared axes and directions made exact (knit/axes.js): coaxial faces meet without a kink
  if (axes) {
    const ta = performance.now();
    const r = shareAxes(mesh, surfs, axesOptions);
    surfs = r.surfaces;
    Object.assign(info, r.info, { axes_ms: Math.round(performance.now() - ta) });
  }
  // each build's time (ms_clean, ms_plain, ms_mesh_clean, ms_mesh_plain): where a part's seconds went
  // (BUILD.PROF: each build's problems too, probs_<name>)
  const attempt = (sf, name) => {
    const t = performance.now();
    try {
      const r = buildRepaired(oc, mesh, sf, { maxBuilds: 1, onProgress });
      if (BUILD.PROF && r?.state) info[`probs_${name}`] = Object.fromEntries(Object.entries(r.state.problems).map(([k, v]) => [k, [...v].sort((x, y) => x - y)]));
      return r;
    } catch (err) {
      info.exception = String(err?.message ?? err).slice(0, 120);
      if (globalThis.process?.env?.KNIT_DEBUG) info.exception_stack = String(err?.stack ?? '').split(/\r?\n/).slice(0, 8).join(' | ');
      return null;
    } finally {
      info[`ms_${name}`] = Math.round(performance.now() - t);
    }
  };
  let r = attempt(surfs, 'clean');
  // The clean build (shared axes, exact edges, contact loci) is not analytic everywhere the plain one is:
  // on 274 Onshape parts each was fully analytic on parts the other was not (deterministic gates, no
  // time budget: clean 250, plain 254, 8 one way and 5 the other). So a clean build that is not fully
  // analytic is retried plain, time permitting, and the plain one is kept if it is.
  const clean = axes || BUILD.EXACT_EDGES || BUILD.CONTACT;
  // progressive (repair.js progressiveRecovery): those methods are tried where the part fails, not on all of it
  if (!progressive && clean && retryPlain && !(r?.state?.ok) && now() < end - (deadline === Infinity ? 0 : deadline / 4)) {
    const saved = { ...BUILD };
    BUILD.EXACT_EDGES = false; BUILD.CONTACT = false;
    let r2 = null;
    try { r2 = attempt(plainSurfs, 'plain'); } finally { Object.assign(BUILD, saved); }
    info.plain_retry = 1;
    // the plain build replaces the clean one only when it is fully analytic: otherwise the clean one goes on
    // to the facet tiers (a plain build that grades better but is not analytic fell to facets where the
    // clean one made a hybrid, 00005406_001)
    if (r2?.state?.ok || (!r?.state && r2?.state)) { r = r2; info.plain_retry_won = 1; }
  }
  // ssi's edges (build.js ssiArcs) are not analytic everywhere the mesh's own points are (6 of the 275 Onshape
  // parts need the mesh's, 2026-11-01). So a part that is still not fully analytic is built again from the points
  // the mesh has, clean then plain, time permitting, and that build is kept when it is fully analytic.
  const timeLeft = () => now() < end - (deadline === Infinity ? 0 : deadline / 4);
  if (!progressive && BUILD.SSI && !(r?.state?.ok) && timeLeft()) {
    const saved = { ...BUILD };
    BUILD.SSI = false;
    let r3 = null;
    try {
      r3 = attempt(surfs, 'mesh_clean');
      if (clean && retryPlain && !(r3?.state?.ok) && timeLeft()) {
        BUILD.EXACT_EDGES = false; BUILD.CONTACT = false;
        r3 = attempt(plainSurfs, 'mesh_plain');
      }
    } finally { Object.assign(BUILD, saved); }
    info.mesh_points_retry = 1;
    if (r3?.state?.ok) { r = r3; info.mesh_points_won = 1; }
  }
  if (r) { shape = r.shape; state = r.state; Object.assign(info, r.info); }
  if (state && (state.ok || !fallback)) {
    out = { shape, sc: state.sc, tier: state.ok ? 'analytic' : 'none', facetTris: 0, facetArea: 0 };
  } else if (state && progressive && !openMesh(mesh)) {
    const state0 = state;
    // each place that fails mended on its own (other edge methods, then facets); the planar and faceted tiers only
    // when no whole build of what it chose is a valid solid. The planar tier's facets: every patch but the planes.
    let planarArea = 0, totalArea = 0;
    for (let t = 0; t < mesh.T.length; t++) { totalArea += mesh.area[t]; if (state.surfaces.get(state.labels[t])?.[0] !== 'plane') planarArea += mesh.area[t]; }
    const planarShare = planarArea / Math.max(totalArea, 1e-300);
    const t = performance.now();
    // its trials stop where the whole-part retries did, a quarter of the deadline early, to leave its whole build time
    const recover = (maxFacetArea) => progressiveRecovery(oc, mesh, state0, shape, state0.surfaces, plainSurfs, info,
      end - (deadline === Infinity ? 0 : deadline / 4), onProgress, { methodsMs: 4 * (info.ms_clean ?? Infinity), maxFacetArea });
    let p = recover(planarShare);
    info.ms_progressive = Math.round(performance.now() - t);
    if (p.tier) { out = { shape: p.shape, sc: p.sc, tier: p.tier, facetTris: p.facetTris, facetArea: p.facetArea, demoted: p.demoted }; state = p.state; }
    // a hybrid with more of its area as facets than the planar tier would have is the worse export: planar is tried
    // too, and kept if it is recreated (00002311_011: 0.5% facets planar, 83% hybrid)
    if (p.tier === 'hybrid' && p.facetArea > planarShare + 1e-4) {
      const f = facetFallback(oc, mesh, shape, state0, info, end, { hybrid: false, faceted: false });
      if (f.tier === 'planar' && f.facetArea < p.facetArea) { out = f; state = state0; info.planar_beat_hybrid = 1; }
    }
    let planarTried = false;
    if (!p.tier && p.overPlanar) {
      // stopped where its facets passed the planar tier's: that tier, and if it is no solid, recovery without the bar
      const f = facetFallback(oc, mesh, shape, state0, info, end, { hybrid: false, faceted: false });
      planarTried = true;
      if (f.tier === 'planar') out = f;
      else {
        info.progressive_rerun = 1;
        p = recover(Infinity);
        if (p.tier) { out = { shape: p.shape, sc: p.sc, tier: p.tier, facetTris: p.facetTris, facetArea: p.facetArea, demoted: p.demoted }; state = p.state; }
      }
    }
    // nothing recovered: the hybrid tier as before (every flagged patch a facet at once, grown by whole builds;
    // 00143363_15 was 27% facets there and all facets after recovery), then planar unless tried, then the mesh
    if (!p.tier && !out) { info.progressive_failed = 1; out = facetFallback(oc, mesh, shape, state0, info, end, { planar: !planarTried }); }
  } else if (state) {
    out = facetFallback(oc, mesh, shape, state, info, end);
  } else if (fallback) {
    // The analytic build threw: the mesh itself is the only tier left.
    const { shape: sh } = buildPolyhedral(oc, mesh);
    const { g, sc } = grade(oc, sh, mesh, {});
    out = { shape: sh, sc, tier: g[0] ? 'faceted' : 'none', bestTier: 'faceted', facetTris: mesh.T.length, facetArea: 1 };
  } else {
    out = { shape: null, sc: {}, tier: 'none', facetTris: 0, facetArea: 0 };
  }
  return finish(out);

  function finish(out) {
  const { sc } = out;
  // slivers out, one face per surface (build.js tidy); a faceted solid is left as it is, and a hybrid or planar one
  // keeps its small faces (they are facets)
  if (tidyUp && sc.solid && sc.closed && sc.valid && out.tier !== 'faceted' && out.tier !== 'none') {
    const t1 = performance.now();
    sc.solid = tidy(oc, sc.solid, mesh, info, { smallFaces: out.tier === 'analytic' });
    info.tidy_ms = Math.round(performance.now() - t1);
  }
  const report = {
    tier: out.tier, bestTier: out.bestTier ?? out.tier, facetTris: out.facetTris, facetArea: out.facetArea,
    closed: !!sc.closed, valid: !!sc.valid, vol_err: sc.vol_err ?? null, free_edges: sc.free_edges ?? null,
    nonmanifold_edges: sc.nonmanifold_edges ?? null, shells: sc.shells ?? null,
    groups: new Set(mesh.F).size, meshPieces: mesh.nBodies, triangles: mesh.T.length,
    secs: Math.round((performance.now() - t0) / 100) / 10, info,
    faces: { ...faceOutcomes(labels, surfaces, state, out), pieces: pieces ? [...pieces] : [] },
  };
  return { solid: sc.solid ?? out.shape, report };
  }
}

/**
 * What became of each painted group, for the studio's conversion report (js/studio/report.js): the kernel's
 * complaints about each patch in the best analytic build, by kind (missing, invalid, orient, area, open);
 * the repair loop's moves; the groups it absorbed into neighbours or that ended on another surface type
 * (repair or shared axes); each group's final surface type; and the groups the facet tier built as facets
 * ('all' when the result is the mesh itself). Group ids are the labels the knitter was given; a freeform
 * group split into pieces adds labels above them (knitSolid's report.faces.pieces: [piece, group]).
 */
const TIERS = ['analytic', 'hybrid', 'planar', 'faceted', 'none'];

/** A body's line in report.bodies. */
function bodyRow(r) {
  return { tier: r.tier, bestTier: r.bestTier, facetTris: r.facetTris, facetArea: r.facetArea, closed: r.closed, valid: r.valid,
    vol_err: r.vol_err, triangles: r.triangles, groups: r.groups, secs: r.secs };
}

/** The solids of a shape, in its order. */
function solidsIn(oc, shape) {
  const out = [];
  if (!shape) return out;
  for (const ex = new oc.TopExp_Explorer_2(shape, oc.TopAbs_ShapeEnum.TopAbs_SOLID, oc.TopAbs_ShapeEnum.TopAbs_SHAPE); ex.More(); ex.Next()) out.push(ex.Current());
  return out;
}

/** A name per solid of `shape` for one body's: its name, numbered when it came out as several solids. */
function solidNames(oc, shape, [name]) {
  const n = solidsIn(oc, shape).length;
  return n > 1 ? Array.from({ length: n }, (_, k) => `${name} ${k + 1}`) : [name];
}

/**
 * MULTIBODY (payload bodies: a body id per triangle, a void's the body around it): each body built and judged on its
 * own (knitOne), so one body's failure or time sends no other to facets, with its share of what is left of the
 * deadline by triangles. The solid is one compound of every body's solids, names one per solid (the payload's, else
 * Body 1..n) for writeStep; report.bodies a line per body, report.tier the worst. onBody({id, name, solid, report})
 * as each is done (the export worker posts it, so a hard stop keeps the finished ones); keep: Map id -> {brep,
 * report} of bodies already built (the worker's earlier run), read instead of built.
 */
function knitBodies(oc, { positions, labels, surfaces, bodies, names, oriented, ids }, options) {
  const { deadline = Infinity, onProgress = null, onBody = null, keep = null } = options;
  const t0 = performance.now();
  const end = now() + deadline;
  const surfMap = surfaces instanceof Map ? surfaces : new Map(Object.entries(surfaces).map(([k, v]) => [Number(k), v]));
  const trisOf = new Map(ids.map((id) => [id, []]));
  for (let t = 0; t < labels.length; t++) trisOf.get(bodies[t]).push(t);
  let next = 0;
  for (const l of labels) next = Math.max(next, l + 1);
  let left = labels.length;
  const builder = new oc.BRep_Builder(), comp = new oc.TopoDS_Compound();
  builder.MakeCompound(comp);
  const rows = [], reports = [], solidNamesOut = [], areas = [];
  ids.forEach((id, i) => {
    const tris = trisOf.get(id), name = names?.[id] ?? `Body ${i + 1}`;
    const pos = new Float64Array(tris.length * 9), lab = new Int32Array(tris.length);
    tris.forEach((t, k) => { for (let j = 0; j < 9; j++) pos[k * 9 + j] = positions[t * 9 + j]; lab[k] = labels[t]; });
    let area = 0;
    for (let k = 0; k < tris.length; k++) {
      const o = k * 9, u = [pos[o + 3] - pos[o], pos[o + 4] - pos[o + 1], pos[o + 5] - pos[o + 2]], v = [pos[o + 6] - pos[o], pos[o + 7] - pos[o + 1], pos[o + 8] - pos[o + 2]];
      area += 0.5 * Math.hypot(u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]);
    }
    areas.push(area);
    let solid, report;
    const kept = keep?.get(id);
    if (kept) {
      solid = readBrep(oc, kept.brep);
      report = kept.report;
    } else {
      const present = new Set(lab);
      const sf = new Map([...surfMap].filter(([g]) => present.has(g)));
      const share = deadline === Infinity ? Infinity : Math.max(0, end - now()) * tris.length / Math.max(left, 1);
      ({ solid, report } = knitOne(oc, { positions: pos, labels: lab, surfaces: sf, oriented },
        { ...options, deadline: share, firstLabel: next, onProgress: onProgress && ((p) => onProgress({ ...p, body: id })) }));
      onBody?.({ id, name, solid, report });
    }
    for (const [piece] of report.faces?.pieces ?? []) next = Math.max(next, piece + 1);
    left -= tris.length;
    reports.push(report);
    rows.push({ id, name, ...bodyRow(report) });
    const ss = solidsIn(oc, solid);
    if (!ss.length && solid) builder.Add(comp, solid);           // no solid (tier none): the best attempt, unnamed
    ss.forEach((sh, k) => { builder.Add(comp, sh); solidNamesOut.push(ss.length > 1 ? `${name} ${k + 1}` : name); });
  });
  const report = mergeReports(reports, rows, areas, trisOf, ids, labels);
  report.secs = Math.round((performance.now() - t0) / 100) / 10;
  return { solid: comp, report, names: solidNamesOut };
}

/** knitBodies' report: the bodies' summed, report.tier the worst, report.bodies a line each. */
function mergeReports(reports, rows, areas, trisOf, ids, labels) {
  const worst = (k) => TIERS[Math.max(...reports.map((r) => TIERS.indexOf(r[k] ?? r.tier)))];
  const sum = (k) => reports.reduce((s, r) => s + (r[k] ?? 0), 0);
  const total = areas.reduce((s, a) => s + a, 0);
  const info = { bodies: reports.length };
  for (const r of reports) for (const [k, v] of Object.entries(r.info ?? {})) {
    if (typeof v === 'number') info[k] = (info[k] ?? 0) + v;
    else if (!(k in info)) info[k] = v;
  }
  const faces = { problems: {}, moves: [], absorbed: [], retyped: [], final: [], demoted: [], pieces: [] };
  const allFaceted = reports.every((r) => r.faces?.demoted === 'all');
  reports.forEach((r, i) => {
    const f = r.faces ?? {};
    for (const [k, v] of Object.entries(f.problems ?? {})) faces.problems[k] = [...(faces.problems[k] ?? []), ...v];
    for (const k of ['moves', 'absorbed', 'retyped', 'final', 'pieces']) faces[k].push(...(f[k] ?? []));
    if (!allFaceted) faces.demoted.push(...(f.demoted === 'all' ? [...new Set(trisOf.get(ids[i]).map((t) => labels[t]))] : f.demoted ?? []));
  });
  if (allFaceted) faces.demoted = 'all';
  return {
    tier: worst('tier'), bestTier: worst('bestTier'), facetTris: sum('facetTris'),
    facetArea: Math.round((reports.reduce((s, r, i) => s + (r.facetArea ?? 0) * areas[i], 0) / Math.max(total, 1e-300)) * 1e4) / 1e4,
    closed: reports.every((r) => r.closed), valid: reports.every((r) => r.valid),
    vol_err: Math.max(...reports.map((r) => r.vol_err ?? 0)), free_edges: sum('free_edges'), nonmanifold_edges: sum('nonmanifold_edges'),
    shells: sum('shells'), groups: new Set(labels).size, meshPieces: sum('meshPieces'), triangles: sum('triangles'),
    info, faces, bodies: rows,
  };
}

function faceOutcomes(labels, surfaces, state, out) {
  const ids = (s) => [...s].sort((a, b) => a - b);
  const faces = {
    problems: {}, moves: state?.moves ?? [], absorbed: [], retyped: [], final: [],
    demoted: (out.bestTier ?? out.tier) === 'faceted' ? 'all' : ids(out.demoted ?? []),
  };
  if (!state) return faces;
  for (const [k, v] of Object.entries(state.problems ?? {})) if (v.size) faces.problems[k] = ids(v);
  const after = new Set(state.labels);
  faces.absorbed = ids(new Set([...labels].filter((g) => g >= 0 && !after.has(g))));
  const before = surfaces instanceof Map ? surfaces : new Map(Object.entries(surfaces).map(([k, v]) => [Number(k), v]));
  for (const [g, s] of [...state.surfaces].sort((a, b) => a[0] - b[0])) {
    faces.final.push([g, s[0]]);
    const was = before.get(g)?.[0];
    if (was && was !== s[0]) faces.retyped.push([g, was, s[0]]);
  }
  return faces;
}

/**
 * Whether the mesh has a boundary (a triangle side with no partner): no build of it is a closed solid, so progressive
 * recovery cannot find one either (all 7 of the 275 Onshape parts that are never recreated, 2 to 6,118 open sides).
 */
function openMesh(mesh) {
  for (let t = 0; t < mesh.T.length; t++) for (let k = 0; k < 3; k++) if (mesh.across(t, mesh.T[t][k], mesh.T[t][(k + 1) % 3]) < 0) return true;
  return false;
}

/** The labels addFreeform made (a split group's later pieces and its unfitted rest): Map label -> its group. */
function piecesOf(before, after) {
  const out = new Map();
  for (let t = 0; t < after.length; t++) if (after[t] !== before[t]) out.set(after[t], before[t]);
  return out;
}

/**
 * The studio's Repair: the export's build (freeform fits, shared axes), then the repair loop (repair.js
 * buildRepaired) until the part is a closed, valid solid of the mesh's volume or the builds or the budget run
 * out. No facet tiers. Each move kept goes to onMove in the input's terms, for the caller to apply as an edit:
 *   {move: 'retype', group, from, to, params, changes}
 *   {move: 'absorb', group, changes} and {move: 'absorb_slivers', groups, changes}
 * changes: [input triangle, its new group, ...]. Groups are the input's labels, except that a freeform piece
 * (addFreeform's split) that is retyped, or whose group is, becomes a group of its own under its piece label:
 * the export then fits each group as the loop built it. Facets moved onto a piece stay in its group.
 * onProgress gets knitSolid's progress, with `trying` (the move about to be built) in the same terms.
 * Returns {ok, labels (per input triangle), surfaces (Map group -> [type, params]: the input's, retypes
 * applied, emptied groups gone), moves, info, problems}.
 */
export function repairGroups(oc, { positions, labels, surfaces, bodies = null, oriented = false }, options = {}) {
  const ids = bodies ? [...new Set(bodies)].sort((a, b) => a - b) : [];
  if (ids.length <= 1) return repairOne(oc, { positions, labels, surfaces, oriented }, options);
  return repairBodies(oc, { positions, labels, surfaces, bodies, oriented, ids }, options);
}

/**
 * Repair BODY BY BODY (the default with several bodies; bodies merge only when the user joins them): each body's
 * loop alone, with its share of what is left of the budget by triangles, so no move reaches across bodies. A group
 * painted across bodies is that group in the first body holding it and a group of its own in each other (a fresh
 * label), split off in the studio only when a retype would otherwise reach across: a retype of a fresh part carries
 * its triangles (a label the studio does not know is a new group), and before a retype of the original group each
 * part still in it is split off by a retype to the type it has ({move: 'retype', split: true, group, from, to,
 * params, changes}). Absorbs move triangles by name and need no split. Moves, labels and problems come back in the
 * input's terms; onProgress gets body: id.
 */
function repairBodies(oc, { positions, labels, surfaces, bodies, oriented, ids }, options) {
  const { budget = LOOP.BUDGET_S, onProgress = null, onMove = null } = options;
  const t0 = performance.now();
  const surfMap = surfaces instanceof Map ? surfaces : new Map(Object.entries(surfaces).map(([k, v]) => [Number(k), v]));
  const trisOf = new Map(ids.map((id) => [id, []]));
  for (let t = 0; t < labels.length; t++) trisOf.get(bodies[t]).push(t);
  let next = 0;
  for (const l of labels) next = Math.max(next, l + 1);
  for (const g of surfMap.keys()) next = Math.max(next, g + 1);
  // groups painted across bodies: a fresh label in every body after the first that holds them
  const firstBody = new Map(), fresh = new Map();          // fresh: `${body}|${group}` -> label
  const runLabels = Int32Array.from(labels), freshTris = new Map(), freshOf = new Map();   // label -> [input tris], label -> group
  for (const id of ids) for (const t of trisOf.get(id)) {
    const g = labels[t];
    if (g < 0) continue;
    if (!firstBody.has(g)) firstBody.set(g, id);
    if (firstBody.get(g) === id) continue;
    const key = `${id}|${g}`;
    if (!fresh.has(key)) { fresh.set(key, next); freshOf.set(next, g); freshTris.set(next, []); next++; }
    runLabels[t] = fresh.get(key);
    freshTris.get(fresh.get(key)).push(t);
  }
  const shown = new Set();                                // fresh labels the studio has been told about
  const out = { labels: Int32Array.from(runLabels), surfaces: new Map(surfMap) };
  for (const [f, g] of freshOf) if (surfMap.has(g)) out.surfaces.set(f, surfMap.get(g));
  const moves = [], problems = {}, info = { bodies: ids.length }, rows = [];
  let ok = true, left = labels.length;
  for (const id of ids) {
    const tris = trisOf.get(id);
    const pos = new Float64Array(tris.length * 9), lab = new Int32Array(tris.length);
    tris.forEach((t, k) => { for (let j = 0; j < 9; j++) pos[k * 9 + j] = positions[t * 9 + j]; lab[k] = runLabels[t]; });
    const present = new Set(lab);
    const sf = new Map([...out.surfaces].filter(([g]) => present.has(g)));
    const share = Math.max(0, budget - (performance.now() - t0) / 1000) * tris.length / Math.max(left, 1);
    const emit = (m) => { for (let i = 0; i < m.changes.length; i += 2) out.labels[m.changes[i]] = m.changes[i + 1]; onMove?.(m); };
    const splitOff = (f) => {
      shown.add(f);
      const changes = [];
      for (const t of freshTris.get(f)) if (out.labels[t] === freshOf.get(f) || out.labels[t] === f) changes.push(t, f);
      return changes;
    };
    const onBodyMove = (m) => {
      const changes = [];
      if (m.move === 'retype' && freshTris.has(m.group) && !shown.has(m.group)) changes.push(...splitOff(m.group));
      if (m.move === 'retype' && !freshTris.has(m.group)) {
        for (const [f, g] of freshOf) {
          if (g !== m.group || shown.has(f)) continue;
          const [type, params] = out.surfaces.get(f) ?? out.surfaces.get(g) ?? [];
          emit({ move: 'retype', split: true, group: f, from: type, to: type, params, changes: splitOff(f) });
        }
      }
      for (let i = 0; i < m.changes.length; i += 2) changes.push(tris[m.changes[i]], m.changes[i + 1]);
      emit({ ...m, changes });
    };
    const r = repairOne(oc, { positions: pos, labels: lab, surfaces: sf, oriented }, { ...options, budget: share, firstLabel: next,
      onProgress: onProgress && ((p) => onProgress({ ...p, body: id })), onMove: onBodyMove });
    tris.forEach((t, k) => { out.labels[t] = r.labels[k]; next = Math.max(next, r.labels[k] + 1); });
    for (const g of sf.keys()) if (!r.surfaces.has(g)) out.surfaces.delete(g);
    for (const [g, v] of r.surfaces) { out.surfaces.set(g, v); next = Math.max(next, g + 1); }
    moves.push(...r.moves);
    for (const [k, v] of Object.entries(r.problems)) problems[k] = [...new Set([...(problems[k] ?? []), ...v])].sort((a, b) => a - b);
    for (const [k, v] of Object.entries(r.info)) if (typeof v === 'number') info[k] = (info[k] ?? 0) + v;
    ok &&= r.ok;
    left -= tris.length;
    rows.push({ id, ok: r.ok, moves: r.moves.length, secs: r.info.secs });
  }
  // a fresh label no move touched is still its group
  for (const [f, g] of freshOf) if (!shown.has(f)) {
    for (const t of freshTris.get(f)) if (out.labels[t] === f) out.labels[t] = g;
    out.surfaces.delete(f);
  }
  info.secs = Math.round((performance.now() - t0) / 100) / 10;
  return { ok, labels: out.labels, surfaces: out.surfaces, moves, info, problems, bodies: rows };
}

/** One body's repair (or the whole mesh's, as before bodies): repairGroups' loop. */
function repairOne(oc, { positions, labels, surfaces, oriented = false }, { budget = LOOP.BUDGET_S, maxBuilds = LOOP.MAX_BUILDS,
  onProgress = null, onMove = null, split = true, axes = true, axesOptions = {}, firstLabel = 0 } = {}) {
  const t0 = performance.now();
  const mesh = new KnitMesh(positions, labels, { oriented });          // voids as voids
  const surfs = surfaces instanceof Map ? new Map(surfaces) : new Map(Object.entries(surfaces).map(([k, v]) => [Number(k), v]));
  const out = { labels: Int32Array.from(labels), surfaces: new Map(surfs) };
  const info = {};
  const unsplit = Int32Array.from(mesh.F);
  Object.assign(info, addFreeform(mesh, surfs, onProgress, { split, firstLabel }));
  const pieces = piecesOf(unsplit, mesh.F);
  // Every build gets its shared axes from the groups and fits as they stand, as the export will make them:
  // the loop used to build a retype on its own fit, which the export never would (00000270_015)
  const prepare = axes ? (lab, sf) => {
    const r = shareAxes(mesh.withLabels(lab), sf, axesOptions);
    if (!info.axes_surfaces) Object.assign(info, r.info);
    return r.surfaces;
  } : null;
  const own = new Set();                         // pieces that are groups of their own
  const groupOf = (l) => (pieces.has(l) && !own.has(l) ? pieces.get(l) : l);
  const kept = (m) => {
    if (m.move === 'retype') {
      if (pieces.has(m.group)) own.add(m.group);
      else for (const [p, g] of pieces) if (g === m.group) own.add(p);
    }
    const changes = [];
    for (let t = 0; t < m.labels.length; t++) {
      const s = mesh.src[t], g = groupOf(m.labels[t]);
      if (out.labels[s] !== g) { out.labels[s] = g; changes.push(s, g); }
    }
    const move = { move: m.move, changes };
    if (m.move === 'retype') {
      const params = m.surfaces.get(m.group)[1];
      out.surfaces.set(m.group, [m.to, params]);
      Object.assign(move, { group: m.group, from: m.from, to: m.to, params });
    } else if (m.move === 'absorb') move.group = groupOf(m.group);
    else move.groups = [...new Set(m.groups.map(groupOf))];
    const present = new Set(out.labels);
    for (const g of [...out.surfaces.keys()]) if (!present.has(g)) out.surfaces.delete(g);
    onMove?.(move);
  };
  const progress = onProgress && ((p) => {
    const trying = p.trying && { ...p.trying, ...(p.trying.group !== undefined ? { group: groupOf(p.trying.group) } : {}),
      ...(p.trying.groups ? { groups: [...new Set(p.trying.groups.map(groupOf))] } : {}) };
    onProgress(trying ? { ...p, trying } : p);
  });
  const r = buildRepaired(oc, mesh, surfs, { fitCandidates: fitCandidatesFor(positions), maxBuilds, budget,
    onProgress: progress, onMove: kept, prepare });
  Object.assign(info, r.info, { secs: Math.round((performance.now() - t0) / 100) / 10 });
  const problems = {};
  for (const [k, v] of Object.entries(r.state.problems ?? {})) if (v.size) problems[k] = [...new Set([...v].map(groupOf))].sort((a, b) => a - b);
  const moves = r.state.moves.map((m) => ({ ...m, ...(m.group !== undefined ? { group: groupOf(m.group) } : {}),
    ...(m.groups ? { groups: [...new Set(m.groups.map(groupOf))] } : {}) }));
  return { ok: r.state.ok, labels: out.labels, surfaces: out.surfaces, moves, info, problems };
}
