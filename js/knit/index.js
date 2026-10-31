/**
 * knit/index.js — a STEP solid from painted groups: the topo knitter (ml/solid_rebuild.py's method,
 * ported), its kernel-judged repair loop, and the facet fallback, with an honest account of what came out.
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
import { buildRepaired, facetFallback, LOOP, now } from './repair.js';
import { buildPolyhedral, grade, tidy, better, BUILD } from './build.js';
import { buildFacets } from '../meshCore.js';
import { attachCurvature } from '../curvature.js';
import { fitRegionAs } from '../gaussMap.js';
import { fitPieces } from '../freeformFit.js';
import { shareAxes } from './axes.js';

export { loadOcct, writeStep } from './occt.js';
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
export function addFreeform(mesh, surfaces, onProgress = null, { split = true } = {}) {
  const byLabel = new Map();
  let next = 0;
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
 * no surface get a fitted B-spline first (addFreeform). `deadline` (s): past
 * it no new repair build or facet tier starts (a running OCCT call cannot be stopped, so callers that
 * must bound the time also need a hard stop, as the studio's worker timeout). mode 'faceted': the mesh
 * as a polyhedral solid only.
 */
export function knitSolid(oc, { positions, labels, surfaces }, { loop = true, fallback = true, budget = LOOP.BUDGET_S,
  maxBuilds = LOOP.MAX_BUILDS, onProgress = null, deadline = Infinity, mode = 'knit', freeform = true, split = true,
  axes = true, axesOptions = {}, tidy: tidyUp = true, retryPlain = true } = {}) {
  const t0 = performance.now();
  const end = now() + deadline;                  // past it: no new repair builds or facet tiers
  const mesh = new KnitMesh(positions, labels);
  let surfs = surfaces instanceof Map ? surfaces : new Map(Object.entries(surfaces).map(([k, v]) => [Number(k), v]));
  const info = {};
  let shape = null, state = null, out;
  if (mode === 'faceted') {
    // The mesh as a solid and nothing else: the studio's answer when the analytic build ran out of time.
    const { shape: sh } = buildPolyhedral(oc, mesh);
    const { g, sc } = grade(oc, sh, mesh, {});
    out = { shape: sh, sc, tier: g[0] ? 'faceted' : 'none', bestTier: 'faceted', facetTris: mesh.T.length, facetArea: 1 };
    return finish(out);
  }
  budget = Math.min(budget, deadline / 2);
  if (freeform) Object.assign(info, addFreeform(mesh, surfs, onProgress, { split }));
  const plainSurfs = new Map(surfs);              // the fits as they came, for the plain retry
  // shared axes and directions made exact (knit/axes.js): coaxial faces meet without a kink
  if (axes) {
    const ta = performance.now();
    const r = shareAxes(mesh, surfs, axesOptions);
    surfs = r.surfaces;
    Object.assign(info, r.info, { axes_ms: Math.round(performance.now() - ta) });
  }
  const attempt = (sf) => {
    try {
      return buildRepaired(oc, mesh, sf, { fitCandidates: loop ? fitCandidatesFor(positions) : null,
        maxBuilds: loop ? maxBuilds : 1, budget, onProgress });
    } catch (err) {
      info.exception = String(err?.message ?? err).slice(0, 120);
      if (globalThis.process?.env?.KNIT_DEBUG) info.exception_stack = String(err?.stack ?? '').split(/\r?\n/).slice(0, 8).join(' | ');
      return null;
    }
  };
  let r = attempt(surfs);
  // The clean build (shared axes, exact edges, contact loci) is not analytic everywhere the plain one is:
  // on 274 Onshape parts each was fully analytic on parts the other was not (deterministic gates, no
  // time budget: clean 250, plain 254, 8 one way and 5 the other). So a clean build that is not fully
  // analytic is retried plain, time permitting, and the plain one is kept if it is.
  const clean = axes || BUILD.EXACT_EDGES || BUILD.CONTACT;
  if (clean && retryPlain && !(r?.state?.ok) && now() < end - (deadline === Infinity ? 0 : deadline / 4)) {
    const saved = { ...BUILD };
    BUILD.EXACT_EDGES = false; BUILD.CONTACT = false;
    let r2 = null;
    try { r2 = attempt(plainSurfs); } finally { Object.assign(BUILD, saved); }
    info.plain_retry = 1;
    // the plain build replaces the clean one only when it is fully analytic: otherwise the clean one goes on
    // to the facet tiers (a plain build that grades better but is not analytic fell to facets where the
    // clean one made a hybrid, 00005406_001)
    if (r2?.state?.ok || (!r?.state && r2?.state)) { r = r2; info.plain_retry_won = 1; }
  }
  if (r) { shape = r.shape; state = r.state; Object.assign(info, r.info); }
  if (state && (state.ok || !fallback)) {
    out = { shape, sc: state.sc, tier: state.ok ? 'analytic' : 'none', facetTris: 0, facetArea: 0 };
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
  // slivers out, one face per surface (build.js tidy); a faceted solid is left as it is
  if (tidyUp && sc.solid && sc.closed && sc.valid && out.tier !== 'faceted' && out.tier !== 'none') {
    const t1 = performance.now();
    sc.solid = tidy(oc, sc.solid, mesh, info);
    info.tidy_ms = Math.round(performance.now() - t1);
  }
  const report = {
    tier: out.tier, bestTier: out.bestTier ?? out.tier, facetTris: out.facetTris, facetArea: out.facetArea,
    closed: !!sc.closed, valid: !!sc.valid, vol_err: sc.vol_err ?? null, free_edges: sc.free_edges ?? null,
    nonmanifold_edges: sc.nonmanifold_edges ?? null, shells: sc.shells ?? null,
    groups: new Set(mesh.F).size, bodies: mesh.nBodies, triangles: mesh.T.length,
    secs: Math.round((performance.now() - t0) / 100) / 10, info,
    faces: faceOutcomes(labels, surfaces, state, out),
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
 * group split into pieces adds labels above them.
 */
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
