/**
 * knit/repair.js — the kernel-judged repair loop and the facet fallback, a port of ml/solid_rebuild.py
 * build_repaired / facet_fallback (measured there: Onshape auto 211 -> 247 of 275 with the loop, then
 * 269 with the fallback; 2026-09-27 and 2026-10-02).
 *
 * The loop: while the part is not a closed, valid solid of the mesh's volume, find the failing patches,
 * try ONE move (absorb the group into the neighbours it lies on, or its next surface type from the
 * fitters), rebuild, keep it only if the part got better. Inference-time signals only. The export never
 * runs it: the studio's Repair does (knit/index.js repairGroups), and each move it keeps becomes an edit
 * the user sees and can undo, so the export builds only what the view shows.
 *
 * The export's own recovery, when its build is not recreated: progressiveRecovery (below) mends each place that
 * fails on its own, other edge methods there, then B-splines fitted to the failing patches (SPLINE), then facets,
 * and only then the facet tiers:
 *   hybrid   the patches still failing (and surfaceless ones) as facets, the rest analytic (knitSolid's
 *            progressive false only: progressiveRecovery makes its hybrids place by place)
 *   planar   only plane patches analytic
 *   faceted  the mesh as a polyhedral solid
 */

import { buildTopo, buildPolyhedral, patchProblems, regionProblems, grade, better, score, strayFaces, VARIANTS } from './build.js';
import { distNormal, unit, sub, cross } from './geometry.js';
import { meshTopology } from './mesh.js';
import { fitGroup } from '../freeformFit.js';

export const LOOP = {
  SMALL_GROUP: 3,            // facets; a failing group this small is absorbed, never refit
  TIE_RATIO: 1.5,            // fits within this factor of the best residual count as tied
  RETYPES: 2,                // alternatives tried per group before absorbing it
  MAX_BUILDS: 40,
  BUDGET_S: 300,
  MAX_RESIDUAL_RATIO: 10,    // a retype candidate fits within this factor of the best candidate
  ABSORB_TOL: 2e-3,          // x diagonal: a larger group's facets move only onto a surface this near
  EXACT: 1e-6,               // x diagonal: a group its own type fits this closely is never retyped (axes.js's bar)
};
const SIMPLER = { plane: 0, sphere: 1, cylinder: 2, cone: 3, torus: 4 };
const KINDS = ['missing', 'invalid', 'orient', 'area', 'open'];

const bump = (info, k, n = 1) => { info[k] = (info[k] ?? 0) + n; };
const mid = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now()) / 1000;

/** Least-squares plane through a group's vertices: the fallback surface without the fitters. */
function planeCandidate(mesh, labels, g) {
  const ids = new Set();
  for (let t = 0; t < labels.length; t++) if (labels[t] === g) for (const v of mesh.T[t]) ids.add(v);
  const P = [...ids].map((v) => mesh.V[v]);
  const c = [0, 0, 0];
  for (const q of P) for (let k = 0; k < 3; k++) c[k] += q[k] / P.length;
  let n;
  if (P.length === 3) n = unit(cross(sub(P[1], P[0]), sub(P[2], P[0])));
  else {
    // smallest eigenvector of the covariance, by inverse power iteration on (C + eps I)
    const C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (const q of P) { const d = sub(q, c); for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) C[i][j] += d[i] * d[j]; }
    const tr = C[0][0] + C[1][1] + C[2][2] || 1;
    for (let i = 0; i < 3; i++) C[i][i] += 1e-12 * tr;
    n = [1, 1, 1];
    for (let it = 0; it < 50; it++) n = unit(solve3(C, n));
  }
  let r = 0;
  for (const q of P) r = Math.max(r, Math.abs((q[0] - c[0]) * n[0] + (q[1] - c[1]) * n[1] + (q[2] - c[2]) * n[2]));
  return { type: 'plane', params: { origin: c, normal: n }, residual: r };
}

function solve3(A, b) {
  const det = (m) => m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  const D = det(A) || 1e-300;
  return [0, 1, 2].map((k) => det(A.map((row, i) => row.map((v, j) => (j === k ? b[i] : v)))) / D);
}

/**
 * Hand group g's facets to neighbouring groups, working in from its border: each facet goes to the
 * adjacent group (with a surface) nearest all three of its vertices; with `tol`, only within it.
 */
function absorbGroup(mesh, labels, surfaces, g, tol = null) {
  const { T, V } = mesh;
  const left = new Set();
  for (let t = 0; t < labels.length; t++) if (labels[t] === g) left.add(t);
  let moved = 0;
  while (left.size) {
    const assign = new Map();
    for (const t of left) {
      let best = null;
      for (let k = 0; k < 3; k++) {
        const n = mesh.across(t, T[t][k], T[t][(k + 1) % 3]);
        if (n < 0) continue;
        const q = labels[n];
        if (q === g || !surfaces.has(q)) continue;
        const [kind, p] = surfaces.get(q);
        const d = Math.max(...T[t].map((v) => Math.abs(distNormal(kind, p, V[v])[0])));
        if (!best || d < best[0]) best = [d, q];
      }
      if (best && (tol === null || best[0] <= tol)) assign.set(t, best[1]);
    }
    if (!assign.size) break;
    for (const [t, q] of assign) { labels[t] = q; left.delete(t); moved++; }
  }
  return moved;
}

/**
 * Candidate surfaces other than the current type: the simplest first among those tied with the best. None
 * for a group its current type fits exactly: the Manta report (2026-10-31) had exact planes retyped to
 * cylinders and exact cylinders to spheres, kept because the build improved (a band with vertices on its two
 * rims only lies exactly on a sphere too).
 */
function alternatives(cands, current, scaleLen) {
  if (current && cands.some((c) => c.params && c.type === current[0] && c.residual <= LOOP.EXACT * scaleLen)) return [];
  cands = cands.filter((c) => c.params && (!current || c.type !== current[0]));
  if (!cands.length) return [];
  const best = Math.min(...cands.map((c) => c.residual));
  const tied = cands.filter((c) => c.residual <= LOOP.TIE_RATIO * best + 1e-7 * scaleLen);
  const rest = cands.filter((c) => !tied.includes(c) && c.residual <= LOOP.MAX_RESIDUAL_RATIO * best + 1e-7 * scaleLen)
    .sort((a, b) => a.residual - b.residual);
  return [...tied.sort((a, b) => (SIMPLER[a.type] ?? 9) - (SIMPLER[b.type] ?? 9)), ...rest];
}

function countOf(labels, g) {
  let n = 0;
  for (let t = 0; t < labels.length; t++) if (labels[t] === g) n++;
  return n;
}

/**
 * buildTopo, then repair moves while the part is not a closed, valid solid of the mesh's volume (with
 * maxBuilds 1, the build alone). fitCandidates({group: soupTris}) -> {group: [{type, params, residual}]}.
 * onProgress({builds, ok, trying?}) before and after each build; onMove({move, ..., labels, surfaces}) after
 * each move kept, with the state it made. prepare(labels, surfaces) -> the surfaces each build gets (the
 * studio's Repair: shared axes, as the export will make them from the same groups); the state keeps the
 * surfaces before it. Returns {shape, info, state} with state = the best build's labels, surfaces, problems
 * and verdict (for facetFallback).
 */
export function buildRepaired(oc, mesh, surfaces, { fitCandidates = null, maxBuilds = LOOP.MAX_BUILDS,
  budget = LOOP.BUDGET_S, onProgress = null, onMove = null, prepare = null } = {}) {
  const t0 = now();
  const log = {};
  let slowest = 0;

  const build = (labels, surfs, first = false) => {
    const t = now();
    try {
      const work = mesh.withLabels(labels);
      const r = buildTopo(oc, work, prepare ? prepare(labels, surfs) : surfs);
      const checked = score(oc, r.shape, work);
      const probs = patchProblems(oc, r.shape, r.faceOf, labels, r.ratios, checked.analyzers, { offFaces: checked.offFaces });
      const { g, sc } = grade(oc, r.shape, work, probs, checked);
      return { g, shape: r.shape, info: r.info, probs, labels, surfs, sc };
    } catch (err) {
      // A move that makes OCCT throw is a rejected move, not a lost part.
      if (first) throw err;
      bump(log, 'trial_exception');
      return null;
    } finally {
      slowest = Math.max(slowest, now() - t);
    }
  };
  // a build is only started when the slowest one so far still fits in the budget
  const timeLeft = () => budget - (now() - t0) > slowest;

  let best = build(Int32Array.from(mesh.F), new Map(surfaces), true);
  let builds = 1;
  // every move tried and whether it was kept, for the conversion report (knitSolid's report.faces)
  const moves = [];
  onProgress?.({ builds, ok: !!best.g[0] });
  log.problems_before = Object.values(best.probs).reduce((s, v) => s + v.size, 0);
  const cands = new Map(), tried = new Set();

  const attempt = (lab, sf, name) => {
    const trial = build(lab, sf);
    builds++;
    bump(log, `${name}_tried`);
    onProgress?.({ builds, ok: !!best.g[0] });
    if (trial && better(trial.g, best.g)) { best = trial; bump(log, `${name}_kept`); return true; }
    return false;
  };

  // Untyped slivers first, all at once: a patch with no surface has no face, so moving its facets to a
  // neighbour cannot lose anything.
  if (!best.g[0] && builds < maxBuilds) {
    const lab = Int32Array.from(best.labels), sf = new Map(best.surfs);
    const slivers = [...best.probs.missing].filter((g) => !sf.has(g) && countOf(lab, g) <= LOOP.SMALL_GROUP);
    let movedAny = 0;
    for (const g of slivers) movedAny += absorbGroup(mesh, lab, sf, g);
    if (slivers.length && movedAny && timeLeft()) {
      onProgress?.({ builds, ok: false, trying: { move: 'absorb_slivers', groups: slivers } });
      const trial = build(lab, sf);
      builds++;
      bump(log, 'sliver_absorb_tried');
      const kept = !!trial && !better(best.g, trial.g);
      if (kept) { best = trial; bump(log, 'slivers_absorbed', slivers.length); }
      moves.push({ move: 'absorb_slivers', groups: slivers, kept });
      if (kept) onMove?.({ move: 'absorb_slivers', groups: slivers, labels: best.labels, surfaces: best.surfs });
    }
    for (const g of slivers) tried.add(`absorb|${g}`);
  }

  while (builds < maxBuilds && timeLeft() && !best.g[0]) {
    const { probs, labels: lab0, surfs: sf0 } = best;
    const order = [];
    for (const k of KINDS) for (const g of [...probs[k]].sort((a, b) => a - b)) order.push(g);
    if (!order.length) break;
    const need = order.filter((g) => !cands.has(g) && countOf(lab0, g) > LOOP.SMALL_GROUP);
    if (need.length) {
      const req = {};
      for (const g of need) {
        req[g] = [];
        for (let t = 0; t < lab0.length; t++) if (lab0[t] === g) req[g].push(mesh.src[t]);
      }
      const got = fitCandidates ? fitCandidates(req) : {};
      for (const g of need) cands.set(g, got[g]?.length ? got[g] : [planeCandidate(mesh, lab0, g)]);
    }
    let move = null;
    for (const g of order) {
      if (countOf(lab0, g) > LOOP.SMALL_GROUP) {
        const alts = alternatives(cands.get(g) ?? [], sf0.get(g), mesh.diag).slice(0, LOOP.RETYPES);
        for (let i = 0; i < alts.length; i++) {
          if (!tried.has(`retype|${g}|${i}`)) { move = ['retype', g, i, alts[i]]; break; }
        }
      }
      if (!move && !tried.has(`absorb|${g}`)) move = ['absorb', g];
      if (move) break;
    }
    if (!move) break;
    tried.add(move[0] === 'retype' ? `retype|${move[1]}|${move[2]}` : `absorb|${move[1]}`);
    const lab = Int32Array.from(lab0), sf = new Map(sf0);
    if (move[0] === 'retype') {
      sf.set(move[1], [move[3].type, move[3].params]);
    } else {
      const big = countOf(lab0, move[1]) > LOOP.SMALL_GROUP;
      if (!absorbGroup(mesh, lab, sf, move[1], big ? LOOP.ABSORB_TOL * mesh.diag : null)) continue;
      if (!lab.includes(move[1])) sf.delete(move[1]);     // fully absorbed; stranded facets keep it
    }
    const record = move[0] === 'retype'
      ? { move: 'retype', group: move[1], from: sf0.get(move[1])?.[0] ?? null, to: move[3].type }
      : { move: 'absorb', group: move[1] };
    onProgress?.({ builds, ok: false, trying: record });
    const kept = attempt(lab, sf, move[0]);
    moves.push({ ...record, kept });
    if (kept) onMove?.({ ...record, labels: best.labels, surfaces: best.surfs });
  }
  const info = { ...best.info };
  for (const [k, v] of Object.entries(log)) info[`repair_${k}`] = v;
  info.repair_builds = builds;
  info.repair_problems_after = Object.values(best.probs).reduce((s, v) => s + v.size, 0);
  for (const [k, v] of Object.entries(best.probs)) if (v.size) info[`repair_left_${k}`] = v.size;
  info.repair_secs = Math.round((now() - t0) * 10) / 10;
  return { shape: best.shape, info, state: { labels: best.labels, surfaces: best.surfs, problems: best.probs, ok: !!best.g[0], g: best.g, sc: best.sc, moves } };
}

export const FALLBACK = { ROUNDS: 3, MAX_FACETS: 6000 };
export { now };

/** buildTopo with every triangle of the patches in `demote` a facet face of its own. */
function buildWithFacets(oc, mesh, labels, surfaces, demote) {
  const lab = Int32Array.from(labels);
  let base = -Infinity;
  for (const l of lab) base = Math.max(base, l);
  base += 1;
  const facetLabels = new Set();
  let next = base;
  for (let t = 0; t < lab.length; t++) {
    if (demote.has(lab[t])) { lab[t] = next; facetLabels.add(next); next++; }
  }
  const sf = new Map([...surfaces].filter(([g]) => !demote.has(g)));
  const work = mesh.withLabels(lab);
  const r = buildTopo(oc, work, sf, { facetLabels });
  const checked = score(oc, r.shape, work);
  const probs = patchProblems(oc, r.shape, r.faceOf, lab, r.ratios, checked.analyzers, { offFaces: checked.offFaces });
  const { g, sc } = grade(oc, r.shape, work, probs, checked);
  return { g, sc, shape: r.shape, probs, base };
}

/**
 * The first fallback tier that recreates the part, else the best-graded of them all. `state` is
 * buildRepaired's; past `deadline` (seconds, performance.now() / 1000) the hybrid and planar tiers are
 * skipped and only the polyhedral solid is tried. Returns {shape, sc, tier, facetTris, facetArea}: tier is analytic, hybrid, planar,
 * faceted, or none (no tier recreated the part; the best of them is still returned).
 */
export function facetFallback(oc, mesh, shape, state, info = {}, deadline = Infinity, { hybrid = true, planar = true, faceted = true } = {}) {
  const t0 = now();
  const { labels, surfaces } = state;
  const present = new Set(labels);
  let totalArea = 0;
  for (const a of mesh.area) totalArea += a;
  const share = (demote) => {
    let n = 0, a = 0;
    for (let t = 0; t < labels.length; t++) if (demote.has(labels[t])) { n++; a += mesh.area[t]; }
    return [n, a / Math.max(totalArea, 1e-300)];
  };
  if (state.ok) return { shape, sc: state.sc, tier: 'analytic', facetTris: 0, facetArea: 0 };
  let best = { g: state.g, shape, sc: state.sc, tier: 'analytic', n: 0, area: 0 };
  const tried = new Set();
  const attempt = (tier, demote) => {
    const key = [...demote].sort((a, b) => a - b).join(',');
    if (!demote.size || tried.has(key)) return null;
    // Out of time: the facet tiers run the full repair over thousands of faces; the polyhedral solid does not.
    if (now() > deadline) { bump(info, 'fallback_skipped_for_time'); return null; }
    tried.add(key);
    const [n, area] = share(demote);
    bump(info, `fallback_${tier}_tried`);
    if (n > FALLBACK.MAX_FACETS) { bump(info, 'fallback_too_many_facets'); return null; }
    let r;
    try {
      r = buildWithFacets(oc, mesh, labels, surfaces, demote);
    } catch {
      bump(info, 'fallback_exception');
      return null;
    }
    if (better(r.g, best.g)) best = { g: r.g, shape: r.shape, sc: r.sc, tier, n, area, demote: new Set(demote) };
    return r;
  };
  const demote = new Set();
  for (const k of KINDS) for (const g of state.problems[k] ?? []) if (present.has(g)) demote.add(g);
  for (const g of present) if (!surfaces.has(g)) demote.add(g);
  for (let round = 0; hybrid && round < FALLBACK.ROUNDS; round++) {
    const r = attempt('hybrid', demote);
    if (!r || r.g[0]) break;
    let grew = false;
    for (const k of KINDS) for (const x of r.probs[k]) if (x < r.base && !demote.has(x)) { demote.add(x); grew = true; }
    if (!grew) break;
  }
  if (!best.g[0] && planar) attempt('planar', new Set([...present].filter((g) => surfaces.get(g)?.[0] !== 'plane')));
  if (!best.g[0] && faceted) {
    bump(info, 'fallback_faceted_tried');
    try {
      const { shape: sh } = buildPolyhedral(oc, mesh);
      const { g, sc } = grade(oc, sh, mesh, {});
      if (better(g, best.g)) best = { g, shape: sh, sc, tier: 'faceted', n: mesh.T.length, area: 1 };
    } catch (err) {
      bump(info, err?.code === 'too_many_facets' ? 'faceted_too_many_triangles' : 'fallback_exception');
    }
  }
  info.fallback_secs = Math.round((now() - t0) * 10) / 10;
  return { shape: best.shape, sc: best.sc, tier: best.g[0] ? best.tier : 'none', bestTier: best.tier,
           facetTris: best.n, facetArea: Math.round(best.area * 1e4) / 1e4, demoted: [...(best.demote ?? [])] };
}

// ── progressive recovery ─────────────────────────────────────────────────────────────────────────────
//
// The part is judged as a whole, but it fails in places: the median failing Onshape part flagged 15 of its 64
// groups (2026-11-04). So instead of rebuilding the whole part with another edge method (plain, then the mesh's
// own points) and then as facets, each place that fails is mended on its own, and only the places are rebuilt
// while they are tried (buildTopo's only, judged by regionProblems). Each pass:
//   1. edge methods: each other method (VARIANTS) on each place's groups, then on them and their neighbours; each
//      place keeps the one that leaves the least area to facets, and a whole build judges them (undone if no better);
//   2. B-splines (SPLINE): the patches step 3 would make facets, each as one B-spline fitted to its own triangles,
//      grown round by round as the facets are; a whole build judges them (undone if no better);
//   3. facets for what that build still flags, grown round by round by the neighbours they break; a whole build
//      judges again, and what it still flags starts the next pass (PASSES).
// A place is problem groups that touch (share a mesh vertex); its region is them and every group touching them, so
// every edge and corner that can change is inside it. Every place is tried in the same trial: they do not touch.

export const PROGRESSIVE = { ROUNDS: 8, PASSES: 3 };
const WHOLE_VARIANTS = ['plain', 'mesh_plain'];
// The B-spline fallback: a patch that would go to facets is first tried as one B-spline fitted to its own triangles
// (js/freeformFit.js), taken when its triangles (vertices, edge midpoints and centres: a CAD mesh's long triangles
// have vertices only on a face's rims, and a fit held only there sagged 0.8 mm between them on a 294 mm part,
// 00005528_003) lie within TOL x diagonal of it, and kept when the whole build leaves less to facets for it.
// MIN_TRIS as addFreeform's (more than FREEFORM_MIN_FACETS); at most SAMPLES points checked per group.
// EDGE_TOL (0: off): a B-spline face built kept only with its edges and inside within EDGE_TOL x diagonal of the mesh
// (build.js strayFaces). Off: at 1e-3 it sent faces 0.003 mm off the mesh to facets (00007996_019), and what it was
// added for was OCCT's tessellation missing its deflection on B-spline faces, not the faces.
export const SPLINE = { ON: true, MIN_TRIS: 5, TOL: 5e-4, SAMPLES: 6000, EDGE_TOL: 0 };

/** Groups sharing a mesh vertex with each group. */
function touching(mesh, labels) {
  const at = new Map();
  for (let t = 0; t < labels.length; t++) for (const v of mesh.T[t]) {
    let s = at.get(v);
    if (!s) at.set(v, (s = new Set()));
    s.add(labels[t]);
  }
  const adj = new Map();
  for (const s of at.values()) {
    if (s.size < 2) continue;
    for (const a of s) {
      let n = adj.get(a);
      if (!n) adj.set(a, (n = new Set()));
      for (const b of s) if (b !== a) n.add(b);
    }
  }
  return adj;
}

/** The groups of `set` in pieces that touch. */
function piecesOfSet(set, adj) {
  const left = new Set(set), out = [];
  while (left.size) {
    const [g0] = left;
    left.delete(g0);
    const c = [g0];
    for (let i = 0; i < c.length; i++) for (const n of adj.get(c[i]) ?? []) if (left.has(n)) { left.delete(n); c.push(n); }
    out.push(new Set(c));
  }
  return out;
}

const grown = (set, adj) => { const r = new Set(set); for (const g of set) for (const n of adj.get(g) ?? []) r.add(n); return r; };
const problemCount = (probs) => { let n = 0; for (const v of Object.values(probs)) n += v.size; return n; };
// breakage a whole build cannot be recreated with (a face off its mesh area or an orient clash sometimes can)
const severe = (probs) => (probs.missing?.size ?? 0) + (probs.invalid?.size ?? 0) + (probs.open?.size ?? 0);
const problemGroups = (probs) => { const s = new Set(); for (const k of KINDS) for (const g of probs[k] ?? []) s.add(g); return s; };
// what to make facets of: the patches that fail themselves first. An orient flag is a pair's (one face uses the
// edge the wrong way, so both are flagged) and mostly follows a broken neighbour: the planes around a broken
// torus (00000890_001) went to facets with it. So orient-only patches are taken when nothing else is left, and of
// them the ones that clash on the largest share of their edges (patchProblems' orientShare: the turned face, not
// the neighbours around it; 00002468_000's four planes and three cylinders all flagged, one turned).
const culprits = (probs, keep = () => true) => {
  const s = new Set();
  for (const k of KINDS) if (k !== 'orient') for (const g of probs[k] ?? []) if (keep(g)) s.add(g);
  if (s.size) return s;
  const flagged = [...(probs.orient ?? [])].filter(keep);
  const share = probs.orientShare;
  const top = share ? Math.max(0, ...flagged.map((g) => share.get(g) ?? 0)) : 0;
  for (const g of flagged) if (!share || (share.get(g) ?? 0) >= 0.5 * top) s.add(g);
  return s;
};

/** labels with every triangle of the groups in `demote` a facet label of its own (above every group). */
function facetize(labels, surfaces, demote) {
  if (!demote.size) return { lab: labels, sf: surfaces, facetLabels: null, base: Infinity };
  const lab = Int32Array.from(labels);
  let base = -Infinity;
  for (const l of lab) base = Math.max(base, l);
  base += 1;
  const facetLabels = new Set();
  let next = base;
  for (let t = 0; t < lab.length; t++) if (demote.has(lab[t])) { lab[t] = next; facetLabels.add(next); next++; }
  return { lab, sf: new Map([...surfaces].filter(([g]) => !demote.has(g))), facetLabels, base };
}

/**
 * Mend the part `state` (buildRepaired's, not fully analytic) as progressive recovery does (above). surfaces are
 * the build's; plainSurfaces the fits as they came (no shared axes), which a group mended by a plain method gets.
 * Returns {shape, sc, state, tier (analytic, hybrid, or null when no whole build was a valid solid of the mesh's
 * volume), facetTris, facetArea, demoted, variants, splined (groups built as B-splines), splineArea, overPlanar
 * (stopped at maxFacetArea)} with the best whole build (state.surfaces with the B-splines in),
 * and counts in info. onProgress gets
 * {builds} (the trials and whole builds so far, after the first) as buildRepaired's does. methodsMs: the time the
 * edge-method trials of a pass may take before the rest are skipped (knitSolid: four of its first build, about what the
 * whole-part retries cost; a place touching big planes is the whole part, 00004062_001's 1,821 groups). maxFacetArea:
 * the share of the area past which facets are not worth it (knitSolid: the planar tier's), where recovery stops.
 */
export function progressiveRecovery(oc, mesh, state, shape, surfaces, plainSurfaces, info = {}, deadline = Infinity, onProgress = null,
  { methodsMs = Infinity, maxFacetArea = Infinity, lastChance = deadline } = {}) {
  const t0 = now();
  let builds = 1;                              // the build that failed, then every trial and whole build here
  const labels = state.labels;
  const adj = touching(mesh, labels);
  let totalArea = 0;
  const areaOf = new Map();
  for (let t = 0; t < labels.length; t++) { totalArea += mesh.area[t]; areaOf.set(labels[t], (areaOf.get(labels[t]) ?? 0) + mesh.area[t]); }
  // a trial's cost: the area its failing patches would take to facets (culprits), then how many problems it has
  const cost = (r) => {
    if (!r) return [Infinity, Infinity];
    let a = 0;
    for (const g of culprits(r.probs)) a += areaOf.get(g) ?? 0;
    return [a, r.n];
  };
  const less = (x, y) => x[0] < y[0] - 1e-12 * totalArea || (Math.abs(x[0] - y[0]) <= 1e-12 * totalArea && x[1] < y[1]);
  const variantOf = new Map();                 // group -> its edge method, where not BUILD's own
  const demote = new Set();                    // groups built as facets
  const splined = new Map();                   // group -> ['bspline', params]: the B-spline fallback (SPLINE)
  const plainish = (v) => v === 'plain' || v === 'mesh_plain';
  const surfacesFor = (vOf, dem, spl = splined) => {
    const sf = new Map(surfaces);
    for (const [g, v] of vOf) if (plainish(v) && plainSurfaces.has(g) && !dem.has(g)) sf.set(g, plainSurfaces.get(g));
    for (const [g, s] of spl) if (!dem.has(g)) sf.set(g, s);
    return sf;
  };
  // a group's B-spline, fitted once to its own triangles; null when it has too few, is one already, or no single map
  // of it holds its vertices within SPLINE.TOL
  const splineCache = new Map();
  // the farthest of a group's triangles' edge midpoints and centres from its B-spline (its vertices are the fit's own)
  const splineOff = (params, V, T) => {
    const step = Math.max(1, Math.ceil((4 * T.length) / SPLINE.SAMPLES));
    let far = 0;
    for (let i = 0; i < T.length; i += step) {
      const [a, b, c] = T[i].map((v) => V[v]);
      for (const x of [mid(a, b), mid(b, c), mid(c, a), [(a[0] + b[0] + c[0]) / 3, (a[1] + b[1] + c[1]) / 3, (a[2] + b[2] + c[2]) / 3]]) {
        far = Math.max(far, Math.abs(distNormal('bspline', params, x)[0]));
        if (far > SPLINE.TOL * mesh.diag) return far;
      }
    }
    return far;
  };
  const splineFor = (g) => {
    if (splineCache.has(g)) return splineCache.get(g);
    let out = null;
    const tris = [];
    for (let t = 0; t < labels.length; t++) if (labels[t] === g) tris.push(t);
    if (tris.length >= SPLINE.MIN_TRIS && surfaces.has(g) && surfaces.get(g)[0] !== 'bspline') {
      const local = new Map(), V = [], T = [];
      for (const t of tris) T.push(mesh.T[t].map((v) => { if (!local.has(v)) { local.set(v, V.length); V.push(mesh.V[v]); } return local.get(v); }));
      try {
        const r = fitGroup(V, T, { scale: mesh.diag });
        debug?.(`  fit ${g}${typeOf(g)[0]}: ${T.length} tris, vertices ${(r.info.max_err ?? NaN).toExponential(2)}, between ${r.params ? (splineOff(r.params, V, T) / mesh.diag).toExponential(2) : '-'} x diag`);
        if (!r.params) bump(info, 'spline_unfit');
        else if (r.info.max_err > SPLINE.TOL || splineOff(r.params, V, T) > SPLINE.TOL * mesh.diag) bump(info, 'spline_too_far');
        else out = ['bspline', r.params];
      } catch {
        bump(info, 'spline_exception');
      }
    }
    splineCache.set(g, out);
    return out;
  };
  // the mesh's topology for each set of facets, and ssi's curves for each pair, shared by every build here
  const topos = new Map(), ssiCache = new Map();
  const topoFor = (work, facetLabels, dem) => {
    const key = [...dem].sort((x, y) => x - y).join(',');
    if (!topos.has(key)) {
      if (topos.size > 8) topos.clear();
      topos.set(key, meshTopology(work, facetLabels ? [...work.F.keys()].filter((t) => facetLabels.has(work.F[t])) : null));
    }
    return topos.get(key);
  };
  // a trial of one region: its analytic faces built alone with the choices so far plus the trial's. Facets stay
  // out (a plane through its own corners needs no check): the edges to them are the chords they make.
  let trialSecs = 0, wholeSecs = 0;          // the last trial's and whole build's time: what the next ones will cost
  const trial = (region, vOf, dem, spl = splined) => {
    const { lab, sf, facetLabels, base } = facetize(labels, surfacesFor(vOf, dem, spl), dem);
    const only = new Set([...region].filter((g) => !dem.has(g)));
    if (!only.size) return { probs: {}, n: 0, base };
    bump(info, 'progressive_trials');
    onProgress?.({ builds: ++builds, ok: false });
    const t1 = now();
    try {
      const work = mesh.withLabels(lab);
      const r = buildTopo(oc, work, sf, { facetLabels, only, variantOf: vOf.size ? vOf : null, topo: topoFor(work, facetLabels, dem), ssiCache });
      const probs = regionProblems(oc, r, only);
      return { probs, n: problemCount(probs), base };
    } catch {
      bump(info, 'progressive_trial_exception');
      return null;
    } finally {
      trialSecs = now() - t1;
    }
  };
  // the whole part with every choice so far, judged as every build is
  const whole = () => {
    const t1 = now();
    try { return wholeBuild(); } finally { wholeSecs = now() - t1; }
  };
  const wholeBuild = () => {
    const { lab, sf, facetLabels, base } = facetize(labels, surfacesFor(variantOf, demote), demote);
    const work = mesh.withLabels(lab);
    bump(info, 'progressive_builds');
    onProgress?.({ builds: ++builds, ok: false });
    const r = buildTopo(oc, work, sf, { facetLabels, variantOf: variantOf.size ? variantOf : null, topo: topoFor(work, facetLabels, demote), ssiCache });
    const checked = score(oc, r.shape, work);
    const probs = patchProblems(oc, r.shape, r.faceOf, lab, r.ratios, checked.analyzers, { offFaces: checked.offFaces });
    // B-spline faces off the mesh along their edges (SPLINE.EDGE_TOL) are failing faces, as any face off OFF_MESH is
    const live = SPLINE.EDGE_TOL > 0 ? [...splined.keys()].filter((g) => !demote.has(g)) : [];
    const stray = live.length ? strayFaces(oc, r.faceOf, live, work, SPLINE.EDGE_TOL * mesh.diag) : new Set();
    for (const g of stray) probs.invalid.add(g);
    if (stray.size) { checked.off_mesh = (checked.off_mesh ?? 0) + stray.size; bump(info, 'spline_stray_faces', stray.size); }
    const { g, sc } = grade(oc, r.shape, work, probs, checked);
    // the problems in the input's groups too (a facet's is its group's), for the conversion report
    const groupOf = new Map();
    if (facetLabels) for (let t = 0; t < lab.length; t++) if (lab[t] >= base) groupOf.set(lab[t], labels[t]);
    const asGroups = Object.fromEntries(Object.entries(probs).map(([k, v]) => [k, new Set([...v].map((x) => groupOf.get(x) ?? x))]));
    return { g, sc, shape: r.shape, probs, asGroups, base, arcs: r.info.facet_arcs ?? 0 };
  };

  const timeUp = () => now() > deadline;
  const debug = globalThis.process?.env?.KNIT_DEBUG_PROGRESSIVE ? (...a) => console.error(`[progressive ${(now() - t0).toFixed(1)} s]`, ...a) : null;
  const typeOf = (g) => surfaces.get(g)?.[0] ?? 'none';
  const show = (probs) => Object.entries(probs).filter(([, v]) => v.size).map(([k, v]) => `${k}:${[...v].map((g) => `${g}${typeOf(g)[0]}`).join(',')}`).join(' ');

  let best = { g: state.g, sc: state.sc, shape, demoted: new Set(), variants: new Map(), splined: new Map(), probs: state.problems, asGroups: state.problems };
  let probs = state.problems, base = Infinity;
  const names = (set) => [...set].map((g) => `${g}${typeOf(g)[0]}`).join(',');
  // the groups to mend: flagged groups that are not facets yet, and groups with no surface (always facets)
  const flaggedNow = () => {
    const out = new Set([...problemGroups(probs)].filter((g) => g < base && !demote.has(g)));
    for (const g of new Set(labels)) if (!surfaces.has(g) && !demote.has(g)) out.add(g);
    return out;
  };
  const chosen = () => `${[...demote].sort((x, y) => x - y)}|${[...variantOf].sort((x, y) => x[0] - y[0]).join(';')}|${[...splined.keys()].sort((x, y) => x - y)}`;
  // the part as a whole with everything chosen so far: true when it is recreated
  let lastJudged = chosen();
  let lastAsGroups = state.problems;            // the last whole build's problems in the input's groups (a facet's is its group's)
  const judge = (stage) => {
    lastJudged = chosen();
    let w;
    try { w = whole(); } catch { bump(info, 'progressive_exception'); return null; }
    debug?.(`whole build (${stage}): g ${JSON.stringify(w.g)} ${show(w.asGroups)}`);
    if (better(w.g, best.g)) best = { ...w, demoted: new Set(demote), variants: new Map(variantOf), splined: new Map(splined) };
    probs = w.probs;
    base = w.base;
    lastAsGroups = w.asGroups;
    return !!w.g[0];
  };
  const within = (p, set) => {
    const q = Object.fromEntries(Object.entries(p).map(([k, v]) => [k, new Set([...v].filter((g) => set.has(g)))]));
    if (p.orientShare) Object.defineProperty(q, 'orientShare', { value: p.orientShare });
    return q;
  };
  const triedMethods = new Set();               // groups whose edge methods were tried
  const splineTried = new Set();                // groups tried as B-splines
  const backTried = new Set();                  // facet groups tried back on a surface (step 4)
  let overPlanar = false;
  debug?.(`start: g ${JSON.stringify(state.g)} ${show(state.problems)}`);
  for (let pass = 0; pass < PROGRESSIVE.PASSES && !best.g[0] && !timeUp(); pass++) {
    // 1. edge methods. Every place at once (places do not touch, so one trial over all their regions judges each in
    // its own): BUILD's own as the regions stand, then each variant on the places' groups, then on the places and
    // every group touching them; each place keeps what left it the least. Then the whole part judges them all, as a
    // build with a patch still flagged can be recreated (00000715_004: built plain, one plane off its mesh area,
    // was analytic).
    const prevVariants = new Map(variantOf);
    let uniform = null;                         // the one method that left the least breakage over every place
    let places = piecesOfSet(flaggedNow(), adj).map((place) => ({ place, mine: [...place].filter((g) => surfaces.has(g)) }))
      .filter(({ mine }) => mine.length && !mine.every((g) => triedMethods.has(g)));
    bump(info, 'progressive_places', places.length);
    const methodsEnd = now() + methodsMs / 1000;
    const methodsUp = () => timeUp() || now() > methodsEnd;
    if (places.length && !timeUp()) {
      const dem = new Set(demote);
      const all = new Set();
      for (const p of places) {
        for (const g of p.mine) triedMethods.add(g);
        for (const g of p.place) if (!surfaces.has(g)) dem.add(g);
        p.region = grown(p.place, adj);
        for (const g of p.region) all.add(g);
      }
      const ref = trial(all, variantOf, dem);
      debug?.(`pass ${pass}: ${places.length} places in ${all.size} groups; clean: ${ref?.n} ${ref ? show(ref.probs) : 'threw'}`);
      const consider = (r, v, wide) => { if (r && severe(r.probs) < (uniform?.severe ?? severe(ref.probs))) uniform = { v, wide, severe: severe(r.probs) }; };
      if (ref) {
        for (const p of places) { const q = within(ref.probs, p.region); p.cost = cost({ probs: q, n: problemCount(q) }); p.n = problemCount(q); }
        let open = places.filter((p) => p.n > 0);
        for (const v of VARIANTS) {
          if (!open.length || methodsUp()) break;
          const vOf = new Map(variantOf);
          for (const p of open) for (const g of p.mine) vOf.set(g, v);
          const r = trial(all, vOf, dem);
          debug?.(`  ${v}: ${r?.n} ${r ? show(r.probs) : 'threw'}`);
          if (!r) continue;
          consider(r, v, false);
          for (const p of open) {
            const q = within(r.probs, p.region), c = cost({ probs: q, n: problemCount(q) });
            if (less(c, p.cost)) { p.cost = c; p.n = problemCount(q); p.choice = v; }
          }
          open = open.filter((p) => p.n > 0);
        }
        // ...and on the places with every group touching them, built one ring further out: what breaks a place can
        // be its neighbours' surfaces (shared axes adjusted all 32 of 00009274_009's; its planes 8 and 9 failed under
        // every method of their own and the whole part built plain was fully analytic). Kept for a place when its own
        // region then has no problem (the whole build judges the rest: 00003838_001's two places wanted plain and
        // wide mesh_plain, and each one's ring held the other).
        const outerAll = new Set();
        for (const p of open) { p.outer = grown(p.region, adj); for (const g of p.outer) outerAll.add(g); }
        for (const v of VARIANTS) {
          if (!open.length || methodsUp()) break;
          const vOf = new Map(variantOf);
          for (const p of open) for (const g of p.region) if (!dem.has(g)) vOf.set(g, v);
          const r = trial(outerAll, vOf, dem);
          debug?.(`  wide ${v} (${outerAll.size}): ${r?.n} ${r ? show(r.probs) : 'threw'}`);
          if (!r) continue;
          consider(r, v, true);
          for (const p of open) if (!problemCount(within(r.probs, p.region))) { p.wide = v; p.n = 0; }
          open = open.filter((p) => p.n > 0);
        }
        for (const p of places) {
          if (p.wide) { for (const g of p.region) if (!dem.has(g)) variantOf.set(g, p.wide); bump(info, `progressive_wide_${p.wide}`); }
          else if (p.choice) { for (const g of p.mine) variantOf.set(g, p.choice); bump(info, `progressive_${p.choice}`); }
        }
      }
    }
    // the methods chosen stay only if the whole part is better for them (a method can trade a failing face for orient
    // flags that the whole build still fails on, and its facets then go elsewhere: mesh_clean on 00002299_000)
    // Undone, the method that left the least breakage over every place is tried on them all alike (00003456_007:
    // its places chose apart and the whole build got worse; built from the mesh's points it was recreated, every
    // flag left a face off its area)
    const undo = (was) => {
      variantOf.clear();
      for (const [g, v] of prevVariants) variantOf.set(g, v);
      probs = was.probs;
      base = was.base ?? Infinity;
      lastAsGroups = was.asGroups ?? lastAsGroups;
      lastJudged = chosen();
    };
    let done = false;
    if (chosen() !== lastJudged) {
      const was = best;
      if (judge('methods')) break;
      if (best === was) {
        bump(info, 'progressive_methods_undone');
        debug?.('  methods undone');
        undo(was);
        if (uniform && !timeUp()) {
          for (const p of places) for (const g of uniform.wide ? p.region : p.mine) if (surfaces.has(g) && !demote.has(g)) variantOf.set(g, uniform.v);
          debug?.(`  every place ${uniform.wide ? 'wide ' : ''}${uniform.v}`);
          if (chosen() !== lastJudged) {
            done = judge('one method');
            if (!done && best === was) { bump(info, 'progressive_methods_undone'); undo(was); }
            else bump(info, `progressive_uniform_${uniform.v}`);
          }
        }
      }
    }
    // ...and once, the whole part by one method: every group plain, then from the mesh's points too (00050238_3:
    // no place's method recreated it, the whole part built plain was fully analytic, as today's main retries it).
    // Kept only when recreated.
    for (const v of pass === 0 && !done ? WHOLE_VARIANTS : []) {
      if (timeUp()) break;
      const keep = new Map(variantOf), was = { probs, base, lastAsGroups };
      for (const g of surfaces.keys()) if (!demote.has(g)) variantOf.set(g, v);
      if (chosen() === lastJudged) continue;
      if (judge(`whole part ${v}`)) { done = true; bump(info, `progressive_whole_${v}`); break; }
      variantOf.clear();
      for (const [g, x] of keep) variantOf.set(g, x);
      ({ probs, base, lastAsGroups } = was);
      lastJudged = chosen();
    }
    if (done) break;
    if (timeUp()) break;
    // 2. B-splines (SPLINE): the patches the facets would take, each as a B-spline fitted to its own triangles, grown
    // round by round by the neighbours they break as the facets are, judged in each round's region; then the whole
    // build judges, and they stay only if it is better for them. What it still flags goes on to facets.
    // only with time left for the facets after them: B-splines that run the clock out leave the part to the mesh as
    // facets where the facets alone made a hybrid (00000765_001: 29% facets on main, all facets with them)
    const splineTime = now() + 2 * wholeSecs + 4 * trialSecs < deadline;
    if (SPLINE.ON && !splineTime) bump(info, 'progressive_splines_skipped_for_time');
    if (SPLINE.ON && splineTime) {
      const flaggedS = flaggedNow();
      const spl = new Map(splined), added = new Set();
      const take = (g) => {
        if (spl.has(g) || splineTried.has(g)) return false;
        splineTried.add(g);
        const s = splineFor(g);
        if (!s) return false;
        spl.set(g, s);
        added.add(g);
        return true;
      };
      for (const g of culprits(probs, (x) => flaggedS.has(x) && surfaces.has(x))) take(g);
      for (let round = 0; added.size && round < PROGRESSIVE.ROUNDS && !timeUp(); round++) {
        const r = trial(grown(added, adj), variantOf, demote, spl);
        bump(info, 'progressive_spline_rounds');
        debug?.(`  B-splines ${names(added)}: ${r?.n} ${r ? show(r.probs) : 'threw'}`);
        if (!r || r.n === 0) break;
        let more = false;
        for (const g of culprits(r.probs, (x) => x < r.base && !demote.has(x) && surfaces.has(x))) if (take(g)) more = true;
        if (!more) break;
      }
      if (added.size && !timeUp()) {
        // kept when they leave less of the area to facets than the build before them: fewer problems is not enough,
        // as B-splines can break planes beside them (00007502_007: 45 problems to 32, but 1.0% facets to 1.7%)
        const keepVariants = new Map(variantOf), keepWas = { probs, base, lastAsGroups };
        // the area the facet step would start from (step 3's fix: surfaceless flagged groups and the culprits)
        const toFacets = () => {
          const fl = flaggedNow(), f = new Set([...fl].filter((g) => !surfaces.has(g)));
          for (const g of culprits(probs, (x) => fl.has(x))) f.add(g);
          let x = 0;
          for (const g of f) x += areaOf.get(g) ?? 0;
          return x;
        };
        const before = toFacets();
        for (const g of added) { splined.set(g, spl.get(g)); variantOf.delete(g); }
        if (judge('B-splines')) { bump(info, 'progressive_splines_kept', added.size); break; }
        const after = toFacets();
        if (!(after < before - 1e-9 * totalArea)) {
          // no less for facets: undone, and their patches go on to facets as before
          bump(info, 'progressive_splines_undone');
          debug?.(`  B-splines undone (area to facets ${(after / totalArea).toFixed(4)} vs ${(before / totalArea).toFixed(4)})`);
          for (const g of added) splined.delete(g);
          variantOf.clear();
          for (const [g, v] of keepVariants) variantOf.set(g, v);
          ({ probs, base, lastAsGroups } = keepWas);
          lastJudged = chosen();
        } else { bump(info, 'progressive_splines_kept', added.size); debug?.(`  B-splines kept (area to facets ${(after / totalArea).toFixed(4)} vs ${(before / totalArea).toFixed(4)})`); }
      }
      if (timeUp()) break;
    }
    // 3. facets for what the whole build still flags (patches that fail themselves first), grown round by round by
    // the neighbours they break, judged in each round's region: every place at once too
    const flagged = flaggedNow();
    if (!flagged.size) break;
    const fix = new Set([...flagged].filter((g) => !surfaces.has(g)));
    for (const g of culprits(probs, (x) => flagged.has(x))) fix.add(g);
    // more of the area as facets than the planar tier would make: no hybrid of these is the better export, and the
    // facet builds are the slow ones (curved faces bounded by hundreds of facet sides: 33 s a build on 00009079_017)
    const facetShare = () => { let x = 0; for (const g of new Set([...demote, ...fix])) x += areaOf.get(g) ?? 0; return x / Math.max(totalArea, 1e-300); };
    let over = false;
    for (let round = 0; fix.size && round < PROGRESSIVE.ROUNDS && !timeUp(); round++) {
      if (facetShare() > maxFacetArea) { over = true; break; }
      const dem = new Set([...demote, ...fix]);
      const r = trial(grown(fix, adj), variantOf, dem);
      bump(info, 'progressive_facet_rounds');
      debug?.(`  facets ${names(fix)}: ${r?.n} ${r ? show(r.probs) : 'threw'}`);
      if (!r || r.n === 0) break;
      let more = false;
      for (const g of culprits(r.probs, (x) => x < r.base && !dem.has(x))) { fix.add(g); more = true; }
      if (!more) break;
    }
    if (over || facetShare() > maxFacetArea) { overPlanar = true; bump(info, 'progressive_over_planar'); debug?.('  facets past the planar tier'); break; }
    for (const g of fix) { demote.add(g); variantOf.delete(g); splined.delete(g); }
    // nothing new chosen: the whole build would be the last one again
    if (chosen() === lastJudged) { bump(info, 'progressive_stalled'); break; }
    if (judge('facets')) break;
    // 4. facets that fail themselves (a thin facet turned over where its corners went onto curved neighbours:
    // 00007996_019's plane 2, 00003304_000's cylinder 56) leave no flagged patch for the next pass, and the export
    // went to the mesh as facets: each such group back on a surface (its B-spline, else its own), kept if the whole
    // build is better for it
    const back = [...problemGroups(lastAsGroups)].filter((g) => demote.has(g) && !backTried.has(g) && surfaces.has(g));
    if (back.length && !timeUp()) {
      const was = best, keepWas = { probs, base, lastAsGroups };
      for (const g of back) {
        backTried.add(g);
        demote.delete(g);
        const sp = SPLINE.ON ? splineFor(g) : null;
        if (sp) splined.set(g, sp);
      }
      debug?.(`  facets back on surfaces: ${names(back)}`);
      if (judge('facets back on surfaces')) { bump(info, 'progressive_facets_back', back.length); break; }
      if (best === was) {
        for (const g of back) { demote.add(g); splined.delete(g); }
        ({ probs, base, lastAsGroups } = keepWas);
        lastJudged = chosen();
        // ...and else the groups around them as facets too: a facet turns over where a neighbour's surface pulls its
        // corner (00003502_006's torus 119: back on its surface it broke five faces; main recreated it as a hybrid)
        const ring = [...grown(new Set(back), adj)].filter((g) => !demote.has(g) && surfaces.has(g));
        if (ring.length && !timeUp()) {
          for (const g of ring) { demote.add(g); variantOf.delete(g); splined.delete(g); }
          debug?.(`  facets around them: ${names(ring)}`);
          if (judge('facets around failing facets')) { bump(info, 'progressive_facets_ring', ring.length); break; }
        }
      } else bump(info, 'progressive_facets_back', back.length);
    }
  }
  // no whole build recreated with B-splines in play: out of time, every group the last whole build flagged to facets at
  // once, in one more whole build before lastChance (the time the facet tiers keep). Their trials take the time the
  // facets alone would have used (00000130_005 recreated at 112.7 s of a 112.5 s budget, so under load it was not).
  // ...and with B-splines in a solid that is off in volume (00191936_0: closed, valid, no problem, 1.08% off where the
  // facets alone made 1.0% or less), the B-splines themselves to facets in one more whole build
  const facetShareOf = () => { let x = 0; for (const g of demote) x += areaOf.get(g) ?? 0; return x / Math.max(totalArea, 1e-300); };
  const lastChanceFor = (groups, stage) => {
    if (!groups.length || best.g[0] || now() + wholeSecs >= lastChance) return;
    for (const g of groups) { demote.add(g); variantOf.delete(g); splined.delete(g); }
    if (facetShareOf() <= maxFacetArea && judge(stage)) bump(info, 'progressive_last_chance');
  };
  if (!best.g[0] && splined.size && !overPlanar) {
    if (timeUp()) lastChanceFor([...problemGroups(lastAsGroups)].filter((g) => !demote.has(g)), 'last chance, flagged to facets');
    lastChanceFor([...splined.keys()], 'last chance, B-splines to facets');
  }
  let n = 0, a = 0;
  for (let t = 0; t < labels.length; t++) if (best.demoted.has(labels[t])) { n++; a += mesh.area[t]; }
  info.progressive_secs = Math.round((now() - t0) * 10) / 10;
  if (best.arcs) info.progressive_facet_arcs = best.arcs;
  if (best.variants.size) info.progressive_variant_groups = best.variants.size;
  // the B-spline fallback's groups in the best build, with their share of the area
  let sa = 0;
  for (let t = 0; t < labels.length; t++) if (best.splined.has(labels[t])) sa += mesh.area[t];
  if (best.splined.size) info.progressive_spline_groups = best.splined.size;
  const finalSurfaces = new Map(state.surfaces);
  for (const [g, s] of best.splined) finalSurfaces.set(g, s);
  // every triangle a facet (a part of one group) is the faceted tier, as the old fallback called it
  return { shape: best.shape, sc: best.sc, tier: best.g[0] ? (n === labels.length ? 'faceted' : n ? 'hybrid' : 'analytic') : null, overPlanar,
    facetTris: n, facetArea: Math.round((a / Math.max(totalArea, 1e-300)) * 1e4) / 1e4, demoted: [...best.demoted], variants: best.variants,
    splined: [...best.splined.keys()], splineArea: Math.round((sa / Math.max(totalArea, 1e-300)) * 1e4) / 1e4,
    state: { ...state, surfaces: finalSurfaces, problems: best.asGroups, ok: !!best.g[0] && !n, g: best.g, sc: best.sc } };
}
