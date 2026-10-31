/**
 * knit/repair.js — the kernel-judged repair loop and the facet fallback, a port of ml/solid_rebuild.py
 * build_repaired / facet_fallback (measured there: Onshape auto 211 -> 247 of 275 with the loop, then
 * 269 with the fallback; 2026-09-27 and 2026-10-02).
 *
 * The loop: while the part is not a closed, valid solid of the mesh's volume, find the failing patches,
 * try ONE move (absorb the group into the neighbours it lies on, or its next surface type from the
 * fitters), rebuild, keep it only if the part got better. Inference-time signals only.
 *
 * The fallback, tried only when the loop's best is not recreated:
 *   hybrid   the patches still failing (and surfaceless ones) as facets, the rest analytic
 *   planar   only plane patches analytic
 *   faceted  the mesh as a polyhedral solid
 */

import { buildTopo, buildPolyhedral, patchProblems, grade, better } from './build.js';
import { distNormal, unit, sub, cross } from './geometry.js';

export const LOOP = {
  SMALL_GROUP: 3,            // facets; a failing group this small is absorbed, never refit
  TIE_RATIO: 1.5,            // fits within this factor of the best residual count as tied
  RETYPES: 2,                // alternatives tried per group before absorbing it
  MAX_BUILDS: 40,
  BUDGET_S: 300,
  MAX_RESIDUAL_RATIO: 10,    // a retype candidate fits within this factor of the best candidate
  ABSORB_TOL: 2e-3,          // x diagonal: a larger group's facets move only onto a surface this near
};
const SIMPLER = { plane: 0, sphere: 1, cylinder: 2, cone: 3, torus: 4 };
const KINDS = ['missing', 'invalid', 'orient', 'area', 'open'];

const bump = (info, k, n = 1) => { info[k] = (info[k] ?? 0) + n; };
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

/** Candidate surfaces other than the current type: the simplest first among those tied with the best. */
function alternatives(cands, current, scaleLen) {
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
 * buildTopo, then repair moves while the part is not a closed, valid solid of the mesh's volume.
 * fitCandidates({group: soupTris}) -> {group: [{type, params, residual}]}. Returns {shape, info, state}
 * with state = the best build's labels, surfaces, problems and verdict (for facetFallback).
 */
export function buildRepaired(oc, mesh, surfaces, { fitCandidates = null, maxBuilds = LOOP.MAX_BUILDS,
  budget = LOOP.BUDGET_S, onProgress = null } = {}) {
  const t0 = now();
  const log = {};
  let slowest = 0;

  const build = (labels, surfs, first = false) => {
    const t = now();
    try {
      const work = mesh.withLabels(labels);
      const r = buildTopo(oc, work, surfs);
      const probs = patchProblems(oc, r.shape, r.faceOf, labels, r.ratios);
      const { g, sc } = grade(oc, r.shape, work, probs);
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
  if (!best.g[0]) {
    const lab = Int32Array.from(best.labels), sf = new Map(best.surfs);
    const slivers = [...best.probs.missing].filter((g) => !sf.has(g) && countOf(lab, g) <= LOOP.SMALL_GROUP);
    let movedAny = 0;
    for (const g of slivers) movedAny += absorbGroup(mesh, lab, sf, g);
    if (slivers.length && movedAny && timeLeft()) {
      const trial = build(lab, sf);
      builds++;
      bump(log, 'sliver_absorb_tried');
      const kept = !!trial && !better(best.g, trial.g);
      if (kept) { best = trial; bump(log, 'slivers_absorbed', slivers.length); }
      moves.push({ move: 'absorb_slivers', groups: slivers, kept });
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
    const kept = attempt(lab, sf, move[0]);
    moves.push(move[0] === 'retype' ? { move: 'retype', group: move[1], to: move[3].type, kept } : { move: 'absorb', group: move[1], kept });
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
  const probs = patchProblems(oc, r.shape, r.faceOf, lab, r.ratios);
  const { g, sc } = grade(oc, r.shape, work, probs);
  return { g, sc, shape: r.shape, probs, base };
}

/**
 * The first fallback tier that recreates the part, else the best-graded of them all. `state` is
 * buildRepaired's; past `deadline` (seconds, performance.now() / 1000) the hybrid and planar tiers are
 * skipped and only the polyhedral solid is tried. Returns {shape, sc, tier, facetTris, facetArea}: tier is analytic, hybrid, planar,
 * faceted, or none (no tier recreated the part; the best of them is still returned).
 */
export function facetFallback(oc, mesh, shape, state, info = {}, deadline = Infinity) {
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
  for (let round = 0; round < FALLBACK.ROUNDS; round++) {
    const r = attempt('hybrid', demote);
    if (!r || r.g[0]) break;
    let grew = false;
    for (const k of KINDS) for (const x of r.probs[k]) if (x < r.base && !demote.has(x)) { demote.add(x); grew = true; }
    if (!grew) break;
  }
  if (!best.g[0]) attempt('planar', new Set([...present].filter((g) => surfaces.get(g)?.[0] !== 'plane')));
  if (!best.g[0]) {
    bump(info, 'fallback_faceted_tried');
    try {
      const { shape: sh } = buildPolyhedral(oc, mesh);
      const { g, sc } = grade(oc, sh, mesh, {});
      if (better(g, best.g)) best = { g, shape: sh, sc, tier: 'faceted', n: mesh.T.length, area: 1 };
    } catch {
      bump(info, 'fallback_exception');
    }
  }
  info.fallback_secs = Math.round((now() - t0) * 10) / 10;
  return { shape: best.shape, sc: best.sc, tier: best.g[0] ? best.tier : 'none', bestTier: best.tier,
           facetTris: best.n, facetArea: Math.round(best.area * 1e4) / 1e4, demoted: [...(best.demote ?? [])] };
}
