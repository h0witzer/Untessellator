/**
 * paintCore.js — the fill itself, with no model runtime attached.
 *
 * Everything the studio's fill does, taking the model as a `runner(features, n)` function, so the
 * same code runs in the browser (js/paintTool.js wraps onnxruntime-web around it) and in Node
 * (test/autoSeed.mjs wraps onnxruntime-node), and what the harness measures is what ships.
 *
 * Methods for one fill:
 *   plain      paintFrom from the seed, no model (the type must be given, or the model is still
 *              asked for the type on auto)
 *   model      the model's proposed region, grown and refit by paintFrom
 *   hyp        the model's SURFACE HYPOTHESIS grows a region first, then paintFrom takes over
 *   veto       paintFrom with an adaptive angle tolerance and a membership veto
 *   hyp+veto   both
 */

import { buildFacets, buildAdjacency, dot } from './meshCore.js';
import { paintFrom, surfaceEval, creaseFor, creaseComponent, creaseTableFor } from './regionGrow.js';
import { fitRegionAs, classifyRegion } from './gaussMap.js';
import { attachCurvature } from './curvature.js';
import { buildFeatures, proposeRegion, surfaceHypothesis, fitterFor, patchNeighbours } from './seedNet.js';
import { refineFeatures } from './refineFeatures.js';
import { cutFrom } from './cutFill.js';
import { attachEdgeGraph } from './edgeGraph.js';

/** The crease barrier for a region with no analytic type (the model's own region). */
export const FREEFORM_CREASE = 30;

/** Primitive complexity, simplest first: the order classifyRegion prefers. */
export const ORDER_RANK = { plane: 0, cylinder: 1, cone: 2, sphere: 3, torus: 4 };

const DEG = Math.PI / 180;
const COS15 = Math.cos(15 * DEG);

/**
 * Facets, adjacency and curvature from flat non-indexed positions (9 per triangle). Done once per
 * mesh: adjacency is the expensive part.
 */
export function preparePaintPositions(positions) {
  const P = positions instanceof Float64Array ? positions : Float64Array.from(positions);
  const facets = buildFacets(P);
  const adjacency = buildAdjacency(P, facets);
  attachCurvature(P, facets);   // the torus fitter uses it
  const sizes = [];
  for (let t = 0; t < facets.count; t++) sizes.push(Math.sqrt(Math.max(facets.areas[t], 0)));
  sizes.sort((a, b) => a - b);
  return { facets, adjacency, tolD: 0.5 * (sizes[sizes.length >> 1] || 1) };
}

/**
 * typeFinalGroup's exactness bar, x the part diagonal: a fit whose every vertex is this close is the
 * group's own surface at the mesh's precision (float32 coordinates quantise at ~6e-8 of their size).
 */
export const EXACT_TYPE_BAR = 1e-6;

const diagonals = new WeakMap();
function partDiagonal(facets) {
  if (!diagonals.has(facets)) {
    const P = facets.positions;
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < P.length; i += 3) for (let k = 0; k < 3; k++) {
      if (P[i + k] < lo[k]) lo[k] = P[i + k];
      if (P[i + k] > hi[k]) hi[k] = P[i + k];
    }
    diagonals.set(facets, Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]));
  }
  return diagonals.get(facets);
}

/** Largest distance of the vertices of `tris` from a surface, and how many distinct vertices there are. */
export function vertexDeviation(tris, facets, type, params) {
  const ev = surfaceEval(type, params);
  if (!ev) return { d: Infinity, n: 0 };
  const P = facets.positions;
  const seen = new Set();
  let d = 0;
  for (const t of tris) {
    for (let k = 0; k < 3; k++) {
      const o = t * 9 + k * 3;
      const key = P[o] + '_' + P[o + 1] + '_' + P[o + 2];
      if (seen.has(key)) continue;
      seen.add(key);
      d = Math.max(d, Math.abs(ev.distance([P[o], P[o + 1], P[o + 2]])));
    }
  }
  return { d, n: seen.size };
}

/** Degrees of freedom of each primitive: an exact fit is evidence only on several times as many vertices. */
const DOF = { plane: 3, cylinder: 5, cone: 6, sphere: 4, torus: 7 };

/**
 * The type a FINISHED group exports with. A fill picks its type on the model's first patch, before
 * the group has grown, and keeps it: on the Onshape set, whole pure groups that a plane fits to 1e-9 of
 * the diagonal had been painted as tori, cones or cylinders. next.BREP.io's cascade (plane, cylinder,
 * cone, sphere, torus: the first that fits) gets those right.
 *
 * classifyRegion on the final group is the better typer on PURE groups (98% right on the Onshape set,
 * 92-95% on OCCT/NURBS/gmsh meshes) but not on impure ones, where re-typing every group hurt game
 * meshes (memory: surface-scorecard). What tells them apart without the truth: a pure group of a CAD
 * tessellation has its vertices ON its surface, an impure one has no surface that fits it exactly. So
 * the classifier's type replaces the painted one only when it fits every vertex within `bar` x the part
 * diagonal, on at least `support` x its degrees of freedom distinct vertices (stl2step's peel uses 3x).
 * Without that, a low-poly group is fitted "exactly" by any type with enough parameters: on the game
 * meshes 89 right cylinders of 7-10 vertices became tori, spheres and cones. Otherwise the painted type
 * stays. An untyped group (the fill
 * found no analytic surface) takes the classifier's type, or a plane through 1-2 facets within
 * `looseTol` x the diagonal.
 *
 * @returns {{type: string|null, changed: boolean, exact: boolean}}
 */
export function typeFinalGroup(tris, facets, painted, { bar = EXACT_TYPE_BAR, looseTol = 5e-4, support = 3 } = {}) {
  const set = tris instanceof Set ? tris : new Set(tris);
  const diag = partDiagonal(facets);
  const keep = { type: painted ?? null, changed: false, exact: false };
  if (set.size < 3) {
    if (painted || !set.size) return keep;
    const plane = fitRegionAs('plane', set, facets);
    return plane.params && vertexDeviation(set, facets, 'plane', plane.params).d <= looseTol * diag
      ? { type: 'plane', changed: true, exact: false } : keep;
  }
  const cls = classifyRegion(set, facets);
  if (!(cls.type in ORDER_RANK)) return keep;
  if (!painted) return { type: cls.type, changed: true, exact: false };
  if (cls.type === painted) return keep;
  const { d, n } = vertexDeviation(set, facets, cls.type, cls.params);
  return d <= bar * diag && n >= support * DOF[cls.type]
    ? { type: cls.type, changed: true, exact: true } : keep;
}

/**
 * The piece of `region` connected to `seed` across edges below `creaseDeg`, through region facets
 * only; when the seed is not in the region, its largest such piece.
 */
export function connectedPiece(seed, region, adjacency, creaseDeg) {
  if (region.has(seed)) return creaseComponent(seed, region, adjacency, creaseDeg);
  const seen = new Set();
  let best = new Set();
  for (const t of region) {
    if (seen.has(t)) continue;
    const piece = creaseComponent(t, region, adjacency, creaseDeg);
    for (const u of piece) seen.add(u);
    if (piece.size > best.size) best = piece;
  }
  return best;
}

const centroid = (facets, t) =>
  [facets.centroids[t * 3], facets.centroids[t * 3 + 1], facets.centroids[t * 3 + 2]];
const normal = (facets, t) =>
  [facets.normals[t * 3], facets.normals[t * 3 + 1], facets.normals[t * 3 + 2]];

/** regionGrow's growOn test against a given surface, flooding from `seedSet`. */
function growAgainst(seedSet, mesh, type, params, tolD, cosTol, taken = null) {
  const { facets, adjacency } = mesh;
  const creaseDeg = creaseFor(creaseTableFor(adjacency),type);
  const ev = surfaceEval(type, params);
  if (!ev) return new Set();
  const fits = (t) => {
    const c = centroid(facets, t);
    if (Math.abs(ev.distance(c)) > tolD) return false;
    return Math.abs(dot(ev.normal(c), normal(facets, t))) >= cosTol;
  };
  const region = new Set();
  const queue = [];
  for (const s of seedSet) if (!(taken && taken[s]) && fits(s)) { region.add(s); queue.push(s); }
  while (queue.length) {
    const t = queue.pop();
    for (const e of adjacency.get(t) || []) {
      if (region.has(e.neighbor) || (taken && taken[e.neighbor]) || e.angle >= creaseDeg || !fits(e.neighbor)) continue;
      region.add(e.neighbor);
      queue.push(e.neighbor);
    }
  }
  return region;
}

/**
 * paintFrom's grow/refit loop with two acceptance changes: the angle tolerance adapts to the
 * region's own normal deviation (3 x p90, clamped 4-15 deg), and a facet must also have
 * membership probability >= `tau` (default 0.5) from a model patch seeded at the accepted facet
 * next to it. `wall` is a regionGrow wall predicate (the learned boundary), honoured as paintFrom
 * does; `onQuery(facet, p)` sees every membership probability the veto consults.
 */
export async function paintWithVeto(seed, type, mesh, seedRegion, runner, taken = null, extended = false,
                                     { tau = 0.5, wall = null, onQuery = null } = {}) {
  const w = wall && (!wall.types || wall.types.includes(type)) ? wall : null;
  const { facets, adjacency, tolD } = mesh;
  const creaseDeg = creaseFor(creaseTableFor(adjacency),type);
  const prob = new Map();
  let calls = 0;
  const patch = async (centre) => {
    const { features, ids, n } = buildFeatures(centre, facets, adjacency, 512, null, taken, extended);
    const { sameLogit } = await runner(features, n, patchNeighbours(ids, adjacency));
    calls++;
    for (let i = 0; i < n; i++) if (!prob.has(ids[i])) prob.set(ids[i], 1 / (1 + Math.exp(-sameLogit[i])));
  };
  await patch(seed);

  const grow = async (seedSet, params) => {
    const ev = surfaceEval(type, params);
    if (!ev) return new Set();
    const dev = t => Math.acos(Math.min(1, Math.abs(dot(ev.normal(centroid(facets, t)), normal(facets, t))))) / DEG;
    const devs = [...seedSet].map(dev).sort((a, b) => a - b);
    const angle = Math.min(15, Math.max(4, 3 * (devs[Math.floor(0.9 * (devs.length - 1))] ?? 5)));
    const cosTol = Math.cos(angle * DEG);
    const fits = t => Math.abs(ev.distance(centroid(facets, t))) <= tolD
      && Math.abs(dot(ev.normal(centroid(facets, t)), normal(facets, t))) >= cosTol;
    const region = new Set();
    const queue = [];
    for (const s of seedSet) {
      if ((taken && taken[s]) || !fits(s) || (prob.has(s) && prob.get(s) < tau)) continue;
      region.add(s); queue.push(s);
    }
    while (queue.length) {
      const t = queue.pop();
      for (const e of adjacency.get(t) || []) {
        const nb = e.neighbor;
        if (region.has(nb) || (taken && taken[nb]) || e.angle >= creaseDeg || (w && w(t, e)) || !fits(nb)) continue;
        if (!prob.has(nb)) await patch(t);
        if (onQuery && prob.has(nb)) onQuery(nb, prob.get(nb));
        if ((prob.get(nb) ?? 1) < tau) continue;
        region.add(nb); queue.push(nb);
      }
    }
    return region;
  };

  let region = creaseComponent(seed, seedRegion, adjacency, creaseDeg, w);
  let fit = fitRegionAs(type, region, facets);
  if (fit.degenerate || !fit.params) return { region: new Set([seed]), calls };
  for (let i = 0; i < 4; i++) {
    const grown = await grow(region, fit.params);
    if (grown.size === 0) break;
    const refit = fitRegionAs(type, grown, facets);
    if (refit.degenerate || !refit.params) break;
    const stable = grown.size === region.size;
    region = grown;
    fit = refit;
    if (stable) break;
  }
  return { region, calls };
}

/**
 * REFINED FILLS: a second model re-labels the 512-facet patch around the seed GIVEN a first fill (its
 * region, its fitted surface, every facet's distance and angle to that surface, and the learned
 * boundary along paths from the seed: js/refineFeatures.js), then a minimum cut re-chooses the region
 * with the refiner's verdict as the per-facet term inside its patch and the learned boundary pricing
 * the cut (js/cutFill.js, memberReplacesFit). The cut is what turns the refiner's facet-by-facet verdict
 * into one outline: it closes gaps and ignores isolated misjudgements.
 *
 * Measured on 40 held-out ABC parts x 4 meshers with the second edge model (runs/edge2), clean single
 * fills (true type): wall alone 78/65/69/48%, refiner 85/72/76/52%, refiner + cut 87/81/82/54%
 * (test/growEval.mjs crf:a=2;l=4;r=1). Auto-paint area clean 75.3/75.3/76.0/65.8 -> 81.8/82.5/85.4/70.3
 * (test/autoSeed.mjs --refine --crf). Refining twice gains nothing; one pass is the fixed point.
 *
 * `cut` is the refiner's logit threshold (1, chosen on the even parts); inside the cut a facet's term
 * is alpha * clamp((logit - cut) / scale, -1, 1) and an edge costs lambda * length * (1 - p).
 */
export const REFINE_DEFAULTS = Object.freeze({ cut: 1, crf: true, alpha: 2, lambda: 4, scale: 4 });

/**
 * One refinement of `region` (a fill of type `fitter` from `seed`). `runner` is the refiner's model
 * runner ((features, n, graph) => { sameLogit }). Facets marked in `taken` are never added.
 * Returns the new region, or the old one when the fill has no fittable surface.
 */
export async function refineFill(seed, fitter, region, mesh, runner, taken = null, opts = {}) {
  const o = { ...REFINE_DEFAULTS, ...opts };
  const { facets, adjacency } = mesh;
  const fit = fitRegionAs(fitter, region, facets);
  if (!fit.params) return region;
  const r = refineFeatures(seed, facets, adjacency, { region, type: fitter, params: fit.params }, -1);
  const { sameLogit } = await runner(r.features, r.n, r.graph);
  const keep = new Set(region);
  const member = new Map();
  for (let i = 0; i < r.n; i++) {
    const t = r.ids[i];
    if (taken && taken[t]) continue;
    member.set(t, 0.5 + 0.5 * Math.max(-1, Math.min(1, (sameLogit[i] - o.cut) / o.scale)));
    if (sameLogit[i] >= o.cut) keep.add(t); else keep.delete(t);
  }
  // The refiner's per-facet membership, for callers that keep a confidence per painted facet.
  o.onMember?.(member);
  keep.add(seed);
  const refined = creaseComponent(seed, keep, adjacency, creaseFor(creaseTableFor(adjacency), fitter));
  if (!o.crf) return refined;
  // Edge ids and lengths for the cut, built once per mesh.
  mesh.graph ??= attachEdgeGraph(facets.positions, facets, adjacency);
  return cutFrom(seed, fitter, facets, adjacency, mesh.graph, {
    lambda: o.lambda, member, alpha: o.alpha, memberReplacesFit: true, seedRegion: refined,
    ...(taken ? { taken } : {}),
  }).region;
}

/**
 * Fill from one seed triangle.
 *
 * @param {Function|null} runner  model runner (features, n) => {sameLogit, typeLogit, surface};
 *                                may be null only for method 'plain' with a given type
 * @param {number} seed
 * @param {object} mesh           from preparePaintPositions
 * @param {object} opts
 * @param {'plain'|'model'|'hyp'|'veto'|'hyp+veto'} opts.method
 * @param {string} opts.type      'auto' or a fitter type
 * @param {Uint8Array} [opts.taken]  facets painted into OTHER groups: the model's patch walks
 *        around them and no growth enters them. Simulated on 80 ABC parts with simpler faces painted
 *        first (test/fillEval.mjs --sequential): clean fills on non-plane faces 56 -> 60% OCCT,
 *        47 -> 51% NURBS, 46 -> 58% gmsh against dropping them only afterwards.
 * @param {boolean} [opts.extended] the model takes the v6 inputs (js/seedFeatures.js)
 * @param {object} [opts.refine]  { runner, ...REFINE_DEFAULTS overrides }: refine the fill (refineFill)
 */
export async function paintWith(runner, seed, mesh, { method, type, taken = null, extended = false, grow = {}, reportFit = true,
                                                      refine = null }) {
  // `reportFit`: fit the final region for the caller's message. Auto-paint refits every group
  // once at the end instead, and skips this per fill.
  // `grow`: extra paintFrom options for every growth (tangent wall, crease table, angleTol,
  // tolScale); empty means regionGrow's defaults.
  const t0 = performance.now();
  const { facets, adjacency } = mesh;
  const out = { region: new Set([seed]), modelRegion: null, predicted: null, confidence: null,
                fitter: null, fit: null, calls: 0, notes: [] };

  let p = null;
  if (method !== 'plain' || type === 'auto') {
    p = await proposeRegion(seed, facets, adjacency, runner, { excluded: taken, extended });
    out.calls++;
    out.predicted = p.type;
    out.confidence = p.confidence;
    out.modelRegion = p.region;
  }

  // On auto the MODEL proposes the region and the FITTERS choose the type, simplest first:
  // classifyRegion accepts the simplest primitive within 1% (plane < cylinder < cone < sphere <
  // torus), because a cone can always imitate a cylinder and the type head calls cylinders cones.
  // CAPPED BY THE MODEL: when that pick is more complex than the model's own type and the model's
  // type still fits within 5%, the model's type wins, or a large sphere takes cylinders and cones
  // whose own fit narrowly missed 1%.
  // Measured on 80 ABC parts (test/fillEval.mjs --ordered, s1): true cylinders named Cylinder
  // 58 -> 76% on OCCT meshes, 54 -> 69% NURBS, 10 -> 57% gmsh; cones 73-80% kept; clean fills
  // 50 -> 52% OCCT, 42 -> 46% NURBS, 36 -> 42% gmsh, 31 -> 29% game.
  let kind = type;
  if (type === 'auto') {
    const cls = classifyRegion(creaseComponent(seed, p.region, adjacency, FREEFORM_CREASE), facets);
    const modelKind = fitterFor(p.type);
    let chosen = cls.type;
    if (modelKind) {
      const own = Math.min(Infinity, ...cls.candidates.filter((c) => c.type === modelKind).map((c) => c.relResidual));
      if ((chosen === 'nurbs' || ORDER_RANK[chosen] > ORDER_RANK[modelKind]) && own <= 0.05) chosen = modelKind;
    }
    out.orderedType = chosen;
    kind = chosen === 'nurbs' ? null : chosen;
  }
  if (!kind) {
    out.region = connectedPiece(seed, p.region, adjacency, FREEFORM_CREASE);
    out.notes.push('No analytic surface fits the model\'s region, so this is the model\'s own region.');
    out.ms = performance.now() - t0;
    return out;
  }
  out.fitter = kind;

  if (method === 'plain') {
    out.region = paintFrom(seed, kind, facets, adjacency, { ...grow, taken }).region;
  } else if (method === 'model') {
    out.region = paintFrom(seed, kind, facets, adjacency, { ...grow, seedRegion: p.region, taken }).region;
  } else if (method === 'veto') {
    const r = await paintWithVeto(seed, kind, mesh, p.region, runner, taken, extended);
    out.region = r.region;
    out.calls += r.calls;
  } else {
    const hyp = surfaceHypothesis(kind, p.surface, facets, seed, p.scale);
    let start = p.region;
    if (!p.surface) {
      out.notes.push('This model has no surface output; started from its region instead.');
    } else if (!hyp) {
      out.notes.push('The predicted surface was degenerate; started from the model region instead.');
    } else {
      out.hypothesis = hyp;
      const reachable = creaseComponent(seed, p.region, adjacency, creaseFor(creaseTableFor(adjacency),hyp.type));
      const grown = growAgainst(reachable, mesh, hyp.type, hyp.params, mesh.tolD, COS15, taken);
      if (grown.size >= 3) start = grown;
      else out.notes.push('Almost nothing matched the predicted surface; started from the model region.');
      out.fitter = hyp.type;
    }
    if (method === 'hyp') {
      out.region = paintFrom(seed, out.fitter, facets, adjacency, { ...grow, seedRegion: start, taken }).region;
    } else {
      const r = await paintWithVeto(seed, out.fitter, mesh, start, runner, taken, extended);
      out.region = r.region;
      out.calls += r.calls;
    }
  }

  // One connected piece. Growth starts from every seed-region facet that fits the surface, so
  // facets that did not fit can leave pieces joined only through triangles the fill rejected; a
  // group in two pieces would be a face in two pieces. Keep the seed's piece (or the largest), and
  // leave the rest unpainted for another fill.
  out.region = connectedPiece(seed, out.region, adjacency, creaseFor(creaseTableFor(adjacency),out.fitter));

  if (refine?.runner && !(taken && taken[seed])) {
    const { runner: refineRunner, ...refineOpts } = refine;
    out.region = await refineFill(seed, out.fitter, out.region, mesh, refineRunner, taken,
                                  { ...refineOpts, onMember: (m) => { out.member = m; } });
    out.refined = true;
  }

  if (out.region.size <= 1) out.notes.push(`The ${out.fitter} fit failed on this region, so only the clicked triangle is filled.`);
  if (reportFit) {
    const fit = fitRegionAs(out.fitter, out.region, facets);
    out.fit = fit.params ? { params: fit.params, relResidual: fit.relResidual } : { reason: fit.reason };
  }
  out.ms = performance.now() - t0;
  return out;
}
