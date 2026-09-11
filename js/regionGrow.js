/**
 * regionGrow.js — primitive-driven segmentation.
 *
 * WHY THIS EXISTS
 *
 * Crease-angle flood fill decides whether two adjacent facets belong together by
 * comparing their normals to EACH OTHER. That fails on real tessellated CAD in
 * two directions at once:
 *
 *   - Tangent-continuous faces (a fillet meeting a wall, a blend running into a
 *     plane) have a near-zero dihedral angle, so they merge no matter how low
 *     the threshold goes.
 *   - A coarsely tessellated cylinder has large dihedral angles BETWEEN ITS OWN
 *     facets, so it shatters unless the threshold is raised.
 *
 * On the WatchStand fixtures those two requirements overlap — the dihedral
 * histogram is continuous, with no separating value — so no single crease angle
 * can work, and ~41% of the mesh ends up in merged blobs.
 *
 * This module instead grows a region by testing each candidate facet against a
 * FITTED SURFACE: does its centroid lie on the surface, and does its normal
 * agree with the surface normal there? Tangency between neighbours is then
 * irrelevant — a fillet's facets simply do not lie on the adjoining plane.
 *
 * Follows the shape of Schnabel, Wahl & Klein, "Efficient RANSAC for
 * Point-Cloud Shape Detection" (2007), specialised to a connected mesh so
 * growth can use edge adjacency rather than a spatial index.
 *
 * STATUS: work in progress, harness-only — NOT wired into the browser app.
 * Scored against test/fixtures/WatchStandTruth.step (77 faces, of which 62 are
 * reachable with the primitives we have; test/oracle.mjs measures that ceiling):
 *
 *     crease-angle flood fill ............ 24/77
 *     this module, best parameters ....... 26/77
 *
 * NEGATIVE RESULT, do not redo: restoring RANSAC's "propose against the whole
 * mesh, then commit best-supported first" ordering was tried and scored WORSE
 * (14/77). Proposals grown against an unclaimed mesh over-grow across face
 * boundaries, so ranking by support hands the part to whichever candidate
 * over-grew the most. Support-ranking needs an over-growth penalty (Schnabel
 * uses an MDL-style score) before it can help here. Tightening the distance
 * tolerance does not help either — the score is flat at ~26 across a sweep of
 * tolScale 0.05-1.5 against angleTol 8-25.
 *
 * The remaining gap to 62 is commit order and seeding, not fitting: oracle.mjs
 * shows every one of the 62 analytic faces has >=10 facets lying on its exact
 * ground-truth surface, and the fitters recover a 0.5 mm radius to 1e-6 when
 * handed the right facets.
 */

import { fitRegionAs } from './gaussMap.js';
import { dot, cross, sub, norm, normalize } from './meshCore.js';

const TYPES = ['plane', 'cylinder', 'cone', 'sphere'];

/**
 * Signed distance to a primitive, and its unit normal at the closest point.
 * @returns {{distance:(p:number[])=>number, normal:(p:number[])=>number[]}|null}
 */
export function surfaceEval(type, p) {
  if (!p) return null;

  if (type === 'plane') {
    return {
      distance: q => dot(sub(q, p.origin), p.normal),
      normal: () => p.normal,
    };
  }

  if (type === 'cylinder') {
    const { axis, axisPoint, radius } = p;
    const radial = (q) => {
      const d = sub(q, axisPoint);
      const z = dot(d, axis);
      return [d[0] - z * axis[0], d[1] - z * axis[1], d[2] - z * axis[2]];
    };
    return {
      distance: q => norm(radial(q)) - radius,
      normal: q => normalize(radial(q)),
    };
  }

  if (type === 'cone') {
    const { axis, apex, halfAngle } = p;
    const ca = Math.cos(halfAngle), sa = Math.sin(halfAngle);
    const frame = (q) => {
      const d = sub(q, apex);
      const z = dot(d, axis);
      const r = [d[0] - z * axis[0], d[1] - z * axis[1], d[2] - z * axis[2]];
      return { z, r, rn: norm(r) };
    };
    return {
      // Perpendicular distance to the cone surface in the (r, z) half-plane.
      distance: (q) => { const f = frame(q); return f.rn * ca - f.z * sa; },
      normal: (q) => {
        const f = frame(q);
        if (f.rn < 1e-12) return axis;
        const u = [f.r[0] / f.rn, f.r[1] / f.rn, f.r[2] / f.rn];
        // Outward normal tilts away from the axis by the half-angle.
        return normalize([
          u[0] * ca - axis[0] * sa,
          u[1] * ca - axis[1] * sa,
          u[2] * ca - axis[2] * sa,
        ]);
      },
    };
  }

  if (type === 'sphere') {
    const { center, radius } = p;
    return {
      distance: q => norm(sub(q, center)) - radius,
      normal: q => normalize(sub(q, center)),
    };
  }

  return null;
}

const centroidOf = (facets, t) =>
  [facets.centroids[t * 3], facets.centroids[t * 3 + 1], facets.centroids[t * 3 + 2]];
const normalOf = (facets, t) =>
  [facets.normals[t * 3], facets.normals[t * 3 + 1], facets.normals[t * 3 + 2]];

/**
 * Flood outward from a seed set, accepting facets that lie on the given surface.
 *
 * @param {Set<number>} seed
 * @param {object} facets
 * @param {Map} adjacency
 * @param {string} type
 * @param {object} params
 * @param {number} tolD    max |distance| from the surface
 * @param {number} cosTolA min |cos| between facet normal and surface normal
 * @param {Uint8Array} taken  facets already committed to another region
 */
function growOn(seed, facets, adjacency, type, params, tolD, cosTolA, taken) {
  const ev = surfaceEval(type, params);
  if (!ev) return new Set();

  const fits = (t) => {
    const c = centroidOf(facets, t);
    if (Math.abs(ev.distance(c)) > tolD) return false;
    return Math.abs(dot(normalOf(facets, t), ev.normal(c))) >= cosTolA;
  };

  const region = new Set();
  const queue = [];
  for (const s of seed) if (!taken[s] && fits(s)) { region.add(s); queue.push(s); }

  while (queue.length) {
    const t = queue.pop();
    for (const e of adjacency.get(t) || []) {
      const nb = e.neighbor;
      if (taken[nb] || region.has(nb)) continue;
      if (!fits(nb)) continue;
      region.add(nb);
      queue.push(nb);
    }
  }
  return region;
}

/** Small connected neighbourhood around a seed, used to propose a primitive. */
function neighbourhood(seed, adjacency, taken, want) {
  const out = new Set([seed]);
  const queue = [seed];
  while (queue.length && out.size < want) {
    const t = queue.shift();
    for (const e of adjacency.get(t) || []) {
      if (taken[e.neighbor] || out.has(e.neighbor)) continue;
      out.add(e.neighbor);
      queue.push(e.neighbor);
      if (out.size >= want) break;
    }
  }
  return out;
}

/** Median edge length, used to scale the distance tolerance to the mesh. */
function medianFacetSize(facets) {
  const s = [];
  for (let t = 0; t < facets.count; t++) s.push(Math.sqrt(Math.max(facets.areas[t], 0)));
  s.sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)] || 1;
}

/**
 * Segment a mesh into analytic primitives by RANSAC-style region growing.
 *
 * @param {object} facets     from meshCore.buildFacets
 * @param {Map} adjacency     from meshCore.buildAdjacency
 * @param {object} [opts]
 * @param {number} [opts.tolScale=1.5]   distance tolerance, in median facet sizes
 * @param {number} [opts.angleTol=25]    max facet-to-surface normal deviation, degrees
 * @param {number} [opts.minRegion=6]    smallest region to accept as a primitive
 * @param {number} [opts.seedSize=16]    facets in a proposal neighbourhood
 * @param {number} [opts.supportSlack=0.9] keep the simplest type within this
 *        fraction of the best support
 * @returns {Array<{region:Set<number>, type:string, params:object, relResidual:number}>}
 */
export function growPrimitives(facets, adjacency, opts = {}) {
  const tolScale = opts.tolScale ?? 1.5;
  const angleTol = opts.angleTol ?? 25;
  const minRegion = opts.minRegion ?? 6;
  const seedSize = opts.seedSize ?? 16;
  const supportSlack = opts.supportSlack ?? 0.9;

  const tolD = tolScale * medianFacetSize(facets);
  const cosTolA = Math.cos(angleTol * Math.PI / 180);

  const taken = new Uint8Array(facets.count);
  const accepted = [];

  // Largest facets first: they carry the most evidence and belong to the
  // biggest faces, which we want to claim before slivers fragment them.
  const order = [...Array(facets.count).keys()]
    .sort((a, b) => facets.areas[b] - facets.areas[a]);

  // Single greedy pass. Re-proposing every seed on every round is O(n x rounds)
  // and far too slow to run in a browser; seeding from the largest untaken facet
  // and committing immediately is O(n) proposals, and the refit-and-regrow step
  // below recovers most of what the exhaustive search would have found.
  for (const seed of order) {
    if (taken[seed]) continue;

    const nb = neighbourhood(seed, adjacency, taken, seedSize);
    if (nb.size < 4) continue;

    const candidates = [];
    for (const type of TYPES) {
      const fit = fitRegionAs(type, nb, facets);
      if (fit.degenerate || !fit.params) continue;
      if (!(fit.relResidual < 0.05)) continue;

      const region = growOn(nb, facets, adjacency, type, fit.params, tolD, cosTolA, taken);
      if (region.size < minRegion) continue;

      // Refit on the full support, then regrow — the seed patch is a poor
      // estimate of a large face, and a better surface claims more of it.
      const refit = fitRegionAs(type, region, facets);
      let final = region, fParams = fit.params, fRes = fit.relResidual;
      if (!refit.degenerate && refit.params) {
        const grown = growOn(region, facets, adjacency, type, refit.params,
                             tolD, cosTolA, taken);
        if (grown.size >= region.size) final = grown;
        fParams = refit.params;
        fRes = refit.relResidual;
      }
      candidates.push({ region: final, type, params: fParams, relResidual: fRes });
    }

    // Parsimony: a more general primitive always fits at least as well and will
    // happily grow across a boundary, so prefer the SIMPLEST type whose support
    // is within `supportSlack` of the best. Choosing purely by region size makes
    // cylinders absorb planes and produces far more of them than the part has.
    let best = null;
    if (candidates.length) {
      const maxSize = Math.max(...candidates.map(c => c.region.size));
      const viable = candidates
        .filter(c => c.region.size >= supportSlack * maxSize)
        .sort((a, b) => TYPES.indexOf(a.type) - TYPES.indexOf(b.type)
                     || b.region.size - a.region.size);
      best = viable[0] ?? null;
    }

    if (!best) continue;
    for (const t of best.region) taken[t] = 1;
    accepted.push(best);
  }

  // Whatever is left over, grouped into connected components, is freeform.
  const leftovers = [];
  for (let t = 0; t < facets.count; t++) {
    if (taken[t]) continue;
    const comp = new Set([t]);
    const q = [t];
    taken[t] = 1;
    while (q.length) {
      const c = q.pop();
      for (const e of adjacency.get(c) || []) {
        if (taken[e.neighbor]) continue;
        taken[e.neighbor] = 1;
        comp.add(e.neighbor);
        q.push(e.neighbor);
      }
    }
    leftovers.push({ region: comp, type: 'nurbs', params: null, relResidual: Infinity });
  }

  return accepted.concat(leftovers);
}

// ── Interactive painting ─────────────────────────────────────────────────────

/**
 * Bucket-fill from a clicked triangle, bounded by SURFACE FIT rather than by
 * crease angle.
 *
 * Crease-angle fill asks whether a triangle's neighbour is parallel to it, which
 * is the wrong question at a tangent junction: a cone running into a tangent
 * planar face has ~0 dihedral angle at the boundary, so the fill spills across
 * no matter how tight the threshold. Asking instead whether each triangle lies
 * on the surface BEING PAINTED stops the fill exactly at the junction, because
 * the planar face's facets are not on the cone.
 *
 * This is the same growth used by growPrimitives(), but with the two things that
 * make automatic segmentation hard supplied by the user: which primitive, and
 * where to start.
 *
 * @param {number} seedTri   triangle index under the cursor
 * @param {string} type      'plane' | 'cylinder' | 'cone' | 'sphere'
 * @param {object} facets    from meshCore.buildFacets
 * @param {Map} adjacency    from meshCore.buildAdjacency
 * @param {object} [opts]
 * @param {number} [opts.tolScale=0.5]  distance tolerance, in median facet sizes
 * @param {number} [opts.angleTol=15]   max facet-to-surface normal deviation, deg
 * @param {number} [opts.seedSize=12]   facets used for the first fit
 * @param {number} [opts.iterations=4]  grow/refit rounds
 * @param {Set<number>} [opts.seedRegion]  paint from an existing selection
 *        instead of a single triangle (shift-click to extend)
 * @returns {{region:Set<number>, type:string, params:object|null,
 *           relResidual:number, degenerate?:boolean, reason?:string}}
 *
 * STATUS (test/paint.mjs): works where crease fill cannot, fails where the face
 * is small relative to its neighbours.
 *
 *   tangent-fillet, which crease fill merges into one 4512-triangle blob
 *   spanning three faces:
 *     paint PLANE on the base ....... purity 100%, recall 100%
 *     paint CYLINDER on the wall .... purity  75%, recall 100%, r=19.986 (true 20)
 *     paint PLANE on the top ........ purity  73%, recall 100%
 *   The ~25% over-reach is the tangent blend: near the junction the fillet
 *   really is inside tolerance of the cylinder. Tightening tolScale trades
 *   that purity back against recall.
 *
 *   WatchStandMedium conical faces .. 0/4, half-angles come back 22, 0.6, 0.57,
 *   89 deg. NOT a tolerance problem: flat at 0/4 across seedSize 12-80 x
 *   tolScale 0.5-1.5. The cause is the seed patch — neighbourhood() is a plain
 *   adjacency BFS, so on a face that is small next to its neighbours it crosses
 *   the boundary and the first fit is poisoned before growth starts.
 *
 * NEXT: seed from a CREASE-bounded patch (locally coherent, never crosses a
 * sharp edge) and then switch to surface-bounded growth. Crease angle is the
 * wrong tool for the boundary but the right one for picking a clean starting
 * patch; this module currently uses it for neither.
 */
export function paintFrom(seedTri, type, facets, adjacency, opts = {}) {
  const tolD = (opts.tolScale ?? 0.5) * medianFacetSize(facets);
  const cosTolA = Math.cos((opts.angleTol ?? 15) * Math.PI / 180);
  const iterations = opts.iterations ?? 4;
  const none = new Uint8Array(facets.count);

  let region = opts.seedRegion && opts.seedRegion.size
    ? new Set(opts.seedRegion)
    : neighbourhood(seedTri, adjacency, none, opts.seedSize ?? 12);

  let fit = fitRegionAs(type, region, facets);
  if (fit.degenerate || !fit.params) {
    return { region: new Set([seedTri]), type, params: null,
             relResidual: Infinity, degenerate: true, reason: fit.reason };
  }

  // A fit from a dozen triangles around the click is a poor estimate of a whole
  // face, so alternate growing and refitting until the region stops changing.
  // Each refit sees more of the surface and claims more of it in turn.
  for (let i = 0; i < iterations; i++) {
    const grown = growOn(region, facets, adjacency, type, fit.params,
                         tolD, cosTolA, none);
    if (grown.size === 0) break;

    const refit = fitRegionAs(type, grown, facets);
    if (refit.degenerate || !refit.params) break;

    const stable = grown.size === region.size;
    region = grown;
    fit = refit;
    if (stable) break;
  }

  return { region, type, params: fit.params, relResidual: fit.relResidual };
}
