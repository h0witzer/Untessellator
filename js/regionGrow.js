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
 *
 * CORRECTION — that last sentence WAS true of cylinders and false of cones.
 *
 * It had been measured on a cylinder and generalised. test/fitoracle.mjs now measures
 * it per type by handing fitRegionAs the EXACT ground-truth facets. Cones originally
 * scored 0 of 4 from perfect input — 22.34 deg for a 9.85, 86.00 for a 45, and a
 * "no measurable taper" rejection for a 59 whose patch was exactly conical.
 *
 * Root cause was in gaussMap.js, in two parts, both now fixed:
 *
 *   - The half-angle came from an R-vs-Z regression, which needs the patch to span
 *     axial extent. A narrow annular band has near-constant R, so the slope read ~0
 *     and a perfectly conical patch was rejected. It now comes from the normals,
 *     |n . axis| = sin(halfAngle), which holds on any patch however narrow.
 *   - The axis was refined by pattern search on the POSITIONAL residual, which has a
 *     degenerate minimum: a wrong axis with a compensating half-angle fits a patch
 *     more closely than the true one (measured 8.4e-4 against 4.0e-3 on the 45 deg
 *     cone), so the search actively walked away from a correct Gauss-map seed. The
 *     axis is now chosen by the variance of (n . axis), which is exactly zero on a
 *     cone's true axis and needs no weighting against anything.
 *
 * After the fix, from perfect input: 9.66 / 8.49 / 58.96 / 44.99 deg against truths of
 * 9.85 / 9.85 / 59.00 / 45.00 — 4 of 4.
 *
 * Downstream, one click each (test/paint.mjs, test/paintnet.mjs):
 *
 *     baseline paintFrom ....................... 0/4 -> 1/4   (fitter fix alone)
 *     paintFrom seeded by a learned proposer ... 1/4 -> 3/4
 *
 * The learned proposer is optional and lives behind js/seedNet.js; without it the
 * converter behaves exactly as before. It helps by replacing the initial patch that
 * the NEGATIVE RESULT above names — neighbourhood()'s plain adjacency BFS.
 *
 * Two further fixes followed from asking why cones were hard at all, given that a CAD
 * tessellator puts its vertices exactly ON the surface:
 *
 *   - Positional terms now fit WELDED VERTICES rather than facet centroids. An
 *     inscribed triangulation puts every centroid strictly inside a convex surface, by
 *     0.69% of the radius at coarse tessellation and 0.06% at fine — a density-
 *     dependent bias that does not average out. Fitting the tangent-fillet wall to
 *     centroids gave r=19.990482 for a true 20; to vertices it gives 20.000000.
 *   - The apex now comes from TANGENT-PLANE CONCURRENCY, a single 3x3 solve. Every
 *     tangent plane of a cone contains the apex, and a tessellator's triangles each
 *     carry an edge along a ruling, so their planes do too: measured |n . (apex - c)|
 *     has median 8.3e-10 on a cleanly tessellated cone. Pinning the apex removes the
 *     slide-to-infinity direction that made the positional objective degenerate.
 *
 * The apex fix mostly shows up in GROWTH, not in the final angle: because growOn now
 * tests against a correct cone from the first iteration, one click recovers 65.4%,
 * 80.7% and 100.0% of three cone faces at 100% purity, against 14.7%, 22.1% and 23.3%
 * before.
 *
 * The axis criterion is also TRIMMED, dropping the worst 20% by deviation, because a
 * variance has no resistance to outliers: a 90.6%-pure region — five foreign facets out
 * of 53 — previously swung the axis by 68 degrees. Swept across 0.05-0.4 the result is
 * flat, so the fraction only has to exceed the impurity a segmenter leaves behind.
 *
 * That 45 deg cone is still the open case: 29.58 deg of error before trimming, 6.38
 * after, against a 2 deg bar. Everything else lands inside 0.15 deg.
 *
 * Method note worth keeping: run test/fitoracle.mjs before optimising any proposer.
 * If a primitive cannot be recovered from perfect input, nothing feeding it can help.
 */

import { fitRegionAs } from './gaussMap.js';
import { dot, cross, sub, norm, normalize } from './meshCore.js';

const TYPES = ['plane', 'cylinder', 'cone', 'sphere', 'torus'];

/** Any unit vector perpendicular to a. Used only for the degenerate on-axis case. */
function perpOf(a) {
  const k = Math.abs(a[0]) < Math.abs(a[1])
    ? (Math.abs(a[0]) < Math.abs(a[2]) ? 0 : 2)
    : (Math.abs(a[1]) < Math.abs(a[2]) ? 1 : 2);
  const e = [0, 0, 0]; e[k] = 1;
  return normalize(cross(a, e));
}

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

  if (type === 'torus') {
    const { axis, center, majorRadius, minorRadius } = p;
    // Nearest point on the tube-centre circle, then distance to the tube around it.
    const tube = (q) => {
      const d = sub(q, center);
      const z = dot(d, axis);
      const r = [d[0] - z * axis[0], d[1] - z * axis[1], d[2] - z * axis[2]];
      const rn = norm(r);
      // On the axis itself the nearest tube point is not unique; pick any radial.
      const u = rn > 1e-12 ? [r[0] / rn, r[1] / rn, r[2] / rn] : perpOf(axis);
      const c = [center[0] + majorRadius * u[0],
                 center[1] + majorRadius * u[1],
                 center[2] + majorRadius * u[2]];
      return { c, z, rn };
    };
    return {
      distance: (q) => {
        const t = tube(q);
        return norm(sub(q, t.c)) - minorRadius;
      },
      normal: (q) => {
        const t = tube(q);
        const v = sub(q, t.c);
        const vn = norm(v);
        return vn > 1e-12 ? [v[0] / vn, v[1] / vn, v[2] / vn] : axis;
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
function growOn(seed, facets, adjacency, type, params, tolD, cosTolA, taken, creaseDeg = Infinity) {
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
      if (e.angle >= creaseDeg) continue;   // a crease is a wall, whatever the fit says
      if (!fits(nb)) continue;
      region.add(nb);
      queue.push(nb);
    }
  }
  return region;
}

/** Small connected neighbourhood around a seed, used to propose a primitive. */
function neighbourhood(seed, adjacency, taken, want, creaseDeg = Infinity) {
  const out = new Set([seed]);
  const queue = [seed];
  while (queue.length && out.size < want) {
    const t = queue.shift();
    for (const e of adjacency.get(t) || []) {
      if (taken[e.neighbor] || out.has(e.neighbor) || e.angle >= creaseDeg) continue;
      out.add(e.neighbor);
      queue.push(e.neighbor);
      if (out.size >= want) break;
    }
  }
  return out;
}

/**
 * CREASE BARRIER: paintFrom never grows across an edge whose dihedral angle is at or above
 * the angle for the surface being painted. A drifting refit cannot carry a fill over a sharp
 * edge, whatever the fit then says about the triangles beyond it.
 *
 * The angle depends on the type because a curved face is tessellated with real dihedrals
 * INSIDE it, while a plane has none. Measured on 80 ABC parts (test/creaseStats.mjs): the
 * within-face dihedral of planes is 0.0 at p99 on every mesher, so 2 degrees costs no reach;
 * cylinders, cones and spheres keep their full reach from about 20-30 degrees on OCCT meshes
 * but need 30 on gmsh and game-style meshes; tori need 45.
 *
 * End to end (test/fillEval.mjs --declared-only, true type, no model), clean fills went from
 * 19% to 56% on OCCT meshes, 12 to 48% on NURBS-converted, 15 to 44% on gmsh and 15 to 31%
 * on game-style meshes. Planes gained most (OCCT plane clean 8% -> 90%). Alternatives on the
 * same clicks: one 30-degree barrier for every type, 50/42/32/27%; plane 5 instead of 2,
 * 51/44/43/30%; curved types at 20 or 15 instead of 30, down to 33% on gmsh and 19% on game,
 * because those meshes put 20-30 degree dihedrals inside a single cylinder.
 * The plane angle assumes CAD-exported meshes: a noisy scan's planes are not flat to 2 degrees.
 *
 * A barrier can never see a TANGENT boundary (a fillet meeting a face), which has no crease;
 * about half the neighbouring faces of different types in ABC meet that way. Those are still
 * the surface test's job.
 */
export const CREASE_BARRIER = Object.freeze({ plane: 2, cylinder: 30, cone: 30, sphere: 30, torus: 45 });

/**
 * The dihedral angle, in degrees, at which growth for `type` treats an edge as a wall.
 * `crease` is a number for every type, a { type: degrees } table, or null for no wall.
 */
export function creaseFor(crease, type) {
  if (typeof crease === 'number') return crease;
  return crease?.[type] ?? Infinity;
}

/**
 * The triangles of `allowed` reachable from `seed` without crossing an edge whose dihedral is
 * at or above `creaseDeg`. The seed itself is always included. With no barrier (Infinity)
 * nothing is cut.
 */
export function creaseComponent(seed, allowed, adjacency, creaseDeg) {
  const out = new Set([seed]);
  if (!Number.isFinite(creaseDeg)) {
    for (const t of allowed) out.add(t);
    return out;
  }
  const stack = [seed];
  while (stack.length) {
    const t = stack.pop();
    for (const e of adjacency.get(t) || []) {
      if (e.angle >= creaseDeg || out.has(e.neighbor) || !allowed.has(e.neighbor)) continue;
      out.add(e.neighbor);
      stack.push(e.neighbor);
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
 * @param {string} type      'plane' | 'cylinder' | 'cone' | 'sphere' | 'torus'
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
 *     paint PLANE on the top ........ purity 100%, recall 100%  (73% before CREASE_BARRIER)
 *   The cylinder's ~25% over-reach is the tangent blend: near the junction the
 *   fillet really is inside tolerance of the cylinder, and a tangent junction
 *   has no crease for the barrier to stop at. Tightening tolScale trades that
 *   purity back against recall.
 *
 *   WatchStandMedium conical faces .. 3/4 with CREASE_BARRIER (was 1/4): the
 *   59 and 45 deg cones now come back to 0.04 deg. The failure before was the
 *   seed patch: neighbourhood() was a plain adjacency BFS, so on a face small
 *   next to its neighbours it crossed the boundary and poisoned the first fit.
 *   The seed patch is now crease-bounded, and so is growth. The remaining miss
 *   is a 9.85 deg cone whose 12-triangle seed recovers only 3.7 deg.
 *
 * Crease angle is still the wrong tool at a TANGENT boundary (no crease), so it
 * is a wall added to the surface test, never a replacement for it.
 */
export function paintFrom(seedTri, type, facets, adjacency, opts = {}) {
  const tolD = (opts.tolScale ?? 0.5) * medianFacetSize(facets);
  const cosTolA = Math.cos((opts.angleTol ?? 15) * Math.PI / 180);
  const iterations = opts.iterations ?? 4;
  const creaseDeg = creaseFor(opts.crease === undefined ? CREASE_BARRIER : opts.crease, type);
  // opts.taken: facets already painted into another group, which growth never enters.
  const none = opts.taken ?? new Uint8Array(facets.count);

  // A supplied region (a model's proposal) is cut back to the part the click can reach without
  // crossing a crease, or a proposal spanning a bend would carry the fill over it.
  let region = opts.seedRegion && opts.seedRegion.size
    ? creaseComponent(seedTri, opts.seedRegion, adjacency, creaseDeg)
    : neighbourhood(seedTri, adjacency, none, opts.seedSize ?? 12, creaseDeg);
  if (opts.taken) for (const t of region) if (t !== seedTri && opts.taken[t]) region.delete(t);

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
                         tolD, cosTolA, none, creaseDeg);
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
