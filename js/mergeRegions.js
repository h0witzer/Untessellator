/**
 * mergeRegions.js — give every painted crumb to the neighbour whose surface explains it.
 *
 * WHAT THE MEASUREMENT CHANGED. This file began as a merge pass: join two regions when one surface
 * still fits their union. test/mergeScreen.mjs took every touching pair of painted regions on 5
 * native parts (658 pairs) and stratified them by the SMALLER piece:
 *
 *     smaller piece    pairs   same face   different
 *     <= 2 tris          227          41         151
 *     3-9 tris           146          38          56
 *     10-49              157           1          76
 *     50+                128           1          87
 *
 * 98% of the merges worth making involve a piece of nine triangles or fewer, and the medium and
 * large strata hold ONE apiece. Auto-paint does not split a face into halves that need rejoining;
 * it leaves crumbs around the edges of faces. So this is not a merge between two faces — it is
 * ABSORPTION of small fragments, and the region that takes a fragment is chosen by COMPETITION
 * among its neighbours rather than by a threshold. That distinction is what makes it safe: the
 * pairs that made every threshold statistic look hopeless are precisely the losing neighbours of a
 * fragment that no threshold could place but a comparison can.
 *
 * The two strata want different evidence, which is why one rule over both was worse than either
 * (thresholds are quoted as: what share of the good merges survives, at what share of bad ones):
 *
 *   - Slivers (<= 2 triangles) are too small to fit anything: a union refit keeps 0% of them at any
 *     budget. What works is asking whether the sliver's facets LIE ON the host's surface —
 *     87% at a 5% bad budget, at 1.06 tolD.
 *   - Fragments of 3-9 triangles carry enough shape for the fitters: refitting host + fragment and
 *     requiring the union to stay under 1.4e-4 of its extent keeps 89% at a 1% bad budget, far
 *     better than any distance test there (32%).
 *
 * Nothing a model predicts reaches this decision; the fitters and the mesh do, as everywhere else.
 */

import { classifyRegion } from './gaussMap.js';
import { surfaceEval } from './regionGrow.js';

/** Types a host may carry: a fragment cannot be absorbed into something unfitted. */
const ANALYTIC = new Set(['plane', 'cylinder', 'cone', 'sphere', 'torus']);

const COS15 = Math.cos(15 * Math.PI / 180);

/** Half the median facet edge, the same length scale preparePaintPositions calls tolD. */
function facetScale(facets) {
  const sizes = [];
  for (let t = 0; t < facets.count; t++) {
    if (facets.degenerate?.[t]) continue;
    sizes.push(Math.sqrt(Math.max(facets.areas[t], 0)));
  }
  sizes.sort((a, b) => a - b);
  return 0.5 * (sizes[sizes.length >> 1] || 1);
}

/**
 * How well a host's surface explains a fragment's facets: the same test growth itself applies
 * (js/paintCore.js growAgainst) — within tolD of the surface, normals within 15 degrees.
 */
function onSurface(facets, fit, list, tolD) {
  const ev = fit?.params ? surfaceEval(fit.type, fit.params) : null;
  if (!ev) return null;
  const C = facets.centroids;
  const N = facets.normals;
  const ds = [];
  let maxAngle = 0;
  let ok = 0;
  for (const t of list) {
    const c = [C[t * 3], C[t * 3 + 1], C[t * 3 + 2]];
    const d = Math.abs(ev.distance(c)) / tolD;
    ds.push(d);
    const n = ev.normal(c);
    const cos = Math.min(1, Math.abs(n[0] * N[t * 3] + n[1] * N[t * 3 + 1] + n[2] * N[t * 3 + 2]));
    const angle = Math.acos(cos) * 180 / Math.PI;
    if (angle > maxAngle) maxAngle = angle;
    if (d <= 1 && cos >= COS15) ok++;
  }
  ds.sort((a, b) => a - b);
  return { p90: ds[Math.floor(0.9 * (ds.length - 1))], max: ds[ds.length - 1],
           angle: maxAngle, frac: ok / list.length };
}

/**
 * Absorb small regions of `seg` (an owner array, -1 unpainted) into the neighbour whose surface
 * best explains them.
 *
 * @param {Int32Array} seg          region id per triangle; NOT modified — a new array is returned
 * @param {object} facets           from preparePaintPositions
 * @param {Map} adjacency           from preparePaintPositions
 * @param {object} [opts]
 *        maxFragment  regions this size or smaller are offered to their neighbours (default 9)
 *        sliverMax    at or below this size, judge by distance alone (default 2)
 *        distTol      a sliver's facets must sit within this many tolD of the host (default 1.06)
 *        angleTol     ...and its normals within this many degrees of the host's (default 40)
 *        uRelTol      a 3-9 fragment's union with its host must fit this well (default 1.4e-4)
 *        maxDist      sanity bound on the union case, in tolD (default 2.61)
 *        tolD         the length scale; default half the median facet edge
 *        passes       absorbing a crumb can expose another; repeat this many times (default 3)
 *        onMerge      called (host, fragment, fit, detail) after each absorption
 * @returns {{seg, merges, fits}} the new owner array, how many fragments were absorbed, and the
 *          refitted surface of every host that took one (Map id -> classifyRegion result).
 */
export function mergeRegions(seg, facets, adjacency, opts = {}) {
  const maxFragment = opts.maxFragment ?? 9;
  const sliverMax = opts.sliverMax ?? 2;
  // 0.337, not the 1.06 that five native parts suggested: across 10 parts x 3 meshers, 1.06 admits
  // far more than a 5% bad-merge budget, and the 20-part run at 1.06 duly turned split faces into
  // impure ones (spill up on every mesher, area clean down on three of four).
  const distTol = opts.distTol ?? 0.337;
  const angleTol = opts.angleTol ?? 40;
  const uRelTol = opts.uRelTol ?? 1.4e-4;
  const maxDist = opts.maxDist ?? 2.61;
  const tolD = opts.tolD ?? facetScale(facets);
  const passes = opts.passes ?? 3;

  const out = Int32Array.from(seg);
  const tris = new Map();
  for (let t = 0; t < out.length; t++) {
    if (out[t] < 0 || facets.degenerate?.[t]) continue;
    const list = tris.get(out[t]);
    if (list) list.push(t);
    else tris.set(out[t], [t]);
  }

  const fitCache = new Map();
  const fitOf = (id) => {
    if (!fitCache.has(id)) fitCache.set(id, classifyRegion(tris.get(id), facets));
    return fitCache.get(id);
  };

  let merges = 0;
  const fits = new Map();

  for (let pass = 0; pass < passes; pass++) {
    // Smallest first: a crumb should pick its host before a larger fragment changes what the
    // neighbourhood looks like.
    const fragments = [...tris.keys()]
      .filter((id) => tris.get(id).length <= maxFragment)
      .sort((a, b) => tris.get(a).length - tris.get(b).length);
    let absorbedAny = false;

    for (const id of fragments) {
      const list = tris.get(id);
      if (!list || list.length > maxFragment) continue;          // absorbed, or grown, this pass

      // Who touches this fragment?
      const neighbours = new Set();
      for (const t of list) {
        for (const e of adjacency.get(t) || []) {
          const r = out[e.neighbor];
          if (r >= 0 && r !== id) neighbours.add(r);
        }
      }

      // Every neighbour bids; the fragment goes to the best bid that also passes its stratum's
      // test. A host must be strictly larger, so two crumbs never swallow each other.
      let best = null;
      for (const host of neighbours) {
        const hostTris = tris.get(host);
        if (!hostTris || hostTris.length <= list.length) continue;
        const fit = fitOf(host);
        if (!ANALYTIC.has(fit.type)) continue;
        const on = onSurface(facets, fit, list, tolD);
        if (!on) continue;

        let ok = false;
        let union = null;
        if (list.length <= sliverMax) {
          ok = on.p90 <= distTol && on.angle <= angleTol;
        } else if (on.p90 <= maxDist) {
          union = classifyRegion([...hostTris, ...list], facets);
          ok = ANALYTIC.has(union.type) && union.relResidual <= uRelTol;
        }
        if (!ok) continue;
        if (!best || on.p90 < best.on.p90) best = { host, on, fit, union };
      }
      if (!best) continue;

      const hostTris = tris.get(best.host);
      for (const t of list) out[t] = best.host;
      hostTris.push(...list);
      tris.delete(id);
      fitCache.delete(best.host);                  // the host changed shape; refit on demand
      fitCache.delete(id);
      const fit = best.union ?? fitOf(best.host);
      fits.set(best.host, fit);
      fits.delete(id);
      merges++;
      absorbedAny = true;
      opts.onMerge?.(best.host, id, fit, {
        sizeA: hostTris.length - list.length, sizeB: list.length,
        p90: best.on.p90, angle: best.on.angle, frac: best.on.frac,
        stratum: list.length <= sliverMax ? 'sliver' : 'fragment',
      });
    }
    if (!absorbedAny) break;
  }

  return { seg: out, merges, fits };
}
