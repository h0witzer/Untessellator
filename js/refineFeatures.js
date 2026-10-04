/**
 * refineFeatures.js — inputs of the fill-REFINEMENT model (experimental): the seed model's patch
 * around the click, plus what the current fill looks like on it.
 *
 * The seed model answers "which facets share this click's face" from geometry alone. A refinement
 * model answers it GIVEN a first attempt: the fill's region, the surface fitted to it, and how far
 * every facet sits from that surface. Interactive image segmentation made the same move (RITM, SAM:
 * the previous mask and the user's clicks are inputs), because correcting an answer is easier than
 * producing one. A NEGATIVE CLICK — the user pointing at a spill — is one more input channel.
 *
 * Per patch facet: the v6 inputs (seedNet.buildFeatures, curvature columns zeroed as the seed model
 * was trained), then REFINE_COLUMNS:
 *   in_region        1 if the fill holds the facet
 *   dist, angle      |distance| / tolD and normal deviation / angleTol to the fill's surface (<= 10)
 *   sdist            signed distance / tolD (-10..10): which side of the surface the facet is on
 *   ebottle, epath   the learned boundary along paths from the seed (test/boundaryScreen.mjs):
 *                    lowest achievable max probability, and lowest sum of -log(1 - p) (<= 20)
 *   elocal           highest boundary probability on the facet's own edges
 *   click, cdist     1 at a negative click, and distance from it / patch scale (10 when none)
 * Shared by the training dump (test/refineDump.mjs) and inference, so both build the same rows.
 */

import { buildFeatures, patchNeighbours, N_FEATURES_V6 } from './seedNet.js';
import { surfaceEval, lowPolyDefaults } from './regionGrow.js';

export const REFINE_COLUMNS = ['in_region', 'dist', 'angle', 'sdist', 'ebottle', 'epath', 'elocal', 'click', 'cdist'];
export const N_REFINE = N_FEATURES_V6 + REFINE_COLUMNS.length;

class Heap {
  constructor() { this.a = []; }
  get size() { return this.a.length; }
  push(k, v) {
    const a = this.a; a.push([k, v]);
    let i = a.length - 1;
    while (i > 0) { const p = (i - 1) >> 1; if (a[p][0] <= a[i][0]) break; [a[p], a[i]] = [a[i], a[p]]; i = p; }
  }
  pop() {
    const a = this.a; const top = a[0]; const last = a.pop();
    if (a.length) {
      a[0] = last; let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1; let m = i;
        if (l < a.length && a[l][0] < a[m][0]) m = l;
        if (r < a.length && a[r][0] < a[m][0]) m = r;
        if (m === i) break;
        [a[m], a[i]] = [a[i], a[m]]; i = m;
      }
    }
    return top;
  }
}

/** Dijkstra from patch index 0 (the seed) over patch edges, combining costs with `step`. */
export function patchPaths(n, nbrs, step) {
  const best = new Float64Array(n).fill(Infinity);
  best[0] = 0;
  const heap = new Heap();
  heap.push(0, 0);
  while (heap.size) {
    const [k, i] = heap.pop();
    if (k > best[i]) continue;
    for (const [j, p] of nbrs[i]) {
      const v = step(k, p);
      if (v < best[j]) { best[j] = v; heap.push(v, j); }
    }
  }
  return best;
}

const sizeCache = new WeakMap();
function medianFacetSize(facets) {
  if (!sizeCache.has(facets)) {
    const s = [];
    for (let t = 0; t < facets.count; t++) s.push(Math.sqrt(Math.max(facets.areas[t], 0)));
    s.sort((a, b) => a - b);
    sizeCache.set(facets, s[s.length >> 1] || 1);
  }
  return sizeCache.get(facets);
}

/**
 * @param {number} seed
 * @param {{region:Set<number>, type:string, params:object|null}} fill
 * @param {number} click   facet id of a negative click, or -1
 * @returns {{ features: Float32Array, ids: number[], n: number, scale: number, graph: object }}
 */
export function refineFeatures(seed, facets, adjacency, fill, click = -1, maxFacets = 512) {
  const noCurv = { valid: new Uint8Array(facets.count) };
  const { features: base, ids, n, scale } = buildFeatures(seed, facets, adjacency, maxFacets, noCurv, null, true);
  const W = N_REFINE;
  const out = new Float32Array(n * W);
  for (let i = 0; i < n; i++) out.set(base.subarray(i * N_FEATURES_V6, (i + 1) * N_FEATURES_V6), i * W);

  const lowPoly = lowPolyDefaults(adjacency);
  const tolD = (lowPoly.tolScale ?? 0.5) * medianFacetSize(facets);
  const angTol = (lowPoly.angleTol ?? 15) * Math.PI / 180;
  const ev = fill.params ? surfaceEval(fill.type, fill.params) : null;
  const C = facets.centroids, N = facets.normals;
  const index = new Map(ids.map((t, i) => [t, i]));
  const nbrs = ids.map((t) => (adjacency.get(t) || []).filter((e) => index.has(e.neighbor))
    .map((e) => [index.get(e.neighbor), e.boundary ?? 0]));
  const bottle = patchPaths(n, nbrs, (k, p) => Math.max(k, p));
  const cost = patchPaths(n, nbrs, (k, p) => k - Math.log(Math.max(1e-4, 1 - p)));
  const cx = click >= 0 ? [C[click * 3], C[click * 3 + 1], C[click * 3 + 2]] : null;

  for (let i = 0; i < n; i++) {
    const t = ids[i];
    let o = i * W + N_FEATURES_V6;
    out[o++] = fill.region.has(t) ? 1 : 0;
    const c = [C[t * 3], C[t * 3 + 1], C[t * 3 + 2]];
    if (ev) {
      const d = ev.distance(c) / tolD;
      const nn = ev.normal(c);
      const cos = Math.min(1, Math.abs(nn[0] * N[t * 3] + nn[1] * N[t * 3 + 1] + nn[2] * N[t * 3 + 2]));
      const a = Math.acos(cos) / angTol;
      out[o++] = Number.isFinite(d) ? Math.min(10, Math.abs(d)) : 10;
      out[o++] = Math.min(10, a);
      out[o++] = Number.isFinite(d) ? Math.max(-10, Math.min(10, d)) : 0;
    } else {
      out[o++] = 10; out[o++] = 10; out[o++] = 0;
    }
    out[o++] = Number.isFinite(bottle[i]) ? bottle[i] : 1;
    out[o++] = Number.isFinite(cost[i]) ? Math.min(20, cost[i]) : 20;
    let local = 0;
    for (const e of adjacency.get(t) || []) local = Math.max(local, e.boundary ?? 0);
    out[o++] = local;
    out[o++] = t === click ? 1 : 0;
    out[o++] = cx ? Math.min(10, Math.hypot(c[0] - cx[0], c[1] - cx[1], c[2] - cx[2]) / scale) : 10;
  }
  return { features: out, ids, n, scale, graph: patchNeighbours(ids, adjacency) };
}

/**
 * Where a user would click to say "not this": the facet nearest the centroid of the largest
 * connected piece of the fill lying on a different surface type from the seed's. -1 if none.
 * Simulation only (it reads the truth `T`).
 */
export function simulatedClick(seed, region, adjacency, facets, T) {
  const seen = new Set();
  let best = null;
  for (const t of region) {
    if (seen.has(t) || T[t] === T[seed]) continue;
    const piece = [t];
    seen.add(t);
    for (let k = 0; k < piece.length; k++) {
      for (const e of adjacency.get(piece[k]) || []) {
        const u = e.neighbor;
        if (!seen.has(u) && region.has(u) && T[u] !== T[seed]) { seen.add(u); piece.push(u); }
      }
    }
    if (!best || piece.length > best.length) best = piece;
  }
  if (!best) return -1;
  const C = facets.centroids;
  let mx = 0, my = 0, mz = 0;
  for (const t of best) { mx += C[t * 3]; my += C[t * 3 + 1]; mz += C[t * 3 + 2]; }
  mx /= best.length; my /= best.length; mz /= best.length;
  let pick = best[0], bd = Infinity;
  for (const t of best) {
    const d = (C[t * 3] - mx) ** 2 + (C[t * 3 + 1] - my) ** 2 + (C[t * 3 + 2] - mz) ** 2;
    if (d < bd) { bd = d; pick = t; }
  }
  return pick;
}
