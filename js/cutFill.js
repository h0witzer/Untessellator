/**
 * cutFill.js — one fill chosen by a MINIMUM CUT instead of a flood (experimental).
 *
 * paintFrom (regionGrow.js) floods: a facet joins when it fits the surface and the edge it is reached
 * over is not a wall. Every decision is local, so a boundary the learned wall scores only in part
 * leaks through its gaps, and a false wall inside a face can split it. Here each grow/refit round
 * instead labels the reachable facets by minimising
 *
 *     E = sum over facets in the fill of  w_t (s_t - 1)          (s_t: residual to the fitted surface,
 *                                                                 1 = at paintFrom's tolerance)
 *       + lambda * sum over cut edges of  (len_e / median len) * (1 - p_e)^gamma
 *
 * where p_e is the learned boundary probability (js/edgeWall.js annotateEdges) and w_t the facet's
 * area over the median. A facet that fits pays for its exclusion, one that does not pays for its
 * inclusion, and the region's outline pays wherever the edge model says "inside a face". So the
 * decision is made per boundary, not per edge: closing a gap costs a few edges, while leaking through
 * it costs every poorly fitting facet beyond. This is the recipe of Kalogerakis et al., "Learning 3D
 * Mesh Segmentation and Labeling" (SIGGRAPH 2010): a per-facet term, a learned boundary term, and a
 * graph cut to combine them.
 *
 * Without edge scores the cut cost is plain length, and the crease barrier stays a hard wall either
 * way, so it runs with no model at all.
 */

import { fitRegionAs } from './gaussMap.js';
import { surfaceEval, creaseFor, creaseTableFor, lowPolyDefaults, creaseComponent } from './regionGrow.js';
import { MaxFlow } from './maxflow.js';

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
const lenCache = new WeakMap();
function medianEdgeLength(graph) {
  if (!lenCache.has(graph)) {
    const s = Array.from(graph.length).filter((v) => v > 0).sort((a, b) => a - b);
    lenCache.set(graph, s[s.length >> 1] || 1);
  }
  return lenCache.get(graph);
}

function seedNeighbourhood(seed, adjacency, taken, want, creaseDeg) {
  const out = new Set([seed]);
  const queue = [seed];
  while (queue.length && out.size < want) {
    const t = queue.shift();
    for (const e of adjacency.get(t) || []) {
      if ((taken && taken[e.neighbor]) || out.has(e.neighbor) || e.angle >= creaseDeg) continue;
      out.add(e.neighbor);
      queue.push(e.neighbor);
      if (out.size >= want) break;
    }
  }
  return out;
}

/**
 * Fill from `seedTri` as `type`, each round a minimum cut. Options as paintFrom (tolScale, angleTol,
 * iterations, crease, seedSize, seedRegion, taken), plus:
 *   lambda     boundary weight per median edge length (default 1)
 *   gamma      exponent on (1 - p) (default 1)
 *   maxScore   facets with a residual score above this are left out of the graph (default 9)
 *   area       weight facets by area / median area, clamped 0.2-5 (default true)
 *   useEdges   read e.boundary into the cut cost (default true)
 *   norm       'max' scores a facet by the worse of (distance / tolerance)^2 and (angle / tolerance)^2,
 *              so score <= 1 is exactly paintFrom's acceptance box; 'sum' adds them (an ellipse
 *              inside that box, stricter at the corners). Default 'sum'.
 *   member     optional Map facet -> membership probability, added as alpha * w * 2 (p - 0.5)
 *   alpha      weight of `member` (default 0)
 *   memberReplacesFit  where `member` has a value, use it INSTEAD of the fit residual
 * `graph` is js/edgeGraph.js's attachEdgeGraph for this mesh (edge ids and lengths).
 */
export function cutFrom(seedTri, type, facets, adjacency, graph, opts = {}) {
  const lowPoly = lowPolyDefaults(adjacency);
  const tolD = (opts.tolScale ?? lowPoly.tolScale ?? 0.5) * medianFacetSize(facets);
  const angleTol = (opts.angleTol ?? lowPoly.angleTol ?? 15) * Math.PI / 180;
  const iterations = opts.iterations ?? 4;
  const creaseDeg = creaseFor(opts.crease === undefined ? creaseTableFor(adjacency) : opts.crease, type);
  const lambda = opts.lambda ?? 1;
  const gamma = opts.gamma ?? 1;
  const maxScore = opts.maxScore ?? 9;
  const useArea = opts.area ?? true;
  const useEdges = opts.useEdges ?? true;
  const member = opts.member ?? null;
  const alpha = opts.alpha ?? 0;
  // memberReplacesFit: where `member` has a value it IS the per-facet term (a learned model's verdict),
  // and the fit residual only decides facets outside the model's patch.
  const replaceFit = !!opts.memberReplacesFit;
  const maxNorm = opts.norm === 'max';
  const taken = opts.taken ?? null;
  const medLen = medianEdgeLength(graph);
  const medArea = medianFacetSize(facets) ** 2;
  const C = facets.centroids, N = facets.normals;

  let region;
  if (opts.seedRegion && opts.seedRegion.size) {
    const allowed = taken ? new Set([...opts.seedRegion].filter((t) => !taken[t])) : opts.seedRegion;
    region = creaseComponent(seedTri, allowed, adjacency, creaseDeg);
  } else {
    region = seedNeighbourhood(seedTri, adjacency, taken, opts.seedSize ?? 12, creaseDeg);
  }
  let fit = fitRegionAs(type, region, facets);
  if (fit.degenerate || !fit.params) {
    return { region: new Set([seedTri]), type, params: null, relResidual: Infinity, degenerate: true, reason: fit.reason };
  }

  const weight = (t) => (useArea ? Math.min(5, Math.max(0.2, facets.areas[t] / medArea)) : 1);
  const edgeCost = (e) => {
    const len = graph.length[e.id] > 0 ? graph.length[e.id] / medLen : 1;
    const p = useEdges && e.boundary !== undefined ? e.boundary : 0;
    return lambda * len * Math.pow(Math.max(0, 1 - p), gamma);
  };

  for (let it = 0; it < iterations; it++) {
    const ev = surfaceEval(type, fit.params);
    if (!ev) break;
    const score = new Map();
    const scoreOf = (t) => {
      let s = score.get(t);
      if (s === undefined) {
        const c = [C[t * 3], C[t * 3 + 1], C[t * 3 + 2]];
        const d = ev.distance(c) / tolD;
        const n = ev.normal(c);
        const cos = Math.min(1, Math.abs(n[0] * N[t * 3] + n[1] * N[t * 3 + 1] + n[2] * N[t * 3 + 2]));
        const a = Math.acos(cos) / angleTol;
        s = !Number.isFinite(d) ? Infinity : maxNorm ? Math.max(d * d, a * a) : d * d + a * a;
        score.set(t, s);
      }
      return s;
    };

    // Domain: facets reachable from the region without crossing a crease or a taken facet, through
    // facets whose score stays within maxScore. Region facets are always in.
    const index = new Map();
    const order = [];
    const push = (t) => { index.set(t, order.length); order.push(t); };
    for (const t of region) if (!(taken && taken[t])) push(t);
    if (!index.has(seedTri)) push(seedTri);
    for (let k = 0; k < order.length; k++) {
      const t = order[k];
      for (const e of adjacency.get(t) || []) {
        const b = e.neighbor;
        if (index.has(b) || e.angle >= creaseDeg || (taken && taken[b])) continue;
        if (!(scoreOf(b) <= maxScore)) continue;
        push(b);
      }
    }

    const g = new MaxFlow(order.length, order.length * 2);
    for (let i = 0; i < order.length; i++) {
      const t = order[i];
      const s = Math.min(scoreOf(t), maxScore);
      let benefit = weight(t) * (1 - s);
      if (member && alpha) {
        const pm = member.get(t);
        if (pm !== undefined) benefit = (replaceFit ? 0 : benefit) + alpha * weight(t) * 2 * (pm - 0.5);
      }
      if (benefit > 0) g.addTweights(i, benefit, 0); else g.addTweights(i, 0, -benefit);
      for (const e of adjacency.get(t) || []) {
        const b = e.neighbor;
        if (e.angle >= creaseDeg || (taken && taken[b])) continue;   // a free cut
        const j = index.get(b);
        if (j === undefined) {
          g.addTweights(i, 0, edgeCost(e));                          // b is out: t in cuts this edge
        } else if (j > i) {
          const c = edgeCost(e);
          g.addEdge(i, j, c, c);
        }
      }
    }
    g.addTweights(index.get(seedTri), 1e12, 0);
    g.maxflow();

    const chosen = new Set();
    for (let i = 0; i < order.length; i++) if (g.onSource(i)) chosen.add(order[i]);
    const grown = creaseComponent(seedTri, chosen, adjacency, creaseDeg);
    if (grown.size === 0) break;
    const refit = fitRegionAs(type, grown, facets);
    if (refit.degenerate || !refit.params) { region = grown; break; }
    const stable = grown.size === region.size && [...grown].every((t) => region.has(t));
    region = grown;
    fit = refit;
    if (stable) break;
  }
  return { region, type, params: fit.params, relResidual: fit.relResidual };
}
