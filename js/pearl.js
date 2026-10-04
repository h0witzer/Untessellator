/**
 * pearl.js — re-assign painted facets among the surfaces auto-paint found, jointly (experimental).
 *
 * Auto-paint claims facets greedily: the first fill to reach a facet keeps it, and later, better
 * surfaces never get the chance. The merge pass (mergeRegions.js) tried to repair that one crumb at
 * a time with thresholds and failed on exactly the choice among competing neighbours. This instead
 * minimises one energy over every facet's label at once, PEARL-style (Isack & Boykov, "Energy-based
 * Geometric Multi-Model Fitting", IJCV 2012): propose (auto-paint's segments are the proposals),
 * expand (alpha-expansion by graph cuts), re-estimate (refit each surface on its facets), repeat.
 *
 *     E = sum over facets   w_t * min(s_t(label), sMax)           s: paintFrom's residual box, so a
 *                                                                  facet within tolerance costs <= 1
 *       + lambda * sum over edges whose labels differ  (len / median len) * (1 - p_e)^gamma
 *
 * p_e is the learned boundary probability, as in cutFill.js. A freeform segment (no analytic type)
 * costs `freeCost` per unit weight on its own facets and cannot spread. Only facets within `rings`
 * of a label's region take part in its expansion, since labels are spatially coherent.
 */

import { fitRegionAs } from './gaussMap.js';
import { surfaceEval, lowPolyDefaults } from './regionGrow.js';
import { MaxFlow } from './maxflow.js';

const ANALYTIC = new Set(['plane', 'cylinder', 'cone', 'sphere', 'torus']);

function median(values) {
  const s = Array.from(values).filter((v) => v > 0).sort((a, b) => a - b);
  return s[s.length >> 1] || 1;
}

/**
 * @param {Int32Array} seg      segment per facet, -1 unpainted (left alone)
 * @param {Array<string|null>} segType  fitter type per segment id
 * @param {object} opts  lambda (1), gamma (1), sMax (9), freeCost (2), rings (3), rounds (2),
 *                       strict (0: with 1, a facet may only switch to a surface that holds it within
 *                       paintFrom's tolerance, score <= 1, so no label can spread onto facets it does
 *                       not fit; with 2, also only facets their own surface does NOT hold), lockFree
 *                       (0: with 1, facets of freeform segments never move), locked (Uint8Array: facets never re-assigned, e.g. the user's groups)
 * @returns {{ seg: Int32Array, changed: number, labels: number }}
 */
export function pearlRefine(facets, adjacency, graph, seg, segType, opts = {}) {
  const lambda = opts.lambda ?? 1;
  const gamma = opts.gamma ?? 1;
  const sMax = opts.sMax ?? 9;
  const freeCost = opts.freeCost ?? 2;
  const rings = opts.rings ?? 3;
  const rounds = opts.rounds ?? 2;
  const locked = opts.locked ?? null;
  const strict = !!opts.strict;
  const n = facets.count;
  const lowPoly = lowPolyDefaults(adjacency);
  const medSize = Math.sqrt(median(facets.areas));
  const tolD = (lowPoly.tolScale ?? 0.5) * medSize;
  const angTol = (lowPoly.angleTol ?? 15) * Math.PI / 180;
  const medArea = medSize * medSize;
  const medLen = median(graph.length);
  const C = facets.centroids, N = facets.normals;
  const weight = new Float64Array(n);
  for (let t = 0; t < n; t++) weight[t] = Math.min(5, Math.max(0.2, facets.areas[t] / medArea));
  const edgeCost = (e) => {
    const len = graph.length[e.id] > 0 ? graph.length[e.id] / medLen : 1;
    const p = e.boundary !== undefined ? e.boundary : 0;
    return lambda * len * Math.pow(Math.max(0, 1 - p), gamma);
  };

  const label = Int32Array.from(seg);
  const original = Int32Array.from(seg);
  const nLabels = segType.length;
  let changed = 0;

  for (let round = 0; round < rounds; round++) {
    // Re-estimate: each analytic label's surface from its current facets.
    const members = Array.from({ length: nLabels }, () => []);
    for (let t = 0; t < n; t++) if (label[t] >= 0) members[label[t]].push(t);
    const ev = new Array(nLabels).fill(null);
    for (let l = 0; l < nLabels; l++) {
      const type = segType[l];
      if (!ANALYTIC.has(type) || members[l].length === 0) continue;
      const fit = fitRegionAs(type, new Set(members[l]), facets);
      if (fit.params && !fit.degenerate) ev[l] = surfaceEval(type, fit.params);
    }
    const cost = (t, l) => {
      if (l < 0) return Infinity;
      const e = ev[l];
      if (!e) return original[t] === l ? freeCost * weight[t] : Infinity;
      const c = [C[t * 3], C[t * 3 + 1], C[t * 3 + 2]];
      const d = e.distance(c) / tolD;
      const nn = e.normal(c);
      const cos = Math.min(1, Math.abs(nn[0] * N[t * 3] + nn[1] * N[t * 3 + 1] + nn[2] * N[t * 3 + 2]));
      const a = Math.acos(cos) / angTol;
      const s = Number.isFinite(d) ? Math.max(d * d, a * a) : sMax;
      return weight[t] * Math.min(s, sMax);
    };
    const pair = (a, b, e) => (a === b ? 0 : edgeCost(e));

    // Expand every analytic label, largest first.
    const order = [...Array(nLabels).keys()].filter((l) => ev[l]).sort((a, b) => members[b].length - members[a].length);
    for (const alpha of order) {
      // Domain: facets within `rings` of alpha's region, painted, not alpha, not locked.
      const inAlpha = [];
      for (let t = 0; t < n; t++) if (label[t] === alpha) inAlpha.push(t);
      if (!inAlpha.length) continue;
      const depth = new Map();
      for (const t of inAlpha) depth.set(t, 0);
      const queue = [...inAlpha];
      for (let k = 0; k < queue.length; k++) {
        const t = queue[k];
        const d = depth.get(t);
        if (d >= rings) continue;
        for (const e of adjacency.get(t) || []) {
          const u = e.neighbor;
          if (depth.has(u) || label[u] < 0 || (locked && locked[u])) continue;
          depth.set(u, d + 1);
          queue.push(u);
        }
      }
      const nodes = queue.filter((t) => label[t] !== alpha);
      if (!nodes.length) continue;
      const index = new Map(nodes.map((t, i) => [t, i]));
      const g = new MaxFlow(nodes.length, nodes.length * 2);
      // unary: E(0) keep (paid on the source side, t -> sink), E(1) switch (source -> t)
      const e0 = new Float64Array(nodes.length);
      const e1 = new Float64Array(nodes.length);
      for (let i = 0; i < nodes.length; i++) {
        const t = nodes[i];
        e0[i] = cost(t, label[t]);
        e1[i] = cost(t, alpha);
        if (strict && e1[i] > weight[t]) e1[i] = Infinity;
        // lockFree: facets of a freeform segment (no analytic surface) stay put. Their cost is a
        // constant, so without the lock ANY surface that fits them within tolerance takes them.
        if (opts.lockFree && !ev[label[t]]) e1[i] = Infinity;
        // strict 2 (REPAIR ONLY): a facet its own surface still explains (score <= 1) never moves.
        if (opts.strict === 2 && e0[i] <= weight[t]) e1[i] = Infinity;
      }
      for (let i = 0; i < nodes.length; i++) {
        const t = nodes[i];
        for (const e of adjacency.get(t) || []) {
          const u = e.neighbor;
          const j = index.get(u);
          if (j === undefined) {
            // u is fixed at its label (alpha, outside the band, unpainted or locked)
            if (label[u] < 0) continue;
            e0[i] += pair(label[t], label[u], e);
            e1[i] += pair(alpha, label[u], e);
          } else if (j > i) {
            const A = pair(label[t], label[u], e), B = pair(label[t], alpha, e);
            const Cc = pair(alpha, label[u], e), D = 0;
            // E(x_i, x_j) = A + (C - A) x_i + (D - C) x_j + (B + C - A - D)(1 - x_i) x_j
            if (Cc - A >= 0) e1[i] += Cc - A; else e0[i] += A - Cc;
            if (D - Cc >= 0) e1[j] += D - Cc; else e0[j] += Cc - D;
            const w = B + Cc - A - D;
            if (w > 0) g.addEdge(i, j, w, 0);
          }
        }
      }
      for (let i = 0; i < nodes.length; i++) {
        const a = e1[i], b = e0[i];
        g.addTweights(i, Number.isFinite(a) ? a : 1e12, Number.isFinite(b) ? b : 1e12);
      }
      g.maxflow();
      for (let i = 0; i < nodes.length; i++) {
        if (g.onSinkSide(i) && Number.isFinite(e1[i])) {
          label[nodes[i]] = alpha;
          changed++;
        }
      }
    }
  }
  const used = new Set();
  for (let t = 0; t < n; t++) if (label[t] >= 0) used.add(label[t]);
  return { seg: label, changed, labels: used.size };
}
