/**
 * edgeFeatures.js — what one mesh edge looks like to the boundary model.
 *
 * Shared by test/edgeScreen.mjs (which dumps these rows with their true labels) and the studio
 * runtime, so the model is scored on exactly the rows it was trained on.
 *
 * An edge has no natural first side, so every pair of per-side values enters as (lower, higher).
 * The groups, and what the probe screen (ml/edge_screen.py, 1.1M labelled edges of 80 ABC parts)
 * found at the threshold that wrongly walls 2% of within-face edges:
 *
 *   dihedral  the bend itself and each side's local bend statistics. Alone it catches 5-26% of
 *             SMOOTH boundaries (under 30 degrees, invisible to the crease barrier).
 *   curv      curvature of each side and the jump across the edge. The big one: 50-81% of smooth
 *             boundaries.
 *   size      triangle areas. Describes the TRIANGULATION, not the surface: it adds a few points
 *             within a mesher and collapses across meshers, like log_area did for the type head.
 *             Kept here so a training run can measure it, excluded from the shipped input set.
 *   spread    eigenvalues of the normals around each side's crease-bounded ring.
 *   fit       CROSS FITS: the closed-form plane, cylinder, cone and sphere fitted to one side's
 *             ring, evaluated at the facet on the OTHER side, plus each side's own best residual
 *             and whether both sides chose the same primitive. Across a real boundary the
 *             neighbour does not lie on my surface.
 */

import { creaseRing, fitPrimitives, eig3symValues } from './seedFeatures.js';
import { distanceAndCos } from './autoPaint.js';

export const RING = 24;
export const RING_CREASE = 30;
export const KINDS = ['plane', 'cylinder', 'cone', 'sphere'];

export const EDGE_GROUPS = {
  dihedral: ['dihedral', 'local_max_lo', 'local_max_hi', 'local_mean_lo', 'local_mean_hi'],
  curv: ['curv_jump', 'curv_rel', 'curv_lo', 'curv_hi'],
  size: ['area_ratio', 'area_rel'],
  spread: ['ev1_lo', 'ev1_hi', 'ev2_lo', 'ev2_hi'],
  fit: [...KINDS.flatMap((k) => [`${k}_d_lo`, `${k}_d_hi`, `${k}_a_lo`, `${k}_a_hi`]), 'own_lo', 'own_hi', 'same_kind'],
};
export const EDGE_COLUMNS = Object.values(EDGE_GROUPS).flat();
export const N_EDGE_FEATURES = EDGE_COLUMNS.length;

/**
 * SECOND STAGE (auto-context, experimental): what the first-stage scores AROUND an edge say. A face
 * boundary is a curve, so an edge whose neighbours along that curve score high is likely boundary
 * too, and a lone high score in a field of low ones is likely a false wall. Per edge, from the
 * first-stage probabilities p:
 *   wing_lo/hi   the higher p of each adjacent triangle's two other edges, (lower, higher) side
 *   cont_lo/hi   the highest p among the other edges meeting at each endpoint vertex: does a curve
 *                continue through it? (lower, higher) endpoint
 *   n_lo/hi      how many edges at each endpoint score above 0.5, (lower, higher)
 * Needs js/edgeGraph.js's attachEdgeGraph for edge ids and vertex incidence.
 */
export const CONTEXT_COLUMNS = ['wing_lo', 'wing_hi', 'cont_lo', 'cont_hi', 'n_lo', 'n_hi'];

export function edgeContext(graph, adjacency, p) {
  const E = graph.edgeCount;
  const out = new Float32Array(E * CONTEXT_COLUMNS.length);
  const { ev, ef, vertStart, vertEdges } = graph;
  const wingOf = (t, id) => {
    let m = 0;
    for (const e of adjacency.get(t) || []) if (e.id !== id && p[e.id] > m) m = p[e.id];
    return m;
  };
  const contOf = (v, id) => {
    let m = 0, n = 0;
    if (v < 0) return [0, 0];
    for (let k = vertStart[v]; k < vertStart[v + 1]; k++) {
      const o = vertEdges[k];
      if (o === id) continue;
      if (p[o] > m) m = p[o];
      if (p[o] > 0.5) n++;
    }
    return [m, n];
  };
  for (let id = 0; id < E; id++) {
    const wa = wingOf(ef[id * 2], id), wb = wingOf(ef[id * 2 + 1], id);
    const [ca, na] = contOf(ev[id * 2], id);
    const [cb, nb] = contOf(ev[id * 2 + 1], id);
    const o = id * 6;
    out[o] = Math.min(wa, wb); out[o + 1] = Math.max(wa, wb);
    out[o + 2] = Math.min(ca, cb); out[o + 3] = Math.max(ca, cb);
    out[o + 4] = Math.min(na, nb) / 4; out[o + 5] = Math.max(na, nb) / 4;
  }
  return out;
}

/**
 * Per-mesh edge feature builder. Per-facet records (ring fits, normal spread, local bends) are
 * computed once on first use and reused by every edge of that facet.
 */
export function edgeFeatures(facets, adjacency) {
  const C = facets.centroids;
  const N = facets.normals;
  const curv = facets.curvature;
  const sizes = [];
  for (let t = 0; t < facets.count; t++) sizes.push(Math.sqrt(Math.max(facets.areas[t], 0)));
  sizes.sort((a, b) => a - b);
  const medianSize = sizes[sizes.length >> 1] || 1;
  const records = new Array(facets.count).fill(null);

  const build = (t) => {
    const ring = creaseRing(t, adjacency, RING, RING_CREASE);
    const fits = fitPrimitives(ring, facets);
    let m0 = 0, m1 = 0, m2 = 0, s00 = 0, s01 = 0, s02 = 0, s11 = 0, s12 = 0, s22 = 0;
    let mx = 0, my = 0, mz = 0;
    for (const u of ring) {
      const a = N[u * 3], b = N[u * 3 + 1], g = N[u * 3 + 2];
      m0 += a; m1 += b; m2 += g;
      s00 += a * a; s01 += a * b; s02 += a * g; s11 += b * b; s12 += b * g; s22 += g * g;
      mx += C[u * 3]; my += C[u * 3 + 1]; mz += C[u * 3 + 2];
    }
    const w = ring.length;
    m0 /= w; m1 /= w; m2 /= w; mx /= w; my /= w; mz /= w;
    const ev = eig3symValues(s00 / w - m0 * m0, s01 / w - m0 * m1, s02 / w - m0 * m2,
                             s11 / w - m1 * m1, s12 / w - m1 * m2, s22 / w - m2 * m2);
    let ss = 0;
    for (const u of ring) ss += (C[u * 3] - mx) ** 2 + (C[u * 3 + 1] - my) ** 2 + (C[u * 3 + 2] - mz) ** 2;
    const radius = Math.sqrt(ss / w) || 1;
    let own = Infinity;
    let ownKind = -1;
    KINDS.forEach((kind, ki) => {
      const fit = fits[kind];
      if (!fit) return;
      let d2 = 0, mis = 0;
      for (const u of ring) {
        const [d, cos] = distanceAndCos(kind, fit, [C[u * 3], C[u * 3 + 1], C[u * 3 + 2]],
                                        [N[u * 3], N[u * 3 + 1], N[u * 3 + 2]]);
        d2 += d * d;
        mis += 1 - Math.min(1, cos);
      }
      const s = Math.sqrt(d2 / w) / radius + mis / w;
      if (Number.isFinite(s) && s < own) { own = s; ownKind = ki; }
    });
    const edges = adjacency.get(t) || [];
    let maxA = 0, sumA = 0;
    for (const e of edges) { if (e.angle > maxA) maxA = e.angle; sumA += e.angle; }
    return { fits, radius, ev, own: Math.min(10, own), ownKind,
             localMax: maxA, localMean: edges.length ? sumA / edges.length : 0 };
  };
  const recordOf = (t) => (records[t] ??= build(t));

  /** The feature row for the edge between facets `a` and `b`, whose dihedral is `angle` degrees. */
  const row = (a, b, angle, into = new Float32Array(N_EDGE_FEATURES), at = 0) => {
    const ra = recordOf(a);
    const rb = recordOf(b);
    const d = Math.hypot(C[a * 3] - C[b * 3], C[a * 3 + 1] - C[b * 3 + 1], C[a * 3 + 2] - C[b * 3 + 2]);
    const ka = curv && curv.valid[a] ? Math.abs(curv.k1[a]) + Math.abs(curv.k2[a]) : 0;
    const kb = curv && curv.valid[b] ? Math.abs(curv.k1[b]) + Math.abs(curv.k2[b]) : 0;
    const jump = Math.abs(ka - kb);
    let i = at;
    into[i++] = angle / 90;
    into[i++] = Math.min(ra.localMax, rb.localMax) / 90;
    into[i++] = Math.max(ra.localMax, rb.localMax) / 90;
    into[i++] = Math.min(ra.localMean, rb.localMean) / 90;
    into[i++] = Math.max(ra.localMean, rb.localMean) / 90;
    into[i++] = Math.min(10, jump * d);
    into[i++] = jump / Math.max(ka, kb, 1e-30);
    into[i++] = Math.min(10, Math.min(ka, kb) * medianSize);
    into[i++] = Math.min(10, Math.max(ka, kb) * medianSize);
    into[i++] = Math.abs(Math.log(Math.max(facets.areas[a], 1e-30) / Math.max(facets.areas[b], 1e-30)));
    into[i++] = Math.log(Math.max(Math.min(facets.areas[a], facets.areas[b]), 1e-30) / (medianSize * medianSize));
    into[i++] = Math.min(ra.ev[1], rb.ev[1]);
    into[i++] = Math.max(ra.ev[1], rb.ev[1]);
    into[i++] = Math.min(ra.ev[2], rb.ev[2]);
    into[i++] = Math.max(ra.ev[2], rb.ev[2]);
    for (const kind of KINDS) {
      const pair = [[ra, b], [rb, a]].map(([rec, other]) => {
        const fit = rec.fits[kind];
        if (!fit) return [10, 1];
        const [dist, cos] = distanceAndCos(kind, fit,
          [C[other * 3], C[other * 3 + 1], C[other * 3 + 2]],
          [N[other * 3], N[other * 3 + 1], N[other * 3 + 2]]);
        return [Number.isFinite(dist) ? Math.min(10, dist / rec.radius) : 10,
                Number.isFinite(cos) ? 1 - Math.min(1, cos) : 1];
      });
      into[i++] = Math.min(pair[0][0], pair[1][0]);
      into[i++] = Math.max(pair[0][0], pair[1][0]);
      into[i++] = Math.min(pair[0][1], pair[1][1]);
      into[i++] = Math.max(pair[0][1], pair[1][1]);
    }
    into[i++] = Math.min(ra.own, rb.own);
    into[i++] = Math.max(ra.own, rb.own);
    into[i++] = ra.ownKind === rb.ownKind && ra.ownKind >= 0 ? 1 : 0;
    return into;
  };

  return { row, recordOf, medianSize };
}
