/**
 * groupType.js — the group-level type model's view of a FINISHED painted group, and its decision.
 *
 * Every painted group becomes one face of the solid, and the exporter fits it with a type: plane,
 * cylinder, cone, sphere, torus, or none (a freeform B-spline). Today that type is a rule
 * (paintCore.typeFinalGroup): the painted type, replaced by classifyRegion's choice only where that fit
 * holds every vertex to 1e-6 of the part diagonal. With the true type on today's groups, surfaces found
 * rise 9.3 points on fill_eval and 3.6 on the Onshape exports (memory: surface-scorecard); re-running the
 * classifier's simplest-within-1% rule gets about 1. This module turns the evidence the rule already
 * computes into a feature row for a learned typer (ml/group_type_train.py), and lets the typer
 * override the rule only where it is confident.
 *
 * FEATURES, one row per group of >= 3 facets (GROUP_FEATURES names them, in order):
 *   per primitive (plane, cylinder, cone, sphere, torus), from classifyRegion's refined candidates:
 *     fit      1 if the fitter returned this primitive
 *     lrel     log10 residual / group extent
 *     lratio   log10 residual / the best candidate's residual (0 for the best)
 *     lvdev    log10 worst vertex deviation from the fitted surface / part diagonal
 *     support  log10 distinct vertices / (3 x the primitive's degrees of freedom)
 *   parameter sanity: cylinder radius, cone half-angle and apex distance, sphere radius, torus radii
 *     and their ratio, each against the group extent
 *   the group: facets, vertices, extent / diagonal, area / extent^2, share of facets on its border,
 *     the Gauss map's planarity, spread and half-angle
 *   today's decisions, one-hot: the painted type, classifyRegion's choice, typeFinalGroup's choice
 * Triangle size is deliberately absent: it describes the mesher, the shortcut the seed model learned.
 *
 * The model never returns geometry. Its type goes to gaussMap.fitRegionAs like any other.
 */

import { classifyRegion } from './gaussMap.js';
import { typeFinalGroup, vertexDeviation, partDiagonal, EXACT_TYPE_BAR } from './paintCore.js';

/** The typer's classes, in output order; 'freeform' exports as no primitive (null). */
export const GROUP_TYPES = ['plane', 'cylinder', 'cone', 'sphere', 'torus', 'freeform'];
const PRIMS = ['plane', 'cylinder', 'cone', 'sphere', 'torus'];
const DOF = { plane: 3, cylinder: 5, cone: 6, sphere: 4, torus: 7 };
const SLOTS = [...PRIMS, null];
const slotName = (t) => t ?? 'none';

export const GROUP_FEATURES = [
  ...PRIMS.flatMap((k) => [`${k}_fit`, `${k}_lrel`, `${k}_lratio`, `${k}_lvdev`, `${k}_support`]),
  'cyl_lradius', 'cone_halfangle', 'cone_lapex', 'sphere_lradius', 'torus_lminor', 'torus_lmajor', 'torus_ratio',
  'lfacets', 'lverts', 'lextent', 'larea', 'border_share', 'gauss_planarity', 'gauss_spread', 'gauss_halfangle',
  ...SLOTS.map((t) => `painted_${slotName(t)}`),
  ...SLOTS.map((t) => `classified_${slotName(t)}`),
  ...SLOTS.map((t) => `rule_${slotName(t)}`),
];
export const N_GROUP_FEATURES = GROUP_FEATURES.length;

// -Infinity (an exact fit: log10 of 0) clamps to `lo`, the best end; NaN (no measurement) to `hi`.
const clamp = (v, lo, hi) => (Number.isNaN(v) ? hi : Math.min(hi, Math.max(lo, v)));
const log10 = (v, lo, hi) => clamp(Math.log10(v), lo, hi);
const dist3 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

/**
 * The feature row of one finished group, with the classifier's result and today's rule decision
 * (so a caller that falls back to the rule does not classify twice). Null for groups under 3 facets,
 * which the rule keeps deciding.
 *
 * @param {Set<number>|number[]} tris  the group's facets
 * @param {object} facets, adjacency   the mesh (paintCore.preparePaintPositions)
 * @param {string|null} painted        the type the group was painted with
 * @returns {{features: Float32Array, cls: object, rule: string|null} | null}
 */
export function groupTypeFeatures(tris, facets, adjacency, painted, { bar = EXACT_TYPE_BAR, looseTol = 5e-4 } = {}) {
  const set = tris instanceof Set ? tris : new Set(tris);
  if (set.size < 3) return null;
  const diag = partDiagonal(facets) || 1;
  const cls = classifyRegion(set, facets);
  const rule = typeFinalGroup(set, facets, painted ?? null, { bar, looseTol, cls }).type;

  // Extent and area on the group's own facets (classifyRegion reports extent only when it picks a type).
  const C = facets.centroids;
  let cx = 0, cy = 0, cz = 0, area = 0;
  for (const t of set) {
    const a = facets.areas[t];
    cx += C[t * 3] * a; cy += C[t * 3 + 1] * a; cz += C[t * 3 + 2] * a; area += a;
  }
  const mean = area > 0 ? [cx / area, cy / area, cz / area] : [C[0], C[1], C[2]];
  let extent = 0;
  for (const t of set) extent = Math.max(extent, dist3([C[t * 3], C[t * 3 + 1], C[t * 3 + 2]], mean));
  if (!(extent > 0)) extent = 1e-12;
  let border = 0;
  for (const t of set) {
    for (const e of adjacency.get(t) || []) if (!set.has(e.neighbor)) { border++; break; }
  }

  const out = new Float32Array(N_GROUP_FEATURES);
  let i = 0;
  const cand = new Map();
  for (const c of cls.candidates || []) if (PRIMS.includes(c.type) && !cand.has(c.type)) cand.set(c.type, c);
  const best = Math.max(Math.min(...[...cand.values()].map((c) => c.residual)), 1e-30);
  let nVerts = 0;
  for (const k of PRIMS) {
    const c = cand.get(k);
    let lvdev = 0;
    if (c?.params) {
      const v = vertexDeviation(set, facets, k, c.params);
      nVerts = Math.max(nVerts, v.n);
      lvdev = log10(v.d / diag, -12, 0);
    }
    out[i++] = c ? 1 : 0;
    out[i++] = c ? log10(c.relResidual, -12, 2) : 2;
    out[i++] = c ? log10(c.residual / best, 0, 6) : 6;
    out[i++] = c ? lvdev : 0;
    out[i++] = 0;   // support, filled once the vertex count is known
  }
  if (!nVerts) nVerts = vertexDeviation(set, facets, 'plane', { origin: mean, normal: [0, 0, 1] }).n;
  PRIMS.forEach((k, j) => { out[j * 5 + 4] = log10(nVerts / (3 * DOF[k]), -2, 4); });

  const cy_ = cand.get('cylinder')?.params, co = cand.get('cone')?.params;
  const sp = cand.get('sphere')?.params, to = cand.get('torus')?.params;
  out[i++] = cy_ ? log10(cy_.radius / extent, -3, 6) : 6;
  out[i++] = co ? clamp(co.halfAngle / (Math.PI / 2), 0, 1) : 0;
  out[i++] = co ? log10(dist3(co.apex, mean) / extent, -3, 6) : 6;
  out[i++] = sp ? log10(sp.radius / extent, -3, 6) : 6;
  out[i++] = to ? log10(to.minorRadius / extent, -3, 6) : 6;
  out[i++] = to ? log10(to.majorRadius / extent, -3, 6) : 6;
  out[i++] = to ? clamp(to.minorRadius / Math.max(to.majorRadius, 1e-30), 0, 1) : 0;

  out[i++] = Math.log10(set.size);
  out[i++] = Math.log10(Math.max(nVerts, 1));
  out[i++] = log10(extent / diag, -8, 0);
  out[i++] = log10(area / (extent * extent), -6, 2);
  out[i++] = border / set.size;
  const g = cls.gauss;
  out[i++] = g ? clamp(g.planarity, 0, 1) : 0;
  out[i++] = g ? clamp(g.spread, 0, 1) : 0;
  out[i++] = g ? clamp(g.halfAngle / (Math.PI / 2), 0, 1) : 0;

  const classified = PRIMS.includes(cls.type) ? cls.type : null;
  for (const t of SLOTS) out[i++] = (painted ?? null) === t ? 1 : 0;
  for (const t of SLOTS) out[i++] = classified === t ? 1 : 0;
  for (const t of SLOTS) out[i++] = rule === t ? 1 : 0;
  if (i !== N_GROUP_FEATURES) throw new Error(`groupTypeFeatures wrote ${i} of ${N_GROUP_FEATURES}`);
  return { features: out, cls, rule };
}

/**
 * The type a finished group exports with: the typer's class where its probability reaches
 * `threshold`, today's rule elsewhere. 'freeform' exports as null (a B-spline).
 *
 * @param {ArrayLike<number>} probs  softmax over GROUP_TYPES
 * @param {string|null} rule         typeFinalGroup's choice for the same group
 */
export function decideGroupType(probs, rule, threshold) {
  let k = 0;
  for (let j = 1; j < probs.length; j++) if (probs[j] > probs[k]) k = j;
  if (!(probs[k] >= threshold)) return { type: rule, by: 'rule', p: probs[k] };
  return { type: GROUP_TYPES[k] === 'freeform' ? null : GROUP_TYPES[k], by: 'model', p: probs[k] };
}
