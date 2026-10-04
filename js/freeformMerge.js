/**
 * freeformMerge.js — one group per freeform face, after auto-paint.
 *
 * Auto-paint knows only primitives, so a freeform face (a B-spline, a sweep) comes out as many primitive
 * groups, each fitting its small piece roughly: 458 groups for the 16 faces of the user's Christmas
 * Ornament, and on 150 ABC parts only 21.7% of freeform faces kept as one group (test/freeformSeg.mjs).
 *
 * What tells those pieces apart without the truth: on a CAD tessellation the vertices lie ON the true
 * surface, so a group on an analytic face fits its primitive exactly (paintCore.typeFinalGroup's bar), and
 * one inside a freeform face does not. Measured on the auto-painted groups of 40 ABC freeform parts, by
 * area: 98% of the groups on analytic faces are exact or too small to judge, 97% of those inside freeform
 * faces are inexact.
 *
 * So adjacent groups that are both inexact are joined across boundaries that are no face edge to the mesh
 * (no crease at `crease` degrees, no learned-boundary wall, under `curvatureFrac` of the boundary's edges
 * jumping in curvature: a fillet meets a plane tangentially but at curvature 1/r against 0), and a joined
 * group is marked freeform: the B-spline fitter's, never a primitive's. Exact and small groups are never
 * joined, and a part whose inexact groups hold under `minInexactShare` of the painted area is left alone.
 *
 * Measured with test/freeformSeg.mjs (freeform faces kept as one pure group, by area) against the
 * studio's auto-paint, on the user's twisty parts and 40 fill_eval parts per mesher:
 *   ungated, crease only    twisty 92%    fill analytic once -9 to -10 points on every mesher
 *   + curvature, no small   twisty 81%    fill analytic once -2.6 to -5.4
 *   + part gate 0.5         twisty 81.5%  fill analytic once -0.0 to -0.2; fill native freeform 3 -> 56%
 * The learned edge wall made it worse (it walls inside twisting faces); a union-fit test cannot tell
 * same-face from different-face pairs (the user's parts' faces join smoothly).
 *
 * The fast metric scores grouping only. Scored with typing (test/surfaceScore.mjs fetm vs etm, analytic
 * faces found once on the right surface type), the 0.5-gated merge cost the 14 Onshape parts it acted on
 * 89.7 -> 63.4 (swallowed 1.6 -> 24.3%): auto-paint's groups on analytic faces are mostly impure, so
 * 76% of their area was 'inexact' and joined freeform pieces. Hence the defaults now:
 *   robust     a group is also exact by its core or by a plane holding 30% of its vertices
 *              (groupExactness): those 14 parts 63.4 -> 74.8
 *   keepFlat   a small group holding a plane (4+ vertices, half of them) is never absorbed
 *   shareOver  'judged': the part gate ignores small groups (the coarse Fencepost's flat sides are a few
 *              large triangles each and held it at 0.35, so it was never merged)
 * Twisty with all three: freeform once by area 85.4% (as before), analytic once back to the unmerged
 * 77.8% (96.3% by area); Fencepost coarse 51 -> 23 groups.
 */

import { classifyRegion } from './gaussMap.js';
import { vertexDeviation, EXACT_TYPE_BAR } from './paintCore.js';
import { meshStepAngle, LOWPOLY_STEP } from './regionGrow.js';

const DOF = { plane: 3, cylinder: 5, cone: 6, sphere: 4, torus: 7 };

/**
 * 'exact' (a primitive fits at mesh precision), 'inexact', or 'small' (too few vertices to tell).
 * robust (needs adjacency): a group that is inexact as a whole is still 'exact' when its core (two rings
 * of boundary triangles peeled off) is, or when one plane through a group triangle holds 30% of its
 * vertices (at least 3 x DOF) to the bar: auto-paint's groups on analytic faces often carry a few triangles
 * of a neighbour, or are two faces in one. On the 14 Onshape parts the merge acted on, by area: groups on
 * analytic faces judged exact 23.6% whole -> 95.3% robust; pieces of freeform faces inexact 99.5 -> 99.2%.
 */
export function groupExactness(tris, facets, diag, { bar = EXACT_TYPE_BAR, support = 3, robust = false, adjacency = null } = {}) {
  const set = tris instanceof Set ? tris : new Set(tris);
  if (set.size < 3) return 'small';
  const cls = classifyRegion(set, facets);
  let st;
  if (!(cls.type in DOF)) st = vertexCount(set, facets) >= support * 3 ? 'inexact' : 'small';
  else {
    const { d, n } = vertexDeviation(set, facets, cls.type, cls.params);
    st = n < support * DOF[cls.type] ? 'small' : d <= bar * diag ? 'exact' : 'inexact';
  }
  if (st !== 'inexact' || !robust || !adjacency) return st;
  if (planeInliers(set, facets, bar * diag, support * DOF.plane)) return 'exact';
  const peel = (s) => {
    const out = new Set();
    for (const t of s) if ((adjacency.get(t) ?? []).every((e) => s.has(e.neighbor))) out.add(t);
    return out;
  };
  const core = peel(peel(set));
  if (core.size >= 3 && core.size < set.size && groupExactness(core, facets, diag, { bar, support }) === 'exact') return 'exact';
  return 'inexact';
}

/** Whether a plane through one of the group's triangles (30 tried, spread) holds 30% of its vertices and at least minIn. */
function planeInliers(set, facets, tol, minIn, share = 0.3, tries = 30) {
  const P = facets.positions, Nr = facets.normals;
  const verts = new Map();
  for (const t of set) for (let k = 0; k < 3; k++) { const o = t * 9 + k * 3; verts.set(P[o] + '_' + P[o + 1] + '_' + P[o + 2], o); }
  const need = Math.max(minIn, share * verts.size);
  const tris = [...set];
  for (let s = 0; s < Math.min(tries, tris.length); s++) {
    const t = tris[(s * 7919) % tris.length], o = t * 9;
    const nx = Nr[t * 3], ny = Nr[t * 3 + 1], nz = Nr[t * 3 + 2];
    const inl = [];
    for (const q of verts.values()) if (Math.abs((P[q] - P[o]) * nx + (P[q + 1] - P[o + 1]) * ny + (P[q + 2] - P[o + 2]) * nz) <= tol) inl.push(q);
    if (inl.length >= need && !collinear(P, inl, tol)) return true;
  }
  return false;
}

/**
 * Whether fewer than 3 of the points (offsets into P; all but 2 of 4) lie off the line through the two farthest apart: a
 * ruled face meshed with vertices on its straight edges only puts a whole edge and one more vertex on a
 * plane through any triangle (the full Star's twisted faces passed as planes before this check).
 */
function collinear(P, idx, tol) {
  const d2 = (a, b) => (P[a] - P[b]) ** 2 + (P[a + 1] - P[b + 1]) ** 2 + (P[a + 2] - P[b + 2]) ** 2;
  let a = idx[0], b = a;
  for (const q of idx) if (d2(q, a) > d2(b, a)) b = q;
  a = b;
  for (const q of idx) if (d2(q, a) > d2(b, a)) b = q;
  const L = Math.sqrt(d2(a, b));
  if (!(L > 0)) return true;
  const u = [(P[b] - P[a]) / L, (P[b + 1] - P[a + 1]) / L, (P[b + 2] - P[a + 2]) / L];
  let off = 0;
  for (const q of idx) {
    const w = [P[q] - P[a], P[q + 1] - P[a + 1], P[q + 2] - P[a + 2]];
    const s = w[0] * u[0] + w[1] * u[1] + w[2] * u[2];
    if (Math.hypot(w[0] - s * u[0], w[1] - s * u[1], w[2] - s * u[2]) > tol && ++off >= Math.min(3, idx.length - 2)) return false;
  }
  return true;
}

/** regionGrow.tangentWall's test, for any facet pair: total curvature jumps by `rel` of the larger, over `abs`. */
function curvatureJump(facets, { abs = 0.1, rel = 0.8 } = {}) {
  const { k1, k2, valid } = facets.curvature;
  const C = facets.centroids;
  return (a, e) => {
    const b = e.neighbor;
    if (!valid[a] || !valid[b]) return false;
    const ka = Math.abs(k1[a]) + Math.abs(k2[a]), kb = Math.abs(k1[b]) + Math.abs(k2[b]);
    const big = Math.max(ka, kb), j = Math.abs(ka - kb);
    if (!(big > 0) || j < rel * big) return false;
    return j * Math.hypot(C[a * 3] - C[b * 3], C[a * 3 + 1] - C[b * 3 + 1], C[a * 3 + 2] - C[b * 3 + 2]) >= abs;
  };
}

function vertexCount(tris, facets) {
  const P = facets.positions, seen = new Set();
  for (const t of tris) for (let k = 0; k < 3; k++) { const o = t * 9 + k * 3; seen.add(P[o] + '_' + P[o + 1] + '_' + P[o + 2]); }
  return seen.size;
}

/**
 * @param owner      Int32Array group per triangle (-1 unpainted), modified in place
 * @param facets     meshCore facets (with positions)
 * @param adjacency  Map triangle -> [{neighbor, angle (deg), boundary?}]
 * @param wall       optional edgeWall predicate (wall(t, e) true = a learned face boundary)
 * @returns {{freeform: Set<number>, merged: number, status: Map<number, string>}}  freeform: the group ids
 *          (after merging) to leave without a primitive
 */
export function freeformMerge(owner, facets, adjacency, { crease = 30, wall = null, bar = EXACT_TYPE_BAR, support = 3,
  curvature = { abs: 0.1, rel: 0.8 }, curvatureFrac = 0.2, joinSmall = false, skipLowPoly = false, minInexactShare = 0.5,
  only = null, joinedOnly = false, absorbSmall = true, shareOver = 'judged', keepFlat = true, islands = 0, robust = true,
  creaseFrac = 0 } = {}) {
  if (skipLowPoly && meshStepAngle(adjacency) >= LOWPOLY_STEP) return { freeform: new Set(), merged: 0, status: new Map(), skipped: 'low-poly' };
  const P = facets.positions;
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < P.length; i += 3) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], P[i + k]); hi[k] = Math.max(hi[k], P[i + k]); }
  const diag = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
  // `only`: the groups that may be joined (the studio's: this auto-paint's, never the user's own)
  const groups = new Map();
  for (let t = 0; t < owner.length; t++) {
    const g = owner[t];
    if (g >= 0 && (!only || only.has(g))) (groups.get(g) ?? groups.set(g, []).get(g)).push(t);
  }
  const status = new Map();
  for (const [g, tris] of groups) status.set(g, groupExactness(tris, facets, diag, { bar, support, robust, adjacency }));
  // Per part: merge only where inexact groups hold at least minInexactShare of the painted area (a part
  // made mostly of freeform faces), so mostly-analytic parts keep their groups.
  // shareOver 'judged': of the area of groups that are exact or inexact only, so a few large triangles
  // too small to judge (a coarse mesh's flat sides) do not vote
  let painted = 0, inexactArea = 0;
  for (const [g, tris] of groups) {
    if (shareOver === 'judged' && status.get(g) === 'small') continue;
    let a = 0;
    for (const t of tris) a += facets.areas[t];
    painted += a;
    if (status.get(g) === 'inexact') inexactArea += a;
  }
  const inexactShare = inexactArea / Math.max(painted, 1e-300);
  if (inexactShare < minInexactShare) return { freeform: new Set(), merged: 0, status, inexactShare, skipped: 'mostly analytic' };

  // A boundary between two groups is hard when any of its mesh edges is a crease or a learned wall, or
  // when a share >= curvatureFrac of its edges jumps in curvature (a fillet meets a plane tangentially but
  // with curvature 1/r against 0; pieces of one freeform face are curvature-continuous).
  // creaseFrac > 0: a boundary is hard by creases only when that share of its edges are creases, not any
  // one: the coarse Ornament's narrow faces are curved bands two or three triangles across, where pieces
  // of one face meet with a few edges just past 30 degrees
  const hard = new Set(), soft = new Set();
  const edgesOf = new Map(), jumpsOf = new Map(), perim = new Map(), creasesOf = new Map();
  const key = (a, b) => (a < b ? `${a}|${b}` : `${b}|${a}`);
  const jump = curvature && facets.curvature ? curvatureJump(facets, curvature) : null;
  for (const [t, edges] of adjacency) {
    const a = owner[t];
    if (a < 0) continue;
    for (const e of edges) {
      const b = owner[e.neighbor];
      if (b !== a) perim.set(a, (perim.get(a) ?? 0) + 1);
      if (b < 0 || b === a || e.neighbor < t || !status.has(a) || !status.has(b)) continue;
      const k = key(a, b);
      if (creaseFrac > 0 && e.angle >= crease) creasesOf.set(k, (creasesOf.get(k) ?? 0) + 1);
      if ((creaseFrac <= 0 && e.angle >= crease) || (wall && wall(t, e))) hard.add(k);
      else soft.add(k);
      edgesOf.set(k, (edgesOf.get(k) ?? 0) + 1);
      if (jump && jump(t, e)) jumpsOf.set(k, (jumpsOf.get(k) ?? 0) + 1);
    }
  }
  if (jump) for (const [k, n] of jumpsOf) if (n >= curvatureFrac * edgesOf.get(k)) hard.add(k);
  for (const [k, n] of creasesOf) if (n >= creaseFrac * edgesOf.get(k)) hard.add(k);
  const parent = new Map([...groups.keys()].map((g) => [g, g]));
  const find = (g) => { while (parent.get(g) !== g) { parent.set(g, parent.get(parent.get(g))); g = parent.get(g); } return g; };
  let merged = 0;
  for (const k of soft) {
    if (hard.has(k)) continue;
    const [a, b] = k.split('|').map(Number);
    if (status.get(a) === 'exact' || status.get(b) === 'exact') continue;
    if (!joinSmall && (status.get(a) !== 'inexact' || status.get(b) !== 'inexact')) continue;
    const ra = find(a), rb = find(b);
    if (ra === rb) continue;
    parent.set(rb, ra);
    merged++;
  }
  // Small groups (too few vertices to judge) left between the pieces of a freeform face: each joins a
  // joined group it borders softly, repeatedly, so slivers inside a B-spline face do not stay faces of
  // their own (Cake Pan (2): 356 groups -> 214 after joining, most of the rest slivers).
  // keepFlat: a small group in which one plane through a triangle holds 4+ vertices and half of them (to
  // the exactness bar) is (mostly) a plane face of few large triangles - a coarse mesh's flat side, often
  // with a stray triangle of its neighbour - and is never absorbed; pieces of a curved face hold 3
  const flatOf = new Map();
  const flat = (g) => {
    if (!flatOf.has(g)) flatOf.set(g, planeInliers(new Set(groups.get(g)), facets, bar * diag, 4, 0.5));
    return flatOf.get(g);
  };
  if (absorbSmall) {
    const sizeOf = () => { const m = new Map(); for (const g of status.keys()) m.set(find(g), (m.get(find(g)) ?? 0) + 1); return m; };
    for (let round = 0; round < 8; round++) {
      const sz = sizeOf();
      let moved = 0;
      for (const k of soft) {
        if (hard.has(k)) continue;
        const [a, b] = k.split('|').map(Number);
        for (const [x, y] of [[a, b], [b, a]]) {
          if (status.get(x) !== 'small' || find(x) !== x) continue;       // a small group still on its own
          if (keepFlat && flat(x)) continue;                                 // a plane face, not a sliver
          const ry = find(y);
          if (ry === x || (sz.get(ry) ?? 1) < 2) continue;                   // into a joined group only
          parent.set(x, ry);
          merged++; moved++;
          break;
        }
      }
      if (!moved) break;
    }
  }
  // islands: a group still on its own (inexact or small, never exact) whose boundary runs for at least
  // `islands` of its length softly along ONE joined group joins it: a piece of a freeform face that
  // auto-paint typed as a primitive, walled off from the other pieces only by the face it sits in.
  if (islands > 0) {
    const pairsOf = new Map();
    for (const k of soft) {
      if (hard.has(k)) continue;
      const [a, b] = k.split('|').map(Number);
      (pairsOf.get(a) ?? pairsOf.set(a, []).get(a)).push([b, edgesOf.get(k)]);
      (pairsOf.get(b) ?? pairsOf.set(b, []).get(b)).push([a, edgesOf.get(k)]);
    }
    for (let round = 0; round < 4; round++) {
      const sz = new Map();
      for (const g of status.keys()) sz.set(find(g), (sz.get(find(g)) ?? 0) + 1);
      let moved = 0;
      for (const [g, st] of status) {
        if (st === 'exact' || find(g) !== g || sz.get(g) > 1 || (keepFlat && flat(g))) continue;
        const by = new Map();
        for (const [o, n] of pairsOf.get(g) ?? []) { const r = find(o); if (r !== g && sz.get(r) > 1) by.set(r, (by.get(r) ?? 0) + n); }
        let best = -1, bn = 0;
        for (const [r, n] of by) if (n > bn) { best = r; bn = n; }
        if (best >= 0 && bn >= islands * (perim.get(g) ?? Infinity)) { parent.set(g, best); merged++; moved++; }
      }
      if (!moved) break;
    }
  }
  // Freeform: the joined groups and every inexact group. joinedOnly (a lone inexact group keeps its
  // primitive) protected impure analytic groups judged whole (Onshape once 93.3 -> 91.3 without it); with
  // robust exactness it changes nothing on Onshape or the ABC freeform set, and it typed the full Star's
  // twisted faces - one pure group each, kept apart by creases - as planes.
  const size = new Map();
  for (const g of status.keys()) size.set(find(g), (size.get(find(g)) ?? 0) + 1);
  const freeform = new Set();
  for (const [g, s] of status) if (s === 'inexact' && (!joinedOnly || size.get(find(g)) > 1)) freeform.add(find(g));
  for (let t = 0; t < owner.length; t++) if (status.has(owner[t])) owner[t] = find(owner[t]);
  return { freeform, merged, status, inexactShare };
}
