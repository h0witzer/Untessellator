/**
 * autoPaint.js — paint whatever is still unpainted, automatically, best seed first.
 *
 * Works at any point: before the user has painted anything, or after, when it only fills what is
 * left and never touches existing groups (they are passed in as `owner`, and every fill treats
 * painted facets as taken: excluded from the model's patch and from growth).
 *
 * SEED ORDER is the lever measured by test/autoSeed.mjs (80 ABC parts, 4 meshers, v6feat, every
 * part painted to completion). Seeding where a single simple primitive best explains the
 * crease-bounded neighbourhood ("coherent") was the only order that never lost to random:
 * surface area recovered clean 75 -> 78% on OCCT meshes, 73 -> 77% NURBS, 53 -> 73% gmsh,
 * 63 -> 64% game-style. Largest facet first matched it on gmsh but lost 5 points on OCCT meshes
 * (their largest facets are slivers across a face), and seeding next to a crease lost 5 too.
 */

import { creaseRing, fitPrimitives } from './seedFeatures.js';

export const COHERENT_RING = 24;
export const COHERENT_CREASE = 30;

/**
 * The curvature-jump wall (regionGrow.tangentWall) auto-paint grows with: a wall where the facets'
 * total curvature differs by at least 80%, for curved fills only. On single clicks it is neutral
 * (it trades spill for incompleteness), but when a whole part is painted a stopped fill leaves
 * facets for the next seed while spill steals them from a neighbour. The first 20 parts looked like a
 * win (area clean 78.5 -> 82.2% OCCT), but all 80 parts do not bear it out: 77.4 -> 77.7% OCCT,
 * 77.5 -> 76.0% NURBS, 73.4 -> 72.7% gmsh, 68.4 -> 68.3% game (test/autoSeed.mjs --tangent c80),
 * with median segments per face 1.06 -> 1.38 on OCCT meshes. So the studio offers it as an option,
 * off by default (settings.autoTangent).
 */
export const AUTO_TANGENT = Object.freeze({ abs: 0.1, rel: 0.8, types: ['cylinder', 'cone', 'sphere', 'torus'] });

const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/** Unsigned distance and |cos| of facet (q, nt) to one fit from seedFeatures.fitPrimitives. */
export function distanceAndCos(kind, fit, q, nt) {
  if (kind === 'plane') {
    return [Math.abs(dot3([q[0] - fit.origin[0], q[1] - fit.origin[1], q[2] - fit.origin[2]], fit.normal)),
            Math.abs(dot3(nt, fit.normal))];
  }
  if (kind === 'sphere') {
    const w = [q[0] - fit.center[0], q[1] - fit.center[1], q[2] - fit.center[2]];
    const y = Math.hypot(w[0], w[1], w[2]);
    return y > 1e-30 ? [Math.abs(y - fit.radius), Math.abs(dot3(nt, w)) / y] : [Infinity, 0];
  }
  const base = kind === 'cylinder' ? fit.center : fit.apex;
  const v = [q[0] - base[0], q[1] - base[1], q[2] - base[2]];
  const al = dot3(v, fit.axis);
  const w = [v[0] - al * fit.axis[0], v[1] - al * fit.axis[1], v[2] - al * fit.axis[2]];
  const y = Math.hypot(w[0], w[1], w[2]);
  if (!(y > 1e-30)) return [Infinity, 0];
  if (kind === 'cylinder') return [Math.abs(y - fit.radius), Math.abs(dot3(nt, w)) / y];
  const sg = al < 0 ? -1 : 1;
  const nrm = [0, 1, 2].map((a) => (fit.cos * w[a]) / y - fit.sin * sg * fit.axis[a]);
  return [Math.abs(y * fit.cos - Math.abs(al) * fit.sin), Math.abs(dot3(nt, nrm))];
}

/** How well one simple primitive explains facet t's crease-bounded neighbourhood: lower is better. */
export function coherenceOf(t, facets, adjacency) {
  const ring = creaseRing(t, adjacency, COHERENT_RING, COHERENT_CREASE);
  if (ring.length < 8) return Infinity;
  const C = facets.centroids, Nr = facets.normals;
  const fits = fitPrimitives(ring, facets);
  let m0 = 0, m1 = 0, m2 = 0;
  for (const u of ring) { m0 += C[u * 3]; m1 += C[u * 3 + 1]; m2 += C[u * 3 + 2]; }
  m0 /= ring.length; m1 /= ring.length; m2 /= ring.length;
  let ss = 0;
  for (const u of ring) ss += (C[u * 3] - m0) ** 2 + (C[u * 3 + 1] - m1) ** 2 + (C[u * 3 + 2] - m2) ** 2;
  const radius = Math.sqrt(ss / ring.length) || 1;
  let best = Infinity;
  for (const kind of ['plane', 'cylinder', 'cone', 'sphere']) {
    const fit = fits[kind];
    if (!fit) continue;
    let d2 = 0, mis = 0;
    for (const u of ring) {
      const [d, cos] = distanceAndCos(kind, fit, [C[u * 3], C[u * 3 + 1], C[u * 3 + 2]], [Nr[u * 3], Nr[u * 3 + 1], Nr[u * 3 + 2]]);
      d2 += d * d;
      mis += 1 - Math.min(1, cos);
    }
    const s = Math.sqrt(d2 / ring.length) / radius + mis / ring.length;
    if (Number.isFinite(s) && s < best) best = s;
  }
  return best;
}

/**
 * Coherence for every facet (Infinity for degenerate facets and unfittable neighbourhoods).
 * Yields to the event loop every `chunk` facets so a page stays responsive.
 */
export async function coherentScores(facets, adjacency, { chunk = 4000, onProgress = null } = {}) {
  const n = facets.count;
  const scores = new Float64Array(n).fill(Infinity);
  for (let t = 0; t < n; t++) {
    if (!facets.degenerate[t]) scores[t] = coherenceOf(t, facets, adjacency);
    if ((t + 1) % chunk === 0) {
      onProgress?.((t + 1) / n);
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  }
  onProgress?.(1);
  return scores;
}

/** Facet ids best seed first: lowest coherence score, larger facet on ties; degenerate facets left out. */
export function seedOrder(scores, facets) {
  const ids = [];
  for (let t = 0; t < facets.count; t++) if (!facets.degenerate[t]) ids.push(t);
  const A = facets.areas;
  return ids.sort((a, b) => scores[a] - scores[b] || A[b] - A[a]);
}

/**
 * Paint every unpainted facet the seeds reach.
 *
 * @param {object}   o
 * @param {object}   o.facets
 * @param {number[]} o.order      seed order (seedOrder)
 * @param {Int32Array} o.owner    group per facet, -1 unpainted; painted facets are never touched
 * @param {Function} o.fill       async (seed, taken: Uint8Array) => { region: Set<number>, ... }
 * @param {Function} o.onSegment  async (triangles: Set<number>, fillResult) => token, per accepted fill
 * @param {number}   [o.minClaim=1]  fills that claim fewer new facets are dropped at once
 * @param {number}   [o.minKeep=1]   segments smaller than this are handed to `onDiscard` AFTER the
 *                                   run, back to unpainted. Leave both at 1: CAD meshes often make a
 *                                   flat rectangle of exactly two triangles, so any minimum of 3
 *                                   throws away real, sometimes large, faces. Measured on 20 parts
 *                                   (test/autoSeed.mjs): area clean 78 -> 69% OCCT, 78 -> 68% NURBS,
 *                                   72 -> 61% game with a minimum of 3, whether applied during the run
 *                                   or after it.
 * @param {Function} [o.onDiscard]   async (triangles: Set<number>, token) => void
 * @param {Function} [o.shouldStop]
 * @param {Function} [o.onProgress]  ({ fills, segments, free }) => void
 */
export async function autoPaint({ facets, order, owner, fill, onSegment, minClaim = 1, minKeep = 1,
                                  onDiscard = null, adjacency = null, absorbUpTo = 0, onAbsorb = null,
                                  forceSeed = false,
                                  shouldStop = () => false, onProgress = null }) {
  const n = facets.count;
  const taken = new Uint8Array(n);
  let free = 0;
  for (let t = 0; t < n; t++) {
    if (owner[t] >= 0 || facets.degenerate[t]) taken[t] = 1;
    else free++;
  }
  const tried = new Uint8Array(n);
  const segOf = new Int32Array(n).fill(-1);   // index into `tokens` for facets this run painted
  const tokens = [];
  const small = [];
  let fills = 0, segments = 0, dropped = 0;
  for (const seed of order) {
    if (shouldStop()) break;
    if (taken[seed] || tried[seed]) continue;
    tried[seed] = 1;
    const result = await fill(seed, taken);
    fills++;
    const claim = new Set();
    for (const t of result.region) if (!taken[t]) claim.add(t);
    if (forceSeed) claim.add(seed);   // the first harness's behaviour, kept for comparison
    // The seed joins only a fill that kept it: a fill that rejected its own seed would otherwise
    // leave it as a one-triangle island of the group. A rejected seed stays free for later fills.
    if (claim.size < minClaim) { dropped++; continue; }
    for (const t of claim) taken[t] = 1;
    free -= claim.size;
    segments++;
    const token = await onSegment(claim, result);
    tokens.push(token);
    for (const t of claim) segOf[t] = tokens.length - 1;
    if (claim.size < minKeep) small.push([claim, token]);
    onProgress?.({ fills, segments, free });
  }
  const stopped = shouldStop();
  // Small segments served as walls while the run went on; now they go back to unpainted.
  let discarded = 0;
  if (onDiscard) {
    for (const [claim, token] of small) {
      await onDiscard(claim, token);
      for (const t of claim) { segOf[t] = -1; taken[t] = 0; }
      free += claim.size;
      discarded++;
    }
  }

  // LEFTOVER ISLANDS. A fill can reject its own seed or leave a stray piece, and those facets stay
  // unpainted inside a face, which costs the face its recall. Each connected piece of unpainted
  // facets of at most `absorbUpTo` facets joins the run's segment it shares the most edges with
  // below COHERENT_CREASE degrees. Groups that existed before the run are never extended.
  let absorbed = 0;
  if (absorbUpTo > 0 && adjacency && onAbsorb && !stopped) {
    const seen = new Uint8Array(n);
    for (let s = 0; s < n; s++) {
      if (taken[s] || seen[s] || facets.degenerate[s]) continue;
      const island = [s];
      seen[s] = 1;
      const votes = new Map();
      for (let i = 0; i < island.length; i++) {
        const t = island[i];
        for (const e of adjacency.get(t) || []) {
          const u = e.neighbor;
          if (!taken[u] && !facets.degenerate[u]) {
            if (!seen[u]) { seen[u] = 1; island.push(u); }
          } else if (segOf[u] >= 0 && e.angle < COHERENT_CREASE) {
            votes.set(segOf[u], (votes.get(segOf[u]) ?? 0) + 1);
          }
        }
      }
      if (island.length > absorbUpTo || votes.size === 0) continue;
      let best = -1, bestVotes = 0;
      for (const [seg, v] of votes) if (v > bestVotes) { best = seg; bestVotes = v; }
      for (const t of island) { taken[t] = 1; segOf[t] = best; }
      await onAbsorb(island, tokens[best]);
      free -= island.length;
      absorbed += island.length;
    }
  }
  return { fills, segments: segments - discarded, dropped, discarded, absorbed, free, stopped };
}
