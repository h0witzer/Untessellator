/**
 * knit/tiers.js — what an export is, in order of worth: analytic (every face on a fitted surface), hybrid (some
 * patches as flat facets), planar (only the planes on surfaces), faceted (the mesh itself), none (no closed solid).
 * Kept apart from knit/index.js so the studio's page can rank exports without loading the knitter.
 */

export const TIERS = ['analytic', 'hybrid', 'planar', 'faceted', 'none'];

/**
 * Which of two exports of the same part is better (negative: a): the tier first, then less of the area as flat
 * facets, then less as fallback B-splines. Repair keeps its changes only when the export after it is not worse
 * than the one before (the head-to-head of 2026-11-16: Repair lowered the tier on 10 of 69 parts it ran on).
 */
export function compareExports(a, b) {
  const rank = (r) => { const i = TIERS.indexOf(r?.tier); return i < 0 ? TIERS.length : i; };
  if (rank(a) !== rank(b)) return rank(a) - rank(b);
  const facets = (a.facetArea ?? 0) - (b.facetArea ?? 0);
  if (Math.abs(facets) > 1e-4) return facets;
  const splines = (a.splineArea ?? 0) - (b.splineArea ?? 0);
  return Math.abs(splines) > 1e-4 ? splines : 0;
}
