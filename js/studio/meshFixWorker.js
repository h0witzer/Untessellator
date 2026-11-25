/**
 * meshFixWorker.js — the mesh fixes in Fix bad areas, worked out off the main thread as a mesh opens and after each
 * edit, so the card can list them and Apply is at once. In order:
 *
 *   pieces  small loose pieces go (the triangles the studio marks)
 *   holes   js/holeFill.js: fins and walls stuck to the surface go, then every hole is capped (on the survey's 45 game
 *           models, closed meshes 33 -> 43, after the import repair)
 *
 * Bodies that pass through each other are no longer joined here: a mesh union (Manifold) cut their clean faces along
 * jagged seams, which auto-paint then split (FPA coal chassis, 2026-11-21). Each body exports as its own solid, as the
 * file had it; fusing them is for the CAD program, on exact solids (2026-11-25).
 *
 * In:  {id, positions: Float32Array (9 per triangle), small: Uint8Array (per triangle) | null, steps: {pieces, holes}}
 * Out: {id, positions, src (each output triangle's input triangle; -1 for a cap), pieces: {removed, triangles},
 *       holes: holeFill's report} or {id, error}
 */

import { fillHoles } from '../holeFill.js';

self.onmessage = async ({ data }) => {
  const { id, positions, small, steps = { pieces: true, holes: true } } = data;
  try {
    let P = positions;
    let src = Int32Array.from({ length: P.length / 9 }, (_, t) => t);
    const after = (next, map) => { P = next; src = Int32Array.from(map, (s) => (s < 0 ? -1 : src[s])); };
    const result = { id, pieces: { removed: 0, triangles: 0 }, holes: null };
    // small loose pieces
    if (steps.pieces && small) {
      const keep = [];
      for (let t = 0; t < small.length; t++) if (!small[t]) keep.push(t);
      if (keep.length < small.length) {
        const next = new Float32Array(keep.length * 9);
        keep.forEach((t, i) => next.set(P.subarray(t * 9, t * 9 + 9), i * 9));
        result.pieces = { removed: data.smallPieces ?? 0, triangles: small.length - keep.length };
        after(next, keep);
      }
    }
    // holes, fins and walls
    if (steps.holes) {
      const r = fillHoles(P);
      result.holes = r.report;
      if (r.report.filled || r.report.removed_triangles) after(Float32Array.from(r.positions), r.src);
    }
    result.positions = P;
    result.src = src;
    self.postMessage(result);
  } catch (error) {
    self.postMessage({ id, error: String(error?.message ?? error) });
  }
};
