/**
 * preview.js — the patch solid as triangles for the studio to show before the STEP is saved, in the shape
 * js/knit/preview.js gives the knitter's solids, but evaluated straight from the patches (no kernel): each patch
 * on a grid of cells, normals from its derivatives, facing out as the STEP's faces do; every edge of the faces'
 * loops once, as a polyline.
 *
 * Out: {positions: Float32Array (9 per triangle), normals: Float32Array (9 per triangle), face: Int32Array (per
 * triangle), types: ['freeform' per face], edges: Float32Array (6 per segment), faces: count}.
 */
import { evalAt } from './bezier.js';
import { orientation } from './patchStep.js';

/**
 * solids: [{ faces, loops }] as for patchStep. Each patch gets cells x cells grid cells, at most 8 and fewer on
 * solids of many faces, so the preview stays near `triangles` triangles.
 */
export function patchPreview(solids, { triangles = 200000 } = {}) {
  const count = solids.reduce((s, x) => s + x.faces.length, 0);
  const cells = Math.max(2, Math.min(8, Math.floor(Math.sqrt(triangles / (2 * count)))));
  const k = cells + 1;
  const positions = new Float32Array(count * cells * cells * 18), normals = new Float32Array(count * cells * cells * 18);
  const face = new Int32Array(count * cells * cells * 2), edges = [];
  const P = new Float64Array(k * k * 3), N = new Float64Array(k * k * 3);
  let t = 0, f0 = 0;
  for (const { faces, loops } of solids) {
    const out = orientation(faces, loops) >= 0 ? 1 : -1;
    const drawn = new Set();
    faces.forEach((f, fi) => {
      const n = f.uCount, poles = f.poles;
      for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) {
        const D = evalAt(n, poles, a / cells, b / cells), o = (a * k + b) * 3;
        P[o] = D[0]; P[o + 1] = D[1]; P[o + 2] = D[2];
        const nx = D[4] * D[8] - D[5] * D[7], ny = D[5] * D[6] - D[3] * D[8], nz = D[3] * D[7] - D[4] * D[6];
        const l = out / (Math.hypot(nx, ny, nz) || 1);
        N[o] = nx * l; N[o + 1] = ny * l; N[o + 2] = nz * l;
      }
      for (let a = 0; a < cells; a++) for (let b = 0; b < cells; b++) {
        const i00 = a * k + b, i10 = (a + 1) * k + b, i11 = (a + 1) * k + b + 1, i01 = a * k + b + 1;
        // counter-clockwise seen from Su x Sv; reversed when the solid's outside is the other way
        for (const tri of out > 0 ? [[i00, i10, i11], [i00, i11, i01]] : [[i00, i11, i10], [i00, i01, i11]]) {
          tri.forEach((i, c) => { positions.set(P.subarray(i * 3, i * 3 + 3), t * 9 + c * 3); normals.set(N.subarray(i * 3, i * 3 + 3), t * 9 + c * 3); });
          face[t++] = f0 + fi;
        }
      }
      // the loop's edges, each once: cells segments along a whole side, proportionally fewer along part of one
      const loop = loops[fi];
      loop.forEach(([a, u0, v0], i) => {
        const [b, u1, v1] = loop[(i + 1) % loop.length], key = a < b ? `${a},${b}` : `${b},${a}`;
        if (drawn.has(key)) return;
        drawn.add(key);
        const m = Math.max(1, Math.round(cells * (Math.abs(u1 - u0) + Math.abs(v1 - v0))));
        let prev = null;
        for (let s = 0; s <= m; s++) {
          const D = evalAt(n, poles, u0 + (u1 - u0) * s / m, v0 + (v1 - v0) * s / m), q = [D[0], D[1], D[2]];
          if (prev) edges.push(...prev, ...q);
          prev = q;
        }
      });
    });
    f0 += faces.length;
  }
  return { positions, normals, face, types: new Array(count).fill('freeform'), edges: Float32Array.from(edges), faces: count };
}
