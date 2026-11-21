/**
 * preview.js — the patch solid as triangles for the studio to show before the STEP is saved, in the shape
 * js/knit/preview.js gives the knitter's solids, but evaluated straight from the patches (no kernel): each patch
 * on a grid of cells, normals from its derivatives, facing out as the STEP's faces do; every patch side once as a
 * polyline edge.
 *
 * Out: {positions: Float32Array (9 per triangle), normals: Float32Array (9 per triangle), face: Int32Array (per
 * triangle), types: ['freeform' per face], edges: Float32Array (6 per segment), faces: count}.
 */
import { evalAt } from './bezier.js';
import { orientation } from './patchStep.js';

/**
 * solids: [{ faces, cageFaces }] as for patchStep. Each patch gets cells x cells grid cells, at most 8 and fewer on
 * solids of many faces, so the preview stays near `triangles` triangles.
 */
export function patchPreview(solids, { triangles = 200000 } = {}) {
  const count = solids.reduce((s, x) => s + x.faces.length, 0);
  const cells = Math.max(2, Math.min(8, Math.floor(Math.sqrt(triangles / (2 * count)))));
  let nTri = 0, nSeg = 0, nFace = 0;
  for (const s of solids) { nTri += s.faces.length * cells * cells * 2; nFace += s.faces.length; nSeg += s.faces.length * 4 * cells; }
  const positions = new Float32Array(nTri * 9), normals = new Float32Array(nTri * 9), face = new Int32Array(nTri);
  const edgeList = new Float32Array(nSeg * 6);
  let t = 0, e = 0, f0 = 0;
  const k = cells + 1, P = new Float64Array(k * k * 3), N = new Float64Array(k * k * 3);
  for (const { faces, cageFaces } of solids) {
    const out = orientation(faces, cageFaces) >= 0 ? 1 : -1;
    const drawn = new Set();
    faces.forEach((f, fi) => {
      const n = f.uCount, poles = f.poles;
      for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) {
        const D = evalAt(n, poles, a / cells, b / cells), o = (a * k + b) * 3;
        P[o] = D[0]; P[o + 1] = D[1]; P[o + 2] = D[2];
        let nx = D[4] * D[8] - D[5] * D[7], ny = D[5] * D[6] - D[3] * D[8], nz = D[3] * D[7] - D[4] * D[6];
        const l = out / (Math.hypot(nx, ny, nz) || 1);
        N[o] = nx * l; N[o + 1] = ny * l; N[o + 2] = nz * l;
      }
      const put = (i) => { positions.set(P.subarray(i * 3, i * 3 + 3), t * 9 + put.c * 3); normals.set(N.subarray(i * 3, i * 3 + 3), t * 9 + put.c * 3); put.c++; };
      for (let a = 0; a < cells; a++) for (let b = 0; b < cells; b++) {
        const i00 = a * k + b, i10 = (a + 1) * k + b, i11 = (a + 1) * k + b + 1, i01 = a * k + b + 1;
        // counter-clockwise seen from Su x Sv; reversed when the solid's outside is the other way
        for (const tri of out > 0 ? [[i00, i10, i11], [i00, i11, i01]] : [[i00, i11, i10], [i00, i01, i11]]) {
          put.c = 0; tri.forEach(put); face[t] = f0 + fi; t++;
        }
      }
      // sides: corner k to corner k + 1 of the loop, each cage edge once
      const q = cageFaces[fi];
      const side = [(s) => [s, 0], (s) => [cells, s], (s) => [cells - s, cells], (s) => [0, cells - s]];
      for (let sd = 0; sd < 4; sd++) {
        const a = q[sd], b = q[(sd + 1) % 4], key = a < b ? `${a},${b}` : `${b},${a}`;
        if (drawn.has(key)) continue;
        drawn.add(key);
        for (let s = 0; s < cells; s++) {
          const [a0, b0] = side[sd](s), [a1, b1] = side[sd](s + 1);
          edgeList.set(P.subarray((a0 * k + b0) * 3, (a0 * k + b0) * 3 + 3), e * 6);
          edgeList.set(P.subarray((a1 * k + b1) * 3, (a1 * k + b1) * 3 + 3), e * 6 + 3);
          e++;
        }
      }
    });
    f0 += faces.length;
  }
  return { positions, normals, face, types: new Array(nFace).fill('freeform'), edges: edgeList.slice(0, e * 6), faces: nFace };
}
