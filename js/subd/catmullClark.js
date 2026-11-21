/**
 * catmullClark.js — Catmull-Clark subdivision of a closed quad cage, and exact points of its limit surface.
 *
 * Each subdivided face remembers where its corners sit in its ROOT face's unit square: corner k of a root
 * face with loop [c0, c1, c2, c3] is (0,0), (1,0), (1,1), (0,1), so u runs c0 -> c1 and v runs c0 -> c3. The
 * limit position of every fine vertex (Halstead, Kass and DeRose 1993) is an exact point of the limit surface,
 * at a dyadic parameter of each root face it touches. Regular faces of the limit surface are bicubic B-spline
 * patches in exactly that parameterisation, which is what makes the samples usable for fitting.
 *
 * Supported: closed manifold cages of quads, no creases. Anything else is refused by name.
 */

/** Face counts per vertex, edges with their two faces, refusing open, non-manifold or non-quad cages. */
export function cageTopology(nv, faces) {
  const edges = new Map();
  const valence = new Int32Array(nv);
  faces.forEach((loop, f) => {
    if (loop.length !== 4) throw new Error(`SUBD_NOT_QUAD: face ${f} has ${loop.length} corners`);
    for (let k = 0; k < 4; k++) {
      const a = loop[k], b = loop[(k + 1) % 4];
      valence[a]++;
      const key = a < b ? a * nv + b : b * nv + a;
      const e = edges.get(key);
      if (!e) edges.set(key, { a: Math.min(a, b), b: Math.max(a, b), uses: [{ face: f, side: k }] });
      else e.uses.push({ face: f, side: k });
    }
  });
  for (const e of edges.values()) {
    if (e.uses.length !== 2) throw new Error(`SUBD_NOT_CLOSED: edge ${e.a}-${e.b} has ${e.uses.length} faces`);
  }
  return { edges, valence };
}

/** One Catmull-Clark step. V: Float64Array (3 per vertex); faces: Int32Array (4 per face); uv: 8 per face. */
function step(V, faces, uv) {
  const nv = V.length / 3, nf = faces.length / 4;
  const edgeIndex = new Map();
  const edgeEnds = [];
  const edgeFaces = [];
  const faceEdges = new Int32Array(nf * 4);
  for (let f = 0; f < nf; f++) {
    for (let k = 0; k < 4; k++) {
      const a = faces[f * 4 + k], b = faces[f * 4 + ((k + 1) & 3)];
      const key = a < b ? a * nv + b : b * nv + a;
      let e = edgeIndex.get(key);
      if (e === undefined) {
        e = edgeEnds.length / 2;
        edgeIndex.set(key, e);
        edgeEnds.push(a, b);
        edgeFaces.push(f, -1);
      } else edgeFaces[e * 2 + 1] = f;
      faceEdges[f * 4 + k] = e;
    }
  }
  const ne = edgeEnds.length / 2;
  const out = new Float64Array((nv + ne + nf) * 3);
  const fp = nv + ne;                                         // face points after vertices and edges
  for (let f = 0; f < nf; f++) {
    for (let c = 0; c < 3; c++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += V[faces[f * 4 + k] * 3 + c];
      out[(fp + f) * 3 + c] = s / 4;
    }
  }
  for (let e = 0; e < ne; e++) {
    const a = edgeEnds[e * 2], b = edgeEnds[e * 2 + 1], f0 = edgeFaces[e * 2], f1 = edgeFaces[e * 2 + 1];
    for (let c = 0; c < 3; c++) {
      out[(nv + e) * 3 + c] = (V[a * 3 + c] + V[b * 3 + c] + out[(fp + f0) * 3 + c] + out[(fp + f1) * 3 + c]) / 4;
    }
  }
  // vertex points: (Q + 2R + (n - 3) S) / n
  const n = new Int32Array(nv);
  const Q = new Float64Array(nv * 3), R = new Float64Array(nv * 3);
  for (let f = 0; f < nf; f++) {
    for (let k = 0; k < 4; k++) {
      const v = faces[f * 4 + k], next = faces[f * 4 + ((k + 1) & 3)], prev = faces[f * 4 + ((k + 3) & 3)];
      n[v]++;
      for (let c = 0; c < 3; c++) {
        Q[v * 3 + c] += out[(fp + f) * 3 + c];
        // each incident edge is seen from both of its faces: half a midpoint from each
        R[v * 3 + c] += (V[v * 3 + c] + V[next * 3 + c]) / 4 + (V[v * 3 + c] + V[prev * 3 + c]) / 4;
      }
    }
  }
  for (let v = 0; v < nv; v++) {
    for (let c = 0; c < 3; c++) {
      const q = Q[v * 3 + c] / n[v], r = R[v * 3 + c] / n[v];
      out[v * 3 + c] = (q + 2 * r + (n[v] - 3) * V[v * 3 + c]) / n[v];
    }
  }
  const nfaces = new Int32Array(nf * 16);
  const nuv = new Float64Array(nf * 32);
  for (let f = 0; f < nf; f++) {
    const U = (k, c) => uv[f * 8 + ((k + 4) & 3) * 2 + c];
    for (let k = 0; k < 4; k++) {
      const child = f * 4 + k;
      nfaces[child * 4] = faces[f * 4 + k];
      nfaces[child * 4 + 1] = nv + faceEdges[f * 4 + k];
      nfaces[child * 4 + 2] = fp + f;
      nfaces[child * 4 + 3] = nv + faceEdges[f * 4 + ((k + 3) & 3)];
      for (let c = 0; c < 2; c++) {
        const ctr = (U(0, c) + U(1, c) + U(2, c) + U(3, c)) / 4;
        nuv[child * 8 + c] = U(k, c);
        nuv[child * 8 + 2 + c] = (U(k, c) + U(k + 1, c)) / 2;
        nuv[child * 8 + 4 + c] = ctr;
        nuv[child * 8 + 6 + c] = (U(k - 1, c) + U(k, c)) / 2;
      }
    }
  }
  return { V: out, faces: nfaces, uv: nuv };
}

/** Limit positions of every vertex of a closed all-quad mesh: (n^2 v + 4 sum e + sum f) / (n (n + 5)). */
export function limitPositions(V, faces) {
  const nv = V.length / 3, nf = faces.length / 4;
  const n = new Int32Array(nv);
  const E = new Float64Array(nv * 3), F = new Float64Array(nv * 3);
  for (let f = 0; f < nf; f++) {
    for (let k = 0; k < 4; k++) {
      const v = faces[f * 4 + k];
      const a = faces[f * 4 + ((k + 1) & 3)], d = faces[f * 4 + ((k + 2) & 3)], b = faces[f * 4 + ((k + 3) & 3)];
      n[v]++;
      for (let c = 0; c < 3; c++) {
        E[v * 3 + c] += (V[a * 3 + c] + V[b * 3 + c]) / 2;    // each edge neighbour is shared by two faces
        F[v * 3 + c] += V[d * 3 + c];
      }
    }
  }
  const L = new Float64Array(nv * 3);
  for (let v = 0; v < nv; v++) {
    const k = n[v];
    for (let c = 0; c < 3; c++) L[v * 3 + c] = (k * k * V[v * 3 + c] + 4 * E[v * 3 + c] + F[v * 3 + c]) / (k * (k + 5));
  }
  return L;
}

/**
 * The cage subdivided `levels` times: positions V, quads F, and per root face the (2^levels + 1)^2 grid of fine
 * vertex ids, ids[f][i * N + j] for the vertex at (u, v) = (i, j) / 2^levels of root face f.
 */
export function subdivide(vertices, faces, levels) {
  let V = Float64Array.from(vertices.flat());
  let F = Int32Array.from(faces.flat());
  let uv = new Float64Array(faces.length * 8);
  for (let f = 0; f < faces.length; f++) uv.set([0, 0, 1, 0, 1, 1, 0, 1], f * 8);
  for (let l = 0; l < levels; l++) ({ V, faces: F, uv } = step(V, F, uv));
  const N = (1 << levels) + 1, perRoot = 1 << (2 * levels);
  const ids = faces.map(() => new Int32Array(N * N).fill(-1));
  const nfine = F.length / 4;
  for (let g = 0; g < nfine; g++) {
    const root = Math.floor(g / perRoot);                    // children keep their root's block of indices
    for (let k = 0; k < 4; k++) {
      const i = Math.round(uv[g * 8 + k * 2] * (N - 1)), j = Math.round(uv[g * 8 + k * 2 + 1] * (N - 1));
      ids[root][i * N + j] = F[g * 4 + k];
    }
  }
  return { V, F, ids, N };
}

/**
 * Exact limit-surface samples on a (2^levels + 1)^2 grid per root face: grids[f] is a Float64Array, index
 * ((i * N) + j) * 3 + c for the point at (u, v) = (i, j) / 2^levels.
 */
export function limitGrids(vertices, faces, levels) {
  const { V, F, ids, N } = subdivide(vertices, faces, levels);
  const L = limitPositions(V, F);
  const grids = ids.map((g) => {
    const out = new Float64Array(N * N * 3);
    for (let k = 0; k < N * N; k++) out.set(L.subarray(g[k] * 3, g[k] * 3 + 3), k * 3);
    return out;
  });
  return { grids, N };
}
