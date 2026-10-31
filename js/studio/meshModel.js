/**
 * meshModel.js — the loaded mesh as the studio draws and paints it.
 *
 * Pure: no GL, no DOM. Triangle t owns positions[t*9 .. t*9+8], the same order as the
 * three.js geometry, meshCore.buildFacets and paintTool, so a triangle index means the
 * same triangle everywhere.
 *
 *   neighbours[t*3+k]  the triangle across edge k (corner k to corner k+1), or -1 for an
 *                      open or non-manifold edge
 *   openEdges          the open edges (one triangle), each as a corner t*3+k
 *   nonManifoldEdges   the edges three or more triangles share, each once, as a corner of one of them
 *   facetNormals       one normal per corner, all three the facet's: faceted shading
 *   smoothNormals      area-weighted welded-vertex normals, falling back to the facet
 *                      normal where they differ by more than 60 degrees so a crease stays sharp
 */

const COS_KEEP_SMOOTH = Math.cos(60 * Math.PI / 180);

export function buildMeshModel(positions) {
  const P = positions;
  const count = Math.floor(P.length / 9);

  let min = [Infinity, Infinity, Infinity];
  let max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < count * 9; i += 3) {
    for (let a = 0; a < 3; a++) {
      if (P[i + a] < min[a]) min[a] = P[i + a];
      if (P[i + a] > max[a]) max[a] = P[i + a];
    }
  }
  if (!count) { min = [0, 0, 0]; max = [0, 0, 0]; }
  const centre = [0, 1, 2].map(a => (min[a] + max[a]) / 2);
  const radius = Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]) / 2 || 1;

  // Weld corners that share a position, to a millionth of the model's size.
  const quantum = 1e6 / (2 * radius);
  const weld = new Int32Array(count * 3);
  const ids = new Map();
  for (let c = 0; c < count * 3; c++) {
    const o = c * 3;
    const key = `${Math.round(P[o] * quantum)},${Math.round(P[o + 1] * quantum)},${Math.round(P[o + 2] * quantum)}`;
    let id = ids.get(key);
    if (id === undefined) { id = ids.size; ids.set(key, id); }
    weld[c] = id;
  }
  const vertexCount = ids.size;
  ids.clear();

  const neighbours = new Int32Array(count * 3).fill(-1);
  const firstOwner = new Map();
  const nonManifold = new Map();   // key -> a corner on the edge
  for (let t = 0; t < count; t++) {
    for (let k = 0; k < 3; k++) {
      const a = weld[t * 3 + k];
      const b = weld[t * 3 + (k + 1) % 3];
      if (a === b) continue;
      const key = a < b ? a * vertexCount + b : b * vertexCount + a;
      if (nonManifold.has(key)) continue;
      const other = firstOwner.get(key);
      if (other === undefined) {
        firstOwner.set(key, t * 3 + k);
      } else if (other >= 0) {
        neighbours[t * 3 + k] = Math.floor(other / 3);
        neighbours[other] = t;
        firstOwner.set(key, -1);
      } else {
        // A third owner: the edge is non-manifold, so nothing is adjacent across it.
        nonManifold.set(key, t * 3 + k);
      }
    }
  }
  // Undo any pair recorded before a third owner of the same edge turned up.
  if (nonManifold.size) {
    for (let t = 0; t < count; t++) {
      for (let k = 0; k < 3; k++) {
        const a = weld[t * 3 + k];
        const b = weld[t * 3 + (k + 1) % 3];
        const key = a < b ? a * vertexCount + b : b * vertexCount + a;
        if (nonManifold.has(key)) neighbours[t * 3 + k] = -1;
      }
    }
  }

  // an edge whose first owner was never matched has one triangle: open
  const open = [];
  for (const corner of firstOwner.values()) if (corner >= 0) open.push(corner);
  const openEdges = Int32Array.from(open);
  const nonManifoldEdges = Int32Array.from(nonManifold.values());

  const facetNormals = new Float32Array(count * 9);
  const vertexSums = new Float64Array(vertexCount * 3);
  const unit = new Float32Array(count * 3);
  for (let t = 0; t < count; t++) {
    const o = t * 9;
    const ux = P[o + 3] - P[o], uy = P[o + 4] - P[o + 1], uz = P[o + 5] - P[o + 2];
    const vx = P[o + 6] - P[o], vy = P[o + 7] - P[o + 1], vz = P[o + 8] - P[o + 2];
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const len = Math.hypot(nx, ny, nz);
    if (len > 0) {
      unit[t * 3] = nx / len; unit[t * 3 + 1] = ny / len; unit[t * 3 + 2] = nz / len;
    }
    for (let k = 0; k < 3; k++) {
      facetNormals.set([unit[t * 3], unit[t * 3 + 1], unit[t * 3 + 2]], o + k * 3);
      const v = weld[t * 3 + k] * 3;
      // The unnormalised cross product is twice the area along the normal: area weighting for free.
      vertexSums[v] += nx; vertexSums[v + 1] += ny; vertexSums[v + 2] += nz;
    }
  }
  const smoothNormals = new Float32Array(count * 9);
  for (let t = 0; t < count; t++) {
    for (let k = 0; k < 3; k++) {
      const v = weld[t * 3 + k] * 3;
      const len = Math.hypot(vertexSums[v], vertexSums[v + 1], vertexSums[v + 2]) || 1;
      const sx = vertexSums[v] / len, sy = vertexSums[v + 1] / len, sz = vertexSums[v + 2] / len;
      const agree = sx * unit[t * 3] + sy * unit[t * 3 + 1] + sz * unit[t * 3 + 2];
      const o = t * 9 + k * 3;
      if (agree >= COS_KEEP_SMOOTH) smoothNormals.set([sx, sy, sz], o);
      else smoothNormals.set([unit[t * 3], unit[t * 3 + 1], unit[t * 3 + 2]], o);
    }
  }

  return { count, positions: P, centre, radius, neighbours, unitNormals: unit, facetNormals, smoothNormals, openEdges, nonManifoldEdges };
}

function pushEdge(out, P, t, k) {
  const a = t * 9 + k * 3;
  const b = t * 9 + ((k + 1) % 3) * 3;
  out.push(P[a], P[a + 1], P[a + 2], P[b], P[b + 1], P[b + 2]);
}

/**
 * The drawn edges of the body: every edge between triangles of different owners (a group
 * boundary, or painted against unpainted), every open edge, and every crease sharper than
 * `creaseDeg`. Returns segment endpoints, six numbers per segment.
 */
export function bodyEdges(model, owner, creaseDeg = 30) {
  const { count, positions: P, neighbours, unitNormals: N } = model;
  const cosCrease = Math.cos(creaseDeg * Math.PI / 180);
  const out = [];
  for (let t = 0; t < count; t++) {
    for (let k = 0; k < 3; k++) {
      const nb = neighbours[t * 3 + k];
      if (nb >= 0 && nb < t) continue;   // drawn from the other side already
      let draw = nb < 0 || owner[nb] !== owner[t];
      if (!draw) {
        const d = N[t * 3] * N[nb * 3] + N[t * 3 + 1] * N[nb * 3 + 1] + N[t * 3 + 2] * N[nb * 3 + 2];
        draw = d < cosCrease;
      }
      if (draw) pushEdge(out, P, t, k);
    }
  }
  return new Float32Array(out);
}

/** The outline of a set of triangles: its edges that border anything outside it. */
export function outline(model, triangles, inside) {
  const { positions: P, neighbours } = model;
  const out = [];
  for (const t of triangles) {
    for (let k = 0; k < 3; k++) {
      const nb = neighbours[t * 3 + k];
      if (nb < 0 || !inside(nb)) pushEdge(out, P, t, k);
    }
  }
  return new Float32Array(out);
}
