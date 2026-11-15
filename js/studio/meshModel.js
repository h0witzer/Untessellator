/**
 * meshModel.js — the loaded mesh as the studio draws and paints it.
 *
 * Pure: no GL, no DOM. Triangle t owns positions[t*9 .. t*9+8], the same order as the
 * three.js geometry, meshCore.buildFacets and paintTool, so a triangle index means the
 * same triangle everywhere.
 *
 *   neighbours[t*3+k]  the triangle across edge k (corner k to corner k+1) in the same body, or -1 for an open
 *                      edge or one left unpaired (js/bodies.js pairs the triangles round an edge on 3+ triangles
 *                      body by body, as the knitter does, so fills work on bodies that touch)
 *   openEdges          the open edges (one triangle), each as a corner t*3+k
 *   nonManifoldEdges   the edges three or more triangles share that the pairing could not resolve (some triangle on
 *                      it pairs with none), each once, as a corner of one of them; bodies touching cleanly are not
 *   bodyOf, bodies     each triangle's body and the bodies (js/bodies.js findBodies: largest first, with their
 *                      shells, voids, volume and bounds); shellOf, shells likewise
 *   hidden             set by the studio: 1 per triangle of a body hidden in the Bodies list (or null); the drawn
 *                      edges and outlines leave those out
 *   flipped            how many triangles were turned round so every body faces out and every void in: the flips
 *                      are made IN PLACE on `positions`, which the export and the fills read too
 *   facetNormals       one normal per corner, all three the facet's: faceted shading
 *   smoothNormals      area-weighted welded-vertex normals, falling back to the facet
 *                      normal where they differ by more than 60 degrees so a crease stays sharp
 */

import { findBodies } from '../bodies.js';

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

  // Bodies, shells and the windings (js/bodies.js), from the welded corners.
  const V = new Float64Array(vertexCount * 3);
  for (let c = count * 3 - 1; c >= 0; c--) V.set([P[c * 3], P[c * 3 + 1], P[c * 3 + 2]], weld[c] * 3);
  const { flip, across, bodyOf, bodies, shellOf, shells } = findBodies(weld, V);
  let flipped = 0;
  for (let t = 0; t < count; t++) {
    if (!flip[t]) continue;
    flipped++;
    const o = t * 9;
    for (let a = 0; a < 3; a++) { const x = P[o + 3 + a]; P[o + 3 + a] = P[o + 6 + a]; P[o + 6 + a] = x; }
    const w = weld[t * 3 + 1]; weld[t * 3 + 1] = weld[t * 3 + 2]; weld[t * 3 + 2] = w;
  }
  const neighbours = across;

  // open edges (one triangle) and the edges on 3+ triangles that some triangle could not pair across
  const owners = new Map();
  for (let t = 0; t < count; t++) {
    for (let k = 0; k < 3; k++) {
      const a = weld[t * 3 + k], b = weld[t * 3 + (k + 1) % 3];
      if (a === b) continue;
      const key = a < b ? a * vertexCount + b : b * vertexCount + a;
      const list = owners.get(key);
      if (list) list.push(t * 3 + k); else owners.set(key, [t * 3 + k]);
    }
  }
  const open = [], nonManifold = [];
  for (const list of owners.values()) {
    if (list.length === 1) open.push(list[0]);
    else if (list.length > 2 && list.some((corner) => neighbours[corner] < 0)) nonManifold.push(list[0]);
  }
  owners.clear();
  const openEdges = Int32Array.from(open);
  const nonManifoldEdges = Int32Array.from(nonManifold);

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

  return { count, positions: P, centre, radius, neighbours, unitNormals: unit, facetNormals, smoothNormals, openEdges, nonManifoldEdges,
    bodyOf, bodies, shellOf, shells, flipped };
}

function pushEdge(out, P, t, k) {
  const a = t * 9 + k * 3;
  const b = t * 9 + ((k + 1) % 3) * 3;
  out.push(P[a], P[a + 1], P[a + 2], P[b], P[b + 1], P[b + 2]);
}

/**
 * The drawn edges of the body: every edge between triangles of different owners (a group
 * boundary, or painted against unpainted), every open edge, and every crease sharper than
 * `creaseDeg`; with `keep(t)`, only those triangles' edges. Returns segment endpoints, six numbers per segment.
 */
export function bodyEdges(model, owner, creaseDeg = 30, keep = null) {
  const { count, positions: P, neighbours, unitNormals: N, hidden } = model;
  const cosCrease = Math.cos(creaseDeg * Math.PI / 180);
  const out = [];
  for (let t = 0; t < count; t++) {
    if (hidden?.[t] || (keep && !keep(t))) continue;
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
  const { positions: P, neighbours, hidden } = model;
  const out = [];
  for (const t of triangles) {
    if (hidden?.[t]) continue;
    for (let k = 0; k < 3; k++) {
      const nb = neighbours[t * 3 + k];
      if (nb < 0 || !inside(nb)) pushEdge(out, P, t, k);
    }
  }
  return new Float32Array(out);
}
