/**
 * faceGrouper.js — mesh face segmentation by dihedral angle
 *
 * Segments every triangle in the mesh into groups where adjacent triangles share
 * an edge whose dihedral angle is ≤ creaseAngleDeg.
 *
 * Adjacency comes from meshCore rather than a three.js graph so that this module —
 * and everything downstream of it — carries no three.js dependency and can run
 * headlessly under node. meshCore also recomputes geometric face normals from
 * vertex winding instead of trusting the source file's normals.
 */

import { buildFacets, buildAdjacency, segment } from './meshCore.js';

/**
 * Flat position array for a BufferGeometry, without requiring three.js.
 * Accepts either a real BufferAttribute (which exposes `.array`) or any
 * accessor-shaped stand-in.
 *
 * @param {object} geometry  non-indexed source geometry
 * @returns {ArrayLike<number>}
 */
export function positionsOf(geometry) {
  const attr = geometry.attributes.position;
  if (attr.array) return attr.array;
  const out = new Float64Array(attr.count * 3);
  for (let i = 0; i < attr.count; i++) {
    out[i * 3] = attr.getX(i);
    out[i * 3 + 1] = attr.getY(i);
    out[i * 3 + 2] = attr.getZ(i);
  }
  return out;
}

/**
 * Segment a non-indexed BufferGeometry into face groups.
 *
 * @param {object} geometry        non-indexed source geometry
 * @param {number} creaseAngleDeg  max dihedral angle within a group (0–180°)
 * @returns {Array<{triangleIndices: Set<number>}>}
 */
export function groupFaces(geometry, creaseAngleDeg = 30) {
  const positions = positionsOf(geometry);
  const facets = buildFacets(positions);
  const adjacency = buildAdjacency(positions, facets);
  return segment(adjacency, facets.count, creaseAngleDeg)
    .map(region => ({ triangleIndices: region }));
}

/**
 * Extract deduplicated vertex positions for a set of triangle indices.
 *
 * @param {THREE.BufferGeometry} geometry
 * @param {Set<number>} triangleIndices
 * @returns {Float32Array}  flat [x,y,z, x,y,z, …] for every vertex in the group
 */
export function extractGroupVertices(geometry, triangleIndices) {
  const posAttr = geometry.attributes.position;
  const out = new Float32Array(triangleIndices.size * 9);
  let i = 0;
  for (const t of triangleIndices) {
    for (let v = 0; v < 3; v++) {
      const idx = t * 3 + v;
      out[i++] = posAttr.getX(idx);
      out[i++] = posAttr.getY(idx);
      out[i++] = posAttr.getZ(idx);
    }
  }
  return out;
}

/**
 * Extract face normals for a set of triangle indices.
 * Uses per-vertex normals from the geometry if available, otherwise computes
 * per-triangle normals on the fly.
 *
 * @param {THREE.BufferGeometry} geometry
 * @param {Set<number>} triangleIndices
 * @returns {Float32Array}  flat [nx,ny,nz, …] one per vertex
 */
export function extractGroupNormals(geometry, triangleIndices) {
  const posAttr = geometry.attributes.position;
  const nrmAttr = geometry.attributes.normal;
  const out = new Float32Array(triangleIndices.size * 9);
  let i = 0;

  for (const t of triangleIndices) {
    if (nrmAttr) {
      for (let v = 0; v < 3; v++) {
        const idx = t * 3 + v;
        out[i++] = nrmAttr.getX(idx);
        out[i++] = nrmAttr.getY(idx);
        out[i++] = nrmAttr.getZ(idx);
      }
    } else {
      // compute face normal on the fly
      const ax = posAttr.getX(t*3),   ay = posAttr.getY(t*3),   az = posAttr.getZ(t*3);
      const bx = posAttr.getX(t*3+1), by = posAttr.getY(t*3+1), bz = posAttr.getZ(t*3+1);
      const cx = posAttr.getX(t*3+2), cy = posAttr.getY(t*3+2), cz = posAttr.getZ(t*3+2);
      const ex = bx-ax, ey = by-ay, ez = bz-az;
      const fx = cx-ax, fy = cy-ay, fz = cz-az;
      let nx = ey*fz - ez*fy, ny = ez*fx - ex*fz, nz = ex*fy - ey*fx;
      const len = Math.sqrt(nx*nx + ny*ny + nz*nz) || 1;
      nx /= len; ny /= len; nz /= len;
      for (let v = 0; v < 3; v++) { out[i++] = nx; out[i++] = ny; out[i++] = nz; }
    }
  }
  return out;
}

/**
 * Extract the ordered boundary vertex loop for a face group.
 * Returns the longest continuous loop found, or null if no valid loop exists.
 *
 * @param {THREE.BufferGeometry} geometry
 * @param {Set<number>} triangleIndices
 * @returns {Array<[number,number,number]>|null}  ordered list of [x,y,z] positions
 */
export function extractBoundaryLoop(geometry, triangleIndices) {
  const posAttr = geometry.attributes.position;
  const QUANT = 1e4;
  const qk = (x, y, z) =>
    `${Math.round(x * QUANT)}_${Math.round(y * QUANT)}_${Math.round(z * QUANT)}`;

  // Count how many times each edge appears in this group
  const edgeCount = new Map();
  const edgeVerts = new Map();

  for (const t of triangleIndices) {
    const verts = [];
    for (let v = 0; v < 3; v++) {
      const idx = t * 3 + v;
      verts.push([posAttr.getX(idx), posAttr.getY(idx), posAttr.getZ(idx)]);
    }
    for (let e = 0; e < 3; e++) {
      const a = verts[e], b = verts[(e + 1) % 3];
      const ka = qk(...a), kb = qk(...b);
      const key = ka < kb ? `${ka}|${kb}` : `${kb}|${ka}`;
      edgeCount.set(key, (edgeCount.get(key) || 0) + 1);
      if (!edgeVerts.has(key)) edgeVerts.set(key, [a, b]);
    }
  }

  // Boundary edges appear only once
  const boundary = [];
  for (const [key, count] of edgeCount) {
    if (count === 1) boundary.push(edgeVerts.get(key));
  }
  if (boundary.length < 3) return null;

  // Chain edges into an ordered loop
  const adj = new Map();
  for (let i = 0; i < boundary.length; i++) {
    const [a, b] = boundary[i];
    const ka = qk(...a), kb = qk(...b);
    if (!adj.has(ka)) adj.set(ka, []);
    if (!adj.has(kb)) adj.set(kb, []);
    adj.get(ka).push({ next: kb, pos: b, edgeIdx: i });
    adj.get(kb).push({ next: ka, pos: a, edgeIdx: i });
  }

  const visited = new Uint8Array(boundary.length);
  let bestLoop = null;

  for (let startIdx = 0; startIdx < boundary.length; startIdx++) {
    if (visited[startIdx]) continue;
    const [startA] = boundary[startIdx];
    const startKey = qk(...startA);
    const loop = [startA];
    let curKey = qk(...boundary[startIdx][1]);
    visited[startIdx] = 1;

    for (let step = 0; step < boundary.length; step++) {
      const neighbors = adj.get(curKey) || [];
      const next = neighbors.find(n => !visited[n.edgeIdx]);
      if (!next) break;
      visited[next.edgeIdx] = 1;
      if (next.next === startKey) break;
      loop.push(next.pos);
      curKey = next.next;
    }

    if (loop.length >= 3 && (!bestLoop || loop.length > bestLoop.length)) {
      bestLoop = loop;
    }
  }

  return bestLoop;
}
