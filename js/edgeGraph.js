/**
 * edgeGraph.js — mesh edges as things in their own right: one id per undirected edge, its two welded
 * vertices and its length, and which edges meet at each vertex.
 *
 * buildAdjacency (meshCore.js) keeps only facet-to-facet links, which is all a flood fill needs.
 * Anything that reasons along a boundary CURVE needs more: which edge continues which. Hysteresis on
 * the learned wall (edgeWall.hysteresisWall) links wall edges through shared vertices, and the
 * min-cut fill (cutFill.js) prices a cut by its length. The weld is meshCore's own, so the edges
 * here are exactly the adjacency's manifold edges.
 *
 * `attachEdgeGraph` also writes `e.id` onto both adjacency entries of every edge.
 */

import { weldQuantum } from './meshCore.js';

/**
 * @returns {{ edgeCount:number, vertexCount:number,
 *             ev:Int32Array,          two vertex ids per edge
 *             ef:Int32Array,          two facets per edge (lower id first)
 *             length:Float64Array,    edge length
 *             vertStart:Int32Array, vertEdges:Int32Array }   edges at each vertex (CSR)
 */
export function attachEdgeGraph(positions, facets, adjacency, quant = weldQuantum(positions)) {
  const P = positions;
  const ids = new Map();
  const where = [];                       // first position offset seen for each vertex id
  const fv = new Int32Array(facets.count * 3).fill(-1);
  for (let t = 0; t < facets.count; t++) {
    if (facets.degenerate && facets.degenerate[t]) continue;
    for (let k = 0; k < 3; k++) {
      const o = t * 9 + k * 3;
      const key = Math.round(P[o] * quant) + '_' + Math.round(P[o + 1] * quant) + '_' + Math.round(P[o + 2] * quant);
      let id = ids.get(key);
      if (id === undefined) { id = where.length; ids.set(key, id); where.push(o); }
      fv[t * 3 + k] = id;
    }
  }

  let edgeCount = 0;
  for (const [a, list] of adjacency) for (const e of list) if (e.neighbor > a) edgeCount++;
  const ev = new Int32Array(edgeCount * 2);
  const ef = new Int32Array(edgeCount * 2);
  const length = new Float64Array(edgeCount);
  let next = 0;
  for (const [a, list] of adjacency) {
    for (const e of list) {
      const b = e.neighbor;
      if (b < a) continue;
      // The shared edge is the two vertices the facets have in common. Two identical triangles
      // share all three; any two of them will do for such a pathological pair.
      const shared = [];
      for (let i = 0; i < 3 && shared.length < 2; i++) {
        const v = fv[a * 3 + i];
        if (v >= 0 && (fv[b * 3] === v || fv[b * 3 + 1] === v || fv[b * 3 + 2] === v)) shared.push(v);
      }
      const id = next++;
      e.id = id;
      for (const back of adjacency.get(b) || []) if (back.neighbor === a && back.id === undefined) { back.id = id; break; }
      ev[id * 2] = shared[0] ?? -1;
      ev[id * 2 + 1] = shared[1] ?? -1;
      ef[id * 2] = a;
      ef[id * 2 + 1] = b;
      if (shared.length === 2) {
        const p = where[shared[0]], q = where[shared[1]];
        length[id] = Math.hypot(P[p] - P[q], P[p + 1] - P[q + 1], P[p + 2] - P[q + 2]);
      }
    }
  }

  const vertexCount = where.length;
  const vertStart = new Int32Array(vertexCount + 1);
  for (let i = 0; i < edgeCount * 2; i++) if (ev[i] >= 0) vertStart[ev[i] + 1]++;
  for (let v = 0; v < vertexCount; v++) vertStart[v + 1] += vertStart[v];
  const fill = vertStart.slice(0, vertexCount);
  const vertEdges = new Int32Array(vertStart[vertexCount]);
  for (let id = 0; id < edgeCount; id++) {
    for (let k = 0; k < 2; k++) {
      const v = ev[id * 2 + k];
      if (v >= 0) vertEdges[fill[v]++] = id;
    }
  }
  return { edgeCount, vertexCount, ev, ef, length, vertStart, vertEdges, facetVerts: fv };
}

/** The boundary probability of every edge (NaN where the adjacency carries no score). */
export function edgeScores(graph, adjacency) {
  const p = new Float64Array(graph.edgeCount).fill(NaN);
  for (const [, list] of adjacency) for (const e of list) if (e.id !== undefined && e.boundary !== undefined) p[e.id] = e.boundary;
  return p;
}
