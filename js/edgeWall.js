/**
 * edgeWall.js — score every mesh edge with the learned face-boundary model, and turn that into a
 * wall for growth.
 *
 * The crease barrier cannot see a tangent junction: a fillet running into a plane has no dihedral
 * to notice, and that is where fills spill (with the true type declared, 23% of clicks spill on
 * OCCT meshes and 43% on gmsh). ml/edge_screen.py showed the boundary IS recognisable from the
 * edge's neighbourhood — at a threshold that wrongly walls 2% of within-face edges, a probe caught
 * 84% of smooth (under 30 degree) boundaries on OCCT meshes, against 0% for the crease rule by
 * construction. ml/edge_train.py trains the shipped scorer and writes its thresholds.
 *
 * Scoring is per MESH, not per fill: every edge is scored once in a few batched calls and the
 * probability is written onto the adjacency entries, so the growth loop only compares a number.
 *
 * The model takes the columns js/edgeFeatures.js builds, in the order its sidecar names.
 */

import { edgeFeatures, edgeContext, EDGE_COLUMNS, N_EDGE_FEATURES, CONTEXT_COLUMNS } from './edgeFeatures.js';
import { edgeScores } from './edgeGraph.js';

export const EDGE_BATCH = 65536;

/**
 * Score every manifold edge and annotate the adjacency with `e.boundary` (a probability), on both
 * directions of each edge.
 *
 * @param {object}   facets
 * @param {Map}      adjacency
 * @param {Function} runner  async (features: Float32Array, n: number) => logits (Float32Array-like)
 * @param {string[]} columns the model's input columns, from its sidecar; defaults to all of them
 * @returns {{ edges: number, ms: number }}
 */
export async function annotateEdges(facets, adjacency, runner, columns = EDGE_COLUMNS) {
  const t0 = performance.now();
  const take = columns.map((c) => EDGE_COLUMNS.indexOf(c));
  if (take.some((i) => i < 0)) throw new Error(`unknown edge feature column in ${columns.join(', ')}`);
  const { row } = edgeFeatures(facets, adjacency);
  const scratch = new Float32Array(N_EDGE_FEATURES);

  // One list of (facet, edge) pairs, each edge once; the partner entry is annotated with it.
  const pairs = [];
  for (const [a, edges] of adjacency) {
    for (const e of edges) if (e.neighbor > a) pairs.push([a, e]);
  }
  const width = take.length;
  for (let start = 0; start < pairs.length; start += EDGE_BATCH) {
    const chunk = pairs.slice(start, start + EDGE_BATCH);
    const features = new Float32Array(chunk.length * width);
    for (let i = 0; i < chunk.length; i++) {
      const [a, e] = chunk[i];
      row(a, e.neighbor, e.angle, scratch, 0);
      for (let c = 0; c < width; c++) features[i * width + c] = scratch[take[c]];
    }
    const logits = await runner(features, chunk.length);
    for (let i = 0; i < chunk.length; i++) {
      const [a, e] = chunk[i];
      const p = 1 / (1 + Math.exp(-logits[i]));
      e.boundary = p;
      // The same edge seen from the other side.
      for (const back of adjacency.get(e.neighbor) || []) if (back.neighbor === a) back.boundary = p;
    }
  }
  return { edges: pairs.length, ms: performance.now() - t0 };
}

/**
 * Two-stage (stacked) scoring, experimental: the first-stage model annotates every edge, then a
 * second model re-scores each edge from its first-stage features PLUS the first-stage scores around
 * it (edgeFeatures.CONTEXT_COLUMNS), and its probability replaces e.boundary. `graph` is
 * js/edgeGraph.js's attachEdgeGraph for this mesh.
 */
export async function annotateEdgesStacked(facets, adjacency, graph, runner1, columns1, runner2, columns2) {
  const t0 = performance.now();
  await annotateEdges(facets, adjacency, runner1, columns1);
  const ctx = edgeContext(graph, adjacency, edgeScores(graph, adjacency));
  const K = CONTEXT_COLUMNS.length;
  const take = columns2.map((c) => {
    const i = EDGE_COLUMNS.indexOf(c);
    if (i >= 0) return i;
    const j = CONTEXT_COLUMNS.indexOf(c);
    if (j < 0) throw new Error(`unknown second-stage column ${c}`);
    return N_EDGE_FEATURES + j;
  });
  const { row } = edgeFeatures(facets, adjacency);
  const scratch = new Float32Array(N_EDGE_FEATURES + K);
  const pairs = [];
  for (const [a, edges] of adjacency) for (const e of edges) if (e.neighbor > a) pairs.push([a, e]);
  const width = take.length;
  for (let start = 0; start < pairs.length; start += EDGE_BATCH) {
    const chunk = pairs.slice(start, start + EDGE_BATCH);
    const features = new Float32Array(chunk.length * width);
    for (let i = 0; i < chunk.length; i++) {
      const [a, e] = chunk[i];
      row(a, e.neighbor, e.angle, scratch, 0);
      for (let k = 0; k < K; k++) scratch[N_EDGE_FEATURES + k] = ctx[e.id * K + k];
      for (let c = 0; c < width; c++) features[i * width + c] = scratch[take[c]];
    }
    const logits = await runner2(features, chunk.length);
    for (let i = 0; i < chunk.length; i++) {
      const [a, e] = chunk[i];
      const p = 1 / (1 + Math.exp(-logits[i]));
      e.boundary = p;
      for (const back of adjacency.get(e.neighbor) || []) if (back.neighbor === a) back.boundary = p;
    }
  }
  return { edges: pairs.length, ms: performance.now() - t0 };
}

/**
 * A wall predicate for regionGrow (`opts.wall`) from annotated adjacency: an edge at or above
 * `threshold` is a face boundary and growth stops there. Edges with no score (not annotated) are
 * never walls, so a mesh that was not scored behaves as before.
 */
export function edgeWall(threshold, types = null) {
  const wall = (_t, e) => e.boundary !== undefined && e.boundary >= threshold;
  // Which fill types it applies to (null = all). paintFrom skips it for the others: PLANE fills
  // regress with it (clean 94 -> 90% OCCT, 94 -> 80% NURBS) because a large flat face can pick up a
  // wall inside it, and the 2 degree plane crease already stops a plane at a tangent junction.
  wall.types = types;
  wall.threshold = threshold;
  return wall;
}

/**
 * HYSTERESIS (Canny's rule, along mesh edges; experimental): an edge scoring at least `hi` is a wall,
 * and an edge scoring at least `lo` becomes one when it CONTINUES a wall — when a chain of edges
 * scoring at least `lo`, meeting at shared vertices, links it to an edge at `hi`. The idea is that a
 * face boundary is a curve: a weak edge on the same curve as strong ones is probably boundary too,
 * and a flood fill leaks through any single gap. With `minChain`, a wall whose chain holds fewer edges
 * is dropped, since a lone false wall inside a face can still cut a thin one.
 *
 * Needs js/edgeGraph.js's attachEdgeGraph (edge ids and vertices) on an annotated adjacency.
 */
export function hysteresisWall(graph, adjacency, hi, lo, { types = null, minChain = 1 } = {}) {
  const E = graph.edgeCount;
  const p = new Float64Array(E);
  for (const [, list] of adjacency) for (const e of list) if (e.id !== undefined) p[e.id] = e.boundary ?? 0;
  const root = new Int32Array(E);
  for (let i = 0; i < E; i++) root[i] = i;
  const find = (i) => { while (root[i] !== i) { root[i] = root[root[i]]; i = root[i]; } return i; };
  const { vertStart, vertEdges } = graph;
  for (let v = 0; v + 1 < vertStart.length; v++) {
    let first = -1;
    for (let k = vertStart[v]; k < vertStart[v + 1]; k++) {
      const id = vertEdges[k];
      if (!(p[id] >= lo)) continue;
      if (first < 0) { first = id; continue; }
      const a = find(first), b = find(id);
      if (a !== b) root[a] = b;
    }
  }
  const strong = new Uint8Array(E);
  const size = new Int32Array(E);
  for (let i = 0; i < E; i++) {
    if (!(p[i] >= lo)) continue;
    const r = find(i);
    size[r]++;
    if (p[i] >= hi) strong[r] = 1;
  }
  const isWall = new Uint8Array(E);
  let walls = 0;
  for (let i = 0; i < E; i++) {
    if (!(p[i] >= lo)) continue;
    const r = find(i);
    if (strong[r] && size[r] >= minChain) { isWall[i] = 1; walls++; }
  }
  const wall = (_t, e) => e.id !== undefined && isWall[e.id] === 1;
  wall.types = types;
  wall.walls = walls;
  wall.isWall = isWall;
  return wall;
}

/** Is this adjacency annotated? */
export function hasEdgeScores(adjacency) {
  for (const [, edges] of adjacency) {
    for (const e of edges) return e.boundary !== undefined;
  }
  return false;
}
