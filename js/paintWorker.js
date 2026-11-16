/**
 * paintWorker.js — the studio's fills off the main thread. A fill runs the seed model, grows and refits its region
 * and may refine it with a second model: tens of milliseconds to seconds, which froze the viewport while it ran.
 * The learned boundary's scoring of every edge (seconds on a big mesh) and auto-paint's seed scores run here too.
 *
 * The worker keeps its own copy of the mesh's facets and adjacency (paintCore.preparePaintPositions on the same
 * positions, so triangle indices match the page's), and the edge scores live only on this copy: growth is their
 * only reader. Jobs run one at a time, in the order they came, so a fill never sees half-scored edges.
 *
 * Messages in (each with an `id` the reply carries, and every one but `load` with the `key` of the mesh it is for):
 *   {op: 'load', key, positions: Float32Array (9 per triangle)}   this mesh replaces the last; its scores are dropped
 *   {op: 'scoreEdges', url, wallTypes}   score every edge with the boundary model at `url` (absolute); the wall stops
 *                                        fills of `wallTypes` at the model's 2% false-wall threshold
 *   {op: 'coherence'}                    autoPaint.coherentScores, Float64Array
 *   {op: 'fill', seed, opts}             paintTool.paintAt's options, with `wall: true` for the learned wall (when
 *                                        the edges are scored) in place of a wall function
 *   {op: 'typer', url, features, n}      the group typer's probabilities and threshold
 *   {op: 'freeformMerge', owner, only}   freeformMerge.js on a copy of the owners: {owner, freeform (group ids)}
 *   {op: 'typeGroups', groups: [{id, tris: Int32Array, type, freeform}], typerUrl}   auto-paint's last stage, each
 *                                        new group's export type and fit: [{id, type, typedBy, typer, fit}]
 * Replies: {id, result} or {id, error}; a long job also sends {id, progress} as it goes. A fill's result is paintWith's, its region an Int32Array and without
 * the parts the page never reads (the model's own region, the refiner's per-facet membership).
 */

import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.wasm.min.mjs';   // = ortModels.ORT_URL
import { ortModels } from './ortModels.js';
import { preparePaintPositions, paintWith } from './paintCore.js';
import { annotateEdges, edgeWall } from './edgeWall.js';
import { coherentScores } from './autoPaint.js';
import { typeFinalGroup } from './paintCore.js';
import { fitRegionAs } from './gaussMap.js';
import { freeformMerge } from './freeformMerge.js';
import { groupTypeFeatures, decideGroupType, GROUP_TYPES, N_GROUP_FEATURES } from './groupType.js';

const models = ortModels(ort);

/** The mesh in hand: {key, mesh (preparePaintPositions), scored: {url, wall} | null}. */
let current = null;

const ops = {
  load({ key, positions }) {
    current = { key, mesh: preparePaintPositions(positions), scored: null };
    return { triangles: current.mesh.facets.count };
  },

  async scoreEdges({ url, wallTypes }) {
    if (current.scored?.url === url) return { url, edges: 0, ms: 0, cached: true };
    const { runner, meta } = await models.edgeScorerFor(url);
    const { facets, adjacency } = current.mesh;
    const { edges, ms } = await annotateEdges(facets, adjacency, runner, meta.columns);
    const logit = meta.thresholds['fpr0.02'].logit;
    current.scored = { url, wall: edgeWall(1 / (1 + Math.exp(-logit)), wallTypes) };
    return { url, edges, ms };
  },

  coherence() {
    return coherentScores(current.mesh.facets, current.mesh.adjacency, { chunk: Infinity });
  },

  async fill({ seed, opts }) {
    const { method, type, modelUrl, taken = null, extended = false, grow = {}, wall = false, reportFit = true,
            refineUrl = null } = opts;
    const needsModel = method !== 'plain' || type === 'auto';
    const runner = needsModel ? await models.runnerFor(modelUrl) : null;
    const refine = refineUrl ? { runner: await models.runnerFor(refineUrl) } : null;
    const growWith = wall && current.scored ? { ...grow, wall: current.scored.wall } : grow;
    const out = await paintWith(runner, seed, current.mesh, { method, type, taken, extended, grow: growWith, reportFit, refine });
    const { modelRegion, member, hypothesis, ...rest } = out;
    return { ...rest, region: Int32Array.from(out.region) };
  },

  async typer({ url, features, n }) {
    const typer = await models.groupTyperFor(url);
    // copied: the runtime's output array is its own, and only a buffer of ours may be handed over
    return { probs: Float32Array.from(await typer.run(features, n)), threshold: typer.threshold };
  },

  freeformMerge({ owner, only }) {
    const { facets, adjacency } = current.mesh;
    const { freeform } = freeformMerge(owner, facets, adjacency, { only: new Set(only) });
    return { owner, freeform: [...freeform] };
  },

  /**
   * Auto-paint's finished groups, typed and fitted as runAutoPaint did on the page: the learned typer decides where it
   * is confident (groups of 3 or more triangles), typeFinalGroup the rest; freeform-merged groups stay untyped. A
   * typer that fails to load leaves every group to typeFinalGroup. Progress: {phase: 'weigh' | 'type', done, of}.
   */
  async typeGroups({ groups, typerUrl }, progress) {
    const { facets, adjacency } = current.mesh;
    const sets = new Map(groups.map((g) => [g.id, new Set(g.tris)]));
    const learned = new Map();
    const typer = typerUrl ? await models.groupTyperFor(typerUrl).catch(() => null) : null;
    const tick = clock(progress);
    if (typer) {
      const rows = [];
      groups.forEach((g, i) => {
        if (!g.freeform) {
          const f = groupTypeFeatures(sets.get(g.id), facets, adjacency, g.type);
          if (f) rows.push([g.id, f]);
        }
        tick({ phase: 'weigh', done: i + 1, of: groups.length });
      });
      if (rows.length) {
        const X = new Float32Array(rows.length * N_GROUP_FEATURES), k = GROUP_TYPES.length;
        rows.forEach(([, f], i) => X.set(f.features, i * N_GROUP_FEATURES));
        const probs = await typer.run(X, rows.length);
        rows.forEach(([id, f], i) => learned.set(id, decideGroupType(probs.subarray(i * k, i * k + k), f.rule, typer.threshold)));
      }
    }
    return groups.map((g, i) => {
      const tris = sets.get(g.id);
      let type = g.type, typedBy, by = null;
      if (g.freeform) { type = null; typedBy = 'freeform merge'; }
      else if (learned.has(g.id)) {
        const d = learned.get(g.id);
        type = d.type;
        by = { by: d.by, p: Math.round(d.p * 1000) / 1000 };
        typedBy = d.by === 'model' ? 'group typer' : 'typer rule';
      } else {
        const typed = typeFinalGroup(tris, facets, g.type);
        if (typed.changed) type = typed.type;
        typedBy = typed.changed ? 'exact rule' : 'fill';
      }
      let fit = null;
      if (type && type !== 'skip' && tris.size) {
        const r = fitRegionAs(type, tris, facets);
        fit = { params: r.params ?? null, relResidual: r.relResidual, reason: r.reason };
      }
      tick({ phase: 'type', done: i + 1, of: groups.length });
      return { id: g.id, type, typedBy, typer: by, fit };
    });
  },
};

/** A progress reporter that sends at most every 40 ms, and always the last step. */
function clock(progress) {
  let last = 0;
  return (p) => {
    const now = performance.now();
    if (now - last < 40 && p.done < p.of) return;
    last = now;
    progress(p);
  };
}

/** The buffers a result can hand over rather than copy. */
const transferables = (result) => [result?.region, result?.probs, result?.owner, result instanceof Float64Array ? result : null]
  .filter((a) => ArrayBuffer.isView(a)).map((a) => a.buffer);

let queue = Promise.resolve();
self.onmessage = ({ data }) => {
  queue = queue.then(async () => {
    const { id, op, key } = data;
    try {
      if (!(op in ops)) throw new Error(`paint worker: no op ${op}`);
      if (op !== 'load' && op !== 'typer' && current?.key !== key) throw new Error('paint worker: that mesh is no longer loaded');
      const result = await ops[op](data, (progress) => self.postMessage({ id, progress }));
      self.postMessage({ id, result }, transferables(result));
    } catch (error) {
      self.postMessage({ id, error: String(error?.message ?? error) });
    }
  });
};
