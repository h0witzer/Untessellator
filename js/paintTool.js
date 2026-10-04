/**
 * paintTool.js — the browser's fill: js/paintCore.js with onnxruntime-web attached.
 *
 * Loaded only when the user first paints, so the ONNX runtime is never fetched otherwise. The fill
 * logic itself lives in paintCore.js, shared with the Node harnesses (test/autoSeed.mjs), so what
 * they measure is what the studio runs.
 */

import * as ort from 'onnxruntime-web';
import { preparePaintPositions, paintWith } from './paintCore.js';
import { graphInputs } from './seedNet.js';

export { FREEFORM_CREASE, ORDER_RANK, typeFinalGroup } from './paintCore.js';
export { freeformMerge } from './freeformMerge.js';

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
ort.env.wasm.numThreads = 1;   // no cross-origin isolation here, so threads are unavailable anyway

const sessions = new Map();

/** One ONNX session per model URL, created on first use. */
export function loadModel(url) {
  if (!sessions.has(url)) {
    sessions.set(url, ort.InferenceSession.create(url, { executionProviders: ['wasm'] }));
  }
  return sessions.get(url);
}

const sidecars = new Map();
/** A model's sidecar JSON (ml/export_onnx.py, ml/refine_train.py), or {} when it has none. */
function sidecarFor(url) {
  if (!sidecars.has(url)) {
    sidecars.set(url, fetch(url.replace(/\.onnx$/, '.json'))
      .then((r) => (r.ok ? r.json() : {})).catch(() => ({})));
  }
  return sidecars.get(url);
}

function makeRunner(session, edgeDim = 1) {
  // A message-passing model (ml/model.py MeshConv, e.g. v7gnn) also takes the patch graph that
  // seedNet.patchNeighbours builds; other models ignore it.
  const wantsGraph = session.inputNames.includes('nbr');
  return async (features, n, graph = null) => {
    const feeds = {
      features: new ort.Tensor('float32', features, [1, n, features.length / n]),
      mask: new ort.Tensor('float32', new Float32Array(n).fill(1), [1, n]),
    };
    if (wantsGraph) {
      if (!graph) throw new Error('this model needs the patch graph (seedNet.patchNeighbours)');
      const g = graphInputs(graph, n, edgeDim);
      feeds.nbr = new ort.Tensor('int64', ...g.nbr);
      feeds.eattr = new ort.Tensor('float32', ...g.eattr);
    }
    const out = await session.run(feeds);
    return {
      sameLogit: out.same_logit.data,
      typeLogit: Array.from(out.type_logit.data),
      surface: out.surface ? out.surface.data : null,
    };
  };
}

/** The model runner for a URL, for callers that fill many times (auto-paint). */
export async function runnerFor(url) {
  const [session, meta] = await Promise.all([loadModel(url), sidecarFor(url)]);
  return makeRunner(session, meta.edge_dim ?? 1);
}

/**
 * The learned face-boundary scorer (js/edgeWall.js) and its sidecar: one session per URL.
 * Returns `{ runner, meta }`, where meta names the input columns and the thresholds, each labelled
 * by the share of within-face edges it wrongly walls.
 */
export async function edgeScorerFor(url) {
  if (!sessions.has(url)) {
    sessions.set(url, Promise.all([
      ort.InferenceSession.create(url, { executionProviders: ['wasm'] }),
      fetch(url.replace(/\.onnx$/, '.json')).then((r) => r.json()),
    ]).then(([session, meta]) => ({
      meta,
      runner: async (features, n) => {
        const out = await session.run({
          features: new ort.Tensor('float32', features, [n, features.length / n]),
        });
        return out.logit.data;
      },
    })));
  }
  return sessions.get(url);
}

/**
 * Facets, adjacency and curvature for a loaded three.js geometry. Triangle order matches the
 * geometry, so viewer and studio triangle indices apply.
 */
export function preparePaintMesh(geometry) {
  const geo = geometry.index ? geometry.toNonIndexed() : geometry;
  return preparePaintPositions(geo.attributes.position.array);
}

/**
 * Fill from one clicked triangle. See paintCore.paintWith for the options; `modelUrl` names the
 * model, loaded on first use. `refineUrl` (the fill-refinement model) refines the fill with
 * paintCore.refineFill; it needs the mesh's edges scored by the boundary model first.
 */
export async function paintAt(seed, mesh, { method, type, modelUrl, taken = null, extended = false,
                                             grow = {}, reportFit = true, refineUrl = null }) {
  const needsModel = method !== 'plain' || type === 'auto';
  const runner = needsModel ? await runnerFor(modelUrl) : null;
  const refine = refineUrl ? { runner: await runnerFor(refineUrl) } : null;
  return paintWith(runner, seed, mesh, { method, type, taken, extended, grow, reportFit, refine });
}
