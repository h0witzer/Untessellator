/**
 * ortModels.js — the browser's ONNX model runners, for whichever copy of onnxruntime-web the caller has: the page
 * imports it through index.html's import map (js/paintTool.js), the paint worker by URL (js/paintWorker.js; a worker
 * does not see the page's import map). Pass the `ort` module; every session is created once per model URL.
 */

import { graphInputs } from './seedNet.js';

/** onnxruntime-web: the version index.html's import map names, and where its WebAssembly comes from. */
export const ORT_VERSION = '1.22.0';
export const ORT_URL = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/ort.wasm.min.mjs`;

export function ortModels(ort) {
  ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
  ort.env.wasm.numThreads = 1;   // no cross-origin isolation here, so threads are unavailable anyway

  const sessions = new Map();
  /** One ONNX session per model URL, created on first use. */
  const loadModel = (url) => {
    if (!sessions.has(url)) sessions.set(url, ort.InferenceSession.create(url, { executionProviders: ['wasm'] }));
    return sessions.get(url);
  };

  const sidecars = new Map();
  /** A model's sidecar JSON (ml/export_onnx.py, ml/refine_train.py), or {} when it has none. */
  const sidecarFor = (url) => {
    if (!sidecars.has(url)) {
      sidecars.set(url, fetch(url.replace(/\.onnx$/, '.json'))
        .then((r) => (r.ok ? r.json() : {})).catch(() => ({})));
    }
    return sidecars.get(url);
  };

  const makeRunner = (session, edgeDim = 1) => {
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
  };

  /**
   * The group-level type model (js/groupType.js, ml/group_type_train.py) and the threshold its sidecar
   * names: `{ run: async (features, n) => probs, threshold }`. Probabilities are softmax over GROUP_TYPES.
   */
  const groupTyperFor = async (url) => {
    const [session, meta] = await Promise.all([loadModel(url), sidecarFor(url)]);
    return {
      threshold: meta.threshold ?? 1.01,
      run: async (features, n) => (await session.run({
        features: new ort.Tensor('float32', features, [n, features.length / n]),
      })).probs.data,
    };
  };

  /** The model runner for a URL, for callers that fill many times (auto-paint). */
  const runnerFor = async (url) => {
    const [session, meta] = await Promise.all([loadModel(url), sidecarFor(url)]);
    return makeRunner(session, meta.edge_dim ?? 1);
  };

  const scorers = new Map();
  /**
   * The learned face-boundary scorer (js/edgeWall.js) and its sidecar: one session per URL.
   * Returns `{ runner, meta }`, where meta names the input columns and the thresholds, each labelled
   * by the share of within-face edges it wrongly walls.
   */
  const edgeScorerFor = (url) => {
    if (!scorers.has(url)) {
      scorers.set(url, Promise.all([
        loadModel(url),
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
    return scorers.get(url);
  };

  return { loadModel, runnerFor, groupTyperFor, edgeScorerFor };
}
