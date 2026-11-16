/**
 * paintTool.js — the browser's fill: js/paintCore.js with onnxruntime-web attached.
 *
 * Loaded only when the user first paints, so the ONNX runtime is never fetched otherwise. The fill
 * logic itself lives in paintCore.js, shared with the Node harnesses (test/autoSeed.mjs), so what
 * they measure is what the studio runs. The studio runs its fills in js/paintWorker.js, which loads the
 * same code; on the page this module serves the work that stays there (typing and merging groups).
 */

import * as ort from 'onnxruntime-web';
import { preparePaintPositions, paintWith } from './paintCore.js';
import { ortModels } from './ortModels.js';

export { FREEFORM_CREASE, ORDER_RANK, typeFinalGroup } from './paintCore.js';
export { freeformMerge } from './freeformMerge.js';
export { groupTypeFeatures, decideGroupType, GROUP_TYPES, N_GROUP_FEATURES } from './groupType.js';

// The sessions and runners (js/ortModels.js), shared with the paint worker (js/paintWorker.js).
export const { loadModel, runnerFor, groupTyperFor, edgeScorerFor } = ortModels(ort);

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
