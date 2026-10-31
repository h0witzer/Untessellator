/**
 * knit/worker.js — the studio's STEP export and Repair off the main thread: the geometry kernel's builds
 * run for seconds to minutes and would freeze the page.
 *
 * Message in:  {positions: Float64Array (9 per triangle), labels: Int32Array, surfaces: [[id, [type,
 *               params]], ...], options: {budget?, fallback?, deadline?, mode?}, repair?: true}
 * Messages out: {type: 'status', text} while the kernel loads, {type: 'progress', builds, ok, trying?} per
 *               build, then {type: 'done', step, report} or {type: 'error', message}. With repair: no STEP;
 *               {type: 'move', move} for each move the loop keeps (knit/index.js repairGroups), then
 *               {type: 'done', repair: {ok, labels, surfaces: [[id, [type, params]], ...], moves, info, problems}}.
 */

import { loadOcct, knitSolid, repairGroups, writeStep } from './index.js';

self.onmessage = async ({ data }) => {
  const { positions, labels, surfaces, options = {}, repair = false } = data;
  try {
    const oc = await loadOcct({ onStatus: (text) => self.postMessage({ type: 'status', text }) });
    const onProgress = (p) => self.postMessage({ type: 'progress', ...p });
    if (repair) {
      self.postMessage({ type: 'status', text: 'Building the solid to find what fails…' });
      const r = repairGroups(oc, { positions, labels, surfaces: new Map(surfaces) },
        { ...options, onProgress, onMove: (move) => self.postMessage({ type: 'move', move }) });
      self.postMessage({ type: 'done', repair: { ...r, surfaces: [...r.surfaces] } });
      return;
    }
    self.postMessage({ type: 'status', text: 'Building the solid…' });
    const { solid, report } = knitSolid(oc, { positions, labels, surfaces: new Map(surfaces) }, { ...options, onProgress });
    self.postMessage({ type: 'status', text: 'Writing STEP…' });
    const step = solid ? writeStep(oc, solid) : null;
    self.postMessage({ type: 'done', step, report });
  } catch (err) {
    self.postMessage({ type: 'error', message: String(err?.message ?? err) });
  }
};
