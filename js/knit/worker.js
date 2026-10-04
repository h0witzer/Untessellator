/**
 * knit/worker.js — the studio's STEP export off the main thread: the geometry kernel's builds and
 * repairs run for seconds to minutes and would freeze the page.
 *
 * Message in:  {positions: Float64Array (9 per triangle), labels: Int32Array, surfaces: [[id, [type,
 *               params]], ...], options: {budget?, loop?, fallback?}}
 * Messages out: {type: 'status', text} while the kernel loads, {type: 'progress', builds, ok} per
 *               build, then {type: 'done', step, report} or {type: 'error', message}.
 */

import { loadOcct, knitSolid, writeStep } from './index.js';

self.onmessage = async ({ data }) => {
  const { positions, labels, surfaces, options = {} } = data;
  try {
    const oc = await loadOcct({ onStatus: (text) => self.postMessage({ type: 'status', text }) });
    self.postMessage({ type: 'status', text: 'Building the solid…' });
    const { solid, report } = knitSolid(oc, { positions, labels, surfaces: new Map(surfaces) },
      { ...options, onProgress: (p) => self.postMessage({ type: 'progress', ...p }) });
    self.postMessage({ type: 'status', text: 'Writing STEP…' });
    const step = solid ? writeStep(oc, solid) : null;
    self.postMessage({ type: 'done', step, report });
  } catch (err) {
    self.postMessage({ type: 'error', message: String(err?.message ?? err) });
  }
};
