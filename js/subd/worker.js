/**
 * subd/worker.js — the freeform STEP export (freeform.js) off the main thread, with the knit worker's messages so
 * the studio can run either: no geometry kernel to load (QuadriFlow, 0.6 MB, starts loading with the worker).
 *
 * Message in:  {positions: Float64Array (9 per triangle), bodies?: Int32Array (a body id per triangle), names?:
 *               {id: name}, part?: name, options?: {quads?}}
 * Messages out: {type: 'status', text} as it goes; then {type: 'done', step, report, preview} (step null when no
 *               body could be built; report.bodies says why for each) or {type: 'error', message}.
 */
import { freeformSolid } from './freeform.js';
import { loadQuadriFlow } from './retopo.js';

loadQuadriFlow().catch(() => {});                 // warm; a failed load is retried, and reported, by the export

self.onmessage = async ({ data }) => {
  const { positions, bodies = null, names = null, part = 'Part', options = {} } = data;
  try {
    const onStatus = (text) => self.postMessage({ type: 'status', text });
    const { step, report, preview } = await freeformSolid(positions, { bodies, names, part, quads: options.quads ?? null, onStatus });
    const buffers = preview ? [preview.positions.buffer, preview.normals.buffer, preview.face.buffer, preview.edges.buffer] : [];
    self.postMessage({ type: 'done', step, report, preview }, buffers);
  } catch (err) {
    self.postMessage({ type: 'error', message: String(err?.message ?? err) });
  }
};
