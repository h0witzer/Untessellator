/**
 * knit/worker.js — the studio's STEP export and Repair off the main thread: the geometry kernel's builds
 * run for seconds to minutes and would freeze the page. One build per worker (a build that throws can leave
 * the kernel's memory corrupt), but the kernel starts loading as soon as the worker does: the studio starts the
 * next worker when a mesh opens and after each build, so Export rarely waits for the 50 MB kernel.
 *
 * Message in:  {positions: Float64Array (9 per triangle), labels: Int32Array, surfaces: [[id, [type,
 *               params]], ...], options: {budget?, fallback?, deadline?, mode?}, repair?: true}
 * Messages out: {type: 'status', text} while the kernel loads, {type: 'progress', builds, ok, trying?} per
 *               build, then {type: 'done', step, report, preview} or {type: 'error', message}; preview is the
 *               solid tessellated for the studio to show before saving (knit/preview.js), or null. With repair: no STEP;
 *               {type: 'move', move} for each move the loop keeps (knit/index.js repairGroups), then
 *               {type: 'done', repair: {ok, labels, surfaces: [[id, [type, params]], ...], moves, info, problems}}.
 */

import { loadOcct, knitSolid, repairGroups, writeStep } from './index.js';
import { previewMesh } from './preview.js';

// warm: loading now, before any build is asked for
let ready = false;
const kernel = loadOcct().then((oc) => { ready = true; return oc; });
kernel.catch(() => {});                           // a failed warm load is retried, with its status, by the build

self.onmessage = async ({ data }) => {
  const { positions, labels, surfaces, options = {}, repair = false } = data;
  try {
    const onStatus = (text) => self.postMessage({ type: 'status', text });
    if (!ready) onStatus('Still loading the geometry kernel (about 50 MB the first time, then cached)…');
    const oc = await kernel.catch(() => loadOcct({ onStatus }));
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
    // the preview: tessellated to a thousandth of the part's diagonal; without one the studio saves at once
    let preview = null;
    if (step) {
      self.postMessage({ type: 'status', text: 'Tessellating the solid for the preview…' });
      try {
        preview = previewMesh(oc, solid, 1e-3 * diagonal(positions));
      } catch (err) {
        console.warn('no preview:', err?.message ?? err);
      }
    }
    const buffers = preview ? [preview.positions.buffer, preview.normals.buffer, preview.face.buffer, preview.edges.buffer] : [];
    self.postMessage({ type: 'done', step, report, preview }, buffers);
  } catch (err) {
    self.postMessage({ type: 'error', message: String(err?.message ?? err) });
  }
};

/** The diagonal of the triangles' bounding box. */
function diagonal(P) {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < P.length; i++) { const k = i % 3; if (P[i] < lo[k]) lo[k] = P[i]; if (P[i] > hi[k]) hi[k] = P[i]; }
  return Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
}
