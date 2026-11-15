/**
 * knit/worker.js — the studio's STEP export and Repair off the main thread: the geometry kernel's builds
 * run for seconds to minutes and would freeze the page. One build per worker (a build that throws can leave
 * the kernel's memory corrupt), but the kernel starts loading as soon as the worker does: the studio starts the
 * next worker when a mesh opens and after each build, so Export rarely waits for the 50 MB kernel.
 *
 * Message in:  {positions: Float64Array (9 per triangle), labels: Int32Array, surfaces: [[id, [type,
 *               params]], ...], options: {budget?, fallback?, deadline?, mode?, keep?}, repair?: true,
 *               bodies?: Int32Array (a body id per triangle), names?: {id: name}, oriented?: true, part?: name}
 * Messages out: {type: 'status', text} while the kernel loads, {type: 'progress', builds, ok, trying?, body?} per
 *               build; with several bodies {type: 'body', id, name, brep, report} as each is built (BRep text: a
 *               hard stop keeps these, and the next run gets them back as options.keep, [[id, {brep, report}]], with
 *               mode 'faceted' for the rest); then {type: 'done', step, report, preview} or {type: 'error', message}; preview is the
 *               solid tessellated for the studio to show before saving (knit/preview.js), or null. With repair: no STEP;
 *               {type: 'move', move} for each move the loop keeps (knit/index.js repairGroups), then
 *               {type: 'done', repair: {ok, labels, surfaces: [[id, [type, params]], ...], moves, info, problems}}.
 */

import { loadOcct, knitSolid, repairGroups, writeStep, brepText } from './index.js';
import { previewMesh } from './preview.js';

// warm: loading now, before any build is asked for
let ready = false;
const kernel = loadOcct().then((oc) => { ready = true; return oc; });
kernel.catch(() => {});                           // a failed warm load is retried, with its status, by the build

self.onmessage = async ({ data }) => {
  const { positions, labels, surfaces, options = {}, repair = false, bodies = null, names = null, oriented = false, part = 'Part' } = data;
  try {
    const onStatus = (text) => self.postMessage({ type: 'status', text });
    if (!ready) onStatus('Still loading the geometry kernel (about 50 MB the first time, then cached)…');
    const oc = await kernel.catch(() => loadOcct({ onStatus }));
    const onProgress = (p) => self.postMessage({ type: 'progress', ...p });
    if (repair) {
      self.postMessage({ type: 'status', text: 'Building the solid to find what fails…' });
      const r = repairGroups(oc, { positions, labels, surfaces: new Map(surfaces), oriented },
        { ...options, onProgress, onMove: (move) => self.postMessage({ type: 'move', move }) });
      self.postMessage({ type: 'done', repair: { ...r, surfaces: [...r.surfaces] } });
      return;
    }
    self.postMessage({ type: 'status', text: 'Building the solid…' });
    const keep = options.keep ? new Map(options.keep) : null;
    const onBody = ({ id, name, solid: s, report: r }) => {
      let brep = null;
      try { brep = s ? brepText(oc, s) : null; } catch (err) { console.warn('no BRep for body', id, err?.message ?? err); }
      self.postMessage({ type: 'body', id, name, brep, report: r });
    };
    const { solid, report, names: solidNames } = knitSolid(oc, { positions, labels, surfaces: new Map(surfaces), bodies, names, oriented },
      { ...options, keep, onProgress, onBody });
    self.postMessage({ type: 'status', text: 'Writing STEP…' });
    const step = solid ? writeStep(oc, solid, solidNames ? { names: solidNames, part } : {}) : null;
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
