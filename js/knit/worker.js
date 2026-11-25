/**
 * knit/worker.js — the studio's STEP export and Repair off the main thread: the geometry kernel's builds
 * run for seconds to minutes and would freeze the page. One build per worker (a build that throws can leave
 * the kernel's memory corrupt), but the kernel starts loading as soon as the worker does: the studio starts the
 * next worker when a mesh opens and after each build, so Export rarely waits for the 50 MB kernel.
 *
 * A part with freeform groups (painted groups with no surface) goes the face path first (knit/facePath.js: one
 * trimmed face per group, the STEP written without the kernel, which only reads it back to judge and tessellate it),
 * and to the knitter when that is not closed, valid and of the mesh's volume (options.facePath false: the knitter
 * only); report.info.path is 'faces' when it was taken, report.info.face_path why not when it was not. A knitted
 * solid with facets in it (or no solid) then gets the face path once more, with the groups the knitter could build
 * only as facets fitted as B-splines (knit/facePath.js faceAfterKnit); report.info.face_after_knit says why not
 * when the knitter's result stays. Before that rung the worker posts {type: 'interim', step, report}: the knitter's
 * result, which the studio keeps if it stops the worker during the rung.
 *
 * Message in:  {positions: Float64Array (9 per triangle), labels: Int32Array, surfaces: [[id, [type,
 *               params]], ...], options: {budget?, fallback?, deadline?, mode?, keep?, facePath?}, repair?: true,
 *               bodies?: Int32Array (a body id per triangle), names?: {id: name}, oriented?: true, part?: name}
 * Messages out: {type: 'status', text} while the kernel loads, {type: 'progress', builds, ok, trying?, body?} per
 *               build; with several bodies {type: 'body', id, name, brep, report} as each is built (BRep text: a
 *               hard stop keeps these, and the next run gets them back as options.keep, [[id, {brep, report}]], with
 *               mode 'faceted' for the rest); then {type: 'done', step, report, preview} or {type: 'error', message}; preview is the
 *               solid tessellated for the studio to show before saving (knit/preview.js), or null. With repair: no STEP;
 *               {type: 'move', move} for each move the loop keeps (knit/index.js repairGroups), then
 *               {type: 'done', repair: {ok, labels, surfaces: [[id, [type, params]], ...], moves, info, problems,
 *               bodies?}}: with several bodies each is repaired alone (knit/index.js repairGroups), bodies a line each.
 */

import { loadOcct, knitSolid, repairGroups, writeStep, brepText } from './index.js';
import { previewMesh } from './preview.js';
import { hasFreeform, tryFacePath, faceAfterKnit } from './facePath.js';

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
      const r = repairGroups(oc, { positions, labels, surfaces: new Map(surfaces), bodies, oriented },
        { ...options, onProgress, onMove: (move) => self.postMessage({ type: 'move', move }) });
      self.postMessage({ type: 'done', repair: { ...r, surfaces: [...r.surfaces] } });
      return;
    }
    const keep = options.keep ? new Map(options.keep) : null;
    const tStart = performance.now();
    // a part with freeform groups: one face per painted group first (knit/facePath.js), taken when it is as good as
    // the knitter's own bar; else the knitter, as before
    let faceTried = null, faceSecs = 0;
    if (options.facePath !== false && !keep && hasFreeform(labels, new Map(surfaces))) {
      const tf = performance.now();
      self.postMessage({ type: 'status', text: 'Building one face per painted group…' });
      const fp = tryFacePath(oc, { positions, labels, surfaces: new Map(surfaces), oriented }, { part, budgetS: (options.deadline ?? 150) / 3 });
      if (fp.step) {
        const { preview } = fp;
        const buffers = preview ? [preview.positions.buffer, preview.normals.buffer, preview.face.buffer, preview.edges.buffer] : [];
        self.postMessage({ type: 'done', step: fp.step, report: fp.report, preview }, buffers);
        return;
      }
      faceTried = fp;
      faceSecs = (performance.now() - tf) / 1000;
    }
    // the studio's hard stop counts from the start: the knitter's deadline less what the face path took (a fifth of it
    // at least), so it still ends when it would have
    const knitOptions = { ...options };
    if (faceSecs && Number.isFinite(options.deadline)) knitOptions.deadline = Math.max(options.deadline / 5, options.deadline - faceSecs);
    self.postMessage({ type: 'status', text: 'Building the solid…' });
    const onBody = ({ id, name, solid: s, report: r }) => {
      let brep = null;
      try { brep = s ? brepText(oc, s) : null; } catch (err) { console.warn('no BRep for body', id, err?.message ?? err); }
      self.postMessage({ type: 'body', id, name, brep, report: r });
    };
    const { solid, report, names: solidNames } = knitSolid(oc, { positions, labels, surfaces: new Map(surfaces), bodies, names, oriented },
      { ...knitOptions, keep, onProgress, onBody });
    if (faceTried) report.info = { ...report.info, face_path: faceTried.rejected, face_path_info: faceTried.info };
    let interimStep = null;
    // never facets: a solid with facets in it (or none at all) gets the face path as one more rung, the facet groups
    // fitted as B-splines (knit/facePath.js faceAfterKnit); the knitter's result stays when that does not pass
    if (!keep && options.facePath !== false && (!solid || (report.facetArea ?? 0) > 0)) {
      // the knitter's own result first, so a hard stop during the rung keeps it rather than faceting the whole mesh
      if (solid) {
        try { interimStep = writeStep(oc, solid, solidNames ? { names: solidNames, part } : {}); self.postMessage({ type: 'interim', step: interimStep, report }); }
        catch (err) { console.warn('no interim STEP:', err?.message ?? err); }
      }
      self.postMessage({ type: 'status', text: 'Fitting B-spline faces where the solid has facets…' });
      const left = (options.deadline ?? 150) - (performance.now() - tStart) / 1000;
      const fa = faceAfterKnit(oc, { positions, labels, surfaces: new Map(surfaces), oriented }, report,
        { part, budgetS: Math.max(30, left), triedAsPainted: !!faceTried });
      if (fa.step) {
        const { preview } = fa;
        const buffers = preview ? [preview.positions.buffer, preview.normals.buffer, preview.face.buffer, preview.edges.buffer] : [];
        self.postMessage({ type: 'done', step: fa.step, report: fa.report, preview }, buffers);
        return;
      }
      report.info = { ...report.info, face_after_knit: fa.rejected };
    }
    self.postMessage({ type: 'status', text: 'Writing STEP…' });
    // (the interim STEP is this one: the rung leaves the knitter's solid as it was)
    const step = solid ? interimStep ?? writeStep(oc, solid, solidNames ? { names: solidNames, part } : {}) : null;
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
