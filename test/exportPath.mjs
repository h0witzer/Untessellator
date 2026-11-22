// The studio's Export STEP as the worker runs it (js/knit/worker.js), on one painted part: the face path first for a
// part with freeform groups (js/knit/facePath.js), the knitter (knitSolid, the studio's 150 s deadline) when that is
// not as good; --knit-only: the knitter only, as before the face path. Input as test/faceExport.mjs's (<stem>.pos,
// .fid, .fits.json: test/studioExport.mjs --dump). Writes <out>/step/<stem>.step; one JSON line on stdout.
// --face-only: the face path alone (nothing written when it is not taken).
// node test/exportPath.mjs <dump dir> <stem> <out dir> [--knit-only | --face-only] [--deadline 150]
import fs from 'node:fs';
import path from 'node:path';
import { loadOcct, knitSolid, writeStep } from '../js/knit/index.js';

const args = process.argv.slice(2);
const flag = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args.splice(i, 2)[1] : d; };
const bool = (k) => { const i = args.indexOf(k); if (i < 0) return false; args.splice(i, 1); return true; };
const KNIT_ONLY = bool('--knit-only'), FACE_ONLY = bool('--face-only'), DEADLINE = Number(flag('--deadline', '150'));
const [dir, stem, outDir] = args;
const rd = (f, T) => { const b = fs.readFileSync(path.join(dir, f)); return new T(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); };
const positions = Float64Array.from(rd(`${stem}.pos`, Float32Array)), labels = rd(`${stem}.fid`, Int32Array);
const surfaces = new Map(JSON.parse(fs.readFileSync(path.join(dir, `${stem}.fits.json`), 'utf8')).faces.map(({ id, type, params }) => [id, [type, params]]));
const oc = await loadOcct();
const t0 = performance.now();
// (the face path loaded only when used: --knit-only runs in a tree without it, as main before it)
const { hasFreeform, tryFacePath } = KNIT_ONLY ? { hasFreeform: () => false } : await import('../js/knit/facePath.js');
const row = { stem, triangles: labels.length, groups: new Set(labels).size, freeform: hasFreeform(labels, surfaces) };
let step = null, report = null;
if (!KNIT_ONLY && row.freeform) {
  const fp = tryFacePath(oc, { positions, labels, surfaces }, { part: stem, budgetS: DEADLINE / 3 });
  if (fp.step) { step = fp.step; report = fp.report; } else row.face_rejected = fp.rejected;
  const fi = fp.info ?? fp.report?.info;
  row.face_info = fi ? { faces: fi.step_faces, analytic: fi.analytic_faces, fitted: fi.fitted_faces,
    splits: fi.splits?.length ?? 0, edge_tol: fi.edge_tol, face_ms: fi.face_ms, check_ms: fi.check_ms, kernel_rounds: fi.kernel_rounds } : null;
}
if (!step && !FACE_ONLY) {
  const { solid, report: r } = knitSolid(oc, { positions, labels, surfaces }, { deadline: DEADLINE });
  report = r;
  if (solid) step = writeStep(oc, solid);
}
row.path = report?.info?.path ?? 'knit';
Object.assign(row, { tier: report?.tier, facetArea: report?.facetArea, vol_err: report?.vol_err, closed: report?.closed, valid: report?.valid,
  secs: Math.round((performance.now() - t0) / 100) / 10 });
if (step) { fs.mkdirSync(path.join(outDir, 'step'), { recursive: true }); fs.writeFileSync(path.join(outDir, 'step', `${stem}.step`), step); }
console.log(JSON.stringify(row));
