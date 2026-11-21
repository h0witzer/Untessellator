// The face path (js/subd/faceExport.js) on one painted part, as test/knit.mjs runs the knitter on it: <stem>.pos
// (float32 soup), <stem>.fid (group per triangle) and <stem>.fits.json ({faces: [{id, type, params}]}: the studio's
// surfaces; untyped groups absent), as test/studioExport.mjs --dump writes them. Writes <out>/step/<stem>.step, reads
// it back with OCCT (solids, faces, closed, valid) and dumps the input and OCCT's tessellation (<out>/dump) for the
// two-way distance. One JSON line on stdout.
// node test/faceExport.mjs <dump dir> <stem> <out dir>
import fs from 'node:fs';
import path from 'node:path';
import { faceExport } from '../js/subd/faceExport.js';

const [dir, stem, outDir] = process.argv.slice(2);
const rd = (f, T) => { const b = fs.readFileSync(path.join(dir, f)); return new T(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); };
const P = rd(`${stem}.pos`, Float32Array), labels = rd(`${stem}.fid`, Int32Array);
const fits = JSON.parse(fs.readFileSync(path.join(dir, `${stem}.fits.json`), 'utf8'));
const surfaces = new Map(fits.faces.map(({ id, type, params }) => [id, [type, params]]));
const positions = Float64Array.from(P);
const t0 = performance.now();
const r = faceExport({ positions, labels, surfaces }, { part: stem });
const out = { stem, triangles: labels.length, groups: new Set(labels).size, ms: Math.round(performance.now() - t0) };
if (r.error) { console.log(JSON.stringify({ ...out, error: r.error.slice(0, 300) })); process.exit(0); }
Object.assign(out, { stepFaces: r.report.info.step_faces, analytic: r.report.info.analytic_faces, fitted: r.report.info.fitted_faces,
  splits: r.report.info.splits, edgeTol: r.report.info.edge_tol });
fs.mkdirSync(path.join(outDir, 'step'), { recursive: true });
fs.writeFileSync(path.join(outDir, 'step', `${stem}.step`), r.step);
try {
  const { loadOcct } = await import('../js/knit/index.js');
  const { score, subShapes } = await import('../js/knit/build.js');
  const { previewMesh } = await import('../js/knit/preview.js');
  const oc = await loadOcct();
  oc.FS.writeFile('/tmp/in.step', r.step);
  const reader = new oc.STEPControl_Reader_1();
  if (reader.ReadFile('/tmp/in.step') !== oc.IFSelect_ReturnStatus.IFSelect_RetDone) out.occt = 'unreadable';
  else {
    reader.TransferRoots(new oc.Message_ProgressRange_1());
    const shape = reader.OneShape(), sc = score(oc, shape, { volume: () => null });
    out.occt = { solids: subShapes(oc, shape, oc.TopAbs_ShapeEnum.TopAbs_SOLID).length, faces: subShapes(oc, shape, oc.TopAbs_ShapeEnum.TopAbs_FACE).length, closed: sc.closed, valid: sc.valid };
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < P.length; i++) { const k = i % 3; lo[k] = Math.min(lo[k], P[i]); hi[k] = Math.max(hi[k], P[i]); }
    const pm = previewMesh(oc, shape, 2e-4 * Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]));
    fs.mkdirSync(path.join(outDir, 'dump'), { recursive: true });
    fs.writeFileSync(path.join(outDir, 'dump', `${stem}.in.bin`), Buffer.from(P.buffer, P.byteOffset, P.byteLength));
    fs.writeFileSync(path.join(outDir, 'dump', `${stem}.out.bin`), Buffer.from(Float32Array.from(pm.positions).buffer));
  }
} catch (e) { out.occt = `threw: ${String(e?.message ?? e).slice(0, 160)}`; }
console.log(JSON.stringify(out));
