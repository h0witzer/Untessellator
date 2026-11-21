// One part of the face suite (test/faceSuite.mjs): an eval part (.pos/.fid/.typ) with its TRUE faces as the
// painted groups, built as one trimmed B-spline face per face (js/subd/faceSolid.js faceSolidSplit: a face no single
// surface carries is split, and counted), the STEP read
// back with OCCT, every face graded by its surface's worst distance to the face's mesh vertices (exact samples of
// the true surface), and the input plus OCCT's tessellation dumped for the two-way distance (scratch twosided.py).
// node test/faceSuitePart.mjs <mesh dir> <stem> <out dir>   -> one JSON line on stdout
import fs from 'node:fs';
import path from 'node:path';
import { faceSolidSplit } from '../js/subd/faceSolid.js';
import { faceStep } from '../js/subd/faceStep.js';
import { findBodies } from '../js/bodies.js';

const [meshDir, stem, outDir] = process.argv.slice(2);
const TYPES = ['Plane', 'Cylinder', 'Cone', 'Sphere', 'Torus', 'Revolution', 'Extrusion', 'BSpline', 'Other'];
const rd = (f, T) => { const b = fs.readFileSync(path.join(meshDir, f)); return new T(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); };
const P = rd(`${stem}.pos`, Float32Array), F = rd(`${stem}.fid`, Int32Array), TY = rd(`${stem}.typ`, Uint8Array);
const nT = F.length, out = { stem, triangles: nT, faces: new Set(F).size };

// weld on exact coordinates
const key = new Map(), V = [], idx = new Int32Array(nT * 3);
for (let i = 0; i < nT * 3; i++) {
  const k = `${P[i * 3]},${P[i * 3 + 1]},${P[i * 3 + 2]}`;
  if (!key.has(k)) { key.set(k, V.length); V.push([P[i * 3], P[i * 3 + 1], P[i * 3 + 2]]); }
  idx[i] = key.get(k);
}
const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
for (const p of V) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], p[k]); hi[k] = Math.max(hi[k], p[k]); }
const diag = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
const Vflat = Float64Array.from(V.flat());
const found = findBodies(idx, Vflat);
out.bodies = found.bodies.length;

const t0 = performance.now();
const solids = [], faceRows = [], splitFaces = [];
let error = null;
for (let b = 0; b < found.bodies.length && !error; b++) {
  const body = found.bodies[b];
  if (!body.closed) { error = 'open body'; break; }
  if (body.shells.length > 1) { error = 'body with a void (more than one shell)'; break; }
  const tris = [];
  for (let t = 0; t < nT; t++) if (found.bodyOf[t] === b) tris.push(t);
  const local = new Map(), Vb = [], Tb = [], Lb = [];
  const at = (v) => { if (!local.has(v)) { local.set(v, Vb.length); Vb.push(V[v]); } return local.get(v); };
  for (const t of tris) {
    const [a, c, d] = [idx[t * 3], idx[t * 3 + 1], idx[t * 3 + 2]];
    Tb.push(found.flip[t] ? [at(a), at(d), at(c)] : [at(a), at(c), at(d)]);
    Lb.push(F[t]);
  }
  let r;
  try { r = faceSolidSplit(Vb, Tb, Lb, { scale: diag }); } catch (e) { error = `threw: ${String(e?.message ?? e).slice(0, 160)}`; break; }
  for (const sp of r.splits ?? []) splitFaces.push(sp);
  if (r.error) { error = r.error.slice(0, 300); break; }
  solids.push({ faces: r.faces, edges: r.edges, points: r.points, name: `Body ${b + 1}`, tolerance: r.tolerance, report: r.report });
  // a face's stored type: the most common over its triangles
  const typeOf = new Map();
  tris.forEach((t) => { const m = typeOf.get(F[t]) ?? new Map(); m.set(TY[t], (m.get(TY[t]) ?? 0) + 1); typeOf.set(F[t], m); });
  for (const f of r.faces) {
    const m = typeOf.get(f.origin ?? f.label), type = TYPES[[...m].sort((x, y) => y[1] - x[1])[0][0]];
    faceRows.push({ type, err: f.info.max_err, param: f.info.param, spans: f.info.spans });
  }
}
out.ms = performance.now() - t0;
if (splitFaces.length) out.splits = splitFaces;
if (error) { out.error = error; console.log(JSON.stringify(out)); process.exit(0); }
out.stepFaces = solids.reduce((s, x) => s + x.faces.length, 0);
out.edgeTol = Math.max(...solids.map((s) => s.report.edgeTol));
out.faceErr = faceRows;
const tol = Math.max(...solids.map((s) => s.tolerance));
const step = faceStep(solids, { name: stem, tol });
fs.mkdirSync(path.join(outDir, 'step'), { recursive: true });
fs.writeFileSync(path.join(outDir, 'step', `${stem}.step`), step);

// OCCT: closed, valid, faces
try {
  const { loadOcct } = await import('../js/knit/index.js');
  const { score, subShapes } = await import('../js/knit/build.js');
  const { previewMesh } = await import('../js/knit/preview.js');
  const oc = await loadOcct();
  oc.FS.writeFile('/tmp/in.step', step);
  const reader = new oc.STEPControl_Reader_1();
  if (reader.ReadFile('/tmp/in.step') !== oc.IFSelect_ReturnStatus.IFSelect_RetDone) out.occt = 'unreadable';
  else {
    reader.TransferRoots(new oc.Message_ProgressRange_1());
    const shape = reader.OneShape(), sc = score(oc, shape, { volume: () => null });
    out.occt = { solids: subShapes(oc, shape, oc.TopAbs_ShapeEnum.TopAbs_SOLID).length, faces: subShapes(oc, shape, oc.TopAbs_ShapeEnum.TopAbs_FACE).length, closed: sc.closed, valid: sc.valid };
    const pm = previewMesh(oc, shape, 2e-4 * diag);
    fs.mkdirSync(path.join(outDir, 'dump'), { recursive: true });
    fs.writeFileSync(path.join(outDir, 'dump', `${stem}.in.bin`), Buffer.from(P.buffer, P.byteOffset, P.byteLength));
    fs.writeFileSync(path.join(outDir, 'dump', `${stem}.out.bin`), Buffer.from(Float32Array.from(pm.positions).buffer));
  }
} catch (e) { out.occt = `threw: ${String(e?.message ?? e).slice(0, 160)}`; }
console.log(JSON.stringify(out));
