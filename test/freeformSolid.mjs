// The freeform export as the studio would run it (js/subd/freeform.js on a triangle soup), timed; --check reads the
// STEP back with OCCT. Meshes: mesh.json ({vertices, faces}) or .stl. Several meshes with --merge make one
// multibody soup (each mesh offset along x) to exercise named bodies.
// node test/freeformSolid.mjs <mesh.json | file.stl> ... [--merge] [--quads N] [--check] [--out file.step]
import fs from 'node:fs';
import { freeformSolid } from '../js/subd/freeform.js';

const args = process.argv.slice(2);
const flag = (k) => { const i = args.indexOf(k); return i >= 0 ? args.splice(i, 2)[1] : null; };
const quads = flag('--quads'), out = flag('--out');
const check = args.includes('--check'), merge = args.includes('--merge');
const files = args.filter((a) => !a.startsWith('--'));

function soupOf(file) {
  if (file.toLowerCase().endsWith('.stl')) {
    const buf = fs.readFileSync(file), n = buf.readUInt32LE(80), P = new Float64Array(n * 9);
    for (let t = 0; t < n; t++) for (let k = 0; k < 9; k++) P[t * 9 + k] = buf.readFloatLE(84 + t * 50 + 12 + k * 4);
    return P;
  }
  const m = JSON.parse(fs.readFileSync(file, 'utf8')), P = new Float64Array(m.faces.length * 9);
  m.faces.forEach((t, i) => t.forEach((v, k) => P.set(m.vertices[v], i * 9 + k * 3)));
  return P;
}
const runs = merge ? [files] : files.map((f) => [f]);
let oc = null;
for (const group of runs) {
  let soup = new Float64Array(0), shift = 0;
  for (const f of group) {
    const P = soupOf(f);
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < P.length; i += 3) { lo = Math.min(lo, P[i]); hi = Math.max(hi, P[i]); }
    for (let i = 0; i < P.length; i += 3) P[i] += shift - lo;
    shift += (hi - lo) * 1.2;
    const S = new Float64Array(soup.length + P.length); S.set(soup); S.set(P, soup.length); soup = S;
  }
  const t0 = performance.now();
  const r = await freeformSolid(soup, { quads: quads ? Number(quads) : null, part: 'Test' });
  const ms = performance.now() - t0;
  const label = group.map((f) => f.split(/[\/]/).slice(-2).join('/')).join(' + ');
  console.log(`${label}: ${(ms / 1000).toFixed(2)} s, STEP ${r.step ? (r.step.length / 1e6).toFixed(1) + ' MB' : 'none'}, preview ${r.preview ? r.preview.face.length + ' triangles' : 'none'}`);
  for (const b of r.report.bodies) console.log('  ', b.error ? `${b.name}: ${b.error}` : `${b.name}: ${b.triangles} triangles -> ${b.quads} quads (${b.stars} stars) -> ${b.faces} faces, ${b.subdivisions} subdivisions, rms ${b.rms.toPrecision(3)} max ${b.max.toPrecision(3)}, remesh ${b.ms.remesh.toFixed(0)} ms, fit ${b.ms.fit.toFixed(0)} ms`);
  if (out && r.step) fs.writeFileSync(out, r.step);
  if (check && r.step) {
    const { loadOcct } = await import('../js/knit/index.js');
    const { score, subShapes } = await import('../js/knit/build.js');
    oc ??= await loadOcct();
    oc.FS.writeFile('/tmp/in.step', r.step);
    const reader = new oc.STEPControl_Reader_1();
    if (reader.ReadFile('/tmp/in.step') !== oc.IFSelect_ReturnStatus.IFSelect_RetDone) { console.log('   check: OCCT could not read it'); continue; }
    reader.TransferRoots(new oc.Message_ProgressRange_1());
    const shape = reader.OneShape();
    const sc = score(oc, shape, { volume: () => null });
    console.log(`   check: solids ${subShapes(oc, shape, oc.TopAbs_ShapeEnum.TopAbs_SOLID).length}, faces ${subShapes(oc, shape, oc.TopAbs_ShapeEnum.TopAbs_FACE).length}, closed ${sc.closed}, valid ${sc.valid}`);
  }
}
