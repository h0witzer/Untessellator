// Triangle mesh -> QuadriFlow quad cage -> cage fitted so its NURBS surface lies on the mesh -> G2 NURBS -> STEP
// text (js/subd/patchStep.js, no kernel), timed per stage. Accuracy: the fit's own points (mesh vertices) and,
// held out, the triangles' centroids. --check reads the STEP back with OCCT (untimed) and scores it as the knitter
// scores a build: closed, valid, volume against the mesh's.
// node test/meshToStep.mjs <mesh.json> [quads] [--out file.step] [--check]
import fs from 'node:fs';
import { quadRemesh } from '../js/subd/retopo.js';
import { fitCage, surfaceDistances } from '../js/subd/fitCage.js';
import { patchStep } from '../js/subd/patchStep.js';

const args = process.argv.slice(2);
const oi = args.indexOf('--out');
const out = oi >= 0 ? args.splice(oi, 2)[1] : null;
const check = args.includes('--check');
const [meshPath, target = '300'] = args.filter((a) => !a.startsWith('--'));
const mesh = JSON.parse(fs.readFileSync(meshPath, 'utf8'));
const t0 = performance.now();
const cage = await quadRemesh(mesh.vertices, mesh.faces, { faces: Number(target) });
if (cage.error) { console.log(cage.error); process.exit(1); }
const t1 = performance.now();
const fit = fitCage(cage, mesh.vertices, { rounds: 4 });
const t2 = performance.now();
const step = patchStep(fit);
const t3 = performance.now();
if (out) fs.writeFileSync(out, step);
const h = fit.stats.history[fit.stats.history.length - 1];
const centroids = mesh.faces.map((t) => [0, 1, 2].map((c) => (mesh.vertices[t[0]][c] + mesh.vertices[t[1]][c] + mesh.vertices[t[2]][c]) / 3));
const held = surfaceDistances(fit.faces, centroids);
const heldRms = Math.sqrt(held.distances.reduce((s, d) => s + d * d, 0) / held.distances.length) / held.diag;
const heldMax = held.distances.reduce((a, d) => Math.max(a, d), 0) / held.diag;
const name = meshPath.split(/[\\/]/).slice(-2, -1)[0];
// a truth.json beside the mesh (points on the surface the mesh was tessellated from) gives the true error
const truthPath = meshPath.replace(/mesh\.json$/, 'truth.json');
let truth = '';
if (truthPath !== meshPath && fs.existsSync(truthPath)) {
  const td = surfaceDistances(fit.faces, JSON.parse(fs.readFileSync(truthPath, 'utf8')));
  const rms = Math.sqrt(td.distances.reduce((s, d) => s + d * d, 0) / td.distances.length) / td.diag;
  truth = `, truth rms ${(100 * rms).toFixed(4)} % max ${(100 * td.distances.reduce((a, d) => Math.max(a, d), 0) / td.diag).toFixed(3)} %`;
}
console.log(`${name}: ${mesh.faces.length} triangles -> ${cage.faces.length} quads (${(t1 - t0).toFixed(0)} ms)`,
  `-> fit ${(t2 - t1).toFixed(0)} ms (map ${fit.stats.ms.map.toFixed(0)} ms), ${fit.stats.subdivisions} subdivisions, ${fit.faces.length} NURBS faces`,
  `-> STEP ${(t3 - t2).toFixed(0)} ms (${(step.length / 1e6).toFixed(1)} MB) | total ${((t3 - t0) / 1000).toFixed(2)} s`,
  `| vertices rms ${(100 * h.rms).toFixed(4)} % max ${(100 * h.max).toFixed(3)} %, centroids rms ${(100 * heldRms).toFixed(4)} % max ${(100 * heldMax).toFixed(3)} %${truth} of size`);

if (check) {
  const { loadOcct } = await import('../js/knit/index.js');
  const { score, subShapes } = await import('../js/knit/build.js');
  const oc = await loadOcct();
  let vol = 0;
  for (const [a, b, c] of mesh.faces) {
    const [x, y, z] = [mesh.vertices[a], mesh.vertices[b], mesh.vertices[c]];
    vol += (x[0] * (y[1] * z[2] - y[2] * z[1]) - x[1] * (y[0] * z[2] - y[2] * z[0]) + x[2] * (y[0] * z[1] - y[1] * z[0])) / 6;
  }
  const r0 = performance.now();
  oc.FS.writeFile('/tmp/in.step', step);
  const reader = new oc.STEPControl_Reader_1();
  if (reader.ReadFile('/tmp/in.step') !== oc.IFSelect_ReturnStatus.IFSelect_RetDone) { console.log('  check: OCCT could not read the file'); process.exit(1); }
  reader.TransferRoots(new oc.Message_ProgressRange_1());
  const shape = reader.OneShape();
  const r1 = performance.now();
  const sc = score(oc, shape, { volume: () => Math.abs(vol) });
  console.log(`  check: OCCT read ${(r1 - r0).toFixed(0)} ms, ${subShapes(oc, shape, oc.TopAbs_ShapeEnum.TopAbs_FACE).length} faces, solids ${subShapes(oc, shape, oc.TopAbs_ShapeEnum.TopAbs_SOLID).length},`,
    `closed ${sc.closed}, valid ${sc.valid}, free edges ${sc.free_edges}, volume error ${sc.volume ? `${(100 * Math.abs(sc.volume - Math.abs(vol)) / Math.abs(vol)).toFixed(3)} %` : 'n/a'}`);
}
