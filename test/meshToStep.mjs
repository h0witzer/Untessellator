// Triangle mesh -> QuadriFlow quad cage -> cage fitted so its NURBS surface lies on the mesh -> G2 NURBS ->
// STEP solid, timed per stage. node test/meshToStep.mjs <mesh.json> [quadTarget] [--out file.step]
import fs from 'node:fs';
import { quadRemesh } from '../js/subd/retopo.js';
import { fitCage } from '../js/subd/fitCage.js';
import { patchSolid } from '../js/knit/patches.js';
import { loadOcct, writeStep } from '../js/knit/index.js';

const args = process.argv.slice(2);
const oi = args.indexOf('--out');
const out = oi >= 0 ? args.splice(oi, 2)[1] : null;
const [meshPath, target = '300'] = args;
const mesh = JSON.parse(fs.readFileSync(meshPath, 'utf8'));
const oc = await loadOcct();
const t0 = performance.now();
const cage = await quadRemesh(mesh.vertices, mesh.faces, { faces: Number(target) });
if (cage.error) { console.log(cage.error); process.exit(1); }
const t1 = performance.now();
const fit = fitCage(cage, mesh.vertices, { rounds: 4 });
const t2 = performance.now();
const { solid, valid, ms } = patchSolid(oc, fit, { check: true });
const t3 = performance.now();
const step = writeStep(oc, solid);
const t4 = performance.now();
if (out) fs.writeFileSync(out, step);
const h = fit.stats.history;
console.log(`${meshPath.split(/[\\/]/).slice(-2, -1)[0]}: ${mesh.faces.length} triangles -> ${cage.faces.length} quads (${(t1 - t0).toFixed(0)} ms)`,
  `-> fit ${(t2 - t1).toFixed(0)} ms (map ${fit.stats.ms.map.toFixed(0)} ms), distance to mesh rms ${(100 * h[h.length - 1].rms).toFixed(4)} %`,
  `max ${(100 * h[h.length - 1].max).toFixed(3)} % of size, ${fit.stats.subdivisions} subdivisions, ${fit.faces.length} NURBS faces`,
  `-> solid ${ms.build.toFixed(0)} ms (valid ${valid}, check ${ms.check.toFixed(0)} ms) -> STEP ${(t4 - t3).toFixed(0)} ms | total ${((t4 - t0) / 1000).toFixed(1)} s`);
