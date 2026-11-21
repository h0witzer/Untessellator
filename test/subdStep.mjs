// Cage -> NURBS patches (js/subd/fastNurbs.js) -> closed solid (js/knit/patches.js) -> STEP, timed per stage.
// node test/subdStep.mjs <dir-with-cage.json> ... [--out <dir>]
import fs from 'node:fs';
import path from 'node:path';
import { fastSubdToNurbs } from '../js/subd/fastNurbs.js';
import { patchSolid } from '../js/knit/patches.js';
import { loadOcct, writeStep } from '../js/knit/index.js';

const args = process.argv.slice(2);
const oi = args.indexOf('--out');
const outDir = oi >= 0 ? args.splice(oi, 2)[1] : null;
const t0 = performance.now();
const oc = await loadOcct();
console.log(`OCCT loaded in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
for (const dir of args) {
  const cage = JSON.parse(fs.readFileSync(path.join(dir, 'cage.json'), 'utf8'));
  const ta = performance.now();
  const r = fastSubdToNurbs(cage.vertices, cage.faces);
  const tb = performance.now();
  const { solid, valid, ms } = patchSolid(oc, r);
  const tc = performance.now();
  const props = new oc.GProp_GProps_1();
  oc.BRepGProp.VolumeProperties_1(solid, props, false, false, false);
  const volume = props.Mass();
  const td = performance.now();
  const step = writeStep(oc, solid);
  const te = performance.now();
  if (outDir) fs.writeFileSync(path.join(outDir, path.basename(dir) + '.step'), step);
  console.log(path.basename(dir).padEnd(12), `${r.faces.length} faces |`, `nurbs ${(tb - ta).toFixed(1)} ms,`,
    `solid ${ms.build.toFixed(0)} ms, check ${ms.check.toFixed(0)} ms (valid ${valid}), volume ${volume.toFixed(4)},`,
    `STEP ${(te - td).toFixed(0)} ms (${(step.length / 1024).toFixed(0)} KB) | total ${(te - ta).toFixed(0)} ms`);
}
