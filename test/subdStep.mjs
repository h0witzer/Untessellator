// Cage -> NURBS patches (js/subd/fastNurbs.js) -> STEP text (js/subd/patchStep.js), timed; then OCCT reads the
// file back (untimed) as the judge: one closed valid solid, and its volume.
// node test/subdStep.mjs <dir-with-cage.json | cage.obj> ... [--out <dir>]
import fs from 'node:fs';
import path from 'node:path';
import { fastSubdToNurbs } from '../js/subd/fastNurbs.js';
import { patchStep } from '../js/subd/patchStep.js';
import { loadOcct } from '../js/knit/index.js';
import { score } from '../js/knit/build.js';
import { parseQuadCage } from '../js/subd/objCage.js';

const args = process.argv.slice(2);
const oi = args.indexOf('--out');
const outDir = oi >= 0 ? args.splice(oi, 2)[1] : null;
const oc = await loadOcct();
for (const dir of args) {
  const cage = dir.toLowerCase().endsWith('.obj') ? parseQuadCage(fs.readFileSync(dir, 'utf8'))
    : JSON.parse(fs.readFileSync(path.join(dir, 'cage.json'), 'utf8'));
  if (cage.error) { console.log(path.basename(dir), 'not a quad cage:', cage.error); continue; }
  const ta = performance.now();
  const r = fastSubdToNurbs(cage.vertices, cage.faces);
  const tb = performance.now();
  const step = patchStep(r);
  const tc = performance.now();
  if (outDir) fs.writeFileSync(path.join(outDir, path.basename(dir) + '.step'), step);
  oc.FS.writeFile('/tmp/in.step', step);
  const reader = new oc.STEPControl_Reader_1();
  const read = reader.ReadFile('/tmp/in.step') === oc.IFSelect_ReturnStatus.IFSelect_RetDone;
  let judged = 'OCCT could not read it';
  if (read) {
    reader.TransferRoots(new oc.Message_ProgressRange_1());
    const sc = score(oc, reader.OneShape(), { volume: () => null });
    judged = `read back: closed ${sc.closed}, valid ${sc.valid}, volume ${sc.volume?.toFixed(4)}`;
  }
  console.log(path.basename(dir).padEnd(12), `${r.faces.length} faces |`, `nurbs ${(tb - ta).toFixed(1)} ms,`,
    `STEP ${(tc - tb).toFixed(0)} ms (${(step.length / 1024).toFixed(0)} KB) | total ${(tc - ta).toFixed(0)} ms | ${judged}`);
}
