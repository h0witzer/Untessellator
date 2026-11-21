// SubD cage -> NURBS faces (js/subd/toNurbs.js) on saved cages; writes <dir>/untess.json for the bake-off checker.
// node test/subdNurbs.mjs <dir-with-cage.json> ...
import fs from 'node:fs';
import path from 'node:path';
import { subdToNurbs } from '../js/subd/toNurbs.js';

for (const dir of process.argv.slice(2)) {
  const cage = JSON.parse(fs.readFileSync(path.join(dir, 'cage.json'), 'utf8'));
  const t = performance.now();
  const { faces, stats } = subdToNurbs(cage.vertices, cage.faces);
  const seconds = (performance.now() - t) / 1000;
  fs.writeFileSync(path.join(dir, 'untess.json'), JSON.stringify({ seconds, faces, stats }));
  const cvs = faces.reduce((s, f) => s + f.uCount * f.vCount, 0);
  console.log(path.basename(dir), `${faces.length} faces (${stats.starFaces} star), ${cvs} CVs,`,
    `${stats.edgesSolved} edges solved, ${stats.steps} steps, residual ${stats.residual.toExponential(2)},`,
    `dropped ${stats.dropped.toExponential(1)}, ${seconds.toFixed(2)} s`,
    `(limit ${(stats.ms.limit / 1000).toFixed(2)}, fit ${(stats.ms.fit / 1000).toFixed(2)}, solve ${(stats.ms.solve / 1000).toFixed(2)})`);
  console.log('  residual by step', stats.history.map((x) => x.toExponential(1)).join(' '));
}
