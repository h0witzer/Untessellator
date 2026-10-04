/**
 * rebuildGroups.mjs — the studio's Export STEP on a given painted segmentation, headless: each group fitted
 * with fitRegionAs for its type (as the studio refits a painted group), untyped groups left to the freeform
 * fitter, then js/knit. For reproducing a user's export from their groups (recovered from their STEP with
 * D:/abc-data/lab/s18_clean/relabel_from_step.py) and judging builder changes on it.
 *
 * Usage:  node test/rebuildGroups.mjs <prefix> <out.step> [--no-axes] [--deadline 150] [--json report.json]
 *   <prefix>.pos (float32 soup), <prefix>.fid (group per triangle), <prefix>.types.json ({group: type|null})
 */

import { readFileSync, writeFileSync } from 'node:fs';
import { preparePaintPositions } from '../js/paintCore.js';
import { fitRegionAs } from '../js/gaussMap.js';
import { loadOcct, knitSolid, writeStep } from '../js/knit/index.js';

const args = process.argv.slice(2);
const flag = (name, dflt) => { const i = args.indexOf(name); if (i < 0) return dflt; const v = args[i + 1]; args.splice(i, 2); return v; };
const bool = (name) => { const i = args.indexOf(name); if (i < 0) return false; args.splice(i, 1); return true; };
const DEADLINE = Number(flag('--deadline', '150'));
const JSON_OUT = flag('--json', null);
const NO_AXES = bool('--no-axes');
const [PREFIX, OUT] = args;
const rd = (p, T) => { const b = readFileSync(p); return new T(b.buffer.slice(b.byteOffset, b.byteOffset + b.length)); };
const positions = Float64Array.from(rd(`${PREFIX}.pos`, Float32Array));
const labels = rd(`${PREFIX}.fid`, Int32Array);
const types = JSON.parse(readFileSync(`${PREFIX}.types.json`, 'utf8'));
const { facets } = preparePaintPositions(positions);
const groups = new Map();
for (let t = 0; t < labels.length; t++) (groups.get(labels[t]) ?? groups.set(labels[t], new Set()).get(labels[t])).add(t);
const surfaces = new Map();
for (const [g, tris] of groups) {
  const type = types[g] ?? null;
  if (!type) continue;
  const fit = fitRegionAs(type, tris, facets);
  if (fit.params) surfaces.set(g, [type, fit.params]);
}
const oc = await loadOcct();
const t0 = performance.now();
const { solid, report } = knitSolid(oc, { positions, labels, surfaces }, { deadline: DEADLINE, axes: !NO_AXES });
writeFileSync(OUT, writeStep(oc, solid));
const row = { prefix: PREFIX, groups: groups.size, surfaces: surfaces.size, secs: +((performance.now() - t0) / 1000).toFixed(1), ...report };
if (JSON_OUT) writeFileSync(JSON_OUT, JSON.stringify(row, null, 1));
console.log(JSON.stringify({ ...row, info: undefined }));
