/**
 * oracle.mjs — how much of the truth model is recoverable from the mesh AT ALL?
 *
 * Usage:  node test/oracle.mjs
 *
 * This ignores segmentation and classification entirely. For each analytic face
 * in the STEP, it takes the EXACT ground-truth surface and asks which mesh
 * facets lie on it (centroid within tolerance, normal in agreement). That is the
 * maximum support any algorithm could ever find for that face.
 *
 * A face supported by only a handful of facets cannot be fitted by anything —
 * not by region growing, not by a human painting faces by hand. This separates
 * "our algorithm is weak" from "the information is not in the file".
 */

import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import { load3MF } from './load3mf.mjs';
import { parseStepFaces, STEP_TO_PRIMITIVE } from './parseStep.mjs';
import { buildFacets } from '../js/meshCore.js';
import { surfaceEval } from '../js/regionGrow.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const FIX = join(HERE, 'fixtures');
const DEG = 180 / Math.PI;

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/** Convert a STEP face record into this project's primitive parameters. */
function toParams(f) {
  const t = STEP_TO_PRIMITIVE[f.type];
  if (!f.axis || !f.location) return null;
  if (t === 'plane') return { type: 'plane', params: { origin: f.location, normal: f.axis } };
  if (t === 'cylinder')
    return { type: 'cylinder', params: { axis: f.axis, axisPoint: f.location, radius: f.radius } };
  if (t === 'sphere')
    return { type: 'sphere', params: { center: f.location, radius: f.radius } };
  if (t === 'cone') {
    // STEP places the reference point where the cone has `radius`; the apex sits
    // radius/tan(halfAngle) back along the axis.
    const d = f.radius / Math.tan(f.halfAngle);
    const apex = [f.location[0] - f.axis[0] * d,
                  f.location[1] - f.axis[1] * d,
                  f.location[2] - f.axis[2] * d];
    return { type: 'cone', params: { axis: f.axis, apex, halfAngle: f.halfAngle } };
  }
  return null;
}

function medianFacetSize(facets) {
  const s = [];
  for (let t = 0; t < facets.count; t++) s.push(Math.sqrt(Math.max(facets.areas[t], 0)));
  s.sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)] || 1;
}

const { faces } = parseStepFaces(join(FIX, 'WatchStandTruth.step'));

for (const name of ['WatchStandCoarse', 'WatchStandMedium']) {
  const { positions } = load3MF(join(FIX, name + '.3mf'));
  const facets = buildFacets(positions);
  const tolD = 1.5 * medianFacetSize(facets);
  const cosTol = Math.cos(25 * Math.PI / 180);

  const buckets = { '0': 0, '1-3': 0, '4-9': 0, '10-29': 0, '30+': 0 };
  const byType = {};
  let claimed = 0;

  const rows = [];
  for (const f of faces) {
    const key = STEP_TO_PRIMITIVE[f.type] ?? f.type;
    byType[key] = byType[key] || { total: 0, fittable: 0 };
    byType[key].total++;

    const prim = toParams(f);
    if (!prim) { buckets['0']++; rows.push({ key, n: 0 }); continue; }
    const ev = surfaceEval(prim.type, prim.params);

    let n = 0;
    for (let t = 0; t < facets.count; t++) {
      const c = [facets.centroids[t * 3], facets.centroids[t * 3 + 1], facets.centroids[t * 3 + 2]];
      if (Math.abs(ev.distance(c)) > tolD) continue;
      const nrm = [facets.normals[t * 3], facets.normals[t * 3 + 1], facets.normals[t * 3 + 2]];
      if (Math.abs(dot(nrm, ev.normal(c))) < cosTol) continue;
      n++;
    }
    claimed += n;
    rows.push({ key, n });
    if (n === 0) buckets['0']++;
    else if (n <= 3) buckets['1-3']++;
    else if (n <= 9) buckets['4-9']++;
    else if (n <= 29) buckets['10-29']++;
    else buckets['30+']++;
    if (n >= 4) byType[key].fittable++;
  }

  console.log(`\n${'='.repeat(72)}\n${name} — ${facets.count} triangles, ` +
    `tolerance ${tolD.toExponential(2)} m\n${'='.repeat(72)}`);
  console.log('max support per truth face (facets lying on the exact surface):');
  for (const [k, v] of Object.entries(buckets))
    console.log(`  ${k.padEnd(7)} facets : ${String(v).padStart(3)} faces` +
      (k === '0' || k === '1-3' ? '   <- unfittable by anything' : ''));

  console.log('\nfaces with >=4 supporting facets, by type:');
  for (const [k, v] of Object.entries(byType))
    console.log(`  ${k.padEnd(10)} ${v.fittable}/${v.total}`);

  const ceiling = Object.values(byType).reduce((s, v) => s + v.fittable, 0);
  console.log(`\n  CEILING: ${ceiling}/77 faces have enough mesh to fit at all`);
}
