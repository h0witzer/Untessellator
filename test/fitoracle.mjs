/**
 * fitoracle.mjs — can the fitters recover a surface from the EXACT truth facets?
 *
 * Usage:  node test/fitoracle.mjs
 *
 * oracle.mjs asks whether the information is present in the mesh. This asks the next
 * question, which had been assumed rather than measured: given a perfect segmentation,
 * does the fitter get the right answer?
 *
 * It exists because that assumption turned out to be false. The regionGrow.js header
 * records "Fitting is not the bottleneck: handed the right facets, the fitters recover
 * a 0.5 mm cylinder radius to six significant figures." That measurement was made on a
 * CYLINDER and quietly generalised to every primitive. It does not hold for cones.
 *
 * Handing fitRegionAs the exact ground-truth facet set for each WatchStand cone:
 *
 *    9.85 deg,  477 facets  ->  22.34 deg   (err 12.48)
 *    9.85 deg,  244 facets  ->  31.21 deg   (err 21.36)
 *   59.00 deg,   30 facets  ->  degenerate  ("no measurable taper")
 *   45.00 deg,   50 facets  ->  86.00 deg   (err 41.00)
 *
 * 0 of 4, with perfect input. So the 0/4 that paint.mjs reports for automatic
 * segmentation is not, on this fixture, a segmentation result at all — a correct region
 * still fails. Any work aimed at better segmentation is capped by this until the cone
 * fitter is fixed.
 *
 * Run this before optimising any proposer. If a primitive fails here, improving the
 * region that feeds it cannot help.
 */

import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import { load3MF } from './load3mf.mjs';
import { parseStepFaces } from './parseStep.mjs';
import { buildFacets } from '../js/meshCore.js';
import { surfaceEval } from '../js/regionGrow.js';
import { fitRegionAs } from '../js/gaussMap.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const FIX = join(HERE, 'fixtures');
const DEG = 180 / Math.PI;
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

const { positions } = load3MF(join(FIX, 'WatchStandMedium.3mf'));
const facets = buildFacets(positions);
const { faces } = parseStepFaces(join(FIX, 'WatchStandTruth.step'));

const sizes = [];
for (let t = 0; t < facets.count; t++) sizes.push(Math.sqrt(facets.areas[t]));
sizes.sort((a, b) => a - b);
const tolD = 1.5 * sizes[sizes.length >> 1];
const cosTol = Math.cos(25 * Math.PI / 180);

/** Facets that genuinely lie on an analytic surface, by the same test oracle.mjs uses. */
function truthFacets(ev) {
  const set = new Set();
  for (let t = 0; t < facets.count; t++) {
    const c = [facets.centroids[t * 3], facets.centroids[t * 3 + 1], facets.centroids[t * 3 + 2]];
    if (Math.abs(ev.distance(c)) > tolD) continue;
    const n = [facets.normals[t * 3], facets.normals[t * 3 + 1], facets.normals[t * 3 + 2]];
    if (Math.abs(dot(n, ev.normal(c))) < cosTol) continue;
    set.add(t);
  }
  return set;
}

console.log('\nfit-from-truth oracle — WatchStandMedium\n');
console.log('  each fitter is handed the EXACT facets of the surface. Any failure here');
console.log('  cannot be fixed by better segmentation.\n');

let pass = 0, total = 0;

// ── cones ────────────────────────────────────────────────────────────────────
{
  const seen = new Set();
  for (const f of faces.filter(f => f.type === 'CONICAL_SURFACE' && f.axis && f.location)) {
    const key = (f.halfAngle * DEG).toFixed(2) + '|' + f.radius.toFixed(5);
    if (seen.has(key)) continue;
    seen.add(key);
    const d = f.radius / Math.tan(f.halfAngle);
    const apex = [f.location[0] - f.axis[0] * d, f.location[1] - f.axis[1] * d,
                  f.location[2] - f.axis[2] * d];
    const truth = truthFacets(surfaceEval('cone', { axis: f.axis, apex, halfAngle: f.halfAngle }));
    if (truth.size < 4) continue;
    total++;
    const fit = fitRegionAs('cone', truth, facets);
    const got = fit.params ? fit.params.halfAngle * DEG : NaN;
    const err = Math.abs(got - f.halfAngle * DEG);
    const ok = err < 2;
    if (ok) pass++;
    console.log(`  cone   ${(f.halfAngle * DEG).toFixed(2).padStart(6)} deg` +
      `  ${String(truth.size).padStart(4)} facets  ->  ` +
      (fit.degenerate ? 'DEGENERATE'.padStart(9) : `${got.toFixed(2).padStart(9)}`) +
      `   err ${(Number.isFinite(err) ? err.toFixed(2) : '  n/a').padStart(6)}   ${ok ? 'ok' : 'FAIL'}` +
      (fit.reason ? `   ${fit.reason}` : ''));
  }
}

// ── cylinders, as the control ────────────────────────────────────────────────
{
  const seen = new Set();
  let n = 0;
  for (const f of faces.filter(f => f.type === 'CYLINDRICAL_SURFACE' && f.axis && f.location)) {
    const key = f.radius.toFixed(5);
    if (seen.has(key)) continue;
    seen.add(key);
    const truth = truthFacets(surfaceEval('cylinder',
      { axis: f.axis, axisPoint: f.location, radius: f.radius }));
    if (truth.size < 4) continue;
    if (n++ >= 6) break;
    total++;
    const fit = fitRegionAs('cylinder', truth, facets);
    const got = fit.params ? fit.params.radius : NaN;
    const err = Math.abs(got - f.radius);
    const ok = err < 0.05 * f.radius;
    if (ok) pass++;
    console.log(`  cyl    r=${f.radius.toFixed(3).padStart(8)}` +
      `  ${String(truth.size).padStart(4)} facets  ->  ` +
      (fit.degenerate ? 'DEGENERATE'.padStart(9) : `r=${got.toFixed(3).padStart(7)}`) +
      `   err ${(Number.isFinite(err) ? err.toFixed(3) : '  n/a').padStart(6)}   ${ok ? 'ok' : 'FAIL'}` +
      (fit.reason ? `   ${fit.reason}` : ''));
  }
}

console.log(`\n  recovered from perfect input: ${pass}/${total}`);
console.log('  Cones fail here. Cylinders do not — which is why the earlier');
console.log('  "fitting is not the bottleneck" measurement, made on a cylinder,');
console.log('  did not catch it.\n');
