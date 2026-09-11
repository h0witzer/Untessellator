/**
 * forced.mjs — verify user-forced surface types produce honest fits.
 *
 * Usage:  node test/makeFixtures.mjs && node test/forced.mjs
 *
 * For every group in every fixture, force each surface type in turn and report
 * the relative residual. Two properties matter:
 *
 *   1. Forcing the CORRECT type must reproduce the ground-truth parameters.
 *   2. Forcing a WRONG type must still return a real best-fit of that primitive
 *      (large residual, honest), or declare itself degenerate — never invent
 *      parameters and present them as a fit.
 *
 * The original classifyGroupAs() violates (2): on failure it returns fabricated
 * parameters (cylinder axis = the region's plane normal, radius = half the
 * region extent; cone = a hardcoded 30 degree half-angle) with rms = Infinity.
 * Those are flagged FABRICATED below.
 */

import { readFileSync, readdirSync } from 'node:fs';
import { dirname, join, basename } from 'node:path';
import { fileURLToPath } from 'node:url';

import { buildFacets, buildAdjacency, segment } from '../js/meshCore.js';
import { fitRegionAs } from '../js/gaussMap.js';
import { classifyGroupAs } from '../js/surfaceFitter.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const FIX = join(HERE, 'fixtures');
const DEG = 180 / Math.PI;
const CREASE = 10;
const TYPES = ['plane', 'cylinder', 'cone', 'sphere'];

function readSTL(path) {
  const buf = readFileSync(path);
  const tris = buf.readUInt32LE(80);
  const positions = new Float64Array(tris * 9);
  for (let t = 0; t < tris; t++) {
    const o = 84 + t * 50;
    for (let i = 0; i < 9; i++) positions[t * 9 + i] = buf.readFloatLE(o + 12 + i * 4);
  }
  return positions;
}

const attr = arr => ({
  array: arr, count: arr.length / 3,
  getX: i => arr[i * 3], getY: i => arr[i * 3 + 1], getZ: i => arr[i * 3 + 2],
});

/** Short description of the fitted parameters, for eyeballing correctness. */
function describe(type, p) {
  if (!p) return '—';
  if (type === 'plane' && p.normal)
    return `n=[${p.normal.map(v => v.toFixed(2)).join(',')}]`;
  if (type === 'cylinder' && p.axis)
    return `r=${p.radius.toFixed(3)} ax=[${p.axis.map(v => v.toFixed(2)).join(',')}]`;
  if (type === 'cone' && p.axis)
    return `${(p.halfAngle * DEG).toFixed(1)}° ax=[${p.axis.map(v => v.toFixed(2)).join(',')}]`;
  if (type === 'sphere' && p.center)
    return `r=${p.radius.toFixed(3)}`;
  return '—';
}

/** Detect the original's fabricated-parameter failure mode. */
function isFabricated(res) {
  return res && res.params && !isFinite(res.rms);
}

let fabricatedCount = 0, honestCount = 0, correctRecovered = 0, correctTotal = 0;

for (const file of readdirSync(FIX).filter(f => f.endsWith('.stl'))) {
  const name = basename(file, '.stl');
  const positions = readSTL(join(FIX, file));
  const { truth, segOf } = JSON.parse(readFileSync(join(FIX, name + '.truth.json'), 'utf8'));

  const facets = buildFacets(positions);
  const adjacency = buildAdjacency(positions, facets);
  const groups = segment(adjacency, facets.count, CREASE);
  const geometry = { attributes: { position: attr(positions), normal: attr(positions) } };

  console.log(`\n${'='.repeat(96)}\n${name}\n${'='.repeat(96)}`);

  groups.forEach((region, gi) => {
    const votes = new Map();
    for (const t of region) votes.set(segOf[t], (votes.get(segOf[t]) || 0) + facets.areas[t]);
    const seg = [...votes.entries()].sort((a, b) => b[1] - a[1])[0][0];
    const gt = truth[seg];
    if (!gt) return;

    const gtLabel = gt.type === 'cone' ? `cone ${gt.halfAngleDeg.toFixed(0)}°`
      : (gt.radius !== undefined ? `${gt.type} r=${gt.radius.toFixed(2)}` : gt.type);
    console.log(`\n  Group ${gi + 1}  (truth: ${gtLabel})`);
    console.log('    forced      OLD classifyGroupAs                 NEW fitRegionAs');
    console.log('    ' + '-'.repeat(88));

    for (const type of TYPES) {
      const oldR = classifyGroupAs(type, { triangleIndices: region }, geometry);
      const newR = fitRegionAs(type, region, facets);

      const oldFab = isFabricated(oldR);
      if (oldFab) fabricatedCount++;

      const oldTxt = oldFab
        ? `FABRICATED  ${describe(type, oldR.params)}`
        : (oldR.type !== type ? `fell back to ${oldR.type}` : describe(type, oldR.params));

      let newTxt;
      if (newR.degenerate) {
        newTxt = `declined: ${newR.reason}`;
        honestCount++;
      } else {
        newTxt = `${describe(type, newR.params)}  rel=${newR.relResidual.toExponential(1)}`;
        honestCount++;
      }

      // Does forcing the true type recover the true parameters?
      let flag = '  ';
      if (type === gt.type) {
        correctTotal++;
        const p = newR.params;
        let ok = false;
        if (type === 'cone') ok = p && Math.abs(p.halfAngle * DEG - gt.halfAngleDeg) < 0.5;
        else if (type === 'cylinder' || type === 'sphere')
          ok = p && Math.abs(p.radius - gt.radius) / gt.radius < 0.01;
        else ok = !!p;
        if (ok) { correctRecovered++; flag = '<-'; } else flag = 'XX';
      }

      console.log(`  ${flag}  ${type.padEnd(10)}  ${oldTxt.padEnd(34)}  ${newTxt}`);
    }
  });
}

console.log(`\n${'#'.repeat(96)}`);
console.log(`forced fits where OLD returned fabricated parameters: ${fabricatedCount}`);
console.log(`forced fits where NEW returned a real fit or declined: ${honestCount}`);
console.log(`forcing the TRUE type recovers ground truth: ${correctRecovered}/${correctTotal}`);
console.log('#'.repeat(96));
