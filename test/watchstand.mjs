/**
 * watchstand.mjs — score the pipeline against a real STEP ground truth.
 *
 * Usage:  node test/watchstand.mjs [creaseAngle]
 *
 * Loads each WatchStand 3MF, segments and classifies it, then greedily matches
 * the fitted primitives against the ADVANCED_FACEs in WatchStandTruth.step.
 * A truth face counts as recovered when a fitted region of the same primitive
 * type agrees on the defining parameters within tolerance.
 */

import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import { load3MF } from './load3mf.mjs';
import { parseStepFaces, STEP_TO_PRIMITIVE } from './parseStep.mjs';
import { buildFacets, buildAdjacency, segment } from '../js/meshCore.js';
import { classifyRegion } from '../js/gaussMap.js';
import { growPrimitives } from '../js/regionGrow.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const FIX = join(HERE, 'fixtures');
const DEG = 180 / Math.PI;

const RAD_TOL = 0.03;    // 3% of radius
const ANG_TOL = 5;       // degrees, axis/normal direction
const CONE_TOL = 2;      // degrees, half-angle

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
/** Unsigned angle between two directions, in degrees (sign is arbitrary here). */
const dirAngle = (a, b) =>
  Math.acos(Math.min(1, Math.abs(dot(a, b)))) * DEG;

function matches(truth, fit) {
  const want = STEP_TO_PRIMITIVE[truth.type];
  if (want !== fit.type) return false;
  const p = fit.params;
  if (!p) return false;

  if (want === 'plane')
    return truth.axis ? dirAngle(truth.axis, p.normal) < ANG_TOL : true;

  if (want === 'cylinder') {
    if (Math.abs(p.radius - truth.radius) / truth.radius > RAD_TOL) return false;
    return truth.axis ? dirAngle(truth.axis, p.axis) < ANG_TOL : true;
  }

  if (want === 'cone') {
    const ha = Math.abs(p.halfAngle * DEG - truth.halfAngle * DEG);
    if (ha > CONE_TOL) return false;
    return truth.axis ? dirAngle(truth.axis, p.axis) < ANG_TOL : true;
  }

  if (want === 'sphere')
    return Math.abs(p.radius - truth.radius) / truth.radius <= RAD_TOL;

  return false;
}

function scoreModel(name, crease) {
  const { positions } = load3MF(join(FIX, name + '.3mf'));
  const facets = buildFacets(positions);
  const adjacency = buildAdjacency(positions, facets, 1e7);
  let regions, fits;
  if (process.env.SEG === 'grow') {
    fits = growPrimitives(facets, adjacency, {
      tolScale: Number(process.env.TOLD || 1.5),
      angleTol: Number(process.env.ANGT || 25),
      minRegion: Number(process.env.MINR || 6),
      supportSlack: Number(process.env.SLACK || 0.9),
    }).map(f => ({ ...f, region: f.region }));
    regions = fits.map(f => f.region);
  } else {
    regions = segment(adjacency, facets.count, crease);
    fits = regions.map(r => ({ region: r, ...classifyRegion(r, facets) }));
  }
  const { faces } = parseStepFaces(join(FIX, 'WatchStandTruth.step'));

  // Greedy one-to-one matching, largest regions first so a real face is not
  // consumed by a stray sliver that happens to fit.
  const pool = fits.filter(f => f.params).sort((a, b) => b.region.size - a.region.size);
  const used = new Set();
  const recovered = {};
  const total = {};

  for (const truth of faces) {
    const key = STEP_TO_PRIMITIVE[truth.type] ?? truth.type;
    total[key] = (total[key] || 0) + 1;
    for (let i = 0; i < pool.length; i++) {
      if (used.has(i)) continue;
      if (matches(truth, pool[i])) { used.add(i); recovered[key] = (recovered[key] || 0) + 1; break; }
    }
  }

  const tally = {};
  for (const f of fits) tally[f.type] = (tally[f.type] || 0) + 1;

  return { name, crease, tris: facets.count, regions: regions.length, tally, recovered, total };
}

const crease = Number(process.argv[2] || 10);
const rows = ['WatchStandCoarse', 'WatchStandMedium', 'WatchStandFine']
  .map(n => scoreModel(n, crease));

const keys = ['plane', 'cylinder', 'cone', 'sphere', 'torus', 'bspline'];
const truthTotals = rows[0].total;

console.log(`\ncrease angle ${crease}°   |   truth = 77 faces ` +
  `(${keys.filter(k => truthTotals[k]).map(k => truthTotals[k] + ' ' + k).join(', ')})\n`);
console.log('model'.padEnd(20) + 'tris'.padStart(7) + 'regions'.padStart(9) +
  '   ' + keys.map(k => k.slice(0, 4).padStart(8)).join('') + '     TOTAL');
console.log('-'.repeat(94));

for (const r of rows) {
  const cells = keys.map(k => {
    const t = r.total[k] || 0;
    if (!t) return ''.padStart(8);
    return `${r.recovered[k] || 0}/${t}`.padStart(8);
  }).join('');
  const rec = keys.reduce((s, k) => s + (r.recovered[k] || 0), 0);
  console.log(r.name.padEnd(20) + String(r.tris).padStart(7) +
    String(r.regions).padStart(9) + '   ' + cells + `     ${rec}/77`);
}

console.log('\nfitted primitive inventory per model:');
for (const r of rows) {
  console.log('  ' + r.name.padEnd(20) +
    Object.entries(r.tally).sort((a, b) => b[1] - a[1])
      .map(([k, v]) => `${k}:${v}`).join('  '));
}
