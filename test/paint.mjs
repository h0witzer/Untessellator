/**
 * paint.mjs — verify primitive-bounded bucket fill.
 *
 * Usage:  node test/paint.mjs
 *
 * Two questions:
 *   1. On tangent-fillet — the fixture crease-angle fill has never been able to
 *      separate at any threshold — does painting a type stop at the junction?
 *   2. On the real WatchStand, can a single click recover one of the 12 conical
 *      faces that automatic segmentation finds 0 of?
 *
 * Purity = fraction of the painted region that really belongs to the target
 * face. Recall = fraction of that face's available facets the paint captured.
 */

import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import { load3MF } from './load3mf.mjs';
import { parseStepFaces, STEP_TO_PRIMITIVE } from './parseStep.mjs';
import { buildFacets, buildAdjacency, segment } from '../js/meshCore.js';
import { paintFrom, surfaceEval } from '../js/regionGrow.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const FIX = join(HERE, 'fixtures');
const DEG = 180 / Math.PI;
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

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

const report = (label, painted, truthSet, extra) => {
  let hit = 0;
  for (const t of painted) if (truthSet.has(t)) hit++;
  const purity = 100 * hit / painted.size;
  const recall = 100 * hit / truthSet.size;
  console.log(`  ${label.padEnd(34)} painted ${String(painted.size).padStart(4)}` +
    `   purity ${purity.toFixed(1).padStart(5)}%   recall ${recall.toFixed(1).padStart(5)}%` +
    (extra ? '   ' + extra : ''));
};

// ── 1. tangent-fillet ────────────────────────────────────────────────────────

console.log('\ntangent-fillet  (crease fill merges cylinder + torus fillet + top plane)');
{
  const positions = readSTL(join(FIX, 'tangent-fillet.stl'));
  const { segOf } = JSON.parse(readFileSync(join(FIX, 'tangent-fillet.truth.json'), 'utf8'));
  const facets = buildFacets(positions);
  const adjacency = buildAdjacency(positions, facets);

  const creaseRegions = segment(adjacency, facets.count, 10);
  const biggest = creaseRegions.sort((a, b) => b.size - a.size)[0];
  console.log(`  crease fill @10 deg               largest region ${biggest.size} of ${facets.count} triangles` +
    `  (spans segments ${[...new Set([...biggest].map(t => segOf[t]))].join(',')})`);

  for (const [segIdx, type, name] of [[1, 'cylinder', 'paint CYLINDER on the wall'],
                                      [0, 'plane', 'paint PLANE on the base'],
                                      [3, 'plane', 'paint PLANE on the top face']]) {
    const truthSet = new Set();
    for (let t = 0; t < facets.count; t++) if (segOf[t] === segIdx) truthSet.add(t);
    if (!truthSet.size) continue;
    const seed = [...truthSet][Math.floor(truthSet.size / 2)];
    const r = paintFrom(seed, type, facets, adjacency);
    const d = r.params?.radius !== undefined ? `r=${r.params.radius.toFixed(3)}` : '';
    report(name, r.region, truthSet, d);
  }
}

// ── 2. WatchStand conical faces ──────────────────────────────────────────────

console.log('\nWatchStandMedium  (automatic segmentation recovers 0 of 12 cones)');
{
  const { positions } = load3MF(join(FIX, 'WatchStandMedium.3mf'));
  const facets = buildFacets(positions);
  const adjacency = buildAdjacency(positions, facets, 1e7);
  const { faces } = parseStepFaces(join(FIX, 'WatchStandTruth.step'));

  const medianSize = (() => {
    const s = [];
    for (let t = 0; t < facets.count; t++) s.push(Math.sqrt(facets.areas[t]));
    s.sort((a, b) => a - b); return s[s.length >> 1];
  })();
  const tolD = 1.5 * medianSize;
  const cosTol = Math.cos(25 * Math.PI / 180);

  const cones = faces.filter(f => f.type === 'CONICAL_SURFACE' && f.axis && f.location);
  const seen = new Set();
  let recovered = 0, attempted = 0;

  for (const f of cones) {
    const key = (f.halfAngle * DEG).toFixed(2) + '|' + f.radius.toFixed(5);
    if (seen.has(key)) continue;          // one representative per distinct cone
    seen.add(key);

    const d = f.radius / Math.tan(f.halfAngle);
    const apex = [f.location[0] - f.axis[0] * d,
                  f.location[1] - f.axis[1] * d,
                  f.location[2] - f.axis[2] * d];
    const ev = surfaceEval('cone', { axis: f.axis, apex, halfAngle: f.halfAngle });

    const truthSet = new Set();
    for (let t = 0; t < facets.count; t++) {
      const c = [facets.centroids[t * 3], facets.centroids[t * 3 + 1], facets.centroids[t * 3 + 2]];
      if (Math.abs(ev.distance(c)) > tolD) continue;
      const n = [facets.normals[t * 3], facets.normals[t * 3 + 1], facets.normals[t * 3 + 2]];
      if (Math.abs(dot(n, ev.normal(c))) < cosTol) continue;
      truthSet.add(t);
    }
    if (truthSet.size < 4) continue;
    attempted++;

    const seed = [...truthSet][Math.floor(truthSet.size / 2)];
    const r = paintFrom(seed, 'cone', facets, adjacency, {
      seedSize: Number(process.env.SEEDN||12), tolScale: Number(process.env.TOLD||0.5),
      angleTol: Number(process.env.ANGT||15), iterations: 6 });
    const got = r.params ? r.params.halfAngle * DEG : NaN;
    const err = Math.abs(got - f.halfAngle * DEG);
    if (err < 2) recovered++;
    report(`click a ${(f.halfAngle * DEG).toFixed(2)}deg cone`, r.region, truthSet,
      `halfAngle ${got.toFixed(2)}deg (err ${err.toFixed(2)}deg)`);
  }
  console.log(`\n  distinct cones recovered by one click each: ${recovered}/${attempted}`);
}
