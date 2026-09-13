/**
 * paintnet.mjs — does the learned seed proposer beat the plain BFS seed patch?
 *
 * Usage:  node test/paintnet.mjs [path/to/seedfill.onnx]
 *
 * This is the kill criterion from docs/ml-experiment-scope.md, run against the numbers
 * test/paint.mjs already records:
 *
 *   - WatchStand conical faces: 0/4 recovered, and flat across seedSize 12-80 crossed
 *     with tolScale 0.5-1.5, so it is structural rather than a tuning problem.
 *     regionGrow.js diagnoses it: "neighbourhood() is a plain adjacency BFS, so on a
 *     face that is small relative to its neighbours the seed patch crosses the boundary
 *     and poisons the initial fit before growth begins."
 *   - tangent-fillet: base plane at 100% purity and 100% recall, cylinder at 100%
 *     recall with r=19.986 against a true 20, but the cylinder and top face over-reach
 *     about 25% into the tangent blend.
 *
 * THE INTEGRATION IS DELIBERATELY MINIMAL. paintFrom already takes opts.seedRegion, so
 * the model replaces precisely the component named in that diagnosis — the initial
 * patch — and every downstream step (grow, refit, tolerance) is untouched. If the model
 * helps, it helps by handing the existing machinery a cleaner start. If it is absent,
 * nothing changes.
 *
 * Two modes are reported because they answer different questions:
 *   seeded  — model proposes the start, existing grow/refit runs. The shippable one.
 *   direct  — model's region fitted once, no growth. Shows what the model alone knows.
 */

import { readFileSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

import { load3MF } from './load3mf.mjs';
import { parseStepFaces } from './parseStep.mjs';
import { buildFacets, buildAdjacency } from '../js/meshCore.js';
import { paintFrom, surfaceEval } from '../js/regionGrow.js';
import { fitRegionAs } from '../js/gaussMap.js';
import { proposeRegion, TYPE_NAMES , N_FEATURES } from '../js/seedNet.js';

const require = createRequire(import.meta.url);
const ort = require('onnxruntime-node');

const HERE = dirname(fileURLToPath(import.meta.url));
const FIX = join(HERE, 'fixtures');
const DEG = 180 / Math.PI;
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

const MODEL = process.argv[2] || 'D:/abc-data/runs/v2ctrl/seedfill.onnx';
if (!existsSync(MODEL)) {
  console.error(`no model at ${MODEL}\nTrain one, or pass a path.`);
  process.exit(2);
}

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

const session = await ort.InferenceSession.create(MODEL);
const runner = async (features, n) => {
  const x = new ort.Tensor('float32', features, [1, n, N_FEATURES]);
  const m = new ort.Tensor('float32', new Float32Array(n).fill(1), [1, n]);
  const out = await session.run({ features: x, mask: m });
  return { sameLogit: out.same_logit.data, typeLogit: Array.from(out.type_logit.data) };
};

function score(painted, truthSet) {
  let hit = 0;
  for (const t of painted) if (truthSet.has(t)) hit++;
  return { purity: 100 * hit / painted.size, recall: 100 * hit / truthSet.size, size: painted.size };
}

const row = (label, s, extra = '') =>
  console.log(`    ${label.padEnd(22)} painted ${String(s.size).padStart(4)}` +
    `   purity ${s.purity.toFixed(1).padStart(5)}%   recall ${s.recall.toFixed(1).padStart(5)}%` +
    (extra ? '   ' + extra : ''));

let netLatency = [];

async function propose(seed, facets, adjacency) {
  const t0 = performance.now();
  const p = await proposeRegion(seed, facets, adjacency, runner);
  netLatency.push(performance.now() - t0);
  return p;
}

// ── 1. tangent-fillet ────────────────────────────────────────────────────────

console.log(`\nmodel: ${MODEL}`);
console.log('\ntangent-fillet  (baseline over-reaches ~25% into the tangent blend)\n');
{
  const positions = readSTL(join(FIX, 'tangent-fillet.stl'));
  const { segOf } = JSON.parse(readFileSync(join(FIX, 'tangent-fillet.truth.json'), 'utf8'));
  const facets = buildFacets(positions);
  const adjacency = buildAdjacency(positions, facets);

  for (const [segIdx, type, name] of [[1, 'cylinder', 'wall (cylinder)'],
                                      [0, 'plane', 'base (plane)'],
                                      [3, 'plane', 'top face (plane)']]) {
    const truthSet = new Set();
    for (let t = 0; t < facets.count; t++) if (segOf[t] === segIdx) truthSet.add(t);
    if (!truthSet.size) continue;
    const seed = [...truthSet][Math.floor(truthSet.size / 2)];

    console.log(`  ${name}   truth ${truthSet.size} facets`);
    const base = paintFrom(seed, type, facets, adjacency);
    row('baseline paintFrom', score(base.region, truthSet),
        base.params?.radius !== undefined ? `r=${base.params.radius.toFixed(3)}` : '');

    const p = await propose(seed, facets, adjacency);
    row('model direct', score(p.region, truthSet),
        `predicted ${p.type} (${(100 * p.confidence).toFixed(0)}%)`);

    const seeded = paintFrom(seed, type, facets, adjacency, { seedRegion: p.region });
    row('model-seeded paintFrom', score(seeded.region, truthSet),
        seeded.params?.radius !== undefined ? `r=${seeded.params.radius.toFixed(3)}` : '');
    console.log('');
  }
}

// ── 2. WatchStand conical faces ──────────────────────────────────────────────

console.log('WatchStandMedium  (baseline recovers 0 of 4 distinct cones)\n');
{
  const { positions } = load3MF(join(FIX, 'WatchStandMedium.3mf'));
  const facets = buildFacets(positions);
  const adjacency = buildAdjacency(positions, facets, 1e7);
  const { faces } = parseStepFaces(join(FIX, 'WatchStandTruth.step'));

  const sizes = [];
  for (let t = 0; t < facets.count; t++) sizes.push(Math.sqrt(facets.areas[t]));
  sizes.sort((a, b) => a - b);
  const tolD = 1.5 * sizes[sizes.length >> 1];
  const cosTol = Math.cos(25 * Math.PI / 180);

  const cones = faces.filter(f => f.type === 'CONICAL_SURFACE' && f.axis && f.location);
  const seen = new Set();
  let baseOk = 0, seededOk = 0, typeOk = 0, attempted = 0;

  for (const f of cones) {
    const key = (f.halfAngle * DEG).toFixed(2) + '|' + f.radius.toFixed(5);
    if (seen.has(key)) continue;
    seen.add(key);

    const d = f.radius / Math.tan(f.halfAngle);
    const apex = [f.location[0] - f.axis[0] * d, f.location[1] - f.axis[1] * d,
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
    const trueAngle = f.halfAngle * DEG;
    console.log(`  ${trueAngle.toFixed(2)}deg cone   truth ${truthSet.size} facets`);

    const opts = { seedSize: 12, tolScale: 0.5, angleTol: 15, iterations: 6 };
    const base = paintFrom(seed, 'cone', facets, adjacency, opts);
    const baseAngle = base.params ? base.params.halfAngle * DEG : NaN;
    const baseErr = Math.abs(baseAngle - trueAngle);
    if (baseErr < 2) baseOk++;
    row('baseline paintFrom', score(base.region, truthSet),
        `halfAngle ${baseAngle.toFixed(2)} (err ${baseErr.toFixed(2)})`);

    const p = await propose(seed, facets, adjacency);
    if (p.type === 'Cone') typeOk++;
    const direct = fitRegionAs('cone', p.region, facets);
    const dAngle = direct.params ? direct.params.halfAngle * DEG : NaN;
    row('model direct', score(p.region, truthSet),
        `predicted ${p.type} (${(100 * p.confidence).toFixed(0)}%), halfAngle ${dAngle.toFixed(2)}`);

    const seeded = paintFrom(seed, 'cone', facets, adjacency, { ...opts, seedRegion: p.region });
    const sAngle = seeded.params ? seeded.params.halfAngle * DEG : NaN;
    const sErr = Math.abs(sAngle - trueAngle);
    if (sErr < 2) seededOk++;
    row('model-seeded paintFrom', score(seeded.region, truthSet),
        `halfAngle ${sAngle.toFixed(2)} (err ${sErr.toFixed(2)})`);
    console.log('');
  }

  console.log(`  cones recovered (halfAngle within 2 deg of truth):`);
  console.log(`    baseline paintFrom       ${baseOk}/${attempted}`);
  console.log(`    model-seeded paintFrom   ${seededOk}/${attempted}`);
  console.log(`    model predicted Cone     ${typeOk}/${attempted}`);
}

const mean = netLatency.reduce((a, b) => a + b, 0) / netLatency.length;
console.log(`\n  model calls ${netLatency.length}, mean ${mean.toFixed(1)} ms ` +
  `(features + onnxruntime-node, includes k-ring BFS)\n`);
