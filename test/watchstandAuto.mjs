/**
 * watchstandAuto.mjs — the studio's auto-paint on the WatchStand meshes, scored against the real STEP.
 *
 * Usage:  node test/watchstandAuto.mjs
 *
 * test/watchstand.mjs scores segmentations of the three WatchStand 3MFs by surface recovery: a truth
 * face counts as recovered when a region of the same primitive type fits its defining parameters
 * (greedy, largest region first). This runs the studio's auto-paint (js/autoPaint.js around
 * js/paintCore.js, as test/autoSeed.mjs does) in two configurations and scores each the same way:
 *
 *   before   v6feat, boundary model v1 wall, first pass only
 *   after    v7gnn, boundary model v2 wall, refined fills (paintCore.refineFill: refiner + minimum cut)
 *
 * It is one hand-made real part, not a benchmark: read it as a check that the ABC gains survive
 * outside ABC, next to the group count a user would see.
 */

import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';
import { load3MF } from './load3mf.mjs';
import { parseStepFaces, STEP_TO_PRIMITIVE } from './parseStep.mjs';
import { preparePaintPositions, paintWith } from '../js/paintCore.js';
import { coherentScores, seedOrder, autoPaint } from '../js/autoPaint.js';
import { annotateEdges, edgeWall } from '../js/edgeWall.js';
import { fitRegionAs } from '../js/gaussMap.js';

const ort = createRequire(import.meta.url)('onnxruntime-node');
const HERE = dirname(fileURLToPath(import.meta.url));
const FIX = join(HERE, 'fixtures');
const MODELS = join(HERE, '..', 'models');
const DEG = 180 / Math.PI;
const RAD_TOL = 0.03, ANG_TOL = 5, CONE_TOL = 2;   // test/watchstand.mjs's tolerances
const CURVED = ['cylinder', 'cone', 'sphere', 'torus'];

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const dirAngle = (a, b) => Math.acos(Math.min(1, Math.abs(dot(a, b)))) * DEG;
function matches(truth, fit) {
  const want = STEP_TO_PRIMITIVE[truth.type];
  if (want !== fit.type || !fit.params) return false;
  const p = fit.params;
  if (want === 'plane') return truth.axis ? dirAngle(truth.axis, p.normal) < ANG_TOL : true;
  if (want === 'cylinder') {
    if (Math.abs(p.radius - truth.radius) / truth.radius > RAD_TOL) return false;
    return truth.axis ? dirAngle(truth.axis, p.axis) < ANG_TOL : true;
  }
  if (want === 'cone') {
    if (Math.abs(p.halfAngle * DEG - truth.halfAngle * DEG) > CONE_TOL) return false;
    return truth.axis ? dirAngle(truth.axis, p.axis) < ANG_TOL : true;
  }
  if (want === 'sphere') return Math.abs(p.radius - truth.radius) / truth.radius <= RAD_TOL;
  return false;
}

const graphFeeds = (session, graph, n) => (session.inputNames.includes('nbr') && graph
  ? { nbr: new ort.Tensor('int64', graph.nbr, [1, n, 3]), eattr: new ort.Tensor('float32', graph.eattr, [1, n, 3]) }
  : {});
async function modelRunner(file) {
  const s = await ort.InferenceSession.create(join(MODELS, file));
  return async (features, n, graph) => {
    const out = await s.run({
      features: new ort.Tensor('float32', features, [1, n, features.length / n]),
      mask: new ort.Tensor('float32', new Float32Array(n).fill(1), [1, n]),
      ...graphFeeds(s, graph, n),
    });
    return { sameLogit: out.same_logit.data, typeLogit: Array.from(out.type_logit.data), surface: out.surface ? out.surface.data : null };
  };
}
async function edgeModel(file) {
  const meta = JSON.parse(readFileSync(join(MODELS, file.replace(/\.onnx$/, '.json')), 'utf8'));
  const s = await ort.InferenceSession.create(join(MODELS, file));
  return { meta, run: async (f, n) => (await s.run({ features: new ort.Tensor('float32', f, [n, f.length / n]) })).logit.data };
}

const ALL_CONFIGS = {
  before: { seed: 'seedfill-v6feat.onnx', edge: 'edge-boundary.onnx', refine: null },
  after: { seed: 'seedfill-v7gnn.onnx', edge: 'edge-boundary-v2.onnx', refine: 'fill-refiner.onnx' },
  // one change at a time from `before`, to see which part moves the group count
  'v2 only': { seed: 'seedfill-v6feat.onnx', edge: 'edge-boundary-v2.onnx', refine: null },
  'v7 only': { seed: 'seedfill-v7gnn.onnx', edge: 'edge-boundary.onnx', refine: null },
  'v2+refine': { seed: 'seedfill-v6feat.onnx', edge: 'edge-boundary-v2.onnx', refine: 'fill-refiner.onnx' },
  'v2+v7': { seed: 'seedfill-v7gnn.onnx', edge: 'edge-boundary-v2.onnx', refine: null },
  'v1+v7+refine': { seed: 'seedfill-v7gnn.onnx', edge: 'edge-boundary.onnx', refine: 'fill-refiner.onnx' },
};
const pick = process.argv.indexOf('--configs');
const CONFIGS = Object.fromEntries((pick >= 0 ? process.argv[pick + 1].split(',') : ['before', 'after']).map((k) => [k, ALL_CONFIGS[k]]));
const PARTS = process.argv.includes('--medium') ? ['WatchStandMedium'] : ['WatchStandCoarse', 'WatchStandMedium', 'WatchStandFine'];
const { faces: truthFaces } = parseStepFaces(join(FIX, 'WatchStandTruth.step'));

for (const [name, cfg] of Object.entries(CONFIGS)) {
  const runner = await modelRunner(cfg.seed);
  const edge = await edgeModel(cfg.edge);
  const refine = cfg.refine ? { runner: await modelRunner(cfg.refine) } : null;
  for (const part of PARTS) {
    const t0 = performance.now();
    const { positions } = load3MF(join(FIX, `${part}.3mf`));
    const mesh = preparePaintPositions(positions);
    const { facets, adjacency } = mesh;
    await annotateEdges(facets, adjacency, edge.run, edge.meta.columns);
    const wall = edgeWall(1 / (1 + Math.exp(-edge.meta.thresholds['fpr0.02'].logit)), CURVED);
    const order = seedOrder(await coherentScores(facets, adjacency, { chunk: 1e9 }), facets);
    const seg = new Int32Array(facets.count).fill(-1);
    const segType = [];
    const result = await autoPaint({
      facets, order, owner: new Int32Array(facets.count).fill(-1), minClaim: 1, minKeep: 1,
      adjacency, absorbUpTo: 8,
      onAbsorb: (island, id) => { for (const t of island) seg[t] = id; },
      fill: (seed, taken) => paintWith(runner, seed, mesh, {
        method: 'model', type: 'auto', taken, extended: true, reportFit: false, grow: { wall },
        ...(refine ? { refine } : {}),
      }),
      onSegment: (claim, out) => {
        const id = segType.length;
        for (const t of claim) seg[t] = id;
        segType.push(out.fitter ?? null);
        return id;
      },
    });
    // Fit every group as its type, then match truth faces greedily, largest group first.
    const groups = segType.map((type, id) => ({ type, region: new Set() }));
    for (let t = 0; t < facets.count; t++) if (seg[t] >= 0) groups[seg[t]].region.add(t);
    const fits = groups.filter((g) => g.type && g.region.size).map((g) => ({ ...g, ...fitRegionAs(g.type, g.region, facets), type: g.type }));
    const pool = fits.filter((f) => f.params).sort((a, b) => b.region.size - a.region.size);
    const used = new Set();
    const recovered = {}, total = {};
    for (const truth of truthFaces) {
      const key = STEP_TO_PRIMITIVE[truth.type] ?? truth.type;
      total[key] = (total[key] || 0) + 1;
      for (let i = 0; i < pool.length; i++) {
        if (used.has(i) || !matches(truth, pool[i])) continue;
        used.add(i);
        recovered[key] = (recovered[key] || 0) + 1;
        break;
      }
    }
    const rec = Object.values(recovered).reduce((a, b) => a + b, 0);
    const kinds = ['plane', 'cylinder', 'cone', 'sphere', 'torus'].filter((k) => total[k])
      .map((k) => `${k} ${recovered[k] || 0}/${total[k]}`).join('  ');
    console.log(`${name.padEnd(7)} ${part.padEnd(17)} ${facets.count} tris  ${result.segments} groups  recovered ${rec}/${truthFaces.length}   ${kinds}   ${((performance.now() - t0) / 1000).toFixed(1)} s`);
  }
}
