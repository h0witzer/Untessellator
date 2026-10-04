/**
 * boundaryScreen.mjs — would the seed model gain from seeing the learned face boundary?
 *
 * Usage:  node test/boundaryScreen.mjs <dir> <out prefix> --model <seedfill.onnx>
 *                                       --edge-wall <run dir> --edge-wall-even <run dir>
 *         <dir> from ml/export_fill_eval.py; writes <prefix>.feat / .meta / .json for
 *         ml/boundary_screen.py.
 *
 * The model's best membership input so far is `bottleneck`, the largest dihedral on the gentlest
 * path from the seed (test/featureScreen.mjs): membership is a connectivity question and a
 * per-facet network cannot trace paths itself. This dumps the same 512-facet patches the model sees
 * with its exact v6 inputs, its own membership logit, and path features built from the learned
 * edge scores (js/edgeWall.js) instead of dihedrals:
 *
 *   ebottle   the lowest achievable MAXIMUM boundary probability over paths from the seed
 *   epath     the lowest achievable SUM of -log(1 - p) over paths from the seed (capped at 20)
 *   elocal    the highest boundary probability on the facet's own edges
 *
 * Edge scores are held out on every part: odd parts are scored by the model trained on even parts
 * (--edge-wall) and even parts by the one trained on odd parts (--edge-wall-even).
 */

import { readFileSync, openSync, writeSync, closeSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { createRequire } from 'node:module';
import { buildFacets, buildAdjacency } from '../js/meshCore.js';
import { attachCurvature } from '../js/curvature.js';
import { annotateEdges } from '../js/edgeWall.js';
import { buildFeatures, TYPE_NAMES, N_FEATURES_V6 } from '../js/seedNet.js';
import { EXT_FEATURE_NAMES } from '../js/seedFeatures.js';

const ort = createRequire(import.meta.url)('onnxruntime-node');
const args = process.argv.slice(2);
function flag(name, dflt) {
  const i = args.indexOf(name);
  if (i < 0) return dflt;
  const v = args[i + 1];
  args.splice(i, 2);
  return v;
}
const MODEL = flag('--model', null);
const EDGE_ODD = flag('--edge-wall', null);
const EDGE_EVEN = flag('--edge-wall-even', null);
const [DIR, OUT] = args;
if (!DIR || !OUT || !MODEL || !EDGE_ODD || !EDGE_EVEN) {
  console.error('usage: node test/boundaryScreen.mjs <dir> <out prefix> --model m.onnx --edge-wall A --edge-wall-even B');
  process.exit(2);
}

const V6_NAMES = ['dx', 'dy', 'dz', 'nx', 'ny', 'nz', 'dot_seed', 'log_area', 'ring', 'dist', 'k1s', 'k2s', ...EXT_FEATURE_NAMES];
const GROUPS = { v6: V6_NAMES.filter((c) => c !== 'k1s' && c !== 'k2s'), logit: ['v6logit'], edge: ['ebottle', 'epath', 'elocal'] };
const COLUMNS = [...V6_NAMES, 'v6logit', 'ebottle', 'epath', 'elocal'];
const NCOL = COLUMNS.length;
const SEEDS_PER_TYPE = 2;
const MAX_SEEDS_PER_PART = 10;

const session = await ort.InferenceSession.create(MODEL);
const runModel = async (features, n) => (await session.run({
  features: new ort.Tensor('float32', features, [1, n, features.length / n]),
  mask: new ort.Tensor('float32', new Float32Array(n).fill(1), [1, n]),
})).same_logit.data;
async function edgeRunner(dir) {
  const meta = JSON.parse(readFileSync(join(dir, 'edge.json'), 'utf8'));
  const s = await ort.InferenceSession.create(join(dir, 'edge.onnx'), { intraOpNumThreads: 2, interOpNumThreads: 1 });
  return { meta, run: async (f, n) => (await s.run({ features: new ort.Tensor('float32', f, [n, f.length / n]) })).logit.data };
}
const edgeOdd = await edgeRunner(EDGE_ODD);
const edgeEven = await edgeRunner(EDGE_EVEN);

const copy = (b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.length);
const load = (stem) => ({
  P: Float64Array.from(new Float32Array(copy(readFileSync(join(DIR, `${stem}.pos`))))),
  F: new Int32Array(copy(readFileSync(join(DIR, `${stem}.fid`)))),
  T: new Uint8Array(copy(readFileSync(join(DIR, `${stem}.typ`)))),
});
let x = 7;
const rng = () => (x = (x * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;

class Heap {
  constructor() { this.a = []; }
  get size() { return this.a.length; }
  push(k, v) {
    const a = this.a; a.push([k, v]);
    let i = a.length - 1;
    while (i > 0) { const p = (i - 1) >> 1; if (a[p][0] <= a[i][0]) break; [a[p], a[i]] = [a[i], a[p]]; i = p; }
  }
  pop() {
    const a = this.a; const top = a[0]; const last = a.pop();
    if (a.length) {
      a[0] = last; let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1; let m = i;
        if (l < a.length && a[l][0] < a[m][0]) m = l;
        if (r < a.length && a[r][0] < a[m][0]) m = r;
        if (m === i) break;
        [a[m], a[i]] = [a[i], a[m]]; i = m;
      }
    }
    return top;
  }
}
function paths(n, nbrs, step) {
  const best = new Float64Array(n).fill(Infinity);
  best[0] = 0;
  const heap = new Heap();
  heap.push(0, 0);
  while (heap.size) {
    const [k, i] = heap.pop();
    if (k > best[i]) continue;
    for (const [j, p] of nbrs[i]) {
      const v = step(k, p);
      if (v < best[j]) { best[j] = v; heap.push(v, j); }
    }
  }
  return best;
}

const manifest = JSON.parse(readFileSync(join(DIR, 'manifest.json'), 'utf8'));
const reference = manifest.variants[0];
const featFd = openSync(`${OUT}.feat`, 'w');
const metaFd = openSync(`${OUT}.meta`, 'w');
let rows = 0, seedId = 0;
const t0 = performance.now();

for (const [partIdx, entry] of manifest.parts.entries()) {
  const ref = entry.variants[reference];
  if (!ref) continue;
  const base = load(ref.stem);
  const facesByType = new Map();
  for (let t = 0; t < base.F.length; t++) {
    if (!facesByType.has(base.T[t])) facesByType.set(base.T[t], new Set());
    facesByType.get(base.T[t]).add(base.F[t]);
  }
  const seedFaces = [];
  for (const faces of facesByType.values()) {
    const arr = [...faces];
    for (let i = 0; i < SEEDS_PER_TYPE && arr.length; i++) seedFaces.push(arr.splice(Math.floor(rng() * arr.length), 1)[0]);
  }
  seedFaces.splice(MAX_SEEDS_PER_PART);
  const edge = partIdx % 2 === 0 ? edgeEven : edgeOdd;

  for (const [mesherIdx, mesher] of manifest.variants.entries()) {
    const v = entry.variants[mesher];
    if (!v) continue;
    const { P, F, T } = mesher === reference ? base : load(v.stem);
    const facets = buildFacets(P);
    const adjacency = buildAdjacency(P, facets);
    attachCurvature(P, facets);
    await annotateEdges(facets, adjacency, edge.run, edge.meta.columns);
    const noCurv = { valid: new Uint8Array(facets.count) };
    const byFace = new Map();
    for (let t = 0; t < facets.count; t++) {
      if (facets.degenerate[t]) continue;
      if (!byFace.has(F[t])) byFace.set(F[t], []);
      byFace.get(F[t]).push(t);
    }
    for (const face of seedFaces) {
      const tris = byFace.get(face);
      if (!tris || tris.length < 3) continue;
      const seed = tris[Math.floor(rng() * tris.length)];
      const { features, ids, n } = buildFeatures(seed, facets, adjacency, 512, noCurv, null, true);
      const logit = await runModel(features, n);
      const index = new Map(ids.map((t, i) => [t, i]));
      const nbrs = ids.map((t) => (adjacency.get(t) || []).filter((e) => index.has(e.neighbor))
        .map((e) => [index.get(e.neighbor), e.boundary ?? 0]));
      const bottle = paths(n, nbrs, (k, p) => Math.max(k, p));
      const cost = paths(n, nbrs, (k, p) => k - Math.log(Math.max(1e-4, 1 - p)));
      const block = new Float32Array(n * NCOL);
      const meta = new Int32Array(n * 5);
      for (let i = 0; i < n; i++) {
        const o = i * NCOL;
        for (let c = 0; c < N_FEATURES_V6; c++) block[o + c] = features[i * N_FEATURES_V6 + c];
        let c = N_FEATURES_V6;
        block[o + c++] = logit[i];
        block[o + c++] = Number.isFinite(bottle[i]) ? bottle[i] : 1;
        block[o + c++] = Number.isFinite(cost[i]) ? Math.min(20, cost[i]) : 20;
        let local = 0;
        for (const e of adjacency.get(ids[i]) || []) local = Math.max(local, e.boundary ?? 0);
        block[o + c++] = local;
        meta.set([seedId, mesherIdx, partIdx, T[seed], F[ids[i]] === F[seed] ? 1 : 0], i * 5);
      }
      writeSync(featFd, new Uint8Array(block.buffer));
      writeSync(metaFd, new Uint8Array(meta.buffer));
      rows += n;
      seedId++;
    }
  }
  if (partIdx % 10 === 9) console.log(`  ${partIdx + 1} parts, ${seedId} seeds, ${rows} rows, ${((performance.now() - t0) / 1000).toFixed(0)} s`);
}
closeSync(featFd);
closeSync(metaFd);
writeFileSync(`${OUT}.json`, JSON.stringify({ ncol: NCOL, columns: COLUMNS, groups: GROUPS, meshers: manifest.variants,
  typeNames: TYPE_NAMES, rows, seeds: seedId }, null, 1));
console.log(`wrote ${rows} rows from ${seedId} seeds to ${OUT}.*`);
