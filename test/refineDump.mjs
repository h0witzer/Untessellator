/**
 * refineDump.mjs — training and evaluation rows for the fill-refinement model (js/refineFeatures.js).
 *
 * Usage:  node test/refineDump.mjs <dir> <out prefix> --edge-wall <run dir> [--edge-wall-even <run dir>]
 *                                   [--shard k/n] [--parity all|odd|even]
 *
 * For every click (chosen as test/fillEval.mjs chooses them), the fill the studio makes today — the
 * true type declared, paintFrom with the learned wall on curved types — and one 512-facet patch
 * around the click with that fill's state as extra inputs. When the fill spills, a second row set is
 * written with a simulated NEGATIVE CLICK on the spill (refineFeatures.simulatedClick), if the click
 * lands inside the patch.
 *
 * Files: <prefix>.feat (float32, N_REFINE per facet), <prefix>.meta (int32 per facet: patch, mesher,
 * part, seed type, same face, same type, in patch region), <prefix>.nbr (int32, 3 per facet: patch
 * graph), <prefix>.eattr (float32, 3 per facet: those edges' dihedral / 90), <prefix>.eprob (float32,
 * 3 per facet: those edges' learned boundary probability), <prefix>.json. Training data must come from parts no model in the loop trained on: the edge
 * model's scores are in the inputs, so pass one that did not see these parts.
 */

import { readFileSync, openSync, writeSync, closeSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { createRequire } from 'node:module';
import { buildFacets, buildAdjacency } from '../js/meshCore.js';
import { paintFrom } from '../js/regionGrow.js';
import { attachCurvature } from '../js/curvature.js';
import { annotateEdges, edgeWall } from '../js/edgeWall.js';
import { TYPE_NAMES, fitterFor } from '../js/seedNet.js';
import { refineFeatures, simulatedClick, REFINE_COLUMNS, N_REFINE } from '../js/refineFeatures.js';

const ort = createRequire(import.meta.url)('onnxruntime-node');
const args = process.argv.slice(2);
function flag(name, dflt) {
  const i = args.indexOf(name);
  if (i < 0) return dflt;
  const v = args[i + 1];
  args.splice(i, 2);
  return v;
}
const EDGE_ODD = flag('--edge-wall', null);
const EDGE_EVEN = flag('--edge-wall-even', null);
const SHARD = flag('--shard', '0/1').split('/').map(Number);
const PARITY = flag('--parity', 'all');
const [DIR, OUT] = args;
if (!DIR || !OUT || !EDGE_ODD) { console.error('usage: node test/refineDump.mjs <dir> <out prefix> --edge-wall <run dir>'); process.exit(2); }

async function loadEdge(dir) {
  const meta = JSON.parse(readFileSync(join(dir, 'edge.json'), 'utf8'));
  const s = await ort.InferenceSession.create(join(dir, 'edge.onnx'), { intraOpNumThreads: 1, interOpNumThreads: 1 });
  const thr = 1 / (1 + Math.exp(-meta.thresholds['fpr0.02'].logit));
  return { meta, thr, run: async (f, n) => (await s.run({ features: new ort.Tensor('float32', f, [n, f.length / n]) })).logit.data };
}
const edgeOdd = await loadEdge(EDGE_ODD);
const edgeEven = EDGE_EVEN ? await loadEdge(EDGE_EVEN) : edgeOdd;
const CURVED = ['cylinder', 'cone', 'sphere', 'torus'];

const copy = (b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.length);
const load = (stem) => ({
  P: Float64Array.from(new Float32Array(copy(readFileSync(join(DIR, `${stem}.pos`))))),
  F: new Int32Array(copy(readFileSync(join(DIR, `${stem}.fid`)))),
  T: new Uint8Array(copy(readFileSync(join(DIR, `${stem}.typ`)))),
});

const manifest = JSON.parse(readFileSync(join(DIR, 'manifest.json'), 'utf8'));
const reference = manifest.variants[0];
const featFd = openSync(`${OUT}.feat`, 'w');
const metaFd = openSync(`${OUT}.meta`, 'w');
const nbrFd = openSync(`${OUT}.nbr`, 'w');
const eattrFd = openSync(`${OUT}.eattr`, 'w');
const eprobFd = openSync(`${OUT}.eprob`, 'w');
const patches = [];
let rows = 0;
// One generator per part (seeded by its index), so shards need not load each other's meshes. These
// are training and validation clicks; growEval scores the real ones.
let x = 99;
const rng = () => (x = (x * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
const t0 = performance.now();

function write(r, meta) {
  const n = r.n;
  writeSync(featFd, new Uint8Array(r.features.buffer, r.features.byteOffset, n * N_REFINE * 4));
  writeSync(metaFd, new Uint8Array(meta.buffer));
  const nb = new Int32Array(n * 3);
  for (let i = 0; i < n * 3; i++) nb[i] = Number(r.graph.nbr[i]);
  writeSync(nbrFd, new Uint8Array(nb.buffer));
  writeSync(eattrFd, new Uint8Array(r.graph.eattr.buffer, r.graph.eattr.byteOffset, n * 3 * 4));
  writeSync(eprobFd, new Uint8Array(r.graph.eprob.buffer, r.graph.eprob.byteOffset, n * 3 * 4));
  rows += n;
}

for (const [partIdx, entry] of manifest.parts.entries()) {
  if (PARITY !== 'all' && (partIdx % 2 === 0) !== (PARITY === 'even')) continue;
  if (partIdx % SHARD[1] !== SHARD[0]) continue;
  x = (partIdx * 2654435761 + 99) & 0x7fffffff;
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
    for (let i = 0; i < 2 && arr.length; i++) seedFaces.push(arr.splice(Math.floor(rng() * arr.length), 1)[0]);
  }
  seedFaces.splice(10);
  const edge = partIdx % 2 === 0 ? edgeEven : edgeOdd;

  for (const [mesherIdx, mesher] of manifest.variants.entries()) {
    const v = entry.variants[mesher];
    if (!v) continue;
    const { P, F, T } = mesher === reference ? base : load(v.stem);
    const facets = buildFacets(P);
    const byFace = new Map();
    for (let t = 0; t < facets.count; t++) {
      if (facets.degenerate[t]) continue;
      if (!byFace.has(F[t])) byFace.set(F[t], []);
      byFace.get(F[t]).push(t);
    }
    const seeds = [];
    for (const face of seedFaces) {
      const tris = byFace.get(face);
      if (!tris || tris.length < 3) continue;
      seeds.push([face, tris[Math.floor(rng() * tris.length)], tris]);
    }
    const adjacency = buildAdjacency(P, facets);
    attachCurvature(P, facets);
    await annotateEdges(facets, adjacency, edge.run, edge.meta.columns);
    const wall = edgeWall(edge.thr, CURVED);
    for (const [face, seed, tris] of seeds) {
      const tf = fitterFor(TYPE_NAMES[T[seed]]);
      if (!tf) continue;
      const fill = paintFrom(seed, tf, facets, adjacency, { wall });
      if (!fill.params) continue;
      const truth = new Set(tris);
      let hit = 0, sameType = 0;
      for (const t of fill.region) { if (truth.has(t)) hit++; else if (T[t] === T[seed]) sameType++; }
      const purity = (hit + sameType) / (fill.region.size || 1);
      const recall = hit / truth.size;
      const clicks = [-1];
      if (purity < 0.9) {
        const c = simulatedClick(seed, fill.region, adjacency, facets, T);
        if (c >= 0) clicks.push(c);
      }
      for (const click of clicks) {
        const r = refineFeatures(seed, facets, adjacency, { region: fill.region, type: tf, params: fill.params }, click);
        if (click >= 0 && !r.ids.includes(click)) continue;
        const pid = patches.length;
        patches.push({ part: partIdx, mesher: mesherIdx, face, seed, type: T[seed], click, n: r.n,
                       purity, recall, faceSize: truth.size, fillSize: fill.region.size });
        const meta = new Int32Array(r.n * 7);
        for (let i = 0; i < r.n; i++) {
          const t = r.ids[i];
          meta.set([pid, mesherIdx, partIdx, T[seed], F[t] === face ? 1 : 0, T[t] === T[seed] ? 1 : 0,
                    fill.region.has(t) ? 1 : 0], i * 7);
        }
        write(r, meta);
      }
    }
  }
  if (partIdx % 50 === SHARD[0]) console.log(`  part ${partIdx}, ${patches.length} patches, ${rows} rows, ${((performance.now() - t0) / 1000).toFixed(0)} s`);
}
closeSync(featFd);
closeSync(metaFd);
closeSync(nbrFd);
closeSync(eattrFd);
closeSync(eprobFd);
writeFileSync(`${OUT}.json`, JSON.stringify({ ncol: N_REFINE, refineColumns: REFINE_COLUMNS, meshers: manifest.variants,
  typeNames: TYPE_NAMES, rows, patches }, null, 0));
console.log(`wrote ${rows} rows from ${patches.length} patches to ${OUT}.*`);
