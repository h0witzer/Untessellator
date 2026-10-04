/**
 * edgeScreen.mjs — is this edge a B-rep face boundary? Dump labelled mesh edges for probing.
 *
 * Usage:  node test/edgeScreen.mjs <dir> <out prefix> [--interior-rate 0.25] [--parts N]
 *         <dir> from ml/export_fill_eval.py; writes <prefix>.feat (float32 rows), <prefix>.meta
 *         (int32: part, mesher, sameFace, sameType, dihedral x 10) and <prefix>.json.
 *
 * WHY. Spill is what keeps fills from being clean: with the TRUE type declared, 23% of clicks on
 * OCCT meshes and 43% on gmsh still spill (test/fillEval.mjs). The hand-built walls only go so far:
 * a crease angle cannot see a tangent junction, and a curvature jump is noisy on planes and split
 * true faces when auto-paint used it. But every edge of every baked part carries an exact label,
 * so "is this a face boundary" can be learned rather than tuned.
 *
 * The feature row itself is js/edgeFeatures.js, shared with the studio runtime so the model is
 * scored on exactly the rows it will be given. ml/edge_screen.py probes these; ml/edge_train.py
 * trains the shipped scorer.
 *
 * All boundary edges are kept; interior edges are sampled at `--interior-rate` (recorded in the
 * header, so a probe can weight them back).
 *
 * SECOND STAGE: with `--stage1-even <run dir> --stage1-odd <run dir>`, every edge of each mesh is
 * first scored by a first-stage model that did NOT train on that part (even-indexed parts by the
 * one named for even parts), and the rows gain js/edgeFeatures.js's CONTEXT_COLUMNS: what the
 * first-stage scores around the edge say. That is the stacked (auto-context) training set.
 */

import { readFileSync, openSync, writeSync, closeSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { createRequire } from 'node:module';
import { annotateEdges } from '../js/edgeWall.js';
import { attachEdgeGraph, edgeScores } from '../js/edgeGraph.js';
import { buildFacets, buildAdjacency } from '../js/meshCore.js';
import { attachCurvature } from '../js/curvature.js';
import { edgeFeatures, edgeContext, EDGE_COLUMNS, EDGE_GROUPS, N_EDGE_FEATURES, CONTEXT_COLUMNS } from '../js/edgeFeatures.js';
import { TYPE_NAMES } from '../js/seedNet.js';

const args = process.argv.slice(2);
function flag(name, dflt) {
  const i = args.indexOf(name);
  if (i < 0) return dflt;
  const v = args[i + 1];
  args.splice(i, 2);
  return v;
}
const INTERIOR_RATE = Number(flag('--interior-rate', '0.25'));
const PART_LIMIT = Number(flag('--parts', '0'));
const STAGE1_EVEN = flag('--stage1-even', null);
const STAGE1_ODD = flag('--stage1-odd', null);
const [DIR, OUT] = args;
const ort = createRequire(import.meta.url)('onnxruntime-node');
async function stage1(dir) {
  const meta = JSON.parse(readFileSync(join(dir, 'edge.json'), 'utf8'));
  const s = await ort.InferenceSession.create(join(dir, 'edge.onnx'), { intraOpNumThreads: 2, interOpNumThreads: 1 });
  return { meta, run: async (f, n) => (await s.run({ features: new ort.Tensor('float32', f, [n, f.length / n]) })).logit.data };
}
const S1 = STAGE1_EVEN && STAGE1_ODD ? { even: await stage1(STAGE1_EVEN), odd: await stage1(STAGE1_ODD) } : null;
const NCOL = N_EDGE_FEATURES + (S1 ? CONTEXT_COLUMNS.length : 0);
if (!DIR || !OUT) { console.error('usage: node test/edgeScreen.mjs <dir> <out prefix>'); process.exit(2); }

const copy = (b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.length);
const load = (stem) => ({
  P: Float64Array.from(new Float32Array(copy(readFileSync(join(DIR, `${stem}.pos`))))),
  F: new Int32Array(copy(readFileSync(join(DIR, `${stem}.fid`)))),
  T: new Uint8Array(copy(readFileSync(join(DIR, `${stem}.typ`)))),
});
let rand = 12345;
const rng = () => (rand = (rand * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;

const manifest = JSON.parse(readFileSync(join(DIR, 'manifest.json'), 'utf8'));
const parts = PART_LIMIT ? manifest.parts.slice(0, PART_LIMIT) : manifest.parts;
const featFd = openSync(`${OUT}.feat`, 'w');
const metaFd = openSync(`${OUT}.meta`, 'w');
let rows = 0, boundaries = 0;
const t0 = performance.now();

for (const [partIdx, entry] of parts.entries()) {
  for (const [mesherIdx, mesher] of manifest.variants.entries()) {
    const v = entry.variants[mesher];
    if (!v) continue;
    const { P, F, T } = load(v.stem);
    const facets = buildFacets(P);
    const adjacency = buildAdjacency(P, facets);
    attachCurvature(P, facets);
    const { row } = edgeFeatures(facets, adjacency);
    let context = null;
    if (S1) {
      const m = partIdx % 2 === 0 ? S1.even : S1.odd;
      await annotateEdges(facets, adjacency, m.run, m.meta.columns);
      const graph = attachEdgeGraph(P, facets, adjacency);
      context = { graph, c: edgeContext(graph, adjacency, edgeScores(graph, adjacency)) };
    }

    const block = [];
    const meta = [];
    const scratch = new Float32Array(N_EDGE_FEATURES);
    for (const [a, edges] of adjacency) {
      for (const e of edges) {
        const b = e.neighbor;
        if (b < a || facets.degenerate[a] || facets.degenerate[b]) continue;
        const boundary = F[a] !== F[b];
        if (!boundary && rng() > INTERIOR_RATE) continue;
        row(a, b, e.angle, scratch, 0);
        block.push(...scratch);
        if (context) for (let k = 0; k < CONTEXT_COLUMNS.length; k++) block.push(context.c[e.id * CONTEXT_COLUMNS.length + k]);
        meta.push(partIdx, mesherIdx, boundary ? 1 : 0, T[a] === T[b] ? 1 : 0, Math.round(e.angle * 10));
        rows++;
        if (boundary) boundaries++;
      }
    }
    writeSync(featFd, new Uint8Array(Float32Array.from(block).buffer));
    writeSync(metaFd, new Uint8Array(Int32Array.from(meta).buffer));
  }
  if (partIdx % 10 === 9) console.log(`  ${partIdx + 1} parts, ${rows} edges (${boundaries} boundaries), ${((performance.now() - t0) / 1000).toFixed(0)} s`);
}

closeSync(featFd);
closeSync(metaFd);
writeFileSync(`${OUT}.json`, JSON.stringify({
  ncol: NCOL, columns: S1 ? [...EDGE_COLUMNS, ...CONTEXT_COLUMNS] : EDGE_COLUMNS,
  groups: S1 ? { ...EDGE_GROUPS, context: CONTEXT_COLUMNS } : EDGE_GROUPS, meshers: manifest.variants,
  typeNames: TYPE_NAMES, rows, boundaries, interiorRate: INTERIOR_RATE,
}, null, 1));
console.log(`wrote ${rows} edges (${boundaries} boundaries, interior sampled at ${INTERIOR_RATE}) to ${OUT}.*`);
