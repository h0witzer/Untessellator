/**
 * showcase.mjs — run the held-out assets through the real pipeline.
 *
 * Usage:  node test/showcase.mjs [dir] [seeds-per-model]
 *
 * WHAT THIS IS, AND IS NOT
 *
 * This is a SMOKE TEST across geometry the tuning never saw: does the pipeline survive
 * a Blender sculpt, a 225k-facet CAD export with degenerate slivers in it, a teapot
 * made of Bezier patches, and an exact analytic torus? Does it classify them
 * plausibly, and how long does it take?
 *
 * It is NOT validation. Fourteen hand-picked models are a broader basis than one
 * WatchStand, but they are still hand-picked, and picked by us. Proving the tool works
 * means reconstructing a substantial portion of ABC from mesh input and scoring against
 * the STEP each model came from — 24,790 baked models with per-facet ground truth and
 * 200,000 STEP files are already on disk for exactly that. See docs/ml-experiment-scope.md.
 *
 * Read the output as "nothing fell over and the answers look sane", not as a score.
 *
 * Ground truth exists for only part of the set:
 *   torus_exact_*   an exact torus, R=1.0 r=0.4 — but there is NO torus fitter yet,
 *                   so the best available answer is whatever the model predicts
 *   teapot_*        28 bicubic Bezier patches
 *   blender_donut   the recognisable one; no analytic truth
 *   3DBenchy        CAD-origin; its STEP is public domain but not in the repo
 */

import { readFileSync, readdirSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { createRequire } from 'node:module';

import { buildFacets, buildAdjacency, segment } from '../js/meshCore.js';
import { paintFrom } from '../js/regionGrow.js';
import { proposeRegion, TYPE_NAMES, fitterFor } from '../js/seedNet.js';
import { attachCurvature } from '../js/curvature.js';

const require = createRequire(import.meta.url);
const ort = require('onnxruntime-node');

const DIR = process.argv[2] || 'D:/abc-data/showcase';
const SEEDS = Number(process.argv[3] || 40);
const MODEL = 'D:/abc-data/runs/v1/seedfill.onnx';

function readSTL(path) {
  const b = readFileSync(path);
  if (b.length > 84) {
    const n = b.readUInt32LE(80);
    if (84 + n * 50 === b.length) {
      const P = new Float64Array(n * 9);
      for (let t = 0; t < n; t++) {
        const o = 84 + t * 50;
        for (let i = 0; i < 9; i++) P[t * 9 + i] = b.readFloatLE(o + 12 + i * 4);
      }
      return P;
    }
  }
  const v = [...b.toString('utf8').matchAll(/vertex\s+(\S+)\s+(\S+)\s+(\S+)/g)];
  const P = new Float64Array(v.length * 3);
  v.forEach((m, i) => { P[i * 3] = +m[1]; P[i * 3 + 1] = +m[2]; P[i * 3 + 2] = +m[3]; });
  return P;
}

if (!existsSync(MODEL)) { console.error(`no model at ${MODEL}`); process.exit(2); }
const session = await ort.InferenceSession.create(MODEL);
const runner = async (features, n) => {
  const out = await session.run({
    features: new ort.Tensor('float32', features, [1, n, 10]),
    mask: new ort.Tensor('float32', new Float32Array(n).fill(1), [1, n]),
  });
  return { sameLogit: out.same_logit.data, typeLogit: Array.from(out.type_logit.data) };
};

const files = readdirSync(DIR).filter(f => f.endsWith('.stl')).sort();
console.log(`\nshowcase smoke test — ${files.length} assets, ${SEEDS} seeds each`);
console.log(`model: ${MODEL}\n`);
console.log(`  ${'asset'.padEnd(26)} ${'facets'.padStart(8)} ${'deg'.padStart(4)} ${'comp'.padStart(5)}` +
  `  ${'claimed'.padStart(8)} ${'ms'.padStart(6)}   predicted types`);
console.log(`  ${'-'.repeat(26)} ${'-'.repeat(8)} ${'-'.repeat(4)} ${'-'.repeat(5)}  ${'-'.repeat(8)} ${'-'.repeat(6)}`);

for (const name of files) {
  let P;
  try { P = readSTL(join(DIR, name)); } catch (e) { console.log(`  ${name}: unreadable (${e.message})`); continue; }
  const facets = buildFacets(P);
  if (facets.count < 16) { console.log(`  ${name}: too few facets`); continue; }

  let deg = 0;
  for (let t = 0; t < facets.count; t++) if (facets.degenerate[t]) deg++;
  const adjacency = buildAdjacency(P, facets);
  attachCurvature(P, facets);   // one pass per mesh; gives fitTorus a local handle on r

  // connected components, to expose meshes that are secretly in pieces
  const seenC = new Uint8Array(facets.count);
  let comps = 0;
  for (let s = 0; s < facets.count; s++) {
    if (seenC[s] || facets.degenerate[s]) continue;
    comps++;
    const st = [s]; seenC[s] = 1;
    while (st.length) {
      const t = st.pop();
      for (const e of adjacency.get(t) || []) if (!seenC[e.neighbor]) { seenC[e.neighbor] = 1; st.push(e.neighbor); }
    }
  }

  // sample seeds, propose, and fit through the existing path where a fitter exists
  const rng = (() => { let x = 12345; return () => (x = (x * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff; })();
  const typeCount = {};
  const claimed = new Set();
  let ms = 0, fitted = 0, attempted = 0, resid = 0;

  for (let k = 0; k < SEEDS; k++) {
    let seed = Math.floor(rng() * facets.count);
    let guard = 0;
    while (facets.degenerate[seed] && guard++ < 50) seed = Math.floor(rng() * facets.count);
    if (facets.degenerate[seed]) continue;

    const t0 = performance.now();
    const p = await proposeRegion(seed, facets, adjacency, runner);
    ms += performance.now() - t0;
    typeCount[p.type] = (typeCount[p.type] || 0) + 1;

    const fitter = fitterFor(p.type);
    if (!fitter) { for (const t of p.region) claimed.add(t); continue; }
    attempted++;
    const r = paintFrom(seed, fitter, facets, adjacency, { seedRegion: p.region });
    if (r.params && isFinite(r.relResidual)) { fitted++; resid += r.relResidual; }
    for (const t of r.region) claimed.add(t);
  }

  const top = Object.entries(typeCount).sort((a, b) => b[1] - a[1])
    .map(([k, v]) => `${k} ${v}`).join(', ');
  console.log(`  ${name.padEnd(26)} ${String(facets.count).padStart(8)} ${String(deg).padStart(4)} ` +
    `${String(comps).padStart(5)}  ${(100 * claimed.size / facets.count).toFixed(1).padStart(7)}% ` +
    `${(ms / SEEDS).toFixed(1).padStart(6)}   ${top}`);
  if (attempted) {
    console.log(`  ${''.padEnd(26)} ${'fits'.padStart(8)} ${String(fitted).padStart(4)}/${String(attempted).padEnd(4)}` +
      `  mean relResidual ${(resid / Math.max(fitted, 1)).toExponential(2)}`);
  }
}

console.log(`\n  deg = degenerate facets, comp = connected components,`);
console.log(`  claimed = fraction of the mesh covered by ${SEEDS} seeded fills,`);
console.log(`  ms = mean per model call (k-ring BFS + features + onnxruntime-node).`);
console.log(`\n  Reminder: smoke test, not validation. See the module header.\n`);
