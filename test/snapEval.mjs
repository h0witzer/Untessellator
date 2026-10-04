/**
 * snapEval.mjs — should a click's fill start from the most coherent facet under the brush?
 *
 * Usage:  node test/snapEval.mjs <dir> <model.onnx> [--method model|hyp] [--json out.json]
 *
 * Automatic painting gained most from choosing WHERE to seed (test/autoSeed.mjs: seeding where one
 * primitive best explains the crease-bounded neighbourhood). A click fixes the face but not the
 * best starting facet on it. This moves the seed, before filling, to the facet with the lowest
 * autoPaint.coherenceOf score among facets within R of the clicked one (centroid distance, reached
 * across edges below 30 degrees, so never over a crease), R = k x the median facet size. The fill is
 * scored against the CLICKED facet's face, so a snap onto a neighbouring face counts against it.
 *
 * Same clicks as test/fillEval.mjs (2 faces per type, 10 per part, one random facet each).
 * Fills are js/paintCore.paintWith with type 'auto' (the studio's model-capped ordered type).
 */

import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { createRequire } from 'node:module';

import { preparePaintPositions, paintWith } from '../js/paintCore.js';
import { coherenceOf } from '../js/autoPaint.js';
import { TYPE_NAMES, proposeRegion } from '../js/seedNet.js';

const ort = createRequire(import.meta.url)('onnxruntime-node');
const args = process.argv.slice(2);
function flag(name, dflt) {
  const i = args.indexOf(name);
  if (i < 0) return dflt;
  const v = args[i + 1];
  args.splice(i, 2);
  return v;
}
const METHOD = flag('--method', 'model');
const JSON_OUT = flag('--json', null);
const [DIR, MODEL] = args;
if (!DIR || !MODEL) { console.error('usage: node test/snapEval.mjs <dir> <model.onnx> [--method model|hyp]'); process.exit(2); }
// Variants: k = brush radius in median facet sizes; region = candidates limited to the model's
// proposed region from the click (facets it already puts on the clicked face); margin = move only
// when the best candidate's score is below margin x the clicked facet's.
// First run (plain k 3/6/12): clean fills fell 3-8 points on CAD meshes, because 8-48% of snaps left
// the clicked face.
const VARIANTS = [
  { k: 0 },
  { k: 3 },
  { k: 6, region: true },
  { k: 12, region: true },
  { k: 6, margin: 0.5 },
  { k: 6, region: true, margin: 0.5 },
];
const variantLabel = (v) => (v.k ? `snap ${v.k}${v.region ? ' in model region' : ''}${v.margin ? ` margin ${v.margin}` : ''}` : 'no snap');

const session = await ort.InferenceSession.create(MODEL, { intraOpNumThreads: 2, interOpNumThreads: 1 });
let extended = false;
try { extended = JSON.parse(readFileSync(MODEL.replace(/\.onnx$/, '.json'), 'utf8')).features === 'v6'; } catch { /* base */ }
const runner = async (features, n) => {
  const out = await session.run({
    features: new ort.Tensor('float32', features, [1, n, features.length / n]),
    mask: new ort.Tensor('float32', new Float32Array(n).fill(1), [1, n]),
  });
  return { sameLogit: out.same_logit.data, typeLogit: Array.from(out.type_logit.data), surface: out.surface ? out.surface.data : null };
};

const copy = (b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.length);
const load = (stem) => ({
  P: Float64Array.from(new Float32Array(copy(readFileSync(join(DIR, `${stem}.pos`))))),
  F: new Int32Array(copy(readFileSync(join(DIR, `${stem}.fid`)))),
  T: new Uint8Array(copy(readFileSync(join(DIR, `${stem}.typ`)))),
});
let x = 99;
const rng = () => (x = (x * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;

/** Facets within `radius` of the seed's centroid, reached across edges below 30 degrees. */
function brushFacets(seed, facets, adjacency, radius) {
  const C = facets.centroids;
  const out = [seed];
  const seen = new Set([seed]);
  const stack = [seed];
  while (stack.length) {
    const t = stack.pop();
    for (const e of adjacency.get(t) || []) {
      const u = e.neighbor;
      if (seen.has(u) || e.angle >= 30) continue;
      seen.add(u);
      const d = Math.hypot(C[u * 3] - C[seed * 3], C[u * 3 + 1] - C[seed * 3 + 1], C[u * 3 + 2] - C[seed * 3 + 2]);
      if (d > radius) continue;
      out.push(u);
      stack.push(u);
    }
  }
  return out;
}

const manifest = JSON.parse(readFileSync(join(DIR, 'manifest.json'), 'utf8'));
const reference = manifest.variants[0];
const rows = [];
const t0 = performance.now();

for (const [pi, entry] of manifest.parts.entries()) {
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

  for (const mesher of manifest.variants) {
    const v = entry.variants[mesher];
    if (!v) continue;
    const { P, F, T } = mesher === reference ? base : load(v.stem);
    const mesh = preparePaintPositions(P);
    const { facets, adjacency } = mesh;
    const sizes = [];
    for (let t = 0; t < facets.count; t++) sizes.push(Math.sqrt(Math.max(facets.areas[t], 0)));
    sizes.sort((a, b) => a - b);
    const median = sizes[sizes.length >> 1] || 1;
    const byFace = new Map();
    for (let t = 0; t < facets.count; t++) {
      if (facets.degenerate[t]) continue;
      if (!byFace.has(F[t])) byFace.set(F[t], []);
      byFace.get(F[t]).push(t);
    }
    const scoreCache = new Map();
    const score = (t) => { if (!scoreCache.has(t)) scoreCache.set(t, coherenceOf(t, facets, adjacency)); return scoreCache.get(t); };

    for (const face of seedFaces) {
      const tris = byFace.get(face);
      if (!tris || tris.length < 3) continue;
      const click = tris[Math.floor(rng() * tris.length)];
      const truth = new Set(tris);
      const truthType = TYPE_NAMES[T[click]];
      const proposal = VARIANTS.some((v) => v.region)
        ? await proposeRegion(click, facets, adjacency, runner, { extended })
        : null;
      for (const v of VARIANTS) {
        const k = v.k;
        let seed = click;
        if (k > 0) {
          const own = score(click);
          let best = own;
          let bestSeed = click;
          for (const u of brushFacets(click, facets, adjacency, k * median)) {
            if (v.region && !proposal.region.has(u)) continue;
            const s = score(u);
            if (s < best || (s === best && facets.areas[u] > facets.areas[bestSeed])) { best = s; bestSeed = u; }
          }
          if (!v.margin || best < v.margin * own) seed = bestSeed;
        }
        const out = await paintWith(runner, seed, mesh, { method: METHOD, type: 'auto', extended });
        let hit = 0, sameType = 0;
        for (const t of out.region) {
          if (truth.has(t)) hit++;
          else if (T[t] === T[click]) sameType++;
        }
        const n = out.region.size || 1;
        rows.push({ part: pi, mesher, variant: variantLabel(v), truthType, moved: seed !== click, offFace: !truth.has(seed),
                    purity: (hit + sameType) / n, recall: hit / truth.size,
                    typeOk: (out.fitter ?? 'bspline') === truthType.toLowerCase() });
      }
    }
  }
  if (pi % 10 === 9) console.log(`  ${pi + 1} parts, ${((performance.now() - t0) / 1000).toFixed(0)} s`);
}

const pc = (v) => `${Math.round(100 * v)}%`.padStart(4);
const frac = (rs, fn) => rs.filter(fn).length / (rs.length || 1);
const clean = (r) => r.purity >= 0.95 && r.recall >= 0.95;
console.log(`\nsnap evaluation, method ${METHOD}, ${rows.length / VARIANTS.length} clicks, ${((performance.now() - t0) / 1000).toFixed(0)} s`);
for (const mesher of manifest.variants) {
  console.log(`  ${mesher}`);
  for (const v of VARIANTS) {
    const rs = rows.filter((r) => r.mesher === mesher && r.variant === variantLabel(v));
    console.log(`    ${variantLabel(v).padEnd(32)}  clean ${pc(frac(rs, clean))}` +
      `  spill ${pc(frac(rs, (r) => r.purity < 0.9))}  incomplete ${pc(frac(rs, (r) => r.recall < 0.9))}` +
      `  type ${pc(frac(rs, (r) => r.typeOk))}  moved ${pc(frac(rs, (r) => r.moved))}  off the face ${pc(frac(rs, (r) => r.offFace))}`);
  }
}
if (JSON_OUT) writeFileSync(JSON_OUT, JSON.stringify(rows));
