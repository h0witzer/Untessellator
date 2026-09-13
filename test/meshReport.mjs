/**
 * meshReport.mjs — label-free report for meshes that have no CAD ground truth.
 *
 * Usage:  node test/meshReport.mjs <file|dir> [seeds] [--json out.json] [--bogus 1e-2]
 *         SEEDNET=path/to/seedfill.onnx overrides the model. Without a model the
 *         behaviour lines are skipped and everything else still runs.
 *
 * WHY THIS EXISTS
 *
 * oracle.mjs and rescore.py score against a STEP. Game assets, sculpts and hand-built
 * cosplay meshes have none, and they are a target in their own right
 * (docs/game-assets-scope.md). Without truth, four things can still be measured:
 *
 *   topology   components, open and non-manifold edges. Does a solid even exist?
 *   pattern    triangle-area spread and crease-group counts. How unlike a CAD tessellator?
 *   polygons   (OBJ only) triangle / quad / n-gon share. Artist topology survives here
 *              and nowhere else: STL and three's OBJLoader both fan-triangulate it away.
 *   behaviour  seeded fills: predicted types, fit residuals, and BOGUS FITS, seeds whose
 *              own fit residual says the predicted surface is not there. On a mesh with no
 *              analytic structure a converter that trusts its type head emits surfaces that
 *              do not follow the shape; the residual is what should route those regions to
 *              a freeform path instead.
 *
 * Topology uses the SAME weld as meshCore.buildAdjacency (absolute quantum 1e-4), so an
 * open edge here is an edge the pipeline cannot cross, not a different definition of one.
 *
 * DUPLICATES are counted always and removed only with --dedupe. Measured: Kenney's Blaster
 * Kit OBJs repeat about half their triangles with the same winding, which makes every
 * edge four-owned, so buildAdjacency finds no neighbours and each triangle is its own
 * component. That is a hygiene failure the converter has to survive, not a property of
 * the shape, so the report shows both.
 *
 * CURVATURE must match how the model was trained. v2ctrl is the control arm, trained with
 * columns 10-11 zeroed (rescore.py --zero-curvature says the same). seedNet.buildFeatures
 * fills them whenever attachCurvature has run, so a harness that attaches curvature for
 * the torus fitter silently feeds v2ctrl inputs it never saw. --curvature off (default)
 * zeroes the model's columns while the fitters keep real curvature; use --curvature on
 * with SEEDNET pointing at v2curv.
 *
 * This is not hypothetical. The first version of this report fed curvature to v2ctrl and
 * read masterchief.stl as Plane for 49 of 60 seeds. Zeroed as trained, the same seeds are
 * Cone 37, BSpline 9, Plane 1. showcase.mjs has the same mismatch.
 *
 * NOT VALIDATION. A low bogus-fit rate means the tool stopped claiming surfaces that are
 * not there; it says nothing about whether what it claimed instead is any good. The
 * --bogus threshold is provisional until calibrated against genuine planes in ABC.
 */

import { readFileSync, readdirSync, statSync, writeFileSync, existsSync } from 'node:fs';
import { join, basename, extname } from 'node:path';
import { createRequire } from 'node:module';

import { buildFacets, buildAdjacency, segment } from '../js/meshCore.js';
import { paintFrom } from '../js/regionGrow.js';
import { proposeRegion, fitterFor, N_FEATURES } from '../js/seedNet.js';
import { attachCurvature } from '../js/curvature.js';

const args = process.argv.slice(2);
function flag(name, dflt) {
  const i = args.indexOf(name);
  if (i < 0) return dflt;
  const v = args[i + 1];
  args.splice(i, 2);
  return v;
}
function boolFlag(name) {
  const i = args.indexOf(name);
  if (i < 0) return false;
  args.splice(i, 1);
  return true;
}
const JSON_OUT = flag('--json', null);
const BOGUS = Number(flag('--bogus', 1e-2));
const CURVATURE = flag('--curvature', 'off');
const DEDUPE = boolFlag('--dedupe');
const TARGET = args[0];
const SEEDS = Number(args[1] || 60);
const MODEL = process.env.SEEDNET || 'D:/abc-data/runs/v2ctrl/seedfill.onnx';
const QUANT = 1e4;   // must match buildAdjacency's default

if (!TARGET) {
  console.error('usage: node test/meshReport.mjs <file|dir> [seeds] [--json out.json]');
  process.exit(2);
}

// ── Readers ──────────────────────────────────────────────────────────────────

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
      return { P, polys: null };
    }
  }
  const v = [...b.toString('utf8').matchAll(/vertex\s+(\S+)\s+(\S+)\s+(\S+)/g)];
  const P = new Float64Array(v.length * 3);
  v.forEach((m, i) => { P[i * 3] = +m[1]; P[i * 3 + 1] = +m[2]; P[i * 3 + 2] = +m[3]; });
  return { P, polys: null };
}

/** Fan-triangulate exactly as three's OBJLoader does, but count polygon sizes first. */
function readOBJ(path) {
  const V = [];
  const tris = [];
  const polys = { tri: 0, quad: 0, ngon: 0 };
  for (const line of readFileSync(path, 'utf8').split('\n')) {
    if (line.startsWith('v ')) {
      const [, x, y, z] = line.trim().split(/\s+/);
      V.push(+x, +y, +z);
    } else if (line.startsWith('f ')) {
      const idx = line.trim().split(/\s+/).slice(1).map(tok => {
        const i = parseInt(tok, 10);                 // "7/3/2" parses as 7
        return i < 0 ? V.length / 3 + i : i - 1;
      });
      if (idx.length < 3) continue;
      polys[idx.length === 3 ? 'tri' : idx.length === 4 ? 'quad' : 'ngon']++;
      for (let j = 1; j < idx.length - 1; j++) tris.push(idx[0], idx[j], idx[j + 1]);
    }
  }
  const P = new Float64Array(tris.length * 3);
  tris.forEach((vi, k) => {
    P[k * 3] = V[vi * 3]; P[k * 3 + 1] = V[vi * 3 + 1]; P[k * 3 + 2] = V[vi * 3 + 2];
  });
  return { P, polys };
}

/** Drop repeated triangles, ignoring winding, on the pipeline's weld quantum. */
function dedupe(P) {
  const key = o => Math.round(P[o] * QUANT) + '_' + Math.round(P[o + 1] * QUANT) + '_' +
    Math.round(P[o + 2] * QUANT);
  const n = P.length / 9;
  const seen = new Set();
  const keep = [];
  for (let t = 0; t < n; t++) {
    const k = [key(t * 9), key(t * 9 + 3), key(t * 9 + 6)].sort().join('|');
    if (seen.has(k)) continue;
    seen.add(k);
    keep.push(t);
  }
  const Q = new Float64Array(keep.length * 9);
  keep.forEach((t, i) => Q.set(P.subarray(t * 9, t * 9 + 9), i * 9));
  return { P: Q, removed: n - keep.length };
}

// ── Measures ─────────────────────────────────────────────────────────────────

function topology(P, facets) {
  const key = o => Math.round(P[o] * QUANT) + '_' + Math.round(P[o + 1] * QUANT) + '_' +
    Math.round(P[o + 2] * QUANT);
  const verts = new Set();
  const edges = new Map();
  for (let t = 0; t < facets.count; t++) {
    if (facets.degenerate[t]) continue;
    const k = [key(t * 9), key(t * 9 + 3), key(t * 9 + 6)];
    for (let e = 0; e < 3; e++) {
      verts.add(k[e]);
      const a = k[e], b = k[(e + 1) % 3];
      const ek = a < b ? a + '|' + b : b + '|' + a;
      edges.set(ek, (edges.get(ek) || 0) + 1);
    }
  }
  let open = 0, nonManifold = 0;
  for (const n of edges.values()) { if (n === 1) open++; else if (n > 2) nonManifold++; }
  return { weldedVertices: verts.size, edges: edges.size, openEdges: open, nonManifoldEdges: nonManifold };
}

function components(facets, adjacency) {
  const seen = new Uint8Array(facets.count);
  const sizes = [];
  for (let s = 0; s < facets.count; s++) {
    if (seen[s] || facets.degenerate[s]) continue;
    let c = 0;
    const st = [s]; seen[s] = 1;
    while (st.length) {
      const t = st.pop(); c++;
      for (const e of adjacency.get(t) || []) if (!seen[e.neighbor]) { seen[e.neighbor] = 1; st.push(e.neighbor); }
    }
    sizes.push(c);
  }
  return sizes.sort((a, b) => b - a);
}

const pct = (sorted, p) => sorted.length ? sorted[Math.floor(p * (sorted.length - 1))] : NaN;

function areaSpread(facets) {
  const A = [];
  for (let t = 0; t < facets.count; t++) if (!facets.degenerate[t]) A.push(facets.areas[t]);
  A.sort((a, b) => a - b);
  return { p10: pct(A, 0.1), p50: pct(A, 0.5), p90: pct(A, 0.9), ratio90to10: pct(A, 0.9) / pct(A, 0.1) };
}

async function behaviour(facets, adjacency, runner) {
  const rng = (() => { let x = 12345; return () => (x = (x * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff; })();
  // an all-invalid curvature object makes buildFeatures write zeros; fitters still read facets.curvature
  const proposeOpts = CURVATURE === 'on' ? {} : { curvature: { valid: new Uint8Array(facets.count) } };
  const types = {};
  const residuals = [];
  const claimed = new Set();
  const byType = {};
  let attempted = 0, fitted = 0, ms = 0;

  for (let k = 0; k < SEEDS; k++) {
    let seed = Math.floor(rng() * facets.count);
    for (let g = 0; facets.degenerate[seed] && g < 50; g++) seed = Math.floor(rng() * facets.count);
    if (facets.degenerate[seed]) continue;

    const t0 = performance.now();
    const p = await proposeRegion(seed, facets, adjacency, runner, proposeOpts);
    ms += performance.now() - t0;
    types[p.type] = (types[p.type] || 0) + 1;

    const fitter = fitterFor(p.type);
    if (!fitter) { for (const t of p.region) claimed.add(t); continue; }
    attempted++;
    const r = paintFrom(seed, fitter, facets, adjacency, { seedRegion: p.region });
    const ok = r.params && isFinite(r.relResidual);
    if (ok) { fitted++; residuals.push(r.relResidual); }
    const s = byType[p.type] || (byType[p.type] = { seeds: 0, fitted: 0, bogus: 0, failed: 0 });
    s.seeds++;
    if (!ok) s.failed++;
    else { s.fitted++; if (r.relResidual > BOGUS) s.bogus++; }
    for (const t of r.region) claimed.add(t);
  }
  residuals.sort((a, b) => a - b);
  return {
    seeds: SEEDS, types, attempted, fitted,
    residual: { p50: pct(residuals, 0.5), p90: pct(residuals, 0.9),
                mean: residuals.reduce((s, v) => s + v, 0) / Math.max(residuals.length, 1) },
    byType, bogus: Object.values(byType).reduce((a, s) => a + s.bogus, 0), bogusThreshold: BOGUS,
    claimedFraction: claimed.size / facets.count, msPerCall: ms / SEEDS,
  };
}

// ── Run ──────────────────────────────────────────────────────────────────────

let runner = null;
if (existsSync(MODEL)) {
  const ort = createRequire(import.meta.url)('onnxruntime-node');
  const session = await ort.InferenceSession.create(MODEL);
  runner = async (features, n) => {
    const out = await session.run({
      features: new ort.Tensor('float32', features, [1, n, N_FEATURES]),
      mask: new ort.Tensor('float32', new Float32Array(n).fill(1), [1, n]),
    });
    return { sameLogit: out.same_logit.data, typeLogit: Array.from(out.type_logit.data) };
  };
}

const files = statSync(TARGET).isDirectory()
  ? readdirSync(TARGET).filter(f => /\.(stl|obj)$/i.test(f)).sort().map(f => join(TARGET, f))
  : [TARGET];

console.log(`\nmesh report — ${files.length} file(s), ` +
  `${runner ? `${SEEDS} seeds, model ${MODEL}, curvature ${CURVATURE}` : 'no model'}` +
  `${DEDUPE ? ', deduplicated' : ''}\n`);
const reports = [];

for (const path of files) {
  const raw = extname(path).toLowerCase() === '.obj' ? readOBJ(path) : readSTL(path);
  const { polys } = raw;
  const dup = dedupe(raw.P);
  const P = DEDUPE ? dup.P : raw.P;
  const facets = buildFacets(P);
  if (facets.count < 16) { console.log(`  ${basename(path)}: too few facets`); continue; }
  let degenerate = 0;
  for (let t = 0; t < facets.count; t++) if (facets.degenerate[t]) degenerate++;

  const adjacency = buildAdjacency(P, facets);
  attachCurvature(P, facets);

  let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < P.length; i += 3) for (let k = 0; k < 3; k++) {
    lo[k] = Math.min(lo[k], P[i + k]); hi[k] = Math.max(hi[k], P[i + k]);
  }
  const comps = components(facets, adjacency);
  const rep = {
    file: basename(path), triangles: facets.count, degenerate,
    duplicateTriangles: dup.removed, deduplicated: DEDUPE, curvature: CURVATURE,
    diag: Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]),
    ...topology(P, facets),
    components: comps.length, largestComponents: comps.slice(0, 5),
    area: areaSpread(facets),
    creaseGroupsPerTriangle: Object.fromEntries([5, 15, 30].map(d =>
      [`${d}deg`, segment(adjacency, facets.count, d).length / facets.count])),
    polygons: polys,
    behaviour: runner ? await behaviour(facets, adjacency, runner) : null,
  };
  reports.push(rep);

  const f3 = v => (isFinite(v) ? v.toExponential(2) : '-');
  const crease = Object.entries(rep.creaseGroupsPerTriangle).map(([k, v]) => `${k} ${v.toFixed(3)}`).join('  ');
  console.log(`  ${rep.file}`);
  console.log(`    size       ${rep.triangles} tris, ${rep.weldedVertices} verts, diag ${rep.diag.toFixed(2)}, ${degenerate} degenerate, ` +
    `${dup.removed} duplicate${DEDUPE ? ' (removed)' : ''}`);
  console.log(`    topology   ${rep.components} components (largest ${rep.largestComponents.join(', ')}), ` +
    `${rep.openEdges} open edges, ${rep.nonManifoldEdges} non-manifold`);
  console.log(`    pattern    area p90/p10 ${rep.area.ratio90to10.toFixed(1)}   crease groups per tri: ${crease}`);
  if (polys) {
    const n = polys.tri + polys.quad + polys.ngon;
    console.log(`    polygons   ${n} faces: tri ${(100 * polys.tri / n).toFixed(1)}%  ` +
      `quad ${(100 * polys.quad / n).toFixed(1)}%  ngon ${(100 * polys.ngon / n).toFixed(1)}%`);
  }
  if (rep.behaviour) {
    const b = rep.behaviour;
    const top = Object.entries(b.types).sort((x, y) => y[1] - x[1]).map(([k, v]) => `${k} ${v}`).join(', ');
    console.log(`    types      ${top}`);
    console.log(`    fits       ${b.fitted}/${b.attempted}, relResidual p50 ${f3(b.residual.p50)} p90 ${f3(b.residual.p90)}, ` +
      `claimed ${(100 * b.claimedFraction).toFixed(1)}%`);
    const bad = Object.entries(b.byType).filter(([, s]) => s.bogus || s.failed)
      .map(([k, s]) => `${k} ${s.bogus}+${s.failed}/${s.seeds}`).join(', ');
    console.log(`    bogus      ${b.bogus} of ${b.fitted} fits above ${f3(BOGUS)}; bogus+failed/seeds: ${bad || 'none'}`);
  }
  console.log('');
}

if (JSON_OUT) {
  writeFileSync(JSON_OUT, JSON.stringify(reports, null, 2));
  console.log(`wrote ${JSON_OUT}`);
}
