/**
 * realPairs.mjs — the studio's auto-paint on REAL exports, scored against the STEP each came from.
 *
 * Usage:  node test/realPairs.mjs <pairs.json> [--configs before,v1stack,v2stack] [--max-tris 300000]
 *                                  [--json out.json]
 *         pairs.json: [{ "name": "...", "step": "path.step", "mesh": "path.stl|.3mf" }, ...]
 *
 * ABC's meshes all come from our own tessellators (OCCT, gmsh, a game-art bake); a user's STL comes
 * from their CAD system's exporter. For any part saved both ways, this scores auto-paint the way
 * test/watchstand.mjs scores the WatchStand: a STEP face (plane, cylinder, cone, sphere) counts as
 * recovered when a painted group of the same type fits its defining parameters (radius within 3%,
 * axis within 5 degrees, half-angle within 2 degrees; greedy, largest group first). No per-triangle
 * labels are needed, so any pair works. Tori and freeform faces are counted but not matchable here.
 *
 * Units and frames: the mesh is scaled to the STEP's units by comparing its bounding box with the
 * STEP's vertex bounding box, snapped to a standard factor (mm, cm, m, inch). A pair whose ratio is
 * no standard factor, or whose boxes do not agree after scaling (a different part with the same
 * name, a rescaled or re-oriented copy), is reported and skipped.
 *
 * Caveat: planes are matched by normal direction alone, so on a part with many parallel planes a
 * configuration that makes more groups has more candidates to match. Read recovered counts next to
 * the group counts.
 *
 * Configurations, as in test/watchstandAuto.mjs:
 *   before    v6feat, boundary model v1 wall, first pass only (the studio's defaults before 2026-09-25)
 *   v1stack   v7gnn, boundary model v1 wall, refined fills
 *   v2stack   v7gnn, boundary model v2 wall, refined fills (the new default)
 */

import { readFileSync, writeFileSync } from 'node:fs';
import { join, dirname, extname } from 'node:path';
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
const MODELS = join(HERE, '..', 'models');
const DEG = 180 / Math.PI;
const RAD_TOL = 0.03, ANG_TOL = 5, CONE_TOL = 2;
const CURVED = ['cylinder', 'cone', 'sphere', 'torus'];

const args = process.argv.slice(2);
const flag = (name, dflt) => { const i = args.indexOf(name); if (i < 0) return dflt; const v = args[i + 1]; args.splice(i, 2); return v; };
const JSON_OUT = flag('--json', null);
const ALL = {
  before: { seed: 'seedfill-v6feat.onnx', edge: 'edge-boundary.onnx', refine: null },
  v1stack: { seed: 'seedfill-v7gnn.onnx', edge: 'edge-boundary.onnx', refine: 'fill-refiner.onnx' },
  v2stack: { seed: 'seedfill-v7gnn.onnx', edge: 'edge-boundary-v2.onnx', refine: 'fill-refiner.onnx' },
};
const CONFIGS = flag('--configs', 'before,v1stack,v2stack').split(',');
// Meshes past this are skipped: a threaded part exported at millions of triangles exhausts the heap.
const MAX_TRIS = Number(flag('--max-tris', '300000'));
const [PAIRS] = args;
if (!PAIRS) { console.error('usage: node test/realPairs.mjs <pairs.json> [--configs ...]'); process.exit(2); }

// ── mesh and STEP geometry ───────────────────────────────────────────────────────────────────────
function loadMesh(path) {
  if (extname(path).toLowerCase() === '.3mf') return load3MF(path, { maxBytes: 250e6 }).positions;
  const buf = readFileSync(path);
  const count = buf.length >= 84 ? buf.readUInt32LE(80) : 0;
  if (buf.length === 84 + count * 50) {
    const P = new Float64Array(count * 9);
    for (let t = 0; t < count; t++) for (let i = 0; i < 9; i++) P[t * 9 + i] = buf.readFloatLE(84 + t * 50 + 12 + i * 4);
    return P;
  }
  const v = [...buf.toString('latin1').matchAll(/vertex\s+(\S+)\s+(\S+)\s+(\S+)/g)].map((m) => [Number(m[1]), Number(m[2]), Number(m[3])]);
  return Float64Array.from(v.flat());
}
function boxOf(points) {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i + 2 < points.length; i += 3) for (let a = 0; a < 3; a++) { lo[a] = Math.min(lo[a], points[i + a]); hi[a] = Math.max(hi[a], points[i + a]); }
  return { lo, hi, size: [0, 1, 2].map((a) => hi[a] - lo[a]) };
}
/** Bounding box of the STEP's VERTEX_POINTs (the solid's corners), in file units. */
function stepVertexBox(path) {
  const text = readFileSync(path, 'latin1');
  const pts = new Map();
  for (const m of text.matchAll(/#(\d+)\s*=\s*CARTESIAN_POINT\s*\(\s*'[^']*'\s*,\s*\(([^)]*)\)\s*\)/g)) {
    pts.set(m[1], m[2].split(',').map(Number));
  }
  const out = [];
  for (const m of text.matchAll(/VERTEX_POINT\s*\(\s*'[^']*'\s*,\s*#(\d+)\s*\)/g)) {
    const p = pts.get(m[1]);
    if (p && p.length === 3) out.push(...p);
  }
  return boxOf(out);
}
/** Scale taking mesh units to STEP units, and whether the two boxes then agree (sorted extents within 5%). */
function unitScale(meshBox, stepBox) {
  const dm = Math.hypot(...meshBox.size), ds = Math.hypot(...stepBox.size);
  let k = ds / dm, unit = false;
  for (const c of [1, 10, 0.1, 1000, 0.001, 25.4, 1 / 25.4, 100, 0.01]) if (Math.abs(k / c - 1) < 0.03) { k = c; unit = true; break; }
  const a = [...meshBox.size].map((v) => v * k).sort((p, q) => p - q), b = [...stepBox.size].sort((p, q) => p - q);
  const agree = a.every((v, i) => Math.abs(v - b[i]) <= 0.05 * Math.max(b[2], 1e-9) + 1e-9);
  return { k, agree, unit };
}

// ── scoring (test/watchstand.mjs's matcher) ──────────────────────────────────────────────────────
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

// ── models ───────────────────────────────────────────────────────────────────────────────────────
const graphFeeds = (session, graph, n) => (session.inputNames.includes('nbr') && graph
  ? { nbr: new ort.Tensor('int64', graph.nbr, [1, n, 3]), eattr: new ort.Tensor('float32', graph.eattr, [1, n, 3]) } : {});
const runners = new Map();
async function modelRunner(file) {
  if (!runners.has(file)) {
    const s = await ort.InferenceSession.create(join(MODELS, file));
    runners.set(file, async (features, n, graph) => {
      const out = await s.run({
        features: new ort.Tensor('float32', features, [1, n, features.length / n]),
        mask: new ort.Tensor('float32', new Float32Array(n).fill(1), [1, n]),
        ...graphFeeds(s, graph, n),
      });
      return { sameLogit: out.same_logit.data, typeLogit: Array.from(out.type_logit.data), surface: out.surface ? out.surface.data : null };
    });
  }
  return runners.get(file);
}
const edges = new Map();
async function edgeModel(file) {
  if (!edges.has(file)) {
    const meta = JSON.parse(readFileSync(join(MODELS, file.replace(/\.onnx$/, '.json')), 'utf8'));
    const s = await ort.InferenceSession.create(join(MODELS, file));
    edges.set(file, { meta, run: async (f, n) => (await s.run({ features: new ort.Tensor('float32', f, [n, f.length / n]) })).logit.data });
  }
  return edges.get(file);
}

// ── the run ──────────────────────────────────────────────────────────────────────────────────────
const pairs = JSON.parse(readFileSync(PAIRS, 'utf8'));
const rows = [];
for (const pair of pairs) {
  let P, truth, stepBox;
  try {
    P = loadMesh(pair.mesh);
    truth = parseStepFaces(pair.step).faces;
    stepBox = stepVertexBox(pair.step);
  } catch (e) {
    console.log(`${pair.name}: could not read (${e.message})`);
    continue;
  }
  if (P.length / 9 > MAX_TRIS) {
    console.log(`${pair.name}: SKIPPED — ${(P.length / 9).toFixed(0)} triangles (over --max-tris ${MAX_TRIS})`);
    continue;
  }
  const { k, agree, unit } = unitScale(boxOf(P), stepBox);
  // A scale that is no unit conversion means a different part or a rescaled copy. Radii then never
  // match, and planes (matched by normal alone) reward whichever configuration makes more groups.
  if (!agree || !unit || !truth.length) {
    console.log(`${pair.name}: SKIPPED — ${!truth.length ? 'no faces parsed from the STEP'
      : !agree ? 'mesh and STEP boxes disagree (different part or frame)'
        : `scale x${k.toPrecision(3)} is not a unit conversion (different part or a rescaled copy)`}`);
    continue;
  }
  const scaled = Float64Array.from(P, (v) => v * k);
  const analytic = truth.filter((f) => ['plane', 'cylinder', 'cone', 'sphere'].includes(STEP_TO_PRIMITIVE[f.type])).length;
  const line = [];
  for (const cname of CONFIGS) {
    const cfg = ALL[cname];
    const t0 = performance.now();
    const mesh = preparePaintPositions(scaled);
    const { facets, adjacency } = mesh;
    const edge = await edgeModel(cfg.edge);
    await annotateEdges(facets, adjacency, edge.run, edge.meta.columns);
    const wall = edgeWall(1 / (1 + Math.exp(-edge.meta.thresholds['fpr0.02'].logit)), CURVED);
    const runner = await modelRunner(cfg.seed);
    const refine = cfg.refine ? { runner: await modelRunner(cfg.refine) } : null;
    const order = seedOrder(await coherentScores(facets, adjacency, { chunk: 1e9 }), facets);
    const seg = new Int32Array(facets.count).fill(-1);
    const segType = [];
    const result = await autoPaint({
      facets, order, owner: new Int32Array(facets.count).fill(-1), minClaim: 1, minKeep: 1, adjacency, absorbUpTo: 8,
      onAbsorb: (island, id) => { for (const t of island) seg[t] = id; },
      fill: (seed, taken) => paintWith(runner, seed, mesh, {
        method: 'model', type: 'auto', taken, extended: true, reportFit: false, grow: { wall }, ...(refine ? { refine } : {}),
      }),
      onSegment: (claim, out) => { const id = segType.length; for (const t of claim) seg[t] = id; segType.push(out.fitter ?? null); return id; },
    });
    const groups = segType.map((type) => ({ type, region: new Set() }));
    for (let t = 0; t < facets.count; t++) if (seg[t] >= 0) groups[seg[t]].region.add(t);
    const pool = groups.filter((g) => g.type && g.region.size)
      .map((g) => ({ ...fitRegionAs(g.type, g.region, facets), type: g.type, size: g.region.size }))
      .filter((f) => f.params).sort((a, b) => b.size - a.size);
    const used = new Set();
    let rec = 0;
    for (const f of truth) {
      for (let i = 0; i < pool.length; i++) {
        if (used.has(i) || !matches(f, pool[i])) continue;
        used.add(i); rec++; break;
      }
    }
    const row = { pair: pair.name, config: cname, tris: facets.count, faces: truth.length, analytic, groups: result.segments,
                  recovered: rec, s: (performance.now() - t0) / 1000 };
    rows.push(row);
    line.push(`${cname} ${String(result.segments).padStart(4)} groups ${String(rec).padStart(3)}/${analytic}`);
  }
  console.log(`${pair.name.slice(0, 34).padEnd(34)} ${String(rows.at(-1)?.tris ?? '').padStart(7)} tris  x${k === 1 ? '1' : k.toPrecision(3)}  |  ${line.join('  |  ')}`);
}
if (rows.length) {
  console.log('\ntotals over scored pairs (recovered out of analytic STEP faces):');
  for (const c of CONFIGS) {
    const rs = rows.filter((r) => r.config === c);
    const sum = (key) => rs.reduce((s, r) => s + r[key], 0);
    console.log(`  ${c.padEnd(8)} groups ${sum('groups')}  recovered ${sum('recovered')}/${sum('analytic')}  (${(sum('s')).toFixed(0)} s)`);
  }
}
if (JSON_OUT) writeFileSync(JSON_OUT, JSON.stringify(rows));
