/**
 * autoSeed.mjs — does the ORDER of seeds matter when a part is painted automatically?
 *
 * Usage:  node test/autoSeed.mjs <dir> <model.onnx> --strategy <name> [--meshers native,gmsh]
 *                                 [--parts N] [--json out.json]
 *         <dir> from ml/export_fill_eval.py.
 *
 * Each part is painted end to end the way the studio fills: pick a free seed, let the model
 * propose a region (with painted facets excluded from its patch), choose the type with the
 * model-capped ordered rule, grow with paintFrom (crease barrier, painted facets taken), claim
 * the result, repeat until every facet is painted or the fill budget runs out. Only the seed
 * ORDER differs between strategies:
 *
 *   random     uniformly random free facet (what clicking anywhere amounts to)
 *   largest    largest free facet first
 *   coherent   the facet whose crease-bounded 24-facet neighbourhood is best explained by a
 *              single primitive (lowest classifyRegion residual): the Gauss-map classifier used
 *              to choose where to start, not to label anything
 *   interior   farthest from any crease (geodesic over facet centroids)
 *   crease     nearest a crease first
 *
 * Scores per part: the share of true B-rep faces, and of surface area, recovered CLEAN (one
 * segment holds >= 95% of the face and is >= 95% that face's surface type, as in fillEval),
 * segments per true face (over-segmentation) and fills used.
 */

import { readFileSync, writeFileSync } from 'node:fs';
import { join, basename, dirname } from 'node:path';
import { createRequire } from 'node:module';

import { buildFacets, buildAdjacency } from '../js/meshCore.js';
import { paintFrom, creaseComponent } from '../js/regionGrow.js';
import { classifyRegion } from '../js/gaussMap.js';
import { attachCurvature } from '../js/curvature.js';
import { proposeRegion, TYPE_NAMES, fitterFor } from '../js/seedNet.js';
import { creaseRing, fitPrimitives } from '../js/seedFeatures.js';

const ort = createRequire(import.meta.url)('onnxruntime-node');

const args = process.argv.slice(2);
function flag(name, dflt) {
  const i = args.indexOf(name);
  if (i < 0) return dflt;
  const v = args[i + 1];
  args.splice(i, 2);
  return v;
}
const STRATEGY = flag('--strategy', 'random');
const MESHERS = flag('--meshers', null)?.split(',');
const PART_LIMIT = Number(flag('--parts', '0'));
const JSON_OUT = flag('--json', null);
const [DIR, MODEL] = args;
if (!DIR || !MODEL || !['random', 'largest', 'coherent', 'interior', 'crease'].includes(STRATEGY)) {
  console.error('usage: node test/autoSeed.mjs <dir> <model.onnx> --strategy random|largest|coherent|interior|crease');
  process.exit(2);
}

const CREASE = 30;
const ORDER_RANK = { plane: 0, cylinder: 1, cone: 2, sphere: 3, torus: 4 };

const session = await ort.InferenceSession.create(MODEL, { intraOpNumThreads: 2, interOpNumThreads: 1 });
const metaPath = MODEL.replace(/\.onnx$/, '.json');
let extended = false;
try { extended = JSON.parse(readFileSync(metaPath, 'utf8')).features === 'v6'; } catch { /* base model */ }
const run = async (features, n) => {
  const out = await session.run({
    features: new ort.Tensor('float32', features, [1, n, features.length / n]),
    mask: new ort.Tensor('float32', new Float32Array(n).fill(1), [1, n]),
  });
  return { sameLogit: out.same_logit.data, typeLogit: Array.from(out.type_logit.data) };
};

const copy = (b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.length);
const load = (stem) => ({
  P: Float64Array.from(new Float32Array(copy(readFileSync(join(DIR, `${stem}.pos`))))),
  F: new Int32Array(copy(readFileSync(join(DIR, `${stem}.fid`)))),
  T: new Uint8Array(copy(readFileSync(join(DIR, `${stem}.typ`)))),
});

let x = 31;
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

/** Geodesic distance over facet centroids from every facet touching a crease or an open edge. */
function creaseDistance(facets, adjacency) {
  const n = facets.count;
  const dist = new Float64Array(n).fill(Infinity);
  const heap = new Heap();
  for (let t = 0; t < n; t++) {
    if (facets.degenerate[t]) continue;
    const edges = adjacency.get(t) || [];
    if (edges.length < 3 || edges.some((e) => e.angle >= CREASE)) { dist[t] = 0; heap.push(0, t); }
  }
  const C = facets.centroids;
  while (heap.size) {
    const [k, t] = heap.pop();
    if (k > dist[t]) continue;
    for (const e of adjacency.get(t) || []) {
      const u = e.neighbor;
      const v = k + Math.hypot(C[t * 3] - C[u * 3], C[t * 3 + 1] - C[u * 3 + 1], C[t * 3 + 2] - C[u * 3 + 2]);
      if (v < dist[u]) { dist[u] = v; heap.push(v, u); }
    }
  }
  return dist;
}

const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/** Unsigned distance and |cos| of facet (q, nt) to a fit from seedFeatures.fitPrimitives. */
function distanceAndCos(kind, fit, q, nt) {
  if (kind === 'plane') {
    return [Math.abs(dot3([q[0] - fit.origin[0], q[1] - fit.origin[1], q[2] - fit.origin[2]], fit.normal)), Math.abs(dot3(nt, fit.normal))];
  }
  if (kind === 'sphere') {
    const w = [q[0] - fit.center[0], q[1] - fit.center[1], q[2] - fit.center[2]];
    const y = Math.hypot(...w);
    return y > 1e-30 ? [Math.abs(y - fit.radius), Math.abs(dot3(nt, w)) / y] : [Infinity, 0];
  }
  const base = kind === 'cylinder' ? fit.center : fit.apex;
  const v = [q[0] - base[0], q[1] - base[1], q[2] - base[2]];
  const al = dot3(v, fit.axis);
  const w = [v[0] - al * fit.axis[0], v[1] - al * fit.axis[1], v[2] - al * fit.axis[2]];
  const y = Math.hypot(...w);
  if (!(y > 1e-30)) return [Infinity, 0];
  if (kind === 'cylinder') return [Math.abs(y - fit.radius), Math.abs(dot3(nt, w)) / y];
  const sg = al < 0 ? -1 : 1;
  const nrm = [0, 1, 2].map((a) => (fit.cos * w[a]) / y - fit.sin * sg * fit.axis[a]);
  return [Math.abs(y * fit.cos - Math.abs(al) * fit.sin), Math.abs(dot3(nt, nrm))];
}

/** Facet order for the strategy: best seed first. */
function seedOrder(facets, adjacency) {
  const n = facets.count;
  const ids = [];
  for (let t = 0; t < n; t++) if (!facets.degenerate[t]) ids.push(t);
  const A = facets.areas;
  if (STRATEGY === 'random') {
    for (let i = ids.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [ids[i], ids[j]] = [ids[j], ids[i]]; }
    return ids;
  }
  if (STRATEGY === 'largest') return ids.sort((a, b) => A[b] - A[a]);
  if (STRATEGY === 'interior' || STRATEGY === 'crease') {
    const d = creaseDistance(facets, adjacency);
    const fin = (v) => (Number.isFinite(v) ? v : -1);
    return STRATEGY === 'interior'
      ? ids.sort((a, b) => fin(d[b]) - fin(d[a]) || A[b] - A[a])
      : ids.sort((a, b) => fin(d[a]) - fin(d[b]) || A[b] - A[a]);
  }
  // coherent: the closed-form fits of js/seedFeatures.js on the crease-bounded 24-facet ring (the
  // full classifyRegion per facet took over ten minutes for three parts). Score = the best
  // primitive's RMS centroid distance over the ring radius plus its mean normal misfit.
  const C = facets.centroids, Nr = facets.normals;
  const score = new Float64Array(n).fill(Infinity);
  for (const t of ids) {
    const ring = creaseRing(t, adjacency, 24, CREASE);
    if (ring.length < 8) continue;
    const fits = fitPrimitives(ring, facets);
    let m0 = 0, m1 = 0, m2 = 0;
    for (const u of ring) { m0 += C[u * 3]; m1 += C[u * 3 + 1]; m2 += C[u * 3 + 2]; }
    m0 /= ring.length; m1 /= ring.length; m2 /= ring.length;
    let ss = 0;
    for (const u of ring) ss += (C[u * 3] - m0) ** 2 + (C[u * 3 + 1] - m1) ** 2 + (C[u * 3 + 2] - m2) ** 2;
    const radius = Math.sqrt(ss / ring.length) || 1;
    let best = Infinity;
    for (const kind of ['plane', 'cylinder', 'cone', 'sphere']) {
      const fit = fits[kind];
      if (!fit) continue;
      let d2 = 0, mis = 0;
      for (const u of ring) {
        const q = [C[u * 3], C[u * 3 + 1], C[u * 3 + 2]];
        const nt = [Nr[u * 3], Nr[u * 3 + 1], Nr[u * 3 + 2]];
        const [d, cos] = distanceAndCos(kind, fit, q, nt);
        d2 += d * d;
        mis += 1 - Math.min(1, cos);
      }
      const s = Math.sqrt(d2 / ring.length) / radius + mis / ring.length;
      if (Number.isFinite(s) && s < best) best = s;
    }
    score[t] = best;
  }
  return ids.sort((a, b) => score[a] - score[b] || A[b] - A[a]);
}

function cappedType(seed, p, facets, adjacency, taken) {
  const region = new Set([...p.region].filter((t) => !taken[t]));
  const cls = classifyRegion(creaseComponent(seed, region, adjacency, CREASE), facets);
  const mk = fitterFor(p.type);
  let chosen = cls.type;
  if (mk) {
    const own = Math.min(Infinity, ...cls.candidates.filter((c) => c.type === mk).map((c) => c.relResidual));
    if ((chosen === 'nurbs' || ORDER_RANK[chosen] > ORDER_RANK[mk]) && own <= 0.05) chosen = mk;
  }
  return chosen === 'nurbs' ? null : chosen;
}

function scorePart(F, T, seg, facets) {
  const faces = new Map();   // face -> { tris, area, type }
  for (let t = 0; t < facets.count; t++) {
    if (facets.degenerate[t]) continue;
    const f = faces.get(F[t]) ?? { tris: [], area: 0, type: T[t] };
    f.tris.push(t); f.area += facets.areas[t];
    faces.set(F[t], f);
  }
  const segSize = new Map();
  const segTypeCount = new Map();   // seg -> Map(type -> count)
  for (let t = 0; t < facets.count; t++) {
    if (facets.degenerate[t] || seg[t] < 0) continue;
    segSize.set(seg[t], (segSize.get(seg[t]) ?? 0) + 1);
    const m = segTypeCount.get(seg[t]) ?? new Map();
    m.set(T[t], (m.get(T[t]) ?? 0) + 1);
    segTypeCount.set(seg[t], m);
  }
  let clean = 0, cleanArea = 0, total = 0, totalArea = 0;
  for (const f of faces.values()) {
    total++; totalArea += f.area;
    const overlap = new Map();
    for (const t of f.tris) if (seg[t] >= 0) overlap.set(seg[t], (overlap.get(seg[t]) ?? 0) + 1);
    let best = -1, bestN = 0;
    for (const [s, k] of overlap) if (k > bestN) { best = s; bestN = k; }
    if (best < 0) continue;
    const recall = bestN / f.tris.length;
    const purity = (segTypeCount.get(best).get(f.type) ?? 0) / segSize.get(best);
    if (recall >= 0.95 && purity >= 0.95) { clean++; cleanArea += f.area; }
  }
  return { faces: total, clean: clean / total, cleanArea: cleanArea / totalArea, segments: segSize.size };
}

const manifest = JSON.parse(readFileSync(join(DIR, 'manifest.json'), 'utf8'));
const meshers = MESHERS ?? manifest.variants;
const rows = [];
const t0 = performance.now();
const parts = PART_LIMIT ? manifest.parts.slice(0, PART_LIMIT) : manifest.parts;

for (const [pi, entry] of parts.entries()) {
  for (const mesher of meshers) {
    const v = entry.variants[mesher];
    if (!v) continue;
    const { P, F, T } = load(v.stem);
    const facets = buildFacets(P);
    const adjacency = buildAdjacency(P, facets);
    attachCurvature(P, facets);
    const zeroCurv = { valid: new Uint8Array(facets.count) };
    const ts = performance.now();
    const order = seedOrder(facets, adjacency);
    const orderMs = performance.now() - ts;

    const taken = new Uint8Array(facets.count);
    const seg = new Int32Array(facets.count).fill(-1);
    const nFaces = new Set(F).size;
    const budget = Math.max(300, 4 * nFaces);
    let fills = 0, cursor = 0, nextSeg = 0;
    while (fills < budget) {
      while (cursor < order.length && taken[order[cursor]]) cursor++;
      if (cursor >= order.length) break;
      const seed = order[cursor];
      const p = await proposeRegion(seed, facets, adjacency, run, { curvature: zeroCurv, excluded: taken, extended });
      const k = cappedType(seed, p, facets, adjacency, taken);
      const region = k
        ? paintFrom(seed, k, facets, adjacency, { seedRegion: p.region, taken }).region
        : creaseComponent(seed, new Set([...p.region].filter((t) => !taken[t])), adjacency, CREASE);
      region.add(seed);
      const id = nextSeg++;
      for (const t of region) if (!taken[t]) { taken[t] = 1; seg[t] = id; }
      fills++;
    }
    const s = scorePart(F, T, seg, facets);
    let painted = 0;
    for (let t = 0; t < facets.count; t++) if (taken[t]) painted++;
    rows.push({ part: pi, mesher, strategy: STRATEGY, ...s, fills, painted: painted / facets.count,
                ms: performance.now() - ts, orderMs });
  }
  if (pi % 10 === 9) console.log(`  ${pi + 1} parts, ${((performance.now() - t0) / 1000).toFixed(0)} s`);
}

const mean = (a) => a.reduce((s, v) => s + v, 0) / (a.length || 1);
console.log(`\n${STRATEGY}: ${parts.length} parts, model ${basename(dirname(MODEL))}${extended ? ' (v6 inputs)' : ''}, ${((performance.now() - t0) / 1000).toFixed(0)} s`);
for (const mesher of meshers) {
  const rs = rows.filter((r) => r.mesher === mesher);
  if (!rs.length) continue;
  console.log(`  ${mesher.padEnd(7)} faces clean ${(100 * mean(rs.map((r) => r.clean))).toFixed(1)}%` +
    `  area clean ${(100 * mean(rs.map((r) => r.cleanArea))).toFixed(1)}%` +
    `  segments/face ${mean(rs.map((r) => r.segments / r.faces)).toFixed(2)}` +
    `  fills ${mean(rs.map((r) => r.fills)).toFixed(0)}  painted ${(100 * mean(rs.map((r) => r.painted))).toFixed(0)}%` +
    `  ${(mean(rs.map((r) => r.ms)) / 1000).toFixed(1)} s/part`);
}
if (JSON_OUT) writeFileSync(JSON_OUT, JSON.stringify(rows));
