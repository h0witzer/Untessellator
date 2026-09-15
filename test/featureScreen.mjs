/**
 * featureScreen.mjs — candidate per-facet inputs for the seed-fill model, computed on the same
 * 512-facet patches the model sees, and dumped for ml/feature_screen.py to probe.
 *
 * Usage:  node test/featureScreen.mjs <dir> <out prefix>
 *         <dir> from ml/export_fill_eval.py; writes <prefix>.feat (float32 rows),
 *         <prefix>.meta (int32: seed, mesher, part, seed type, same face) and <prefix>.json.
 *
 * Retraining costs a night per idea, so this screens ideas first: small probe networks trained on
 * half the parts, scored on the other half and on every mesher, with and without each group.
 * A useful input must help AND transfer across meshers (the type head's failure so far is that
 * it reads the mesher's pattern: see docs/ml-experiment-scope.md).
 *
 * Column groups (the first two are the model's current inputs):
 *   base      dx dy dz nx ny nz dot_seed log_area ring dist
 *   curv      k1s k2s                     (trained out of s1/s3, with --no-curvature)
 *   crease    bottleneck                  largest dihedral on the gentlest path from the seed
 *             local_crease                largest dihedral on the facet's own edges
 *   geodesic  geo                         shortest centroid path from the seed / patch scale
 *   normals   nev0 nev1 nev2              eigenvalues of the covariance of normals within
 *                                         0.35 patch radii: plane ~0/0/0, cylinder 0/0/+,
 *                                         sphere 0/+/+ (a metric radius, not a ring count)
 *   seedfit   {plane,cyl,cone,sphere}_{d,a}  distance / scale and 1-|cos| to each primitive
 *                                         fitted to a crease-bounded 32-facet patch at the seed
 *   shape     quality edge_ratio          triangle shape: a mesher-pattern CONTROL, expected to
 *                                         help on the mesher it trains on and not transfer
 */

import { readFileSync, openSync, writeSync, closeSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { buildFacets, buildAdjacency } from '../js/meshCore.js';
import { surfaceEval } from '../js/regionGrow.js';
import { fitRegionAs } from '../js/gaussMap.js';
import { attachCurvature } from '../js/curvature.js';
import { buildFeatures, TYPE_NAMES } from '../js/seedNet.js';

const [DIR, OUT] = process.argv.slice(2);
if (!DIR || !OUT) { console.error('usage: node test/featureScreen.mjs <dir> <out prefix>'); process.exit(2); }

const GROUPS = {
  base: ['dx', 'dy', 'dz', 'nx', 'ny', 'nz', 'dot_seed', 'log_area', 'ring', 'dist'],
  curv: ['k1s', 'k2s'],
  crease: ['bottleneck', 'local_crease'],
  geodesic: ['geo'],
  normals: ['nev0', 'nev1', 'nev2'],
  seedfit: ['plane_d', 'plane_a', 'cyl_d', 'cyl_a', 'cone_d', 'cone_a', 'sphere_d', 'sphere_a'],
  shape: ['quality', 'edge_ratio'],
};
const COLUMNS = Object.values(GROUPS).flat();
const NCOL = COLUMNS.length;
const SEEDS_PER_TYPE = 2;
const MAX_SEEDS_PER_PART = 10;
const FIT_TYPES = ['plane', 'cylinder', 'cone', 'sphere'];
const NORMAL_RADIUS = 0.35;

const copy = (b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.length);
const load = (stem) => ({
  P: Float64Array.from(new Float32Array(copy(readFileSync(join(DIR, `${stem}.pos`))))),
  F: new Int32Array(copy(readFileSync(join(DIR, `${stem}.fid`)))),
  T: new Uint8Array(copy(readFileSync(join(DIR, `${stem}.typ`)))),
});

let x = 7;
const rng = () => (x = (x * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;

/** Minimal binary heap of [key, value] pairs, smallest key first. */
class Heap {
  constructor() { this.a = []; }
  get size() { return this.a.length; }
  push(k, v) {
    const a = this.a;
    a.push([k, v]);
    let i = a.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (a[p][0] <= a[i][0]) break;
      [a[p], a[i]] = [a[i], a[p]];
      i = p;
    }
  }
  pop() {
    const a = this.a;
    const top = a[0];
    const last = a.pop();
    if (a.length) {
      a[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1;
        let m = i;
        if (l < a.length && a[l][0] < a[m][0]) m = l;
        if (r < a.length && a[r][0] < a[m][0]) m = r;
        if (m === i) break;
        [a[m], a[i]] = [a[i], a[m]];
        i = m;
      }
    }
    return top;
  }
}

/** Dijkstra over the patch from index 0 (the seed), combining path cost with `step`. */
function patchPaths(n, nbrs, step) {
  const best = new Float64Array(n).fill(Infinity);
  best[0] = 0;
  const heap = new Heap();
  heap.push(0, 0);
  while (heap.size) {
    const [k, i] = heap.pop();
    if (k > best[i]) continue;
    for (const [j, w] of nbrs[i]) {
      const v = step(k, w, i, j);
      if (v < best[j]) { best[j] = v; heap.push(v, j); }
    }
  }
  return best;
}

/** Eigenvalues of a symmetric 3x3 matrix, ascending (trigonometric method). */
function eig3sym(a00, a01, a02, a11, a12, a22) {
  const p1 = a01 * a01 + a02 * a02 + a12 * a12;
  if (p1 < 1e-30) return [a00, a11, a22].sort((u, v) => u - v);
  const q = (a00 + a11 + a22) / 3;
  const p = Math.sqrt(((a00 - q) ** 2 + (a11 - q) ** 2 + (a22 - q) ** 2 + 2 * p1) / 6);
  const b00 = (a00 - q) / p, b11 = (a11 - q) / p, b22 = (a22 - q) / p;
  const b01 = a01 / p, b02 = a02 / p, b12 = a12 / p;
  const r = (b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b12 * b02) + b02 * (b01 * b12 - b11 * b02)) / 2;
  const phi = r <= -1 ? Math.PI / 3 : r >= 1 ? 0 : Math.acos(r) / 3;
  const e1 = q + 2 * p * Math.cos(phi);
  const e3 = q + 2 * p * Math.cos(phi + (2 * Math.PI) / 3);
  return [e3, 3 * q - e1 - e3, e1];
}

function creasePatch(seed, adjacency, want, deg = 30) {
  const out = new Set([seed]);
  const queue = [seed];
  while (queue.length && out.size < want) {
    const t = queue.shift();
    for (const e of adjacency.get(t) || []) {
      if (out.has(e.neighbor) || e.angle >= deg) continue;
      out.add(e.neighbor);
      queue.push(e.neighbor);
      if (out.size >= want) break;
    }
  }
  return out;
}

const manifest = JSON.parse(readFileSync(join(DIR, 'manifest.json'), 'utf8'));
const reference = manifest.variants[0];
const featFd = openSync(`${OUT}.feat`, 'w');
const metaFd = openSync(`${OUT}.meta`, 'w');
let rows = 0;
let seedId = 0;
const t0 = performance.now();

manifest.parts.forEach((entry, partIdx) => {
  const ref = entry.variants[reference];
  if (!ref) return;
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

  manifest.variants.forEach((mesher, mesherIdx) => {
    const v = entry.variants[mesher];
    if (!v) return;
    const { P, F, T } = mesher === reference ? base : load(v.stem);
    const facets = buildFacets(P);
    const adjacency = buildAdjacency(P, facets);
    attachCurvature(P, facets);
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
      const { features, ids, n, scale } = buildFeatures(seed, facets, adjacency, 512);
      const index = new Map(ids.map((t, i) => [t, i]));
      const nbrs = ids.map((t) => (adjacency.get(t) || [])
        .filter((e) => index.has(e.neighbor)).map((e) => [index.get(e.neighbor), e.angle]));
      const bottleneck = patchPaths(n, nbrs, (k, w) => Math.max(k, w));
      const cx = (i) => facets.centroids[ids[i] * 3], cy = (i) => facets.centroids[ids[i] * 3 + 1], cz = (i) => facets.centroids[ids[i] * 3 + 2];
      const geo = patchPaths(n, nbrs, (k, w, i, j) => k + Math.hypot(cx(i) - cx(j), cy(i) - cy(j), cz(i) - cz(j)));

      const fits = FIT_TYPES.map((k) => {
        const f = fitRegionAs(k, creasePatch(seed, adjacency, 32), facets);
        return f.params ? surfaceEval(k, f.params) : null;
      });

      const block = new Float32Array(n * NCOL);
      const meta = new Int32Array(n * 5);
      const r2 = NORMAL_RADIUS * NORMAL_RADIUS;
      for (let i = 0; i < n; i++) {
        const t = ids[i];
        const o = i * NCOL;
        for (let c = 0; c < 12; c++) block[o + c] = features[i * 12 + c];
        let c = 12;
        block[o + c++] = Math.min(180, bottleneck[i]) / 90;
        let local = 0;
        for (const e of adjacency.get(t) || []) local = Math.max(local, e.angle);
        block[o + c++] = local / 90;
        block[o + c++] = Math.min(20, geo[i] / scale);

        // Normal covariance within a metric radius, in the canonical frame.
        let w = 0, m0 = 0, m1 = 0, m2 = 0, s00 = 0, s01 = 0, s02 = 0, s11 = 0, s12 = 0, s22 = 0;
        const px = features[i * 12], py = features[i * 12 + 1], pz = features[i * 12 + 2];
        for (let j = 0; j < n; j++) {
          const dx = features[j * 12] - px, dy = features[j * 12 + 1] - py, dz = features[j * 12 + 2] - pz;
          if (dx * dx + dy * dy + dz * dz > r2) continue;
          const a = features[j * 12 + 3], b = features[j * 12 + 4], g = features[j * 12 + 5];
          w++; m0 += a; m1 += b; m2 += g;
          s00 += a * a; s01 += a * b; s02 += a * g; s11 += b * b; s12 += b * g; s22 += g * g;
        }
        m0 /= w; m1 /= w; m2 /= w;
        const ev = eig3sym(s00 / w - m0 * m0, s01 / w - m0 * m1, s02 / w - m0 * m2, s11 / w - m1 * m1, s12 / w - m1 * m2, s22 / w - m2 * m2);
        block[o + c++] = Math.max(0, ev[0]);
        block[o + c++] = Math.max(0, ev[1]);
        block[o + c++] = Math.max(0, ev[2]);

        const q = [facets.centroids[t * 3], facets.centroids[t * 3 + 1], facets.centroids[t * 3 + 2]];
        const nt = [facets.normals[t * 3], facets.normals[t * 3 + 1], facets.normals[t * 3 + 2]];
        for (const fit of fits) {
          if (!fit) { block[o + c++] = 10; block[o + c++] = 1; continue; }
          const d = Math.abs(fit.distance(q)) / scale;
          const sn = fit.normal(q);
          const cos = Math.abs(sn[0] * nt[0] + sn[1] * nt[1] + sn[2] * nt[2]);
          block[o + c++] = Number.isFinite(d) ? Math.min(10, d) : 10;
          block[o + c++] = Number.isFinite(cos) ? 1 - Math.min(1, cos) : 1;
        }

        const po = t * 9;
        const L = [0, 1, 2].map((k) => {
          const a = po + k * 3, b = po + ((k + 1) % 3) * 3;
          return Math.hypot(P[a] - P[b], P[a + 1] - P[b + 1], P[a + 2] - P[b + 2]);
        });
        const sumSq = L[0] ** 2 + L[1] ** 2 + L[2] ** 2;
        block[o + c++] = sumSq > 0 ? (4 * Math.sqrt(3) * facets.areas[t]) / sumSq : 0;
        block[o + c++] = Math.min(10, Math.log(Math.max(...L) / Math.max(1e-30, Math.min(...L))));

        meta.set([seedId, mesherIdx, partIdx, T[seed], F[t] === F[seed] ? 1 : 0], i * 5);
      }
      writeSync(featFd, new Uint8Array(block.buffer));
      writeSync(metaFd, new Uint8Array(meta.buffer));
      rows += n;
      seedId++;
    }
  });
  if (partIdx % 10 === 9) console.log(`  ${partIdx + 1} parts, ${seedId} seeds, ${rows} rows, ${((performance.now() - t0) / 1000).toFixed(0)} s`);
});

closeSync(featFd);
closeSync(metaFd);
writeFileSync(`${OUT}.json`, JSON.stringify({
  ncol: NCOL, columns: COLUMNS, groups: GROUPS, meshers: manifest.variants, typeNames: TYPE_NAMES, rows, seeds: seedId,
}, null, 1));
console.log(`wrote ${rows} rows from ${seedId} seeds to ${OUT}.*`);
