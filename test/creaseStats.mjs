/**
 * creaseStats.mjs — how well a dihedral-angle barrier separates B-rep faces, per surface type.
 *
 * Usage:  node test/creaseStats.mjs <dir>        (<dir> from ml/export_fill_eval.py)
 *
 * The studio draws an edge wherever neighbouring triangles meet at 30 degrees or more, and on
 * CAD meshes those lines fall on face boundaries. If a fill could never cross such an edge,
 * a drifting refit could not carry it into the neighbour. Two questions decide the threshold:
 *
 *   RECALL CEILING  With edges >= theta as walls, how much of its own face can a fill from a
 *                   random triangle reach? Per face this is sum(component^2) / size^2 over the
 *                   face's crease-bounded components. A curved face tessellated coarsely has
 *                   large dihedrals INSIDE it, so theta must sit above those for its type.
 *   SEALING         Of the edges between two different faces, how many are >= theta, and how
 *                   many neighbouring face pairs have EVERY shared edge >= theta (a pair with
 *                   one gap is not sealed)? Split into different-type pairs and same-type pairs,
 *                   because ABC splits one surface into several faces (coplanar splits, seams),
 *                   and those boundaries are flat by construction.
 */

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { buildFacets, buildAdjacency } from '../js/meshCore.js';
import { TYPE_NAMES } from '../js/seedNet.js';

const DIR = process.argv[2];
if (!DIR) { console.error('usage: node test/creaseStats.mjs <dir>'); process.exit(2); }
const THETAS = [5, 10, 15, 20, 30, 45, 60];

const copy = (b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.length);
const load = (stem) => ({
  P: Float64Array.from(new Float32Array(copy(readFileSync(join(DIR, `${stem}.pos`))))),
  F: new Int32Array(copy(readFileSync(join(DIR, `${stem}.fid`)))),
  T: new Uint8Array(copy(readFileSync(join(DIR, `${stem}.typ`)))),
});

const manifest = JSON.parse(readFileSync(join(DIR, 'manifest.json'), 'utf8'));
// stats[mesher] = { within: {type: {angles:[], recall: {theta: [..]}}}, boundary: {diff|same: {...}} }
const stats = {};

function find(parent, i) {
  while (parent[i] !== i) { parent[i] = parent[parent[i]]; i = parent[i]; }
  return i;
}

for (const entry of manifest.parts) {
  for (const mesher of manifest.variants) {
    const v = entry.variants[mesher];
    if (!v) continue;
    const { P, F, T } = load(v.stem);
    const facets = buildFacets(P);
    const adjacency = buildAdjacency(P, facets);
    const s = (stats[mesher] ??= { within: {}, boundary: { diff: newBoundary(), same: newBoundary() } });

    const faceSize = new Map();
    for (let t = 0; t < facets.count; t++) {
      if (facets.degenerate[t]) continue;
      faceSize.set(F[t], (faceSize.get(F[t]) ?? 0) + 1);
    }
    const parents = THETAS.map(() => Int32Array.from({ length: facets.count }, (_, i) => i));
    const pairs = new Map();   // "fa|fb" -> { kind, minAngle, n }

    for (const [a, edges] of adjacency) {
      for (const { neighbor: b, angle } of edges) {
        if (b < a) continue;
        if (F[a] === F[b]) {
          const w = (s.within[TYPE_NAMES[T[a]]] ??= { angles: [], recall: THETAS.map(() => []), faces: 0 });
          w.angles.push(angle);
          THETAS.forEach((theta, i) => {
            if (angle < theta) {
              const ra = find(parents[i], a), rb = find(parents[i], b);
              if (ra !== rb) parents[i][ra] = rb;
            }
          });
        } else {
          const kind = T[a] === T[b] ? 'same' : 'diff';
          const bd = s.boundary[kind];
          bd.edges++;
          THETAS.forEach((theta, i) => { if (angle >= theta) bd.edgesAbove[i]++; });
          const key = F[a] < F[b] ? `${F[a]}|${F[b]}` : `${F[b]}|${F[a]}`;
          const p = pairs.get(key);
          if (p) p.min = Math.min(p.min, angle);
          else pairs.set(key, { kind, min: angle });
        }
      }
    }
    for (const { kind, min } of pairs.values()) {
      const bd = s.boundary[kind];
      bd.pairs++;
      THETAS.forEach((theta, i) => { if (min >= theta) bd.pairsSealed[i]++; });
    }

    // Expected recall ceiling per face and theta.
    const faceType = new Map();
    for (let t = 0; t < facets.count; t++) if (!facets.degenerate[t]) faceType.set(F[t], TYPE_NAMES[T[t]]);
    THETAS.forEach((theta, i) => {
      const comp = new Map();   // face -> Map(root -> count)
      for (let t = 0; t < facets.count; t++) {
        if (facets.degenerate[t]) continue;
        const r = find(parents[i], t);
        const m = comp.get(F[t]) ?? new Map();
        m.set(r, (m.get(r) ?? 0) + 1);
        comp.set(F[t], m);
      }
      for (const [face, m] of comp) {
        const n = faceSize.get(face);
        if (n < 3) continue;
        let sq = 0;
        for (const c of m.values()) sq += c * c;
        const w = s.within[faceType.get(face)];
        if (w) w.recall[i].push(sq / (n * n));
      }
    });
  }
}

function newBoundary() {
  return { edges: 0, edgesAbove: THETAS.map(() => 0), pairs: 0, pairsSealed: THETAS.map(() => 0) };
}

const pct = (v) => `${(100 * v).toFixed(0)}%`.padStart(5);
const quant = (arr, q) => {
  if (!arr.length) return NaN;
  const s = Float64Array.from(arr).sort();
  return s[Math.min(s.length - 1, Math.floor(q * s.length))];
};
const mean = (arr) => arr.reduce((a, b) => a + b, 0) / (arr.length || 1);

console.log(`${manifest.parts.length} parts; thresholds ${THETAS.join(', ')} deg\n`);
for (const mesher of manifest.variants) {
  const s = stats[mesher];
  if (!s) continue;
  console.log(`== ${mesher}`);
  console.log('  within-face dihedral (deg)           p50    p90    p99   max  | expected recall ceiling at theta (faces >= 95% in brackets)');
  for (const type of TYPE_NAMES) {
    const w = s.within[type];
    if (!w || !w.angles.length) continue;
    const cells = THETAS.map((theta, i) => `${theta}:${pct(mean(w.recall[i]))}[${pct(w.recall[i].filter((r) => r >= 0.95).length / (w.recall[i].length || 1))}]`);
    console.log(`  ${type.padEnd(10)} ${String(w.recall[0].length).padStart(5)} faces ${quant(w.angles, 0.5).toFixed(1).padStart(6)} ${quant(w.angles, 0.9).toFixed(1).padStart(6)} ${quant(w.angles, 0.99).toFixed(1).padStart(6)} ${quant(w.angles, 1).toFixed(0).padStart(5)}  | ${cells.join(' ')}`);
  }
  for (const kind of ['diff', 'same']) {
    const bd = s.boundary[kind];
    console.log(`  boundary, ${kind === 'diff' ? 'different types' : 'same type     '}: edges >= theta ${THETAS.map((t, i) => `${t}:${pct(bd.edgesAbove[i] / bd.edges)}`).join(' ')}`);
    console.log(`  ${' '.repeat(33)}pairs sealed   ${THETAS.map((t, i) => `${t}:${pct(bd.pairsSealed[i] / bd.pairs)}`).join(' ')}   (${bd.pairs} pairs)`);
  }
  console.log('');
}
