/**
 * tangentStats.mjs — can a curvature jump find the face boundaries a crease angle cannot?
 *
 * Usage:  node test/tangentStats.mjs <dir>        (<dir> from ml/export_fill_eval.py)
 *
 * The crease barrier (regionGrow.CREASE_BARRIER) stops growth at sharp edges, but about half the
 * neighbouring faces of different types meet TANGENTIALLY: a fillet running into a plane or a
 * cylinder, with no dihedral to see. What changes there is curvature: zero on the plane, 1/r on
 * the fillet. This measures two per-edge signals against the true faces:
 *
 *   curv   jump in total principal curvature |k1| + |k2| (js/curvature.js, per facet) between the
 *          two facets, times the distance between their centroids (dimensionless: roughly the
 *          extra turning, in radians, one step across the edge adds)
 *   turn   the same jump for each facet's mean normal turning rate over its edges (dihedral in
 *          radians / centroid distance), which needs no curvature estimate
 *
 * An edge is FLAGGED when the jump is at least `abs` AND at least `rel` of the larger of the two
 * values (so noise on two nearly flat facets never flags). For each signal and threshold:
 *   smooth boundary edges flagged   edges between different faces with dihedral < 30 degrees,
 *                                   split into different-type and same-type pairs
 *   smooth pairs sealed             neighbouring face pairs whose every shared edge is a crease
 *                                   (>= 30 degrees) or flagged
 *   within-face edges flagged       false barriers, per type
 *   recall ceiling                  the expected share of its own face a fill from a random facet
 *                                   can reach with creases (the per-type barrier) and flagged
 *                                   edges as walls; compare the crease-only row
 */

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { buildFacets, buildAdjacency } from '../js/meshCore.js';
import { attachCurvature } from '../js/curvature.js';
import { CREASE_BARRIER } from '../js/regionGrow.js';
import { TYPE_NAMES } from '../js/seedNet.js';

const DIR = process.argv[2];
if (!DIR) { console.error('usage: node test/tangentStats.mjs <dir>'); process.exit(2); }

const SIGNALS = ['curv', 'turn'];
const ABS = [0.02, 0.05, 0.1, 0.2];
const REL = [0.5, 0.8];
const CONFIGS = [{ signal: 'none' }];
for (const signal of SIGNALS) for (const abs of ABS) for (const rel of REL) CONFIGS.push({ signal, abs, rel });
const label = (c) => (c.signal === 'none' ? 'crease only' : `${c.signal} >=${c.abs} & >=${Math.round(c.rel * 100)}%`);

const copy = (b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.length);
const load = (stem) => ({
  P: Float64Array.from(new Float32Array(copy(readFileSync(join(DIR, `${stem}.pos`))))),
  F: new Int32Array(copy(readFileSync(join(DIR, `${stem}.fid`)))),
  T: new Uint8Array(copy(readFileSync(join(DIR, `${stem}.typ`)))),
});
const creaseDegFor = (typeName) => CREASE_BARRIER[typeName.toLowerCase()] ?? 30;

function find(parent, i) {
  while (parent[i] !== i) { parent[i] = parent[parent[i]]; i = parent[i]; }
  return i;
}

const manifest = JSON.parse(readFileSync(join(DIR, 'manifest.json'), 'utf8'));
const stats = {};   // mesher -> per config accumulators

for (const entry of manifest.parts) {
  for (const mesher of manifest.variants) {
    const v = entry.variants[mesher];
    if (!v) continue;
    const { P, F, T } = load(v.stem);
    const facets = buildFacets(P);
    const adjacency = buildAdjacency(P, facets);
    attachCurvature(P, facets);
    const curv = facets.curvature;
    const n = facets.count;
    const C = facets.centroids;

    const kappa = new Float64Array(n).fill(NaN);
    const turn = new Float64Array(n).fill(0);
    for (let t = 0; t < n; t++) {
      if (curv && curv.valid[t]) kappa[t] = Math.abs(curv.k1[t]) + Math.abs(curv.k2[t]);
      const edges = adjacency.get(t) || [];
      let s = 0;
      for (const e of edges) {
        const u = e.neighbor;
        const d = Math.hypot(C[t * 3] - C[u * 3], C[t * 3 + 1] - C[u * 3 + 1], C[t * 3 + 2] - C[u * 3 + 2]) || 1e-30;
        s += (e.angle * Math.PI / 180) / d;
      }
      turn[t] = edges.length ? s / edges.length : 0;
    }

    const edgeList = [];   // [a, b, angle, jumpCurv, relCurv, jumpTurn, relTurn]
    for (const [a, edges] of adjacency) {
      for (const e of edges) {
        const b = e.neighbor;
        if (b < a) continue;
        const d = Math.hypot(C[a * 3] - C[b * 3], C[a * 3 + 1] - C[b * 3 + 1], C[a * 3 + 2] - C[b * 3 + 2]);
        const jc = Number.isFinite(kappa[a]) && Number.isFinite(kappa[b]) ? Math.abs(kappa[a] - kappa[b]) : 0;
        const rc = jc / Math.max(kappa[a] || 0, kappa[b] || 0, 1e-30);
        const jt = Math.abs(turn[a] - turn[b]);
        const rt = jt / Math.max(turn[a], turn[b], 1e-30);
        edgeList.push([a, b, e.angle, jc * d, rc, jt * d, rt]);
      }
    }

    const s = (stats[mesher] ??= CONFIGS.map(() => ({
      boundDiff: [0, 0], boundSame: [0, 0], pairs: [0, 0], within: {}, recall: {},
    })));
    CONFIGS.forEach((cfg, ci) => {
      const acc = s[ci];
      const flagged = (row) => cfg.signal !== 'none' && (cfg.signal === 'curv'
        ? row[3] >= cfg.abs && row[4] >= cfg.rel
        : row[5] >= cfg.abs && row[6] >= cfg.rel);
      const parent = Int32Array.from({ length: n }, (_, i) => i);
      const pairs = new Map();
      for (const row of edgeList) {
        const [a, b, angle] = row;
        const flag = flagged(row);
        if (F[a] === F[b]) {
          const type = TYPE_NAMES[T[a]];
          const w = (acc.within[type] ??= [0, 0]);
          w[1]++;
          if (flag) w[0]++;
          if (angle < creaseDegFor(type) && !flag) {
            const ra = find(parent, a), rb = find(parent, b);
            if (ra !== rb) parent[ra] = rb;
          }
        } else {
          const key = F[a] < F[b] ? `${F[a]}|${F[b]}` : `${F[b]}|${F[a]}`;
          const p = pairs.get(key) ?? { sealed: true, smooth: false };
          if (angle < 30) {
            p.smooth = true;
            const bucket = T[a] === T[b] ? acc.boundSame : acc.boundDiff;
            bucket[1]++;
            if (flag) bucket[0]++;
            else p.sealed = false;
          }
          pairs.set(key, p);
        }
      }
      for (const p of pairs.values()) {
        if (!p.smooth) continue;
        acc.pairs[1]++;
        if (p.sealed) acc.pairs[0]++;
      }
      const comp = new Map();
      const size = new Map();
      const typeOf = new Map();
      for (let t = 0; t < n; t++) {
        if (facets.degenerate[t]) continue;
        size.set(F[t], (size.get(F[t]) ?? 0) + 1);
        typeOf.set(F[t], TYPE_NAMES[T[t]]);
        const r = find(parent, t);
        const m = comp.get(F[t]) ?? new Map();
        m.set(r, (m.get(r) ?? 0) + 1);
        comp.set(F[t], m);
      }
      for (const [face, m] of comp) {
        const total = size.get(face);
        if (total < 3) continue;
        let sq = 0;
        for (const c of m.values()) sq += c * c;
        (acc.recall[typeOf.get(face)] ??= []).push(sq / (total * total));
      }
    });
  }
}

const pct = (a, b) => (b ? `${Math.round((100 * a) / b)}%` : '-').padStart(4);
const mean = (arr) => (arr && arr.length ? arr.reduce((x, y) => x + y, 0) / arr.length : NaN);
const TYPES = ['Plane', 'Cylinder', 'Cone', 'Sphere', 'Torus', 'BSpline'];
console.log(`${manifest.parts.length} parts. Boundary columns: smooth (<30 deg) edges between different faces flagged, different / same type; pairs sealed.`);
console.log('Within-face columns: false-barrier edges, then recall ceiling per type (crease + flag as walls).\n');
for (const mesher of manifest.variants) {
  const s = stats[mesher];
  if (!s) continue;
  console.log(`== ${mesher}`);
  console.log(`  ${'config'.padEnd(20)} bnd diff  bnd same  sealed  | false barriers ${TYPES.map((t) => t.slice(0, 5).padStart(6)).join('')} | recall ${TYPES.map((t) => t.slice(0, 5).padStart(6)).join('')}`);
  CONFIGS.forEach((cfg, ci) => {
    const a = s[ci];
    const fb = TYPES.map((t) => pct(...(a.within[t] ?? [0, 0])).padStart(6)).join('');
    const rc = TYPES.map((t) => `${Math.round(100 * mean(a.recall[t]))}%`.padStart(6)).join('');
    console.log(`  ${label(cfg).padEnd(20)} ${pct(...a.boundDiff).padStart(8)}  ${pct(...a.boundSame).padStart(8)}  ${pct(...a.pairs).padStart(6)}  |                ${fb} |        ${rc}`);
  });
  console.log('');
}
