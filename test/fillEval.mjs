/**
 * fillEval.mjs — end-to-end bucket fill, one simulated click at a time, through the JS the
 * app would ship.
 *
 * Usage:  node test/fillEval.mjs <dir> <model.onnx>[,<model.onnx>...] [--json out.json]
 *                                [--curvature off|on]
 *         <dir> is written by ml/export_fill_eval.py. A model is named by its run
 *         directory, so pass D:/abc-data/runs/<run>/seedfill.onnx.
 *
 * WHY THIS EXISTS
 *
 * rescore.py scores the model's 512-facet patch. The user sees something else: that region
 * grown and refit by paintFrom, or the raw region when the predicted type has no fitter.
 * Only that decides whether a fill stops at the right boundary. The first run (v2ctrl, 80
 * validation parts) found a clean fill on 38% of clicks on OCCT meshes and 21% on gmsh.
 * Declaring the true type instead of trusting the model raised OCCT to 45%; no model at
 * all gave 12%. So the model's value is the seed region more than the type, and the growth
 * step causes most of the spill.
 *
 * One click is one seed triangle on a known B-rep face, and the same faces are clicked on
 * every mesher. Methods:
 *   auto            the model's type and region; paintFrom grows it when the type has a fitter
 *   declared+model  the TRUE type, seeded with the model's region
 *   declared        the TRUE type with paintFrom's own BFS seed: no model, computed once
 *   model only      the model's raw region, no growth
 *
 * PURITY COMES IN TWO KINDS, because ABC splits one geometric surface into several faces
 * (coplanar splits, periodic seams):
 *   strict   facets on the seed's own face
 *   surface  also counting faces of the same canonical type. After fitted growth that is
 *            almost always the same surface, because growOn admits only facets within
 *            tolerance of the fitted plane, cylinder or sphere. "model only" and types with
 *            no fitter carry no such guarantee.
 * clean = surface purity >= 95% and recall >= 95%. spill = surface purity < 90%.
 * incomplete = recall < 90%.
 *
 * CURVATURE: --curvature off (default) zeroes the model's curvature columns while the
 * fitters keep real curvature, as every --no-curvature run was trained. Exports from
 * ml/export_onnx.py also zero trained-out columns inside the graph.
 */

import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { join, basename, dirname } from 'node:path';
import { createRequire } from 'node:module';

import { buildFacets, buildAdjacency } from '../js/meshCore.js';
import { paintFrom, creaseComponent } from '../js/regionGrow.js';
import { classifyRegion } from '../js/gaussMap.js';
import { proposeRegion, TYPE_NAMES, fitterFor, N_FEATURES } from '../js/seedNet.js';
import { attachCurvature } from '../js/curvature.js';

const ort = createRequire(import.meta.url)('onnxruntime-node');

const args = process.argv.slice(2);
function flag(name, dflt) {
  const i = args.indexOf(name);
  if (i < 0) return dflt;
  const v = args[i + 1];
  args.splice(i, 2);
  return v;
}
const JSON_OUT = flag('--json', null);
const CURVATURE = flag('--curvature', 'off');
// Crease barriers for the declared (no model) fills: edges at or above the angle are walls.
// Several comma-separated presets run side by side on the same clicks.
const CREASE_PRESETS = {
  none: null,
  all30: 30,
  typedA: { plane: 5, cylinder: 30, cone: 30, sphere: 30, torus: 45 },
  typedB: { plane: 10, cylinder: 30, cone: 45, sphere: 30, torus: 45 },
  typedC: { plane: 5, cylinder: 45, cone: 45, sphere: 45, torus: 60 },
  typedD: { plane: 5, cylinder: 20, cone: 20, sphere: 20, torus: 45 },
  typedE: { plane: 5, cylinder: 15, cone: 15, sphere: 20, torus: 30 },
  typedF: { plane: 2, cylinder: 30, cone: 30, sphere: 30, torus: 45 },
};
const CREASES = flag('--crease', 'none').split(',');
for (const c of CREASES) if (!(c in CREASE_PRESETS)) { console.error(`unknown crease preset ${c}`); process.exit(2); }
const DECLARED_ONLY = args.includes('--declared-only');
if (DECLARED_ONLY) args.splice(args.indexOf('--declared-only'), 1);
// ORDERED TYPE: the geometry picks the type instead of the model's type head. classifyRegion fits
// every primitive to a crease-bounded patch and accepts the SIMPLEST whose relative residual is
// under 1% (plane < cylinder < cone < sphere < torus), since a cone can always imitate a cylinder.
//   ordered <n>          no model: an n-facet crease-bounded BFS patch around the click
//   <model> ordered type the model's region, clipped to the click's crease component
const ORDERED = args.includes('--ordered');
if (ORDERED) args.splice(args.indexOf('--ordered'), 1);
const ORDERED_SIZES = [32, 128];
const ORDER_CREASE = 30;

const ORDER_RANK = { plane: 0, cylinder: 1, cone: 2, sphere: 3, torus: 4 };

// SEQUENTIAL PAINTING: does it help to paint simpler surfaces first and keep them out of later
// fills? Only seeds on non-plane faces are clicked, with ground-truth faces already "painted":
//   planes    every Plane face (other than the seed's own)
//   simpler   every face of a simpler type (plane < cylinder < cone < sphere < torus)
// and three ways of using that, all with the model-capped ordered type the studio ships:
//   base      fill as now, painted facets dropped afterwards (what the studio already does)
//   grow      painted facets excluded from growth (paintFrom opts.taken)
//   all       also dropped from the model's 512-facet patch (kRing walks around them)
// The painting is an oracle: real earlier fills would be imperfect.
const SEQUENTIAL = args.includes('--sequential');
if (SEQUENTIAL) args.splice(args.indexOf('--sequential'), 1);
const SEQ_RANK = { Plane: 0, Cylinder: 1, Cone: 2, Sphere: 3, Torus: 4 };
const without = (set, mask) => new Set([...set].filter(t => !mask[t]));

/** The studio's auto type: simplest fitting primitive, capped by the model's own type. */
function cappedType(seed, p, facets, adjacency) {
  const cls = classifyRegion(creaseComponent(seed, p.region, adjacency, ORDER_CREASE), facets);
  const mk = fitterFor(p.type);
  let chosen = cls.type;
  if (mk) {
    const own = Math.min(Infinity, ...cls.candidates.filter(c => c.type === mk).map(c => c.relResidual));
    if ((chosen === 'nurbs' || ORDER_RANK[chosen] > ORDER_RANK[mk]) && own <= 0.05) chosen = mk;
  }
  return chosen === 'nurbs' ? null : chosen;
}
function creasePatch(seed, adjacency, want, allowed = null) {
  const out = new Set([seed]);
  const queue = [seed];
  while (queue.length && out.size < want) {
    const t = queue.shift();
    for (const e of adjacency.get(t) || []) {
      if (out.has(e.neighbor) || e.angle >= ORDER_CREASE || (allowed && !allowed.has(e.neighbor))) continue;
      out.add(e.neighbor);
      queue.push(e.neighbor);
      if (out.size >= want) break;
    }
  }
  return out;
}
const TYPE_BY_FITTER = { plane: 'Plane', cylinder: 'Cylinder', cone: 'Cone', sphere: 'Sphere', torus: 'Torus', nurbs: 'BSpline' };
const orderedTypeOf = (region, facets) => TYPE_BY_FITTER[classifyRegion(region, facets).type] ?? 'BSpline';
const declaredMethod = c => (c === 'none' ? 'declared' : `declared crease=${c}`);

const [DIR, MODEL_LIST] = args;
if (!DIR || (!MODEL_LIST && !DECLARED_ONLY)) {
  console.error('usage: node test/fillEval.mjs <dir> <model.onnx>[,...] [--json out.json] [--curvature off|on]\n' +
    '                                [--crease none,all30,typedA,...] [--declared-only]');
  process.exit(2);
}
const SEEDS_PER_TYPE = 2;
const MAX_SEEDS_PER_PART = 10;

const models = [];
for (const path of DECLARED_ONLY ? [] : MODEL_LIST.split(',')) {
  const session = await ort.InferenceSession.create(path);
  // ml/export_onnx.py writes a sidecar naming the input set; v6 models take the extended columns.
  const metaPath = path.replace(/\.onnx$/, '.json');
  const meta = existsSync(metaPath) ? JSON.parse(readFileSync(metaPath, 'utf8')) : {};
  models.push({
    name: basename(dirname(path)),
    extended: meta.features === 'v6',
    run: async (features, n) => {
      const out = await session.run({
        features: new ort.Tensor('float32', features, [1, n, features.length / n]),
        mask: new ort.Tensor('float32', new Float32Array(n).fill(1), [1, n]),
      });
      return { sameLogit: out.same_logit.data, typeLogit: Array.from(out.type_logit.data) };
    },
  });
}

const copy = b => b.buffer.slice(b.byteOffset, b.byteOffset + b.length);
function load(stem) {
  return {
    P: Float64Array.from(new Float32Array(copy(readFileSync(join(DIR, stem + '.pos'))))),
    F: new Int32Array(copy(readFileSync(join(DIR, stem + '.fid')))),
    T: new Uint8Array(copy(readFileSync(join(DIR, stem + '.typ')))),
  };
}

let x = 99;
const rng = () => (x = (x * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
const rows = [];

function score(method, model, mesher, seed, T, predType, region, truth, extra = {}) {
  let hit = 0, sameType = 0;
  for (const t of region) {
    if (truth.has(t)) hit++;
    else if (T[t] === T[seed]) sameType++;
  }
  const n = region.size || 1;
  rows.push({ method, model, mesher, truthType: TYPE_NAMES[T[seed]], predType, faceSize: truth.size,
              purity: hit / n, surfacePurity: (hit + sameType) / n, recall: hit / truth.size, ...extra });
}

const manifest = JSON.parse(readFileSync(join(DIR, 'manifest.json'), 'utf8'));
const reference = manifest.variants[0];
const t0 = performance.now();

for (const entry of manifest.parts) {
  const ref = entry.variants[reference];
  if (!ref) continue;
  // seed FACES are picked once, on the reference mesh, so every mesher clicks the same faces
  const base = load(ref.stem);
  const facesByType = new Map();
  for (let t = 0; t < base.F.length; t++) {
    if (!facesByType.has(base.T[t])) facesByType.set(base.T[t], new Set());
    facesByType.get(base.T[t]).add(base.F[t]);
  }
  const seedFaces = [];
  for (const faces of facesByType.values()) {
    const arr = [...faces];
    for (let i = 0; i < SEEDS_PER_TYPE && arr.length; i++) {
      seedFaces.push(arr.splice(Math.floor(rng() * arr.length), 1)[0]);
    }
  }
  seedFaces.splice(MAX_SEEDS_PER_PART);

  for (const mesher of manifest.variants) {
    const v = entry.variants[mesher];
    if (!v) continue;
    const { P, F, T } = mesher === reference ? base : load(v.stem);
    const facets = buildFacets(P);
    const adjacency = buildAdjacency(P, facets);
    attachCurvature(P, facets);
    const proposeOpts = CURVATURE === 'on' ? {} : { curvature: { valid: new Uint8Array(facets.count) } };
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
      const truth = new Set(tris);
      const truthType = TYPE_NAMES[T[seed]];
      const tf = fitterFor(truthType);

      if (SEQUENTIAL) {
        if (!tf || truthType === 'Plane') continue;
        const rank = SEQ_RANK[truthType];
        const masks = {};
        for (const [name, keep] of [['planes', r => r === 0], ['simpler', r => r < rank]]) {
          const m = new Uint8Array(facets.count);
          for (let t = 0; t < facets.count; t++) {
            const r = SEQ_RANK[TYPE_NAMES[T[t]]];
            if (F[t] !== face && r !== undefined && keep(r)) m[t] = 1;
          }
          masks[name] = m;
        }
        const fillWith = (k, p, taken) => (k
          ? paintFrom(seed, k, facets, adjacency, { seedRegion: p.region, ...(taken ? { taken } : {}) }).region
          : creaseComponent(seed, taken ? without(p.region, taken) : p.region, adjacency, ORDER_CREASE));
        for (const model of models) {
          const p0 = await proposeRegion(seed, facets, adjacency, model.run, { ...proposeOpts, extended: model.extended });
          const k0 = cappedType(seed, p0, facets, adjacency);
          const g0 = fillWith(k0, p0, null);
          for (const [name, m] of Object.entries(masks)) {
            const t0 = TYPE_BY_FITTER[k0] ?? 'BSpline';
            score(`seq base/${name}`, model.name, mesher, seed, T, t0, without(g0, m), truth, { modelType: p0.type });
            score(`seq grow/${name}`, model.name, mesher, seed, T, t0, fillWith(k0, p0, m), truth, { modelType: p0.type });
            const p2 = await proposeRegion(seed, facets, adjacency, model.run, { ...proposeOpts, excluded: m, extended: model.extended });
            const k2 = cappedType(seed, { ...p2, region: without(p2.region, m) }, facets, adjacency);
            score(`seq all/${name}`, model.name, mesher, seed, T, TYPE_BY_FITTER[k2] ?? 'BSpline',
                  fillWith(k2, p2, m), truth, { modelType: p2.type });
          }
        }
        continue;
      }
      if (tf) {
        for (const c of CREASES) {
          score(declaredMethod(c), '-', mesher, seed, T, truthType,
                paintFrom(seed, tf, facets, adjacency, { crease: CREASE_PRESETS[c] }).region, truth);
        }
      }
      if (ORDERED) {
        for (const size of ORDERED_SIZES) {
          const patch = creasePatch(seed, adjacency, size);
          const name = orderedTypeOf(patch, facets);
          const f = fitterFor(name);
          score(`ordered ${size}`, '-', mesher, seed, T, name,
                f ? paintFrom(seed, f, facets, adjacency).region : patch, truth);
        }
      }

      for (const model of models) {
        const p = await proposeRegion(seed, facets, adjacency, model.run, { ...proposeOpts, extended: model.extended });
        score('model only', model.name, mesher, seed, T, p.type, p.region, truth);
        const f = fitterFor(p.type);
        score('auto', model.name, mesher, seed, T, p.type,
              f ? paintFrom(seed, f, facets, adjacency, { seedRegion: p.region }).region : p.region, truth);
        if (tf) {
          score('declared+model', model.name, mesher, seed, T, truthType,
                paintFrom(seed, tf, facets, adjacency, { seedRegion: p.region }).region, truth);
        }
        if (ORDERED) {
          const patch = creaseComponent(seed, p.region, adjacency, ORDER_CREASE);
          const name = orderedTypeOf(patch, facets);
          const of = fitterFor(name);
          score('ordered type', model.name, mesher, seed, T, name,
                of ? paintFrom(seed, of, facets, adjacency, { seedRegion: p.region }).region : patch, truth);
          // ORDERED, CAPPED BY THE MODEL: the simplest type within 1%, unless that is MORE complex
          // than the model's own type and the model's type still fits within 5%. Keeps a cylinder
          // the model calls a cone a cylinder, without letting a large sphere take a cylinder whose
          // own fit narrowly missed 1%.
          const mk = f;
          const cls = classifyRegion(patch, facets);
          let capped = cls.type;
          if (mk) {
            const own = Math.min(...cls.candidates.filter(c => c.type === mk).map(c => c.relResidual), Infinity);
            const moreComplex = capped === 'nurbs' || ORDER_RANK[capped] > ORDER_RANK[mk];
            if (moreComplex && own <= 0.05) capped = mk;
          }
          const cname = TYPE_BY_FITTER[capped] ?? 'BSpline';
          const cf = fitterFor(cname);
          score('ordered+model', model.name, mesher, seed, T, cname,
                cf ? paintFrom(seed, cf, facets, adjacency, { seedRegion: p.region }).region : patch, truth);
        }
      }
    }
  }
}

const med = a => { const s = [...a].sort((p, q) => p - q); return s.length ? s[s.length >> 1] : NaN; };
const pc = v => `${(100 * v).toFixed(0)}%`.padStart(4);
const frac = (rs, fn) => rs.filter(fn).length / rs.length;
const clean = r => r.surfacePurity >= 0.95 && r.recall >= 0.95;

function line(label, rs) {
  if (!rs.length) return;
  console.log(`  ${label.padEnd(30)} n ${String(rs.length).padStart(4)}` +
    `  type ${pc(frac(rs, r => r.predType === r.truthType))}` +
    `  clean ${pc(frac(rs, clean))} (strict ${pc(frac(rs, r => r.purity >= 0.95 && r.recall >= 0.95))})` +
    `  spill ${pc(frac(rs, r => r.surfacePurity < 0.9))}` +
    `  incomplete ${pc(frac(rs, r => r.recall < 0.9))}` +
    `  recall p50 ${pc(med(rs.map(r => r.recall)))}`);
}

console.log(`\n${manifest.parts.length} parts, ${rows.length} scored fills, ` +
  `${((performance.now() - t0) / 1000).toFixed(0)} s, curvature ${CURVATURE}`);
console.log('clean: surface purity and recall >= 95% (strict: own face only). spill: surface purity < 90%. ' +
  'incomplete: recall < 90%.\n');

for (const mesher of manifest.variants) {
  const at = rows.filter(r => r.mesher === mesher);
  if (!at.length) continue;
  console.log(mesher);
  for (const c of CREASES) line(`declared (no model) ${c}`, at.filter(r => r.method === declaredMethod(c)));
  if (ORDERED) for (const size of ORDERED_SIZES) line(`ordered ${size} (no model)`, at.filter(r => r.method === `ordered ${size}`));
  for (const model of models) {
    for (const method of ['auto', ...(ORDERED ? ['ordered type', 'ordered+model'] : []), 'declared+model', 'model only']) {
      line(`${model.name} ${method}`, at.filter(r => r.model === model.name && r.method === method));
    }
  }
  console.log('');
}

if (CREASES.length > 1) {
  console.log('declared, by true type: clean / spill / incomplete per crease preset');
  console.log(`  ${'mesher / type'.padEnd(22)}${CREASES.map(c => c.padStart(22)).join('')}`);
  for (const mesher of manifest.variants) {
    for (const t of TYPE_NAMES) {
      const cells = CREASES.map(c => {
        const rs = rows.filter(r => r.mesher === mesher && r.method === declaredMethod(c) && r.truthType === t);
        return rs.length
          ? `${pc(frac(rs, clean))}/${pc(frac(rs, r => r.surfacePurity < 0.9))}/${pc(frac(rs, r => r.recall < 0.9))}`.padStart(22)
          : '-'.padStart(22);
      });
      if (cells.some(x => x.trim() !== '-')) console.log(`  ${`${mesher} ${t}`.padEnd(22)}${cells.join('')}`);
    }
  }
  console.log('');
}

if (SEQUENTIAL) {
  console.log('sequential painting, seeds on non-plane faces (oracle painting). base: fill as now, painted dropped afterwards;\n' +
    'grow: painted excluded from growth; all: also dropped from the model patch. Types: the model head, and the capped type used.');
  const methods = ['base', 'grow', 'all'].flatMap(k => ['planes', 'simpler'].map(m => `seq ${k}/${m}`))
    .sort((a, b) => a.split('/')[1].localeCompare(b.split('/')[1]));
  for (const mesher of manifest.variants) {
    for (const truth of ['all', 'Cylinder', 'Cone', 'Sphere', 'Torus']) {
      const at = rows.filter(r => r.mesher === mesher && r.method.startsWith('seq ') && (truth === 'all' || r.truthType === truth));
      if (!at.length) continue;
      console.log(`  ${mesher} ${truth}`);
      for (const method of methods) {
        const rs = at.filter(r => r.method === method);
        if (!rs.length) continue;
        console.log(`    ${method.padEnd(18)} n ${String(rs.length).padStart(4)}  model type ${pc(frac(rs, r => r.modelType === r.truthType))}` +
          `  capped type ${pc(frac(rs, r => r.predType === r.truthType))}  clean ${pc(frac(rs, clean))}` +
          `  spill ${pc(frac(rs, r => r.surfacePurity < 0.9))}  incomplete ${pc(frac(rs, r => r.recall < 0.9))}`);
      }
    }
  }
  console.log('');
}

if (ORDERED) {
  // Where each method's type goes, for the common true types: the cone-for-cylinder question.
  const methods = [...models.flatMap(m => [[`${m.name} auto`, r => r.model === m.name && r.method === 'auto'],
                                           [`${m.name} ordered`, r => r.model === m.name && r.method === 'ordered type'],
                                           [`${m.name} ordered+model`, r => r.model === m.name && r.method === 'ordered+model']]),
                   ...ORDERED_SIZES.map(s => [`ordered ${s}`, r => r.method === `ordered ${s}`])];
  console.log('predicted type by true type (share of clicks; clean fill in brackets)');
  for (const mesher of manifest.variants) {
    for (const truth of ['Plane', 'Cylinder', 'Cone', 'Sphere', 'Torus']) {
      console.log(`  ${mesher} ${truth}`);
      for (const [label, pick] of methods) {
        const rs = rows.filter(r => r.mesher === mesher && r.truthType === truth && pick(r));
        if (!rs.length) continue;
        const counts = new Map();
        for (const r of rs) counts.set(r.predType, (counts.get(r.predType) ?? 0) + 1);
        const top = [...counts].sort((a, b) => b[1] - a[1]).slice(0, 4).map(([t, n]) => `${t} ${pc(n / rs.length)}`).join(', ');
        console.log(`    ${label.padEnd(22)} ${top}   [clean ${pc(frac(rs, clean))}]`);
      }
    }
  }
  console.log('');
}

console.log('auto, clean fill by true type (type accuracy in brackets)');
console.log(`  ${'mesher / type'.padEnd(22)}${models.map(m => m.name.padStart(18)).join('')}`);
for (const mesher of manifest.variants) {
  for (const t of TYPE_NAMES) {
    const cells = models.map(m => {
      const rs = rows.filter(r => r.mesher === mesher && r.model === m.name && r.method === 'auto' && r.truthType === t);
      return rs.length ? `${pc(frac(rs, clean))} [${pc(frac(rs, r => r.predType === t))}]`.padStart(18) : '-'.padStart(18);
    });
    if (cells.some(c => c.trim() !== '-')) console.log(`  ${`${mesher} ${t}`.padEnd(22)}${cells.join('')}`);
  }
}

if (JSON_OUT) {
  writeFileSync(JSON_OUT, JSON.stringify(rows));
  console.log(`\nwrote ${rows.length} rows to ${JSON_OUT}`);
}
