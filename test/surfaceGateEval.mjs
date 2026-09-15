// Surface-hypothesis gated fills, end to end in JS, on the fillEval clicks.
//
// Usage: node test/surfaceGateEval.mjs <model.onnx with a surface output>[,...] [parts]
//
// For each click the model gives membership, type and a surface hypothesis (surface_targets.py
// conventions). The hypothesis is placed in the world from the seed centroid, seed facet normal,
// patch frame and scale. Fills compared on identical clicks:
//   paintFrom           today: model region -> paintFrom with the type
//   hyp only            grow from the model region against the HYPOTHESIS surface only
//   hyp -> fit          hyp-only region handed to paintFrom (exact fitters take over)
//   fit & hyp           paintFrom's fill, kept only where facets also sit within a loose
//                       tolerance (2 x tolD, 25 deg) of the hypothesis, connected to the seed
// each under the model's predicted type ("auto") and the true type ("declared").
import { readFileSync } from 'node:fs';
import { join, basename, dirname } from 'node:path';
import { createRequire } from 'node:module';
import { pathToFileURL, fileURLToPath } from 'node:url';

const REPO = fileURLToPath(new URL('..', import.meta.url));
const imp = p => import(pathToFileURL(join(REPO, p)).href);
const { buildFacets, buildAdjacency } = await imp('js/meshCore.js');
const { paintFrom, surfaceEval } = await imp('js/regionGrow.js');
const { fitRegionAs } = await imp('js/gaussMap.js');
const { buildFeatures, TYPE_NAMES, fitterFor, N_FEATURES } = await imp('js/seedNet.js');
const { attachCurvature } = await imp('js/curvature.js');
const ort = createRequire(pathToFileURL(join(REPO, 'package.json')).href)('onnxruntime-node');

const DIR = process.env.FILL_EVAL_DIR || 'D:/abc-data/fill_eval';
const MODEL_PATHS = process.argv[2].split(',');
const PART_LIMIT = Number(process.argv[3] || 1e9);
const SEEDS_PER_TYPE = 2;
const MAX_SEEDS_PER_PART = 10;
const DEG = Math.PI / 180;
const METHODS = ['paintFrom', 'paintFrom+', 'hyp only', 'hyp -> fit', 'hyp -> fit+', 'hyp3x only', 'hyp3x -> fit', 'fit & hyp'];

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const addS = (a, b, s) => [a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2]];
const unit = (v) => { const l = Math.hypot(v[0], v[1], v[2]); return l > 1e-300 ? [v[0] / l, v[1] / l, v[2] / l] : null; };

// copy of js/seedNet.js frameFromNormal (not exported); rows map world -> frame
function frameFromNormal(n) {
  const len = Math.hypot(n[0], n[1], n[2]) || 1e-30;
  const nz = [n[0] / len, n[1] / len, n[2] / len];
  const a = [Math.abs(nz[0]), Math.abs(nz[1]), Math.abs(nz[2])];
  let k = 0;
  if (a[1] < a[k]) k = 1;
  if (a[2] < a[k]) k = 2;
  const axis = [0, 0, 0];
  axis[k] = 1;
  let t1 = [nz[1] * axis[2] - nz[2] * axis[1], nz[2] * axis[0] - nz[0] * axis[2], nz[0] * axis[1] - nz[1] * axis[0]];
  const l1 = Math.hypot(t1[0], t1[1], t1[2]) || 1e-30;
  t1 = [t1[0] / l1, t1[1] / l1, t1[2] / l1];
  const t2 = [nz[1] * t1[2] - nz[2] * t1[1], nz[2] * t1[0] - nz[0] * t1[2], nz[0] * t1[1] - nz[1] * t1[0]];
  return [t1, t2, nz];
}

/** surface_targets.reconstruct, in world coordinates and regionGrow.surfaceEval's format. */
function hypothesis(kind, surf, q, n, R, scale) {
  const kn = Math.sinh(surf[3]);
  if (kind === 'plane' || Math.abs(kn) < 1e-6) return { type: 'plane', params: { origin: q, normal: n } };
  const s = kn > 0 ? 1 : -1;
  const d = scale / Math.abs(kn);
  const foot = addS(q, n, -s * d);
  if (kind === 'sphere') return { type: 'sphere', params: { center: foot, radius: d } };
  const aw = unit([
    surf[0] * R[0][0] + surf[1] * R[1][0] + surf[2] * R[2][0],
    surf[0] * R[0][1] + surf[1] * R[1][1] + surf[2] * R[2][1],
    surf[0] * R[0][2] + surf[1] * R[1][2] + surf[2] * R[2][2],
  ]);
  if (!aw) return null;
  if (kind === 'cylinder') return { type: 'cylinder', params: { axis: aw, axisPoint: foot, radius: d } };
  if (kind === 'cone') {
    const sina = Math.min(Math.abs(dot(n, aw)), 0.999999);
    if (sina < 1e-6) return { type: 'cylinder', params: { axis: aw, axisPoint: foot, radius: d } };
    const sg = s * dot(n, aw) >= 0 ? 1 : -1;
    const aOpen = [-sg * aw[0], -sg * aw[1], -sg * aw[2]];
    return { type: 'cone', params: { axis: aOpen, apex: addS(foot, aOpen, -d / sina), halfAngle: Math.asin(sina) } };
  }
  if (kind === 'torus') {
    const ks = Math.sinh(surf[4]);
    if (Math.abs(ks) < 1e-9) return null;
    const na = dot(n, aw);
    const up = unit([n[0] - na * aw[0], n[1] - na * aw[1], n[2] - na * aw[2]]);
    if (!up) return null;
    const side = (ks > 0 ? 1 : -1) * s;
    const bigR = scale / Math.abs(ks);
    return { type: 'torus', params: { axis: aw, center: addS(foot, up, -side * bigR), majorRadius: bigR, minorRadius: d } };
  }
  return null;
}

function medianFacetSize(facets) {
  const s = [];
  for (let t = 0; t < facets.count; t++) s.push(Math.sqrt(Math.max(facets.areas[t], 0)));
  s.sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)] || 1;
}

/** growOn from js/regionGrow.js (not exported), with a membership predicate for 'fit & hyp'. */
function grow(seedSet, facets, adjacency, type, params, tolD, cosTol, allowed = null) {
  const ev = surfaceEval(type, params);
  if (!ev) return new Set();
  const fits = (t) => {
    if (allowed && !allowed.has(t)) return false;
    const c = [facets.centroids[t * 3], facets.centroids[t * 3 + 1], facets.centroids[t * 3 + 2]];
    if (Math.abs(ev.distance(c)) > tolD) return false;
    const sn = ev.normal(c);
    return Math.abs(sn[0] * facets.normals[t * 3] + sn[1] * facets.normals[t * 3 + 1] + sn[2] * facets.normals[t * 3 + 2]) >= cosTol;
  };
  const region = new Set();
  const queue = [];
  for (const s of seedSet) if (fits(s)) { region.add(s); queue.push(s); }
  while (queue.length) {
    const t = queue.pop();
    for (const e of adjacency.get(t) || []) {
      const nb = e.neighbor;
      if (region.has(nb) || !fits(nb)) continue;
      region.add(nb); queue.push(nb);
    }
  }
  return region;
}

/**
 * paintFrom with the two acceptance fixes measured on 2026-09-13: an adaptive angle tolerance
 * (3 x the region's own p90 normal deviation, clamped 4-15 deg) and a membership veto
 * (probability >= 0.5 from patches seeded at accepted frontier facets). "+" in method names.
 */
async function paintPlus(seedTri, type, facets, adjacency, seedRegion, model, tolD) {
  const prob = new Map();
  const patch = async (centre) => {
    const { features, ids, n } = buildFeatures(centre, facets, adjacency, 512);
    const out = await model.session.run({
      features: new ort.Tensor('float32', features, [1, n, N_FEATURES]),
      mask: new ort.Tensor('float32', new Float32Array(n).fill(1), [1, n]),
    });
    const same = out.same_logit.data;
    for (let i = 0; i < n; i++) if (!prob.has(ids[i])) prob.set(ids[i], 1 / (1 + Math.exp(-same[i])));
  };
  await patch(seedTri);
  const cen = t => [facets.centroids[t * 3], facets.centroids[t * 3 + 1], facets.centroids[t * 3 + 2]];
  const nrm = t => [facets.normals[t * 3], facets.normals[t * 3 + 1], facets.normals[t * 3 + 2]];

  async function growVeto(seedSet, params) {
    const ev = surfaceEval(type, params);
    if (!ev) return new Set();
    const dotAt = t => dot(ev.normal(cen(t)), nrm(t));
    const devs = [...seedSet].map(s => Math.acos(Math.min(1, Math.abs(dotAt(s)))) / DEG).sort((a, b) => a - b);
    const angle = Math.min(15, Math.max(4, 3 * (devs[Math.floor(0.9 * (devs.length - 1))] ?? 5)));
    const cosTol = Math.cos(angle * DEG);
    const fits = t => Math.abs(ev.distance(cen(t))) <= tolD && Math.abs(dotAt(t)) >= cosTol;
    const region = new Set();
    const queue = [];
    for (const s of seedSet) {
      if (!fits(s) || (prob.has(s) && prob.get(s) < 0.5)) continue;
      region.add(s); queue.push(s);
    }
    while (queue.length) {
      const t = queue.pop();
      for (const e of adjacency.get(t) || []) {
        const nb = e.neighbor;
        if (region.has(nb) || !fits(nb)) continue;
        if (!prob.has(nb)) await patch(t);
        if ((prob.get(nb) ?? 1) < 0.5) continue;
        region.add(nb); queue.push(nb);
      }
    }
    return region;
  }

  let region = new Set(seedRegion);
  let fit = fitRegionAs(type, region, facets);
  if (fit.degenerate || !fit.params) return new Set([seedTri]);
  for (let i = 0; i < 4; i++) {
    const grown = await growVeto(region, fit.params);
    if (grown.size === 0) break;
    const refit = fitRegionAs(type, grown, facets);
    if (refit.degenerate || !refit.params) break;
    const stable = grown.size === region.size;
    region = grown;
    fit = refit;
    if (stable) break;
  }
  return region;
}

const models = [];
for (const path of MODEL_PATHS) {
  const session = await ort.InferenceSession.create(path);
  if (!session.outputNames.includes('surface')) throw new Error(`${path} has no surface output`);
  models.push({ name: basename(dirname(path)), session });
}

async function predict(model, seed, facets, adjacency) {
  const { features, ids, n } = buildFeatures(seed, facets, adjacency, 512);
  const out = await model.session.run({
    features: new ort.Tensor('float32', features, [1, n, N_FEATURES]),
    mask: new ort.Tensor('float32', new Float32Array(n).fill(1), [1, n]),
  });
  const same = out.same_logit.data, tl = out.type_logit.data;
  const region = new Set([seed]);
  for (let i = 0; i < n; i++) if (1 / (1 + Math.exp(-same[i])) >= 0.5) region.add(ids[i]);
  let best = 0;
  for (let i = 1; i < tl.length; i++) if (tl[i] > tl[best]) best = i;
  let sumSq = 0;
  const sx = facets.centroids[seed * 3], sy = facets.centroids[seed * 3 + 1], sz = facets.centroids[seed * 3 + 2];
  for (const t of ids) {
    const dx = facets.centroids[t * 3] - sx, dy = facets.centroids[t * 3 + 1] - sy, dz = facets.centroids[t * 3 + 2] - sz;
    sumSq += dx * dx + dy * dy + dz * dz;
  }
  const nrm = [facets.normals[seed * 3], facets.normals[seed * 3 + 1], facets.normals[seed * 3 + 2]];
  return { region, typeName: TYPE_NAMES[best], surf: Array.from(out.surface.data), scale: Math.sqrt(sumSq / n) || 1,
           R: frameFromNormal(nrm), q: [sx, sy, sz], n: nrm };
}

const copy = b => b.buffer.slice(b.byteOffset, b.byteOffset + b.length);
const load = stem => ({
  P: Float64Array.from(new Float32Array(copy(readFileSync(join(DIR, stem + '.pos'))))),
  F: new Int32Array(copy(readFileSync(join(DIR, stem + '.fid')))),
  T: new Uint8Array(copy(readFileSync(join(DIR, stem + '.typ')))),
});
let x = 99;
const rng = () => (x = (x * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
const rows = [];
const manifest = JSON.parse(readFileSync(join(DIR, 'manifest.json'), 'utf8'));
const reference = manifest.variants[0];
const MESHERS = ['native', 'gmsh', 'game'];
const t0 = performance.now();

for (const entry of manifest.parts.slice(0, PART_LIMIT)) {
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
    for (let i = 0; i < SEEDS_PER_TYPE && arr.length; i++) seedFaces.push(arr.splice(Math.floor(rng() * arr.length), 1)[0]);
  }
  seedFaces.splice(MAX_SEEDS_PER_PART);

  for (const mesher of manifest.variants) {
    const v = entry.variants[mesher];
    if (!v) continue;
    const { P, F, T } = mesher === reference ? base : load(v.stem);
    if (!MESHERS.includes(mesher)) { for (const face of seedFaces) rng(); continue; }  // keep the click sequence aligned
    const facets = buildFacets(P);
    const adjacency = buildAdjacency(P, facets);
    attachCurvature(P, facets);
    const tolD = 0.5 * medianFacetSize(facets);
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
      const score = (model, method, region) => {
        let hit = 0, same = 0;
        for (const t of region) { if (truth.has(t)) hit++; else if (T[t] === T[seed]) same++; }
        const n = region.size || 1;
        rows.push({ model, mesher, method, truthType, surfacePurity: (hit + same) / n, recall: hit / truth.size });
      };
      for (const model of models) {
        const p = await predict(model, seed, facets, adjacency);
        for (const [mode, typeName] of [['auto', p.typeName], ['declared', truthType]]) {
          const kind = fitterFor(typeName);
          if (!kind) {                                   // no analytic surface: the model region is the fill
            if (mode === 'auto') for (const m of METHODS) score(model.name, `${mode} ${m}`, p.region);
            continue;
          }
          const fitted = paintFrom(seed, kind, facets, adjacency, { seedRegion: p.region }).region;
          score(model.name, `${mode} paintFrom`, fitted);
          const plus = await paintPlus(seed, kind, facets, adjacency, p.region, model, tolD);
          score(model.name, `${mode} paintFrom+`, plus);
          const hyp = hypothesis(kind, p.surf, p.q, p.n, p.R, p.scale);
          if (!hyp) {
            for (const m of METHODS.slice(2)) score(model.name, `${mode} ${m}`, m.endsWith('+') ? plus : fitted);
            continue;
          }
          let hr = grow(p.region, facets, adjacency, hyp.type, hyp.params, tolD, Math.cos(15 * DEG));
          if (!hr.size) hr = new Set([seed]);
          score(model.name, `${mode} hyp only`, hr);
          score(model.name, `${mode} hyp -> fit`, hr.size >= 3 ? paintFrom(seed, hyp.type, facets, adjacency, { seedRegion: hr }).region : hr);
          score(model.name, `${mode} hyp -> fit+`, hr.size >= 3 ? await paintPlus(seed, hyp.type, facets, adjacency, hr, model, tolD) : hr);
          // The prediction is locally good but drifts across a patch (s1: axis p50 3-5 deg,
          // |kn| error ~9%), so a strict gate under-fills. Looser: 3 x tolD and 20 deg.
          let hl = grow(p.region, facets, adjacency, hyp.type, hyp.params, 3 * tolD, Math.cos(20 * DEG));
          if (!hl.size) hl = new Set([seed]);
          score(model.name, `${mode} hyp3x only`, hl);
          score(model.name, `${mode} hyp3x -> fit`, hl.size >= 3 ? paintFrom(seed, hyp.type, facets, adjacency, { seedRegion: hl }).region : hl);
          const both = grow(new Set([seed]), facets, adjacency, hyp.type, hyp.params, 2 * tolD, Math.cos(25 * DEG), fitted);
          score(model.name, `${mode} fit & hyp`, both.size ? both : fitted);
        }
      }
    }
  }
}

const pc = v => `${(100 * v).toFixed(0)}%`.padStart(4);
console.log(`\n${rows.length} fills, ${((performance.now() - t0) / 1000).toFixed(0)} s`);
console.log('clean: surface purity and recall >= 95%. spill: surface purity < 90%. incomplete: recall < 90%.');
for (const mesher of MESHERS) {
  console.log(`\n${mesher}`);
  for (const model of models) for (const mode of ['auto', 'declared']) for (const m of METHODS) {
    const rs = rows.filter(r => r.mesher === mesher && r.model === model.name && r.method === `${mode} ${m}`);
    if (!rs.length) continue;
    const f = fn => rs.filter(fn).length / rs.length;
    console.log(`  ${`${model.name} ${mode} ${m}`.padEnd(34)} n ${String(rs.length).padStart(4)}  clean ${pc(f(r => r.surfacePurity >= 0.95 && r.recall >= 0.95))}` +
      `  spill ${pc(f(r => r.surfacePurity < 0.9))}  incomplete ${pc(f(r => r.recall < 0.9))}`);
  }
  for (const model of models) {
    console.log(`    ${model.name} declared, clean by true type: ` + ['Plane', 'Cylinder', 'Cone', 'Sphere', 'Torus'].map(ty => {
      const cell = METHODS.map(m => {
        const rs = rows.filter(r => r.mesher === mesher && r.model === model.name && r.method === `declared ${m}` && r.truthType === ty);
        return rs.length ? pc(rs.filter(r => r.surfacePurity >= 0.95 && r.recall >= 0.95).length / rs.length).trim() : '-';
      }).join('/');
      return `${ty} ${cell}`;
    }).join('  ') + '   (' + METHODS.join(' / ') + ')');
  }
}
