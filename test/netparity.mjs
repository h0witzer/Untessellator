/**
 * netparity.mjs — prove js/seedNet.js builds the same features as ml/patches.py.
 *
 * Usage:  node test/netparity.mjs
 *
 * The model's weights were fitted against the exact feature arrangement in
 * ml/patches.py. If this side disagrees — a transposed frame axis, a differently
 * normalised area, a ring index off by one — the model does not fail loudly, it just
 * gets quietly worse on inputs that look superficially fine. That failure mode is
 * indistinguishable from "the model is mediocre", which is the worst possible thing to
 * be unsure about when deciding whether to ship it.
 *
 * So: run both implementations over the same fixture meshes and the same seeds, and
 * compare feature blocks elementwise.
 *
 * The canonical frame has one genuinely arbitrary degree of freedom — the choice of
 * tangent axes perpendicular to the seed normal. Both sides resolve it the same way
 * (least-aligned world axis, first minimum wins on ties), so the comparison is exact
 * rather than up-to-rotation. If that ever diverges, the dx/dy columns will disagree
 * while dz, the normals' z component and dist stay identical — a distinctive signature
 * worth recognising rather than debugging blind.
 */

import { readFileSync, writeFileSync, mkdtempSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { tmpdir } from 'node:os';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

import { buildFacets, buildAdjacency } from '../js/meshCore.js';
import { buildFeatures, kRing, N_FEATURES, N_FEATURES_V6 } from '../js/seedNet.js';
import { EXT_FEATURE_NAMES } from '../js/seedFeatures.js';
import { attachCurvature } from '../js/curvature.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const FIX = join(HERE, 'fixtures');
const PY = 'D:/abc-data/venv/Scripts/python.exe';
const ML = join(HERE, '..', 'ml');

function readSTL(path) {
  const buf = readFileSync(path);
  const tris = buf.readUInt32LE(80);
  const positions = new Float64Array(tris * 9);
  for (let t = 0; t < tris; t++) {
    const o = 84 + t * 50;
    for (let i = 0; i < 9; i++) positions[t * 9 + i] = buf.readFloatLE(o + 12 + i * 4);
  }
  return positions;
}

const PY_DRIVER = `
import sys, json
import numpy as np
sys.path.insert(0, r"${ML.replace(/\\/g, '\\\\')}")
from meshcore import build_facets, build_adjacency, adjacency_csr, facet_curvature, k_ring
from patches import extract_patch

cfg = json.load(open(sys.argv[1]))
pos = np.array(cfg["positions"], dtype=np.float64).reshape(-1, 9)
facets = build_facets(pos)
pairs, ang = build_adjacency(pos, facets, cfg["quant"])
indptr, indices = adjacency_csr(pairs, facets["count"])
curv = facet_curvature(pos, facets)
fid = np.zeros(facets["count"], np.int32)
fty = np.zeros(facets["count"], np.uint8)
out = []
for seed in cfg["seeds"]:
    p = extract_patch(facets, indptr, indices, fid, fty, int(seed),
                      cfg["max_facets"], rng=None, spin=False, curv=curv)
    out.append({"ids": p["ids"].tolist(), "features": p["features"].ravel().tolist()}
               if p is not None else None)
json.dump(out, open(sys.argv[2], "w"))
`;

function pythonFeatures(positions, seeds, maxFacets, quant) {
  const dir = mkdtempSync(join(tmpdir(), 'netparity-'));
  const cfgPath = join(dir, 'cfg.json');
  const outPath = join(dir, 'out.json');
  const drvPath = join(dir, 'drv.py');
  writeFileSync(cfgPath, JSON.stringify({
    positions: Array.from(positions), seeds, max_facets: maxFacets, quant,
  }));
  writeFileSync(drvPath, PY_DRIVER);
  execFileSync(PY, [drvPath, cfgPath, outPath], { stdio: ['ignore', 'inherit', 'inherit'] });
  return JSON.parse(readFileSync(outPath, 'utf8'));
}

function compare(label, positions, seeds, maxFacets, quant) {
  const facets = buildFacets(positions);
  const adjacency = buildAdjacency(positions, facets, quant);
  attachCurvature(positions, facets);
  const py = pythonFeatures(positions, seeds, maxFacets, quant);

  let worstFeat = 0, idMismatch = 0, sizeMismatch = 0, checked = 0, worstCol = -1;
  for (let s = 0; s < seeds.length; s++) {
    const ref = py[s];
    const got = buildFeatures(seeds[s], facets, adjacency, maxFacets);
    if (!ref) continue;
    checked++;
    if (ref.ids.length !== got.n) { sizeMismatch++; continue; }
    for (let i = 0; i < got.n; i++) if (ref.ids[i] !== got.ids[i]) { idMismatch++; break; }
    for (let i = 0; i < got.n * N_FEATURES; i++) {
      const d = Math.abs(ref.features[i] - got.features[i]);
      if (d > worstFeat) { worstFeat = d; worstCol = i % N_FEATURES; }
    }
  }
  const ok = !sizeMismatch && !idMismatch && worstFeat < 2e-5;
  console.log(`  ${label.padEnd(26)} seeds ${String(checked).padStart(3)}` +
    `   patch-size mismatches ${sizeMismatch}   id-order mismatches ${idMismatch}` +
    `   max|feature diff| ${worstFeat.toExponential(2)}` +
    (worstCol >= 0 ? ` (col ${worstCol})` : '') + `   ${ok ? 'ok' : 'MISMATCH'}`);
  return ok;
}

console.log('\nfeature parity: js/seedNet.js vs ml/patches.py\n');

let allOk = true;

{
  const positions = readSTL(join(FIX, 'tangent-fillet.stl'));
  const facets = buildFacets(positions);
  const seeds = [0, 1, 7, 100, 523, 1500, 3000, facets.count - 1]
    .filter(s => s >= 0 && s < facets.count);
  allOk = compare('tangent-fillet', positions, seeds, 512, 1e4) && allOk;
  allOk = compare('tangent-fillet (256)', positions, seeds, 256, 1e4) && allOk;
}

for (const name of ['ball-seat.stl', 'cone-sweep.stl', 'countersunk-plate.stl']) {
  const positions = readSTL(join(FIX, name));
  const facets = buildFacets(positions);
  const step = Math.max(1, Math.floor(facets.count / 7));
  const seeds = [];
  for (let s = 0; s < facets.count && seeds.length < 7; s += step) seeds.push(s);
  allOk = compare(name.replace('.stl', ''), positions, seeds, 512, 1e4) && allOk;
}

// ── v6: the extended columns (js/seedFeatures.js vs ml/seedfeatures.py), with and without
// facets excluded as if already painted into other groups.

const PY_DRIVER_V6 = `
import sys, json
import numpy as np
sys.path.insert(0, r"${ML.replace(/\\/g, '\\\\')}")
from meshcore import build_facets, build_adjacency, facet_curvature
from patches import extract_patch
from seedfeatures import adjacency_csr_angles

cfg = json.load(open(sys.argv[1]))
pos = np.array(cfg["positions"], dtype=np.float64).reshape(-1, 9)
facets = build_facets(pos)
pairs, ang = build_adjacency(pos, facets, cfg["quant"])
indptr, indices, eang = adjacency_csr_angles(pairs, ang, facets["count"])
curv = facet_curvature(pos, facets)
fid = np.zeros(facets["count"], np.int32)
fty = np.zeros(facets["count"], np.uint8)
exc = np.array(cfg["excluded"], dtype=bool) if cfg["excluded"] is not None else None
out = []
for seed in cfg["seeds"]:
    p = extract_patch(facets, indptr, indices, fid, fty, int(seed), cfg["max_facets"], rng=None,
                      spin=False, curv=curv, extended=True, eangles=eang, excluded=exc)
    out.append({"ids": p["ids"].tolist(), "features": p["features"].ravel().tolist()}
               if p is not None else None)
json.dump(out, open(sys.argv[2], "w"))
`;

function compareV6(label, positions, seeds, maxFacets, quant, excludeEvery) {
  const facets = buildFacets(positions);
  const adjacency = buildAdjacency(positions, facets, quant);
  attachCurvature(positions, facets);
  let excluded = null;
  if (excludeEvery) {
    excluded = new Uint8Array(facets.count);
    for (let t = 0; t < facets.count; t++) if (t % excludeEvery === 3 && !seeds.includes(t)) excluded[t] = 1;
  }
  const dir = mkdtempSync(join(tmpdir(), 'netparity6-'));
  writeFileSync(join(dir, 'cfg.json'), JSON.stringify({
    positions: Array.from(positions), seeds, max_facets: maxFacets, quant,
    excluded: excluded ? Array.from(excluded) : null,
  }));
  writeFileSync(join(dir, 'drv.py'), PY_DRIVER_V6);
  execFileSync(PY, [join(dir, 'drv.py'), join(dir, 'cfg.json'), join(dir, 'out.json')], { stdio: ['ignore', 'inherit', 'inherit'] });
  const py = JSON.parse(readFileSync(join(dir, 'out.json'), 'utf8'));

  const W = N_FEATURES_V6;
  const worst = new Float64Array(W);
  let idMismatch = 0, sizeMismatch = 0, checked = 0;
  for (let s = 0; s < seeds.length; s++) {
    const ref = py[s];
    if (!ref) continue;
    const got = buildFeatures(seeds[s], facets, adjacency, maxFacets, null, excluded, true);
    checked++;
    if (ref.ids.length !== got.n) { sizeMismatch++; continue; }
    for (let i = 0; i < got.n; i++) if (ref.ids[i] !== got.ids[i]) { idMismatch++; break; }
    for (let i = 0; i < got.n * W; i++) {
      const d = Math.abs(ref.features[i] - got.features[i]);
      if (d > worst[i % W]) worst[i % W] = d;
    }
  }
  const baseWorst = Math.max(...worst.slice(0, N_FEATURES));
  const extWorst = Math.max(...worst.slice(N_FEATURES));
  const extCol = worst.indexOf(extWorst);
  const ok = !sizeMismatch && !idMismatch && baseWorst < 2e-5 && extWorst < 1e-4;
  console.log(`  ${label.padEnd(34)} seeds ${String(checked).padStart(3)}   size/id mismatches ${sizeMismatch}/${idMismatch}` +
    `   base ${baseWorst.toExponential(2)}   extended ${extWorst.toExponential(2)} (${EXT_FEATURE_NAMES[extCol - N_FEATURES] ?? '-'})   ${ok ? 'ok' : 'MISMATCH'}`);
  return ok;
}

console.log('\nv6 extended-feature parity: js/seedFeatures.js vs ml/seedfeatures.py\n');
for (const name of ['tangent-fillet.stl', 'ball-seat.stl', 'cone-sweep.stl', 'countersunk-plate.stl']) {
  const positions = readSTL(join(FIX, name));
  const facets = buildFacets(positions);
  const step = Math.max(1, Math.floor(facets.count / 6));
  const seeds = [];
  for (let s = 0; s < facets.count && seeds.length < 6; s += step) seeds.push(s);
  allOk = compareV6(name.replace('.stl', ''), positions, seeds, 512, 1e4, 0) && allOk;
  allOk = compareV6(`${name.replace('.stl', '')} (painted)`, positions, seeds, 512, 1e4, 7) && allOk;
}

console.log(allOk ? '\nPASS — features agree\n' : '\nFAIL — features diverge\n');
process.exit(allOk ? 0 : 1);
