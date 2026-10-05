/**
 * groupTypeDump.mjs — training rows for the group-level type model (js/groupType.js,
 * ml/group_type_train.py), from finished auto-paint segmentations.
 *
 * Usage:  node test/groupTypeDump.mjs <mesh dir> <truth dir> <seg dir> <out prefix> [--shard k/n] [--ff json]
 *
 *   <mesh dir>   fill_eval-format parts (<stem>.pos, <stem>.fid)
 *   <truth dir>  <part>.srf from ml/export_fit_truth.py or export_abc_obj_eval.py --truth-only
 *   <seg dir>    test/autoSeed.mjs --dump-seg output (<stem>.seg, <stem>.json with the painted types)
 *
 * Each part is segmented as the export path sees it: the painted groups, then js/freeformMerge.js (as
 * test/surfaceScore.mjs's fexact variant). Every group freeformMerge leaves to typeFinalGroup, with
 * at least 3 facets, gives one row: groupType.groupTypeFeatures, and as its label the CANONICAL type
 * of the true face holding most of its area ('freeform' when that face is no primitive), with the
 * group's purity (that face's share of the group's area) and its coverage (the group's share of
 * that face).
 *
 * Files: <prefix>.feat (float32, N_GROUP_FEATURES per row) and <prefix>.rows.jsonl (one JSON object per
 * row: stem, mesher, part, segment, label, purity, coverage, area share of the part, facets, painted,
 * classified, rule) plus <prefix>.json naming the columns.
 */

import { readFileSync, writeFileSync, readdirSync, openSync, writeSync, closeSync, existsSync } from 'node:fs';
import { join } from 'node:path';

import { preparePaintPositions } from '../js/paintCore.js';
import { freeformMerge } from '../js/freeformMerge.js';
import { groupTypeFeatures, GROUP_FEATURES, GROUP_TYPES, N_GROUP_FEATURES } from '../js/groupType.js';

const args = process.argv.slice(2);
const flag = (name, dflt) => { const i = args.indexOf(name); if (i < 0) return dflt; const v = args[i + 1]; args.splice(i, 2); return v; };
const SHARD = flag('--shard', '0/1').split('/').map(Number);
const FF_OPTS = JSON.parse(flag('--ff', '{}'));
const [MESH_DIR, TRUTH_DIR, SEG_DIR, OUT] = args;
if (!OUT) { console.error('usage: node test/groupTypeDump.mjs <mesh dir> <truth dir> <seg dir> <out prefix> [--shard k/n]'); process.exit(2); }

const KIND = ['plane', 'cylinder', 'cone', 'sphere', 'torus'];
const rd = (p, T) => { const b = readFileSync(p); return new T(b.buffer.slice(b.byteOffset, b.byteOffset + b.length)); };

const stems = readdirSync(SEG_DIR).filter((f) => f.endsWith('.json')).map((f) => f.slice(0, -5)).sort()
  .filter((_, i) => i % SHARD[1] === SHARD[0]);
const featFd = openSync(`${OUT}.feat`, 'w');
const rowsFd = openSync(`${OUT}.rows.jsonl`, 'w');
let rows = 0, parts = 0;
const t0 = Date.now();
for (const stem of stems) {
  const meta = JSON.parse(readFileSync(join(SEG_DIR, `${stem}.json`), 'utf8'));
  const srf = join(TRUTH_DIR, `${meta.part}.srf`);
  if (!existsSync(srf) || !existsSync(join(MESH_DIR, `${stem}.pos`))) continue;
  const P = Float64Array.from(rd(join(MESH_DIR, `${stem}.pos`), Float32Array));
  const F = rd(join(MESH_DIR, `${stem}.fid`), Int32Array);
  const truth = rd(srf, Float64Array);
  const seg = Int32Array.from(rd(join(SEG_DIR, `${stem}.seg`), Int32Array));
  const { facets, adjacency } = preparePaintPositions(P);
  if (F.length !== facets.count || seg.length !== facets.count) { console.error(`${stem}: facet counts differ`); continue; }
  // the true type of every face id: the CANONICAL kind (a B-spline that is really a torus is a torus)
  const kindOf = (id) => {
    if (id < 0 || id * 12 + 12 > truth.length) return 'freeform';
    const canon = truth[id * 12 + 1];
    return canon >= 0 && canon <= 4 ? KIND[canon] : 'freeform';
  };
  const faceArea = new Map();
  let total = 0;
  for (let t = 0; t < facets.count; t++) {
    if (facets.degenerate[t]) continue;
    faceArea.set(F[t], (faceArea.get(F[t]) ?? 0) + facets.areas[t]);
    total += facets.areas[t];
  }
  const { freeform } = freeformMerge(seg, facets, adjacency, FF_OPTS);
  const groups = new Map();
  for (let t = 0; t < facets.count; t++) {
    if (seg[t] < 0 || facets.degenerate[t]) continue;
    const g = groups.get(seg[t]) ?? { tris: [], area: 0, byFace: new Map() };
    g.tris.push(t); g.area += facets.areas[t];
    g.byFace.set(F[t], (g.byFace.get(F[t]) ?? 0) + facets.areas[t]);
    groups.set(seg[t], g);
  }
  for (const [id, g] of groups) {
    if (freeform.has(id) || g.tris.length < 3) continue;
    const painted = meta.types[id] ?? null;
    const r = groupTypeFeatures(new Set(g.tris), facets, adjacency, painted);
    if (!r) continue;
    let maj = -1, a = -1;
    for (const [f, fa] of g.byFace) if (fa > a) { a = fa; maj = f; }
    writeSync(featFd, new Uint8Array(r.features.buffer, r.features.byteOffset, N_GROUP_FEATURES * 4));
    writeSync(rowsFd, JSON.stringify({
      stem, mesher: meta.mesher, part: meta.part, segment: id, label: kindOf(maj),
      purity: a / g.area, coverage: a / (faceArea.get(maj) || a), areaShare: g.area / total, facets: g.tris.length,
      painted, classified: GROUP_TYPES.includes(r.cls.type) ? r.cls.type : null, rule: r.rule,
    }) + '\n');
    rows++;
  }
  parts++;
  if (parts % 25 === 0) console.log(`  ${parts} parts, ${rows} groups, ${((Date.now() - t0) / 1000).toFixed(0)} s`);
}
closeSync(featFd);
closeSync(rowsFd);
writeFileSync(`${OUT}.json`, JSON.stringify({ columns: GROUP_FEATURES, classes: GROUP_TYPES, rows, parts }, null, 1));
console.log(`wrote ${rows} groups from ${parts} parts to ${OUT}.*`);
