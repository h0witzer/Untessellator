/**
 * freeformSeg.mjs — how whole the auto-painter keeps FREEFORM faces (B-spline, extrusion, revolution,
 * other), which test/surfaceScore.mjs does not score: a freeform face has no analytic surface to find, but
 * the solid needs it in ONE group, the group a B-spline is fitted to.
 *
 * Usage:  node test/freeformSeg.mjs <mesh dir> <seg dir> [--json out.json] [--merge] [--crease 30]
 *
 *   --merge  js/freeformMerge.js on each segmentation first (as the studio's auto-paint end does)
 *
 *   <mesh dir>  .pos / .fid / .typ (+ manifest.json), from ml/export_abc_obj_eval.py --freeform or
 *               ml/export_real_pairs.py
 *   <seg dir>   test/autoSeed.mjs --dump-seg: <stem>.seg (group per triangle, -1 unpainted), <stem>.json
 *
 * Per true face, with L the group holding most of its area:
 *   once    L holds >= 95% of the face and is >= 95% that face (the face is one pure group)
 *   pieces  groups holding >= 5% of the face
 * reported for freeform and, as a check, analytic faces; and per part, groups per true face.
 */

import { readFileSync, writeFileSync, readdirSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { preparePaintPositions } from '../js/paintCore.js';
import { freeformMerge } from '../js/freeformMerge.js';
import { annotateEdges, edgeWall } from '../js/edgeWall.js';
import { createRequire } from 'node:module';

const args = process.argv.slice(2);
const flag = (name, dflt) => { const i = args.indexOf(name); if (i < 0) return dflt; const v = args[i + 1]; args.splice(i, 2); return v; };
const JSON_OUT = flag('--json', null);
const CREASE = Number(flag('--crease', '30'));
// --ff 'curv=0.2;small=0;lowpoly=1;judged=1': freeformMerge options (curvature-jump boundary share, small joins,
// low-poly skip, the part gate's share over judged groups only, flat small groups never absorbed, island joins)
const FF = Object.fromEntries((flag('--ff', '') || '').split(';').filter(Boolean).map((kv) => kv.split('=')).map(([k, v]) => [k, Number(v)]));
const MERGE = args.includes('--merge');
if (MERGE) args.splice(args.indexOf('--merge'), 1);
// --edge-wall <run dir> --edge fpr2: the learned face boundary separates groups too (the studio's edge model)
const EDGE_DIR = flag('--edge-wall', null);
const EDGE = flag('--edge', 'fpr2');
let edgeRunner = null, edgeMeta = null;
if (EDGE_DIR) {
  const ort = createRequire(import.meta.url)('onnxruntime-node');
  edgeMeta = JSON.parse(readFileSync(join(EDGE_DIR, 'edge.json'), 'utf8'));
  const sess = await ort.InferenceSession.create(join(EDGE_DIR, 'edge.onnx'), { intraOpNumThreads: 2, interOpNumThreads: 1 });
  edgeRunner = async (features, n) => (await sess.run({ features: new ort.Tensor('float32', features, [n, features.length / n]) })).logit.data;
}
const [MESH_DIR, SEG_DIR] = args;
const TYPES = ['plane', 'cylinder', 'cone', 'sphere', 'torus', 'revolution', 'extrusion', 'bspline', 'other'];
const FREEFORM = new Set([5, 6, 7, 8]);
const copy = (b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.length);

const rows = [];
for (const f of readdirSync(SEG_DIR).filter((x) => x.endsWith('.seg')).sort()) {
  const stem = f.slice(0, -4);
  if (!existsSync(join(MESH_DIR, `${stem}.fid`))) continue;
  const P = new Float32Array(copy(readFileSync(join(MESH_DIR, `${stem}.pos`))));
  const F = new Int32Array(copy(readFileSync(join(MESH_DIR, `${stem}.fid`))));
  const Ty = new Uint8Array(copy(readFileSync(join(MESH_DIR, `${stem}.typ`))));
  const seg = new Int32Array(copy(readFileSync(join(SEG_DIR, f))));
  let merged = 0, freeformGroups = 0, inexactShare = null;
  if (MERGE) {
    const { facets, adjacency } = preparePaintPositions(Float64Array.from(P));
    let wall = null;
    if (edgeRunner) {
      await annotateEdges(facets, adjacency, edgeRunner, edgeMeta.columns);
      const key = { fpr1: 'fpr0.01', fpr2: 'fpr0.02', fpr5: 'fpr0.05' }[EDGE];
      wall = edgeWall(1 / (1 + Math.exp(-edgeMeta.thresholds[key].logit)));
    }
    const r = freeformMerge(seg, facets, adjacency, { crease: CREASE, wall,
      ...(FF.curv !== undefined ? { curvature: { abs: 0.1, rel: 0.8 }, curvatureFrac: FF.curv } : {}),
      ...(FF.small !== undefined ? { joinSmall: !!FF.small } : {}), ...(FF.lowpoly ? { skipLowPoly: true } : {}),
      ...(FF.share !== undefined ? { minInexactShare: FF.share } : {}), ...(FF.singles ? { joinedOnly: false } : {}), ...(FF.absorb !== undefined ? { absorbSmall: !!FF.absorb } : {}),
      ...(FF.judged ? { shareOver: 'judged' } : {}), ...(FF.keepflat ? { keepFlat: true } : {}),
      ...(FF.islands ? { islands: FF.islands } : {}), ...(FF.robust ? { robust: true } : {}),
      ...(FF.cfrac !== undefined ? { creaseFrac: FF.cfrac } : {}) });
    inexactShare = r.inexactShare ?? 0;
    merged = r.merged;
    freeformGroups = r.freeform.size;
  }
  const n = F.length;
  const area = new Float64Array(n);
  for (let t = 0; t < n; t++) {
    const o = t * 9;
    const u = [P[o + 3] - P[o], P[o + 4] - P[o + 1], P[o + 5] - P[o + 2]], v = [P[o + 6] - P[o], P[o + 7] - P[o + 1], P[o + 8] - P[o + 2]];
    area[t] = 0.5 * Math.hypot(u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]);
  }
  const faces = new Map(), groups = new Map();
  for (let t = 0; t < n; t++) {
    const fc = faces.get(F[t]) ?? faces.set(F[t], { area: 0, typ: Ty[t], by: new Map() }).get(F[t]);
    fc.area += area[t];
    if (seg[t] >= 0) {
      fc.by.set(seg[t], (fc.by.get(seg[t]) ?? 0) + area[t]);
      groups.set(seg[t], (groups.get(seg[t]) ?? 0) + area[t]);
    }
  }
  const tally = { freeform: { faces: 0, once: 0, pieces: 0, area: 0, onceArea: 0 }, analytic: { faces: 0, once: 0, pieces: 0, area: 0, onceArea: 0 } };
  for (const fc of faces.values()) {
    const k = FREEFORM.has(fc.typ) ? 'freeform' : 'analytic';
    const t = tally[k];
    t.faces++; t.area += fc.area;
    let L = -1, la = 0;
    for (const [g, a] of fc.by) if (a > la) { la = a; L = g; }
    t.pieces += [...fc.by.values()].filter((a) => a >= 0.05 * fc.area).length;
    if (L >= 0 && la >= 0.95 * fc.area && la >= 0.95 * groups.get(L)) { t.once++; t.onceArea += fc.area; }
  }
  rows.push({ stem, faces: faces.size, groups: groups.size, merged, freeformGroups, inexactShare, ...tally });
}

const report = (rs, label) => {
  const sum = (k, f) => rs.reduce((s, r) => s + r[k][f], 0);
  for (const k of ['freeform', 'analytic']) {
    const faces = sum(k, 'faces');
    if (!faces) continue;
    console.log(`${label.padEnd(8)} ${k.padEnd(8)} faces ${String(faces).padStart(5)}  once ${(100 * sum(k, 'once') / faces).toFixed(1).padStart(5)}%` +
      `  once by area ${(100 * sum(k, 'onceArea') / sum(k, 'area')).toFixed(1).padStart(5)}%  pieces per face ${(sum(k, 'pieces') / faces).toFixed(2)}`);
  }
};
const meshers = [...new Set(rows.map((r) => r.stem.match(/_(native|nurbs|gmsh|game)$/)?.[1]).filter(Boolean))];
if (meshers.length > 1) for (const m of meshers) report(rows.filter((r) => r.stem.endsWith(`_${m}`)), m);
else report(rows, 'all');
const gpf = rows.map((r) => r.groups / r.faces).sort((a, b) => a - b);
console.log(`parts ${rows.length}  groups per true face: median ${gpf[gpf.length >> 1]?.toFixed(2)}  mean ${(gpf.reduce((a, b) => a + b, 0) / gpf.length).toFixed(2)}`);
if (JSON_OUT) writeFileSync(JSON_OUT, JSON.stringify(rows));
