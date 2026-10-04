/**
 * knit.mjs — js/knit (the studio's solid builder) on one part's patches, the same input
 * ml/solid_rebuild.py --fid-dir takes, so the port can be held against the Python measurements.
 *
 * Usage:  node test/knit.mjs <mesh dir> <patch dir> <stem> [--out row.json] [--step out.step]
 *                            [--no-loop] [--no-fallback] [--budget 300] [--deadline s] [--faceted] [--no-split] [--no-axes]
 *                            [--no-tidy] [--no-exact-edges] [--no-contact] [--axes-opts '{"TANGENT_DEG":0}']
 *
 *   <mesh dir>   <stem>.pos (float32 triangle soup)
 *   <patch dir>  <stem>.fid (a group per triangle) and <stem>.json (test/exportPatchFits.mjs format:
 *                {faces: [{id, type, params}]}), as test/surfaceScore.mjs --patches-out writes them
 *
 * One part per process, as the Python harness does: run many with test/knitRebuild.mjs.
 */

import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { loadOcct, knitSolid, writeStep, BUILD } from '../js/knit/index.js';

const args = process.argv.slice(2);
const flag = (name, dflt) => { const i = args.indexOf(name); if (i < 0) return dflt; const v = args[i + 1]; args.splice(i, 2); return v; };
const bool = (name) => { const i = args.indexOf(name); if (i < 0) return false; args.splice(i, 1); return true; };
const OUT = flag('--out', null);
const STEP = flag('--step', null);
const BUDGET = Number(flag('--budget', '300'));
const NO_LOOP = bool('--no-loop');
const NO_SPLIT = bool('--no-split');
const NO_AXES = bool('--no-axes');
const NO_TIDY = bool('--no-tidy');
const AXES_OPTS = JSON.parse(flag('--axes-opts', '{}'));
const NO_EXACT_EDGES = bool('--no-exact-edges');
const NO_CONTACT = bool('--no-contact');
if (NO_EXACT_EDGES) BUILD.EXACT_EDGES = false;
if (NO_CONTACT) BUILD.CONTACT = false;
const NO_FALLBACK = bool('--no-fallback');
const DEADLINE = Number(flag('--deadline', 'Infinity'));
const FACETED = bool('--faceted');
const [MESH_DIR, PATCH_DIR, STEM] = args;
if (!STEM) { console.error('usage: node test/knit.mjs <mesh dir> <patch dir> <stem>'); process.exit(2); }

const copy = (b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.length);
const positions = Float64Array.from(new Float32Array(copy(readFileSync(join(MESH_DIR, `${STEM}.pos`)))));
const labels = new Int32Array(copy(readFileSync(join(PATCH_DIR, `${STEM}.fid`))));
// <stem>.fits.json (test/studioExport.mjs --dump) or <stem>.json (surfaceScore --patches-out)
const fitsPath = existsSync(join(PATCH_DIR, `${STEM}.fits.json`)) ? join(PATCH_DIR, `${STEM}.fits.json`) : join(PATCH_DIR, `${STEM}.json`);
const fits = JSON.parse(readFileSync(fitsPath, 'utf8'));
const surfaces = new Map(fits.faces.filter((f) => f.params).map((f) => [f.id, [f.type, f.params]]));

const t0 = performance.now();
const oc = await loadOcct();
const loadSecs = (performance.now() - t0) / 1000;
let row;
try {
  const { solid, report } = knitSolid(oc, { positions, labels, surfaces }, { loop: !NO_LOOP, fallback: !NO_FALLBACK, budget: BUDGET, deadline: DEADLINE, mode: FACETED ? 'faceted' : 'knit', split: !NO_SPLIT, axes: !NO_AXES, axesOptions: AXES_OPTS, tidy: !NO_TIDY });
  row = { stem: STEM, ...report, load_secs: Math.round(loadSecs * 10) / 10 };
  if (STEP && solid) writeFileSync(STEP, writeStep(oc, solid));
} catch (err) {
  row = { stem: STEM, tier: 'none', closed: false, valid: false, exception: String(err?.stack ?? err).slice(0, 300) };
}
if (OUT) writeFileSync(OUT, JSON.stringify(row));
console.log(JSON.stringify({ ...row, info: undefined }));
