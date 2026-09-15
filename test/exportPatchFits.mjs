/**
 * exportPatchFits.mjs — the fitted surface of every labelled patch, for ml/solid_rebuild.py.
 *
 * Usage:  node test/exportPatchFits.mjs <outDir> [parts] [meshers]
 *
 * Patches are the true B-rep faces and their type the true (canonical) type, i.e. a perfect
 * classifier / painting session. The surface is what the export path fits to that patch
 * (surfaceFitter.classifyGroupAsGauss, as studio.js exportStep calls it). One JSON per part x
 * mesher:
 *
 *   { part, mesher, stem, faces: [{ id, type, params | null }] }
 *
 * type is 'nurbs' for faces that are not an analytic primitive; params is null when the fit
 * degenerated. The mesh itself is read by the Python side from the fill_eval .pos/.fid files.
 */

import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';

const REPO = fileURLToPath(new URL('..', import.meta.url));
const { classifyGroupAsGauss } = await import(pathToFileURL(join(REPO, 'js/surfaceFitter.js')).href);

const [OUT, PARTS = '1e9', MESHERS = 'native,gmsh,game'] = process.argv.slice(2);
if (!OUT) { console.error('usage: node test/exportPatchFits.mjs <outDir> [parts] [meshers]'); process.exit(2); }
mkdirSync(OUT, { recursive: true });
const MESH_DIR = process.env.FILL_EVAL_DIR || 'D:/abc-data/fill_eval';
const TRUTH_DIR = process.env.FIT_TRUTH_DIR || 'D:/abc-data/fit_truth';
const KIND = ['plane', 'cylinder', 'cone', 'sphere', 'torus'];
const copy = b => b.buffer.slice(b.byteOffset, b.byteOffset + b.length);

const manifest = JSON.parse(readFileSync(join(MESH_DIR, 'manifest.json'), 'utf8'));
let files = 0;
const t0 = performance.now();
for (const entry of manifest.parts.slice(0, Number(PARTS))) {
  let truth;
  try { truth = new Float64Array(copy(readFileSync(join(TRUTH_DIR, entry.part + '.srf')))); } catch { continue; }
  for (const mesher of MESHERS.split(',')) {
    const v = entry.variants[mesher];
    if (!v) continue;
    const P = new Float32Array(copy(readFileSync(join(MESH_DIR, v.stem + '.pos'))));
    const F = new Int32Array(copy(readFileSync(join(MESH_DIR, v.stem + '.fid'))));
    const geometry = { attributes: { position: { array: P, count: P.length / 3,
      getX: i => P[i * 3], getY: i => P[i * 3 + 1], getZ: i => P[i * 3 + 2] } }, index: null };
    const byFace = new Map();
    for (let t = 0; t < F.length; t++) {
      if (!byFace.has(F[t])) byFace.set(F[t], []);
      byFace.get(F[t]).push(t);
    }
    const faces = [];
    for (const [id, tris] of [...byFace].sort((a, b) => a[0] - b[0])) {
      const canon = truth[id * 12 + 1];
      const type = canon <= 4 ? KIND[canon] : 'nurbs';
      if (type === 'nurbs') { faces.push({ id, type, params: null }); continue; }
      const s = classifyGroupAsGauss(type, { triangleIndices: new Set(tris) }, geometry);
      faces.push({ id, type, params: s.type === type && !s.degenerate ? s.params : null });
    }
    writeFileSync(join(OUT, `${v.stem}.json`), JSON.stringify({ part: entry.part, mesher, stem: v.stem, faces }));
    files++;
  }
}
console.log(`wrote ${files} patch-fit files to ${OUT} in ${((performance.now() - t0) / 1000).toFixed(0)} s`);
