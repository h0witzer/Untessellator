/**
 * guideBracket.mjs — writes samples/SampleBracket.guide.json, the data the guide (js/studio/guide.js) runs on:
 * the bracket's answer key (each triangle's true face and each face's type) and the paint with mistakes planted in
 * it that the guide's second round loads.
 *
 * Usage:  node test/studioExport.mjs <dir> samples/SampleBracket.stl --dump
 *         node test/guideBracket.mjs <dir>/SampleBracket.fid <dir>/SampleBracket.fits.json
 *
 * The key is Auto-paint's grouping of the bracket, which is the true one: the one-shot export of that grouping
 * matches the Onshape STEP face for face (24 of 24, worst face 5 nm off, ml/step_match.py), so it is checked here
 * against that STEP's face types before it is written. The planted paint is made from the key by four rules, each
 * a mistake the guide teaches one fix for (measured in the page on f2b7aa2):
 *   spill   the top face's triangles along one countersink painted into the countersink's cone: the export check
 *           flags the cone, Find fixes re-fills them into the top face
 *   split   the bottom face split into two plane groups: the check passes it, Find fixes merges them
 *   bend    the bend's fillet painted into the top face: the check flags the top face, Find fixes can't mend it, so
 *           it is fixed by hand (Delete group, then the bucket as Cylinder and as Plane)
 *   hole    the big hole typed as a cone: a cone fits it to a half angle of 1e-4 rad, so nothing flags it; Color by
 *           Type shows it, and Fit as Cylinder fixes it
 * Faces are found by type and size, not by group id, so a rerun of the dump gives the same data.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const STL = join(ROOT, 'samples', 'SampleBracket.stl');
const OUT = join(ROOT, 'samples', 'SampleBracket.guide.json');
// the Onshape STEP's faces by type (ml/step_match.py on SampleBracket.step)
const TRUTH = { plane: 8, cylinder: 10, cone: 4, torus: 2 };

const [fidPath, fitsPath] = process.argv.slice(2);
if (!fidPath || !fitsPath) { console.error('usage: node test/guideBracket.mjs <SampleBracket.fid> <SampleBracket.fits.json>'); process.exit(1); }

// the mesh, as the studio opens it: a binary STL, triangles in file order
const buf = readFileSync(STL);
const n = buf.readUInt32LE(80);
if (buf.length !== 84 + 50 * n) throw new Error('SampleBracket.stl is not a binary STL');
const P = new Float32Array(n * 9);
for (let t = 0; t < n; t++) for (let k = 0; k < 9; k++) P[t * 9 + k] = buf.readFloatLE(84 + 50 * t + 12 + 4 * k);

const fidBuf = readFileSync(fidPath);
const fid = new Int32Array(fidBuf.buffer, fidBuf.byteOffset, fidBuf.length / 4);
if (fid.length !== n) throw new Error(`the dump has ${fid.length} triangles, the STL ${n}`);
const typeOf = new Map(JSON.parse(readFileSync(fitsPath, 'utf8')).faces.map((f) => [f.id, f.type]));

// faces numbered by their first triangle
const faceOf = new Map();
const key = new Int32Array(n);
for (let t = 0; t < n; t++) {
  if (fid[t] < 0) throw new Error(`triangle ${t} is unpainted in the dump`);
  if (!faceOf.has(fid[t])) faceOf.set(fid[t], faceOf.size);
  key[t] = faceOf.get(fid[t]);
}
const faces = [...faceOf.keys()].map((g) => typeOf.get(g));
const counts = {};
for (const type of faces) counts[type] = (counts[type] ?? 0) + 1;
if (JSON.stringify(Object.entries(counts).sort()) !== JSON.stringify(Object.entries(TRUTH).sort())) {
  throw new Error(`the dump's faces ${JSON.stringify(counts)} are not the STEP's ${JSON.stringify(TRUTH)}: not a key`);
}

const area = (t) => {
  const o = t * 9;
  const ax = P[o + 3] - P[o], ay = P[o + 4] - P[o + 1], az = P[o + 5] - P[o + 2];
  const bx = P[o + 6] - P[o], by = P[o + 7] - P[o + 1], bz = P[o + 8] - P[o + 2];
  return 0.5 * Math.hypot(ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx);
};
const centroid = (t) => [0, 1, 2].map((k) => (P[t * 9 + k] + P[t * 9 + 3 + k] + P[t * 9 + 6 + k]) / 3);
const faceArea = faces.map(() => 0);
const faceCentre = faces.map(() => [0, 0, 0]);
for (let t = 0; t < n; t++) {
  const a = area(t), c = centroid(t);
  faceArea[key[t]] += a;
  for (let k = 0; k < 3; k++) faceCentre[key[t]][k] += a * c[k];
}
faceCentre.forEach((c, f) => { for (let k = 0; k < 3; k++) c[k] /= faceArea[f]; });
const ranked = (type) => faces.map((ty, f) => f).filter((f) => faces[f] === type).sort((a, b) => faceArea[b] - faceArea[a]);

// the faces the mistakes are planted on
const [bottom, top] = ranked('plane');                 // the base's underside (3,852 mm2), then its top (2,940 mm2)
const [fillet, hole] = ranked('cylinder');             // the bend's fillet (471 mm2), then the upright's big hole (251 mm2)
const cone = ranked('cone').sort((a, b) => faceCentre[b][0] - faceCentre[a][0] || faceCentre[b][1] - faceCentre[a][1])[0];

// triangle adjacency by shared corners (the STL's corners are exact)
const corner = new Map();
const vid = new Int32Array(n * 3);
for (let i = 0; i < n * 3; i++) {
  const k = `${P[i * 3]},${P[i * 3 + 1]},${P[i * 3 + 2]}`;
  if (!corner.has(k)) corner.set(k, corner.size);
  vid[i] = corner.get(k);
}
const byEdge = new Map();
for (let t = 0; t < n; t++) {
  for (let e = 0; e < 3; e++) {
    const a = vid[t * 3 + e], b = vid[t * 3 + ((e + 1) % 3)];
    const k = a < b ? `${a},${b}` : `${b},${a}`;
    (byEdge.get(k) ?? byEdge.set(k, []).get(k)).push(t);
  }
}
const touches = (t, f) => {
  for (let e = 0; e < 3; e++) {
    const a = vid[t * 3 + e], b = vid[t * 3 + ((e + 1) % 3)];
    if (byEdge.get(a < b ? `${a},${b}` : `${b},${a}`).some((u) => u !== t && key[u] === f)) return true;
  }
  return false;
};

const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
for (let i = 0; i < n * 3; i++) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], P[i * 3 + k]); hi[k] = Math.max(hi[k], P[i * 3 + k]); }
const midX = (lo[0] + hi[0]) / 2;

const spill = [], half = [];
for (let t = 0; t < n; t++) {
  if (key[t] === top && touches(t, cone)) spill.push(t);
  if (key[t] === bottom && centroid(t)[0] > midX + 5) half.push(t);
}

// the planted paint: the key's faces, the bottom's far half as a group of its own, the fillet in the top, the spill
// in the cone, the hole typed as a cone
const types = [...faces, 'plane'];
types[hole] = 'cone';
const half2 = faces.length;
const spilled = new Set(spill), halved = new Set(half);
// (the fillet's own group ends up with no triangles; the loader makes groups only for the ones that have some)
const owner = Array.from(key, (f, t) => (spilled.has(t) ? cone : halved.has(t) ? half2 : f === fillet ? top : f));

const round = (v) => Math.round(v * 10) / 10;
const data = {
  about: 'The guide\'s sample: an Onshape coarse STL of a bracket, its true faces (key) and a paint with four planted mistakes (test/guideBracket.mjs).',
  file: 'samples/SampleBracket.stl',
  triangles: n,
  faces,
  roles: { top, bottom, fillet, hole, cone },
  areas: Object.fromEntries(Object.entries({ top, bottom, fillet, hole, cone }).map(([k, f]) => [k, round(faceArea[f])])),
  key: Array.from(key),
  spill,
  half,
  planted: { types, owner },
};
writeFileSync(OUT, `${JSON.stringify(data)}\n`);
console.log(`wrote ${OUT}: ${n} triangles, ${faces.length} faces (${JSON.stringify(counts)}); spill ${spill.length}, half ${half.length};`,
  `top ${round(faceArea[top])}, bottom ${round(faceArea[bottom])}, fillet ${round(faceArea[fillet])}, hole ${round(faceArea[hole])}, cone ${round(faceArea[cone])} mm2`);
