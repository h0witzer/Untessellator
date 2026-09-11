/**
 * classify.mjs — headless A/B of the original classifier against the Gauss-map
 * classifier, scored on fixtures with exact analytic ground truth.
 *
 * Usage:  node test/makeFixtures.mjs && node test/classify.mjs
 *
 * The original classifier is exercised twice:
 *   [file]   normals taken from the STL as the app does today (the fixtures
 *            write zero facet normals, which real exporters commonly emit)
 *   [geom]   normals recomputed from vertex winding
 * so the normal-provenance defect and the cone-axis defect are visible apart
 * from one another rather than confounded.
 */

import { readFileSync, readdirSync } from 'node:fs';
import { dirname, join, basename } from 'node:path';
import { fileURLToPath } from 'node:url';

import { buildFacets, buildAdjacency, segment } from '../js/meshCore.js';
import { classifyRegion } from '../js/gaussMap.js';
import { classifyGroup } from '../js/surfaceFitter.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const FIX = join(HERE, 'fixtures');
const DEG = 180 / Math.PI;
const CREASE = 10;   // below the 15 deg cone/cylinder junction in cone-sweep

// ── Binary STL reader ────────────────────────────────────────────────────────

function readSTL(path) {
  const buf = readFileSync(path);
  const tris = buf.readUInt32LE(80);
  const positions = new Float64Array(tris * 9);
  const fileNormals = new Float64Array(tris * 9);
  for (let t = 0; t < tris; t++) {
    const o = 84 + t * 50;
    const nx = buf.readFloatLE(o), ny = buf.readFloatLE(o + 4), nz = buf.readFloatLE(o + 8);
    for (let v = 0; v < 3; v++) {
      fileNormals[t * 9 + v * 3] = nx;
      fileNormals[t * 9 + v * 3 + 1] = ny;
      fileNormals[t * 9 + v * 3 + 2] = nz;
    }
    for (let i = 0; i < 9; i++) positions[t * 9 + i] = buf.readFloatLE(o + 12 + i * 4);
  }
  return { tris, positions, fileNormals };
}

/** Minimal BufferAttribute stand-in so the original classifier runs unmodified. */
const attr = arr => ({
  array: arr,
  count: arr.length / 3,
  getX: i => arr[i * 3],
  getY: i => arr[i * 3 + 1],
  getZ: i => arr[i * 3 + 2],
});

/** Per-vertex normals replicated from per-facet geometric normals. */
function facetNormalsAsVertexNormals(facets) {
  const out = new Float64Array(facets.count * 9);
  for (let t = 0; t < facets.count; t++)
    for (let v = 0; v < 3; v++) {
      out[t * 9 + v * 3] = facets.normals[t * 3];
      out[t * 9 + v * 3 + 1] = facets.normals[t * 3 + 1];
      out[t * 9 + v * 3 + 2] = facets.normals[t * 3 + 2];
    }
  return out;
}

// ── Scoring ──────────────────────────────────────────────────────────────────

/** Compare a classification against the ground-truth segment it covers. */
function score(truth, result) {
  const got = result.type;
  if (got !== truth.type) return { ok: false, detail: got };

  const p = result.params || {};
  if (truth.type === 'cone') {
    const err = Math.abs(p.halfAngle * DEG - truth.halfAngleDeg);
    return { ok: err < 0.5, detail: `${(p.halfAngle * DEG).toFixed(2)}° (err ${err.toFixed(2)}°)` };
  }
  if (truth.type === 'cylinder' || truth.type === 'sphere') {
    const err = Math.abs(p.radius - truth.radius);
    const rel = err / truth.radius;
    return { ok: rel < 0.01, detail: `r=${p.radius.toFixed(3)} (err ${(rel * 100).toFixed(2)}%)` };
  }
  return { ok: true, detail: 'ok' };
}

function run(name) {
  const { tris, positions, fileNormals } = readSTL(join(FIX, name + '.stl'));
  const { truth, segOf } = JSON.parse(readFileSync(join(FIX, name + '.truth.json'), 'utf8'));

  const facets = buildFacets(positions);
  const adjacency = buildAdjacency(positions, facets);
  const groups = segment(adjacency, facets.count, CREASE);

  const geomNormals = facetNormalsAsVertexNormals(facets);
  const geoFile = { attributes: { position: attr(positions), normal: attr(fileNormals) } };
  const geoGeom = { attributes: { position: attr(positions), normal: attr(geomNormals) } };

  console.log(`\n${'='.repeat(78)}\n${name}  —  ${tris} triangles, ` +
              `${groups.length} regions at ${CREASE}° crease\n${'='.repeat(78)}`);
  console.log(
    'truth'.padEnd(20) + 'old [file]'.padEnd(13) + 'old [geom]'.padEnd(13) + 'NEW (gauss map)');
  console.log('-'.repeat(78));

  const tally = { oldFile: 0, oldGeom: 0, gauss: 0, total: 0 };

  for (const region of groups) {
    // Attribute each region to the ground-truth segment holding most of its area.
    const votes = new Map();
    for (const t of region) votes.set(segOf[t], (votes.get(segOf[t]) || 0) + facets.areas[t]);
    const segs = [...votes.entries()].sort((a, b) => b[1] - a[1]);
    const seg = segs[0][0];
    const pure = segs.length === 1;
    const gt = truth[seg];
    if (!gt) continue;

    const group = { triangleIndices: region };
    const oldF = classifyGroup(group, geoFile);
    const oldG = classifyGroup(group, geoGeom);
    const nw = classifyRegion(region, facets);

    const label = gt.type === 'cone' ? `cone ${gt.halfAngleDeg.toFixed(0)}°`
      : gt.type === 'cylinder' || gt.type === 'sphere' ? `${gt.type} r=${gt.radius.toFixed(2)}`
      : gt.type;

    const sF = score(gt, oldF), sG = score(gt, oldG), sN = score(gt, nw);
    tally.total++;
    if (sF.ok) tally.oldFile++;
    if (sG.ok) tally.oldGeom++;
    if (sN.ok) tally.gauss++;

    const mark = s => (s.ok ? '✓ ' : '✗ ');
    console.log(
      (label + (pure ? '' : ' *merged')).padEnd(20) +
      (mark(sF) + oldF.type).padEnd(13) +
      (mark(sG) + oldG.type).padEnd(13) +
      mark(sN) + nw.type.padEnd(9) + (sN.ok ? sN.detail : `got ${sN.detail}`));
  }

  console.log('-'.repeat(78));
  console.log(`score: old[file] ${tally.oldFile}/${tally.total}   ` +
              `old[geom] ${tally.oldGeom}/${tally.total}   ` +
              `NEW ${tally.gauss}/${tally.total}`);
  return tally;
}

const names = readdirSync(FIX).filter(f => f.endsWith('.stl')).map(f => basename(f, '.stl'));
const totals = { oldFile: 0, oldGeom: 0, gauss: 0, total: 0 };
for (const n of names) {
  const t = run(n);
  for (const k of Object.keys(totals)) totals[k] += t[k];
}
console.log(`\n${'#'.repeat(78)}`);
console.log(`TOTAL   old[file] ${totals.oldFile}/${totals.total}   ` +
            `old[geom] ${totals.oldGeom}/${totals.total}   ` +
            `NEW ${totals.gauss}/${totals.total}`);
console.log('#'.repeat(78));
