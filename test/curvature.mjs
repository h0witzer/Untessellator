/**
 * curvature.mjs — check facet curvature against surfaces whose answer is known exactly.
 *
 * Usage:  node test/curvature.mjs
 *
 * Curvature estimation on a mesh is approximate by nature, so the only way to know
 * whether it is good enough to fit with is to generate surfaces whose principal
 * curvatures are known in closed form and compare.
 *
 *     plane      k1 = 0,     k2 = 0
 *     cylinder   k1 = 1/R,   k2 = 0
 *     sphere     k1 = 1/R,   k2 = 1/R
 *     cone       k2 = 0 along the ruling; k1 = cos(a) / d at axial distance d
 *     torus      k1 = 1/r EVERYWHERE, k2 = cos(v) / (R + r cos v)
 *
 * The torus row is the reason this module exists. 1/r holds at every point regardless of
 * where it sits, so the minor radius is readable from a patch far too small for the
 * tube-centre relation in gaussMap.js to see a circle.
 *
 * Tolerances below are loose on purpose. The estimator uses area-weighted vertex normals
 * from a discrete mesh, so it converges with tessellation density rather than being
 * exact; what matters is whether the error is small enough to SEED a fit, not whether it
 * could replace one.
 */

import { buildFacets } from '../js/meshCore.js';
import { facetCurvature, regionCurvature } from '../js/curvature.js';

const push = (out, a, b, c) => { out.push(...a, ...b, ...c); };

function meshPlane(nu = 40, nv = 40, size = 2) {
  const out = [];
  const at = (i, j) => [size * (i / nu - 0.5), size * (j / nv - 0.5), 0];
  for (let i = 0; i < nu; i++) for (let j = 0; j < nv; j++) {
    push(out, at(i, j), at(i + 1, j), at(i + 1, j + 1));
    push(out, at(i, j), at(i + 1, j + 1), at(i, j + 1));
  }
  return new Float64Array(out);
}

function meshCylinder(R = 1.3, H = 3, nu = 96, nv = 24) {
  const out = [];
  const at = (i, j) => {
    const th = 2 * Math.PI * i / nu;
    return [R * Math.cos(th), R * Math.sin(th), H * (j / nv - 0.5)];
  };
  for (let i = 0; i < nu; i++) for (let j = 0; j < nv; j++) {
    push(out, at(i, j), at(i + 1, j), at(i + 1, j + 1));
    push(out, at(i, j), at(i + 1, j + 1), at(i, j + 1));
  }
  return new Float64Array(out);
}

function meshSphere(R = 0.85, nu = 96, nv = 48) {
  const out = [];
  const at = (i, j) => {
    const th = 2 * Math.PI * i / nu, ph = Math.PI * j / nv;
    return [R * Math.sin(ph) * Math.cos(th), R * Math.sin(ph) * Math.sin(th), R * Math.cos(ph)];
  };
  for (let i = 0; i < nu; i++) for (let j = 0; j < nv; j++) {
    push(out, at(i, j), at(i + 1, j), at(i + 1, j + 1));
    push(out, at(i, j), at(i + 1, j + 1), at(i, j + 1));
  }
  return new Float64Array(out);
}

function meshTorus(R = 1.0, r = 0.4, nu = 128, nv = 64) {
  const out = [];
  const at = (i, j) => {
    const u = 2 * Math.PI * i / nu, v = 2 * Math.PI * j / nv;
    return [(R + r * Math.cos(v)) * Math.cos(u), (R + r * Math.cos(v)) * Math.sin(u), r * Math.sin(v)];
  };
  for (let i = 0; i < nu; i++) for (let j = 0; j < nv; j++) {
    push(out, at(i, j), at(i + 1, j), at(i + 1, j + 1));
    push(out, at(i, j), at(i + 1, j + 1), at(i, j + 1));
  }
  return new Float64Array(out);
}

function meshCone(halfAngle = Math.PI / 4, z0 = 0.5, z1 = 2.0, nu = 96, nv = 24) {
  const out = [];
  const tan = Math.tan(halfAngle);
  const at = (i, j) => {
    const th = 2 * Math.PI * i / nu, z = z0 + (z1 - z0) * (j / nv), rr = z * tan;
    return [rr * Math.cos(th), rr * Math.sin(th), z];
  };
  for (let i = 0; i < nu; i++) for (let j = 0; j < nv; j++) {
    push(out, at(i, j), at(i + 1, j), at(i + 1, j + 1));
    push(out, at(i, j), at(i + 1, j + 1), at(i, j + 1));
  }
  return new Float64Array(out);
}

function check(label, positions, expect) {
  const facets = buildFacets(positions);
  const curv = facetCurvature(positions, facets);
  const all = new Set();
  for (let t = 0; t < facets.count; t++) all.add(t);
  const s = regionCurvature(all, curv);
  if (!s) { console.log(`  ${label.padEnd(30)} no valid curvature`); return false; }
  const a1 = Math.abs(s.k1), a2 = Math.abs(s.k2);
  const e1 = Math.abs(a1 - expect.k1), e2 = Math.abs(a2 - expect.k2);
  const tol1 = Math.max(0.05 * Math.max(expect.k1, 1e-9), 0.05);
  const tol2 = Math.max(0.05 * Math.max(expect.k2, 1e-9), 0.05);
  const ok = e1 < tol1 && e2 < tol2;
  console.log(`  ${label.padEnd(30)} |k1| ${a1.toFixed(4).padStart(8)} (want ${expect.k1.toFixed(4)})` +
    `   |k2| ${a2.toFixed(4).padStart(8)} (want ${expect.k2.toFixed(4)})   ${ok ? 'ok' : 'FAIL'}`);
  return ok;
}

console.log('\nfacet curvature vs closed form\n');
let pass = true;
pass = check('plane', meshPlane(), { k1: 0, k2: 0 }) && pass;
pass = check('cylinder R=1.3', meshCylinder(1.3), { k1: 1 / 1.3, k2: 0 }) && pass;
pass = check('sphere R=0.85', meshSphere(0.85), { k1: 1 / 0.85, k2: 1 / 0.85 }) && pass;
pass = check('torus R=1.0 r=0.4', meshTorus(1.0, 0.4), { k1: 1 / 0.4, k2: 0 }) && pass;
pass = check('torus R=2.0 r=0.25', meshTorus(2.0, 0.25), { k1: 1 / 0.25, k2: 0 }) && pass;
pass = check('cone 45 deg', meshCone(Math.PI / 4), { k1: Math.abs(Math.cos(Math.PI / 4)) / 1.25, k2: 0 }) && pass;

// ── the point of the exercise ────────────────────────────────────────────────
//
// Can a SMALL patch of a torus give the minor radius, where the tube-centre relation
// in gaussMap.js needs enough of the tube to see a circle?
console.log('\nminor radius from a small patch — the case fitTorus cannot do\n');
{
  const R = 1.0, r = 0.4, nu = 128, nv = 64;
  const positions = meshTorus(R, r, nu, nv);
  const facets = buildFacets(positions);
  const curv = facetCurvature(positions, facets);
  for (const frac of [1.0, 0.25, 0.05, 0.01, 0.004]) {
    const want = Math.max(8, Math.round(facets.count * frac));
    // a contiguous block of the (u,v) grid, i.e. a genuine patch rather than a scatter
    const region = new Set();
    const perU = 2 * nv;
    const cols = Math.max(1, Math.ceil(want / perU));
    for (let i = 0; i < cols && region.size < want; i++) {
      for (let k = 0; k < perU && region.size < want; k++) region.add(i * perU + k);
    }
    const s = regionCurvature(region, curv);
    if (!s) { console.log(`  ${String(region.size).padStart(6)} facets  no estimate`); continue; }
    const rEst = 1 / Math.abs(s.k1);
    console.log(`  ${String(region.size).padStart(6)} facets (${(100 * region.size / facets.count).toFixed(1).padStart(5)}% of the torus)` +
      `   r estimate ${rEst.toFixed(4)}   error ${((rEst - r) / r * 100).toFixed(2).padStart(6)}%`);
  }
}

console.log(pass ? '\nPASS\n' : '\nFAIL\n');
process.exit(pass ? 0 : 1);
