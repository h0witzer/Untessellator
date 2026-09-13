/**
 * fitbias.mjs — do the fitters invert the tessellator, or lose to it?
 *
 * Usage:  node test/fitbias.mjs
 *
 * A CAD tessellator places its vertices exactly ON the surface and inscribes triangles
 * between them. So the surface is fully recoverable in principle: the vertices are exact
 * samples of it. Any error a fitter shows on a clean synthetic primitive is therefore
 * self-inflicted, and this measures how much.
 *
 * THE TELL IS DENSITY DEPENDENCE. A fitter that works from vertices should recover its
 * parameters to near machine precision at every tessellation density. One that works
 * from facet centroids inherits the sagitta — centroids of an inscribed triangle lie
 * strictly INSIDE a convex surface, by O(theta^2) — so its error shrinks by roughly 4x
 * each time the density doubles. That signature is unmistakable and is what this test
 * looks for.
 *
 * Not every use of a centroid is a bug, which is why this measures rather than greps.
 * A facet's PLANE is exactly determined by its three vertices and the centroid lies in
 * it, so relations built on tangent planes — the cone's apex-from-concurrency — are
 * exact with centroids. Relations that step ALONG a normal are not: the point where a
 * facet's normal is the true surface normal is not the centroid.
 */

import { buildFacets } from '../js/meshCore.js';
import { fitRegionAs } from '../js/gaussMap.js';
import { attachCurvature } from '../js/curvature.js';

const push = (out, a, b, c) => { out.push(...a, ...b, ...c); };
const grid = (nu, nv, at, wrapU = true, wrapV = true) => {
  const out = [];
  for (let i = 0; i < nu; i++) for (let j = 0; j < nv; j++) {
    const i1 = wrapU ? (i + 1) % nu : i + 1, j1 = wrapV ? (j + 1) % nv : j + 1;
    push(out, at(i, j), at(i1, j), at(i1, j1));
    push(out, at(i, j), at(i1, j1), at(i, j1));
  }
  return new Float64Array(out);
};

const cylinder = (R, H) => (nu, nv) => grid(nu, nv, (i, j) => {
  const th = 2 * Math.PI * i / nu;
  return [R * Math.cos(th), R * Math.sin(th), H * (j / nv - 0.5)];
}, true, false);

const sphere = (R) => (nu, nv) => grid(nu, nv, (i, j) => {
  const th = 2 * Math.PI * i / nu, ph = Math.PI * j / nv;
  return [R * Math.sin(ph) * Math.cos(th), R * Math.sin(ph) * Math.sin(th), R * Math.cos(ph)];
}, true, false);

const torus = (R, r) => (nu, nv) => grid(nu, nv, (i, j) => {
  const u = 2 * Math.PI * i / nu, v = 2 * Math.PI * j / nv;
  return [(R + r * Math.cos(v)) * Math.cos(u), (R + r * Math.cos(v)) * Math.sin(u), r * Math.sin(v)];
});

const cone = (half, z0, z1) => (nu, nv) => grid(nu, nv, (i, j) => {
  const th = 2 * Math.PI * i / nu, z = z0 + (z1 - z0) * (j / nv), rr = z * Math.tan(half);
  return [rr * Math.cos(th), rr * Math.sin(th), z];
}, true, false);

const plane = (size) => (nu, nv) => grid(nu, nv, (i, j) =>
  [size * (i / nu - 0.5), size * (j / nv - 0.5), 0], false, false);

const DENSITIES = [[24, 8], [48, 16], [96, 32], [192, 64]];

function run(label, make, type, measure) {
  console.log(`\n  ${label}`);
  console.log(`    ${'facets'.padStart(8)}  ${'recovered'.padStart(30)}  ${'rel error'.padStart(11)}   ratio`);
  let prev = null;
  for (const [nu, nv] of DENSITIES) {
    const positions = make(nu, nv);
    const facets = buildFacets(positions);
    attachCurvature(positions, facets);
    const region = new Set();
    for (let t = 0; t < facets.count; t++) region.add(t);
    const fit = fitRegionAs(type, region, facets);
    if (!fit.params) { console.log(`    ${String(facets.count).padStart(8)}  DEGENERATE`); prev = null; continue; }
    const { text, err } = measure(fit.params);
    const ratio = prev && err > 0 ? (prev / err).toFixed(1) + 'x' : '  -';
    console.log(`    ${String(facets.count).padStart(8)}  ${text.padStart(30)}  ${err.toExponential(2).padStart(11)}   ${ratio.padStart(6)}`);
    prev = err;
  }
}

console.log('\ninverting the tessellator: parameter error vs density');
console.log('a ~4x ratio per density step means the fitter is eating the sagitta;');
console.log('a flat ratio near machine precision means it is reading the vertices.');

run('plane (z = 0)', plane(2), 'plane', (p) => ({
  text: `|n.z| ${Math.abs(p.normal[2]).toFixed(12)}`,
  err: Math.abs(1 - Math.abs(p.normal[2])) + Math.abs(p.origin[2]),
}));

run('cylinder R = 1.3', cylinder(1.3, 3), 'cylinder', (p) => ({
  text: `R ${p.radius.toFixed(10)}`, err: Math.abs(p.radius - 1.3) / 1.3,
}));

run('cone half-angle 30 deg', cone(Math.PI / 6, 0.5, 2.0), 'cone', (p) => ({
  text: `half ${(p.halfAngle * 180 / Math.PI).toFixed(8)} deg`,
  err: Math.abs(p.halfAngle - Math.PI / 6) / (Math.PI / 6),
}));

run('sphere R = 0.85', sphere(0.85), 'sphere', (p) => ({
  text: `R ${p.radius.toFixed(10)}`, err: Math.abs(p.radius - 0.85) / 0.85,
}));

run('torus R = 1.0, r = 0.4', torus(1.0, 0.4), 'torus', (p) => ({
  text: `R ${p.majorRadius.toFixed(6)} r ${p.minorRadius.toFixed(6)}`,
  err: Math.max(Math.abs(p.majorRadius - 1.0), Math.abs(p.minorRadius - 0.4) / 0.4),
}));

console.log('');
