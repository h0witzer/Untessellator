/**
 * freeform.js — a whole freeform mesh (sculpt, SubD or T-spline tessellation, scan) as a NURBS solid per body, no
 * kernel: QuadriFlow quad cage (retopo.js), cage fitted so its G2 surface lies on the mesh (fitCage.js), bi-3
 * patches with bi-6 caps at star points (fastNurbs.js), STEP written directly (patchStep.js), preview from the
 * patches (preview.js). Bodies are the studio's (js/bodies.js): each closed one becomes a solid; open bodies and
 * bodies with voids are refused with a reason.
 *
 * Meshes whose triangles QuadriFlow cannot follow (CAD slivers) are evened out first (isotropic.js).
 *
 * In:  positions (9 per triangle, the studio's export soup); bodies (a body id per triangle) and names ({id: name})
 *      optional; quads: the quad count to aim for per body (default from the body's vertex count, at most 1000:
 *      about four mesh vertices per cage vertex, so the data determines the cage).
 * Out: {step, preview, report: {bodies: [{name, triangles, quads, faces, stars, subdivisions, rms, max, relRms, relMax,
 *      conditioned?, error?}], ms}}: rms and max the fitted surface's distance to the body's mesh vertices in model
 *      units, rel* the same over the body's diagonal (a body with sharp edges or points shows here: the surface is G2
 *      everywhere); conditioned: the triangle count of the evened-out mesh the cage came from.
 */
import { findBodies } from '../bodies.js';
import { quadRemesh } from './retopo.js';
import { isotropicRemesh, projector } from './isotropic.js';
import { fitCage, fitMap } from './fitCage.js';
import { groupCage } from './groupCage.js';
import { groupLinearMap } from './groupNurbs.js';
import { patchStep } from './patchStep.js';
import { patchPreview } from './preview.js';

/** The triangle soup welded on exact coordinates: { V (3 per vertex), idx (3 per triangle) }. */
function weld(positions) {
  const ids = new Map(), V = [], idx = new Int32Array(positions.length / 3);
  for (let i = 0; i < idx.length; i++) {
    const x = positions[i * 3], y = positions[i * 3 + 1], z = positions[i * 3 + 2], key = `${x},${y},${z}`;
    let v = ids.get(key);
    if (v === undefined) { v = V.length / 3; ids.set(key, v); V.push(x, y, z); }
    idx[i] = v;
  }
  return { V: Float64Array.from(V), idx };
}

export async function freeformSolid(positions, { bodies = null, names = null, groups = null, part = 'Part', quads = null, onStatus = () => {} } = {}) {
  const t0 = performance.now();
  const { V, idx } = weld(positions);
  const found = findBodies(idx, V);
  const bodyOf = bodies ?? found.bodyOf;
  const byBody = new Map();
  for (let t = 0; t < bodyOf.length; t++) {
    if (!byBody.has(bodyOf[t])) byBody.set(bodyOf[t], []);
    byBody.get(bodyOf[t]).push(t);
  }
  const solids = [], report = [];
  for (const [id, tris] of byBody) {
    const name = names?.[id] ?? `Body ${solids.length + report.filter((r) => r.error).length + 1}`;
    const row = { name, triangles: tris.length };
    report.push(row);
    const shells = new Set(tris.map((t) => found.shellOf[t]));
    if ([...shells].some((s) => !found.shells[s].closed)) { row.error = 'open: the freeform solid needs a closed mesh'; continue; }
    if (shells.size > 1) { row.error = 'has a cavity or a second shell: not supported by the freeform solid yet'; continue; }
    // the body's triangles, wound outward, on its own vertices
    const local = new Map(), vertices = [], faces = [];
    const at = (v) => { let i = local.get(v); if (i === undefined) { i = vertices.length; local.set(v, i); vertices.push([V[v * 3], V[v * 3 + 1], V[v * 3 + 2]]); } return i; };
    for (const t of tris) {
      const [a, b, c] = [idx[t * 3], idx[t * 3 + 1], idx[t * 3 + 2]];
      faces.push(found.flip[t] ? [at(a), at(c), at(b)] : [at(a), at(b), at(c)]);
    }
    const target = quads ?? Math.max(100, Math.min(1000, Math.round(vertices.length / 4)));
    const labels = groups ? tris.map((t) => groups[t]) : null;
    if (labels && new Set(labels).size > 1) {
      try {
        const r = groupSolid(vertices, faces, labels, target, onStatus, name);
        if (r.error) { row.error = r.error; continue; }
        Object.assign(row, r.row);
        solids.push({ name, faces: r.fit.faces, loops: r.fit.loops, faceGroup: r.faceGroup });
      } catch (err) {
        row.error = String(err?.message ?? err);
      }
      continue;
    }
    try {
      onStatus(`${name}: quad cage…`);
      const ta = performance.now();
      const cage = await cageFor(vertices, faces, target, onStatus, name, row);
      if (cage.error) { row.error = cage.error; continue; }
      onStatus(`${name}: fitting ${cage.faces.length} quads…`);
      const tb = performance.now();
      const data = quadrature(vertices, faces, Math.sqrt(meshArea(vertices, faces) / cage.faces.length) / 2);
      const fit = fitCage(cage, data.points, { rounds: 4, weights: data.weights });
      const tc = performance.now();
      const h = fit.stats.history[fit.stats.history.length - 1], diag = fitDiag(vertices);
      Object.assign(row, { quads: cage.faces.length, faces: fit.faces.length, stars: starCount(cage.faces),
        subdivisions: fit.stats.subdivisions, rms: h.rms * diag, max: h.max * diag, relRms: h.rms, relMax: h.max,
        ms: { remesh: tb - ta, fit: tc - tb } });
      solids.push({ name, faces: fit.faces, loops: fit.loops });
    } catch (err) {
      row.error = String(err?.message ?? err);
    }
  }
  if (!solids.length) return { step: null, preview: null, report: { bodies: report, ms: { total: performance.now() - t0 } } };
  onStatus('Writing STEP…');
  const td = performance.now();
  const step = patchStep(solids, { name: part });
  const te = performance.now();
  const preview = patchPreview(solids);
  const tf = performance.now();
  return { step, preview, solids, report: { bodies: report, ms: { step: te - td, preview: tf - te, total: tf - t0 } } };
}

/**
 * A body painted in groups (its faces): a cage laid out from the groups' borders (groupCage.js), converted with
 * sharp borders (groupNurbs.js) and fitted with each group's surface points held to its own faces.
 */
function groupSolid(vertices, faces, labels, target, onStatus, name) {
  const ta = performance.now();
  onStatus(`${name}: cage from its faces…`);
  const cage = groupCage(vertices, faces, labels, { quads: target });
  if (cage.error) return cage;
  const map = groupLinearMap(cage.vertices.length, cage.groups, { ends: cage.ends });
  const h = Math.sqrt(meshArea(vertices, faces) / map.faces.length) / 2;
  const points = [], weights = [], pointGroup = [];
  cage.groups.forEach((g, gi) => {
    const d = quadrature(vertices, g.triangles.map((t) => faces[t]), h);
    d.points.forEach((p, i) => { points.push(p); weights.push(d.weights[i]); pointGroup.push(gi); });
  });
  onStatus(`${name}: fitting ${cage.groups.length} faces…`);
  const tb = performance.now();
  // two-sided: each group's faces are also pulled onto that group's triangles
  const onGroup = cage.groups.map((g) => {
    const T = g.triangles.map((t) => faces[t]);
    let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (const t of T) for (const v of t) for (let d = 0; d < 3; d++) { lo[d] = Math.min(lo[d], vertices[v][d]); hi[d] = Math.max(hi[d], vertices[v][d]); }
    return projector(vertices, T, Math.max(Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) / 24, 1e-9));
  });
  const fit = fitMap(map, cage.vertices, cage.groups.flatMap((g) => g.faces), points,
    { rounds: 4, weights: Float64Array.from(weights), pointGroup: Int32Array.from(pointGroup), project: (g, p) => onGroup[g](p) });
  const tc = performance.now();
  const last = fit.stats.history[fit.stats.history.length - 1], diag = fitDiag(vertices);
  return { fit, faceGroup: map.faceGroup, row: { groups: cage.groups.length, n: cage.n, quads: map.faces.length, faces: fit.faces.length,
    kinds: cage.groups.reduce((m, g) => ({ ...m, [g.kind]: (m[g.kind] ?? 0) + 1 }), {}), subdivisions: 0, rms: last.rms * diag, max: last.max * diag,
    relRms: last.rms, relMax: last.max, ms: { remesh: tb - ta, fit: tc - tb } } };
}

/**
 * QuadriFlow's cage for the body. Its field follows the input's triangles: an edge longer than twice the quad size
 * gets split into a fan of slivers it cannot follow (CAD exports' long flat faces), so such a mesh, or one QuadriFlow
 * fails on, is first remeshed isotropically to half the quad size (isotropic.js). A mesh of fine detail is left as it
 * is: remeshing would smooth what is finer than the quads.
 */
async function cageFor(vertices, faces, target, onStatus, name, row) {
  let area = 0, longest = 0;
  for (const [a, b, c] of faces) {
    const P = [vertices[a], vertices[b], vertices[c]];
    const u = [0, 1, 2].map((i) => P[1][i] - P[0][i]), w = [0, 1, 2].map((i) => P[2][i] - P[0][i]);
    area += Math.hypot(u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]) / 2;
    for (let k = 0; k < 3; k++) longest = Math.max(longest, Math.hypot(...[0, 1, 2].map((i) => P[(k + 1) % 3][i] - P[k][i])));
  }
  const quadSize = Math.sqrt(area / target);
  const conditioned = () => {
    onStatus(`${name}: evening out the triangles…`);
    const m = isotropicRemesh(vertices, faces, quadSize / 2);
    row.conditioned = m.faces.length;
    onStatus(`${name}: quad cage…`);
    return quadRemesh(m.vertices, m.faces, { faces: target });
  };
  if (longest > 2 * quadSize) return conditioned();
  const cage = await quadRemesh(vertices, faces, { faces: target });
  return cage.error ? conditioned() : cage;
}

function meshArea(vertices, faces) {
  let area = 0;
  for (const [a, b, c] of faces) area += triArea(vertices[a], vertices[b], vertices[c]);
  return area;
}
function triArea(A, B, C) {
  const u = [B[0] - A[0], B[1] - A[1], B[2] - A[2]], w = [C[0] - A[0], C[1] - A[1], C[2] - A[2]];
  return Math.hypot(u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]) / 2;
}

/**
 * Points and weights for the integral of squared distance over the mesh, no two more than about h apart: a
 * triangle no longer than h gives a third of its area to each corner; a longer one (a CAD face's long triangles,
 * whose corners are all on the face's edges) is bisected on its longest edge until every piece is, and each
 * piece's centroid carries the piece's area. h is the quadrature's spacing (half the quad size, so every
 * patch has data inside it), not a smoothing weight.
 */
function quadrature(vertices, faces, h) {
  const weight = new Float64Array(vertices.length), extra = [], extraWeight = [];
  const len = (p, q) => Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
  const mid = (p, q) => [(p[0] + q[0]) / 2, (p[1] + q[1]) / 2, (p[2] + q[2]) / 2];
  for (const [ia, ib, ic] of faces) {
    const A = vertices[ia], B = vertices[ib], C = vertices[ic];
    if (Math.max(len(A, B), len(B, C), len(C, A)) <= h) {
      const a = triArea(A, B, C) / 3;
      weight[ia] += a; weight[ib] += a; weight[ic] += a;
      continue;
    }
    const stack = [[A, B, C]];
    while (stack.length) {
      const [p, q, r] = stack.pop(), e0 = len(p, q), e1 = len(q, r), e2 = len(r, p), m = Math.max(e0, e1, e2);
      if (m <= h) {
        extra.push([(p[0] + q[0] + r[0]) / 3, (p[1] + q[1] + r[1]) / 3, (p[2] + q[2] + r[2]) / 3]);
        extraWeight.push(triArea(p, q, r));
      } else if (e0 === m) { const s = mid(p, q); stack.push([p, s, r], [s, q, r]); }
      else if (e1 === m) { const s = mid(q, r); stack.push([p, q, s], [p, s, r]); }
      else { const s = mid(r, p); stack.push([p, q, s], [s, q, r]); }
    }
  }
  const points = [], weights = [];
  vertices.forEach((v, i) => { if (weight[i] > 0) { points.push(v); weights.push(weight[i]); } });
  return { points: points.concat(extra), weights: Float64Array.from(weights.concat(extraWeight)) };
}

function starCount(faces) {
  const deg = new Map();
  for (const q of faces) for (const v of q) deg.set(v, (deg.get(v) ?? 0) + 1);
  let n = 0;
  for (const d of deg.values()) if (d !== 4) n++;
  return n;
}

function fitDiag(P) {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const p of P) for (let c = 0; c < 3; c++) { lo[c] = Math.min(lo[c], p[c]); hi[c] = Math.max(hi[c], p[c]); }
  return Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
}
