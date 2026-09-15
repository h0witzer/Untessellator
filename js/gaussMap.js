/**
 * gaussMap.js — analytic surface classification via normal-space (Gauss map)
 * hypothesis generation plus positional least-squares verification.
 *
 * WHY THIS REPLACES THE ORIGINAL classifyGroup():
 *
 * The original derived a cylinder/cone axis from the smallest eigenvector of the
 * UNCENTERED second moment of the normals, Sum(n n^T). For a cone of half-angle a
 * that matrix has M_axis = N sin^2(a) and M_inplane = N cos^2(a)/2, so the axis is
 * only the smallest eigenvector while sin^2(a) < cos^2(a)/2, i.e. a < 35.26 deg.
 * Above that the returned "axis" is an arbitrary in-plane direction and the fit
 * silently fails. That threshold excludes every countersink (41-60 deg), every
 * 45 deg chamfer, and every drill point (59 deg) — i.e. essentially every cone
 * that appears in a machined part.
 *
 * The fix is to fit a plane in NORMAL space using the CENTERED covariance:
 *
 *     n . u = d      with u = smallest eigenvector of cov(n), d = mean(n) . u
 *
 *     |d| ~ 1, residual ~ 0  -> plane
 *     |d| ~ 0, residual ~ 0  -> cylinder, axis = u
 *     0 < |d| < 1, res ~ 0   -> cone, axis = u, half-angle = asin|d|
 *     residual large         -> sphere or freeform
 *
 * This recovers the axis for a cone of ANY half-angle, and works on partial
 * patches (the original derived the half-angle from the mean normal, which is
 * only valid for a full 360 deg revolution).
 *
 * The Gauss map is used only to PROPOSE candidates. Every candidate is then
 * refit positionally and scored by area-weighted RMS point-to-surface distance,
 * because normal-space geometry alone is degenerate for narrow arcs: a 20 deg
 * cylindrical band is nearly indistinguishable from a wide cone in normal space.
 * Positional verification resolves those cases, and removes the need for the
 * original's hand-tuned bias table and magic radius/spread thresholds.
 */

import {
  eigen3, solve3, dot, cross, sub, norm, normalize, perp,
} from './meshCore.js';

/** Gather area-weighted facet data for one region. */
export function gather(triIndices, facets) {
  const idx = [...triIndices];
  const n = idx.length;
  const P = new Float64Array(n * 3);   // centroids
  const N = new Float64Array(n * 3);   // unit geometric normals
  const W = new Float64Array(n);       // areas (weights)
  let wSum = 0;

  for (let i = 0; i < n; i++) {
    const t = idx[i];
    P[i * 3] = facets.centroids[t * 3];
    P[i * 3 + 1] = facets.centroids[t * 3 + 1];
    P[i * 3 + 2] = facets.centroids[t * 3 + 2];
    N[i * 3] = facets.normals[t * 3];
    N[i * 3 + 1] = facets.normals[t * 3 + 1];
    N[i * 3 + 2] = facets.normals[t * 3 + 2];
    W[i] = facets.areas[t];
    wSum += facets.areas[t];
  }

  // Welded vertices of the same region, for POSITIONAL terms only.
  //
  // Normals stay on the facets. Averaging unit normals at a vertex and renormalising
  // shortens the average, which on a cone inflates |n . axis| and therefore biases the
  // half-angle upward -- the exact quantity the axis criterion depends on. So position
  // comes from vertices (unbiased) and orientation from facets (exact): each term uses
  // whichever sampling is correct for it.
  //
  // Each vertex carries a third of its incident in-region facet area, which answers
  // meshCore's reason for preferring centroids in the first place: raw vertex counting
  // over-weights densely tessellated regions, area weighting does not.
  let VP = null, VW = null, vn = 0;
  if (facets.positions) {
    const acc = new Map();
    for (let i = 0; i < n; i++) {
      const t = idx[i];
      for (let k = 0; k < 3; k++) {
        const o = t * 9 + k * 3;
        const x = facets.positions[o], y = facets.positions[o + 1], z = facets.positions[o + 2];
        // EXACT coordinates, not a quantised key.
        //
        // buildAdjacency welds with an absolute quantum (1e4 by default), which is
        // right for topology but wrong here: WatchStand's radii run 0.001-0.025 model
        // units, so a 1e-4 quantum is several percent of a small feature and merges
        // genuinely distinct vertices. Doing that collapsed a true r=0.003 cylinder
        // to a fitted 0.053.
        //
        // Welding is only a de-duplication convenience for positional fitting, and the
        // two errors are not symmetric: UNDER-welding just lists a vertex twice with
        // half the weight each, which changes nothing, while OVER-welding moves points
        // off the surface. So match exact coordinates and accept the occasional
        // duplicate. A tessellator emits identical values for a shared vertex.
        const key = x + '_' + y + '_' + z;
        const e = acc.get(key);
        if (e) e.w += W[i] / 3;
        else acc.set(key, { x, y, z, w: W[i] / 3 });
      }
    }
    vn = acc.size;
    VP = new Float64Array(vn * 3);
    VW = new Float64Array(vn);
    let j = 0;
    for (const e of acc.values()) {
      VP[j * 3] = e.x; VP[j * 3 + 1] = e.y; VP[j * 3 + 2] = e.z; VW[j] = e.w; j++;
    }
  }
  // Curvature hint, when the caller has attached one to `facets` (see
  // curvature.js attachCurvature). Medians rather than means, because curvature on a
  // discrete mesh is noisy and a proposed region is never perfectly pure.
  let kMajor = null;
  if (facets.curvature && facets.curvature.valid) {
    const a = [];
    for (let i = 0; i < n; i++) {
      const t = idx[i];
      if (facets.curvature.valid[t]) a.push(Math.abs(facets.curvature.k1[t]));
    }
    if (a.length >= 3) { a.sort((x, y) => x - y); kMajor = a[a.length >> 1]; }
  }

  return { n, P, N, W, wSum, vn, VP, VW, kMajor };
}

/** Positional sample set: welded vertices when available, facet centroids otherwise. */
function positional(g) {
  return g.vn ? { n: g.vn, P: g.VP, W: g.VW, wSum: g.VW.reduce((a, b) => a + b, 0) }
              : { n: g.n, P: g.P, W: g.W, wSum: g.wSum };
}

/** Area-weighted mean of a flat vec3 array. */
function weightedMean3(A, W, n, wSum) {
  let x = 0, y = 0, z = 0;
  for (let i = 0; i < n; i++) {
    x += W[i] * A[i * 3];
    y += W[i] * A[i * 3 + 1];
    z += W[i] * A[i * 3 + 2];
  }
  return [x / wSum, y / wSum, z / wSum];
}

/** Area-weighted covariance (about the weighted mean) of a flat vec3 array. */
function weightedCov3(A, W, n, wSum, mean) {
  let c00 = 0, c01 = 0, c02 = 0, c11 = 0, c12 = 0, c22 = 0;
  for (let i = 0; i < n; i++) {
    const w = W[i];
    const x = A[i * 3] - mean[0];
    const y = A[i * 3 + 1] - mean[1];
    const z = A[i * 3 + 2] - mean[2];
    c00 += w * x * x; c01 += w * x * y; c02 += w * x * z;
    c11 += w * y * y; c12 += w * y * z; c22 += w * z * z;
  }
  const s = 1 / wSum;
  return [c00 * s, c01 * s, c02 * s, c11 * s, c12 * s, c22 * s];
}

// ── Positional fits ──────────────────────────────────────────────────────────

function fitPlane(g) {
  const { N } = g;
  const { n, P, W, wSum } = positional(g);          // vertices: on the surface
  const origin = weightedMean3(P, W, n, wSum);
  const { vectors } = eigen3(weightedCov3(P, W, n, wSum, origin));
  let normal = vectors[0];

  // PCA sign is arbitrary; orient outward to agree with the facet normals.
  const meanN = weightedMean3(N, g.W, g.n, g.wSum);
  if (dot(normal, meanN) < 0) normal = [-normal[0], -normal[1], -normal[2]];

  let e = 0;
  for (let i = 0; i < n; i++) {
    const d = (P[i * 3] - origin[0]) * normal[0]
            + (P[i * 3 + 1] - origin[1]) * normal[1]
            + (P[i * 3 + 2] - origin[2]) * normal[2];
    e += W[i] * d * d;
  }
  return { type: 'plane', params: { origin, normal }, residual: Math.sqrt(e / wSum) };
}

/** Axial coordinate + radial distance of every facet centroid about an axis. */
function axialFrame(g, axis, origin) {
  const { n, P } = g.P ? g : positional(g);
  const Z = new Float64Array(n);
  const R = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const dx = P[i * 3] - origin[0];
    const dy = P[i * 3 + 1] - origin[1];
    const dz = P[i * 3 + 2] - origin[2];
    const z = dx * axis[0] + dy * axis[1] + dz * axis[2];
    const rx = dx - z * axis[0], ry = dy - z * axis[1], rz = dz - z * axis[2];
    Z[i] = z;
    R[i] = Math.sqrt(rx * rx + ry * ry + rz * rz);
  }
  return { Z, R };
}

function fitCylinder(g, axis) {
  const { n, P, W, wSum } = positional(g);          // vertices: on the surface
  const origin = weightedMean3(P, W, n, wSum);
  const xh = perp(axis);
  const yh = normalize(cross(axis, xh));

  // Weighted algebraic circle fit in the plane perpendicular to the axis:
  // x^2 + y^2 = 2 cx x + 2 cy y + k,  k = R^2 - cx^2 - cy^2
  const A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  const b = [0, 0, 0];
  const U = new Float64Array(n), V = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    const dx = P[i * 3] - origin[0];
    const dy = P[i * 3 + 1] - origin[1];
    const dz = P[i * 3 + 2] - origin[2];
    const z = dx * axis[0] + dy * axis[1] + dz * axis[2];
    const rx = dx - z * axis[0], ry = dy - z * axis[1], rz = dz - z * axis[2];
    const u = rx * xh[0] + ry * xh[1] + rz * xh[2];
    const v = rx * yh[0] + ry * yh[1] + rz * yh[2];
    U[i] = u; V[i] = v;

    const row = [2 * u, 2 * v, 1];
    const rhs = u * u + v * v;
    const w = W[i];
    for (let j = 0; j < 3; j++) {
      b[j] += w * rhs * row[j];
      for (let k = 0; k < 3; k++) A[j][k] += w * row[j] * row[k];
    }
  }

  const sol = solve3(A, b);
  if (!sol) return null;
  const r2 = sol[2] + sol[0] * sol[0] + sol[1] * sol[1];
  if (!(r2 > 0) || !isFinite(r2)) return null;
  const radius = Math.sqrt(r2);

  const axisPoint = [
    origin[0] + sol[0] * xh[0] + sol[1] * yh[0],
    origin[1] + sol[0] * xh[1] + sol[1] * yh[1],
    origin[2] + sol[0] * xh[2] + sol[1] * yh[2],
  ];

  let e = 0;
  for (let i = 0; i < n; i++) {
    const du = U[i] - sol[0], dv = V[i] - sol[1];
    const d = Math.sqrt(du * du + dv * dv) - radius;
    e += W[i] * d * d;
  }
  return {
    type: 'cylinder',
    params: { axis, axisPoint, radius },
    residual: Math.sqrt(e / wSum),
  };
}

/**
 * Apex by tangent-plane concurrency — no axis, no search, one 3x3 solve.
 *
 * Every tangent plane of a cone contains the apex, because the ruling through the
 * touch point passes through it. A standard tessellator samples a cone at constant
 * angles and two heights, so every triangle has one edge lying ALONG a ruling, and a
 * plane containing that edge contains the apex exactly. Measured on a cleanly
 * tessellated 59 deg cone: |n . (apex - centroid)| has median 8.3e-10 and max 1.3e-8,
 * i.e. the relation is exact, not approximate.
 *
 * So the apex is the common point of all facet planes:  n_j . x = n_j . p_j.
 * Least squares gives the normal equations  (sum w n n^T) x = sum w (n.p) n.
 *
 * This matters because it removes the degeneracy that made cones hard. Fitting a cone
 * by distance alone lets the axis tilt, the apex slide and the half-angle open together
 * while the surface barely moves through a bounded patch — which is how a 45 deg cone
 * was fitted as 86 deg with a LOWER residual than the truth. Pinning the apex first
 * removes that whole direction from the problem.
 *
 * Conditioning note: cone normals lie on a plane in normal space that does not pass
 * through the origin, so they span R^3 and the system is full rank — but on a narrow
 * arc they are nearly collinear, and the solve degrades. Callers must check the result
 * rather than assume it.
 */
export function coneApexFromPlanes(g) {
  const { n, P, N, W } = g;
  const A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  const b = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    const w = W[i];
    const nx = N[i * 3], ny = N[i * 3 + 1], nz = N[i * 3 + 2];
    const d = nx * P[i * 3] + ny * P[i * 3 + 1] + nz * P[i * 3 + 2];
    const r = [nx, ny, nz];
    for (let j = 0; j < 3; j++) {
      b[j] += w * d * r[j];
      for (let k = 0; k < 3; k++) A[j][k] += w * r[j] * r[k];
    }
  }
  const x = solve3(A, b);
  return x && isFinite(x[0]) && isFinite(x[1]) && isFinite(x[2]) ? x : null;
}

const TRIM_FRACTION = 0.2;

export function fitCone(g, axis) {
  const { n, N, W, wSum } = g;                      // facets: exact normals
  const pos = positional(g);                        // vertices: on the surface
  const origin = weightedMean3(pos.P, pos.W, pos.n, pos.wSum);

  // HALF-ANGLE FROM THE NORMALS, not from an R-vs-Z regression.
  //
  // The regression needs the patch to span enough AXIAL extent for the radius to
  // visibly change. A narrow annular band has almost constant R, so the slope comes
  // out ~0 and the fit is rejected as "no taper" even when the patch is perfectly
  // conical — which is exactly how a 59 deg cone with 30 facets, sitting on its true
  // axis, was reported degenerate (test/fitoracle.mjs).
  //
  // Every outward normal of a cone makes the same angle with the axis:
  // |n . a| = sin(halfAngle), independently of where the facet sits. That holds on
  // any patch, however narrow, and needs no axial extent at all. It is the Gauss-map
  // relation this module is named for, applied per facet rather than to the mean.
  // TWO ESTIMATES OF THE HALF-ANGLE, scored against each other rather than ranked a
  // priori, because each is better than the other in a different regime.
  //
  //   from NORMALS   |n . axis| = sin(halfAngle). Needs no apex, so it cannot inherit
  //                  an apex error - robust on a contaminated region. But it uses the
  //                  normal of a SECANT facet, tilted from the true surface normal by
  //                  O(theta^2), so it eats the sagitta: measured error fell exactly 4x
  //                  per density step, 7.09e-3 at 384 facets to 1.11e-4 at 24,576.
  //
  //   from VERTICES  every vertex lies exactly on the cone, so the angle between
  //                  (v - apex) and the axis IS the half-angle, with no approximation -
  //                  1.5e-15 on a synthetic cone at every density. But it DEPENDS on the
  //                  apex, and on a real region a slightly wrong apex makes it worse
  //                  than the normals: adopting it outright took fitoracle's cones from
  //                  4/4 to 2/4 and paint.mjs from 1/4 to 0/4.
  //
  // So compute both and keep whichever explains the points better. Recorded because the
  // obvious reading - "vertices are on the surface, therefore always prefer them" - is
  // exactly the conclusion that regressed.
  let sAbs = 0;
  for (let i = 0; i < n; i++) {
    sAbs += W[i] * Math.abs(N[i * 3] * axis[0] + N[i * 3 + 1] * axis[1] + N[i * 3 + 2] * axis[2]);
  }
  const angles = [Math.asin(Math.min(1, Math.max(0, sAbs / wSum)))];

  const planeApex = coneApexFromPlanes(g);

  // GATE THE VERTEX ESTIMATE ON WHETHER THE PLANES ACTUALLY CONCUR.
  //
  // It is only exact when the mesh really is an inscribed tessellation of a cone, with
  // an edge of every triangle lying along a ruling. That is true of a CAD tessellator's
  // output and false of a downloaded mesh scored against a reconstructed STEP, and the
  // difference is directly measurable as how nearly the facet planes pass through the
  // solved apex. Measured, relative to patch extent:
  //
  //     synthetic inscribed cone .......... 8.4e-16
  //     WatchStand 59 deg, clean patch .... 2.1e-6
  //     WatchStand 45 deg ................. 8.8e-3
  //     WatchStand 9.85 deg ............... 3.9e-2 / 6.0e-2
  //
  // Three orders of magnitude of daylight, so the exact cut hardly matters; 1e-4 sits in
  // the gap. Without this gate the vertex estimate is offered everywhere and the
  // residual picks it on the 45 deg cone, giving 47.08 where the normals give 44.98.
  let concur = Infinity;
  if (planeApex) {
    const gp = g.P;
    let acc = 0, accw = 0;
    for (let i = 0; i < n; i++) {
      acc += W[i] * Math.abs(
        N[i * 3] * (planeApex[0] - gp[i * 3]) +
        N[i * 3 + 1] * (planeApex[1] - gp[i * 3 + 1]) +
        N[i * 3 + 2] * (planeApex[2] - gp[i * 3 + 2]));
      accw += W[i];
    }
    let ext = 0;
    for (let i = 0; i < n; i++) {
      const d = norm(sub([gp[i * 3], gp[i * 3 + 1], gp[i * 3 + 2]], origin));
      if (d > ext) ext = d;
    }
    if (accw > 0 && ext > 0) concur = (acc / accw) / ext;
  }

  if (planeApex && pos.n >= 3 && concur < 1e-4) {
    for (const sgn of [1, -1]) {
      let acc = 0, accw = 0;
      for (let i = 0; i < pos.n; i++) {
        const dx = pos.P[i * 3] - planeApex[0], dy = pos.P[i * 3 + 1] - planeApex[1],
              dz = pos.P[i * 3 + 2] - planeApex[2];
        const L = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (!(L > 1e-14)) continue;
        const c = sgn * (dx * axis[0] + dy * axis[1] + dz * axis[2]) / L;
        acc += pos.W[i] * Math.acos(Math.min(1, Math.max(-1, c)));
        accw += pos.W[i];
      }
      if (accw > 0) {
        const a = acc / accw;
        if (a > 1e-4 && a < 1.5533) { angles.push(a); break; }
      }
    }
  }

  let halfAngle = null, ax = null, apex = null, bestCost = Infinity;
  for (const cand of angles) {
    if (!(cand > 1e-4) || cand > 1.5533) continue;          // (0, 89 deg)
    const tanA = Math.tan(cand);

  // Orientation is not determined by the half-angle, so place the apex for both
  // senses of the axis and keep whichever explains the region better. Cheaper and
  // more reliable than inferring it from a normal's sign, which flips on a mesh
  // whose winding is inconsistent.
  // Apex candidates: the tangent-plane solve (exact where the tessellation is clean —
  // measured 1.4e-9 on a 59 deg cone) and the axial average as a fallback for when the
  // plane system is ill-conditioned on a narrow arc. Both are scored below.

  for (const sg of [1, -1]) {
    const cnd = [axis[0] * sg, axis[1] * sg, axis[2] * sg];
    const fr = axialFrame(pos, cnd, origin);
    // On the cone, R = tan(a) * (Z - z0), so each sample votes for an apex position.
    let z0 = 0;
    for (let i = 0; i < pos.n; i++) z0 += pos.W[i] * (fr.Z[i] - fr.R[i] / tanA);
    z0 /= pos.wSum;
    const axialApex = [origin[0] + z0 * cnd[0], origin[1] + z0 * cnd[1],
                       origin[2] + z0 * cnd[2]];
    for (const ap of (planeApex ? [planeApex, axialApex] : [axialApex])) {
      const ca = Math.cos(cand), sa = Math.sin(cand);
      const f2 = axialFrame(pos, cnd, ap);
    let e = 0;
    for (let i = 0; i < pos.n; i++) {
      const d = f2.R[i] * ca - f2.Z[i] * sa;
      e += pos.W[i] * d * d;
    }
      if (e < bestCost) { bestCost = e; ax = cnd; apex = ap; halfAngle = cand; }
    }
    }
  }
  if (!apex || !isFinite(apex[0]) || halfAngle === null) return null;

  // Perpendicular distance from (r, z) to the cone surface in the half-plane.
  const ca = Math.cos(halfAngle), sa = Math.sin(halfAngle);
  const fr = axialFrame(pos, ax, apex);

  // Region scale, so the angular term below lands in the same units as distance.
  let s2 = 0;
  for (let i = 0; i < pos.n; i++) {
    const dx = pos.P[i * 3] - origin[0], dy = pos.P[i * 3 + 1] - origin[1],
          dz = pos.P[i * 3 + 2] - origin[2];
    s2 += pos.W[i] * (dx * dx + dy * dy + dz * dz);
  }
  const scale = Math.sqrt(s2 / pos.wSum) || 1;

  // Positional error, plus how far each facet normal is from the cone's own
  // normal there. See SCORE vs RESIDUAL below for why both are needed.
  const NA = new Float64Array(n);
  let e = 0, eN = 0, sd = 0, sdd = 0;
  for (let i = 0; i < pos.n; i++) {
    const d = fr.R[i] * ca - fr.Z[i] * sa;
    e += pos.W[i] * d * d;
  }
  const { P } = g;
  for (let i = 0; i < n; i++) {
    const px = P[i * 3] - apex[0], py = P[i * 3 + 1] - apex[1], pz = P[i * 3 + 2] - apex[2];
    const z = px * ax[0] + py * ax[1] + pz * ax[2];
    let rx = px - z * ax[0], ry = py - z * ax[1], rz = pz - z * ax[2];
    const rn = Math.sqrt(rx * rx + ry * ry + rz * rz);
    if (rn > 1e-12) { rx /= rn; ry /= rn; rz /= rn; }
    const nx = rx * ca - ax[0] * sa, ny = ry * ca - ax[1] * sa, nz = rz * ca - ax[2] * sa;
    const c = N[i * 3] * nx + N[i * 3 + 1] * ny + N[i * 3 + 2] * nz;
    eN += W[i] * Math.max(0, 1 - c * c);      // sin^2 of the disagreement angle

    // n . axis, whose SPREAD is the real axis discriminator (see below).
    const na = N[i * 3] * ax[0] + N[i * 3 + 1] * ax[1] + N[i * 3 + 2] * ax[2];
    NA[i] = na;
    sd += W[i] * na;
    sdd += W[i] * na * na;
  }

  // THE AXIS TEST IS THE SPREAD OF (n . axis), NOT THE POSITIONAL RESIDUAL.
  //
  // On a cone's true axis every outward normal makes the same angle with it, so
  // n . axis is CONSTANT — that is the Gauss-map relation this module is built on,
  // and it is exact, not approximate. Tilt the axis and the value spreads out.
  //
  // Position alone cannot see this. A patch of a 45 deg cone is fitted almost as
  // closely by a wrong axis with a compensating half-angle: measured on WatchStand,
  // the positional residual at a 6.8 deg wrong axis was 8.4e-4 against 4.0e-3 at the
  // TRUE one, so a position-only search actively prefers the wrong answer and walks
  // away from a correct Gauss-map seed. Adding the variance term makes the true axis
  // the minimum, which is what it should have been all along.
  let varNA = Math.max(0, sdd / wSum - (sd / wSum) * (sd / wSum));

  // TRIMMED, because a handful of foreign facets otherwise decide the axis.
  //
  // A region proposed by any real segmenter is not perfectly pure. Measured on
  // WatchStand: a 45 deg cone region that is 90.6% pure — five foreign facets out of
  // 53 — swung the fitted axis by 68 degrees, because those five inflate the variance
  // at the TRUE axis and some other axis then scores lower. The criterion is a
  // variance, so it has no resistance to outliers at all.
  //
  // Dropping the worst TRIM_FRACTION by deviation restores it. This is not a tuned
  // constant in the sense the earlier AXIS_TRUST was: it only has to exceed the
  // impurity a segmenter realistically leaves behind, and the result is flat across a
  // wide range of values.
  if (n >= 8) {
    const mean = sd / wSum;
    const order = Array.from(NA.subarray(0, n).keys())
      .sort((i, j) => Math.abs(NA[i] - mean) - Math.abs(NA[j] - mean));
    const keep = Math.max(4, Math.floor(n * (1 - TRIM_FRACTION)));
    let tw = 0, ts = 0, tss = 0;
    for (let k = 0; k < keep; k++) {
      const i = order[k];
      tw += W[i]; ts += W[i] * NA[i]; tss += W[i] * NA[i] * NA[i];
    }
    if (tw > 0) varNA = Math.max(0, tss / tw - (ts / tw) * (ts / tw));
  }

  return {
    type: 'cone',
    params: { axis: ax, apex, halfAngle },
    residual: Math.sqrt(e / wSum),
    score: Math.sqrt(e / pos.wSum + scale * scale * (eN / wSum)),
    // The axis is chosen by this alone — see above. Exact, and free of any weight.
    axisCost: varNA,
  };
}

/**
 * Fit a torus via the TUBE-CENTRE relation.
 *
 * The obvious approach - least squares over all seven parameters (centre, axis, R, r)
 * against point-to-surface distance - fails the same way the cone did, and for the same
 * reason. The objective has a degenerate valley: tilt the axis, trade R against r, and
 * the surface stays near a bounded patch. It also has three structural escapes, since
 * R -> 0 is a sphere, R -> infinity is a cylinder, and r -> 0 makes any small patch fit.
 *
 * There is an exact relation that removes the search instead. For an outward normal n
 * at a surface point p, the centre of the tube is exactly
 *
 *     q = p - r n
 *
 * and for the CORRECT r every q lies on one circle: radius R, in the plane through the
 * centre perpendicular to the axis. So r is the value that makes {p - r n} planar, which
 * is a one-dimensional search with a well-behaved objective, and everything else then
 * follows from a plane fit and a circle fit - both linear.
 *
 * Measured on a generated torus of R=1.0, r=0.4: the planarity objective is unimodal in
 * r, minimised at r=0.400 with R=0.9999 on a fine mesh and r=0.395 / R=0.9987 on a
 * coarse one.
 *
 * KNOWN LIMIT, and it is the partial-patch problem again in a new place. The tube-centre
 * relation needs enough of the tube to see that the q form a CIRCLE. On a small patch
 * they form a short arc, and many r values make a short arc planar, so the fit degrades
 * exactly where the cone's did. Measured: fitting a whole generated torus gives a
 * relative residual of 1.5e-8, while fitting small seeded patches of the same mesh gives
 * 7.5e-2 to 2.2e-1.
 *
 * Curvature (curvature.js) was built to resolve this and only PARTLY does, which is
 * worth recording so it is not retried expecting more. The principal curvature around
 * the tube is exactly 1/r everywhere, and the estimator recovers it well: r to 2.5% from
 * a 66-facet patch, 0.4% of the surface. But r was not the binding constraint. R is, and
 * R genuinely requires seeing enough of the tube to form a circle - no local quantity can
 * supply it. End to end the hint moved seeded torus fits from 13/40 to 16/40 on a fine
 * mesh and made a coarse one slightly worse.
 *
 * Nor can the residual arbitrate between a curvature-seeded fit and a free one: on a
 * small patch a wrong (R, r) pair genuinely achieves the lower residual, which is the
 * degeneracy restated rather than a scoring bug. Growth is what supplies R - a correct r
 * at least lets growth start from something sane.
 *
 * The residual also DISCRIMINATES, which matters more than the parameters. The same
 * torus displaced by 9% of its minor radius - a sculpted doughnut rather than a
 * machined one - fits 22x worse (6.7e-2 against 3.1e-3). A sculpted surface should be
 * reported as freeform, not as a torus that happens to be nearby, and that judgement
 * belongs to the residual gate rather than to the classifier.
 */
/**
 * Centre of the circle through q_i = p_i - s n_i, in the plane through `m` normal to
 * `axis`: a weighted algebraic (Kasa) fit, u^2 + v^2 + D u + E v + F = 0, which is linear
 * in D, E, F. Returns null when the points do not constrain a circle.
 */
function spineCentre(P, N, W, n, s, axis, m) {
  const e1 = perp(axis);
  const e2 = cross(axis, e1);
  const A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  const b = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    const d = [P[i * 3] - s * N[i * 3] - m[0],
               P[i * 3 + 1] - s * N[i * 3 + 1] - m[1],
               P[i * 3 + 2] - s * N[i * 3 + 2] - m[2]];
    const u = dot(d, e1), v = dot(d, e2), w = W[i];
    const row = [u, v, 1];
    const rhs = -(u * u + v * v);
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) A[r][c] += w * row[r] * row[c];
      b[r] += w * row[r] * rhs;
    }
  }
  const x = solve3(A, b);
  if (!x || !x.every(Number.isFinite)) return null;
  const cu = -x[0] / 2, cv = -x[1] / 2;
  return [m[0] + cu * e1[0] + cv * e2[0],
          m[1] + cu * e1[1] + cv * e2[1],
          m[2] + cu * e1[2] + cv * e2[2]];
}

function fitTorus(g) {
  const { n, P, N, W, wSum } = g;
  const pos = positional(g);                          // vertices, for the refinement
  const origin = weightedMean3(P, W, n, wSum);

  // Patch extent bounds the plausible tube radius; a torus whose tube dwarfs the patch
  // is indistinguishable from a cylinder, and one far smaller is indistinguishable from
  // a plane, so there is no point searching outside this.
  let ext = 0;
  for (let i = 0; i < n; i++) {
    const d = norm(sub([P[i * 3], P[i * 3 + 1], P[i * 3 + 2]], origin));
    if (d > ext) ext = d;
  }
  if (!(ext > 0)) return null;

  // Planarity of {p - s*r*n}, plus the circle it forms.
  const probe = (r, sgn) => {
    const Q = new Float64Array(n * 3);
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < 3; k++) Q[i * 3 + k] = P[i * 3 + k] - sgn * r * N[i * 3 + k];
    }
    const m = weightedMean3(Q, W, n, wSum);
    const { values, vectors } = eigen3(weightedCov3(Q, W, n, wSum, m));
    return { planar: Math.sqrt(Math.max(0, values[0])), axis: vectors[0], centre: m };
  };

  // CURVATURE HINT. On a small patch the tube-centre relation is degenerate - the q form
  // a short arc, and many r make a short arc planar - which is the same partial-patch
  // failure the cone had. The principal curvature around the tube is exactly 1/r at
  // every point of a torus, so it pins r from a patch far too small to show a circle.
  // Measured on a generated torus: a 66-facet patch, 0.4% of the surface, gives r to
  // 2.5%. Used to BRACKET the search rather than replace it, since the estimate is
  // approximate and the planarity objective is still the more accurate of the two.
  const hint = g.kMajor && g.kMajor > 1e-12 ? 1 / g.kMajor : null;

  // Coarse scan then golden-section. The objective is unimodal in r but the scan guards
  // against the trivial minimum near r = 0, where any small patch looks planar.
  let best = null;
  const minima = [];
  for (const sgn of [1, -1]) {
    const lo = hint ? Math.max(0.02 * ext, 0.5 * hint) : 0.02 * ext;
    const hi = hint ? Math.min(4.0 * ext, 2.0 * hint) : 2.0 * ext;
    if (!(hi > lo)) continue;
    const steps = 48;
    let bi = null;
    const scan = [];
    for (let i = 0; i <= steps; i++) {
      const r = lo + (hi - lo) * (i / steps);
      const pr = probe(r, sgn);
      scan.push({ ...pr, r, sgn });
      if (!bi || pr.planar < bi.planar) bi = { ...pr, r };
    }
    for (let i = 1; i < steps; i++) {
      if (scan[i].planar <= scan[i - 1].planar && scan[i].planar <= scan[i + 1].planar) minima.push(scan[i]);
    }
    if (!bi) continue;
    const span = (hi - lo) / steps;
    let a = Math.max(lo, bi.r - span), b = Math.min(hi, bi.r + span);
    const phi = 0.6180339887;
    let c1 = b - phi * (b - a), c2 = a + phi * (b - a);
    let f1 = probe(c1, sgn).planar, f2 = probe(c2, sgn).planar;
    for (let it = 0; it < 40 && b - a > 1e-9 * ext; it++) {
      if (f1 < f2) { b = c2; c2 = c1; f2 = f1; c1 = b - phi * (b - a); f1 = probe(c1, sgn).planar; }
      else { a = c1; c1 = c2; f1 = f2; c2 = a + phi * (b - a); f2 = probe(c2, sgn).planar; }
    }
    const r = 0.5 * (a + b);
    const pr = probe(r, sgn);
    if (!best || pr.planar < best.planar) best = { ...pr, r, sgn };
  }
  if (!best || !(best.r > 0)) return null;

  // CENTRE FROM A CIRCLE FIT, NOT THE MEAN. The tube centres q = p - r n lie on the
  // spine circle, so the torus centre is that circle's centre. Their mean is the centre
  // only when q covers the whole ring, and CAD fillets rarely do: on ABC validation parts
  // they span a median 80-85 deg of it, and the mean of an arc sits inside the arc.
  // Measured against the true STEP surfaces with the true r, so only the estimator
  // differed, on 684 native tori: centre error 0.90 R and R error 61% from the mean,
  // 0.009 R and 0.7% from the circle fit (gmsh: 0.054 R and 3.8%). Rings spanning
  // 300 deg or more were fine either way, which is why generated test tori never showed it.
  const ax = normalize(best.axis);
  const c0 = spineCentre(P, N, W, n, best.sgn * best.r, ax, best.centre) ?? best.centre;

  // Refine on the VERTICES, which lie exactly on the surface where centroids do not.
  // With the axis and centre fixed, each vertex gives rho and z about the axis, and its
  // distance to the tube-centre circle is sqrt((rho-R)^2 + z^2), which should equal r.
  const rho = new Float64Array(pos.n), zz = new Float64Array(pos.n);
  let wAll = 0;
  for (let i = 0; i < pos.n; i++) {
    const dx = pos.P[i * 3] - c0[0], dy = pos.P[i * 3 + 1] - c0[1], dz = pos.P[i * 3 + 2] - c0[2];
    const z = dx * ax[0] + dy * ax[1] + dz * ax[2];
    const rx = dx - z * ax[0], ry = dy - z * ax[1], rz = dz - z * ax[2];
    rho[i] = Math.sqrt(rx * rx + ry * ry + rz * rz);
    zz[i] = z;
    wAll += pos.W[i];
  }
  const tubeDist = (R, i) => {
    const dr = rho[i] - R;
    return Math.sqrt(dr * dr + zz[i] * zz[i]);
  };
  // Free r: R is wherever the tube distances are most consistent, r their mean.
  const spread = (R) => {
    let sd = 0, sdd = 0;
    for (let i = 0; i < pos.n; i++) {
      const d = tubeDist(R, i);
      sd += pos.W[i] * d; sdd += pos.W[i] * d * d;
    }
    const mean = sd / wAll;
    return { r: mean, v: Math.max(0, sdd / wAll - mean * mean) };
  };
  // Fixed r: R is wherever the tube distances sit closest to the given r.
  const misfit = (R, rFix) => {
    let e = 0;
    for (let i = 0; i < pos.n; i++) {
      const d = tubeDist(R, i) - rFix;
      e += pos.W[i] * d * d;
    }
    return e / wAll;
  };
  const golden = (lo0, hi0, f) => {
    let a = lo0, b = hi0;
    const phi = 0.6180339887;
    let c1 = b - phi * (b - a), c2 = a + phi * (b - a);
    let f1 = f(c1), f2 = f(c2);
    for (let it = 0; it < 60 && b - a > 1e-12 * ext; it++) {
      if (f1 < f2) { b = c2; c2 = c1; f2 = f1; c1 = b - phi * (b - a); f1 = f(c1); }
      else { a = c1; c1 = c2; f1 = f2; c2 = a + phi * (b - a); f2 = f(c2); }
    }
    return 0.5 * (a + b);
  };
  let R0 = 0;
  for (let i = 0; i < pos.n; i++) R0 += pos.W[i] * rho[i];
  R0 /= wAll;

  // TWO CANDIDATES, CHOSEN BY EVIDENCE rather than by a size threshold.
  //
  // The planarity route estimates r globally and is the more accurate of the two when
  // the patch shows enough of the tube to form a circle. The curvature route estimates
  // r locally and is the only one that works when it does not. Rather than guess where
  // the crossover is, fit both and keep whichever explains the vertices better.
  //
  // This matters because the refinement below re-derives r from the axis and centre,
  // so on a small patch a bad axis produces a bad r no matter how the planarity search
  // was bracketed - bracketing alone was measured to change nothing at all.
  const candidates = [];
  {
    const R = golden(Math.max(1e-12, R0 - 2 * best.r), R0 + 2 * best.r, (x) => spread(x).v);
    candidates.push({ R, r: spread(R).r });
  }
  if (hint) {
    const R = golden(Math.max(1e-12, R0 - 2 * hint), R0 + 2 * hint, (x) => misfit(x, hint));
    candidates.push({ R, r: hint });
  }

  let R = null, rr = null, bestE = Infinity;
  for (const cand of candidates) {
    if (!(cand.R > 0) || !(cand.r > 0) || !isFinite(cand.R) || !isFinite(cand.r)) continue;
    if (cand.r >= cand.R) continue;   // self-intersecting tube: not a torus
    const e = misfit(cand.R, cand.r);
    if (e < bestE) { bestE = e; R = cand.R; rr = cand.r; }
  }

  let fit = null;
  if (R !== null) {
    // True point-to-surface residual, on the vertices.
    let e = 0, ws = 0;
    for (let i = 0; i < pos.n; i++) {
      const dr = rho[i] - R;
      const d = Math.sqrt(dr * dr + zz[i] * zz[i]) - rr;
      e += pos.W[i] * d * d; ws += pos.W[i];
    }
    fit = {
      type: 'torus',
      params: { axis: ax, center: c0, majorRadius: R, minorRadius: rr },
      residual: Math.sqrt(e / ws),
    };
  }

  // Every candidate, including the self-intersecting ones rejected above, is a START for
  // the geometric refinement; only its result has to be a proper torus.
  // Starts share one axis, so refineBest's axis de-duplication must not collapse them.
  const starts = [];

  // MORE THAN ONE TUBE HYPOTHESIS. Everything above commits to the single deepest planarity
  // minimum, and every start built from it shares its axis and centre. On fat tori - R/r of
  // 1.2 to 2, the fillets around small holes - that minimum is often the wrong one: scored
  // against the true STEP surfaces, 31 of 40 remaining native/gmsh misses had a true surface
  // fitting their vertices 1e4 times better than the fit (test/fitTruth.mjs, a basin check),
  // and almost all were such tori. So the next-best local minima of the scan, for both tube
  // senses, and the curvature estimate of r each get their own axis, centre and R as starts.
  const tubeStarts = minima.sort((a, b) => a.planar - b.planar).slice(0, 4);
  if (hint) for (const sgn of [1, -1]) tubeStarts.push({ ...probe(hint, sgn), r: hint, sgn });
  for (const m of tubeStarts) {
    const axm = normalize(m.axis);
    const cm = spineCentre(P, N, W, n, m.sgn * m.r, axm, m.centre) ?? m.centre;
    let Rm = 0;
    for (let i = 0; i < n; i++) {
      const dx = P[i * 3] - m.sgn * m.r * N[i * 3] - cm[0];
      const dy = P[i * 3 + 1] - m.sgn * m.r * N[i * 3 + 1] - cm[1];
      const dz = P[i * 3 + 2] - m.sgn * m.r * N[i * 3 + 2] - cm[2];
      const z = dx * axm[0] + dy * axm[1] + dz * axm[2];
      Rm += W[i] * Math.sqrt(Math.max(0, dx * dx + dy * dy + dz * dz - z * z));
    }
    Rm /= wSum;
    if (Rm > 0 && isFinite(Rm)) {
      starts.push({ startOnly: true, params: { axis: axm, center: cm, majorRadius: Rm, minorRadius: m.r } });
    }
  }
  for (const cand of candidates) {
    if (cand.R > 0 && cand.r > 0 && isFinite(cand.R) && isFinite(cand.r)) {
      starts.push({ startOnly: true,
                    params: { axis: ax, center: c0, majorRadius: cand.R, minorRadius: cand.r } });
    }
  }
  return refineBest(g, pos, fit ? [fit, ...starts] : starts, 'torus', refineTorus, false) ?? fit;
}

// ── Geometric refinement ─────────────────────────────────────────────────────
//
// Every estimate above is ALGEBRAIC: a relation that holds exactly on the true surface
// (tangent planes concur at a cone's apex, tube centres lie on a circle), solved in
// closed form. That makes them good at finding the basin and poor at its bottom. Off an
// exact surface each relation weights error differently from distance, and several
// read facet normals, which on an inscribed mesh are secant normals tilted by O(theta^2).
//
// Scored against the true STEP surfaces of 80 validation parts (test/fitTruth.mjs), that
// left cones and tori well short of what the data supports even where the mesh vertices
// sit on the true surface to 1e-8 of the part:
//
//                      native OCCT            gmsh
//     cone   dev<1e-4    39.9%                5.5%
//     torus  dev<1e-5    12.3%, 11% degen     7.9%
//
// The fix is the standard one: finish with least squares on the true geometric distance
// of the VERTICES, started from the algebraic estimate. The residual is exactly zero at
// the truth when the vertices are on the surface, so where the start is in the basin the
// refinement converges to it regardless of how the start was obtained.

/** Gaussian elimination with partial pivoting on a small dense system. */
function solveN(A, b) {
  const k = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < k; c++) {
    let p = c;
    for (let r = c + 1; r < k; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    if (!(Math.abs(M[p][c]) > 1e-300)) return null;
    if (p !== c) { const t = M[c]; M[c] = M[p]; M[p] = t; }
    for (let r = c + 1; r < k; r++) {
      const f = M[r][c] / M[c][c];
      if (f === 0) continue;
      for (let j = c; j <= k; j++) M[r][j] -= f * M[c][j];
    }
  }
  const x = new Array(k);
  for (let i = k - 1; i >= 0; i--) {
    let s = M[i][k];
    for (let j = i + 1; j < k; j++) s -= M[i][j] * x[j];
    x[i] = s / M[i][i];
  }
  return x.every(Number.isFinite) ? x : null;
}

/**
 * Levenberg-Marquardt with a forward-difference Jacobian.
 *
 * @param {number[]} x0      start
 * @param {number} m         residual count
 * @param {(x:number[], out:Float64Array)=>void} resid   fills out with WEIGHTED residuals
 * @param {number[]} steps   finite-difference step per parameter
 * @returns {{x:number[], cost:number}|null}  cost = sum of squared weighted residuals
 */
function levenbergMarquardt(x0, m, resid, steps, maxIter = 100) {
  const k = x0.length;
  let x = [...x0];
  let f = new Float64Array(m);
  resid(x, f);
  const sumSq = (v) => { let s = 0; for (let i = 0; i < m; i++) s += v[i] * v[i]; return s; };
  let cost = sumSq(f);
  if (!isFinite(cost)) return null;
  const J = new Float64Array(m * k);
  const fp = new Float64Array(m);
  let fn = new Float64Array(m);
  let lam = 1e-3;

  for (let it = 0; it < maxIter && cost > 0; it++) {
    for (let j = 0; j < k; j++) {
      const xp = [...x];
      xp[j] += steps[j];
      resid(xp, fp);
      for (let i = 0; i < m; i++) J[i * k + j] = (fp[i] - f[i]) / steps[j];
    }
    const A = Array.from({ length: k }, () => new Array(k).fill(0));
    const g = new Array(k).fill(0);
    for (let i = 0; i < m; i++) {
      const o = i * k;
      for (let a = 0; a < k; a++) {
        const ja = J[o + a];
        if (ja === 0) continue;
        g[a] += ja * f[i];
        for (let b = a; b < k; b++) A[a][b] += ja * J[o + b];
      }
    }
    let dmax = 0;
    for (let a = 0; a < k; a++) {
      for (let b = 0; b < a; b++) A[a][b] = A[b][a];
      dmax = Math.max(dmax, A[a][a]);
    }
    if (!(dmax > 0)) break;

    let accepted = false;
    while (lam < 1e10) {
      const M = A.map((row, a) => row.map((v, b) => (a === b ? v + lam * Math.max(v, 1e-12 * dmax) : v)));
      const dx = solveN(M, g.map(v => -v));
      if (dx) {
        const xn = x.map((v, j) => v + dx[j]);
        resid(xn, fn);
        const cn = sumSq(fn);
        if (cn < cost) {
          const gain = cost - cn;
          x = xn; [f, fn] = [fn, f]; cost = cn;
          lam = Math.max(lam * 0.3, 1e-12);
          accepted = gain > 1e-14 * cost ? true : 'converged';
          break;
        }
      }
      lam *= 10;
    }
    if (accepted !== true) break;
  }
  return { x, cost };
}

/** Tangent frame of a unit axis: the rotation parameters below perturb along these. */
function tangentFrame(axis) {
  const t1 = perp(axis);
  return [t1, normalize(cross(axis, t1))];
}

/** Region extent about the weighted mean of a positional sample set. */
function extentOf(pos) {
  const c = weightedMean3(pos.P, pos.W, pos.n, pos.wSum);
  let ext = 0;
  for (let i = 0; i < pos.n; i++) {
    ext = Math.max(ext, Math.hypot(pos.P[i * 3] - c[0], pos.P[i * 3 + 1] - c[1], pos.P[i * 3 + 2] - c[2]));
  }
  return { c, ext: ext || 1 };
}

/** Below this fraction of the region extent a refined RMS is exact: no other start can beat it. */
const EXACT_REL = 1e-6;

/**
 * Robust least squares on geometric distance: iteratively reweighted Levenberg-Marquardt
 * with Cauchy weights 1 / (1 + (d / s)^2), s = 1.4826 x the median |distance|.
 *
 * Plain least squares has no resistance to foreign facets, and every region a segmenter
 * proposes carries some. Refined without any robustness, fitoracle's 45 deg WatchStand
 * cone (a tolerance-built truth set that sweeps in a few neighbours) went from 44.99 to
 * 64.69 deg: the refinement walked to the wrong axis the positional residual has always
 * preferred there.
 *
 * A HARD trim was tried first - keep the best-fitting 80% of vertex weight, refit, repeat,
 * the same fraction the cone's axis criterion uses - and was worse, for a reason worth
 * keeping: on a small face the vertices that are dropped are the ones that DISCRIMINATE.
 * Ten vertices on two rings of a cone, less two, are fitted exactly by other cones. It lost
 * a WatchStand cone in paint.mjs (4/4 -> 3/4) and moved fitoracle's 45 deg cone only to
 * 49.68. A smooth weight keeps every vertex; on clean data the median residual describes
 * all of them, the weights stay near 1, and this is ordinary least squares.
 *
 * @returns {{x:number[], rms:number, sigma:number, d:Float64Array, d0:Float64Array}|null}
 *          rms over ALL vertices unweighted (the residual callers threshold on), d and d0
 *          the distances at the result and at the start
 */
function robustFit(pos, x0, steps, dist, ext) {
  const n = pos.n;
  const wn = Float64Array.from(pos.W, w => w / pos.wSum);
  const d = new Float64Array(n);
  const floor = 1e-12 * ext;
  const scaleOf = () => {
    const a = Array.from(d, Math.abs).sort((p, q) => p - q);
    return Math.max(1.4826 * a[a.length >> 1], floor);
  };
  dist(x0, d);
  if (!d.every(Number.isFinite)) return null;
  const d0 = Float64Array.from(d);
  let x = x0, sigma = scaleOf();
  for (let round = 0; round < 8; round++) {
    const sw = Float64Array.from(wn, (w, i) => Math.sqrt(w / (1 + (d[i] / sigma) ** 2)));
    const lm = levenbergMarquardt(x, n, (xx, o) => {
      dist(xx, o);
      for (let i = 0; i < n; i++) o[i] *= sw[i];
    }, steps);
    if (!lm) break;
    x = lm.x;
    dist(x, d);
    const next = scaleOf();
    const settled = Math.abs(next - sigma) <= 0.01 * sigma;
    sigma = next;
    if (settled) break;
  }
  let e = 0;
  for (let i = 0; i < n; i++) e += wn[i] * d[i] * d[i];
  return { x, rms: Math.sqrt(e), sigma, d: Float64Array.from(d), d0 };
}

/**
 * Refine a cone on vertex distance.
 *
 * NOT parameterised by the apex. On a shallow cone the apex sits many patch-widths away
 * and slides a long way for a tiny change in the surface, so an apex parameterisation is
 * ill-conditioned exactly where cones are hardest - and it has no limit as the half-angle
 * goes to zero. Instead: a reference point on the axis beside the patch, the radius r0
 * there, and the half-angle. The generator in the (rho, z) half-plane is then
 * rho = r0 + z tan(a), whose perpendicular distance is (rho - r0 - z tan a) cos a, and
 * a = 0 is simply a cylinder. The apex is recovered at the end.
 */
function refineCone(pos, p0) {
  const { c, ext } = extentOf(pos);
  const ax0 = normalize(p0.axis);
  const [t1, t2] = tangentFrame(ax0);
  const zc = dot(sub(c, p0.apex), ax0);
  const c0 = [p0.apex[0] + zc * ax0[0], p0.apex[1] + zc * ax0[1], p0.apex[2] + zc * ax0[2]];
  const unpack = (x) => ({
    ref: [c0[0] + (x[0] * t1[0] + x[1] * t2[0]) * ext,
          c0[1] + (x[0] * t1[1] + x[1] * t2[1]) * ext,
          c0[2] + (x[0] * t1[2] + x[1] * t2[2]) * ext],
    axis: normalize([ax0[0] + x[2] * t1[0] + x[3] * t2[0],
                     ax0[1] + x[2] * t1[1] + x[3] * t2[1],
                     ax0[2] + x[2] * t1[2] + x[3] * t2[2]]),
    r0: x[4] * ext,
    a: x[5],
  });
  const dist = (x, out) => {
    const { ref, axis, r0, a } = unpack(x);
    const ca = Math.cos(a), ta = Math.tan(a);
    for (let i = 0; i < pos.n; i++) {
      const dx = pos.P[i * 3] - ref[0], dy = pos.P[i * 3 + 1] - ref[1], dz = pos.P[i * 3 + 2] - ref[2];
      const z = dx * axis[0] + dy * axis[1] + dz * axis[2];
      const rho = Math.sqrt(Math.max(0, dx * dx + dy * dy + dz * dz - z * z));
      out[i] = (rho - r0 - z * ta) * ca;
    }
  };
  const x0 = [0, 0, 0, 0, (zc * Math.tan(p0.halfAngle)) / ext, p0.halfAngle];
  const t = robustFit(pos, x0, new Array(6).fill(1e-7), dist, ext);
  if (!t) return null;
  t.fit = null;
  let { ref, axis, r0, a } = unpack(t.x);
  if (a < 0) { a = -a; axis = [-axis[0], -axis[1], -axis[2]]; }
  if (!(a > 1e-4) || !(a < 1.5533) || !isFinite(t.rms)) return t;   // (0, 89 deg), as fitCone
  // radius r0 at `ref` shrinks to zero a distance r0 / tan(a) back along the (opening) axis
  const back = r0 / Math.tan(a);
  const apex = [ref[0] - back * axis[0], ref[1] - back * axis[1], ref[2] - back * axis[2]];
  if (apex.every(Number.isFinite)) {
    t.fit = { type: 'cone', params: { axis, apex, halfAngle: a }, residual: t.rms };
  }
  return t;
}

/** Refine a cylinder on vertex distance: axis point (2), axis (2), radius. */
function refineCylinder(pos, p0) {
  const { c, ext } = extentOf(pos);
  const ax0 = normalize(p0.axis);
  const [t1, t2] = tangentFrame(ax0);
  const zc = dot(sub(c, p0.axisPoint), ax0);
  const c0 = [p0.axisPoint[0] + zc * ax0[0], p0.axisPoint[1] + zc * ax0[1], p0.axisPoint[2] + zc * ax0[2]];
  const unpack = (x) => ({
    point: [c0[0] + (x[0] * t1[0] + x[1] * t2[0]) * ext,
            c0[1] + (x[0] * t1[1] + x[1] * t2[1]) * ext,
            c0[2] + (x[0] * t1[2] + x[1] * t2[2]) * ext],
    axis: normalize([ax0[0] + x[2] * t1[0] + x[3] * t2[0],
                     ax0[1] + x[2] * t1[1] + x[3] * t2[1],
                     ax0[2] + x[2] * t1[2] + x[3] * t2[2]]),
    radius: x[4] * ext,
  });
  const dist = (x, out) => {
    const { point, axis, radius } = unpack(x);
    for (let i = 0; i < pos.n; i++) {
      const dx = pos.P[i * 3] - point[0], dy = pos.P[i * 3 + 1] - point[1], dz = pos.P[i * 3 + 2] - point[2];
      const z = dx * axis[0] + dy * axis[1] + dz * axis[2];
      out[i] = Math.sqrt(Math.max(0, dx * dx + dy * dy + dz * dz - z * z)) - radius;
    }
  };
  const t = robustFit(pos, [0, 0, 0, 0, p0.radius / ext], new Array(5).fill(1e-7), dist, ext);
  if (!t) return null;
  t.fit = null;
  const { point, axis, radius } = unpack(t.x);
  if (radius > 0 && point.every(Number.isFinite) && isFinite(t.rms)) {
    t.fit = { type: 'cylinder', params: { axis, axisPoint: point, radius }, residual: t.rms };
  }
  return t;
}

/** Refine a torus on vertex distance: centre (3), axis (2), R, r. */
function refineTorus(pos, p0) {
  const { ext } = extentOf(pos);
  const ax0 = normalize(p0.axis);
  const [t1, t2] = tangentFrame(ax0);
  const c0 = p0.center;
  const unpack = (x) => ({
    center: [c0[0] + (x[0] * t1[0] + x[1] * t2[0] + x[2] * ax0[0]) * ext,
             c0[1] + (x[0] * t1[1] + x[1] * t2[1] + x[2] * ax0[1]) * ext,
             c0[2] + (x[0] * t1[2] + x[1] * t2[2] + x[2] * ax0[2]) * ext],
    axis: normalize([ax0[0] + x[3] * t1[0] + x[4] * t2[0],
                     ax0[1] + x[3] * t1[1] + x[4] * t2[1],
                     ax0[2] + x[3] * t1[2] + x[4] * t2[2]]),
    R: x[5] * ext,
    r: x[6] * ext,
  });
  const dist = (x, out) => {
    const { center, axis, R, r } = unpack(x);
    for (let i = 0; i < pos.n; i++) {
      const dx = pos.P[i * 3] - center[0], dy = pos.P[i * 3 + 1] - center[1], dz = pos.P[i * 3 + 2] - center[2];
      const z = dx * axis[0] + dy * axis[1] + dz * axis[2];
      const rho = Math.sqrt(Math.max(0, dx * dx + dy * dy + dz * dz - z * z));
      out[i] = Math.hypot(rho - R, z) - r;
    }
  };
  const x0 = [0, 0, 0, 0, 0, p0.majorRadius / ext, p0.minorRadius / ext];
  const t = robustFit(pos, x0, new Array(7).fill(1e-7), dist, ext);
  if (!t) return null;
  t.fit = null;
  const { center, axis, R, r } = unpack(t.x);
  if (R > 0 && r > 0 && center.every(Number.isFinite) && isFinite(t.rms)) {
    t.fit = { type: 'torus', params: { axis, center, majorRadius: R, minorRadius: r }, residual: t.rms };
  }
  return t;
}

/** Refine a plane on vertex distance: normal (2), offset along it. */
function refinePlane(pos, p0) {
  const { c, ext } = extentOf(pos);
  const n0 = normalize(p0.normal);
  const [t1, t2] = tangentFrame(n0);
  const o0 = p0.origin;
  const unpack = (x) => ({
    normal: normalize([n0[0] + x[0] * t1[0] + x[1] * t2[0],
                       n0[1] + x[0] * t1[1] + x[1] * t2[1],
                       n0[2] + x[0] * t1[2] + x[1] * t2[2]]),
    origin: [o0[0] + x[2] * ext * n0[0], o0[1] + x[2] * ext * n0[1], o0[2] + x[2] * ext * n0[2]],
  });
  const dist = (x, out) => {
    const { normal, origin } = unpack(x);
    for (let i = 0; i < pos.n; i++) {
      out[i] = (pos.P[i * 3] - origin[0]) * normal[0] + (pos.P[i * 3 + 1] - origin[1]) * normal[1]
             + (pos.P[i * 3 + 2] - origin[2]) * normal[2];
    }
  };
  const t = robustFit(pos, [0, 0, 0], [1e-7, 1e-7, 1e-7], dist, ext);
  if (!t) return null;
  t.fit = null;
  const { normal, origin } = unpack(t.x);
  if (normal.every(Number.isFinite) && isFinite(t.rms)) {
    // Keep the reported origin beside the region: the region's centre, dropped onto the plane.
    const h = dot(sub(c, origin), normal);
    t.fit = { type: 'plane', params: { origin: [c[0] - h * normal[0], c[1] - h * normal[1], c[2] - h * normal[2]], normal },
              residual: t.rms };
  }
  return t;
}

/** Refine a sphere on vertex distance: centre (3), radius. */
function refineSphere(pos, p0) {
  const { ext } = extentOf(pos);
  const c0 = p0.center;
  const unpack = (x) => ({
    center: [c0[0] + x[0] * ext, c0[1] + x[1] * ext, c0[2] + x[2] * ext],
    radius: x[3] * ext,
  });
  const dist = (x, out) => {
    const { center, radius } = unpack(x);
    for (let i = 0; i < pos.n; i++) {
      out[i] = Math.hypot(pos.P[i * 3] - center[0], pos.P[i * 3 + 1] - center[1], pos.P[i * 3 + 2] - center[2]) - radius;
    }
  };
  const t = robustFit(pos, [0, 0, 0, p0.radius / ext], new Array(4).fill(1e-7), dist, ext);
  if (!t) return null;
  t.fit = null;
  const { center, radius } = unpack(t.x);
  if (radius > 0 && center.every(Number.isFinite) && isFinite(t.rms)) {
    t.fit = { type: 'sphere', params: { center, radius }, residual: t.rms };
  }
  return t;
}

/**
 * One-facet plane hypotheses: the plane of a facet, for a few facets spread through the region.
 *
 * Every refinement starts from the region's own statistics, and on an impure region those
 * are pulled by the foreign facets - worst when the foreign facets are LARGE, since every
 * estimate here is area-weighted. OCCT meshes a planar cap as a fan of long slivers, so a
 * cylinder region with three cap facets can carry more cap area than cylinder area, and the
 * PCA plane of a plane region with a few wall facets is simply the wrong plane. A facet of
 * the true plane lies exactly in it, so its plane is an exact minimal-sample hypothesis in
 * the RANSAC sense, needing no estimate from the region at all.
 */
function planeStarts(g, k = 8) {
  const meanN = weightedMean3(g.N, g.W, g.n, g.wSum);
  const out = [];
  const stride = Math.max(1, Math.floor(g.n / k));
  for (let i = 0; i < g.n && out.length < k; i += stride) {
    let normal = [g.N[i * 3], g.N[i * 3 + 1], g.N[i * 3 + 2]];
    if (!(norm(normal) > 0.5)) continue;
    if (dot(normal, meanN) < 0) normal = [-normal[0], -normal[1], -normal[2]];
    out.push({ startOnly: true, params: { origin: [g.P[i * 3], g.P[i * 3 + 1], g.P[i * 3 + 2]], normal } });
  }
  return out;
}

/** Sine of the angle between each facet normal and the surface's normal at its centroid. */
function normalMisfit(type, p, g) {
  const out = new Float64Array(g.n);
  for (let i = 0; i < g.n; i++) {
    const x = g.P[i * 3], y = g.P[i * 3 + 1], z = g.P[i * 3 + 2];
    let s;
    if (type === 'plane') {
      s = p.normal;
    } else if (type === 'sphere') {
      s = [x - p.center[0], y - p.center[1], z - p.center[2]];
    } else {
      const a = normalize(p.axis);
      const o = type === 'cylinder' ? p.axisPoint : type === 'cone' ? p.apex : p.center;
      const dx = x - o[0], dy = y - o[1], dz = z - o[2];
      const t = dx * a[0] + dy * a[1] + dz * a[2];
      let rx = dx - t * a[0], ry = dy - t * a[1], rz = dz - t * a[2];
      const rn = Math.hypot(rx, ry, rz);
      if (rn > 0) { rx /= rn; ry /= rn; rz /= rn; }
      if (type === 'cylinder') {
        s = [rx, ry, rz];
      } else if (type === 'cone') {
        const ca = Math.cos(p.halfAngle), sa = Math.sin(p.halfAngle);
        s = [rx * ca - a[0] * sa, ry * ca - a[1] * sa, rz * ca - a[2] * sa];
      } else {
        s = [dx - p.majorRadius * rx, dy - p.majorRadius * ry, dz - p.majorRadius * rz];
      }
    }
    const sl = Math.hypot(s[0], s[1], s[2]);
    const c = sl > 0 ? Math.abs(g.N[i * 3] * s[0] + g.N[i * 3 + 1] * s[1] + g.N[i * 3 + 2] * s[2]) / sl : 0;
    out[i] = Math.sqrt(Math.max(0, 1 - Math.min(1, c * c)));
  }
  return out;
}

const madOf = (a) => {
  const s = Array.from(a, Math.abs).sort((p, q) => p - q);
  return s.length ? 1.4826 * s[s.length >> 1] : Infinity;
};

/**
 * Refine several starts; return the best of the starts and their refinements.
 *
 * Multi-start because the failures left after refining a single start were all BASIN
 * errors, not precision errors: faces of 6-30 facets whose start sat 30-90 deg from the
 * true axis, where no local method helps.
 *
 * CANDIDATES ARE JUDGED ON POSITIONS AND NORMALS TOGETHER, each by a Cauchy cost
 * sum w log(1 + (e / s)^2) at ONE scale shared by all candidates - the smallest robust scale
 * any candidate reaches, never below what float32 coordinates can represent. Shared, because
 * each fit's own scale would make them incomparable; Cauchy, because plain squares would hand
 * the choice back to foreign facets, whose cost this grows only logarithmically.
 *
 * Positions alone were tried first and lost 61 game-art cylinders of 4-8 facets that the
 * analytic start had right (dev 1e-5 -> 1e-3 of the part). Six to eight vertices barely
 * constrain a five-parameter cylinder, and least squares found another one through them. But
 * a low-poly prism's facet normals are EXACTLY perpendicular to the true axis, which is why
 * the Gauss-map start was right; the normal term restores that evidence. Normals also carry
 * what quantisation erases on narrow bands: on WatchStand's 29-facet 59 deg cones, vertices
 * at 3e-9 cannot tell 59.00 from 58.68 deg, while the facet normals disagree 10x.
 *
 * @param {object} g           gathered region (facet centroids and normals)
 * @param {object} pos         positional sample set
 * @param {Array<{params:object, startOnly?:boolean}>} starts  fits, or bare starts marked startOnly
 * @param {string} type
 * @param {Function} refine    refinePlane / refineCylinder / refineCone / refineSphere / refineTorus
 * @param {boolean} [dedupe]   refine starts whose axes agree within ~1 deg only once
 */
function refineBest(g, pos, starts, type, refine, dedupe = true) {
  const { ext } = extentOf(pos);
  const cands = [];
  const axes = [];
  for (const s of starts) {
    if (!s || !s.params) continue;
    if (dedupe) {
      const a = normalize(s.params.axis ?? s.params.normal);
      if (axes.some(b => Math.abs(dot(a, b)) > 0.9998)) continue;
      axes.push(a);
    }
    const r = refine(pos, s.params);
    if (!r) continue;
    if (!s.startOnly) cands.push({ fit: s, d: r.d0 });
    if (r.fit) {
      cands.push({ fit: r.fit, d: r.d });
      if (r.fit.residual <= EXACT_REL * ext) break;
    }
  }
  if (cands.length <= 1) return cands.length ? cands[0].fit : null;

  // float32 coordinates are exact to ~6e-8 of their magnitude; below that neither distances
  // nor facet normals (that quantum over an edge length) carry information.
  let big = 0;
  for (let i = 0; i < pos.n * 3; i++) big = Math.max(big, Math.abs(pos.P[i]));
  const quantum = 6e-8 * big;
  const edge = Math.sqrt(2 * g.wSum / g.n);
  let sp = Infinity, sn = Infinity;
  for (const c of cands) {
    c.e = normalMisfit(type, c.fit.params, g);
    sp = Math.min(sp, madOf(c.d));
    sn = Math.min(sn, madOf(c.e));
  }
  sp = Math.max(sp, quantum, 1e-12 * ext);
  sn = Math.max(sn, edge > 0 ? quantum / edge : 0, 1e-12);

  let best = null, bestCost = Infinity;
  for (const c of cands) {
    let cost = 0;
    for (let i = 0; i < pos.n; i++) cost += pos.W[i] * Math.log1p((c.d[i] / sp) ** 2);
    for (let i = 0; i < g.n; i++) cost += g.W[i] * Math.log1p((c.e[i] / sn) ** 2);
    if (cost < bestCost) { bestCost = cost; best = c.fit; }
  }
  return best;
}

/**
 * Fit a sphere by normal-line concurrency: every surface normal of a sphere
 * passes through its centre, so minimise Sum w |(I - n n^T)(c - p)|^2.
 * This is far better conditioned than the original 4-parameter algebraic fit
 * on positions alone, which needed a radius <= 5*extent guard to suppress
 * near-flat patches being "fit" by an enormous sphere.
 */
function fitSphere(g) {
  const { n, P, N, W, wSum } = g;
  const A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  const b = [0, 0, 0];

  for (let i = 0; i < n; i++) {
    const w = W[i];
    const nx = N[i * 3], ny = N[i * 3 + 1], nz = N[i * 3 + 2];
    const Pm = [
      [1 - nx * nx, -nx * ny, -nx * nz],
      [-ny * nx, 1 - ny * ny, -ny * nz],
      [-nz * nx, -nz * ny, 1 - nz * nz],
    ];
    const p = [P[i * 3], P[i * 3 + 1], P[i * 3 + 2]];
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        A[r][c] += w * Pm[r][c];
        b[r] += w * Pm[r][c] * p[c];
      }
    }
  }

  let centre = solve3(A, b);
  if (!centre) return null;

  // REFINE ON THE VERTICES. Normal concurrency above is well conditioned and gives a
  // good centre, but it is driven by facet CENTROIDS, and for a triangle inscribed in a
  // sphere the foot of the perpendicular from the centre is the circumcentre, not the
  // centroid. That mismatch is O(theta^2), so it shrank by exactly 4x per density step
  // and left the radius 2.5% short on a coarse mesh (0.8289 for a true 0.85).
  //
  // The vertices carry no such error - a tessellator puts them exactly on the surface -
  // so finish with the standard algebraic fit over them, which is exact for points on a
  // sphere: |v|^2 - 2 c.v + |c|^2 = R^2 is linear in (c, k) with k = |c|^2 - R^2.
  const pos = positional(g);
  if (pos.n >= 4) {
    const M = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
    const rhs = [0, 0, 0, 0];
    for (let i = 0; i < pos.n; i++) {
      const w = pos.W[i];
      const x = pos.P[i * 3], y = pos.P[i * 3 + 1], z = pos.P[i * 3 + 2];
      const row = [2 * x, 2 * y, 2 * z, 1];
      const val = x * x + y * y + z * z;
      for (let r = 0; r < 4; r++) {
        rhs[r] += w * val * row[r];
        for (let c = 0; c < 4; c++) M[r][c] += w * row[r] * row[c];
      }
    }
    const sol = solve4(M, rhs);
    if (sol) {
      const r2 = sol[3] + sol[0] * sol[0] + sol[1] * sol[1] + sol[2] * sol[2];
      if (r2 > 0 && isFinite(r2)) centre = [sol[0], sol[1], sol[2]];
    }
  }

  // Radius from the vertices too, for the same reason.
  let radius = 0, rw = 0;
  for (let i = 0; i < pos.n; i++) {
    radius += pos.W[i] * norm(sub([pos.P[i * 3], pos.P[i * 3 + 1], pos.P[i * 3 + 2]], centre));
    rw += pos.W[i];
  }
  radius = rw > 0 ? radius / rw : 0;
  if (!(radius > 1e-9) || !isFinite(radius)) return null;

  let e = 0, ew = 0;
  for (let i = 0; i < pos.n; i++) {
    const d = norm(sub([pos.P[i * 3], pos.P[i * 3 + 1], pos.P[i * 3 + 2]], centre)) - radius;
    e += pos.W[i] * d * d; ew += pos.W[i];
  }
  return { type: 'sphere', params: { center: centre, radius }, residual: Math.sqrt(e / Math.max(ew, 1e-30)) };
}

/** Gauss-Jordan on a 4x4, for the algebraic sphere fit. */
function solve4(A, b) {
  const m = A.map((row, i) => [...row, b[i]]);
  let scale = 0;
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) scale = Math.max(scale, Math.abs(m[i][j]));
  if (!(scale > 0)) return null;
  for (let i = 0; i < 4; i++) {
    let piv = i;
    for (let r = i + 1; r < 4; r++) if (Math.abs(m[r][i]) > Math.abs(m[piv][i])) piv = r;
    if (Math.abs(m[piv][i]) < 1e-14 * scale) return null;
    if (piv !== i) { const t = m[i]; m[i] = m[piv]; m[piv] = t; }
    for (let r = 0; r < 4; r++) {
      if (r === i) continue;
      const f = m[r][i] / m[i][i];
      for (let c = i; c < 5; c++) m[r][c] -= f * m[i][c];
    }
  }
  return [m[0][4] / m[0][0], m[1][4] / m[1][1], m[2][4] / m[2][2], m[3][4] / m[3][3]];
}

// ── Classification ───────────────────────────────────────────────────────────

const ORDER = { plane: 0, cylinder: 1, cone: 2, sphere: 3, torus: 4 };

/**
 * Classify one facet region.
 *
 * @param {Set<number>|number[]} triIndices
 * @param {{count,centroids,normals,areas}} facets  from meshCore.buildFacets
 * @param {{tol?:number, maxTol?:number, ratio?:number}} [opts]
 *        ratio  ...and whose residual is within this factor of the best candidate's
 *        tol    accept the simplest surface whose relative residual is under this
 *        maxTol beyond this nothing is considered an analytic fit -> 'nurbs'
 * @returns {{type,params,residual,relResidual,gauss,candidates}}
 */
export function classifyRegion(triIndices, facets, opts = {}) {
  const tol = opts.tol ?? 0.01;
  const maxTol = opts.maxTol ?? 0.05;
  const g = gather(triIndices, facets);
  if (g.n < 3 || g.wSum <= 0) {
    return { type: 'nurbs', params: null, residual: Infinity, relResidual: Infinity,
             gauss: null, candidates: [] };
  }

  // Region extent, used to make residuals scale-free.
  const pMean = weightedMean3(g.P, g.W, g.n, g.wSum);
  let extent = 0;
  for (let i = 0; i < g.n; i++) {
    const d = norm(sub([g.P[i * 3], g.P[i * 3 + 1], g.P[i * 3 + 2]], pMean));
    if (d > extent) extent = d;
  }
  if (!(extent > 0)) extent = 1;

  // ── Gauss map: fit a plane to the unit normals using the CENTERED covariance.
  const meanN = weightedMean3(g.N, g.W, g.n, g.wSum);
  const { values, vectors } = eigen3(weightedCov3(g.N, g.W, g.n, g.wSum, meanN));
  const u = vectors[0];
  const d = dot(meanN, u);
  const planarity = Math.sqrt(Math.max(0, values[0]));   // deviation from normal-plane
  const spread = Math.sqrt(Math.max(0, values[2]));      // normal-cloud extent
  const gauss = { axis: u, d, planarity, spread, halfAngle: Math.asin(Math.min(1, Math.abs(d))) };

  // ── Hypotheses. The Gauss map proposes; the positional fit disposes.
  const candidates = [];
  const push = c => { if (c && isFinite(c.residual)) candidates.push(c); };

  // EVERY CANDIDATE IS REFINED before any is compared. The choice below asks whether a simpler
  // primitive explains the region as well as the best one, and that is only meaningful between
  // fits that have each reached their own optimum: set a refined torus beside an unrefined
  // cylinder and the torus wins regions it has no business winning.
  const pos = positional(g);
  const plane = fitPlane(g);
  push(refineBest(g, pos, [plane, ...planeStarts(g)], 'plane', refinePlane) ?? plane);

  // Narrow bands are degenerate in normal space (the two smallest eigenvalues
  // are close and either may be the true axis), so try both as axis candidates.
  const axisCandidates = [vectors[0]];
  if (values[1] > 0 && (values[0] / values[1]) > 0.1) axisCandidates.push(vectors[1]);

  const cyls = [], cones = [];
  for (const ax of axisCandidates) {
    const cy = fitCylinder(g, ax);
    if (cy) cyls.push(cy);
    const co = fitCone(g, ax);
    if (co) cones.push(co);
  }
  push(refineBest(g, pos, cyls, 'cylinder', refineCylinder));
  push(refineBest(g, pos, cones, 'cone', refineCone));
  const sphere = fitSphere(g);
  push(sphere && (refineBest(g, pos, [sphere], 'sphere', refineSphere, false) ?? sphere));

  // A torus has seven parameters against a cylinder's five, so on residual alone it
  // could impersonate the simpler primitives. Measured on tangent-fillet, it does not:
  // on the true cylinder it scores 3.5e-1 against the cylinder's 2.0e-8, and on the
  // planes it is rejected outright. The guards inside fitTorus are what close that off
  // - bounding the tube radius to the patch extent and requiring r < R rules out the
  // R -> infinity (cylinder) and R -> 0 (sphere) escapes. On the actual fillet it wins
  // by four orders of magnitude, 2.0e-6 against the next best 2.0e-2.
  push(fitTorus(g));

  if (!candidates.length) {
    return { type: 'nurbs', params: null, residual: Infinity, relResidual: Infinity,
             gauss, candidates: [] };
  }

  for (const c of candidates) c.relResidual = c.residual / extent;

  // Prefer the SIMPLEST surface that explains the data, not merely the lowest
  // residual — a more general primitive always fits at least as well, so raw
  // residual comparison across differing degrees of freedom is not meaningful.
  //
  // "Explains the data" is RELATIVE TO THE BEST FIT, not a fixed fraction of the region. A flat
  // 1% of extent let a plane take a shallow cylindrical arc whose sagitta is under 1% of it, and
  // for export that is a face whose edges miss their neighbours by up to 1% of the face. A
  // simpler primitive is accepted when its residual is within `ratio` of the best candidate's -
  // or of the float32 quantum, below which no two fits differ measurably - and still under tol.
  //
  // Measured on 40 validation parts x 3 meshers, true type returned (test/fitTruth.mjs --ratio):
  //
  //     ratio        cylinder  cone   sphere  torus
  //     2              85.4    95.4   93.2    94.9
  //     4              87.2    97.5   94.2    94.5
  //     8              89.1    98.2   94.2    92.8
  //     unlimited      87.5    93.3   92.2    73.2    (the old rule; 7.6% of cylinders -> plane)
  //
  // 4 rather than 8 because the real downloaded mesh does better with it (watchstand.mjs:
  // WatchStandCoarse 21/77 at 4, 19 at 8). The cost of any finite ratio is on IMPURE crease
  // regions of that mesh: a torus absorbs foreign facets well enough to win, taking two
  // WatchStandMedium cones (19 -> 17/77) at every ratio tried.
  let big = 0;
  for (let i = 0; i < pos.n * 3; i++) big = Math.max(big, Math.abs(pos.P[i]));
  const bestResidual = Math.max(Math.min(...candidates.map(c => c.residual)), 6e-8 * big);
  const ratio = opts.ratio ?? 4;
  const acceptable = candidates
    .filter(c => c.relResidual <= tol && c.residual <= ratio * bestResidual)
    .sort((a, b) => ORDER[a.type] - ORDER[b.type] || a.relResidual - b.relResidual);

  const best = acceptable[0]
    ?? [...candidates].sort((a, b) => a.relResidual - b.relResidual)[0];

  if (best.relResidual > maxTol) {
    return { type: 'nurbs', params: null, residual: best.residual,
             relResidual: best.relResidual, gauss, candidates };
  }
  return { ...best, extent, gauss, candidates };
}

// ── Forced (user-specified) surface fitting ──────────────────────────────────
//
// When the user picks the surface type, classification is out of the picture —
// the only remaining task is the best-fit of ONE known primitive to the facet
// set. That is a well-posed problem, so this must never fail silently and must
// never invent parameters. The original classifyGroupAs() fabricated a cylinder
// whose axis was the region's PLANE NORMAL (perpendicular to any real cylinder
// axis) with radius = half the region extent, and a cone with a hardcoded 30
// degree half-angle, whenever its fit did not converge. That is why forcing a
// type produced geometry unrelated to the selected faces.
//
// Robustness here comes from multi-start plus refinement rather than from a
// single analytic guess: seed the axis from several independent estimates, then
// walk it downhill on the true positional residual.

/** Independent axis seeds for a surface of revolution. */
function axisSeeds(g) {
  const seeds = [];
  const meanN = weightedMean3(g.N, g.W, g.n, g.wSum);
  const gm = eigen3(weightedCov3(g.N, g.W, g.n, g.wSum, meanN));
  seeds.push(gm.vectors[0], gm.vectors[1], gm.vectors[2]);

  // A long, narrow band gives a poor normal-space axis but an excellent
  // positional one: the patch is elongated ALONG the axis.
  const pMean = weightedMean3(g.P, g.W, g.n, g.wSum);
  const pc = eigen3(weightedCov3(g.P, g.W, g.n, g.wSum, pMean));
  seeds.push(pc.vectors[2], pc.vectors[1]);

  // The mean normal is perpendicular to the axis of a cylinder, so anything
  // orthogonal to it is a reasonable starting guess.
  const mn = normalize(meanN);
  seeds.push(perp(mn));
  return seeds.map(normalize);
}

/**
 * How good is an axis?
 *
 * `axisCost` when the fit supplies one — for cones that is the variance of
 * (n . axis), which is EXACTLY zero on the true axis and needs no weighting against
 * the positional term. Everything else falls back to the positional residual.
 */
function axisCost(f) {
  if (!f) return Infinity;
  if (isFinite(f.axisCost)) return f.axisCost;
  return isFinite(f.score) ? f.score : f.residual;
}

/**
 * Walk an axis downhill on the positional residual by pattern search over the
 * unit sphere. Derivative-free, so it works identically for cylinders and cones
 * and cannot diverge — each step is accepted only if the residual improves.
 */
function refineAxis(g, axis0, fitFn, maxIter = 80) {
  let axis = normalize(axis0);
  let best = fitFn(g, axis);
  let step = 0.2;                        // radians
  const cost = axisCost;

  for (let it = 0; it < maxIter && step > 1e-7; it++) {
    const t1 = perp(axis);
    const t2 = normalize(cross(axis, t1));
    let improved = false;

    for (const [du, dv] of [[1, 0], [-1, 0], [0, 1], [0, -1],
                            [1, 1], [1, -1], [-1, 1], [-1, -1]]) {
      const cand = normalize([
        axis[0] + step * (du * t1[0] + dv * t2[0]),
        axis[1] + step * (du * t1[1] + dv * t2[1]),
        axis[2] + step * (du * t1[2] + dv * t2[2]),
      ]);
      const f = fitFn(g, cand);
      if (f && isFinite(f.residual) && cost(f) < cost(best)) {
        best = f; axis = cand; improved = true;
      }
    }
    if (!improved) step *= 0.5;
  }
  return best;
}

/**
 * Best-fit a SPECIFIC surface type to a facet region.
 *
 * @param {'plane'|'cylinder'|'cone'|'sphere'|'nurbs'} type
 * @param {Set<number>|number[]} triIndices
 * @param {object} facets  from meshCore.buildFacets
 * @returns {{type,params,residual,relResidual,extent,degenerate?:boolean,reason?:string}}
 */
export function fitRegionAs(type, triIndices, facets) {
  const g = gather(triIndices, facets);
  // One triangle already fixes a plane exactly. OCCT meshes a rectangular planar face as TWO
  // triangles, and refusing those (the old limit of three for every type) left every such face of a
  // native mesh without a surface, so no solid could close around it.
  if (g.n < (type === 'plane' ? 1 : 3) || g.wSum <= 0) {
    return { type, params: null, residual: Infinity, relResidual: Infinity,
             degenerate: true, reason: 'too few triangles' };
  }

  const pMean = weightedMean3(g.P, g.W, g.n, g.wSum);
  let extent = 0;
  for (let i = 0; i < g.n; i++) {
    const d = norm(sub([g.P[i * 3], g.P[i * 3 + 1], g.P[i * 3 + 2]], pMean));
    if (d > extent) extent = d;
  }
  if (!(extent > 0)) extent = 1;

  const done = (fit, reason) => fit
    ? { ...fit, extent, relResidual: fit.residual / extent }
    : { type, params: null, residual: Infinity, relResidual: Infinity,
        extent, degenerate: true, reason };

  if (type === 'plane') {
    const base = fitPlane(g);
    return done(refineBest(g, positional(g), [base, ...planeStarts(g)], 'plane', refinePlane) ?? base,
                'plane fit degenerate');
  }
  if (type === 'sphere') {
    const base = fitSphere(g);
    return done(base && (refineBest(g, positional(g), [base], 'sphere', refineSphere, false) ?? base),
                'normals do not converge on a centre');
  }
  if (type === 'torus')  return done(fitTorus(g), 'no consistent tube: {p - r n} is not planar for any r');
  if (type === 'nurbs')  return { type: 'nurbs', params: null, residual: Infinity,
                                  relResidual: Infinity, extent };

  if (type === 'cylinder' || type === 'cone') {
    const fitFn = type === 'cylinder' ? fitCylinder : fitCone;
    const cost = axisCost;
    let best = null;

    const found = [];
    for (const seed of axisSeeds(g)) {
      const r = refineAxis(g, seed, fitFn);
      if (!r || !isFinite(r.residual)) continue;
      found.push(r);
      if (cost(r) < cost(best)) best = r;
    }
    if (best) {
      // The axis criterion's choice first, so it is kept unless a refined fit beats it.
      found.sort((a, b) => cost(a) - cost(b));
      const lm = refineBest(g, positional(g), [best, ...found], type,
                            type === 'cone' ? refineCone : refineCylinder);
      if (lm) best = lm;
    }
    return done(best, type === 'cone'
      ? 'no apex: the region has no measurable taper (a cone with zero half-angle is a cylinder)'
      : 'no consistent axis for a cylinder');
  }

  return { type, params: null, residual: Infinity, relResidual: Infinity,
           extent, degenerate: true, reason: 'unknown surface type' };
}
