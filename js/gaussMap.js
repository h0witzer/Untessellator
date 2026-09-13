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
  return { n, P, N, W, wSum, vn, VP, VW };
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
  let sAbs = 0;
  for (let i = 0; i < n; i++) {
    sAbs += W[i] * Math.abs(N[i * 3] * axis[0] + N[i * 3 + 1] * axis[1] + N[i * 3 + 2] * axis[2]);
  }
  const halfAngle = Math.asin(Math.min(1, Math.max(0, sAbs / wSum)));
  if (!(halfAngle > 1e-4) || halfAngle > 1.5533) return null;    // (0, 89 deg)
  const tanA = Math.tan(halfAngle);

  // Orientation is not determined by the half-angle, so place the apex for both
  // senses of the axis and keep whichever explains the region better. Cheaper and
  // more reliable than inferring it from a normal's sign, which flips on a mesh
  // whose winding is inconsistent.
  // Apex candidates: the tangent-plane solve (exact where the tessellation is clean —
  // measured 1.4e-9 on a 59 deg cone) and the axial average as a fallback for when the
  // plane system is ill-conditioned on a narrow arc. Both are scored below.
  const planeApex = coneApexFromPlanes(g);

  let ax = axis, apex = null, bestCost = Infinity;
  for (const s of [1, -1]) {
    const cand = [axis[0] * s, axis[1] * s, axis[2] * s];
    const fr = axialFrame(pos, cand, origin);
    // On the cone, R = tan(a) * (Z - z0), so each sample votes for an apex position.
    let z0 = 0;
    for (let i = 0; i < pos.n; i++) z0 += pos.W[i] * (fr.Z[i] - fr.R[i] / tanA);
    z0 /= pos.wSum;
    const axialApex = [origin[0] + z0 * cand[0], origin[1] + z0 * cand[1],
                       origin[2] + z0 * cand[2]];
    for (const ap of (planeApex ? [planeApex, axialApex] : [axialApex])) {
    const ca = Math.cos(halfAngle), sa = Math.sin(halfAngle);
    const f2 = axialFrame(pos, cand, ap);
    let e = 0;
    for (let i = 0; i < pos.n; i++) {
      const d = f2.R[i] * ca - f2.Z[i] * sa;
      e += pos.W[i] * d * d;
    }
      if (e < bestCost) { bestCost = e; ax = cand; apex = ap; }
    }
  }
  if (!apex || !isFinite(apex[0])) return null;

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
 * What would resolve it is curvature rather than more search. At any point of a torus the
 * principal curvature around the tube is exactly 1/r, independent of where the point sits,
 * so r is readable from a small patch without ever seeing the circle - the same shape of
 * fix as taking the cone's half-angle from |n . axis| instead of an R-vs-Z regression.
 * That needs curvature estimation on a mesh, which does not exist here yet.
 *
 * The residual also DISCRIMINATES, which matters more than the parameters. The same
 * torus displaced by 9% of its minor radius - a sculpted doughnut rather than a
 * machined one - fits 22x worse (6.7e-2 against 3.1e-3). A sculpted surface should be
 * reported as freeform, not as a torus that happens to be nearby, and that judgement
 * belongs to the residual gate rather than to the classifier.
 */
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

  // Coarse scan then golden-section. The objective is unimodal in r but the scan guards
  // against the trivial minimum near r = 0, where any small patch looks planar.
  let best = null;
  for (const sgn of [1, -1]) {
    const lo = 0.02 * ext, hi = 2.0 * ext;
    const steps = 48;
    let bi = null;
    for (let i = 0; i <= steps; i++) {
      const r = lo + (hi - lo) * (i / steps);
      const pr = probe(r, sgn);
      if (!bi || pr.planar < bi.planar) bi = { ...pr, r };
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

  // Refine R and r on the VERTICES, which lie exactly on the surface where centroids do
  // not. With the axis and centre fixed, each vertex gives rho and z about the axis, and
  // its distance to the tube-centre circle is sqrt((rho-R)^2 + z^2), which should equal r.
  const ax = normalize(best.axis);
  const c0 = best.centre;
  const rho = new Float64Array(pos.n), zz = new Float64Array(pos.n);
  for (let i = 0; i < pos.n; i++) {
    const dx = pos.P[i * 3] - c0[0], dy = pos.P[i * 3 + 1] - c0[1], dz = pos.P[i * 3 + 2] - c0[2];
    const z = dx * ax[0] + dy * ax[1] + dz * ax[2];
    const rx = dx - z * ax[0], ry = dy - z * ax[1], rz = dz - z * ax[2];
    rho[i] = Math.sqrt(rx * rx + ry * ry + rz * rz);
    zz[i] = z;
  }
  const spread = (R) => {
    let sw = 0, sd = 0, sdd = 0;
    for (let i = 0; i < pos.n; i++) {
      const dr = rho[i] - R;
      const d = Math.sqrt(dr * dr + zz[i] * zz[i]);
      sw += pos.W[i]; sd += pos.W[i] * d; sdd += pos.W[i] * d * d;
    }
    const mean = sd / sw;
    return { r: mean, v: Math.max(0, sdd / sw - mean * mean) };
  };
  let R0 = 0;
  for (let i = 0; i < pos.n; i++) R0 += pos.W[i] * rho[i];
  R0 /= pos.W.reduce((a, b) => a + b, 0);
  let lo = Math.max(1e-12, R0 - 2 * best.r), hi = R0 + 2 * best.r;
  const phi = 0.6180339887;
  let c1 = hi - phi * (hi - lo), c2 = lo + phi * (hi - lo);
  let f1 = spread(c1).v, f2 = spread(c2).v;
  for (let it = 0; it < 60 && hi - lo > 1e-12 * ext; it++) {
    if (f1 < f2) { hi = c2; c2 = c1; f2 = f1; c1 = hi - phi * (hi - lo); f1 = spread(c1).v; }
    else { lo = c1; c1 = c2; f1 = f2; c2 = lo + phi * (hi - lo); f2 = spread(c2).v; }
  }
  const R = 0.5 * (lo + hi);
  const rr = spread(R).r;
  if (!(R > 0) || !(rr > 0) || !isFinite(R) || !isFinite(rr)) return null;
  // A torus whose tube reaches the axis is self-intersecting; treat it as not a torus.
  if (rr >= R) return null;

  // True point-to-surface residual, on the vertices.
  let e = 0, ws = 0;
  for (let i = 0; i < pos.n; i++) {
    const dr = rho[i] - R;
    const d = Math.sqrt(dr * dr + zz[i] * zz[i]) - rr;
    e += pos.W[i] * d * d; ws += pos.W[i];
  }
  return {
    type: 'torus',
    params: { axis: ax, center: c0, majorRadius: R, minorRadius: rr },
    residual: Math.sqrt(e / ws),
  };
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

  const centre = solve3(A, b);
  if (!centre) return null;

  let signedR = 0;
  for (let i = 0; i < n; i++) {
    signedR += W[i] * (
      (P[i * 3] - centre[0]) * N[i * 3] +
      (P[i * 3 + 1] - centre[1]) * N[i * 3 + 1] +
      (P[i * 3 + 2] - centre[2]) * N[i * 3 + 2]);
  }
  signedR /= wSum;
  const radius = Math.abs(signedR);
  if (!(radius > 1e-9) || !isFinite(radius)) return null;

  let e = 0;
  for (let i = 0; i < n; i++) {
    const d = norm(sub([P[i * 3], P[i * 3 + 1], P[i * 3 + 2]], centre)) - radius;
    e += W[i] * d * d;
  }
  return { type: 'sphere', params: { center: centre, radius }, residual: Math.sqrt(e / wSum) };
}

// ── Classification ───────────────────────────────────────────────────────────

const ORDER = { plane: 0, cylinder: 1, cone: 2, sphere: 3, torus: 4 };

/**
 * Classify one facet region.
 *
 * @param {Set<number>|number[]} triIndices
 * @param {{count,centroids,normals,areas}} facets  from meshCore.buildFacets
 * @param {{tol?:number, maxTol?:number}} [opts]
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

  push(fitPlane(g));

  // Narrow bands are degenerate in normal space (the two smallest eigenvalues
  // are close and either may be the true axis), so try both as axis candidates.
  const axisCandidates = [vectors[0]];
  if (values[1] > 0 && (values[0] / values[1]) > 0.1) axisCandidates.push(vectors[1]);

  for (const ax of axisCandidates) {
    push(fitCylinder(g, ax));
    push(fitCone(g, ax));
  }
  push(fitSphere(g));

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
  const acceptable = candidates
    .filter(c => c.relResidual <= tol)
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
  if (g.n < 3 || g.wSum <= 0) {
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

  if (type === 'plane')  return done(fitPlane(g), 'plane fit degenerate');
  if (type === 'sphere') return done(fitSphere(g), 'normals do not converge on a centre');
  if (type === 'torus')  return done(fitTorus(g), 'no consistent tube: {p - r n} is not planar for any r');
  if (type === 'nurbs')  return { type: 'nurbs', params: null, residual: Infinity,
                                  relResidual: Infinity, extent };

  if (type === 'cylinder' || type === 'cone') {
    const fitFn = type === 'cylinder' ? fitCylinder : fitCone;
    const cost = axisCost;
    let best = null;

    for (const seed of axisSeeds(g)) {
      const r = refineAxis(g, seed, fitFn);
      if (r && isFinite(r.residual) && cost(r) < cost(best)) best = r;
    }
    return done(best, type === 'cone'
      ? 'no apex: the region has no measurable taper (a cone with zero half-angle is a cylinder)'
      : 'no consistent axis for a cylinder');
  }

  return { type, params: null, residual: Infinity, relResidual: Infinity,
           extent, degenerate: true, reason: 'unknown surface type' };
}
