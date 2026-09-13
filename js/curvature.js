/**
 * curvature.js — per-facet principal curvatures on a triangle mesh.
 *
 * WHY THIS EXISTS
 *
 * Every fitter in this project has hit the same wall: a small patch does not contain
 * enough of a surface to pin its GLOBAL parameters. The cone's half-angle could not be
 * read from an R-vs-Z regression on a narrow band; the torus's tube radius cannot be
 * read from the tube-centre relation on a short arc, because many radii make a short arc
 * planar. Both were solved, or need solving, by finding a quantity that is LOCAL.
 *
 * Principal curvature is that quantity for every primitive we fit:
 *
 *     plane      k1 = 0,      k2 = 0
 *     cylinder   k1 = 1/R,    k2 = 0
 *     cone       k1 varies,   k2 = 0          (zero along the ruling)
 *     sphere     k1 = 1/R,    k2 = 1/R
 *     torus      k1 = 1/r     everywhere,     k2 = cos(v) / (R + r cos v)
 *
 * The torus line is the one worth staring at: the curvature around the tube is exactly
 * 1/r at EVERY point, independent of where the point sits on the surface. So the minor
 * radius is readable from an arbitrarily small patch, which is precisely what the
 * tube-centre relation cannot do.
 *
 * METHOD
 *
 * Rusinkiewicz, "Estimating Curvatures and Their Derivatives on Triangle Meshes" (2004).
 * For a triangle, the second fundamental form II maps a tangent direction to the rate of
 * change of the surface normal along it. Each edge gives one sample of that map — the
 * edge vector and the difference of the vertex normals across it — so the three edges
 * give six linear equations in the three unknowns of the symmetric 2x2 form. Solve by
 * least squares in the triangle's own tangent basis, then diagonalise.
 *
 * Vertex normals are area-weighted averages of the incident facet normals. That is the
 * standard input for this method and is appropriate here even though averaged normals
 * are the wrong thing for FITTING (see gaussMap.js, where averaging then renormalising
 * biases a cone's half-angle): here the signal is how the normal VARIES, not its value.
 *
 * SIGN CONVENTION: normals point outward, so a convex surface has positive curvature.
 * A sphere of radius R gives k1 = k2 = +1/R. Callers comparing against an analytic
 * radius should generally use magnitudes, since a mesh with inconsistent winding will
 * flip the sign without changing the geometry.
 */

import { eigen3 } from './meshCore.js';

/**
 * Area-weighted vertex normals, welded by exact coordinates.
 *
 * Exact matching rather than a quantised key, for the same reason gaussMap.js welds
 * exactly: under-welding merely splits a vertex into two with half the weight each and
 * changes nothing here, while over-welding averages normals across a genuine crease and
 * invents curvature that is not there.
 */
export function vertexNormals(positions, facets) {
  const acc = new Map();
  const ids = new Int32Array(facets.count * 3).fill(-1);
  const list = [];

  for (let t = 0; t < facets.count; t++) {
    if (facets.degenerate && facets.degenerate[t]) continue;
    const w = facets.areas[t];
    for (let k = 0; k < 3; k++) {
      const o = t * 9 + k * 3;
      const x = positions[o], y = positions[o + 1], z = positions[o + 2];
      const key = x + '_' + y + '_' + z;
      let e = acc.get(key);
      if (e === undefined) {
        e = list.length;
        acc.set(key, e);
        list.push([0, 0, 0]);
      }
      const n = list[e];
      n[0] += w * facets.normals[t * 3];
      n[1] += w * facets.normals[t * 3 + 1];
      n[2] += w * facets.normals[t * 3 + 2];
      ids[t * 3 + k] = e;
    }
  }

  const N = new Float64Array(list.length * 3);
  for (let i = 0; i < list.length; i++) {
    const n = list[i];
    const L = Math.hypot(n[0], n[1], n[2]);
    if (L > 1e-20) { N[i * 3] = n[0] / L; N[i * 3 + 1] = n[1] / L; N[i * 3 + 2] = n[2] / L; }
  }
  return { normals: N, ids, count: list.length };
}

/**
 * Principal curvatures per facet.
 *
 * @returns {{k1: Float64Array, k2: Float64Array, valid: Uint8Array}}
 *   k1 is the larger-magnitude principal curvature, k2 the smaller. `valid` is 0 where
 *   the facet is degenerate or the system was too ill-conditioned to trust.
 */
export function facetCurvature(positions, facets, vn = null) {
  const V = vn || vertexNormals(positions, facets);
  const n = facets.count;
  const k1 = new Float64Array(n);
  const k2 = new Float64Array(n);
  const valid = new Uint8Array(n);

  for (let t = 0; t < n; t++) {
    if (facets.degenerate && facets.degenerate[t]) continue;
    const o = t * 9;
    const p = [
      [positions[o], positions[o + 1], positions[o + 2]],
      [positions[o + 3], positions[o + 4], positions[o + 5]],
      [positions[o + 6], positions[o + 7], positions[o + 8]],
    ];
    const nrm = [facets.normals[t * 3], facets.normals[t * 3 + 1], facets.normals[t * 3 + 2]];

    // Tangent basis for this facet.
    let u = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
    let uL = Math.hypot(u[0], u[1], u[2]);
    if (!(uL > 1e-20)) continue;
    u = [u[0] / uL, u[1] / uL, u[2] / uL];
    const v = [
      nrm[1] * u[2] - nrm[2] * u[1],
      nrm[2] * u[0] - nrm[0] * u[2],
      nrm[0] * u[1] - nrm[1] * u[0],
    ];

    const vnOf = (i) => {
      const id = V.ids[t * 3 + i];
      return id < 0 ? nrm : [V.normals[id * 3], V.normals[id * 3 + 1], V.normals[id * 3 + 2]];
    };
    const n0 = vnOf(0), n1 = vnOf(1), n2 = vnOf(2);

    // Edge i is opposite vertex i; dn is the normal difference across the same edge.
    const edges = [
      [[p[2][0] - p[1][0], p[2][1] - p[1][1], p[2][2] - p[1][2]], [n2[0] - n1[0], n2[1] - n1[1], n2[2] - n1[2]]],
      [[p[0][0] - p[2][0], p[0][1] - p[2][1], p[0][2] - p[2][2]], [n0[0] - n2[0], n0[1] - n2[1], n0[2] - n2[2]]],
      [[p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]], [n1[0] - n0[0], n1[1] - n0[1], n1[2] - n0[2]]],
    ];

    // Six equations in (a, b, c) for II = [[a, b], [b, c]]:
    //   a (e.u) + b (e.v) = dn.u
    //   b (e.u) + c (e.v) = dn.v
    const A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    const rhs = [0, 0, 0];
    const add = (row, r) => {
      for (let i = 0; i < 3; i++) {
        rhs[i] += row[i] * r;
        for (let j = 0; j < 3; j++) A[i][j] += row[i] * row[j];
      }
    };
    for (const [e, dn] of edges) {
      const eu = e[0] * u[0] + e[1] * u[1] + e[2] * u[2];
      const ev = e[0] * v[0] + e[1] * v[1] + e[2] * v[2];
      const du = dn[0] * u[0] + dn[1] * u[1] + dn[2] * u[2];
      const dv = dn[0] * v[0] + dn[1] * v[1] + dn[2] * v[2];
      add([eu, ev, 0], du);
      add([0, eu, ev], dv);
    }

    const sol = solve3sym(A, rhs);
    if (!sol) continue;
    const [a, b, c] = sol;

    // Eigenvalues of the symmetric 2x2 form.
    const tr = a + c, det = a * c - b * b;
    const disc = Math.max(0, tr * tr / 4 - det);
    const rt = Math.sqrt(disc);
    const e1 = tr / 2 + rt, e2 = tr / 2 - rt;
    if (!isFinite(e1) || !isFinite(e2)) continue;
    if (Math.abs(e1) >= Math.abs(e2)) { k1[t] = e1; k2[t] = e2; }
    else { k1[t] = e2; k2[t] = e1; }
    valid[t] = 1;
  }
  return { k1, k2, valid };
}

/** Symmetric 3x3 solve with a conditioning guard. */
function solve3sym(A, b) {
  const m = [
    [A[0][0], A[0][1], A[0][2], b[0]],
    [A[1][0], A[1][1], A[1][2], b[1]],
    [A[2][0], A[2][1], A[2][2], b[2]],
  ];
  let scale = 0;
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) scale = Math.max(scale, Math.abs(m[i][j]));
  if (!(scale > 0)) return null;
  for (let i = 0; i < 3; i++) {
    let piv = i;
    for (let r = i + 1; r < 3; r++) if (Math.abs(m[r][i]) > Math.abs(m[piv][i])) piv = r;
    if (Math.abs(m[piv][i]) < 1e-14 * scale) return null;
    if (piv !== i) { const tmp = m[i]; m[i] = m[piv]; m[piv] = tmp; }
    for (let r = 0; r < 3; r++) {
      if (r === i) continue;
      const f = m[r][i] / m[i][i];
      for (let cc = i; cc < 4; cc++) m[r][cc] -= f * m[i][cc];
    }
  }
  return [m[0][3] / m[0][0], m[1][3] / m[1][1], m[2][3] / m[2][2]];
}

/**
 * Robust summary of a region's curvature. Uses medians, because a region proposed by any
 * real segmenter carries a few foreign facets and curvature estimates on a mesh are
 * noisy at the best of times.
 *
 * @returns {{k1: number, k2: number, n: number}|null} medians of the two principal
 *   curvatures over the valid facets of the region.
 */
export function regionCurvature(region, curv) {
  const a = [], b = [];
  for (const t of region) {
    if (!curv.valid[t]) continue;
    a.push(curv.k1[t]);
    b.push(curv.k2[t]);
  }
  if (a.length < 3) return null;
  a.sort((x, y) => x - y);
  b.sort((x, y) => x - y);
  return { k1: a[a.length >> 1], k2: b[b.length >> 1], n: a.length };
}

/**
 * Compute curvature once for a mesh and attach it to the facets object, so that
 * gaussMap's gather() can pick up a per-region hint without recomputing.
 *
 * Curvature is a whole-mesh quantity and costs a pass over every facet, so it must not
 * be recomputed per region: a seeded fill asks for hundreds of regions on one mesh.
 */
export function attachCurvature(positions, facets) {
  if (!facets.curvature) facets.curvature = facetCurvature(positions, facets);
  return facets;
}
