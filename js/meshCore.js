/**
 * meshCore.js — dependency-free mesh primitives shared by the browser app and
 * the headless test harness.
 *
 * Nothing here imports three.js, so the segmentation + fitting pipeline can be
 * exercised under plain node. Browser-side modules such as faceGrouper.js
 * are thin adapters over this.
 *
 * Key difference from the original pipeline: fitting operates on FACETS
 * (centroid + geometric normal + area), never on raw vertices. Raw vertices are
 * counted once per incident triangle, which silently weights densely-tessellated
 * regions more heavily; and vertex normals read from an STL header are frequently
 * zero or unnormalized. Geometric normals recomputed here are always valid.
 */

// ── 3x3 linear algebra ───────────────────────────────────────────────────────

/**
 * Eigen-decomposition of a symmetric 3x3 matrix by cyclic Jacobi rotation.
 * @param {number[]} m  packed upper triangle [m00, m01, m02, m11, m12, m22]
 * @returns {{values: number[], vectors: number[][]}} ascending by eigenvalue
 */
export function eigen3(m) {
  const a = [
    [m[0], m[1], m[2]],
    [m[1], m[3], m[4]],
    [m[2], m[4], m[5]],
  ];
  const V = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];

  for (let iter = 0; iter < 100; iter++) {
    let off = 0, p = 0, q = 1;
    for (let i = 0; i < 3; i++)
      for (let j = i + 1; j < 3; j++)
        if (Math.abs(a[i][j]) > off) { off = Math.abs(a[i][j]); p = i; q = j; }
    if (off < 1e-15) break;

    const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
    const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(1 + theta * theta));
    const c = 1 / Math.sqrt(1 + t * t), s = t * c;
    const apq = a[p][q];

    a[p][p] -= t * apq;
    a[q][q] += t * apq;
    a[p][q] = a[q][p] = 0;
    for (let r = 0; r < 3; r++) {
      if (r === p || r === q) continue;
      const apr = a[p][r], aqr = a[q][r];
      a[p][r] = a[r][p] = c * apr - s * aqr;
      a[q][r] = a[r][q] = s * apr + c * aqr;
    }
    for (let r = 0; r < 3; r++) {
      const vp = V[r][p], vq = V[r][q];
      V[r][p] = c * vp - s * vq;
      V[r][q] = s * vp + c * vq;
    }
  }

  const order = [0, 1, 2].sort((i, j) => a[i][i] - a[j][j]);
  return {
    values: order.map(i => a[i][i]),
    vectors: order.map(i => [V[0][i], V[1][i], V[2][i]]),
  };
}

/** Solve a 3x3 system by Gaussian elimination with partial pivoting. */
export function solve3(A, b) {
  const M = [[...A[0], b[0]], [...A[1], b[1]], [...A[2], b[2]]];
  for (let col = 0; col < 3; col++) {
    let piv = col;
    for (let r = col + 1; r < 3; r++)
      if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    [M[col], M[piv]] = [M[piv], M[col]];
    if (Math.abs(M[col][col]) < 1e-13) return null;
    for (let r = col + 1; r < 3; r++) {
      const f = M[r][col] / M[col][col];
      for (let c = col; c <= 3; c++) M[r][c] -= f * M[col][c];
    }
  }
  const x = new Array(3);
  for (let i = 2; i >= 0; i--) {
    x[i] = M[i][3];
    for (let j = i + 1; j < 3; j++) x[i] -= M[i][j] * x[j];
    x[i] /= M[i][i];
  }
  return x;
}

export const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
export const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const norm = v => Math.sqrt(dot(v, v));

export function normalize(v) {
  const n = norm(v);
  return n > 1e-14 ? [v[0] / n, v[1] / n, v[2] / n] : [0, 0, 1];
}

/** An arbitrary but numerically stable unit vector perpendicular to v. */
export function perp(v) {
  const a = v.map(Math.abs);
  const i = a.indexOf(Math.min(...a));
  const e = [0, 0, 0]; e[i] = 1;
  return normalize(cross(v, e));
}

// ── Facets ───────────────────────────────────────────────────────────────────

/**
 * Derive per-triangle centroid, geometric unit normal, and area.
 *
 * Normals are recomputed from vertex winding rather than read from the source
 * file, so malformed STL facet normals (zero vectors are common) cannot poison
 * downstream fitting.
 *
 * @param {ArrayLike<number>} positions  flat non-indexed [x,y,z, ...], 9 per triangle
 * @returns {{count:number, centroids:Float64Array, normals:Float64Array, areas:Float64Array}}
 */
export function buildFacets(positions) {
  const count = Math.floor(positions.length / 9);
  const centroids = new Float64Array(count * 3);
  const normals = new Float64Array(count * 3);
  const areas = new Float64Array(count);
  const degenerate = new Uint8Array(count);

  for (let t = 0; t < count; t++) {
    const o = t * 9;
    const ax = positions[o],     ay = positions[o + 1], az = positions[o + 2];
    const bx = positions[o + 3], by = positions[o + 4], bz = positions[o + 5];
    const cx = positions[o + 6], cy = positions[o + 7], cz = positions[o + 8];

    const ux = bx - ax, uy = by - ay, uz = bz - az;
    const vx = cx - ax, vy = cy - ay, vz = cz - az;
    const nx = uy * vz - uz * vy;
    const ny = uz * vx - ux * vz;
    const nz = ux * vy - uy * vx;

    const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
    areas[t] = 0.5 * len;
    // DEGENERATE FACET: zero area, so it has no meaningful normal and is not surface.
    //
    // Real STLs carry these. 3DBenchy - the most-printed model there is - ships 563 of
    // them out of 225,706, of which 552 have two literally identical vertices. They
    // are invisible: zero area means nothing to see. But they are not harmless. A
    // sliver with vertices (A, B, B) contributes the edge A-B TWICE, so an ordinary
    // 2-manifold edge is counted as 4-manifold, buildAdjacency's "exactly two owners"
    // test rejects it, and the traversal walls off there. On Benchy that split the
    // mesh into 300 components under flood fill - which is what a bucket fill hitting
    // an invisible seam and refusing to continue actually is.
    //
    // They also carry a ZERO normal, which would poison any fit that averaged it in.
    if (!(len > 1e-20)) {
      degenerate[t] = 1;
    } else {
      normals[t * 3]     = nx / len;
      normals[t * 3 + 1] = ny / len;
      normals[t * 3 + 2] = nz / len;
    }
    centroids[t * 3]     = (ax + bx + cx) / 3;
    centroids[t * 3 + 1] = (ay + by + cy) / 3;
    centroids[t * 3 + 2] = (az + bz + cz) / 3;
  }
  // The raw positions ride along so fitters can use VERTICES for positional terms.
  // A CAD tessellator puts its vertices exactly ON the surface and the triangles
  // inscribe it, so every facet centroid sits strictly INSIDE a convex surface by
  // O(theta^2). Measured on OCCT output over 23 cylinders: vertices land within
  // 1e-10 of the true surface while every centroid is inside, by 0.69% of the radius
  // at coarse tessellation and 0.06% at fine. That is a density-dependent BIAS, not
  // noise, and it does not average out -- fitting the tangent-fillet wall to
  // centroids gives r=19.990482 for a true 20, and to vertices gives 20.000000.
  return { count, centroids, normals, areas, positions, degenerate };
}

/**
 * The inverse welding grid for a mesh: a millionth of its bounding-box diagonal.
 *
 * This was a fixed 1e4 (a 0.0001-unit grid), which is right for a 100 mm part and wrong for
 * one modelled in metres: WatchStandMedium.3mf is 0.09 m across, so the grid was 0.1 mm and
 * merged the vertices of its crush ribs. 708 of its 4,820 triangles lost a neighbour, which
 * walls fills off exactly as degenerate triangles would. For a part about 100 units across
 * the relative grid is the old one.
 */
export function weldQuantum(positions) {
  let lo = [Infinity, Infinity, Infinity];
  let hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i + 2 < positions.length; i += 3) {
    for (let a = 0; a < 3; a++) {
      const v = positions[i + a];
      if (v < lo[a]) lo[a] = v;
      if (v > hi[a]) hi[a] = v;
    }
  }
  const diagonal = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
  return diagonal > 0 && Number.isFinite(diagonal) ? 1e6 / diagonal : 1e4;
}

/**
 * Build triangle adjacency keyed on quantized shared edges.
 *
 * @param {ArrayLike<number>} positions  flat non-indexed vertex positions
 * @param {{count:number, normals:Float64Array}} facets  from buildFacets
 * @param {number} quant  inverse position quantum for vertex welding
 * @returns {Map<number, Array<{neighbor:number, angle:number}>>} angle in degrees
 */
export function buildAdjacency(positions, facets, quant = weldQuantum(positions)) {
  const key = (x, y, z) =>
    Math.round(x * quant) + '_' + Math.round(y * quant) + '_' + Math.round(z * quant);

  const edgeMap = new Map();
  for (let t = 0; t < facets.count; t++) {
    // Degenerate facets are excluded, not merely ignored: leaving them in inflates
    // edge multiplicity and makes real edges look non-manifold. See buildFacets.
    if (facets.degenerate && facets.degenerate[t]) continue;
    const o = t * 9;
    const k = [
      key(positions[o],     positions[o + 1], positions[o + 2]),
      key(positions[o + 3], positions[o + 4], positions[o + 5]),
      key(positions[o + 6], positions[o + 7], positions[o + 8]),
    ];
    for (let e = 0; e < 3; e++) {
      const a = k[e], b = k[(e + 1) % 3];
      const ek = a < b ? a + '|' + b : b + '|' + a;
      const hit = edgeMap.get(ek);
      if (hit) hit.push(t); else edgeMap.set(ek, [t]);
    }
  }

  const adjacency = new Map();
  for (let t = 0; t < facets.count; t++) adjacency.set(t, []);
  for (const tris of edgeMap.values()) {
    if (tris.length !== 2) continue;      // boundary or non-manifold edge
    const [a, b] = tris;
    const d = Math.max(-1, Math.min(1,
      facets.normals[a * 3]     * facets.normals[b * 3] +
      facets.normals[a * 3 + 1] * facets.normals[b * 3 + 1] +
      facets.normals[a * 3 + 2] * facets.normals[b * 3 + 2]));
    const angle = Math.acos(d) * 180 / Math.PI;
    adjacency.get(a).push({ neighbor: b, angle });
    adjacency.get(b).push({ neighbor: a, angle });
  }
  return adjacency;
}

/** BFS flood fill across edges whose dihedral angle is within thresholdDeg. */
export function floodFill(seed, adjacency, thresholdDeg, visited) {
  const region = new Set([seed]);
  const queue = [seed];
  while (queue.length) {
    const t = queue.pop();
    for (const edge of adjacency.get(t) || []) {
      if (edge.angle > thresholdDeg || region.has(edge.neighbor)) continue;
      if (visited && visited[edge.neighbor]) continue;
      region.add(edge.neighbor);
      queue.push(edge.neighbor);
    }
  }
  return region;
}

/** Segment every triangle into crease-bounded regions. */
export function segment(adjacency, count, thresholdDeg) {
  const visited = new Uint8Array(count);
  const groups = [];
  for (let t = 0; t < count; t++) {
    if (visited[t]) continue;
    const region = floodFill(t, adjacency, thresholdDeg, null);
    for (const i of region) visited[i] = 1;
    groups.push(region);
  }
  return groups;
}
