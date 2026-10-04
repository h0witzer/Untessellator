/**
 * seedNet.js — learned seeded fill: propose a region and a primitive type from one click.
 *
 * Optional. The converter works exactly as before without it; this is a third proposer
 * alongside crease fill and `paintFrom`, behind the same shape of interface. It emits
 * only `{ region, type }` — never geometry, never topology. The existing fitters in
 * surfaceFitter.js do all the actual surface work.
 *
 * FEATURE PARITY IS THE WHOLE GAME
 *
 * Every number built here must match ml/patches.py bit for bit in construction order,
 * normalisation and sign, because the weights were fitted against that exact
 * arrangement. A transposed axis or a differently-scaled area is not a small error — it
 * is a different input distribution, and the model degrades silently rather than
 * failing. `test/netparity.mjs` checks this side against the Python side on real meshes
 * and refuses to pass on drift.
 *
 * The ten features, in order (see FEATURE_NAMES in ml/patches.py):
 *   0-2  position relative to the seed centroid, in the canonical frame, / RMS radius
 *   3-5  unit normal in the canonical frame
 *   6    normal · seed normal
 *   7    log(area / mean patch area), clipped to +/-8
 *   8    BFS ring index / max ring
 *   9    |relative position| / RMS radius
 *   10-11 principal curvatures * RMS radius (the local surface-type signal)
 *
 * CANONICAL FRAME: the patch is rotated so the SEED'S NORMAL POINTS ALONG +Z. Training
 * additionally span the patch randomly about Z; at inference we do not, because a fixed
 * choice is one sample from the distribution the model was trained to be invariant to.
 */

export const TYPE_NAMES = ['Plane', 'Cylinder', 'Cone', 'Sphere', 'Torus',
  'Revolution', 'Extrusion', 'BSpline', 'Other'];

import { writeExtendedFeatures, N_EXT } from './seedFeatures.js';

export const N_FEATURES = 12;
/** Width of a v6 model's input: the 12 above plus js/seedFeatures.js's extended columns. */
export const N_FEATURES_V6 = N_FEATURES + N_EXT;

/** Types the converter has an analytic fitter for. Others route to the NURBS fallback. */
export const FITTABLE = new Set(['plane', 'cylinder', 'cone', 'sphere', 'torus']);

/** Map a model type name onto the converter's fitter vocabulary. */
export function fitterFor(typeName) {
  const lower = typeName.toLowerCase();
  return FITTABLE.has(lower) ? lower : null;
}

/**
 * Breadth-first walk over the dual graph, mirroring k_ring() in ml/meshcore.py.
 * Returns facet ids in BFS order plus each one's ring index.
 */
export function kRing(seed, adjacency, maxFacets, maxRings = 64, excluded = null) {
  const seen = new Set([seed]);
  const ids = [seed];
  const rings = [0];
  let frontier = [seed];
  for (let r = 1; r <= maxRings && ids.length < maxFacets; r++) {
    const next = [];
    for (const f of frontier) {
      for (const e of adjacency.get(f) || []) {
        // `excluded` (facets already painted into another group) is not part of the Python
        // mirror: with it the patch walks around them, as an experiment in test/fillEval.mjs.
        if (seen.has(e.neighbor) || (excluded && excluded[e.neighbor])) continue;
        seen.add(e.neighbor);
        next.push(e.neighbor);
      }
    }
    if (!next.length) break;
    // Python takes np.unique() of the frontier's neighbours, which sorts them. Match
    // that, or the ring contents agree but their order — and so the truncation at
    // maxFacets — does not.
    next.sort((a, b) => a - b);
    for (const f of next) {
      if (ids.length >= maxFacets) break;
      ids.push(f);
      rings.push(r);
    }
    frontier = next;
  }
  return { ids, rings };
}

/** Rotation whose rows map world coordinates into the seed's frame (n -> +Z). */
function frameFromNormal(n) {
  const len = Math.hypot(n[0], n[1], n[2]) || 1e-30;
  const nz = [n[0] / len, n[1] / len, n[2] / len];
  // Least-aligned world axis, matching np.argmin(np.abs(n)) including its tie-break
  // (first minimum wins).
  const a = [Math.abs(nz[0]), Math.abs(nz[1]), Math.abs(nz[2])];
  let k = 0;
  if (a[1] < a[k]) k = 1;
  if (a[2] < a[k]) k = 2;
  const axis = [0, 0, 0];
  axis[k] = 1;
  let t1 = [nz[1] * axis[2] - nz[2] * axis[1],
            nz[2] * axis[0] - nz[0] * axis[2],
            nz[0] * axis[1] - nz[1] * axis[0]];
  const l1 = Math.hypot(t1[0], t1[1], t1[2]) || 1e-30;
  t1 = [t1[0] / l1, t1[1] / l1, t1[2] / l1];
  const t2 = [nz[1] * t1[2] - nz[2] * t1[1],
              nz[2] * t1[0] - nz[0] * t1[2],
              nz[0] * t1[1] - nz[1] * t1[0]];
  return [t1, t2, nz];
}

/**
 * Build the (n, 10) feature block for one seed.
 * @returns {{features: Float32Array, ids: number[], n: number}}
 */
export function buildFeatures(seed, facets, adjacency, maxFacets = 512, curv = null, excluded = null, extended = false) {
  const width = extended ? N_FEATURES_V6 : N_FEATURES;
  curv = curv || facets.curvature || null;
  const { ids, rings } = kRing(seed, adjacency, maxFacets, 64, excluded);
  const n = ids.length;
  const R = frameFromNormal([facets.normals[seed * 3],
                             facets.normals[seed * 3 + 1],
                             facets.normals[seed * 3 + 2]]);
  const sx = facets.centroids[seed * 3];
  const sy = facets.centroids[seed * 3 + 1];
  const sz = facets.centroids[seed * 3 + 2];
  const sn = [facets.normals[seed * 3], facets.normals[seed * 3 + 1], facets.normals[seed * 3 + 2]];

  const d = new Float64Array(n * 3);
  const nrm = new Float64Array(n * 3);
  let sumSq = 0;
  let sumArea = 0;
  let maxRing = 0;
  for (let i = 0; i < n; i++) {
    const t = ids[i];
    const px = facets.centroids[t * 3] - sx;
    const py = facets.centroids[t * 3 + 1] - sy;
    const pz = facets.centroids[t * 3 + 2] - sz;
    for (let r = 0; r < 3; r++) d[i * 3 + r] = R[r][0] * px + R[r][1] * py + R[r][2] * pz;
    const nx = facets.normals[t * 3], ny = facets.normals[t * 3 + 1], nz2 = facets.normals[t * 3 + 2];
    for (let r = 0; r < 3; r++) nrm[i * 3 + r] = R[r][0] * nx + R[r][1] * ny + R[r][2] * nz2;
    sumSq += d[i * 3] * d[i * 3] + d[i * 3 + 1] * d[i * 3 + 1] + d[i * 3 + 2] * d[i * 3 + 2];
    sumArea += facets.areas[t];
    if (rings[i] > maxRing) maxRing = rings[i];
  }
  const scale = Math.sqrt(sumSq / n) || 1;
  const meanArea = sumArea / n || 1;
  const ringDiv = Math.max(1, maxRing);

  const out = new Float32Array(n * width);
  for (let i = 0; i < n; i++) {
    const t = ids[i];
    const dx = d[i * 3] / scale, dy = d[i * 3 + 1] / scale, dz = d[i * 3 + 2] / scale;
    const o = i * width;
    out[o] = dx; out[o + 1] = dy; out[o + 2] = dz;
    out[o + 3] = nrm[i * 3]; out[o + 4] = nrm[i * 3 + 1]; out[o + 5] = nrm[i * 3 + 2];
    out[o + 6] = facets.normals[t * 3] * sn[0] + facets.normals[t * 3 + 1] * sn[1]
               + facets.normals[t * 3 + 2] * sn[2];
    const la = Math.log(Math.max(facets.areas[t], 1e-30) / meanArea);
    out[o + 7] = Math.min(8, Math.max(-8, la));
    out[o + 8] = rings[i] / ringDiv;
    out[o + 9] = Math.hypot(dx, dy, dz);
    if (curv && curv.valid && curv.valid[t]) {
      out[o + 10] = Math.min(20, Math.max(-20, curv.k1[t] * scale));
      out[o + 11] = Math.min(20, Math.max(-20, curv.k2[t] * scale));
    }
  }
  if (extended) {
    const dScaled = new Float64Array(n * 3);
    for (let i = 0; i < n * 3; i++) dScaled[i] = d[i] / scale;
    writeExtendedFeatures(out, width, N_FEATURES, ids, seed, facets, adjacency, dScaled, nrm, scale, excluded);
  }
  return { features: out, ids, n, scale };
}

/**
 * Patch-local neighbours for a message-passing model (ml/model.py MeshConv): for patch facet i, the
 * patch indices of its (at most three) neighbours, -1 where one is missing or outside the patch, and
 * each edge's dihedral / 90. Mirrors ml/patches.py patch_graph. The model max-pools over the three
 * slots, so their order does not matter. Models without message passing ignore it.
 */
export function patchNeighbours(ids, adjacency) {
  const n = ids.length;
  const index = new Map();
  for (let i = 0; i < n; i++) index.set(ids[i], i);
  const nbr = new BigInt64Array(n * 3).fill(-1n);
  const eattr = new Float32Array(n * 3);
  // The learned boundary probability of the same edges (js/edgeWall.js annotateEdges), 0 when the
  // mesh has not been scored: a second edge attribute for refiners trained with --edge-dim 2.
  const eprob = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    let k = 0;
    for (const e of adjacency.get(ids[i]) || []) {
      if (k >= 3) break;
      const j = index.get(e.neighbor);
      if (j === undefined) continue;
      nbr[i * 3 + k] = BigInt(j);
      eattr[i * 3 + k] = e.angle / 90;
      eprob[i * 3 + k] = e.boundary ?? 0;
      k++;
    }
  }
  return { nbr, eattr, eprob };
}

/**
 * The surface a surface-head model predicted under `seed`, placed in the world in
 * regionGrow.surfaceEval's format. `surface` is the model's 5-vector (ml/surface_targets.py):
 * the axis in the patch frame, asinh(kn) and asinh(ks), where kn = scale / d and d is the
 * distance along the seed facet's normal to the axis, centre or tube centre.
 *
 * It exists to START growth. The analytic fitters still decide the final surface: measured
 * on ABC validation parts, handing a hypothesis-grown region to paintFrom beats paintFrom from
 * the model's region, while using the hypothesis itself as the boundary under-fills.
 *
 * @param {'plane'|'cylinder'|'cone'|'sphere'|'torus'} kind
 * @param {ArrayLike<number>} surface
 * @param {object} facets
 * @param {number} seed
 * @param {number} scale   the patch scale buildFeatures used for this seed
 * @returns {{type:string, params:object}|null}
 */
export function surfaceHypothesis(kind, surface, facets, seed, scale) {
  if (!surface) return null;
  const q = [facets.centroids[seed * 3], facets.centroids[seed * 3 + 1], facets.centroids[seed * 3 + 2]];
  const n = [facets.normals[seed * 3], facets.normals[seed * 3 + 1], facets.normals[seed * 3 + 2]];
  const along = (p, v, s) => [p[0] + s * v[0], p[1] + s * v[1], p[2] + s * v[2]];
  const unit = (v) => {
    const l = Math.hypot(v[0], v[1], v[2]);
    return l > 1e-300 ? [v[0] / l, v[1] / l, v[2] / l] : null;
  };

  const kn = Math.sinh(surface[3]);
  if (kind === 'plane' || Math.abs(kn) < 1e-6) return { type: 'plane', params: { origin: q, normal: n } };
  const s = kn > 0 ? 1 : -1;
  const d = scale / Math.abs(kn);
  const foot = along(q, n, -s * d);
  if (kind === 'sphere') return { type: 'sphere', params: { center: foot, radius: d } };

  // frame rows map world -> frame, so the world axis is R^T a
  const R = frameFromNormal(n);
  const a = unit([
    surface[0] * R[0][0] + surface[1] * R[1][0] + surface[2] * R[2][0],
    surface[0] * R[0][1] + surface[1] * R[1][1] + surface[2] * R[2][1],
    surface[0] * R[0][2] + surface[1] * R[1][2] + surface[2] * R[2][2],
  ]);
  if (!a) return null;
  const na = n[0] * a[0] + n[1] * a[1] + n[2] * a[2];
  if (kind === 'cylinder') return { type: 'cylinder', params: { axis: a, axisPoint: foot, radius: d } };
  if (kind === 'cone') {
    const sina = Math.min(Math.abs(na), 0.999999);
    if (sina < 1e-6) return { type: 'cylinder', params: { axis: a, axisPoint: foot, radius: d } };
    const sg = s * na >= 0 ? 1 : -1;
    const open = [-sg * a[0], -sg * a[1], -sg * a[2]];        // radius grows along `open`
    return { type: 'cone', params: { axis: open, apex: along(foot, open, -d / sina), halfAngle: Math.asin(sina) } };
  }
  if (kind === 'torus') {
    const ks = Math.sinh(surface[4]);
    if (Math.abs(ks) < 1e-9) return null;
    const up = unit([n[0] - na * a[0], n[1] - na * a[1], n[2] - na * a[2]]);
    if (!up) return null;
    const side = (ks > 0 ? 1 : -1) * s;
    const bigR = scale / Math.abs(ks);
    return { type: 'torus', params: { axis: a, center: along(foot, up, -side * bigR), majorRadius: bigR, minorRadius: d } };
  }
  return null;
}

/**
 * Run the model for one seed.
 *
 * `run` is an injected async (features, mask, n) => {sameLogit, typeLogit}, so this
 * module stays free of any runtime import — onnxruntime-node under test,
 * onnxruntime-web in the browser.
 */
export async function proposeRegion(seed, facets, adjacency, run, opts = {}) {
  const maxFacets = opts.maxFacets ?? 512;
  const threshold = opts.threshold ?? 0.5;
  const { features, ids, n, scale } = buildFeatures(seed, facets, adjacency, maxFacets, opts.curvature, opts.excluded, opts.extended);
  const { sameLogit, typeLogit, surface } = await run(features, n, patchNeighbours(ids, adjacency));

  const region = new Set();
  for (let i = 0; i < n; i++) {
    if (1 / (1 + Math.exp(-sameLogit[i])) >= threshold) region.add(ids[i]);
  }
  region.add(seed);   // the seed is on its own surface by definition

  let best = 0;
  for (let i = 1; i < typeLogit.length; i++) if (typeLogit[i] > typeLogit[best]) best = i;
  const probs = softmax(typeLogit);
  return {
    region,
    type: TYPE_NAMES[best],
    fitter: fitterFor(TYPE_NAMES[best]),
    confidence: probs[best],
    probs,
    patchSize: n,
    // present only when the model has a surface head (ml/surface_targets.py)
    surface: surface ? Array.from(surface) : null,
    scale,
  };
}

function softmax(v) {
  let m = -Infinity;
  for (const x of v) if (x > m) m = x;
  let s = 0;
  const out = new Array(v.length);
  for (let i = 0; i < v.length; i++) { out[i] = Math.exp(v[i] - m); s += out[i]; }
  for (let i = 0; i < v.length; i++) out[i] /= s;
  return out;
}
