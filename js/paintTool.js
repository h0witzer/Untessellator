/**
 * paintTool.js — experimental click-to-fill, for trying the learned fill aids by feel.
 *
 * Not part of the conversion pipeline, and loaded only when the user first paints, so the
 * ONNX runtime is never fetched otherwise. It wraps the deterministic pieces (meshCore,
 * regionGrow.paintFrom, the gaussMap fitters) and the optional learned proposer (seedNet),
 * so one click can be filled several ways on the same mesh:
 *
 *   plain      paintFrom from the click, no model (the type comes from the Type box, or the
 *              model's prediction when that is left on auto)
 *   model      the model's proposed region, grown and refit by paintFrom
 *   hyp        the model's SURFACE HYPOTHESIS grows a region first, then paintFrom takes over
 *   veto       paintFrom with an adaptive angle tolerance and a membership veto
 *   hyp+veto   both
 *
 * Measured on 80 ABC validation parts (docs/ml-experiment-scope.md, "Surface hypothesis head"),
 * clean fills with the true type: hyp beats model by about 8 points on CAD-exported meshes;
 * veto helps isotropic (gmsh-like) meshes when the model was trained on them; neither helps
 * low-poly game art. These numbers are what "by feel" is checking against.
 */

import * as ort from 'onnxruntime-web';
import { buildFacets, buildAdjacency, dot } from './meshCore.js';
import { paintFrom, surfaceEval, creaseFor, creaseComponent, CREASE_BARRIER } from './regionGrow.js';

/** The crease barrier for a region with no analytic type (the model's own region). */
export const FREEFORM_CREASE = 30;

/** Primitive complexity, simplest first: the order classifyRegion prefers. */
const ORDER_RANK = { plane: 0, cylinder: 1, cone: 2, sphere: 3, torus: 4 };
import { fitRegionAs, classifyRegion } from './gaussMap.js';
import { attachCurvature } from './curvature.js';
import { buildFeatures, proposeRegion, surfaceHypothesis, fitterFor, N_FEATURES } from './seedNet.js';

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
ort.env.wasm.numThreads = 1;   // no cross-origin isolation here, so threads are unavailable anyway

const DEG = Math.PI / 180;
const COS15 = Math.cos(15 * DEG);
const sessions = new Map();

/** One ONNX session per model URL, created on first use. */
export function loadModel(url) {
  if (!sessions.has(url)) {
    sessions.set(url, ort.InferenceSession.create(url, { executionProviders: ['wasm'] }));
  }
  return sessions.get(url);
}

function makeRunner(session) {
  return async (features, n) => {
    const out = await session.run({
      features: new ort.Tensor('float32', features, [1, n, features.length / n]),
      mask: new ort.Tensor('float32', new Float32Array(n).fill(1), [1, n]),
    });
    return {
      sameLogit: out.same_logit.data,
      typeLogit: Array.from(out.type_logit.data),
      surface: out.surface ? out.surface.data : null,
    };
  };
}

/**
 * Facets, adjacency and curvature for a loaded geometry. Done once per mesh: adjacency is the
 * expensive part. Triangle order matches the geometry, so viewer.pickTriangle indices apply.
 */
export function preparePaintMesh(geometry) {
  const geo = geometry.index ? geometry.toNonIndexed() : geometry;
  const P = Float64Array.from(geo.attributes.position.array);
  const facets = buildFacets(P);
  const adjacency = buildAdjacency(P, facets);
  attachCurvature(P, facets);   // the torus fitter uses it
  const sizes = [];
  for (let t = 0; t < facets.count; t++) sizes.push(Math.sqrt(Math.max(facets.areas[t], 0)));
  sizes.sort((a, b) => a - b);
  return { facets, adjacency, tolD: 0.5 * (sizes[sizes.length >> 1] || 1) };
}

const centroid = (facets, t) =>
  [facets.centroids[t * 3], facets.centroids[t * 3 + 1], facets.centroids[t * 3 + 2]];
const normal = (facets, t) =>
  [facets.normals[t * 3], facets.normals[t * 3 + 1], facets.normals[t * 3 + 2]];

/** regionGrow's growOn test against a given surface, flooding from `seedSet`. */
function growAgainst(seedSet, mesh, type, params, tolD, cosTol, taken = null) {
  const { facets, adjacency } = mesh;
  const creaseDeg = creaseFor(CREASE_BARRIER, type);
  const ev = surfaceEval(type, params);
  if (!ev) return new Set();
  const fits = (t) => {
    const c = centroid(facets, t);
    if (Math.abs(ev.distance(c)) > tolD) return false;
    return Math.abs(dot(ev.normal(c), normal(facets, t))) >= cosTol;
  };
  const region = new Set();
  const queue = [];
  for (const s of seedSet) if (!(taken && taken[s]) && fits(s)) { region.add(s); queue.push(s); }
  while (queue.length) {
    const t = queue.pop();
    for (const e of adjacency.get(t) || []) {
      if (region.has(e.neighbor) || (taken && taken[e.neighbor]) || e.angle >= creaseDeg || !fits(e.neighbor)) continue;
      region.add(e.neighbor);
      queue.push(e.neighbor);
    }
  }
  return region;
}

/**
 * paintFrom's grow/refit loop with two acceptance changes: the angle tolerance adapts to the
 * region's own normal deviation (3 x p90, clamped 4-15 deg), and a facet must also have
 * membership probability >= 0.5 from a model patch seeded at the accepted facet next to it.
 */
async function paintWithVeto(seed, type, mesh, seedRegion, runner, taken = null, extended = false) {
  const { facets, adjacency, tolD } = mesh;
  const creaseDeg = creaseFor(CREASE_BARRIER, type);
  const prob = new Map();
  let calls = 0;
  const patch = async (centre) => {
    const { features, ids, n } = buildFeatures(centre, facets, adjacency, 512, null, taken, extended);
    const { sameLogit } = await runner(features, n);
    calls++;
    for (let i = 0; i < n; i++) if (!prob.has(ids[i])) prob.set(ids[i], 1 / (1 + Math.exp(-sameLogit[i])));
  };
  await patch(seed);

  const grow = async (seedSet, params) => {
    const ev = surfaceEval(type, params);
    if (!ev) return new Set();
    const dev = t => Math.acos(Math.min(1, Math.abs(dot(ev.normal(centroid(facets, t)), normal(facets, t))))) / DEG;
    const devs = [...seedSet].map(dev).sort((a, b) => a - b);
    const angle = Math.min(15, Math.max(4, 3 * (devs[Math.floor(0.9 * (devs.length - 1))] ?? 5)));
    const cosTol = Math.cos(angle * DEG);
    const fits = t => Math.abs(ev.distance(centroid(facets, t))) <= tolD
      && Math.abs(dot(ev.normal(centroid(facets, t)), normal(facets, t))) >= cosTol;
    const region = new Set();
    const queue = [];
    for (const s of seedSet) {
      if ((taken && taken[s]) || !fits(s) || (prob.has(s) && prob.get(s) < 0.5)) continue;
      region.add(s); queue.push(s);
    }
    while (queue.length) {
      const t = queue.pop();
      for (const e of adjacency.get(t) || []) {
        const nb = e.neighbor;
        if (region.has(nb) || (taken && taken[nb]) || e.angle >= creaseDeg || !fits(nb)) continue;
        if (!prob.has(nb)) await patch(t);
        if ((prob.get(nb) ?? 1) < 0.5) continue;
        region.add(nb); queue.push(nb);
      }
    }
    return region;
  };

  let region = creaseComponent(seed, seedRegion, adjacency, creaseDeg);
  let fit = fitRegionAs(type, region, facets);
  if (fit.degenerate || !fit.params) return { region: new Set([seed]), calls };
  for (let i = 0; i < 4; i++) {
    const grown = await grow(region, fit.params);
    if (grown.size === 0) break;
    const refit = fitRegionAs(type, grown, facets);
    if (refit.degenerate || !refit.params) break;
    const stable = grown.size === region.size;
    region = grown;
    fit = refit;
    if (stable) break;
  }
  return { region, calls };
}

/**
 * Fill from one clicked triangle.
 *
 * @param {number} seed
 * @param {object} mesh     from preparePaintMesh
 * @param {object} opts
 * @param {'plain'|'model'|'hyp'|'veto'|'hyp+veto'} opts.method
 * @param {string} opts.type      'auto' or a fitter type
 * @param {string} opts.modelUrl
 */
export async function paintAt(seed, mesh, { method, type, modelUrl, taken = null, extended = false }) {
  // `extended`: the model takes the v6 inputs (js/seedFeatures.js), named in its export's
  // seedfill.json by ml/export_onnx.py.
  // `taken` marks facets already painted into OTHER groups. The model's patch walks around them
  // and no growth enters them. Simulated on 80 ABC parts with simpler faces painted first
  // (test/fillEval.mjs --sequential, oracle painting): clean fills on non-plane faces
  // 56 -> 60% OCCT, 47 -> 51% NURBS, 46 -> 58% gmsh against dropping them only afterwards;
  // gmsh tori 33 -> 53%, spheres 50 -> 68%.
  const t0 = performance.now();
  const { facets, adjacency } = mesh;
  const out = { region: new Set([seed]), modelRegion: null, predicted: null, confidence: null,
                fitter: null, fit: null, calls: 0, notes: [] };

  let p = null;
  let runner = null;
  if (method !== 'plain' || type === 'auto') {
    runner = makeRunner(await loadModel(modelUrl));
    p = await proposeRegion(seed, facets, adjacency, runner, { excluded: taken, extended });
    out.calls++;
    out.predicted = p.type;
    out.confidence = p.confidence;
    out.modelRegion = p.region;
  }

  // On auto the MODEL proposes the region and the FITTERS choose the type, simplest first:
  // classifyRegion accepts the simplest primitive within 1% (plane < cylinder < cone < sphere <
  // torus), because a cone can always imitate a cylinder and the type head calls cylinders cones.
  // CAPPED BY THE MODEL: when that pick is more complex than the model's own type and the model's
  // type still fits within 5%, the model's type wins, or a large sphere takes cylinders and cones
  // whose own fit narrowly missed 1%.
  // Measured on 80 ABC parts (test/fillEval.mjs --ordered, s1): true cylinders named Cylinder
  // 58 -> 76% on OCCT meshes, 54 -> 69% NURBS, 10 -> 57% gmsh; cones 73-80% kept; clean fills
  // 50 -> 52% OCCT, 42 -> 46% NURBS, 36 -> 42% gmsh, 31 -> 29% game. Overall type accuracy on
  // OCCT meshes drops 68 -> 62%, because freeform faces (BSpline, Extrusion) get primitive names.
  let kind = type;
  if (type === 'auto') {
    const cls = classifyRegion(creaseComponent(seed, p.region, adjacency, FREEFORM_CREASE), facets);
    const modelKind = fitterFor(p.type);
    let chosen = cls.type;
    if (modelKind) {
      const own = Math.min(Infinity, ...cls.candidates.filter((c) => c.type === modelKind).map((c) => c.relResidual));
      if ((chosen === 'nurbs' || ORDER_RANK[chosen] > ORDER_RANK[modelKind]) && own <= 0.05) chosen = modelKind;
    }
    out.orderedType = chosen;
    kind = chosen === 'nurbs' ? null : chosen;
  }
  if (!kind) {
    out.region = creaseComponent(seed, p.region, adjacency, FREEFORM_CREASE);
    out.notes.push('No analytic surface fits the model\'s region, so this is the model\'s own region.');
    out.ms = performance.now() - t0;
    return out;
  }
  out.fitter = kind;

  if (method === 'plain') {
    out.region = paintFrom(seed, kind, facets, adjacency, { taken }).region;
  } else if (method === 'model') {
    out.region = paintFrom(seed, kind, facets, adjacency, { seedRegion: p.region, taken }).region;
  } else if (method === 'veto') {
    const r = await paintWithVeto(seed, kind, mesh, p.region, runner, taken, extended);
    out.region = r.region;
    out.calls += r.calls;
  } else {
    const hyp = surfaceHypothesis(kind, p.surface, facets, seed, p.scale);
    let start = p.region;
    if (!p.surface) {
      out.notes.push('This model has no surface output; started from its region instead.');
    } else if (!hyp) {
      out.notes.push('The predicted surface was degenerate; started from the model region instead.');
    } else {
      out.hypothesis = hyp;
      const reachable = creaseComponent(seed, p.region, adjacency, creaseFor(CREASE_BARRIER, hyp.type));
      const grown = growAgainst(reachable, mesh, hyp.type, hyp.params, mesh.tolD, COS15, taken);
      if (grown.size >= 3) start = grown;
      else out.notes.push('Almost nothing matched the predicted surface; started from the model region.');
      out.fitter = hyp.type;
    }
    if (method === 'hyp') {
      out.region = paintFrom(seed, out.fitter, facets, adjacency, { seedRegion: start, taken }).region;
    } else {
      const r = await paintWithVeto(seed, out.fitter, mesh, start, runner, taken, extended);
      out.region = r.region;
      out.calls += r.calls;
    }
  }

  if (out.region.size <= 1) out.notes.push(`The ${out.fitter} fit failed on this region, so only the clicked triangle is filled.`);
  const fit = fitRegionAs(out.fitter, out.region, facets);
  out.fit = fit.params ? { params: fit.params, relResidual: fit.relResidual } : { reason: fit.reason };
  out.ms = performance.now() - t0;
  return out;
}
