/**
 * growEval.mjs — growth experiments on declared fills: the true type is given, so every difference
 * between methods is the GROWTH step's (walls, vetoes, cuts), not the type's.
 *
 * Usage:  node test/growEval.mjs <dir> --edge-wall <run dir> [--edge-wall-even <run dir>]
 *                                [--methods flood,wall,oracle,...] [--parity odd|even|all]
 *                                [--model <seedfill.onnx>] [--leaks <method>] [--json out.json]
 *
 * Clicks are chosen exactly as test/fillEval.mjs chooses them (same generator, same faces, same
 * seed triangles), so `wall` here reproduces fillEval's `declared crease=default edge=fpr2np` row.
 *
 * `--edge-wall-even` scores EVEN parts with a second edge model (one trained on the odd parts), so
 * both halves of the set carry held-out boundary scores: tune on even, report on odd.
 *
 * Methods (a trailing `@m` seeds the fill with the --model's proposed region instead of paintFrom's
 * 12-facet BFS; the type is always the true one):
 *   flood                paintFrom, crease barrier only
 *   wall[:fprN]          paintFrom with the learned wall at that threshold, curved types only (the
 *                        shipped default is fpr2)
 *   oracle               wall, plus: a facet of another B-rep face is refused when that face's own
 *                        (true-face) fitted surface explains it clearly better than the fill's
 *   hyst:<hi>/<lo>[/<minChain>]   wall by hysteresis (edgeWall.hysteresisWall); hi/lo are fpr1/fpr2/
 *                        fpr5 or a probability
 *   veto[:tau]           paintCore.paintWithVeto from the --model's region with the wall on: adaptive
 *                        angle, and a facet needs membership >= tau (default 0.5) from a patch at
 *                        the accepted facet beside it. Every consulted probability is recorded with
 *                        its truth (JSON `vetoQueries`), to set tau by a false-reject rate.
 *   refine               the fill-refinement model (--refine <onnx>, js/refineFeatures.js) re-labels
 *                        the patch around the click given the `wall` fill; the region is what it keeps
 *   refine+grow          ... then paintFrom regrows from it, never entering facets it rejected
 *   click@refine[+grow]  a simulated NEGATIVE CLICK on the spill (refineFeatures.simulatedClick),
 *                        only where the `wall` fill spilled; clean fills are left as they are
 *   click@geo            the same click without a model: the simplest surface fitted around the click
 *                        claims the fill's facets it explains better than the fill's own surface
 *                        (connected to the click), and paintFrom regrows without them
 *   studio:<method>[+r]  the studio's own fill (paintCore.paintWith: the --model's region, the type chosen
 *                        automatically, the wall), method model/hyp/veto/plain; `+r` adds the refined
 *                        fill (paintCore.refineFill with --refine). The TYPE is not declared here.
 *   cut[:k=v;k=v]        cutFill.cutFrom; keys lambda, gamma, maxScore, area (0/1), edges (0/1),
 *                        max (1: score by the worse of distance and angle, paintFrom's own box)
 *
 * `--leaks <method>` explains that method's fills: for each spill, whether the boundary it crossed
 * was mostly walled (it went through gaps) or not walled at all; for each incomplete fill, what
 * stopped it inside the true face (crease, wall, or the fit test).
 */

import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { createRequire } from 'node:module';

import { buildFacets, buildAdjacency, dot } from '../js/meshCore.js';
import { paintFrom, surfaceEval, creaseFor, creaseTableFor, lowPolyDefaults, creaseComponent } from '../js/regionGrow.js';
import { fitRegionAs } from '../js/gaussMap.js';
import { annotateEdges, annotateEdgesStacked, edgeWall, hysteresisWall } from '../js/edgeWall.js';
import { attachEdgeGraph } from '../js/edgeGraph.js';
import { cutFrom } from '../js/cutFill.js';
import { paintWithVeto, paintWith } from '../js/paintCore.js';
import { refineFeatures, simulatedClick } from '../js/refineFeatures.js';
import { classifyRegion } from '../js/gaussMap.js';
import { creaseRing } from '../js/seedFeatures.js';
import { attachCurvature } from '../js/curvature.js';
import { proposeRegion, TYPE_NAMES, fitterFor, graphInputs } from '../js/seedNet.js';

const ort = createRequire(import.meta.url)('onnxruntime-node');
/** The patch graph a message-passing model takes (seedNet.patchNeighbours, seedNet.graphInputs), if it
 * takes one; `edgeDim` 2 adds the learned boundary probability for refiners trained with --edge-dim 2. */
const graphFeeds = (session, graph, n, edgeDim = 1) => {
  if (!(session.inputNames.includes('nbr') && graph)) return {};
  const g = graphInputs(graph, n, edgeDim);
  return { nbr: new ort.Tensor('int64', ...g.nbr), eattr: new ort.Tensor('float32', ...g.eattr) };
};
/** A model's edge_dim from its sidecar JSON (1 when it has none). */
const edgeDimOf = (onnxPath) => {
  try { return JSON.parse(readFileSync(onnxPath.replace(/\.onnx$/, '.json'), 'utf8')).edge_dim ?? 1; } catch { return 1; }
};

const args = process.argv.slice(2);
function flag(name, dflt) {
  const i = args.indexOf(name);
  if (i < 0) return dflt;
  const v = args[i + 1];
  args.splice(i, 2);
  return v;
}
const PARITY = flag('--parity', 'odd');
const EDGE_DIR = flag('--edge-wall', null);
const EDGE_DIR_EVEN = flag('--edge-wall-even', null);
// A second-stage (stacked) edge model re-scores every edge from the first stage's scores around it;
// its sidecar's thresholds then name the walls. Both stages must be held out on the scored parts.
const EDGE_STAGE2 = flag('--edge-stage2', null);
const METHODS = flag('--methods', 'flood,wall').split(',');
const MODEL = flag('--model', null);
const REFINE = flag('--refine', null);
const LEAKS = flag('--leaks', null);
// Region-level features of every fill, for a verifier that picks among candidate fills (test step 8).
const VERIFIER = args.includes('--verifier-features');
if (VERIFIER) args.splice(args.indexOf('--verifier-features'), 1);
const JSON_OUT = flag('--json', null);
const PART_LIMIT = Number(flag('--parts', '0'));
const [DIR] = args;
if (!DIR) { console.error('usage: node test/growEval.mjs <dir> --edge-wall <run dir> [--methods ...]'); process.exit(2); }

const CURVED = ['cylinder', 'cone', 'sphere', 'torus'];
const SEEDS_PER_TYPE = 2;
const MAX_SEEDS_PER_PART = 10;

async function loadEdge(dir) {
  if (!dir) return null;
  const meta = JSON.parse(readFileSync(join(dir, 'edge.json'), 'utf8'));
  const session = await ort.InferenceSession.create(join(dir, 'edge.onnx'), { intraOpNumThreads: 2, interOpNumThreads: 1 });
  const run = async (features, n) => (await session.run({ features: new ort.Tensor('float32', features, [n, features.length / n]) })).logit.data;
  const prob = (name) => {
    if (/^[0-9.]+$/.test(name)) return Number(name);
    const key = { fpr1: 'fpr0.01', fpr2: 'fpr0.02', fpr5: 'fpr0.05' }[name];
    const entry = key && meta.thresholds[key];
    if (!entry) throw new Error(`unknown edge threshold ${name}`);
    return 1 / (1 + Math.exp(-entry.logit));
  };
  return { meta, run, prob };
}
const edgeOdd = await loadEdge(EDGE_DIR);
const edgeEven = EDGE_DIR_EVEN ? await loadEdge(EDGE_DIR_EVEN) : edgeOdd;
const edgeStage2 = EDGE_STAGE2 ? await loadEdge(EDGE_STAGE2) : null;

let model = null;
if (MODEL) {
  const session = await ort.InferenceSession.create(MODEL);
  const metaPath = MODEL.replace(/\.onnx$/, '.json');
  const meta = existsSync(metaPath) ? JSON.parse(readFileSync(metaPath, 'utf8')) : {};
  model = {
    extended: meta.features === 'v6',
    run: async (features, n, graph) => {
      const out = await session.run({
        features: new ort.Tensor('float32', features, [1, n, features.length / n]),
        mask: new ort.Tensor('float32', new Float32Array(n).fill(1), [1, n]),
        ...graphFeeds(session, graph, n),
      });
      return { sameLogit: out.same_logit.data, typeLogit: Array.from(out.type_logit.data),
               surface: out.surface ? out.surface.data : null };
    },
  };
}
if (METHODS.some((m) => m.endsWith('@m')) && !model) { console.error('@m methods need --model'); process.exit(2); }
let refiner = null;
if (REFINE) {
  const session = await ort.InferenceSession.create(REFINE);
  const refineEdgeDim = edgeDimOf(REFINE);
  refiner = {
    run: async (features, n, graph) => (await session.run({
      features: new ort.Tensor('float32', features, [1, n, features.length / n]),
      mask: new ort.Tensor('float32', new Float32Array(n).fill(1), [1, n]),
      ...graphFeeds(session, graph, n, refineEdgeDim),
    })).same_logit.data,
  };
}
if (METHODS.some((m) => m.startsWith('refine') || m.startsWith('click@refine')) && !refiner) {
  console.error('refine methods need --refine <onnx>'); process.exit(2);
}

const copy = (b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.length);
const load = (stem) => ({
  P: Float64Array.from(new Float32Array(copy(readFileSync(join(DIR, `${stem}.pos`))))),
  F: new Int32Array(copy(readFileSync(join(DIR, `${stem}.fid`)))),
  T: new Uint8Array(copy(readFileSync(join(DIR, `${stem}.typ`)))),
});

// ── growth with hooks: paintFrom's loop, reproduced so a veto can see the current surface ──────────
const sizeCache = new WeakMap();
function medianFacetSize(facets) {
  if (!sizeCache.has(facets)) {
    const s = [];
    for (let t = 0; t < facets.count; t++) s.push(Math.sqrt(Math.max(facets.areas[t], 0)));
    s.sort((a, b) => a - b);
    sizeCache.set(facets, s[s.length >> 1] || 1);
  }
  return sizeCache.get(facets);
}

function growLab(seedTri, type, facets, adjacency, { wall = null, veto = null, seedRegion = null } = {}) {
  const lowPoly = lowPolyDefaults(adjacency);
  const tolD = (lowPoly.tolScale ?? 0.5) * medianFacetSize(facets);
  const cosTolA = Math.cos((lowPoly.angleTol ?? 15) * Math.PI / 180);
  const creaseDeg = creaseFor(creaseTableFor(adjacency), type);
  const w = wall && (!wall.types || wall.types.includes(type)) ? wall : null;
  let region;
  if (seedRegion && seedRegion.size) region = creaseComponent(seedTri, seedRegion, adjacency, creaseDeg, w);
  else {
    region = new Set([seedTri]);
    const queue = [seedTri];
    while (queue.length && region.size < 12) {
      const t = queue.shift();
      for (const e of adjacency.get(t) || []) {
        if (region.has(e.neighbor) || e.angle >= creaseDeg || (w && w(t, e))) continue;
        region.add(e.neighbor);
        queue.push(e.neighbor);
        if (region.size >= 12) break;
      }
    }
  }
  let fit = fitRegionAs(type, region, facets);
  if (fit.degenerate || !fit.params) return { region: new Set([seedTri]), params: null };
  const C = facets.centroids, N = facets.normals;
  for (let i = 0; i < 4; i++) {
    const ev = surfaceEval(type, fit.params);
    const fits = (t) => {
      const c = [C[t * 3], C[t * 3 + 1], C[t * 3 + 2]];
      if (Math.abs(ev.distance(c)) > tolD) return false;
      return Math.abs(dot([N[t * 3], N[t * 3 + 1], N[t * 3 + 2]], ev.normal(c))) >= cosTolA;
    };
    const grown = new Set();
    const queue = [];
    for (const s of region) if (fits(s) && !(veto && veto(s, ev, fit.params))) { grown.add(s); queue.push(s); }
    while (queue.length) {
      const t = queue.pop();
      for (const e of adjacency.get(t) || []) {
        const nb = e.neighbor;
        if (grown.has(nb) || e.angle >= creaseDeg || (w && w(t, e))) continue;
        if (!fits(nb) || (veto && veto(nb, ev, fit.params))) continue;
        grown.add(nb);
        queue.push(nb);
      }
    }
    if (grown.size === 0) break;
    const refit = fitRegionAs(type, grown, facets);
    if (refit.degenerate || !refit.params) break;
    const stable = grown.size === region.size;
    region = grown;
    fit = refit;
    if (stable) break;
  }
  return { region, params: fit.params };
}

/** Normalised residual of facet t to a surface evaluator: (d / tolD)^2 + (angle / tol)^2. */
function residualScore(ev, facets, t, tolD, angTol) {
  const C = facets.centroids, N = facets.normals;
  const c = [C[t * 3], C[t * 3 + 1], C[t * 3 + 2]];
  const d = ev.distance(c) / tolD;
  const n = ev.normal(c);
  const cos = Math.min(1, Math.abs(n[0] * N[t * 3] + n[1] * N[t * 3 + 1] + n[2] * N[t * 3 + 2]));
  const a = Math.acos(cos) / angTol;
  return Number.isFinite(d) ? d * d + a * a : Infinity;
}

// ── region features for a fill verifier ──
/**
 * What a whole fill looks like, without the truth: size, how well one surface explains it, and how
 * its outline sits on the learned boundary and on creases; how many facets just outside would also
 * fit (a stopped fill), and how many walls it holds inside (a leak through a gap).
 */
function regionFeatures(region, type, facets, adjacency, graph, wallThr) {
  const fit = fitRegionAs(type, region, facets);
  const lowPoly = lowPolyDefaults(adjacency);
  const tolD = (lowPoly.tolScale ?? 0.5) * medianFacetSize(facets);
  const angTol = (lowPoly.angleTol ?? 15) * Math.PI / 180;
  const creaseDeg = creaseFor(creaseTableFor(adjacency), type);
  const ev = fit.params ? surfaceEval(type, fit.params) : null;
  let outline = 0, outP = 0, outWall = 0, outCrease = 0, inside = 0, inWall = 0, outFit = 0;
  const seenOut = new Set();
  let area = 0;
  for (const t of region) {
    area += facets.areas[t];
    for (const e of adjacency.get(t) || []) {
      const u = e.neighbor;
      const p = e.boundary ?? 0;
      if (region.has(u)) {
        if (u > t) { inside++; if (p >= wallThr) inWall++; }
        continue;
      }
      outline++;
      outP += p;
      if (p >= wallThr) outWall++;
      if (e.angle >= creaseDeg) outCrease++;
      if (ev && !seenOut.has(u)) {
        seenOut.add(u);
        if (residualScore(ev, facets, u, tolD, angTol) <= 1) outFit++;
      }
    }
  }
  const types = ['plane', 'cylinder', 'cone', 'sphere', 'torus'];
  return [Math.log(region.size + 1), Math.log(area / (medianFacetSize(facets) ** 2) + 1),
          Math.log10(Math.max(1e-12, fit.relResidual ?? 1)), outline ? outP / outline : 0,
          outline ? outWall / outline : 0, outline ? outCrease / outline : 0,
          seenOut.size ? outFit / seenOut.size : 0, inside ? inWall / inside : 0,
          Math.log(outline + 1) - 0.5 * Math.log(region.size + 1), meshStepOf(adjacency) / 90,
          ...types.map((k) => (k === type ? 1 : 0))];
}
const stepCache = new WeakMap();
function meshStepOf(adjacency) {
  if (!stepCache.has(adjacency)) {
    const a = [];
    for (const [t, es] of adjacency) for (const e of es) if (e.neighbor > t && e.angle >= 1 && e.angle < 60) a.push(e.angle);
    a.sort((p, q) => p - q);
    stepCache.set(adjacency, a.length ? a[a.length >> 1] : 0);
  }
  return stepCache.get(adjacency);
}

// ── scoring ──────────────────────────────────────────────────────────────────────────────────────
const rows = [];
const leakRows = [];
const vetoQueries = [];
const stopRows = [];
function score(method, mesher, part, seed, T, F, region, truth) {
  let hit = 0, sameType = 0;
  for (const t of region) {
    if (truth.has(t)) hit++;
    else if (T[t] === T[seed]) sameType++;
  }
  const n = region.size || 1;
  const row = { method, mesher, part, face: F[seed], truthType: TYPE_NAMES[T[seed]], faceSize: truth.size,
                size: region.size, purity: hit / n, surfacePurity: (hit + sameType) / n, recall: hit / truth.size };
  rows.push(row);
  return row;
}

// ── leak anatomy ─────────────────────────────────────────────────────────────────────────────────
function leakAnatomy(ctx, row, region, truth, wallPred, finalFit, type) {
  const { facets, adjacency, F, T, seed, mesher } = ctx;
  const creaseDeg = creaseFor(creaseTableFor(adjacency), type);
  const home = T[seed];
  if (row.surfacePurity < 0.9) {
    // Foreign faces entered, by different canonical type.
    const foreignFacets = new Map();
    for (const t of region) if (T[t] !== home) {
      if (!foreignFacets.has(F[t])) foreignFacets.set(F[t], []);
      foreignFacets.get(F[t]).push(t);
    }
    const homeFaces = new Set();
    for (const t of region) if (T[t] === home) homeFaces.add(F[t]);
    for (const [G, tris] of foreignFacets) {
      // Every mesh edge between a home-type face the fill holds and G, and how many were walls.
      let boundary = 0, walled = 0, crossed = 0, crossedDirect = 0;
      const gSet = new Set(tris);
      for (const t of gSet) {
        for (const e of adjacency.get(t) || []) {
          const a = e.neighbor;
          if (!homeFaces.has(F[a]) || T[a] !== home) continue;
          boundary++;
          const isW = wallPred ? wallPred(a, e) : false;
          if (isW) walled++;
          if (region.has(a)) { crossed++; if (F[a] === F[seed]) crossedDirect++; }
        }
      }
      // All edges of G with the home faces (including where the fill did not reach).
      let fullBoundary = 0, fullWalled = 0;
      for (const t of ctx.byFace.get(G) || []) {
        for (const e of adjacency.get(t) || []) {
          const a = e.neighbor;
          if (!homeFaces.has(F[a]) || T[a] !== home) continue;
          fullBoundary++;
          if (wallPred && wallPred(a, e)) fullWalled++;
        }
      }
      // Depth: rings from the crossing into G's filled facets.
      const depth = new Map();
      const q = [];
      for (const t of gSet) {
        for (const e of adjacency.get(t) || []) {
          if (region.has(e.neighbor) && T[e.neighbor] === home) { depth.set(t, 1); q.push(t); break; }
        }
      }
      for (let k = 0; k < q.length; k++) {
        const t = q[k];
        for (const e of adjacency.get(t) || []) {
          if (gSet.has(e.neighbor) && !depth.has(e.neighbor)) { depth.set(e.neighbor, depth.get(t) + 1); q.push(e.neighbor); }
        }
      }
      const maxDepth = q.length ? Math.max(...depth.values()) : 0;
      // Scores of the crossed edges and of the unwalled ones on the boundary.
      leakRows.push({ mesher, type: row.truthType, foreignType: TYPE_NAMES[T[tris[0]]], spilled: tris.length,
                      boundary, walled, crossed, crossedDirect, fullBoundary, fullWalled,
                      indirect: q.length === 0, maxDepth, faceSize: row.faceSize });
    }
  }
  if (row.recall < 0.9) {
    // What stops the fill inside its own face: edges from filled truth facets to unfilled ones.
    const ev = finalFit ? surfaceEval(type, finalFit) : null;
    const lowPoly = lowPolyDefaults(adjacency);
    const tolD = (lowPoly.tolScale ?? 0.5) * medianFacetSize(facets);
    const cosTol = Math.cos((lowPoly.angleTol ?? 15) * Math.PI / 180);
    let crease = 0, wall = 0, misfit = 0, other = 0;
    for (const t of region) {
      if (!truth.has(t)) continue;
      for (const e of adjacency.get(t) || []) {
        const b = e.neighbor;
        if (!truth.has(b) || region.has(b)) continue;
        if (e.angle >= creaseDeg) crease++;
        else if (wallPred && (!wallPred.types || wallPred.types.includes(type)) && wallPred(t, e)) wall++;
        else if (ev) {
          const c = [facets.centroids[b * 3], facets.centroids[b * 3 + 1], facets.centroids[b * 3 + 2]];
          const nb = [facets.normals[b * 3], facets.normals[b * 3 + 1], facets.normals[b * 3 + 2]];
          const fitsB = Math.abs(ev.distance(c)) <= tolD && Math.abs(dot(nb, ev.normal(c))) >= cosTol;
          if (!fitsB) misfit++; else other++;
        } else other++;
      }
    }
    const reached = [...truth].filter((t) => region.has(t)).length;
    stopRows.push({ mesher, type: row.truthType, crease, wall, misfit, other, recall: row.recall,
                    faceSize: row.faceSize, reached });
  }
}

// ── the run ──────────────────────────────────────────────────────────────────────────────────────
const manifest = JSON.parse(readFileSync(join(DIR, 'manifest.json'), 'utf8'));
const reference = manifest.variants[0];
let x = 99;
const rng = () => (x = (x * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
const t0 = performance.now();
let partsDone = 0;

function parseCut(spec) {
  const o = {};
  for (const kv of (spec || '').split(';').filter(Boolean)) {
    const [k, v] = kv.split('=');
    o[k] = Number(v);
  }
  return {
    lambda: o.lambda ?? 1, gamma: o.gamma ?? 1, maxScore: o.maxScore ?? 9,
    area: (o.area ?? 1) !== 0, useEdges: (o.edges ?? 1) !== 0, norm: o.max ? 'max' : 'sum',
  };
}

for (const [partIndex, entry] of manifest.parts.entries()) {
  if (PARITY !== 'all' && (partIndex % 2 === 0) !== (PARITY === 'even')) continue;
  if (PART_LIMIT && partsDone >= PART_LIMIT) break;
  partsDone++;
  const ref = entry.variants[reference];
  if (!ref) continue;
  const base = load(ref.stem);
  const facesByType = new Map();
  for (let t = 0; t < base.F.length; t++) {
    if (!facesByType.has(base.T[t])) facesByType.set(base.T[t], new Set());
    facesByType.get(base.T[t]).add(base.F[t]);
  }
  const seedFaces = [];
  for (const faces of facesByType.values()) {
    const arr = [...faces];
    for (let i = 0; i < SEEDS_PER_TYPE && arr.length; i++) seedFaces.push(arr.splice(Math.floor(rng() * arr.length), 1)[0]);
  }
  seedFaces.splice(MAX_SEEDS_PER_PART);
  const edge = partIndex % 2 === 0 ? edgeEven : edgeOdd;

  for (const mesher of manifest.variants) {
    const v = entry.variants[mesher];
    if (!v) continue;
    const { P, F, T } = mesher === reference ? base : load(v.stem);
    const facets = buildFacets(P);
    const adjacency = buildAdjacency(P, facets);
    attachCurvature(P, facets);
    const graph = attachEdgeGraph(P, facets, adjacency);
    if (edge && edgeStage2) {
      await annotateEdgesStacked(facets, adjacency, graph, edge.run, edge.meta.columns, edgeStage2.run, edgeStage2.meta.columns);
    } else if (edge) {
      await annotateEdges(facets, adjacency, edge.run, edge.meta.columns);
    }
    const thresholds = edgeStage2 ?? edge;
    const byFace = new Map();
    for (let t = 0; t < facets.count; t++) {
      if (facets.degenerate[t]) continue;
      if (!byFace.has(F[t])) byFace.set(F[t], []);
      byFace.get(F[t]).push(t);
    }
    // Walls built once per mesh.
    const walls = new Map();
    const wallFor = (spec) => {
      if (!walls.has(spec)) {
        let w = null;
        if (spec.startsWith('wall')) {
          const thr = spec.split(':')[1] ?? 'fpr2';
          w = edgeWall(thresholds.prob(thr), CURVED);
        } else if (spec.startsWith('hyst:')) {
          const [hi, lo, minChain] = spec.slice(5).split('/');
          w = hysteresisWall(graph, adjacency, thresholds.prob(hi), thresholds.prob(lo), { types: CURVED, minChain: Number(minChain ?? 1) });
        } else if (spec === 'oracle') {
          w = edgeWall(thresholds.prob('fpr2'), CURVED);
        }
        walls.set(spec, w);
      }
      return walls.get(spec);
    };
    // True-face surfaces, for the oracle, fitted on demand.
    const trueSurface = new Map();
    const surfaceOfFace = (G) => {
      if (!trueSurface.has(G)) {
        const tris = byFace.get(G);
        const kind = tris ? fitterFor(TYPE_NAMES[T[tris[0]]]) : null;
        let ev = null;
        if (kind && tris.length >= 3) {
          const f = fitRegionAs(kind, new Set(tris), facets);
          if (f.params && !f.degenerate) ev = surfaceEval(kind, f.params);
        }
        trueSurface.set(G, ev);
      }
      return trueSurface.get(G);
    };
    const lowPoly = lowPolyDefaults(adjacency);
    const tolD = (lowPoly.tolScale ?? 0.5) * medianFacetSize(facets);
    const angTol = (lowPoly.angleTol ?? 15) * Math.PI / 180;

    for (const face of seedFaces) {
      const tris = byFace.get(face);
      if (!tris || tris.length < 3) continue;
      const seed = tris[Math.floor(rng() * tris.length)];
      const truth = new Set(tris);
      const tf = fitterFor(TYPE_NAMES[T[seed]]);
      if (!tf) continue;
      let proposal = null;
      let baseFill = null;
      const wallFill = () => (baseFill ??= paintFrom(seed, tf, facets, adjacency, { wall: wallFor('wall') }));
      const creaseDeg = creaseFor(creaseTableFor(adjacency), tf);
      const spilled = (region) => {
        let own = 0;
        for (const t of region) if (T[t] === T[seed]) own++;
        return own / (region.size || 1) < 0.9;
      };
      const applyRefiner = async (fill, click, cut = 0) => {
        const r = refineFeatures(seed, facets, adjacency, { region: fill.region, type: tf, params: fill.params }, click);
        if (click >= 0 && !r.ids.includes(click)) return null;
        const logits = await refiner.run(r.features, r.n, r.graph);
        const keep = new Set(fill.region);
        const rejected = new Uint8Array(facets.count);
        const logit = new Map();
        for (let i = 0; i < r.n; i++) {
          const t = r.ids[i];
          logit.set(t, logits[i]);
          if (logits[i] >= cut) keep.add(t); else { keep.delete(t); if (t !== seed) rejected[t] = 1; }
        }
        keep.add(seed);
        return { region: creaseComponent(seed, keep, adjacency, creaseDeg), rejected, logit };
      };
      for (const spec of METHODS) {
        const withModel = spec.endsWith('@m');
        const name = withModel ? spec.slice(0, -2) : spec;
        let seedRegion = null;
        if (withModel) {
          proposal ??= await proposeRegion(seed, facets, adjacency, model.run,
            { curvature: { valid: new Uint8Array(facets.count) }, extended: model.extended });
          seedRegion = proposal.region;
        }
        let out;
        let wallPred = null;
        if (name === 'flood') {
          out = paintFrom(seed, tf, facets, adjacency, seedRegion ? { seedRegion } : {});
        } else if (name.startsWith('wall') || name.startsWith('hyst:')) {
          wallPred = wallFor(name);
          out = paintFrom(seed, tf, facets, adjacency, { wall: wallPred, ...(seedRegion ? { seedRegion } : {}) });
        } else if (name === 'labwall') {
          // growLab without a veto must reproduce paintFrom with the wall exactly (a harness check)
          wallPred = wallFor('wall');
          out = growLab(seed, tf, facets, adjacency, { wall: wallPred, seedRegion });
        } else if (name === 'oracle') {
          wallPred = wallFor('oracle');
          const veto = (t, ev) => {
            if (F[t] === face) return false;
            const other = surfaceOfFace(F[t]);
            if (!other) return false;
            const mine = residualScore(ev, facets, t, tolD, angTol);
            const theirs = residualScore(other, facets, t, tolD, angTol);
            return theirs < 0.5 * mine;
          };
          out = growLab(seed, tf, facets, adjacency, { wall: wallPred, veto, seedRegion });
        } else if (name.startsWith('veto')) {
          const tau = Number(name.split(':')[1] ?? 0.5);
          proposal ??= await proposeRegion(seed, facets, adjacency, model.run,
            { curvature: { valid: new Uint8Array(facets.count) }, extended: model.extended });
          wallPred = wallFor('wall');
          const mesh = { facets, adjacency, tolD: 0.5 * medianFacetSize(facets) };
          const record = name === METHODS.find((m) => m.startsWith('veto'))
            ? (t, p) => vetoQueries.push([mesher === 'native' ? 0 : mesher === 'nurbs' ? 1 : mesher === 'gmsh' ? 2 : 3,
                                          p, F[t] === face ? 1 : 0, T[t] === T[seed] ? 1 : 0])
            : null;
          out = await paintWithVeto(seed, tf, mesh, proposal.region, model.run, null, model.extended,
                                    { tau, wall: wallPred, onQuery: record });
        } else if (name.split(':')[0] === 'refine' || name.split(':')[0] === 'refine+grow') {
          const fill = wallFill();
          const r = fill.params ? await applyRefiner(fill, -1, Number(name.split(':')[1] ?? 0)) : null;
          if (!r) out = fill;
          else if (name.split(':')[0] === 'refine') out = { region: r.region, params: null };
          else out = paintFrom(seed, tf, facets, adjacency, { seedRegion: r.region, wall: wallFor('wall'), taken: r.rejected });
        } else if (name.startsWith('studio:')) {
          const refineOn = name.endsWith('+r');
          const method = name.slice(7, refineOn ? -2 : undefined);
          const mesh = { facets, adjacency, tolD: 0.5 * medianFacetSize(facets), graph };
          const res = await paintWith(model.run, seed, mesh, {
            method, type: 'auto', extended: model.extended, reportFit: false,
            grow: { wall: wallFor('wall') },
            ...(refineOn ? { refine: { runner: async (f, n, g) => ({ sameLogit: await refiner.run(f, n, g) }) } } : {}),
          });
          out = { region: res.region, params: null };
        } else if (name.startsWith('iter:')) {
          // ITERATED refinement: refine, refit the surface to the refined region, refine again (n passes)
          const [, nStr, cutStr] = name.split(':');
          const cut = Number(cutStr ?? 1);
          let fill = wallFill();
          out = fill;
          for (let k = 0; k < Number(nStr) && fill.params; k++) {
            const r = await applyRefiner(fill, -1, cut);
            if (!r) break;
            const refit = fitRegionAs(tf, r.region, facets);
            out = { region: r.region, params: refit.params ?? null };
            if (!refit.params) break;
            fill = { region: r.region, params: refit.params };
          }
        } else if (name.startsWith('crf:')) {
          // KALOGERAKIS RECIPE: minimum cut with the REFINER's verdict as the per-facet term inside its
          // patch (logit - c, scaled by s, clamped to +/-1), the fit residual outside it, and the learned
          // boundary pricing cut edges. Keys a (alpha), l (lambda), r (1: refiner replaces the fit term),
          // c (logit centre, default 1: the refiner's tuned cut), s (scale, default 4).
          const o = {};
          for (const kv of name.slice(4).split(';').filter(Boolean)) { const [k, v] = kv.split('='); o[k] = Number(v); }
          const fill = wallFill();
          out = fill;
          const r = fill.params ? await applyRefiner(fill, -1, o.c ?? 1) : null;
          if (r) {
            const member = new Map();
            for (const [t, lg] of r.logit) member.set(t, 0.5 + 0.5 * Math.max(-1, Math.min(1, (lg - (o.c ?? 1)) / (o.s ?? 4))));
            out = cutFrom(seed, tf, facets, adjacency, graph, {
              lambda: o.l ?? 1, member, alpha: o.a ?? 2, memberReplacesFit: (o.r ?? 1) !== 0, seedRegion: r.region,
            });
          }
        } else if (name.startsWith('refclick')) {
          // refine (logit cut from the name, default 1), then ONE negative click if it still spills
          const cut = Number(name.split(':')[1] ?? 1);
          const fill = wallFill();
          out = fill;
          const r = fill.params ? await applyRefiner(fill, -1, cut) : null;
          if (r) {
            out = { region: r.region, params: null };
            if (spilled(r.region)) {
              const refit = fitRegionAs(tf, r.region, facets);
              const click = refit.params ? simulatedClick(seed, r.region, adjacency, facets, T) : -1;
              if (click >= 0) {
                const r2 = await applyRefiner({ region: r.region, params: refit.params }, click, cut);
                if (r2) out = { region: r2.region, params: null };
              }
            }
          }
        } else if (name.startsWith('click@')) {
          const fill = wallFill();
          out = fill;
          const click = fill.params && spilled(fill.region) ? simulatedClick(seed, fill.region, adjacency, facets, T) : -1;
          if (click >= 0 && name.startsWith('click@refine')) {
            const r = await applyRefiner(fill, click, Number(name.split(':')[1] ?? 0));
            if (r) {
              out = name.split(':')[0] === 'click@refine' ? { region: r.region, params: null }
                : paintFrom(seed, tf, facets, adjacency, { seedRegion: r.region, wall: wallFor('wall'), taken: r.rejected });
            }
          } else if (click >= 0 && name === 'click@geo') {
            const cls = classifyRegion(new Set(creaseRing(click, adjacency, 24, 30)), facets);
            const evC = cls.params ? surfaceEval(cls.type, cls.params) : null;
            const evF = surfaceEval(tf, fill.params);
            if (evC && evF) {
              const taken = new Uint8Array(facets.count);
              const comp = [click];
              taken[click] = 1;
              for (let k = 0; k < comp.length; k++) {
                for (const e of adjacency.get(comp[k]) || []) {
                  const u = e.neighbor;
                  if (taken[u] || !fill.region.has(u) || u === seed) continue;
                  if (residualScore(evC, facets, u, tolD, angTol) < residualScore(evF, facets, u, tolD, angTol)) {
                    taken[u] = 1;
                    comp.push(u);
                  }
                }
              }
              const rest = new Set([...fill.region].filter((t) => !taken[t]));
              out = paintFrom(seed, tf, facets, adjacency, { seedRegion: rest, wall: wallFor('wall'), taken });
            }
          }
        } else if (name.startsWith('cut')) {
          const o = parseCut(name.split(':')[1]);
          out = cutFrom(seed, tf, facets, adjacency, graph, { ...o, ...(seedRegion ? { seedRegion } : {}) });
        } else {
          console.error(`unknown method ${spec}`);
          process.exit(2);
        }
        const row = score(spec, mesher, partIndex, seed, T, F, out.region, truth);
        if (VERIFIER) row.vf = regionFeatures(out.region, tf, facets, adjacency, graph, thresholds.prob('fpr2'));
        if (LEAKS === spec) {
          leakAnatomy({ facets, adjacency, F, T, seed, mesher, byFace }, row, out.region, truth,
                      wallPred ?? wallFor('wall'), out.params, tf);
        }
      }
    }
  }
  if (partsDone % 10 === 0) console.log(`  ${partsDone} parts, ${rows.length} fills, ${((performance.now() - t0) / 1000).toFixed(0)} s`);
}

// ── report ───────────────────────────────────────────────────────────────────────────────────────
const quantile = (a, f) => { const s = [...a].sort((p, q) => p - q); return s.length ? s[Math.min(s.length - 1, Math.floor(f * s.length))] : NaN; };
const med = (a) => quantile(a, 0.5);
const pc = (v) => `${(100 * v).toFixed(0)}%`.padStart(4);
const frac = (rs, fn) => rs.filter(fn).length / (rs.length || 1);
const clean = (r) => r.surfacePurity >= 0.95 && r.recall >= 0.95;
const spill = (r) => r.surfacePurity < 0.9;
const incomplete = (r) => r.recall < 0.9;

console.log(`\n${partsDone} parts (${PARITY}), ${rows.length} fills, ${((performance.now() - t0) / 1000).toFixed(0)} s`);
console.log('clean: surface purity and recall >= 95%. spill: surface purity < 90%. incomplete: recall < 90%.\n');
for (const mesher of manifest.variants) {
  const at = rows.filter((r) => r.mesher === mesher);
  if (!at.length) continue;
  console.log(mesher);
  for (const m of METHODS) {
    const rs = at.filter((r) => r.method === m);
    console.log(`  ${m.padEnd(34)} n ${String(rs.length).padStart(4)}  clean ${pc(frac(rs, clean))}` +
      ` (strict ${pc(frac(rs, (r) => r.purity >= 0.95 && r.recall >= 0.95))})  spill ${pc(frac(rs, spill))}` +
      `  incomplete ${pc(frac(rs, incomplete))}  recall p50 ${pc(med(rs.map((r) => r.recall)))}`);
  }
}
console.log('\nclean / spill / incomplete by true type');
console.log(`  ${'mesher / type'.padEnd(20)}${METHODS.map((m) => m.slice(0, 20).padStart(22)).join('')}`);
for (const mesher of manifest.variants) {
  for (const t of ['Plane', 'Cylinder', 'Cone', 'Sphere', 'Torus']) {
    const cells = METHODS.map((m) => {
      const rs = rows.filter((r) => r.mesher === mesher && r.method === m && r.truthType === t);
      return rs.length ? `${pc(frac(rs, clean))}/${pc(frac(rs, spill))}/${pc(frac(rs, incomplete))}`.padStart(22) : '-'.padStart(22);
    });
    if (cells.some((c) => c.trim() !== '-')) console.log(`  ${`${mesher} ${t}`.padEnd(20)}${cells.join('')}`);
  }
}
if (METHODS.length > 1) {
  // Best of the methods per click: what a perfect chooser among them would reach.
  console.log('\nbest of the methods per click (clean by any), against the first method');
  const key = (r) => `${r.mesher}|${r.part}|${r.face}`;
  const byClick = new Map();
  for (const r of rows) { if (!byClick.has(key(r))) byClick.set(key(r), []); byClick.get(key(r)).push(r); }
  for (const mesher of manifest.variants) {
    const clicks = [...byClick.values()].filter((rs) => rs[0].mesher === mesher);
    if (!clicks.length) continue;
    const any = frac(clicks, (rs) => rs.some(clean));
    const first = frac(clicks, (rs) => rs.some((r) => r.method === METHODS[0] && clean(r)));
    console.log(`  ${mesher.padEnd(8)} ${METHODS[0]} ${pc(first)}   best of ${METHODS.length} ${pc(any)}`);
  }
}
if (LEAKS) {
  console.log(`\nLEAK ANATOMY for ${LEAKS}: each (spilled fill, foreign face) entry`);
  for (const mesher of manifest.variants) {
    const L = leakRows.filter((r) => r.mesher === mesher);
    if (!L.length) continue;
    const total = L.reduce((s, r) => s + r.spilled, 0);
    const bucket = (r) => (r.indirect ? 'indirect' : r.fullBoundary === 0 ? 'no shared edge'
      : r.fullWalled / r.fullBoundary >= 0.8 ? 'walled >= 80% (gaps)'
      : r.fullWalled / r.fullBoundary >= 0.2 ? 'walled 20-80%' : 'walled < 20%');
    const groups = new Map();
    for (const r of L) { const b = bucket(r); if (!groups.has(b)) groups.set(b, []); groups.get(b).push(r); }
    console.log(`  ${mesher}: ${L.length} entries, ${total} spilled facets`);
    for (const [b, rs] of [...groups].sort()) {
      const n = rs.reduce((s, r) => s + r.spilled, 0);
      console.log(`    ${b.padEnd(22)} entries ${pc(rs.length / L.length)}  spilled facets ${pc(n / total)}` +
        `  crossed edges p50 ${med(rs.map((r) => r.crossed))}  depth p50 ${med(rs.map((r) => r.maxDepth))} p90 ${quantile(rs.map((r) => r.maxDepth), 0.9)}` +
        `  foreign ${[...new Set(rs.map((r) => r.foreignType))].slice(0, 4).join('/')}`);
    }
  }
  console.log(`\nSTOP REASONS for ${LEAKS}: incomplete fills, share of the edges where growth stopped inside the true face`);
  for (const mesher of manifest.variants) {
    const S = stopRows.filter((r) => r.mesher === mesher);
    if (!S.length) continue;
    const sum = (k) => S.reduce((s, r) => s + r[k], 0);
    const all = sum('crease') + sum('wall') + sum('misfit') + sum('other');
    const dominant = (r) => ['crease', 'wall', 'misfit', 'other'].reduce((a, k) => (r[k] > r[a] ? k : a), 'crease');
    const dom = new Map();
    for (const r of S) dom.set(dominant(r), (dom.get(dominant(r)) ?? 0) + 1);
    console.log(`  ${mesher.padEnd(8)} ${S.length} fills; edges crease ${pc(sum('crease') / all)} wall ${pc(sum('wall') / all)}` +
      ` misfit ${pc(sum('misfit') / all)} other ${pc(sum('other') / all)}; dominant per fill ${
        [...dom].map(([k, n]) => `${k} ${pc(n / S.length)}`).join(', ')}`);
  }
}
if (JSON_OUT) {
  writeFileSync(JSON_OUT, JSON.stringify({ rows, leakRows, stopRows, vetoQueries }));
  console.log(`\nwrote ${rows.length} rows to ${JSON_OUT}`);
}
