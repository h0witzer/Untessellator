/**
 * surfaceScore.mjs — an automatic segmentation scored by what a B-rep needs, against the true surfaces.
 *
 * Usage:  node test/surfaceScore.mjs <mesh dir> <truth dir> <seg dir> [--variants a,b] [--tol 5e-4]
 *                                    [--meshers native,gmsh] [--shard k/n] [--json out.json]
 *                                    [--patches-out dir]
 *
 *   <mesh dir>   test/fillEval.mjs's format (.pos/.fid/.typ, manifest.json), from ml/export_fill_eval.py
 *                or ml/export_abc_obj_eval.py
 *   <truth dir>  <part>.srf, the true surface of every face id, from ml/export_fit_truth.py or
 *                export_abc_obj_eval.py --truth-only
 *   <seg dir>    test/autoSeed.mjs --dump-seg: painted segments and their types, the refiner's
 *                membership per facet, the boundary model's largest probability per facet
 *
 * WHY. The solid rebuild (ml/solid_rebuild.py, topo) takes each segment boundary, corner to corner,
 * and projects its vertices onto BOTH fitted surfaces, so where a boundary runs between two correct
 * surfaces hardly matters, and a triangle within tolerance of both is no error at all. What a solid
 * needs is every face present, in one piece, on the right surface. "Clean" (one segment holds 95% of
 * a face and is 95% its type) punishes harmless boundary slop, misses spurious slivers, and never asks
 * whether a group got a surface at all. So this scores, per true analytic face, with L the segment
 * holding most of the face's area:
 *   found     L carries a surface of the true type within 1e-3 (found3) or 1e-4 (found4) x the part
 *             diagonal of every point of the true face (geo3: any type)
 *   once      found3, and L holds >= 95% of the face (one piece). Two true faces on one surface (a
 *             cylinder in two halves) merged into one segment are both found once: same surface.
 *   swallowed not found, and L belongs mostly to another face
 * and per part: segments per true face, SPURIOUS segments (the L of no face: extra faces in a solid),
 * and the painted area on the wrong face (not its segment's majority face), split into HARMLESS
 * (every vertex within 1e-3 x diagonal of that majority face's true surface), HARMFUL, and UNKNOWN
 * (a freeform majority face).
 *
 * VARIANTS, named segmentation/surfaces. Surfaces are gaussMap.fitRegionAs (the export path's fitter)
 * with the segment's painted type, fitted on:
 *   all      every facet (today's export)
 *   ring1    facets at least 2 edge-steps from any other segment; ring2 at least 3
 *   member   facets the refiner scored >= 0.9, and ring1's rule where it scored none
 *   edge     facets whose edges all score below the boundary model's fpr2 threshold
 *   trim     every facet, then again without the 20% farthest from the first fit
 *   (a core under max(3, 10%) of its segment falls back to every facet)
 * Segmentations:
 *   base     as painted. A group the painting left untyped (its fill found no analytic surface: often a
 *            flat face meshed as two triangles) gets no surface, as in the studio's refit.
 *   retype   base, with every untyped group classified on its own facets (gaussMap.classifyRegion)
 *   reclass  base, with EVERY group classified on its own final facets (simplest type within 1%),
 *            instead of the type its fill chose from the model's first patch; the painted type stays
 *            where classifyRegion finds none. rtm = reclass, then triage and merge.
 *   exact    base, typed by paintCore.typeFinalGroup: the classifier's type only where it fits every
 *            vertex within --exact-bar x diagonal (a pure group), the painted type elsewhere, untyped
 *            groups as retype. etm = exact, then triage and merge.
 *   fexact   js/freeformMerge.js first (adjacent groups neither of which fits a primitive exactly, joined
 *            across crease-free boundaries; a joined inexact group stays untyped, for a B-spline), then
 *            exact on the rest. fetm = fexact, then triage and merge.
 *   gexact   fexact with the group-level type model (js/groupType.js, --group-typer <onnx>) deciding the
 *            type of every group of >= 3 facets where its top probability reaches the sidecar's threshold
 *            (or --typer-threshold); typeFinalGroup elsewhere. getm = gexact, then triage and merge.
 *   oracletype  base, each group typed as its majority face truly is: what perfect typing of the
 *            painted groups would give (uses the truth; a bound, not a method)
 *   truth    the true faces with their true types: the fitter's own ceiling
 * and the truth-free passes, on retype, fitted on ring1 cores:
 *   triage   a facet off its own segment's core surface (a vertex beyond tol) moves to an adjacent
 *            segment whose core surface it is on, repeatedly, so a deep spill peels back ring by ring
 *   merge    a segment whose vertices lie (95%) on an adjacent segment's core surface joins it,
 *            smaller into larger
 *   tm       triage, then merge; tmo also turns each connected run of >= 3 facets that fits neither
 *            its segment nor a neighbour into a segment of its own (a face the painting swallowed)
 *   oldmerge js/mergeRegions.js with its defaults (rejected 2026-09 by clean area)
 *   pearl    js/pearl.js with its defaults, edges scored by the painting's boundary model (rejected
 *            2026-09-24 by clean area)
 *   tol = max(2 x the core's 90th-percentile vertex residual, --tol x diagonal)
 *
 * --patches-out writes, per variant in PATCH_VARIANTS, the segment of every facet (<stem>.fid) and its
 * surface (<stem>.json, test/exportPatchFits.mjs's format) for ml/solid_rebuild.py --fid-dir.
 */

import { readFileSync, writeFileSync, readdirSync, mkdirSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { createRequire } from 'node:module';

import { preparePaintPositions, typeFinalGroup, EXACT_TYPE_BAR } from '../js/paintCore.js';
import { freeformMerge } from '../js/freeformMerge.js';
import { groupTypeFeatures, decideGroupType, N_GROUP_FEATURES, GROUP_TYPES } from '../js/groupType.js';
import { fitRegionAs, classifyRegion } from '../js/gaussMap.js';
import { mergeRegions } from '../js/mergeRegions.js';
import { pearlRefine } from '../js/pearl.js';
import { attachEdgeGraph } from '../js/edgeGraph.js';
import { annotateEdges } from '../js/edgeWall.js';

const ort = createRequire(import.meta.url)('onnxruntime-node');
const args = process.argv.slice(2);
const flag = (name, dflt) => { const i = args.indexOf(name); if (i < 0) return dflt; const v = args[i + 1]; args.splice(i, 2); return v; };
const ALL_VARIANTS = ['base/all', 'retype/all', 'retype/ring1', 'retype/ring2', 'retype/member', 'retype/edge',
  'retype/trim', 'truth/all', 'truth/ring1', 'triage/ring1', 'merge/ring1', 'tm/ring1', 'tmo/ring1', 'oldmerge/ring1',
  'pearl/ring1'];
const VARIANTS = flag('--variants', ALL_VARIANTS.join(',')).split(',');
const TOL = Number(flag('--tol', '5e-4'));
const EXACT_BAR = Number(flag('--exact-bar', String(EXACT_TYPE_BAR)));
const MESHERS = flag('--meshers', null)?.split(',');
const SHARD = flag('--shard', null)?.split('/').map(Number);
const JSON_OUT = flag('--json', null);
const PATCHES_OUT = flag('--patches-out', null);
const DEBUG = flag('--debug', null);
// --ff '{"robust":true}': freeformMerge options for the fexact / fetm variants (default: its defaults)
const FF_OPTS = JSON.parse(flag('--ff', '{}'));
const TYPER_PATH = flag('--group-typer', null);
const TYPER_THR = flag('--typer-threshold', null);
const TYPER = TYPER_PATH ? {
  session: await ort.InferenceSession.create(TYPER_PATH),
  threshold: Number(TYPER_THR ?? JSON.parse(readFileSync(TYPER_PATH.replace(/\.onnx$/, '.json'), 'utf8')).threshold),
} : null;
// --faces: also keep every analytic face's verdict (kind, L's type, share, whether L is its own, dev).
const FACES = args.includes('--faces');
if (FACES) args.splice(args.indexOf('--faces'), 1);   // a stem: print every analytic face's verdict per variant
const PATCH_VARIANTS = ['base/all', 'retype/all', 'retype/ring1', 'tm/ring1', 'tmo/ring1', 'truth/all', 'reclass/all',
  'rtm/ring1', 'oracletype/all', 'exact/all', 'etm/ring1', 'fexact/all', 'fetm/ring1', 'getm/ring1'];
const [MESH_DIR, TRUTH_DIR, SEG_DIR] = args;
if (!MESH_DIR || !TRUTH_DIR || !SEG_DIR) {
  console.error('usage: node test/surfaceScore.mjs <mesh dir> <truth dir> <seg dir> [--variants ...]');
  process.exit(2);
}

const KIND = ['plane', 'cylinder', 'cone', 'sphere', 'torus'];
const FOUND3 = 1e-3, FOUND4 = 1e-4, HARMLESS = 1e-3;

// ── surfaces (gaussMap's param format; as test/fitTruth.mjs) ───────────────────────────────────
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const len = (v) => Math.hypot(v[0], v[1], v[2]);
const unit = (v) => { const l = len(v); return l > 1e-300 ? [v[0] / l, v[1] / l, v[2] / l] : [0, 0, 1]; };
const anyPerp = (a) => unit(Math.abs(a[0]) < 0.9 ? [0, -a[2], a[1]] : [-a[2], 0, a[0]]);

/** Signed distance and unit normal of point q to a surface. Cones are double cones. */
function evalSurface(type, p, q) {
  if (type === 'plane') {
    const n = unit(p.normal);
    return { d: dot(sub(q, p.origin), n), n };
  }
  if (type === 'sphere') {
    const v = sub(q, p.center); const r = len(v);
    return { d: r - p.radius, n: r > 1e-300 ? [v[0] / r, v[1] / r, v[2] / r] : [0, 0, 1] };
  }
  const a = unit(p.axis);
  if (type === 'cylinder') {
    const v = sub(q, p.axisPoint); const z = dot(v, a);
    const rad = [v[0] - z * a[0], v[1] - z * a[1], v[2] - z * a[2]]; const r = len(rad);
    return { d: r - p.radius, n: r > 1e-300 ? [rad[0] / r, rad[1] / r, rad[2] / r] : anyPerp(a) };
  }
  if (type === 'cone') {
    const v = sub(q, p.apex); const z = dot(v, a);
    const rad = [v[0] - z * a[0], v[1] - z * a[1], v[2] - z * a[2]]; const r = len(rad);
    const u = r > 1e-300 ? [rad[0] / r, rad[1] / r, rad[2] / r] : anyPerp(a);
    const ca = Math.cos(p.halfAngle), sa = Math.sin(p.halfAngle);
    const s = z >= 0 ? 1 : -1;
    const along = r * sa + Math.abs(z) * ca;
    if (along < 0) return { d: Math.hypot(r, z), n: unit(v) };
    return { d: r * ca - Math.abs(z) * sa,
             n: unit([u[0] * ca - s * a[0] * sa, u[1] * ca - s * a[1] * sa, u[2] * ca - s * a[2] * sa]) };
  }
  if (type === 'torus') {
    const v = sub(q, p.center); const z = dot(v, a);
    const rad = [v[0] - z * a[0], v[1] - z * a[1], v[2] - z * a[2]]; const r = len(rad);
    const u = r > 1e-300 ? [rad[0] / r, rad[1] / r, rad[2] / r] : anyPerp(a);
    const w = sub(v, [p.majorRadius * u[0], p.majorRadius * u[1], p.majorRadius * u[2]]);
    const wl = len(w);
    return { d: wl - p.minorRadius, n: wl > 1e-300 ? [w[0] / wl, w[1] / wl, w[2] / wl] : u };
  }
  throw new Error(type);
}

/** fit_truth row -> gaussMap params. `c` orients a cone onto the face's nappe. */
function truthParams(kind, s, c) {
  const P = [s[0], s[1], s[2]], A = [s[3], s[4], s[5]];
  if (kind === 'plane') return { origin: P, normal: unit(A) };
  if (kind === 'cylinder') return { axisPoint: P, axis: unit(A), radius: s[6] };
  if (kind === 'sphere') return { center: P, radius: s[6] };
  if (kind === 'torus') return { center: P, axis: unit(A), majorRadius: s[6], minorRadius: s[7] };
  let a = unit(A);
  if (dot(sub(c, P), a) < 0) a = [-a[0], -a[1], -a[2]];
  return { apex: P, axis: a, halfAngle: s[6] };
}

// ── one part ─────────────────────────────────────────────────────────────────────────────────────
const copy = (b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.length);
const rd = (path, T) => new T(copy(readFileSync(path)));

const edgeModels = new Map();
async function edgeModel(dir) {
  if (!edgeModels.has(dir)) {
    const meta = JSON.parse(readFileSync(join(dir, 'edge.json'), 'utf8'));
    const s = await ort.InferenceSession.create(join(dir, 'edge.onnx'), { intraOpNumThreads: 2, interOpNumThreads: 1 });
    edgeModels.set(dir, { meta, thr: 1 / (1 + Math.exp(-meta.thresholds['fpr0.02'].logit)),
      run: async (f, n) => (await s.run({ features: new ort.Tensor('float32', f, [n, f.length / n]) })).logit.data });
  }
  return edgeModels.get(dir);
}

async function scorePart(stem, manifestEntry, mesher) {
  const meta = JSON.parse(readFileSync(join(SEG_DIR, `${stem}.json`), 'utf8'));
  const P = Float64Array.from(rd(join(MESH_DIR, `${stem}.pos`), Float32Array));
  const F = rd(join(MESH_DIR, `${stem}.fid`), Int32Array);
  const truth = rd(join(TRUTH_DIR, `${meta.part}.srf`), Float64Array);
  const seg0 = rd(join(SEG_DIR, `${stem}.seg`), Int32Array);
  const member = rd(join(SEG_DIR, `${stem}.member`), Float32Array);
  const pedge = rd(join(SEG_DIR, `${stem}.pedge`), Float32Array);
  const mesh = preparePaintPositions(P);
  const { facets, adjacency } = mesh;
  const n = facets.count;
  if (F.length !== n || seg0.length !== n) throw new Error(`${stem}: facet counts differ`);
  const edgeDir = meta.edge ? meta.edge.split(' ')[0] : null;
  const edge = edgeDir ? await edgeModel(edgeDir) : null;

  let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < P.length; i += 3) for (let k = 0; k < 3; k++) {
    if (P[i + k] < lo[k]) lo[k] = P[i + k];
    if (P[i + k] > hi[k]) hi[k] = P[i + k];
  }
  const diag = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
  const vtx = (t, k) => [P[t * 9 + k * 3], P[t * 9 + k * 3 + 1], P[t * 9 + k * 3 + 2]];
  /** Largest |distance| of facet t's vertices from a surface. */
  const vdist = (t, type, params) => {
    let m = 0;
    for (let k = 0; k < 3; k++) m = Math.max(m, Math.abs(evalSurface(type, params, vtx(t, k)).d));
    return m;
  };
  const vdistMax = (tris, type, params) => tris.reduce((m, t) => Math.max(m, vdist(t, type, params)), 0);

  // True faces: facets, area, kind, parameters, and samples (face vertices projected onto the truth).
  const faces = new Map();
  for (let t = 0; t < n; t++) {
    if (facets.degenerate[t]) continue;
    const f = faces.get(F[t]) ?? { tris: [], area: 0 };
    f.tris.push(t); f.area += facets.areas[t];
    faces.set(F[t], f);
  }
  for (const [id, f] of faces) {
    const s = id >= 0 && id * 12 + 12 <= truth.length ? truth.subarray(id * 12, id * 12 + 12) : null;
    f.kind = s && s[0] === s[1] && s[0] <= 4 ? KIND[s[0]] : null;
    if (!f.kind) continue;
    const seen = new Set(), verts = [];
    let cx = 0, cy = 0, cz = 0;
    for (const t of f.tris) for (let k = 0; k < 3; k++) {
      const q = vtx(t, k), key = `${q[0]}_${q[1]}_${q[2]}`;
      if (seen.has(key)) continue;
      seen.add(key); verts.push(q);
      cx += q[0]; cy += q[1]; cz += q[2];
    }
    f.params = truthParams(f.kind, s.subarray(2), [cx / verts.length, cy / verts.length, cz / verts.length]);
    f.samples = verts.map((q) => { const e = evalSurface(f.kind, f.params, q); return [q[0] - e.d * e.n[0], q[1] - e.d * e.n[1], q[2] - e.d * e.n[2]]; });
  }
  const analytic = [...faces].filter(([, f]) => f.kind);
  const totalArea = [...faces.values()].reduce((s, f) => s + f.area, 0);

  // ── segment bookkeeping ──
  const segmentsOf = (seg) => {
    const S = new Map();
    for (let t = 0; t < n; t++) {
      if (seg[t] < 0 || facets.degenerate[t]) continue;
      const s = S.get(seg[t]) ?? { tris: [], area: 0, faceArea: new Map() };
      s.tris.push(t); s.area += facets.areas[t];
      s.faceArea.set(F[t], (s.faceArea.get(F[t]) ?? 0) + facets.areas[t]);
      S.set(seg[t], s);
    }
    for (const s of S.values()) {
      let best = null, a = -1;
      for (const [f, fa] of s.faceArea) if (fa > a) { a = fa; best = f; }
      s.maj = best;
    }
    return S;
  };
  /** Edge-steps from each painted facet to the nearest facet of another segment (1 = on the boundary). */
  const ringDepth = (seg) => {
    const depth = new Int32Array(n).fill(0);
    let frontier = [];
    for (let t = 0; t < n; t++) {
      if (seg[t] < 0) continue;
      for (const e of adjacency.get(t) ?? []) if (seg[e.neighbor] !== seg[t]) { depth[t] = 1; frontier.push(t); break; }
    }
    let d = 1;
    while (frontier.length) {
      const next = [];
      for (const t of frontier) for (const e of adjacency.get(t) ?? []) {
        const u = e.neighbor;
        if (seg[u] === seg[t] && depth[u] === 0) { depth[u] = d + 1; next.push(u); }
      }
      frontier = next; d++;
    }
    for (let t = 0; t < n; t++) if (seg[t] >= 0 && depth[t] === 0) depth[t] = 1e9;   // a segment with no boundary
    return depth;
  };

  // ── fitting ──
  const coreOf = (how, tris, depth) => {
    let c;
    if (how === 'all' || how === 'trim') return tris;
    if (how === 'ring1') c = tris.filter((t) => depth[t] >= 2);
    else if (how === 'ring2') c = tris.filter((t) => depth[t] >= 3);
    else if (how === 'member') c = tris.filter((t) => (Number.isNaN(member[t]) ? depth[t] >= 2 : member[t] >= 0.9));
    else if (how === 'edge') c = tris.filter((t) => !(pedge[t] >= (edge ? edge.thr : Infinity)));
    else throw new Error(how);
    return c.length >= Math.max(3, 0.1 * tris.length) ? c : tris;
  };
  /** 90th percentile |distance| of up to 2000 of the region's vertices from its fit: how far its own
   *  facets stray, the scale below which another facet is on the same surface. */
  const residualSpread = (tris, type, params) => {
    const ds = [];
    const step = Math.max(1, Math.floor(tris.length / 667));
    for (let i = 0; i < tris.length; i += step) for (let k = 0; k < 3; k++) ds.push(Math.abs(evalSurface(type, params, vtx(tris[i], k)).d));
    ds.sort((a, b) => a - b);
    return ds.length ? ds[Math.floor(0.9 * (ds.length - 1))] : 0;
  };
  const fitSegment = (type, tris, how, depth) => {
    if (!type || !KIND.includes(type)) return null;
    const core = coreOf(how, tris, depth);
    let fit = fitRegionAs(type, new Set(core), facets);
    if (!fit.params || fit.degenerate) return null;
    if (how === 'trim' && tris.length >= 5) {
      const ranked = tris.map((t) => [vdist(t, type, fit.params), t]).sort((a, b) => a[0] - b[0]);
      const kept = ranked.slice(0, Math.max(3, Math.ceil(0.8 * ranked.length))).map((x) => x[1]);
      const again = fitRegionAs(type, new Set(kept), facets);
      if (again.params && !again.degenerate) fit = again;
    }
    const tol = Math.max(2 * residualSpread(core, type, fit.params), TOL * diag);
    return { type, params: fit.params, tol };
  };
  const fitAll = (seg, types, how) => {
    const S = segmentsOf(seg);
    const depth = ringDepth(seg);
    const surf = new Map();
    for (const [id, s] of S) surf.set(id, fitSegment(types[id], s.tris, how, depth));
    return surf;
  };

  // ── the truth-free passes ──
  function triage(seg, types, surf, splitOrphans) {
    const out = Int32Array.from(seg);
    const newTypes = [...types];
    let moved = 0;
    for (let pass = 0; pass < 60; pass++) {
      let changed = 0;
      for (let t = 0; t < n; t++) {
        const s = out[t];
        if (s < 0) continue;
        const own = surf.get(s);
        if (own && vdist(t, own.type, own.params) <= own.tol) continue;
        let best = -1, bestD = Infinity;
        for (const e of adjacency.get(t) ?? []) {
          const c = out[e.neighbor];
          if (c < 0 || c === s) continue;
          const sc = surf.get(c);
          if (!sc) continue;
          const d = vdist(t, sc.type, sc.params);
          if (d <= sc.tol && d < bestD) { best = c; bestD = d; }
        }
        if (best >= 0) { out[t] = best; changed++; }
      }
      moved += changed;
      if (!changed) break;
    }
    // Leftovers: facets on neither their own nor any neighbour's surface. A connected run of them
    // inside one segment is a face the painting swallowed; it becomes a segment of its own.
    const orphan = new Uint8Array(n);
    for (let t = 0; t < n; t++) {
      const s = out[t];
      if (s < 0) continue;
      const own = surf.get(s);
      if (own && vdist(t, own.type, own.params) > own.tol) orphan[t] = 1;
    }
    let orphans = 0, newSegs = 0, next = 0;
    for (let t = 0; t < n; t++) if (out[t] >= next) next = out[t] + 1;
    if (!splitOrphans) {
      for (let t = 0; t < n; t++) orphans += orphan[t];
      return { seg: out, types: newTypes, moved, orphans, newSegs };
    }
    const seen = new Uint8Array(n);
    for (let t = 0; t < n; t++) {
      if (!orphan[t] || seen[t]) continue;
      const comp = [t];
      seen[t] = 1;
      for (let i = 0; i < comp.length; i++) for (const e of adjacency.get(comp[i]) ?? []) {
        const u = e.neighbor;
        if (orphan[u] && !seen[u] && out[u] === out[t]) { seen[u] = 1; comp.push(u); }
      }
      orphans += comp.length;
      if (comp.length < 3) continue;
      const cls = classifyRegion(new Set(comp), facets);
      const id = next++;
      newTypes[id] = cls.type === 'nurbs' ? null : cls.type;
      for (const u of comp) out[u] = id;
      newSegs++;
    }
    return { seg: out, types: newTypes, moved, orphans, newSegs };
  }

  function mergeSame(seg, types, surf) {
    const out = Int32Array.from(seg);
    let merges = 0;
    for (let round = 0; round < 30; round++) {
      const S = segmentsOf(out);
      // Adjacent segment pairs.
      const pairs = new Set();
      for (let t = 0; t < n; t++) {
        if (out[t] < 0) continue;
        for (const e of adjacency.get(t) ?? []) {
          const u = e.neighbor;
          if (out[u] < 0 || out[u] === out[t]) continue;
          const a = Math.min(out[t], out[u]), b = Math.max(out[t], out[u]);
          pairs.add(a * 1e6 + b);
        }
      }
      // Smaller onto larger, smallest first; each segment moves or receives once per round.
      const cand = [...pairs].map((k) => [Math.floor(k / 1e6), k % 1e6])
        .map(([a, b]) => (S.get(a).area <= S.get(b).area ? [a, b] : [b, a]))
        .sort((x, y) => S.get(x[0]).area - S.get(y[0]).area);
      const touched = new Set();
      let did = 0;
      for (const [x, y] of cand) {
        if (touched.has(x) || touched.has(y)) continue;
        const sy = surf.get(y);
        if (!sy) continue;
        let on = 0, all = 0;
        for (const t of S.get(x).tris) for (let k = 0; k < 3; k++) {
          all++;
          if (Math.abs(evalSurface(sy.type, sy.params, vtx(t, k)).d) <= sy.tol) on++;
        }
        if (on < 0.95 * all) continue;
        for (const t of S.get(x).tris) out[t] = y;
        touched.add(x); touched.add(y);
        did++;
      }
      merges += did;
      if (!did) break;
    }
    return { seg: out, types, merges };
  }

  // ── the scorecard ──
  function scorecard(seg, types, surf) {
    const S = segmentsOf(seg);
    // L per face: the segment holding most of its area.
    const share = new Map();
    for (const [id, s] of S) for (const [f, a] of s.faceArea) {
      const cur = share.get(f);
      if (!cur || a > cur.a) share.set(f, { seg: id, a });
    }
    let found3 = 0, found4 = 0, geo3 = 0, once = 0, swallowed = 0, split = 0;
    let found3A = 0, onceA = 0, analyticA = 0;
    const byKind = {};
    const faceRows = FACES ? [] : null;
    for (const [id, f] of analytic) {
      analyticA += f.area;
      const sh = share.get(id);
      const L = sh ? sh.seg : -1;
      const whole = sh ? sh.a >= 0.95 * f.area : false;
      const sf = L >= 0 ? surf.get(L) : null;
      let dev = Infinity;
      if (sf) { dev = 0; for (const q of f.samples) dev = Math.max(dev, Math.abs(evalSurface(sf.type, sf.params, q).d)); dev /= diag; }
      const typed = sf && sf.type === f.kind;
      const ok3 = typed && dev < FOUND3;
      if (DEBUG === stem) {
        console.log(`    face ${id} ${f.kind} tris ${f.tris.length} L ${L}:${L >= 0 ? types[L] : '-'} share ${sh ? (sh.a / f.area).toFixed(2) : 0}` +
          ` maj(L) ${L >= 0 ? S.get(L).maj : '-'} fit ${sf ? sf.type : '-'} dev ${dev.toExponential(1)}`);
      }
      if (faceRows) faceRows.push({ face: id, kind: f.kind, tris: f.tris.length, share: sh ? sh.a / f.area : 0,
                                    L: L >= 0 ? (sf ? sf.type : null) : undefined, own: L >= 0 && S.get(L).maj === id, dev });
      const k = byKind[f.kind] ??= { faces: 0, found3: 0, once: 0, swallowed: 0 };
      k.faces++;
      if (ok3) { found3++; found3A += f.area; k.found3++; }
      if (typed && dev < FOUND4) found4++;
      if (dev < FOUND3) geo3++;
      if (ok3 && whole) { once++; onceA += f.area; k.once++; }
      if (!ok3 && L >= 0 && S.get(L).maj !== id) { swallowed++; k.swallowed++; }
      if (!whole) split++;
    }
    const isL = new Set([...share.values()].map((x) => x.seg));
    const spurious = [...S.keys()].filter((id) => !isL.has(id)).length;
    // Painted area on the wrong face, harmless or not.
    let wrongA = 0, harmlessA = 0, harmfulA = 0, unknownA = 0, unpaintedA = 0;
    for (let t = 0; t < n; t++) {
      if (facets.degenerate[t]) continue;
      const a = facets.areas[t];
      if (seg[t] < 0) { unpaintedA += a; continue; }
      const m = S.get(seg[t]).maj;
      if (F[t] === m) continue;
      wrongA += a;
      const mf = faces.get(m);
      if (!mf.kind) { unknownA += a; continue; }
      if (vdist(t, mf.kind, mf.params) <= HARMLESS * diag) harmlessA += a; else harmfulA += a;
    }
    // Clean, as autoSeed scores it: one segment holds >= 95% of the face and is >= 95% its type.
    let clean = 0, cleanA = 0;
    const T = rd(join(MESH_DIR, `${stem}.typ`), Uint8Array);
    const segType = new Map();
    for (const [id, s] of S) {
      const m = new Map();
      for (const t of s.tris) m.set(T[t], (m.get(T[t]) ?? 0) + 1);
      segType.set(id, m);
    }
    for (const [id, f] of faces) {
      const overlap = new Map();
      for (const t of f.tris) if (seg[t] >= 0) overlap.set(seg[t], (overlap.get(seg[t]) ?? 0) + 1);
      let best = -1, bn = 0;
      for (const [s, c] of overlap) if (c > bn) { bn = c; best = s; }
      if (best < 0) continue;
      const purity = (segType.get(best).get(T[f.tris[0]]) ?? 0) / S.get(best).tris.length;
      if (bn / f.tris.length >= 0.95 && purity >= 0.95) { clean++; cleanA += f.area; }
    }
    const nA = analytic.length || 1;
    return {
      analytic: analytic.length, faces: faces.size, segments: S.size, spurious,
      found3: found3 / nA, found4: found4 / nA, geo3: geo3 / nA, once: once / nA, swallowed: swallowed / nA, split: split / nA,
      found3Area: found3A / (analyticA || 1), onceArea: onceA / (analyticA || 1),
      segPerFace: S.size / faces.size, spuriousPerFace: spurious / faces.size, clean: clean / faces.size, cleanArea: cleanA / totalArea,
      wrongArea: wrongA / totalArea, harmlessArea: harmlessA / totalArea, harmfulArea: harmfulA / totalArea,
      unknownArea: unknownA / totalArea, unpaintedArea: unpaintedA / totalArea, byKind, ...(faceRows ? { faceRows } : {}),
    };
  }

  // ── the variants ──
  const rows = [];
  const types0 = meta.types;
  const truthSeg = Int32Array.from(F);
  const truthTypes = [];
  for (const [id, f] of faces) truthTypes[id] = f.kind;
  const cache = new Map();
  const segmentation = async (name) => {
    if (cache.has(name)) return cache.get(name);
    let r;
    if (name === 'base') r = { seg: seg0, types: types0 };
    else if (name === 'truth') r = { seg: truthSeg, types: truthTypes };
    else if (name === 'retype') {
      const types = [...types0];
      let retyped = 0;
      for (const [id, sg] of segmentsOf(seg0)) {
        if (types[id]) continue;
        const cls = classifyRegion(new Set(sg.tris), facets);
        if (KIND.includes(cls.type)) { types[id] = cls.type; retyped++; continue; }
        // classifyRegion wants 3 facets; a flat rectangle is often 2. A plane through them all.
        if (sg.tris.length < 3) {
          const plane = fitRegionAs('plane', new Set(sg.tris), facets);
          if (plane.params && vdistMax(sg.tris, 'plane', plane.params) <= TOL * diag) { types[id] = 'plane'; retyped++; }
        }
      }
      r = { seg: seg0, types, info: { retyped } };
    } else if (name === 'reclass' || name === 'oracletype') {
      const types = [...types0];
      let changed = 0;
      for (const [id, sg] of segmentsOf(seg0)) {
        let t = null;
        if (name === 'oracletype') t = faces.get(sg.maj).kind;
        else {
          const cls = classifyRegion(new Set(sg.tris), facets);
          if (KIND.includes(cls.type)) t = cls.type;
          else if (!types[id] && sg.tris.length < 3) {
            const plane = fitRegionAs('plane', new Set(sg.tris), facets);
            if (plane.params && vdistMax(sg.tris, 'plane', plane.params) <= TOL * diag) t = 'plane';
          }
          t ??= types[id];
        }
        if (t !== types[id]) changed++;
        types[id] = t;
      }
      r = { seg: seg0, types, info: { retyped: changed } };
    } else if (name === 'exact') {
      const types = [...types0];
      let changed = 0, exact = 0;
      for (const [id, sg] of segmentsOf(seg0)) {
        const r = typeFinalGroup(sg.tris, facets, types[id], { bar: EXACT_BAR, looseTol: TOL });
        if (r.changed) { changed++; exact += r.exact; }
        types[id] = r.type;
      }
      r = { seg: seg0, types, info: { retyped: changed, exactRetyped: exact } };
    } else if (name === 'fexact') {
      const seg = Int32Array.from(seg0);
      const { freeform, merged } = freeformMerge(seg, facets, adjacency, FF_OPTS);
      const types = [...types0];
      let changed = 0;
      for (const [id, sg] of segmentsOf(seg)) {
        if (freeform.has(id)) { types[id] = null; continue; }
        const r = typeFinalGroup(sg.tris, facets, types[id], { bar: EXACT_BAR, looseTol: TOL });
        if (r.changed) changed++;
        types[id] = r.type;
      }
      r = { seg, types, info: { retyped: changed, merged, freeform: freeform.size } };
    } else if (name === 'gexact') {
      if (!TYPER) throw new Error('gexact needs --group-typer <onnx>');
      const seg = Int32Array.from(seg0);
      const { freeform, merged } = freeformMerge(seg, facets, adjacency, FF_OPTS);
      const types = [...types0];
      let changed = 0, byModel = 0;
      const pending = [];
      for (const [id, sg] of segmentsOf(seg)) {
        if (freeform.has(id)) { types[id] = null; continue; }
        const f = groupTypeFeatures(new Set(sg.tris), facets, adjacency, types[id] ?? null, { bar: EXACT_BAR, looseTol: TOL });
        if (!f) {
          const r = typeFinalGroup(sg.tris, facets, types[id], { bar: EXACT_BAR, looseTol: TOL });
          if (r.changed) changed++;
          types[id] = r.type;
          continue;
        }
        pending.push({ id, f });
      }
      if (pending.length) {
        const X = new Float32Array(pending.length * N_GROUP_FEATURES);
        pending.forEach((q, i) => X.set(q.f.features, i * N_GROUP_FEATURES));
        const out = await TYPER.session.run({ features: new ort.Tensor('float32', X, [pending.length, N_GROUP_FEATURES]) });
        const probs = out.probs.data, k = GROUP_TYPES.length;
        pending.forEach((q, i) => {
          const d = decideGroupType(probs.subarray(i * k, i * k + k), q.f.rule, TYPER.threshold);
          if (d.type !== (types[q.id] ?? null)) changed++;
          if (d.by === 'model' && d.type !== q.f.rule) byModel++;
          types[q.id] = d.type;
        });
      }
      r = { seg, types, info: { retyped: changed, byModel, merged, freeform: freeform.size } };
    } else if (name === 'rtm' || name === 'etm' || name === 'fetm' || name === 'getm') {
      const base = await segmentation({ rtm: 'reclass', etm: 'exact', fetm: 'fexact', getm: 'gexact' }[name]);
      const surf = fitAll(base.seg, base.types, 'ring1');
      const tr = triage(base.seg, base.types, surf, false);
      const mg = mergeSame(tr.seg, tr.types, fitAll(tr.seg, tr.types, 'ring1'));
      r = { seg: mg.seg, types: mg.types, info: { ...base.info, moved: tr.moved, orphans: tr.orphans, merges: mg.merges } };
    } else if (['triage', 'merge', 'tm', 'tmo'].includes(name)) {
      const base = await segmentation('retype');
      const surf = fitAll(base.seg, base.types, 'ring1');
      let cur = base, info = { ...base.info };
      if (name !== 'merge') {
        const tr = triage(cur.seg, cur.types, surf, name === 'tmo');
        cur = { seg: tr.seg, types: tr.types };
        Object.assign(info, { moved: tr.moved, orphans: tr.orphans, newSegs: tr.newSegs });
      }
      if (name !== 'triage') {
        const surf2 = name === 'merge' ? surf : fitAll(cur.seg, cur.types, 'ring1');
        const mg = mergeSame(cur.seg, cur.types, surf2);
        cur = { seg: mg.seg, types: mg.types };
        info.merges = mg.merges;
      }
      r = { ...cur, info };
    } else if (name === 'oldmerge') {
      const base = await segmentation('retype');
      const types = [...base.types];
      const m = mergeRegions(Int32Array.from(base.seg), facets, adjacency, { onMerge: (keep, gone, fit) => { types[keep] = fit.type; } });
      r = { seg: m.seg, types, info: { merges: m.merges } };
    } else if (name === 'pearl') {
      const base = await segmentation('retype');
      if (edge) await annotateEdges(facets, adjacency, edge.run, edge.meta.columns);
      const graph = attachEdgeGraph(P, facets, adjacency);
      const pr = pearlRefine(facets, adjacency, graph, Int32Array.from(base.seg), [...base.types], {});
      r = { seg: pr.seg, types: base.types, info: { changed: pr.changed } };
    } else throw new Error(name);
    cache.set(name, r);
    return r;
  };
  for (const v of VARIANTS) {
    const [segName, how] = v.split('/');
    const t0 = performance.now();
    const s = await segmentation(segName);
    const surf = fitAll(s.seg, s.types, how);
    if (DEBUG === stem) console.log(`  ${stem} ${v}`);
    const card = scorecard(s.seg, s.types, surf);
    rows.push({ part: meta.part, mesher, stem, variant: v, ms: performance.now() - t0, ...(s.info ?? {}), ...card });
    if (PATCHES_OUT && PATCH_VARIANTS.includes(v)) {
      const dir = join(PATCHES_OUT, v.replace('/', '-'));
      mkdirSync(dir, { recursive: true });
      // The rebuild needs every facet in a patch: an unpainted one joins the neighbouring segment it
      // touches most.
      const lab = Int32Array.from(s.seg);
      for (let pass = 0; pass < 50; pass++) {
        let left = 0;
        for (let t = 0; t < n; t++) {
          if (lab[t] >= 0) continue;
          const votes = new Map();
          for (const e of adjacency.get(t) ?? []) if (lab[e.neighbor] >= 0) votes.set(lab[e.neighbor], (votes.get(lab[e.neighbor]) ?? 0) + 1);
          let best = -1, bv = 0;
          for (const [k, c] of votes) if (c > bv) { bv = c; best = k; }
          if (best >= 0) lab[t] = best; else left++;
        }
        if (!left) break;
      }
      writeFileSync(join(dir, `${stem}.fid`), new Uint8Array(lab.buffer, lab.byteOffset, lab.byteLength));
      const facesOut = [];
      for (const id of [...new Set(lab)].filter((x) => x >= 0).sort((a, b) => a - b)) {
        const sf = surf.get(id);
        facesOut.push({ id, type: sf ? sf.type : 'nurbs', params: sf ? sf.params : null });
      }
      writeFileSync(join(dir, `${stem}.json`), JSON.stringify({ part: meta.part, mesher, stem, faces: facesOut }));
    }
  }
  return rows;
}

// ── the run ──────────────────────────────────────────────────────────────────────────────────────
const manifest = JSON.parse(readFileSync(join(MESH_DIR, 'manifest.json'), 'utf8'));
const byStem = new Map();
for (const entry of manifest.parts) for (const [mesher, v] of Object.entries(entry.variants)) byStem.set(v.stem, { entry, mesher });
const stems = readdirSync(SEG_DIR).filter((f) => f.endsWith('.seg')).map((f) => f.slice(0, -4))
  .filter((s) => byStem.has(s) && (!MESHERS || MESHERS.includes(byStem.get(s).mesher)))
  .filter((s) => existsSync(join(TRUTH_DIR, `${byStem.get(s).entry.part}.srf`)))
  .sort().filter((_, i) => !SHARD || i % SHARD[1] === SHARD[0]);
const rows = [];
const t0 = performance.now();
for (const [i, stem] of stems.entries()) {
  const { entry, mesher } = byStem.get(stem);
  try {
    rows.push(...await scorePart(stem, entry, mesher));
  } catch (e) {
    console.log(`  ${stem}: ${e.message}`);
  }
  if (i % 10 === 9) {
    console.log(`  ${i + 1}/${stems.length} parts, ${((performance.now() - t0) / 1000).toFixed(0)} s`);
    if (JSON_OUT) writeFileSync(JSON_OUT, JSON.stringify(rows));
  }
}
if (JSON_OUT) writeFileSync(JSON_OUT, JSON.stringify(rows));

const mean = (a) => a.reduce((s, v) => s + v, 0) / (a.length || 1);
const pc = (x) => (100 * x).toFixed(1).padStart(5);
console.log(`\n${SEG_DIR}: ${stems.length} parts, tol ${TOL}, ${((performance.now() - t0) / 1000).toFixed(0)} s`);
console.log('variant            found3 found4   geo3   once swallow split | found3A onceA | seg/face spur/face | clean cleanA | wrong = harmless + harmful + unknown (area %)');
for (const mesher of [...new Set(rows.map((r) => r.mesher))]) {
  console.log(mesher);
  for (const v of VARIANTS) {
    const rs = rows.filter((r) => r.mesher === mesher && r.variant === v);
    if (!rs.length) continue;
    const m = (k) => mean(rs.map((r) => r[k]));
    console.log(`  ${v.padEnd(16)} ${pc(m('found3'))}  ${pc(m('found4'))}  ${pc(m('geo3'))}  ${pc(m('once'))}  ${pc(m('swallowed'))} ${pc(m('split'))} |` +
      `  ${pc(m('found3Area'))}  ${pc(m('onceArea'))} | ${m('segPerFace').toFixed(2).padStart(8)} ${m('spuriousPerFace').toFixed(2).padStart(9)} | ${pc(m('clean'))} ${pc(m('cleanArea'))} |` +
      `  ${pc(m('wrongArea'))} = ${pc(m('harmlessArea'))} + ${pc(m('harmfulArea'))} + ${pc(m('unknownArea'))}`);
  }
}
