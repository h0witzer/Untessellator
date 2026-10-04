/**
 * mergeScreen.mjs — what actually separates a merge worth making from one that ruins a face.
 *
 * Usage:  node test/mergeScreen.mjs <dir> <model.onnx> [--parts N] [--meshers native,gmsh]
 *                                   [--out pairs.tsv]
 *         <dir> from ml/export_fill_eval.py.
 *
 * WHY THIS EXISTS. js/mergeRegions.js joins two neighbouring regions when one surface still
 * explains their union. Three hand-picked residual thresholds in a row dissolved parts into blobs,
 * so this dumps the DISTRIBUTION instead of assuming it: every touching pair of painted regions,
 * the geometry of both pieces and of their union, and the truth label — whether the two pieces are
 * parts of the SAME B-rep face (a merge that should happen) or of different ones (a merge that
 * must not).
 *
 * THE FIRST RUN KILLED RESIDUAL-ONLY MERGING. Over 658 touching pairs on 5 native parts, the 1st
 * percentile of DIFFERENT-face pairs had a union residual of exactly zero, and at a budget of 1%
 * bad merges no residual rule kept more than 0-50% of the good merges. Different faces routinely
 * fit one surface EXACTLY — two coplanar faces, two faces on one cylinder — so "does the union
 * fit" carries almost no signal about whether the union is one face.
 *
 * What should carry signal is whether the two surfaces are the SAME surface: their axes aligned,
 * their radii equal, their curvature compatible across the edge they share. That is what this
 * version measures, alongside the residuals, so the rule can be read off the data.
 *
 * A pair is only labelled `same` when both pieces are also PURE (>= 90% of each piece lies in that
 * one face). Two spilled regions that happen to straddle the same face are not a merge anyone
 * wants; they are already broken, and joining them cannot fix them.
 */

import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { createRequire } from 'node:module';

import { preparePaintPositions, paintWith } from '../js/paintCore.js';
import { coherentScores, seedOrder, autoPaint } from '../js/autoPaint.js';
import { classifyRegion } from '../js/gaussMap.js';
import { surfaceEval } from '../js/regionGrow.js';

const ort = createRequire(import.meta.url)('onnxruntime-node');

const args = process.argv.slice(2);
function flag(name, dflt) {
  const i = args.indexOf(name);
  if (i < 0) return dflt;
  const v = args[i + 1];
  args.splice(i, 2);
  return v;
}
const PART_LIMIT = Number(flag('--parts', '10'));
const MESHERS = flag('--meshers', null)?.split(',');
const OUT = flag('--out', null);
const [DIR, MODEL] = args;
if (!DIR || !MODEL) {
  console.error('usage: node test/mergeScreen.mjs <dir> <model.onnx> [--parts N] [--out pairs.tsv]');
  process.exit(2);
}

const session = await ort.InferenceSession.create(MODEL, { intraOpNumThreads: 2, interOpNumThreads: 1 });
let extended = false;
try { extended = JSON.parse(readFileSync(MODEL.replace(/\.onnx$/, '.json'), 'utf8')).features === 'v6'; } catch { /* base model */ }
const runner = async (features, n) => {
  const out = await session.run({
    features: new ort.Tensor('float32', features, [1, n, features.length / n]),
    mask: new ort.Tensor('float32', new Float32Array(n).fill(1), [1, n]),
  });
  return { sameLogit: out.same_logit.data, typeLogit: Array.from(out.type_logit.data), surface: out.surface ? out.surface.data : null };
};

const copy = (b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.length);
const load = (stem) => ({
  P: Float64Array.from(new Float32Array(copy(readFileSync(join(DIR, `${stem}.pos`))))),
  F: new Int32Array(copy(readFileSync(join(DIR, `${stem}.fid`)))),
  T: new Uint8Array(copy(readFileSync(join(DIR, `${stem}.typ`)))),
});

// ── surface parameters, compared ─────────────────────────────────────────────
//
// One reference direction, radius and point per surface, so pairs of any two types can be
// compared by the same few numbers. Orientation is arbitrary (PCA signs flip, axes may point
// either way), so directions are compared unsigned.

const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const dist3 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

/** Plane: its normal. Cylinder, cone, torus: the axis. Sphere: no preferred direction. */
function axisOf(fit) {
  const p = fit.params;
  if (!p) return null;
  if (fit.type === 'plane') return p.normal;
  return p.axis ?? null;
}

/** The characteristic radius, where one exists: cones are described by their half-angle instead. */
function radiusOf(fit) {
  const p = fit.params;
  if (!p) return null;
  if (fit.type === 'cylinder' || fit.type === 'sphere') return p.radius;
  if (fit.type === 'torus') return p.minorRadius;
  return null;
}

/** A point ON or IN the surface, for offset comparisons. */
function pointOf(fit) {
  const p = fit.params;
  if (!p) return null;
  return p.origin ?? p.axisPoint ?? p.center ?? p.apex ?? null;
}

/** Unsigned angle between two directions, in degrees. */
function axisAngle(a, b) {
  if (!a || !b) return NaN;
  const c = Math.min(1, Math.abs(dot3(a, b)));
  return Math.acos(c) * 180 / Math.PI;
}

/**
 * Do two WHOLE true faces lie on one and the same analytic surface?
 *
 * STEP translation splits periodic surfaces, so one cylinder routinely arrives as two half-cylinder
 * faces with different ids, one axis and one radius (this corpus's documented label noise). Calling
 * such a pair a forbidden merge would be scoring the dataset's seams rather than our geometry, so
 * they are bucketed separately.
 *
 * This compares the faces themselves — every triangle the truth assigns to each — not the painted
 * fragments, whose small local fits are exactly what cannot be trusted here.
 */
function sameSurface(fa, fb, tolD) {
  if (!fa?.params || !fb?.params || fa.type !== fb.type) return false;
  const ax = axisOf(fa);
  const bx = axisOf(fb);
  if (ax && bx && axisAngle(ax, bx) > 1) return false;
  const ra = radiusOf(fa);
  const rb = radiusOf(fb);
  if (ra != null && rb != null && Math.abs(ra - rb) > 1e-3 * Math.max(ra, rb, 1e-300)) return false;
  const pa = pointOf(fa);
  const pb = pointOf(fb);
  if (!pa || !pb) return true;
  const d = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
  // A plane is fixed by its normal and offset; a cylinder or cone by its axis LINE, so sliding the
  // reference point along the axis means nothing. Measure only the part that matters.
  if (fa.type === 'plane') return Math.abs(dot3(d, ax)) <= 0.05 * tolD;
  if (ax) {
    const along = dot3(d, ax);
    const perp = Math.hypot(d[0] - along * ax[0], d[1] - along * ax[1], d[2] - along * ax[2]);
    return perp <= 0.05 * tolD;
  }
  return dist3(pa, pb) <= 0.05 * tolD;
}

/**
 * Curvature compatibility across the edges the two regions SHARE, in the same terms as
 * regionGrow.tangentWall: the relative jump in total principal curvature |k1| + |k2|. A fillet
 * meeting the cylinder it blends into has no crease but a large curvature jump, and that is
 * exactly the merge that must not happen.
 */
function sharedCurvatureJump(facets, shared) {
  const c = facets.curvature;
  if (!c) return { median: NaN, max: NaN };
  const { k1, k2, valid } = c;
  const jumps = [];
  for (const [a, b] of shared) {
    if (!valid[a] || !valid[b]) continue;
    const ka = Math.abs(k1[a]) + Math.abs(k2[a]);
    const kb = Math.abs(k1[b]) + Math.abs(k2[b]);
    const big = Math.max(ka, kb);
    jumps.push(big > 0 ? Math.abs(ka - kb) / big : 0);
  }
  if (!jumps.length) return { median: NaN, max: NaN };
  jumps.sort((x, y) => x - y);
  return { median: jumps[jumps.length >> 1], max: jumps[jumps.length - 1] };
}

/** Mean total curvature of a region, so two regions can be compared as wholes. */
function meanCurvature(facets, list) {
  const c = facets.curvature;
  if (!c) return NaN;
  const { k1, k2, valid } = c;
  let sum = 0, n = 0;
  for (const t of list) {
    if (!valid[t]) continue;
    sum += Math.abs(k1[t]) + Math.abs(k2[t]);
    n++;
  }
  return n ? sum / n : NaN;
}

const COS15 = Math.cos(15 * Math.PI / 180);

/**
 * Does one region's surface already explain the OTHER region's triangles?
 *
 * The decisive asymmetry: painting splits a face into a large piece and slivers, and a sliver's own
 * fit is noise — over 5 native parts, fragments of ONE face disagreed on axis by 80 degrees at the
 * median and shared a fitted type only 10 times in 81. Comparing two fits therefore compares a
 * surface against nothing. Comparing the small piece's FACETS against the big piece's SURFACE asks
 * the question that has an answer, and it is the same test growth itself uses (js/paintCore.js
 * growAgainst): within tolD of the surface, normals within 15 degrees.
 */
function onSurfaceStats(facets, fit, list, tolD) {
  const ev = fit && fit.params ? surfaceEval(fit.type, fit.params) : null;
  if (!ev) return { frac: NaN, p90: NaN, max: NaN, angle: NaN };
  const C = facets.centroids;
  const N = facets.normals;
  const ds = [];
  let maxAngle = 0;
  let ok = 0;
  for (const t of list) {
    const c = [C[t * 3], C[t * 3 + 1], C[t * 3 + 2]];
    const d = Math.abs(ev.distance(c)) / tolD;
    ds.push(d);
    const n = ev.normal(c);
    const cos = Math.min(1, Math.abs(n[0] * N[t * 3] + n[1] * N[t * 3 + 1] + n[2] * N[t * 3 + 2]));
    const angle = Math.acos(cos) * 180 / Math.PI;
    if (angle > maxAngle) maxAngle = angle;
    if (d <= 1 && cos >= COS15) ok++;
  }
  ds.sort((a, b) => a - b);
  return { frac: ok / list.length, p90: ds[Math.floor(0.9 * (ds.length - 1))],
           max: ds[ds.length - 1], angle: maxAngle };
}

/** Half the median facet edge — the length scale preparePaintPositions calls tolD. */
function facetScale(facets) {
  const sizes = [];
  for (let t = 0; t < facets.count; t++) {
    if (facets.degenerate[t]) continue;
    sizes.push(Math.sqrt(Math.max(facets.areas[t], 0)));
  }
  sizes.sort((a, b) => a - b);
  return 0.5 * (sizes[sizes.length >> 1] || 1);
}

const manifest = JSON.parse(readFileSync(join(DIR, 'manifest.json'), 'utf8'));
const meshers = MESHERS ?? manifest.variants;
const parts = PART_LIMIT ? manifest.parts.slice(0, PART_LIMIT) : manifest.parts;

const COLUMNS = ['part', 'mesher', 'label', 'sizeA', 'sizeB', 'edges', 'sameType',
                 'typeA', 'typeB', 'align', 'radiusRel', 'offsetRel', 'perpRel',
                 'kJump', 'kJumpMax', 'kRel',
                 'resA', 'relA', 'resB', 'relB', 'uType', 'uRes', 'uRel', 'tolD',
                 'onFrac', 'onP90', 'onMax', 'onAngle'];
const rows = [];
const t0 = performance.now();

for (const [pi, entry] of parts.entries()) {
  for (const mesher of meshers) {
    const v = entry.variants[mesher];
    if (!v) continue;
    const { P, F } = load(v.stem);
    const mesh = preparePaintPositions(P);
    const { facets, adjacency } = mesh;
    const tolD = facetScale(facets);

    // Paint the part exactly as the studio's auto-paint would.
    const seg = new Int32Array(facets.count).fill(-1);
    let nextSeg = 0;
    await autoPaint({
      facets, order: seedOrder(await coherentScores(facets, adjacency, { chunk: 1e9 }), facets),
      owner: new Int32Array(facets.count).fill(-1), minClaim: 1, minKeep: 1,
      onDiscard: (claim) => { for (const t of claim) seg[t] = -1; },
      adjacency, absorbUpTo: 8,
      onAbsorb: (island, id) => { for (const t of island) seg[t] = id; },
      fill: (seed, taken) => paintWith(runner, seed, mesh, { method: 'model', type: 'auto', taken, extended, reportFit: false }),
      onSegment: (claim) => { const id = nextSeg++; for (const t of claim) seg[t] = id; return id; },
    });

    // Regions, their own fits, and how much of each lies in its dominant true face.
    const tris = new Map();
    for (let t = 0; t < facets.count; t++) {
      if (seg[t] < 0 || facets.degenerate[t]) continue;
      const list = tris.get(seg[t]);
      if (list) list.push(t);
      else tris.set(seg[t], [t]);
    }
    const info = new Map();
    for (const [id, list] of tris) {
      const votes = new Map();
      for (const t of list) votes.set(F[t], (votes.get(F[t]) ?? 0) + 1);
      let face = -1, n = 0;
      for (const [f, k] of votes) if (k > n) { face = f; n = k; }
      const fit = classifyRegion(list, facets);
      info.set(id, { fit, face, purity: n / list.length, size: list.length,
                     meanK: meanCurvature(facets, list) });
    }

    // Whole true faces, fitted once each, so a pair can be asked whether its two faces are really
    // one surface that STEP split.
    const faceTris = new Map();
    for (let t = 0; t < facets.count; t++) {
      if (facets.degenerate[t]) continue;
      const list = faceTris.get(F[t]);
      if (list) list.push(t);
      else faceTris.set(F[t], [t]);
    }
    const faceFitCache = new Map();
    const faceFit = (f) => {
      if (!faceFitCache.has(f)) {
        const list = faceTris.get(f);
        faceFitCache.set(f, list && list.length >= 3 ? classifyRegion(list, facets) : null);
      }
      return faceFitCache.get(f);
    };

    // Every touching pair, once, with the facet pairs along the edges they share.
    const pairs = new Map();
    for (const [a, edges] of adjacency) {
      const ra = seg[a];
      if (ra < 0) continue;
      for (const e of edges) {
        const rb = seg[e.neighbor];
        if (rb < 0 || rb === ra) continue;
        const key = ra < rb ? `${ra}|${rb}` : `${rb}|${ra}`;
        const seen = pairs.get(key);
        if (seen) seen.push([a, e.neighbor]);
        else pairs.set(key, [[a, e.neighbor]]);
      }
    }

    for (const [key, shared] of pairs) {
      const [a, b] = key.split('|').map(Number);
      const ia = info.get(a);
      const ib = info.get(b);
      if (!ia || !ib) continue;
      const fa = ia.fit;
      const fb = ib.fit;

      const align = axisAngle(axisOf(fa), axisOf(fb));
      const ra = radiusOf(fa);
      const rb = radiusOf(fb);
      const radiusRel = (ra != null && rb != null && Math.max(ra, rb) > 0)
        ? Math.abs(ra - rb) / Math.max(ra, rb) : NaN;
      const pa = pointOf(fa);
      const pb = pointOf(fb);
      const offsetRel = (pa && pb) ? dist3(pa, pb) / tolD : NaN;
      // For planes the only offset that matters is along the normal: two coplanar faces sit at
      // zero perpendicular offset however far apart their origins are.
      const na = axisOf(fa);
      const perpRel = (pa && pb && na)
        ? Math.abs(dot3([pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]], na)) / tolD : NaN;
      const k = sharedCurvatureJump(facets, shared);
      const bigK = Math.max(ia.meanK, ib.meanK);
      const kRel = bigK > 0 ? Math.abs(ia.meanK - ib.meanK) / bigK : 0;

      // The big piece proposes the surface; the small piece is tested against it.
      const bigIsA = ia.size >= ib.size;
      const bigFit = (bigIsA ? ia : ib).fit;
      const smallTris = tris.get(bigIsA ? b : a);
      const on = onSurfaceStats(facets, bigFit, smallTris, tolD);

      const union = classifyRegion([...tris.get(a), ...tris.get(b)], facets);
      const pure = ia.purity >= 0.9 && ib.purity >= 0.9;
      const label = !pure ? 'dirty'
        : ia.face === ib.face ? 'same'
        : sameSurface(faceFit(ia.face), faceFit(ib.face), tolD) ? 'split'
        : 'diff';
      rows.push([pi, mesher, label, ia.size, ib.size, shared.length, fa.type === fb.type ? 1 : 0,
                 fa.type, fb.type, align, radiusRel, offsetRel, perpRel,
                 k.median, k.max, kRel,
                 fa.residual, fa.relResidual, fb.residual, fb.relResidual,
                 union.type, union.residual, union.relResidual, tolD,
                 on.frac, on.p90, on.max, on.angle]);
    }
  }
  if (pi % 5 === 4) console.log(`  ${pi + 1} parts, ${((performance.now() - t0) / 1000).toFixed(0)} s, ${rows.length} pairs`);
}

const counts = { same: 0, split: 0, diff: 0, dirty: 0 };
for (const r of rows) counts[r[2]]++;
console.log(`\n${rows.length} touching pairs over ${parts.length} parts:`);
console.log(`  ${counts.same} same face — merge these`);
console.log(`  ${counts.split} different faces that ARE one surface (STEP's periodic split) — merging is harmless, scored separately`);
console.log(`  ${counts.diff} genuinely different surfaces — never merge these`);
console.log(`  ${counts.dirty} involving an already-impure region — no right answer`);

const scored = rows.filter((r) => r[2] !== 'dirty');
const C = Object.fromEntries(COLUMNS.map((c, i) => [c, i]));

// Same-type is a necessary condition worth stating on its own before any threshold.
const sameTyped = (label) => scored.filter((r) => r[2] === label && r[C.sameType] === 1).length;
const total = (label) => scored.filter((r) => r[2] === label).length;
console.log(`\nsame fitted type: ${sameTyped('same')}/${total('same')} of good merges,` +
  ` ${sameTyped('diff')}/${total('diff')} of bad ones` +
  ` — necessary, nowhere near sufficient`);

/**
 * Each candidate rule is a scalar where SMALLER means more compatible. Report where the good and
 * bad merges sit, and how many good merges survive a threshold that admits only 1% of the bad.
 */
const RULES = {
  'align (deg)':          (r) => r[C.align],
  'radiusRel':            (r) => r[C.radiusRel],
  'perpRel (tolD)':       (r) => r[C.perpRel],
  'offsetRel (tolD)':     (r) => r[C.offsetRel],
  'kJump (shared edges)': (r) => r[C.kJump],
  'kJumpMax':             (r) => r[C.kJumpMax],
  'kRel (region means)':  (r) => r[C.kRel],
  'uRel (union fit)':     (r) => r[C.uRel],
  'off-surface share':    (r) => 1 - r[C.onFrac],
  'onDist p90 (tolD)':    (r) => r[C.onP90],
  'onDist max (tolD)':    (r) => r[C.onMax],
  'onAngle max (deg)':    (r) => r[C.onAngle],
};
const q = (a, p) => (a.length ? a[Math.min(a.length - 1, Math.floor(p * a.length))] : NaN);
for (const [name, f] of Object.entries(RULES)) {
  const same = scored.filter((r) => r[2] === 'same').map(f).filter((x) => isFinite(x)).sort((a, b) => a - b);
  const diff = scored.filter((r) => r[2] === 'diff').map(f).filter((x) => isFinite(x)).sort((a, b) => a - b);
  if (!same.length || !diff.length) { console.log(`\n  ${name}: not defined for enough pairs`); continue; }
  const cap = q(diff, 0.01);
  const kept = same.filter((x) => x <= cap).length;
  console.log(`\n  ${name}   (defined for ${same.length} good, ${diff.length} bad)`);
  console.log(`    good  p50 ${q(same, 0.5).toExponential(2)}  p90 ${q(same, 0.9).toExponential(2)}`);
  console.log(`    bad   p1  ${q(diff, 0.01).toExponential(2)}  p10 ${q(diff, 0.1).toExponential(2)}  p50 ${q(diff, 0.5).toExponential(2)}`);
  console.log(`    at 1% bad merges: keeps ${(100 * kept / same.length).toFixed(0)}% of good (threshold ${cap.toExponential(2)})`);
}

// The rule as it would actually be written: same type, aligned axes, matching radii, no curvature
// jump across the shared edges. Swept coarsely, since the point is which combination separates.
console.log('\ncombined rule: same type AND align <= A AND kJump <= K (radiusRel <= 0.05 when defined)');
for (const A of [1, 5, 15]) {
  for (const K of [0.05, 0.2, 0.5]) {
    const ok = (r) => r[C.sameType] === 1
      && (!isFinite(r[C.align]) || r[C.align] <= A)
      && (!isFinite(r[C.kJump]) || r[C.kJump] <= K)
      && (!isFinite(r[C.radiusRel]) || r[C.radiusRel] <= 0.05);
    const good = scored.filter((r) => r[2] === 'same');
    const bad = scored.filter((r) => r[2] === 'diff');
    console.log(`  align <= ${String(A).padStart(2)} deg, kJump <= ${K}:` +
      `  keeps ${(100 * good.filter(ok).length / (good.length || 1)).toFixed(0)}% of good,` +
      `  admits ${(100 * bad.filter(ok).length / (bad.length || 1)).toFixed(0)}% of bad`);
  }
}

// The asymmetric rule, as it would be written: does the small piece lie ON the big piece's surface?
console.log('\non-surface rule: onFrac >= F AND onDist p90 <= D (tolD)');
for (const Fr of [1, 0.95, 0.9]) {
  for (const D of [0.5, 1, 2]) {
    const ok = (r) => isFinite(r[C.onFrac]) && r[C.onFrac] >= Fr && r[C.onP90] <= D;
    const good = scored.filter((r) => r[2] === 'same');
    const bad = scored.filter((r) => r[2] === 'diff');
    console.log(`  onFrac >= ${Fr}, p90 <= ${D}:` +
      `  keeps ${(100 * good.filter(ok).length / (good.length || 1)).toFixed(0)}% of good,` +
      `  admits ${(100 * bad.filter(ok).length / (bad.length || 1)).toFixed(0)}% of bad`);
  }
}

if (OUT) {
  writeFileSync(OUT, [COLUMNS.join('\t'), ...rows.map((r) => r.join('\t'))].join('\n'));
  console.log(`\nwrote ${OUT}`);
}
