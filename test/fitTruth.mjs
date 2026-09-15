/**
 * fitTruth.mjs — every analytic fitter, handed the exact facets of every true B-rep face,
 * scored against the exact STEP surface.
 *
 * Usage:  node test/fitTruth.mjs [parts] [meshers] [--dump file.json] [--types plane,cone]
 *
 * fitoracle.mjs asks this question on one fixture. This asks it on the validation parts
 * exported by ml/export_fill_eval.py (meshes) and ml/export_fit_truth.py (true surfaces),
 * so fitter changes are measured on thousands of faces under several meshers rather than
 * argued from a handful.
 *
 * WHAT MATTERS FOR EXPORT is not the parameter error in isolation but how far the fitted
 * surface strays from the true one over the face it has to bound. Two neighbouring faces
 * only sew into a solid when both surfaces pass through their shared edge within the sewing
 * tolerance. So the headline metric is DEV: the largest distance from the fitted surface of
 * any point of the true face (mesh vertices projected onto the true surface), divided by the
 * part's bounding-box diagonal. FLOOR is the same for the raw mesh vertices against the true
 * surface — what the mesher itself put there, which no fitter can be blamed for.
 *
 * --contaminate f hands each fitter the face plus f x its facet count of neighbouring foreign
 * facets, nearest rings first: the impure region any segmenter actually proposes.
 *
 * --jitter k perturbs the mesh vertices (shared vertices move together) and still scores against
 * the true surface. On exact tessellations fitting the mesh and recovering the surface coincide;
 * under noise they do not, and only the second is the goal.
 */

import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';

const REPO = fileURLToPath(new URL('..', import.meta.url));
const imp = p => import(pathToFileURL(join(REPO, p)).href);
const { buildFacets, buildAdjacency } = await imp('js/meshCore.js');
// GAUSSMAP=path: score another copy of the fitters (e.g. a saved baseline) on the same faces.
const { fitRegionAs, classifyRegion } = process.env.GAUSSMAP
  ? await import(pathToFileURL(process.env.GAUSSMAP).href) : await imp('js/gaussMap.js');
const { attachCurvature } = await imp('js/curvature.js');

const argv = process.argv.slice(2);
const flag = (name) => { const i = argv.indexOf(name); if (i < 0) return null; const v = argv[i + 1]; argv.splice(i, 2); return v; };
const DUMP = flag('--dump');
const RATIO = flag('--ratio');   // classifyRegion opts.ratio, when measuring it
// --contaminate f: add f x (face facets) foreign facets from the rings around each face, as an
// impure region a segmenter would propose. dev is still measured on the true face only.
const CONTAMINATE = Number(flag('--contaminate') || 0);
// --jitter k: move every welded vertex by isotropic Gaussian noise of k x the median edge length
// before fitting. dev is still measured against the TRUE surface at the ORIGINAL vertices, so this
// scores recovery of the seed surface from a mesh that no longer lies on it - not fit to the mesh.
const JITTER = Number(flag('--jitter') || 0);
const TYPES = (flag('--types') || 'plane,cylinder,cone,sphere,torus').split(',');
const NOCLASSIFY = argv.includes('--no-classify') ? (argv.splice(argv.indexOf('--no-classify'), 1), true) : false;
const PART_LIMIT = Number(argv[0] || 1e9);
const MESHERS = (argv[1] || 'native,gmsh,game').split(',');
const MESH_DIR = process.env.FILL_EVAL_DIR || 'D:/abc-data/fill_eval';
const TRUTH_DIR = process.env.FIT_TRUTH_DIR || 'D:/abc-data/fit_truth';
const KIND = ['plane', 'cylinder', 'cone', 'sphere', 'torus'];
const DEG = 180 / Math.PI;
const MOVED = 1e-4;   // floor above this: the mesher displaced the face, see report

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const len = (v) => Math.hypot(v[0], v[1], v[2]);
const unit = (v) => { const l = len(v); return l > 1e-300 ? [v[0] / l, v[1] / l, v[2] / l] : [0, 0, 1]; };
const anyPerp = (a) => unit(Math.abs(a[0]) < 0.9 ? [0, -a[2], a[1]] : [-a[2], 0, a[0]]);

/** Signed distance and unit normal of point q to a surface, in gaussMap's param format. */
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
    // Distance to the full double cone: the nearer of the two nappes, so a fit whose axis
    // points the "wrong" way is judged by the surface it describes, not by its convention.
    const v = sub(q, p.apex); const z = dot(v, a);
    const rad = [v[0] - z * a[0], v[1] - z * a[1], v[2] - z * a[2]]; const r = len(rad);
    const u = r > 1e-300 ? [rad[0] / r, rad[1] / r, rad[2] / r] : anyPerp(a);
    const ca = Math.cos(p.halfAngle), sa = Math.sin(p.halfAngle);
    const s = z >= 0 ? 1 : -1;
    // In the (r, z) half-plane the generator is the ray (sa, s*ca) from the origin. Beyond
    // the apex the nearest point is the apex itself.
    const along = r * sa + Math.abs(z) * ca;
    if (along < 0) return { d: Math.hypot(r, z), n: unit(v) };
    const d = r * ca - Math.abs(z) * sa;
    return { d, n: unit([u[0] * ca - s * a[0] * sa, u[1] * ca - s * a[1] * sa, u[2] * ca - s * a[2] * sa]) };
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

/** surface_targets.py sidecar row -> gaussMap params. `c` orients a cone onto the face's nappe. */
function truthParams(kind, s, c) {
  const P = [s[0], s[1], s[2]], A = [s[3], s[4], s[5]];
  if (kind === 'plane') return { origin: P, normal: unit(A) };
  if (kind === 'cylinder') return { axisPoint: P, axis: unit(A), radius: s[6] };
  if (kind === 'sphere') return { center: P, radius: s[6] };
  if (kind === 'torus') return { center: P, axis: unit(A), majorRadius: s[6], minorRadius: s[7] };
  if (kind === 'cone') {
    let a = unit(A);
    if (dot(sub(c, P), a) < 0) a = [-a[0], -a[1], -a[2]];
    return { apex: P, axis: a, halfAngle: s[6] };
  }
}

const axisErr = (a, b) => Math.acos(Math.min(1, Math.abs(dot(unit(a), unit(b))))) * DEG;

/** Parameter errors, named so the report can pick them by type. Lengths relative to `ref`. */
function paramErrors(kind, t, f, faceScale) {
  if (kind === 'plane') return { axisDeg: axisErr(t.normal, f.normal) };
  if (kind === 'cylinder') return {
    axisDeg: axisErr(t.axis, f.axis),
    radiusRel: Math.abs(f.radius - t.radius) / t.radius,
  };
  if (kind === 'sphere') return {
    radiusRel: Math.abs(f.radius - t.radius) / t.radius,
    centreRel: len(sub(f.center, t.center)) / t.radius,
  };
  if (kind === 'cone') return {
    axisDeg: axisErr(t.axis, f.axis),
    halfDeg: Math.abs(f.halfAngle - t.halfAngle) * DEG,
    apexRel: len(sub(f.apex, t.apex)) / faceScale,
  };
  if (kind === 'torus') return {
    axisDeg: axisErr(t.axis, f.axis),
    majorRel: Math.abs(f.majorRadius - t.majorRadius) / t.majorRadius,
    minorRel: Math.abs(f.minorRadius - t.minorRadius) / t.minorRadius,
    centreRel: len(sub(f.center, t.center)) / t.majorRadius,
  };
}

const copy = b => b.buffer.slice(b.byteOffset, b.byteOffset + b.length);
const manifest = JSON.parse(readFileSync(join(MESH_DIR, 'manifest.json'), 'utf8'));
const rows = [];
const t0 = performance.now();

for (const entry of manifest.parts.slice(0, PART_LIMIT)) {
  let truth;
  try { truth = new Float64Array(copy(readFileSync(join(TRUTH_DIR, entry.part + '.srf')))); }
  catch { continue; }
  const nFaces = truth.length / 12;

  for (const mesher of MESHERS) {
    const v = entry.variants[mesher];
    if (!v) continue;
    const P0 = Float64Array.from(new Float32Array(copy(readFileSync(join(MESH_DIR, v.stem + '.pos')))));
    let P = P0;
    if (JITTER > 0) {
      const edges = [];
      for (let o = 0; o + 8 < P0.length; o += 9) {
        edges.push(Math.hypot(P0[o + 3] - P0[o], P0[o + 4] - P0[o + 1], P0[o + 5] - P0[o + 2]));
      }
      edges.sort((a, b) => a - b);
      const sigma = JITTER * (edges[edges.length >> 1] || 0);
      let seed = 12345;
      for (const ch of v.stem) seed = (seed * 31 + ch.charCodeAt(0)) >>> 0;
      const rand = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) + 0.5) / 4294967296;
      const gauss = () => Math.sqrt(-2 * Math.log(rand())) * Math.cos(2 * Math.PI * rand());
      const moved = new Map();
      P = new Float64Array(P0.length);
      for (let o = 0; o < P0.length; o += 3) {
        const key = P0[o] + '_' + P0[o + 1] + '_' + P0[o + 2];
        let d = moved.get(key);
        if (!d) { d = [sigma * gauss(), sigma * gauss(), sigma * gauss()]; moved.set(key, d); }
        P[o] = P0[o] + d[0]; P[o + 1] = P0[o + 1] + d[1]; P[o + 2] = P0[o + 2] + d[2];
      }
    }
    const F = new Int32Array(copy(readFileSync(join(MESH_DIR, v.stem + '.fid'))));
    const facets = buildFacets(P);
    attachCurvature(P, facets);
    const adjacency = CONTAMINATE > 0 ? buildAdjacency(P, facets) : null;

    let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < P.length; i += 3) for (let k = 0; k < 3; k++) {
      if (P[i + k] < lo[k]) lo[k] = P[i + k];
      if (P[i + k] > hi[k]) hi[k] = P[i + k];
    }
    const diag = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);

    const byFace = new Map();
    for (let t = 0; t < facets.count; t++) {
      if (facets.degenerate[t]) continue;
      const f = F[t];
      if (!byFace.has(f)) byFace.set(f, []);
      byFace.get(f).push(t);
    }

    for (const [face, tris] of byFace) {
      if (face < 0 || face >= nFaces) continue;
      const s = truth.subarray(face * 12, face * 12 + 12);
      if (s[0] !== s[1] || s[0] > 4) continue;
      const kind = KIND[s[0]];
      if (!TYPES.includes(kind)) continue;
      if (tris.length < 3) continue;

      // Unique vertices of the face, and the face centroid for cone orientation.
      const seen = new Set(), verts = [];
      let cx = 0, cy = 0, cz = 0;
      for (const t of tris) for (let k = 0; k < 3; k++) {
        const o = t * 9 + k * 3;
        const key = P0[o] + '_' + P0[o + 1] + '_' + P0[o + 2];
        if (seen.has(key)) continue;
        seen.add(key);
        verts.push([P0[o], P0[o + 1], P0[o + 2]]);
        cx += P0[o]; cy += P0[o + 1]; cz += P0[o + 2];
      }
      const centre = [cx / verts.length, cy / verts.length, cz / verts.length];
      const tp = truthParams(kind, s.subarray(2), centre);
      let faceScale = 0;
      for (const q of verts) faceScale = Math.max(faceScale, len(sub(q, centre)));

      // Floor: mesh vertices against the true surface. Samples: those vertices ON the truth.
      let floor = 0;
      const samples = verts.map(q => {
        const e = evalSurface(kind, tp, q);
        floor = Math.max(floor, Math.abs(e.d));
        return [q[0] - e.d * e.n[0], q[1] - e.d * e.n[1], q[2] - e.d * e.n[2]];
      });

      const row = { part: entry.part, mesher, face, kind, facets: tris.length, diag,
                    floor: floor / diag, faceRel: faceScale / diag };
      const tf = performance.now();
      let region = tris;
      if (CONTAMINATE > 0) {
        const want = Math.max(1, Math.round(CONTAMINATE * tris.length));
        const inFace = new Set(tris), seenT = new Set(tris), foreign = [];
        let ring = [...tris];
        while (foreign.length < want && ring.length) {
          const next = [];
          for (const t of ring) for (const e of adjacency.get(t) || []) {
            if (seenT.has(e.neighbor)) continue;
            seenT.add(e.neighbor);
            next.push(e.neighbor);
          }
          for (const t of next) if (!inFace.has(t) && foreign.length < want) foreign.push(t);
          ring = next;
        }
        region = [...tris, ...foreign];
      }
      const fit = fitRegionAs(kind, region, facets);
      row.ms = performance.now() - tf;
      if (!fit.params || fit.degenerate) {
        row.degenerate = true;
        row.dev = Infinity;
      } else {
        let dev = 0;
        for (const q of samples) dev = Math.max(dev, Math.abs(evalSurface(kind, fit.params, q).d));
        row.dev = dev / diag;
        Object.assign(row, paramErrors(kind, tp, fit.params, faceScale));
      }
      if (!NOCLASSIFY) {
        const c = classifyRegion(region, facets, RATIO ? { ratio: Number(RATIO) } : {});
        row.auto = c.type;
      }
      rows.push(row);
    }
  }
}

// ── report ───────────────────────────────────────────────────────────────────
const quant = (a, q) => { if (!a.length) return NaN; const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(q * s.length))]; };
const e = (x) => Number.isFinite(x) ? x.toExponential(1).padStart(8) : '     inf';
const pc = (x) => `${(100 * x).toFixed(1)}%`.padStart(6);
const PARAMS = {
  plane: ['axisDeg'], cylinder: ['axisDeg', 'radiusRel'], sphere: ['radiusRel', 'centreRel'],
  cone: ['axisDeg', 'halfDeg', 'apexRel'], torus: ['axisDeg', 'majorRel', 'minorRel', 'centreRel'],
};

console.log(`\nfit-from-truth${CONTAMINATE ? ` (contaminated ${CONTAMINATE})` : ''}${JITTER ? ` (jitter ${JITTER} x edge)` : ''}: ${rows.length} faces, ${((performance.now() - t0) / 1000).toFixed(0)} s`);
console.log('dev = max distance, fitted surface to true face, / part diagonal.  floor = mesh vertices to truth.');
console.log('ok<N = dev under 1e-N.  auto = classifyRegion returns the true type.\n');
for (const mesher of MESHERS) {
  console.log(mesher);
  for (const kind of KIND) {
    // A face the MESH moved off its true surface (game_bake's rigid "shells", mostly) says
    // nothing about the fitter: dev can never beat floor. Counted, then excluded.
    const all = rows.filter(r => r.mesher === mesher && r.kind === kind);
    const rs = all.filter(r => r.floor < MOVED);
    const moved = all.length - rs.length;
    if (!rs.length) continue;
    const devs = rs.map(r => r.dev);
    const frac = (fn) => rs.filter(fn).length / rs.length;
    let line = `  ${kind.padEnd(9)} n ${String(rs.length).padStart(5)}  moved ${String(moved).padStart(4)}  degen ${pc(frac(r => r.degenerate))}` +
      `  floor p50 ${e(quant(rs.map(r => r.floor), 0.5))}  dev p50 ${e(quant(devs, 0.5))} p90 ${e(quant(devs, 0.9))}` +
      `  ok<3 ${pc(frac(r => r.dev < 1e-3))} ok<4 ${pc(frac(r => r.dev < 1e-4))} ok<5 ${pc(frac(r => r.dev < 1e-5))}`;
    if (!NOCLASSIFY) line += `  auto ${pc(frac(r => r.auto === kind))}`;
    line += `  ms p50 ${quant(rs.map(r => r.ms), 0.5).toFixed(1)}`;
    console.log(line);
    const ok = rs.filter(r => !r.degenerate);
    console.log('            ' + PARAMS[kind].map(k => `${k} p50 ${e(quant(ok.map(r => r[k]), 0.5))} p90 ${e(quant(ok.map(r => r[k]), 0.9))}`).join('   '));
  }
}
if (!NOCLASSIFY) {
  console.log('\nauto confusion (all meshers): true -> returned');
  for (const kind of KIND) {
    const rs = rows.filter(r => r.kind === kind);
    if (!rs.length) continue;
    const c = {};
    for (const r of rs) c[r.auto] = (c[r.auto] || 0) + 1;
    console.log(`  ${kind.padEnd(9)} ` + Object.entries(c).sort((a, b) => b[1] - a[1]).map(([k, n]) => `${k} ${pc(n / rs.length).trim()}`).join('  '));
  }
}
if (DUMP) writeFileSync(DUMP, JSON.stringify(rows));
console.log('');
