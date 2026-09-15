/**
 * emitTruth.mjs — where do EMITTED surfaces land, for patches labelled with their true type?
 *
 * Usage:  THREE_MODULE=<path to three.module.js> node test/emitTruth.mjs [parts] [meshers] [--out rows.json]
 *
 * fitTruth.mjs scores the fitted PARAMETERS. This scores what is drawn and exported from them,
 * which is what a user actually sees: the labelled patch (the true B-rep face's triangles, typed
 * with its true type) goes through the same calls the app makes -
 *
 *   classifyGroupAsGauss(type)                     the export adapter (studio.js exportStep)
 *   computeAnalyticalBoundaries + brepVisualizer   the preview meshes (main.js)
 *
 * - and every emitted point is scored against the TRUE surface and the TRUE patch:
 *
 *   dev      max distance of emitted points from the true surface / part diagonal
 *   excess   fraction of emitted points farther than 2 median edges from the true patch's TRIANGLES
 *            (not its vertices: sliver meshes leave vertex gaps wider than that tolerance)
 *   axIoU    cylinders and cones: overlap of the emitted and true axial ranges, along the TRUE axis
 *
 * A band slid along its own cone lies ON the cone, so dev alone cannot see it; axIoU can.
 *
 * --out writes the fits and bounds for ml/emit_truth_occ.py, which rebuilds the STEP faces with
 * OCCT exactly as js/brepBuilder.js does and scores those. Without THREE_MODULE the preview is
 * skipped and only the rows are written.
 */

import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { registerHooks } from 'node:module';

const THREE_MODULE = process.env.THREE_MODULE;
if (THREE_MODULE) {
  const url = pathToFileURL(THREE_MODULE).href;
  registerHooks({ resolve: (spec, ctx, next) => (spec === 'three' ? { url, shortCircuit: true } : next(spec, ctx)) });
}
const REPO = fileURLToPath(new URL('..', import.meta.url));
const imp = p => import(pathToFileURL(join(REPO, p)).href);
const { classifyGroupAsGauss } = await imp('js/surfaceFitter.js');
const { computeAnalyticalBoundaries } = await imp('js/brepBuilder.js');
const vis = THREE_MODULE ? await imp('js/brepVisualizer.js') : null;
const THREE = THREE_MODULE ? await import(pathToFileURL(THREE_MODULE).href) : null;

const argv = process.argv.slice(2);
const flag = (name) => { const i = argv.indexOf(name); if (i < 0) return null; const v = argv[i + 1]; argv.splice(i, 2); return v; };
const OUT = flag('--out');
const PART_LIMIT = Number(argv[0] || 1e9);
const MESHERS = (argv[1] || 'native,gmsh').split(',');
const MESH_DIR = process.env.FILL_EVAL_DIR || 'D:/abc-data/fill_eval';
const TRUTH_DIR = process.env.FIT_TRUTH_DIR || 'D:/abc-data/fit_truth';
const KIND = ['plane', 'cylinder', 'cone', 'sphere', 'torus'];
const MAX_VERTS = 400;

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/** Distance from point p to triangle t (9 numbers): closest point by region (Ericson, RTCD 5.1.5). */
function pointTriangle(p, t) {
  const a = [t[0], t[1], t[2]], b = [t[3], t[4], t[5]], c = [t[6], t[7], t[8]];
  const ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
  const ap = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
  const at = (u, v) => [a[0] + u * ab[0] + v * ac[0], a[1] + u * ab[1] + v * ac[1], a[2] + u * ab[2] + v * ac[2]];
  const d = (q) => Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
  const d1 = dot(ab, ap), d2 = dot(ac, ap);
  if (d1 <= 0 && d2 <= 0) return d(a);
  const bp = [p[0] - b[0], p[1] - b[1], p[2] - b[2]];
  const d3 = dot(ab, bp), d4 = dot(ac, bp);
  if (d3 >= 0 && d4 <= d3) return d(b);
  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) return d(at(d1 / (d1 - d3), 0));
  const cp = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
  const d5 = dot(ab, cp), d6 = dot(ac, cp);
  if (d6 >= 0 && d5 <= d6) return d(c);
  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) return d(at(0, d2 / (d2 - d6)));
  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
    const w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    return d([b[0] + w * (c[0] - b[0]), b[1] + w * (c[1] - b[1]), b[2] + w * (c[2] - b[2])]);
  }
  const den = 1 / (va + vb + vc);
  return d(at(vb * den, vc * den));
}
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const len = (v) => Math.hypot(v[0], v[1], v[2]);
const unit = (v) => { const l = len(v); return l > 1e-300 ? [v[0] / l, v[1] / l, v[2] / l] : [0, 0, 1]; };

/** Unsigned distance of q to the TRUE surface (sidecar layout, cone on its face's nappe). */
function truthDistance(kind, s, q, faceCentre) {
  const P = [s[0], s[1], s[2]];
  let a = unit([s[3], s[4], s[5]]);
  const v = sub(q, P);
  if (kind === 'plane') return Math.abs(dot(v, a));
  if (kind === 'sphere') return Math.abs(len(v) - s[6]);
  const z = dot(v, a);
  const rho = Math.sqrt(Math.max(0, dot(v, v) - z * z));
  if (kind === 'cylinder') return Math.abs(rho - s[6]);
  if (kind === 'torus') return Math.abs(Math.hypot(rho - s[6], z) - s[7]);
  if (dot(sub(faceCentre, P), a) < 0) a = [-a[0], -a[1], -a[2]];
  const zz = dot(v, a);
  return Math.abs(rho * Math.cos(s[6]) - zz * Math.sin(s[6]));   // the face's nappe only
}

/**
 * The planar boundary the export used BEFORE planeBoundary, for ml/emit_truth_occ.py to rebuild
 * as the old face: brepBuilder._orderBoundaryLoop, verbatim in logic, over the group's boundary
 * edges (the same edge set its group-to-group map yields on a closed mesh), 1e4 vertex keys.
 */
function oldPlaneLoop(tris, P) {
  const qk = (i) => `${Math.round(P[i] * 1e4)}_${Math.round(P[i + 1] * 1e4)}_${Math.round(P[i + 2] * 1e4)}`;
  const count = new Map();
  for (const t of tris) for (let e = 0; e < 3; e++) {
    const a = t * 9 + e * 3, b = t * 9 + ((e + 1) % 3) * 3;
    const ka = qk(a), kb = qk(b);
    const key = ka < kb ? `${ka}|${kb}` : `${kb}|${ka}`;
    const hit = count.get(key);
    if (hit) hit.n++; else count.set(key, { n: 1, a, b });
  }
  const pairs = [...count.values()].filter(e => e.n === 1).map(e => [e.a, e.b]);
  if (pairs.length < 3) return null;
  const adj = new Map(), vidx = new Map();
  for (const [a, b] of pairs) {
    const ka = qk(a), kb = qk(b);
    if (!adj.has(ka)) { adj.set(ka, []); vidx.set(ka, a); }
    if (!adj.has(kb)) { adj.set(kb, []); vidx.set(kb, b); }
    adj.get(ka).push({ key: kb, idx: b });
    adj.get(kb).push({ key: ka, idx: a });
  }
  const loop = [], visited = new Set();
  let curKey = [...adj.keys()][0], curIdx = vidx.get(curKey);
  while (!visited.has(curKey)) {
    visited.add(curKey);
    loop.push([P[curIdx], P[curIdx + 1], P[curIdx + 2]]);
    let next = null;
    for (const nb of adj.get(curKey) ?? []) if (!visited.has(nb.key)) { next = nb; break; }
    if (!next) break;
    curKey = next.key; curIdx = next.idx;
  }
  return loop.length >= 3 ? loop : null;
}

const copy = b => b.buffer.slice(b.byteOffset, b.byteOffset + b.length);
const manifest = JSON.parse(readFileSync(join(MESH_DIR, 'manifest.json'), 'utf8'));
const rows = [];
const t0 = performance.now();

for (const entry of manifest.parts.slice(0, PART_LIMIT)) {
  let truth;
  try { truth = new Float64Array(copy(readFileSync(join(TRUTH_DIR, entry.part + '.srf')))); } catch { continue; }
  for (const mesher of MESHERS) {
    const variant = entry.variants[mesher];
    if (!variant) continue;
    const P = Float32Array.from(new Float32Array(copy(readFileSync(join(MESH_DIR, variant.stem + '.pos')))));
    const F = new Int32Array(copy(readFileSync(join(MESH_DIR, variant.stem + '.fid'))));
    const geometry = {
      attributes: { position: { array: P, count: P.length / 3,
        getX: i => P[i * 3], getY: i => P[i * 3 + 1], getZ: i => P[i * 3 + 2] } },
      index: null,
    };
    let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < P.length; i += 3) for (let k = 0; k < 3; k++) {
      lo[k] = Math.min(lo[k], P[i + k]); hi[k] = Math.max(hi[k], P[i + k]);
    }
    const diag = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);

    const byFace = new Map();
    for (let t = 0; t < F.length; t++) {
      if (!byFace.has(F[t])) byFace.set(F[t], []);
      byFace.get(F[t]).push(t);
    }
    for (const [face, tris] of byFace) {
      const s = truth.subarray(face * 12, face * 12 + 12);
      if (s[0] !== s[1] || s[0] > 4 || tris.length < 3) continue;
      const kind = KIND[s[0]];
      const sp = s.subarray(2);

      const seen = new Set(), verts = [];
      const edges = [];
      for (const t of tris) {
        const o = t * 9;
        for (let k = 0; k < 3; k++) {
          const a = o + k * 3, b = o + ((k + 1) % 3) * 3;
          edges.push(Math.hypot(P[a] - P[b], P[a + 1] - P[b + 1], P[a + 2] - P[b + 2]));
          const key = P[a] + '_' + P[a + 1] + '_' + P[a + 2];
          if (!seen.has(key)) { seen.add(key); verts.push([P[a], P[a + 1], P[a + 2]]); }
        }
      }
      edges.sort((x, y) => x - y);
      const tol = 2 * edges[edges.length >> 1];
      const centre = verts.reduce((c, q) => [c[0] + q[0] / verts.length, c[1] + q[1] / verts.length, c[2] + q[2] / verts.length], [0, 0, 0]);
      const floor = Math.max(...verts.map(q => truthDistance(kind, sp, q, centre))) / diag;

      const part = { triangleIndices: new Set(tris) };
      part.surface = classifyGroupAsGauss(kind, part, geometry);
      const emitted = part.surface.type === kind && !part.surface.degenerate;
      const { boundaries } = computeAnalyticalBoundaries([part], geometry);
      const bound = boundaries.get(0) ?? null;

      const row = { part: entry.part, mesher, face, kind, facets: tris.length, diag, tol, floor, emitted,
                    fit: emitted ? part.surface.params : null, bound, truth: Array.from(sp), centre };
      const stride = Math.max(1, Math.ceil(verts.length / MAX_VERTS));
      row.verts = verts.filter((_, i) => i % stride === 0);
      row.tris = tris.map(t => Array.from(P.subarray(t * 9, t * 9 + 9)));
      if (kind === 'plane') row.oldLoop = oldPlaneLoop(tris, P);

      if (vis && emitted) {
        const scored = [];
        for (const [name, build] of [
          ['faces', () => vis.buildAnalyticalFacesMesh([part], geometry, boundaries, {})],
          ['wire', () => vis.buildBrepOverlay([part], geometry, {})],
        ]) {
          const pts = [];
          const obj = build();
          obj.updateMatrixWorld(true);
          obj.traverse((o) => {
            const pa = o.geometry?.attributes?.position;
            if (!pa) return;
            const v = new THREE.Vector3();
            for (let i = 0; i < pa.count; i++) {
              v.fromBufferAttribute(pa, i).applyMatrix4(o.matrixWorld);
              pts.push([v.x, v.y, v.z]);
            }
          });
          if (!pts.length) { scored.push([name, null]); continue; }
          let dev = 0, far = 0;
          for (const q of pts) {
            dev = Math.max(dev, truthDistance(kind, sp, q, centre));
            let near = Infinity;
            for (const tr of row.tris) { const d = pointTriangle(q, tr); if (d < near) near = d; if (near <= tol) break; }
            if (near > tol) far++;
          }
          const m = { dev: dev / diag, excess: far / pts.length, n: pts.length };
          if (kind === 'cylinder' || kind === 'cone') {
            const a = unit([sp[3], sp[4], sp[5]]);
            const ax = (q) => dot(q, a);
            const e = pts.map(ax), t = verts.map(ax);
            const e0 = Math.min(...e), e1 = Math.max(...e), t0 = Math.min(...t), t1 = Math.max(...t);
            const inter = Math.max(0, Math.min(e1, t1) - Math.max(e0, t0));
            const uni = Math.max(e1, t1) - Math.min(e0, t0);
            m.axIoU = uni > 0 ? inter / uni : 1;
          }
          scored.push([name, m]);
        }
        row.preview = Object.fromEntries(scored);
      }
      rows.push(row);
    }
  }
}

// ── preview report ───────────────────────────────────────────────────────────
const q = (a, p) => { if (!a.length) return NaN; const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(p * s.length))]; };
const e = (x) => (Number.isFinite(x) ? x.toExponential(1).padStart(8) : '     n/a');
const pc = (x) => `${(100 * x).toFixed(1)}%`.padStart(6);
console.log(`\nemitted-surface truth check: ${rows.length} labelled faces, ${((performance.now() - t0) / 1000).toFixed(0)} s`);
console.log('dev = max emitted-point distance from the true surface / diagonal; excess = share of emitted points off the true patch (by triangle distance);');
console.log('axIoU = emitted vs true axial range along the true axis. Faces the mesher moved (floor > 1e-4) excluded.\n');
for (const mesher of MESHERS) {
  console.log(mesher);
  for (const kind of KIND) {
    const rs = rows.filter(r => r.mesher === mesher && r.kind === kind && r.floor < 1e-4);
    if (!rs.length) continue;
    let line = `  ${kind.padEnd(9)} n ${String(rs.length).padStart(5)}  emitted ${pc(rs.filter(r => r.emitted).length / rs.length)}`;
    if (vis) {
      for (const name of ['faces', 'wire']) {
        const ms = rs.map(r => r.preview?.[name]).filter(Boolean);
        const drawn = ms.length / rs.length;
        line += `  | ${name} drawn ${pc(drawn)}`;
        if (!ms.length) continue;
        line += ` dev p50 ${e(q(ms.map(m => m.dev), 0.5))} ok<4 ${pc(ms.filter(m => m.dev < 1e-4).length / ms.length)}` +
                ` excess p50 ${pc(q(ms.map(m => m.excess), 0.5))} p90 ${pc(q(ms.map(m => m.excess), 0.9))}`;
        const io = ms.map(m => m.axIoU).filter(Number.isFinite);
        if (io.length) line += ` axIoU p50 ${q(io, 0.5).toFixed(2)} p10 ${q(io, 0.1).toFixed(2)}`;
      }
    }
    console.log(line);
  }
}
if (OUT) writeFileSync(OUT, JSON.stringify(rows));
console.log('');
