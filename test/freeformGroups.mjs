// The freeform export by groups (a part's faces): an STL's faces found by flood fill over edges bent less than
// --crease degrees (what one paint click per face gives), exported with freeformSolid({ groups }), timed, read back
// with OCCT (--check), and every shared edge measured from the output: the gap everywhere; inside a group the
// normal angle and the G2 error (normal curvature across the edge from both sides); on a border the angle the two
// faces meet at.
// node test/freeformGroups.mjs <file.stl> ... [--crease 20] [--quads N] [--check] [--out file.step]
import fs from 'node:fs';
import { freeformSolid } from '../js/subd/freeform.js';
import { bernstein } from '../js/subd/kpCap.js';

const args = process.argv.slice(2);
const flag = (k) => { const i = args.indexOf(k); return i >= 0 ? args.splice(i, 2)[1] : null; };
const crease = Number(flag('--crease') ?? 20), quads = flag('--quads'), out = flag('--out');
const check = args.includes('--check');
const files = args.filter((a) => !a.startsWith('--'));

function readStl(file) {
  const buf = fs.readFileSync(file), n = buf.readUInt32LE(80), P = new Float64Array(n * 9);
  for (let t = 0; t < n; t++) for (let k = 0; k < 9; k++) P[t * 9 + k] = buf.readFloatLE(84 + t * 50 + 12 + k * 4);
  return P;
}

function floodGroups(P, deg) {
  const n = P.length / 9, key = new Map(), idx = new Int32Array(n * 3);
  for (let i = 0; i < n * 3; i++) {
    const k = `${P[i * 3]},${P[i * 3 + 1]},${P[i * 3 + 2]}`;
    if (!key.has(k)) key.set(k, key.size);
    idx[i] = key.get(k);
  }
  const N = [];
  for (let t = 0; t < n; t++) {
    const o = t * 9, u = [P[o + 3] - P[o], P[o + 4] - P[o + 1], P[o + 5] - P[o + 2]], w = [P[o + 6] - P[o], P[o + 7] - P[o + 1], P[o + 8] - P[o + 2]];
    const c = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]], l = Math.hypot(...c) || 1;
    N.push(c.map((x) => x / l));
  }
  const edges = new Map();
  for (let t = 0; t < n; t++) for (let k = 0; k < 3; k++) {
    const a = idx[t * 3 + k], b = idx[t * 3 + (k + 1) % 3], e = a < b ? `${a},${b}` : `${b},${a}`;
    if (!edges.has(e)) edges.set(e, []);
    edges.get(e).push(t);
  }
  const adj = Array.from({ length: n }, () => []), cos = Math.cos(deg * Math.PI / 180);
  for (const ts of edges.values()) {
    if (ts.length !== 2) continue;
    const [s, t] = ts;
    if (N[s][0] * N[t][0] + N[s][1] * N[t][1] + N[s][2] * N[t][2] >= cos) { adj[s].push(t); adj[t].push(s); }
  }
  const g = new Int32Array(n).fill(-1);
  let count = 0;
  for (let s = 0; s < n; s++) {
    if (g[s] >= 0) continue;
    const stack = [s];
    g[s] = count;
    while (stack.length) { const t = stack.pop(); for (const u of adj[t]) if (g[u] < 0) { g[u] = count; stack.push(u); } }
    count++;
  }
  return { groups: g, count };
}

function evalPatch(f, u, v) {                       // derivatives up to order 2 of a Bezier patch
  const p = f.uCount - 1, Bu = bernstein(p, u, 2), Bv = bernstein(p, v, 2), r = {};
  for (const [a, b] of [[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]]) {
    const x = [0, 0, 0];
    for (let i = 0; i <= p; i++) for (let j = 0; j <= p; j++) {
      const w = Bu[a][i] * Bv[b][j];
      if (w) for (let c = 0; c < 3; c++) x[c] += w * f.poles[(i * f.vCount + j) * 3 + c];
    }
    r[`${a}${b}`] = x;
  }
  return r;
}
const sub = (a, b) => a.map((x, i) => x - b[i]);
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const unit = (a) => { const n = Math.hypot(...a) || 1; return a.map((x) => x / n); };
function curvatureAcross(d, n, D) {                 // normal curvature along D, and |k| max
  const Su = d['10'], Sv = d['01'], E = dot(Su, Su), F = dot(Su, Sv), G = dot(Sv, Sv);
  const L = dot(d['20'], n), M = dot(d['11'], n), N = dot(d['02'], n), det = E * G - F * F;
  const bu = dot(Su, D), bv = dot(Sv, D), a = (G * bu - F * bv) / det, b = (E * bv - F * bu) / det;
  const k = (a * a * L + 2 * a * b * M + b * b * N) / (a * a * E + 2 * a * b * F + b * b * G);
  const H = (E * N - 2 * F * M + G * L) / (2 * det), K = (L * N - M * M) / det;
  return [k, Math.abs(H) + Math.sqrt(Math.max(H * H - K, 0))];
}

function continuity({ faces, loops, faceGroup }, diag) {
  const owner = new Map();
  loops.forEach((loop, f) => loop.forEach(([a, u0, v0], i) => {
    const [b, u1, v1] = loop[(i + 1) % loop.length];
    owner.set(`${a},${b}`, [f, [u0, v0], [u1, v1]]);
  }));
  const r = { gap: 0, innerAngle: 0, innerG2: 0, borderMin: 180, borderMax: 0, inner: 0, border: 0, open: 0 };
  for (const [key, [f, p0, p1]] of owner) {
    const [a, b] = key.split(',').map(Number);
    if (a > b) continue;
    const other = owner.get(`${b},${a}`);
    if (!other) { r.open++; continue; }
    const [g, q0, q1] = other, same = faceGroup[f] === faceGroup[g];
    if (same) r.inner++; else r.border++;
    for (let m = 0; m <= 16; m++) {
      const s = m / 16, u = p0[0] + s * (p1[0] - p0[0]), v = p0[1] + s * (p1[1] - p0[1]);
      const ug = q1[0] + s * (q0[0] - q1[0]), vg = q1[1] + s * (q0[1] - q1[1]);
      const dA = evalPatch(faces[f], u, v), dB = evalPatch(faces[g], ug, vg);
      r.gap = Math.max(r.gap, Math.hypot(...sub(dA['00'], dB['00'])) / diag);
      const nA = unit(cross(dA['10'], dA['01'])), nB = unit(cross(dB['10'], dB['01']));
      const angle = Math.acos(Math.max(-1, Math.min(1, dot(nA, nB)))) * 180 / Math.PI;
      const sine = (d) => Math.hypot(...cross(d['10'], d['01'])) / ((Math.hypot(...d['10']) * Math.hypot(...d['01'])) || 1);
      if (!same && Math.min(sine(dA), sine(dB)) < 0.05) { r.degenerate = (r.degenerate ?? 0) + 1; continue; }
      if (!same) {
        if (m > 0 && m < 16) {
          if (angle < r.borderMin) { r.borderMin = angle; r.minAt = dA['00']; }
          if (angle > r.borderMax) { r.borderMax = angle; r.maxAt = dA['00']; }
        }
        continue;
      }
      r.innerAngle = Math.max(r.innerAngle, angle);
      const along = dA['10'].map((x, c) => x * (p1[0] - p0[0]) + dA['01'][c] * (p1[1] - p0[1]));
      const D = unit(cross(nA, unit(along)));
      const [kA, mA] = curvatureAcross(dA, nA, D), [kB, mB] = curvatureAcross(dB, nB, D);
      const g2 = Math.abs(kA - kB) / Math.max(mA, mB, 1 / diag);
      if (g2 > r.innerG2) { r.innerG2 = g2; r.g2At = dA['00']; r.g2Group = faceGroup[f]; r.g2Face = f; }
    }
  }
  // skew: the smallest sine of the angle between a face's two directions, over a 5 x 5 grid of each face
  r.skewed = 0; r.minSine = 1;
  faces.forEach((f) => {
    let m = 1;
    for (let i = 0; i <= 4; i++) for (let j = 0; j <= 4; j++) {
      const d = evalPatch(f, i / 4, j / 4), c = Math.hypot(...cross(d['10'], d['01'])) / ((Math.hypot(...d['10']) * Math.hypot(...d['01'])) || 1);
      m = Math.min(m, c);
    }
    r.minSine = Math.min(r.minSine, m);
    if (m < 0.1) r.skewed++;
  });
  return r;
}

let oc = null;
for (const file of files) {
  // an STL's faces come from the flood fill; a mesh.json { vertices, faces, labels } brings its own
  let P, groups, count;
  if (file.endsWith('.json')) {
    const m = JSON.parse(fs.readFileSync(file, 'utf8'));
    P = new Float64Array(m.faces.length * 9);
    m.faces.forEach((t, i) => t.forEach((v, k) => P.set(m.vertices[v], i * 9 + k * 3)));
    groups = Int32Array.from(m.labels); count = new Set(m.labels).size;
  } else ({ groups, count } = floodGroups(P = readStl(file), crease));
  const t0 = performance.now();
  const r = await freeformSolid(P, { groups, part: 'Test', quads: quads ? Number(quads) : null });
  const ms = performance.now() - t0;
  console.log(`${file.split(/[\\/]/).pop()}: ${count} faces at ${crease} deg, ${(ms / 1000).toFixed(2)} s`);
  for (const b of r.report.bodies) {
    console.log('  ', b.error ? `${b.name}: ${b.error}` : `${b.name}: ${b.groups ?? 1} groups ${JSON.stringify(b.kinds ?? {})}, ${b.faces} patches, `
      + `rms ${(b.relRms * 100).toFixed(3)} % max ${(b.relMax * 100).toFixed(3)} % of size, cage ${b.ms.remesh.toFixed(0)} ms, fit ${b.ms.fit.toFixed(0)} ms`);
  }
  if (!r.solids?.length) continue;
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < P.length; i += 3) for (let c = 0; c < 3; c++) { lo[c] = Math.min(lo[c], P[i + c]); hi[c] = Math.max(hi[c], P[i + c]); }
  const diag = Math.hypot(...hi.map((h, c) => h - lo[c]));
  for (const s of r.solids) {
    if (!s.faceGroup) continue;
    const c = continuity(s, diag);
    console.log(`   edges: ${c.inner} inside faces (angle ${c.innerAngle.toExponential(1)} deg, G2 ${(100 * c.innerG2).toExponential(1)} %), `
      + `${c.border} on borders (faces meet at ${c.borderMin.toFixed(1)}-${c.borderMax.toFixed(1)} deg), gap ${c.gap.toExponential(1)} of size, open ${c.open}, ${c.degenerate ?? 0} border samples at degenerate points skipped; ${c.skewed} patches skewed past sine 0.1 (least ${c.minSine.toFixed(3)})`);
    if (process.env.WHERE) console.log(`   least at ${c.minAt?.map((x) => x.toFixed(1))}, most at ${c.maxAt?.map((x) => x.toFixed(1))}; worst G2 at ${c.g2At?.map((x) => x.toFixed(1))} (group ${c.g2Group}, patch ${c.g2Face} of degree ${s.faces[c.g2Face]?.uCount - 1})`);
  }
  if (out && r.step) fs.writeFileSync(out, r.step);
  if (check && r.step) {
    const { loadOcct } = await import('../js/knit/index.js');
    const { score, subShapes } = await import('../js/knit/build.js');
    oc ??= await loadOcct();
    oc.FS.writeFile('/tmp/in.step', r.step);
    const reader = new oc.STEPControl_Reader_1();
    if (reader.ReadFile('/tmp/in.step') !== oc.IFSelect_ReturnStatus.IFSelect_RetDone) { console.log('   check: OCCT could not read it'); continue; }
    reader.TransferRoots(new oc.Message_ProgressRange_1());
    const shape = reader.OneShape(), sc = score(oc, shape, { volume: () => null });
    console.log(`   check: solids ${subShapes(oc, shape, oc.TopAbs_ShapeEnum.TopAbs_SOLID).length}, faces ${subShapes(oc, shape, oc.TopAbs_ShapeEnum.TopAbs_FACE).length}, `
      + `closed ${sc.closed}, valid ${sc.valid}`);
  }
}
