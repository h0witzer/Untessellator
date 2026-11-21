// fastSubdToNurbs on saved cages: time, and G2 across every edge measured from the output (the same test as the
// bake-off checker: two G1 surfaces are G2 along a curve iff their normal curvatures across it agree).
// node test/subdFast.mjs <dir-with-cage.json> ... [--variant '{"flipSpoke":true}'] [--write]
import fs from 'node:fs';
import path from 'node:path';
import { fastSubdToNurbs } from '../js/subd/fastNurbs.js';
import { bernstein } from '../js/subd/kpCap.js';

const args = process.argv.slice(2);
const vi = args.indexOf('--variant');
const variant = vi >= 0 ? JSON.parse(args.splice(vi, 2)[1]) : {};
const write = args.includes('--write');
const dirs = args.filter((a) => !a.startsWith('--'));

function evalPatch(f, u, v) {                       // derivatives up to order 2 of a Bezier patch
  const p = f.uCount - 1, Bu = bernstein(p, u, 2), Bv = bernstein(p, v, 2), out = {};
  for (const [a, b] of [[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]]) {
    const x = [0, 0, 0];
    for (let i = 0; i <= p; i++) for (let j = 0; j <= p; j++) {
      const w = Bu[a][i] * Bv[b][j];
      if (w) for (let c = 0; c < 3; c++) x[c] += w * f.poles[(i * f.vCount + j) * 3 + c];
    }
    out[`${a}${b}`] = x;
  }
  return out;
}
const sub = (a, b) => a.map((x, i) => x - b[i]);
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const unit = (a) => { const n = Math.hypot(...a) || 1; return a.map((x) => x / n); };
function curvatureAcross(d, n, D) {                 // normal curvature along D and |k|max
  const Su = d['10'], Sv = d['01'], E = dot(Su, Su), F = dot(Su, Sv), G = dot(Sv, Sv);
  const L = dot(d['20'], n), M = dot(d['11'], n), N = dot(d['02'], n), det = E * G - F * F;
  const bu = dot(Su, D), bv = dot(Sv, D), a = (G * bu - F * bv) / det, b = (E * bv - F * bu) / det;
  const k = (a * a * L + 2 * a * b * M + b * b * N) / (a * a * E + 2 * a * b * F + b * b * G);
  const H = (E * N - 2 * F * M + G * L) / (2 * det), K = (L * N - M * M) / det;
  return [k, Math.abs(H) + Math.sqrt(Math.max(H * H - K, 0))];
}

for (const dir of dirs) {
  const cage = JSON.parse(fs.readFileSync(path.join(dir, 'cage.json'), 'utf8'));
  const t = performance.now();
  const r = fastSubdToNurbs(cage.vertices, cage.faces, { variant });
  const ms = performance.now() - t;
  const { faces, loops } = r, vertices = cage.vertices;
  const lo = [0, 1, 2].map((c) => Math.min(...vertices.map((p) => p[c]))), hi = [0, 1, 2].map((c) => Math.max(...vertices.map((p) => p[c])));
  const diag = Math.hypot(...hi.map((h, c) => h - lo[c]));
  // each loop segment a -> b with its uv ends; the face across runs it b -> a (T-junction sides included)
  const owner = new Map();
  loops.forEach((loop, f) => loop.forEach(([a, u0, v0], i) => { const [b, u1, v1] = loop[(i + 1) % loop.length]; owner.set(`${a},${b}`, [f, [u0, v0], [u1, v1]]); }));
  let worstAngle = 0, worstG2 = 0, worstGap = 0, edges = 0;
  for (const [key, [f, p0, p1]] of owner) {
    const [a, b] = key.split(',').map(Number);
    if (a > b) continue;
    const [g, q0, q1] = owner.get(`${b},${a}`);
    edges++;
    for (let m = 0; m <= 24; m++) {
      const s = m / 24, u = p0[0] + s * (p1[0] - p0[0]), v = p0[1] + s * (p1[1] - p0[1]);
      const ug = q1[0] + s * (q0[0] - q1[0]), vg = q1[1] + s * (q0[1] - q1[1]);
      const dA = evalPatch(faces[f], u, v), dB = evalPatch(faces[g], ug, vg);
      worstGap = Math.max(worstGap, Math.hypot(...sub(dA['00'], dB['00'])) / diag);
      const nA = unit(cross(dA['10'], dA['01'])); let nB = unit(cross(dB['10'], dB['01']));
      if (dot(nA, nB) < 0) nB = nB.map((x) => -x);
      worstAngle = Math.max(worstAngle, Math.acos(Math.min(1, dot(nA, nB))) * 180 / Math.PI);
      const along = dA['10'].map((x, c) => x * (p1[0] - p0[0]) + dA['01'][c] * (p1[1] - p0[1]));
      const D = unit(cross(nA, unit(along)));
      const [kA, mA] = curvatureAcross(dA, nA, D), [kB, mB] = curvatureAcross(dB, nB, D);
      worstG2 = Math.max(worstG2, Math.abs(kA - kB) / Math.max(mA, mB, 1 / diag));
    }
  }
  const cvs = faces.reduce((s, f) => s + f.uCount * f.vCount, 0);
  console.log(path.basename(dir), JSON.stringify(variant), `${ms.toFixed(1)} ms (build ${r.stats.ms.build.toFixed(1)}),`,
    `${faces.length} faces, ${r.stats.stars} stars, ${r.stats.subdivisions} subdivisions, ${cvs} CVs |`,
    `gap ${worstGap.toExponential(1)} angle ${worstAngle.toExponential(1)} deg G2 ${(100 * worstG2).toExponential(2)} %`);
  if (write) fs.writeFileSync(path.join(dir, 'untess_fast.json'), JSON.stringify({ seconds: ms / 1000, faces: r.faces, stats: r.stats }));
}
