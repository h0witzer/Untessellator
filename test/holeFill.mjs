/**
 * holeFill.mjs — js/holeFill.js on shapes with a known answer: closed afterwards (every edge once each way), with
 * the right volume and cap area. A cube missing a face, a sphere missing a cap, an L-shaped (concave) hole, two
 * holes touching at a corner, a fin on an edge and a wall inside (both go, by the winding number), and an open thin
 * plate standing alone (kept and capped).
 *
 * Usage:  node test/holeFill.mjs      (exit 1 when a shape fails)
 */

import { fillHoles } from '../js/holeFill.js';

let failures = 0;

const quad = (a, b, c, d) => [[a, b, c], [a, c, d]];
const soup = (tris) => Float64Array.from(tris.flatMap((t) => t.flat()));
function check(name, P, want = {}) {
  const r = fillHoles(P);
  const out = r.positions, n = out.length / 9;
  const key = (i) => `${out[i * 3]},${out[i * 3 + 1]},${out[i * 3 + 2]}`;
  const dir = new Map();
  let vol = 0, capArea = 0;
  for (let t = 0; t < n; t++) {
    const v = [0, 1, 2].map((k) => key(t * 3 + k));
    for (let k = 0; k < 3; k++) { const e = `${v[k]}|${v[(k + 1) % 3]}`; dir.set(e, (dir.get(e) ?? 0) + 1); }
    const a = [0, 1, 2].map((j) => out[t * 9 + j]), b = [0, 1, 2].map((j) => out[t * 9 + 3 + j]), c = [0, 1, 2].map((j) => out[t * 9 + 6 + j]);
    vol += (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])) / 6;
    if (r.src[t] < 0) { const u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], w = [c[0] - a[0], c[1] - a[1], c[2] - a[2]]; capArea += Math.hypot(u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]) / 2; }
  }
  let open = 0, bad = 0;
  for (const [e, c] of dir) { const [x, y] = e.split('|'); const back = dir.get(`${y}|${x}`) ?? 0; if (!back) open++; if (c > 1) bad++; }
  const ok = open === 0 && bad === 0 && (want.vol === undefined || Math.abs(vol - want.vol) < (want.tol ?? 1e-6)) && (want.capArea === undefined || Math.abs(capArea - want.capArea) < 1e-6);
  if (!ok) failures++;
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${name.padEnd(30)} open ${open}, doubled ${bad}, volume ${vol.toFixed(5)}${want.vol !== undefined ? ` (want ${want.vol})` : ''}, cap area ${capArea.toFixed(4)}${want.capArea !== undefined ? ` (want ${want.capArea})` : ''}, ${JSON.stringify({ filled: r.report.filled, caps: r.report.cap_triangles, walls: r.report.walls_removed, fins: r.report.fins_removed, flatLeft: r.report.flat_left, sheets: r.report.sheets_split, skipped: r.report.skipped })}`);
  return r;
}
const C = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]];
const cubeFaces = [quad(C[0], C[3], C[2], C[1]), quad(C[4], C[5], C[6], C[7]), quad(C[0], C[1], C[5], C[4]), quad(C[2], C[3], C[7], C[6]), quad(C[1], C[2], C[6], C[5]), quad(C[0], C[4], C[7], C[3])];
// 1. a cube with its top missing
check('cube, top missing', soup(cubeFaces.filter((_, i) => i !== 1).flat()), { vol: 1, capArea: 1 });
// 2. a sphere with a cap missing: closed, and the faired cap bulges (volume between the flat cap's and the sphere's)
{
  const tris = [], N = 24, M = 48;
  const pt = (i, j) => { const th = (Math.PI * i) / N, ph = (2 * Math.PI * (j % M)) / M; return [Math.sin(th) * Math.cos(ph), Math.sin(th) * Math.sin(ph), Math.cos(th)]; };
  for (let i = 0; i < N; i++) for (let j = 0; j < M; j++) {
    if (i < 5) continue;                                             // the cap round the pole, gone
    const a = pt(i, j), b = pt(i + 1, j), c = pt(i + 1, j + 1), d = pt(i, j + 1);
    if (i === N - 1) tris.push([a, b, d]); else tris.push([a, b, c], [a, c, d]);
  }
  const r = check('sphere, cap missing', soup(tris));
  const capPts = [];
  for (let t = 0; t < r.positions.length / 9; t++) if (r.src[t] < 0) for (let k = 0; k < 3; k++) capPts.push(Math.hypot(r.positions[t * 9 + k * 3], r.positions[t * 9 + k * 3 + 1], r.positions[t * 9 + k * 3 + 2]));
  console.log(`     cap points off the sphere (radius 1): min ${Math.min(...capPts).toFixed(3)}; a flat cap's middle would be at ${Math.cos(5 * Math.PI / N).toFixed(3)}`);
}
// 3. an L-shaped hole in a box's top (concave): the cap fills exactly the L
{
  const top = [], g = (x, y) => [x, y, 1];
  const cells = [];
  for (let x = 0; x < 3; x++) for (let y = 0; y < 3; y++) cells.push([x, y]);
  const hole = new Set(['0,0', '1,0', '2,0', '0,1', '0,2']);                // an L of five cells
  for (const [x, y] of cells) if (!hole.has(`${x},${y}`)) top.push(...quad(g(x, y), g(x + 1, y), g(x + 1, y + 1), g(x, y + 1)));
  check('box, L-shaped hole (concave)', soup([...stitchSides(3), ...top]), { vol: 9, capArea: 5 });
}
function stitchSides(s) {
  // a 3 x 3 x 1 box's bottom and sides, its sides' top edges cut at each unit, to meet the top grid
  const tris = [];
  for (let x = 0; x < s; x++) for (let y = 0; y < s; y++) tris.push(...quad([x, y, 0], [x, y + 1, 0], [x + 1, y + 1, 0], [x + 1, y, 0]));
  for (let i = 0; i < s; i++) {
    tris.push(...quad([i, 0, 0], [i + 1, 0, 0], [i + 1, 0, 1], [i, 0, 1]));
    tris.push(...quad([s, i, 0], [s, i + 1, 0], [s, i + 1, 1], [s, i, 1]));
    tris.push(...quad([i + 1, s, 0], [i, s, 0], [i, s, 1], [i + 1, s, 1]));
    tris.push(...quad([0, i + 1, 0], [0, i, 0], [0, i, 1], [0, i + 1, 1]));
  }
  return tris;
}
// 4. two holes touching at a corner (a figure 8) in a 2 x 1 grid top
{
  const top = [], g = (x, y) => [x, y, 1];
  // a 3 x 3 top with cells (0,0) and (1,1) missing: they touch at the point (1,1)
  for (let x = 0; x < 3; x++) for (let y = 0; y < 3; y++) if (!((x === 0 && y === 0) || (x === 1 && y === 1))) top.push(...quad(g(x, y), g(x + 1, y), g(x + 1, y + 1), g(x, y + 1)));
  check('two holes touching at a corner', soup([...stitchSides(3), ...top]), { vol: 9, capArea: 2 });
}
// 5. a cube with a fin on one edge: the fin goes
check('cube + a fin on an edge', soup([...cubeFaces.flat(), [C[4], C[5], [0.5, 0.5, 2]]]), { vol: 1, capArea: 0 });
// 6. a cube with a wall inside, its edges on the cube's: the wall goes
check('cube + a wall inside', soup([...cubeFaces.flat(), ...quad(C[0], C[2], C[6], C[4])]), { vol: 1, capArea: 0 });
// 7. an open thin plate standing alone (1 x 1 x 0.01, its top missing): kept and capped
{
  const t = 0.01, Q = C.map(([x, y, z]) => [x, y, z * t]);
  const faces = [quad(Q[0], Q[3], Q[2], Q[1]), quad(Q[0], Q[1], Q[5], Q[4]), quad(Q[2], Q[3], Q[7], Q[6]), quad(Q[1], Q[2], Q[6], Q[5]), quad(Q[0], Q[4], Q[7], Q[3])];
  check('thin plate, top missing (kept)', soup(faces.flat()), { vol: 0.01, capArea: 1 });
}
if (failures) { console.log(`${failures} shapes failed`); process.exit(1); }
console.log('holeFill ok');
