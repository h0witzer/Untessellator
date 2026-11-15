// test/bodies.mjs — js/bodies.js on boxes whose bodies are known: apart, touching (a conformal shared face, so a
// wall of opposite-wound pairs), overlapping, hollow (a void wound inward), a box in a void, a box embedded wound
// outward, the whole file inside out, and triangles flipped at random. Then any meshes given on the command line,
// with their body count, shells, voids and time.
//   node test/bodies.mjs [file.stl ...]
import { readFileSync } from 'node:fs';
import { findBodies, pairEdges } from '../js/bodies.js';

function box(lo, hi, n = 4, inward = false) {
  const T = [];
  for (let ax = 0; ax < 3; ax++) for (const side of [0, 1]) {
    const [u, v] = [0, 1, 2].filter((a) => a !== ax);
    const P = (a, b) => { const p = [0, 0, 0]; p[ax] = side ? hi[ax] : lo[ax]; p[u] = lo[u] + ((hi[u] - lo[u]) * a) / n; p[v] = lo[v] + ((hi[v] - lo[v]) * b) / n; return p; };
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const q = [P(i, j), P(i + 1, j), P(i + 1, j + 1), P(i, j + 1)];
      for (let t of [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]) {
        const e1 = t[1].map((x, k) => x - t[0][k]), e2 = t[2].map((x, k) => x - t[0][k]);
        const nrm = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
        if ((nrm[ax] * (side ? 1 : -1) < 0) !== inward) t = [t[0], t[2], t[1]];
        T.push(t);
      }
    }
  }
  return T;
}

/** Exact weld of a soup of [[x,y,z] x3] triangles. */
function indexed(tris) {
  const ids = new Map(), V = [], idx = new Int32Array(tris.length * 3);
  tris.forEach((t, i) => t.forEach((p, k) => {
    const key = p.join(',');
    let id = ids.get(key);
    if (id === undefined) { id = ids.size; ids.set(key, id); V.push(...p); }
    idx[i * 3 + k] = id;
  }));
  return { idx, V: Float64Array.from(V) };
}

let failures = 0;
const check = (name, ok, detail) => { if (!ok) failures++; console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}${ok ? '' : `: ${detail}`}`); };
const round = (x) => Math.round(x * 1000) / 1000;

function expect(name, tris, want) {
  const { idx, V } = indexed(tris);
  const r = findBodies(idx, V);
  const got = r.bodies.map((b) => ({ volume: round(b.volume), voids: b.voids, closed: b.closed }));
  check(name, JSON.stringify(got) === JSON.stringify(want), `got ${JSON.stringify(got)}, want ${JSON.stringify(want)}`);
  // no pair crosses bodies, and every body is closed through its pairs
  let crossing = 0;
  for (let t = 0; t < r.bodyOf.length; t++) for (let k = 0; k < 3; k++) {
    const m = r.across[t * 3 + k];
    if (m >= 0 && r.bodyOf[m] !== r.bodyOf[t]) crossing++;
  }
  check(`${name}: no pair crosses bodies`, crossing === 0, `${crossing} pairs cross`);
  return r;
}

const flipAll = (T) => T.map((t) => [t[0], t[2], t[1]]);
const B = (lo, hi, inward) => box(lo, hi, 4, inward);

expect('apart', [...B([0, 0, 0], [20, 20, 20]), ...B([30, 0, 0], [50, 20, 20])],
  [{ volume: 8000, voids: 0, closed: true }, { volume: 8000, voids: 0, closed: true }]);
expect('touching (shared conformal face)', [...B([0, 0, 0], [20, 20, 20]), ...B([20, 0, 0], [40, 20, 20])],
  [{ volume: 8000, voids: 0, closed: true }, { volume: 8000, voids: 0, closed: true }]);
expect('overlapping', [...B([0, 0, 0], [20, 20, 20]), ...B([10, 5, 5], [30, 15, 15])],
  [{ volume: 8000, voids: 0, closed: true }, { volume: 2000, voids: 0, closed: true }]);
const hollow = [...B([0, 0, 0], [40, 40, 40]), ...B([10, 10, 10], [30, 30, 30], true)];
expect('hollow (void wound inward)', hollow, [{ volume: 56000, voids: 1, closed: true }]);
expect('box in a void', [...hollow, ...B([14, 14, 14], [26, 26, 26])],
  [{ volume: 56000, voids: 1, closed: true }, { volume: 1728, voids: 0, closed: true }]);
expect('embedded box wound outward', [...B([0, 0, 0], [40, 40, 40]), ...B([10, 10, 10], [30, 30, 30])],
  [{ volume: 64000, voids: 0, closed: true }, { volume: 8000, voids: 0, closed: true }]);
const inverted = expect('hollow, whole file inside out', flipAll(hollow), [{ volume: 56000, voids: 1, closed: true }]);
check('inside-out file: every triangle flipped back', inverted.flip.every((f) => f === 1), 'some kept');
let seed = 7;
const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
const noisy = hollow.map((t) => (rand() < 0.2 ? [t[0], t[2], t[1]] : t));
const fixed = expect('hollow, 20% of triangles flipped', noisy, [{ volume: 56000, voids: 1, closed: true }]);
check('noisy: exactly the flipped ones reversed', noisy.every((t, i) => (fixed.flip[i] === 1) === (t !== hollow[i])), 'mismatch');
const open = B([0, 0, 0], [20, 20, 20]).slice(2);
expect('an open box and a closed one', [...open, ...B([30, 0, 0], [50, 20, 20])],
  [{ volume: 8000, voids: 0, closed: true }, { volume: 0, voids: 0, closed: false }]);

// pairEdges on a manifold edge pairs the two; on the touching wall, each box's wall pairs inside its box
{
  const { idx, V } = indexed([...B([0, 0, 0], [20, 20, 20]), ...B([20, 0, 0], [40, 20, 20])]);
  const across = pairEdges(idx, V);
  const half = idx.length / 6;
  let bad = 0;
  for (let t = 0; t < idx.length / 3; t++) for (let k = 0; k < 3; k++) if (across[t * 3 + k] < 0 || (across[t * 3 + k] < half) !== (t < half)) bad++;
  check('touching: every edge pairs within its box', bad === 0, `${bad} edges`);
}

// meshes from the command line
for (const file of process.argv.slice(2)) {
  const buf = readFileSync(file);
  const n = buf.readUInt32LE(80);
  const tris = [];
  for (let t = 0; t < n; t++) {
    const o = 84 + t * 50 + 12;
    tris.push([0, 1, 2].map((k) => [0, 1, 2].map((a) => buf.readFloatLE(o + k * 12 + a * 4))));
  }
  const { idx, V } = indexed(tris);
  const t0 = performance.now();
  const r = findBodies(idx, V);
  const ms = performance.now() - t0;
  const name = file.split(/[\\/]/).pop();
  console.log(`${name}: ${n} triangles, ${r.shells.length} shells, ${r.bodies.length} bodies `
    + `[${r.bodies.map((b) => `${b.closed ? '' : 'open '}${round(b.volume)}${b.voids ? ` (${b.voids} void${b.voids > 1 ? 's' : ''})` : ''}`).join(', ')}], `
    + `${r.flip.reduce((s, f) => s + f, 0)} flipped, ${ms.toFixed(0)} ms`);
}
if (failures) { console.log(`${failures} failed`); process.exit(1); }
console.log('all passed');
