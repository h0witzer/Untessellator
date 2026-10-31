/**
 * meshDamage.mjs — clean meshes damaged the way exported and downloaded STLs are, with the truth carried along,
 * so the pipeline's tolerance (and js/meshRepair.js) can be measured on parts that export well when clean.
 *
 * Usage:  node test/meshDamage.mjs <out dir> <mesh dir> <stem> [<stem> ...] [--ops a,b] [--seed 1] [--repair]
 *
 * Writes <out>/<op>/<stem>.stl and <stem>.src (Int32: the clean triangle each damaged one came from, -1 for none),
 * and with --repair also <out>/<op>+rep/ (the damaged mesh through repairMesh, src composed back to the clean
 * mesh). <out>/clean/ holds the meshes as they were. Score the exports with test/studioExport.mjs --dump.
 *
 *   holes      10 triangles removed at random
 *   patch      a connected patch of 1% of the triangles removed (one larger hole, on whatever face it lands)
 *   dupes      5% of the triangles written twice, same winding
 *   flip       5% of the triangles reversed
 *   unweld     every corner of every triangle moved independently by up to 2e-7 x the diagonal: no shared
 *              vertex is bit-identical any more (floats written per triangle by a careless exporter)
 *   tjunction  2% of the interior edges split on one side only: a crack along each. The new vertex is the edge's
 *              midpoint, which on a curved face lies inside the surface (chord sag); with --truth <dir> (the
 *              part's .srf files; the true face ids are <mesh dir>/<stem>.fid) only edges inside one analytic
 *              face are split and the vertex is moved onto that face's true surface, as an exporter's would be
 *   needles    as tjunction, with each crack filled by a zero-area triangle (what some exporters write)
 *   collapsed  2% extra triangles with two identical corners (3DBenchy carries 552 of these)
 */

import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { repairMesh } from '../js/meshRepair.js';
import { distNormal, unit, dot, sub } from '../js/knit/geometry.js';

const args = process.argv.slice(2);
const flag = (name, dflt) => { const i = args.indexOf(name); if (i < 0) return dflt; const v = args[i + 1]; args.splice(i, 2); return v; };
const ALL_OPS = ['holes', 'patch', 'dupes', 'flip', 'unweld', 'tjunction', 'needles', 'collapsed'];
const OPS = flag('--ops', ALL_OPS.join(',')).split(',');
const SEED = Number(flag('--seed', '1'));
const TRUTH = flag('--truth', null);
const KIND = ['plane', 'cylinder', 'cone', 'sphere', 'torus'];
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
/** True surface per triangle (null off analytic faces), from <mesh dir>/<stem>.fid and <truth>/<part>.srf. */
function truthSurfaces(stem, P) {
  const fb = readFileSync(join(MESH_DIR, `${stem}.fid`)), F = new Int32Array(copy(fb));
  const sb = readFileSync(join(TRUTH, `${stem.replace(/_[a-z]+$/, '')}.srf`)), srf = new Float64Array(copy(sb));
  const cen = new Map();
  for (let t = 0; t < F.length; t++) {
    const c = cen.get(F[t]) ?? [0, 0, 0, 0];
    for (let k = 0; k < 3; k++) for (let j = 0; j < 3; j++) c[j] += P[t * 9 + k * 3 + j];
    c[3] += 3; cen.set(F[t], c);
  }
  const surf = new Map();
  for (const [id, c] of cen) {
    const s = id >= 0 && id * 12 + 12 <= srf.length ? srf.subarray(id * 12, id * 12 + 12) : null;
    const kind = s && s[0] === s[1] && s[0] <= 4 ? KIND[s[0]] : null;
    surf.set(id, kind ? [kind, truthParams(kind, s.subarray(2), [c[0] / c[3], c[1] / c[3], c[2] / c[3]])] : null);
  }
  return { F, surf };
}
const REPAIR = args.includes('--repair');
if (REPAIR) args.splice(args.indexOf('--repair'), 1);
const [OUT, MESH_DIR, ...STEMS] = args;

const copy = (b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.length);
function rng(seed) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
function writeStl(path, P) {
  const n = P.length / 9;
  const buf = Buffer.alloc(84 + 50 * n);
  buf.writeUInt32LE(n, 80);
  for (let t = 0; t < n; t++) for (let k = 0; k < 9; k++) buf.writeFloatLE(P[t * 9 + k], 84 + 50 * t + 12 + 4 * k);
  writeFileSync(path, buf);
}
const save = (dir, stem, P, src) => {
  mkdirSync(dir, { recursive: true });
  writeStl(join(dir, `${stem}.stl`), P);
  writeFileSync(join(dir, `${stem}.src`), Buffer.from(Int32Array.from(src).buffer));
};

/** Indexed form of a soup (exact weld), and back. */
function indexed(P) {
  const keys = new Map(), V = [], T = [];
  for (let t = 0; t < P.length / 9; t++) {
    const tri = [];
    for (let k = 0; k < 3; k++) {
      const o = t * 9 + k * 3, key = `${P[o]},${P[o + 1]},${P[o + 2]}`;
      let i = keys.get(key);
      if (i === undefined) { i = V.length; keys.set(key, i); V.push([P[o], P[o + 1], P[o + 2]]); }
      tri.push(i);
    }
    T.push(tri);
  }
  return { V, T };
}
const soup = (V, T) => { const P = new Float64Array(T.length * 9); T.forEach((tri, i) => tri.forEach((v, k) => { for (let j = 0; j < 3; j++) P[i * 9 + k * 3 + j] = V[v][j]; })); return P; };

function damage(op, P, rand, truth = null) {
  const { V, T } = indexed(P);
  const n = T.length;
  let src = T.map((_, i) => i);
  const pick = (k) => { const s = new Set(); while (s.size < Math.min(k, n)) s.add(Math.floor(rand() * n)); return s; };
  const edgesOf = () => {
    const m = new Map();
    T.forEach((tri, i) => { for (let k = 0; k < 3; k++) { const u = tri[k], v = tri[(k + 1) % 3], key = Math.min(u, v) * V.length + Math.max(u, v); (m.get(key) ?? m.set(key, []).get(key)).push([i, u, v]); } });
    return m;
  };
  let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const p of V) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], p[k]); hi[k] = Math.max(hi[k], p[k]); }
  const diag = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
  if (op === 'holes' || op === 'patch') {
    let gone;
    if (op === 'holes') gone = pick(10);
    else {
      const E = edgesOf(), nb = T.map(() => []);
      for (const l of E.values()) if (l.length === 2) { nb[l[0][0]].push(l[1][0]); nb[l[1][0]].push(l[0][0]); }
      const want = Math.max(10, Math.round(0.01 * n));
      gone = new Set([Math.floor(rand() * n)]);
      const q = [...gone];
      for (let qi = 0; qi < q.length && gone.size < want; qi++) for (const u of nb[q[qi]]) if (!gone.has(u) && gone.size < want) { gone.add(u); q.push(u); }
    }
    const keep = T.map((_, i) => !gone.has(i));
    return { P: soup(V, T.filter((_, i) => keep[i])), src: src.filter((_, i) => keep[i]) };
  }
  if (op === 'dupes') {
    const extra = [...pick(Math.round(0.05 * n))];
    return { P: soup(V, [...T, ...extra.map((i) => T[i])]), src: [...src, ...extra] };
  }
  if (op === 'flip') {
    const f = pick(Math.round(0.05 * n));
    return { P: soup(V, T.map((tri, i) => (f.has(i) ? [...tri].reverse() : tri))), src };
  }
  if (op === 'unweld') {
    const Q = soup(V, T);
    const amp = 2e-7 * diag;
    for (let i = 0; i < Q.length; i++) Q[i] += (2 * rand() - 1) * amp;
    return { P: Q, src };
  }
  if (op === 'tjunction' || op === 'needles') {
    const E = edgesOf();
    const interior = [...E.values()].filter((l) => l.length === 2);
    const want = Math.round(0.02 * interior.length);
    const used = new Set();
    const T2 = T.map((t) => t.slice()), src2 = [...src], dead = new Set();
    let done = 0;
    for (let tries = 0; done < want && tries < 20 * want; tries++) {
      const l = interior[Math.floor(rand() * interior.length)];
      const [[t1, u, v], [t2]] = l;
      if (used.has(t1) || used.has(t2)) continue;
      let mid = [(V[u][0] + V[v][0]) / 2, (V[u][1] + V[v][1]) / 2, (V[u][2] + V[v][2]) / 2];
      if (truth) {
        // only inside one analytic face, and onto its true surface
        const f = truth.F[t1], s = truth.surf.get(f);
        if (truth.F[t2] !== f || !s) continue;
        const [d, nn] = distNormal(s[0], s[1], mid);
        mid = [mid[0] - d * nn[0], mid[1] - d * nn[1], mid[2] - d * nn[2]];
      }
      used.add(t1); used.add(t2);
      const m = V.length;
      V.push(mid);
      // t1 runs u -> v: split it at m; t2 keeps the long edge
      const c = T[t1].find((x) => x !== u && x !== v);
      dead.add(t1);
      T2.push([u, m, c], [m, v, c]); src2.push(t1, t1);
      if (op === 'needles') { T2.push([m, u, v]); src2.push(-1); }
      done++;
    }
    const keep = T2.map((_, i) => !dead.has(i));
    return { P: soup(V, T2.filter((_, i) => keep[i])), src: src2.filter((_, i) => keep[i]) };
  }
  if (op === 'collapsed') {
    const extra = [...pick(Math.round(0.02 * n))];
    return { P: soup(V, [...T, ...extra.map((i) => [T[i][0], T[i][0], T[i][1]])]), src: [...src, ...extra.map(() => -1)] };
  }
  throw new Error(op);
}

for (const stem of STEMS) {
  const P = Float64Array.from(new Float32Array(copy(readFileSync(join(MESH_DIR, `${stem}.pos`)))));
  save(join(OUT, 'clean'), stem, P, Array.from({ length: P.length / 9 }, (_, i) => i));
  OPS.forEach((op, k) => {
    const rand = rng(SEED * 1000 + k * 7919 + [...stem].reduce((h, c) => (h * 31 + c.charCodeAt(0)) >>> 0, 0));
    const d = damage(op, P, rand, TRUTH && (op === 'tjunction' || op === 'needles') ? truthSurfaces(stem, P) : null);
    save(join(OUT, op), stem, d.P, d.src);
    if (REPAIR) {
      const r = repairMesh(d.P);
      save(join(OUT, `${op}+rep`), stem, r.positions, Array.from(r.src, (s) => (s < 0 ? -1 : d.src[s])));
    }
  });
  console.log(stem);
}
