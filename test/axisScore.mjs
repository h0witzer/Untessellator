/**
 * axisScore.mjs — js/knit/axes.js (shared axes) on fitted patches, judged against the true surfaces.
 *
 * Usage:  node test/axisScore.mjs <mesh dir> <truth dir> <patch dir> [--json out.json] [--limit n]
 *
 *   <mesh dir>   .pos (float32 soup) and .fid (true face per triangle), as test/surfaceScore.mjs reads
 *   <truth dir>  <part>.srf, the true surface of every face id (part = the stem less its _mesher suffix)
 *   <patch dir>  <stem>.fid (a group per triangle) and <stem>.json ({faces: [{id, type, params}]}), as
 *                surfaceScore --patches-out writes them and test/knit.mjs reads them
 *
 * Per pair of ADJACENT groups whose majority true faces are on one axis (axis angle < 1e-6 rad, lines
 * within 1e-6 x diagonal): is our pair exactly coaxial (angle < 1e-9 rad), before and after; and the
 * same for planes on one direction (parallel normals). False shares: pairs made exactly coaxial or
 * parallel whose true faces are not. Per group whose majority true face is analytic and of its own
 * type: the largest distance of that face's vertices from our surface (found4: < 1e-4 x diagonal),
 * before and after, and how many got worse.
 */

import { readFileSync, readdirSync, writeFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { KnitMesh } from '../js/knit/mesh.js';
import { shareAxes } from '../js/knit/axes.js';
import { distNormal, dot, sub, cross, norm, unit, tangentLocus } from '../js/knit/geometry.js';

const args = process.argv.slice(2);
const flag = (name, dflt) => { const i = args.indexOf(name); if (i < 0) return dflt; const v = args[i + 1]; args.splice(i, 2); return v; };
const JSON_OUT = flag('--json', null);
const LIMIT = Number(flag('--limit', '1e9'));
// --opts '{"TANGENT_DEG":0}': shareAxes options (knit/axes.js AXES)
const OPTS = JSON.parse(flag('--opts', '{}'));
const [MESH_DIR, TRUTH_DIR, PATCH_DIR] = args;
const KIND = ['plane', 'cylinder', 'cone', 'sphere', 'torus'];
const AXIAL = new Set(['cylinder', 'cone', 'torus']);
const copy = (b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.length);
const rd = (path, T) => new T(copy(readFileSync(path)));

function truthParams(kind, s) {
  const P = [s[0], s[1], s[2]], A = unit([s[3], s[4], s[5]]);
  if (kind === 'plane') return { origin: P, normal: A };
  if (kind === 'cylinder') return { axisPoint: P, axis: A, radius: s[6] };
  if (kind === 'sphere') return { center: P, radius: s[6] };
  if (kind === 'torus') return { center: P, axis: A, majorRadius: s[6], minorRadius: s[7] };
  return { apex: P, axis: A, halfAngle: s[6] };
}
const lineOf = (kind, p) => [kind === 'cylinder' ? p.axisPoint : kind === 'cone' ? p.apex : p.center, unit(p.axis)];
// atan2 of |a x b| and |a . b|: acos cannot resolve angles below ~1.5e-8 rad
const angle = (a, b) => Math.atan2(norm(cross(a, b)), Math.abs(dot(a, b)));
function lineDist([o1, d1], [o2]) { const v = sub(o2, o1); return norm(sub(v, d1.map((x) => x * dot(v, d1)))); }
/** Relation of two surfaces: 'coaxial' (axial both), 'parallel' (planes), or null. */
function relation(k1, p1, k2, p2, diag, angTol, offTol) {
  if (AXIAL.has(k1) && AXIAL.has(k2)) {
    const l1 = lineOf(k1, p1), l2 = lineOf(k2, p2);
    return angle(l1[1], l2[1]) < angTol && lineDist(l1, l2) < offTol * diag ? 'coaxial' : null;
  }
  if (k1 === 'plane' && k2 === 'plane') {
    const a = angle(unit(p1.normal), unit(p2.normal));
    return a < angTol ? 'parallel' : Math.abs(a - Math.PI / 2) < angTol ? 'perpendicular' : null;
  }
  // an axis and a plane: the end face (normal along the axis) or a face along it (normal across)
  const [ka, pa, pp] = AXIAL.has(k1) && k2 === 'plane' ? [k1, p1, p2] : AXIAL.has(k2) && k1 === 'plane' ? [k2, p2, p1] : [];
  if (ka) {
    const a = angle(lineOf(ka, pa)[1], unit(pp.normal));
    return a < angTol ? 'end' : Math.abs(a - Math.PI / 2) < angTol ? 'along' : null;
  }
  return null;
}

const stems = readdirSync(PATCH_DIR).filter((f) => f.endsWith('.fid')).map((f) => f.slice(0, -4)).sort().slice(0, LIMIT);
const REL = ['coaxial', 'parallel', 'perpendicular', 'end', 'along', 'tangent'];
const zero = () => Object.fromEntries(REL.map((r) => [r, 0]));
const tot = { parts: 0, pairsTrue: zero(), exactBefore: zero(), exactAfter: zero(),
  falseBefore: 0, falseAfter: 0, falseEx: [], worseEx: [], falseKinds: {}, missed: {}, missedKinds: {}, examples: [], groups: 0, found4Before: 0, found4After: 0, found3Before: 0, found3After: 0, worse: 0, better: 0, ms: 0 };
const info = {};
const rows = [];
for (const stem of stems) {
  const part = stem.replace(/_[a-z]+$/, '');
  if (!existsSync(join(TRUTH_DIR, `${part}.srf`)) || !existsSync(join(MESH_DIR, `${stem}.pos`))) continue;
  const positions = Float64Array.from(rd(join(MESH_DIR, `${stem}.pos`), Float32Array));
  const Ftrue = rd(join(MESH_DIR, `${stem}.fid`), Int32Array);
  const labels = rd(join(PATCH_DIR, `${stem}.fid`), Int32Array);
  const truth = rd(join(TRUTH_DIR, `${part}.srf`), Float64Array);
  const fits = JSON.parse(readFileSync(join(PATCH_DIR, `${stem}.json`), 'utf8'));
  const surfaces = new Map((fits.faces ?? []).filter((f) => f.type && f.params).map((f) => [f.id, [f.type, f.params]]));
  const mesh = new KnitMesh(positions, labels);
  const t0 = performance.now();
  const { surfaces: after, info: inf, rejected } = shareAxes(mesh, surfaces, OPTS);
  tot.ms += performance.now() - t0;
  for (const [k, v] of Object.entries(inf)) info[k] = (info[k] ?? 0) + v;
  const diag = mesh.diag;
  // true face per kept triangle; majority true face per label
  const tf = Int32Array.from(mesh.src, (s) => Ftrue[s]);
  const trueSurf = (f) => {
    const s = f >= 0 && f * 12 + 12 <= truth.length ? truth.subarray(f * 12, f * 12 + 12) : null;
    return s && s[0] === s[1] && s[0] <= 4 ? [KIND[s[0]], truthParams(KIND[s[0]], s.subarray(2))] : null;
  };
  const cnt = new Map();
  for (let t = 0; t < mesh.T.length; t++) { const k = `${mesh.F[t]}|${tf[t]}`; cnt.set(k, (cnt.get(k) ?? 0) + 1); }
  const major = new Map();
  for (const [k, n] of cnt) { const [g, f] = k.split('|').map(Number); if (!major.has(g) || major.get(g)[1] < n) major.set(g, [f, n]); }
  // adjacency between labels
  const adj = new Set();
  for (let t = 0; t < mesh.T.length; t++) {
    const [a, b, c] = mesh.T[t];
    for (const [u, v] of [[a, b], [b, c], [c, a]]) {
      const o = mesh.across(t, u, v);
      if (o >= 0 && mesh.F[o] !== mesh.F[t]) adj.add(mesh.F[t] < mesh.F[o] ? `${mesh.F[t]}|${mesh.F[o]}` : `${mesh.F[o]}|${mesh.F[t]}`);
    }
  }
  const row = { stem, coaxialTrue: 0, coaxialExactBefore: 0, coaxialExactAfter: 0, falseAfter: 0, worse: 0 };
  for (const k of adj) {
    const [g1, g2] = k.split('|').map(Number);
    const s1 = surfaces.get(g1), s2 = surfaces.get(g2);
    if (!s1 || !s2 || !major.has(g1) || !major.has(g2)) continue;
    const f1 = major.get(g1)[0], f2 = major.get(g2)[0];
    const t1 = trueSurf(f1), t2 = trueSurf(f2);
    if (!t1 || !t2) continue;
    // tangency is its own relation: the true surfaces touch (contact locus), ours exactly tangent?
    if (tangentLocus(t1, t2, diag)) {
      tot.pairsTrue.tangent++;
      if (tangentLocus(s1, s2, diag)) tot.exactBefore.tangent++;
      if (tangentLocus(after.get(g1), after.get(g2), diag)) tot.exactAfter.tangent++;
    }
    const rel = relation(t1[0], t1[1], t2[0], t2[1], diag, 1e-6, 1e-6);
    const a1 = after.get(g1), a2 = after.get(g2);
    const exactB = relation(s1[0], s1[1], s2[0], s2[1], diag, 1e-9, 1e-9);
    const exactA = relation(a1[0], a1[1], a2[0], a2[1], diag, 1e-9, 1e-9);
    if (rel && (rel === relation(s1[0], s1[1], s2[0], s2[1], diag, 0.05, 0.05))) {
      tot.pairsTrue[rel]++;
      if (exactB === rel) tot.exactBefore[rel]++;
      if (exactA === rel) tot.exactAfter[rel]++;
      if (rel === 'coaxial') {
        row.coaxialTrue++; if (exactB) row.coaxialExactBefore++; if (exactA) row.coaxialExactAfter++;
        if (!exactA) {
          const l1 = lineOf(s1[0], s1[1]), l2 = lineOf(s2[0], s2[1]);
          const why = rejected.has(g1) || rejected.has(g2) ? 'rejected' : angle(l1[1], l2[1]) > Math.PI / 180 ? 'angle>1deg'
            : lineDist(l1, l2) > 1e-3 * diag ? 'offset>1e-3' : 'other';
          tot.missed[why] = (tot.missed[why] ?? 0) + 1;
          if (why === 'rejected') {
            for (const [g, tsf] of [[g1, t1], [g2, t2]]) if (rejected.has(g)) {
              const k = `${surfaces.get(g)[0]}(true ${tsf[0]})`;
              tot.missedKinds[k] = (tot.missedKinds[k] ?? 0) + 1;
            }
          }
          if (tot.examples.length < 40) tot.examples.push({ stem, g1, g2, k: [s1[0], s2[0]], why, deg: +(angle(l1[1], l2[1]) * 180 / Math.PI).toPrecision(3), off: +(lineDist(l1, l2) / diag).toPrecision(3) });
        }
      }
    } else if (!rel) {
      if (exactB) tot.falseBefore++;
      if (exactA) {
        tot.falseAfter++; row.falseAfter++;
        const l1 = AXIAL.has(t1[0]) ? lineOf(t1[0], t1[1]) : null, l2 = AXIAL.has(t2[0]) ? lineOf(t2[0], t2[1]) : null;
        const typed = a1[0] === t1[0] && a2[0] === t2[0] ? 'right types' : 'wrong type';
        const tr = relation(t1[0], t1[1], t2[0], t2[1], diag, 0.02, 0.02);   // close to the relation in truth?
        const fk = `${exactA}/${typed}/${tr ? 'truth within 1deg' : 'truth far'}`;
        tot.falseKinds[fk] = (tot.falseKinds[fk] ?? 0) + 1;
        if (tot.falseEx.length < 15 && typed === 'right types') tot.falseEx.push({ stem, g1, g2, rel: exactA, ours: [a1[0], a2[0]], truth: [t1[0], t2[0]],
          trueDeg: l1 && l2 ? +(angle(l1[1], l2[1]) * 180 / Math.PI).toPrecision(3) : null, trueOff: l1 && l2 ? +(lineDist(l1, l2) / diag).toPrecision(3) : null,
          offPerpDeg: t1[0] === 'plane' && t2[0] === 'plane' ? +((angle(unit(t1[1].normal), unit(t2[1].normal)) - Math.PI / 2) * 180 / Math.PI).toPrecision(3) : null });
      }
    }
  }
  // part-wide design intent: equal sizes (any two groups, not only neighbours) and planes through an axis
  const VAL = { cylinder: (q) => q.radius, cone: (q) => q.halfAngle, torus: (q) => q.minorRadius };
  const glist = [...surfaces.keys()].filter((g) => major.has(g));
  for (let i = 0; i < glist.length; i++) for (let j = i + 1; j < glist.length; j++) {
    const g1 = glist[i], g2 = glist[j];
    const f1 = major.get(g1)[0], f2 = major.get(g2)[0];
    if (f1 === f2) continue;
    const t1 = trueSurf(f1), t2 = trueSurf(f2);
    if (!t1 || !t2) continue;
    const s1 = surfaces.get(g1), s2 = surfaces.get(g2), a1 = after.get(g1), a2 = after.get(g2);
    if (t1[0] === t2[0] && VAL[t1[0]] && s1[0] === t1[0] && s2[0] === t2[0] && a1[0] === t1[0] && a2[0] === t2[0]) {
      const key = `equal_${t1[0]}`, V = VAL[t1[0]];
      const tv1 = V(t1[1]), tv2 = V(t2[1]), tol = t1[0] === 'cone' ? 1e-7 : 1e-6 * Math.max(Math.abs(tv1), 1e-300);
      const same = (x, y, rel) => Math.abs(x - y) <= rel * Math.max(Math.abs(x), 1e-300);
      if (Math.abs(tv1 - tv2) <= tol) {
        tot.pairsTrue[key] = (tot.pairsTrue[key] ?? 0) + 1;
        if (same(V(s1[1]), V(s2[1]), 1e-12)) tot.exactBefore[key] = (tot.exactBefore[key] ?? 0) + 1;
        if (same(V(a1[1]), V(a2[1]), 1e-12)) tot.exactAfter[key] = (tot.exactAfter[key] ?? 0) + 1;
      } else if (!same(tv1, tv2, 1e-4) && same(V(a1[1]), V(a2[1]), 1e-12) && !same(V(s1[1]), V(s2[1]), 1e-12)) {
        tot.falseValue = (tot.falseValue ?? 0) + 1;
      }
    }
    // a plane through an axis
    const pa = t1[0] === 'plane' && AXIAL.has(t2[0]) ? [g1, g2, t1, t2] : t2[0] === 'plane' && AXIAL.has(t1[0]) ? [g2, g1, t2, t1] : null;
    if (pa) {
      const [gp, ga, tp, ta] = pa;
      const through = (P, A) => { const n = unit(P[1].normal), [o, d] = lineOf(A[0], A[1]); return Math.abs(dot(n, d)) < 1e-9 && Math.abs(dot(sub(o, P[1].origin), n)) <= 1e-9 * diag; };
      if (surfaces.get(gp)[0] === 'plane' && AXIAL.has(surfaces.get(ga)[0]) && after.get(gp)[0] === 'plane' && AXIAL.has(after.get(ga)[0])) {
        const n = unit(tp[1].normal), [o, d] = lineOf(ta[0], ta[1]);
        if (Math.abs(dot(n, d)) < 1e-6 && Math.abs(dot(sub(o, tp[1].origin), n)) <= 1e-6 * diag) {
          tot.pairsTrue.through = (tot.pairsTrue.through ?? 0) + 1;
          if (through(surfaces.get(gp), surfaces.get(ga))) tot.exactBefore.through = (tot.exactBefore.through ?? 0) + 1;
          if (through(after.get(gp), after.get(ga))) tot.exactAfter.through = (tot.exactAfter.through ?? 0) + 1;
        }
      }
    }
  }
  // recovery of each group's true face
  const vertsOfFace = new Map();
  for (let t = 0; t < mesh.T.length; t++) {
    let s = vertsOfFace.get(tf[t]);
    if (!s) vertsOfFace.set(tf[t], (s = new Set()));
    for (const v of mesh.T[t]) s.add(v);
  }
  for (const [g, s] of surfaces) {
    if (!major.has(g)) continue;
    const ts = trueSurf(major.get(g)[0]);
    if (!ts || ts[0] !== s[0]) continue;
    const V = [...vertsOfFace.get(major.get(g)[0])].map((v) => mesh.V[v]);
    const dev = (sp) => V.reduce((m, x) => Math.max(m, Math.abs(distNormal(sp[0], sp[1], x)[0])), 0) / diag;
    const b = dev(s), a = dev(after.get(g));
    tot.groups++;
    if (b < 1e-4) tot.found4Before++;
    if (a < 1e-4) tot.found4After++;
    if (b < 1e-3) tot.found3Before++;
    if (a < 1e-3) tot.found3After++;
    if (a > 2 * b + 1e-5) {
      tot.worse++; row.worse++;
      if (tot.worseEx.length < 60) tot.worseEx.push({ stem, g, kind: s[0], after: after.get(g)[0], before: +b.toPrecision(3), afterDev: +a.toPrecision(3), rejected: rejected.has(g) });
    }
    if (b > 2 * a + 1e-5) tot.better++;
  }
  tot.parts++;
  rows.push(row);
}
const pct = (a, b) => (b ? `${((100 * a) / b).toFixed(1)}%` : '-');
console.log(`${PATCH_DIR}: ${tot.parts} parts, shareAxes ${(tot.ms / Math.max(tot.parts, 1)).toFixed(0)} ms/part`);
for (const rel of [...REL, 'equal_cylinder', 'equal_cone', 'equal_torus', 'through']) {
  console.log(`  adjacent ${rel} in truth: ${tot.pairsTrue[rel]}   exact before ${pct(tot.exactBefore[rel] ?? 0, tot.pairsTrue[rel] ?? 0)}   after ${pct(tot.exactAfter[rel] ?? 0, tot.pairsTrue[rel] ?? 0)}`);
}
console.log(`  false shares (exact, not in truth): before ${tot.falseBefore}  after ${tot.falseAfter}   equal sizes made where the truth differs: ${tot.falseValue ?? 0}`);
console.log('  coaxial pairs left inexact, by reason:', JSON.stringify(tot.missed), 'rejected kinds:', JSON.stringify(tot.missedKinds));
for (const e of tot.examples) console.log('    ', JSON.stringify(e));
console.log('  false shares after, by kind:', JSON.stringify(tot.falseKinds));
for (const e of tot.worseEx) console.log('   worse:', JSON.stringify(e));
for (const e of tot.falseEx) console.log('   false:', JSON.stringify(e));
console.log(`  groups on their true type: ${tot.groups}   found4 ${pct(tot.found4Before, tot.groups)} -> ${pct(tot.found4After, tot.groups)}   found3 ${pct(tot.found3Before, tot.groups)} -> ${pct(tot.found3After, tot.groups)}   worse (x2 and +1e-5): ${tot.worse}  better: ${tot.better}`);
console.log('  ', JSON.stringify(info));
if (JSON_OUT) writeFileSync(JSON_OUT, JSON.stringify({ tot, info, rows }));
