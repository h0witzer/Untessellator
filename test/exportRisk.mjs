/**
 * exportRisk.mjs — does js/knit/preflight.js see a bad export coming? Its group signals against the true faces,
 * and its part signals against what the knitter actually built from the same groups.
 *
 * Usage:  node test/exportRisk.mjs <mesh dir> <truth dir> <patch dir> <knit dir> [--json out.json] [--shard k/n]
 *         node test/exportRisk.mjs --report a.json [b.json ...]      (the tables, from saved rows)
 *
 *   <mesh dir>   .pos (float32 soup) and .fid (true face per triangle), as test/surfaceScore.mjs reads them
 *   <truth dir>  <part>.srf, the true surface of every face
 *   <patch dir>  <stem>.fid and <stem>.json: painted groups and their fits (surfaceScore --patches-out)
 *   <knit dir>   <stem>.json: test/knitRebuild.mjs's row for those patches (tier, closed, valid, vol_err)
 *
 * A group is BAD when its majority true face is analytic and the group
 *   type      carries another type, or no surface at all
 *   surface   carries the right type more than 1e-3 x diagonal off the true surface (at the face's own vertices)
 *   harm      holds more than 2% of its area on other faces AND off this face's true surface by more than
 *             1e-3 x diagonal (another half of the same cylinder is no harm)
 * Groups whose majority face is freeform are left out of the group scores. A part is BORKED when its export is
 * not fully analytic.
 */

import { readFileSync, writeFileSync, readdirSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { preflight } from '../js/knit/preflight.js';
import { distNormal, unit, dot, sub } from '../js/knit/geometry.js';

const args = process.argv.slice(2);
const flag = (name, dflt) => { const i = args.indexOf(name); if (i < 0) return dflt; const v = args[i + 1]; args.splice(i, 2); return v; };
const KIND = ['plane', 'cylinder', 'cone', 'sphere', 'torus'];
const OFF = 1e-3, HARM_SHARE = 0.02;

if (args[0] === '--report') { report(args.slice(1).flatMap((f) => JSON.parse(readFileSync(f, 'utf8')))); process.exit(0); }

const JSON_OUT = flag('--json', null);
const SHARD = flag('--shard', null)?.split('/').map(Number);
const [MESH_DIR, TRUTH_DIR, PATCH_DIR, KNIT_DIR] = args;
const copy = (b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.length);
const rd = (path, T) => new T(copy(readFileSync(path)));

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

function scorePart(stem) {
  const fits = JSON.parse(readFileSync(join(PATCH_DIR, `${stem}.json`), 'utf8'));
  const part = fits.part ?? stem.replace(/_[a-z]+$/, '');
  const P = Float64Array.from(rd(join(MESH_DIR, `${stem}.pos`), Float32Array));
  const TF = rd(join(MESH_DIR, `${stem}.fid`), Int32Array);
  const labels = rd(join(PATCH_DIR, `${stem}.fid`), Int32Array);
  const truth = rd(join(TRUTH_DIR, `${part}.srf`), Float64Array);
  const knit = JSON.parse(readFileSync(join(KNIT_DIR, `${stem}.json`), 'utf8'));
  const surfaces = new Map(fits.faces.filter((f) => f.params).map((f) => [f.id, [f.type, f.params]]));
  const t0 = performance.now();
  const pf = preflight(P, labels, surfaces);
  const secs = (performance.now() - t0) / 1000;
  const { diag } = pf;
  const n = TF.length;
  const vtx = (t, k) => [P[t * 9 + k * 3], P[t * 9 + k * 3 + 1], P[t * 9 + k * 3 + 2]];
  const area = (t) => {
    const a = vtx(t, 0), b = vtx(t, 1), c = vtx(t, 2);
    const u = sub(b, a), v = sub(c, a);
    return 0.5 * Math.hypot(u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]);
  };
  // true faces
  const faces = new Map();
  for (let t = 0; t < n; t++) {
    const f = faces.get(TF[t]) ?? { tris: [], c: [0, 0, 0], m: 0 };
    f.tris.push(t);
    for (let k = 0; k < 3; k++) { const q = vtx(t, k); f.c[0] += q[0]; f.c[1] += q[1]; f.c[2] += q[2]; f.m++; }
    faces.set(TF[t], f);
  }
  for (const [id, f] of faces) {
    const s = id >= 0 && id * 12 + 12 <= truth.length ? truth.subarray(id * 12, id * 12 + 12) : null;
    f.kind = s && s[0] === s[1] && s[0] <= 4 ? KIND[s[0]] : null;
    if (f.kind) f.params = truthParams(f.kind, s.subarray(2), f.c.map((x) => x / f.m));
  }
  const dTrue = (f, q) => Math.abs(distNormal(f.kind, f.params, q)[0]);
  // groups against the truth
  const byG = new Map();
  for (let t = 0; t < n; t++) if (labels[t] >= 0) (byG.get(labels[t]) ?? byG.set(labels[t], []).get(labels[t])).push(t);
  const rows = [];
  for (const [g, tris] of byG) {
    const pr = pf.groups.get(g);
    if (!pr) continue;
    const fa = new Map();
    let A = 0;
    for (const t of tris) { const a = area(t); A += a; fa.set(TF[t], (fa.get(TF[t]) ?? 0) + a); }
    let M = null, best = -1;
    for (const [f, a] of fa) if (a > best) { best = a; M = f; }
    const mf = faces.get(M);
    const row = { g, tris: tris.length, type: pr.type, dev: pr.dev, seam: pr.seam, seamOff: pr.seamOff, arc: pr.arc,
      flags: pr.flags, maj: mf.kind, share: best / A, bad: null, why: [] };
    if (mf.kind) {
      if (!pr.type) row.why.push('missing');
      else if (pr.type !== mf.kind) row.why.push('type');
      if (pr.type) {
        // the surface at the majority face's own vertices, projected onto the true surface
        const s = surfaces.get(g);
        let off = 0;
        for (const t of tris) {
          if (TF[t] !== M) continue;
          for (let k = 0; k < 3; k++) {
            const q = vtx(t, k);
            const [d, nn] = distNormal(mf.kind, mf.params, q);
            const p = [q[0] - d * nn[0], q[1] - d * nn[1], q[2] - d * nn[2]];
            off = Math.max(off, Math.abs(distNormal(s[0], s[1], p)[0]));
          }
        }
        row.off = off / diag;
        if (pr.type === mf.kind && row.off > OFF) row.why.push('surface');
      }
      let harm = 0;
      for (const t of tris) {
        if (TF[t] === M) continue;
        let far = false;
        for (let k = 0; k < 3; k++) if (dTrue(mf, vtx(t, k)) > OFF * diag) far = true;
        if (far) harm += area(t);
      }
      row.harm = harm / A;
      if (row.harm > HARM_SHARE) row.why.push('harm');
      row.bad = row.why.length > 0;
    }
    rows.push(row);
  }
  return { stem, tier: knit.tier, recreated: !!(knit.closed && knit.valid && knit.vol_err != null && knit.vol_err < 0.01),
    mesh: pf.mesh, part: pf.part, secs: Math.round(secs * 100) / 100, groups: rows };
}

if (MESH_DIR) {
  let stems = readdirSync(PATCH_DIR).filter((f) => f.endsWith('.fid')).map((f) => f.slice(0, -4))
    .filter((s) => existsSync(join(KNIT_DIR, `${s}.json`))).sort();
  if (SHARD) stems = stems.filter((_, i) => i % SHARD[1] === SHARD[0]);
  const out = [];
  for (const stem of stems) {
    try {
      const r = scorePart(stem);
      out.push(r);
      const bad = r.groups.filter((g) => g.bad).length;
      console.log(`${stem} ${r.tier} groups ${r.groups.length} bad ${bad} flagged ${r.part.flagged} worstDev ${r.part.worstDev.toExponential(1)} worstSeam ${r.part.worstSeam.toExponential(1)} open ${r.mesh.openEdges} ${r.secs}s`);
    } catch (err) { console.log(`${stem} ERROR ${String(err?.stack ?? err).slice(0, 200)}`); }
  }
  if (JSON_OUT) writeFileSync(JSON_OUT, JSON.stringify(out));
  report(out);
}

// ── tables ──
function auc(pos, neg) {
  // probability a random positive scores above a random negative (ties half)
  if (!pos.length || !neg.length) return NaN;
  const all = [...pos.map((v) => [v, 1]), ...neg.map((v) => [v, 0])].sort((a, b) => a[0] - b[0]);
  let rank = 0, sumPos = 0;
  for (let i = 0; i < all.length;) {
    let j = i;
    while (j < all.length && all[j][0] === all[i][0]) j++;
    const r = (i + j + 1) / 2;
    for (let k = i; k < j; k++) if (all[k][1]) sumPos += r;
    rank = j; i = j;
  }
  void rank;
  return (sumPos - (pos.length * (pos.length + 1)) / 2) / (pos.length * neg.length);
}
function pct(a, b) { return b ? `${((100 * a) / b).toFixed(1)}%` : '-'; }

function report(all) {
  // a SHATTERED part (most of its groups a triangle or two) would swamp every pooled group figure
  const shattered = (p) => p.groups.filter((g) => g.tris <= 2).length > 0.5 * p.groups.length && p.groups.length > 100;
  const sh = all.filter(shattered);
  if (sh.length) console.log(`
shattered parts (left out of the GROUP tables): ${sh.map((p) => `${p.stem} (${p.groups.length} groups, ${p.tier})`).join(', ')}`);
  groupTables(all.filter((p) => !shattered(p)));
  partTables(all);
}

function groupTables(parts) {
  const groups = parts.flatMap((p) => p.groups.filter((g) => g.bad !== null));
  const bad = groups.filter((g) => g.bad), good = groups.filter((g) => !g.bad);
  console.log(`\n${parts.length} parts, ${groups.length} groups on analytic faces: ${bad.length} bad (` +
    ['type', 'missing', 'surface', 'harm'].map((w) => `${w} ${bad.filter((g) => g.why.includes(w)).length}`).join(', ') + ')');
  const sig = {
    dev: (g) => g.dev ?? Infinity,
    seam: (g) => g.seam ?? 0,
    'max(dev,seam)': (g) => Math.max(g.dev ?? Infinity, g.seam ?? 0),
    'narrow arc': (g) => (g.arc === null ? 0 : 360 - g.arc),
    shard: (g) => (g.flags.includes('shard') ? 1 : 0),
  };
  console.log('\nGROUPS: AUC of each signal for bad vs good');
  for (const [k, f] of Object.entries(sig)) console.log(`  ${k.padEnd(14)} ${auc(bad.map(f), good.map(f)).toFixed(3)}`);
  console.log('\n  threshold (x diag) on max(dev, seam): bad caught / good flagged');
  for (const th of [1e-6, 1e-5, 1e-4, 1e-3]) {
    const f = sig['max(dev,seam)'];
    console.log(`  ${th.toExponential(0).padEnd(8)} ${pct(bad.filter((g) => f(g) > th).length, bad.length).padStart(7)} / ${pct(good.filter((g) => f(g) > th).length, good.length).padStart(7)}`);
  }
  console.log('\n  by flag (the defaults in PREFLIGHT): bad with flag / good with flag');
  for (const fl of ['off its surface', 'seam gap', 'narrow arc', 'shard', 'no surface (B-spline)', 'no surface (facets)']) {
    console.log(`  ${fl.padEnd(22)} ${String(bad.filter((g) => g.flags.includes(fl)).length).padStart(5)} / ${good.filter((g) => g.flags.includes(fl)).length}`);
  }
  console.log('  any flag               ' + `${String(bad.filter((g) => g.flags.length).length).padStart(5)} / ${good.filter((g) => g.flags.length).length}`);
  for (const w of ['type', 'missing', 'surface', 'harm']) {
    const b = bad.filter((g) => g.why.includes(w));
    console.log(`  caught by any flag, ${w.padEnd(8)} ${pct(b.filter((g) => g.flags.length).length, b.length)} of ${b.length}`);
  }

}

function partTables(parts) {
  const borked = parts.filter((p) => p.tier !== 'analytic'), fine = parts.filter((p) => p.tier === 'analytic');
  const worst = (p) => Math.max(p.part.worstDev, p.part.worstSeam);
  console.log(`\nPARTS: ${borked.length} of ${parts.length} exports not fully analytic (` +
    Object.entries(borked.reduce((m, p) => ((m[p.tier] = (m[p.tier] ?? 0) + 1), m), {})).map(([k, v]) => `${k} ${v}`).join(', ') + ')');
  const ps = {
    'worst dev/seam': worst,
    'flagged groups': (p) => p.part.flagged,
    'shards': (p) => p.part.shards ?? 0,
    'open mesh': (p) => (p.mesh.closed ? 0 : 1),
    'bad groups (truth)': (p) => p.groups.filter((g) => g.bad).length,
  };
  for (const [k, f] of Object.entries(ps)) console.log(`  AUC ${k.padEnd(19)} ${auc(borked.map(f), fine.map(f)).toFixed(3)}`);
  console.log('\n  warn when the worst group is over: borked warned / fine warned');
  for (const th of [1e-6, 1e-5, 1e-4, 1e-3]) {
    console.log(`  ${th.toExponential(0).padEnd(8)} ${pct(borked.filter((p) => worst(p) > th || !p.mesh.closed).length, borked.length).padStart(7)} / ${pct(fine.filter((p) => worst(p) > th || !p.mesh.closed).length, fine.length).padStart(7)}`);
  }
  const anyBad = parts.filter((p) => p.groups.some((g) => g.bad));
  console.log(`\n  parts with a bad group (truth): ${anyBad.length}; of them exported fully analytic anyway: ${anyBad.filter((p) => p.tier === 'analytic').length}`);
  console.log(`  borked exports with no bad group (truth): ${borked.filter((p) => !p.groups.some((g) => g.bad)).map((p) => `${p.stem}(${p.tier}${p.mesh.closed ? '' : ', open'})`).join(' ') || 'none'}`);
}
