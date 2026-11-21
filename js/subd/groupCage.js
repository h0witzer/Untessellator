/**
 * groupCage.js — a quad cage for a mesh painted in groups (a part's faces), laid out from the groups' borders.
 *
 * Borders are the mesh edges between two groups; corners are the vertices where three or more groups meet; a
 * border runs corner to corner as a chain, or round a closed ring with no corner. At a corner where one group's
 * border runs on nearly straight (a T-point: interior angle over flatAngle), that group is "flat" there: the
 * point is no corner of its layout, its border curve runs smoothly through, and the other groups' curves follow it
 * (the `ends` returned below: the border curves' end conditions for groupNurbs.js).
 *
 * Every chain gets an even number of cage edges, from its length and the target quad size, raised where a group's
 * layout needs more. Layouts, by the group's border loops and corners (k: corners where it is not flat):
 * - one loop, k >= 3: split from a centre into k quad grids, one per corner (centre: a star of valence k);
 * - one loop, k = 2: two corner grids and two middle grids round two valence-3 stars;
 * - one loop, k = 1: the point opposite the corner is made a corner of this group alone, then as k = 2;
 * - one loop, k = 0: an O-grid (a core grid with a ring of four grids; four valence-3 stars);
 * - two loops, no corners: a ring grid, no stars.
 * Points off the border are put on the group's triangles (closest point); grid insides are Coons patches of their
 * sides. The fit (fitCage.js) then moves every point.
 *
 * Returns { vertices: [[x, y, z]...], groups: [{ label, faces: [[a, b, c, d]...], triangles: [t...], kind }],
 * ends: Map "corner,next" -> { W0, Wm1 } (stencils [[vertex, weight]...]), counts } or { error }. Quads are wound
 * like the mesh's triangles (outward when they are).
 */
import { projector } from './isotropic.js';
import { envelope } from './envelopeCholesky.js';

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const lerp = (a, b, w) => [a[0] + w * (b[0] - a[0]), a[1] + w * (b[1] - a[1]), a[2] + w * (b[2] - a[2])];

export function groupCage(vertices, triangles, labels, { quads = 600, flatAngle = 150 } = {}) {
  const nv = vertices.length;
  const half = new Map();
  triangles.forEach((t, f) => { for (let k = 0; k < 3; k++) half.set(t[k] * nv + t[(k + 1) % 3], f); });
  const labelsAt = Array.from({ length: nv }, () => new Set());
  triangles.forEach((t, f) => t.forEach((v) => labelsAt[v].add(labels[f])));
  const isCorner = (v) => labelsAt[v].size >= 3;
  let totalArea = 0;
  const triArea = triangles.map(([a, b, c]) => {
    const A = Math.hypot(...cross(sub(vertices[b], vertices[a]), sub(vertices[c], vertices[a]))) / 2;
    totalArea += A;
    return A;
  });
  const h = Math.sqrt(totalArea / quads);

  // groups and their border loops (group on the left)
  const byLabel = new Map();
  triangles.forEach((t, f) => { if (!byLabel.has(labels[f])) byLabel.set(labels[f], []); byLabel.get(labels[f]).push(f); });
  const groups = [], problems = [];
  // the border half-edge after a -> b (in triangle t): turn round b through the group's own triangles
  const nextBorder = (t, b, label) => {
    let tri = t, c = triangles[t][(triangles[t].indexOf(b) + 1) % 3];
    for (let guard = 0; guard < 1000; guard++) {
      const u = half.get(c * nv + b);
      if (u === undefined || labels[u] !== label) return [b, c, tri];
      const q = triangles[u];
      tri = u; c = q[(q.indexOf(b) + 1) % 3];
    }
    return null;
  };
  for (const [label, tris] of byLabel) {
    const border = [];
    for (const f of tris) {
      const t = triangles[f];
      for (let k = 0; k < 3; k++) {
        const a = t[k], b = t[(k + 1) % 3], g = half.get(b * nv + a);
        if (g === undefined) return { error: `the mesh is open at edge ${a}-${b}` };
        if (labels[g] !== label) border.push([a, b, f]);
      }
    }
    if (!border.length) { problems.push(`group ${label} has no border (a whole closed body)`); continue; }
    const seen = new Set(), loops = [];
    for (const [a0, b0, f0] of border) {
      if (seen.has(a0 * nv + b0)) continue;
      const loop = [];
      let [a, b, f] = [a0, b0, f0];
      do { seen.add(a * nv + b); loop.push(a); [a, b, f] = nextBorder(f, b, label); } while (a !== a0 || b !== b0);
      loops.push(loop);
    }
    const visits = new Map();
    for (const loop of loops) for (const v of loop) visits.set(v, (visits.get(v) ?? 0) + 1);
    if ([...visits.values()].some((c) => c > 1)) { problems.push(`group ${label}: its border touches itself at a point`); continue; }
    // genus: V - E + F = 2 - loops for a sphere with holes
    const vs = new Set(), es = new Set();
    for (const f of tris) { const t = triangles[f]; for (let k = 0; k < 3; k++) { vs.add(t[k]); const a = t[k], b = t[(k + 1) % 3]; es.add(a < b ? a * nv + b : b * nv + a); } }
    if (vs.size - es.size + tris.length !== 2 - loops.length) { problems.push(`group ${label} has a handle (not a disk with holes)`); continue; }
    groups.push({ label, tris, loops });
  }

  // chains, shared by the two groups beside them
  const chains = [], chainKey = new Map();
  const polyLength = (verts, ring) => {
    const len = [0];
    const n = ring ? verts.length + 1 : verts.length;
    for (let i = 1; i < n; i++) len.push(len[i - 1] + Math.hypot(...sub(vertices[verts[i % verts.length]], vertices[verts[i - 1]])));
    return len;
  };
  const register = (verts, ring) => {
    if (ring) {
      const key = 'r' + [...verts].sort((a, b) => a - b).join(',');
      if (chainKey.has(key)) {
        const c = chainKey.get(key), S = chains[c].verts, i = S.indexOf(verts[0]);
        return { c, rev: S[(i + 1) % S.length] !== verts[1] };
      }
      chainKey.set(key, chains.length);
      chains.push({ verts, ring: true, len: polyLength(verts, true) });
      return { c: chains.length - 1, rev: false };
    }
    const k = verts.join(','), kr = [...verts].reverse().join(',');
    if (chainKey.has(k)) return { c: chainKey.get(k), rev: false };
    if (chainKey.has(kr)) return { c: chainKey.get(kr), rev: true };
    chainKey.set(k, chains.length);
    chains.push({ verts, ring: false, len: polyLength(verts, false) });
    return { c: chains.length - 1, rev: false };
  };
  for (const g of groups) {
    g.loopChains = g.loops.map((loop) => {
      const ci = loop.map((v, i) => (isCorner(v) ? i : -1)).filter((i) => i >= 0);
      if (!ci.length) return [register(loop, true)];
      const rot = loop.slice(ci[0]).concat(loop.slice(0, ci[0]));
      const at = ci.map((i) => i - ci[0]);
      return at.map((s, j) => register(rot.slice(s, (at[j + 1] ?? rot.length) + 1).concat(j + 1 === at.length ? [rot[0]] : []), false));
    });
  }
  const totalLen = (c) => chains[c].len[chains[c].len.length - 1];
  // a point at arc length s along a chain as a group walks it
  const along = (e, s) => {
    const ch = chains[e.c], L = totalLen(e.c), len = ch.len, verts = ch.ring ? [...ch.verts, ch.verts[0]] : ch.verts;
    const t = e.rev ? L - s : s;
    let i = 1;
    while (i < len.length - 1 && len[i] < t) i++;
    return lerp(vertices[verts[i - 1]], vertices[verts[i]], (t - len[i - 1]) / ((len[i] - len[i - 1]) || 1));
  };
  const startOf = (e) => { const v = chains[e.c].verts; return e.rev ? v[v.length - 1] : v[0]; };

  // flat groups at corners: the largest interior angle over flatAngle, one per corner
  const normalAt = (g, p) => {
    const n = [0, 0, 0];
    for (const f of g.tris) {
      const t = triangles[f];
      if (!t.includes(p)) continue;
      const c = cross(sub(vertices[t[1]], vertices[t[0]]), sub(vertices[t[2]], vertices[t[0]]));
      n[0] += c[0]; n[1] += c[1]; n[2] += c[2];
    }
    return n;
  };
  const flatAt = new Map();                        // corner vertex -> [group, angle]
  groups.forEach((g, gi) => g.loopChains.forEach((entries) => {
    if (entries.length === 1 && chains[entries[0].c].ring) return;
    entries.forEach((out, j) => {
      const inc = entries[(j + entries.length - 1) % entries.length], p = startOf(out), P = vertices[p];
      const dOut = sub(along(out, Math.min(0.2 * totalLen(out.c), 2 * h)), P);
      const dIn = sub(along(inc, totalLen(inc.c) - Math.min(0.2 * totalLen(inc.c), 2 * h)), P);
      let angle = Math.atan2(dot(normalAt(g, p), cross(dOut, dIn)) / (Math.hypot(...normalAt(g, p)) || 1), dot(dOut, dIn)) * 180 / Math.PI;
      if (angle < 0) angle += 360;
      if (angle > flatAngle && (!flatAt.has(p) || flatAt.get(p)[1] < angle)) flatAt.set(p, [gi, angle]);
    });
  }));
  // each loop as sides between the group's own corners (none on a ring or an all-flat loop)
  for (const [gi, g] of groups.entries()) {
    g.sides = g.loopChains.map((entries) => {
      const own = entries.map((e, j) => (chains[e.c].ring || flatAt.get(startOf(e))?.[0] === gi ? -1 : j)).filter((j) => j >= 0);
      if (!own.length) return { corners: 0, sides: [entries] };
      const rot = entries.slice(own[0]).concat(entries.slice(0, own[0])), at = own.map((j) => j - own[0]);
      return { corners: own.length, sides: at.map((s, i) => rot.slice(s, at[i + 1] ?? rot.length)) };
    });
    const ks = g.sides.map((l) => l.corners);
    if (g.sides.length === 1) g.kind = ks[0] >= 3 ? (ks[0] % 2 ? 'split' : 'cutsplit') : ks[0] === 2 ? 'bigon' : ks[0] === 1 ? 'teardrop' : 'ogrid';
    else if (g.sides.length === 2 && ks[0] === 0 && ks[1] === 0) g.kind = 'ring';
    else if (g.sides.length === 2 && Math.min(...ks) === 0 && Math.max(...ks) >= 2) { g.kind = 'holed'; if (ks[0] === 0) g.sides.reverse(); }
    else problems.push(`group ${g.label}: ${g.sides.length} border loops with ${ks.join('+')} corners`);
  }
  if (problems.length) return { error: problems.join('; '), problems };

  // edge counts per chain: even, from length, raised until every group's layout fits
  const count = chains.map((ch, c) => Math.max(ch.ring ? 8 : 2, 2 * Math.round(totalLen(c) / (2 * h))));
  const sideLen = (side) => side.reduce((s, e) => s + count[e.c], 0);
  const splitParts = (L) => {                       // x with x[i] + x[i+2] = L[i], balanced where free; null if none
    const k = L.length, x = new Array(k).fill(0), starts = k % 2 ? [0] : [0, 1];
    let closure = 0;
    for (const s of starts) {
      const cyc = [s];
      for (let i = (s + 2) % k; i !== s; i = (i + 2) % k) cyc.push(i);
      const m = cyc.length, c = [0], sg = [1];      // x[cyc[j]] = c[j] + sg[j] t
      for (let j = 1; j <= m; j++) { c.push(L[cyc[j - 1]] - c[j - 1]); sg.push(-sg[j - 1]); }
      let t;
      if (m % 2) t = c[m] / 2;
      else {
        closure += Math.abs(c[m]);
        let lo = -Infinity, hi = Infinity;          // c[j] + sg[j] t >= 1 for every j
        for (let j = 0; j < m; j++) { if (sg[j] > 0) lo = Math.max(lo, 1 - c[j]); else hi = Math.min(hi, c[j] - 1); }
        t = Math.round(lo <= hi ? (lo + hi) / 2 : lo);
      }
      cyc.forEach((i, j) => { x[i] = c[j] + sg[j] * t; });
    }
    return { x, closure };
  };
  // a k-split fits sides S when its parts are whole and its centre (a star unless k = 4) is two edges or more from
  // the border: one edge from it works, but a sharp corner's grid then shrinks to one skewed quad (the Star's tips)
  const short = (S) => {                           // how far a k-split of sides S is from fitting (0: fits)
    if (S.some((v) => v < 1)) return 1000;
    const { x, closure } = splitParts(S), need = S.length === 4 ? 1 : 2;
    return closure + (S.reduce((s, v) => s + v, 0) % 2) + x.reduce((s, v) => s + Math.max(0, need - v), 0);
  };
  const fits = (S) => short(S) === 0;
  const cands = (n) => [...new Set([Math.ceil(n / 2), Math.floor(n / 2), Math.round(n / 3), Math.round(2 * n / 3), 1, n - 1])].filter((v) => v >= 1 && v <= n - 1);
  // an even-cornered face: a cut across corner j (from b edges before it to a edges after it, c edges long) leaves a
  // triangle and a face with one corner more, both odd, so no side has to equal another
  const planCache = new Map();
  const cutPlan = (L) => {
    const key = 'c' + L.join(',');
    if (planCache.has(key)) return planCache.get(key);
    const k = L.length;
    let best = null;
    for (let j = 0; j < k; j++) {
      const Lp = L[(j + k - 1) % k], Ln = L[j];
      for (const a of cands(Ln)) for (const b of cands(Lp)) {
        const target = Math.max(1, Math.round(0.7 * (a + b)));
        let found = null;
        for (let dc = 0; dc <= a + b + 8 && !found; dc++) for (const c of dc ? [target + dc, target - dc] : [target]) {
          if (c < 1) continue;
          const poly = [Lp - b, c, Ln - a];
          for (let i = 1; i <= k - 2; i++) poly.push(L[(j + i) % k]);
          if (fits([a, c, b]) && fits(poly)) { found = c; break; }
        }
        if (found === null) continue;
        const score = Math.abs(a - Ln / 2) + Math.abs(b - Lp / 2) + Math.abs(found - target);
        if (!best || score < best.score) best = { j, a, b, c: found, score };
      }
    }
    if (!best) {                                    // how near the nearest cut comes, for raising counts
      let near = Infinity;
      for (let j = 0; j < k; j++) {
        const Lp = L[(j + k - 1) % k], Ln = L[j];
        for (const a of cands(Ln)) for (const b of cands(Lp)) for (let c = 1; c <= a + b + 8; c++) {
          const poly = [Lp - b, c, Ln - a];
          for (let i = 1; i <= k - 2; i++) poly.push(L[(j + i) % k]);
          near = Math.min(near, short([a, c, b]) + short(poly));
        }
      }
      best = { near: Number.isFinite(near) ? near : 1000 };
    }
    planCache.set(key, best);
    return best;
  };
  // a face with a hole and k >= 2 corners round its outer border: a cut from each outer side's middle to the hole
  // makes k five-sided regions, one round each corner (sides: half side, cut, hole arc, cut, half side)
  const holePlan = (L, Lh) => {
    const key = 'h' + L.join(',') + '|' + Lh;
    if (planCache.has(key)) return planCache.get(key);
    const k = L.length, a = L.map((n) => Math.ceil(n / 2)), b = L.map((n, i) => n - a[i]);
    const ext = L.map((_, i) => a[i] + b[(i + k - 1) % k]), tot = ext.reduce((s, x) => s + x, 0);
    const arc = ext.map((e) => Math.max(1, Math.round(Lh * e / tot)));
    let diff = Lh - arc.reduce((s, x) => s + x, 0);
    for (let i = 0; diff !== 0 && i < 100 * k; i++) { const j = i % k; if (diff > 0) { arc[j]++; diff--; } else if (arc[j] > 1) { arc[j]--; diff++; } }
    const c = L.map(() => Math.max(1, Math.round(tot / (2 * k))));
    const pent = (i) => [a[i], c[i], arc[i], c[(i + k - 1) % k], b[(i + k - 1) % k]];
    // whole parts: every region's perimeter even (shift hole split points; the last one follows from the total)
    for (let i = 0; i < k - 1; i++) if (pent(i).reduce((s, x) => s + x, 0) % 2) { if (arc[i] > 1) { arc[i]--; arc[i + 1]++; } else { arc[i]++; arc[i + 1]--; } }
    let plan = null, near = 1000;
    if (!diff && arc.every((x) => x >= 1)) {
      for (let it = 0; it < 200; it++) {
        const bad = [...Array(k).keys()].filter((i) => !fits(pent(i)));
        if (!bad.length) { plan = { a, b, arc, c }; break; }
        near = Math.min(near, bad.reduce((s, i) => s + short(pent(i)), 0));
        for (const i of bad) { c[i] += 2; c[(i + k - 1) % k] += 2; }   // by two: every perimeter keeps its parity
      }
    }
    plan ??= { near };
    planCache.set(key, plan);
    return plan;
  };
  const deficit = (g) => {                          // 0 when the group's layout fits its sides' counts
    const loop = g.sides[0], L = loop.sides.map(sideLen);
    if (g.kind === 'split') return short(L);
    if (g.kind === 'cutsplit') return cutPlan(L).near ?? 0;
    if (g.kind === 'bigon') return L.reduce((s, v) => s + Math.max(0, 6 - v), 0);
    if (g.kind === 'teardrop') return Math.max(0, 12 - L[0]) + (L[0] % 4 ? 2 : 0);
    if (g.kind === 'ogrid') return Math.max(0, 8 - L[0]);
    if (g.kind === 'holed') return holePlan(L, sideLen(g.sides[1].sides[0])).near ?? 0;
    return Math.abs(sideLen(g.sides[0].sides[0]) - sideLen(g.sides[1].sides[0]));
  };
  for (let round = 0, changed = true; changed; round++) {
    if (round > 200) return { error: 'could not fit the faces\' layouts to common edge counts' };
    changed = false;
    for (const g of groups) {
      for (let guard = 0; deficit(g) > 0; guard++) {
        if (guard > 400) return { error: `group ${g.label}: no edge counts fit its layout` };
        // grow the chain (the most coarsely divided one of some side) that most reduces the deficit
        let best = null;
        const now = deficit(g);
        for (const loop of g.sides) for (const side of loop.sides) {
          const e = side.reduce((a, b) => (totalLen(a.c) / count[a.c] >= totalLen(b.c) / count[b.c] ? a : b));
          count[e.c] += 2;
          const d = deficit(g);
          count[e.c] -= 2;
          if (!best || d < best.d) best = { c: e.c, d };
        }
        if (best.d >= now) {                        // nothing helps alone: grow every side
          for (const loop of g.sides) for (const side of loop.sides) count[side[0].c] += 2;
        } else count[best.c] += 2;
        changed = true;
        if (process.env.CAGE_DEBUG) { g.bumps = (g.bumps ?? 0) + 1; }
      }
    }
  }

  // a layout far past the quad budget means the faces' corners and sides fight (a face with many uneven sides):
  // refuse rather than build it
  const estimate = groups.reduce((s, g) => { const L = g.sides.flatMap((l) => l.sides.map(sideLen)), P = L.reduce((t, v) => t + v, 0); return s + P * P / 16; }, 0);
  if (estimate > 20 * quads) return { error: `the faces' layouts need about ${Math.round(estimate)} quads for ${quads} asked: some face has too many uneven sides for the layouts yet` };

  // cage points: chain samples by arc length
  const X = [], at = (p) => { X.push(p); return X.length - 1; };
  const cornerId = new Map();
  const cornerOf = (v) => { if (!cornerId.has(v)) cornerId.set(v, at([...vertices[v]])); return cornerId.get(v); };
  chains.forEach((ch, c) => {
    const L = totalLen(c), n = count[c], fwd = { c, rev: false };
    if (ch.ring) { ch.ids = Array.from({ length: n }, (_, j) => at(along(fwd, L * j / n))); return; }
    ch.ids = [cornerOf(ch.verts[0])];
    for (let j = 1; j < n; j++) ch.ids.push(at(along(fwd, L * j / n)));
    ch.ids.push(cornerOf(ch.verts[ch.verts.length - 1]));
  });
  const idsOf = (e) => (e.rev ? [...chains[e.c].ids].reverse() : chains[e.c].ids);

  // end conditions of the border curves at corners (groupNurbs.js): natural ends at the corner, or where a group is
  // flat, its smooth curve through the corner and the other chains ending on that curve's point there
  const ends = new Map();
  const set = (p, b, W0, Wm1) => ends.set(`${p},${b}`, { W0, Wm1 });
  const atCorner = new Map();                      // corner vertex -> [{ c, first }] chain ends there
  chains.forEach((ch, c) => {
    if (ch.ring) return;
    const n = ch.ids.length - 1, a = ch.verts[0], b = ch.verts[ch.verts.length - 1];
    if (!atCorner.has(a)) atCorner.set(a, []);
    if (!atCorner.has(b)) atCorner.set(b, []);
    atCorner.get(a).push({ c, first: ch.ids[1] });
    atCorner.get(b).push({ c, first: ch.ids[n - 1] });
  });
  for (const [v, list] of atCorner) {
    const P = cornerOf(v), flat = flatAt.get(v);
    if (!flat) { for (const { first } of list) set(P, first, [[P, 1]], [[P, 2], [first, -1]]); continue; }
    // the flat group's two chains at v: its border runs from one into the other
    const g = groups[flat[0]], mine = new Set();
    for (const entries of g.loopChains) for (const e of entries) if (!chains[e.c].ring && (startOf(e) === v || chains[e.c].verts.includes(v))) mine.add(e.c);
    const own = list.filter((x) => mine.has(x.c));
    if (own.length !== 2) { for (const { first } of list) set(P, first, [[P, 1]], [[P, 2], [first, -1]]); flatAt.delete(v); continue; }
    const [b, q] = [own[0].first, own[1].first], star = [[b, 1 / 6], [P, 4 / 6], [q, 1 / 6]];
    for (const { c, first } of list) {
      if (c === own[0].c) set(P, first, [[P, 1]], [[q, 1]]);
      else if (c === own[1].c) set(P, first, [[P, 1]], [[b, 1]]);
      else set(P, first, star, [[b, 2 / 6], [P, 8 / 6], [q, 2 / 6], [first, -1]]);
    }
  }

  // layouts. Every layout is made of k-splits (a region with corners split from a centre into one grid per corner)
  // and a few fixed patterns (two-corner lens, O-grid, ring). A single-loop face is also drawn on a flat domain
  // whose sides are as long as their edge counts (a polygon through its corners, a lens for two corners, a circle
  // for none), its triangles mapped there with fixed border (Tutte: one-to-one onto a convex domain): the layout's
  // own points (centres, stars) are placed there. Grid lines and grid insides are blended in 3D and put on the
  // face's triangles by closest point (a CAD face's fan of slivers maps the domain too unevenly to carry grids).
  const out = [];
  for (const [gi, g] of groups.entries()) {
    const T = g.tris.map((f) => triangles[f]);
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (const t of T) for (const v of t) for (let d = 0; d < 3; d++) { lo[d] = Math.min(lo[d], vertices[v][d]); hi[d] = Math.max(hi[d], vertices[v][d]); }
    const onGroup = projector(vertices, T, Math.max(Math.hypot(...sub(hi, lo)) / 16, 1e-9));
    const point3 = (p) => at(onGroup(p));
    const faces = [], edges = new Map();
    const edge = (a, b, n) => {                     // n + 1 ids from a to b; its inside points made once
      const key = a < b ? `${a},${b}` : `${b},${a}`;
      if (!edges.has(key)) {
        const A3 = X[Math.min(a, b)], B3 = X[Math.max(a, b)], e = [Math.min(a, b)];
        for (let j = 1; j < n; j++) e.push(point3(lerp(A3, B3, j / n)));
        e.push(Math.max(a, b));
        edges.set(key, e);
      }
      const e = edges.get(key);
      if (e.length !== n + 1) throw new Error(`group ${g.label}: an inner edge used with ${e.length - 1} and ${n} segments`);
      return a < b ? e : [...e].reverse();
    };
    const region = (bottom, right, top, left) => {  // P0->P1, P1->P2, P3->P2, P0->P3
      const a = bottom.length - 1, b = left.length - 1;
      if (right.length - 1 !== b || top.length - 1 !== a || bottom[0] !== left[0] || bottom[a] !== right[0] || top[a] !== right[b] || top[0] !== left[b]) {
        throw new Error(`group ${g.label}: a layout grid whose sides do not meet`);
      }
      const G = Array.from({ length: a + 1 }, () => new Int32Array(b + 1));
      for (let s = 0; s <= a; s++) { G[s][0] = bottom[s]; G[s][b] = top[s]; }
      for (let t = 0; t <= b; t++) { G[0][t] = left[t]; G[a][t] = right[t]; }
      const Q = (s, t) => X[G[s][t]];
      for (let s = 1; s < a; s++) for (let t = 1; t < b; t++) {
        const u = s / a, v = t / b, p = [0, 0, 0];
        for (let d = 0; d < 3; d++) {
          p[d] = (1 - v) * Q(s, 0)[d] + v * Q(s, b)[d] + (1 - u) * Q(0, t)[d] + u * Q(a, t)[d]
            - ((1 - u) * (1 - v) * Q(0, 0)[d] + u * (1 - v) * Q(a, 0)[d] + (1 - u) * v * Q(0, b)[d] + u * v * Q(a, b)[d]);
        }
        G[s][t] = point3(p);
      }
      for (let s = 0; s < a; s++) for (let t = 0; t < b; t++) faces.push([G[s][t], G[s + 1][t], G[s + 1][t + 1], G[s][t + 1]]);
    };
    // a k-split of a region given as its sides (ids corner to corner, in loop order); centre: given, or where its
    // spokes come nearest to their edge counts times the region's mean edge length
    const splitRegion = (sides, centre = null) => {
      const k = sides.length, L = sides.map((s) => s.length - 1), { x } = splitParts(L);
      const mid = sides.map((s, i) => s[x[i]]);
      if (centre === null) {
        let len = 0, cnt = 0;
        for (const s of sides) for (let j = 1; j < s.length; j++) { len += Math.hypot(...sub(X[s[j]], X[s[j - 1]])); cnt++; }
        const ell = len / cnt, c = [0, 0, 0];
        mid.forEach((m) => { for (let d = 0; d < 3; d++) c[d] += X[m][d] / k; });
        relax([c], mid.map((m, i) => [c, [...X[m]], x[(i + 1) % k] * ell]));
        centre = point3(c);
      }
      for (let i = 0; i < k; i++) {
        const j = (i + k - 1) % k, xn = x[(i + 1) % k], prev = sides[j];
        region(sides[i].slice(0, x[i] + 1), edge(mid[i], centre, xn), edge(mid[j], centre, x[i]), prev.slice(prev.length - 1 - xn).reverse());
      }
    };
    // each loop: cage ids (cyclic, from its first own corner), their arc length s, the corners' positions in it
    const loops = g.sides.map((loop) => {
      const ids = [], idS = [], meshV = [], meshS = [], cornersAt = [];
      let s0 = 0;
      for (const side of loop.sides) {
        cornersAt.push(ids.length);
        for (const e of side) {
          const ch = chains[e.c], Lc = totalLen(e.c), n = count[e.c], cid = idsOf(e);
          const verts = ch.ring ? [...ch.verts, ch.verts[0]] : ch.verts, vv = e.rev ? [...verts].reverse() : verts;
          const lens = e.rev ? ch.len.map((x) => Lc - x).reverse() : ch.len;
          for (let j = 0; j < n; j++) { ids.push(cid[j]); idS.push(s0 + Lc * j / n); }
          for (let j = 0; j < vv.length - 1; j++) { meshV.push(vv[j]); meshS.push(s0 + lens[j]); }
          s0 += Lc;
        }
      }
      const L = loop.sides.map(sideLen);
      const pt = (i) => ids[((i % ids.length) + ids.length) % ids.length];
      const seg = (i, n, dir = 1) => Array.from({ length: n + 1 }, (_, j) => pt(i + dir * j));
      return { ids, idS, meshV, meshS, cornersAt, L, total: s0, pt, seg, sides: cornersAt.map((I, i) => seg(I, L[i])) };
    });

    if (g.kind === 'ring') {
      const A = loops[0].ids, B = loops[1].ids, L = A.length;
      let o = 0, best = Infinity;
      for (let t = 0; t < L; t++) {
        let s = 0;
        for (let j = 0; j < L; j += Math.max(1, Math.floor(L / 32))) s += Math.hypot(...sub(X[A[j]], X[B[((t - j) % L + L) % L]]));
        if (s < best) { best = s; o = t; }
      }
      let mean = 0;
      for (let j = 0; j < L; j++) mean += Math.hypot(...sub(X[A[j]], X[B[((o - j) % L + L) % L]])) / L;
      const r = Math.max(1, Math.round(mean / h));
      const col = Array.from({ length: L }, (_, j) => {
        const a = A[j], b = B[((o - j) % L + L) % L], c = [a];
        for (let k = 1; k < r; k++) c.push(point3(lerp(X[a], X[b], k / r)));
        c.push(b);
        return c;
      });
      for (let j = 0; j < L; j++) for (let k = 0; k < r; k++) { const c = col[j], d = col[(j + 1) % L]; faces.push([c[k], d[k], d[k + 1], c[k + 1]]); }
    } else if (g.kind === 'holed') {
      // five-sided regions round each outer corner: cuts from the outer sides' middles to the hole
      const O = loops[0], H = loops[1], k = O.L.length, plan = holePlan(O.L, H.ids.length);
      const m = O.sides.map((s, i) => s[plan.a[i]]);
      // hole points: the hole runs the other way round; start where the first cut is shortest
      const Lh = H.ids.length, starts = [];
      let acc = 0;
      for (let i = 0; i < k; i++) { starts.push(acc); acc += plan.arc[(i + 1) % k]; }
      let o = 0, best = Infinity;
      for (let t = 0; t < Lh; t++) {
        let s = 0;
        for (let i = 0; i < k; i++) s += Math.hypot(...sub(X[m[i]], X[H.pt(t - starts[i])]));
        if (s < best) { best = s; o = t; }
      }
      const hIdx = starts.map((st) => o - st), hp = hIdx.map((i) => H.pt(i));
      const cut = m.map((mi, i) => edge(mi, hp[i], plan.c[i]));
      for (let i = 0; i < k; i++) {
        const j = (i + k - 1) % k;
        const arc = H.seg(hIdx[i], plan.arc[i]);   // from h_i along the hole (its own direction) to h_(i-1)
        if (arc[arc.length - 1] !== hp[j]) throw new Error(`group ${g.label}: hole split points out of step`);
        splitRegion([O.sides[i].slice(0, plan.a[i] + 1), cut[i], arc, [...cut[j]].reverse(), O.sides[j].slice(plan.a[j])]);
      }
    } else if (g.kind === 'cutsplit') {
      const lp = loops[0], k = lp.L.length, plan = cutPlan(lp.L), { j, a, b, c } = plan, jp = (j + k - 1) % k;
      const before = lp.sides[jp], after = lp.sides[j], Lp = lp.L[jp];
      const p = before[Lp - b], q = after[a], cutPQ = edge(p, q, c);
      splitRegion([after.slice(0, a + 1), [...cutPQ].reverse(), before.slice(Lp - b)]);
      const poly = [before.slice(0, Lp - b + 1), cutPQ, after.slice(a)];
      for (let i = 1; i <= k - 2; i++) poly.push(lp.sides[(j + i) % k]);
      splitRegion(poly);
    } else {
      // single loop on a flat domain: split (odd k), lens (2 corners or 1), O-grid (none)
      const lp = loops[0], { ids, idS, meshV, meshS, L: Lsides, total, pt, seg } = lp;
      const sOf = new Map(ids.map((id, i) => [id, idS[i]]));
      let breaks = lp.cornersAt.map((I) => idS[I]), curve;
      if (g.kind === 'split') {
        const C = cyclicPolygon(Lsides);
        curve = (i, t) => lerp2(C[i], C[(i + 1) % C.length], t);
      } else if (g.kind === 'bigon' || g.kind === 'teardrop') {
        let LA = Lsides[0], LB = Lsides[1];
        if (g.kind === 'teardrop') { LA = LB = ids.length / 2; breaks = [0, idS[ids.length / 2]]; }
        const D = 0.9003 * Math.min(LA, LB), arcs = [lensArc(LA, D, -1), lensArc(LB, D, 1)];
        curve = (i, t) => arcs[i](t);
      } else {
        const R = ids.length / (2 * Math.PI);
        curve = (i, t) => [R * Math.cos(2 * Math.PI * t), R * Math.sin(2 * Math.PI * t)];
      }
      const border = (s) => {
        let i = breaks.length - 1;
        while (i > 0 && breaks[i] > s + 1e-12) i--;
        const end = breaks[i + 1] ?? total;
        return curve(i, (s - breaks[i]) / ((end - breaks[i]) || 1));
      };
      const UV = new Map();
      meshV.forEach((v, j) => UV.set(v, border(meshS[j])));
      const innerV = [...new Set(T.flat())].filter((v) => !UV.has(v)), idx = new Map(innerV.map((v, i) => [v, i]));
      if (innerV.length) {
        const nb = innerV.map(() => new Set());
        for (const t of T) for (let k = 0; k < 3; k++) for (const [a, b] of [[t[k], t[(k + 1) % 3]], [t[(k + 1) % 3], t[k]]]) if (idx.has(a)) nb[idx.get(a)].add(b);
        const A = envelope(innerV.length, nb.map((s) => new Set([...s].filter((v) => idx.has(v)).map((v) => idx.get(v)))));
        const bu = new Float64Array(innerV.length), bv = new Float64Array(innerV.length);
        nb.forEach((s, i) => {
          A.add(i, i, s.size);
          for (const w of s) {
            if (idx.has(w)) { if (idx.get(w) < i) A.add(i, idx.get(w), -1); } else { const p = UV.get(w); bu[i] += p[0]; bv[i] += p[1]; }
          }
        });
        A.factor();
        const u = A.solve(bu), v = A.solve(bv);
        innerV.forEach((w, i) => UV.set(w, [u[i], v[i]]));
      }
      const to3 = locator(T, UV, vertices);
      const P2 = new Map();
      ids.forEach((id) => { if (!P2.has(id)) P2.set(id, border(sOf.get(id))); });
      const point = (uv) => { const id = at(to3(uv)); P2.set(id, uv); return id; };
      const bigon = (I0, LA, I1, LB) => {
        let m = Math.max(2, 2 * Math.round(Math.min(LA, LB) / 6));
        m = Math.min(m, Math.min(LA, LB) - 4);
        if ((LA - m) % 2) m -= 1;
        const r = (LA - m) / 2, s = (LB - m) / 2;
        const a1 = pt(I0 + r), a2 = pt(I0 + r + m), b2 = pt(I1 + s), b1 = pt(I1 + s + m);
        const uv0 = mid2(P2.get(a1), P2.get(b1)), uv1 = mid2(P2.get(a2), P2.get(b2));
        relax([uv0, uv1], [[uv0, P2.get(a1), s], [uv0, P2.get(b1), r], [uv1, P2.get(a2), s], [uv1, P2.get(b2), r], [uv0, uv1, m]]);
        const p0 = point(uv0), p1 = point(uv1);
        region(seg(I0, r), edge(a1, p0, s), edge(b1, p0, r), seg(I0, s, -1));
        region(seg(I0 + r, m), edge(a2, p1, s), edge(p0, p1, m), edge(a1, p0, s));
        region(seg(I1 + s, m), edge(b1, p0, r), edge(p1, p0, m), edge(b2, p1, r));
        region(seg(I1, s), edge(b2, p1, r), edge(a2, p1, s), seg(I1, r, -1));
      };
      if (g.kind === 'split') {
        const k = Lsides.length, { x } = splitParts(Lsides);
        const mid = lp.cornersAt.map((I, i) => pt(I + x[i]));
        const c = [0, 0];
        mid.forEach((m) => { c[0] += P2.get(m)[0] / k; c[1] += P2.get(m)[1] / k; });
        relax([c], mid.map((m, i) => [c, [...P2.get(m)], x[(i + 1) % k]]));
        splitRegion(lp.sides, point(c));
      } else if (g.kind === 'bigon') {
        bigon(0, Lsides[0], Lsides[0], Lsides[1]);
      } else if (g.kind === 'teardrop') {
        // the point opposite the corner becomes a corner of this group alone: its neighbour's curve runs smoothly
        // through it, and both of this group's curves there follow that curve
        const half = ids.length / 2, p = ids[half];
        set(p, ids[half - 1], [[p, 1]], [[ids[half + 1], 1]]);
        set(p, ids[half + 1], [[p, 1]], [[ids[half - 1], 1]]);
        bigon(0, half, half, half);
      } else {
        const L = ids.length, c0 = Math.max(2, Math.floor(L / 4)), c1 = L / 2 - c0;
        const xs = [0, c0, c0 + c1, 2 * c0 + c1], arc = [c0, c1, c0, c1], r = Math.max(2, Math.round(L / (4 * Math.PI)));
        const q = xs.map((i) => point(P2.get(ids[i]).map((w) => w / 2)));
        for (let j = 0; j < 4; j++) region(seg(xs[j], arc[j]), edge(ids[xs[(j + 1) % 4]], q[(j + 1) % 4], r), edge(q[j], q[(j + 1) % 4], arc[j]), edge(ids[xs[j]], q[j], r));
        region(edge(q[0], q[1], c0), edge(q[1], q[2], c1), edge(q[3], q[2], c0), edge(q[0], q[3], c1));
      }
    }
    out.push({ label: g.label, faces, triangles: g.tris, kind: g.kind });
  }
  if (process.env.CAGE_DEBUG) {
    const init = chains.map((ch, c) => Math.max(ch.ring ? 8 : 2, 2 * Math.round(totalLen(c) / (2 * h))));
    console.log('initial count sum', init.reduce((s, x) => s + x, 0), 'final', count.reduce((s, x) => s + x, 0));
    groups.map((g, i) => [i, g.kind, g.bumps ?? 0, g.sides.map((l) => l.sides.map((side) => `${sideLen(side)}(${side.map((e) => init[e.c]).join('+')})`).join(' '))]).filter((r) => r[2] > 0).sort((a, b) => b[2] - a[2]).slice(0, 12).forEach((r) => console.log('  ', JSON.stringify(r)));
  }
  return { vertices: X, groups: out, ends, counts: count };
}

const lerp2 = (a, b, t) => [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
const mid2 = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];

/** Corners of a convex polygon with these side lengths, counter-clockwise: on a circle where they can be. */
function cyclicPolygon(L) {
  const k = L.length, big = Math.max(...L);
  const angles = (R) => L.map((l) => 2 * Math.asin(Math.min(1, l / (2 * R))));
  let theta;
  if (angles(big / 2).reduce((s, a) => s + a, 0) >= 2 * Math.PI) {
    let lo = big / 2, hi = big * k;
    for (let it = 0; it < 100; it++) { const R = (lo + hi) / 2; if (angles(R).reduce((s, a) => s + a, 0) > 2 * Math.PI) lo = R; else hi = R; }
    theta = angles((lo + hi) / 2);
  } else {
    const sum = L.reduce((s, l) => s + l, 0);         // the circle's centre would be outside it: angles by length
    theta = L.map((l) => 2 * Math.PI * l / sum);
  }
  const R = L.reduce((s, l, i) => s + l / (2 * Math.sin(theta[i] / 2)), 0) / k, C = [];
  let phi = -Math.PI / 2 - theta[0] / 2;
  for (let i = 0; i < k; i++) { C.push([R * Math.cos(phi), R * Math.sin(phi)]); phi += theta[i]; }
  return C;
}

/** A circular arc of length L between (0, 0) and (D, 0): below the chord left to right (side -1), or above it right to left (side 1). */
function lensArc(L, D, side) {
  let lo = 1e-6, hi = 2 * Math.PI - 1e-6;
  for (let it = 0; it < 100; it++) { const a = (lo + hi) / 2; if (a / (2 * Math.sin(a / 2)) < L / D) lo = a; else hi = a; }
  const a = (lo + hi) / 2, R = D / (2 * Math.sin(a / 2)), d = R * Math.cos(a / 2);
  if (side < 0) return (t) => { const th = -Math.PI / 2 - a / 2 + t * a; return [D / 2 + R * Math.cos(th), d + R * Math.sin(th)]; };
  return (t) => { const th = Math.PI / 2 - a / 2 + t * a; return [D / 2 + R * Math.cos(th), -d + R * Math.sin(th)]; };
}

/** Move the free points so the springs [p, q, length] come to their lengths (p, q: coordinate arrays, shared by reference). */
function relax(free, springs) {
  const isFree = new Set(free);
  for (let it = 0; it < 400; it++) for (const [p, q, len] of springs) {
    const d = p.map((x, i) => q[i] - x), l = Math.hypot(...d) || 1e-12, e = (l - len) / l;
    const fp = isFree.has(p), fq = isFree.has(q), wp = fp ? (fq ? 0.5 : 1) : 0, wq = fq ? (fp ? 0.5 : 1) : 0;
    for (let i = 0; i < p.length; i++) { p[i] += wp * e * d[i]; q[i] -= wq * e * d[i]; }
  }
}

/** Domain point -> the mesh point it maps to: the containing triangle's barycentric combination (the nearest triangle when outside). */
function locator(T, UV, vertices) {
  const lo = [Infinity, Infinity], hi = [-Infinity, -Infinity];
  for (const p of UV.values()) for (let d = 0; d < 2; d++) { lo[d] = Math.min(lo[d], p[d]); hi[d] = Math.max(hi[d], p[d]); }
  const n = Math.max(1, Math.ceil(Math.sqrt(T.length / 2))), cw = (hi[0] - lo[0]) / n || 1, chh = (hi[1] - lo[1]) / n || 1;
  const cell = (x, d) => Math.min(n - 1, Math.max(0, Math.floor((x - lo[d]) / (d ? chh : cw))));
  const grid = Array.from({ length: n * n }, () => []);
  T.forEach((t, f) => {
    const P = t.map((v) => UV.get(v));
    const i0 = cell(Math.min(P[0][0], P[1][0], P[2][0]), 0), i1 = cell(Math.max(P[0][0], P[1][0], P[2][0]), 0);
    const j0 = cell(Math.min(P[0][1], P[1][1], P[2][1]), 1), j1 = cell(Math.max(P[0][1], P[1][1], P[2][1]), 1);
    for (let i = i0; i <= i1; i++) for (let j = j0; j <= j1; j++) grid[i * n + j].push(f);
  });
  const bary = (p, f) => {
    const [a, b, c] = T[f].map((v) => UV.get(v));
    const det = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]);
    if (!det) return null;
    const l1 = ((p[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (p[1] - a[1])) / det;
    const l2 = ((b[0] - a[0]) * (p[1] - a[1]) - (p[0] - a[0]) * (b[1] - a[1])) / det;
    return [1 - l1 - l2, l1, l2];
  };
  return (p) => {
    let best = null, bestOut = Infinity;
    const ci = cell(p[0], 0), cj = cell(p[1], 1);
    for (let r = 0; r <= n; r++) {
      for (let i = ci - r; i <= ci + r; i++) for (let j = cj - r; j <= cj + r; j++) {
        if (i < 0 || j < 0 || i >= n || j >= n || (Math.abs(i - ci) !== r && Math.abs(j - cj) !== r)) continue;
        for (const f of grid[i * n + j]) {
          const w = bary(p, f);
          if (!w) continue;
          const outside = Math.max(0, -w[0]) + Math.max(0, -w[1]) + Math.max(0, -w[2]);
          if (outside < bestOut) { bestOut = outside; best = [f, w]; }
        }
      }
      if (best && (bestOut === 0 || r >= 1)) break;
    }
    const [f, w] = best, cl = w.map((x) => Math.max(0, x)), s = cl[0] + cl[1] + cl[2];
    const [a, b, c] = T[f].map((v) => vertices[v]);
    return [0, 1, 2].map((d) => (cl[0] * a[d] + cl[1] * b[d] + cl[2] * c[d]) / s);
  };
}
