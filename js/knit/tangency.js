/**
 * knit/tangency.js — faces that meet smoothly in the mesh, made exactly tangent all at once.
 *
 * Two faces whose surfaces' normals agree within TANGENT_DEG along their shared border are tangent by design:
 * a fillet and its walls, a chain of arcs, a cone between two chamfer planes. Fitted one by one they miss by the
 * fits' noise. A gap g between a plane and a cylinder of radius R is a crease of sqrt(2 g / R) radians, and the
 * edge between two surfaces that nearly touch is ill-conditioned: their intersection wanders along the contact,
 * or there is none. The builder puts the edge of an exactly tangent pair on its contact line or circle
 * (geometry.js tangentLocus), so the pairs are made exact here.
 *
 * Every surface of a connected set of tangent pairs is refitted to its own vertices, each weighted by its
 * surface's noise, subject to every tangency of the set at once. What the shared-axes step made exact stays
 * exact: members of one line stay coaxial, of one plane coplanar, of one direction parallel. Equal sizes among
 * the set's faces (axes.js valueGroups) are held equal in the same solve.
 *
 * Shared directions are held, and turn only when that makes more pairs exact: then a shared direction is one
 * unknown for all its faces, constrained by all their vertices, so it turns only as far as the tangencies need.
 * On the Base each corner cone touches two chamfer planes whose fitted inclinations differed by 1.3e-6, and no
 * half-angle is tangent to both until the chamfer directions agree. The part's main directions never turn, nor
 * those an intent or mirror relation set; a turning direction stays square to the directions it was squared to.
 *
 * The method is Gauss-Newton with the constraints linearized at each step. Each face's own unknowns are
 * eliminated first (its normal equations are small), then the shared directions', leaving the constraints'
 * multipliers. Every member must still pass the shared step's acceptance; a round that fails drops the pair
 * pulling hardest on each failing member (the largest first-order cost, multiplier x violation) and solves again.
 *
 * Pairs, the cases tangentLocus builds edges on:
 *   plane - cylinder     the axis along the plane, at distance r from it
 *   cylinder - cylinder  parallel axes r1 + r2 (or |r1 - r2|) apart
 *   plane - cone         the apex on the plane; the plane's normal at the sine of the half-angle to the axis
 *   coaxial pairs        cylinder, cone, torus, sphere and planes square to the axis, about one shared line:
 *                        their profiles in (rho, z) tangent (a line at distance r from a circle's centre, two
 *                        circles r1 + r2 or |r1 - r2| apart)
 *
 * This replaces a greedy pass that refitted one surface at a time against its more precise partners: a
 * cylinder against planes only (no chains of arcs), a cone against one of its two chamfer planes.
 */

import { dot, sub, add, scale, cross, norm, unit, distNormal } from './geometry.js';

const KINDS = new Set(['plane', 'cylinder', 'cone', 'torus', 'sphere']);
const REVOLVED = new Set(['cylinder', 'cone', 'torus', 'sphere']);
const SQUARE = 1e-10;        // two fixed directions this close to square / parallel are exactly so
const ITERS = 30;
const EXACT = 1e-13;         // converged: every constraint within this (x the part's size for lengths)
const MADE = 1e-9;           // a pair is made when within this; tangentLocus asks 1e-7 of lengths, 1e-9 of angles
// A set of more pairs is solved in connected chunks of at most this many, in turn, each with the faces outside it
// held and directions held: the dense solve grows with the cube of its constraints (a set of 2422 pairs did not
// finish in 30 minutes)
const CHUNK = 300;

/** Orthonormal (e1, e2) spanning the plane square to d. */
function frameOf(d) {
  const ref = Math.abs(d[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  const e1 = unit(cross(d, ref));
  return [e1, cross(d, e1)];
}
const turn = (a, t1, t2) => { const [e1, e2] = frameOf(a); return unit(add(a, add(scale(e1, t1), scale(e2, t2)))); };
const axisOrigin = (kind, p) => (kind === 'cylinder' ? p.axisPoint : kind === 'cone' ? p.apex : p.center);
const dirOf = (kind, p) => (kind === 'plane' ? p.normal : kind === 'sphere' ? null : p.axis);

/**
 * A local chart of a surface about its parameters p: {n, steps, apply(x, dir) -> params}. dir, when given, is the
 * surface's direction (its shared direction's, signed); otherwise a free surface turns by its own last two
 * unknowns and a fixed one keeps its own. Positions are taken at the patch (the centroid's foot on the axis), so a
 * turn pivots there; a cone is held by its radius at that foot and its half-angle (the apex of a near-cylindrical
 * cone is far away and badly placed).
 */
function chartOf(kind, p, P, mode, len) {
  const c = scale(P.reduce((s, x) => add(s, x), [0, 0, 0]), 1 / P.length);
  const L = 1e-6 * len, A = 1e-6, free = mode === 'free';
  if (kind === 'sphere') {
    return { n: 4, steps: [L, L, L, L], apply: (x) => ({ ...p, center: add(p.center, [x[0], x[1], x[2]]), radius: p.radius + x[3] }) };
  }
  if (kind === 'plane') {
    const n0 = unit(p.normal), c0 = sub(c, scale(n0, dot(sub(c, p.origin), n0)));
    return { n: free ? 3 : 1, steps: [L, A, A],
      apply: (x, dir) => ({ ...p, normal: dir ?? (free ? turn(n0, x[1], x[2]) : n0), origin: add(c0, scale(n0, x[0])) }) };
  }
  const a0 = unit(p.axis), [e1, e2] = frameOf(a0), o = axisOrigin(kind, p);
  const z0 = dot(sub(c, o), a0), c0 = add(o, scale(a0, z0));
  const at = (x) => add(c0, add(scale(e1, x[0]), scale(e2, x[1])));
  if (kind === 'cylinder') {
    return { n: free ? 5 : 3, steps: [L, L, L, A, A],
      apply: (x, dir) => ({ ...p, axis: dir ?? (free ? turn(a0, x[3], x[4]) : a0), axisPoint: at(x), radius: p.radius + x[2] }) };
  }
  if (kind === 'cone') {
    const s = z0 >= 0 ? 1 : -1, rho0 = Math.abs(z0) * Math.tan(p.halfAngle);
    return { n: free ? 6 : 4, steps: [L, L, L, A, A, A],
      apply: (x, dir) => {
        const a = dir ?? (free ? turn(a0, x[4], x[5]) : a0), q = at(x), al = p.halfAngle + x[3];
        return { ...p, axis: a, apex: sub(q, scale(a, (s * (rho0 + x[2])) / Math.tan(al))), halfAngle: al };
      } };
  }
  if (kind === 'torus') {
    return { n: free ? 7 : 5, steps: [L, L, L, L, L, A, A],
      apply: (x, dir) => ({ ...p, axis: dir ?? (free ? turn(a0, x[5], x[6]) : a0),
        center: add(add(p.center, add(scale(e1, x[0]), scale(e2, x[1]))), scale(a0, x[2])),
        majorRadius: p.majorRadius + x[3], minorRadius: p.minorRadius + x[4] }) };
  }
  return null;
}

/** A revolved surface (or a plane square to the axis) as its profile in (rho, z) about (o, d). */
function profile(kind, p, o, d) {
  const zOf = (x) => dot(sub(x, o), d);
  if (kind === 'cylinder') return [{ line: [1, 0], c: p.radius }];
  if (kind === 'plane') return [{ line: [0, 1], c: zOf(p.origin) }];
  if (kind === 'cone') {
    const za = zOf(p.apex), ca = Math.cos(p.halfAngle), sa = Math.sin(p.halfAngle);
    return [{ line: [ca, -sa], c: -sa * za }, { line: [ca, sa], c: sa * za }];   // the two nappes
  }
  if (kind === 'torus') return [{ circle: [p.majorRadius, zOf(p.center)], r: p.minorRadius }];
  if (kind === 'sphere') return [{ circle: [0, zOf(p.center)], r: p.radius }];
  return [];
}

/** The tangency gap of two profile elements, its case (sg) fixed by the configuration at the start. */
function profileGap(A, B, sg) {
  if (A.line && B.line) return null;
  if (A.circle && B.line) [A, B] = [B, A];
  if (A.line) {
    const v = A.line[0] * B.circle[0] + A.line[1] * B.circle[1] - A.c;
    return sg === undefined ? { sg: Math.sign(v) || 1 } : sg * v - B.r;
  }
  const dd = Math.hypot(A.circle[0] - B.circle[0], A.circle[1] - B.circle[1]);
  if (sg === undefined) {
    const ext = Math.abs(dd - (A.r + B.r)), int = Math.abs(dd - Math.abs(A.r - B.r));
    return { sg: ext <= int ? 0 : Math.sign(A.r - B.r) || 1 };
  }
  return sg === 0 ? dd - (A.r + B.r) : dd - sg * (A.r - B.r);
}

/**
 * The equations making a pair tangent, from their configuration now: {eqs: [{f(q1, q2), ang}]} (ang: the value
 * is an angle's sine or cosine, else a length), or {why} when the pair is not one of the cases above or its fixed
 * directions forbid it. turns1 / turns2: whether that surface's direction can move in the solve.
 */
function pairEquations(k1, p1, k2, p2, turns1, turns2, coaxial) {
  if (k2 === 'plane' && k1 !== 'plane') return swap(pairEquations(k2, p2, k1, p1, turns2, turns1, coaxial));
  if (coaxial) {
    if (k1 === 'sphere' && k2 === 'sphere') return { why: 'kinds' };
    // the frame from the revolved member: the second when the first is a plane (planes come first) or a sphere
    const frame = (q1, q2) => (k1 === 'plane' ? [axisOrigin(k2, q2), unit(q2.axis)]
      : k1 === 'sphere' ? [q1.center, unit(q2.axis)] : [axisOrigin(k1, q1), unit(q1.axis)]);
    const [o, d] = frame(p1, p2);
    const A = profile(k1, p1, o, d), B = profile(k2, p2, o, d);
    let best = null;
    A.forEach((ea, i) => B.forEach((eb, j) => {
      const s = profileGap(ea, eb);
      if (!s) return;
      const g = Math.abs(profileGap(ea, eb, s.sg));
      if (!best || g < best.g) best = { i, j, sg: s.sg, g };
    }));
    if (!best) return { why: 'profiles' };
    return { eqs: [{ f: (q1, q2) => { const [oo, dd] = frame(q1, q2); return profileGap(profile(k1, q1, oo, dd)[best.i], profile(k2, q2, oo, dd)[best.j], best.sg); } }] };
  }
  if (k1 === 'plane' && k2 === 'cylinder') {
    const eqs = [];
    const sq = (q1, q2) => dot(unit(q1.normal), unit(q2.axis));
    if (turns1 || turns2) eqs.push({ f: sq, ang: true });
    else if (Math.abs(sq(p1, p2)) > SQUARE) return { why: 'unsquare' };
    const s = Math.sign(dot(sub(p2.axisPoint, p1.origin), unit(p1.normal))) || 1;
    eqs.push({ f: (q1, q2) => s * dot(sub(q2.axisPoint, q1.origin), unit(q1.normal)) - q2.radius });
    return { eqs };
  }
  if (k1 === 'plane' && k2 === 'cone') {
    const sn = Math.sign(dot(unit(p1.normal), unit(p2.axis))) || 1;
    return { eqs: [
      { f: (q1, q2) => dot(sub(q2.apex, q1.origin), unit(q1.normal)) },
      { f: (q1, q2) => sn * dot(unit(q1.normal), unit(q2.axis)) - Math.sin(q2.halfAngle), ang: true },
    ] };
  }
  if (k1 === 'cylinder' && k2 === 'cylinder') {
    const eqs = [];
    if (turns1 || turns2) {
      eqs.push({ f: (q1, q2) => dot(unit(q2.axis), frameOf(unit(q1.axis))[0]), ang: true },
        { f: (q1, q2) => dot(unit(q2.axis), frameOf(unit(q1.axis))[1]), ang: true });
    } else if (norm(cross(unit(p1.axis), unit(p2.axis))) > SQUARE) return { why: 'unparallel' };
    const gap = (q1, q2) => { const a = unit(q1.axis), v = sub(q2.axisPoint, q1.axisPoint); return norm(sub(v, scale(a, dot(v, a)))); };
    const dd = gap(p1, p2), ext = Math.abs(dd - (p1.radius + p2.radius)), int = Math.abs(dd - Math.abs(p1.radius - p2.radius));
    const sg = ext <= int ? 0 : Math.sign(p1.radius - p2.radius) || 1;
    eqs.push({ f: (q1, q2) => (sg === 0 ? gap(q1, q2) - (q1.radius + q2.radius) : gap(q1, q2) - sg * (q1.radius - q2.radius)) });
    return { eqs };
  }
  return { why: 'kinds' };
}
function swap(r) {
  if (!r?.eqs) return r;
  return { ...r, eqs: r.eqs.map((e) => ({ ...e, f: (q1, q2) => e.f(q2, q1) })) };
}

/** label pair "a|b" (a < b) -> Set of mesh vertices on their shared boundary. */
function boundaries(mesh) {
  const out = new Map();
  for (let t = 0; t < mesh.T.length; t++) {
    const [a, b, c] = mesh.T[t];
    for (const [u, v] of [[a, b], [b, c], [c, a]]) {
      const o = mesh.across(t, u, v);
      if (o < 0 || mesh.F[o] === mesh.F[t] || o < t) continue;
      const k = `${Math.min(mesh.F[t], mesh.F[o])}|${Math.max(mesh.F[t], mesh.F[o])}`;
      let set = out.get(k);
      if (!set) out.set(k, (set = new Set()));
      set.add(u).add(v);
    }
  }
  return out;
}

/** Solve A x = b (dense, partial pivoting); a column whose pivot vanishes (rel x the largest entry) gets 0. */
function solveDense(A, b, rel = 1e-13) {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  let big = 0;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) big = Math.max(big, Math.abs(A[i][j]));
  const dead = new Array(n).fill(false);
  const rowOf = new Array(n).fill(-1), used = new Array(n).fill(false);
  for (let c = 0; c < n; c++) {
    let piv = -1, pv = rel * big;
    for (let r = 0; r < n; r++) if (!used[r] && Math.abs(M[r][c]) > pv) { pv = Math.abs(M[r][c]); piv = r; }
    if (piv < 0) { dead[c] = true; continue; }
    used[piv] = true; rowOf[c] = piv;
    for (let r = 0; r < n; r++) {
      if (r === piv) continue;
      const f = M[r][c] / M[piv][c];
      if (f) for (let k = c; k <= n; k++) M[r][k] -= f * M[piv][k];
    }
  }
  return Array.from({ length: n }, (_, c) => (dead[c] ? 0 : M[rowOf[c]][n] / M[rowOf[c]][c]));
}

const mv = (M, v) => M.map((row) => row.reduce((s, x, j) => s + x * v[j], 0));
const dotv = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
function inverse(H) {
  const n = H.length, out = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let j = 0; j < n; j++) solveDense(H, Array.from({ length: n }, (_, i) => (i === j ? 1 : 0)), 1e-300).forEach((v, i) => (out[i][j] = v));
  return out;
}

/**
 * One connected set. members: labels; mode(g): 'free' | 'fixed' | an entity (a shared direction cluster it follows,
 * with sign sgn(g)); ents: the turning clusters (kept square to their perpOf and refBy clusters); cons:
 * [{g, h, f, ang}], f of the two labels' params (a label outside members is held). Returns {params, dirs, lambda,
 * viol} (viol relative: lengths over len), or null if the solve diverges.
 */
function solveSet(members, ents, cons, surfaces, pts, sigma, mode, sgn, len) {
  const kind = (g) => surfaces.get(g)[0];
  let cur = new Map(members.map((g) => [g, surfaces.get(g)[1]]));
  const at = (P, g) => P.get(g) ?? surfaces.get(g)[1];           // a partner outside the members is held
  let dirs = new Map(ents.map((e) => [e, e.d]));
  const E = ents.length, ei = new Map(ents.map((e, i) => [e, i]));
  // the perpendicular constraints of the entities: dot(d, f) = 0 with f a fixed direction or another entity's
  const perps = [];
  for (const e of ents) {
    for (const f of e.perpOf ?? []) perps.push(ei.has(f) ? { e, f2: f } : { e, f: f.d });
    for (const j of e.refBy ?? []) if (!ei.has(j)) perps.push({ e, f: j.d });     // (an entity's own perpOf covers it)
  }
  const m = cons.length + perps.length;
  const viol = (P, D) => {
    let v = 0;
    for (const c of cons) { const x = Math.abs(c.f(at(P, c.g), at(P, c.h))); v = Math.max(v, c.ang ? x : x / len); }
    for (const { e, f, f2 } of perps) v = Math.max(v, Math.abs(dot(D.get(e), f2 ? D.get(f2) : f)));
    return v;
  };
  let lambda = new Array(m).fill(0);
  for (let it = 0; it < ITERS; it++) {
    const charts = new Map();
    for (const g of members) charts.set(g, chartOf(kind(g), cur.get(g), pts.get(g), mode(g) === 'free' ? 'free' : 'fixed', len));
    const entOf = (g) => { const md = mode(g); return md === 'free' || md === 'fixed' ? null : md; };
    const dirFor = (g, e, z) => { const d = dirs.get(e); const dd = z ? turn(d, z[0], z[1]) : d; return scale(dd, sgn(g)); };
    const paramsAt = (g, x, z) => { const e = entOf(g); return charts.get(g).apply(x, e ? dirFor(g, e, z) : undefined); };
    // each member's normal equations on its own vertices (central differences): own unknowns x, entity unknowns z
    const blk = new Map();
    for (const g of members) {
      const ch = charts.get(g), P = pts.get(g), w = 1 / sigma(g) ** 2, n = ch.n, x0 = new Array(n).fill(0), e = entOf(g);
      const p0 = paramsAt(g, x0, null);
      const r0 = P.map((x) => distNormal(kind(g), p0, x)[0]);
      const col = (pp, pm, h) => P.map((x) => (distNormal(kind(g), pp, x)[0] - distNormal(kind(g), pm, x)[0]) / (2 * h));
      const Jx = [], Jz = [];
      for (let j = 0; j < n; j++) {
        const xp = x0.slice(), xm = x0.slice(); xp[j] = ch.steps[j]; xm[j] = -ch.steps[j];
        Jx.push(col(paramsAt(g, xp, null), paramsAt(g, xm, null), ch.steps[j]));
      }
      if (e) for (const zj of [[1e-6, 0], [0, 1e-6]]) Jz.push(col(paramsAt(g, x0, zj), paramsAt(g, x0, zj.map((v) => -v)), 1e-6));
      const gram = (U, W) => U.map((u) => W.map((v) => w * dotv(u, v)));
      const A = gram(Jx, Jx);
      let tr = 0;
      for (let i = 0; i < n; i++) tr += A[i][i];
      for (let i = 0; i < n; i++) A[i][i] += 1e-12 * (A[i][i] + tr / n) + 1e-300;
      const Ai = inverse(A), bx = Jx.map((u) => w * dotv(u, r0));
      const b = { Ai, bx, Aibx: mv(Ai, bx), e };
      if (e) { b.B = gram(Jx, Jz); b.C = gram(Jz, Jz); b.bz = Jz.map((u) => w * dotv(u, r0)); b.AiB = [0, 1].map((k) => mv(Ai, b.B.map((row) => row[k]))); }
      blk.set(g, b);
    }
    // constraint values and gradients: rows over each member's own unknowns and over the entities' (2 each)
    const base = new Map(members.map((g) => [g, paramsAt(g, new Array(charts.get(g).n).fill(0), null)]));
    const rows = [];          // [{own: [[g, row]], ent: Map e -> [2], val}]
    for (const c of cons) {
      const own = [], ent = new Map();
      const ev = (pg, ph) => c.f(pg, ph);
      for (const [g, first] of [[c.g, true], [c.h, false]]) {
        const ch = charts.get(g);
        if (!ch) continue;
        const n = ch.n, row = new Array(n).fill(0);
        const other = first ? at(base, c.h) : at(base, c.g);
        const f1 = (p) => (first ? ev(p, other) : ev(other, p));
        for (let j = 0; j < n; j++) {
          const xp = new Array(n).fill(0), xm = new Array(n).fill(0); xp[j] = ch.steps[j]; xm[j] = -ch.steps[j];
          row[j] = (f1(paramsAt(g, xp, null)) - f1(paramsAt(g, xm, null))) / (2 * ch.steps[j]);
        }
        own.push([g, row]);
      }
      for (const e of new Set([c.g, c.h].filter((g) => charts.has(g)).map(entOf).filter(Boolean))) {
        const zr = [0, 1].map((k) => {
          const zp = k ? [0, 1e-6] : [1e-6, 0], zm = zp.map((v) => -v);
          const pg = (z) => (charts.has(c.g) && entOf(c.g) === e ? paramsAt(c.g, new Array(charts.get(c.g).n).fill(0), z) : at(base, c.g));
          const ph = (z) => (charts.has(c.h) && entOf(c.h) === e ? paramsAt(c.h, new Array(charts.get(c.h).n).fill(0), z) : at(base, c.h));
          return (ev(pg(zp), ph(zp)) - ev(pg(zm), ph(zm))) / 2e-6;
        });
        ent.set(e, zr);
      }
      rows.push({ own, ent, val: ev(at(base, c.g), at(base, c.h)) });
    }
    for (const { e, f, f2 } of perps) {
      const d = dirs.get(e), o = f2 ? dirs.get(f2) : f;
      const grad = (u, w) => [0, 1].map((k) => (dot(turn(u, k ? 0 : 1e-6, k ? 1e-6 : 0), w) - dot(turn(u, k ? 0 : -1e-6, k ? -1e-6 : 0), w)) / 2e-6);
      const ent = new Map([[e, grad(d, o)]]);
      if (f2) ent.set(f2, grad(o, d));
      rows.push({ own: [], ent, val: dot(d, o) });
    }
    // the reduced system in [dz (2 per entity); lambda]
    const N = 2 * E + m, M = Array.from({ length: N }, () => new Array(N).fill(0)), rhs = new Array(N).fill(0);
    for (const g of members) {
      const b = blk.get(g);
      if (!b.e) continue;
      const o = 2 * ei.get(b.e);
      for (let i = 0; i < 2; i++) {
        for (let j = 0; j < 2; j++) M[o + i][o + j] += b.C[i][j] - dotv(b.B.map((r) => r[i]), b.AiB[j]);
        rhs[o + i] += -b.bz[i] + dotv(b.B.map((r) => r[i]), b.Aibx);
      }
    }
    const AiRow = rows.map((r) => new Map(r.own.map(([g, row]) => [g, mv(blk.get(g).Ai, row)])));
    rows.forEach((r, k) => {
      const K = 2 * E + k;
      rhs[K] = -r.val;
      for (const [g, row] of r.own) rhs[K] += dotv(row, blk.get(g).Aibx);
      for (const [e, zr] of r.ent) { const o = 2 * ei.get(e); for (let i = 0; i < 2; i++) { M[o + i][K] += zr[i]; M[K][o + i] += zr[i]; } }
      for (const [g, row] of r.own) {
        const b = blk.get(g);
        if (b.e) { const o = 2 * ei.get(b.e); for (let i = 0; i < 2; i++) { const v = dotv(b.B.map((rr) => rr[i]), AiRow[k].get(g)); M[o + i][K] -= v; M[K][o + i] -= v; } }
      }
      for (let l = k; l < rows.length; l++) {
        let s = 0;
        for (const [g, row] of r.own) { const y = AiRow[l].get(g); if (y) s += dotv(row, y); }
        M[K][2 * E + l] -= s;
        if (l !== k) M[2 * E + l][K] -= s;
      }
    });
    // the directions eliminated too (their block is the vertices' stiffness, positive definite), leaving the
    // multipliers' system S lambda = r with S = G H^-1 G^T positive semidefinite; scaled to a unit diagonal, a
    // dependent constraint's pivot vanishes against 1 whatever the stiffness (eliminating in one system left a
    // squareness row, which has no unknowns of its own, a pivot of 1e-15 against entries of 1e15). A pivot under
    // 1e-7 is a dependent constraint: at 1e-12 the Base's nearly dependent rows (a corner's stack of cones and
    // cylinders) took 0.3 mm steps and never converged; at 1e-9 or 1e-7 it converges in three
    const Z = 2 * E, Mzz = M.slice(0, Z).map((r) => r.slice(0, Z)), Mzl = M.slice(0, Z).map((r) => r.slice(Z));
    const Mzzi = Z ? inverse(Mzz) : [];
    const colsOf = (k) => Mzl.map((r) => r[k]);
    const MzziMzl = Array.from({ length: m }, (_, k) => (Z ? mv(Mzzi, colsOf(k)) : []));
    const Mzzirz = Z ? mv(Mzzi, rhs.slice(0, Z)) : [];
    const S = Array.from({ length: m }, (_, k) => Array.from({ length: m }, (_, l) => -(M[Z + k][Z + l] - (Z ? dotv(colsOf(k), MzziMzl[l]) : 0))));
    const r = Array.from({ length: m }, (_, k) => -(rhs[Z + k] - (Z ? dotv(colsOf(k), Mzzirz) : 0)));
    const dg = S.map((row, k) => (row[k] > 0 ? 1 / Math.sqrt(row[k]) : 0));
    const y = solveDense(S.map((row, k) => row.map((v, l) => v * dg[k] * dg[l])), r.map((v, k) => v * dg[k]), 1e-7);
    lambda = y.map((v, k) => v * dg[k]);
    const zsol = Z ? Mzzirz.map((v, i) => v - lambda.reduce((acc, lk, k) => acc + MzziMzl[k][i] * lk, 0)) : [];
    const u = [...zsol, ...lambda];
    // the step: each member's own unknowns from the entities' turn and the multipliers
    const dz = new Map(ents.map((e, i) => [e, [u[2 * i], u[2 * i + 1]]]));
    const next = new Map();
    let moved = 0;
    for (const g of members) {
      const b = blk.get(g), n = charts.get(g).n, v = b.bx.slice();
      if (b.e) { const z = dz.get(b.e); for (let j = 0; j < n; j++) v[j] += b.B[j][0] * z[0] + b.B[j][1] * z[1]; }
      rows.forEach((r, k) => { for (const [h, row] of r.own) if (h === g) for (let j = 0; j < n; j++) v[j] += row[j] * lambda[k]; });
      const d = mv(b.Ai, v).map((x) => -x);
      next.set(g, paramsAt(g, d, b.e ? dz.get(b.e) : null));
      moved = Math.max(moved, ...d.map((x, j) => Math.abs(x) / (charts.get(g).steps[j] * 1e6)));
    }
    for (const e of ents) moved = Math.max(moved, Math.abs(dz.get(e)[0]), Math.abs(dz.get(e)[1]));
    dirs = new Map(ents.map((e) => [e, turn(dirs.get(e), dz.get(e)[0], dz.get(e)[1])]));
    cur = next;
    const v = viol(cur, dirs);
    if (!Number.isFinite(v)) return null;
    if (v <= EXACT && moved <= 1e-12) break;
  }
  return { params: cur, dirs, lambda, viol: viol(cur, dirs) };
}

/**
 * Make the mesh's tangent pairs exact. surfaces is changed in place (Map label -> [kind, params]).
 * sigma(g): the vertex noise of g's fit (its vertices weigh 1 / sigma^2); accepts(g, kind, params): the shared
 * step's check; pts: each label's inlier vertices; locks: the shared step's (dirOf, lines, planeSet, ...). Labels
 * made tangent go into tangentSet. Returns info.
 */
export function solveTangency(mesh, surfaces, { sigma, accepts, pts, cfg, locks, tangentSet, values = [], solvedSet = new Set() }) {
  const t0 = performance.now();
  const info = { tangent_pairs: 0, tangent_made: 0, tangent_rejected: 0 };
  const cosT = Math.cos((cfg.TANGENT_DEG * Math.PI) / 180);
  const len = mesh.diag;
  const lineOf = new Map();
  for (const L of locks.lines) for (const g of L.members) lineOf.set(g, L);
  const usable = (g) => { const s = surfaces.get(g); return !!(s && s[1] && KINDS.has(s[0]) && pts.has(g) && pts.get(g).length); };
  // a label's direction: 'free' when not shared (its own fit's, or rejected by the shared step); a shared one is
  // held ('fixed') or, when the set is solved turning, the shared direction itself (an entity) unless a frame one
  const clusterOf = (g) => locks.dirOf?.get(g) ?? null;
  const modeOf = (turning) => (g) => { const k = clusterOf(g); return !k ? 'free' : k.frame || !turning ? 'fixed' : k; };
  const sgn = (g) => { const k = clusterOf(g), s = surfaces.get(g); return k ? Math.sign(dot(unit(dirOf(s[0], s[1])), k.d)) || 1 : 1; };
  // the candidate pairs: normals within TANGENT_DEG along most of the shared border
  const cands = [];
  for (const [k, verts] of boundaries(mesh)) {
    const [g1, g2] = k.split('|').map(Number);
    if (!usable(g1) || !usable(g2)) continue;
    const s1 = surfaces.get(g1), s2 = surfaces.get(g2);
    const cs = [...verts].map((v) => Math.abs(dot(distNormal(s1[0], s1[1], mesh.V[v])[1], distNormal(s2[0], s2[1], mesh.V[v])[1])));
    cs.sort((a, b) => a - b);
    if (!(cs[cs.length >> 1] >= cosT)) continue;
    info.tangent_pairs++;
    // coaxial: on one shared line, or a sphere / a plane square to the other's axis
    const L1 = lineOf.get(g1), L2 = lineOf.get(g2);
    let coaxial = !!(L1 && L1 === L2);
    if (!coaxial) {
      const [ka, pa, kb, pb] = s1[0] !== 'plane' && s1[0] !== 'sphere' ? [s1[0], s1[1], s2[0], s2[1]] : [s2[0], s2[1], s1[0], s1[1]];
      if (REVOLVED.has(ka) && ka !== 'sphere') {
        const d = unit(pa.axis), o = axisOrigin(ka, pa);
        if (kb === 'sphere') { const v = sub(pb.center, o); coaxial = norm(sub(v, scale(d, dot(v, d)))) <= 1e-9 * len; }
        if (kb === 'plane' && ka !== 'cylinder') coaxial = Math.abs(Math.abs(dot(unit(pb.normal), d)) - 1) <= 1e-9;
      }
    }
    cands.push({ g1, g2, coaxial });
  }
  // equal sizes (axes.js valueGroups): a group touching a set of tangent faces is held equal within it
  const VALUE = { cylinder: (q) => q.radius, cone: (q) => q.halfAngle, torus: (q) => q.minorRadius };
  const groupOf = new Map();
  for (const k of values) for (const it of k.items) if (usable(it.g)) groupOf.set(it.g, k);
  const equationsOf = (p, mode) => {
    if (p.value) return { eqs: [{ f: (q1, q2) => VALUE[p.value](q1) - VALUE[p.value](q2), ang: p.value === 'cone' }] };
    const turns = (g) => mode(g) !== 'fixed' && surfaces.get(g)[0] !== 'sphere';
    const s1 = surfaces.get(p.g1), s2 = surfaces.get(p.g2);
    return pairEquations(s1[0], s1[1], s2[0], s2[1], turns(p.g1), turns(p.g2), p.coaxial);
  };
  // a pair no mode gives equations (kinds or profiles with no contact here: two planes, a cone and a cylinder) joins
  // no set: 00004062_001's 2410 such pairs made one set of 1817 faces
  for (let i = cands.length - 1; i >= 0; i--) {
    const r = equationsOf(cands[i], modeOf(true));
    if (r?.eqs || r?.why === 'unsquare' || r?.why === 'unparallel') continue;
    info[`tangent_skipped_${r?.why ?? 'kinds'}`] = (info[`tangent_skipped_${r?.why ?? 'kinds'}`] ?? 0) + 1;
    cands.splice(i, 1);
  }
  if (!cands.length) return info;
  // turning directions square to each other turn together (each keeps the other square); a turning direction stays
  // square to every direction it was squared to, or that was squared to it
  const clusters = new Set(locks.dirOf?.values() ?? []);
  const linked = new Map();
  for (const k of clusters) {
    k.refBy = [];
    for (const f of k.perpOf ?? []) {
      if (k.frame || f.frame) continue;
      (linked.get(k) ?? linked.set(k, []).get(k)).push(f);
      (linked.get(f) ?? linked.set(f, []).get(f)).push(k);
    }
  }
  for (const k of clusters) for (const f of k.perpOf ?? []) f.refBy?.push(k);
  // sets: tangent pairs joined, with the labels that move with any member (its line's, its plane's, and its shared
  // direction's when that may turn)
  const turnMode = modeOf(true), holdMode = modeOf(false);
  const parent = new Map();
  const find = (x) => { while (parent.get(x) !== x) { parent.set(x, parent.get(parent.get(x))); x = parent.get(x); } return x; };
  const join = (a, b) => { for (const x of [a, b]) if (!parent.has(x)) parent.set(x, x); parent.set(find(a), find(b)); };
  for (const p of cands) join(p.g1, p.g2);
  for (let grew = true; grew; ) {
    grew = false;
    for (const g of [...parent.keys()]) {
      const md = turnMode(g);
      const dirMates = typeof md === 'object' ? [md, ...(linked.get(md) ?? [])].flatMap((k) => k.items.map((it) => it.g)) : [];
      const sizeMates = groupOf.get(g)?.items.map((it) => it.g) ?? [];
      const mates = [...(lineOf.get(g)?.members ?? []), ...(locks.planeSet.get(g) ?? []), ...dirMates, ...sizeMates].filter(usable);
      for (const h of mates) if (!parent.has(h) || find(h) !== find(g)) { join(g, h); grew = true; }
    }
  }
  const sets = new Map();
  for (const g of parent.keys()) (sets.get(find(g)) ?? sets.set(find(g), []).get(find(g))).push(g);
  for (const members of sets.values()) {
    const inSet = new Set(members);
    const mine = cands.filter((p) => inSet.has(p.g1));
    if (!mine.length) continue;
    for (const k of new Set(members.map((g) => groupOf.get(g)).filter(Boolean))) {
      const ms = k.items.map((it) => it.g).filter((g) => inSet.has(g));
      for (const h of ms.slice(1)) mine.push({ g1: ms[0], g2: h, value: k.kind });
    }
    info.tangent_sets = (info.tangent_sets ?? 0) + 1;
    // the locks the set keeps: a line's members on its first member's axis, a plane set's on one plane
    const lockCons = [];
    const seenLines = new Set();
    for (const g of members) {
      const L = lineOf.get(g);
      if (L && !seenLines.has(L)) {
        seenLines.add(L);
        const ms = L.members.filter((h) => inSet.has(h)), k1 = surfaces.get(ms[0])?.[0];
        for (const h of ms.slice(1)) for (const e of [0, 1]) {
          const k2 = surfaces.get(h)[0];
          lockCons.push({ g: ms[0], h, f: (q1, q2) => dot(sub(axisOrigin(k2, q2), axisOrigin(k1, q1)), frameOf(unit(q1.axis))[e]) });
        }
      }
      const ps = locks.planeSet.get(g);
      if (ps && surfaces.get(g)[0] === 'plane') {
        const ms = ps.filter((h) => inSet.has(h));
        if (ms[0] === g) for (const h of ms.slice(1)) lockCons.push({ g, h, f: (q1, q2) => dot(sub(q2.origin, q1.origin), unit(q1.normal)) });
      }
    }
    // the most pairs that can be made exact with every member still passing the shared step's check. A round that
    // fails drops pairs: one per failing member, the one pulling hardest on it (multiplier x violation); or, when
    // the set does not solve exactly, every pair the solve left off exact (one at a time took a solve per pair).
    // whole: every pair or nothing
    const solvePairs = (ms, list, mode, whole = false) => {
      const inM = new Set(ms);
      const lc = lockCons.filter((c) => inM.has(c.g) || inM.has(c.h));
      const ents = [...new Set(ms.map(mode).filter((md) => typeof md === 'object'))];
      const why = {};
      let active = [];
      for (const p of list) {
        const r = equationsOf(p, mode);
        if (r?.eqs) active.push({ ...p, eqs: r.eqs });
        else why[r?.why ?? 'kinds'] = (why[r?.why ?? 'kinds'] ?? 0) + 1;
      }
      if (whole && active.length < list.length) return null;
      const now = (c) => Math.abs(c.f(surfaces.get(c.g)[1], surfaces.get(c.h)[1])) / (c.ang ? 1 : len);
      while (active.length) {
        const cons = [...lc];
        for (const p of active) for (const e of p.eqs) cons.push({ g: p.g1, h: p.g2, f: e.f, ang: e.ang, pair: p });
        const res = solveSet(ms, ents, cons, surfaces, pts, sigma, mode, sgn, len);
        const exact = res && res.viol <= MADE;
        const failing = exact ? ms.filter((g) => !accepts(g, surfaces.get(g)[0], res.params.get(g))) : [];
        if (exact && !failing.length) return { active, res, ents, why };
        if (whole) return null;
        const drop = new Set();
        if (!exact && res) {
          const at = (g) => res.params.get(g) ?? surfaces.get(g)[1];
          for (const c of cons) if (c.pair && Math.abs(c.f(at(c.g), at(c.h))) / (c.ang ? 1 : len) > MADE) drop.add(c.pair);
        }
        for (const g of failing) {
          let worst = null, wv = -1;
          cons.forEach((c, k) => {
            if (!c.pair || (c.g !== g && c.h !== g)) return;
            const v = Math.abs(res.lambda[k] ?? 0) * now(c);
            if (v > wv) { wv = v; worst = c.pair; }
          });
          if (worst) drop.add(worst);
        }
        if (!drop.size) {
          let worst = null, wv = -1;
          for (const c of cons) if (c.pair && now(c) > wv) { wv = now(c); worst = c.pair; }
          if (!worst) break;
          drop.add(worst);
        }
        active = active.filter((p) => !drop.has(p));
      }
      return { active: [], res: null, ents, why };
    };
    const commit = (ms, best, list) => {
      for (const [k, n] of Object.entries(best.why)) info[`tangent_skipped_${k}`] = (info[`tangent_skipped_${k}`] ?? 0) + n;
      const made = best.active.filter((p) => !p.value), sizes = best.active.filter((p) => p.value);
      info.tangent_rejected += list.filter((p) => !p.value).length - made.length - Object.values(best.why).reduce((a, n) => a + n, 0);
      if (!made.length) return;
      for (const g of ms) { surfaces.set(g, [surfaces.get(g)[0], best.res.params.get(g)]); solvedSet.add(g); }
      for (const e of best.ents) e.d = best.res.dirs.get(e);
      for (const p of made) { tangentSet.add(p.g1); tangentSet.add(p.g2); }
      info.tangent_made += made.length;
      if (sizes.length) info.values_with_tangency = (info.values_with_tangency ?? 0) + sizes.length;
      if (best.ents.length) info.tangent_directions_turned = (info.tangent_directions_turned ?? 0) + best.ents.length;
    };
    if (mine.length > CHUNK) {
      // in connected chunks, in turn: faces outside a chunk held (a lock or equal size to one still holds)
      info.tangent_chunked = (info.tangent_chunked ?? 0) + 1;
      const byMember = new Map();
      for (const p of mine) for (const g of [p.g1, p.g2]) (byMember.get(g) ?? byMember.set(g, []).get(g)).push(p);
      const done = new Set();
      for (const seed of mine) {
        if (done.has(seed)) continue;
        const chunk = [], queue = [seed.g1], seen = new Set(queue);
        for (let q = 0; q < queue.length && chunk.length < CHUNK; q++) for (const p of byMember.get(queue[q])) {
          if (done.has(p) || chunk.length >= CHUNK) continue;
          done.add(p); chunk.push(p);
          for (const g of [p.g1, p.g2]) if (!seen.has(g)) { seen.add(g); queue.push(g); }
        }
        const cm = [...new Set(chunk.flatMap((p) => [p.g1, p.g2]))];
        commit(cm, solvePairs(cm, chunk, holdMode), chunk);
      }
      continue;
    }
    // shared directions turn only when that makes more pairs exact than holding them (turning moves every face of
    // the direction, and on 00005828_013 it cost the solid for no pair); holding is pruned only when turning cannot
    // make every pair
    let best = solvePairs(members, mine, holdMode, true);
    if (!best && members.some((g) => typeof turnMode(g) === 'object')) {
      const t = solvePairs(members, mine, turnMode);
      if (t.active.length < mine.length) { const h = solvePairs(members, mine, holdMode); best = t.active.length > h.active.length ? t : h; } else best = t;
    }
    best ??= solvePairs(members, mine, holdMode);
    commit(members, best, mine);
  }
  info.tangent_ms = Math.round(performance.now() - t0);
  return info;
}
