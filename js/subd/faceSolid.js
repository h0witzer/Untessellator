/**
 * faceSolid.js — a mesh painted in groups (a part's faces) as a solid of trimmed B-spline faces, one face per
 * group, the way CAD models are built:
 * - each face one bicubic B-spline surface (faceFit.js fitGroup): a four-sided face's over a rectangle whose sides
 *   are the face's (its rows run along the face and across it, its edges are the surface's own), any other face's
 *   a little larger than the face and trimmed;
 * - each border between two faces one edge curve, fitted once to the mesh border and ending on the corner points,
 *   so neighbours share it;
 * - each face trimmed by its edges' curves projected into its surface's parameters (pcurves, on the edge curve's
 *   knots so a parameter means the same point on both);
 * - a face whose surface closes round (a ring: a cylinder wall) gets a seam edge along the surface's closing line,
 *   with its rings' vertices where that line crosses them.
 * Watertight to the fits' tolerance (each edge curve lies on both its faces' surfaces within it), not exactly.
 *
 * In: vertices [[x, y, z]], triangles [[a, b, c]] wound outward, labels (one group per triangle).
 * Out: { faces, edges, points, report } for faceStep.js, or { error, problems }.
 */
import { fitGroup, splitNormals, meshNormals, FIT } from './faceFit.js';
import { Surface, nzBasis } from '../nurbs.js';
import { AnalyticSurface, ANALYTIC } from './analytic.js';
import { projectOnto, tangentLocus, ontoLocus, exactCurve, distNormal } from '../knit/geometry.js';

const sub = (a, b) => a.map((x, i) => x - b[i]);
const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const len = (a) => Math.hypot(...a);
const add3 = (a, b) => a.map((x, i) => x + b[i]), scale3 = (a, s) => a.map((x) => x * s), unit3 = (a) => scale3(a, 1 / (len(a) || 1));
const CORNER_TURN = Math.PI / 9;                   // 20 degrees

/** Groups, their border loops (group on the left) as chains, and the chains (corner to corner, or a ring). */
export function faceTopology(vertices, triangles, labels) {
  const nv = vertices.length, half = new Map();
  triangles.forEach((t, f) => { for (let k = 0; k < 3; k++) half.set(t[k] * nv + t[(k + 1) % 3], f); });
  const labelsAt = Array.from({ length: nv }, () => new Set());
  triangles.forEach((t, f) => t.forEach((v) => labelsAt[v].add(labels[f])));
  const byLabel = new Map();
  triangles.forEach((t, f) => { if (!byLabel.has(labels[f])) byLabel.set(labels[f], []); byLabel.get(labels[f]).push(f); });
  // the border half-edge after a -> b (in triangle t): turn round b through the group's own triangles
  const nextBorder = (t, b, label) => {
    let tri = t, c = triangles[t][(triangles[t].indexOf(b) + 1) % 3];
    for (let guard = 0; guard < 10000; guard++) {
      const u = half.get(c * nv + b);
      if (u === undefined || labels[u] !== label) return [b, c, tri];
      tri = u; c = triangles[u][(triangles[u].indexOf(b) + 1) % 3];
    }
    throw new Error('faceTopology: a vertex fan that does not close');
  };
  const groups = [], problems = [], failed = [];
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
    if (!border.length) { problems.push(`face ${label} has no border (a whole closed body)`); failed.push(label); continue; }
    const seen = new Set(), loops = [];
    for (const [a0, b0, f0] of border) {
      if (seen.has(a0 * nv + b0)) continue;
      const loop = [];
      let [a, b, f] = [a0, b0, f0];
      do { seen.add(a * nv + b); loop.push(a); [a, b, f] = nextBorder(f, b, label); } while (a !== a0 || b !== b0);
      loops.push(loop);
    }
    groups.push({ label, tris, loops });
  }
  // corners: where three or more groups meet, or where a group's border passes twice (a pinch)
  const corner = new Uint8Array(nv);
  for (let v = 0; v < nv; v++) if (labelsAt[v].size >= 3) corner[v] = 1;
  for (const g of groups) {
    const visits = new Map();
    for (const loop of g.loops) for (const v of loop) visits.set(v, (visits.get(v) ?? 0) + 1);
    for (const [v, c] of visits) if (c > 1) corner[v] = 1;
  }
  const chains = [], chainKey = new Map();
  const register = (verts, ring) => {
    if (ring) {
      const key = 'r' + [...verts].sort((a, b) => a - b).join(',');
      if (chainKey.has(key)) {
        const c = chainKey.get(key), S = chains[c].verts, i = S.indexOf(verts[0]);
        return { c, rev: S[(i + 1) % S.length] !== verts[1] };
      }
      chainKey.set(key, chains.length);
      chains.push({ verts, ring: true, uses: [] });
      return { c: chains.length - 1, rev: false };
    }
    const k = verts.join(','), kr = [...verts].reverse().join(',');
    if (chainKey.has(k)) return { c: chainKey.get(k), rev: false };
    if (chainKey.has(kr)) return { c: chainKey.get(kr), rev: true };
    chainKey.set(k, chains.length);
    chains.push({ verts, ring: false, uses: [] });
    return { c: chains.length - 1, rev: false };
  };
  groups.forEach((g, gi) => {
    g.loopChains = g.loops.map((loop) => {
      const ci = loop.map((v, i) => (corner[v] ? i : -1)).filter((i) => i >= 0);
      if (!ci.length) return [register(loop, true)];
      const rot = loop.slice(ci[0]).concat(loop.slice(0, ci[0]));
      const at = ci.map((i) => i - ci[0]);
      return at.map((s, j) => register(rot.slice(s, (at[j + 1] ?? rot.length) + 1).concat(j + 1 === at.length ? [rot[0]] : []), false));
    });
    g.loopChains.forEach((entries) => entries.forEach((e) => chains[e.c].uses.push({ g: gi, rev: e.rev })));
  });
  return { groups, chains, problems, failed };
}

// ── B-spline curves: { p, knots (flat, clamped on [0, 1]), poles: [[...]] } ─────────────────────────────────────

/**
 * Clamped cubic knots on [0, 1] with `spans` spans placed by the data's parameters ts (sorted, from 0 to 1): knot j
 * at fractional index j (m - 1) / spans along them, so each span holds its share of the points (uniform knots on
 * unevenly spaced points leave spans with none, decided by the regulariser alone; on even spacing these are
 * uniform).
 */
function dataKnots(ts, spans, p = 3) {
  const m = ts.length, inner = [];
  for (let j = 1; j < spans; j++) {
    const x = (j * (m - 1)) / spans, i = Math.min(m - 2, Math.floor(x)), a = x - i;
    inner.push((1 - a) * ts[i] + a * ts[i + 1]);
  }
  return [...Array(p + 1).fill(0), ...inner, ...Array(p + 1).fill(1)];
}

export function curveAt(c, t, nder = 0) {
  const b = nzBasis(c.knots, c.p, Math.min(1, Math.max(0, t)), nder), dim = c.poles[0].length;
  return Array.from({ length: nder + 1 }, (_, d) => {
    const x = new Array(dim).fill(0);
    for (let i = 0; i <= c.p; i++) { const w = b.ders[d][i], P = c.poles[b.first + i]; for (let k = 0; k < dim; k++) x[k] += w * P[k]; }
    return x;
  });
}

/** Dense symmetric positive definite solve (Cholesky), B: n x dim. */
function solveSPD(A, B, n, dim) {
  const L = new Float64Array(n * n);
  for (let j = 0; j < n; j++) {
    let s = A[j * n + j];
    for (let k = 0; k < j; k++) s -= L[j * n + k] ** 2;
    const d = Math.sqrt(Math.max(s, 1e-300));
    L[j * n + j] = d;
    for (let i = j + 1; i < n; i++) {
      let t = A[i * n + j];
      for (let k = 0; k < j; k++) t -= L[i * n + k] * L[j * n + k];
      L[i * n + j] = t / d;
    }
  }
  const X = B.map((r) => [...r]);
  for (let c = 0; c < dim; c++) {
    for (let i = 0; i < n; i++) { let t = X[i][c]; for (let k = 0; k < i; k++) t -= L[i * n + k] * X[k][c]; X[i][c] = t / L[i * n + i]; }
    for (let i = n - 1; i >= 0; i--) { let t = X[i][c]; for (let k = i + 1; k < n; k++) t -= L[k * n + i] * X[k][c]; X[i][c] = t / L[i * n + i]; }
  }
  return X;
}

/**
 * A cubic B-spline through pts[0] and pts[last] (fixed) nearest the points at parameters ts (default: chord
 * length): spans (knots placed by the data, dataKnots) doubled from `spans` until every point is within tol, while
 * there are fewer control points than points and at most maxSpans; failing that, the interpolant (as many control
 * points as points, its knots each the average of three parameters, NURBS Book 9.8, which keeps the system well
 * posed however unevenly the points are spaced); the closest of these. knots: fit on these instead (a pcurve on
 * its edge's). closed: pts[0] = pts[last] and the curve is tangent-continuous there (its first and last legs in
 * line, in the ratio of the end spans). A second difference term, a millionth of the data's weight, only settles
 * control points no point reaches.
 */
export function fitCurve(pts, { ts = null, closed = false, tol = 0, spans = 1, maxSpans = 256, knots: given = null } = {}) {
  if (!given && tol > 0 && pts.length > 2) {
    // points within tol of the last one kept are the same point at the fit's resolution (a split landing beside a
    // vertex); the ends stay
    const keepIdx = [0];
    for (let i = 1; i < pts.length - 1; i++) if (len(sub(pts[i], pts[keepIdx[keepIdx.length - 1]])) > tol) keepIdx.push(i);
    if (keepIdx.length > 1 && len(sub(pts[pts.length - 1], pts[keepIdx[keepIdx.length - 1]])) <= tol) keepIdx.pop();
    keepIdx.push(pts.length - 1);
    if (keepIdx.length < pts.length) { pts = keepIdx.map((i) => pts[i]); if (ts) ts = keepIdx.map((i) => ts[i]); }
  }
  const dim = pts[0].length, m = pts.length;
  if (!ts) {
    const d = [0];
    for (let i = 1; i < m; i++) d.push(d[i - 1] + len(sub(pts[i], pts[i - 1])));
    ts = d.map((x) => x / (d[m - 1] || 1));
  }
  const fitOn = (knots) => {
    const n = knots.length - 4;
    // poles as affine combinations of the free ones: P_i = const_i + sum_j M[i][j] Q_j
    const fixedA = pts[0], fixedB = pts[m - 1];
    const free = [];
    for (let i = 1; i < n - 1; i++) if (!(closed && i === 1)) free.push(i);
    const nf = free.length, col = new Map(free.map((i, j) => [i, j]));
    const r = knots[4] / Math.max(1 - knots[knots.length - 5], 1e-300);   // first span over last
    const rowOf = (i) => {                        // [const, [[j, w]...]]
      if (i === 0) return [fixedA, []];
      if (i === n - 1) return [fixedB, []];
      if (closed && i === 1) return [fixedA.map((x) => (1 + r) * x), [[col.get(n - 2), -r]]];
      return [new Array(dim).fill(0), [[col.get(i), 1]]];
    };
    const rows = Array.from({ length: n }, (_, i) => rowOf(i));
    const A = new Float64Array(nf * nf), B = Array.from({ length: nf }, () => new Array(dim).fill(0));
    let trace = 0;
    const addRow = (coef, target, w) => {        // coef: [[poleIndex, weight]]: sum coef P = target
      const lin = new Map(), cst = new Array(dim).fill(0);
      for (const [i, a] of coef) {
        const [c0, l] = rows[i];
        for (let k = 0; k < dim; k++) cst[k] += a * c0[k];
        for (const [j, b] of l) lin.set(j, (lin.get(j) ?? 0) + a * b);
      }
      const L = [...lin];
      for (const [j, a] of L) {
        for (const [j2, b] of L) A[j * nf + j2] += w * a * b;
        for (let k = 0; k < dim; k++) B[j][k] += w * a * (target[k] - cst[k]);
        trace += w * a * a;
      }
    };
    for (let k = 0; k < m; k++) {
      const b = nzBasis(knots, 3, ts[k], 0);
      addRow(Array.from(b.ders[0], (w, i) => [b.first + i, w]).filter(([, w]) => w), pts[k], 1);
    }
    const mu = 1e-6 * (trace / Math.max(nf, 1) || 1);
    for (let i = 1; i < n - 1; i++) addRow([[i - 1, 1], [i, -2], [i + 1, 1]], new Array(dim).fill(0), mu);
    const Q = nf ? solveSPD(A, B, nf, dim) : [];
    const poles = rows.map(([c0, l]) => { const x = [...c0]; for (const [j, w] of l) for (let k = 0; k < dim; k++) x[k] += w * Q[j][k]; return x; });
    const curve = { p: 3, knots, poles };
    let err = 0;
    for (let k = 0; k < m; k++) err = Math.max(err, len(sub(curveAt(curve, ts[k])[0], pts[k])));
    curve.err = err;
    return curve;
  };
  if (given) return fitOn(given);
  let best = null;
  const keep = (c) => { if (!best || c.err < best.err) best = c; return c.err <= tol; };
  // (closed, the tangent condition takes a control point: one more span to interpolate)
  const full = closed ? m - 2 : m - 3;
  for (let s = Math.max(spans, closed ? 4 : 1); s < full && s <= maxSpans; s *= 2) if (keep(fitOn(dataKnots(ts, s)))) return best;
  if (full >= 1 && full <= maxSpans) {
    if (closed) keep(fitOn(dataKnots(ts, full)));
    else {
      const inner = [];
      for (let j = 1; j <= m - 4; j++) inner.push((ts[j] + ts[j + 1] + ts[j + 2]) / 3);
      keep(fitOn([0, 0, 0, 0, ...inner, 1, 1, 1, 1]));
    }
  }
  return best ?? fitOn(dataKnots(ts, 1));
}

/** The closest point to x on the row of S where coordinate k (0: u, 1: v) is val: the other parameter, by Newton
 * from t (clamped to [lo, hi], halving a step that does not get closer). */
function isoClosest(S, x, k, val, t, lo, hi) {
  const at = (s) => (k === 0 ? S.eval(val, s, 2) : S.eval(s, val, 2));
  let d = at(t), r = sub(d.S, x), cur = dot(r, r);
  for (let it = 0; it < 16; it++) {
    const D1 = k === 0 ? d.Sv : d.Su, D2 = k === 0 ? d.Svv : d.Suu;
    const g = dot(r, D1), h = dot(D1, D1) + dot(r, D2);
    let step = h > 0 ? -g / h : -g / Math.max(dot(D1, D1), 1e-300), nt, nd, nr, nx;
    for (let tries = 0; tries < 8; tries++, step /= 2) {
      nt = Math.min(hi, Math.max(lo, t + step)); nd = at(nt); nr = sub(nd.S, x); nx = dot(nr, nr);
      if (nx <= cur) break;
    }
    if (!(nx <= cur)) break;
    const moved = Math.abs(nt - t);
    t = nt; d = nd; r = nr; cur = nx;
    if (moved <= 1e-14 * (hi - lo)) break;
  }
  return t;
}

// ── the solid ────────────────────────────────────────────────────────────────────────────────────────────────

export function faceSolid(vertices, triangles, labels, { scale = null, tol = FIT.TOL, fits = null, surfaces = null } = {}) {
  const t0 = performance.now();
  const topo = faceTopology(vertices, triangles, labels);
  if (topo.error) return topo;
  const { groups, chains } = topo, problems = [...topo.problems], failed = new Set(topo.failed);
  const fail = (label, msg) => { problems.push(`face ${label}: ${msg}`); failed.add(label); };
  const stop = () => ({ error: problems.join('; '), problems, failed: [...failed] });
  if (problems.length) return stop();
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const p of vertices) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], p[k]); hi[k] = Math.max(hi[k], p[k]); }
  const diag = scale ?? len(sub(hi, lo)), absTol = tol * diag;

  // each group's own mesh, and the analytic surfaces given (the app's, exact: only framed here for their faces)
  const meshes = groups.map((g) => {
    const local = new Map(), Vg = [];
    const Tg = g.tris.map((f) => triangles[f].map((v) => { if (!local.has(v)) { local.set(v, Vg.length); Vg.push(vertices[v]); } return local.get(v); }));
    return { local, Vg, Tg };
  });
  const analyticOf = groups.map((g, gi) => {
    const given = surfaces?.get(g.label);
    return given && ANALYTIC.has(given[0]) ? new AnalyticSurface(given[0], given[1], meshes[gi].Vg) : null;
  });
  // a border a fitted face shares with an analytic one lies on that surface: between two border vertices, the straight
  // line between their places on it in its own parameters ((u, v), unwrapped along the border) — exact for the edges
  // analytic faces mostly have (a cylinder's, cone's or torus's circles, their straight lines), the chord on a plane,
  // and never past a corner the border turns (one smooth curve through them all overshoots there); moved by each
  // vertex's own offset from the surface (zero where the surface is exact; carried linearly between vertices), so it
  // runs through the vertices and follows the surface between them. A border it does not follow (below) is left to the
  // edges' refinement onto both surfaces. Its points between the vertices
  // (three to a segment) are data for the fitted face too, so its surface meets the analytic one along the whole
  // border, not only at the vertices, and they are the edge's points (onA). Not where the curve strays from a
  // segment's chord by more than the sagitta of the turn the border makes there (the bound the edges' refinement
  // keeps, below): then the surface does not follow the border (a cone typed onto a coarse tip swings round its apex).
  const extraOf = groups.map(() => []), analyticNbrs = groups.map(() => new Set()), onA = new Map();
  chains.forEach((ch, c) => {
    if (ch.uses.length !== 2) return;
    const [ua, ub] = ch.uses;
    if (!!analyticOf[ua.g] === !!analyticOf[ub.g]) return;
    const A = analyticOf[ua.g] ?? analyticOf[ub.g], gF = analyticOf[ua.g] ? ub.g : ua.g;
    if (surfaces?.has(groups[gF].label)) return;      // a B-spline given: not fitted here
    const verts = ch.ring ? [...ch.verts, ch.verts[0]] : ch.verts;
    let prev = null;
    const off = [];
    const uv = verts.map((v) => { const r = A.closest(vertices[v], prev); prev = [r.u, r.v]; off.push(sub(vertices[v], r.S)); return prev; });
    const { local } = meshes[gF], ins = [], extra = [];
    const n = verts.length, turn = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      if (!ch.ring && (i === 0 || i === n - 1)) continue;
      const ip = ch.ring ? (i - 1 + n - 1) % (n - 1) : i - 1, inx = ch.ring ? (i + 1) % (n - 1) : i + 1;
      const d1 = sub(vertices[verts[i]], vertices[verts[ip]]), d2 = sub(vertices[verts[inx]], vertices[verts[i]]), l1 = len(d1), l2 = len(d2);
      if (l1 > 0 && l2 > 0) turn[i] = Math.acos(Math.max(-1, Math.min(1, dot(d1, d2) / (l1 * l2))));
    }
    let follows = true;
    for (let i = 0; i + 1 < verts.length && follows; i++) {
      const a0 = vertices[verts[i]], ab = sub(vertices[verts[i + 1]], a0), L = len(ab);
      const sag = (L / 2) * Math.tan(Math.max(turn[i], turn[i + 1]) / 4) + absTol;
      const within = (pts3) => pts3.every((x) => !(L > 0) || len(cross(sub(x, a0), ab)) / L <= sag);
      const shift = (on, w) => on.map((y, k) => y + off[i][k] + w * (off[i + 1][k] - off[i][k]));
      const seg = [1, 2, 3].map((j) => shift(A.eval(...[0, 1].map((k) => uv[i][k] + (j / 4) * (uv[i + 1][k] - uv[i][k]))).S, j / 4));
      if (!within(seg)) { follows = false; break; }
      seg.forEach((x, j) => extra.push({ x, a: local.get(verts[i]), b: local.get(verts[i + 1]), w: (j + 1) / 4 }));
      ins.push(seg);
    }
    if (!follows) return;
    for (const x of extra) extraOf[gF].push(x);
    onA.set(c, { ins, g: gF });
    analyticNbrs[gF].add(groups[ua.g === gF ? ub.g : ua.g].label);
  });

  // four-sided faces: a fitted face with one border loop whose corners, where the border turns by more than
  // CORNER_TURN in the face's tangent plane at a change of neighbour (a chain's end), are four. Its sides go to the
  // fitter (faceFit.js rectParam), so its surface's edges are the face's own and its rows follow the face. (The share
  // of the ABC freeform set's faces with four such corners is the same at 20 and 35 degrees: a CAD face's corners
  // are tangent breaks, not the border's curvature between vertices.)
  const sidesOf = (gi) => {
    const g = groups[gi];
    if (g.loops.length !== 1 || g.loopChains[0].length < 3) return null;
    const loop = g.loops[0], n = loop.length, { local, Vg, Tg } = meshes[gi], turns = [];
    const starts = new Set(g.loopChains[0].map((e) => { const v = chains[e.c].verts; return e.rev ? v[v.length - 1] : v[0]; }));
    const { vn } = meshNormals(Vg, Tg);
    const at = [];
    for (let i = 0; i < n; i++) {
      const v = loop[i];
      if (!starts.has(v)) continue;
      const nr = vn[local.get(v)], ln = len(nr);
      let d0 = sub(vertices[v], vertices[loop[(i - 1 + n) % n]]), d1 = sub(vertices[loop[(i + 1) % n]], vertices[v]);
      if (ln > 0) { const u = nr.map((x) => x / ln); d0 = sub(d0, u.map((x) => x * dot(d0, u))); d1 = sub(d1, u.map((x) => x * dot(d1, u))); }
      const l0 = len(d0), l1 = len(d1);
      const turn = l0 > 0 && l1 > 0 ? Math.acos(Math.max(-1, Math.min(1, dot(d0, d1) / (l0 * l1)))) : 0;
      if (turn > CORNER_TURN) { at.push(i); turns.push(turn); }
    }
    if (at.length !== 4 && at.length !== 3) return null;
    const sides = at.map((i0, s) => {
      const side = [];
      for (let i = i0; ; i = (i + 1) % n) { side.push(local.get(loop[i])); if (i === at[(s + 1) % at.length]) break; }
      return side;
    });
    if (at.length === 4) return sides;
    // three corners: the sharpest is the tip, where the face's width runs out (the border doubling back on itself, as
    // a fillet's two rails meet); the rectangle's side there collapses onto it: [rail into the tip, the tip, rail out
    // of it, the third side]
    const t = turns.indexOf(Math.max(...turns));
    return [sides[(t + 2) % 3], [local.get(loop[at[t]])], sides[t], sides[(t + 1) % 3]];
  };

  // a four-sided face's sides are its surface's boundary rows, and a side with few vertices leaves its row to
  // extrapolation from the inside (a tapered face's short end, stretched in the map, overshoots back and forth). So
  // each side's segments get points between their vertices too (three to a segment) from the smooth curve through
  // the side's vertices (a single segment: its chord, which a CAD mesher leaves single only where the edge is
  // straight): data, but soft (an estimate: no refinement to reach them). Not where the segment has points already
  // (a border on an analytic face: points of that surface).
  const sidePoints = (Vg, sides, have) => {
    const done = new Set(have.map((e) => (e.a < e.b ? `${e.a}_${e.b}` : `${e.b}_${e.a}`))), out = [];
    for (const side of sides) {
      const pts = side.map((l) => Vg[l]), d = [0];
      for (let i = 1; i < pts.length; i++) d.push(d[i - 1] + len(sub(pts[i], pts[i - 1])));
      if (!(d[d.length - 1] > 0)) continue;
      const ts = d.map((x) => x / d[d.length - 1]), C = fitCurve(pts, { ts });
      for (let i = 0; i + 1 < side.length; i++) {
        const a = side[i], b = side[i + 1];
        if (done.has(a < b ? `${a}_${b}` : `${b}_${a}`)) continue;
        for (const w of [0.25, 0.5, 0.75]) out.push({ x: curveAt(C, ts[i] + w * (ts[i + 1] - ts[i]))[0], a, b, w, soft: true });
      }
    }
    return out;
  };

  // one surface per group
  const faces = groups.map((g, gi) => {
    const { local, Vg, Tg } = meshes[gi];
    // surfaces: the app's own for a group (its analytic types exact, as given: only framed for this face, its vertices'
    // parameters their projections); any other group is fitted here. The mesh's vertices are the data (a CAD mesh's
    // are on the true surface; its triangles' insides are chords), with its borders on analytic faces. fits: a cache
    // across calls that differ only in other faces (faceSolidSplit's retries), by label and analytic neighbours.
    const given = surfaces?.get(g.label) ?? null, analytic = !!analyticOf[gi];
    // which way the surface faces: its normal at each vertex's place on it against the mesh's (outward, area
    // weighted) there, by area. Where it faces the other way under more than one triangle's worth of the face, it
    // folds under it (a fit across a cusp or crease the face has inside it). (Not counted: vertices where the mesh
    // itself folds over, meshNormals.)
    const { vn, va, trusted, area } = meshNormals(Vg, Tg);
    const facing = (S2, vertexUV) => {
      let vote = 0, against = 0, measured = 0;
      for (const [l, uv] of vertexUV) {
        if (!trusted[l]) continue;
        const s1 = Math.sign(dot(vn[l], S2.normal(uv[0], uv[1])[1]));
        vote += s1 * va[l]; measured += va[l];
        if (s1 < 0) against += va[l];
      }
      return { vote, folded: Math.min(against, measured - against) };
    };
    let fit, S;
    if (given) {
      S = analytic ? analyticOf[gi] : Surface.fromParams(given[1]);
      const vertexUV = new Map();
      let maxErr = 0, sq = 0;
      Vg.forEach((q, l) => { const r = S.closest(q); vertexUV.set(l, [r.u, r.v]); const e = len(sub(r.S, q)); maxErr = Math.max(maxErr, e); sq += e * e; });
      fit = { params: analytic ? null : given[1], vertexUV, info: { param: given[0], spans: [0, 0], max_err: maxErr / diag, rms_err: Math.sqrt(sq / Vg.length) / diag } };
    } else {
      // a four-sided face's own route first; where it gives no surface, or one that folds under the face, the
      // trimmed route (the face's map's box)
      const nbrKey = `${g.label}|${[...analyticNbrs[gi]].sort((a, b) => a - b).join(',')}`;
      const fitWith = (sides) => {
        const key = `${nbrKey}|${sides ? sides.map((s) => s[0]).join(',') : ''}`;
        const f = fits?.get(key) ?? fitGroup(Vg, Tg, { scale: diag, extra: sides ? [...extraOf[gi], ...sidePoints(Vg, sides, extraOf[gi])] : extraOf[gi], sides });
        fits?.set(key, f);
        return f;
      };
      const sides = sidesOf(gi);
      fit = fitWith(sides);
      if (sides && (!fit.params || facing(Surface.fromParams(fit.params), fit.vertexUV).folded > area / Tg.length)) fit = fitWith(null);
      if (!fit.params) { fail(g.label, `no single surface fits it (${fit.info.fail})`); return null; }
      S = Surface.fromParams(fit.params);
    }
    const periodic = analytic ? S.periodicU : !!fit.params.u_periodic;
    // a surface that folds under its face: none for this face, which faceSolidSplit then cuts in two
    const { vote, folded } = facing(S, fit.vertexUV);
    if (folded > area / Tg.length) { fail(g.label, `its surface folds under it (${((100 * folded) / area).toFixed(1)} % of its area)`); return null; }
    const seed = new Map();
    for (const [v, l] of local) { const uv = fit.vertexUV.get(l); if (uv) seed.set(v, uv); }
    return { g, gi, S, params: fit.params, periodic, analytic, sense: vote >= 0, info: fit.info, seed, shift: 0, sideOf: fit.sideOf ?? null };
  });
  if (problems.length) return stop();

  // working chains: their points and end vertices (ids into points). A face whose surface closes round has its
  // seam on the surface's closing line (u = the domain's start); each of its two loops crosses that line once, and
  // the chain crossing it is split there (a ring opens there), so every edge's pcurve stays inside the domain.
  const points = [], pointOf = new Map();
  const vertexAt = (v) => { if (!pointOf.has(v)) { pointOf.set(v, points.length); points.push([...vertices[v]]); } return pointOf.get(v); };
  const work = chains.map((ch) => ({
    pts: ch.verts.map((v) => [...vertices[v]]), vid: [...ch.verts], ring: ch.ring, closed: false,
    v0: ch.ring ? -1 : vertexAt(ch.verts[0]), v1: ch.ring ? -1 : vertexAt(ch.verts[ch.verts.length - 1]),
  }));
  const loopsOf = groups.map((g) => g.loopChains.map((l) => l.map((e) => ({ ...e }))));
  for (const [c, { ins, g }] of onA) {
    if (faces[g].info.exact) continue;
    const ch = work[c], pts = ch.ring ? [...ch.pts, ch.pts[0]] : ch.pts, vid = ch.ring ? [...ch.vid, ch.vid[0]] : ch.vid;
    const outP = [pts[0]], outV = [vid[0]];
    for (let i = 0; i + 1 < pts.length; i++) {
      for (const x of ins[i]) { outP.push(x); outV.push(-1); }
      outP.push(pts[i + 1]); outV.push(vid[i + 1]);
    }
    ch.pts = ch.ring ? outP.slice(0, -1) : outP;
    ch.vid = ch.ring ? outV.slice(0, -1) : outV;
    ch.onA = true;
  }
  const seedFor = (f, v) => {
    const uv = v >= 0 ? f.seed.get(v) : null;
    if (!uv) return null;
    if (!f.periodic) return uv;
    const [u0, u1] = f.S.dom, P = u1 - u0;
    return [u0 + ((((uv[0] - f.shift - u0) % P) + P) % P), uv[1]];
  };
  // each border where its two faces meet: the mesh gives its vertices only, and the smooth curve through a coarse
  // ring of them is not the edge between them (a hexagon's spline is not its circle). Where that curve's midpoint
  // between two vertices strays more than tol from either face's surface, the point is moved onto both (projected
  // onto each in turn until they agree: alternating projections, which converge onto the intersection where the
  // faces cross) and kept, and each half is checked again (three levels at most); the edge curve is then fitted
  // through vertices and these points alike. A point is kept only where its correction is more than its own
  // uncertainty: the surfaces are right to their fit (e, at least tol), and two surfaces at angle a locate their
  // intersection only to e / sin a across it (faces meeting near tangentially do not say where their edge is: the
  // mesh border does); nor where the projections do not come within tol of each other; and only within what the
  // mesh allows: its border is inscribed in the true edge, which strays from a segment's chord by no more than the
  // sagitta of the turn the border makes there, (L / 2) tan(turn / 4) (between vertices neither surface has data,
  // and a surface's guess there is no better than the mesh's).
  const onBoth = (x, fa, fb, sa, sb) => {
    let a = fa.S.closest(x, sa), b = fb.S.closest(a.S, sb);
    for (let it = 0; it < 30 && len(sub(a.S, b.S)) > absTol / 8; it++) {
      a = fa.S.closest(b.S, [a.u, a.v]);
      b = fb.S.closest(a.S, [b.u, b.v]);
    }
    if (len(sub(a.S, b.S)) > absTol) return null;
    const na = fa.S.normal(a.u, a.v)[1], nb = fb.S.normal(b.u, b.v)[1];
    const sin = len(cross(na, nb)), e = Math.max(absTol, fa.info.max_err * diag, fb.info.max_err * diag);
    return { y: a.S.map((q, k) => (q + b.S[k]) / 2), unc: e / Math.max(sin, 1e-12) };
  };
  chains.forEach((c0, c) => {
    if (c0.uses.length !== 2 || work[c].onA) return;
    const ch = work[c], [fa, fb] = c0.uses.map((u) => faces[u.g]);
    const pts = ch.ring ? [...ch.pts, ch.pts[0]] : ch.pts, vid = ch.ring ? [...ch.vid, ch.vid[0]] : ch.vid, d = [0];
    for (let i = 1; i < pts.length; i++) d.push(d[i - 1] + len(sub(pts[i], pts[i - 1])));
    const ts = d.map((x) => x / (d[d.length - 1] || 1));
    const C = fitCurve(pts, { ts, closed: ch.ring, tol: absTol });
    // the border's turn at each vertex (a chain's ends are corners: no turn of its own there)
    const n = pts.length, turn = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const ip = ch.ring ? (i - 1 + n - 1) % (n - 1) : i - 1, inx = ch.ring ? (i + 1) % (n - 1) : i + 1;
      if (!ch.ring && (i === 0 || i === n - 1)) continue;
      const d1 = sub(pts[i], pts[ip]), d2 = sub(pts[inx], pts[i]), l1 = len(d1), l2 = len(d2);
      if (l1 > 0 && l2 > 0) turn[i] = Math.acos(Math.max(-1, Math.min(1, dot(d1, d2) / (l1 * l2))));
    }
    const outP = [pts[0]], outV = [vid[0]];
    for (let i = 0; i + 1 < pts.length; i++) {
      const sa = seedFor(fa, vid[i]) ?? seedFor(fa, vid[i + 1]), sb = seedFor(fb, vid[i]) ?? seedFor(fb, vid[i + 1]);
      const A = pts[i], AB = sub(pts[i + 1], A), L = len(AB);
      const sag = (L / 2) * Math.tan(Math.max(turn[i], turn[i + 1]) / 4) + absTol;
      const offChord = (y) => (L > 0 ? len(cross(sub(y, A), AB)) / L : len(sub(y, A)));
      const ins = [];
      const refine = (ta, tb, depth) => {
        const x = curveAt(C, (ta + tb) / 2)[0], r = onBoth(x, fa, fb, sa, sb);
        if (!r || len(sub(x, r.y)) <= Math.max(absTol, r.unc) || offChord(r.y) > sag) return;
        if (depth < 2) refine(ta, (ta + tb) / 2, depth + 1);
        ins.push(r.y);
        if (depth < 2) refine((ta + tb) / 2, tb, depth + 1);
      };
      refine(ts[i], ts[i + 1], 0);
      // in order along the segment, strictly inside it (a point moved past its neighbour would fold the edge back
      // on itself)
      let last = 0;
      for (const y of ins) {
        const f = L > 0 ? dot(sub(y, A), AB) / (L * L) : 0;
        if (f > last && f < 1) { outP.push(y); outV.push(-1); last = f; }
      }
      outP.push(pts[i + 1]); outV.push(vid[i + 1]);
    }
    if (outP.length === pts.length) return;
    ch.pts = ch.ring ? outP.slice(0, -1) : outP;
    ch.vid = ch.ring ? outV.slice(0, -1) : outV;
  });
  const split = (c, j, X) => {                     // X: between stored points j and j + 1 (cyclically for a ring)
    const ch = work[c], n = ch.pts.length, q = points.length;
    points.push(X);
    if (ch.ring) {
      ch.pts = [X, ...ch.pts.slice(j + 1), ...ch.pts.slice(0, j + 1), X];
      ch.vid = [-1, ...ch.vid.slice(j + 1), ...ch.vid.slice(0, j + 1), -1];
      ch.ring = false; ch.closed = true; ch.v0 = ch.v1 = q;
      return q;
    }
    const c2 = work.length;
    work.push({ pts: [X, ...ch.pts.slice(j + 1)], vid: [-1, ...ch.vid.slice(j + 1)], ring: false, closed: false, v0: q, v1: ch.v1 });
    // (a closed chain split again, at a second face's seam, is two open ones)
    ch.pts = [...ch.pts.slice(0, j + 1), X]; ch.vid = [...ch.vid.slice(0, j + 1), -1]; ch.v1 = q; ch.closed = false;
    for (const loops of loopsOf) for (const l of loops) for (let i = 0; i < l.length; i++) {
      if (l[i].c !== c) continue;
      l.splice(i, 1, ...(l[i].rev ? [{ c: c2, rev: true }, { c, rev: true }] : [{ c, rev: false }, { c: c2, rev: false }]));
      i++;
    }
    return q;
  };
  // a loop's points in its direction on face f's surface, each with the stored chain segment it starts and its u
  // unwrapped along the loop; ends: where each chain starts; turn: how many times the loop goes round the surface
  // (a rim once, either way; a hole in the wall not at all)
  const trace = (f, l) => {
    const { S } = f, [u0, u1] = S.dom, P = u1 - u0, seq = [], ends = [];
    for (const e of l) {
      const ch = work[e.c], n = ch.pts.length;
      ends.push(seq.length);
      if (ch.ring) for (let k = 0; k < n; k++) { const i = e.rev ? (n - k) % n : k; seq.push({ x: ch.pts[i], v: ch.vid[i], c: e.c, j: e.rev ? (n - k - 1 + n) % n : k, rev: e.rev }); }
      else for (let k = 0; k + 1 < n; k++) { const i = e.rev ? n - 1 - k : k; seq.push({ x: ch.pts[i], v: ch.vid[i], c: e.c, j: e.rev ? n - 2 - k : k, rev: e.rev }); }
    }
    let prev = null, last = null;
    for (const s2 of seq) {
      const r = S.closest(s2.x, seedFor(f, s2.v) ?? prev);
      prev = [r.u, r.v];
      let u = r.u;
      if (last !== null) { while (u - last > P / 2) u -= P; while (last - u > P / 2) u += P; }
      s2.u = last = u;
    }
    let back = seq[0].u - last;                    // the step back to the start, the short way
    while (back > P / 2) back -= P;
    while (back < -P / 2) back += P;
    return { seq, ends, turn: Math.round((last + back - seq[0].u) / P) };
  };
  // the segments of a traced loop that cross the line u = us0 (or its copies a period apart)
  const crossings = ({ seq, turn }, P, us0) => {
    const hits = [];
    for (let k = 0; k < seq.length; k++) {
      const a = seq[k].u, b = k + 1 < seq.length ? seq[k + 1].u : seq[0].u + turn * P;
      if (Math.floor((a - us0) / P) !== Math.floor((b - us0) / P)) hits.push(k);
    }
    return hits;
  };
  const seamAt = new Map();                        // face -> Map(rim loop -> the vertex where it crosses the seam)
  for (const f of faces) {
    if (!f.periodic) continue;
    const loops = loopsOf[f.gi];
    placeSeam(f, loops.map((l) => trace(f, l)));
    const { S } = f, [u0, u1] = S.dom, P = u1 - u0, at = new Map();
    seamAt.set(f.gi, at);
    loops.forEach((l, li) => {
      const tr = trace(f, l), { seq, turn } = tr, hits = crossings(tr, P, u0);
      if (Math.abs(turn) > 1) { fail(f.g.label, `a border goes round its surface ${Math.abs(turn)} times`); return; }
      if (hits.length !== Math.abs(turn)) { fail(f.g.label, `a border crosses its seam ${hits.length} times`); return; }
      if (!turn) return;                           // a hole in the wall
      const k = hits[0], s2 = seq[k], s3 = seq[(k + 1) % seq.length];
      const ua = s2.u, ub = k + 1 < seq.length ? s3.u : seq[0].u + turn * P;
      const target = u0 + Math.max(Math.floor((ua - u0) / P), Math.floor((ub - u0) / P)) * P;
      // the crossing on the smooth curve through the chain's points (the edge curve it will be fitted to), not on
      // the chord between two of them (off the surface by its sagitta), by bisection on the projected u (the
      // curve is not linear in u); the segment from stored point j to j + 1 in the chain's own direction
      const ch = work[s2.c], pts = ch.ring ? [...ch.pts, ch.pts[0]] : ch.pts, d = [0];
      for (let i = 1; i < pts.length; i++) d.push(d[i - 1] + len(sub(pts[i], pts[i - 1])));
      const ts = d.map((x) => x / (d[d.length - 1] || 1));
      const C = fitCurve(pts, { ts, closed: ch.ring || ch.closed, tol: absTol });
      const [ta, tb, uA, uB] = s2.rev ? [ts[s2.j], ts[s2.j + 1], ub, ua] : [ts[s2.j], ts[s2.j + 1], ua, ub];
      const r0 = S.closest(s2.x, seedFor(f, s2.v)), seedUV = [r0.u, r0.v];
      let lo3 = 0, hi3 = 1;
      for (let it = 0; it < 40; it++) {
        const wm = (lo3 + hi3) / 2, r = S.closest(curveAt(C, ta + wm * (tb - ta))[0], seedUV);
        let u = r.u;
        const guess = uA + wm * (uB - uA);
        while (u - guess > P / 2) u -= P;
        while (guess - u > P / 2) u += P;
        if ((u - target) * (uB - uA) < 0) lo3 = wm; else hi3 = wm;
      }
      const w = Math.min(1 - 1e-6, Math.max(1e-6, (lo3 + hi3) / 2));
      at.set(li, split(s2.c, s2.j, curveAt(C, ta + w * (tb - ta))[0]));
    });
  }
  if (problems.length) return stop();
  function placeSeam(f, traces) {
    // the seam may sit on any knot (rotating a uniform periodic net by whole spans moves it, the surface unchanged),
    // on an analytic surface at any angle (its frame turned): the one each rim crosses once and no hole crosses, the
    // rims' crossings furthest from their corners (in points along the loop)
    const [u0, u1] = f.S.dom, P = u1 - u0;
    const wrap = f.analytic ? 720 : f.params.poles.length - 3, h = P / wrap;
    let best = null;
    for (let j = 0; j < wrap; j++) {
      const us0 = u0 + j * h;
      let bad = 0, room = Infinity;
      for (const tr of traces) {
        const hits = crossings(tr, P, us0), n = tr.seq.length;
        bad += Math.abs(hits.length - Math.abs(tr.turn));
        if (tr.turn && hits.length) room = Math.min(room, ...tr.ends.map((e) => Math.min(Math.abs(hits[0] - e), n - Math.abs(hits[0] - e))));
      }
      if (!best || bad < best.bad || (bad === best.bad && room > best.room)) best = { j, bad, room };
    }
    if (!best.j) return;
    if (f.analytic) { f.S.rotate(best.j * h); f.shift = best.j * h; return; }
    const P2 = f.params.poles.slice(0, wrap), rot = P2.slice(best.j).concat(P2.slice(0, best.j));
    f.params = { ...f.params, poles: rot.concat(rot.slice(0, 3)) };
    f.S = Surface.fromParams(f.params);
    f.shift = best.j * h;                          // a point at u on the old net is at u - shift on the new one
  }
  const usesOf = work.map(() => []);
  loopsOf.forEach((loops, gi) => loops.forEach((l) => l.forEach((e) => usesOf[e.c].push({ g: gi, rev: e.rev }))));
  for (const ch of work) if (ch.ring) { const q = points.length; points.push(ch.pts[0]); ch.v0 = ch.v1 = q; }

  // EXACT EDGES where two analytic faces meet. Their surfaces are given exactly (and made exact together:
  // faceExport.js runs the shared-axes step), so where they meet is known exactly: the curve through the mesh's
  // border points left gaps of 1e-4 to 5e-4 of the part on cylinder-plane, torus-plane and cylinder-torus edges (the
  // painted lab parts' face path), where the knitter builds the true line or circle. So, as the knitter does:
  // - a corner where only analytic faces meet goes onto all their surfaces (one least-squares point; a seam vertex onto
  //   its closing-round face's seam plane too), when that moves it by the fits' tolerance at most. Not where a fitted
  //   face meets them: its surface does not pass through the moved point, and five of the lab's sixteen face-path
  //   parts went invalid or untessellable that way;
  // - each border between two analytic faces goes onto their intersection: a tangent pair's contact line or circle,
  //   else every border point projected onto both; where those points make one line or circle (within 1e-9 of the
  //   part), that is the edge, written as such (faceStep.js: a LINE or CIRCLE, no pcurves: a reader projects those onto
  //   analytic surfaces exactly); else the curve through them, refined onto both until within 1e-9 between them.
  const givenOf = (gi) => (faces[gi].analytic ? surfaces.get(faces[gi].g.label) : null);
  const EXACT_TOL = 1e-9 * diag;
  {
    const facesAt = new Map(), seamPlanes = new Map();
    work.forEach((ch, c) => {
      for (const q of [ch.v0, ch.v1]) for (const u of usesOf[c]) (facesAt.get(q) ?? facesAt.set(q, new Set()).get(q)).add(u.g);
    });
    for (const [gi, at] of seamAt) {
      const f = faces[gi];
      if (!f.analytic) continue;
      for (const q of at.values()) (seamPlanes.get(q) ?? seamPlanes.set(q, []).get(q)).push(['plane', { origin: f.S.C, normal: f.S.y }]);
    }
    for (const [q, fs] of facesAt) {
      if ([...fs].some((gi) => !faces[gi].analytic)) continue;
      const surfs = [...[...fs].map(givenOf), ...(seamPlanes.get(q) ?? [])];
      const x = projectOnto(surfs, points[q], diag);
      // a CAD mesh's corner vertex is on the true corner: moved by the fits' error at most, else the fits meet
      // elsewhere (a loose torus fit put its corners 1e-3 of the part away, and the solid went invalid)
      if (Math.max(...surfs.map(([k, p]) => Math.abs(distNormal(k, p, x)[0]))) <= absTol && len(sub(x, points[q])) <= absTol) points[q] = x;
    }
    for (const ch of work) { ch.pts[0] = points[ch.v0]; if (!ch.ring) ch.pts[ch.pts.length - 1] = points[ch.v1]; }
  }
  work.forEach((ch, c) => {
    const us = usesOf[c];
    if (us.length !== 2 || !faces[us[0].g].analytic || !faces[us[1].g].analytic) return;
    const s1 = givenOf(us[0].g), s2 = givenOf(us[1].g);
    const locus = tangentLocus(s1, s2, diag);
    const onBothExact = (x) => (locus ? ontoLocus(locus, x) : projectOnto([s1, s2], x, diag));
    const off = (x) => Math.max(Math.abs(distNormal(s1[0], s1[1], x)[0]), Math.abs(distNormal(s2[0], s2[1], x)[0]));
    // (a ring's points do not repeat its first; a closed chain, a ring split at a seam, starts and ends on that vertex)
    const closedLoop = ch.ring || ch.closed, loop = (P) => (ch.ring ? [...P, P[0]] : P);
    const pts = ch.pts.map((x, i) => (i === 0 || (!ch.ring && i === ch.pts.length - 1) ? x : onBothExact(x) ?? x));
    // a CAD mesh's border vertices are on the true edge: onto both surfaces they move by the fits' error at most. More
    // means the surfaces meet elsewhere (two that nearly touch, not made tangent: their nearest common point runs off
    // along the border's normal, and 00004411_009's solid left its mesh); and the border keeps its curve
    if (pts.some((x, i) => off(x) > absTol || len(sub(x, ch.pts[i])) > absTol)) return;
    ch.pts = pts;
    // three points onto both inside each segment first: two or three points always make a circle (a short border
    // between a cylinder and an oblique plane passed as one, and the solid left the mesh)
    const dense = [];
    const L0 = loop(pts);
    for (let i = 0; i + 1 < L0.length; i++) {
      dense.push(L0[i]);
      const seg = len(sub(L0[i + 1], L0[i]));
      for (const w of [0.25, 0.5, 0.75]) {
        const x = L0[i].map((v, k) => v + w * (L0[i + 1][k] - v)), y = onBothExact(x);
        if (y && off(y) <= EXACT_TOL && len(sub(y, x)) <= seg / 4 + absTol) dense.push(y);
      }
    }
    dense.push(L0[L0.length - 1]);
    // a tangent pair's contact line or circle is its edge as it stands (its points are on both surfaces only to the
    // tangency's own tolerance, 1e-7 of the part, which a 1e-9 test of the points refused: 64 of 8 parts' borders)
    const ex = locus ? (locus.line ? { line: locus.line } : { circle: { ...locus.circle } }) : exactCurve(dense, EXACT_TOL);
    if (ex) {
      // the arc's way round: from the first point through the border's own points (a circle's normal turned to it)
      if (ex.circle) {
        const { c: C, n, r } = ex.circle, x0 = unit3(sub(pts[0], C)), y0 = cross(n, x0);
        const ang = (x) => { const d = sub(x, C); const a = Math.atan2(dot(d, y0), dot(d, x0)); return a < 0 ? a + 2 * Math.PI : a; };
        const mid = ang(pts[Math.max(1, pts.length >> 1)]), end = closedLoop ? 2 * Math.PI : ang(points[ch.v1]);
        const ccw = closedLoop ? ang(pts[1]) < Math.PI : mid < end;
        ch.exact = { circle: { c: C, n: ccw ? n : n.map((v) => -v), r, x: x0 } };
        const span = closedLoop ? 2 * Math.PI : ccw ? end : 2 * Math.PI - end;
        const N = ccw ? n : n.map((v) => -v), Y = cross(N, x0);
        ch.exactSpan = span;
        ch.sample = (t) => add3(C, add3(scale3(x0, r * Math.cos(t * span)), scale3(Y, r * Math.sin(t * span))));
      } else {
        const a = points[ch.v0], d = sub(points[ch.v1], a), L = len(d);
        ch.exact = { line: { o: a, d: unit3(d), len: L } };
        ch.sample = (t) => add3(a, scale3(d, t));
      }
      // the arc as written must run through the border's own points (not the rest of its circle): each point's
      // distance from the arc itself (measured against a polygon of it, its sag past the tolerance on large radii
      // refused 56 good circles)
      const on = (x) => {
        if (ch.exact.line) { const { o, d, len: L } = ch.exact.line, t = Math.max(0, Math.min(L, dot(sub(x, o), d))); return len(sub(x, add3(o, scale3(d, t)))); }
        const { c: C, n: N, r, x: X0 } = ch.exact.circle, Y = cross(N, X0), d = sub(x, C), z = dot(d, N), rad = sub(d, scale3(N, z));
        let a = Math.atan2(dot(rad, Y), dot(rad, X0));
        if (a < 0) a += 2 * Math.PI;
        if (a <= ch.exactSpan + 1e-9) return Math.hypot(len(rad) - r, z);
        return Math.min(len(sub(x, ch.sample(0))), len(sub(x, ch.sample(1))));
      };
      if (dense.every((x) => on(x) <= absTol)) return;
      delete ch.exact; delete ch.sample; delete ch.exactSpan;
    }
    // a general intersection: points onto both between the border's, until the curve through them is within
    // EXACT_TOL of both surfaces between them (four halvings at most)
    for (let round = 0; round < 4; round++) {
      const P = loop(ch.pts), V = loop(ch.vid), C = fitCurve(P, { closed: closedLoop, tol: EXACT_TOL }), d = [0];
      for (let i = 1; i < P.length; i++) d.push(d[i - 1] + len(sub(P[i], P[i - 1])));
      const outP = [P[0]], outV = [V[0]];
      let added = 0;
      for (let i = 0; i + 1 < P.length; i++) {
        const x = curveAt(C, (d[i] + d[i + 1]) / 2 / (d[d.length - 1] || 1))[0];
        if (off(x) > EXACT_TOL) { const y = onBothExact(x); if (y && off(y) <= EXACT_TOL && len(sub(y, x)) <= absTol) { outP.push(y); outV.push(-1); added++; } }
        outP.push(P[i + 1]); outV.push(V[i + 1]);
      }
      if (!added) break;
      ch.pts = ch.ring ? outP.slice(0, -1) : outP;
      ch.vid = ch.ring ? outV.slice(0, -1) : outV;
    }
    ch.onBothExact = true;
  });
  for (const ch of work) {
    const closedLoop = ch.ring || ch.closed;
    if (ch.sample) {
      // the exact curve's own points, enough for the B-spline the pcurves are fitted against to follow it closely
      const m = 64, ts = Array.from({ length: m + 1 }, (_, j) => j / m), P = ts.map((t) => ch.sample(t));
      if (closedLoop) P[m] = P[0];
      ch.curve = fitCurve(P, { ts, closed: closedLoop, tol: EXACT_TOL });
    } else ch.curve = fitCurve(ch.ring ? [...ch.pts, ch.pts[0]] : ch.pts, { closed: closedLoop, tol: ch.onBothExact ? EXACT_TOL : absTol });
  }

  // pcurves: the edge curve's points projected onto each of its faces' surfaces, fitted on the edge's knots
  let edgeTol = 0;
  // an edge along a side of a four- or three-sided face lies on the surface's boundary row there: its pcurve is that
  // side's line in the parameters, each sample placed along the row (the other parameter held). Which line: the held
  // coordinate its vertices share (the face's fit reports each border vertex's; a three-sided face's tip is on both
  // rails, its own held u being the row it collapses to). A free projection there is ill-posed at a collapsed tip.
  const isoOf = (face, ch) => {
    const so = face.sideOf;
    if (!so) return null;
    const { local } = meshes[face.gi];
    let cand = null, tipIn = false;
    for (const v of ch.vid) {
      if (v < 0) continue;
      const l = local.get(v);
      if (l === so.tip) { tipIn = true; continue; }
      const h = so.held.get(l);
      if (!h) return null;
      const here = [0, 1].filter((k) => !Number.isNaN(h[k])).map((k) => [k, h[k]]);
      cand = cand ? cand.filter(([k, x]) => here.some(([k2, x2]) => k2 === k && x2 === x)) : here;
      if (!cand.length) return null;
    }
    if (tipIn && cand) { const tu = so.held.get(so.tip)?.[0]; cand = cand.filter(([k, x]) => !(k === 0 && x !== tu)); }
    return cand?.length === 1 ? cand[0] : null;
  };
  // a pcurve on the edge curve's knots, each span split f ways: an edge whose curve is on both surfaces exactly (between
  // analytic faces) is refined until its pcurves are too (on its own knots they missed by 1e-5 of the part)
  const refined = (knots, f) => {
    if (f === 1) return knots;
    const u = [...new Set(knots)], lo = knots.filter((k) => k === knots[0]).length, out = [];
    for (let i = 0; i + 1 < u.length; i++) for (let j = 0; j < f; j++) out.push(u[i] + ((u[i + 1] - u[i]) * j) / f);
    return [...Array(lo - 1).fill(u[0]), ...out, ...Array(lo).fill(u[u.length - 1])];
  };
  const pcurveOn = (face, curve, ch) => {
    const target = ch.onBothExact || ch.exact ? 10 * EXACT_TOL : Infinity;
    let best = null;
    for (const f of [1, 2, 4, 8]) {
      const r = pcurveAt(face, curve, ch, f);
      if (r.error) return r;
      if (!best || r.tol < best.tol) best = r;
      if (best.tol <= target) break;
    }
    edgeTol = Math.max(edgeTol, best.tol);
    return best;
  };
  const pcurveAt = (face, curve, ch, f) => {
    const { S, periodic } = face, [u0, u1] = S.dom, P = u1 - u0;
    const knots = refined(curve.knots, f), spans = new Set(knots).size - 1, m = Math.max(24, 8 * spans);
    const ts = Array.from({ length: m + 1 }, (_, j) => j / m), uv = [];
    const iso = isoOf(face, ch);
    if (iso) {
      const [k, val] = iso, [a, b] = k === 0 ? [S.dom[2], S.dom[3]] : [S.dom[0], S.dom[1]];
      let t = null;
      for (const s of ts) {
        const x = curveAt(curve, s)[0];
        if (t === null) {                          // the first: the nearest of samples along the row, then refined
          let best = Infinity;
          for (let j = 0; j <= 64; j++) { const y = a + ((b - a) * j) / 64, q = k === 0 ? S.eval(val, y).S : S.eval(y, val).S, e = len(sub(q, x)); if (e < best) { best = e; t = y; } }
        }
        t = isoClosest(S, x, k, val, t, a, b);
        uv.push(k === 0 ? [val, t] : [t, val]);
      }
      const pc = fitCurve(uv, { ts, knots });
      let et = 0;
      for (const s of ts) et = Math.max(et, len(sub(S.eval(...curveAt(pc, s)[0]).S, curveAt(curve, s)[0])));
      return { pc, dir: Math.sign(uv[m][0] - uv[0][0]), tol: et, iso: true };
    }
    // the chain's points' chord-length parameters (as the curve was fitted), to find each sample's nearest vertex
    const pts = ch.ring ? [...ch.pts, ch.pts[0]] : ch.pts, vid = ch.ring ? [...ch.vid, ch.vid[0]] : ch.vid, d = [0];
    for (let i = 1; i < pts.length; i++) d.push(d[i - 1] + len(sub(pts[i], pts[i - 1])));
    const tv = d.map((x) => x / (d[d.length - 1] || 1));
    let prev = null, near = 0;
    for (const t of ts) {
      while (near + 1 < tv.length && Math.abs(tv[near + 1] - t) <= Math.abs(tv[near] - t)) near++;
      let sd = null;
      for (let o = 0; o < vid.length && !sd; o++) sd = seedFor(face, vid[near - o] ?? -1) ?? seedFor(face, vid[near + o] ?? -1);
      // the closer of the projections from the last sample and from the nearest vertex's place (either can stall
      // in a local minimum on the domain's edge when the other does not)
      const x = curveAt(curve, t)[0];
      let r = S.closest(x, prev ?? sd);
      if (prev && sd) { const r2 = S.closest(x, sd); if (len(sub(r2.S, x)) < len(sub(r.S, x))) r = r2; }
      let u = r.u;
      if (periodic && uv.length) { while (u - uv[uv.length - 1][0] > P / 2) u -= P; while (uv[uv.length - 1][0] - u > P / 2) u += P; }
      uv.push([u, r.v]);
      prev = [r.u, r.v];
    }
    if (periodic) {
      // inside the domain: an edge ending on the seam ends at u0 or u1, whichever side it comes from
      // (placed by its middle: an end on the seam may project to either side of it, and is snapped onto it)
      const lo2 = Math.min(...uv.map((q) => q[0])), hi2 = Math.max(...uv.map((q) => q[0]));
      const k = Math.floor(((lo2 + hi2) / 2 - u0) / P);
      if (hi2 - k * P > u1 + 1e-3 * P || lo2 - k * P < u0 - 1e-3 * P) return { error: 'an edge still crosses the seam of a face that closes round' };
      uv.forEach((q) => {
        q[0] -= k * P;
        if (Math.abs(q[0] - u0) < 1e-3 * P) q[0] = u0;
        if (Math.abs(q[0] - u1) < 1e-3 * P) q[0] = u1;
        q[0] = Math.min(u1, Math.max(u0, q[0]));
      });
    }
    if (S.periodicV) {
      // a torus's v (continuous along the curve, each sample seeded by the last) in the face's window, by its middle
      const mid = (Math.min(...uv.map((q) => q[1])) + Math.max(...uv.map((q) => q[1]))) / 2;
      const k = Math.round((mid - S.vRef) / (2 * Math.PI));
      if (k) for (const q of uv) q[1] -= 2 * Math.PI * k;
    }
    const pc = fitCurve(uv, { ts, knots });
    let et = 0;
    for (const t of ts) et = Math.max(et, len(sub(S.eval(...curveAt(pc, t)[0]).S, curveAt(curve, t)[0])));
    return { pc, dir: Math.sign(uv[m][0] - uv[0][0]), tol: et };
  };
  const edges = work.map((ch, c) => {
    const e = { curve: ch.curve, v0: ch.v0, v1: ch.v1, pcurves: [] };
    if (ch.exact) {
      // written as the line or circle itself; its gap, its own distance from both surfaces
      const [s1, s2] = usesOf[c].map((u) => givenOf(u.g));
      let gap = 0;
      for (let j = 0; j <= 32; j++) { const x = ch.sample(j / 32); for (const [k, p] of [s1, s2]) gap = Math.max(gap, Math.abs(distNormal(k, p, x)[0])); }
      Object.assign(e, { exact: ch.exact, exactGap: gap });
    }
    return e;
  });
  work.forEach((ch, c) => {
    for (const use of usesOf[c]) {
      const r = pcurveOn(faces[use.g], ch.curve, ch);
      if (r.error) { fail(faces[use.g].g.label, r.error); continue; }
      edges[c].pcurves.push({ face: use.g, pc: r.pc, rev: use.rev, dir: r.dir, tol: r.tol, iso: !!r.iso });
    }
  });
  if (problems.length) return stop();

  // each face's bounds; a closing-round face: one loop, its two borders joined by a seam edge on the closing line
  const out = faces.map((f) => ({ params: f.analytic ? f.S.stepParams() : f.params, sense: f.sense, bounds: [], label: f.g.label, info: f.info }));
  const pcOf = (e, gi) => edges[e.c].pcurves.find((x) => x.face === gi && x.rev === e.rev).pc;
  const uvAt = (e, gi, t) => curveAt(pcOf(e, gi), e.rev ? 1 - t : t)[0];
  for (const f of faces) {
    const loops = loopsOf[f.gi];
    // (an analytic face on a surface that closes round but not closing round itself, a fillet's quarter, has its
    // seam off it and is bounded as any other)
    if (!f.periodic || (f.analytic && !seamAt.get(f.gi)?.size)) {
      const areas = loops.map((entries) => {
        let a = 0;
        for (const e of entries) for (let j = 0; j < 16; j++) { const p = uvAt(e, f.gi, j / 16), q = uvAt(e, f.gi, (j + 1) / 16); a += p[0] * q[1] - p[1] * q[0]; }
        // (and across any gap between one edge's end and the next's start: a three-sided face's collapsed side)
        entries.forEach((e, i) => { const p = uvAt(e, f.gi, 1), q = uvAt(entries[(i + 1) % entries.length], f.gi, 0); a += p[0] * q[1] - p[1] * q[0]; });
        return Math.abs(a);
      });
      const outer = areas.indexOf(Math.max(...areas));
      loops.forEach((entries, li) => out[f.gi].bounds.push({ outer: li === outer, edges: entries.map((e) => ({ e: e.c, forward: !e.rev })) }));
      // a fitted face's share of its surface's domain (the rest of the net hangs outside the face)
      if (!f.analytic) {
        const [u0, u1, v0, v1] = f.S.dom;
        f.info.cover = (areas[outer] - areas.reduce((s, a, li) => (li === outer ? s : s + a), 0)) / 2 / ((u1 - u0) * (v1 - v0));
      }
      continue;
    }
    // its two rims, each from its seam vertex; A: the one whose u rises along it (it ends on the far side of the
    // seam); any other loop a hole in the wall
    const at = seamAt.get(f.gi), rims = [...at.keys()];
    if (rims.length !== 2) { fail(f.g.label, `closes round with ${rims.length} rims`); continue; }
    const q = rims.map((li) => at.get(li));
    const fromSeam = rims.map((li, r) => {
      const l = loops[li], i = l.findIndex((e) => (e.rev ? edges[e.c].v1 : edges[e.c].v0) === q[r]);
      return l.slice(i).concat(l.slice(0, i));
    });
    const rise = fromSeam.map((l) => uvAt(l[l.length - 1], f.gi, 1)[0] - uvAt(l[0], f.gi, 0)[0]);
    const [A, B, qa, qb] = rise[0] > 0 ? [fromSeam[0], fromSeam[1], q[0], q[1]] : [fromSeam[1], fromSeam[0], q[1], q[0]];
    const { S } = f, [u0, u1] = S.dom;
    const va = points[qa], vb = points[qb];
    // the seam's ends where the rims' own pcurves meet it (a fresh projection of the seam vertices can land on
    // another part of the surface, and the wire would not close)
    const wa = uvAt(A[0], f.gi, 0)[1], wb = uvAt(B[0], f.gi, 0)[1];
    const m = 24, ts = Array.from({ length: m + 1 }, (_, j) => j / m);
    const samples = ts.map((t, j) => (j === 0 ? va : j === m ? vb : S.eval(u0, wa + t * (wb - wa)).S));
    const curve = fitCurve(samples, { ts, tol: absTol });
    const line = (u) => ({ p: 1, knots: [0, 0, 1, 1], poles: [[u, wa], [u, wb]] });
    const seam = edges.length;
    edges.push({ curve, v0: qa, v1: qb, seam: { face: f.gi, pcurves: [line(u1), line(u0)] } });
    out[f.gi].bounds.push({ outer: true, edges: [...A.map((e) => ({ e: e.c, forward: !e.rev })), { e: seam, forward: true }, ...B.map((e) => ({ e: e.c, forward: !e.rev })), { e: seam, forward: false }] });
    loops.forEach((l, li) => { if (!at.has(li)) out[f.gi].bounds.push({ outer: false, edges: l.map((e) => ({ e: e.c, forward: !e.rev })) }); });
  }
  if (problems.length) return stop();
  // the loosest edges: each edge's worst distance from either face's surface, with the faces it joins
  const loose = edges.filter((e) => e.pcurves?.length).map((e) => ({ tol: Math.max(...e.pcurves.map((p) => p.tol)) / diag,
    faces: e.pcurves.map((p) => faces[p.face].g.label), on: e.pcurves.map((p) => (faces[p.face].analytic ? faces[p.face].S.type : `fit (${faces[p.face].info.param})`)) }));
  loose.sort((a, b) => b.tol - a.tol);
  // the edges by what meets there, each side 'side' (a side of a four- or three-sided face's net), 'trimmed' (any
  // other fitted face), or its analytic type: how many, and their gaps (each edge's worst distance from either face)
  const edgeGaps = {};
  for (const e of edges) {
    if (e.pcurves?.length !== 2) continue;
    const k = e.pcurves.map((p) => (faces[p.face].analytic ? faces[p.face].S.type : p.iso ? 'side' : 'trimmed')).sort().join('-');
    const g = edgeGaps[k] ?? (edgeGaps[k] = { n: 0, gaps: [] });
    g.n++;
    g.gaps.push((e.exact ? e.exactGap : Math.max(...e.pcurves.map((p) => p.tol))) / diag);
    if (e.exact) g.exact = (g.exact ?? 0) + 1;
  }

  // the volume between the mesh (its triangles are chords) and the faces' surfaces: per triangle its area times the
  // surface's mean offset outward from it, the mean of the offsets at its sides' midpoints (exact for a quadratic
  // bulge). A solid on these surfaces has the mesh's volume plus this; what a volume check sees beyond it is no fit.
  let bulge = 0;
  for (const f of faces) {
    const sgn = f.sense ? 1 : -1;
    for (const ti of f.g.tris) {
      const [a, b, c] = triangles[ti].map((v) => vertices[v]), A = len(cross(sub(b, a), sub(c, a))) / 2;
      if (!(A > 0)) continue;
      let d = 0;
      for (const [p, q] of [[a, b], [b, c], [c, a]]) {
        const mid = p.map((x, k) => (x + q[k]) / 2), r = f.S.closest(mid, seedFor(f, triangles[ti][0]));
        d += dot(sub(r.S, mid), f.S.normal(r.u, r.v)[1]) * sgn / 3;
      }
      bulge += A * d;
    }
  }
  const report = {
    bulgeVolume: bulge,
    looseEdges: loose.slice(0, 10),
    edgeGaps,
    faces: out.length, edges: edges.length, maxFit: Math.max(...faces.map((f) => f.info.max_err)), edgeTol: edgeTol / diag,
    ms: performance.now() - t0,
  };
  return { faces: out, edges, points, tolerance: Math.max(edgeTol * 1.5, 1e-7 * diag), report };
}

/**
 * faceSolid, splitting only the faces no single surface covers: a face whose map, fit or seam fails is cut in two
 * where its normals turn most (freeformFit's splitNormals: normal-cone clustering) and the solid built again, the
 * other faces' fits kept; up to `rounds` times. A painted face goes out as one STEP face wherever one surface can
 * carry it, and as the fewest pieces this finds where not. A face whose given surface (opts.surfaces) fails it (a
 * cone typed onto a few triangles across its apex) is fitted freely instead, before any split. split: faces to cut
 * before the first build, each given by its triangles (indices into triangles: the same on every run, so a later
 * call can name a piece an earlier one made, faceExport's kernel verdicts): those triangles become a face of their
 * own and it is cut. Out as faceSolid's, each face with `origin` (its painted label) and `tris` (its triangles),
 * `splits`: [{ origin, pieces }] and `retyped`: [[label, given type]].
 */
export function faceSolidSplit(vertices, triangles, labels, { rounds = 4, split = null, ...opts } = {}) {
  const lab = Array.from(labels), origin = new Map(), fits = new Map(), splits = new Map(), retyped = [];
  const surfaces = new Map(opts.surfaces ?? []);
  let next = 1, r = null;
  for (const l of lab) if (l + 1 > next) next = l + 1;   // (a loop: too many triangles to spread into Math.max in a browser)
  const cutFace = (label) => {
    const tris = [];
    lab.forEach((l, t) => { if (l === label) tris.push(t); });
    const pieces = tris.length > 1 ? splitNormals(vertices, triangles, tris) : [tris];
    if (pieces.length < 2) return false;
    const root = origin.get(label) ?? label;
    for (const piece of pieces) { for (const t of piece) lab[t] = next; origin.set(next, root); next++; }
    splits.set(root, (splits.get(root) ?? 1) - 1 + pieces.length);
    return true;
  };
  for (const tris of split ?? []) {
    const was = labels[tris[0]];
    if (surfaces.has(was)) continue;              // a given surface is not cut
    const label = next++;
    for (const t of tris) lab[t] = label;
    origin.set(label, was);
    cutFace(label);
  }
  for (let round = 0; ; round++) {
    r = faceSolid(vertices, triangles, lab, { ...opts, surfaces, fits });
    if (!r.error || !r.failed?.length || round === rounds) break;
    let cut = false;
    for (const label of r.failed) {
      if (surfaces.has(label)) { retyped.push([label, surfaces.get(label)[0]]); surfaces.delete(label); cut = true; continue; }
      if (cutFace(label)) cut = true;
    }
    if (!cut) break;
  }
  const out = { ...r, splits: [...splits].map(([o, pieces]) => ({ origin: o, pieces })), retyped };
  const trisOf = new Map();
  lab.forEach((l, t) => (trisOf.get(l) ?? trisOf.set(l, []).get(l)).push(t));
  for (const f of r.faces ?? []) { f.origin = origin.get(f.label) ?? f.label; f.tris = trisOf.get(f.label) ?? []; }
  return out;
}
