/**
 * faceSolid.js — a mesh painted in groups (a part's faces) as a solid of trimmed B-spline faces, one face per
 * group, the way CAD models are built:
 * - each face one bicubic B-spline surface a little larger than the face (js/freeformFit.js fitGroup, given points
 *   inside the face's long triangles too, so a CAD face's mesh is data all over it);
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

const sub = (a, b) => a.map((x, i) => x - b[i]);
const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const len = (a) => Math.hypot(...a);

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
    extraOf[gF].push(...extra);
    onA.set(c, { ins, g: gF });
    analyticNbrs[gF].add(groups[ua.g === gF ? ub.g : ua.g].label);
  });

  // one surface per group
  const faces = groups.map((g, gi) => {
    const { local, Vg, Tg } = meshes[gi];
    // surfaces: the app's own for a group (its analytic types exact, as given: only framed for this face, its vertices'
    // parameters their projections); any other group is fitted here. The mesh's vertices are the data (a CAD mesh's
    // are on the true surface; its triangles' insides are chords), with its borders on analytic faces. fits: a cache
    // across calls that differ only in other faces (faceSolidSplit's retries), by label and analytic neighbours.
    const given = surfaces?.get(g.label) ?? null, analytic = !!analyticOf[gi];
    let fit, S;
    if (given) {
      S = analytic ? analyticOf[gi] : Surface.fromParams(given[1]);
      const vertexUV = new Map();
      let maxErr = 0, sq = 0;
      Vg.forEach((q, l) => { const r = S.closest(q); vertexUV.set(l, [r.u, r.v]); const e = len(sub(r.S, q)); maxErr = Math.max(maxErr, e); sq += e * e; });
      fit = { params: analytic ? null : given[1], vertexUV, info: { param: given[0], spans: [0, 0], max_err: maxErr / diag, rms_err: Math.sqrt(sq / Vg.length) / diag } };
    } else {
      const key = `${g.label}|${[...analyticNbrs[gi]].sort((a, b) => a - b).join(',')}`;
      fit = fits?.get(key) ?? fitGroup(Vg, Tg, { scale: diag, extra: extraOf[gi] });
      fits?.set(key, fit);
      if (!fit.params) { fail(g.label, `no single surface fits it (${fit.info.fail})`); return null; }
      S = Surface.fromParams(fit.params);
    }
    const periodic = analytic ? S.periodicU : !!fit.params.u_periodic;
    // which way the surface faces: its normal at each vertex's place on it against the mesh's (outward, area
    // weighted) there, by area. Where it faces the other way under more than one triangle's worth of the face, it
    // folds under it (a fit across a cusp or crease the face has inside it): no surface for this face, which
    // faceSolidSplit then cuts in two. (Not counted: vertices where the mesh itself folds over, meshNormals.)
    const { vn, va, trusted, area } = meshNormals(Vg, Tg);
    let vote = 0, against = 0, measured = 0;
    for (const [l, uv] of fit.vertexUV) {
      if (!trusted[l]) continue;
      const s1 = Math.sign(dot(vn[l], S.normal(uv[0], uv[1])[1]));
      vote += s1 * va[l]; measured += va[l];
      if (s1 < 0) against += va[l];
    }
    const folded = Math.min(against, measured - against);
    if (folded > area / Tg.length) { fail(g.label, `its surface folds under it (${((100 * folded) / area).toFixed(1)} % of its area)`); return null; }
    const seed = new Map();
    for (const [v, l] of local) { const uv = fit.vertexUV.get(l); if (uv) seed.set(v, uv); }
    return { g, gi, S, params: fit.params, periodic, analytic, sense: vote >= 0, info: fit.info, seed, shift: 0 };
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
  for (const ch of work) {
    if (ch.ring) { const q = points.length; points.push(ch.pts[0]); ch.v0 = ch.v1 = q; ch.curve = fitCurve([...ch.pts, ch.pts[0]], { closed: true, tol: absTol }); }
    else ch.curve = fitCurve(ch.pts, { closed: ch.closed, tol: absTol });
  }
  const usesOf = work.map(() => []);
  loopsOf.forEach((loops, gi) => loops.forEach((l) => l.forEach((e) => usesOf[e.c].push({ g: gi, rev: e.rev }))));

  // pcurves: the edge curve's points projected onto each of its faces' surfaces, fitted on the edge's knots
  let edgeTol = 0;
  const pcurveOn = (face, curve, ch) => {
    const { S, periodic } = face, [u0, u1] = S.dom, P = u1 - u0;
    const spans = new Set(curve.knots).size - 1, m = Math.max(24, 8 * spans);
    const ts = Array.from({ length: m + 1 }, (_, j) => j / m), uv = [];
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
    const pc = fitCurve(uv, { ts, knots: curve.knots });
    let et = 0;
    for (const t of ts) et = Math.max(et, len(sub(S.eval(...curveAt(pc, t)[0]).S, curveAt(curve, t)[0])));
    edgeTol = Math.max(edgeTol, et);
    return { pc, dir: Math.sign(uv[m][0] - uv[0][0]), tol: et };
  };
  const edges = work.map((ch) => ({ curve: ch.curve, v0: ch.v0, v1: ch.v1, pcurves: [] }));
  work.forEach((ch, c) => {
    for (const use of usesOf[c]) {
      const r = pcurveOn(faces[use.g], ch.curve, ch);
      if (r.error) { fail(faces[use.g].g.label, r.error); continue; }
      edges[c].pcurves.push({ face: use.g, pc: r.pc, rev: use.rev, dir: r.dir, tol: r.tol });
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
        return Math.abs(a);
      });
      const outer = areas.indexOf(Math.max(...areas));
      loops.forEach((entries, li) => out[f.gi].bounds.push({ outer: li === outer, edges: entries.map((e) => ({ e: e.c, forward: !e.rev })) }));
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
 * cone typed onto a few triangles across its apex) is fitted freely instead, before any split. Out as faceSolid's,
 * each face with `origin` (its painted label), `splits`: [{ origin, pieces }] and `retyped`: [[label, given type]].
 */
export function faceSolidSplit(vertices, triangles, labels, { rounds = 4, ...opts } = {}) {
  const lab = Array.from(labels), origin = new Map(), fits = new Map(), splits = new Map(), retyped = [];
  const surfaces = new Map(opts.surfaces ?? []);
  let next = Math.max(...lab) + 1, r = null;
  for (let round = 0; ; round++) {
    r = faceSolid(vertices, triangles, lab, { ...opts, surfaces, fits });
    if (!r.error || !r.failed?.length || round === rounds) break;
    let cut = false;
    for (const label of r.failed) {
      if (surfaces.has(label)) { retyped.push([label, surfaces.get(label)[0]]); surfaces.delete(label); cut = true; continue; }
      const tris = [];
      lab.forEach((l, t) => { if (l === label) tris.push(t); });
      const pieces = tris.length > 1 ? splitNormals(vertices, triangles, tris) : [tris];
      if (pieces.length < 2) continue;
      const root = origin.get(label) ?? label;
      for (const piece of pieces) { for (const t of piece) lab[t] = next; origin.set(next, root); next++; }
      splits.set(root, (splits.get(root) ?? 1) - 1 + pieces.length);
      cut = true;
    }
    if (!cut) break;
  }
  const out = { ...r, splits: [...splits].map(([o, pieces]) => ({ origin: o, pieces })), retyped };
  for (const f of r.faces ?? []) f.origin = origin.get(f.label) ?? f.label;
  return out;
}
