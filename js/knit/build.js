/**
 * knit/build.js — the B-rep from a labelled mesh and its fitted surfaces, in OpenCASCADE (opencascade.js
 * 2.x, passed in as `oc`). A port of ml/solid_rebuild.py build_topo / score / patch_problems /
 * build_polyhedral; the reasons behind each step are recorded there with their measurements.
 *
 *   corner  a mesh vertex where >= 3 patches of one body meet, placed on ALL its patches' surfaces
 *   edge    each run of boundary between two patches, corner to corner: ONE curve through the boundary
 *           vertices projected onto both surfaces, shared by the two faces (straight between planes
 *           and along facets)
 *   face    the patch's surface, U-reversed where its natural normal points into the solid, bounded by
 *           its edges in mesh winding
 *   repair  one ShapeFix_Shape over the analytic faces, then natural-bound wires it added are dropped; facets
 *           join after it, with the edges and vertices it replaced
 */

import { distNormal, projectOnto, seamAway, unfoldChain, interpolateChain, unit, sub, add, dot, scale, norm, cross,
  tangentLocus, ontoLocus, cornerOnLocus, exactCurve, evalBSpline } from './geometry.js';
import { meshTopology, BODY_STRIDE } from './mesh.js';
import { ssiNow } from './ssi.js';

const bump = (info, k, n = 1) => { info[k] = (info[k] ?? 0) + n; };
// BUILD.PROF = {}: ms per build stage, summed over every build (profiling only)
const tick = (k, t) => { if (BUILD.PROF) BUILD.PROF[k] = (BUILD.PROF[k] ?? 0) + performance.now() - t; return performance.now(); };

function pnt(oc, p) { return new oc.gp_Pnt_3(p[0], p[1], p[2]); }
function dir(oc, v) { return new oc.gp_Dir_4(v[0], v[1], v[2]); }
function enumOf(oc) {
  return {
    FACE: oc.TopAbs_ShapeEnum.TopAbs_FACE, EDGE: oc.TopAbs_ShapeEnum.TopAbs_EDGE, WIRE: oc.TopAbs_ShapeEnum.TopAbs_WIRE,
    SHELL: oc.TopAbs_ShapeEnum.TopAbs_SHELL, SHAPE: oc.TopAbs_ShapeEnum.TopAbs_SHAPE, VERTEX: oc.TopAbs_ShapeEnum.TopAbs_VERTEX,
    REVERSED: oc.TopAbs_Orientation.TopAbs_REVERSED,
    IN: oc.TopAbs_State.TopAbs_IN,
  };
}

/** Every sub-shape of `shape` of one type, in explorer order. */
export function subShapes(oc, shape, type) {
  const out = [];
  const ex = new oc.TopExp_Explorer_2(shape, type, oc.TopAbs_ShapeEnum.TopAbs_SHAPE);
  while (ex.More()) { out.push(ex.Current()); ex.Next(); }
  ex.delete();
  return out;
}

/** The OCCT surface for a fitted patch (gaussMap params), framed with its seam away from the patch. */
export function geomSurface(oc, kind, p, patchPts) {
  if (kind === 'bspline') return bsplineSurface(oc, p);
  const c = [0, 0, 0];
  for (const q of patchPts) for (let k = 0; k < 3; k++) c[k] += q[k] / patchPts.length;
  let s;
  if (kind === 'plane') {
    s = new oc.Geom_Plane_3(pnt(oc, p.origin), dir(oc, unit(p.normal)));
  } else if (kind === 'sphere') {
    const mean = unit(patchPts.reduce((acc, q) => [acc[0] + q[0] - p.center[0], acc[1] + q[1] - p.center[1], acc[2] + q[2] - p.center[2]], [0, 0, 0]));
    const ref = Math.abs(mean[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
    const axis = unit(cross(mean, ref));           // poles away from the patch
    const x = seamAway(axis, mean);
    s = new oc.Geom_SphericalSurface_1(new oc.gp_Ax3_3(pnt(oc, p.center), dir(oc, axis), dir(oc, x)), p.radius);
  } else {
    let a = unit(p.axis);
    let o = kind === 'cylinder' ? p.axisPoint : kind === 'cone' ? p.apex : p.center;
    let x = seamAway(a, sub(c, o));
    if (kind === 'cone') {
      let m = 0;
      for (const q of patchPts) m += dot(sub(q, o), a);
      if (m < 0) { a = scale(a, -1); x = seamAway(a, sub(c, o)); }
      s = new oc.Geom_ConicalSurface_1(new oc.gp_Ax3_3(pnt(oc, o), dir(oc, a), dir(oc, x)), p.halfAngle, 0);
    } else if (kind === 'cylinder') {
      // Frame origin on the axis beside the patch: equivalent fits then give equivalent faces.
      o = [o[0] + dot(sub(c, o), a) * a[0], o[1] + dot(sub(c, o), a) * a[1], o[2] + dot(sub(c, o), a) * a[2]];
      s = new oc.Geom_CylindricalSurface_1(new oc.gp_Ax3_3(pnt(oc, o), dir(oc, a), dir(oc, x)), p.radius);
    } else if (kind === 'torus') {
      s = new oc.Geom_ToroidalSurface_1(new oc.gp_Ax3_3(pnt(oc, o), dir(oc, a), dir(oc, x)), p.majorRadius, p.minorRadius);
    } else {
      throw new Error(`no surface for ${kind}`);
    }
  }
  return new oc.Handle_Geom_Surface_2(s);
}

/**
 * Geom_BSplineSurface for js/freeformFit.js params (flat knots -> distinct knots + multiplicities).
 * u_periodic params (rings and tubes) hold the unclamped form, the first pu rows of poles repeated at
 * the end; OCCT gets the distinct poles and the domain's knots with multiplicity 1, periodic in u, and
 * treats the face like a cylinder's (nurbs.py to_occ).
 */
function bsplineSurface(oc, p) {
  if (p.u_periodic) {
    const pu = p.pu ?? 3, n = p.poles.length - pu;
    return bsplineSurface(oc, { ...p, poles: p.poles.slice(0, n), tu: p.tu.slice(pu, pu + n + 1), u_periodic: false, periodicU: true });
  }
  const nu = p.poles.length, nv = p.poles[0].length;
  const P = new oc.TColgp_Array2OfPnt_2(1, nu, 1, nv);
  for (let i = 0; i < nu; i++) for (let j = 0; j < nv; j++) P.SetValue(i + 1, j + 1, pnt(oc, p.poles[i][j]));
  const knots = (t) => {
    const k = [], m = [];
    for (const x of t) {
      if (k.length && Math.abs(x - k[k.length - 1]) <= 1e-14 * Math.max(1, Math.abs(x))) m[m.length - 1]++;
      else { k.push(x); m.push(1); }
    }
    const K = new oc.TColStd_Array1OfReal_2(1, k.length), M = new oc.TColStd_Array1OfInteger_2(1, m.length);
    k.forEach((x, i) => K.SetValue(i + 1, x));
    m.forEach((x, i) => M.SetValue(i + 1, x));
    return [K, M];
  };
  const [ku, mu] = knots(p.tu), [kv, mv] = knots(p.tv);
  return new oc.Handle_Geom_Surface_2(new oc.Geom_BSplineSurface_1(P, ku, kv, mu, mv, p.pu ?? 3, p.pv ?? 3, !!p.periodicU, false));
}

const EXACT_EDGE_TOL = 1e-8;
const CONTACT_TOL = 1e-7;
/** Switches for experiments (test/knit.mjs): exact line / circle edges, contact loci, ssi's edges, exact pinches,
 * pcurves for spline edges, facets through the repair. */
export const BUILD = { EXACT_EDGES: true, CONTACT: true, SSI: true, PINCH: true, PCURVES: true, FIX_FACETS: false, FACET_ARCS: true };

/**
 * Where a facet meets a curved face: the curve its plane cuts that surface in, between the two corners (on both),
 * as points in the facet's plane, or null where the chord will do (it is on the surface, or the cut is not a
 * small bulge off it). A chord is no edge of the curved face, which covers it by tolerance, and that broke the
 * neighbours of facets in turn (each hybrid round demoted the curved faces beside the last one's facets).
 */
function facetArc(q, p0, p1, [kind, prm], scaleLen, k = 6) {
  const n = unit(cross(sub(q[1], q[0]), sub(q[2], q[0])));
  const ch = sub(p1, p0);
  if (!(norm(ch) > 1e-9 * scaleLen) || !n.every(Number.isFinite)) return null;
  const d = unit(cross(n, ch));
  const height = Math.max(...q.map((x) => Math.abs(dot(sub(x, p0), d))));
  const pts = [p0];
  let worst = 0;
  for (let i = 1; i < k; i++) {
    const m = add(p0, scale(ch, i / k));
    let s = 0;
    for (let it = 0; it < 30; it++) {
      const [dist, nrm] = distNormal(kind, prm, add(m, scale(d, s)));
      const den = dot(nrm, d);
      if (Math.abs(den) < 0.1) return null;          // the surface runs nearly along the plane here
      s -= dist / den;
      if (Math.abs(dist) <= 1e-12 * scaleLen) break;
    }
    const x = add(m, scale(d, s));
    if (!(Math.abs(distNormal(kind, prm, x)[0]) <= 1e-10 * scaleLen)) return null;
    worst = Math.max(worst, Math.abs(s));
    pts.push(x);
  }
  pts.push(p1);
  if (worst <= 1e-9 * scaleLen || worst > 0.25 * height) return null;
  return pts;
}

/**
 * Whether the arc `cut` (facetArc's, from one corner of the facet q to another) crosses the facet's other two sides
 * or one of `others` (the arcs it already has), in the facet's plane. The height bound above caps how far an arc
 * bulges, not the angle it leaves a corner at: into a facet with a sharp corner it crossed the side there, and the
 * facet was invalid (00143363_15: one facet of a plane spoiled the hybrid build that was valid with chords).
 */
function arcCrosses(cut, q, others, scaleLen) {
  const p0 = cut[0], p1 = cut[cut.length - 1];
  const n = unit(cross(sub(q[1], q[0]), sub(q[2], q[0]))), e1 = unit(sub(p1, p0)), e2 = cross(n, e1);
  const at = (x) => { const w = sub(x, p0); return [dot(w, e1), dot(w, e2)]; };
  const c = q.reduce((m, x) => (Math.abs(dot(sub(x, p0), e2)) > Math.abs(dot(sub(m, p0), e2)) ? x : m));
  const eps = 1e-12 * scaleLen * scaleLen;
  const side = (a, b, x) => (b[0] - a[0]) * (x[1] - a[1]) - (b[1] - a[1]) * (x[0] - a[0]);
  const meet = (a, b, x, y) => {
    const s1 = side(a, b, x), s2 = side(a, b, y), s3 = side(x, y, a), s4 = side(x, y, b);
    return ((s1 > eps && s2 < -eps) || (s1 < -eps && s2 > eps)) && ((s3 > eps && s4 < -eps) || (s3 < -eps && s4 > eps));
  };
  const P = cut.map(at), C = at(c), A = at(p0), B = at(p1);
  const same = (x, y) => Math.hypot(x[0] - y[0], x[1] - y[1]) <= 1e-9 * scaleLen;
  const lines = [[A, C], [B, C], ...others.map((o) => o.map(at))];
  for (const L of lines) {
    for (let i = 0; i + 1 < P.length; i++) for (let j = 0; j + 1 < L.length; j++) {
      // segments meeting at a corner they share are not a crossing
      if ([P[i], P[i + 1]].some((x) => same(x, L[j]) || same(x, L[j + 1]))) continue;
      if (meet(P[i], P[i + 1], L[j], L[j + 1])) return true;
    }
  }
  return false;
}

/** An edge on an exact line or circle (exactCurve) from v0 to v1, following the chain's direction. */
function exactEdge(oc, ex, pts, closed, v0, v1) {
  if (ex.line) return closed ? null : new oc.BRepBuilderAPI_MakeEdge_2(v0, v1);
  let { c, n, x } = ex.circle;
  const angle = (q, nn) => {
    const w = sub(q, c), y = cross(nn, x);
    let t = Math.atan2(dot(w, y), dot(w, x));
    return t < 0 ? t + 2 * Math.PI : t;
  };
  // the chain runs with increasing angle: its second point just past 0, not just short of 2 pi
  if (angle(pts[1], n) > Math.PI) n = scale(n, -1);
  const t1 = closed ? 2 * Math.PI : angle(pts[pts.length - 1], n);
  if (!(t1 > 1e-9)) return null;
  const ax = new oc.gp_Ax2_2(pnt(oc, c), dir(oc, n), dir(oc, x));
  const crv = new oc.Geom_Circle_2(ax, ex.circle.r);
  ax.delete();
  return new oc.BRepBuilderAPI_MakeEdge_29(new oc.Handle_Geom_Curve_2(crv), v0, v1, 0, t1);
}

/** A B-spline edge through `pts` between two vertices. */
function chainEdge(oc, pts, closed, v0, v1, out = null) {
  const { poles, knots, mults, degree, t } = interpolateChain(pts, closed);
  if (out) { out.t = t ?? knots; out.curve = { poles, knots, mults, degree }; }
  const P = new oc.TColgp_Array1OfPnt_2(1, poles.length);
  poles.forEach((q, i) => P.SetValue(i + 1, pnt(oc, q)));
  const K = new oc.TColStd_Array1OfReal_2(1, knots.length);
  knots.forEach((k, i) => K.SetValue(i + 1, k));
  const M = new oc.TColStd_Array1OfInteger_2(1, mults.length);
  mults.forEach((m, i) => M.SetValue(i + 1, m));
  const crv = new oc.Geom_BSplineCurve_1(P, K, M, degree, false);
  return new oc.BRepBuilderAPI_MakeEdge_29(new oc.Handle_Geom_Curve_2(crv), v0, v1, knots[0], knots[knots.length - 1]);
}

/**
 * EXACT EDGES FROM SSI (knit/ssi.js). Each pair of analytic surfaces is intersected once, as ssi patches over
 * their groups' vertices and SSI.PAD x diagonal more, and an edge between them follows that curve from corner to
 * corner instead of a spline through the mesh's own points, which a coarse mesh leaves few of (Onshape meshed
 * 0.57 mm holes as squares, 00004314_026: the spline through 4 points bulged 85 µm). The curve must pass within
 * SSI.FOLLOW x diagonal of every point of the chain, which are on both surfaces too, or the chain keeps them: so
 * does a pair with a B-spline side, a cone over its apex, and a pair ssi refuses (a contact it cannot prove).
 * Returns (a, b, pts, closed) -> the edge's points, its corners first and last, or null.
 */
export const SSI = { PAD: 0.05, FOLLOW: 1e-4, SAG: 0.05, CORNER: 1e-2 };

function ssiArcs(mesh, surf, patchTris, info, cache = new Map()) {
  const ssi = ssiNow();
  if (!ssi) return () => null;
  const { V, T, diag } = mesh;
  const pad = SSI.PAD * diag, follow = SSI.FOLLOW * diag;
  const perp = (d) => unit(cross(d, Math.abs(d[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0]));
  const vertsOf = (g) => {
    const seen = new Set();
    for (const t of patchTris.get(g) ?? []) for (const v of T[t]) seen.add(v);
    return [...seen].map((v) => V[v]);
  };
  const range = (vals) => [Math.min(...vals) - pad, Math.max(...vals) + pad];
  // the shortest arc (degrees) holding every angle (radians), `more` radians wider each way: a whole surface would
  // set ssi's resolution by its own size (a torus fitted with R 2537 to a 50 mm part's fillet, 00003838_001)
  const arc = (angles, more) => {
    const deg = angles.map((t) => ((t * 180) / Math.PI + 360) % 360).sort((x, y) => x - y);
    let gap = 360 - deg[deg.length - 1] + deg[0], from = deg[0];
    for (let i = 1; i < deg.length; i++) if (deg[i] - deg[i - 1] > gap) { gap = deg[i] - deg[i - 1]; from = deg[i]; }
    const wide = 360 - gap + (2 * more * 180) / Math.PI;
    if (wide >= 360) return [0, 360];
    const a0 = from - (more * 180) / Math.PI;
    return [a0, a0 + wide];
  };
  const around = (q, o, a, x, y) => { const w = sub(q, o); return Math.atan2(dot(w, y), dot(w, x)); };
  const spec = (g) => {
    const [kind, p] = surf(g) ?? [];
    const pts = vertsOf(g);
    if (!pts.length) return null;
    if (kind === 'plane') {
      const n = unit(p.normal), x = perp(n), y = cross(n, x);
      const [u0, u1] = range(pts.map((q) => dot(sub(q, p.origin), x))), [v0, v1] = range(pts.map((q) => dot(sub(q, p.origin), y)));
      return [ssi.KIND.plane, ...p.origin, ...x, ...y, u0, u1, v0, v1];
    }
    if (kind === 'cylinder') {
      const a = unit(p.axis), x = perp(a), y = cross(a, x);
      return [ssi.KIND.cylinder, ...p.axisPoint, ...x, ...y, p.radius, ...arc(pts.map((q) => around(q, p.axisPoint, a, x, y)), pad / p.radius),
        ...range(pts.map((q) => dot(sub(q, p.axisPoint), a)))];
    }
    if (kind === 'cone') {
      let a = unit(p.axis);
      let h = pts.map((q) => dot(sub(q, p.apex), a));
      if (Math.max(...h) <= 0) { a = scale(a, -1); h = h.map((v) => -v); }
      if (Math.min(...h) <= 0) return null;                   // both nappes: not one ssi patch
      const x = perp(a), y = cross(a, x), [h0, h1] = range(h);
      const [a0, a1] = arc(pts.map((q) => around(q, p.apex, a, x, y)), pad / (Math.min(...h) * Math.tan(p.halfAngle) || 1));
      if (a0 !== 0 || a1 !== 360) return null;                   // ssi's cone is whole about its axis
      return [ssi.KIND.cone, ...p.apex, ...x, ...y, (p.halfAngle * 180) / Math.PI, Math.max(h0, Math.min(...h) / 2), h1];
    }
    if (kind === 'sphere') {
      const x = [1, 0, 0], y = [0, 1, 0];
      return [ssi.KIND.sphere, ...p.center, ...x, ...y, p.radius, 0, 360, -90, 90];
    }
    if (kind === 'torus') {
      const a = unit(p.axis), x = perp(a), y = cross(a, x);
      const R = p.majorRadius, r = p.minorRadius;
      const minor = pts.map((q) => { const w = sub(q, p.center), z = dot(w, a); return Math.atan2(z, Math.hypot(dot(w, x), dot(w, y)) - R); });
      return [ssi.KIND.torus, ...p.center, ...x, ...y, R, r, ...arc(pts.map((q) => around(q, p.center, a, x, y)), pad / R), ...arc(minor, pad / r)];
    }
    return null;                                              // a B-spline: not given to ssi yet
  };
  // one intersection per pair, kept across builds that give the pair the same surfaces (`cache`)
  const curvesOf = new Map();
  const pairCurves = (a, b) => {
    const key = a < b ? `${a}|${b}` : `${b}|${a}`;
    const held = cache.get(key);
    if (held && held.sa === surf(a < b ? a : b) && held.sb === surf(a < b ? b : a)) return held.curves;
    if (!curvesOf.has(key)) {
      let curves = null;
      const sa = spec(a), sb = spec(b);
      if (sa && sb) {
        try {
          curves = ssi.intersect(sa, sb).curves.map((c) => ({ closed: c.closed, pts: c.points.map((q) => q.p) }));
          bump(info, 'ssi_pairs');
        } catch (err) {
          bump(info, `ssi_refused_${err.code ?? 'error'}`);
        }
      }
      curvesOf.set(key, curves);
      cache.set(key, { sa: surf(a < b ? a : b), sb: surf(a < b ? b : a), curves });
    }
    return curvesOf.get(key);
  };
  // where q falls along polyline P: [segment + fraction, distance less the chord's sag allowance]
  const locate = (P, closed, q) => {
    let best = [0, Infinity];
    const n = closed ? P.length : P.length - 1;
    for (let i = 0; i < n; i++) {
      const p0 = P[i], d = sub(P[(i + 1) % P.length], p0), dd = dot(d, d);
      const t = dd > 0 ? Math.min(1, Math.max(0, dot(sub(q, p0), d) / dd)) : 0;
      // the polyline is chords of the curve: a point on the curve sits up to its chord's sag off it
      const e = Math.max(0, norm(sub(q, add(p0, scale(d, t)))) - SSI.SAG * Math.sqrt(dd));
      if (e < best[1]) best = [i + t, e];
    }
    return best;
  };
  return (a, b, pts, closed) => {
    const curves = pairCurves(a, b);
    if (!curves?.length) return null;
    let pick = null;
    for (const c of curves) {
      if (closed && !c.closed) continue;
      const at = pts.map((q) => locate(c.pts, c.closed, q));
      // the chain's own points are on both surfaces, so on the curve; its corners are on three or more, which
      // fits put a little off it (8e-4 x diagonal on 00001439_023): they need only be near
      const ends = closed ? [0] : [0, pts.length - 1];
      const worst = Math.max(0, ...at.filter((_, i) => !ends.includes(i)).map(([, e]) => e));
      if (worst > follow || ends.some((i) => at[i][1] > SSI.CORNER * diag)) continue;
      if (!pick || worst < pick.worst) pick = { c, at, worst };
    }
    if (!pick) { bump(info, 'ssi_not_followed'); return null; }
    const { c, at } = pick, P = c.pts, N = P.length, s = at.map(([x]) => x);
    const span = c.closed ? N : N - 1;
    const ahead = (x, y) => (c.closed ? (((y - x) % span) + span) % span : y - x);   // from x forward to y
    const s0 = s[0], s1 = closed ? s0 : s[s.length - 1];
    // the chain's own direction along the curve, from its second point
    const forward = c.closed ? ahead(s0, s[1]) < span / 2 : s[1] >= s0;
    const len = closed ? span : forward ? ahead(s0, s1) : ahead(s1, s0);
    if (!c.closed && !closed && (forward ? s1 < s0 : s1 > s0)) return null;
    const out = [pts[0]];
    const gap = 1e-7 * diag;
    for (let k = 1; k < N + 1; k++) {
      // the curve's vertices strictly between the corners, in the chain's direction
      const idx = forward ? Math.floor(s0) + k : Math.ceil(s0) - k;
      if (!c.closed && (idx < 0 || idx >= N)) break;
      const i = ((idx % N) + N) % N;
      const along = forward ? ahead(s0, i) : ahead(i, s0);
      if (!(along > 0) || along >= len) { if (along >= len) break; continue; }
      if (norm(sub(P[i], out[out.length - 1])) > gap && norm(sub(P[i], pts[closed ? 0 : pts.length - 1])) > gap) out.push(P[i]);
    }
    if (!closed) out.push(pts[pts.length - 1]);
    return out.length >= (closed ? 3 : 2) ? out : null;
  };
}

/**
 * PINCHES: a corner one face's boundary passes twice, two of its loops touching there (a hole tangent to a
 * plate's edge, 00005406_001). The face's two traces are tangent at that corner, and fits a few nm off make
 * them cross either side of it instead (1e-4 mm apart there): BRepCheck calls the face invalid, and the shape
 * repair splits it, filling the holes. Where the face is a plane, one trace a circle (a cylinder square to it)
 * and the other a line (another plane) or a circle, the circle (the smaller, of two) takes the radius that makes
 * them touch when that moves it no more than PINCH_TOL x diagonal, and the corner goes on the touching point
 * (into pinchPos). Returns the surfaces: a copy when a radius moved, the caller's otherwise.
 */
const PINCH_TOL = 1e-6;

function tangentPinches(mesh, surfaces, { fb, chains }, pinchPos, info) {
  const ck = (v, b) => v * BODY_STRIDE + b;
  const ends = new Map();                         // "face|corner" -> the neighbor across each chain end there
  for (const { a, b, path } of chains) {
    for (const v of [path[0], path[path.length - 1]]) {
      const c = ck(v, fb.get(a));
      for (const [f, g] of [[a, b], [b, a]]) {
        if (f < 0) continue;
        const key = `${f}|${c}`;
        (ends.get(key) ?? ends.set(key, []).get(key)).push(g);
      }
    }
  }
  let out = surfaces;
  for (const [key, gs] of ends) {
    if (gs.length < 4) continue;                  // passed once: two chain ends
    const [f, c] = key.split('|').map(Number);
    const sf = out.get(f), nb = [...new Set(gs)];
    if (sf?.[0] !== 'plane' || nb.length !== 2 || nb.some((g) => g < 0)) continue;
    const n = unit(sf[1].normal), o = sf[1].origin;
    const trace = (g) => {
      const [k, p] = out.get(g) ?? [];
      if (k === 'cylinder' && norm(cross(unit(p.axis), n)) <= 1e-9) {
        const a = unit(p.axis), t = dot(sub(o, p.axisPoint), n) / dot(a, n);
        return { g, circle: { c: add(p.axisPoint, scale(a, t)), r: p.radius } };
      }
      if (k === 'plane' && norm(cross(unit(p.normal), n)) > 1e-9) {
        return { g, line: { o: projectOnto([sf, out.get(g)], o, mesh.diag), d: unit(cross(n, unit(p.normal))) } };
      }
      return null;
    };
    const ts = nb.map(trace);
    if (ts.some((t) => !t)) continue;
    const circles = ts.filter((t) => t.circle), line = ts.find((t) => t.line);
    let g, r, at;
    if (circles.length === 1 && line) {
      const { c: cc } = circles[0].circle, { o: lo, d } = line.line;
      at = add(lo, scale(d, dot(sub(cc, lo), d)));                  // the foot of the centre on the line
      g = circles[0].g; r = norm(sub(cc, at));
    } else if (circles.length === 2) {
      const [A, B] = circles[0].circle.r <= circles[1].circle.r ? circles : [circles[1], circles[0]];   // A moves
      const gap = norm(sub(A.circle.c, B.circle.c));
      const outside = Math.abs(gap - (A.circle.r + B.circle.r)) < Math.abs(gap - (B.circle.r - A.circle.r));
      r = outside ? gap - B.circle.r : B.circle.r - gap;
      at = add(B.circle.c, scale(unit(sub(A.circle.c, B.circle.c)), B.circle.r));
      g = A.g;
    } else continue;
    const was = out.get(g)[1].radius;
    if (!(r > 0) || Math.abs(r - was) > PINCH_TOL * mesh.diag) { bump(info, 'pinch_left'); continue; }
    if (out === surfaces) out = new Map(surfaces);
    out.set(g, ['cylinder', { ...out.get(g)[1], radius: r }]);
    pinchPos.set(c, at);
    bump(info, 'pinches_made_exact');
  }
  return out;
}

/** Edge methods for a group in progressive recovery (repair.js): clean is BUILD's own (exact lines and circles,
 * contact loci, ssi's curves), plain drops the exact edges and contact loci, the mesh ones ssi's curves. */
export const VARIANTS = ['plain', 'mesh_clean', 'mesh_plain'];
const flagsOf = (v) => ({
  exact: BUILD.EXACT_EDGES && v !== 'plain' && v !== 'mesh_plain',
  contact: BUILD.CONTACT && v !== 'plain' && v !== 'mesh_plain',
  ssi: BUILD.SSI && v !== 'mesh_clean' && v !== 'mesh_plain',
});

/**
 * Build the shell. surfaces: Map label -> [kind, params]. facetLabels: labels of ONE triangle each,
 * built as a plane through its three corners with straight edges (the facet fallback). only: the labels whose
 * faces are built (null: all), for a trial of one region (repair.js progressive recovery); an edge to a patch
 * outside it is built as the whole build would build it and stays a boundary (boundaryEdges), not a free edge.
 * variantOf: Map label -> one of VARIANTS, the edge method of every edge and corner touching that patch (others
 * BUILD's). Returns {shape, faceOf: Map label -> face (or the pieces the repair split it into), info, ratios,
 * boundaryEdges}.
 */
export function buildTopo(oc, mesh, surfaces, { facetLabels = null, only = null, variantOf = null, topo: topoIn = null, ssiCache = undefined } = {}) {
  const E = enumOf(oc);
  const { V, T, F, diag: scaleLen } = mesh;
  const info = {};
  const facets = facetLabels ?? new Set();
  const forced = facets.size ? [...F.keys()].filter((t) => facets.has(F[t])) : null;
  let tp = performance.now();
  const topo = topoIn ?? meshTopology(mesh, forced);
  tp = tick('topology', tp);
  const { fb, facesAt, corners, chains } = topo;
  Object.assign(info, topo.info);
  const pinchPos = new Map();
  if (BUILD.PINCH) surfaces = tangentPinches(mesh, surfaces, topo, pinchPos, info);
  const surf = (f) => surfaces.get(f) ?? null;
  const ck = (v, b) => v * BODY_STRIDE + b;
  const inside = only ? (g) => only.has(g) : () => true;
  const baseFlags = flagsOf(null);
  const flagsAt = (groups) => {
    if (variantOf) for (const g of groups) { const v = variantOf.get(g); if (v) return flagsOf(v); }
    return baseFlags;
  };

  const patchTris = new Map();
  for (let t = 0; t < T.length; t++) {
    let l = patchTris.get(F[t]);
    if (!l) patchTris.set(F[t], (l = []));
    l.push(t);
  }
  const labelsSorted = [...patchTris.keys()].sort((a, b) => a - b);

  // corners: on every surface meeting there (the min-norm Gauss-Newton point). Known artifact: where
  // nearly coplanar surfaces meet at a sharp tip that point is ill-conditioned (Star.stl's tips moved
  // 100 mm out of a 580 mm part: thin fins inside a valid solid of the right volume). Keeping such
  // corners on their mesh vertex, with or without a matching vertex tolerance, made Star worse
  // (analytic -> 21-31% facets, and OCCT 7.9 then read the STEP as invalid), so it is left as measured.
  const vertex = new Map(), cornerPos = new Map(), arcsOf = new Map();   // arcsOf: a facet's arcs so far
  const place = (c) => {
    const fl = flagsAt(facesAt.get(c));
    const v = Math.floor(c / BODY_STRIDE);
    const ss = [...facesAt.get(c)].map(surf).filter(Boolean);
    let x = pinchPos.get(c) ?? (ss.length ? projectOnto(ss, V[v], scaleLen) : V[v]);
    // a corner of two tangent surfaces lies on their contact locus: solved along it, not by projection
    if (fl.contact && !pinchPos.has(c)) tangent: for (let i = 0; i < ss.length; i++) for (let j = i + 1; j < ss.length; j++) {
      const L = tangentLocus(ss[i], ss[j], scaleLen);
      if (!L) continue;
      // only where the corner then lies on every one of its surfaces: a 4-face corner that cannot be on
      // the locus and on the rest (inconsistent fits) keeps its projected position
      const y = cornerOnLocus(L, ss.filter((_, k) => k !== i && k !== j), x, scaleLen);
      if (y && ss.every(([k, p]) => Math.abs(distNormal(k, p, y)[0]) <= CONTACT_TOL * scaleLen)) { x = y; bump(info, 'corner_on_contact'); }
      break tangent;
    }
    return x;
  };
  for (const c of corners) {
    if (only && ![...facesAt.get(c)].some(inside)) continue;
    const x = place(c);
    cornerPos.set(c, x);
    const gp = pnt(oc, x);
    vertex.set(c, new oc.BRepBuilderAPI_MakeVertex(gp).Vertex());
    gp.delete();
  }
  // a corner no face built here touches, placed as the whole build places it (a facet's far corner, for its arcs)
  const cornerAt = (c) => cornerPos.get(c) ?? (corners.has(c) ? place(c) : null);

  tp = tick('topo_corners', tp);
  // edges
  const ssiArc = BUILD.SSI ? ssiArcs(mesh, surf, patchTris, info, ssiCache) : () => null;
  const edges = [];
  for (const { a, b, path } of chains) {
    if (only && !inside(a) && !(b >= 0 && inside(b))) continue;
    const fl = flagsAt([a, b]);
    const ss = [surf(a), b >= 0 ? surf(b) : null].filter(Boolean);
    const bd = fb.get(a);
    let locus = fl.contact && ss.length === 2 ? tangentLocus(ss[0], ss[1], scaleLen) : null;
    // the edge goes onto the locus only between corners on it (else it would jump at its ends)
    if (locus) for (const v of [path[0], path[path.length - 1]]) {
      const cp = cornerPos.get(ck(v, bd));
      const on = cp && ontoLocus(locus, cp);
      if (!cp || !on || norm(sub(on, cp)) > CONTACT_TOL * scaleLen) { locus = null; break; }
    }
    const pts = path.map((i) => {
      const cp = cornerPos.get(ck(i, bd));
      if (cp) return cp;
      const x = ss.length ? projectOnto(ss, V[i], scaleLen) : V[i];
      return (locus && ontoLocus(locus, x)) || x;        // a tangent pair's edge: on the contact locus
    });
    if (locus) bump(info, 'edges_on_contact');
    const closed = path[0] === path[path.length - 1];
    const bothPlanes = ss.length === 2 && ss[0][0] === 'plane' && ss[1][0] === 'plane';
    // A facet's edges are its own straight sides (every facet vertex is a corner, so the chain is one
    // mesh edge), except where it meets a curved face: there the curve its plane cuts that surface in (facetArc).
    const fa = facets.has(a) ? a : b >= 0 && facets.has(b) ? b : null;
    let cut = null;
    if (BUILD.FACET_ARCS && fa !== null && !closed && path.length === 2 && ss.length === 1 && ss[0][0] !== 'plane' && ss[0][0] !== 'bspline') {
      const q = T[patchTris.get(fa)[0]].map((v) => cornerAt(ck(v, bd)));
      if (q.every(Boolean)) cut = facetArc(q, pts[0], pts[1], ss[0], scaleLen);
      if (cut && arcCrosses(cut, q, arcsOf.get(fa) ?? [], scaleLen)) { cut = null; bump(info, 'facet_arc_crossing'); }
      if (cut) { bump(info, 'facet_arcs'); arcsOf.set(fa, [...(arcsOf.get(fa) ?? []), cut]); }
    }
    const straight = bothPlanes || (fa !== null && !cut);
    const v0 = vertex.get(ck(path[0], bd)), v1 = vertex.get(ck(path[path.length - 1], bd));
    let pc = null;
    try {
      let mk;
      if (cut) {
        pc = { pts: cut, closed: false };
        mk = chainEdge(oc, cut, false, v0, v1, pc);
      } else if (straight && !closed) {
        mk = new oc.BRepBuilderAPI_MakeEdge_2(v0, v1);
      } else {
        const body = closed ? pts.slice(0, -1) : pts;
        let clean = [body[0]];
        for (const q of body.slice(1)) if (norm(sub(q, clean[clean.length - 1])) > 1e-9 * scaleLen) clean.push(q);
        if (!closed) clean[clean.length - 1] = pts[pts.length - 1];
        clean = unfoldChain(clean, closed, info);
        if (clean.length < 2 || (closed && clean.length < 3)) { bump(info, 'edge_too_short'); continue; }
        // the curve between the mesh's points too, not only through them, where two surfaces cross (ssiArcs); a
        // tangent pair's edge is on its contact locus already
        const arc = fl.ssi && ss.length === 2 && !locus ? ssiArc(a, b, clean, closed) : null;
        if (arc) { clean = arc; bump(info, 'edges_ssi'); }
        // an exact line or circle where the two surfaces meet in one: CAD tools treat analytic edges
        // as the intersections they are (fillets, face deletion); a spline through them is only close
        const ex = fl.exact && ss.length === 2 ? exactCurve(closed ? [...clean, clean[0]] : clean, EXACT_EDGE_TOL * scaleLen) : null;
        mk = ex ? exactEdge(oc, ex, closed ? [...clean, clean[0]] : clean, closed, v0, v1) : null;
        if (mk && mk.IsDone()) bump(info, ex.line ? 'edges_line' : 'edges_circle');
        else { pc = { pts: clean, closed }; mk = chainEdge(oc, clean, closed, v0, v1, pc); }
      }
      if (!mk.IsDone()) { bump(info, 'edge_failed'); continue; }
      edges.push({ a, b, s: ck(path[0], bd), t: ck(path[path.length - 1], bd), e: mk.Edge(), pc, bound: !!only && b >= 0 && inside(a) !== inside(b) });
    } catch (err) {
      bump(info, 'edge_exception');
    }
  }

  tp = tick('edges', tp);
  // faces
  const byFace = new Map();
  const push = (f, item) => { let l = byFace.get(f); if (!l) byFace.set(f, (l = [])); l.push(item); };
  for (const { a, b, s, t, e } of edges) {
    if (inside(a)) push(a, [s, t, e]);
    if (b >= 0 && inside(b)) push(b, [t, s, e.Reversed()]);
  }
  const builder = new oc.BRep_Builder();
  const shell = new oc.TopoDS_Shell();
  builder.MakeShell(shell);
  const faceOf = new Map();
  const facetFaces = [];
  for (const f of labelsSorted) {
    if (!inside(f)) continue;
    const tris = patchTris.get(f);
    const sp = surf(f);
    const facet = facets.has(f);
    if (!sp && !facet) { bump(info, 'face_no_surface'); continue; }
    let s, kind = null, p = null;
    if (facet) {
      // The plane through the facet's three corners, normal by the mesh winding: outward.
      const q = T[tris[0]].map((v) => cornerPos.get(ck(v, fb.get(f))));
      const nrm = cross(sub(q[1], q[0]), sub(q[2], q[0]));
      if (!byFace.get(f) || norm(nrm) <= 1e-12 * scaleLen * scaleLen) { bump(info, 'facet_degenerate'); continue; }
      s = new oc.Handle_Geom_Surface_2(new oc.Geom_Plane_3(pnt(oc, q[0]), dir(oc, unit(nrm))));
    } else {
      [kind, p] = sp;
      const ptsIdx = new Set();
      for (const t of tris) for (const v of T[t]) ptsIdx.add(v);
      try {
        s = geomSurface(oc, kind, p, [...ptsIdx].map((v) => V[v]));
      } catch {
        bump(info, 'surface_failed');
        continue;
      }
    }
    if (!byFace.get(f)) {
      // A patch with no boundary is a whole closed surface (a ball, a ring): its natural bounds.
      if (kind === 'sphere' || kind === 'torus') {
        let g = 0;
        for (const t of tris) {
          const cen = T[t].reduce((acc, v) => [acc[0] + V[v][0] / 3, acc[1] + V[v][1] / 3, acc[2] + V[v][2] / 3], [0, 0, 0]);
          g += dot(distNormal(kind, p, cen)[1], mesh.N[t]);
        }
        const nat = new oc.BRepBuilderAPI_MakeFace_8(g < 0 ? s.get().UReversed() : s, 1e-7);
        if (nat.IsDone()) {
          const fc = nat.Face();
          builder.Add(shell, fc);
          faceOf.set(f, fc);
          bump(info, 'faces'); bump(info, 'natural_faces');
          continue;
        }
      }
      bump(info, 'face_no_edges');
      continue;
    }
    // loops in mesh winding: follow end corner -> an unused edge starting there
    const todo = byFace.get(f).slice();
    const wires = [];
    while (todo.length) {
      const [s0, t0, e0] = todo.shift();
      const loop = [e0];
      let cur = t0;
      while (cur !== s0) {
        const k = todo.findIndex(([s2]) => s2 === cur);
        if (k < 0) break;
        const [, t2, e2] = todo.splice(k, 1)[0];
        loop.push(e2);
        cur = t2;
      }
      if (cur !== s0) bump(info, 'open_loop');
      const w = new oc.TopoDS_Wire();
      builder.MakeWire(w);
      for (const e2 of loop) builder.Add(w, e2);
      wires.push(w);
    }
    // A surface whose natural normal points INTO the solid (judged over the patch's own triangles) is
    // replaced by its U-reversed copy, so every face is FORWARD with the mesh-order wires.
    if (!facet) {
      let g = 0;
      for (const t of tris) {
        const cen = T[t].reduce((acc, v) => [acc[0] + V[v][0] / 3, acc[1] + V[v][1] / 3, acc[2] + V[v][2] / 3], [0, 0, 0]);
        g += dot(distNormal(kind, p, cen)[1], mesh.N[t]);
      }
      if (g < 0) { s = s.get().UReversed(); bump(info, 'reversed_surfaces'); }
    }
    const face = new oc.TopoDS_Face();
    builder.MakeFace_2(face, s, 1e-7);
    for (const w of wires) builder.Add(face, w);
    if (facet && !BUILD.FIX_FACETS) facetFaces.push(face);
    else builder.Add(shell, face);
    faceOf.set(f, face);
    bump(info, 'faces');
    if (facet) bump(info, 'facet_faces');
  }

  tp = tick('faces', tp);
  if (BUILD.PCURVES) addPcurves(oc, builder, edges, faceOf, facets, info, scaleLen);
  tp = tick('pcurves', tp);
  BUILD.DUMP?.(oc, shell, mesh);   // the shell as handed to the kernel (test/knit.mjs --dump-kernel)

  // ONE repair pass over the analytic faces: ShapeFix_Shape records every replacement in one context and
  // applies it to all faces, so shared edges stay shared. Facets stay out of it: a plane through its own three
  // corners with straight sides needs nothing, and the repair cost as much per facet as per surface (8 ms a
  // face, 86 s a build of 00005493_012's 10,969 faces) and moved their corners by up to its 1e-7 mm tolerance.
  // The chords where facets meet curved faces are those faces' edges too, so they are repaired (tolerance,
  // pcurve) with them; the facets then take whatever the repair replaced.
  let result = shell, ctx = null;
  if (subShapes(oc, shell, E.FACE).length) {
    // with facets out, the analytic faces are islands: ShapeFix_Shell would orient (or split) each on its own, and
    // the shell with the facets back failed BRepCheck NotConnected (00003840_000). A compound gets the face fixes
    // only, in one context.
    let input = shell;
    if (facetFaces.length) {
      input = new oc.TopoDS_Compound();
      builder.MakeCompound(input);
      for (const fc of subShapes(oc, shell, E.FACE)) builder.Add(input, fc);
    }
    const fix = new oc.ShapeFix_Shape_2(input);
    fix.Perform(new oc.Message_ProgressRange_1());
    result = fix.Shape();
    ctx = fix.Context().get();
  }
  tp = tick('shapefix', tp);
  if (facetFaces.length) {
    const all = [...subShapes(oc, result, E.FACE), ...facetFaces.map((fc) => (ctx ? ctx.Apply_2(fc, E.SHAPE) : fc))];
    result = connectedShells(oc, builder, all);
    bump(info, 'facets_unrepaired', facetFaces.length);
  }
  const remapped = new Map();
  for (const [f, fc] of faceOf) {
    const r = ctx ? ctx.Apply_2(fc, E.SHAPE) : fc;
    if (r.IsNull()) { bump(info, 'face_removed_by_repair'); continue; }
    if (r.ShapeType() === E.FACE) remapped.set(f, oc.TopoDS.Face_1(r));
    else if (subShapes(oc, r, E.FACE).length) { remapped.set(f, r); bump(info, 'face_split_by_repair'); }
  }

  // DROP NATURAL-BOUND WIRES the repair added: on a face that has real boundary edges, a wire with no
  // edge shared by another face is not part of the patch's boundary (a torus face otherwise becomes
  // the whole torus minus the patch).
  const m = new oc.TopTools_IndexedDataMapOfShapeListOfShape_1();
  oc.TopExp.MapShapesAndAncestors(result, E.EDGE, E.FACE, m);
  // a region's edges to patches outside it are real boundary, though no face across them is built
  const boundaryEdges = new oc.TopTools_IndexedMapOfShape_1();
  if (only) for (const ed of edges) {
    if (!ed.bound) continue;
    const r = ctx ? ctx.Apply_2(ed.e, E.SHAPE) : ed.e;
    if (!r.IsNull()) for (const x of subShapes(oc, r, E.EDGE)) boundaryEdges.Add(x);
  }
  const shared = (e, face) => {
    if (only && boundaryEdges.Contains(e)) return true;
    const i = m.FindIndex(e);
    if (i <= 0) return false;
    const l = m.FindFromIndex(i);
    if (l.Size() > 2) return true;
    return (l.Size() >= 1 && !l.First_1().IsSame(face)) || (l.Size() === 2 && !l.Last_1().IsSame(face));
  };
  const kept = [];
  const replaced = [];
  for (const fc0 of subShapes(oc, result, E.FACE)) {
    const fc = oc.TopoDS.Face_1(fc0);
    const all = subShapes(oc, fc, E.WIRE);
    const keep = all.filter((w) => subShapes(oc, w, E.EDGE).some((e) => shared(e, fc)));
    if (keep.length && keep.length < all.length) {
      const nf = new oc.TopoDS_Face();
      builder.MakeFace_2(nf, oc.BRep_Tool.Surface_2(fc), oc.BRep_Tool.Tolerance_1(fc));
      for (const w of keep) builder.Add(nf, w);
      const oriented = oc.TopoDS.Face_1(nf.Oriented(fc.Orientation_1()));
      kept.push(oriented);
      replaced.push([fc, oriented]);
      bump(info, 'natural_bound_wires_removed', all.length - keep.length);
    } else {
      kept.push(fc);
    }
  }
  if (replaced.length) {
    for (const [f, fc] of remapped) {
      const hit = replaced.find(([old]) => old.IsSame(fc));
      if (hit) remapped.set(f, hit[1]);
    }
    result = connectedShells(oc, builder, kept);
  }

  // Area of every patch's final face against the patch's mesh area: a face that took the wrong region
  // of its surface shows up here even when the shell is closed and valid.
  tp = tick('shells_wires', tp);
  const ratios = new Map();
  for (const [f, fc] of remapped) {
    // a facet out of the repair is its own triangle
    if (facets.has(f) && !BUILD.FIX_FACETS) { ratios.set(f, 1); continue; }
    let area = 0;
    for (const t of patchTris.get(f)) area += mesh.area[t];
    try {
      const g = new oc.GProp_GProps_1();
      oc.BRepGProp.SurfaceProperties_1(fc, g, false, false);
      ratios.set(f, Math.abs(g.Mass()) / Math.max(area, 1e-300));
      g.delete();
    } catch {
      ratios.set(f, NaN);
    }
  }
  tick('ratios', tp);
  return { shape: result, faceOf: remapped, info, ratios, boundaryEdges };
}

/**
 * One shell per piece of `faces` joined by shared edges (a compound of them when there are several), as ShapeFix_Shell
 * splits a shell: one shell over separate pieces fails BRepCheck NotConnected (00003840_000's four).
 */
function connectedShells(oc, builder, faces) {
  const E = enumOf(oc);
  const byEdge = new oc.TopTools_IndexedMapOfShape_1();
  const edgesOf = faces.map((fc) => subShapes(oc, fc, E.EDGE).map((e) => byEdge.Add(e)));
  const parent = faces.map((_, i) => i);
  const find = (i) => { while (parent[i] !== i) i = parent[i] = parent[parent[i]]; return i; };
  const first = new Map();
  edgesOf.forEach((es, i) => {
    for (const k of es) {
      if (!first.has(k)) first.set(k, i);
      else parent[find(i)] = find(first.get(k));
    }
  });
  const pieces = new Map();
  faces.forEach((fc, i) => { const r = find(i); if (!pieces.has(r)) pieces.set(r, []); pieces.get(r).push(fc); });
  const shells = [...pieces.values()].map((fs) => {
    const sh = new oc.TopoDS_Shell();
    builder.MakeShell(sh);
    for (const fc of fs) builder.Add(sh, fc);
    return sh;
  });
  if (shells.length === 1) return shells[0];
  const comp = new oc.TopoDS_Compound();
  builder.MakeCompound(comp);
  for (const sh of shells) builder.Add(comp, sh);
  return comp;
}

const PCURVE_TOL = 1e-7, PCURVE_ON = 1e-2, PCURVE_OVERSHOOT = 0.5;

/**
 * PCURVES for the spline edges: each one's curve in the parameters of both its faces, through the surface's own
 * (u, v) at the same points and the same parameters as the 3D spline. Without them ShapeFix projects every spline
 * onto both surfaces, which on a plain build (no exact lines or circles) is most of its time: 31 s against 0.5 s
 * for the clean build of the same 42 faces, 00000890_001. One that strays from the 3D curve by more than
 * PCURVE_TOL x diagonal beyond twice the curve's own distance from the surface is left to ShapeFix, and so is one
 * whose points are off the surface by more than PCURVE_ON x diagonal, or whose poles go round a periodic surface more
 * than PCURVE_OVERSHOOT periods beyond its points (the interpolation overshooting; a thread's helix goes round 3.6
 * times through its own points, 00003819_009): an edge to a corner far off its surfaces (00046240_0, 2.5e5 mm out)
 * passed the check above, as the curve's own distance is allowed, with a pcurve 609,773 rad round a torus, and
 * ShapeFix ran for hours on it.
 */
function addPcurves(oc, builder, edges, faceOf, facets, info, scaleLen) {
  const sas = new Map();
  const analysis = (f, face) => {
    let a = sas.get(f);
    if (!a) {
      const h = oc.BRep_Tool.Surface_2(face);
      const g = h.get();
      a = { g, sa: new oc.ShapeAnalysis_Surface(h), up: g.IsUPeriodic() ? g.UPeriod() : 0, vp: g.IsVPeriodic() ? g.VPeriod() : 0 };
      sas.set(f, a);
    }
    return a;
  };
  const wrap = (x, ref, period) => (period ? x - period * Math.round((x - ref) / period) : x);
  for (const { a, b, e, pc } of edges) {
    if (!pc || a === b) continue;
    for (const f of [a, b]) {
      const face = f >= 0 && !facets.has(f) ? faceOf.get(f) : null;
      if (!face) continue;
      const { g, sa, up, vp } = analysis(f, face);
      const uv = [];
      let gap = 0;
      for (const q of pc.pts) {
        const gp = pnt(oc, q);
        const p2 = sa.ValueOfUV(gp, 1e-9);
        let u = p2.X(), v = p2.Y();
        gap = Math.max(gap, sa.Gap());
        gp.delete(); p2.delete();
        if (uv.length) { const [u0, v0] = uv[uv.length - 1]; u = wrap(u, u0, up); v = wrap(v, v0, vp); }
        uv.push([u, v, 0]);
      }
      let closeWith = null;
      if (pc.closed) {
        const [u0, v0] = uv[0], [ul, vl] = uv[uv.length - 1];
        closeWith = [wrap(u0, ul, up), wrap(v0, vl, vp), 0];
      }
      if (!(gap <= PCURVE_ON * scaleLen)) { bump(info, 'pcurve_off_surface'); continue; }
      try {
        const { poles, knots, mults, degree } = interpolateChain(uv, pc.closed, { t: pc.t, closeWith });
        const span = (P, k) => { let lo = Infinity, hi = -Infinity; for (const q of P) { lo = Math.min(lo, q[k]); hi = Math.max(hi, q[k]); } return hi - lo; };
        const over = (k, period) => period && span(poles, k) > span(uv, k) + PCURVE_OVERSHOOT * period;
        if (over(0, up) || over(1, vp)) { bump(info, 'pcurve_overshoot'); continue; }
        // Checked halfway between its points, where the two interpolations part most: the pcurve's point must be
        // as near the 3D curve's as the surface is (the tolerance ShapeFix's projection would give the edge), else
        // ShapeFix projects it. Interpolated (u, v) strays where the chain is sparse and turns: 5 points round a
        // cone, 2.8e4 mm off on 00002498_000, which made every check after it crawl.
        let excess = 0;
        const c2d = { poles, knots, mults, degree };
        for (let i = 0; i + 1 < pc.t.length; i++) {
          const tm = (pc.t[i] + pc.t[i + 1]) / 2;
          const [u, v] = evalBSpline(c2d, tm);
          const q = evalBSpline(pc.curve, tm);
          const sp = g.Value(u, v);
          const gq = pnt(oc, q);
          sa.ValueOfUV(gq, 1e-9).delete();
          excess = Math.max(excess, norm(sub([sp.X(), sp.Y(), sp.Z()], q)) - 2 * sa.Gap());
          sp.delete(); gq.delete();
        }
        if (!(excess <= PCURVE_TOL * scaleLen)) { bump(info, 'pcurve_off'); continue; }
        const P = new oc.TColgp_Array1OfPnt2d_2(1, poles.length);
        poles.forEach((q, i) => { const g2 = new oc.gp_Pnt2d_3(q[0], q[1]); P.SetValue(i + 1, g2); g2.delete(); });
        const K = new oc.TColStd_Array1OfReal_2(1, knots.length);
        knots.forEach((k, i) => K.SetValue(i + 1, k));
        const M = new oc.TColStd_Array1OfInteger_2(1, mults.length);
        mults.forEach((m, i) => M.SetValue(i + 1, m));
        const c2 = new oc.Geom2d_BSplineCurve_1(P, K, M, degree, false);
        builder.UpdateEdge_3(e, new oc.Handle_Geom2d_Curve_2(c2), face, 1e-7);
        bump(info, 'pcurves');
      } catch {
        bump(info, 'pcurve_failed');
      }
    }
    builder.SameParameter(e, false);
  }
}

/**
 * The mesh as a solid: a planar face per triangle, one straight edge per paired mesh edge, a vertex per
 * (mesh vertex, body), a shell per body. No repair pass: planes need no stored pcurves.
 */
export function buildPolyhedral(oc, mesh) {
  const { V, T } = mesh;
  const builder = new oc.BRep_Builder();
  const shells = new Map();
  const info = {};
  const verts = new Map(), pending = new Map();
  const vert = (v, bd) => {
    const key = v * BODY_STRIDE + bd;
    if (!verts.has(key)) {
      const gp = pnt(oc, V[v]);
      verts.set(key, new oc.BRepBuilderAPI_MakeVertex(gp).Vertex());
      gp.delete();
    }
    return verts.get(key);
  };
  for (let t = 0; t < T.length; t++) {
    const q = T[t].map((v) => V[v]);
    const nrm = cross(sub(q[1], q[0]), sub(q[2], q[0]));
    if (norm(nrm) <= 1e-12 * mesh.diag * mesh.diag) { bump(info, 'facet_degenerate'); continue; }
    const bd = mesh.body[t];
    const wire = new oc.TopoDS_Wire();
    builder.MakeWire(wire);
    for (let k = 0; k < 3; k++) {
      const u = T[t][k], v = T[t][(k + 1) % 3];
      const id = mesh.eid(u, v);
      let e = pending.get(t * 2 ** 26 + id);
      if (e) pending.delete(t * 2 ** 26 + id);
      else {
        const mk = new oc.BRepBuilderAPI_MakeEdge_2(vert(u, bd), vert(v, bd));
        if (!mk.IsDone()) { bump(info, 'edge_failed'); continue; }
        e = mk.Edge();
        const n = mesh.across(t, u, v);
        if (n >= 0) pending.set(n * 2 ** 26 + id, e.Reversed());   // the partner runs it the other way
      }
      builder.Add(wire, e);
    }
    const face = new oc.TopoDS_Face();
    builder.MakeFace_2(face, new oc.Handle_Geom_Surface_2(new oc.Geom_Plane_3(pnt(oc, q[0]), dir(oc, unit(nrm)))), 1e-7);
    builder.Add(face, wire);
    if (!shells.has(bd)) {
      const sh = new oc.TopoDS_Shell();
      builder.MakeShell(sh);
      shells.set(bd, sh);
    }
    builder.Add(shells.get(bd), face);
    bump(info, 'facet_faces');
  }
  const comp = new oc.TopoDS_Compound();
  builder.MakeCompound(comp);
  for (const sh of shells.values()) builder.Add(comp, sh);
  return { shape: comp, info };
}

const SHELL_FIX_MAX_FACES = 3000;

/**
 * Closed (no free or non-manifold edges), BRepCheck-valid, and the volume against the mesh's: one solid
 * per shell, turned outward. `solid` is the compound of solids to export.
 */
// ── on the mesh ──────────────────────────────────────────────────────────────────────────────────────
//
// A build can be a closed, valid solid of the mesh's volume and still not be the part: 00110180_0 (Plasticity) went
// out hybrid with an edge 328 mm long between corners 36 mm apart, 128 mm off the mesh, under a 138 mm tolerance. Its
// vertices, points along its edges and points inside its faces must lie within OFF_MESH x diagonal of the mesh (score's off_mesh, grade), and
// the faces of any that do not are flagged invalid (patchProblems), so recovery and the facet tiers rebuild them.
export const OFF_MESH = 1e-2;
const meshGrids = new WeakMap();

/** The mesh's triangles in a grid of cells OFF_MESH x diagonal wide, for nearMesh. */
function meshGrid(mesh) {
  let grid = meshGrids.get(mesh.T);
  if (grid) return grid;
  const { V, T } = mesh, h = OFF_MESH * mesh.diag, cells = new Map();
  const key = (i, j, k) => `${i},${j},${k}`;
  T.forEach((tri, t) => {
    const lo = [0, 1, 2].map((a) => Math.floor(Math.min(...tri.map((v) => V[v][a])) / h));
    const hi = [0, 1, 2].map((a) => Math.floor(Math.max(...tri.map((v) => V[v][a])) / h));
    for (let i = lo[0]; i <= hi[0]; i++) for (let j = lo[1]; j <= hi[1]; j++) for (let k = lo[2]; k <= hi[2]; k++) {
      const c = key(i, j, k);
      let l = cells.get(c);
      if (!l) cells.set(c, (l = []));
      l.push(t);
    }
  });
  grid = { h, cells, key };
  meshGrids.set(mesh.T, grid);
  return grid;
}

/** Distance from p to triangle abc (Ericson's closest point). */
function triangleDistance(p, a, b, c) {
  const ab = sub(b, a), ac = sub(c, a), ap = sub(p, a);
  const d1 = dot(ab, ap), d2 = dot(ac, ap);
  if (d1 <= 0 && d2 <= 0) return norm(ap);
  const bp = sub(p, b), d3 = dot(ab, bp), d4 = dot(ac, bp);
  if (d3 >= 0 && d4 <= d3) return norm(bp);
  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) return norm(sub(ap, scale(ab, d1 / (d1 - d3))));
  const cp = sub(p, c), d5 = dot(ab, cp), d6 = dot(ac, cp);
  if (d6 >= 0 && d5 <= d6) return norm(cp);
  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) return norm(sub(ap, scale(ac, d2 / (d2 - d6))));
  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) return norm(sub(bp, scale(sub(c, b), (d4 - d3) / ((d4 - d3) + (d5 - d6)))));
  const den = 1 / (va + vb + vc);
  return norm(sub(ap, add(scale(ab, vb * den), scale(ac, vc * den))));
}

/** Whether p is within OFF_MESH x diagonal of the mesh. */
function nearMesh(mesh, p) {
  const { h, cells, key } = meshGrid(mesh), { V, T } = mesh;
  const c = p.map((x) => Math.floor(x / h));
  for (let i = -1; i <= 1; i++) for (let j = -1; j <= 1; j++) for (let k = -1; k <= 1; k++) {
    for (const t of cells.get(key(c[0] + i, c[1] + j, c[2] + k)) ?? []) {
      const [a, b, d] = T[t];
      if (triangleDistance(p, V[a], V[b], V[d]) <= h) return true;
    }
  }
  return false;
}

/**
 * The faces of `shape` with a vertex or a point along an edge (a quarter, half and three quarters) off the mesh, or a
 * point inside them: a 3 x 3 grid over the face's (u, v) bounds, the points the face holds. A face can bulge between
 * good edges (00129368_1's torus 11: every edge on the mesh, half of it up to 2.2 mm off on a 55 mm part). Facets
 * (planes of three edges, through mesh corners) are not sampled inside.
 */
export function offMeshFaces(oc, shape, mesh) {
  const E = enumOf(oc);
  const m = new oc.TopTools_IndexedDataMapOfShapeListOfShape_1();
  oc.TopExp.MapShapesAndAncestors(shape, E.EDGE, E.FACE, m);
  const faces = [];
  for (let i = 1; i <= m.Extent(); i++) {
    const e = oc.TopoDS.Edge_1(m.FindKey(i));
    if (oc.BRep_Tool.Degenerated(e)) continue;
    const ad = new oc.BRepAdaptor_Curve_2(e);
    const t0 = ad.FirstParameter(), t1 = ad.LastParameter();
    let off = false;
    for (const f of [0, 0.25, 0.5, 0.75, 1]) {
      const q = ad.Value(t0 + f * (t1 - t0));
      off ||= !nearMesh(mesh, [q.X(), q.Y(), q.Z()]);
      q.delete();
      if (off) break;
    }
    ad.delete();
    if (!off) continue;
    const lst = m.FindFromIndex(i);
    if (lst.Size() >= 1) faces.push(lst.First_1());                 // an edge of a valid solid has one or two faces
    if (lst.Size() >= 2) faces.push(lst.Last_1());
  }
  m.delete();
  const PLANE = oc.GeomAbs_SurfaceType.GeomAbs_Plane;
  for (const fc of subShapes(oc, shape, E.FACE)) {
    const face = oc.TopoDS.Face_1(fc);
    const sa = new oc.BRepAdaptor_Surface_2(face, true);
    if (sa.GetType() === PLANE && subShapes(oc, face, E.EDGE).length <= 3) { sa.delete(); continue; }
    const u0 = sa.FirstUParameter(), u1 = sa.LastUParameter(), v0 = sa.FirstVParameter(), v1 = sa.LastVParameter();
    const cls = new oc.BRepTopAdaptor_FClass2d(face, 1e-9);
    let off = false;
    for (const fu of [0.25, 0.5, 0.75]) {
      for (const fv of [0.25, 0.5, 0.75]) {
        const uv = new oc.gp_Pnt2d_3(u0 + fu * (u1 - u0), v0 + fv * (v1 - v0));
        if (cls.Perform(uv, false) === E.IN) {
          const p = sa.Value(uv.X(), uv.Y());
          off = !nearMesh(mesh, [p.X(), p.Y(), p.Z()]);
          p.delete();
        }
        uv.delete();
        if (off) break;
      }
      if (off) break;
    }
    cls.delete();
    sa.delete();
    if (off) faces.push(fc);
  }
  return faces;
}

export function score(oc, shape, mesh) {
  const E = enumOf(oc);
  const out = {};
  const m = new oc.TopTools_IndexedDataMapOfShapeListOfShape_1();
  oc.TopExp.MapShapesAndAncestors(shape, E.EDGE, E.FACE, m);
  let free = 0, bad = 0;
  for (let i = 1; i <= m.Extent(); i++) {
    const lst = m.FindFromIndex(i);
    const n = lst.Size();
    const e = oc.TopoDS.Edge_1(m.FindKey(i));
    if (oc.BRep_Tool.Degenerated(e)) continue;                       // apex / pole: one face by nature
    if (n === 1 && oc.BRep_Tool.IsClosed_2(e, oc.TopoDS.Face_1(lst.First_1()))) continue;   // seam
    if (n === 1) free++;
    else if (n > 2) bad++;
  }
  out.edges = m.Extent();
  out.free_edges = free;
  out.nonmanifold_edges = bad;
  m.delete();
  const shells = subShapes(oc, shape, E.SHELL).map((s) => oc.TopoDS.Shell_1(s));
  out.shells = shells.length;
  out.closed = free === 0 && bad === 0 && shells.length >= 1;
  out.valid = false;
  out.vol_err = null;
  try {
    const builder = new oc.BRep_Builder();
    const comp = new oc.TopoDS_Compound();
    builder.MakeCompound(comp);
    let valid = true, total = 0;
    // A shape with a free or non-manifold edge is no valid solid, whatever its shells' repair does: checking
    // it costs as much as a build and changes nothing but the volume tiebreak (grade), so it is not checked.
    const check = out.closed;
    const solidOf = (shell) => {
      let solid = new oc.TopoDS_Solid();
      builder.MakeSolid(solid);
      builder.Add(solid, shell);
      const cls = new oc.BRepClass3d_SolidClassifier_2(solid);
      let tc = performance.now();
      cls.PerformInfinitePoint(1e-7);
      tc = tick('classify', tc);
      if (cls.State() === E.IN) solid = oc.TopoDS.Solid_1(solid.Reversed());
      if (!check) return [solid, false, null];
      const ana = new oc.BRepCheck_Analyzer(solid, true, false);
      const okv = ana.IsValid_2();
      tick('brepcheck', tc);
      return [solid, okv, ana];
    };
    // each shell's check as built, which patchProblems reads its faces from (one check per build, not two)
    out.analyzers = [];
    for (const sh of shells) {
      // The shell as built first: its faces follow the mesh winding, made consistent by KnitMesh, so it
      // usually needs nothing. ml/solid_rebuild.py runs ShapeFix_Shell's whole Perform, but in this
      // kernel (OCCT 7.6) its per-face pass can make a valid shell unorientable (00032949_19 gmsh: valid
      // raw and orientation-fixed, UnorientableShape on 2 faces after Perform; OCCT 7.9 keeps it valid),
      // and its orientation pass is quadratic: 776 s on a 33,802-facet shell. So an invalid shell gets
      // the orientation fix, then Perform, and only up to SHELL_FIX_MAX_FACES faces.
      let [solid, ok, ana] = solidOf(sh);
      if (ana) out.analyzers.push([sh, ana]);
      const tsf = performance.now();
      if (!ok && check && subShapes(oc, sh, E.FACE).length <= SHELL_FIX_MAX_FACES) {
        const orient = new oc.ShapeFix_Shell_1();
        orient.Init(sh);
        orient.FixFaceOrientation(sh, true, false);
        const [s1, ok1] = solidOf(orient.Shell());
        if (ok1) [solid, ok] = [s1, ok1];
        else {
          const fix = new oc.ShapeFix_Shell_2(sh);
          fix.Perform(new oc.Message_ProgressRange_1());
          const [s2, ok2] = solidOf(fix.Shell());
          if (ok2) [solid, ok] = [s2, ok2];
        }
      }
      tick('score_shellfix', tsf);
      valid = valid && ok;
      const g = new oc.GProp_GProps_1();
      const tv = performance.now();
      oc.BRepGProp.VolumeProperties_1(solid, g, false, false, false);
      tick('volume', tv);
      total += g.Mass();
      builder.Add(comp, solid);
    }
    out.valid = valid && shells.length >= 1;
    out.volume = total;
    // on the mesh (OFF_MESH): only a solid that would otherwise pass is worth the look
    if (out.closed && out.valid) {
      const to = performance.now();
      out.offFaces = offMeshFaces(oc, shape, mesh);
      out.off_mesh = out.offFaces.length;
      tick('off_mesh', to);
    }
    const mv = mesh.volume();
    out.vol_err = mv ? Math.abs(total - mv) / Math.abs(mv) : null;
    out.solid = comp;
  } catch (err) {
    out.error = String(err?.message ?? err).slice(0, 120);
  }
  return out;
}

export const LOOP_AREA_RATIO = 1.25;   // a face this far from its patch's mesh area took the wrong region

/**
 * Patches whose face is missing (not built), invalid (BRepCheck), orient (uses a shared edge in the same
 * direction as the face across it), area (off its patch's mesh area by more than LOOP_AREA_RATIO), or
 * open (bounded by a free edge). Faces are checked as they sit in `result`. analyzers: score's [shell, check]
 * per shell; analyzer: one check holding every face (regionProblems); boundary: edges that are no free edge
 * though only one face here uses them (a region's edges to patches outside it).
 */
export function patchProblems(oc, result, faceOf, labels, ratios, analyzers = null, { analyzer = null, boundary = null, offFaces = null } = {}) {
  const E = enumOf(oc);
  const tpp = performance.now();
  const ownMap = new oc.TopTools_IndexedMapOfShape_1();
  const owner = [];
  for (const [f, sh] of faceOf) {
    for (const fc of subShapes(oc, sh, E.FACE)) {
      const i = ownMap.Add(fc);
      owner[i] = f;
    }
  }
  const ownerOf = (fc) => { const i = ownMap.FindIndex(fc); return i > 0 ? owner[i] : undefined; };
  const faces = subShapes(oc, result, E.FACE);
  // score's check of the shell (as built, so the same faces) when it holds every face; else one check of `result`.
  // Not each shell's own solid check on a part of several: in a solid's context 34 faces of 00001745_008's two
  // shells failed, against 2 checked together.
  if (!analyzer && analyzers?.length === 1 && subShapes(oc, analyzers[0][0], E.FACE).length === faces.length) analyzer = analyzers[0][1];
  analyzer ??= new oc.BRepCheck_Analyzer(result, true, false);
  const checkOf = () => analyzer;
  const invalid = new Set(), present = new Set();
  for (const fc of faces) {
    const f = ownerOf(fc);
    if (f === undefined) continue;
    present.add(f);
    if (invalid.has(f)) continue;
    try {
      const ana = checkOf(fc);
      if (!ana.IsValid_1(fc)) { invalid.add(f); continue; }
      for (const type of [E.WIRE, E.EDGE]) {
        if (subShapes(oc, fc, type).some((s) => !ana.IsValid_1(s))) { invalid.add(f); break; }
      }
    } catch {
      invalid.add(f);
    }
  }
  // faces with an edge off the mesh (score's offFaces): valid, but not the part
  for (const fc of offFaces ?? []) { const f = ownerOf(fc); if (f !== undefined) invalid.add(f); }
  const edgeMap = new oc.TopTools_IndexedMapOfShape_1();
  const use = [];
  for (const fc of faces) {
    const f = ownerOf(fc);
    const face = oc.TopoDS.Face_1(fc);
    // the explorer gives each edge as the face uses it, its own orientation already composed with the face's: taking
    // the face's again flagged every face the repair turned over, and its neighbours (6 clashes on a valid box)
    for (const es of subShapes(oc, fc, E.EDGE)) {
      const e = oc.TopoDS.Edge_1(es);
      if (oc.BRep_Tool.Degenerated(e) || oc.BRep_Tool.IsClosed_2(e, face)) continue;
      const i = edgeMap.Add(e);
      (use[i] ??= []).push([f, e.Orientation_1() === E.REVERSED]);
    }
  }
  const opened = new Set(), orient = new Set();
  const pairs = new Map(), clashes = new Map();   // per patch: its edges shared with another face, and those used alike
  for (let i = 1; i < use.length; i++) {
    const lst = use[i];
    if (!lst) continue;
    if (lst.length === 1 && boundary && boundary.Contains(edgeMap.FindKey(i))) continue;
    if (lst.length === 1 && lst[0][0] !== undefined) opened.add(lst[0][0]);
    else if (lst.length === 2) {
      const alike = lst[0][1] === lst[1][1];
      for (const [f] of lst) {
        if (f === undefined) continue;
        pairs.set(f, (pairs.get(f) ?? 0) + 1);
        if (alike) { orient.add(f); clashes.set(f, (clashes.get(f) ?? 0) + 1); }
      }
    }
  }
  const area = new Set();
  for (const [f, r] of ratios ?? []) if (!(r >= 1 / LOOP_AREA_RATIO && r <= LOOP_AREA_RATIO)) area.add(f);
  const missing = new Set([...new Set(labels)].filter((g) => !present.has(g)));
  for (const f of invalid) orient.delete(f);
  const minus = (s, ...others) => new Set([...s].filter((x) => !others.some((o) => o.has(x))));
  tick('problems', tpp);
  const out = { missing, invalid, orient, area: minus(area, invalid, orient), open: minus(opened, invalid, orient) };
  // which of an orient pair is the turned face: the share of each flagged patch's shared edges used alike (a face
  // turned over clashes on all of them, each neighbour on the one it shares). Not enumerable: the kinds are Sets.
  Object.defineProperty(out, 'orientShare', { value: new Map([...orient].map((f) => [f, (clashes.get(f) ?? 0) / Math.max(pairs.get(f) ?? 1, 1)])) });
  return out;
}

/**
 * The problems of a region built alone (buildTopo's only = region): its faces checked as a compound, faces and
 * their wires, edges and vertices (no solid: the rest of the part is not there), and free edges only where
 * both sides are in the region. What a trial of a region is judged by; the part is still judged by a whole build.
 */
export function regionProblems(oc, r, region) {
  const E = enumOf(oc);
  const t = performance.now();
  const builder = new oc.BRep_Builder();
  const comp = new oc.TopoDS_Compound();
  builder.MakeCompound(comp);
  for (const fc of subShapes(oc, r.shape, E.FACE)) builder.Add(comp, fc);
  const analyzer = new oc.BRepCheck_Analyzer(comp, true, false);
  tick('region_check', t);
  return patchProblems(oc, r.shape, r.faceOf, [...region], r.ratios, null, { analyzer, boundary: r.boundaryEdges });
}

/** (recreated, closed & valid, fewer problems, fewer free edges, smaller volume error), to compare. */
export function grade(oc, shape, mesh, problems, sc = score(oc, shape, mesh)) {
  const solid = !!(sc.closed && sc.valid);
  const ve = sc.vol_err;
  const ok = solid && ve !== null && ve < 0.01 && !sc.off_mesh;
  let nprob = 0;
  for (const v of Object.values(problems)) nprob += v.size;
  return { g: [ok ? 1 : 0, solid ? 1 : 0, -nprob, -(sc.free_edges ?? 0), -(ve ?? 1e9)], sc };
}

export function better(a, b) {
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return a[i] > b[i];
  return false;
}

// ── tidy ─────────────────────────────────────────────────────────────────────────────────────────────
//
// What a CAD user meets after import: tiny sliver faces cannot be deleted or edited, and one surface in
// several faces is several faces to select. Where a group's boundary pinches beside a loose fit (the
// user's chain: a cylinder strip 0.05 off its vertices), the shape repair splits the pinched loop off
// as its own face, 1e-8 of the area. So a finished solid gets two OCCT passes, each kept only when the
// solid stays closed, valid and within the volume bar:
//   ShapeFix_FixSmallFace          faces smaller than SMALL_FACE x diagonal (spots and strips) removed
//   ShapeUpgrade_UnifySameDomain   adjacent faces on one surface merged, edges on one line or circle joined
export const TIDY = { SMALL_FACE: 8e-4 };

export function tidy(oc, shape, mesh, info, { smallFaces = true } = {}) {
  const E = enumOf(oc);
  const nFaces = (sh) => subShapes(oc, sh, E.FACE).length;
  const good = (sc) => sc.closed && sc.valid && sc.vol_err !== null && sc.vol_err < 0.01 && !sc.off_mesh;
  const nEdges = (sh) => subShapes(oc, sh, E.EDGE).length;
  // A pass is kept only if it loosens nothing: removing a sliver or joining edges covers the gap it leaves with
  // tolerance, 1e-5 -> 0.254 mm on 00000034_014 for one sliver, 0.004 -> 0.049 mm on 00008869_006 for 16 joined edges.
  const tolOf = (sh) => Math.max(oc.BRep_Tool.MaxTolerance(sh, E.VERTEX), oc.BRep_Tool.MaxTolerance(sh, E.EDGE));
  let cur = shape, faces = nFaces(shape);
  const tol0 = tolOf(shape);
  const tight = (sh) => tolOf(sh) <= tol0 * 1.01;
  const copyOf = (sh) => new oc.BRepBuilderAPI_Copy_2(sh, true, false).Shape();
  // A pass that removed or joined nothing is not checked: a check (score) costs as much as a build, and on most
  // parts neither pass changes anything (2026-11-01: tidy was 91 of the 399 s on 229 fully analytic parts).
  // smallFaces false (a solid with facets): FixSmallFace removes a small face by moving its neighbours' vertices
  // within its 10 x SMALL_FACE tolerance, which tilted facets out of their planes by up to 20 degrees under a
  // 0.61 mm tolerance (00008335_005, test/facetAudit.mjs) while the volume error stayed the same.
  if (smallFaces) try {
    const f = new oc.ShapeFix_FixSmallFace();
    f.Init(copyOf(cur));
    f.SetPrecision(TIDY.SMALL_FACE * mesh.diag);
    f.SetMaxTolerance(10 * TIDY.SMALL_FACE * mesh.diag);
    f.Perform();
    const fixed = f.FixShape();
    if (nFaces(fixed) < faces) {
      const sc = score(oc, fixed, mesh);
      const n = sc.solid ? nFaces(sc.solid) : faces;
      if (good(sc) && n < faces) {
        if (tight(sc.solid)) { info.tidy_small_faces = faces - n; cur = sc.solid; faces = n; }
        else bump(info, 'tidy_rejected_loose');
      }
    }
  } catch (err) { info.tidy_small_error = String(err?.message ?? err).slice(0, 80); }
  try {
    const u = new oc.ShapeUpgrade_UnifySameDomain_2(copyOf(cur), true, true, false);
    u.SetLinearTolerance(1e-7 * mesh.diag);
    u.SetAngularTolerance(1e-9);
    u.Build();
    const unified = u.Shape();
    if (nFaces(unified) < faces || nEdges(unified) < nEdges(cur)) {
      const sc = score(oc, unified, mesh);
      if (good(sc) && !tight(sc.solid)) bump(info, 'tidy_rejected_loose');
      else if (good(sc)) {
        const n = nFaces(sc.solid);
        if (n < faces) info.tidy_unified_faces = faces - n;
        const ne = nEdges(sc.solid), ne0 = nEdges(cur);
        if (ne < ne0) info.tidy_unified_edges = ne0 - ne;
        cur = sc.solid;
      }
    }
  } catch (err) { info.tidy_unify_error = String(err?.message ?? err).slice(0, 80); }
  return cur;
}
