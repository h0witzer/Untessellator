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
 *   repair  one ShapeFix_Shape over the shell, then natural-bound wires it added are dropped
 */

import { distNormal, projectOnto, seamAway, unfoldChain, interpolateChain, unit, sub, dot, scale, norm, cross,
  tangentLocus, ontoLocus, cornerOnLocus, exactCurve } from './geometry.js';
import { meshTopology, BODY_STRIDE } from './mesh.js';

const bump = (info, k, n = 1) => { info[k] = (info[k] ?? 0) + n; };

function pnt(oc, p) { return new oc.gp_Pnt_3(p[0], p[1], p[2]); }
function dir(oc, v) { return new oc.gp_Dir_4(v[0], v[1], v[2]); }
function enumOf(oc) {
  return {
    FACE: oc.TopAbs_ShapeEnum.TopAbs_FACE, EDGE: oc.TopAbs_ShapeEnum.TopAbs_EDGE, WIRE: oc.TopAbs_ShapeEnum.TopAbs_WIRE,
    SHELL: oc.TopAbs_ShapeEnum.TopAbs_SHELL, SHAPE: oc.TopAbs_ShapeEnum.TopAbs_SHAPE, REVERSED: oc.TopAbs_Orientation.TopAbs_REVERSED,
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
/** Switches for experiments (test/knit.mjs): exact line / circle edges and contact loci. */
export const BUILD = { EXACT_EDGES: true, CONTACT: true };

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
function chainEdge(oc, pts, closed, v0, v1) {
  const { poles, knots, mults, degree } = interpolateChain(pts, closed);
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
 * Build the shell. surfaces: Map label -> [kind, params]. facetLabels: labels of ONE triangle each,
 * built as a plane through its three corners with straight edges (the facet fallback). Returns
 * {shape, faceOf: Map label -> face (or the pieces the repair split it into), info, ratios}.
 */
export function buildTopo(oc, mesh, surfaces, { facetLabels = null } = {}) {
  const E = enumOf(oc);
  const { V, T, F, diag: scaleLen } = mesh;
  const info = {};
  const facets = facetLabels ?? new Set();
  const forced = facets.size ? [...F.keys()].filter((t) => facets.has(F[t])) : null;
  const topo = meshTopology(mesh, forced);
  const { fb, facesAt, corners, chains } = topo;
  Object.assign(info, topo.info);
  const surf = (f) => surfaces.get(f) ?? null;
  const ck = (v, b) => v * BODY_STRIDE + b;

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
  const vertex = new Map(), cornerPos = new Map();
  for (const c of corners) {
    const v = Math.floor(c / BODY_STRIDE);
    const ss = [...facesAt.get(c)].map(surf).filter(Boolean);
    let x = ss.length ? projectOnto(ss, V[v], scaleLen) : V[v];
    // a corner of two tangent surfaces lies on their contact locus: solved along it, not by projection
    if (BUILD.CONTACT) tangent: for (let i = 0; i < ss.length; i++) for (let j = i + 1; j < ss.length; j++) {
      const L = tangentLocus(ss[i], ss[j], scaleLen);
      if (!L) continue;
      // only where the corner then lies on every one of its surfaces: a 4-face corner that cannot be on
      // the locus and on the rest (inconsistent fits) keeps its projected position
      const y = cornerOnLocus(L, ss.filter((_, k) => k !== i && k !== j), x, scaleLen);
      if (y && ss.every(([k, p]) => Math.abs(distNormal(k, p, y)[0]) <= CONTACT_TOL * scaleLen)) { x = y; bump(info, 'corner_on_contact'); }
      break tangent;
    }
    cornerPos.set(c, x);
    const gp = pnt(oc, x);
    vertex.set(c, new oc.BRepBuilderAPI_MakeVertex(gp).Vertex());
    gp.delete();
  }

  // edges
  const edges = [];
  for (const { a, b, path } of chains) {
    const ss = [surf(a), b >= 0 ? surf(b) : null].filter(Boolean);
    const bd = fb.get(a);
    let locus = BUILD.CONTACT && ss.length === 2 ? tangentLocus(ss[0], ss[1], scaleLen) : null;
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
    // mesh edge): a chord where it meets a curved face, which the repair covers by tolerance.
    const straight = bothPlanes || facets.has(a) || facets.has(b);
    const v0 = vertex.get(ck(path[0], bd)), v1 = vertex.get(ck(path[path.length - 1], bd));
    try {
      let mk;
      if (straight && !closed) {
        mk = new oc.BRepBuilderAPI_MakeEdge_2(v0, v1);
      } else {
        const body = closed ? pts.slice(0, -1) : pts;
        let clean = [body[0]];
        for (const q of body.slice(1)) if (norm(sub(q, clean[clean.length - 1])) > 1e-9 * scaleLen) clean.push(q);
        if (!closed) clean[clean.length - 1] = pts[pts.length - 1];
        clean = unfoldChain(clean, closed, info);
        if (clean.length < 2 || (closed && clean.length < 3)) { bump(info, 'edge_too_short'); continue; }
        // an exact line or circle where the two surfaces meet in one: CAD tools treat analytic edges
        // as the intersections they are (fillets, face deletion); a spline through them is only close
        const ex = BUILD.EXACT_EDGES && ss.length === 2 ? exactCurve(closed ? [...clean, clean[0]] : clean, EXACT_EDGE_TOL * scaleLen) : null;
        mk = ex ? exactEdge(oc, ex, closed ? [...clean, clean[0]] : clean, closed, v0, v1) : null;
        if (mk && mk.IsDone()) bump(info, ex.line ? 'edges_line' : 'edges_circle');
        else mk = chainEdge(oc, clean, closed, v0, v1);
      }
      if (!mk.IsDone()) { bump(info, 'edge_failed'); continue; }
      edges.push({ a, b, s: ck(path[0], bd), t: ck(path[path.length - 1], bd), e: mk.Edge() });
    } catch (err) {
      bump(info, 'edge_exception');
    }
  }

  // faces
  const byFace = new Map();
  const push = (f, item) => { let l = byFace.get(f); if (!l) byFace.set(f, (l = [])); l.push(item); };
  for (const { a, b, s, t, e } of edges) {
    push(a, [s, t, e]);
    if (b >= 0) push(b, [t, s, e.Reversed()]);
  }
  const builder = new oc.BRep_Builder();
  const shell = new oc.TopoDS_Shell();
  builder.MakeShell(shell);
  const faceOf = new Map();
  for (const f of labelsSorted) {
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
    builder.Add(shell, face);
    faceOf.set(f, face);
    bump(info, 'faces');
    if (facet) bump(info, 'facet_faces');
  }

  // ONE repair pass over the whole shell: ShapeFix_Shape records every replacement in one context and
  // applies it to all faces, so shared edges stay shared.
  const fix = new oc.ShapeFix_Shape_2(shell);
  fix.Perform(new oc.Message_ProgressRange_1());
  let result = fix.Shape();
  const ctx = fix.Context().get();
  const remapped = new Map();
  for (const [f, fc] of faceOf) {
    const r = ctx.Apply_2(fc, E.SHAPE);
    if (r.IsNull()) { bump(info, 'face_removed_by_repair'); continue; }
    if (r.ShapeType() === E.FACE) remapped.set(f, oc.TopoDS.Face_1(r));
    else if (subShapes(oc, r, E.FACE).length) { remapped.set(f, r); bump(info, 'face_split_by_repair'); }
  }

  // DROP NATURAL-BOUND WIRES the repair added: on a face that has real boundary edges, a wire with no
  // edge shared by another face is not part of the patch's boundary (a torus face otherwise becomes
  // the whole torus minus the patch).
  const m = new oc.TopTools_IndexedDataMapOfShapeListOfShape_1();
  oc.TopExp.MapShapesAndAncestors(result, E.EDGE, E.FACE, m);
  const shared = (e, face) => {
    const i = m.FindIndex(e);
    if (i <= 0) return false;
    const l = m.FindFromIndex(i);
    if (l.Size() > 2) return true;
    return (l.Size() >= 1 && !l.First_1().IsSame(face)) || (l.Size() === 2 && !l.Last_1().IsSame(face));
  };
  const cleaned = new oc.TopoDS_Shell();
  builder.MakeShell(cleaned);
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
      builder.Add(cleaned, oriented);
      replaced.push([fc, oriented]);
      bump(info, 'natural_bound_wires_removed', all.length - keep.length);
    } else {
      builder.Add(cleaned, fc);
    }
  }
  if (replaced.length) {
    for (const [f, fc] of remapped) {
      const hit = replaced.find(([old]) => old.IsSame(fc));
      if (hit) remapped.set(f, hit[1]);
    }
    result = cleaned;
  }

  // Area of every patch's final face against the patch's mesh area: a face that took the wrong region
  // of its surface shows up here even when the shell is closed and valid.
  const ratios = new Map();
  for (const [f, fc] of remapped) {
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
  return { shape: result, faceOf: remapped, info, ratios };
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
    const solidOf = (shell) => {
      let solid = new oc.TopoDS_Solid();
      builder.MakeSolid(solid);
      builder.Add(solid, shell);
      const cls = new oc.BRepClass3d_SolidClassifier_2(solid);
      cls.PerformInfinitePoint(1e-7);
      if (cls.State() === E.IN) solid = oc.TopoDS.Solid_1(solid.Reversed());
      return [solid, new oc.BRepCheck_Analyzer(solid, true, false).IsValid_2()];
    };
    for (const sh of shells) {
      // The shell as built first: its faces follow the mesh winding, made consistent by KnitMesh, so it
      // usually needs nothing. ml/solid_rebuild.py runs ShapeFix_Shell's whole Perform, but in this
      // kernel (OCCT 7.6) its per-face pass can make a valid shell unorientable (00032949_19 gmsh: valid
      // raw and orientation-fixed, UnorientableShape on 2 faces after Perform; OCCT 7.9 keeps it valid),
      // and its orientation pass is quadratic: 776 s on a 33,802-facet shell. So an invalid shell gets
      // the orientation fix, then Perform, and only up to SHELL_FIX_MAX_FACES faces.
      let [solid, ok] = solidOf(sh);
      if (!ok && subShapes(oc, sh, E.FACE).length <= SHELL_FIX_MAX_FACES) {
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
      valid = valid && ok;
      const g = new oc.GProp_GProps_1();
      oc.BRepGProp.VolumeProperties_1(solid, g, false, false, false);
      total += g.Mass();
      builder.Add(comp, solid);
    }
    out.valid = valid && shells.length >= 1;
    out.volume = total;
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
 * open (bounded by a free edge). Faces are checked as they sit in `result`.
 */
export function patchProblems(oc, result, faceOf, labels, ratios) {
  const E = enumOf(oc);
  const ownMap = new oc.TopTools_IndexedMapOfShape_1();
  const owner = [];
  for (const [f, sh] of faceOf) {
    for (const fc of subShapes(oc, sh, E.FACE)) {
      const i = ownMap.Add(fc);
      owner[i] = f;
    }
  }
  const ownerOf = (fc) => { const i = ownMap.FindIndex(fc); return i > 0 ? owner[i] : undefined; };
  const ana = new oc.BRepCheck_Analyzer(result, true, false);
  const invalid = new Set(), present = new Set();
  const faces = subShapes(oc, result, E.FACE);
  for (const fc of faces) {
    const f = ownerOf(fc);
    if (f === undefined) continue;
    present.add(f);
    if (invalid.has(f)) continue;
    try {
      if (!ana.IsValid_1(fc)) { invalid.add(f); continue; }
      for (const type of [E.WIRE, E.EDGE]) {
        if (subShapes(oc, fc, type).some((s) => !ana.IsValid_1(s))) { invalid.add(f); break; }
      }
    } catch {
      invalid.add(f);
    }
  }
  const edgeMap = new oc.TopTools_IndexedMapOfShape_1();
  const use = [];
  for (const fc of faces) {
    const f = ownerOf(fc);
    const face = oc.TopoDS.Face_1(fc);
    const flipped = fc.Orientation_1() === E.REVERSED;
    for (const es of subShapes(oc, fc, E.EDGE)) {
      const e = oc.TopoDS.Edge_1(es);
      if (oc.BRep_Tool.Degenerated(e) || oc.BRep_Tool.IsClosed_2(e, face)) continue;
      const i = edgeMap.Add(e);
      (use[i] ??= []).push([f, (e.Orientation_1() === E.REVERSED) !== flipped]);
    }
  }
  const opened = new Set(), orient = new Set();
  for (const lst of use) {
    if (!lst) continue;
    if (lst.length === 1 && lst[0][0] !== undefined) opened.add(lst[0][0]);
    else if (lst.length === 2 && lst[0][1] === lst[1][1]) for (const [f] of lst) if (f !== undefined) orient.add(f);
  }
  const area = new Set();
  for (const [f, r] of ratios ?? []) if (!(r >= 1 / LOOP_AREA_RATIO && r <= LOOP_AREA_RATIO)) area.add(f);
  const missing = new Set([...new Set(labels)].filter((g) => !present.has(g)));
  for (const f of invalid) orient.delete(f);
  const minus = (s, ...others) => new Set([...s].filter((x) => !others.some((o) => o.has(x))));
  return { missing, invalid, orient, area: minus(area, invalid, orient), open: minus(opened, invalid, orient) };
}

/** (recreated, closed & valid, fewer problems, fewer free edges, smaller volume error), to compare. */
export function grade(oc, shape, mesh, problems) {
  const sc = score(oc, shape, mesh);
  const solid = !!(sc.closed && sc.valid);
  const ve = sc.vol_err;
  const ok = solid && ve !== null && ve < 0.01;
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

export function tidy(oc, shape, mesh, info) {
  const E = enumOf(oc);
  const nFaces = (sh) => subShapes(oc, sh, E.FACE).length;
  const good = (sc) => sc.closed && sc.valid && sc.vol_err !== null && sc.vol_err < 0.01;
  let cur = shape, faces = nFaces(shape);
  try {
    const f = new oc.ShapeFix_FixSmallFace();
    f.Init(cur);
    f.SetPrecision(TIDY.SMALL_FACE * mesh.diag);
    f.SetMaxTolerance(10 * TIDY.SMALL_FACE * mesh.diag);
    f.Perform();
    const sc = score(oc, f.FixShape(), mesh);
    const n = sc.solid ? nFaces(sc.solid) : faces;
    if (good(sc) && n < faces) { info.tidy_small_faces = faces - n; cur = sc.solid; faces = n; }
  } catch (err) { info.tidy_small_error = String(err?.message ?? err).slice(0, 80); }
  try {
    const u = new oc.ShapeUpgrade_UnifySameDomain_2(cur, true, true, false);
    u.SetLinearTolerance(1e-7 * mesh.diag);
    u.SetAngularTolerance(1e-9);
    u.Build();
    const sc = score(oc, u.Shape(), mesh);
    if (good(sc)) {
      const n = nFaces(sc.solid);
      if (n < faces) info.tidy_unified_faces = faces - n;
      cur = sc.solid;
    }
  } catch (err) { info.tidy_unify_error = String(err?.message ?? err).slice(0, 80); }
  return cur;
}
