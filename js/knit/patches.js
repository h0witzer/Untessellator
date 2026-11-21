/**
 * knit/patches.js — a closed solid from untrimmed Bezier patches that meet along shared boundary curves (the output
 * of js/subd/fastNurbs.js): one vertex per cage vertex, one edge per cage edge with its 3D curve taken from a patch
 * boundary and a straight pcurve in each face's parameter square, one face per patch. Valid by construction: there
 * is nothing to project, sew or repair, because neighbouring patches share their boundary curves exactly and with
 * the same parameter.
 */

function curveOf(oc, poles, p) {                     // a Bezier curve as a clamped B-spline over [0, 1]
  const P = new oc.TColgp_Array1OfPnt_2(1, p + 1);
  poles.forEach((q, i) => { const g = new oc.gp_Pnt_3(q[0], q[1], q[2]); P.SetValue(i + 1, g); g.delete(); });
  const K = new oc.TColStd_Array1OfReal_2(1, 2), M = new oc.TColStd_Array1OfInteger_2(1, 2);
  K.SetValue(1, 0); K.SetValue(2, 1); M.SetValue(1, p + 1); M.SetValue(2, p + 1);
  return new oc.Handle_Geom_Curve_2(new oc.Geom_BSplineCurve_1(P, K, M, p, false));
}

function lineOf(oc, a, b) {                          // a 2D segment over [0, 1] as a degree-1 B-spline
  const P = new oc.TColgp_Array1OfPnt2d_2(1, 2);
  const g0 = new oc.gp_Pnt2d_3(a[0], a[1]), g1 = new oc.gp_Pnt2d_3(b[0], b[1]);
  P.SetValue(1, g0); P.SetValue(2, g1); g0.delete(); g1.delete();
  const K = new oc.TColStd_Array1OfReal_2(1, 2), M = new oc.TColStd_Array1OfInteger_2(1, 2);
  K.SetValue(1, 0); K.SetValue(2, 1); M.SetValue(1, 2); M.SetValue(2, 2);
  return new oc.Handle_Geom2d_Curve_2(new oc.Geom2d_BSplineCurve_1(P, K, M, 1, false));
}

function surfaceOf(oc, f) {
  const p = f.uCount - 1, q = f.vCount - 1;
  const P = new oc.TColgp_Array2OfPnt_2(1, f.uCount, 1, f.vCount);
  for (let i = 0; i < f.uCount; i++) for (let j = 0; j < f.vCount; j++) {
    const o = (i * f.vCount + j) * 3, g = new oc.gp_Pnt_3(f.poles[o], f.poles[o + 1], f.poles[o + 2]);
    P.SetValue(i + 1, j + 1, g); g.delete();
  }
  const knots = (deg) => {
    const K = new oc.TColStd_Array1OfReal_2(1, 2), M = new oc.TColStd_Array1OfInteger_2(1, 2);
    K.SetValue(1, 0); K.SetValue(2, 1); M.SetValue(1, deg + 1); M.SetValue(2, deg + 1);
    return [K, M];
  };
  const [ku, mu] = knots(p), [kv, mv] = knots(q);
  return new oc.Handle_Geom_Surface_2(new oc.Geom_BSplineSurface_1(P, ku, kv, mu, mv, p, q, false, false));
}

/** The poles of side k of a patch, running from loop corner k to corner k + 1. */
function sidePoles(f, k) {
  const p = f.uCount - 1, at = (i, j) => { const o = (i * f.vCount + j) * 3; return [f.poles[o], f.poles[o + 1], f.poles[o + 2]]; };
  const idx = Array.from({ length: p + 1 }, (_, m) => m);
  if (k === 0) return idx.map((m) => at(m, 0));
  if (k === 1) return idx.map((m) => at(p, m));
  if (k === 2) return idx.map((m) => at(p - m, p));
  return idx.map((m) => at(0, p - m));
}

const UV = [[0, 0], [1, 0], [1, 1], [0, 1]];

/**
 * A solid from patches on a closed quad cage. Returns { solid, valid (null if unchecked), ms }.
 * faces[f]: Bezier patch of cage face f (fastNurbs output); cageFaces: loops, counter-clockwise from outside.
 */
export function patchSolid(oc, { faces, cageFaces }, { check = true, tol = 1e-7 } = {}) {
  const t0 = performance.now();
  const builder = new oc.BRep_Builder();
  const vertex = new Map();
  cageFaces.forEach((q, f) => {
    q.forEach((v, k) => {
      if (vertex.has(v)) return;
      const c = sidePoles(faces[f], k)[0], g = new oc.gp_Pnt_3(c[0], c[1], c[2]);
      vertex.set(v, new oc.BRepBuilderAPI_MakeVertex(g).Vertex());
      g.delete();
    });
  });
  const edges = new Map();                            // "a,b" with a < b -> TopoDS_Edge running a -> b
  const edgeOf = (a, b, f, k) => {
    const key = a < b ? `${a},${b}` : `${b},${a}`;
    if (!edges.has(key)) {
      let poles = sidePoles(faces[f], k);
      if (a > b) poles = poles.slice().reverse();       // the edge runs from the smaller vertex id
      const [s, e] = a < b ? [a, b] : [b, a];
      const mk = new oc.BRepBuilderAPI_MakeEdge_29(curveOf(oc, poles, poles.length - 1), vertex.get(s), vertex.get(e), 0, 1);
      edges.set(key, mk.Edge());
    }
    return edges.get(key);
  };
  const shell = new oc.TopoDS_Shell();
  builder.MakeShell(shell);
  cageFaces.forEach((q, f) => {
    const face = new oc.TopoDS_Face();
    builder.MakeFace_2(face, surfaceOf(oc, faces[f]), tol);
    const wire = new oc.TopoDS_Wire();
    builder.MakeWire(wire);
    for (let k = 0; k < 4; k++) {
      const a = q[k], b = q[(k + 1) % 4], e = edgeOf(a, b, f, k);
      const forward = a < b;
      // the pcurve follows the edge's own direction (smaller vertex id first)
      const [p0, p1] = forward ? [UV[k], UV[(k + 1) % 4]] : [UV[(k + 1) % 4], UV[k]];
      builder.UpdateEdge_3(e, lineOf(oc, p0, p1), face, tol);
      builder.Add(wire, forward ? e : e.Reversed());
    }
    builder.Add(face, wire);
    builder.Add(shell, face);
  });
  for (const e of edges.values()) builder.SameParameter(e, true);
  let solid = new oc.TopoDS_Solid();
  builder.MakeSolid(solid);
  builder.Add(solid, shell);
  const cls = new oc.BRepClass3d_SolidClassifier_2(solid);
  cls.PerformInfinitePoint(1e-7);
  if (cls.State() === oc.TopAbs_State.TopAbs_IN) solid = oc.TopoDS.Solid_1(solid.Reversed());
  const t1 = performance.now();
  let valid = null;
  if (check) valid = new oc.BRepCheck_Analyzer(solid, true, false).IsValid_2();
  return { solid, valid, ms: { build: t1 - t0, check: performance.now() - t1 } };
}
