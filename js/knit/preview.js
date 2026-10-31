/**
 * knit/preview.js — the built solid as triangles, for the studio to show before the STEP is saved: each face as
 * the kernel tessellates it (BRepMesh), with its surface type and normals smoothed within the face (a crease at
 * every face boundary, as CAD draws a solid), and every edge as line segments.
 *
 * Out: {positions: Float32Array (9 per triangle), normals: Float32Array (9 per triangle), face: Int32Array (per
 * triangle), types: [per face: 'plane', 'cylinder', 'cone', 'sphere', 'torus' or 'freeform' (any other surface)],
 * edges: Float32Array (6 per segment), faces: count}. A face the kernel could not tessellate is left out.
 */

import { subShapes } from './build.js';

/** Tessellate `shape` to within `deflection` (model units) and return its triangles and edges. */
export function previewMesh(oc, shape, deflection) {
  new oc.BRepMesh_IncrementalMesh_2(shape, deflection, false, 0.35, false).delete();
  const S = oc.GeomAbs_SurfaceType;
  const typeOf = new Map([[S.GeomAbs_Plane, 'plane'], [S.GeomAbs_Cylinder, 'cylinder'], [S.GeomAbs_Cone, 'cone'], [S.GeomAbs_Sphere, 'sphere'],
    [S.GeomAbs_Torus, 'torus']]);
  const reversed = oc.TopAbs_Orientation.TopAbs_REVERSED;
  const positions = [], normals = [], faceOf = [], types = [];
  for (const sub of subShapes(oc, shape, oc.TopAbs_ShapeEnum.TopAbs_FACE)) {
    const face = oc.TopoDS.Face_1(sub);
    const loc = new oc.TopLoc_Location_1();
    const handle = oc.BRep_Tool.Triangulation(face, loc, 0);
    if (handle.IsNull()) { loc.delete(); continue; }
    const tri = handle.get();
    const trsf = loc.Transformation();
    const nodes = new Float64Array(tri.NbNodes() * 3);
    for (let i = 1; i <= tri.NbNodes(); i++) {
      const p = tri.Node(i).Transformed(trsf);
      nodes.set([p.X(), p.Y(), p.Z()], (i - 1) * 3);
      p.delete();
    }
    const flip = face.Orientation_1() === reversed;
    const corners = [];
    for (let i = 1; i <= tri.NbTriangles(); i++) {
      const t = tri.Triangle(i);
      const a = t.Value(1) - 1, b = t.Value(2) - 1, c = t.Value(3) - 1;
      corners.push(...(flip ? [a, c, b] : [a, b, c]));
    }
    // normals smoothed over the face's own triangles, area-weighted
    const vn = new Float64Array(nodes.length);
    for (let i = 0; i < corners.length; i += 3) {
      const [a, b, c] = [corners[i] * 3, corners[i + 1] * 3, corners[i + 2] * 3];
      const u = [nodes[b] - nodes[a], nodes[b + 1] - nodes[a + 1], nodes[b + 2] - nodes[a + 2]];
      const v = [nodes[c] - nodes[a], nodes[c + 1] - nodes[a + 1], nodes[c + 2] - nodes[a + 2]];
      const n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
      for (const o of [a, b, c]) for (let k = 0; k < 3; k++) vn[o + k] += n[k];
    }
    const index = types.length;
    const adaptor = new oc.BRepAdaptor_Surface_2(face, true);
    types.push(typeOf.get(adaptor.GetType()) ?? 'freeform');
    adaptor.delete();
    for (const i of corners) {
      const o = i * 3, len = Math.hypot(vn[o], vn[o + 1], vn[o + 2]) || 1;
      positions.push(nodes[o], nodes[o + 1], nodes[o + 2]);
      normals.push(vn[o] / len, vn[o + 1] / len, vn[o + 2] / len);
    }
    for (let i = 0; i < corners.length; i += 3) faceOf.push(index);
    loc.delete();
  }
  // each edge once (a shared edge is met from both its faces), sampled along its curve
  const edges = [];
  const map = new oc.TopTools_IndexedMapOfShape_1();
  oc.TopExp.MapShapes_1(shape, oc.TopAbs_ShapeEnum.TopAbs_EDGE, map);
  const LINE = oc.GeomAbs_CurveType.GeomAbs_Line;
  for (let i = 1; i <= map.Extent(); i++) {
    const edge = oc.TopoDS.Edge_1(map.FindKey(i));
    if (oc.BRep_Tool.Degenerated(edge)) continue;
    const curve = new oc.BRepAdaptor_Curve_2(edge);
    const u0 = curve.FirstParameter(), u1 = curve.LastParameter();
    const n = curve.GetType() === LINE ? 1 : 32;
    let prev = null;
    for (let k = 0; k <= n; k++) {
      const p = curve.Value(u0 + ((u1 - u0) * k) / n);
      const q = [p.X(), p.Y(), p.Z()];
      p.delete();
      if (prev) edges.push(...prev, ...q);
      prev = q;
    }
    curve.delete();
  }
  map.delete();
  return {
    positions: Float32Array.from(positions), normals: Float32Array.from(normals), face: Int32Array.from(faceOf),
    types, edges: Float32Array.from(edges), faces: types.length,
  };
}
