/**
 * knit/preview.js — the built solid as triangles, for the studio to show before the STEP is saved: each face as
 * the kernel tessellates it (BRepMesh), with its surface type and normals smoothed within the face (a crease at
 * every face boundary, as CAD draws a solid), and every edge as line segments along the tessellation's own
 * boundary, so a line never floats off the faces it bounds. Seams of closed surfaces are not drawn.
 *
 * Out: {positions: Float32Array (9 per triangle), normals: Float32Array (9 per triangle), face: Int32Array (per
 * triangle), types: [per face: 'plane', 'cylinder', 'cone', 'sphere', 'torus' or 'freeform' (any other surface)],
 * edges: Float32Array (6 per segment), faces: count}. A face the kernel could not tessellate is left out.
 */

import { subShapes } from './build.js';

/**
 * The tessellation's angle: about 7 degrees a facet, finer than most CAD exports' meshes, so the solid never
 * looks coarser than the mesh it came from (at 20 degrees its cylinders were visibly polygonal).
 */
const ANGLE = 0.12;

/** Tessellate `shape` to within `deflection` (model units) and return its triangles and edges. */
export function previewMesh(oc, shape, deflection) {
  new oc.BRepMesh_IncrementalMesh_2(shape, deflection, false, ANGLE, false).delete();
  const S = oc.GeomAbs_SurfaceType;
  const typeOf = new Map([[S.GeomAbs_Plane, 'plane'], [S.GeomAbs_Cylinder, 'cylinder'], [S.GeomAbs_Cone, 'cone'], [S.GeomAbs_Sphere, 'sphere'],
    [S.GeomAbs_Torus, 'torus']]);
  const reversed = oc.TopAbs_Orientation.TopAbs_REVERSED;
  const EDGE = oc.TopAbs_ShapeEnum.TopAbs_EDGE;
  const positions = [], normals = [], faceOf = [], types = [], edges = [];
  const drawn = new oc.TopTools_IndexedMapOfShape_1();   // edges done: a shared edge is met from both its faces
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
    // the face's edges, through the nodes the tessellation put on them
    for (const e of subShapes(oc, face, EDGE)) {
      const edge = oc.TopoDS.Edge_1(e);
      if (drawn.Contains(edge) || oc.BRep_Tool.Degenerated(edge) || oc.BRep_Tool.IsClosed_2(edge, face)) continue;
      drawn.Add(edge);
      const poly = oc.BRep_Tool.PolygonOnTriangulation_1(edge, handle, loc);
      if (poly.IsNull()) { sampleEdge(oc, edge, edges); continue; }
      const p = poly.get();
      for (let k = 1; k < p.NbNodes(); k++) {
        const a = (p.Node(k) - 1) * 3, b = (p.Node(k + 1) - 1) * 3;
        edges.push(nodes[a], nodes[a + 1], nodes[a + 2], nodes[b], nodes[b + 1], nodes[b + 2]);
      }
    }
    loc.delete();
  }
  drawn.delete();
  return {
    positions: Float32Array.from(positions), normals: Float32Array.from(normals), face: Int32Array.from(faceOf),
    types, edges: Float32Array.from(edges), faces: types.length,
  };
}

/** An edge the tessellation has no polygon for, sampled along its curve instead. */
function sampleEdge(oc, edge, out) {
  const curve = new oc.BRepAdaptor_Curve_2(edge);
  const u0 = curve.FirstParameter(), u1 = curve.LastParameter();
  const n = curve.GetType() === oc.GeomAbs_CurveType.GeomAbs_Line ? 1 : 32;
  let prev = null;
  for (let k = 0; k <= n; k++) {
    const p = curve.Value(u0 + ((u1 - u0) * k) / n);
    const q = [p.X(), p.Y(), p.Z()];
    p.delete();
    if (prev) out.push(...prev, ...q);
    prev = q;
  }
  curve.delete();
}
