/**
 * facetAudit.mjs — how far a knitted solid's facets sit from what they were built from: every planar face with
 * three edges and three vertices, against its own corners and against the mesh triangle it came from.
 *
 *   off_plane    a vertex's distance from its face's plane (/ diagonal): the plane and its corners agree
 *   tilt_deg     the plane's normal against the normal of its three corners (degrees)
 *   normal_deg   the face's outward normal against the mesh triangle's (degrees): split into facets whose three
 *                corners are mesh vertices exactly ("exact") and those with a corner moved onto a neighbouring
 *                analytic surface ("moved", by design)
 *   moved        a corner's distance from its mesh vertex (/ diagonal)
 *   tol          the largest vertex or edge tolerance on a facet (mm, and / diagonal)
 */

const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm = (a) => Math.hypot(a[0], a[1], a[2]);
const angleDeg = (a, b) => Math.atan2(norm(cross(a, b)), dot(a, b)) * 180 / Math.PI;

export function facetAudit(oc, shape, positions) {
  // the mesh: welded vertices (exact float keys, as KnitMesh welds them), triangles by their corner ids
  const ids = new Map(), P = [];
  const id = (o) => {
    const k = `${positions[o]},${positions[o + 1]},${positions[o + 2]}`;
    let i = ids.get(k);
    if (i === undefined) { i = P.length; ids.set(k, i); P.push([positions[o], positions[o + 1], positions[o + 2]]); }
    return i;
  };
  const tris = new Map();
  for (let o = 0; o + 9 <= positions.length; o += 9) {
    const c = [id(o), id(o + 3), id(o + 6)];
    const n = cross(sub(P[c[1]], P[c[0]]), sub(P[c[2]], P[c[0]]));
    tris.set([...c].sort((a, b) => a - b).join(','), n);
  }
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const p of P) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], p[k]); hi[k] = Math.max(hi[k], p[k]); }
  const diag = norm(sub(hi, lo));
  const cell = diag / 2000;
  const grid = new Map();
  const cellOf = (p) => p.map((x, k) => Math.floor((x - lo[k]) / cell));
  P.forEach((p, i) => { const k = cellOf(p).join(','); let l = grid.get(k); if (!l) grid.set(k, (l = [])); l.push(i); });
  const nearest = (p) => {
    const c = cellOf(p);
    let best = -1, bd = Infinity;
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (let dz = -1; dz <= 1; dz++) {
      for (const i of grid.get(`${c[0] + dx},${c[1] + dy},${c[2] + dz}`) ?? []) {
        const d = norm(sub(P[i], p));
        if (d < bd) { bd = d; best = i; }
      }
    }
    return [best, bd];
  };

  const E = oc.TopAbs_ShapeEnum;
  const out = { facets: 0, exact: 0, moved_facets: 0, unmatched: 0, flipped: 0, off_plane: 0, tilt_deg: 0,
    normal_deg_exact: 0, exact_over_1e6deg: 0, normal_deg_moved: 0, moved: 0, tol_mm: 0, tol: 0, planes: 0, plane_off: 0,
    moved_hist: { 0: 0, '<1e-12': 0, '<1e-9': 0, '<1e-6': 0, '>=1e-6': 0 } };
  const movedAngles = [];
  const ex = new oc.TopExp_Explorer_2(shape, E.TopAbs_FACE, E.TopAbs_SHAPE);
  for (; ex.More(); ex.Next()) {
    const face = oc.TopoDS.Face_1(ex.Current());
    const ad = new oc.BRepAdaptor_Surface_2(face, true);
    if (ad.GetType().value !== 0) { ad.delete(); continue; }
    const pln = ad.Plane();
    const d = pln.Axis().Direction(), l = pln.Location();
    let n = [d.X(), d.Y(), d.Z()];
    const loc = [l.X(), l.Y(), l.Z()];
    if (face.Orientation_1() === E.TopAbs_REVERSED) n = n.map((x) => -x);
    const vm = new oc.TopTools_IndexedMapOfShape_1(), em = new oc.TopTools_IndexedMapOfShape_1();
    oc.TopExp.MapShapes_1(face, E.TopAbs_VERTEX, vm);
    oc.TopExp.MapShapes_1(face, E.TopAbs_EDGE, em);
    const verts = [];
    for (let i = 1; i <= vm.Extent(); i++) verts.push(oc.TopoDS.Vertex_1(vm.FindKey(i)));
    const pts = verts.map((v) => { const p = oc.BRep_Tool.Pnt(v); return [p.X(), p.Y(), p.Z()]; });
    let off = 0;
    for (const p of pts) off = Math.max(off, Math.abs(dot(sub(p, loc), n)));
    out.planes++;
    out.plane_off = Math.max(out.plane_off, off / diag);
    if (pts.length !== 3 || em.Extent() !== 3) { vm.delete(); em.delete(); ad.delete(); continue; }
    out.facets++;
    out.off_plane = Math.max(out.off_plane, off / diag);
    const own = cross(sub(pts[1], pts[0]), sub(pts[2], pts[0]));
    const tilt = angleDeg(own, n);
    out.tilt_deg = Math.max(out.tilt_deg, Math.min(tilt, 180 - tilt));
    let tol = 0;
    for (const v of verts) tol = Math.max(tol, oc.BRep_Tool.Tolerance_3(v));
    for (let i = 1; i <= em.Extent(); i++) tol = Math.max(tol, oc.BRep_Tool.Tolerance_2(oc.TopoDS.Edge_1(em.FindKey(i))));
    out.tol_mm = Math.max(out.tol_mm, tol);
    out.tol = Math.max(out.tol, tol / diag);
    const near = pts.map(nearest);
    const moved = Math.max(...near.map(([, dd]) => dd));
    const m = tris.get(near.map(([i]) => i).sort((a, b) => a - b).join(','));
    if (!m) out.unmatched++;
    else {
      const a = angleDeg(n, m);
      if (a > 90) out.flipped++;
      out.moved_hist[moved === 0 ? '0' : moved < 1e-12 * diag ? '<1e-12' : moved < 1e-9 * diag ? '<1e-9' : moved < 1e-6 * diag ? '<1e-6' : '>=1e-6'] += 1;
      if (moved === 0) {
        out.exact++;
        out.normal_deg_exact = Math.max(out.normal_deg_exact, a);
        if (a > 1e-6) out.exact_over_1e6deg++;
      } else {
        out.moved_facets++;
        out.moved = Math.max(out.moved, moved / diag);
        movedAngles.push(a);
      }
    }
    vm.delete(); em.delete(); ad.delete();
  }
  ex.delete();
  movedAngles.sort((a, b) => a - b);
  out.normal_deg_moved = movedAngles.length ? movedAngles[movedAngles.length - 1] : 0;
  out.normal_deg_moved_p99 = movedAngles.length ? movedAngles[Math.floor(0.99 * (movedAngles.length - 1))] : 0;
  return out;
}

/** The largest vertex and edge tolerance anywhere in `shape` (mm): what the repairs let the solid's gaps grow to. */
export function maxTolerance(oc, shape) {
  const E = oc.TopAbs_ShapeEnum;
  return { vertex: oc.BRep_Tool.MaxTolerance(shape, E.TopAbs_VERTEX), edge: oc.BRep_Tool.MaxTolerance(shape, E.TopAbs_EDGE) };
}
