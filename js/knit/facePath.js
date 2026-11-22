/**
 * knit/facePath.js — the export's first try for a part with freeform groups: the face path (js/subd/faceExport.js:
 * one trimmed face per painted group, the analytic groups on their own surfaces, the rest fitted B-splines; the STEP
 * written there), judged by the knitter's bar for its own builds (build.js grade: closed, valid, no face off the mesh,
 * within 1% of the mesh's volume), the kernel only reading the file back to judge and tessellate it. The volume: the
 * mesh's plus what lies between its chords and the faces' surfaces (faceSolid's bulgeVolume), since true surfaces
 * are off a coarse mesh by its sagitta (00000606_012's own CAD is 1.4% off its mesh's volume); what is left is what a
 * volume check is for, a face inside out or a piece missing. And its surfaces near the mesh all over, not only at
 * score's few points: every face tessellated (the studio's preview, 1e-3 x diagonal), every point of that within
 * OFF_MESH (the knitter's 1% of the diagonal) of the mesh and 99% of them within half that (this branch's suites'
 * bar); and the mesh near the solid all over the same way (each triangle's centroid, by its area: faces that stay on
 * the mesh can still leave part of it uncovered, a trim cutting into its face). When it is not that good, the caller builds with the knitter as before (knitSolid), so no part exports worse
 * than it did.
 *
 * hasFreeform: whether the payload has a group the knitter would fit a B-spline to (no surface, more than
 * FREEFORM_MIN_FACETS triangles). tryFacePath: { step, report, shape } (report as knitSolid's, tier 'analytic',
 * info.path 'faces') or { rejected: why, info } for the knitter's report.
 */
import { faceExport } from '../subd/faceExport.js';
import { readStep } from './occt.js';
import { KnitMesh } from './mesh.js';
import { score, subShapes, OFF_MESH } from './build.js';
import { previewMesh } from './preview.js';
import { FREEFORM_MIN_FACETS } from './index.js';

export function hasFreeform(labels, surfaces) {
  const count = new Map();
  for (const g of labels) if (g >= 0 && !surfaces.has(g)) count.set(g, (count.get(g) ?? 0) + 1);
  for (const n of count.values()) if (n > FREEFORM_MIN_FACETS) return true;
  return false;
}

export function tryFacePath(oc, payload, { part = 'Part', budgetS = Infinity } = {}) {
  // the kernel's verdict names faces (invalid, or not tessellated): fitted ones are cut and the solid built again,
  // twice at most and within budgetS (the knitter's time after it is what is left)
  const t0 = performance.now(), split = [], seen = new Set();
  let last = null;
  for (let round = 0; round < 3; round++) {
    const r = attempt(oc, payload, part, split);
    (r.info ?? r.report.info).kernel_rounds = round;
    if (r.step) return r;
    last = r;
    const add = (r.bad ?? []).filter((k) => k && !seen.has(`${k.b}:${k.tris.join(',')}`));
    if (!add.length || (performance.now() - t0) / 1000 > budgetS) break;
    for (const k of add) { seen.add(`${k.b}:${k.tris.join(',')}`); split.push(k); }
  }
  return { rejected: last.rejected, info: last.info };
}

function attempt(oc, { positions, labels, surfaces, oriented = false }, part, split) {
  const t0 = performance.now();
  let fx;
  try { fx = faceExport({ positions, labels, surfaces }, { part, split }); } catch (err) { return { rejected: `threw: ${String(err?.message ?? err).slice(0, 160)}`, info: {} }; }
  const info = { ...fx.report.info, face_ms: Math.round(performance.now() - t0) };
  if (fx.error) return { rejected: fx.error, info };
  const shape = readStep(oc, fx.step);
  if (!shape) return { rejected: 'the kernel could not read it', info };
  const mesh = new KnitMesh(positions, labels, { oriented });
  const sc = score(oc, shape, mesh), vm = mesh.volume();
  // the volume against the mesh's plus the bulge (score's vol_err is against the mesh's alone)
  const volErr = sc.volume != null && vm ? Math.abs(Math.abs(sc.volume) - (Math.abs(vm) + info.bulge_volume)) / Math.abs(vm) : null;
  info.check_ms = Math.round(performance.now() - t0) - info.face_ms;
  info.vol_err_mesh = sc.vol_err;
  info.vol_err_surfaces = volErr;
  const faces = subShapes(oc, shape, oc.TopAbs_ShapeEnum.TopAbs_FACE);
  // the faces the kernel calls invalid, as faceExport's keys (the file's order is the read shape's)
  const keyed = (idx) => (faces.length === fx.faceKeys.length ? idx.map((i) => fx.faceKeys[i]).filter(Boolean) : []);
  if (!(sc.closed && sc.valid && !sc.off_mesh && volErr !== null && volErr < 0.01)) {
    const why = !sc.closed ? 'not closed' : !sc.valid ? 'not valid' : sc.off_mesh ? `${sc.off_mesh} faces off the mesh` : `volume ${((volErr ?? 1) * 100).toFixed(2)}% off`;
    let bad = [];
    if (sc.closed && !sc.valid) {
      const ana = new oc.BRepCheck_Analyzer(shape, true, false), W = oc.TopAbs_ShapeEnum, idx = [];
      faces.forEach((fc, i) => {
        if (!ana.IsValid_1(fc) || subShapes(oc, fc, W.TopAbs_WIRE).some((w) => !ana.IsValid_1(w)) || subShapes(oc, fc, W.TopAbs_EDGE).some((e) => !ana.IsValid_1(e))) idx.push(i);
      });
      ana.delete();
      bad = keyed(idx);
    }
    return { rejected: why, info, bad };
  }
  let preview;
  try { preview = previewMesh(oc, shape, 1e-3 * mesh.diag); } catch { return { rejected: 'the kernel could not tessellate it', info }; }
  // (the preview leaves out a face it cannot tessellate, and counts only those it drew: a face left with no
  // triangulation after it is one)
  if (preview.faces < faces.length) {
    const idx = [];
    faces.forEach((fc, i) => { const loc = new oc.TopLoc_Location_1(); if (oc.BRep_Tool.Triangulation(oc.TopoDS.Face_1(fc), loc, 0).IsNull()) idx.push(i); loc.delete(); });
    return { rejected: `${faces.length - preview.faces} faces the kernel could not tessellate`, info, bad: keyed(idx) };
  }
  const { far, beyond } = strays(preview.positions, positions, mesh.diag);
  info.preview_far = far; info.preview_beyond = beyond;
  if (beyond > 0 || far > 0.01 * (preview.positions.length / 3)) return { rejected: `off the mesh (${beyond} points past 1%, ${far} past 0.5% of its size)`, info };
  const nT = positions.length / 9, cen = new Float64Array(nT * 3), area = new Float64Array(nT);
  for (let t = 0; t < nT; t++) {
    const o = t * 9, e1 = [0, 1, 2].map((k) => positions[o + 3 + k] - positions[o + k]), e2 = [0, 1, 2].map((k) => positions[o + 6 + k] - positions[o + k]);
    area[t] = Math.hypot(e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]) / 2;
    for (let k = 0; k < 3; k++) cen[t * 3 + k] = (positions[o + k] + positions[o + 3 + k] + positions[o + 6 + k]) / 3;
  }
  const back = strays(cen, preview.positions, mesh.diag, area);
  info.mesh_far = back.far / (back.total || 1); info.mesh_beyond = back.beyond / (back.total || 1);
  if (back.beyond > 0 || back.far > 0.01 * back.total) return { rejected: `the mesh off the solid (${((100 * back.beyond) / back.total).toFixed(2)}% of its area past 1%, ${((100 * back.far) / back.total).toFixed(2)}% past 0.5% of its size)`, info };
  const report = { ...fx.report, closed: true, valid: true, vol_err: volErr, free_edges: sc.free_edges ?? 0,
    secs: Math.round((performance.now() - t0) / 100) / 10, info };
  return { step: fx.step, report, shape, preview };
}

/**
 * How many of the points `pts` (flat xyz) lie farther than OFF_MESH / 2 x diag from the triangle soup P (far), and
 * farther than OFF_MESH x diag (beyond): exact point-triangle distances over a grid of the triangles. w: a weight per
 * point (far, beyond and total then sums of them).
 */
function strays(pts, P, diag, w = null) {
  const h = (OFF_MESH / 2) * diag, cells = new Map(), key = (x, y, z) => `${x},${y},${z}`;
  const nT = P.length / 9;
  for (let t = 0; t < nT; t++) {
    const o = t * 9, lo = [0, 1, 2].map((k) => Math.floor(Math.min(P[o + k], P[o + 3 + k], P[o + 6 + k]) / h)), hi = [0, 1, 2].map((k) => Math.floor(Math.max(P[o + k], P[o + 3 + k], P[o + 6 + k]) / h));
    for (let x = lo[0]; x <= hi[0]; x++) for (let y = lo[1]; y <= hi[1]; y++) for (let z = lo[2]; z <= hi[2]; z++) (cells.get(key(x, y, z)) ?? cells.set(key(x, y, z), []).get(key(x, y, z))).push(t);
  }
  const dist2 = (p, t) => {
    const o = t * 9, a = [P[o], P[o + 1], P[o + 2]], b = [P[o + 3], P[o + 4], P[o + 5]], c = [P[o + 6], P[o + 7], P[o + 8]];
    const sb = (u, v) => [u[0] - v[0], u[1] - v[1], u[2] - v[2]], dt = (u, v) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    const ab = sb(b, a), ac = sb(c, a), ap = sb(p, a), d1 = dt(ab, ap), d2 = dt(ac, ap);
    let q;
    if (d1 <= 0 && d2 <= 0) q = a;
    else {
      const bp = sb(p, b), d3 = dt(ab, bp), d4 = dt(ac, bp), cp = sb(p, c), d5 = dt(ab, cp), d6 = dt(ac, cp);
      const vc = d1 * d4 - d3 * d2, vb = d5 * d2 - d1 * d6, va = d3 * d6 - d5 * d4;
      if (d3 >= 0 && d4 <= d3) q = b;
      else if (d6 >= 0 && d5 <= d6) q = c;
      else if (vc <= 0 && d1 >= 0 && d3 <= 0) { const v = d1 / (d1 - d3); q = [a[0] + v * ab[0], a[1] + v * ab[1], a[2] + v * ab[2]]; }
      else if (vb <= 0 && d2 >= 0 && d6 <= 0) { const w = d2 / (d2 - d6); q = [a[0] + w * ac[0], a[1] + w * ac[1], a[2] + w * ac[2]]; }
      else if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) { const w = (d4 - d3) / ((d4 - d3) + (d5 - d6)); q = [b[0] + w * (c[0] - b[0]), b[1] + w * (c[1] - b[1]), b[2] + w * (c[2] - b[2])]; }
      else { const den = 1 / (va + vb + vc), v = vb * den, w = vc * den; q = [a[0] + ab[0] * v + ac[0] * w, a[1] + ab[1] * v + ac[1] * w, a[2] + ab[2] * v + ac[2] * w]; }
    }
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2;
  };
  // nearest within the cells r rings round p (2 rings reach 2h = OFF_MESH x diag)
  const near = (p, rings) => {
    const c = p.map((x) => Math.floor(x / h));
    let best = Infinity;
    for (let x = c[0] - rings; x <= c[0] + rings; x++) for (let y = c[1] - rings; y <= c[1] + rings; y++) for (let z = c[2] - rings; z <= c[2] + rings; z++) {
      for (const t of cells.get(key(x, y, z)) ?? []) best = Math.min(best, dist2(p, t));
    }
    return Math.sqrt(best);
  };
  let far = 0, beyond = 0, total = 0;
  for (let i = 0; i < pts.length; i += 3) {
    const p = [pts[i], pts[i + 1], pts[i + 2]], wi = w ? w[i / 3] : 1;
    total += wi;
    if (near(p, 1) <= h) continue;
    far += wi;
    if (near(p, 2) > 2 * h) beyond += wi;
  }
  return { far, beyond, total };
}
