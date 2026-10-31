/**
 * knit/axes.js — the part's shared axes and directions, made exact before the knit.
 *
 * Each patch's surface is fitted on its own, so two halves of one cylinder, a shaft's steps or a bore and
 * its counterbore come out with axes a hundredth of a degree apart, and the solid has a kink at every
 * edge between them. CAD has one axis there. So, before the build:
 *
 *   directions   cylinder, cone and torus axes and plane normals closer than ANGLE_DEG are one
 *                direction (greedy, heaviest first, against the cluster's running mean: no chaining),
 *                and directions within ANGLE_DEG of perpendicular to a heavier one are made exactly
 *                perpendicular to it (to two: their cross product)
 *   axis lines   inside a direction, axes whose lines pass within OFFSET x the part diagonal are one line;
 *                sphere centres that close to a line move onto it
 *   refit        every member refitted about its fixed line: cylinder radius; cone apex and angle (a line
 *                in the (z, rho) profile); torus and sphere by Gauss-Newton on the profile circle; planes
 *                keep their offset. Cylinders on one line with radii within RADIUS, planes of one direction
 *                with offsets within OFFSET, become the same surface exactly
 *   check        a snapped surface is kept only if its RMS distance to its own vertices stays within
 *                ACCEPT x the original + FLOOR x diagonal; otherwise it keeps its own fit and the shared
 *                values are recomputed without it
 *
 * Bsplines and surfaceless groups are left alone.
 */

import { dot, sub, add, scale, cross, norm, unit, distNormal } from './geometry.js';

export const AXES = { ANGLE_DEG: 1, OFFSET: 1e-3, RADIUS: 1e-3, ACCEPT: 1.5, FLOOR: 1e-6, ROUNDS: 3, INLIER: 5, TANGENT_DEG: 3,
  CHORD_Q: 0.9,
  PRECISION_MIN: 1e-9, SIGMA_K: 10, FLOAT: 1e-7, AXIS_COND: 100, LOW_REDUNDANCY: 3, EXACT: 1e-6,
  MIRROR_ROUNDS: 3, VALUE_REL: 1e-3, VALUE_ANGLE_DEG: 0.5,
  INTENT_DEG: 0.25, INTENT_REL: 2e-4, INTENT_FLOOR: 2e-5, THROUGH: true, CONE_REL: 0.01 };
const AXIAL = new Set(['cylinder', 'cone', 'torus']);
const DOF = { plane: 3, cylinder: 5, cone: 6, sphere: 4, torus: 7 };
const SNAPPABLE = new Set([...AXIAL, 'plane', 'sphere']);
const originOf = (kind, p) => (kind === 'cylinder' ? p.axisPoint : kind === 'cone' ? p.apex : p.center);

/**
 * Each surfaced label's INLIER vertices: within max(INLIER x the median distance, FLOOR x diagonal) of
 * its own surface. Auto-paint's groups often carry a few triangles of a neighbour; their own fits are
 * robust to those, and the refits and the check must be too.
 */
function inliersByLabel(mesh, surfaces, cfg) {
  const all = pointsByLabel(mesh), out = new Map();
  for (const [g, P] of all) {
    const s = surfaces.get(g);
    if (!s || !s[1] || !SNAPPABLE.has(s[0])) continue;
    const r = P.map((x) => Math.abs(distNormal(s[0], s[1], x)[0]));
    const med = [...r].sort((a, b) => a - b)[r.length >> 1] ?? 0;
    const band = Math.max(cfg.INLIER * med, cfg.FLOOR * mesh.diag);
    const kept = P.filter((_, i) => r[i] <= band);
    out.set(g, kept.length >= Math.min(P.length, 6) ? kept : P);
  }
  return out;
}

/** Unique vertex positions of every label's triangles. */
function pointsByLabel(mesh) {
  const seen = new Map();
  for (let t = 0; t < mesh.T.length; t++) {
    const g = mesh.F[t];
    if (g < 0) continue;
    let s = seen.get(g);
    if (!s) seen.set(g, (s = new Set()));
    for (const v of mesh.T[t]) s.add(v);
  }
  const out = new Map();
  for (const [g, s] of seen) out.set(g, [...s].map((v) => mesh.V[v]));
  return out;
}

function maxDev(kind, p, pts) {
  let m = 0;
  for (const x of pts) m = Math.max(m, Math.abs(distNormal(kind, p, x)[0]));
  return m;
}

/** Per curved label: the CHORD_Q quantile of its triangle centroids' distances to its own surface. */
function chordByLabel(mesh, surfaces, cfg) {
  const d = new Map();
  for (let t = 0; t < mesh.T.length; t++) {
    const g = mesh.F[t], s = surfaces.get(g);
    if (!s || !s[1] || !SNAPPABLE.has(s[0]) || s[0] === 'plane') continue;
    const [a, b, c] = mesh.T[t].map((v) => mesh.V[v]);
    const cen = [(a[0] + b[0] + c[0]) / 3, (a[1] + b[1] + c[1]) / 3, (a[2] + b[2] + c[2]) / 3];
    (d.get(g) ?? d.set(g, []).get(g)).push(Math.abs(distNormal(s[0], s[1], cen)[0]));
  }
  const out = new Map();
  for (const [g, l] of d) { l.sort((x, y) => x - y); out.set(g, l[Math.min(l.length - 1, Math.floor(cfg.CHORD_Q * l.length))]); }
  return out;
}

function rms(kind, p, pts) {
  let s = 0;
  for (const x of pts) { const d = distNormal(kind, p, x)[0]; s += d * d; }
  return Math.sqrt(s / Math.max(pts.length, 1));
}

/** (z along d from o, rho from the line) of each point. */
function profile(pts, o, d) {
  return pts.map((x) => {
    const v = sub(x, o), z = dot(v, d);
    return [z, norm(sub(v, scale(d, z)))];
  });
}

/** Gauss-Newton on sum (sqrt((rho - R)^2 + (z - zc)^2) - r)^2; R fixed at 0 for a sphere. */
function fitProfileCircle(zr, R0, zc0, r0, sphere) {
  let R = sphere ? 0 : R0, zc = zc0, r = r0;
  for (let it = 0; it < 20; it++) {
    const n = sphere ? 2 : 3;
    const JtJ = Array.from({ length: n }, () => new Array(n).fill(0)), Jtr = new Array(n).fill(0);
    for (const [z, rho] of zr) {
      const a = rho - R, b = z - zc, l = Math.hypot(a, b) || 1e-300, res = l - r;
      const J = sphere ? [-b / l, -1] : [-a / l, -b / l, -1];
      for (let i = 0; i < n; i++) { Jtr[i] += J[i] * res; for (let j = 0; j < n; j++) JtJ[i][j] += J[i] * J[j]; }
    }
    for (let i = 0; i < n; i++) JtJ[i][i] *= 1 + 1e-9;
    const step = solve(JtJ, Jtr.map((v) => -v));
    if (!step) break;
    if (sphere) { zc += step[0]; r += step[1]; } else { R += step[0]; zc += step[1]; r += step[2]; }
    if (Math.max(...step.map(Math.abs)) < 1e-14 * (Math.abs(R) + Math.abs(r) + 1)) break;
  }
  return { R, zc, r };
}

function solve(A, b) {
  const n = b.length, M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    [M[c], M[piv]] = [M[piv], M[c]];
    if (!(Math.abs(M[c][c]) > 1e-300)) return null;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = M[r][c] / M[c][c];
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  return M.map((row, i) => row[n] / row[i]);
}

const median = (a) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[s.length >> 1] : 0; };

/**
 * The profile points within TRIM x the median residual of a first fit (FLOOR-scaled at least): the
 * refits must be as robust as the fits they replace. A 3-triangle sliver across two faces kept its exact
 * plane under a robust fit; refitted through all its vertices it moved 3e-5 of the diagonal off the truth
 * while fitting those vertices better.
 */
function trimmed(zr, residual, floor) {
  const r = zr.map(residual), cut = Math.max(TRIM * median(r.map(Math.abs)), floor);
  const kept = zr.filter((_, i) => Math.abs(r[i]) <= cut);
  return kept.length >= Math.min(zr.length, 4) ? kept : zr;
}
const TRIM = 3;

function coneLine(zr) {
  let sz = 0, sr = 0, szz = 0, szr = 0;
  for (const [z, rho] of zr) { sz += z; sr += rho; szz += z * z; szr += z * rho; }
  const n = zr.length, den = n * szz - sz * sz;
  if (!(Math.abs(den) > 1e-300)) return null;
  const m = (n * szr - sz * sr) / den;
  return { m, b: (sr - m * sz) / n };
}

/** A surface of `kind` refitted to `pts` about the fixed line (o, d); null if it cannot be. floor: a length. */
function refitAbout(kind, p, pts, o, d, floor = 0) {
  let zr = profile(pts, o, d);
  if (kind === 'cylinder') return { ...p, axisPoint: o, axis: d, radius: median(zr.map(([, rho]) => rho)) };
  if (kind === 'cone') {
    // rho = m z + b over the patch (apex at z = -b / m), refitted on the trimmed profile
    let L = coneLine(zr);
    if (L) { zr = trimmed(zr, ([z, rho]) => rho - L.m * z - L.b, floor); L = coneLine(zr); }
    if (!L || !(Math.abs(L.m) > 1e-9)) return null;
    const { m, b } = L;
    return { ...p, apex: add(o, scale(d, -b / m)), axis: d, halfAngle: Math.atan(Math.abs(m)) };
  }
  if (kind === 'torus') {
    let R0, zc0, r0;
    if (p.majorRadius !== undefined) { R0 = p.majorRadius; zc0 = dot(sub(p.center, o), d); r0 = p.minorRadius; }
    else {
      // rho^2 + z^2 = 2 R rho + 2 zc z + c: an algebraic start for a profile circle
      const A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], bb = [0, 0, 0];
      for (const [z, rho] of zr) {
        const row = [2 * rho, 2 * z, 1], y = rho * rho + z * z;
        for (let i = 0; i < 3; i++) { bb[i] += row[i] * y; for (let j = 0; j < 3; j++) A[i][j] += row[i] * row[j]; }
      }
      const x = solve(A, bb);
      if (!x) return null;
      [R0, zc0] = x;
      r0 = Math.sqrt(Math.max(x[2] + R0 * R0 + zc0 * zc0, 0));
    }
    let { R, zc, r } = fitProfileCircle(zr, R0, zc0, r0, false);
    zr = trimmed(zr, ([z, rho]) => Math.hypot(rho - R, z - zc) - r, floor);
    ({ R, zc, r } = fitProfileCircle(zr, R, zc, r, false));
    if (!(R > 0 && r > 0)) return null;
    return { ...p, center: add(o, scale(d, zc)), axis: d, majorRadius: R, minorRadius: r };
  }
  if (kind === 'sphere') {
    const zc0 = dot(sub(p.center, o), d);
    let { zc, r } = fitProfileCircle(zr, 0, zc0, p.radius, true);
    zr = trimmed(zr, ([z, rho]) => Math.hypot(rho, z - zc) - r, floor);
    ({ zc, r } = fitProfileCircle(zr, 0, zc, r, true));
    if (!(r > 0)) return null;
    return { ...p, center: add(o, scale(d, zc)), radius: r };
  }
  return null;
}

/** Greedy clustering, heaviest first: an item joins the first cluster `near` accepts, else starts one. */
function greedy(items, near, merge) {
  const clusters = [];
  for (const it of [...items].sort((a, b) => b.w - a.w)) {
    const c = clusters.find((k) => near(k, it));
    if (c) { c.items.push(it); merge(c, it); } else { const k = { items: [it] }; merge(k, it, true); clusters.push(k); }
  }
  return clusters;
}

/**
 * Snap `surfaces` (Map label -> [kind, params]) of a KnitMesh to shared directions and axes. Returns
 * {surfaces (a new Map; unchanged entries are the same objects), info}.
 */
export function shareAxes(mesh, surfaces, opts = {}) {
  const cfg = { ...AXES, ...opts };
  const offTol = cfg.OFFSET * mesh.diag;
  const pts = inliersByLabel(mesh, surfaces, cfg);
  // Precision weights: a member counts by its vertices over its fit's squared RMS, so exact fits set
  // the shared values and loose ones (a 14-vertex cone on a cylinder strip) follow them.
  // Precision-aware tolerances: a plane's normal is known to about its RMS over its extent, so exact
  // planes join a direction (or share an offset) only when they truly agree - a draft angle or a small
  // step is real - while loose fits on noisy meshes still pool up to ANGLE_DEG / OFFSET. An axis is
  // worse conditioned than its RMS says (a strip), so its uncertainty is AXIS_COND times a plane's: with
  // none, a 4-triangle plane exact on its face joined an axis 0.42 degrees off and tilted with it.
  // Noise is per vertex: a vertex's BEST residual on any surface whose inliers it is among (between an
  // exact plane and a loose cylinder the vertex is exact; the cylinder's fit is not - with the worst
  // residual, single-triangle planes on exact meshes moved 0.17 off their true faces), and a group's
  // floor the RMS of its own vertices' noise. A fit on few vertices is exact by construction,
  // not precise: on the user's chain a 4-vertex chamfer plane sat exactly on two vertices of a cone's
  // ring that are 0.015 off the CAD, at 44.9 degrees against its 45.0 partner, and outweighed the
  // 34-vertex cone. Exact planes on exact vertices keep a floor near the float limit. The precision
  // also counts redundancy: RMS x sqrt(n / (n - DOF)), none at all with n <= DOF.
  const rms0 = new Map(), weight = new Map(), sigma = new Map(), floor = new Map(), noiseOf = new Map();
  const vnoise = new Map();
  for (const [g, P] of pts) {
    const s = surfaces.get(g);
    rms0.set(g, rms(s[0], s[1], P));
    for (const x of P) vnoise.set(x, Math.min(vnoise.get(x) ?? Infinity, Math.abs(distNormal(s[0], s[1], x)[0])));
  }
  for (const [g, P] of pts) floor.set(g, Math.sqrt(P.reduce((a, x) => a + vnoise.get(x) ** 2, 0) / P.length));
  for (const [g, P] of pts) {
    const s = surfaces.get(g), r = rms0.get(g);
    const red = Math.max(P.length - (DOF[s[0]] ?? 3), 0.25);   // never 0: a cluster started by it would have no direction
    // a fit with redundancy (n - DOF >= LOW_REDUNDANCY) estimates its own noise; one with less borrows
    // its vertices' noise (the 4-vertex chamfer plane): exact planes beside loose fits stay exact
    const sh = red > 0 ? r * Math.sqrt(P.length / red) : Infinity;
    noiseOf.set(g, red >= cfg.LOW_REDUNDANCY ? sh : Math.max(Number.isFinite(sh) ? sh : 0, floor.get(g)));
    const se = Math.max(noiseOf.get(g), cfg.PRECISION_MIN * mesh.diag);
    weight.set(g, red * (cfg.FLOOR * mesh.diag / se) ** 2);
    const c = scale(P.reduce((a, x) => add(a, x), [0, 0, 0]), 1 / P.length);
    const ext = Math.sqrt(P.reduce((a, x) => a + dot(sub(x, c), sub(x, c)), 0) / P.length);
    const sg = Math.max(r, cfg.FLOAT * mesh.diag) / Math.max(ext, 1e-12 * mesh.diag);
    sigma.set(g, s[0] === 'plane' ? sg : s[0] === 'sphere' ? Infinity : cfg.AXIS_COND * sg);
  }
  // The mesh's own resolution for each surface: how far its triangles' centroids sit from the surface it was
  // fitted with (the chordal sag; 0 for a plane). A shared or tangent surface whose vertices all stay within
  // that is indistinguishable from the free fit at this tessellation, so it is accepted even when the free
  // fit chased the tessellation's error more closely (the user's chain: a chamfer cone's outer ring of
  // vertices sits 0.015 off its circle, and the free cone tilted 0.08 degrees to follow it).
  const chord = chordByLabel(mesh, surfaces, cfg);
  // Intent relations (parallel or mirror within INTENT_DEG, equal sizes within INTENT_REL) override data
  // that is exact but slightly inconsistent - the user's chain: a 4-vertex chamfer plane exact at 44.90
  // degrees, its partners at 45.00. Their members may stray up to INTENT_FLOOR x diagonal.
  const intent = new Set(), info0 = {};
  // a fit with redundancy (n - DOF >= LOW_REDUNDANCY) is firm: intent never moves it past its precision
  const firm = (g) => pts.get(g).length - (DOF[surfaces.get(g)[0]] ?? 3) >= cfg.LOW_REDUNDANCY;
  // Intent only where the mesh cannot resolve it: the median fit of its firm curved surfaces above the
  // exactness bar. On an exact CAD tessellation a 0.1-degree difference is real (draft faces: intent
  // moved 29 exact fill-set planes off their true faces); on the user's chain the curved surfaces fit to
  // about 1e-5 of the diagonal and the 0.1 degree between mirror chamfers is the tessellation's.
  const curvedFirm = [...pts.keys()].filter((g) => surfaces.get(g)[0] !== 'plane' && firm(g)).map((g) => rms0.get(g));
  const noisy = curvedFirm.length > 0 && median(curvedFirm) > cfg.EXACT * mesh.diag;
  info0.mesh_inexact = noisy ? 1 : 0;
  info0.mesh_noise = curvedFirm.length ? median(curvedFirm) / mesh.diag : 0;
  const intentOn = noisy && cfg.INTENT_DEG > 0;
  const accepts = (g, kind, p) => {
    const P = pts.get(g);
    const fl = (intent.has(g) ? cfg.INTENT_FLOOR : cfg.FLOOR) * mesh.diag;
    if (rms(kind, p, P) <= cfg.ACCEPT * Math.max(rms0.get(g), noiseOf.get(g)) + fl) return true;
    // the chordal allowance only where the vertices are visibly off the surface (a fit above the exactness
    // bar): on an exact CAD tessellation the vertices are on the true surface, and moving within the sag
    // put 22 exact fill-set surfaces up to 5e-4 of the diagonal off their true faces
    if (!(rms0.get(g) > cfg.EXACT * mesh.diag)) return false;
    return maxDev(kind, p, P) <= (chord.get(g) ?? 0) + cfg.FLOOR * mesh.diag;
  };
  const angTol = (s1, s2) => Math.min((cfg.ANGLE_DEG * Math.PI) / 180, cfg.SIGMA_K * (s1 + s2));
  const angleOf = (a, b) => Math.atan2(norm(cross(a, b)), Math.abs(dot(a, b)));
  const lenTol = (r1, r2, cap) => Math.min(cap, cfg.SIGMA_K * (r1 + r2) + 10 * cfg.FLOOR * mesh.diag);
  const rejected = new Set();
  const locks = { dir: new Set(), line: new Set(), plane: new Set(), lines: [], planeSet: new Map(), perpTo: new Map() };
  let out = new Map(surfaces), info;
  for (let round = 0; round < cfg.ROUNDS; round++) {
    out = new Map(surfaces);
    info = { axes_surfaces: 0, axes_directions: 0, axes_lines_shared: 0, axes_snapped: 0, axes_rejected: 0,
      axes_same_radius: 0, axes_same_plane: 0, axes_perpendicular: 0 };
    locks.dir.clear(); locks.line.clear(); locks.plane.clear(); locks.lines.length = 0; locks.planeSet.clear(); locks.perpTo.clear();
    intent.clear();
    const items = [];
    for (const [g, s] of surfaces) {
      if (!s || rejected.has(g) || !pts.has(g)) continue;
      const [kind, p] = s;
      const sig = sigma.get(g), r0 = rms0.get(g);
      if (AXIAL.has(kind)) {
        // the axis point nearest the patch: a stored origin far along the axis (a near-cylindrical cone's
        // apex thousands of units out) turns a hundredth of a degree into a large offset
        const d = unit(p.axis), o = originOf(kind, p), P = pts.get(g);
        const c = scale(P.reduce((a, x) => add(a, x), [0, 0, 0]), 1 / P.length);
        items.push({ g, kind, p, d, o: add(o, scale(d, dot(sub(c, o), d))), w: weight.get(g), sig, r0, firm: firm(g) });
      }
      else if (kind === 'plane') items.push({ g, kind, p, d: unit(p.normal), w: weight.get(g), sig, r0, firm: firm(g) });
    }
    info.axes_surfaces = items.length;
    // 1. directions (antipodal), then exact perpendiculars to heavier directions
    const dirs = greedy(items, (k, it) => angleOf(k.d, it.d) <= angTol(k.sig, it.sig), (k, it, first) => {
      if (first) { k.sum = scale(it.d, it.w); k.w = it.w; k.sig = it.sig; }
      else { k.sum = add(k.sum, scale(it.d, Math.sign(dot(k.sum, it.d)) * it.w)); k.w += it.w; k.sig = Math.min(k.sig, it.sig); }
      k.d = unit(k.sum);
    });
    const fixed = [];
    for (const k of dirs) {
      k.changed = k.items.length > 1;          // a lone direction moves only to become perpendicular
      const perp = fixed.filter((f) => Math.abs(Math.PI / 2 - angleOf(f.d, k.d)) <= angTol(f.sig, k.sig));
      if (perp.length === 1) k.d = unit(sub(k.d, scale(perp[0].d, dot(perp[0].d, k.d))));
      else if (perp.length >= 2) {
        const c = cross(perp[0].d, perp[1].d);
        if (norm(c) > 0.5) k.d = scale(unit(c), Math.sign(dot(c, k.d)) || 1);
        else k.d = unit(sub(k.d, scale(perp[0].d, dot(perp[0].d, k.d))));
      }
      if (perp.length) { info.axes_perpendicular++; k.changed = true; k.perpTo = perp.slice(0, 2).map((f) => f.d); }
      fixed.push(k);
    }
    // 1b. Mirror symmetry: two directions that are mirror images across one of the part's main directions
    // (one several faces share) within their precision become exact mirror images, at their weighted
    // mean - the user's chain had chamfer planes at 44.9 and 45.0 degrees, mirror images across the lug's
    // mid-plane; they now share whatever angle the data gives, not a round number. Mirroring across a main
    // direction keeps a direction square to anything along or across it, so the perpendiculars stay exact;
    // a few rounds settle parts with several mirrors (left/right and top/bottom).
    // 1a. Parallel intent: directions within INTENT_DEG that precision kept apart (an exact few-vertex plane)
    // are one direction; their members are judged with the intent floor
    const intentRad = intentOn ? (cfg.INTENT_DEG * Math.PI) / 180 : 0;
    for (let i = 0; i < dirs.length; i++) {
      const a = dirs[i];
      if (!a) continue;
      for (let j = i + 1; j < dirs.length; j++) {
        const b = dirs[j];
        if (!b || angleOf(a.d, b.d) > intentRad) continue;
        // intent moves only what cannot defend itself: two firm directions that disagree stay apart (on
        // exact CAD meshes they really differ - merging moved 77 exact fill-set surfaces off their faces)
        const fa = a.items.some((it) => it.firm), fb = b.items.some((it) => it.firm);
        if (fa && fb && angleOf(a.d, b.d) > angTol(a.sig, b.sig)) continue;
        const sg = Math.sign(dot(a.d, b.d)) || 1;
        const moved = fa ? b.items : fb ? a.items : [...a.items, ...b.items];
        a.d = fa ? a.d : fb ? scale(b.d, sg) : unit(add(scale(a.d, a.w), scale(b.d, sg * b.w)));
        a.w += b.w; a.items.push(...b.items); a.changed = a.merged = true;
        a.sig = Math.min(a.sig, b.sig);
        if (b.perpTo && !a.perpTo) a.perpTo = b.perpTo;
        for (const it of moved) intent.add(it.g);
        dirs[j] = null;
        info.axes_parallel_intent = (info.axes_parallel_intent ?? 0) + 1;
      }
    }
    for (let i = dirs.length - 1; i >= 0; i--) if (!dirs[i]) dirs.splice(i, 1);
    // keep exact perpendiculars after the merges - for the merged directions only: recomputing every one
    // moved 00003840's directions enough to reject a torus and lose its solid (planar -> faceted)
    for (const k of dirs) if (k.merged && k.perpTo) {
      const [f1, f2] = k.perpTo;
      k.d = f2 ? scale(unit(cross(f1, f2)), Math.sign(dot(cross(f1, f2), k.d)) || 1) : unit(sub(k.d, scale(f1, dot(f1, k.d))));
    }
    // mirror planes come from the part's frame only: its heaviest shared direction and the shared directions
    // exactly square to it (an oblique chamfer direction with four faces once mirrored the frame's x onto z)
    const shared = dirs.filter((k) => k.items.length > 1).sort((u, v) => v.w - u.w);
    const mains = shared.length ? [shared[0], ...shared.slice(1).filter((k) => Math.abs(dot(k.d, shared[0].d)) <= 1e-12)].slice(0, 3) : [];
    const reflect = (v, m) => sub(v, scale(m, 2 * dot(v, m)));
    // (only on inexact meshes: on exact CAD tessellations mirror pairs are already exact by the data, and
    // averaging them cost exact perpendiculars to oblique axes - fill perpendicular 98.9 -> 98.4%)
    for (let round = 0; round < (intentOn ? cfg.MIRROR_ROUNDS : 0); round++) {
      for (const m of mains) {
        const used = new Set();
        for (const a of dirs) {
          if (a === m || used.has(a)) continue;
          const am = Math.abs(dot(a.d, m.d));
          if (am < 1e-6 || am > 1 - 1e-6) continue;       // along or square to the mirror: its own image
          let best = null, ba = Infinity;
          for (const b of dirs) {
            if (b === a || b === m || used.has(b)) continue;
            const t = angleOf(a.d, reflect(b.d, m.d));
            if (t <= Math.max(angTol(a.sig, b.sig), intentRad) && t < ba) { ba = t; best = b; }
          }
          if (!best || ba === 0) continue;
          const fa = a.items.some((it) => it.firm), fb = best.items.some((it) => it.firm);
          if (fa && fb && ba > angTol(a.sig, best.sig)) { used.add(a); continue; }
          const mb = reflect(best.d, m.d), sg = Math.sign(dot(a.d, mb)) || 1;
          const avg = fa ? a.d : fb ? scale(mb, sg) : unit(add(scale(a.d, a.w), scale(mb, sg * best.w)));
          a.d = avg; best.d = scale(reflect(avg, m.d), sg);
          a.changed = best.changed = true;
          for (const it of fa ? best.items : fb ? a.items : [...a.items, ...best.items]) intent.add(it.g);
          used.add(a).add(best);
          if (round === 0) info.axes_mirror_pairs = (info.axes_mirror_pairs ?? 0) + 1;
        }
      }
    }
    info.axes_directions = dirs.length;
    // a direction several faces share is locked; one only squared to another may turn while it stays square
    for (const k of dirs) for (const it of k.items) {
      if (k.items.length > 1) locks.dir.add(it.g);
      else if (k.perpTo) locks.perpTo.set(it.g, k.perpTo);
    }
    // 2. axis lines inside each direction; 3. refits
    const lines = [];
    for (const k of dirs) {
      const d = k.d;
      const axial = k.items.filter((it) => it.kind !== 'plane');
      const flat = (o) => sub(o, scale(d, dot(o, d)));   // the line's point in the plane through 0, normal d
      const ls = greedy(axial.map((it) => ({ ...it, q: flat(it.o) })), (L, it) => norm(sub(L.q, it.q)) <= offTol, (L, it, first) => {
        if (first) { L.sq = scale(it.q, it.w); L.w = it.w; } else { L.sq = add(L.sq, scale(it.q, it.w)); L.w += it.w; }
        L.q = scale(L.sq, 1 / L.w);
      });
      for (const L of ls) {
        lines.push({ o: L.q, d });
        if (L.items.length > 1) info.axes_lines_shared++;
        else if (!k.changed) continue;          // alone on its line and in its direction: its own fit stands
        const cyl = [];
        for (const it of L.items) {
          if (L.items.length > 1) {
            locks.line.add(it.g);
            if (it === L.items[0]) locks.lines.push({ o: L.q, d, members: L.items.map((m) => m.g) });
          }
          const np = refitAbout(it.kind, it.p, pts.get(it.g), L.q, d, cfg.FLOOR * mesh.diag);
          if (np) { out.set(it.g, [it.kind, np]); if (it.kind === 'cylinder') cyl.push(it.g); }
        }
        // one radius for cylinders on one line whose radii agree
        const byR = greedy(cyl.map((g) => ({ g, r: out.get(g)[1].radius, w: weight.get(g), r0: rms0.get(g) })),
          (c, it) => Math.abs(c.r - it.r) <= lenTol(c.r0, it.r0, cfg.RADIUS * c.r),
          (c, it, first) => { if (first) { c.sr = it.r * it.w; c.w = it.w; c.r0 = it.r0; } else { c.sr += it.r * it.w; c.w += it.w; c.r0 = Math.min(c.r0, it.r0); } c.r = c.sr / c.w; });
        for (const c of byR) if (c.items.length > 1) {
          info.axes_same_radius += c.items.length - 1;
          for (const it of c.items) out.set(it.g, ['cylinder', { ...out.get(it.g)[1], radius: c.r }]);
        }
      }
      // planes: the exact direction, one offset for planes of this direction that agree
      const planes = k.items.filter((it) => it.kind === 'plane').map((it) => {
        const s = Math.sign(dot(it.d, d)) || 1, n = scale(d, s);
        const P = pts.get(it.g);
        const c = scale(P.reduce((a, x) => add(a, x), [0, 0, 0]), 1 / P.length);
        return { g: it.g, n, off: median(P.map((x) => dot(x, d))), c, w: it.w, r0: it.r0 };   // w: the precision weight
      });
      const byOff = greedy(planes, (c, it) => Math.abs(c.off - it.off) <= lenTol(c.r0, it.r0, offTol),
        (c, it, first) => { if (first) { c.so = it.off * it.w; c.w = it.w; c.r0 = it.r0; } else { c.so += it.off * it.w; c.w += it.w; c.r0 = Math.min(c.r0, it.r0); } c.off = c.so / c.w; });
      for (const c of byOff) {
        if (c.items.length > 1) {
          info.axes_same_plane += c.items.length - 1;
          const set = c.items.map((it) => it.g);
          for (const it of c.items) { locks.plane.add(it.g); locks.planeSet.set(it.g, set); }
        }
        else if (!k.changed) continue;
        for (const it of c.items) {
          const origin = add(it.c, scale(d, c.off - dot(it.c, d)));
          out.set(it.g, ['plane', { ...surfaces.get(it.g)[1], origin, normal: it.n }]);
        }
      }
    }
    // a plane along an axis that passes within its precision of the axis contains it (a cut through a
    // shaft's axis, a slot's mid-plane)
    if (cfg.THROUGH) for (const L of lines) {
      for (const [g, sp] of out) {
        if (!sp || sp[0] !== 'plane' || rejected.has(g) || !pts.has(g)) continue;
        const n = unit(sp[1].normal);
        if (Math.abs(dot(n, L.d)) > 1e-12) continue;
        const dist = dot(sub(L.o, sp[1].origin), n);
        if (dist === 0 || Math.abs(dist) > lenTol(rms0.get(g), rms0.get(g), offTol)) continue;
        out.set(g, ['plane', { ...sp[1], origin: add(sp[1].origin, scale(n, dist)) }]);
        info.axes_plane_through_axis = (info.axes_plane_through_axis ?? 0) + 1;
      }
    }
    // spheres: a centre near a shared line moves onto it
    for (const [g, s] of surfaces) {
      if (!s || s[0] !== 'sphere' || rejected.has(g) || !pts.has(g)) continue;
      const c = s[1].center;
      let best = null, bd = offTol;
      for (const L of lines) {
        const v = sub(c, L.o), dist = norm(sub(v, scale(L.d, dot(v, L.d))));
        if (dist <= bd) { bd = dist; best = L; }
      }
      if (best) { const np = refitAbout('sphere', s[1], pts.get(g), best.o, best.d, cfg.FLOOR * mesh.diag); if (np) out.set(g, ['sphere', np]); }
    }
    // 4. the check: each changed surface must still fit its own vertices, or it keeps its own fit. Never
    // another type: a type change is the studio's Repair, where the user sees it (the Manta report: two
    // rim-only bands passed as tori about the fan axis on their vertices, 90 degrees off their normals)
    let newly = 0;
    for (const [g, s] of out) {
      if (s === surfaces.get(g) || accepts(g, s[0], s[1])) continue;
      rejected.add(g);
      newly++;
    }
    if (!newly) break;
  }
  for (const g of rejected) out.set(g, surfaces.get(g));
  info.axes_rejected = rejected.size;
  for (const g of rejected) { locks.dir.delete(g); locks.line.delete(g); locks.plane.delete(g); }
  const tangentSet = new Set();
  if (cfg.TANGENT_DEG > 0) Object.assign(info, makeTangent(mesh, out, weight, accepts, pts, cfg, locks, tangentSet));
  if (cfg.VALUE_REL > 0) Object.assign(info, shareValues(mesh, out, weight, accepts, pts, rms0, noisy ? cfg : { ...cfg, INTENT_REL: 0, INTENT_DEG: 0 }, locks, tangentSet, intent));
  Object.assign(info, info0);
  const byKind = {};
  for (const g of rejected) { const k = surfaces.get(g)[0]; byKind[k] = (byKind[k] ?? 0) + 1; }
  for (const [k, n] of Object.entries(byKind)) info[`axes_rejected_${k}`] = n;
  for (const [g, s] of out) if (s !== surfaces.get(g)) info.axes_snapped++;
  return { surfaces: out, info, rejected };
}


// ── tangency ────────────────────────────────────────────────────────────────────────────────────────
//
// Faces that meet smoothly in the mesh (their surfaces' normals within TANGENT_DEG along the shared
// boundary) are made exactly tangent: a gap g between a plane and a cylinder of radius R shows as a
// crease of sqrt(2 g / R) radians, so 0.0002 on the user's chain was a 0.3-degree kink. Members are taken
// most precise first and each is refitted against its MORE precise tangent partners only; the partners'
// constraints are linear in the member's parameters:
//   cylinder   its cross-section circle (centre, radius), tangent to planes along its axis
//   torus      its profile circle (R, zc, r), tangent to a coaxial cylinder (R +- r = Rc) and to planes
//              square to the axis (zc +- r = z)
//   sphere     the same profile with R = 0
//   cone       on its fixed axis, one tangent plane fixes it: the apex where the axis meets the plane,
//              the half-angle from the plane's normal
//   plane      moved (and, for a cone, turned) onto its more precise cylinder or cone
// Every result passes the same acceptance as the shared axes, or the member keeps its surface.

/** label pair "a|b" (a < b) -> Set of mesh vertices on their shared boundary. */
function boundaries(mesh) {
  const out = new Map();
  for (let t = 0; t < mesh.T.length; t++) {
    const [a, b, c] = mesh.T[t];
    for (const [u, v] of [[a, b], [b, c], [c, a]]) {
      const o = mesh.across(t, u, v);
      if (o < 0 || mesh.F[o] === mesh.F[t] || o < t) continue;
      const g1 = Math.min(mesh.F[t], mesh.F[o]), g2 = Math.max(mesh.F[t], mesh.F[o]);
      const k = `${g1}|${g2}`;
      let set = out.get(k);
      if (!set) out.set(k, (set = new Set()));
      set.add(u).add(v);
    }
  }
  return out;
}

const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/** Circle fit, sum (|q - c| - r)^2 over 2D points q, subject to linear constraints A [cx, cy, r] = b. */
function constrainedCircle(Q, x0, A, b) {
  // project x0 onto the constraint set (least-norm correction)
  const M = A.map((a) => A.map((c) => dot3(a, c)));
  const res = A.map((a, i) => b[i] - dot3(a, x0));
  const y = solve(M, res);
  if (!y) return null;
  let x = x0.map((xi, j) => xi + A.reduce((s, a, i) => s + a[j] * y[i], 0));
  // an orthonormal basis of the null space of A
  const rows = [];
  for (const a of A) {
    let v = a.slice();
    for (const r of rows) { const d = dot3(v, r); v = v.map((xx, i) => xx - d * r[i]); }
    const l = Math.hypot(...v);
    if (!(l > 1e-12)) return null;                     // dependent constraints
    rows.push(v.map((xx) => xx / l));
  }
  const N = [];
  for (const e of [[1, 0, 0], [0, 1, 0], [0, 0, 1]]) {
    let w = e.slice();
    for (const r of [...rows, ...N]) { const d = dot3(w, r); w = w.map((xx, i) => xx - d * r[i]); }
    const l = Math.hypot(...w);
    if (l > 1e-9) N.push(w.map((xx) => xx / l));
  }
  for (let it = 0; it < 30 && N.length; it++) {
    const k = N.length, JtJ = Array.from({ length: k }, () => new Array(k).fill(0)), Jtr = new Array(k).fill(0);
    for (const [u, v] of Q) {
      const du = u - x[0], dv = v - x[1], l = Math.hypot(du, dv) || 1e-300, r = l - x[2];
      const Jx = [-du / l, -dv / l, -1];
      const Jz = N.map((n) => dot3(Jx, n));
      for (let i = 0; i < k; i++) { Jtr[i] += Jz[i] * r; for (let j = 0; j < k; j++) JtJ[i][j] += Jz[i] * Jz[j]; }
    }
    for (let i = 0; i < k; i++) JtJ[i][i] = JtJ[i][i] * (1 + 1e-9) + 1e-300;
    const z = solve(JtJ, Jtr.map((v) => -v));
    if (!z) break;
    x = x.map((xi, j) => xi + N.reduce((s2, n, i) => s2 + n[j] * z[i], 0));
    if (Math.max(...z.map(Math.abs)) < 1e-14 * (Math.abs(x[2]) + 1)) break;
  }
  return x;
}

/**
 * Least squares sum r(x)^2 subject to linear constraints A x = b, x in R^n: x projected onto the
 * constraints, then Gauss-Newton in their null space. resid(x) -> {r: [], J: [[n]]}. null if the
 * constraints are dependent.
 */
function constrainedLsq(n, resid, x0, A0, b0) {
  const dotn = (u, v) => u.reduce((s2, ui, i) => s2 + ui * v[i], 0);
  // an independent subset of the constraints (redundant ones, e.g. two rims on one line touching the same
  // pair of planes, are checked after the solve instead)
  const A = [], b = [], basis = [];
  A0.forEach((a, k) => {
    let v = a.slice();
    for (const r of basis) { const d = dotn(v, r); v = v.map((x, i) => x - d * r[i]); }
    const l = Math.hypot(...v);
    if (l > 1e-9 * Math.hypot(...a)) { basis.push(v.map((x) => x / l)); A.push(a); b.push(b0[k]); }
  });
  if (A.length) {
    const M = A.map((a) => A.map((c) => dotn(a, c)));
    const y = solve(M, A.map((a, i) => b[i] - dotn(a, x0)));
    if (!y) return null;
    x0 = x0.map((xi, j) => xi + A.reduce((s2, a, i) => s2 + a[j] * y[i], 0));
  }
  const rows = [];
  for (const a of A) {
    let v = a.slice();
    for (const r of rows) { const d = dotn(v, r); v = v.map((x, i) => x - d * r[i]); }
    const l = Math.hypot(...v);
    if (!(l > 1e-12)) return null;
    rows.push(v.map((x) => x / l));
  }
  const N = [];
  for (let k = 0; k < n; k++) {
    let w = Array.from({ length: n }, (_, i) => (i === k ? 1 : 0));
    for (const r of [...rows, ...N]) { const d = dotn(w, r); w = w.map((x, i) => x - d * r[i]); }
    const l = Math.hypot(...w);
    if (l > 1e-9) N.push(w.map((x) => x / l));
  }
  let x = x0;
  for (let it = 0; it < 30 && N.length; it++) {
    const { r, J } = resid(x), k = N.length;
    const JtJ = Array.from({ length: k }, () => new Array(k).fill(0)), Jtr = new Array(k).fill(0);
    for (let m = 0; m < r.length; m++) {
      const Jz = N.map((nn) => dotn(J[m], nn));
      for (let i = 0; i < k; i++) { Jtr[i] += Jz[i] * r[m]; for (let j = 0; j < k; j++) JtJ[i][j] += Jz[i] * Jz[j]; }
    }
    for (let i = 0; i < k; i++) JtJ[i][i] = JtJ[i][i] * (1 + 1e-9) + 1e-300;
    const z = solve(JtJ, Jtr.map((v) => -v));
    if (!z) break;
    x = x.map((xi, j) => xi + N.reduce((s2, nn, i) => s2 + nn[j] * z[i], 0));
    if (Math.max(...z.map(Math.abs)) < 1e-14 * (Math.max(...x.map(Math.abs)) + 1)) break;
  }
  // the redundant constraints must hold too (they are consistent, or the case is not tangency)
  const scale0 = Math.max(...x.map(Math.abs), 1);
  for (let k = 0; k < A0.length; k++) if (Math.abs(dotn(A0[k], x) - b0[k]) > 1e-9 * scale0) return null;
  return x;
}

/** Orthonormal (e1, e2) spanning the plane square to d. */
function frameOf(d) {
  const ref = Math.abs(d[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  const e1 = unit(cross(d, ref));
  return [e1, cross(d, e1)];
}

function makeTangent(mesh, surfaces, weight, accepts, pts, cfg, locks, tangentSet = new Set()) {
  const info = { tangent_pairs: 0, tangent_made: 0, tangent_rejected: 0 };
  const cosT = Math.cos((cfg.TANGENT_DEG * Math.PI) / 180);
  const par = 1e-9;                                    // "exactly" along / square, after the shared directions
  const partners = new Map();                          // label -> labels meeting it tangentially
  for (const [k, verts] of boundaries(mesh)) {
    const [g1, g2] = k.split('|').map(Number);
    const s1 = surfaces.get(g1), s2 = surfaces.get(g2);
    if (!s1 || !s2 || !s1[1] || !s2[1] || !SNAPPABLE.has(s1[0]) || !SNAPPABLE.has(s2[0])) continue;
    const cs = [...verts].map((v) => Math.abs(dot(distNormal(s1[0], s1[1], mesh.V[v])[1], distNormal(s2[0], s2[1], mesh.V[v])[1])));
    cs.sort((a, b) => a - b);
    if (!(cs[cs.length >> 1] >= cosT)) continue;      // median normal angle over TANGENT_DEG: a real edge
    info.tangent_pairs++;
    (partners.get(g1) ?? partners.set(g1, []).get(g1)).push(g2);
    (partners.get(g2) ?? partners.set(g2, []).get(g2)).push(g1);
  }
  const W = (g) => weight.get(g) ?? 0;
  // Shared lines first: the line's position in its cross-section and the radius of every cylinder on it,
  // fitted to the cylinders' vertices subject to every tangent plane of any of them (a lug's rim is
  // coaxial with its hole and tangent to both straight sides: the sides fix where the axis is). The other
  // members are refitted about the moved line; the line moves only if every member still passes.
  const doneLine = new Set();
  for (const L of locks.lines) {
    const d = L.d, [e1, e2] = frameOf(d), o = L.o;
    const to2 = (x) => { const v = sub(x, o); return [dot(v, e1), dot(v, e2)]; };
    const cyl = L.members.filter((g) => surfaces.get(g)?.[0] === 'cylinder' && pts.has(g));
    const n = 2 + cyl.length, A = [], b = [], used = [];
    cyl.forEach((g, i) => {
      for (const h of partners.get(g) ?? []) {
        const [k2, q] = surfaces.get(h);
        if (k2 !== 'plane' || W(h) <= W(g) || L.members.includes(h)) continue;
        const nn = unit(q.normal);
        if (Math.abs(dot(nn, d)) > par) continue;
        used.push(h);
        const n2 = [dot(nn, e1), dot(nn, e2)], off = dot(sub(q.origin, o), nn), side = off > 0 ? -1 : 1;
        const row = new Array(n).fill(0);
        row[0] = side * n2[0]; row[1] = side * n2[1]; row[2 + i] = -1;
        A.push(row); b.push(side * off);
      }
    });
    if (!A.length) continue;
    const Q = cyl.map((g) => pts.get(g).map(to2));
    const resid = (x) => {
      const r = [], J = [];
      Q.forEach((qs, i) => {
        for (const [u, v] of qs) {
          const du = u - x[0], dv = v - x[1], l = Math.hypot(du, dv) || 1e-300;
          r.push(l - x[2 + i]);
          const row = new Array(n).fill(0);
          row[0] = -du / l; row[1] = -dv / l; row[2 + i] = -1;
          J.push(row);
        }
      });
      return { r, J };
    };
    const x = constrainedLsq(n, resid, [0, 0, ...cyl.map((g) => surfaces.get(g)[1].radius)], A, b);
    if (!x) continue;
    const no = add(o, add(scale(e1, x[0]), scale(e2, x[1])));
    const next = new Map();
    let ok = true;
    for (const g of L.members) {
      const [kind, p] = surfaces.get(g);
      const i = cyl.indexOf(g);
      const np = i >= 0 ? { ...p, axisPoint: no, axis: d, radius: x[2 + i] } : refitAbout(kind, p, pts.get(g), no, d);
      if (!np || !accepts(g, kind, np)) { ok = false; break; }
      next.set(g, [kind, np]);
    }
    if (!ok) { info.tangent_lines_rejected = (info.tangent_lines_rejected ?? 0) + 1; continue; }
    for (const [g, sp] of next) { surfaces.set(g, sp); doneLine.add(g); tangentSet.add(g); }
    for (const h of used) tangentSet.add(h);
    info.tangent_lines = (info.tangent_lines ?? 0) + 1;
  }
  for (const g of [...partners.keys()].sort((a, b) => W(b) - W(a))) {
    if (doneLine.has(g) && surfaces.get(g)[0] === 'cylinder') continue;
    if (!pts.has(g)) continue;
    const [kind, p] = surfaces.get(g);
    // references: more precise partners, and partners the shared step locked (they cannot move, so a
    // movable member follows them whatever their precision)
    const locked = (h) => locks.plane.has(h) || locks.dir.has(h) || locks.line.has(h);
    const ref = partners.get(g).filter((h) => W(h) > W(g) || locked(h)).map((h) => [h, ...surfaces.get(h)]);
    ref.sort((u, v) => (locked(v[0]) - locked(u[0])) || (W(v[0]) - W(u[0])));
    if (!ref.length) continue;
    const P = pts.get(g);
    let np = null;
    const used = [];                                   // the references the constraints came from
    if (kind === 'cylinder') {
      const d = unit(p.axis), [e1, e2] = frameOf(d), o = p.axisPoint;
      const to2 = (x) => { const v = sub(x, o); return [dot(v, e1), dot(v, e2)]; };
      const A = [], b = [];
      for (const [h, k2, q] of ref) {
        if (k2 !== 'plane') continue;
        const n = unit(q.normal);
        if (Math.abs(dot(n, d)) > par) continue;        // a tangent plane runs along the axis
        used.push(h);
        const n2 = [dot(n, e1), dot(n, e2)], off = dot(sub(q.origin, o), n);
        const side = off > 0 ? -1 : 1;                  // the axis lies on this side of the plane
        A.push([side * n2[0], side * n2[1], -1]); b.push(side * off);   // side (n2 . c - off) = r
      }
      if (A.length && locks.line.has(g)) { A.push([1, 0, 0], [0, 1, 0]); b.push(0, 0); }   // on a shared line: radius only
      if (A.length && A.length <= (locks.line.has(g) ? 3 : 2)) {
        const x = constrainedCircle(P.map(to2), [0, 0, p.radius], A, b);
        if (x && x[2] > 0) np = { ...p, axisPoint: add(o, add(scale(e1, x[0]), scale(e2, x[1]))), radius: x[2] };
      }
    } else if (kind === 'torus' || kind === 'sphere') {
      let axisD = kind === 'torus' ? unit(p.axis) : null;
      if (!axisD) { const c = ref.find(([, k2]) => k2 === 'cylinder'); axisD = c ? unit(c[2].axis) : null; if (c) used.push(c[0]); }
      if (axisD) {
        const o = p.center;
        const R0 = kind === 'torus' ? p.majorRadius : 0, r0 = kind === 'torus' ? p.minorRadius : p.radius;
        const A = kind === 'sphere' ? [[1, 0, 0]] : [], b = kind === 'sphere' ? [0] : [];
        for (const [h, k2, q] of ref) {
          if (k2 === 'cylinder') {
            const v = sub(q.axisPoint, o);
            if (Math.abs(Math.abs(dot(unit(q.axis), axisD)) - 1) > par || norm(sub(v, scale(axisD, dot(v, axisD)))) > 1e-9 * mesh.diag) continue;
            used.push(h);
            A.push([1, 0, Math.sign(q.radius - R0) || 1]); b.push(q.radius);            // R + s r = Rc
          } else if (k2 === 'plane') {
            const n = unit(q.normal);
            if (Math.abs(Math.abs(dot(n, axisD)) - 1) > par) continue;
            used.push(h);
            const zp = dot(sub(q.origin, o), axisD);
            A.push([0, 1, Math.sign(zp) || 1]); b.push(zp);                              // zc + s r = z
          }
        }
        const base = kind === 'sphere' ? 1 : 0;
        if (A.length > base && A.length <= 2 + base) {
          const Q = profile(P, o, axisD).map(([z, rho]) => [rho, z]);
          const x = constrainedCircle(Q, [R0, 0, r0], A, b);
          if (x && x[2] > 0) np = kind === 'torus'
            ? { ...p, center: add(o, scale(axisD, x[1])), axis: axisD, majorRadius: x[0], minorRadius: x[2] }
            : { ...p, center: add(o, scale(axisD, x[1])), radius: x[2] };
        }
      }
    } else if (kind === 'cone') {
      const d = unit(p.axis);
      const pl = ref.find(([, k2, q]) => k2 === 'plane' && Math.abs(dot(unit(q.normal), d)) > 1e-3 && Math.abs(dot(unit(q.normal), d)) < 1 - 1e-3);
      if (pl) {
        const q = pl[2], n = unit(q.normal), t = dot(sub(q.origin, p.apex), n) / dot(d, n);
        np = { ...p, apex: add(p.apex, scale(d, t)), halfAngle: Math.asin(Math.abs(dot(n, d))) };
        used.push(pl[0]);
      }
    } else if (kind === 'plane') {
      const n0 = unit(p.normal);
      const cyl = ref.find(([, k2, q]) => k2 === 'cylinder' && Math.abs(dot(unit(q.axis), n0)) <= par);
      const cone = ref.find(([, k2]) => k2 === 'cone');
      if (cyl) {
        const q = cyl[2], off = dot(sub(p.origin, q.axisPoint), n0), sgn = Math.sign(off) || 1;
        np = { ...p, origin: add(p.origin, scale(n0, sgn * q.radius - off)) };
        used.push(cyl[0]);
      } else if (cone && !locks.dir.has(g)) {
        const q = cone[2], a = unit(q.axis), sa = Math.sin(q.halfAngle), ca = Math.cos(q.halfAngle);
        const s = Math.sign(dot(n0, a)) || 1, perp = sub(n0, scale(a, dot(n0, a)));
        const nn = norm(perp) > 1e-9 ? add(scale(a, s * sa), scale(unit(perp), ca)) : null;
        if (nn && (locks.perpTo.get(g) ?? []).every((f) => Math.abs(dot(nn, f)) <= 1e-9)) { np = { ...p, normal: nn, origin: q.apex }; used.push(cone[0]); }
      }
    }
    if (!np) continue;
    // a plane shared by several faces (one offset) moves as a whole, or not at all
    const set = kind === 'plane' ? (locks.planeSet.get(g) ?? [g]) : [g];
    if (set.every((h) => !pts.has(h) || accepts(h, kind, np))) {
      for (const h of set) { surfaces.set(h, [kind, np]); tangentSet.add(h); }
      // the references it was solved against hold the tangency too: the equal-size pass moving one leaves a
      // near-tangent join the kernel builds slowly (00182468_37: 17 s -> 673 s)
      for (const h of used) tangentSet.add(h);
      info.tangent_made++;
    } else info.tangent_rejected++;
  }
  return info;
}

// ── equal sizes ──────────────────────────────────────────────────────────────────────────────────────
//
// Design intent the part repeats: a pattern of identical holes, every chamfer at one angle, every fillet one
// radius. Cylinder radii, cone half-angles and torus tube radii that agree within the fits' precision (capped
// at VALUE_REL relative; cones at VALUE_ANGLE_DEG and CONE_REL) take one shared value, the precision-weighted
// mean; each member then takes that value (nothing else moved) and must pass the usual acceptance.
// Members of an exact tangency - the surface the tangency pass moved and the references its constraints came
// from - keep their own value and share with no one:
//   - moving a reference breaks the tangency, and the kernel builds the near-tangent join slowly (00182468_37:
//     17 s -> 673 s);
//   - a value the tangency fixed comes from the constraint, not the surface's own points: as a group's value it
//     pulled twelve exact hole radii 6e-6 off on 00005406, and the build timed out (with it kept out, 77 s).
// A joint solve that moves the tangent partners with the size would let them share too.
const VALUE_OF = { cylinder: (p) => p.radius, cone: (p) => p.halfAngle, torus: (p) => p.minorRadius };

function shareValues(mesh, surfaces, weight, accepts, pts, rms0, cfg, locks, fixedSet, intent) {
  const info = { values_shared: 0, values_rejected: 0 };
  for (const kind of Object.keys(VALUE_OF)) {
    const items = [];
    for (const [g, sp] of surfaces) {
      if (!sp || sp[0] !== kind || !sp[1] || !pts.has(g) || fixedSet.has(g)) continue;
      const P = pts.get(g);
      const c = scale(P.reduce((a, x) => add(a, x), [0, 0, 0]), 1 / P.length);
      const ext = Math.sqrt(P.reduce((a, x) => a + dot(sub(x, c), sub(x, c)), 0) / P.length) || mesh.diag;
      items.push({ g, p: sp[1], v: VALUE_OF[kind](sp[1]), w: weight.get(g) ?? 0,
        tol: kind === 'cone' ? rms0.get(g) / ext : rms0.get(g) });
    }
    // a cone's angle shares only within CONE_REL of itself too: near-zero angles (a cylinder fitted as a cone)
    // agree in absolute terms but pivoting one onto another moves its apex hundreds of times its distance
    const cap = (v) => (kind === 'cone' ? Math.min((cfg.VALUE_ANGLE_DEG * Math.PI) / 180, cfg.CONE_REL * Math.abs(v)) : cfg.VALUE_REL * Math.abs(v));
    const floor = kind === 'cone' ? 1e-9 : 10 * cfg.FLOOR * mesh.diag;
    const groups = [];
    for (const it of [...items].sort((a, b) => b.w - a.w)) {
      const intentTol = kind === 'cone' ? (cfg.INTENT_DEG * Math.PI) / 180 : cfg.INTENT_REL * Math.abs(it.v);
      const k = groups.find((q) => Math.abs(q.v - it.v) <= Math.min(cap(q.v), Math.max(cfg.SIGMA_K * (q.tol + it.tol) + floor, intentTol)));
      if (!k) { groups.push({ v: it.v, sw: it.w, tol: it.tol, items: [it] }); continue; }
      k.items.push(it);
      k.tol = Math.min(k.tol, it.tol);
      if (it.w > 0) { k.v = (k.v * k.sw + it.v * it.w) / (k.sw + it.w); k.sw += it.w; }
    }
    for (const k of groups) {
      if (k.items.length < 2) continue;
      for (const it of k.items) {
        if (it.v === k.v) continue;
        const np = withValue(kind, it.p, k.v, pts.get(it.g));
        // the intent floor only for a member that joined on the intent tolerance; one that agrees within
        // precision (all of them on an exact mesh) is judged like any other move
        if (Math.abs(it.v - k.v) > cfg.SIGMA_K * (it.tol + k.tol) + floor) intent.add(it.g);
        if (np && accepts(it.g, kind, np)) { surfaces.set(it.g, [kind, np]); info.values_shared++; }
        else info.values_rejected++;
      }
    }
  }
  return info;
}

/**
 * The surface with its size set to v (radius, half-angle or tube radius) and nothing else moved. Refitting
 * the rest about the points cost the kernel: on 00007917 re-estimating two cones' apexes by a median (a
 * 6e-8 shift, angle change 2e-10) made the build 16x slower, the smallest change did not. A cone pivots
 * about its patch: the radius at the points' mean height is kept.
 */
function withValue(kind, p, v, P) {
  if (kind === 'cylinder') return { ...p, radius: v };
  if (kind === 'torus') return { ...p, minorRadius: v };
  if (kind === 'cone') {
    const d = unit(p.axis), zr = profile(P, p.apex, d);
    const zc = zr.reduce((a, [z]) => a + z, 0) / zr.length;
    const sh = zc * (1 - Math.tan(p.halfAngle) / Math.tan(v));
    return { ...p, apex: add(p.apex, scale(d, sh)), halfAngle: v };
  }
  return null;
}
