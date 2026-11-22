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
 *                values are recomputed without it (a member of a pooled line first splits the line)
 *   tangency     faces that meet smoothly made exactly tangent, all at once, with the equal sizes among
 *                them (knit/tangency.js); equal sizes elsewhere shared as before
 *
 * Bsplines and surfaceless groups are left alone.
 */

import { dot, sub, add, scale, cross, norm, unit, distNormal } from './geometry.js';
import { solveTangency } from './tangency.js';

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
  const rejected = new Set(), tight = new Set(), lineMates = new Map();
  // dirOf: each label's direction cluster {d, items, perpOf (the heavier clusters it was squared to), frame}; a frame
  // direction (the part's main ones, or one an intent or mirror relation set) stays where this step put it
  // lines: the shared axis lines and their members; planeSet: each plane's labels sharing one plane
  const locks = { lines: [], planeSet: new Map(), dirOf: new Map() };
  let out = new Map(surfaces), info;
  for (let round = 0; round < cfg.ROUNDS; round++) {
    out = new Map(surfaces);
    info = { axes_surfaces: 0, axes_directions: 0, axes_lines_shared: 0, axes_snapped: 0, axes_rejected: 0,
      axes_same_radius: 0, axes_same_plane: 0, axes_perpendicular: 0 };
    locks.lines.length = 0; locks.planeSet.clear(); locks.dirOf.clear(); lineMates.clear();
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
      if (perp.length) { info.axes_perpendicular++; k.changed = true; k.perpTo = perp.slice(0, 2).map((f) => f.d); k.perpOf = perp.slice(0, 2); }
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
    for (const k of mains) k.frame = true;
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
          a.frame = best.frame = true;
          for (const it of fa ? best.items : fb ? a.items : [...a.items, ...best.items]) intent.add(it.g);
          used.add(a).add(best);
          if (round === 0) info.axes_mirror_pairs = (info.axes_mirror_pairs ?? 0) + 1;
        }
      }
    }
    info.axes_directions = dirs.length;
    // the direction clusters, for the tangency step (a frame direction never turns there)
    for (const k of dirs) if (k.merged) k.frame = true;
    for (const k of dirs) for (const it of k.items) locks.dirOf.set(it.g, k);
    // 2. axis lines inside each direction; 3. refits
    const lines = [];
    for (const k of dirs) {
      const d = k.d;
      const axial = k.items.filter((it) => it.kind !== 'plane');
      const flat = (o) => sub(o, scale(d, dot(o, d)));   // the line's point in the plane through 0, normal d
      // one line within OFFSET; the members of a line that failed the check last round are pooled only within their
      // axes' precision (AXIS_COND times worse than their RMS, as for directions), capped at OFFSET. The Base's inner
      // and outer corner axes, 0.106 apart, pooled into one line that failed every member, so none was made
      // tangent; but a precision cap from the start split a shaft's narrow cones (00000270_015: axes 1-2e-3 off,
      // truly coaxial, whose axis their RMS places far better than they are) and cost the solid
      const ls = greedy(axial.map((it) => ({ ...it, q: flat(it.o), u: cfg.AXIS_COND * it.r0, tight: tight.has(it.g) })),
        (L, it) => norm(sub(L.q, it.q)) <= (L.tight || it.tight ? lenTol(L.u, it.u, offTol) : offTol), (L, it, first) => {
          if (first) { L.sq = scale(it.q, it.w); L.w = it.w; L.u = it.u; L.tight = it.tight; }
          else { L.sq = add(L.sq, scale(it.q, it.w)); L.w += it.w; L.u = Math.min(L.u, it.u); L.tight ||= it.tight; }
          L.q = scale(L.sq, 1 / L.w);
        });
      for (const L of ls) {
        for (const it of L.items) lineMates.set(it.g, L.items.map((m) => m.g));
        lines.push({ o: L.q, d });
        if (L.items.length > 1) info.axes_lines_shared++;
        else if (!k.changed) continue;          // alone on its line and in its direction: its own fit stands
        const cyl = [];
        for (const it of L.items) {
          if (L.items.length > 1 && it === L.items[0]) locks.lines.push({ o: L.q, d, members: L.items.map((m) => m.g) });
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
          for (const it of c.items) locks.planeSet.set(it.g, set);
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
    const tightBefore = new Set(tight);
    for (const [g, s] of out) {
      if (s === surfaces.get(g) || accepts(g, s[0], s[1])) continue;
      // a member of a pooled line splits it first (its members pool by precision next round), then is rejected
      const mates = lineMates.get(g) ?? [];
      if (mates.length > 1 && !tightBefore.has(g) && round < cfg.ROUNDS - 1) { for (const h of mates) tight.add(h); newly++; continue; }
      rejected.add(g);
      newly++;
    }
    if (!newly) break;
  }
  for (const g of rejected) out.set(g, surfaces.get(g));
  info.axes_rejected = rejected.size;
  for (const g of rejected) { locks.dirOf.delete(g); locks.planeSet.delete(g); }
  for (const L of locks.lines) L.members = L.members.filter((g) => !rejected.has(g));
  for (const [g, set] of locks.planeSet) locks.planeSet.set(g, set.filter((h) => !rejected.has(h)));
  const tangentSet = new Set(), solvedSet = new Set();
  const sigmaOf = (g) => Math.max(noiseOf.get(g), cfg.PRECISION_MIN * mesh.diag);
  const groups = cfg.VALUE_REL > 0 ? valueGroups(mesh, out, weight, pts, rms0, noisy ? cfg : { ...cfg, INTENT_REL: 0, INTENT_DEG: 0 }) : [];
  for (const k of groups) for (const it of k.items) if (it.byIntent) intent.add(it.g);
  // faces that meet smoothly made exactly tangent, all at once, with the equal sizes among them (knit/tangency.js)
  if (cfg.TANGENT_DEG > 0) Object.assign(info, solveTangency(mesh, out, { sigma: sigmaOf, accepts, pts, cfg, locks, tangentSet, values: groups, solvedSet }));
  if (cfg.VALUE_REL > 0) Object.assign(info, shareValues(out, groups, accepts, pts, solvedSet));
  Object.assign(info, info0);
  const byKind = {};
  for (const g of rejected) { const k = surfaces.get(g)[0]; byKind[k] = (byKind[k] ?? 0) + 1; }
  for (const [k, n] of Object.entries(byKind)) info[`axes_rejected_${k}`] = n;
  for (const [g, s] of out) if (s !== surfaces.get(g)) info.axes_snapped++;
  return { surfaces: out, info, rejected };
}


// ── equal sizes ──────────────────────────────────────────────────────────────────────────────────────
//
// Design intent the part repeats: a pattern of identical holes, every chamfer at one angle, every fillet one
// radius. Cylinder radii, cone half-angles and torus tube radii that agree within the fits' precision (capped
// at VALUE_REL relative; cones at VALUE_ANGLE_DEG and CONE_REL) take one shared value, the precision-weighted
// mean; each member then takes that value (nothing else moved) and must pass the usual acceptance.
// The groups are found before the tangency step. A group with a member in a set of tangent faces is solved with
// that set (knit/tangency.js: the sizes held equal while the tangencies are made), since moving one member alone
// breaks its tangency (and the kernel builds a near-tangent join slowly: 00182468_37, 17 s -> 673 s), and leaving
// it out cost exact equality where it mattered most: two radius-4 pipes mitred at 30 degrees meet in two planar
// ellipses only when the radii are equal (00009276_011, its elbow failed at 4.0000005 against 3.9999996).
// The other groups are applied here.
const VALUE_OF = { cylinder: (p) => p.radius, cone: (p) => p.halfAngle, torus: (p) => p.minorRadius };

/** The equal-size groups (two or more members): {kind, v, tol, items: [{g, p, v, tol, byIntent}]}. */
function valueGroups(mesh, surfaces, weight, pts, rms0, cfg) {
  const out = [];
  for (const kind of Object.keys(VALUE_OF)) {
    const items = [];
    for (const [g, sp] of surfaces) {
      if (!sp || sp[0] !== kind || !sp[1] || !pts.has(g)) continue;
      const P = pts.get(g);
      const c = scale(P.reduce((a, x) => add(a, x), [0, 0, 0]), 1 / P.length);
      const ext = Math.sqrt(P.reduce((a, x) => a + dot(sub(x, c), sub(x, c)), 0) / P.length) || mesh.diag;
      items.push({ g, p: sp[1], v: VALUE_OF[kind](sp[1]), w: weight.get(g) ?? 0,
        tol: kind === 'cone' ? rms0.get(g) / ext : rms0.get(g) });
    }
    // a cone's angle shares only within CONE_REL of itself too: near-zero angles (a cylinder fitted as a cone)
    // agree in absolute terms but pivoting one onto another moves its apex hundreds of times its distance
    const cap = (v) => (kind === 'cone' ? Math.min((cfg.VALUE_ANGLE_DEG * Math.PI) / 180, cfg.CONE_REL * Math.abs(v)) : cfg.VALUE_REL * Math.abs(v));
    // the floor, FLOOR x diagonal, is about ten times float32's rounding: at ten times that, radii of 0.5 and 0.4997
    // that the fits tell apart to 1e-6 were made equal once tangent faces could share (00002312_012)
    const floor = kind === 'cone' ? 1e-9 : cfg.FLOOR * mesh.diag;
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
      // the intent floor only for a member that joined on the intent tolerance; one that agrees within
      // precision (all of them on an exact mesh) is judged like any other move
      for (const it of k.items) it.byIntent = Math.abs(it.v - k.v) > cfg.SIGMA_K * (it.tol + k.tol) + floor;
      out.push({ kind, v: k.v, tol: k.tol, items: k.items });
    }
  }
  return out;
}

/** Each group's shared value applied to its members (nothing else moved), groups touching `skip` left out. */
function shareValues(surfaces, groups, accepts, pts, skip) {
  const info = { values_shared: 0, values_rejected: 0 };
  for (const k of groups) {
    if (k.items.some((it) => skip.has(it.g))) continue;
    for (const it of k.items) {
      if (it.v === k.v) continue;
      const np = withValue(k.kind, surfaces.get(it.g)[1], k.v, pts.get(it.g));
      if (np && accepts(it.g, k.kind, np)) { surfaces.set(it.g, [k.kind, np]); info.values_shared++; }
      else info.values_rejected++;
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
