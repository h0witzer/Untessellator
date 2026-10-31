/**
 * holeFill.js — close the holes the import repair (meshRepair.js) leaves, and the open edges tangled with
 * non-manifold ones. No three.js, no OCCT.
 *
 * What the import leaves open is rarely a clean loop: on the survey of 45 game models, 19 showcase meshes and 78 ABC
 * parts, every mesh still open after it had edges on 3 or more triangles too, and only 2 to 5 open edges, chains
 * that run into a fin or a wall stuck to the surface. So, in order:
 *
 *   orient      consistent winding across every edge with two triangles
 *   fins, walls the sheets that edges on 3+ triangles cut off are judged by the generalized winding number
 *               (Jacobson et al. 2013) just in front of and behind them: a surface has inside on one side only (mean
 *               near 1/2); a wall inside a body has it on both (near 1), a fin outside on neither (near 0); those go
 *   pair        what is left on 3+ triangles is bodies touching at an edge: round it, each face pairs with the face
 *               next to it on its inside, when that face's inside is it in turn
 *   flat        an open piece standing alone that closes over no volume (FLAT x area^1.5: a stray flat sheet) is
 *               left as it is rather than capped into a sheet of zero thickness; a 1 mm plate 100 mm wide closes
 *               over 0.0035, so a thin part is still capped
 *   loops       open edges join into loops; a loop through a vertex twice (holes touching at a corner) splits there
 *   triangulate each loop by Liepa's minimum weight triangulation (Filling Holes in Meshes, 2003): least worst
 *               dihedral against the rim and its own triangles, then least area; past DP_MAX corners, ear clipping
 *               by the flattest ear
 *   refine      cap edges longer than the rim's are split until they are not, so the cap has points inside
 *   fair        the cap's inner points move to the mean of their neighbours (the rim held), a membrane over the
 *               hole: the studio's export then puts cap points on their group's fitted surface (caps.js), so a
 *               cap on a curved face comes out on it
 *
 * Returns the soup with caps appended, src (each triangle's input triangle; -1 for a cap) and counts.
 */

export const FILL = { DP_MAX: 160, FLAT: 1e-3, REFINE: 1.5, FAIR_ROUNDS: 40, MAX_CAP: 200000 };

export function fillHoles(positions, opts = {}) {
  const cfg = { ...FILL, ...opts };
  const nIn = Math.floor(positions.length / 9);
  const report = { triangles_in: nIn, flipped: 0, sheets_split: 0, walls_removed: 0, fins_removed: 0, removed_triangles: 0,
    flat_left: 0, holes: 0, filled: 0, cap_triangles: 0, skipped: 0 };

  // ── weld exactly (the import repair snapped near corners together) ──
  const ids = new Map(), V = [];
  const T = new Int32Array(nIn * 3);
  for (let c = 0; c < nIn * 3; c++) {
    const x = positions[c * 3], y = positions[c * 3 + 1], z = positions[c * 3 + 2];
    const k = `${x},${y},${z}`;
    let i = ids.get(k);
    if (i === undefined) { i = V.length / 3; ids.set(k, i); V.push(x, y, z); }
    T[c] = i;
  }
  const nV = V.length / 3;
  const live = new Uint8Array(nIn).fill(1);
  for (let t = 0; t < nIn; t++) if (T[t * 3] === T[t * 3 + 1] || T[t * 3 + 1] === T[t * 3 + 2] || T[t * 3] === T[t * 3 + 2]) live[t] = 0;
  const P = (v) => [V[v * 3], V[v * 3 + 1], V[v * 3 + 2]];

  // ── edges: key -> half-edges (t*3+k, the edge from corner k to k+1) ──
  const edges = new Map();
  const ekey = (a, b) => (a < b ? a * nV + b : b * nV + a);
  for (let t = 0; t < nIn; t++) {
    if (!live[t]) continue;
    for (let k = 0; k < 3; k++) {
      const key = ekey(T[t * 3 + k], T[t * 3 + (k + 1) % 3]);
      (edges.get(key) ?? edges.set(key, []).get(key)).push(t * 3 + k);
    }
  }
  const from = (h) => T[h], to = (h) => T[Math.floor(h / 3) * 3 + ((h % 3) + 1) % 3];

  // ── orient: consistent winding over two-triangle edges ──
  {
    const seen = new Uint8Array(nIn);
    const flip = (t) => { const a = T[t * 3 + 1]; T[t * 3 + 1] = T[t * 3 + 2]; T[t * 3 + 2] = a; report.flipped++; };
    for (let s = 0; s < nIn; s++) {
      if (!live[s] || seen[s]) continue;
      seen[s] = 1;
      const queue = [s];
      for (let qi = 0; qi < queue.length; qi++) {
        const t = queue[qi];
        for (let k = 0; k < 3; k++) {
          const u = T[t * 3 + k], v = T[t * 3 + (k + 1) % 3];
          const hs = edges.get(ekey(u, v));
          if (hs.length !== 2) continue;
          const other = Math.floor((hs[0] / 3 | 0) === t ? hs[1] / 3 : hs[0] / 3);
          if (seen[other]) continue;
          // a consistent neighbour runs the edge v -> u
          let same = false;
          for (let j = 0; j < 3; j++) if (T[other * 3 + j] === u && T[other * 3 + (j + 1) % 3] === v) same = true;
          if (same) flip(other);
          seen[other] = 1;
          queue.push(other);
        }
      }
    }
    // half-edge corners moved with the flips: rebuild the table
    edges.clear();
    for (let t = 0; t < nIn; t++) {
      if (!live[t]) continue;
      for (let k = 0; k < 3; k++) {
        const key = ekey(T[t * 3 + k], T[t * 3 + (k + 1) % 3]);
        (edges.get(key) ?? edges.set(key, []).get(key)).push(t * 3 + k);
      }
    }
  }

  const normal = (t) => {
    const a = P(T[t * 3]), b = P(T[t * 3 + 1]), c = P(T[t * 3 + 2]);
    return unit(cross(sub(b, a), sub(c, a)));
  };
  const rebuild = () => {
    edges.clear();
    for (let t = 0; t < nIn; t++) {
      if (!live[t]) continue;
      for (let k = 0; k < 3; k++) {
        const key = ekey(T[t * 3 + k], T[t * 3 + (k + 1) % 3]);
        (edges.get(key) ?? edges.set(key, []).get(key)).push(t * 3 + k);
      }
    }
  };
  const tri = (h) => Math.floor(h / 3);

  // ── fins and walls: the sheets an edge on 3+ triangles cuts off, judged by the generalized winding number ──
  // (Jacobson, Kavan and Sorkine-Hornung 2013). A surface has inside on one side and outside on the other, so the
  // mean of the winding numbers just in front of it and just behind it is near 1/2 (-1/2 wound inward); a wall inside
  // a body has inside on both sides (near 1), a fin outside it neither (near 0). Those go; bodies touching at an edge,
  // and the surfaces round them, stay.
  {
    const sheet = new Int32Array(nIn).fill(-1), sheets = [];
    for (let s0 = 0; s0 < nIn; s0++) {
      if (!live[s0] || sheet[s0] >= 0) continue;
      const id = sheets.length, list = [s0];
      sheet[s0] = id;
      for (let qi = 0; qi < list.length; qi++) {
        const t = list[qi];
        for (let k = 0; k < 3; k++) {
          const hs = edges.get(ekey(T[t * 3 + k], T[t * 3 + (k + 1) % 3]));
          if (hs.length !== 2) continue;
          const o = tri(hs[0]) === t ? tri(hs[1]) : tri(hs[0]);
          if (sheet[o] < 0) { sheet[o] = id; list.push(o); }
        }
      }
      sheets.push(list);
    }
    const cut = new Set();
    for (const hs of edges.values()) if (hs.length >= 3) for (const h of hs) cut.add(sheet[tri(h)]);
    report.sheets_split = cut.size;
    if (cut.size) {
      // every live triangle, flat, for the winding number
      const G = [];
      for (let t = 0; t < nIn; t++) if (live[t]) for (let k = 0; k < 3; k++) G.push(...P(T[t * 3 + k]));
      const g = Float64Array.from(G);
      let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
      for (let i = 0; i < g.length; i++) { lo[i % 3] = Math.min(lo[i % 3], g[i]); hi[i % 3] = Math.max(hi[i % 3], g[i]); }
      const eps = 1e-5 * (Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) || 1);
      const per = Math.max(1, Math.min(7, Math.floor(2e8 / (2 * (g.length / 9) * cut.size))));
      const gwn = (p) => {
        let w = 0;
        for (let i = 0; i < g.length; i += 9) {
          const ax = g[i] - p[0], ay = g[i + 1] - p[1], az = g[i + 2] - p[2];
          const bx = g[i + 3] - p[0], by = g[i + 4] - p[1], bz = g[i + 5] - p[2];
          const cx = g[i + 6] - p[0], cy = g[i + 7] - p[1], cz = g[i + 8] - p[2];
          const la = Math.hypot(ax, ay, az), lb = Math.hypot(bx, by, bz), lc = Math.hypot(cx, cy, cz);
          const det = ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx);
          const div = la * lb * lc + (ax * bx + ay * by + az * bz) * lc + (bx * cx + by * cy + bz * cz) * la + (cx * ax + cy * ay + cz * az) * lb;
          w += 2 * Math.atan2(det, div);
        }
        return w / (4 * Math.PI);
      };
      for (const id of cut) {
        const list = sheets[id];
        const area = (t) => { const a = P(T[t * 3]), b = P(T[t * 3 + 1]), c = P(T[t * 3 + 2]); return Math.hypot(...cross(sub(b, a), sub(c, a))); };
        const samples = [...list].sort((x, y) => area(y) - area(x)).slice(0, per);
        const means = samples.map((t) => {
          const a = P(T[t * 3]), b = P(T[t * 3 + 1]), c = P(T[t * 3 + 2]), n = normal(t);
          const m = [(a[0] + b[0] + c[0]) / 3, (a[1] + b[1] + c[1]) / 3, (a[2] + b[2] + c[2]) / 3];
          return Math.abs((gwn([m[0] + eps * n[0], m[1] + eps * n[1], m[2] + eps * n[2]]) + gwn([m[0] - eps * n[0], m[1] - eps * n[1], m[2] - eps * n[2]])) / 2);
        }).sort((x, y) => x - y);
        const median = means[means.length >> 1];
        if (median > 0.75 || median < 0.25) {
          for (const t of list) live[t] = 0;
          if (median > 0.75) report.walls_removed++; else report.fins_removed++;
          report.removed_triangles += list.length;
        }
      }
      rebuild();
    }
  }

  // ── pair: partner half-edge for each, or -1 (open) ──
  const partner = new Int32Array(nIn * 3).fill(-1);
  for (const hs of edges.values()) {
    if (hs.length === 2) {
      if (from(hs[0]) === to(hs[1])) { partner[hs[0]] = hs[1]; partner[hs[1]] = hs[0]; }
      continue;
    }
    if (hs.length < 3) continue;
    // bodies touching at an edge: around it, each face pairs with the face next to it on its inside, when that
    // face's inside is it in turn
    const a = P(from(hs[0])), b = P(to(hs[0]));
    const e = unit(sub(b, a));
    const r0 = unit(Math.abs(e[0]) < 0.9 ? cross(e, [1, 0, 0]) : cross(e, [0, 1, 0]));
    const r1 = cross(e, r0);
    const faces = hs.map((h) => {
      const t = tri(h);
      const c = P(T[t * 3 + ((h % 3) + 2) % 3]);
      const w = sub(c, a);
      const d = dot(w, e);
      const wp = unit([w[0] - d * e[0], w[1] - d * e[1], w[2] - d * e[2]]);
      const theta = Math.atan2(dot(wp, r1), dot(wp, r0));
      // turning from the face toward its inside (against its normal): +1 if that is increasing theta
      return { h, theta, inward: dot(cross(e, wp), normal(t)) < 0 ? 1 : -1, dir: from(h) === from(hs[0]) ? 1 : -1 };
    }).sort((x, y) => x.theta - y.theta);
    const m = faces.length;
    const insideOf = (i) => (faces[i].inward > 0 ? (i + 1) % m : (i - 1 + m) % m);
    for (let i = 0; i < m; i++) {
      const j = insideOf(i);
      if (j > i && insideOf(j) === i && faces[i].dir !== faces[j].dir) { partner[faces[i].h] = faces[j].h; partner[faces[j].h] = faces[i].h; }
    }
  }

  // ── pieces through the pairing: an open one standing alone that closes over no volume (a stray flat sheet) is
  // left as it is, not capped into a sheet of zero thickness ──
  const noFill = new Uint8Array(nIn);
  {
    const piece = new Int32Array(nIn).fill(-1);
    for (let s0 = 0; s0 < nIn; s0++) {
      if (!live[s0] || piece[s0] >= 0) continue;
      const list = [s0];
      piece[s0] = s0;
      for (let qi = 0; qi < list.length; qi++) {
        const t = list[qi];
        for (let k = 0; k < 3; k++) { const p = partner[t * 3 + k]; if (p >= 0 && piece[tri(p)] < 0) { piece[tri(p)] = s0; list.push(tri(p)); } }
      }
      let area = 0, vol = 0;
      const rim = [];
      for (const t of list) {
        const a = P(T[t * 3]), b = P(T[t * 3 + 1]), c = P(T[t * 3 + 2]);
        area += Math.hypot(...cross(sub(b, a), sub(c, a))) / 2;
        vol += dot(a, cross(b, c)) / 6;
        for (let k = 0; k < 3; k++) if (partner[t * 3 + k] < 0) rim.push(t * 3 + k);
      }
      if (!rim.length) continue;
      const c = [0, 0, 0];
      for (const h of rim) { const q = P(from(h)); for (let k = 0; k < 3; k++) c[k] += q[k] / rim.length; }
      for (const h of rim) vol += dot(P(to(h)), cross(P(from(h)), c)) / 6;
      if (Math.abs(vol) < cfg.FLAT * area ** 1.5) { for (const t of list) noFill[t] = 1; report.flat_left++; }
    }
  }

  // ── loops of open half-edges; each split into simple cycles ──
  const out = new Map();                           // vertex -> open half-edges leaving it
  for (let t = 0; t < nIn; t++) {
    if (!live[t]) continue;
    if (noFill[t]) continue;
    for (let k = 0; k < 3; k++) if (partner[t * 3 + k] < 0) { const h = t * 3 + k; (out.get(from(h)) ?? out.set(from(h), []).get(from(h))).push(h); }
  }
  const cycles = [];
  for (;;) {
    let start = -1;
    for (const l of out.values()) if (l.length) { start = l[0]; break; }
    if (start < 0) break;
    const path = [], at = new Map();
    let h = start, closed = false;
    for (let guard = 0; guard < 1e7; guard++) {
      const l = out.get(from(h));
      l.splice(l.indexOf(h), 1);
      at.set(from(h), path.length);
      path.push(h);
      if (at.has(to(h))) {
        cycles.push(path.splice(at.get(to(h))));
        for (const x of path) out.get(from(x)).push(x);
        closed = true;
        break;
      }
      const next = out.get(to(h));
      if (!next || !next.length) break;
      h = next[0];
    }
    if (!closed) report.skipped++;                 // a chain that does not close: left open
  }
  report.holes = cycles.length + report.skipped;

  // ── triangulate, refine and fair each cycle ──
  const capTris = [];
  const NV = [...V];                               // vertices, with the caps' inner points appended
  for (const cyc of cycles) {
    if (cyc.length < 3) { report.skipped++; continue; }
    // the cap runs each rim edge the other way: the loop in reverse
    const loop = cyc.map((h) => from(h)).reverse();
    // the mesh triangle on loop edge (i, i+1): the half-edge that runs it the other way, cyc[k-2-i]
    const k = cyc.length;
    const across = loop.map((_, i) => Math.floor(cyc[(2 * k - 2 - i) % k] / 3));
    const tris = loop.length <= cfg.DP_MAX ? liepa(loop, across) : ears(loop);
    if (!tris) { report.skipped++; continue; }
    const refined = refineAndFair(tris, loop);
    for (const tri of refined) capTris.push(tri);
    report.filled++;
    if (capTris.length > cfg.MAX_CAP) break;
  }
  report.cap_triangles = capTris.length;

  // ── the soup back: the input's live triangles, then the caps ──
  const keep = [];
  for (let t = 0; t < nIn; t++) if (live[t]) keep.push(t);
  const total = keep.length + capTris.length;
  const pos = new Float64Array(total * 9), src = new Int32Array(total);
  keep.forEach((t, i) => { for (let k = 0; k < 3; k++) for (let j = 0; j < 3; j++) pos[i * 9 + k * 3 + j] = NV[T[t * 3 + k] * 3 + j]; src[i] = t; });
  capTris.forEach((tri, i) => {
    const o = (keep.length + i) * 9;
    for (let k = 0; k < 3; k++) for (let j = 0; j < 3; j++) pos[o + k * 3 + j] = NV[tri[k] * 3 + j];
    src[keep.length + i] = -1;
  });
  report.triangles_out = total;
  return { positions: pos, src, report };

  // ── Liepa: minimum weight triangulation of a loop; weight (worst dihedral, area), the dihedral first ──
  function liepa(loop, across) {
    const n = loop.length;
    const p = loop.map((v) => [NV[v * 3], NV[v * 3 + 1], NV[v * 3 + 2]]);
    // the mesh triangle across loop edge (i, i+1): its normal, for the dihedral at the rim
    const rimNormal = across.map((t) => normal(t));
    const W = new Float64Array(n * n), A = new Float64Array(n * n), L = new Int32Array(n * n).fill(-1);
    const triNormal = (i, m, j) => unit(cross(sub(p[m], p[i]), sub(p[j], p[i])));
    const turn = (n1, n2) => 1 - dot(n1, n2);    // 0 flat continuation, 2 folded back
    // the normal of the triangle across edge (i, j) inside the sub-solution, or the rim's
    const neighbour = (i, j) => (j === i + 1 ? rimNormal[i] : L[i * n + j] >= 0 ? triNormal(i, L[i * n + j], j) : null);
    for (let gap = 2; gap < n; gap++) {
      for (let i = 0; i + gap < n; i++) {
        const j = i + gap;
        let bestW = Infinity, bestA = Infinity, bestM = -1;
        for (let m = i + 1; m < j; m++) {
          const nt = triNormal(i, m, j);
          if (!Number.isFinite(nt[0])) continue;
          const left = neighbour(i, m), right = neighbour(m, j);
          let w = Math.max(W[i * n + m], W[m * n + j], left ? turn(nt, left) : 0, right ? turn(nt, right) : 0);
          if (gap === n - 1) w = Math.max(w, turn(nt, rimNormal[n - 1]));   // the closing edge (n-1, 0) is rim
          const area = A[i * n + m] + A[m * n + j] + Math.hypot(...cross(sub(p[m], p[i]), sub(p[j], p[i]))) / 2;
          if (w < bestW - 1e-12 || (Math.abs(w - bestW) <= 1e-12 && area < bestA)) { bestW = w; bestA = area; bestM = m; }
        }
        if (bestM < 0) return null;
        W[i * n + j] = bestW; A[i * n + j] = bestA; L[i * n + j] = bestM;
      }
    }
    const tris = [];
    const emit = (i, j) => { if (j - i < 2) return; const m = L[i * n + j]; tris.push([loop[i], loop[m], loop[j]]); emit(i, m); emit(m, j); };
    emit(0, n - 1);
    return tris;
  }

  // ── ear clipping for long loops: the flattest ear first ──
  function ears(loop) {
    const idx = [...loop];
    const pt = (v) => [NV[v * 3], NV[v * 3 + 1], NV[v * 3 + 2]];
    // the loop's mean plane normal (Newell), to judge convex corners
    const nrm = [0, 0, 0];
    for (let i = 0; i < idx.length; i++) {
      const a = pt(idx[i]), b = pt(idx[(i + 1) % idx.length]);
      nrm[0] += (a[1] - b[1]) * (a[2] + b[2]); nrm[1] += (a[2] - b[2]) * (a[0] + b[0]); nrm[2] += (a[0] - b[0]) * (a[1] + b[1]);
    }
    const N = unit(nrm);
    const tris = [];
    while (idx.length > 3) {
      let best = -1, bestCost = Infinity;
      for (let i = 0; i < idx.length; i++) {
        const a = pt(idx[(i - 1 + idx.length) % idx.length]), b = pt(idx[i]), c = pt(idx[(i + 1) % idx.length]);
        const n = cross(sub(b, a), sub(c, b));
        const convex = dot(n, N) > 0;
        const ang = Math.acos(Math.max(-1, Math.min(1, dot(unit(sub(a, b)), unit(sub(c, b))))));
        const cost = (convex ? 0 : 10) + ang;
        if (cost < bestCost) { bestCost = cost; best = i; }
      }
      const i = best;
      tris.push([idx[(i - 1 + idx.length) % idx.length], idx[i], idx[(i + 1) % idx.length]]);
      idx.splice(i, 1);
    }
    tris.push([idx[0], idx[1], idx[2]]);
    return tris;
  }

  // ── refine (split long inner edges) and fair (inner points to their neighbours' mean, the rim held) ──
  function refineAndFair(tris, loop) {
    const pt = (v) => [NV[v * 3], NV[v * 3 + 1], NV[v * 3 + 2]];
    let rimLen = 0;
    for (let i = 0; i < loop.length; i++) rimLen += Math.hypot(...sub(pt(loop[(i + 1) % loop.length]), pt(loop[i])));
    const target = (rimLen / loop.length) * cfg.REFINE;
    const rim = new Set(loop);
    const tl = tris.map((t) => [...t]);
    const inner = new Set();
    // longest-edge bisection of edges inside the cap (both sides cap triangles), until none is past target
    for (let round = 0; round < 64; round++) {
      const owners = new Map();
      tl.forEach((t, i) => { for (let k = 0; k < 3; k++) { const a = t[k], b = t[(k + 1) % 3]; const key = a < b ? `${a},${b}` : `${b},${a}`; (owners.get(key) ?? owners.set(key, []).get(key)).push(i); } });
      let split = 0;
      const done = new Set();
      const order = [...owners.entries()].filter(([, o]) => o.length === 2)
        .map(([key, o]) => { const [a, b] = key.split(',').map(Number); return { a, b, o, len: Math.hypot(...sub(pt(a), pt(b))) }; })
        .filter((e) => e.len > target).sort((x, y) => y.len - x.len);
      for (const { a, b, o } of order) {
        if (done.has(o[0]) || done.has(o[1])) continue;
        const m = NV.length / 3;
        const pa = pt(a), pb = pt(b);
        NV.push((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2, (pa[2] + pb[2]) / 2);
        inner.add(m);
        for (const ti of o) {
          const t = tl[ti];
          const k = [0, 1, 2].find((j) => (t[j] === a && t[(j + 1) % 3] === b) || (t[j] === b && t[(j + 1) % 3] === a));
          const x = t[k], y = t[(k + 1) % 3], z = t[(k + 2) % 3];
          tl[ti] = [x, m, z];
          tl.push([m, y, z]);
          done.add(ti); done.add(tl.length - 1);
        }
        split++;
      }
      if (!split) break;
    }
    // fair: the umbrella operator on inner points
    if (inner.size) {
      const nb = new Map([...inner].map((v) => [v, new Set()]));
      for (const t of tl) for (let k = 0; k < 3; k++) { const a = t[k], b = t[(k + 1) % 3]; if (nb.has(a)) nb.get(a).add(b); if (nb.has(b)) nb.get(b).add(a); }
      for (let r = 0; r < cfg.FAIR_ROUNDS; r++) {
        for (const [v, ns] of nb) {
          const c = [0, 0, 0];
          for (const u of ns) for (let k = 0; k < 3; k++) c[k] += NV[u * 3 + k] / ns.size;
          for (let k = 0; k < 3; k++) NV[v * 3 + k] = c[k];
        }
      }
    }
    void rim;
    return tl;
  }
}

function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
function unit(a) { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; }
