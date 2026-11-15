/**
 * bodies.js — which triangles make which solid. No three.js, no OCCT; the painter, the viewer and the export all
 * read a mesh's bodies from here, so they agree.
 *
 *   pairEdges   the triangle across each edge IN THE SAME SOLID. An edge with two triangles pairs them. One with
 *               more (two bodies meshed where they touch, or a wall of opposite-wound pairs between them) is
 *               resolved radially, the knitter's rule: around the edge, a triangle traversing it backwards has
 *               material on its counter-clockwise side and one traversing it forwards on its clockwise side, so
 *               going counter-clockwise each backward triangle pairs with the next forward one. Coincident
 *               triangles tie on angle and the forward one sorts first, so a zero-thickness gap is never paired.
 *   findBodies  shells (triangles joined through paired edges), each wound consistently the way most of its
 *               triangles already were; then nesting. A closed shell inside another closed shell, wound the other
 *               way round from it, is a VOID of that shell's body (CAD exports write a cavity's skin facing in).
 *               Inside but wound the same way, it is a body of its own (an insert modelled overlapping, or a part
 *               sitting in a cavity). Every body's outer shell then faces out and every void in, whatever the
 *               file's overall winding was.
 *
 * Inputs are indexed: idx (3 vertex ids per triangle, already welded; a triangle with a repeated id is degenerate
 * and pairs with nothing) and V (3 coordinates per vertex).
 */

/** Only this many of the largest closed shells are tried as containers. */
const MAX_CONTAINERS = 500;

/** The triangle across edge k (corner k to corner k + 1) of each triangle, in the same solid; -1 if none. */
export function pairEdges(idx, V) {
  const nT = idx.length / 3;
  let nV = 0;
  for (let i = 0; i < idx.length; i++) if (idx[i] >= nV) nV = idx[i] + 1;
  const across = new Int32Array(nT * 3).fill(-1);
  const edges = new Map();     // undirected edge key -> [t * 3 + k, ...]
  for (let t = 0; t < nT; t++) {
    const a = idx[t * 3], b = idx[t * 3 + 1], c = idx[t * 3 + 2];
    if (a === b || b === c || a === c) continue;
    for (let k = 0; k < 3; k++) {
      const u = idx[t * 3 + k], v = idx[t * 3 + (k + 1) % 3];
      const key = u < v ? u * nV + v : v * nV + u;
      const list = edges.get(key);
      if (list) list.push(t * 3 + k); else edges.set(key, [t * 3 + k]);
    }
  }
  for (const list of edges.values()) {
    if (list.length < 2) continue;
    if (list.length === 2) {
      across[list[0]] = Math.floor(list[1] / 3);
      across[list[1]] = Math.floor(list[0] / 3);
      continue;
    }
    // radial order around the edge, from its lower vertex id to its higher
    const c0 = list[0], t0 = Math.floor(c0 / 3), k0 = c0 % 3;
    const p = idx[t0 * 3 + k0], q = idx[t0 * 3 + (k0 + 1) % 3];
    const ea = Math.min(p, q), eb = Math.max(p, q);
    const ex = V[eb * 3] - V[ea * 3], ey = V[eb * 3 + 1] - V[ea * 3 + 1], ez = V[eb * 3 + 2] - V[ea * 3 + 2];
    const el = Math.hypot(ex, ey, ez) || 1;
    const e = [ex / el, ey / el, ez / el];
    const ref = Math.abs(e[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
    let b1 = cross(e, ref);
    const bl = Math.hypot(b1[0], b1[1], b1[2]) || 1;
    b1 = [b1[0] / bl, b1[1] / bl, b1[2] / bl];
    const b2 = cross(e, b1);
    const items = list.map((corner) => {
      const t = Math.floor(corner / 3), k = corner % 3;
      const u = idx[t * 3 + k];
      const w = idx[t * 3 + (k + 2) % 3];
      const rx = V[w * 3] - V[ea * 3], ry = V[w * 3 + 1] - V[ea * 3 + 1], rz = V[w * 3 + 2] - V[ea * 3 + 2];
      const angle = Math.atan2(rx * b2[0] + ry * b2[1] + rz * b2[2], rx * b1[0] + ry * b1[1] + rz * b1[2]);
      const forward = u === ea ? 1 : -1;
      return { angle, forward, t, corner };
    });
    items.sort((x, y) => x.angle - y.angle || (x.forward === 1 ? 0 : 1) - (y.forward === 1 ? 0 : 1) || x.t - y.t);
    const n = items.length;
    for (let i = 0; i < n; i++) {
      if (items[i].forward !== -1) continue;
      const next = items[(i + 1) % n];
      if (next.forward === 1) {
        across[items[i].corner] = next.t;
        across[next.corner] = items[i].t;
      }
    }
  }
  return across;
}

/**
 * The mesh's shells and bodies.
 * Returns {
 *   flip:    Uint8Array per triangle, 1 where its winding must be reversed (outer shells out, voids in)
 *   across:  pairEdges of the triangles as flipped
 *   shellOf: Int32Array per triangle
 *   bodyOf:  Int32Array per triangle, bodies numbered largest volume first (open bodies after closed ones)
 *   shells:  [{triangles, closed, volume (signed, as flipped), lo, hi, body, void}]
 *   bodies:  [{triangles, closed, volume (outer minus voids), shells: [shell ids], voids, lo, hi}]
 * }
 */
export function findBodies(idx, V) {
  const nT = idx.length / 3;
  const flip = new Uint8Array(nT);
  const T = Int32Array.from(idx);
  const reverse = (t) => { const b = T[t * 3 + 1]; T[t * 3 + 1] = T[t * 3 + 2]; T[t * 3 + 2] = b; flip[t] ^= 1; };

  // 1. consistent winding per shell, the way most of its triangles were written; pair again once it is consistent,
  // since the radial rule reads the winding (as the knitter does)
  let across = pairEdges(T, V);
  orientShells(T, across, reverse);
  across = pairEdges(T, V);
  orientShells(T, across, reverse);

  // 2. shells
  const shellOf = new Int32Array(nT).fill(-1);
  const shells = [];
  for (let s = 0; s < nT; s++) {
    if (shellOf[s] >= 0) continue;
    const id = shells.length, list = [s];
    shellOf[s] = id;
    let closed = true;
    for (let i = 0; i < list.length; i++) {
      const t = list[i];
      for (let k = 0; k < 3; k++) {
        const m = across[t * 3 + k];
        if (m < 0) { closed = false; continue; }
        if (shellOf[m] < 0) { shellOf[m] = id; list.push(m); }
      }
    }
    const degenerate = list.length === 1 && isDegenerate(T, s);
    shells.push({ tris: list, triangles: list.length, closed: closed && !degenerate, volume: signedVolume(T, V, list), ...box(T, V, list) });
  }

  // 3. nesting: each closed shell's innermost closed container, containers first
  const order = shells.map((_, i) => i).filter((i) => shells[i].closed).sort((a, b) => Math.abs(shells[b].volume) - Math.abs(shells[a].volume));
  const grids = new Map();
  const gridOf = (i) => grids.get(i) ?? grids.set(i, rayGrid(T, V, shells[i].tris)).get(i);
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const s of shells) for (let a = 0; a < 3; a++) { lo[a] = Math.min(lo[a], s.lo[a]); hi[a] = Math.max(hi[a], s.hi[a]); }
  const diag = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) || 1;
  for (const s of shells) { s.body = -1; s.void = false; }
  const outer = [];
  for (let oi = 0; oi < order.length; oi++) {
    const i = order[oi], si = shells[i];
    let container = -1;
    // smaller volumes first, so the first container found is the innermost; only the MAX_CONTAINERS largest closed
    // shells can contain anything (a mesh with thousands of crumbs stays quick)
    for (let oj = Math.min(oi, MAX_CONTAINERS) - 1; oj >= 0; oj--) {
      const j = order[oj], sj = shells[j];
      if (!boxInside(si, sj, diag * 1e-9)) continue;
      if (inside(T, V, si.tris, gridOf(j), diag)) { container = j; break; }
    }
    const c = container >= 0 ? shells[container] : null;
    if (c && !c.void && Math.sign(si.volume) !== Math.sign(c.volume)) {
      si.void = true;
      si.body = c.body;
    } else {
      si.body = outer.length;
      outer.push(i);
    }
  }
  for (const s of shells) if (!s.closed) { s.body = outer.length; outer.push(shells.indexOf(s)); }

  // 4. outer shells face out, voids in
  for (const s of shells) {
    if (!s.closed) continue;
    if ((s.void && s.volume > 0) || (!s.void && s.volume < 0)) { for (const t of s.tris) reverse(t); s.volume = -s.volume; }
  }

  // 5. bodies, largest first
  let bodies = outer.map((i) => ({ shells: [], voids: 0, triangles: 0, closed: shells[i].closed, volume: 0, lo: [...shells[i].lo], hi: [...shells[i].hi] }));
  shells.forEach((s, i) => {
    const b = bodies[s.body];
    b.shells.push(i);
    b.triangles += s.triangles;
    b.volume += s.closed ? s.volume : 0;
    if (s.void) b.voids++;
  });
  const rank = bodies.map((_, i) => i).sort((a, b) => (bodies[b].closed - bodies[a].closed) || bodies[b].volume - bodies[a].volume || bodies[b].triangles - bodies[a].triangles);
  const newId = new Int32Array(bodies.length);
  rank.forEach((old, i) => { newId[old] = i; });
  bodies = rank.map((old) => bodies[old]);
  for (const s of shells) s.body = newId[s.body];
  const bodyOf = new Int32Array(nT);
  for (let t = 0; t < nT; t++) bodyOf[t] = shells[shellOf[t]].body;
  across = pairEdges(T, V);
  for (const s of shells) delete s.tris;
  return { flip, across, shellOf, bodyOf, shells, bodies };
}

/** Consistent winding over paired edges; each shell keeps the orientation most of its triangles had. */
function orientShells(T, across, reverse) {
  const nT = T.length / 3;
  const seen = new Uint8Array(nT);
  const local = new Uint8Array(nT);    // 1: reversed relative to the shell's seed
  for (let s = 0; s < nT; s++) {
    if (seen[s]) continue;
    seen[s] = 1;
    const list = [s];
    for (let i = 0; i < list.length; i++) {
      const t = list[i];
      for (let k = 0; k < 3; k++) {
        const m = across[t * 3 + k];
        if (m < 0 || seen[m]) continue;
        // t runs its edge k as u -> v (reversed if t is flipped relative to the seed); consistent, m runs it v -> u,
        // so m is flipped when its stored winding runs u -> v
        let u = T[t * 3 + k], v = T[t * 3 + (k + 1) % 3];
        if (local[t]) [u, v] = [v, u];
        local[m] = sameDirection(T, m, u, v) ? 1 : 0;
        seen[m] = 1;
        list.push(m);
      }
    }
    let flipped = 0;
    for (const t of list) flipped += local[t];
    const reverseFlipped = flipped * 2 <= list.length;   // keep the majority's winding
    for (const t of list) if (local[t] === (reverseFlipped ? 1 : 0)) reverse(t);
    for (const t of list) local[t] = 0;
  }
}

/** Whether triangle t traverses u -> v (in its stored winding). */
function sameDirection(T, t, u, v) {
  for (let k = 0; k < 3; k++) if (T[t * 3 + k] === u && T[t * 3 + (k + 1) % 3] === v) return true;
  return false;
}

const isDegenerate = (T, t) => T[t * 3] === T[t * 3 + 1] || T[t * 3 + 1] === T[t * 3 + 2] || T[t * 3] === T[t * 3 + 2];

function signedVolume(T, V, tris) {
  let s = 0;
  for (const t of tris) {
    const a = T[t * 3] * 3, b = T[t * 3 + 1] * 3, c = T[t * 3 + 2] * 3;
    s += V[a] * (V[b + 1] * V[c + 2] - V[b + 2] * V[c + 1])
       - V[a + 1] * (V[b] * V[c + 2] - V[b + 2] * V[c])
       + V[a + 2] * (V[b] * V[c + 1] - V[b + 1] * V[c]);
  }
  return s / 6;
}

function box(T, V, tris) {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const t of tris) for (let k = 0; k < 3; k++) {
    const v = T[t * 3 + k] * 3;
    for (let a = 0; a < 3; a++) { lo[a] = Math.min(lo[a], V[v + a]); hi[a] = Math.max(hi[a], V[v + a]); }
  }
  return { lo, hi };
}

const boxInside = (a, b, tol) => [0, 1, 2].every((k) => a.lo[k] >= b.lo[k] - tol && a.hi[k] <= b.hi[k] + tol);

/**
 * A container's triangles binned over the YZ plane, for rays along +x. Each cell lists the triangles whose YZ
 * bounds overlap it.
 */
function rayGrid(T, V, tris) {
  const lo = [Infinity, Infinity], hi = [-Infinity, -Infinity];
  for (const t of tris) for (let k = 0; k < 3; k++) {
    const v = T[t * 3 + k] * 3;
    lo[0] = Math.min(lo[0], V[v + 1]); hi[0] = Math.max(hi[0], V[v + 1]);
    lo[1] = Math.min(lo[1], V[v + 2]); hi[1] = Math.max(hi[1], V[v + 2]);
  }
  const n = Math.max(1, Math.min(256, Math.ceil(Math.sqrt(tris.length))));
  const sy = (hi[0] - lo[0]) / n || 1, sz = (hi[1] - lo[1]) / n || 1;
  const cells = Array.from({ length: n * n }, () => []);
  const cy = (y) => Math.min(n - 1, Math.max(0, Math.floor((y - lo[0]) / sy)));
  const cz = (z) => Math.min(n - 1, Math.max(0, Math.floor((z - lo[1]) / sz)));
  for (const t of tris) {
    let y0 = Infinity, y1 = -Infinity, z0 = Infinity, z1 = -Infinity;
    for (let k = 0; k < 3; k++) {
      const v = T[t * 3 + k] * 3;
      y0 = Math.min(y0, V[v + 1]); y1 = Math.max(y1, V[v + 1]); z0 = Math.min(z0, V[v + 2]); z1 = Math.max(z1, V[v + 2]);
    }
    for (let i = cy(y0); i <= cy(y1); i++) for (let j = cz(z0); j <= cz(z1); j++) cells[i * n + j].push(t);
  }
  return { cells, n, cy, cz };
}

/** Crossings of the ray from p along +x with the grid's triangles (odd: p is inside). */
function crossings(T, V, grid, px, py, pz) {
  let count = 0;
  for (const t of grid.cells[grid.cy(py) * grid.n + grid.cz(pz)]) {
    const a = T[t * 3] * 3, b = T[t * 3 + 1] * 3, c = T[t * 3 + 2] * 3;
    // point in the triangle's YZ projection, by signed areas
    const d1 = (V[b + 1] - V[a + 1]) * (pz - V[a + 2]) - (V[b + 2] - V[a + 2]) * (py - V[a + 1]);
    const d2 = (V[c + 1] - V[b + 1]) * (pz - V[b + 2]) - (V[c + 2] - V[b + 2]) * (py - V[b + 1]);
    const d3 = (V[a + 1] - V[c + 1]) * (pz - V[c + 2]) - (V[a + 2] - V[c + 2]) * (py - V[c + 1]);
    if (!((d1 > 0 && d2 > 0 && d3 > 0) || (d1 < 0 && d2 < 0 && d3 < 0))) continue;
    const sum = d1 + d2 + d3;
    const x = (d2 * V[a] + d3 * V[b] + d1 * V[c]) / sum;   // barycentric x of the hit
    if (x > px) count++;
  }
  return count;
}

/**
 * Whether a shell lies inside a closed container: up to 7 of its triangles' centroids, each nudged off any
 * axis-aligned edge by an irrational fraction of the diagonal, cast along +x; inside when most cross an odd number
 * of times. A shell that touches its container's skin has some centroids on it; the majority decides.
 */
function inside(T, V, tris, grid, diag) {
  const samples = Math.min(7, tris.length);
  let hits = 0;
  for (let i = 0; i < samples; i++) {
    const t = tris[Math.floor((i * tris.length) / samples)];
    const a = T[t * 3] * 3, b = T[t * 3 + 1] * 3, c = T[t * 3 + 2] * 3;
    const px = (V[a] + V[b] + V[c]) / 3;
    const py = (V[a + 1] + V[b + 1] + V[c + 1]) / 3 + diag * 1.4142135e-7 * (i + 1);
    const pz = (V[a + 2] + V[b + 2] + V[c + 2]) / 3 + diag * 1.7320508e-7 * (i + 1);
    if (crossings(T, V, grid, px, py, pz) % 2 === 1) hits++;
  }
  return hits * 2 > samples;
}

function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
