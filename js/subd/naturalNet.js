/**
 * naturalNet.js — the surface the star faces are fitted to in NURBS Proto's method (from the clean-room
 * specification, section 3.1): the cage subdivided twice and read, face by face, as a uniform bicubic B-spline
 * over the face's 5 x 5 grid of level-2 vertices plus one ring of ghost points, with each star corner moved to
 * its exact limit point. It equals the limit surface everywhere except the single level-2 cell at each star
 * corner, where it is the bicubic B-spline of the irregular control points: smoother than Catmull-Clark there.
 *
 * Written from the specification only (no source read). The corner value is replaced through the clamped
 * corner basis function (1 - 4d)^3, which is what overwriting the clamped net's corner control point does.
 */
import { subdivide, limitPositions } from './catmullClark.js';

/** Neighbours of every fine vertex in counter-clockwise rotation order (faces are counter-clockwise loops). */
function rotations(nv, F) {
  const nextToPrev = Array.from({ length: nv }, () => new Map());
  for (let g = 0; g < F.length / 4; g++) {
    for (let k = 0; k < 4; k++) {
      const v = F[g * 4 + k], next = F[g * 4 + ((k + 1) & 3)], prev = F[g * 4 + ((k + 3) & 3)];
      nextToPrev[v].set(next, prev);
    }
  }
  return (v) => {
    const m = nextToPrev[v], start = m.keys().next().value, ring = [start];
    for (let x = m.get(start); x !== start; x = m.get(x)) ring.push(x);
    return ring;
  };
}

/**
 * Ghost points beyond a corner c (specification 3.1a). eU: c's neighbour along the face side in u, eV: along v;
 * they are adjacent in c's rotation. The ghost continuing the u side past c is the neighbour next to eV on the
 * side away from eU; the ghost continuing the v side is the neighbour next to eU away from eV. At valence 4 these
 * are the true neighbouring-face vertices, and the diagonal is the vertex opposite c in the face through both;
 * at a star corner the diagonal only reaches the corner control point, which is replaced, so c stands in.
 */
function cornerGhosts(ring, c, eU, eV, F, faceOf) {
  const n = ring.length, iu = ring.indexOf(eU), iv = ring.indexOf(eV);
  const awayFrom = (i, other) => (ring[(i + 1) % n] === other ? ring[(i - 1 + n) % n] : ring[(i + 1) % n]);
  const beyondU = awayFrom(iv, eU), beyondV = awayFrom(iu, eV);
  return { beyondU, beyondV, diagonal: n === 4 ? faceOf(c, beyondU, beyondV) : c };
}

/**
 * Samplers of the natural net of every face. Returns { sample(f, u, v) -> [x, y, z] }, exact limit points at
 * the corners. faces: the cage's quads; valence: per cage vertex.
 */
export function naturalNets(vertices, faces) {
  const { V, F, ids, N } = subdivide(vertices, faces, 2);
  const nv = V.length / 3;
  const ring = rotations(nv, F);
  const L = limitPositions(V, F);
  const faceKey = new Map();                     // the fourth vertex of the fine face through c, a, b
  for (let g = 0; g < F.length / 4; g++) {
    const q = [0, 1, 2, 3].map((k) => F[g * 4 + k]);
    for (let k = 0; k < 4; k++) faceKey.set(`${q[k]},${q[(k + 1) & 3]},${q[(k + 3) & 3]}`, q[(k + 2) & 3]);
  }
  const faceOf = (c, a, b) => faceKey.get(`${c},${a},${b}`) ?? faceKey.get(`${c},${b},${a}`);
  const P = (id) => [V[id * 3], V[id * 3 + 1], V[id * 3 + 2]];
  const B = (x) => [(1 - x) ** 3 / 6, (3 * x ** 3 - 6 * x * x + 4) / 6, (-3 * x ** 3 + 3 * x * x + 3 * x + 1) / 6, x ** 3 / 6];
  const spans = N - 1;
  const uniform = (E, u, v) => {
    const ci = Math.min(spans - 1, Math.floor(u * spans)), cj = Math.min(spans - 1, Math.floor(v * spans));
    const bu = B(u * spans - ci), bv = B(v * spans - cj), out = [0, 0, 0];
    for (let a = 0; a < 4; a++) for (let b = 0; b < 4; b++) {
      const p = E[ci + a][cj + b], w = bu[a] * bv[b];
      out[0] += w * p[0]; out[1] += w * p[1]; out[2] += w * p[2];
    }
    return out;
  };
  const build = (G) => {
    const at = (i, j) => G[i * N + j];
    const E = Array.from({ length: N + 2 }, () => new Array(N + 2));   // extended grid, index + 1
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) E[i + 1][j + 1] = P(at(i, j));
    // side ghosts: the fourth neighbour of a side vertex, across the side
    const across = (p, a, b, c) => ring(p).find((x) => x !== a && x !== b && x !== c);
    for (let k = 1; k < N - 1; k++) {
      E[k + 1][0] = P(across(at(k, 0), at(k - 1, 0), at(k + 1, 0), at(k, 1)));
      E[k + 1][N + 1] = P(across(at(k, N - 1), at(k - 1, N - 1), at(k + 1, N - 1), at(k, N - 2)));
      E[0][k + 1] = P(across(at(0, k), at(0, k - 1), at(0, k + 1), at(1, k)));
      E[N + 1][k + 1] = P(across(at(N - 1, k), at(N - 1, k - 1), at(N - 1, k + 1), at(N - 2, k)));
    }
    const corners = [];
    for (const [ci, cj] of [[0, 0], [N - 1, 0], [N - 1, N - 1], [0, N - 1]]) {
      const du = ci === 0 ? 1 : -1, dv = cj === 0 ? 1 : -1;
      const c = at(ci, cj), r = ring(c);
      const g = cornerGhosts(r, c, at(ci + du, cj), at(ci, cj + dv), F, faceOf);
      E[ci + 1 - du][cj + 1] = P(g.beyondU);
      E[ci + 1][cj + 1 - dv] = P(g.beyondV);
      E[ci + 1 - du][cj + 1 - dv] = P(g.diagonal ?? c);
      corners.push({ u: ci / (N - 1), v: cj / (N - 1), irregular: r.length !== 4, limit: [L[c * 3], L[c * 3 + 1], L[c * 3 + 2]] });
    }
    for (const k of corners) k.spline = uniform(E, k.u, k.v);
    return { E, corners };
  };
  const nets = new Array(ids.length);
  const netOf = (f) => nets[f] ?? (nets[f] = build(ids[f]));
  return {
    sample(f, u, v) {
      const net = netOf(f), x = uniform(net.E, u, v);
      for (const k of net.corners) {
        if (!k.irregular) continue;
        const du = Math.abs(u - k.u) * spans, dv = Math.abs(v - k.v) * spans;
        if (du >= 1 || dv >= 1) continue;
        const w = (1 - du) ** 3 * (1 - dv) ** 3;
        for (let c = 0; c < 3; c++) x[c] += w * (k.limit[c] - k.spline[c]);
      }
      return x;
    },
  };
}
