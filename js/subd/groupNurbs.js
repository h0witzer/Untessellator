/**
 * groupNurbs.js — the conversion of js/subd/fastNurbs.js for a cage made of groups (a part's faces): inside a group
 * the surface is G2 (bicubic faces, bi-6 caps at star points), and a group's border is a sharp edge whose curve is
 * the cubic B-spline of the border's cage vertices, so the two groups on either side of a border share that curve
 * exactly: the solid is watertight with no trimming.
 *
 * Each group is extended past its border by mirrored ghost points: a border vertex b with inner neighbour i gets
 * the ghost 2b - i, and a group corner c (on one quad [c, b, y, a]) also gets the corner ghost 4c - 2a - 2b + y.
 * With them every quad next to the border is a regular B-spline face whose border row is exactly the B-spline of
 * the border vertices (the Catmull-Clark boundary rule; corners are interpolated), and the ghosts are fixed linear
 * combinations of real vertices, so the conversion stays one linear map on the cage.
 *
 * A group's border vertices must be on two of its quads, or on one (a corner); groups from groupCage.js are.
 */
import { topology, regularGrid, capFaces, B2BB, bezierKnots } from './fastNurbs.js';

const UV = [[0, 0], [1, 0], [1, 1], [0, 1]];
const R16 = (() => {
  const R = new Float64Array(256);
  for (let a = 0; a < 4; a++) for (let b = 0; b < 4; b++) for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) R[(a * 4 + b) * 16 + i * 4 + j] = B2BB[a][i] * B2BB[b][j];
  return R;
})();

/**
 * groups: [{ faces: [[a, b, c, d]...] }] on cage vertex ids 0..nv-1, quads wound the same way in every group.
 * Returns { faces: [{ uCount, vCount, ids, W }], loops: [[[vertex, u, v] x 4]...], faceGroup: Int32Array } with
 * pole p of face f = sum over a of W[p * ids.length + a] * vertex ids[a] (as subdLinearMap gives it).
 */
export function groupLinearMap(nv, groups, options = {}) {
  const faces = [], loops = [], faceGroup = [];
  groups.forEach((group, gi) => {
    // local numbering: the group's vertices, then its ghosts; each local id is a stencil on cage vertices
    const local = new Map(), stencil = [], globalOf = [];
    const real = (v) => {
      let i = local.get(v);
      if (i === undefined) { i = stencil.length; local.set(v, i); stencil.push([[v, 1]]); globalOf.push(v); }
      return i;
    };
    const F = group.faces.map((q) => q.map(real));
    const nReal = stencil.length, nFaces = F.length;
    const combine = (terms) => {
      const m = new Map();
      for (const [l, w] of terms) for (const [v, x] of stencil[l]) m.set(v, (m.get(v) ?? 0) + w * x);
      return [...m];
    };
    const addPoint = (terms) => { stencil.push(combine(terms)); globalOf.push(-1); return stencil.length - 1; };
    const ghosts = new Map();
    const ghost = (v, x) => {                       // v mirrored through itself away from x: 2v - x
      const key = `${v}|${x}`;
      let i = ghosts.get(key);
      if (i === undefined) { i = addPoint([[v, 2], [x, -1]]); ghosts.set(key, i); }
      return i;
    };
    const owner = new Set(), onQuads = new Int32Array(nReal);
    F.forEach((q) => q.forEach((a, k) => { owner.add(`${a},${q[(k + 1) % 4]}`); onQuads[a]++; }));
    const extra = [], onBorder = new Uint8Array(nReal);
    F.forEach((q) => q.forEach((a, k) => {
      const b = q[(k + 1) % 4];
      if (owner.has(`${b},${a}`)) return;
      onBorder[a] = onBorder[b] = 1;
      extra.push([b, a, ghost(a, q[(k + 3) % 4]), ghost(b, q[(k + 2) % 4])]);
    }));
    for (let v = 0; v < nReal; v++) {
      if (!onBorder[v] || onQuads[v] === 2) continue;
      if (onQuads[v] !== 1) throw new Error(`group ${gi}: a border vertex on ${onQuads[v]} quads (only 1 or 2 are supported)`);
      const f = F.findIndex((q) => q.includes(v)), q = F[f], k = q.indexOf(v);
      const b = q[(k + 1) % 4], y = q[(k + 2) % 4], a = q[(k + 3) % 4];
      const corner = addPoint([[v, 4], [b, -2], [a, -2], [y, 1]]);
      extra.push([v, ghost(v, b), corner, ghost(v, a)]);
    }
    const FF = F.concat(extra), T = topology(stencil.length, FF);
    // pole weights on local ids -> on cage vertices
    const compose = (ids, M, np) => {
      const col = new Map();
      for (const l of ids) for (const [v] of stencil[l]) if (!col.has(v)) col.set(v, col.size);
      const nc = col.size, nl = ids.length, W = new Float64Array(np * nc);
      for (let a = 0; a < nl; a++) {
        const r = stencil[ids[a]].map(([v, w]) => [col.get(v), w]);
        for (let p = 0; p < np; p++) { const m = M[p * nl + a]; if (m) for (const [c, w] of r) W[p * nc + c] += m * w; }
      }
      return { ids: Int32Array.from(col.keys()), W };
    };
    const caps = new Map();
    const capOf = (v) => {
      if (caps.has(v)) return caps.get(v);
      const near = new Set();
      for (const [f] of T.facesAt[v]) for (const w of FF[f]) for (const [g] of T.facesAt[w]) for (const x of FF[g]) near.add(x);
      const L = [...near], dim = L.length, Vl = new Array(stencil.length);
      L.forEach((x, a) => { const e = new Float64Array(dim); e[a] = 1; Vl[x] = e; });
      const out = { L, faces: new Map(capFaces(Vl, FF, T, v, dim, options.variant)) };
      caps.set(v, out);
      return out;
    };
    for (let f = 0; f < nFaces; f++) {
      const q = FF[f], stars = q.filter((v) => T.valence[v] !== 4);
      if (stars.length > 1) throw new Error(`group ${gi}: a face with ${stars.length} star corners`);
      if (stars.length) {
        const c = capOf(stars[0]);
        faces.push({ uCount: 7, vCount: 7, ...compose(c.L, c.faces.get(f), 49) });
      } else {
        const G = regularGrid(q, T), uniq = [...new Set(G)];
        let M = R16;
        if (uniq.length !== 16) {
          M = new Float64Array(16 * uniq.length);
          const at = new Map(uniq.map((v, a) => [v, a]));
          for (let p = 0; p < 16; p++) for (let g = 0; g < 16; g++) M[p * uniq.length + at.get(G[g])] += R16[p * 16 + g];
        }
        faces.push({ uCount: 4, vCount: 4, ...compose(uniq, M, 16) });
      }
      loops.push(q.map((v, k) => [globalOf[v], ...UV[k]]));
      faceGroup.push(gi);
    }
  });
  return { faces, loops, faceGroup: Int32Array.from(faceGroup), subdivisions: 0, bezierKnots };
}
