// Test parts for the export by faces, each face labelled: box (four-cornered faces), cylinder (cornerless caps and
// a ring), lobes (three two-cornered faces), plate (faces with a hole, a ring), tube (rings). mesh.json per part:
// { vertices, faces, labels }, for test/freeformGroups.mjs. node test/groupParts.mjs <out dir>
import fs from 'node:fs';
import path from 'node:path';
const out = process.argv[2];

function mesh() {
  const V = [], F = [], Lb = [], key = new Map();
  const v = (p) => { const k = p.map((x) => (Math.round(x * 1e9) / 1e9 + 0).toFixed(9)).join(","); if (!key.has(k)) { key.set(k, V.length); V.push(p); } return key.get(k); };
  const tri = (a, b, c, l) => { const A = v(a), B = v(b), C = v(c); if (A !== B && B !== C && A !== C) { F.push([A, B, C]); Lb.push(l); } };
  const quad = (a, b, c, d, l) => { tri(a, b, c, l); tri(a, c, d, l); };
  return { V, F, Lb, tri, quad, save(name) { fs.writeFileSync(path.join(out, name + '.json'), JSON.stringify({ vertices: V, faces: F, labels: Lb })); console.log(name, F.length, 'triangles', new Set(Lb).size, 'faces'); } };
}
const TAU = 2 * Math.PI;

// box [-1, 1]^3, m x m quads a face
{
  const M = mesh(), m = 8;
  const faces = [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[-1, 0, 0], [0, 0, 1], [0, 1, 0]], [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
    [[0, -1, 0], [1, 0, 0], [0, 0, 1]], [[0, 0, 1], [1, 0, 0], [0, 1, 0]], [[0, 0, -1], [0, 1, 0], [1, 0, 0]]];
  faces.forEach(([n, u, w], l) => {
    const P = (i, j) => [0, 1, 2].map((d) => n[d] + u[d] * (2 * i / m - 1) + w[d] * (2 * j / m - 1));
    for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) M.quad(P(i, j), P(i + 1, j), P(i + 1, j + 1), P(i, j + 1), l);
  });
  M.save('box');
}
// cylinder r 1, z in [-1, 1]
{
  const M = mesh(), nt = 48, nz = 8, nr = 6;
  const P = (t, z, r = 1) => [r * Math.cos(TAU * t / nt), r * Math.sin(TAU * t / nt), z];
  for (let i = 0; i < nt; i++) for (let j = 0; j < nz; j++) M.quad(P(i, -1 + 2 * j / nz), P(i + 1, -1 + 2 * j / nz), P(i + 1, -1 + 2 * (j + 1) / nz), P(i, -1 + 2 * (j + 1) / nz), 0);
  for (let i = 0; i < nt; i++) for (let k = 0; k < nr; k++) {
    const r0 = k / nr, r1 = (k + 1) / nr;
    M.quad(P(i, 1, r0), P(i, 1, r1), P(i + 1, 1, r1), P(i + 1, 1, r0), 1);
    M.quad(P(i, -1, r0), P(i + 1, -1, r0), P(i + 1, -1, r1), P(i, -1, r1), 2);
  }
  M.save('cylinder');
}
// three-lobed sphere: three lunes meeting along meridians at a crease, poles shared by all three
{
  const M = mesh(), nt = 72, np = 36;
  const P = (t, p) => {
    const phi = TAU * t / nt, th = Math.PI * p / np, frac = ((phi % (TAU / 3)) + TAU / 3) % (TAU / 3) / (TAU / 3);
    const r = 1 + 0.35 * Math.sin(th) * Math.sin(Math.PI * frac);
    return [r * Math.sin(th) * Math.cos(phi), r * Math.sin(th) * Math.sin(phi), Math.cos(th)];
  };
  for (let i = 0; i < nt; i++) for (let j = 0; j < np; j++) M.quad(P(i, j), P(i, j + 1), P(i + 1, j + 1), P(i + 1, j), Math.floor(i / (nt / 3)));
  M.save('lobes');
}
// plate 4 x 4 x 1 with a round hole of radius 1 through it
{
  const M = mesh(), nt = 48, nr = 4, nz = 4;
  const ang = (i) => TAU * i / nt;
  const edgePt = (i) => { const c = Math.cos(ang(i)), s = Math.sin(ang(i)), t = 2 / Math.max(Math.abs(c), Math.abs(s)); return [t * c, t * s]; };
  const hole = (i) => [Math.cos(ang(i)), Math.sin(ang(i))];
  for (const [z, l, up] of [[0.5, 0, true], [-0.5, 1, false]]) {
    for (let i = 0; i < nt; i++) for (let k = 0; k < nr; k++) {
      const a = (ii, kk) => { const h = hole(ii), e = edgePt(ii), w = kk / nr; return [h[0] + w * (e[0] - h[0]), h[1] + w * (e[1] - h[1]), z]; };
      if (up) M.quad(a(i, k), a(i, k + 1), a(i + 1, k + 1), a(i + 1, k), l);
      else M.quad(a(i, k), a(i + 1, k), a(i + 1, k + 1), a(i, k + 1), l);
    }
  }
  // outer sides: one face per side of the square, columns at the top face's border points
  for (let i = 0; i < nt; i++) {
    const e0 = edgePt(i), e1 = edgePt(i + 1), side = Math.floor(((ang(i) + ang(i + 1)) / 2 + Math.PI / 4) / (Math.PI / 2)) % 4;
    for (let j = 0; j < nz; j++) {
      const z0 = -0.5 + j / nz, z1 = -0.5 + (j + 1) / nz;
      M.quad([e0[0], e0[1], z0], [e1[0], e1[1], z0], [e1[0], e1[1], z1], [e0[0], e0[1], z1], 2 + side);
      const h0 = hole(i), h1 = hole(i + 1);
      M.quad([h0[0], h0[1], z0], [h0[0], h0[1], z1], [h1[0], h1[1], z1], [h1[0], h1[1], z0], 6);
    }
  }
  M.save('plate');
}
// tube: outer r 1.5, inner r 1, z in [0, 2]
{
  const M = mesh(), nt = 48, nz = 6, nr = 3;
  const P = (i, r, z) => [r * Math.cos(TAU * i / nt), r * Math.sin(TAU * i / nt), z];
  for (let i = 0; i < nt; i++) {
    for (let j = 0; j < nz; j++) {
      const z0 = 2 * j / nz, z1 = 2 * (j + 1) / nz;
      M.quad(P(i, 1.5, z0), P(i + 1, 1.5, z0), P(i + 1, 1.5, z1), P(i, 1.5, z1), 0);
      M.quad(P(i, 1, z0), P(i, 1, z1), P(i + 1, 1, z1), P(i + 1, 1, z0), 1);
    }
    for (let k = 0; k < nr; k++) {
      const r0 = 1 + 0.5 * k / nr, r1 = 1 + 0.5 * (k + 1) / nr;
      M.quad(P(i, r0, 2), P(i, r1, 2), P(i + 1, r1, 2), P(i + 1, r0, 2), 2);
      M.quad(P(i, r0, 0), P(i + 1, r0, 0), P(i + 1, r1, 0), P(i, r1, 0), 3);
    }
  }
  M.save('tube');
}
