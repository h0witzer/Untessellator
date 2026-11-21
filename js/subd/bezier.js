/**
 * bezier.js — Bezier patch evaluation without allocation: the Bernstein basis with two derivatives, and a square
 * patch's point and derivatives up to order 2. Shared by the fit (footpoints) and the preview.
 */

const BIN = [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1], [1, 4, 6, 4, 1], [1, 5, 10, 10, 5, 1], [1, 6, 15, 20, 15, 6, 1]];
const TP = new Float64Array(8), SP = new Float64Array(8);

/** Bernstein basis of degree p (<= 6) at t, and its first and second derivatives, into b0, b1, b2 (b1, b2 optional). */
export function bern(p, t, b0, b1, b2) {
  const s = 1 - t;
  TP[0] = 1; SP[0] = 1;
  for (let k = 1; k <= p; k++) { TP[k] = TP[k - 1] * t; SP[k] = SP[k - 1] * s; }
  const c1 = (i) => (i < 0 || i > p - 1 ? 0 : BIN[p - 1][i] * TP[i] * SP[p - 1 - i]);
  const c2 = (i) => (i < 0 || i > p - 2 ? 0 : BIN[p - 2][i] * TP[i] * SP[p - 2 - i]);
  for (let i = 0; i <= p; i++) {
    b0[i] = BIN[p][i] * TP[i] * SP[p - i];
    if (b1) b1[i] = p * (c1(i - 1) - c1(i));
    if (b2) b2[i] = p * (p - 1) * (c2(i - 2) - 2 * c2(i - 1) + c2(i));
  }
}

const BU = [new Float64Array(7), new Float64Array(7), new Float64Array(7)];
const BV = [new Float64Array(7), new Float64Array(7), new Float64Array(7)];

/** S, Su, Sv, Suu, Suv, Svv (3 each) of the last evalAt; overwritten by the next call. */
export const D = new Float64Array(18);

/** Point and derivatives up to order 2 of an n x n Bezier patch (poles P, row-major u then v, 3 each) at (u, v), into D. */
export function evalAt(n, P, u, v) {
  const p = n - 1;
  bern(p, u, BU[0], BU[1], BU[2]); bern(p, v, BV[0], BV[1], BV[2]);
  D.fill(0);
  for (let i = 0; i < n; i++) {
    let r0x = 0, r0y = 0, r0z = 0, r1x = 0, r1y = 0, r1z = 0, r2x = 0, r2y = 0, r2z = 0;
    for (let j = 0; j < n; j++) {
      const o = (i * n + j) * 3, x = P[o], y = P[o + 1], z = P[o + 2], a = BV[0][j], b = BV[1][j], c = BV[2][j];
      r0x += a * x; r0y += a * y; r0z += a * z;
      r1x += b * x; r1y += b * y; r1z += b * z;
      r2x += c * x; r2y += c * y; r2z += c * z;
    }
    const a = BU[0][i], b = BU[1][i], c = BU[2][i];
    D[0] += a * r0x; D[1] += a * r0y; D[2] += a * r0z;
    D[3] += b * r0x; D[4] += b * r0y; D[5] += b * r0z;
    D[6] += a * r1x; D[7] += a * r1y; D[8] += a * r1z;
    D[9] += c * r0x; D[10] += c * r0y; D[11] += c * r0z;
    D[12] += b * r1x; D[13] += b * r1y; D[14] += b * r1z;
    D[15] += a * r2x; D[16] += a * r2y; D[17] += a * r2z;
  }
  return D;
}
