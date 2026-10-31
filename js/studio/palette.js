/**
 * palette.js — the surface types' colors, for coloring groups by type.
 *
 * Chosen so the types stay apart AS LIT. The kit's headlight multiplies each color by 0.5 to 1 (ambient 0.5 + 0.5
 * N.L), so a light color in shadow can match a dark one in full light: the palette before this one kept its types
 * apart by lightness, and its cone and sphere, 23 apart side by side, came within 5.9 of each other lit. Here every
 * pair of types, and every type against unpainted gray (lit too), the back-face red and the kit's selected and
 * pre-highlight ambers, stays at least 8.5 OKLab units (x100) apart at any two shades; cone and sphere 30. Skip is
 * left out of that measure, since any two lit grays meet somewhere; its triangle edges, drawn over it, mark it.
 * Side by side, with normal vision at least 15 apart, and under simulated protanopia, deuteranopia and tritanopia
 * (Machado, Oliveira and Fernandes 2009) at least 9. test/palette.mjs checks all three.
 *
 * Found by a max-min search over OKLCH with the plane held to blues (two seeds of eight agree; 2026-10-31); leaving
 * the plane free gained nothing (8.8). The colorblind floor costs under 1: without it the search reached 9.0.
 * Okabe-Ito measured 4.5 lit (sky blue against blue) and its orange 5.0 from the selection amber.
 */

export const TYPE_HEX = Object.freeze({
  plane: '#45a9f1',
  cylinder: '#249b02',
  cone: '#8b56fe',
  sphere: '#e6f028',
  torus: '#df14b3',
  freeform: '#fe7a77',
  skip: '#4d5255',
});

/** A type's color as [r, g, b] in 0..1 (untyped groups are freeform). */
export function typeRgb(type) {
  const hex = TYPE_HEX[type ?? 'freeform'] ?? TYPE_HEX.freeform;
  return [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
}
