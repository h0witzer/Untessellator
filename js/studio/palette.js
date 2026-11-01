/**
 * palette.js — the surface types' colors, for coloring groups by type.
 *
 * TYPE_HEX, the default, is the filament palette of the logo (2026-10-31): the mark's left half is these colors,
 * so the studio paints a part the way its logo looks. Measured as test/palette.mjs measures (OKLab x100) against
 * each other, unpainted gray and the back-face red, with normal vision: 14.2 side by side (torus and freeform) and
 * 6.3 lit (a plane in shade against a cone). The sphere and freeform colors come within 7.1 and 10.7 of the
 * hover and selection ambers, which the viewport draws as outlines on a face, never as a fill. Under simulated color
 * blindness types meet (deuteranopia: cylinder and torus 3.2), so a color-blind viewer turns on Colorblind colors
 * in the view menu.
 *
 * COLORBLIND_HEX was chosen so the types stay apart AS LIT. The kit's headlight multiplies each color by 0.5 to 1
 * (ambient 0.5 + 0.5 N.L), so a light color in shadow can match a dark one in full light: the palette before this one
 * kept its types apart by lightness, and its cone and sphere, 23 apart side by side, came within 5.9 of each other
 * lit. Here every pair of types, and every type against unpainted gray (lit too), the back-face red and the kit's
 * selected and pre-highlight ambers, stays at least 8.5 OKLab units (x100) apart at any two shades; cone and sphere
 * 30. Skip is left out of that measure, since any two lit grays meet somewhere; its triangle edges, drawn over it,
 * mark it. Side by side, with normal vision at least 15 apart, and under simulated protanopia, deuteranopia and
 * tritanopia (Machado, Oliveira and Fernandes 2009) at least 9. test/palette.mjs checks all three.
 *
 * Found by a max-min search over OKLCH with the plane held to blues (two seeds of eight agree; 2026-10-31); leaving
 * the plane free gained nothing (8.8). The colorblind floor costs under 1: without it the search reached 9.0.
 * Okabe-Ito measured 4.5 lit (sky blue against blue) and its orange 5.0 from the selection amber.
 */

export const TYPE_HEX = Object.freeze({
  plane: '#b98cff',
  cylinder: '#2fc59a',
  cone: '#8a4fe0',
  sphere: '#ffc933',
  torus: '#ff5c8a',
  freeform: '#ff7a1a',
  skip: '#4d5255',
});

export const COLORBLIND_HEX = Object.freeze({
  plane: '#45a9f1',
  cylinder: '#249b02',
  cone: '#8b56fe',
  sphere: '#e6f028',
  torus: '#df14b3',
  freeform: '#fe7a77',
  skip: '#4d5255',
});

/** A type's color as [r, g, b] in 0..1 (untyped groups are freeform), from the colorblind palette when asked. */
export function typeRgb(type, colorblind = false) {
  const palette = colorblind ? COLORBLIND_HEX : TYPE_HEX;
  const hex = palette[type ?? 'freeform'] ?? palette.freeform;
  return [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
}
