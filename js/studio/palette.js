/**
 * palette.js — the surface types' colors, for coloring groups by type.
 *
 * Chosen so the types stay apart for color-blind viewers too. Under simulated protanopia, deuteranopia and
 * tritanopia (Machado, Oliveira and Fernandes 2009), every pair of types, and every type against unpainted gray, the
 * kit's selected and pre-highlight colors and the back-face red, is at least 9 OKLab units (x100) apart; with normal
 * vision, at least 15. test/palette.mjs checks it. The types differ in lightness as well as hue, because lightness
 * is what color-blind vision keeps. Planes are the quiet one, since they cover most of a part, and skip is a dark
 * gray: it does not export.
 *
 * Found by a max-min search over OKLCH (seeds agree; 2026-10-31). A hand-picked palette before it measured
 * sphere/freeform 1.3 and plane/cone 2.1 under deuteranopia, and its sphere sat 7.7 from the selection amber.
 */

export const TYPE_HEX = Object.freeze({
  plane: '#b0837e',
  cylinder: '#76830e',
  cone: '#4ffbee',
  sphere: '#3fa6d0',
  torus: '#ba469f',
  freeform: '#646ae1',
  skip: '#595f5d',
});

/** A type's color as [r, g, b] in 0..1 (untyped groups are freeform). */
export function typeRgb(type) {
  const hex = TYPE_HEX[type ?? 'freeform'] ?? TYPE_HEX.freeform;
  return [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
}
