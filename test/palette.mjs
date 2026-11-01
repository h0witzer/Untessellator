/**
 * palette.mjs — js/studio/palette.js's colorblind palette checked for color-blind viewers.
 *
 * Usage:  node test/palette.mjs
 *
 * Simulates protanopia, deuteranopia and tritanopia at full severity (Machado, Oliveira and Fernandes 2009, applied
 * to linear sRGB) and measures every pair of type colors, and every type against the colors it must not be
 * mistaken for, as OKLab distance x100 (about 2 is barely visible). Fails (exit 1) when a pair of COLORBLIND_HEX is
 * closer than NORMAL_MIN with normal vision or CVD_MIN under any simulation, side by side, or closer than LIT_MIN as
 * lit: the kit's headlight scales each color by 0.5 to 1, so a pair is measured at its closest over every two shades
 * (skip left out: any two lit grays meet; its triangle edges mark it). That palette measured 15.0, 9.1 and 8.9.
 * The default, TYPE_HEX (the logo's filament colors, chosen for the brand on 2026-10-31), is measured and printed
 * the same way but not held to the floors; Colorblind colors in the view menu is what the floors guarantee.
 */

import { TYPE_HEX, COLORBLIND_HEX } from '../js/studio/palette.js';

const NORMAL_MIN = 15;
const CVD_MIN = 9;
const LIT_MIN = 8.5;
// the headlight's range (ambient 0.5 + 0.5 N.L, vendor/form-studio-ui/viewport.js PART_GLSL), on the sRGB values
const SHADES = [0.5, 0.6, 0.7, 0.8, 0.9, 1];
// lit like the types: unpainted triangles; the back face is drawn flat, and selection and hover are border lines
const LIT_OTHERS = new Set(['unpainted']);
// What a type color must not be mistaken for: unpainted triangles (studio.js UNPAINTED), the kit's selected and
// pre-highlight colors (vendor/form-studio-ui/viewport.js HIGHLIGHT_STYLE) and an inverted patch (renderer.js).
const OTHERS = { unpainted: '#bdbdbd', selected: '#ffa90a', preHighlight: '#ffc685', backFace: '#73141a' };

const SIMULATIONS = {
  normal: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
  protanopia: [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],
  deuteranopia: [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]],
  tritanopia: [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]],
};

const toLinear = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const linear = (hex, shade = 1) => [1, 3, 5].map((i) => toLinear(shade * parseInt(hex.slice(i, i + 2), 16) / 255));
const clamp = (v) => Math.min(1, Math.max(0, v));
const simulate = (rgb, m) => m.map((row) => clamp(row[0] * rgb[0] + row[1] * rgb[1] + row[2] * rgb[2]));
function oklab([r, g, b]) {
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s, 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s];
}
const distance = (p, q) => 100 * Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);

const lab = (hex, shade) => oklab(linear(hex, shade));

/** Prints a palette's closest pairs in each measure; returns how many fall below the floors when it is held to them. */
function check(name, palette, held) {
  const types = Object.keys(palette);
  const pairs = [];
  for (let i = 0; i < types.length; i++) {
    for (let j = i + 1; j < types.length; j++) pairs.push([types[i], palette[types[i]], types[j], palette[types[j]], true]);
    for (const [other, hex] of Object.entries(OTHERS)) pairs.push([types[i], palette[types[i]], other, hex, LIT_OTHERS.has(other)]);
  }
  console.log(held ? name : `${name} (measured, not held to the floors)`);
  let failed = 0;
  const report = (label, measured, min) => {
    const bad = measured.filter(([d]) => d < min);
    if (held) failed += bad.length;
    console.log(`  ${label.padEnd(12)} closest: ${measured.slice(0, 3).map(([d, p]) => `${p} ${d.toFixed(1)}`).join(', ')}` +
      (!held ? '' : bad.length ? `  FAIL (< ${min}): ${bad.map(([d, p]) => `${p} ${d.toFixed(1)}`).join(', ')}` : `  ok (>= ${min})`));
  };
  for (const [vision, matrix] of Object.entries(SIMULATIONS)) {
    report(vision, pairs.map(([a, ha, b, hb]) => [distance(oklab(simulate(linear(ha), matrix)), oklab(simulate(linear(hb), matrix))), `${a}/${b}`])
      .sort((x, y) => x[0] - y[0]), vision === 'normal' ? NORMAL_MIN : CVD_MIN);
  }
  // as lit, normal vision: the closest the two get over every pair of shades
  report('lit', pairs.filter(([a, , b]) => a !== 'skip' && b !== 'skip').map(([a, ha, b, hb, bothLit]) => {
    let d = Infinity;
    for (const s of SHADES) for (const t of bothLit ? SHADES : [1]) d = Math.min(d, distance(lab(ha, s), lab(hb, t)));
    return [d, `${a}/${b}`];
  }).sort((x, y) => x[0] - y[0]), LIT_MIN);
  return failed;
}

check('default: filament', TYPE_HEX, false);
const failed = check('colorblind', COLORBLIND_HEX, true);
if (failed) { console.log(`${failed} pairs too close`); process.exit(1); }
console.log('palette ok');
