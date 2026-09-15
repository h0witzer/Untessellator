// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Filled icons on a 20 grid, drawn to Onshape's measured conventions.
//
// WHY THESE ARE OURS AND NOT THEIRS. Onshape's icons are their artwork and
// their property; copying the path data out of their sprite would be taking it.
// What is reusable is the CONVENTION, and that was measured
// (`tools/style-probe.mjs`, 2026-09-11):
//
//     viewBox      0 0 20 20      82 of 91 symbols
//     rendered     20 x 20        one to one, never scaled
//     paint        FILLED         201 of 201 shapes; not one stroked
//     colour       currentColor   the button's own colour, so state is free
//
// That last line about paint is the important one. Feather, Lucide, Material
// Symbols Outlined, Heroicons Outline, Phosphor and essentially every popular
// set is STROKED at 1.5 or 2 units on a 24 grid. Dropping one of those into a
// toolbar beside Onshape's reads as a different application instantly, at any
// colour, in either theme, because the weight of the marks is wrong and the
// optical size is wrong with it. These are drawn filled on the 20 grid so they
// sit at the same weight.
//
// HOUSE RULES FOR ADDING ONE
//
//   * 20 x 20 viewBox, and draw to the edges — Onshape's marks are optically
//     large and do not sit in a polite 2px margin.
//   * Fill, never stroke. Where a line is wanted, draw it as a thin filled
//     rectangle, 1.5 to 2 units wide; that is what their sprite does.
//   * No round caps, no joins to tune: a filled shape has neither, which is
//     part of why the set reads as one family.
//   * One shape where possible. A `fill-rule="evenodd"` hole is cheaper than a
//     second path and keeps `currentColor` working.

/**
 * The set. Each entry is the INNER markup of a `0 0 20 20` svg.
 *
 * Named for what the tool does, not for what the picture is, so a caller does
 * not have to know that Subdivide happens to be drawn as a divided square.
 */
export const ICONS = Object.freeze({
  // --- primitives -------------------------------------------------------
  box: '<path d="M10 1 18.5 5.5v9L10 19 1.5 14.5v-9z" fill-rule="evenodd" />',
  sphere:
    '<path d="M10 1a9 9 0 1 0 0 18 9 9 0 0 0 0-18m0 2c1.2 0 2.8 2.6 2.8 7S11.2 17 10 17 7.2 14.4 7.2 10 8.8 3 10 3" fill-rule="evenodd" />',
  cylinder:
    '<path d="M10 1c3.9 0 7 1.3 7 3v12c0 1.7-3.1 3-7 3s-7-1.3-7-3V4c0-1.7 3.1-3 7-3m0 1.6c-3 0-5.4.8-5.4 1.4S7 5.4 10 5.4 15.4 4.6 15.4 4 13 2.6 10 2.6" fill-rule="evenodd" />',
  torus:
    '<path d="M10 3c5 0 9 3.1 9 7s-4 7-9 7-9-3.1-9-7 4-7 9-7m0 4.6c-2.9 0-4.8 1.5-4.8 2.4S7.1 12.4 10 12.4 14.8 10.9 14.8 10 12.9 7.6 10 7.6" fill-rule="evenodd" />',
  face: '<path d="M1.5 4.5h17v11h-17z" fill-rule="evenodd" />',
  tube: '<path d="M10 1c4.4 0 8 1.3 8 3v12c0 1.7-3.6 3-8 3s-8-1.3-8-3V4c0-1.7 3.6-3 8-3m0 1.7c-3.5 0-6.3.7-6.3 1.3S6.5 5.3 10 5.3 16.3 4.6 16.3 4 13.5 2.7 10 2.7m0 2.8c2 0 3.6.5 3.6 1.1S12 7.7 10 7.7 6.4 7.2 6.4 6.6 8 5.5 10 5.5" fill-rule="evenodd" />',
  quadball:
    '<path d="M10 1a9 9 0 1 0 0 18 9 9 0 0 0 0-18M3.1 8.4h3.5v3.2H3.1zm5.2 0h3.4v3.2H8.3zm5.1 0h3.5v3.2h-3.5zM8.3 3.2h3.4v3.4H8.3zm0 10.2h3.4v3.4H8.3z" fill-rule="evenodd" />',

  // --- topology tools ---------------------------------------------------
  subdivide:
    '<path d="M2 2h16v16H2zm1.8 1.8v5.3h5.3V3.8zm7.1 0v5.3h5.3V3.8zm-7.1 7.1v5.3h5.3v-5.3zm7.1 0v5.3h5.3v-5.3z" fill-rule="evenodd" />',
  deleteFace:
    '<path d="M2 2h16v16H2zm1.8 1.8v12.4h12.4V3.8zM6.6 6.1 10 9.5l3.4-3.4 1.3 1.3-3.4 3.4 3.4 3.4-1.3 1.3L10 12.1l-3.4 3.4-1.3-1.3 3.4-3.4-3.4-3.4z" fill-rule="evenodd" />',
  eraseFill:
    '<path d="M2 2h16v16H2zm1.8 1.8v12.4h12.4V3.8zM10 5.4l4.6 4.6L10 14.6 5.4 10z" fill-rule="evenodd" />',
  fillHole:
    '<path d="M2 2h16v16H2zm1.8 1.8v12.4h12.4V3.8zM10 6a4 4 0 1 1 0 8 4 4 0 0 1 0-8" fill-rule="evenodd" />',
  bridge:
    '<path d="M1 3h6v14H1zm12 0h6v14h-6zM7 8.4h6v3.2H7z" fill-rule="evenodd" />',
  crease:
    '<path d="M1 16.5 9.2 3.1l1.6 1L3.3 16.5zm9.6 0L18.8 3.1l1.2.7-.2 12.7zM8.4 16.5h3.2v1.8H8.4z" fill-rule="evenodd" />',
  uncrease:
    '<path d="M1.4 14.6C4.6 8.3 7.4 5.2 10 5.2s5.4 3.1 8.6 9.4l-1.6.8C14 9.6 11.7 7 10 7s-4 2.6-7 8.4z" fill-rule="evenodd" />',
  dissolve:
    '<path d="M2 2h16v16H2zm1.8 1.8v12.4h12.4V3.8zM5.6 9.1h8.8v1.8H5.6z" fill-rule="evenodd" opacity="1" />',
  insertLoop:
    '<path d="M2 2h16v16H2zm1.8 1.8v12.4h12.4V3.8zM9.1 3.8h1.8v12.4H9.1z" fill-rule="evenodd" />',
  thicken:
    '<path d="M2 4.5h12v11H2zm3.5-2.8h12.4v11H16V4.4H5.5z" fill-rule="evenodd" />',
  straighten:
    '<path d="M1.2 15.1 18.4 4.2l1 1.5L2.2 16.6zM3 8.6a1.9 1.9 0 1 1 0 3.8 1.9 1.9 0 0 1 0-3.8m14-4.4a1.9 1.9 0 1 1 0 3.8 1.9 1.9 0 0 1 0-3.8" fill-rule="evenodd" />',
  flatten:
    '<path d="M1 12.4 10 8l9 4.4-9 4.4zM9.1 1.6h1.8v4.6H9.1zM6.4 3.9 10 .8l3.6 3.1z" fill-rule="evenodd" />',
  // Every subpath below is drawn TANGENT to its neighbours rather than
  // overlapping them. Under `evenodd` an overlap is a hole, so two crossing
  // bars would punch a square out of their own intersection -- which is how the
  // four-way Move arrow ends up drawn as one cross outline plus four
  // triangles that only touch it.
  move:
    '<path d="M9.1 4.2h1.8v4.9h4.9v1.8h-4.9v4.9H9.1v-4.9H4.2V9.1h4.9zM10 .8 13 4.2H7zM10 19.2 7 15.8h6zM.8 10 4.2 7v6zM19.2 10 15.8 13V7z" fill-rule="evenodd" />',
  extrude:
    '<path d="M2 13.5h16v4.5H2zM10 1.4 15.2 8h-3.3v4.2H8.1V8H4.8z" fill-rule="evenodd" />',
  extrudeBoundary:
    '<path d="M2.5 14.6h15v2h-15zM2.5 11.4h1.7v2.6H2.5zM15.8 11.4h1.7v2.6h-1.7zM10 1.4 15.2 8h-3.3v4.2H8.1V8H4.8z" fill-rule="evenodd" />',
  // A T-junction, which is literally what a local insert leaves behind: the new
  // edge stops at the seam instead of running the whole ring the way
  // `insertLoop` does.
  insertEdge:
    '<path d="M2 2h16v16H2zm1.8 1.8v12.4h12.4V3.8zM9.1 3.8h1.8v12.4H9.1zM3.8 9.1h5.3v1.8H3.8z" fill-rule="evenodd" />',
  // Two free positions on opposite edges, joined by a seam. The dots sit in
  // gaps cut out of the bars rather than on top of them.
  insertPoint:
    '<path d="M1.5 2.5h1.9v2H1.5zM8.6 2.5h9.9v2H8.6zM6 .9a2.6 2.6 0 1 1 0 5.2A2.6 2.6 0 0 1 6 .9M1.5 15.5h8.9v2H1.5zM15.6 15.5h2.9v2h-2.9zM13 12.4a2.6 2.6 0 1 1 0 5.2 2.6 2.6 0 0 1 0-5.2M5.4 6.6l1.5-.7 6.7 6.5-1.4.9z" fill-rule="evenodd" />',
  unweld:
    '<path d="M1 2h7.1v16H1zM11.9 2H19v16h-7.1zM9.4 3h1.2v3H9.4zM9.4 8.5h1.2v3H9.4zM9.4 14h1.2v3H9.4z" fill-rule="evenodd" />',

  // --- selection modes --------------------------------------------------
  vertex:
    '<path d="M2 2h16v16H2zm1.8 1.8v12.4h12.4V3.8zM10 7.2a2.8 2.8 0 1 1 0 5.6 2.8 2.8 0 0 1 0-5.6" fill-rule="evenodd" />',
  edge: '<path d="M2 2h16v16H2zm1.8 1.8v12.4h12.4V3.8zM4.6 9.1h10.8v1.8H4.6z" fill-rule="evenodd" />',
  faceMode:
    '<path d="M2 2h16v16H2zm1.8 1.8v12.4h12.4V3.8zM5.6 5.6h8.8v8.8H5.6z" fill-rule="evenodd" />',

  // --- history and view -------------------------------------------------
  undo: '<path d="M9.4 4.2V1L3 6l6.4 5V7.6c3 0 5.2 1 6.6 3.4-.5-3.9-3-6.6-6.6-6.8m-6.9 9h1.9c1 2.7 3.6 4.6 6.7 4.6 3.9 0 7-3 7-6.7h-1.9c0 2.7-2.3 4.9-5.1 4.9-2.1 0-3.9-1.2-4.7-3z" fill-rule="evenodd" />',
  redo: '<path d="M10.6 4.2V1L17 6l-6.4 5V7.6c-3 0-5.2 1-6.6 3.4.5-3.9 3-6.6 6.6-6.8m6.9 9h-1.9c-1 2.7-3.6 4.6-6.7 4.6-3.9 0-7-3-7-6.7h1.9c0 2.7 2.3 4.9 5.1 4.9 2.1 0 3.9-1.2 4.7-3z" fill-rule="evenodd" />',
  surface:
    '<path d="M10 1.5 18.5 6 10 10.5 1.5 6zM1.5 8.5 10 13l8.5-4.5v2L10 15l-8.5-4.5z" fill-rule="evenodd" />',
  cage: '<path d="M10 1 18.5 5.5v9L10 19 1.5 14.5v-9zm0 2.1L3.5 6.6v6.8L10 16.9l6.5-3.5V6.6zM10 6a1.6 1.6 0 1 1 0 3.2A1.6 1.6 0 0 1 10 6" fill-rule="evenodd" />',
  edges:
    '<path d="M10 1 18.5 5.5v9L10 19 1.5 14.5v-9zm0 2.1L3.5 6.6v6.8L10 16.9l6.5-3.5V6.6z" fill-rule="evenodd" />',
  commit:
    '<path d="M8.2 14.6 3.6 10l1.5-1.5 3.1 3.1 6.7-6.7L16.4 6z" fill-rule="evenodd" />',
  build:
    '<path d="M10 1.2 17.6 6v8.4L10 18.8 2.4 14.4V6zM4.2 7v6.4L10 16.7l5.8-3.3V7L10 3.7z" fill-rule="evenodd" />',
});

/**
 * Icons another app brings. `registerIcons({ name: innerMarkup })` adds to the
 * set (or overrides one of ours) for every kit component that takes an icon
 * name: the toolbar, the panel, menus. Draw them to the house rules above so
 * they sit at the same weight. Returns the names registered.
 */
const EXTRA = new Map();
export function registerIcons(map) {
  for (const [name, inner] of Object.entries(map)) {
    if (typeof inner !== 'string') throw new Error(`icon ${name}: expected the inner svg markup as a string`);
    EXTRA.set(name, inner);
  }
  return [...Object.keys(map)];
}

/** The inner markup for a name: a registered icon first, then the built-in set. */
export function iconInner(name) {
  return EXTRA.get(name) ?? ICONS[name];
}

/** Is there an icon by this name, built in or registered? */
export function hasIcon(name) {
  return EXTRA.has(name) || Object.hasOwn(ICONS, name);
}

/**
 * One `<svg>`, as markup. `title` becomes the tooltip a label would have been.
 * A `name` that is already `<svg …>` markup is returned as it is, so a caller
 * with a one-off glyph need not register it.
 */
export function iconMarkup(name, { title = '', className = 'os-icon' } = {}) {
  if (typeof name === 'string' && name.trimStart().startsWith('<svg')) return name;
  const inner = iconInner(name);
  if (!inner) throw new Error(`no icon named ${name}`);
  return (
    `<svg class="${className}" viewBox="0 0 20 20" aria-hidden="${title ? 'false' : 'true'}"` +
    `${title ? ` role="img" aria-label="${title}"` : ''}>${inner}</svg>`
  );
}

/** An `<svg>` element, for code that would rather not build markup strings. */
export function iconElement(name, { title = '' } = {}) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 20 20');
  svg.setAttribute('class', 'os-icon');
  if (title) {
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', title);
  } else {
    svg.setAttribute('aria-hidden', 'true');
  }
  svg.innerHTML = iconInner(name) ?? '';
  return svg;
}
