// SPDX-License-Identifier: LGPL-2.1-or-later
//
// How Onshape behaves, as constants and rules a viewport can implement.
//
// Colour and type decide whether a panel looks like the product. This decides
// whether it FEELS like it, which is the part people describe as "native" and
// almost never write down. A viewport with a perfect palette that orbits on the
// left button is wrong in a way that is felt on the first drag.
//
// The colours here were captured from a live session by
// `tools/interaction-probe.mjs`, which screenshots the product MID-GESTURE —
// a box-select net exists only while a button is held, is drawn in WebGL, and
// has no CSS token to read. The rules were confirmed against the product by
// someone who uses it daily.

/**
 * The mouse model. This is the part that surprises people coming from other
 * CAD web apps, and getting it wrong is the fastest way to feel foreign.
 */
export const MOUSE = Object.freeze({
  /** RIGHT button drags the camera. Not the left one, and not with a modifier. */
  orbit: 'right-drag',
  /** LEFT click selects what is under the cursor. */
  select: 'left-click',
  /** LEFT drag draws a selection net — see BOX_SELECT. */
  boxSelect: 'left-drag',
  /** Shift adds to the selection rather than replacing it. */
  addToSelection: 'shift',
  zoom: 'wheel',
});

/**
 * Box select, both directions, measured 2026-09-11.
 *
 * The CAD convention, and Onshape marks which one you are about to get before
 * you let go — by colour AND by whether the border is dashed, so it survives
 * colour blindness and a glance.
 *
 * The alphas are derived: the captured pixels are the token colour composited
 * over the `#333` canvas, so `(captured - 51) / (token - 51)` recovers them.
 */
export const BOX_SELECT = Object.freeze({
  /**
   * Dragging LEFT TO RIGHT takes only what falls entirely INSIDE the box.
   * Solid blue. Captured border rgb(91,115,142), fill rgb(65,73,82).
   */
  window: Object.freeze({
    direction: 'left-to-right',
    takes: 'entirely inside',
    token: '--os-box-select-color',
    colour: Object.freeze({ dark: '#769fcc', light: '#1651b0' }),
    borderAlpha: 0.6,
    fillAlpha: 0.21,
    dashed: false,
  }),
  /**
   * Dragging RIGHT TO LEFT takes anything the box TOUCHES. Dashed amber.
   * Captured border rgb(173,136,28), fill rgb(92,80,44), 12 dashes across a
   * 190px edge — roughly a 6px on, 10px off pattern.
   */
  crossing: Object.freeze({
    direction: 'right-to-left',
    takes: 'anything it touches',
    token: '--os-highlight-outline',
    colour: Object.freeze({ dark: '#dba517', light: '#fab601' }),
    borderAlpha: 0.73,
    fillAlpha: 0.24,
    dashed: true,
    dash: Object.freeze([6, 10]),
  }),
});

/**
 * Selection: a click on an entity toggles it in the set; a box select adds
 * what it catches; a click on EMPTY SPACE clears the set, and so does Space.
 * Running a command does not clear it: the selection is what a tool acts on.
 *
 * (An earlier version of this file claimed an empty click did not clear.
 * It does, in the product; corrected 2026-09-11 by someone who uses it daily.)
 */
export const SELECTION_MODEL = Object.freeze({
  clickToggles: true,
  boxSelectAdds: true,
  clearKey: ' ',
  clearKeyLabel: 'Space',
  clickEmptySpaceClears: true,
});

/**
 * Highlight colours. Onshape selects in AMBER, not in the interface accent.
 *
 * This catches people out because the chrome's accent is the blue `#3e94ff`,
 * so a viewport that highlights in blue looks self-consistent and still reads
 * as the wrong application. The blue is for interface state; the amber is for
 * geometry.
 */
export const HIGHLIGHT = Object.freeze({
  dark: Object.freeze({
    outline: '#dba517',
    fill: '#6a4f06',
    preHighlight: '#fcda7f',
  }),
  light: Object.freeze({
    outline: '#fab601',
    fill: '#fcda7f',
    preHighlight: '#fcda7f',
  }),
});

/**
 * Manipulators — the drag handles.
 *
 * A white line with a dark casing, so it reads against a lit face, a dark face
 * and the flat background without changing colour. That casing is the whole
 * trick and it is why Onshape's handles never disappear into the model.
 *
 * Note the hover and drag colours SWAP between themes rather than lightening:
 * `--os-manipulator-line-color--hover` is `#134e96` on dark and `#a2b9df` on
 * light, and the drag pair is the reverse.
 */
export const MANIPULATOR = Object.freeze({
  dark: Object.freeze({
    line: '#ffffff',
    outline: '#606a72',
    lineHover: '#134e96',
    outlineHover: '#a2b9df',
    lineDrag: '#a2b9df',
    outlineDrag: '#134e96',
    displacement: '#a2b9df',
    resizeOuter: '#d0dbe57f',
    resizeOuterHighlight: '#d0dbe5e5',
  }),
  light: Object.freeze({
    line: '#ffffff',
    outline: '#606a72',
    lineHover: '#a2b9df',
    outlineHover: '#134e96',
    lineDrag: '#134e96',
    outlineDrag: '#a2b9df',
    displacement: '#1651b0',
    resizeOuter: '#1b5faa7f',
    resizeOuterHighlight: '#1b5faae5',
  }),
});

/** The view cube, which every Onshape viewport has in the same corner. */
export const VIEW_CUBE = Object.freeze({
  size: 150,
  insetRight: 15,
  insetTop: 15,
  dark: Object.freeze({
    front: '#333333',
    back: '#999999',
    arrow: '#999999',
    faceHighlight: '#90cef199',
    text: '#ffffff',
    textHover: '#000000',
  }),
  light: Object.freeze({
    front: '#ffffff',
    back: '#cedbe599',
    arrow: '#cedbe599',
    faceHighlight: '#90cef199',
    text: '#000000',
    textHover: '#000000',
  }),
});

/**
 * The frame, measured off a 1600x1000 Part Studio.
 *
 * These are the numbers that make a panel sit in the product rather than beside
 * it: the toolbar's height, the left panel's width, where the canvas starts.
 */
export const LAYOUT = Object.freeze({
  navbarHeight: 40,
  toolbarHeight: 36,
  /** A tool is 32 x 32 with a 1px margin and a 20px icon at a 6px inset; the
   *  70 x 36 first measured here was the undo/redo pair's wrapper. */
  toolbarTool: Object.freeze({ size: 32, margin: 1, icon: 20, inset: 6, radius: 3 }),
  toolbarCaret: Object.freeze({ width: 18, chevron: 10 }),
  toolbarSeparator: Object.freeze({ width: 2, height: 26, margin: 5 }),
  leftRailWidth: 40,
  leftPanelWidth: 200,
  panelHeaderHeight: 30,
  panelFilterHeight: 37,
  /** A list row is 26px; its 20px icon box holds a 17px glyph. The 21 first
   *  written here was the name span, not the row. */
  listRowHeight: 26,
  listIndent: 19,
  leftRailButton: Object.freeze({ width: 39, height: 31 }),
  tabBarHeight: 30,
  tabBarFill: Object.freeze({ dark: '#555555', light: '#eeeeee' }),
  rightRailButton: Object.freeze({ width: 33, height: 34 }),
  splitterWidth: 1,
  menuRowHeight: 22,
});

/** `rgba()` for a measured colour and alpha, for a canvas or a shader. */
export function rgba(hex, alpha) {
  const value = hex.replace('#', '');
  const r = Number.parseInt(value.slice(0, 2), 16);
  const g = Number.parseInt(value.slice(2, 4), 16);
  const b = Number.parseInt(value.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

/**
 * Which box-select mode a drag is in, from its horizontal direction.
 *
 * Takes the start and current x. The rule is direction only — the vertical
 * component does not matter, which is what lets a mostly-vertical drag still
 * mean something definite.
 */
export function boxSelectMode(startX, currentX) {
  return currentX >= startX ? BOX_SELECT.window : BOX_SELECT.crossing;
}
