// SPDX-License-Identifier: LGPL-2.1-or-later
//
// The Onshape-native UI kit, in one import.
//
//     import * as ui from './src/ui/index.js';
//     <link rel="stylesheet" href="./src/ui/onshape.css">
//
// Everything here was measured out of the live product rather than designed;
// docs/UI_KIT.md is how to build an app on it and docs/ONSHAPE_STYLE.md is
// where each number came from.
//
//   shell.js        the frame: toolbar, panel, splitter, stage, rails, tab bar;
//                   and the view-cube corner, wired to a camera
//   toolbar.js      icon tools, toggles, split tools with menus, the labelled
//                   action, separators, the search button
//   panel.js        filter, section headers, list rows with children
//   menu.js         dropdown and context menus
//   tooltip.js      the tooltip, one for the page
//   dialog.js       the docked dialog; Graphics preferences
//   render.js       the surface program, highlight bands and points, SSAO, projections
//   viewcube.js     the view cube, its standard views and its icon
//   camera.js       the orbit camera with the product's rotation model
//   icons.js        filled icons on the 20 grid, and how to add one
//   interaction.js  the gesture model, box select, highlight and manipulator colours
//   viewport.js     the part material, background, edges, and PART_GLSL

export { createShell, mountViewCube, standardCameraMenu } from './shell.js';
export * from './render.js';
export { createToolbar, CARET_SVG } from './toolbar.js';
export { createPanel } from './panel.js';
export { openMenu, closeMenu, isMenuOpen, bindContextMenu, CHECK_SVG } from './menu.js';
export { installTooltips } from './tooltip.js';
export { createDialog, openGraphicsPreferences, GRAPHICS_DEFAULTS } from './dialog.js';
export * from './viewcube.js';
export * from './camera.js';
export * from './icons.js';
export * from './interaction.js';
export * from './viewport.js';
