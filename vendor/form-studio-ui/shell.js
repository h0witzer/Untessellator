// SPDX-License-Identifier: LGPL-2.1-or-later
//
// The frame an Onshape tab lives in, built once, so the next app does not draw
// it again.
//
// MEASURED 2026-09-11 (`tools/style-probe.mjs`, `tools/chrome-extract.mjs`) off
// a 1600 x 1000 Part Studio:
//
//     ┌──────────────────────────────────────────────────────────┐
//     │ toolbar 36                                               │
//     ├────┬─────────┬┬─────────────────────────────────────┬────┤
//     │rail│ panel   ││ stage                               │rail│
//     │ 40 │ 200     ││   the canvas, with the view cube    │    │
//     │    │         ││   150 x 150 inset 15 top-right and  │    │
//     │    │         ││   the camera dropdown under it      │    │
//     ├────┴─────────┴┴─────────────────────────────────────┴────┤
//     │ tab bar 30                                               │
//     └──────────────────────────────────────────────────────────┘
//
// The splitter between the panel and the stage is 1px of #666 with a
// col-resize cursor. An app embedded as an Onshape tab is INSIDE the product's
// navbar, left rail and tab bar already, so those are off by default; a
// standalone page (the style guide, a demo) can turn them on to see the whole
// frame.
//
// `createShell` builds the grid and returns the hosts; `createToolbar`,
// `createPanel` and `mountViewCube` fill them.

import { OrbitCamera } from './camera.js';
import { installTooltips } from './tooltip.js';
import { openMenu, isMenuOpen, CHECK_SVG } from './menu.js';
import { ViewCube, STANDARD_VIEWS, ANIMATION_MS, VIEW_CUBE_ICON_SVG, SCREEN_SIZE_PX } from './viewcube.js';

/**
 * @param {HTMLElement} host   filled with the shell; gets `os-shell`
 * @param {{ theme?: 'dark'|'light', panel?: boolean, collapseHandle?: boolean, leftRail?: boolean, rightRail?: boolean, tabBar?: boolean, tooltips?: boolean }} options
 * @returns {{ element, toolbar, panel, splitter, stage, leftRail, rightRail, tabBar, status, message, hint,
 *             setStatus: (text: string) => void, say: (text: string, bad?: boolean) => void, theme: string }}
 */
export function createShell(host, options = {}) {
  const theme = options.theme ?? 'dark';
  const withPanel = options.panel !== false;
  document.body.dataset.osTheme = theme;
  document.body.classList.add('os-app');
  host.classList.add('os-shell');
  if (!withPanel) host.classList.add('os-shell--no-panel');
  if (options.leftRail) host.classList.add('os-shell--left-rail');
  if (options.tabBar) host.classList.add('os-shell--tab-bar');

  const make = (className, area) => {
    const element = document.createElement('div');
    element.className = className;
    element.style.gridArea = area;
    host.append(element);
    return element;
  };
  const toolbar = make('os-toolbar', 'toolbar');
  const leftRail = options.leftRail ? make('os-rail os-rail--left', 'rail') : null;
  const panel = withPanel ? make('os-panel os-shell-panel', 'panel') : null;
  const splitter = withPanel ? make('os-splitter', 'splitter') : null;
  const stage = make('os-stage', 'stage');
  const tabBar = options.tabBar ? make('os-tab-bar', 'tabs') : null;
  // The product's right rail floats mid-height on the canvas's right edge.
  let rightRail = null;
  if (options.rightRail) {
    rightRail = document.createElement('div');
    rightRail.className = 'os-rail os-rail--right';
    stage.append(rightRail);
  }

  // Their collapse handle: a 33 x 34 button on the panel's edge at mid-height,
  // bordered on its top, right and bottom, that folds the panel away.
  let handle = null;
  if (withPanel && options.collapseHandle !== false) {
    handle = document.createElement('button');
    handle.type = 'button';
    handle.className = 'os-panel-handle';
    handle.setAttribute('data-tooltip', 'Toggle panel');
    handle.setAttribute('aria-label', 'Toggle panel');
    handle.innerHTML = '<svg class="os-icon" viewBox="0 0 20 20" aria-hidden="true"><path d="M2 3h2v14H2zm4 1h12v1.6H6zm0 4h12v1.6H6zm0 4h12v1.6H6zm0 4h12v1.6H6zM3 8.5 1 10l2 1.5z" fill="currentColor"/></svg>';
    handle.addEventListener('click', () => {
      const collapsed = host.classList.toggle('os-shell--panel-collapsed');
      handle.setAttribute('aria-expanded', String(!collapsed));
    });
    stage.append(handle);
  }

  const status = document.createElement('div');
  status.className = 'os-stage-status';
  const message = document.createElement('div');
  message.className = 'os-stage-message';
  const hint = document.createElement('div');
  hint.className = 'os-stage-hint';
  stage.append(status, message, hint);

  if (options.tooltips !== false) installTooltips(document.body);

  return {
    element: host,
    theme,
    toolbar,
    leftRail,
    rightRail,
    panel,
    splitter,
    handle,
    stage,
    tabBar,
    status,
    message,
    hint,
    setStatus(text) {
      status.textContent = text;
    },
    say(text, bad = false) {
      message.textContent = text;
      message.classList.toggle('os-stage-message--bad', bad);
    },
  };
}

/**
 * The view cube in its corner of a stage, its camera dropdown under it, and a
 * camera to drive both — the whole corner, wired.
 *
 * @param {HTMLElement} stage
 * @param {{ theme?: 'dark'|'light', camera?: OrbitCamera, onZoomFit?: () => void,
 *           onPreferences?: () => void,
 *           perspective?: { get: () => boolean, set: (on: boolean) => void },
 *           edges?: { get: () => boolean, set: (on: boolean) => void },
 *           extraMenu?: Array<object> }} options
 *   The menu is the product's, in its order, with the entries an app can honour:
 *   the three projections; Graphics preferences… (); Previous view
 *   (the camera's history); Zoom to fit (); Perspective view and
 *   Shaded with edges as checked toggles (, ).
 * @returns {{ camera: OrbitCamera, cube: ViewCube, canvas: HTMLCanvasElement, button: HTMLElement,
 *             update: () => void }}
 *   Call `update()` every frame after `camera.step(now)`; it hands the cube the
 *   camera's matrix.
 */
export function mountViewCube(stage, options = {}) {
  const theme = options.theme ?? 'dark';
  const camera = options.camera ?? new OrbitCamera({ ...STANDARD_VIEWS.Trimetric, distance: 5 });
  const canvas = document.createElement('canvas');
  canvas.id = 'cube';
  canvas.className = 'os-view-cube';
  const ratio = window.devicePixelRatio || 1;
  canvas.width = SCREEN_SIZE_PX * ratio;
  canvas.height = SCREEN_SIZE_PX * ratio;
  stage.append(canvas);

  const toView = (view) => camera.animateToView(STANDARD_VIEWS[view], ANIMATION_MS.long);
  const cube = new ViewCube(canvas, {
    theme,
    onView: toView,
    onRotate: (direction, angle) => camera.rotateInDirection(direction, angle, ANIMATION_MS.short),
  });

  // Their dropdown: the four-tone cube icon and a caret in a 44 x 28 box,
  // translucent at rest and #666 under the cursor, "Camera and render options".
  const button = document.createElement('div');
  button.className = 'os-graphics-dropdown';
  button.setAttribute('data-tooltip', 'Camera and render options');
  button.setAttribute('role', 'button');
  button.setAttribute('aria-haspopup', 'menu');
  button.innerHTML = `${VIEW_CUBE_ICON_SVG}<b class="caret os-dropdown-caret"></b>`;
  button.addEventListener('pointerdown', (event) => {
    event.preventDefault();
    event.stopPropagation();
  });
  button.addEventListener('click', (event) => {
    event.stopPropagation();
    if (isMenuOpen(button)) return;
    const items = standardCameraMenu({ toView, camera, ...options });
    button.classList.add('open');
    openMenu(items, { anchor: button, align: 'right', minWidth: 224, onClose: () => button.classList.remove('open') });
  });
  stage.append(button);

  return {
    camera,
    cube,
    canvas,
    button,
    update() {
      cube.setOrientation(camera.viewMatrix());
    },
  };
}

/** Their gear, drawn filled on the 20 grid for the menu's 16px icon column. */
const GEAR_SVG = '<svg viewBox="0 0 20 20" width="16" height="16" aria-hidden="true"><path fill="currentColor" d="M8.6 1h2.8l.5 2.3 1.5.7 2-1.3 2 2-1.3 2 .7 1.5 2.3.5v2.8l-2.3.5-.7 1.5 1.3 2-2 2-2-1.3-1.5.7-.5 2.3H8.6l-.5-2.3-1.5-.7-2 1.3-2-2 1.3-2-.7-1.5L1 11.4V8.6l2.3-.5.7-1.5-1.3-2 2-2 2 1.3 1.5-.7zM10 7a3 3 0 1 0 0 6 3 3 0 0 0 0-6z"/></svg>';

/**
 * The camera dropdown's menu, in the product's order, holding the entries the
 * caller can honour. Measured 2026-09-11 (tools/graphics-prefs-extract.mjs):
 *
 *     Isometric / Dimetric / Trimetric
 *     Graphics preferences…            (gear)
 *     Named views… / Previous view     (Previous view disabled with no history)
 *     Zoom to fit / Zoom to window
 *     Perspective view / Orient normal to sketch on edit
 *     Shaded with edges ✓ ▸ / Hidden edges removed ✓ ▸ / Tangent edges visible ✓ ▸
 *     View in high quality
 *     Highlight boundary edges
 *     Section view…
 *
 * Entries without a handler are left out rather than shown dead.
 */
export function standardCameraMenu({ toView, camera, onPreferences, onZoomFit, perspective, edges, extraMenu }) {
  const items = [
    { id: 'Isometric', label: 'Isometric', onSelect: () => toView('Isometric') },
    { id: 'Dimetric', label: 'Dimetric', onSelect: () => toView('Dimetric') },
    { id: 'Trimetric', label: 'Trimetric', onSelect: () => toView('Trimetric') },
  ];
  if (onPreferences) items.push({ separator: true }, { id: 'preferences', label: 'Graphics preferences…', icon: GEAR_SVG, onSelect: onPreferences });
  if (camera) {
    items.push({ separator: true }, {
      id: 'previous', label: 'Previous view', disabled: !camera.hasPrevious(),
      onSelect: () => camera.animateToPrevious(ANIMATION_MS.long),
    });
  }
  if (onZoomFit) items.push({ separator: true }, { id: 'zoomFit', label: 'Zoom to fit', onSelect: onZoomFit });
  if (perspective) {
    items.push({ separator: true }, {
      id: 'perspective', label: 'Perspective view', checked: perspective.get(),
      onSelect: () => perspective.set(!perspective.get()),
    });
  }
  if (edges) {
    items.push({ separator: true }, {
      id: 'edges', label: 'Shaded with edges', checked: edges.get(),
      onSelect: () => edges.set(!edges.get()),
    });
  }
  if (extraMenu?.length) items.push({ separator: true }, ...extraMenu);
  void CHECK_SVG;
  return items;
}
