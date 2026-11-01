// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Onshape's view cube, rebuilt from Onshape's own code rather than from a
// picture of it.
//
// READ OUT OF THE PRODUCT, 2026-09-11, by `tools/viewcube-extract.mjs`, which
// pulled the view-manipulator module out of the live client bundle
// (`webgl.<hash>.js`, the `viewmanipulator` / `viewmanipulatorbuttons` classes)
// and the appearance-constant table out of `woolsthorpe.<hash>.js`, then
// screenshotted the real thing at 3x at rest, under the cursor and mid-drag to
// check the reading. Every number below is one of theirs:
//
//     VIEW_CUBE_SCREEN_SIZE_PX      150        the box the cube lives in
//     VIEW_CUBE_FRONT_SIZE          0.75       the inner plate, as a fraction of a face
//     VIEW_CUBE_BACK_SIZE           0.97       the outer plate
//     VIEW_CUBE_CORNER_SIZE         0.20       plate corner radius, as a fraction of a face
//     VIEW_CUBE_ARROW_WIDTH         0.10       straight arrow base, in box units
//     VIEW_CUBE_ARROW_HEIGHT        0.13       straight arrow height
//     VIEW_CUBE_ARROW_ARC_WIDTH     0.04       the curved arrows' band
//     VIEW_CUBE_TEXT_HEIGHT_PX      12         face labels
//     VIEW_CUBE_AXIS_TEXT_HEIGHT_PX 12         X Y Z
//     VIEW_CUBE_AXIS_GAP            1          the triad sits 1px off the cube's edges
//     VIEW_CUBE_HIDDEN_AXIS_OPACITY 0.3        an axis behind the cube, stippled 4 on 4 off
//     VIEW_CUBE_AXIS_LABEL_FADE     8 -> 2 deg an axis label fades as its axis points at you
//
// WHAT IT ACTUALLY IS, which is not a wireframe and not a solid either:
//
//   * Each face is TWO rounded plates in the face's plane. A 97% plate in
//     `--os-view-cube-back-color` (#999 on dark) with a 75% plate in
//     `--os-view-cube-front-color` (#333, the canvas colour) on top of it. What
//     reads as a light frame around a dark centre is exactly that. There are no
//     drawn edges: the outline geometry exists in their code with isVisible
//     false, and the dark seams between faces are the 3% the plates leave bare.
//   * Each corner is a small DISC, 0.24 of a half-face across, sitting just
//     inside the cube on the corner's diagonal. It shows only through the notch
//     the three plates' rounded corners leave at the vertex, which is the
//     rivet-like dark corner the real cube has.
//   * There are 14 pick regions: 6 faces and 8 corners. Not 26. Edges pick
//     nothing. A face animates to the axis view, a corner to a TRIMETRIC view
//     (30 degrees of azimuth off the nearer of front/back, 30 degrees of
//     elevation), not an isometric.
//   * Hover paints both plates, or the disc, `--os-view-cube-face-highlight`
//     (#90cef1 at 0.60) and swaps the label to its hover colour (#000).
//   * Around it, in screen space: four triangles pointing OUTWARD at the box's
//     edges, and two arc-with-head arrows over the top, 45 to 72 degrees and
//     108 to 135. They rotate the camera 15 degrees (shift: 90, ctrl: 5), not 90.
//   * The triad is three 1px lines the full length of the cube's near-lower-left
//     edges, 1px outside them, in the axis colours, with billboarded labels 3px
//     past their ends. Where an axis runs behind the cube it is stippled at 0.3.
//   * Below the box, right-aligned: a 44 x 28 dropdown, "Camera and render
//     options", with a 20px four-tone cube icon and a caret. Not a home button.
//
// THIS COPY (Untessellator, 2026-10-31) departs from the reading on purpose:
//
//   * The corner discs are painted where the geometry puts them, behind the
//     near plates, so they show only in the vertex notches; the kit painted
//     them over the plates. Only a hovered disc is drawn on top.
//   * A far face shows only its light plate, through the seams.
//   * The plates and discs are faceted (octagons, hexagons) until clicked: the
//     face or corner clicked eases round once the camera has turned to it, and
//     back to facets when the view leaves it. Each near centre plate carries the
//     diagonal an STL splits a face on, fading as it smooths. Hover only lights.
//   * A near face's centre plate picks before the corners; a corner's target is
//     a little wider than its notch.
//   * The colours are the logo's plum, lilac and filament axes.
//
// The cube is handed the camera's view matrix and nothing else, so it cannot
// disagree with the viewport about which way is up. See `src/ui/camera.js`
// for the camera whose rotation model matches the product's drag.

// ---------------------------------------------------------------------------
// constants, theirs
// ---------------------------------------------------------------------------

export const SCREEN_SIZE_PX = 150;
const FRONT_SIZE = 0.75;
const BACK_SIZE = 0.97;
const CORNER_SIZE = 0.2;
const ARROW_WIDTH = 0.1;
const ARROW_HEIGHT = 0.13;
const ARROW_ARC_WIDTH = 0.04;
const TEXT_HEIGHT_PX = 12;
const AXIS_TEXT_HEIGHT_PX = 12;
const TEXT_PADDING_PX = 1;
const AXIS_GAP_PX = 1;
const AXIS_WIDTH_PX = 1;
const HIDDEN_AXIS_OPACITY = 0.3;
const AXIS_LABEL_FADE_START = (8 * Math.PI) / 180;
const AXIS_LABEL_FADE_END = (2 * Math.PI) / 180;
const DEG = Math.PI / 180;

/**
 * The cube's half edge, in box units. Their `zy()`: the box less the arrows,
 * less room for the axis labels. 0.19667 -> a 59px cube in a 150px box.
 */
const HALF = (0.5 - ARROW_WIDTH) / 1.6 - AXIS_TEXT_HEIGHT_PX / SCREEN_SIZE_PX / 1.5;
const HALF_PX = HALF * SCREEN_SIZE_PX;

/** How far a click on an arrow turns the camera. `ROTATE_ANGLE` and friends. */
export const ROTATE_ANGLES = Object.freeze({
  default: 15 * DEG,
  coarse: 90 * DEG,
  fine: 5 * DEG,
});

/** Their `CAMERA_*_ANIMATION_TIME_MS`. */
export const ANIMATION_MS = Object.freeze({ snap: 125, short: 250, long: 500 });
// facets to round, or back, for a face or corner of the cube
const SMOOTH_MS = 240;

export const COLOURS = Object.freeze({
  dark: Object.freeze({
    front: 'rgba(42,26,55,1)',
    back: 'rgba(122,96,152,1)',
    arrow: 'rgba(154,133,176,0.9)',
    highlight: 'rgba(185,140,255,1)',
    seam: 'rgba(122,96,152,0.9)',
    textIdle: '#f5effb',
    textHover: '#2a1638',
    axisX: '#ff5c8a',
    axisY: '#2fc59a',
    axisZ: '#37adf3',
  }),
  light: Object.freeze({
    front: 'rgba(255,255,255,1)',
    back: 'rgba(203,187,220,1)',
    arrow: 'rgba(134,116,154,0.8)',
    highlight: 'rgba(42,22,56,1)',
    seam: 'rgba(203,187,220,1)',
    textIdle: '#2a1638',
    textHover: '#ffffff',
    axisX: '#e8336b',
    axisY: '#16a37b',
    axisZ: '#277fe5',
  }),
});

// ---------------------------------------------------------------------------
// vectors
// ---------------------------------------------------------------------------

const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const scale = (a, k) => [a[0] * k, a[1] * k, a[2] * k];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const norm = (a) => {
  const length = Math.hypot(...a) || 1;
  return scale(a, 1 / length);
};

/** Rotate `v` by `angle` about unit `axis` (Rodrigues). */
export function rotate(v, angle, axis) {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const k = norm(axis);
  return add(add(scale(v, c), scale(cross(k, v), s)), scale(k, dot(k, v) * (1 - c)));
}

// ---------------------------------------------------------------------------
// the standard views, exactly their `gr` table
// ---------------------------------------------------------------------------

/**
 * A camera frame: `right`, `up` and `back` (the direction from the target to
 * the eye) as unit vectors. `direction`, the way the camera looks, is -back.
 */
function frame(right, up, back) {
  return Object.freeze({
    right: Object.freeze(norm(right)),
    up: Object.freeze(norm(up)),
    back: Object.freeze(norm(back)),
    direction: Object.freeze(scale(norm(back), -1)),
  });
}

/** Their `dr(quadrant, top)`: Rz(quadrant * 90 + 30 degrees) then Rx(60 or 120). */
function trimetric(quadrant, top) {
  const z = quadrant * (Math.PI / 2) + Math.PI / 6;
  const x = top ? Math.PI / 3 : (2 * Math.PI) / 3;
  const spin = (v) => rotate(rotate(v, x, [1, 0, 0]), z, [0, 0, 1]);
  return frame(spin([1, 0, 0]), spin([0, 1, 0]), spin([0, 0, 1]));
}

export const STANDARD_VIEWS = Object.freeze({
  XY: frame([1, 0, 0], [0, 1, 0], [0, 0, 1]),
  XYback: frame([1, 0, 0], [0, -1, 0], [0, 0, -1]),
  YZ: frame([0, 1, 0], [0, 0, 1], [1, 0, 0]),
  YZback: frame([0, -1, 0], [0, 0, 1], [-1, 0, 0]),
  XZ: frame([1, 0, 0], [0, 0, 1], [0, -1, 0]),
  XZback: frame([-1, 0, 0], [0, 0, 1], [0, 1, 0]),
  Isometric: (() => {
    const e = Math.sqrt(0.5);
    const t = Math.sqrt(1 / 3);
    const i = Math.sqrt(1 / 6);
    return frame([e, e, 0], [-i, i, 2 * i], [t, -t, t]);
  })(),
  Dimetric: (() => {
    const spin = (v) => rotate(rotate(v, Math.PI / 3, [1, 0, 0]), Math.PI / 4, [0, 0, 1]);
    return frame(spin([1, 0, 0]), spin([0, 1, 0]), spin([0, 0, 1]));
  })(),
  // quadrants: 0 right-front, 1 right-back, 2 left-back, 3 left-front
  Trimetric: trimetric(0, true),
  TrimetricTopRightBack: trimetric(1, true),
  TrimetricTopLeftBack: trimetric(2, true),
  TrimetricTopLeftFront: trimetric(3, true),
  TrimetricBottomRightFront: trimetric(0, false),
  TrimetricBottomRightBack: trimetric(1, false),
  TrimetricBottomLeftBack: trimetric(2, false),
  TrimetricBottomLeftFront: trimetric(3, false),
});

/** The entries of their camera menu, in its order. */
export const MENU_VIEWS = Object.freeze([
  { view: 'Isometric', label: 'Isometric' },
  { view: 'Trimetric', label: 'Trimetric' },
  { view: 'Dimetric', label: 'Dimetric' },
  { view: 'XY', label: 'Top' },
  { view: 'XYback', label: 'Bottom' },
  { view: 'XZ', label: 'Front' },
  { view: 'XZback', label: 'Back' },
  { view: 'YZback', label: 'Left' },
  { view: 'YZ', label: 'Right' },
]);

// ---------------------------------------------------------------------------
// the geometry, theirs
// ---------------------------------------------------------------------------

/**
 * The six faces as they build them: a corner `origin` and the two edge
 * directions `r` and `o` the label runs along and up. Front is -Y, Z is up.
 */
export const FACES = Object.freeze([
  { id: 'Top', label: 'Top', view: 'XY', normal: [0, 0, 1], origin: [-1, -1, 1], r: [1, 0, 0], o: [0, 1, 0] },
  { id: 'Bottom', label: 'Bottom', view: 'XYback', normal: [0, 0, -1], origin: [-1, 1, -1], r: [1, 0, 0], o: [0, -1, 0] },
  { id: 'Front', label: 'Front', view: 'XZ', normal: [0, -1, 0], origin: [-1, -1, -1], r: [1, 0, 0], o: [0, 0, 1] },
  { id: 'Back', label: 'Back', view: 'XZback', normal: [0, 1, 0], origin: [1, 1, -1], r: [-1, 0, 0], o: [0, 0, 1] },
  { id: 'Right', label: 'Right', view: 'YZ', normal: [1, 0, 0], origin: [1, -1, -1], r: [0, 1, 0], o: [0, 0, 1] },
  { id: 'Left', label: 'Left', view: 'YZback', normal: [-1, 0, 0], origin: [-1, 1, -1], r: [0, -1, 0], o: [0, 0, 1] },
].map((face) => Object.freeze({ ...face, kind: 'face' })));

export const CORNERS = Object.freeze((() => {
  const out = [];
  for (const z of [1, -1]) {
    for (const y of [-1, 1]) {
      for (const x of [1, -1]) {
        const id = `Trimetric${z > 0 ? 'Top' : 'Bottom'}${x > 0 ? 'Right' : 'Left'}${y < 0 ? 'Front' : 'Back'}`;
        // Their enum spells the default one plainly.
        const view = id === 'TrimetricTopRightFront' ? 'Trimetric' : id;
        out.push(Object.freeze({ id: view, kind: 'corner', view, normal: norm([x, y, z]), signs: [x, y, z] }));
      }
    }
  }
  return out;
})());

/** The 14 things a click on the cube can pick. */
export const REGIONS = Object.freeze([...FACES, ...CORNERS]);

/** The six controls around the cube, and what each asks the camera to do. */
export const CONTROLS = Object.freeze([
  { id: 'LeftArrow', direction: 'left', title: 'Rotate left' },
  { id: 'RightArrow', direction: 'right', title: 'Rotate right' },
  { id: 'UpArrow', direction: 'up', title: 'Rotate up' },
  { id: 'DownArrow', direction: 'down', title: 'Rotate down' },
  { id: 'CwArrow', direction: 'clockwise', title: 'Rotate clockwise' },
  { id: 'CcwArrow', direction: 'counterclockwise', title: 'Rotate counterclockwise' },
]);

/** A rounded rectangle in 3D: centre, half extents along unit `r`/`o`, corner radius. */
// `cut`: the same points laid on each corner's chord instead of its arc, so the plate is an octagon that can ease
// into the rounded plate point for point.
function roundedRect(centre, r, o, halfR, halfO, radius, segments = 9, cut = false) {
  const points = [];
  const cornerOrder = [[1, 1, 0], [-1, 1, Math.PI / 2], [-1, -1, Math.PI], [1, -1, (3 * Math.PI) / 2]];
  const at = (arcCentre, angle) => add(add(arcCentre, scale(r, Math.cos(angle) * radius)), scale(o, Math.sin(angle) * radius));
  for (const [sr, so, start] of cornerOrder) {
    const arcCentre = add(add(centre, scale(r, sr * (halfR - radius))), scale(o, so * (halfO - radius)));
    const a = at(arcCentre, start), b = at(arcCentre, start + Math.PI / 2);
    for (let i = 0; i <= segments; i += 1) {
      const t = i / segments;
      points.push(cut ? add(a, scale(add(b, scale(a, -1)), t)) : at(arcCentre, start + t * (Math.PI / 2)));
    }
  }
  return points;
}

/** A disc in 3D; `cut`, a hexagon with the same points, to ease into the disc. */
function disc(centre, normal, radius, segments = 36, cut = false) {
  const n = norm(normal);
  const helper = Math.abs(n[2]) > 0.9 ? [1, 0, 0] : [0, 0, 1];
  const u = norm(cross(n, helper));
  const v = cross(n, u);
  const on = (angle) => add(add(centre, scale(u, Math.cos(angle) * radius)), scale(v, Math.sin(angle) * radius));
  const side = segments / 6;
  const points = [];
  for (let i = 0; i < segments; i += 1) {
    if (!cut) { points.push(on((i / segments) * Math.PI * 2)); continue; }
    const j = Math.floor(i / side), t = (i % side) / side;
    const a = on((j / 6) * Math.PI * 2), b = on(((j + 1) / 6) * Math.PI * 2);
    points.push(add(a, scale(add(b, scale(a, -1)), t)));
  }
  return points;
}

/** Everything static about the cube, in pixels at the 150 box. Built once. */
const GEOMETRY = (() => {
  const h = HALF_PX;
  const edge = 2 * h;
  const radius = edge * CORNER_SIZE;
  const faces = FACES.map((face) => {
    const origin = scale(face.origin, h);
    const centre = add(add(origin, scale(face.r, h)), scale(face.o, h));
    return {
      face,
      centre,
      corners: [
        origin,
        add(origin, scale(face.r, edge)),
        add(add(origin, scale(face.r, edge)), scale(face.o, edge)),
        add(origin, scale(face.o, edge)),
      ],
      back: roundedRect(centre, face.r, face.o, (edge * BACK_SIZE) / 2, (edge * BACK_SIZE) / 2, radius),
      front: roundedRect(centre, face.r, face.o, (edge * FRONT_SIZE) / 2, (edge * FRONT_SIZE) / 2, radius),
      backCut: roundedRect(centre, face.r, face.o, (edge * BACK_SIZE) / 2, (edge * BACK_SIZE) / 2, radius, 9, true),
      frontCut: roundedRect(centre, face.r, face.o, (edge * FRONT_SIZE) / 2, (edge * FRONT_SIZE) / 2, radius, 9, true),
    };
  });
  const corners = CORNERS.map((corner) => ({
    corner,
    centre: scale(corner.signs, 0.8 * h),
    polygon: disc(scale(corner.signs, 0.8 * h), corner.signs, 0.24 * h),
    polygonCut: disc(scale(corner.signs, 0.8 * h), corner.signs, 0.24 * h, 36, true),
  }));
  const axisOrigin = [-h - AXIS_GAP_PX, -h - AXIS_GAP_PX, -h - AXIS_GAP_PX];
  const axes = [
    { label: 'X', direction: [1, 0, 0], colour: 'axisX', faces: ['Front', 'Bottom'] },
    { label: 'Y', direction: [0, 1, 0], colour: 'axisY', faces: ['Left', 'Bottom'] },
    { label: 'Z', direction: [0, 0, 1], colour: 'axisZ', faces: ['Left', 'Front'] },
  ].map((axis) => ({ ...axis, from: axisOrigin, to: add(axisOrigin, scale(axis.direction, edge)) }));

  // The controls, in box units with y UP as their GL code has it.
  const V = 0.5;
  const U = ARROW_WIDTH / 2;
  const K = ARROW_HEIGHT / 2;
  const W = V - K;
  const s45 = Math.sin(Math.PI / 4);
  const c45 = Math.cos(Math.PI / 4);
  const sHead = Math.sin(Math.PI / 4 - K / V);
  const cHead = Math.cos(Math.PI / 4 - K / V);
  const px = (p) => [p[0] * SCREEN_SIZE_PX, p[1] * SCREEN_SIZE_PX];
  const arc = (from, to) => {
    const segments = Math.ceil((36 * Math.abs(to - from)) / Math.PI);
    const inner = V - U - ARROW_ARC_WIDTH / 2;
    const outer = V - U + ARROW_ARC_WIDTH / 2;
    const band = [];
    for (let i = 0; i <= segments; i += 1) {
      const a = from + ((to - from) * i) / segments;
      band.push(px([Math.cos(a) * inner, Math.sin(a) * inner]));
    }
    for (let i = segments; i >= 0; i -= 1) {
      const a = from + ((to - from) * i) / segments;
      band.push(px([Math.cos(a) * outer, Math.sin(a) * outer]));
    }
    return band;
  };
  const controls = {
    LeftArrow: [[[-V, 0], [-W, -U], [-W, U]].map(px)],
    RightArrow: [[[V, 0], [W, U], [W, -U]].map(px)],
    UpArrow: [[[0, V], [-U, W], [U, W]].map(px)],
    DownArrow: [[[0, -V], [U, -W], [-U, -W]].map(px)],
    CwArrow: [
      [[c45 * V, s45 * V], [c45 * (V - 2 * U), s45 * (V - 2 * U)], [cHead * (V - U), sHead * (V - U)]].map(px),
      arc(0.25 * Math.PI, 0.4 * Math.PI),
    ],
    CcwArrow: [
      [[-c45 * (V - 2 * U), s45 * (V - 2 * U)], [-c45 * V, s45 * V], [-cHead * (V - U), sHead * (V - U)]].map(px),
      arc(0.6 * Math.PI, 0.75 * Math.PI),
    ],
  };
  return { faces, corners, axes, controls, radius };
})();

// ---------------------------------------------------------------------------
// the cube
// ---------------------------------------------------------------------------

export class ViewCube {
  /**
   * @param {HTMLCanvasElement} canvas  square; drawn at its own pixel size, so
   *   set width/height to 150 * devicePixelRatio for a crisp cube.
   * @param {{ theme?: 'dark'|'light',
   *           onView?: (view: string, region: object) => void,
   *           onRotate?: (direction: string, angle: number) => void }} options
   */
  constructor(canvas, options = {}) {
    this.canvas = canvas;
    this.context = canvas.getContext('2d');
    this.theme = options.theme ?? 'dark';
    this.onView = options.onView ?? (() => {});
    this.onRotate = options.onRotate ?? (() => {});
    this.rotation = [1, 0, 0, 0, 1, 0, 0, 0, 1];
    this.hover = null;
    this.hits = [];
    this.font = resolveFont(canvas);
    // how smooth each face and corner is, 0 faceted .. 1 rounded, and the animation frame easing them
    this.smooth = new Map();
    this._last = 0;
    this._frame = null;
    // the face or corner last clicked, and whether the camera has finished turning to it
    this.active = null;
    this.settled = true;

    canvas.addEventListener('pointermove', (event) => this._onMove(event));
    canvas.addEventListener('pointerleave', () => {
      if (this.hover) {
        this.hover = null;
        this.render();
      }
    });
    // The cube owns its own presses: without stopping them here, a click on it
    // would also start a box select in the viewport underneath.
    canvas.addEventListener('pointerdown', (event) => this._onDown(event));
    canvas.addEventListener('pointerup', (event) => event.stopPropagation());
    canvas.addEventListener('contextmenu', (event) => event.preventDefault());
  }

  /** Hand it the camera's view matrix (column-major 4x4, or a 3x3 rotation). */
  setOrientation(viewMatrix) {
    this.rotation = viewMatrix.length === 9
      ? [...viewMatrix]
      : [
        viewMatrix[0], viewMatrix[1], viewMatrix[2],
        viewMatrix[4], viewMatrix[5], viewMatrix[6],
        viewMatrix[8], viewMatrix[9], viewMatrix[10],
      ];
    // a clicked face or corner stays smooth while its view holds; turn away and it goes back to facets
    if (this.active && this.settled) {
      const region = REGIONS.find((r) => r.id === this.active || r.view === this.active);
      if (!region || this._toView(region.normal)[2] < 0.9) this.active = null;
    }
    this.render();
  }

  _toView(point) {
    const m = this.rotation;
    return [
      m[0] * point[0] + m[3] * point[1] + m[6] * point[2],
      m[1] * point[0] + m[4] * point[1] + m[7] * point[2],
      m[2] * point[0] + m[5] * point[1] + m[8] * point[2],
    ];
  }

  /** Orthographic, like theirs: a point in cube pixels to canvas pixels. */
  _project(point) {
    const size = this.canvas.width;
    const k = size / SCREEN_SIZE_PX;
    const view = this._toView(point);
    return [size / 2 + view[0] * k, size / 2 - view[1] * k, view[2]];
  }

  /** A screen-space point in box pixels (y up) to canvas pixels. */
  _screen(point) {
    const size = this.canvas.width;
    const k = size / SCREEN_SIZE_PX;
    return [size / 2 + point[0] * k, size / 2 - point[1] * k];
  }

  _path(points) {
    const context = this.context;
    context.beginPath();
    context.moveTo(points[0][0], points[0][1]);
    for (let i = 1; i < points.length; i += 1) context.lineTo(points[i][0], points[i][1]);
    context.closePath();
  }

  render() {
    const context = this.context;
    const size = this.canvas.width;
    const k = size / SCREEN_SIZE_PX;
    const palette = COLOURS[this.theme];
    const hover = this.hover?.id ?? '';
    // The cube is a mesh until you click it: the face or corner clicked eases from facets to round while its view
    // holds, then back once the camera turns away. Hover only lights it. SMOOTH_MS each way, a frame at a time.
    const now = performance.now();
    const dt = this._last ? Math.min(64, now - this._last) : 16;
    this._last = now;
    let moving = false;
    const smooth = (id) => {
      const target = this.active === id ? 1 : 0;
      const was = this.smooth.get(id) ?? target;
      const next = was + Math.sign(target - was) * Math.min(Math.abs(target - was), dt / SMOOTH_MS);
      if (next !== target) moving = true;
      this.smooth.set(id, next);
      return next * next * (3 - 2 * next);
    };
    const blend = (cut, round, t) => cut.map((p, i) => [p[0] + (round[i][0] - p[0]) * t, p[1] + (round[i][1] - p[1]) * t, p[2]]);
    context.save();
    context.clearRect(0, 0, size, size);
    context.lineJoin = 'round';
    context.lineCap = 'butt';
    this.hits = [];

    // Which faces look at the camera, and how deep each is.
    const faces = GEOMETRY.faces.map((item) => {
      const normalView = this._toView(item.face.normal);
      const t = smooth(item.face.id);
      return {
        ...item,
        t,
        facing: normalView[2] > 1e-6,
        depth: this._toView(item.centre)[2],
        backPolygon: blend(item.backCut.map((p) => this._project(p)), item.back.map((p) => this._project(p)), t),
        frontPolygon: blend(item.frontCut.map((p) => this._project(p)), item.front.map((p) => this._project(p)), t),
        outline: item.corners.map((p) => this._project(p)),
      };
    });
    const far = faces.filter((f) => !f.facing).sort((a, b) => a.depth - b.depth);
    const near = faces.filter((f) => f.facing).sort((a, b) => a.depth - b.depth);

    // As the product draws it (this file's header): a face turned to the camera is its light 97% plate with the
    // dark 75% plate on it, the frame and the centre; a face turned away shows only its light plate, through the
    // seams. A corner is a disc inside the cube, seen only through the notch the plates leave at the vertex.
    // Hover lights a whole face, both plates, as one wash.
    const paintPlate = (item, polygon, colour) => {
      this._path(polygon);
      context.fillStyle = hover === item.face.id ? palette.highlight : colour;
      context.fill();
    };

    // 1. The far side: light plates, seen through the seams.
    for (const item of far) paintPlate(item, item.backPolygon, palette.back);

    // 2. The corner discs, behind the near faces: they show only in the vertex notches and never cover a face.
    //    Every disc whose corner points at the camera; the far corner is behind everything and is not drawn.
    const discs = GEOMETRY.corners
      .filter((item) => this._toView(item.corner.normal)[2] > 0)
      .map((item) => ({
        ...item,
        polygon: blend(item.polygonCut.map((p) => this._project(p)), item.polygon.map((p) => this._project(p)), smooth(item.corner.id)),
        depth: this._toView(item.centre)[2],
      }))
      .sort((a, b) => a.depth - b.depth);
    for (const item of discs) {
      this._path(item.polygon);
      context.fillStyle = hover === item.corner.id ? palette.highlight : palette.front;
      context.fill();
    }

    // 3. The near side: each face's frame plate and centre plate, the centre with the diagonal an STL splits a face
    //    on (fading as a clicked face smooths), then the labels sheared onto them.
    for (const item of near) {
      paintPlate(item, item.backPolygon, palette.back);
      paintPlate(item, item.frontPolygon, palette.front);
    }
    for (const item of near) {
      if (hover === item.face.id || item.t >= 1) continue;
      const a = item.frontPolygon[5], b = item.frontPolygon[25];
      context.save();
      context.globalAlpha = 1 - item.t;
      context.strokeStyle = palette.seam;
      context.lineWidth = 1.2 * k;
      context.beginPath();
      context.moveTo(a[0], a[1]);
      context.lineTo(b[0], b[1]);
      context.stroke();
      context.restore();
    }
    const label = (item) => {
      const lit = hover === item.face.id;
      const centre = this._project(item.centre);
      const rEnd = this._project(add(item.centre, item.face.r));
      const oEnd = this._project(add(item.centre, item.face.o));
      const rx = rEnd[0] - centre[0];
      const ry = rEnd[1] - centre[1];
      const ox = oEnd[0] - centre[0];
      const oy = oEnd[1] - centre[1];
      // A face seen nearly edge-on has no room for a label; theirs simply
      // becomes a sliver, and so does this.
      context.save();
      context.setTransform(rx, ry, -ox, -oy, centre[0], centre[1]);
      context.font = `${this.font.weight} ${TEXT_HEIGHT_PX}px ${this.font.family}`;
      context.textAlign = 'center';
      context.textBaseline = 'middle';
      context.fillStyle = lit ? palette.textHover : palette.textIdle;
      context.fillText(item.face.label, 0, 0.5);
      context.restore();
    };
    for (const item of near) label(item);
    // a corner under the pointer lights as its whole disc, over the faces, as long as it is hovered
    for (const item of discs) {
      if (hover !== item.corner.id) continue;
      this._path(item.polygon);
      context.fillStyle = palette.highlight;
      context.fill();
    }

    // 4. The triad: three lines the length of the cube's edges, 1px outside
    //    them. An edge is behind the cube when neither face it borders faces
    //    the camera, and then the line is stippled at 0.3.
    //    Their line width is a GL line width: ONE DEVICE PIXEL, whatever the
    //    display density. Measured: at 3x the line is a third of a CSS pixel
    //    and its pixels are the axis colour at about 0.57 over the canvas.
    const facing = new Map(faces.map((f) => [f.face.id, f.facing]));
    context.lineWidth = AXIS_WIDTH_PX;
    for (const axis of GEOMETRY.axes) {
      const from = this._project(axis.from);
      const to = this._project(axis.to);
      const visible = axis.faces.some((id) => facing.get(id));
      context.save();
      context.strokeStyle = palette[axis.colour];
      if (!visible) {
        context.globalAlpha = HIDDEN_AXIS_OPACITY;
        context.setLineDash([4 * k, 4 * k]);
      }
      context.beginPath();
      context.moveTo(from[0], from[1]);
      context.lineTo(to[0], to[1]);
      context.stroke();
      context.restore();
    }

    // 5. The axis labels: billboarded, 3px past the axis ends, fading out as
    //    the axis turns to point at the camera, dimmed behind the cube.
    // In canvas pixels here, not cube pixels: this text is not under the
    // face transform.
    context.font = `${this.font.axisWeight} ${AXIS_TEXT_HEIGHT_PX * k}px ${this.font.family}`;
    context.textAlign = 'center';
    context.textBaseline = 'middle';
    const forward = [this.rotation[2], this.rotation[5], this.rotation[8]];
    for (const axis of GEOMETRY.axes) {
      const angle = Math.acos(Math.min(1, Math.abs(dot(forward, axis.direction))));
      const fade = clamp((angle - AXIS_LABEL_FADE_END) / (AXIS_LABEL_FADE_START - AXIS_LABEL_FADE_END), 0, 1);
      if (fade <= 0) continue;
      const width = context.measureText(axis.label).width / k;
      const reach = 2 * HALF_PX + Math.hypot(width, AXIS_TEXT_HEIGHT_PX) / 2 + 3;
      const at = this._project(add(axis.from, scale(axis.direction, reach)));
      const behind = this._toView(axis.direction)[2] < 0
        && near.some((item) => pointInPolygon(at[0], at[1], item.outline));
      context.save();
      context.globalAlpha = fade * (behind ? HIDDEN_AXIS_OPACITY : 1);
      context.fillStyle = palette[axis.colour];
      context.fillText(axis.label, at[0], at[1]);
      context.restore();
    }

    // 6. The controls, in screen space.
    for (const control of CONTROLS) {
      const lit = hover === control.id;
      context.fillStyle = lit ? palette.highlight : palette.arrow;
      for (const polygon of GEOMETRY.controls[control.id]) {
        const points = polygon.map((p) => this._screen(p));
        this._path(points);
        context.fill();
        this.hits.push({ id: control.id, kind: 'control', control, polygon: points });
      }
    }
    // Pick order: the controls; a near face's centre plate; a corner, over a target a little wider than its rivet so
    // it can be hit; the rest of the near face, its frame; then the far faces through the seams.
    for (const item of near) this.hits.push({ id: item.face.id, kind: 'face', region: item.face, polygon: item.frontPolygon });
    for (const item of discs) {
      const c = this._project(item.centre);
      const target = item.polygon.map((p) => [c[0] + (p[0] - c[0]) * 1.6, c[1] + (p[1] - c[1]) * 1.6]);
      this.hits.push({ id: item.corner.id, kind: 'corner', region: item.corner, polygon: target });
    }
    for (const item of near) this.hits.push({ id: item.face.id, kind: 'face', region: item.face, polygon: item.backPolygon });
    for (const item of far) this.hits.push({ id: item.face.id, kind: 'face', region: item.face, polygon: item.backPolygon });

    context.restore();
    if (moving) this._frame ??= requestAnimationFrame(() => { this._frame = null; this.render(); });
    else this._last = 0;
  }

  hitTest(x, y) {
    for (const hit of this.hits) {
      if (pointInPolygon(x, y, hit.polygon)) return hit;
    }
    return null;
  }

  _local(event) {
    const rect = this.canvas.getBoundingClientRect();
    return [
      ((event.clientX - rect.left) / rect.width) * this.canvas.width,
      ((event.clientY - rect.top) / rect.height) * this.canvas.height,
    ];
  }

  _onMove(event) {
    const [x, y] = this._local(event);
    const hit = this.hitTest(x, y);
    const changed = (hit?.id ?? '') !== (this.hover?.id ?? '');
    this.hover = hit;
    if (changed) this.render();
    this.canvas.style.cursor = hit ? 'pointer' : 'default';
    this.canvas.title = hit?.kind === 'control' ? hit.control.title : '';
  }

  _onDown(event) {
    event.preventDefault();
    event.stopPropagation();
    if (event.button !== 0) return;
    const [x, y] = this._local(event);
    const hit = this.hitTest(x, y);
    if (!hit) return;
    if (hit.kind === 'control') {
      const angle = event.shiftKey
        ? ROTATE_ANGLES.coarse
        : event.ctrlKey
          ? ROTATE_ANGLES.fine
          : ROTATE_ANGLES.default;
      this.onRotate(hit.control.direction, angle);
    } else {
      // smooth it once the camera has turned to it (the turn itself passes through other orientations)
      this.active = hit.region.id;
      this.settled = false;
      setTimeout(() => { this.settled = true; }, ANIMATION_MS.long + 100);
      this.onView(hit.region.view, hit.region);
    }
  }
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function pointInPolygon(x, y, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
    const [xi, yi] = polygon[i];
    const [xj, yj] = polygon[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

/**
 * The label font. Theirs is Flama at 12px, regular. `--os-font` carries the
 * measured fallback chain; without Flama the fallbacks run light at this size,
 * so the weight steps up to keep the labels' visual weight.
 */
function resolveFont(canvas) {
  let family = '';
  try {
    family = getComputedStyle(canvas).getPropertyValue('--os-font').trim();
  } catch {
    family = '';
  }
  if (!family) {
    family = 'Flama, system-ui, "Segoe UI", "Open Sans", "Helvetica Neue", Helvetica, Arial, sans-serif';
  }
  let hasFlama = false;
  try {
    hasFlama = document.fonts?.check?.('12px Flama') ?? false;
  } catch {
    hasFlama = false;
  }
  return { family, weight: hasFlama ? 400 : 600, axisWeight: hasFlama ? 400 : 700 };
}

// ---------------------------------------------------------------------------
// the dropdown under the cube: "Camera and render options" — built by
// shell.js's mountViewCube from menu.js; only the icon lives here
// ---------------------------------------------------------------------------

/** Their `#svg-icon-view-cube-button`, verbatim, with its static four-tone fills. */
export const VIEW_CUBE_ICON_SVG =
  '<svg width="20" height="20" viewBox="0 0 20 20" aria-hidden="true">'
  + '<path d="M10 1.6L17.6 5.8L17.6 14.4L10 18.6L2.4 14.4L2.4 5.8Z" fill="#170c20"></path>'
  + '<path d="M2.9 6.6L9.5 10.3L9.5 17.8L2.9 14.1Z" fill="#8a4fe0"></path>'
  + '<path d="M10.5 10.3L17.1 6.6L17.1 14.1L10.5 17.8Z" fill="#b98cff"></path>'
  + '<path d="M10 2.2L16.7 5.9L10 9.6L3.3 5.9Z" fill="#f5effb"></path>'
  + '</svg>';
