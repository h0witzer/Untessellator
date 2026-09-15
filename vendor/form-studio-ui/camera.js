// SPDX-License-Identifier: LGPL-2.1-or-later
//
// The camera, with Onshape's rotation model.
//
// READ OUT OF THE PRODUCT, 2026-09-11, by `tools/viewcube-extract.mjs`: the
// `ViewController` class in the live client bundle, its `onDrag`, `rotateAbout`,
// `rotateInDirection`, `animateRotation` and `animateTo`, and the constants they
// read from the appearance table. Then checked by right-dragging the real
// product and photographing its cube: 60px right turned the model 40 degrees
// to the right about world Z; 60px down tipped its top toward the viewer;
// after 120px right its Z axis was still vertical on screen.
//
//     ROTATION_CYCLES_PER_SCREEN     3       a drag across the canvas diagonal is
//                                            three full turns: 2 * pi * 3 / hypot(w, h)
//                                            radians per pixel
//     ROTATE_ANGLE                   15 deg  an arrow click; shift 90, ctrl 5
//     CAMERA_SNAP_ANIMATION_TIME_MS  125
//     CAMERA_SHORT_ANIMATION_TIME_MS 250     arrow clicks, linear
//     CAMERA_LONG_ANIMATION_TIME_MS  500     standard views, smootherstep
//     AXIS_ROTATION_SNAP_TOLERANCE   0.001
//
// THE ROTATION IS A TURNTABLE ABOUT WORLD Z, AND THE SIGNS ARE THE MODEL'S
// ------------------------------------------------------------------------
// The product's axis-rotate mode (the one measured) is, per drag pixel:
//
//     rotateAbout(-dx * r * sign(up.z), worldZ)
//     rotateAbout(-dy * r, cameraRight)
//
// applied to the camera's direction, up and eye about the rotation centre.
// The model follows the mouse: drag right and the near face moves right,
// which is the camera moving LEFT around Z, which is a negative angle. The
// camera this replaced added dx to a bearing, so it turned the other way,
// which is the "Z orbit is backwards" that was felt on the first drag.
//
// The `sign(up.z)` is what lets the camera go over the top: past the pole the
// image is upside down and the horizontal sense flips, so the drag still
// follows the mouse. The camera is never clamped. If the camera has been
// rolled (the curved arrows do that), the next drag first levels it, over
// 125ms or 250ms if it is more than 45 degrees off, and ignores the mouse
// until it is level. There is no inertia in this mode.
//
// Arrow clicks are about the CAMERA's axes, not the world's:
//
//     left   +angle about up          right  -angle about up
//     up     +angle about right       down   -angle about right
//     cw     -angle about direction   ccw    +angle about direction
//
// Standard views are reached by interpolating orientation over 500ms with
// their easing, t^3 (t (6t - 15) + 10).

const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const scale = (a, k) => [a[0] * k, a[1] * k, a[2] * k];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const norm = (a) => {
  const length = Math.hypot(...a) || 1;
  return scale(a, 1 / length);
};

function rotateVector(v, angle, axis) {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const k = norm(axis);
  return add(add(scale(v, c), scale(cross(k, v), s)), scale(k, dot(k, v) * (1 - c)));
}

export const ROTATION_CYCLES_PER_SCREEN = 3;
export const AXIS_ROTATION_SNAP_TOLERANCE = 0.001;
export const ANIMATION_MS = Object.freeze({ snap: 125, short: 250, long: 500 });

const WORLD_UP = [0, 0, 1];

/** A unit quaternion [x, y, z, w] from an orthonormal basis given as columns. */
function quaternionFromBasis(right, up, back) {
  const m00 = right[0]; const m01 = up[0]; const m02 = back[0];
  const m10 = right[1]; const m11 = up[1]; const m12 = back[1];
  const m20 = right[2]; const m21 = up[2]; const m22 = back[2];
  const trace = m00 + m11 + m22;
  let q;
  if (trace > 0) {
    const s = Math.sqrt(trace + 1) * 2;
    q = [(m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, s / 4];
  } else if (m00 > m11 && m00 > m22) {
    const s = Math.sqrt(1 + m00 - m11 - m22) * 2;
    q = [s / 4, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s];
  } else if (m11 > m22) {
    const s = Math.sqrt(1 + m11 - m00 - m22) * 2;
    q = [(m01 + m10) / s, s / 4, (m12 + m21) / s, (m02 - m20) / s];
  } else {
    const s = Math.sqrt(1 + m22 - m00 - m11) * 2;
    q = [(m02 + m20) / s, (m12 + m21) / s, s / 4, (m10 - m01) / s];
  }
  const length = Math.hypot(...q) || 1;
  return q.map((v) => v / length);
}

/** The basis back from a quaternion: `{ right, up, back }`. */
function basisFromQuaternion([x, y, z, w]) {
  return {
    right: [1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)],
    up: [2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)],
    back: [2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)],
  };
}

function slerp(a, b, t) {
  let cosine = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  let end = b;
  if (cosine < 0) {
    cosine = -cosine;
    end = b.map((v) => -v);
  }
  if (cosine > 0.9995) {
    const out = a.map((v, i) => v + (end[i] - v) * t);
    const length = Math.hypot(...out) || 1;
    return out.map((v) => v / length);
  }
  const theta = Math.acos(cosine);
  const sinTheta = Math.sin(theta);
  const wa = Math.sin((1 - t) * theta) / sinTheta;
  const wb = Math.sin(t * theta) / sinTheta;
  return a.map((v, i) => v * wa + end[i] * wb);
}

/** Their standard-view easing. */
const smootherstep = (t) => t * t * t * (t * (6 * t - 15) + 10);

export class OrbitCamera {
  /**
   * @param {{ direction: number[], up: number[], distance?: number, center?: number[] }} init
   *   `direction` is the way the camera looks; `up` its up. A frame from
   *   `STANDARD_VIEWS` spreads straight in.
   */
  constructor(init) {
    this.direction = norm(init.direction);
    this.up = norm(init.up);
    this.distance = init.distance ?? 5;
    this.center = [...(init.center ?? [0, 0, 0])];
    this.animation = null;
    /** Frames left behind by standard-view moves, newest last, for "Previous view". */
    this.history = [];
    this._orthonormalise();
  }

  /** A snapshot of the orientation, for the history. */
  frame() {
    return { direction: [...this.direction], up: [...this.up] };
  }

  get right() {
    return norm(cross(this.direction, this.up));
  }

  get back() {
    return scale(this.direction, -1);
  }

  eye() {
    return sub(this.center, scale(this.direction, this.distance));
  }

  isAnimating() {
    return this.animation !== null;
  }

  /** Column-major 4x4 view matrix, the one the viewport and the cube both take. */
  viewMatrix() {
    const z = this.back;
    const x = this.right;
    const y = cross(z, x);
    const eye = this.eye();
    return new Float32Array([
      x[0], y[0], z[0], 0,
      x[1], y[1], z[1], 0,
      x[2], y[2], z[2], 0,
      -dot(x, eye), -dot(y, eye), -dot(z, eye), 1,
    ]);
  }

  /** Their `Camera.rotateAbout`: turn direction and up, and swing the eye about the centre. */
  rotateAbout(angle, axis) {
    if (!angle) return;
    this.direction = rotateVector(this.direction, angle, axis);
    this.up = rotateVector(this.up, angle, axis);
    this._orthonormalise();
  }

  _orthonormalise() {
    this.direction = norm(this.direction);
    const right = norm(cross(this.direction, this.up));
    this.up = norm(cross(right, this.direction));
  }

  /** Set the orientation outright. */
  setFrame(frame) {
    this.direction = norm(frame.direction);
    this.up = norm(frame.up);
    this._orthonormalise();
  }

  /** How far off level the camera is: the angle between right and the horizon. */
  rollError() {
    return Math.asin(Math.max(-1, Math.min(1, dot(this.right, WORLD_UP))));
  }

  /**
   * A right-drag of (dx, dy) canvas pixels over a canvas of (width, height).
   * Returns false while the camera is busy levelling itself and ignored the mouse.
   */
  drag(dx, dy, width, height) {
    if (this.animation) return false;
    const error = this.rollError();
    if (Math.abs(error) > 0) {
      const axis = cross(this.right, WORLD_UP);
      if (Math.abs(error) < AXIS_ROTATION_SNAP_TOLERANCE || Math.hypot(...axis) < 1e-9) {
        if (Math.hypot(...axis) >= 1e-9) this.rotateAbout(-error, axis);
      } else {
        const target = this.clone();
        target.rotateAbout(-error, axis);
        this.animateToFrame(target, Math.abs(error) > Math.PI / 4 ? ANIMATION_MS.short : ANIMATION_MS.snap);
        return false;
      }
    }
    const perPixel = (2 * Math.PI * ROTATION_CYCLES_PER_SCREEN) / (Math.hypot(width, height) || 1);
    const sense = Math.sign(this.up[2]) || 1;
    this.rotateAbout(-dx * perPixel * sense, WORLD_UP);
    this.rotateAbout(-dy * perPixel, this.right);
    return true;
  }

  /** An arrow click: their `rotateInDirection`. */
  rotateInDirection(direction, angle, ms = ANIMATION_MS.short) {
    const up = this.up;
    const forward = this.direction;
    const right = this.right;
    const table = {
      left: [up, angle],
      right: [up, -angle],
      up: [right, angle],
      down: [right, -angle],
      clockwise: [forward, -angle],
      counterclockwise: [forward, angle],
    };
    const entry = table[direction];
    if (!entry) return;
    this.animation = { kind: 'turn', axis: entry[0], total: entry[1], applied: 0, started: null, ms };
  }

  /** A standard view: interpolate there over `ms` with their easing. Ignored mid-animation, as theirs is. */
  animateToView(frame, ms = ANIMATION_MS.long) {
    if (this.animation) return false;
    this.history.push(this.frame());
    if (this.history.length > 32) this.history.shift();
    this.animateToFrame(frame, ms);
    return true;
  }

  /** Their "Previous view": back to where the last standard-view move started. */
  hasPrevious() {
    return this.history.length > 0;
  }

  animateToPrevious(ms = ANIMATION_MS.long) {
    if (this.animation || !this.history.length) return false;
    const frame = this.history.pop();
    this.animateToFrame(frame, ms);
    return true;
  }

  animateToFrame(frame, ms) {
    const from = quaternionFromBasis(this.right, this.up, this.back);
    const target = norm(frame.direction);
    const targetUp = norm(frame.up);
    const targetRight = norm(cross(target, targetUp));
    const to = quaternionFromBasis(targetRight, norm(cross(targetRight, target)), scale(target, -1));
    this.animation = { kind: 'view', from, to, started: null, ms };
  }

  clone() {
    return new OrbitCamera({ direction: this.direction, up: this.up, distance: this.distance, center: this.center });
  }

  /** Advance whatever is animating. Call once per frame. */
  step(now) {
    const animation = this.animation;
    if (!animation) return;
    if (animation.started === null) animation.started = now;
    const t = Math.min(1, (now - animation.started) / animation.ms);
    if (animation.kind === 'turn') {
      // Linear, like theirs.
      const reached = animation.total * t;
      this.rotateAbout(reached - animation.applied, animation.axis);
      animation.applied = reached;
    } else {
      const basis = basisFromQuaternion(slerp(animation.from, animation.to, smootherstep(t)));
      this.direction = scale(norm(basis.back), -1);
      this.up = norm(basis.up);
      this._orthonormalise();
    }
    if (t >= 1) this.animation = null;
  }
}
