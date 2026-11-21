/**
 * analytic.js — a painted group's analytic surface (the studio's [type, params], gaussMap's format) as a surface
 * faceSolid can bound like its own B-splines: eval, normal, closest point, domain, which directions close. Each is
 * parameterised as STEP (ISO 10303-42) parameterises it, so the pcurves faceSolid fits on it are written as they
 * are (faceStep.js):
 *
 *   plane     C + u x + v y
 *   cylinder  C + R (cos u x + sin u y) + v z
 *   cone      C + (R + v tan a)(cos u x + sin u y) + v z          (v along the axis, R the radius at C)
 *   sphere    C + R cos v (cos u x + sin u y) + R sin v z
 *   torus     C + (R + r cos v)(cos u x + sin u y) + r sin v z
 *
 * u (and a torus's v) are angles, periodic in 2 pi. The frame (C, z the axis, x) is chosen for the face's points
 * as the knitter chooses it (js/knit/build.js geomSurface): u = 0 away from the points, a sphere's poles off them,
 * a cone's C on the axis beside them; rotate() moves u = 0 (faceSolid puts a seam where its loops cross it once).
 */

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => Math.hypot(a[0], a[1], a[2]);
const unit = (a) => { const n = norm(a); return n > 0 ? mul(a, 1 / n) : a; };
const TAU = 2 * Math.PI;
const comb = (a, x, b, y) => [a * x[0] + b * y[0], a * x[1] + b * y[1], a * x[2] + b * y[2]];

/** A unit vector perpendicular to z, toward w where w has a part across z (else any). */
function across(z, w) {
  const p = sub(w, mul(z, dot(w, z)));
  if (norm(p) > 1e-9 * Math.max(norm(w), 1e-300)) return unit(p);
  const ref = Math.abs(z[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  return unit(cross(z, ref));
}

export const ANALYTIC = new Set(['plane', 'cylinder', 'cone', 'sphere', 'torus']);

export class AnalyticSurface {
  /**
   * type, params: the studio's surface; pts: the face's vertices (to frame it). The frame: z the axis (a plane's
   * normal; a sphere's: across the points' mean direction, so its poles are off them), x pointing away from the
   * points' mean (u = 0 opposite them).
   */
  constructor(type, params, pts) {
    this.type = type;
    const c = [0, 0, 0];
    for (const q of pts) for (let k = 0; k < 3; k++) c[k] += q[k] / pts.length;
    this.R = 0; this.r = 0; this.tanA = 0;
    if (type === 'plane') {
      this.z = unit(params.normal);
      this.C = add(params.origin, mul(this.z, dot(sub(c, params.origin), this.z)));
      this.x = across(this.z, [1, 0.37, 0.11]);
    } else if (type === 'sphere') {
      this.C = params.center; this.R = params.radius;
      const mean = unit(sub(c, this.C));
      this.z = across(mean, [0.11, 0.37, 1]);          // the poles off the points (across their mean direction)
      this.x = mul(across(this.z, mean), -1);
    } else if (type === 'cylinder' || type === 'cone' || type === 'torus') {
      let z = unit(params.axis);
      const o = type === 'cylinder' ? params.axisPoint : type === 'cone' ? params.apex : params.center;
      if (type === 'cone') {
        // z toward the points' nappe; C on the axis level with their mean, R the radius there
        let m = 0;
        for (const q of pts) m += dot(sub(q, o), z);
        if (m < 0) z = mul(z, -1);
        this.tanA = Math.tan(params.halfAngle);
        const h = dot(sub(c, o), z);
        this.C = add(o, mul(z, h));
        this.R = h * this.tanA;
      } else if (type === 'cylinder') {
        this.C = add(o, mul(z, dot(sub(c, o), z)));
        this.R = params.radius;
      } else {
        this.C = o; this.R = params.majorRadius; this.r = params.minorRadius;
      }
      this.z = z;
      this.x = mul(across(z, sub(c, this.C)), -1);
    } else throw new Error(`no analytic surface ${type}`);
    this.y = cross(this.z, this.x);
    this.periodicU = type !== 'plane';
    this.periodicV = type === 'torus';
    this.vRef = 0;
    // the domain: u a full turn from 0 (or the points' extent, a plane); v the points' extent with a margin (only
    // for what reads it: the seam's line and the size)
    const uv = pts.map((q) => this.param(q));
    if (this.periodicV) {
      // a torus's v window centred on the points' circular mean
      let sc = 0, ss = 0;
      for (const [, v] of uv) { sc += Math.cos(v); ss += Math.sin(v); }
      this.vRef = Math.atan2(ss, sc);
      for (const q of uv) q[1] = this.wrapV(q[1]);
    }
    let v0 = Infinity, v1 = -Infinity, u0 = Infinity, u1 = -Infinity;
    for (const [u, v] of uv) { v0 = Math.min(v0, v); v1 = Math.max(v1, v); u0 = Math.min(u0, u); u1 = Math.max(u1, u); }
    const mv = 0.1 * (v1 - v0) + 1e-9, mu = 0.1 * (u1 - u0) + 1e-9;
    this.dom = this.periodicU ? [0, TAU, v0 - mv, v1 + mv] : [u0 - mu, u1 + mu, v0 - mv, v1 + mv];
    let s = 0;
    for (const q of pts) s = Math.max(s, norm(sub(q, c)));
    this._size = 2 * s;
  }

  size() { return this._size; }
  closed() { return [this.periodicU, this.periodicV]; }
  wrapV(v) { return this.vRef - Math.PI + ((((v - this.vRef + Math.PI) % TAU) + TAU) % TAU); }

  /** Moves u = 0 to angle t of the present frame (a point at u is then at u - t). */
  rotate(t) {
    const x = comb(Math.cos(t), this.x, Math.sin(t), this.y);
    this.y = cross(this.z, x);
    this.x = x;
  }

  /** (u, v) of the surface point nearest x: u in [0, 2 pi) when periodic, a torus's v in its window. */
  param(p) {
    const d = sub(p, this.C), X = dot(d, this.x), Y = dot(d, this.y), Z = dot(d, this.z);
    const fold = (u) => (((u % TAU) + TAU) % TAU);
    switch (this.type) {
      case 'plane': return [X, Y];
      case 'cylinder': return [fold(Math.atan2(Y, X)), Z];
      case 'cone': {
        // nearest on the generator through p's azimuth: rho = R + v tanA, z = v; projected onto that line
        const rho = Math.hypot(X, Y), g = Math.hypot(this.tanA, 1);
        const s = ((rho - this.R) * this.tanA + Z) / g;          // along the unit generator (tanA, 1) / g
        return [fold(Math.atan2(Y, X)), s / g];
      }
      case 'sphere': return [fold(Math.atan2(Y, X)), Math.atan2(Z, Math.hypot(X, Y))];
      default: {
        const u = Math.atan2(Y, X), e = comb(Math.cos(u), this.x, Math.sin(u), this.y);
        const w = sub(d, mul(e, this.R));
        return [fold(u), this.wrapV(Math.atan2(dot(w, this.z), dot(w, e)))];
      }
    }
  }

  /** {S, Su, Sv, Suu, Suv, Svv} at (u, v) (second derivatives when nder >= 2). */
  eval(u, v, nder = 0) {
    const { C, x, y, z } = this;
    const cu = Math.cos(u), su = Math.sin(u);
    const e = comb(cu, x, su, y), ep = comb(-su, x, cu, y);        // radial and its u derivative
    let S, Su, Sv, Suu, Suv, Svv;
    const zero = [0, 0, 0];
    switch (this.type) {
      case 'plane':
        S = add(C, comb(u, x, v, y)); Su = x; Sv = y; Suu = Suv = Svv = zero; break;
      case 'cylinder':
        S = add(add(C, mul(e, this.R)), mul(z, v)); Su = mul(ep, this.R); Sv = z;
        Suu = mul(e, -this.R); Suv = zero; Svv = zero; break;
      case 'cone': {
        const rho = this.R + v * this.tanA;
        S = add(add(C, mul(e, rho)), mul(z, v)); Su = mul(ep, rho); Sv = add(mul(e, this.tanA), z);
        Suu = mul(e, -rho); Suv = mul(ep, this.tanA); Svv = zero; break;
      }
      case 'sphere': {
        const R = this.R, cv = Math.cos(v), sv = Math.sin(v);
        S = add(C, add(mul(e, R * cv), mul(z, R * sv))); Su = mul(ep, R * cv); Sv = add(mul(e, -R * sv), mul(z, R * cv));
        Suu = mul(e, -R * cv); Suv = mul(ep, -R * sv); Svv = add(mul(e, -R * cv), mul(z, -R * sv)); break;
      }
      default: {
        const { R, r } = this, cv = Math.cos(v), sv = Math.sin(v), rho = R + r * cv;
        S = add(C, add(mul(e, rho), mul(z, r * sv))); Su = mul(ep, rho); Sv = add(mul(e, -r * sv), mul(z, r * cv));
        Suu = mul(e, -rho); Suv = mul(ep, -r * sv); Svv = add(mul(e, -r * cv), mul(z, -r * sv));
      }
    }
    return nder >= 2 ? { S, Su, Sv, Suu, Suv, Svv } : { S, Su, Sv };
  }

  /** [point, unit normal (Su x Sv)] at (u, v). */
  normal(u, v) {
    const d = this.eval(u, v, 1), n = cross(d.Su, d.Sv), l = norm(n) || 1e-300;
    return [d.S, mul(n, 1 / l)];
  }

  /**
   * The nearest point to x: {u, v, S}, exact (no iteration). uv0: a seed whose u (and a torus's v) the answer is
   * given nearest to, modulo a turn (callers unwrap along curves).
   */
  closest(p, uv0 = null) {
    let [u, v] = this.param(p);
    if (uv0 && this.periodicU) u += TAU * Math.round((uv0[0] - u) / TAU);
    if (uv0 && this.periodicV) v += TAU * Math.round((uv0[1] - v) / TAU);
    return { u, v, S: this.eval(u, v).S };
  }

  signedDistance(p) {
    const { u, v, S } = this.closest(p), n = this.normal(u, v)[1];
    return [dot(sub(p, S), n), n];
  }

  /** What faceStep writes: the entity, its frame and radii. */
  stepParams() {
    return { analytic: { type: this.type, C: this.C, z: this.z, x: this.x, R: this.R, r: this.r, angle: Math.atan(this.tanA) } };
  }
}
