/**
 * makeFixtures.mjs — generate STL test parts with exact analytic ground truth.
 *
 * Each fixture is a surface of revolution built from a 2D (r, z) profile. Every
 * profile segment revolves into exactly one analytic surface whose type and
 * parameters are known in closed form:
 *
 *   dz == 0            -> plane   (annulus, normal +/-Z)
 *   dr == 0            -> cylinder radius r
 *   otherwise          -> cone    half-angle = atan(|dr/dz|)
 *   arc segment        -> sphere  radius R
 *
 * Revolving guarantees a genuinely connected mesh with real dihedral creases
 * between segments, so this exercises segmentation as well as fitting — unlike
 * feeding the fitters isolated synthetic point clouds.
 *
 * Usage:  node test/makeFixtures.mjs
 * Writes: test/fixtures/<name>.stl  +  test/fixtures/<name>.truth.json
 */

import { writeFileSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const OUT = join(HERE, 'fixtures');

const DEG = Math.PI / 180;

/**
 * Revolve a profile into triangles, tagging each with its source segment.
 *
 * @param {Array<{r:number,z:number}>} profile  closed loop of (r,z) points
 * @param {Array<object>} truth                 one entry per segment
 * @param {number} nTheta                       revolution steps
 * @returns {{positions:number[], segOf:number[]}}
 */
function revolve(profile, truth, nTheta = Number(process.env.NTHETA || 96)) {
  const positions = [];
  const segOf = [];
  const at = (r, th, z) => [r * Math.cos(th), r * Math.sin(th), z];

  for (let s = 0; s < profile.length; s++) {
    const a = profile[s];
    const b = profile[(s + 1) % profile.length];
    const arc = truth[s].a0 !== undefined ? truth[s] : null;

    // Subdivide along the profile so curved segments are properly tessellated
    // and long straight ones do not produce slivers.
    const steps = arc ? (arc.steps ?? 16) : (truth[s].steps ?? 6);

    for (let i = 0; i < steps; i++) {
      const t0 = i / steps, t1 = (i + 1) / steps;
      const p0 = arc ? arcPoint(arc, t0) : lerp(a, b, t0);
      const p1 = arc ? arcPoint(arc, t1) : lerp(a, b, t1);

      for (let j = 0; j < nTheta; j++) {
        const th0 = (j / nTheta) * 2 * Math.PI;
        const th1 = ((j + 1) / nTheta) * 2 * Math.PI;

        const A = at(p0.r, th0, p0.z), B = at(p0.r, th1, p0.z);
        const C = at(p1.r, th1, p1.z), D = at(p1.r, th0, p1.z);

        // Outward winding: profile runs counter-clockwise in (r,z).
        if (p0.r > 1e-9) { positions.push(...A, ...B, ...C); segOf.push(s); }
        if (p1.r > 1e-9) { positions.push(...A, ...C, ...D); segOf.push(s); }
      }
    }
  }
  return { positions, segOf };
}

const lerp = (a, b, t) => ({ r: a.r + (b.r - a.r) * t, z: a.z + (b.z - a.z) * t });

function arcPoint(arc, t) {
  const ang = arc.a0 + (arc.a1 - arc.a0) * t;
  return { r: arc.cr + arc.radius * Math.cos(ang), z: arc.cz + arc.radius * Math.sin(ang) };
}

/** Derive ground truth for a straight profile segment. */
function straightTruth(a, b, steps) {
  const dr = b.r - a.r, dz = b.z - a.z;
  if (Math.abs(dz) < 1e-12) {
    return { type: 'plane', normalZ: dr < 0 ? 1 : -1, steps };
  }
  if (Math.abs(dr) < 1e-12) {
    return { type: 'cylinder', radius: Math.abs(a.r), steps };
  }
  return { type: 'cone', halfAngleDeg: Math.atan(Math.abs(dr / dz)) / DEG, steps };
}

function buildFixture(name, profile, overrides = {}) {
  const truth = profile.map((p, i) => {
    const q = profile[(i + 1) % profile.length];
    return overrides[i] ?? straightTruth(p, q, 6);
  });
  const { positions, segOf } = revolve(profile, truth);
  return { name, positions, segOf, truth };
}

// ── Binary STL writer ────────────────────────────────────────────────────────

function writeSTL(path, positions) {
  const tris = positions.length / 9;
  const buf = Buffer.alloc(84 + tris * 50);
  buf.write('binary STL - stlToStep test fixture', 0, 'ascii');
  buf.writeUInt32LE(tris, 80);

  for (let t = 0; t < tris; t++) {
    const o = 84 + t * 50;
    const p = positions.slice(t * 9, t * 9 + 9);
    // Deliberately write a ZERO facet normal. Many real exporters do exactly
    // this, and it is the condition under which the original pipeline — which
    // trusts geometry.attributes.normal unvalidated — produces silent garbage.
    buf.writeFloatLE(0, o); buf.writeFloatLE(0, o + 4); buf.writeFloatLE(0, o + 8);
    for (let i = 0; i < 9; i++) buf.writeFloatLE(p[i], o + 12 + i * 4);
    buf.writeUInt16LE(0, o + 48);
  }
  writeFileSync(path, buf);
  return tris;
}

// ── Fixtures ─────────────────────────────────────────────────────────────────

const fixtures = [];

// 1. Countersunk plate. The 45 deg countersink is the canonical case the
//    original fitCone rejects outright (threshold is 35.26 deg).
fixtures.push(buildFixture('countersunk-plate', [
  { r: 4,  z: 0  },   // 0 bottom annulus  -> plane
  { r: 20, z: 0  },   // 1 outer wall      -> cylinder r=20
  { r: 20, z: 10 },   // 2 top annulus     -> plane
  { r: 8,  z: 10 },   // 3 countersink     -> cone 45 deg
  { r: 4,  z: 6  },   // 4 bore            -> cylinder r=4
]));

// 2. Cone angle sweep across the real machining angles, bracketing the
//    35.26 deg failure threshold on both sides. Built by walking z upward so
//    the profile stays monotonic (a profile that reverses in z revolves into
//    self-intersecting, non-manifold geometry).
const sweep = (() => {
  const BORE = 2;
  const pts = [{ r: BORE, z: 0 }];
  let r = 30, z = 0;
  pts.push({ r, z });                                  // bottom annulus -> plane
  for (const deg of [15, 41, 59]) {                    // 30/82/118 deg tools
    z += 5;             pts.push({ r, z });            // cylinder
    const dr = 8;
    z += dr / Math.tan(deg * DEG); r -= dr;
    pts.push({ r, z });                                // cone at `deg`
  }
  z += 5; pts.push({ r, z });                          // final cylinder
  pts.push({ r: BORE, z });                            // top annulus -> plane
  return pts;                                          // wraps to bore cylinder
})();
fixtures.push(buildFixture('cone-sweep', sweep));

// 3. Spherical seat, to confirm the normal-concurrency sphere fit.
//    The arc starts at 20 deg so its junction with the cylinder is a real
//    crease (segmentation is not the thing under test here).
//    Segment 2 is the arc; segment 3 lies on the axis (r=0) and vanishes.
const A0 = 20 * DEG, SR = 12;
fixtures.push(buildFixture('ball-seat', [
  { r: 0,             z: -12 },
  { r: SR * Math.cos(A0), z: -12 },
  { r: SR * Math.cos(A0), z: SR * Math.sin(A0) },
  { r: 0,             z: SR },
], {
  2: { type: 'sphere', radius: SR, cr: 0, cz: 0, a0: A0, a1: Math.PI / 2, steps: 20 },
}));

// 4. Tangent fillet — cylinder blending into a top face through a quarter round.
//    BOTH junctions are tangent-continuous, so crease-angle flood fill cannot
//    separate them at any threshold. This is a known limitation of dihedral
//    segmentation, kept here as an explicit regression case rather than hidden.
//    The arc is centred off-axis, so in 3D it revolves into a TORUS, which the
//    classifier has no primitive for — expected to fall out as 'nurbs'.
fixtures.push(buildFixture('tangent-fillet', [
  { r: 0,  z: 0  },
  { r: 20, z: 0  },
  { r: 20, z: 10 },
  { r: 16, z: 14 },
  { r: 0,  z: 14 },
], {
  2: { type: 'torus', radius: 4, cr: 16, cz: 10, a0: 0, a1: Math.PI / 2, steps: 12 },
}));

mkdirSync(OUT, { recursive: true });
for (const f of fixtures) {
  const stl = join(OUT, f.name + '.stl');
  const tris = writeSTL(stl, f.positions);
  writeFileSync(join(OUT, f.name + '.truth.json'),
    JSON.stringify({ name: f.name, truth: f.truth, segOf: f.segOf }, null, 2));
  console.log(`${f.name.padEnd(20)} ${String(tris).padStart(6)} triangles, ` +
              `${f.truth.length} segments: ${f.truth.map(t => t.type).join(', ')}`);
}
console.log('\nwrote fixtures to ' + OUT);
