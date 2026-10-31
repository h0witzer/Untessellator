/**
 * caps.js — the triangles js/meshRepair.js added (hole caps; zero-area fills it kept are not among them), and
 * how the studio treats them. Pure: no GL, no DOM.
 *
 * A cap is a fan from a hole's centroid: it closes the shell but says nothing about the shape, and on a curved
 * face its centre lies inside the surface. Measured on 12 Onshape parts with a 1% patch cut out and repaired
 * (test/meshDamage.mjs, test/studioExport.mjs --synthetic): painted like any other triangles, caps were split
 * into a dozen extra groups and made curved groups inexact (8 of 12 exports fully analytic, 34 s each); with
 * the rules below, 11 of 12 and 7 s, all 12 closed solids. So:
 *
 *   seeds      a cap never seeds an auto-paint group
 *   groups     each connected run of caps joins the group most of the input triangles around it belong to
 *   fits       a group's surface is fitted and typed on its input triangles only
 *   export     a vertex that only caps use is moved onto its group's fitted surface
 */

import { distNormal } from '../knit/geometry.js';

/**
 * The group each run of cap triangles should be in: Map triangle -> group, by vote of the input triangles
 * around the run (model.neighbours: the triangle across each edge, -1 for none). Runs with no painted
 * neighbour are left out.
 */
export function capOwners(owner, synthetic, neighbours) {
  const out = new Map();
  if (!synthetic) return out;
  const seen = new Uint8Array(owner.length);
  for (let s = 0; s < owner.length; s++) {
    if (!synthetic[s] || seen[s]) continue;
    const run = [s], votes = new Map();
    seen[s] = 1;
    for (let i = 0; i < run.length; i++) {
      for (let k = 0; k < 3; k++) {
        const nb = neighbours[run[i] * 3 + k];
        if (nb < 0) continue;
        if (!synthetic[nb]) { if (owner[nb] >= 0) votes.set(owner[nb], (votes.get(owner[nb]) ?? 0) + 1); continue; }
        if (!seen[nb]) { seen[nb] = 1; run.push(nb); }
      }
    }
    let best = -1, n = 0;
    for (const [g, c] of votes) if (c > n) { n = c; best = g; }
    if (best >= 0) for (const t of run) if (owner[t] !== best) out.set(t, best);
  }
  return out;
}

/** A group's triangles to fit and type on: its input triangles, or all of them when it has no others. */
export function fitTriangles(tris, synthetic) {
  if (!synthetic) return tris;
  const kept = [...tris].filter((t) => !synthetic[t]);
  return kept.length ? new Set(kept) : tris;
}

/**
 * Move every vertex that only cap triangles use onto the fitted surface of the one group those caps are in.
 * `positions` (9 per triangle) is changed in place; `surfaces` is Map group -> [type, params] in the same frame.
 */
export function projectCapVertices(positions, labels, synthetic, surfaces) {
  if (!synthetic) return 0;
  const users = new Map();
  for (let t = 0; t < labels.length; t++) {
    for (let k = 0; k < 3; k++) {
      const o = t * 9 + k * 3;
      const key = `${positions[o]},${positions[o + 1]},${positions[o + 2]}`;
      let u = users.get(key);
      if (!u) users.set(key, (u = { cap: true, groups: new Set(), at: [] }));
      if (!synthetic[t]) u.cap = false;
      u.groups.add(labels[t]);
      u.at.push(o);
    }
  }
  let moved = 0;
  for (const u of users.values()) {
    if (!u.cap || u.groups.size !== 1) continue;
    const s = surfaces.get([...u.groups][0]);
    if (!s || !s[1] || s[0] === 'bspline') continue;
    const x = [positions[u.at[0]], positions[u.at[0] + 1], positions[u.at[0] + 2]];
    const [d, n] = distNormal(s[0], s[1], x);
    for (const o of u.at) for (let j = 0; j < 3; j++) positions[o + j] = x[j] - d * n[j];
    moved++;
  }
  return moved;
}
