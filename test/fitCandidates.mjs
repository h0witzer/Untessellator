/**
 * fitCandidates.mjs — every analytic surface type fitted to given triangle groups, for the solid
 * rebuild's repair loop (ml/solid_rebuild.py build_repaired).
 *
 * Usage:  echo '{"pos": "<file>.pos", "groups": {"7": [12, 13, 40]}}' | node test/fitCandidates.mjs
 *
 * Reads one JSON request on stdin: the mesh's .pos file (float32 triangle soup) and, per group id,
 * its triangle indices in that file. Writes, per group, gaussMap.fitRegionAs for plane, cylinder,
 * cone, sphere and torus, best residual first: {"7": [{type, params, residual, relResidual}, ...]}.
 * Types that do not fit (too few triangles, degenerate) are left out.
 *
 * WHY. A face the kernel rejects is often on the wrong surface type (an 89 deg "cone" that is a
 * plane), and a group the painter left untyped has no surface at all. The rebuild then needs the
 * alternatives, from the same fitter the export path uses, to try in turn.
 */

import { readFileSync } from 'node:fs';
import { preparePaintPositions } from '../js/paintCore.js';
import { fitRegionAs } from '../js/gaussMap.js';

const TYPES = ['plane', 'cylinder', 'cone', 'sphere', 'torus'];

const req = JSON.parse(readFileSync(0, 'utf8'));
const buf = readFileSync(req.pos);
const P = Float64Array.from(new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4));
const { facets } = preparePaintPositions(P);

const out = {};
for (const [gid, tris] of Object.entries(req.groups)) {
  const found = [];
  for (const type of req.types ?? TYPES) {
    let fit;
    try {
      fit = fitRegionAs(type, new Set(tris), facets);
    } catch {
      continue;
    }
    if (!fit?.params || !isFinite(fit.residual)) continue;
    found.push({ type, params: fit.params, residual: fit.residual, relResidual: fit.relResidual });
  }
  found.sort((a, b) => a.residual - b.residual);
  out[gid] = found;
}
process.stdout.write(JSON.stringify(out));
