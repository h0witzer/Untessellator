/**
 * parseStep.mjs — minimal AP203/214/242 reader for ground-truth extraction.
 *
 * Pulls every ADVANCED_FACE, resolves its underlying surface entity, and
 * returns the analytic parameters. Enough to score a reverse-engineering
 * pipeline against a real CAD export; not a general STEP implementation.
 */

import { readFileSync } from 'node:fs';

/** Split the DATA section into `#id = TYPE(args)` records. */
function parseEntities(text) {
  const dataStart = text.indexOf('DATA;');
  const body = text.slice(dataStart < 0 ? 0 : dataStart + 5)
    .replace(/\/\*[\s\S]*?\*\//g, '');          // strip block comments

  const entities = new Map();
  let stmt = '';
  let inStr = false;

  for (let i = 0; i < body.length; i++) {
    const c = body[i];
    if (c === "'") inStr = !inStr;
    if (c === ';' && !inStr) {
      const m = /^\s*#(\d+)\s*=\s*([A-Z0-9_]+)\s*\(([\s\S]*)\)\s*$/.exec(stmt);
      if (m) entities.set(+m[1], { type: m[2], raw: m[3] });
      stmt = '';
    } else {
      stmt += c;
    }
  }
  return entities;
}

/** Split an argument list on top-level commas, respecting quotes and nesting. */
function splitArgs(raw) {
  const out = [];
  let depth = 0, cur = '', inStr = false;
  for (let i = 0; i < raw.length; i++) {
    const c = raw[i];
    if (c === "'") inStr = !inStr;
    if (!inStr) {
      if (c === '(') depth++;
      else if (c === ')') depth--;
      else if (c === ',' && depth === 0) { out.push(cur.trim()); cur = ''; continue; }
    }
    cur += c;
  }
  if (cur.trim()) out.push(cur.trim());
  return out;
}

const refOf = s => { const m = /^#(\d+)$/.exec(s.trim()); return m ? +m[1] : null; };
const tripleOf = s => {
  const m = /\(([^)]*)\)/.exec(s);
  return m ? m[1].split(',').map(Number) : null;
};

/**
 * Read a STEP file and return its analytic face inventory.
 *
 * @param {string} path
 * @returns {{faces: Array<object>, counts: Record<string, number>}}
 */
export function parseStepFaces(path) {
  const entities = parseEntities(readFileSync(path, 'latin1'));
  const args = id => {
    const e = entities.get(id);
    return e ? splitArgs(e.raw) : null;
  };

  /** AXIS2_PLACEMENT_3D -> { location, axis } */
  const placement = (id) => {
    const a = args(id);
    if (!a) return null;
    const loc = refOf(a[1]), ax = a[2] ? refOf(a[2]) : null;
    const locE = loc !== null ? entities.get(loc) : null;
    const axE = ax !== null ? entities.get(ax) : null;
    return {
      location: locE ? tripleOf(locE.raw) : null,
      axis: axE ? tripleOf(axE.raw) : null,
    };
  };

  const SURF = new Set(['PLANE', 'CYLINDRICAL_SURFACE', 'CONICAL_SURFACE',
    'SPHERICAL_SURFACE', 'TOROIDAL_SURFACE', 'B_SPLINE_SURFACE_WITH_KNOTS',
    'B_SPLINE_SURFACE', 'SURFACE_OF_REVOLUTION', 'SURFACE_OF_LINEAR_EXTRUSION']);

  const faces = [];
  for (const [id, e] of entities) {
    if (e.type !== 'ADVANCED_FACE') continue;
    const a = splitArgs(e.raw);
    const sid = refOf(a[2]);
    const s = sid !== null ? entities.get(sid) : null;
    if (!s || !SURF.has(s.type)) { faces.push({ id, type: 'unknown' }); continue; }

    const sa = splitArgs(s.raw);
    const p = sa[1] ? placement(refOf(sa[1])) : null;
    const face = { id, stepId: sid, type: s.type, ...(p || {}) };

    if (s.type === 'CYLINDRICAL_SURFACE') face.radius = Number(sa[2]);
    else if (s.type === 'CONICAL_SURFACE') {
      face.radius = Number(sa[2]);
      face.halfAngle = Number(sa[3]);              // radians
    } else if (s.type === 'SPHERICAL_SURFACE') face.radius = Number(sa[2]);
    else if (s.type === 'TOROIDAL_SURFACE') {
      face.majorRadius = Number(sa[2]);
      face.minorRadius = Number(sa[3]);
    }
    faces.push(face);
  }

  const counts = {};
  for (const f of faces) counts[f.type] = (counts[f.type] || 0) + 1;
  return { faces, counts };
}

// Simple map from STEP entity name to this project's primitive vocabulary.
export const STEP_TO_PRIMITIVE = {
  PLANE: 'plane',
  CYLINDRICAL_SURFACE: 'cylinder',
  CONICAL_SURFACE: 'cone',
  SPHERICAL_SURFACE: 'sphere',
  TOROIDAL_SURFACE: 'torus',
  B_SPLINE_SURFACE_WITH_KNOTS: 'bspline',
  B_SPLINE_SURFACE: 'bspline',
  SURFACE_OF_REVOLUTION: 'revolution',
  SURFACE_OF_LINEAR_EXTRUSION: 'extrusion',
};
