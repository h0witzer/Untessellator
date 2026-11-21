/**
 * patchStep.js — STEP (AP214) text for closed solids of untrimmed Bezier patches (the output of
 * js/subd/fastNurbs.js or fitCage.js), written directly: no kernel. Each face is a patch over its whole parameter
 * square, bounded by its loop ([vertex, u, v] entries round the square: the corners, and a side's midpoint where
 * refined neighbours meet a coarse face). One vertex per loop vertex id, one edge per pair of consecutive loop
 * vertices, its 3D curve the patch side between them (a neighbour's side is the same curve with the same
 * parameter), a straight pcurve in each face's square: valid by construction, with nothing to project, sew or
 * repair. Faces point outward; a solid that points inward is turned over (loops reversed, patches transposed).
 * test/subdStep.mjs and test/meshToStep.mjs --check read the files back with OCCT as the judge.
 */
import { evalAt } from './bezier.js';

/** A STEP real: always a decimal point, 15 significant digits, E exponent. */
function real(x) {
  if (x === 0) return '0.';
  const [m0, e] = x.toPrecision(15).split('e');
  const m = m0.includes('.') ? m0.replace(/0+$/, '') : `${m0}.`;
  return e ? `${m}E${e}` : m;
}

const text = (s) => `'${String(s).replace(/'/g, "''")}'`;

/** de Casteljau: the poles of curve P (3D points) restricted to [s0, s1], reversed when s0 > s1. */
function subCurve(P, s0, s1) {
  if (s0 > s1) return subCurve(P, s1, s0).reverse();
  const split = (Q, t) => {                         // [left, right] at t
    const left = [], right = [];
    let row = Q.map((q) => [...q]);
    while (row.length) {
      left.push(row[0]); right.unshift(row[row.length - 1]);
      row = row.slice(1).map((q, i) => q.map((x, c) => (1 - t) * row[i][c] + t * x));
    }
    return [left, right];
  };
  let Q = P;
  if (s1 < 1) Q = split(Q, s1)[0];
  if (s0 > 0) Q = split(Q, s0 / s1)[1];
  return Q;
}

/** The poles of a patch along the straight segment uv0 -> uv1 of its boundary (u or v constant at 0 or 1). */
function segmentPoles(f, uv0, uv1) {
  const p = f.uCount - 1, at = (i, j) => { const o = (i * f.vCount + j) * 3; return [f.poles[o], f.poles[o + 1], f.poles[o + 2]]; };
  const idx = Array.from({ length: p + 1 }, (_, m) => m);
  if (uv0[0] === uv1[0]) {                         // u constant: along v
    const i = uv0[0] === 0 ? 0 : p;
    return subCurve(idx.map((m) => at(i, m)), uv0[1], uv1[1]);
  }
  const j = uv0[1] === 0 ? 0 : f.vCount - 1;
  return subCurve(idx.map((m) => at(m, j)), uv0[0], uv1[0]);
}

/** Six times the signed volume enclosed by the loops (fans through the patch points): positive when they face out. */
export function orientation(faces, loops) {
  let v6 = 0;
  loops.forEach((loop, f) => {
    const c = loop.map(([, u, v]) => { const D = evalAt(faces[f].uCount, faces[f].poles, u, v); return [D[0], D[1], D[2]]; });
    for (let i = 1; i + 1 < c.length; i++) {
      const [x, y, z] = [c[0], c[i], c[i + 1]];
      v6 += x[0] * (y[1] * z[2] - y[2] * z[1]) - x[1] * (y[0] * z[2] - y[2] * z[0]) + x[2] * (y[0] * z[1] - y[1] * z[0]);
    }
  });
  return v6;
}

/**
 * The STEP file: one MANIFOLD_SOLID_BREP per solid, named. solids: [{ faces, loops, name? }] (or one such object).
 * faces[f]: Bezier patch { uCount, vCount, poles (row-major, u then v, 3 each) }; loops[f]: [[vertex, u, v], ...]
 * counter-clockwise in (u, v).
 */
export function patchStep(solids, { name = 'Part', tol = 1e-7 } = {}) {
  if (!Array.isArray(solids)) solids = [solids];
  const lines = [];
  let id = 0;
  const add = (s) => { lines.push(`#${++id}=${s};`); return `#${id}`; };
  const point = (p) => add(`CARTESIAN_POINT('',(${real(p[0])},${real(p[1])},${real(p[2])}))`);

  const app = add(`APPLICATION_CONTEXT('core data for automotive mechanical design processes')`);
  add(`APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,${app})`);
  const product = add(`PRODUCT(${text(name)},${text(name)},'',(${add(`PRODUCT_CONTEXT('',${app},'mechanical')`)}))`);
  add(`PRODUCT_RELATED_PRODUCT_CATEGORY('part',$,(${product}))`);
  const formation = add(`PRODUCT_DEFINITION_FORMATION('','',${product})`);
  const definition = add(`PRODUCT_DEFINITION('design','',${formation},${add(`PRODUCT_DEFINITION_CONTEXT('part definition',${app},'design')`)})`);
  const shapeDef = add(`PRODUCT_DEFINITION_SHAPE('','',${definition})`);
  const mm = add('(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.))');
  const rad = add('(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.))');
  const sr = add('(NAMED_UNIT(*)SI_UNIT($,.STERADIAN.)SOLID_ANGLE_UNIT())');
  const unc = add(`UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(${real(tol)}),${mm},'distance_accuracy_value','confusion accuracy')`);
  const ctx3 = add(`(GEOMETRIC_REPRESENTATION_CONTEXT(3)GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((${unc}))GLOBAL_UNIT_ASSIGNED_CONTEXT((${mm},${rad},${sr}))REPRESENTATION_CONTEXT('Context #1','3D Context with UNIT and UNCERTAINTY'))`);
  const ctx2 = add(`(GEOMETRIC_REPRESENTATION_CONTEXT(2)PARAMETRIC_REPRESENTATION_CONTEXT()REPRESENTATION_CONTEXT('2D SPACE',''))`);

  const closedShell = (faces, loops) => {
    if (orientation(faces, loops) < 0) {
      faces = faces.map((f) => {
        const P = new Array(f.poles.length);
        for (let i = 0; i < f.uCount; i++) for (let j = 0; j < f.vCount; j++) for (let c = 0; c < 3; c++) P[(j * f.uCount + i) * 3 + c] = f.poles[(i * f.vCount + j) * 3 + c];
        return { ...f, uCount: f.vCount, vCount: f.uCount, poles: P };
      });
      loops = loops.map((loop) => [loop[0], ...loop.slice(1).reverse()].map(([v, u, w]) => [v, w, u]));
    }
    const surface = faces.map((f) => {
      const rows = [];
      for (let i = 0; i < f.uCount; i++) {
        const row = [];
        for (let j = 0; j < f.vCount; j++) { const o = (i * f.vCount + j) * 3; row.push(point([f.poles[o], f.poles[o + 1], f.poles[o + 2]])); }
        rows.push(`(${row.join(',')})`);
      }
      const p = f.uCount - 1, q = f.vCount - 1;
      return add(`B_SPLINE_SURFACE_WITH_KNOTS('',${p},${q},(${rows.join(',')}),.UNSPECIFIED.,.F.,.F.,.F.,(${p + 1},${p + 1}),(${q + 1},${q + 1}),(0.,1.),(0.,1.),.UNSPECIFIED.)`);
    });
    // edges: consecutive loop vertices; each runs from its smaller vertex id; its two uses give the pcurves
    const uses = new Map(), vertexAt = new Map();
    loops.forEach((loop, f) => loop.forEach(([a, u0, v0], i) => {
      const [b, u1, v1] = loop[(i + 1) % loop.length], key = a < b ? `${a},${b}` : `${b},${a}`;
      if (!uses.has(key)) uses.set(key, []);
      uses.get(key).push({ f, from: a < b ? [u0, v0] : [u1, v1], to: a < b ? [u1, v1] : [u0, v0] });
    }));
    const vertex = new Map(), edge = new Map();
    for (const [key, list] of uses) {
      if (list.length !== 2) throw new Error(`patchStep: edge ${key} has ${list.length} faces; the solid must be closed and manifold`);
      const full = list.find((x) => Math.abs(x.to[0] - x.from[0]) + Math.abs(x.to[1] - x.from[1]) === 1) ?? list[0];
      const poles = segmentPoles(faces[full.f], full.from, full.to), p = poles.length - 1;
      const [a, b] = key.split(',').map(Number);
      if (!vertex.has(a)) vertex.set(a, add(`VERTEX_POINT('',${point(poles[0])})`));
      if (!vertex.has(b)) vertex.set(b, add(`VERTEX_POINT('',${point(poles[p])})`));
      const curve = add(`B_SPLINE_CURVE_WITH_KNOTS('',${p},(${poles.map(point).join(',')}),.UNSPECIFIED.,.F.,.F.,(${p + 1},${p + 1}),(0.,1.),.UNSPECIFIED.)`);
      const pcurves = list.map(({ f, from, to }) => {
        const pts = [from, to].map((uv) => add(`CARTESIAN_POINT('',(${real(uv[0])},${real(uv[1])}))`));
        const line = add(`B_SPLINE_CURVE_WITH_KNOTS('',1,(${pts.join(',')}),.UNSPECIFIED.,.F.,.F.,(2,2),(0.,1.),.UNSPECIFIED.)`);
        return add(`PCURVE('',${surface[f]},${add(`DEFINITIONAL_REPRESENTATION('',(${line}),${ctx2})`)})`);
      });
      const geometry = add(`SURFACE_CURVE('',${curve},(${pcurves.join(',')}),.CURVE_3D.)`);
      edge.set(key, add(`EDGE_CURVE('',${vertex.get(a)},${vertex.get(b)},${geometry},.T.)`));
    }
    const faceIds = loops.map((loop, f) => {
      const oriented = loop.map(([a], i) => {
        const b = loop[(i + 1) % loop.length][0];
        return add(`ORIENTED_EDGE('',*,*,${edge.get(a < b ? `${a},${b}` : `${b},${a}`)},${a < b ? '.T.' : '.F.'})`);
      });
      const bound = add(`FACE_OUTER_BOUND('',${add(`EDGE_LOOP('',(${oriented.join(',')}))`)},.T.)`);
      return add(`ADVANCED_FACE('',(${bound}),${surface[f]},.T.)`);
    });
    return add(`CLOSED_SHELL('',(${faceIds.join(',')}))`);
  };
  const breps = solids.map((sol, i) => add(`MANIFOLD_SOLID_BREP(${text(sol.name ?? `Body ${i + 1}`)},${closedShell(sol.faces, sol.loops)})`));
  const origin = add(`AXIS2_PLACEMENT_3D('',${point([0, 0, 0])},${add("DIRECTION('',(0.,0.,1.))")},${add("DIRECTION('',(1.,0.,0.))")})`);
  const rep = add(`ADVANCED_BREP_SHAPE_REPRESENTATION('',(${origin},${breps.join(',')}),${ctx3})`);
  add(`SHAPE_DEFINITION_REPRESENTATION(${shapeDef},${rep})`);
  const stamp = new Date().toISOString().slice(0, 19);
  return ['ISO-10303-21;', 'HEADER;', "FILE_DESCRIPTION(('Untessellator NURBS solid'),'2;1');",
    `FILE_NAME(${text(name)},'${stamp}',(''),(''),'Untessellator','Untessellator','');`,
    "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));", 'ENDSEC;', 'DATA;', ...lines, 'ENDSEC;',
    'END-ISO-10303-21;', ''].join('\n');
}
