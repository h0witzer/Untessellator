/**
 * patchStep.js — STEP (AP214) text for a closed solid of untrimmed Bezier patches on a quad cage (the output of
 * js/subd/fastNurbs.js or fitCage.js), written directly: no kernel. One vertex per cage vertex, one edge per cage
 * edge whose 3D curve is a patch side (its neighbour's side is the same curve with the same parameter), a straight
 * pcurve in each face's parameter square, one face per patch: valid by construction, with nothing to project, sew
 * or repair. Faces point outward; a cage that points inward is turned over (loops reversed, patches transposed).
 * test/subdStep.mjs and test/meshToStep.mjs --check read the files back with OCCT as the judge.
 */

/** A STEP real: always a decimal point, 15 significant digits, E exponent. */
function real(x) {
  if (x === 0) return '0.';
  const [m0, e] = x.toPrecision(15).split('e');
  let m = m0.includes('.') ? m0.replace(/0+$/, '') : `${m0}.`;
  if (m.endsWith('-.')) m = m.replace('-.', '-0.');
  return e ? `${m}E${e}` : m;
}

const text = (s) => `'${String(s).replace(/'/g, "''")}'`;
const UV = [[0, 0], [1, 0], [1, 1], [0, 1]];

/** The poles of side k of a patch, running from loop corner k to corner k + 1. */
function sidePoles(f, k) {
  const p = f.uCount - 1, at = (i, j) => { const o = (i * f.vCount + j) * 3; return [f.poles[o], f.poles[o + 1], f.poles[o + 2]]; };
  const idx = Array.from({ length: p + 1 }, (_, m) => m);
  if (k === 0) return idx.map((m) => at(m, 0));
  if (k === 1) return idx.map((m) => at(p, m));
  if (k === 2) return idx.map((m) => at(p - m, p));
  return idx.map((m) => at(0, p - m));
}

/** Six times the signed volume of the quad cage through the patch corners: positive when the loops face outward. */
export function orientation(faces, cageFaces) {
  let v6 = 0;
  cageFaces.forEach((q, f) => {
    const c = [0, 1, 2, 3].map((k) => sidePoles(faces[f], k)[0]);
    for (const [a, b, d] of [[0, 1, 2], [0, 2, 3]]) {
      const [x, y, z] = [c[a], c[b], c[d]];
      v6 += x[0] * (y[1] * z[2] - y[2] * z[1]) - x[1] * (y[0] * z[2] - y[2] * z[0]) + x[2] * (y[0] * z[1] - y[1] * z[0]);
    }
  });
  return v6;
}

/**
 * The STEP file: one MANIFOLD_SOLID_BREP per solid, named. solids: [{ faces, cageFaces, name? }] (or one such
 * object). faces[f]: Bezier patch { uCount, vCount, poles (row-major, u then v, 3 each) } of cage face f;
 * cageFaces[f]: its loop [c0, c1, c2, c3] (u from c0 to c1, v from c0 to c3).
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

  const closedShell = (faces, cageFaces) => {
    if (orientation(faces, cageFaces) < 0) {
      faces = faces.map((f) => {
        const P = new Array(f.poles.length);
        for (let i = 0; i < f.uCount; i++) for (let j = 0; j < f.vCount; j++) for (let c = 0; c < 3; c++) P[(j * f.uCount + i) * 3 + c] = f.poles[(i * f.vCount + j) * 3 + c];
        return { ...f, uCount: f.vCount, vCount: f.uCount, poles: P };
      });
      cageFaces = cageFaces.map(([a, b, c, d]) => [a, d, c, b]);
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
    const vertex = new Map();
    cageFaces.forEach((q, f) => q.forEach((v, k) => {
      if (!vertex.has(v)) vertex.set(v, add(`VERTEX_POINT('',${point(sidePoles(faces[f], k)[0])})`));
    }));
    // each edge runs from its smaller vertex id; its two uses, [face, side], give the pcurves
    const uses = new Map();
    cageFaces.forEach((q, f) => q.forEach((a, k) => {
      const b = q[(k + 1) % 4], key = a < b ? `${a},${b}` : `${b},${a}`;
      if (!uses.has(key)) uses.set(key, []);
      uses.get(key).push([f, k, a < b]);
    }));
    const edge = new Map();
    for (const [key, list] of uses) {
      if (list.length !== 2) throw new Error(`patchStep: edge ${key} has ${list.length} faces; the cage must be closed and manifold`);
      const [f0, k0, fwd0] = list[0];
      const poles = fwd0 ? sidePoles(faces[f0], k0) : sidePoles(faces[f0], k0).reverse();
      const p = poles.length - 1;
      const curve = add(`B_SPLINE_CURVE_WITH_KNOTS('',${p},(${poles.map(point).join(',')}),.UNSPECIFIED.,.F.,.F.,(${p + 1},${p + 1}),(0.,1.),.UNSPECIFIED.)`);
      const pcurves = list.map(([f, k, fwd]) => {
        const [s, e] = fwd ? [UV[k], UV[(k + 1) % 4]] : [UV[(k + 1) % 4], UV[k]];
        const pts = [s, e].map((uv) => add(`CARTESIAN_POINT('',(${real(uv[0])},${real(uv[1])}))`));
        const line = add(`B_SPLINE_CURVE_WITH_KNOTS('',1,(${pts.join(',')}),.UNSPECIFIED.,.F.,.F.,(2,2),(0.,1.),.UNSPECIFIED.)`);
        return add(`PCURVE('',${surface[f]},${add(`DEFINITIONAL_REPRESENTATION('',(${line}),${ctx2})`)})`);
      });
      const [a, b] = key.split(',').map(Number);
      const geometry = add(`SURFACE_CURVE('',${curve},(${pcurves.join(',')}),.CURVE_3D.)`);
      edge.set(key, add(`EDGE_CURVE('',${vertex.get(a)},${vertex.get(b)},${geometry},.T.)`));
    }
    const faceIds = cageFaces.map((q, f) => {
      const oriented = q.map((a, k) => {
        const b = q[(k + 1) % 4];
        return add(`ORIENTED_EDGE('',*,*,${edge.get(a < b ? `${a},${b}` : `${b},${a}`)},${a < b ? '.T.' : '.F.'})`);
      });
      const bound = add(`FACE_OUTER_BOUND('',${add(`EDGE_LOOP('',(${oriented.join(',')}))`)},.T.)`);
      return add(`ADVANCED_FACE('',(${bound}),${surface[f]},.T.)`);
    });
    return add(`CLOSED_SHELL('',(${faceIds.join(',')}))`);
  };
  const breps = solids.map((sol, i) => add(`MANIFOLD_SOLID_BREP(${text(sol.name ?? `Body ${i + 1}`)},${closedShell(sol.faces, sol.cageFaces)})`));
  const origin = add(`AXIS2_PLACEMENT_3D('',${point([0, 0, 0])},${add("DIRECTION('',(0.,0.,1.))")},${add("DIRECTION('',(1.,0.,0.))")})`);
  const rep = add(`ADVANCED_BREP_SHAPE_REPRESENTATION('',(${origin},${breps.join(',')}),${ctx3})`);
  add(`SHAPE_DEFINITION_REPRESENTATION(${shapeDef},${rep})`);
  const stamp = new Date().toISOString().slice(0, 19);
  return ['ISO-10303-21;', 'HEADER;', "FILE_DESCRIPTION(('Untessellator NURBS solid'),'2;1');",
    `FILE_NAME(${text(name)},'${stamp}',(''),(''),'Untessellator','Untessellator','');`,
    "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));", 'ENDSEC;', 'DATA;', ...lines, 'ENDSEC;',
    'END-ISO-10303-21;', ''].join('\n');
}
