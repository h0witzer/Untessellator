/**
 * faceStep.js — STEP (AP214) text for solids of trimmed B-spline faces (js/subd/faceSolid.js), written directly.
 * Each face: one B_SPLINE_SURFACE_WITH_KNOTS and its bounds (edge loops); each edge: its 3D B-spline curve and a
 * pcurve on each face it bounds (SURFACE_CURVE), or two on one face for a seam (SEAM_CURVE). Faces point outward
 * through ADVANCED_FACE's sense; loops run counter-clockwise seen from outside.
 */

function real(x) {
  if (x === 0) return '0.';
  const [m0, e] = x.toPrecision(15).split('e');
  const m = m0.includes('.') ? m0.replace(/0+$/, '') : `${m0}.`;
  return e ? `${m}E${e}` : m;
}
const text = (s) => `'${String(s).replace(/'/g, "''")}'`;

/** Flat knots -> [distinct knots, multiplicities]. */
function knotPairs(flat) {
  const k = [], m = [];
  for (const x of flat) { if (k.length && x === k[k.length - 1]) m[m.length - 1]++; else { k.push(x); m.push(1); } }
  return [k, m];
}

/**
 * solids: [{ faces, edges, points, name? }] (or one). faces[f]: { params: { poles [[[x, y, z]]] (u rows), tu, tv,
 * pu, pv, u_periodic? }, sense, bounds: [{ outer, edges: [{ e, forward }] }] }; edges[e]: { curve: { p, knots,
 * poles }, v0, v1, pcurves: [{ face, pc }] } or with seam: { face, pcurves: [first, second] }; points: vertex xyz.
 */
export function faceStep(solids, { name = 'Part', tol = 1e-6 } = {}) {
  if (!Array.isArray(solids)) solids = [solids];
  const lines = [];
  let id = 0;
  const add = (s) => { lines.push(`#${++id}=${s};`); return `#${id}`; };
  const point = (p) => add(`CARTESIAN_POINT('',(${p.map(real).join(',')}))`);

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

  const curve = (c) => {
    const [k, m] = knotPairs(c.knots);
    return add(`B_SPLINE_CURVE_WITH_KNOTS('',${c.p},(${c.poles.map(point).join(',')}),.UNSPECIFIED.,.F.,.F.,(${m.join(',')}),(${k.map(real).join(',')}),.UNSPECIFIED.)`);
  };
  const surface = (s) => {
    const rows = s.poles.map((row) => `(${row.map(point).join(',')})`);
    const [ku, mu] = knotPairs(s.tu), [kv, mv] = knotPairs(s.tv);
    return add(`B_SPLINE_SURFACE_WITH_KNOTS('',${s.pu ?? 3},${s.pv ?? 3},(${rows.join(',')}),.UNSPECIFIED.,.F.,.F.,.F.,(${mu.join(',')}),(${mv.join(',')}),(${ku.map(real).join(',')}),(${kv.map(real).join(',')}),.UNSPECIFIED.)`);
  };
  const pcurve = (surf, pc) => add(`PCURVE('',${surf},${add(`DEFINITIONAL_REPRESENTATION('',(${curve(pc)}),${ctx2})`)})`);

  const shell = ({ faces, edges, points }) => {
    const surf = faces.map((f) => surface(f.params));
    const vtx = points.map((p) => add(`VERTEX_POINT('',${point(p)})`));
    const edgeIds = edges.map((e) => {
      const c3 = curve(e.curve);
      const geometry = e.seam
        ? add(`SEAM_CURVE('',${c3},(${e.seam.pcurves.map((pc) => pcurve(surf[e.seam.face], pc)).join(',')}),.PCURVE_S1.)`)
        : add(`SURFACE_CURVE('',${c3},(${e.pcurves.map((p) => pcurve(surf[p.face], p.pc)).join(',')}),.PCURVE_S1.)`);
      return add(`EDGE_CURVE('',${vtx[e.v0]},${vtx[e.v1]},${geometry},.T.)`);
    });
    const faceIds = faces.map((f, fi) => {
      const bounds = f.bounds.map((b) => {
        const loop = add(`EDGE_LOOP('',(${b.edges.map(({ e, forward }) => add(`ORIENTED_EDGE('',*,*,${edgeIds[e]},${forward ? '.T.' : '.F.'})`)).join(',')}))`);
        return add(`${b.outer ? 'FACE_OUTER_BOUND' : 'FACE_BOUND'}('',${loop},.T.)`);
      });
      return add(`ADVANCED_FACE('',(${bounds.join(',')}),${surf[fi]},${f.sense ? '.T.' : '.F.'})`);
    });
    return add(`CLOSED_SHELL('',(${faceIds.join(',')}))`);
  };
  const breps = solids.map((s, i) => add(`MANIFOLD_SOLID_BREP(${text(s.name ?? `Body ${i + 1}`)},${shell(s)})`));
  const origin = add(`AXIS2_PLACEMENT_3D('',${point([0, 0, 0])},${add("DIRECTION('',(0.,0.,1.))")},${add("DIRECTION('',(1.,0.,0.))")})`);
  const rep = add(`ADVANCED_BREP_SHAPE_REPRESENTATION('',(${origin},${breps.join(',')}),${ctx3})`);
  add(`SHAPE_DEFINITION_REPRESENTATION(${shapeDef},${rep})`);
  const stamp = new Date().toISOString().slice(0, 19);
  return ['ISO-10303-21;', 'HEADER;', "FILE_DESCRIPTION(('Untessellator NURBS solid'),'2;1');",
    `FILE_NAME(${text(name)},'${stamp}',(''),(''),'Untessellator','Untessellator','');`,
    "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));", 'ENDSEC;', 'DATA;', ...lines, 'ENDSEC;',
    'END-ISO-10303-21;', ''].join('\n');
}
