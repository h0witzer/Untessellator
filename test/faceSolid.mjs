// One trimmed B-spline face per painted face (js/subd/faceSolid.js + faceStep.js): an STL's faces by flood fill
// over edges bent less than --crease degrees, or a mesh.json { vertices, faces, labels } with its own; timed, the
// STEP written (--out) and read back with OCCT (--check): solids, faces (should equal the painted faces), closed,
// valid; and the export's distance to the mesh both ways over its own tessellation.
// --dump dir: the input and OCCT's tessellation of the solid as float32 triangles (scratch twosided.py / look.py).
// node test/faceSolid.mjs <file.stl | mesh.json> ... [--crease 20] [--check] [--out dir] [--dump dir]
import fs from 'node:fs';
import path from 'node:path';
import { faceSolid } from '../js/subd/faceSolid.js';
import { faceStep } from '../js/subd/faceStep.js';

const args = process.argv.slice(2);
const flag = (k) => { const i = args.indexOf(k); return i >= 0 ? args.splice(i, 2)[1] : null; };
const crease = Number(flag('--crease') ?? 20), outDir = flag('--out'), dumpDir = flag('--dump');
const check = args.includes('--check');
const files = args.filter((a) => !a.startsWith('--'));

function load(file) {
  if (file.endsWith('.json')) {
    const m = JSON.parse(fs.readFileSync(file, 'utf8'));
    return { V: m.vertices, T: m.faces, labels: m.labels };
  }
  const buf = fs.readFileSync(file), n = buf.readUInt32LE(80), key = new Map(), V = [], T = [];
  for (let t = 0; t < n; t++) {
    const tri = [];
    for (let c = 0; c < 3; c++) {
      const o = 84 + t * 50 + 12 + c * 12, p = [buf.readFloatLE(o), buf.readFloatLE(o + 4), buf.readFloatLE(o + 8)], k = p.join(',');
      if (!key.has(k)) { key.set(k, V.length); V.push(p); }
      tri.push(key.get(k));
    }
    T.push(tri);
  }
  // faces: flood fill over edges bent less than the crease angle (what one paint click per face gives)
  const N = T.map(([a, b, c]) => { const u = V[b].map((x, i) => x - V[a][i]), w = V[c].map((x, i) => x - V[a][i]); const cr = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]], l = Math.hypot(...cr) || 1; return cr.map((x) => x / l); });
  const half = new Map();
  T.forEach((t, f) => { for (let k = 0; k < 3; k++) half.set(`${t[k]},${t[(k + 1) % 3]}`, f); });
  const cos = Math.cos(crease * Math.PI / 180), labels = new Int32Array(T.length).fill(-1);
  let G = 0;
  for (let s = 0; s < T.length; s++) {
    if (labels[s] >= 0) continue;
    const st = [s]; labels[s] = G;
    while (st.length) {
      const t = st.pop();
      for (let k = 0; k < 3; k++) {
        const u = half.get(`${T[t][(k + 1) % 3]},${T[t][k]}`);
        if (u !== undefined && labels[u] < 0 && N[t][0] * N[u][0] + N[t][1] * N[u][1] + N[t][2] * N[u][2] >= cos) { labels[u] = G; st.push(u); }
      }
    }
    G++;
  }
  return { V, T, labels: [...labels] };
}

let oc = null;
for (const file of files) {
  const { V, T, labels } = load(file);
  const name = path.basename(file).replace(/\.(stl|json)$/i, '');
  const t0 = performance.now();
  const r = faceSolid(V, T, labels);
  if (r.error) { console.log(`${name}: ${r.error}`); continue; }
  const step = faceStep({ faces: r.faces, edges: r.edges, points: r.points, name }, { name, tol: r.tolerance });
  const ms = performance.now() - t0;
  console.log(`${name}: ${new Set(labels).size} painted faces -> ${r.faces.length} STEP faces, ${r.edges.length} edges, ${(ms / 1000).toFixed(2)} s; `
    + `surface fit max ${(r.report.maxFit * 100).toFixed(3)} % of size, edges on their faces within ${(r.report.edgeTol * 100).toFixed(4)} %`);
  if (outDir) fs.writeFileSync(path.join(outDir, `${name}-faces.step`), step);
  if (check) {
    const { loadOcct } = await import('../js/knit/index.js');
    const { score, subShapes } = await import('../js/knit/build.js');
    oc ??= await loadOcct();
    oc.FS.writeFile('/tmp/in.step', step);
    const reader = new oc.STEPControl_Reader_1();
    if (reader.ReadFile('/tmp/in.step') !== oc.IFSelect_ReturnStatus.IFSelect_RetDone) { console.log('   check: OCCT could not read it'); continue; }
    reader.TransferRoots(new oc.Message_ProgressRange_1());
    const shape = reader.OneShape(), sc = score(oc, shape, { volume: () => null });
    console.log(`   check: solids ${subShapes(oc, shape, oc.TopAbs_ShapeEnum.TopAbs_SOLID).length}, faces ${subShapes(oc, shape, oc.TopAbs_ShapeEnum.TopAbs_FACE).length}, closed ${sc.closed}, valid ${sc.valid}`);
    if (dumpDir) {
      const { previewMesh } = await import('../js/knit/preview.js');
      let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
      for (const p of V) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], p[k]); hi[k] = Math.max(hi[k], p[k]); }
      const pm = previewMesh(oc, shape, 2e-4 * Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]));
      const inp = new Float32Array(T.length * 9);
      T.forEach((t, i) => t.forEach((v, k) => inp.set(V[v], i * 9 + k * 3)));
      fs.writeFileSync(path.join(dumpDir, `${name}.in.bin`), Buffer.from(inp.buffer));
      fs.writeFileSync(path.join(dumpDir, `${name}.out.bin`), Buffer.from(Float32Array.from(pm.positions).buffer));
      const idx = path.join(dumpDir, 'index.json'), list = fs.existsSync(idx) ? JSON.parse(fs.readFileSync(idx, 'utf8')).filter((x) => x.name !== name) : [];
      list.push({ name, ms, bodies: [{ quads: r.faces.length, faces: r.faces.length }] });
      fs.writeFileSync(idx, JSON.stringify(list));
    }
  }
}
