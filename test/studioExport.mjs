/**
 * studioExport.mjs — the studio's auto-paint and STEP export on mesh files, headless: the same models,
 * settings and code (js/autoPaint.js, js/paintCore.js, typeFinalGroup, js/knit), with
 * onnxruntime-node in place of the browser runtime. What it writes is what the studio's Export STEP
 * would download after one Auto-paint with the default settings.
 *
 * Usage:  node test/studioExport.mjs <out dir> <file.stl> [<file.stl> ...] [--dump] [--no-axes]
 *
 * Per file: <name>.step and <name>.json (the knitter's report, groups by type, timings). The mesh is
 * centred as js/stlLoader.js does and moved back for the export, as the studio does. --synthetic: where
 * <file>.src exists next to the STL (js/meshRepair.js's src, as test/meshDamage.mjs writes it), triangles with
 * src -1 (hole caps) join the group most of the input triangles around them are in and are left out of typing
 * and fitting, so a flat cap on a curved face does not make its group inexact; they go to the knitter in that
 * group, with the vertices only caps use
 * moved onto that group's fitted surface (a cap's centre would otherwise sit inside a curved face). --dump also writes
 * the painted part as test/knit.mjs input: <name>.pos (float32 soup), <name>.fid (group per triangle) and
 * <name>.fits.json, so the export can be rerun without painting again (positions in float32, so a rerun
 * can differ slightly from the run that wrote them).
 */

import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { join, basename } from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

import { preparePaintPositions, paintWith, typeFinalGroup } from '../js/paintCore.js';
import { coherentScores, seedOrder, autoPaint } from '../js/autoPaint.js';
import { annotateEdges, edgeWall } from '../js/edgeWall.js';
import { fitRegionAs } from '../js/gaussMap.js';
import { freeformMerge } from '../js/freeformMerge.js';
import { loadOcct, knitSolid, writeStep } from '../js/knit/index.js';
import { distNormal } from '../js/knit/geometry.js';

const ROOT = fileURLToPath(new URL('..', import.meta.url));
const ort = createRequire(import.meta.url)('onnxruntime-node');
const args = process.argv.slice(2);
const flag = (name, dflt) => { const i = args.indexOf(name); if (i < 0) return dflt; const v = args[i + 1]; args.splice(i, 2); return v; };
const DEADLINE = Number(flag('--deadline', '150'));   // the studio's EXPORT_DEADLINE_S
const DUMP = args.includes('--dump');
const NO_AXES = args.includes('--no-axes');
const SYNTHETIC = args.includes('--synthetic');
if (SYNTHETIC) args.splice(args.indexOf('--synthetic'), 1);
if (NO_AXES) args.splice(args.indexOf('--no-axes'), 1);
if (DUMP) args.splice(args.indexOf('--dump'), 1);
const [OUT, ...FILES] = args;
if (!OUT || !FILES.length) { console.error('usage: node test/studioExport.mjs <out dir> <file.stl> ...'); process.exit(2); }
mkdirSync(OUT, { recursive: true });

// The studio's defaults (js/studio/studio.js settings): v7 model, method 'model', edge wall on curved
// fills at fpr 2%, refined fills (boundary model v2 + refiner), no tangent wall.
const MODEL = join(ROOT, 'models/seedfill-v7gnn.onnx');
const EDGE_MODEL = join(ROOT, 'models/edge-boundary-v2');
const REFINER = join(ROOT, 'models/fill-refiner.onnx');
const EDGE_WALL_TYPES = ['cylinder', 'cone', 'sphere', 'torus'];

const graphFeeds = (session, graph, n) => (session.inputNames.includes('nbr') && graph
  ? { nbr: new ort.Tensor('int64', graph.nbr, [1, n, 3]), eattr: new ort.Tensor('float32', graph.eattr, [1, n, 3]) }
  : {});
async function seedRunner(path) {
  const session = await ort.InferenceSession.create(path, { intraOpNumThreads: 2, interOpNumThreads: 1 });
  return async (features, n, graph) => {
    const out = await session.run({
      features: new ort.Tensor('float32', features, [1, n, features.length / n]),
      mask: new ort.Tensor('float32', new Float32Array(n).fill(1), [1, n]),
      ...graphFeeds(session, graph, n),
    });
    return { sameLogit: out.same_logit.data, typeLogit: out.type_logit ? Array.from(out.type_logit.data) : null,
             surface: out.surface ? out.surface.data : null };
  };
}
const runner = await seedRunner(MODEL);
const refiner = await seedRunner(REFINER);
const edgeMeta = JSON.parse(readFileSync(`${EDGE_MODEL}.json`, 'utf8'));
const edgeSession = await ort.InferenceSession.create(`${EDGE_MODEL}.onnx`, { intraOpNumThreads: 2, interOpNumThreads: 1 });
const edgeRunner = async (features, n) => (await edgeSession.run({ features: new ort.Tensor('float32', features, [n, features.length / n]) })).logit.data;
const extended = JSON.parse(readFileSync(MODEL.replace(/\.onnx$/, '.json'), 'utf8')).features === 'v6';

/** Triangle soup (float32, as three.js holds it) from a binary or ASCII STL. */
function readStl(path) {
  const buf = readFileSync(path);
  const n = buf.length >= 84 ? buf.readUInt32LE(80) : 0;
  if (buf.length === 84 + 50 * n) {
    const P = new Float32Array(n * 9);
    for (let t = 0; t < n; t++) for (let k = 0; k < 9; k++) P[t * 9 + k] = buf.readFloatLE(84 + 50 * t + 12 + 4 * k);
    return P;
  }
  const vs = [...buf.toString('utf8').matchAll(/vertex\s+(\S+)\s+(\S+)\s+(\S+)/g)].map((m) => [+m[1], +m[2], +m[3]]);
  return Float32Array.from(vs.flat());
}

const oc = await loadOcct();
for (const file of FILES) {
  const name = basename(file).replace(/\.[^.]+$/, '');
  const t0 = performance.now();
  const raw = readStl(file);
  // centred on the bounding box, in float32, as js/stlLoader.js setupGeometry does
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < raw.length; i += 3) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], raw[i + k]); hi[k] = Math.max(hi[k], raw[i + k]); }
  const offset = [0, 1, 2].map((k) => (lo[k] + hi[k]) / 2);
  const centred = Float32Array.from(raw, (v, i) => v - offset[i % 3]);
  const mesh = preparePaintPositions(Float64Array.from(centred));
  const { facets, adjacency } = mesh;
  await annotateEdges(facets, adjacency, edgeRunner, edgeMeta.columns);
  const wall = edgeWall(1 / (1 + Math.exp(-edgeMeta.thresholds['fpr0.02'].logit)), EDGE_WALL_TYPES);
  const order = seedOrder(await coherentScores(facets, adjacency), facets);
  const owner = new Int32Array(facets.count).fill(-1);
  const types = new Map();
  let next = 0;
  const result = await autoPaint({
    facets, order, owner, minKeep: 1, adjacency, absorbUpTo: 8,
    fill: (seed, taken) => paintWith(runner, seed, mesh, {
      method: 'model', type: 'auto', taken, extended, reportFit: false, grow: { wall }, refine: { runner: refiner },
    }),
    onSegment: (claim, out) => { const id = next++; for (const t of claim) owner[t] = id; types.set(id, out.fitter ?? null); return id; },
    onAbsorb: (island, id) => { for (const t of island) owner[t] = id; },
    onDiscard: (claim) => { for (const t of claim) owner[t] = -1; },
  });
  const paintSecs = (performance.now() - t0) / 1000;
  // the studio's end of auto-paint: freeform pieces joined, each finished group's type, then its fit
  const segments = new Set(owner).size - (owner.includes(-1) ? 1 : 0);
  // typing and fitting see the input's own triangles only (--synthetic)
  const srcPath = file.replace(/\.[^.]+$/, '.src');
  const srcBuf = SYNTHETIC && existsSync(srcPath) ? readFileSync(srcPath) : null;
  const src = srcBuf ? new Int32Array(srcBuf.buffer.slice(srcBuf.byteOffset, srcBuf.byteOffset + srcBuf.length)) : null;
  if (src) {
    // a cap says nothing about the shape: each connected run of cap triangles joins the group most of the
    // input triangles around it are in, whatever the painter made of it
    const seen = new Uint8Array(owner.length);
    for (let s0 = 0; s0 < owner.length; s0++) {
      if (src[s0] >= 0 || seen[s0]) continue;
      const comp = [s0], votes = new Map();
      seen[s0] = 1;
      for (let qi = 0; qi < comp.length; qi++) for (const { neighbor: u } of adjacency.get(comp[qi]) ?? []) {
        if (src[u] >= 0) { if (owner[u] >= 0) votes.set(owner[u], (votes.get(owner[u]) ?? 0) + 1); continue; }
        if (!seen[u]) { seen[u] = 1; comp.push(u); }
      }
      let best = -1, n = 0;
      for (const [g, c] of votes) if (c > n) { n = c; best = g; }
      if (best >= 0) for (const t of comp) owner[t] = best;
    }
  }
  const fitOwner = src ? Int32Array.from(owner, (g, t) => (src[t] < 0 ? -1 : g)) : owner;
  const { freeform, inexactShare } = freeformMerge(fitOwner, facets, adjacency);
  const byGroup = new Map();
  for (let t = 0; t < fitOwner.length; t++) if (fitOwner[t] >= 0) (byGroup.get(fitOwner[t]) ?? byGroup.set(fitOwner[t], new Set()).get(fitOwner[t])).add(t);
  const surfaces = new Map();
  const typeCount = {};
  let retyped = 0;
  for (const [id, tris] of byGroup) {
    if (freeform.has(id)) { typeCount.freeform = (typeCount.freeform ?? 0) + 1; continue; }
    const typed = typeFinalGroup(tris, facets, types.get(id));
    if (typed.changed) retyped++;
    const type = typed.type;
    typeCount[type ?? 'none'] = (typeCount[type ?? 'none'] ?? 0) + 1;
    if (!type) continue;
    const fit = fitRegionAs(type, tris, facets);
    if (!fit.params) continue;
    const p = { ...fit.params };
    for (const k of ['origin', 'axisPoint', 'apex', 'center']) if (p[k]) p[k] = p[k].map((v, a) => v + offset[a]);
    surfaces.set(id, [type, p]);
  }
  const positions = Float64Array.from(centred, (v, i) => v + offset[i % 3]);
  if (src) {
    // a vertex only caps use goes onto the fitted surface of the one group its triangles are in
    const users = new Map();
    for (let t = 0; t < owner.length; t++) for (let k = 0; k < 3; k++) {
      const o = t * 9 + k * 3, key = `${positions[o]},${positions[o + 1]},${positions[o + 2]}`;
      const u = users.get(key) ?? users.set(key, { synth: true, groups: new Set(), at: [] }).get(key);
      if (src[t] >= 0) u.synth = false;
      u.groups.add(owner[t]); u.at.push(o);
    }
    for (const u of users.values()) {
      if (!u.synth || u.groups.size !== 1) continue;
      const s = surfaces.get([...u.groups][0]);
      if (!s || s[0] === 'bspline') continue;
      const x = [positions[u.at[0]], positions[u.at[0] + 1], positions[u.at[0] + 2]];
      const [d, n] = distNormal(s[0], s[1], x);
      for (const o of u.at) for (let j = 0; j < 3; j++) positions[o + j] = x[j] - d * n[j];
    }
  }
  if (DUMP) {
    writeFileSync(join(OUT, `${name}.pos`), Buffer.from(Float32Array.from(positions).buffer));
    writeFileSync(join(OUT, `${name}.fid`), Buffer.from(owner.buffer));
    writeFileSync(join(OUT, `${name}.fits.json`), JSON.stringify({ faces: [...surfaces].map(([id, [type, params]]) => ({ id, type, params })) }));
  }
  const t1 = performance.now();
  const { solid, report } = knitSolid(oc, { positions, labels: owner, surfaces }, { deadline: DEADLINE, axes: !NO_AXES });
  const knitSecs = (performance.now() - t1) / 1000;
  if (solid) writeFileSync(join(OUT, `${name}.step`), writeStep(oc, solid));
  const row = { file, name, triangles: facets.count, segments, groups: byGroup.size, inexactShare: Math.round(inexactShare * 100) / 100, unpainted: result.free, retyped, typeCount,
    paint_secs: Math.round(paintSecs * 10) / 10, knit_secs: Math.round(knitSecs * 10) / 10, ...report };
  writeFileSync(join(OUT, `${name}.json`), JSON.stringify(row, null, 1));
  console.log(JSON.stringify({ ...row, info: undefined }));
}
