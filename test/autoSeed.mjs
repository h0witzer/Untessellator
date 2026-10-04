/**
 * autoSeed.mjs — automatic whole-part painting, scored against the B-rep faces.
 *
 * Usage:  node test/autoSeed.mjs <dir> <model.onnx> [--strategy coherent|random|largest]
 *                                 [--method model|hyp] [--meshers native,gmsh] [--parts N]
 *                                 [--min-claim 3] [--shard k/n] [--json out.json] [--dump-seg dir]
 *         <dir> from ml/export_fill_eval.py.
 *
 * Runs the studio's own code: js/autoPaint.js (seed order and loop) around js/paintCore.js (one
 * fill: model region, capped ordered type, crease barrier, painted facets taken). Only
 * onnxruntime-node replaces the browser runtime. Every part is painted to completion from an
 * empty start.
 *
 * Scores per part: the share of true B-rep faces, and of surface area, recovered CLEAN (one
 * segment holds >= 95% of the face and is >= 95% that face's surface type, as in fillEval),
 * segments per true face (over-segmentation), fills used, and the share of facets painted.
 *
 * Measured 2026-09-15 with v6feat and method 'model', the first version of this harness (every
 * fill a segment, no minimum): area clean random 75/73/53/63%, coherent 78/77/73/64%, largest
 * 70/74/74/68%, nearest-crease 70/74/68/67%, farthest-from-crease 72/73/63/65% on
 * native/nurbs/gmsh/game.
 */

import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join, basename, dirname } from 'node:path';
import { createRequire } from 'node:module';

import { preparePaintPositions, paintWith } from '../js/paintCore.js';
import { coherentScores, seedOrder, autoPaint } from '../js/autoPaint.js';
import { annotateEdges, edgeWall } from '../js/edgeWall.js';
import { mergeRegions } from '../js/mergeRegions.js';
import { attachEdgeGraph } from '../js/edgeGraph.js';
import { pearlRefine } from '../js/pearl.js';

import { graphInputs } from '../js/seedNet.js';

const ort = createRequire(import.meta.url)('onnxruntime-node');
/** The patch graph a message-passing model takes (seedNet.patchNeighbours, seedNet.graphInputs), if it
 * takes one; `edgeDim` 2 adds the learned boundary probability for refiners trained with --edge-dim 2. */
const graphFeeds = (session, graph, n, edgeDim = 1) => {
  if (!(session.inputNames.includes('nbr') && graph)) return {};
  const g = graphInputs(graph, n, edgeDim);
  return { nbr: new ort.Tensor('int64', ...g.nbr), eattr: new ort.Tensor('float32', ...g.eattr) };
};
/** A model's edge_dim from its sidecar JSON (1 when it has none). */
const edgeDimOf = (onnxPath) => {
  try { return JSON.parse(readFileSync(onnxPath.replace(/\.onnx$/, '.json'), 'utf8')).edge_dim ?? 1; } catch { return 1; }
};

const args = process.argv.slice(2);
function flag(name, dflt) {
  const i = args.indexOf(name);
  if (i < 0) return dflt;
  const v = args[i + 1];
  args.splice(i, 2);
  return v;
}
const STRATEGY = flag('--strategy', 'coherent');
const METHOD = flag('--method', 'model');
const MESHERS = flag('--meshers', null)?.split(',');
const PART_LIMIT = Number(flag('--parts', '0'));
// ml/edge_train.py trains the learned boundary on EVEN-indexed parts, so any run with the edge wall
// must be scored with `--parity odd`; `--parts N` then takes the first N parts of that parity.
const PARITY = flag('--parity', 'all');
// `--shard k/n` keeps every n-th part from the k-th, so one run can be split across processes; rows
// carry the manifest index, so the shards' --json outputs concatenate.
const SHARD = flag('--shard', null)?.split('/').map(Number);
if (!['all', 'odd', 'even'].includes(PARITY)) { console.error('--parity all|odd|even'); process.exit(2); }
const MIN_CLAIM = Number(flag('--min-claim', '1'));
const MIN_KEEP = Number(flag('--min-keep', '1'));
const ABSORB = Number(flag('--absorb', '8'));
// Fragment absorption (js/mergeRegions.js): offer every region of at most N triangles to the
// neighbour whose surface best explains it. 0 turns the pass off; 9 is the measured stratum.
const MERGE = Number(flag('--merge', '0'));
// PEARL (js/pearl.js): after auto-paint, re-assign facets among the segments' surfaces jointly by
// alpha-expansion. `--pearl lambda=1;rings=3;rounds=2;freeCost=2;sMax=9` (any subset), `--pearl 1`
// for the defaults.
const PEARL = flag('--pearl', null);
// The fill-refinement model (js/refineFeatures.js): every fill is re-labelled around its seed before it
// is claimed; `--refine <onnx>` with `--refine-cut <logit>` (default 1, the single-click setting).
const REFINE = flag('--refine', null);
const REFINE_CUT = Number(flag('--refine-cut', '1'));
// `--crf` (with --refine): each fill is re-chosen by a minimum cut whose per-facet term is the refiner's
// verdict inside its patch (cutFill memberReplacesFit), with the learned boundary pricing cut edges, as
// in test/growEval.mjs crf:a=2;l=4;r=1.
const CRF = args.includes('--crf');
if (CRF) args.splice(args.indexOf('--crf'), 1);
const pearlOpts = {};
if (PEARL && PEARL !== '1') for (const kv of PEARL.split(';')) { const [k, v] = kv.split('='); pearlOpts[k] = Number(v); }
const MERGE_DEBUG = args.includes('--merge-debug');
if (MERGE_DEBUG) args.splice(args.indexOf('--merge-debug'), 1);
const FORCE_SEED = args.includes('--force-seed');
if (FORCE_SEED) args.splice(args.indexOf('--force-seed'), 1);
const JSON_OUT = flag('--json', null);
// `--dump-seg <dir>`: per part and mesher, the final segmentation for test/surfaceScore.mjs:
// <stem>.seg (int32 segment per facet, -1 unpainted), <stem>.member (float32 refiner membership of
// the fill that claimed the facet, NaN without one), <stem>.pedge (float32 largest boundary
// probability over the facet's edges, NaN without an edge model) and <stem>.json (segment types).
const DUMP_SEG = flag('--dump-seg', null);
if (DUMP_SEG) mkdirSync(DUMP_SEG, { recursive: true });
// Curvature-jump walls (regionGrow.tangentWall) for curved fills; same presets as test/fillEval.mjs.
const CURVED = ['cylinder', 'cone', 'sphere', 'torus'];
const TANGENT_PRESETS = {
  none: null,
  c80: { abs: 0.1, rel: 0.8, types: CURVED },
  c65: { abs: 0.1, rel: 0.65, types: CURVED },
  c50: { abs: 0.1, rel: 0.5, types: CURVED },
};
const TANGENT = flag('--tangent', 'none');
if (!(TANGENT in TANGENT_PRESETS)) { console.error(`unknown tangent preset ${TANGENT}`); process.exit(2); }
// The learned face-boundary wall (js/edgeWall.js): `--edge-wall <run dir>` plus a threshold named by
// the share of within-face edges it wrongly walls. `--edge-no-plane` leaves plane fills to the 2
// degree crease, which they already do better with.
const EDGE_WALL_DIR = flag('--edge-wall', null);
const EDGE = flag('--edge', 'none');
const EDGE_NO_PLANE = args.includes('--edge-no-plane');
if (EDGE_NO_PLANE) args.splice(args.indexOf('--edge-no-plane'), 1);
const [DIR, MODEL] = args;
if (!DIR || !MODEL || !['random', 'coherent', 'largest'].includes(STRATEGY)) {
  console.error('usage: node test/autoSeed.mjs <dir> <model.onnx> [--strategy coherent|random|largest] [--method model|hyp]');
  process.exit(2);
}

const session = await ort.InferenceSession.create(MODEL, { intraOpNumThreads: 2, interOpNumThreads: 1 });
let extended = false;
try { extended = JSON.parse(readFileSync(MODEL.replace(/\.onnx$/, '.json'), 'utf8')).features === 'v6'; } catch { /* base model */ }
const runner = async (features, n, graph) => {
  const out = await session.run({
    features: new ort.Tensor('float32', features, [1, n, features.length / n]),
    mask: new ort.Tensor('float32', new Float32Array(n).fill(1), [1, n]),
    ...graphFeeds(session, graph, n),
  });
  return { sameLogit: out.same_logit.data, typeLogit: Array.from(out.type_logit.data), surface: out.surface ? out.surface.data : null };
};

let refiner = null;
if (REFINE) {
  const rs = await ort.InferenceSession.create(REFINE, { intraOpNumThreads: 2, interOpNumThreads: 1 });
  const refineEdgeDim = edgeDimOf(REFINE);
  refiner = async (features, n, graph) => ({ sameLogit: (await rs.run({
    features: new ort.Tensor('float32', features, [1, n, features.length / n]),
    mask: new ort.Tensor('float32', new Float32Array(n).fill(1), [1, n]),
    ...graphFeeds(rs, graph, n, refineEdgeDim),
  })).same_logit.data });
}

let edgeMeta = null;
let edgeRunner = null;
let edgeWallFn = null;
if (EDGE_WALL_DIR && EDGE !== 'none') {
  edgeMeta = JSON.parse(readFileSync(join(EDGE_WALL_DIR, 'edge.json'), 'utf8'));
  const edgeSession = await ort.InferenceSession.create(join(EDGE_WALL_DIR, 'edge.onnx'),
    { intraOpNumThreads: 2, interOpNumThreads: 1 });
  edgeRunner = async (features, n) => {
    const out = await edgeSession.run({ features: new ort.Tensor('float32', features, [n, features.length / n]) });
    return out.logit.data;
  };
  const key = { fpr1: 'fpr0.01', fpr2: 'fpr0.02', fpr5: 'fpr0.05' }[EDGE];
  const entry = key && edgeMeta.thresholds[key];
  if (!entry) { console.error(`unknown edge preset ${EDGE}`); process.exit(2); }
  edgeWallFn = edgeWall(1 / (1 + Math.exp(-entry.logit)),
                        EDGE_NO_PLANE ? ['cylinder', 'cone', 'sphere', 'torus'] : null);
}

const copy = (b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.length);
const load = (stem) => ({
  P: Float64Array.from(new Float32Array(copy(readFileSync(join(DIR, `${stem}.pos`))))),
  F: new Int32Array(copy(readFileSync(join(DIR, `${stem}.fid`)))),
  T: new Uint8Array(copy(readFileSync(join(DIR, `${stem}.typ`)))),
});

let x = 31;
const rng = () => (x = (x * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;

async function orderFor(facets, adjacency) {
  if (STRATEGY === 'coherent') return seedOrder(await coherentScores(facets, adjacency, { chunk: 1e9 }), facets);
  const ids = [];
  for (let t = 0; t < facets.count; t++) if (!facets.degenerate[t]) ids.push(t);
  if (STRATEGY === 'largest') return ids.sort((a, b) => facets.areas[b] - facets.areas[a]);
  for (let i = ids.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [ids[i], ids[j]] = [ids[j], ids[i]]; }
  return ids;
}

/**
 * Why a face is not clean, which sizes the merge pass:
 *   split  the best segment is pure but holds < 95% of the face: the face was cut into pieces,
 *          which a merge of neighbouring segments could repair
 *   spill  the best segment is < 95% the face's surface type: it swallowed a neighbour, which no
 *          merge can undo
 * For split faces it also counts how often the pieces covering at least 5% of the face are
 * ADJACENT and carry the SAME fitted type, i.e. how many a same-type merge would join.
 */
function scorePart(F, T, seg, facets, segType = null, adjacency = null) {
  const faces = new Map();
  for (let t = 0; t < facets.count; t++) {
    if (facets.degenerate[t]) continue;
    const f = faces.get(F[t]) ?? { tris: [], area: 0, type: T[t] };
    f.tris.push(t); f.area += facets.areas[t];
    faces.set(F[t], f);
  }
  const segSize = new Map();
  const segTypeCount = new Map();
  for (let t = 0; t < facets.count; t++) {
    if (facets.degenerate[t] || seg[t] < 0) continue;
    segSize.set(seg[t], (segSize.get(seg[t]) ?? 0) + 1);
    const m = segTypeCount.get(seg[t]) ?? new Map();
    m.set(T[t], (m.get(T[t]) ?? 0) + 1);
    segTypeCount.set(seg[t], m);
  }
  // Which segment pairs touch, for the mergeable count.
  const touching = new Set();
  if (adjacency) {
    for (const [a, edges] of adjacency) {
      for (const e of edges) {
        const b = e.neighbor;
        if (b < a || seg[a] < 0 || seg[b] < 0 || seg[a] === seg[b]) continue;
        touching.add(seg[a] < seg[b] ? `${seg[a]}|${seg[b]}` : `${seg[b]}|${seg[a]}`);
      }
    }
  }
  let clean = 0, cleanArea = 0, total = 0, totalArea = 0;
  let split = 0, splitArea = 0, spill = 0, spillArea = 0, mergeable = 0, mergeableArea = 0;
  for (const f of faces.values()) {
    total++; totalArea += f.area;
    const overlap = new Map();
    for (const t of f.tris) if (seg[t] >= 0) overlap.set(seg[t], (overlap.get(seg[t]) ?? 0) + 1);
    let best = -1, bestN = 0;
    for (const [s, k] of overlap) if (k > bestN) { best = s; bestN = k; }
    if (best < 0) continue;
    const recall = bestN / f.tris.length;
    const purity = (segTypeCount.get(best).get(f.type) ?? 0) / segSize.get(best);
    if (recall >= 0.95 && purity >= 0.95) { clean++; cleanArea += f.area; continue; }
    if (purity < 0.95) { spill++; spillArea += f.area; continue; }
    split++; splitArea += f.area;
    const pieces = [...overlap].filter(([, k]) => k >= 0.05 * f.tris.length).map(([s]) => s);
    if (pieces.length < 2) continue;
    const sameType = segType ? pieces.every((s) => segType[s] && segType[s] === segType[pieces[0]]) : false;
    let joined = true;
    for (const s of pieces) {
      if (s === best) continue;
      const key = s < best ? `${s}|${best}` : `${best}|${s}`;
      if (!touching.has(key)) { joined = false; break; }
    }
    if (sameType && joined) { mergeable++; mergeableArea += f.area; }
  }
  return { faces: total, clean: clean / total, cleanArea: cleanArea / totalArea, segments: segSize.size,
           split: split / total, splitArea: splitArea / totalArea, spill: spill / total,
           spillArea: spillArea / totalArea, mergeable: mergeable / total, mergeableArea: mergeableArea / totalArea };
}

const manifest = JSON.parse(readFileSync(join(DIR, 'manifest.json'), 'utf8'));
const meshers = MESHERS ?? manifest.variants;
const byParity = [...manifest.parts.entries()]
  .filter(([i]) => PARITY === 'all' || (i % 2 === 0) === (PARITY === 'even'));
const sharded = SHARD ? byParity.filter((_, j) => j % SHARD[1] === SHARD[0]) : byParity;
const parts = PART_LIMIT ? sharded.slice(0, PART_LIMIT) : sharded;
const rows = [];
const t0 = performance.now();

for (const [done, [pi, entry]] of parts.entries()) {
  for (const mesher of meshers) {
    const v = entry.variants[mesher];
    if (!v) continue;
    const { P, F, T } = load(v.stem);
    const mesh = preparePaintPositions(P);
    const { facets, adjacency } = mesh;
    const ts = performance.now();
    if (edgeRunner) await annotateEdges(facets, adjacency, edgeRunner, edgeMeta.columns);
    const order = await orderFor(facets, adjacency);
    const seg = new Int32Array(facets.count).fill(-1);
    const member = DUMP_SEG ? new Float32Array(facets.count).fill(NaN) : null;
    const segType = [];
    let nextSeg = 0;
    const result = await autoPaint({
      facets, order, owner: new Int32Array(facets.count).fill(-1), minClaim: MIN_CLAIM, minKeep: MIN_KEEP,
      onDiscard: (claim) => { for (const t of claim) { seg[t] = -1; if (member) member[t] = NaN; } },
      adjacency, absorbUpTo: ABSORB, forceSeed: FORCE_SEED,
      onAbsorb: (island, id) => { for (const t of island) seg[t] = id; },
      // Refined fills run through paintCore.refineFill, the studio's own code (paintWith `refine`).
      fill: (seed, taken) => paintWith(runner, seed, mesh, {
        method: METHOD, type: 'auto', taken, extended, reportFit: false,
        grow: {
          ...(TANGENT_PRESETS[TANGENT] ? { tangent: TANGENT_PRESETS[TANGENT] } : {}),
          ...(edgeWallFn ? { wall: edgeWallFn } : {}),
        },
        ...(refiner ? { refine: { runner: refiner, cut: REFINE_CUT, crf: CRF } } : {}),
      }),
      onSegment: (claim, out) => {
        const id = nextSeg++;
        for (const t of claim) seg[t] = id;
        if (member && out.member) for (const t of claim) { const m = out.member.get(t); if (m !== undefined) member[t] = m; }
        segType[id] = out.fitter ?? null;
        return id;
      },
    });
    let final = seg;
    let merges = 0;
    let pearlChanged = 0;
    if (PEARL) {
      const graph = attachEdgeGraph(P, facets, adjacency);
      const r = pearlRefine(facets, adjacency, graph, seg, segType, pearlOpts);
      final = r.seg;
      pearlChanged = r.changed;
      if (args.includes('--pearl-debug')) {
        // Which true face does each segment mostly hold? A moved facet is FIXED when it leaves a
        // segment dominated by another face for one dominated by its own, BROKEN the other way.
        const dominant = (lab) => {
          const votes = new Map();
          for (let t = 0; t < facets.count; t++) if (lab[t] >= 0) {
            const m = votes.get(lab[t]) ?? new Map();
            m.set(F[t], (m.get(F[t]) ?? 0) + 1);
            votes.set(lab[t], m);
          }
          const dom = new Map();
          for (const [l, m] of votes) dom.set(l, [...m].sort((x, y) => y[1] - x[1])[0][0]);
          return dom;
        };
        const d0 = dominant(seg), d1 = dominant(final);
        let moved = 0, fixed = 0, broken = 0, movedArea = 0, brokenArea = 0;
        const byType = new Map();
        for (let t = 0; t < facets.count; t++) {
          if (seg[t] === final[t] || seg[t] < 0) continue;
          moved++; movedArea += facets.areas[t];
          const was = d0.get(seg[t]) === F[t], now = d1.get(final[t]) === F[t];
          if (!was && now) fixed++;
          if (was && !now) {
            broken++; brokenArea += facets.areas[t];
            const k = `${segType[seg[t]]}->${segType[final[t]]}`;
            byType.set(k, (byType.get(k) ?? 0) + 1);
          }
        }
        console.log(`    part ${pi} ${mesher}: moved ${moved}, fixed ${fixed}, broken ${broken}` +
          ` (area moved ${(100 * movedArea / [...facets.areas].reduce((a, b) => a + b, 0)).toFixed(1)}%,` +
          ` broken ${(100 * brokenArea / [...facets.areas].reduce((a, b) => a + b, 0)).toFixed(1)}%)` +
          `  broken by type ${[...byType].sort((x, y) => y[1] - x[1]).slice(0, 4).map(([k, v]) => `${k} ${v}`).join(', ')}`);
      }
    }
    if (MERGE > 0) {
      // --merge-debug: for the first joins of each part, say what was joined and whether the two
      // pieces really belonged to the same B-rep face. A merge between different true faces is the
      // failure mode to hunt; a merge inside one face is the whole point of the pass.
      let shown = 0, sameFace = 0, crossFace = 0;
      const dominantFace = (id) => {
        const votes = new Map();
        for (let t = 0; t < facets.count; t++) if (seg[t] === id) votes.set(F[t], (votes.get(F[t]) ?? 0) + 1);
        let face = -1, n = 0, size = 0;
        for (const [f, k] of votes) { size += k; if (k > n) { face = f; n = k; } }
        return { face, size };
      };
      const m = mergeRegions(seg, facets, adjacency, {
        maxFragment: MERGE,
        onMerge: (keep, gone, fit, detail) => {
          if (MERGE_DEBUG) {
            const a = dominantFace(keep);
            const b = dominantFace(gone);
            if (a.face === b.face) sameFace++;
            else {
              // The failures are what matter: which fragment went to the wrong host, and on what
              // evidence?
              crossFace++;
              if (shown++ < 12) {
                console.log(`    CROSS host ${keep}(${detail.sizeA} tris, face ${a.face}) took ${gone}` +
                  `(${detail.sizeB} tris, face ${b.face}) as ${detail.stratum} -> ${fit.type}` +
                  `  p90 ${detail.p90.toFixed(2)} tolD  angle ${detail.angle.toFixed(0)} deg` +
                  `  on-surface ${(100 * detail.frac).toFixed(0)}%`);
              }
            }
          }
          segType[keep] = fit.type;
        },
      });
      final = m.seg;
      merges = m.merges;
    }
    let painted = 0;
    for (let t = 0; t < facets.count; t++) if (final[t] >= 0) painted++;
    rows.push({ part: pi, mesher, strategy: STRATEGY, method: METHOD, tangent: TANGENT, merges, pearlChanged,
                ...scorePart(F, T, final, facets, segType, adjacency),
                fills: result.fills, painted: painted / facets.count, ms: performance.now() - ts });
    if (DUMP_SEG) {
      const pedge = new Float32Array(facets.count).fill(NaN);
      if (edgeRunner) {
        for (const [t, es] of adjacency) {
          let m = 0;
          for (const e of es) if (e.boundary > m) m = e.boundary;
          pedge[t] = m;
        }
      }
      const bytes = (a) => new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
      writeFileSync(join(DUMP_SEG, `${v.stem}.seg`), bytes(final));
      writeFileSync(join(DUMP_SEG, `${v.stem}.member`), bytes(member));
      writeFileSync(join(DUMP_SEG, `${v.stem}.pedge`), bytes(pedge));
      writeFileSync(join(DUMP_SEG, `${v.stem}.json`), JSON.stringify({
        part: entry.part, mesher, stem: v.stem, types: segType.map((x) => x ?? null),
        model: basename(MODEL), edge: EDGE_WALL_DIR && EDGE !== 'none' ? `${EDGE_WALL_DIR} ${EDGE}` : null,
        refine: REFINE ? basename(REFINE) : null, crf: CRF }));
    }
  }
  if (done % 10 === 9) console.log(`  ${done + 1} parts, ${((performance.now() - t0) / 1000).toFixed(0)} s`);
}

const mean = (a) => a.reduce((s, v) => s + v, 0) / (a.length || 1);
console.log(`\n${STRATEGY}/${METHOD}/tangent=${TANGENT}: ${parts.length} parts (${PARITY}), model ${basename(dirname(MODEL))}${extended ? ' (v6 inputs)' : ''}, min claim ${MIN_CLAIM}, min keep ${MIN_KEEP}, absorb ${ABSORB},${((performance.now() - t0) / 1000).toFixed(0)} s`);
for (const mesher of meshers) {
  const rs = rows.filter((r) => r.mesher === mesher);
  if (!rs.length) continue;
  console.log(`  ${mesher.padEnd(7)} faces clean ${(100 * mean(rs.map((r) => r.clean))).toFixed(1)}%` +
    `  area clean ${(100 * mean(rs.map((r) => r.cleanArea))).toFixed(1)}%` +
    `  segments/face ${mean(rs.map((r) => r.segments / r.faces)).toFixed(2)}` +
    `  fills ${mean(rs.map((r) => r.fills)).toFixed(0)}  painted ${(100 * mean(rs.map((r) => r.painted))).toFixed(0)}%` +
    `  ${(mean(rs.map((r) => r.ms)) / 1000).toFixed(1)} s/part`);
  console.log(`          not clean: split ${(100 * mean(rs.map((r) => r.split))).toFixed(1)}% of faces` +
    ` (${(100 * mean(rs.map((r) => r.splitArea))).toFixed(1)}% of area), of which same-type and adjacent` +
    ` ${(100 * mean(rs.map((r) => r.mergeable))).toFixed(1)}% (${(100 * mean(rs.map((r) => r.mergeableArea))).toFixed(1)}% of area);` +
    ` spill ${(100 * mean(rs.map((r) => r.spill))).toFixed(1)}% (${(100 * mean(rs.map((r) => r.spillArea))).toFixed(1)}% of area)`);
}
if (JSON_OUT) writeFileSync(JSON_OUT, JSON.stringify(rows));
