/**
 * knit.mjs — js/knit (the studio's solid builder) on one part's patches, the same input
 * ml/solid_rebuild.py --fid-dir takes, so the port can be held against the Python measurements.
 *
 * Usage:  node test/knit.mjs <mesh dir> <patch dir> <stem> [--out row.json] [--step out.step]
 *                            [--repair] [--no-fallback] [--budget 300] [--deadline s] [--faceted] [--no-split] [--no-axes]
 *                            [--no-tidy] [--no-exact-edges] [--no-contact] [--no-ssi] [--no-pinch] [--no-pcurves] [--fix-facets] [--dump-kernel dir] [--facet-audit] [--max-tol] [--no-retry]
 *                            [--axes-opts '{"TANGENT_DEG":0}']
 *
 *   <mesh dir>   <stem>.pos (float32 triangle soup)
 *   <patch dir>  <stem>.fid (a group per triangle) and <stem>.json (test/exportPatchFits.mjs format:
 *                {faces: [{id, type, params}]}), as test/surfaceScore.mjs --patches-out writes them
 *
 * The export as the studio's Export STEP builds it: the groups and types as given. --repair: the studio's
 * flow when that is not analytic, Repair (knit/index.js repairGroups, every move it keeps applied, --budget
 * seconds) and the export again; the row is the second export's, with the first one's tier and the repair's.
 *
 * One part per process, as the Python harness does: run many with test/knitRebuild.mjs.
 */

import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { loadOcct, knitSolid, repairGroups, writeStep, BUILD } from '../js/knit/index.js';
import { subShapes } from '../js/knit/build.js';

const args = process.argv.slice(2);
const flag = (name, dflt) => { const i = args.indexOf(name); if (i < 0) return dflt; const v = args[i + 1]; args.splice(i, 2); return v; };
const bool = (name) => { const i = args.indexOf(name); if (i < 0) return false; args.splice(i, 1); return true; };
const OUT = flag('--out', null);
const STEP = flag('--step', null);
const BUDGET = Number(flag('--budget', '300'));
const REPAIR = bool('--repair');
const NO_SPLIT = bool('--no-split');
const NO_AXES = bool('--no-axes');
const NO_TIDY = bool('--no-tidy');
const NO_RETRY = bool('--no-retry');          // no plain retry of a clean build that is not analytic
const AXES_OPTS = JSON.parse(flag('--axes-opts', '{}'));
const NO_EXACT_EDGES = bool('--no-exact-edges');
const NO_CONTACT = bool('--no-contact');
const NO_SSI = bool('--no-ssi');
const NO_PINCH = bool('--no-pinch');
const NO_PCURVES = bool('--no-pcurves');
if (NO_EXACT_EDGES) BUILD.EXACT_EDGES = false;
if (NO_CONTACT) BUILD.CONTACT = false;
if (NO_SSI) BUILD.SSI = false;
if (NO_PINCH) BUILD.PINCH = false;
if (NO_PCURVES) BUILD.PCURVES = false;
if (bool('--fix-facets')) BUILD.FIX_FACETS = true;   // facets through ShapeFix with the rest, as before 2026-11-03
// every build's shell as handed to the kernel, before ShapeFix: <stem>.<seq>.brep (the shell), .step (it as a
// solid) and .json (mesh volume), for test/kernelBench.mjs
const DUMP_KERNEL = flag('--dump-kernel', null);
// every facet of the result against its own corners and its mesh triangle (test/facetAudit.mjs): row.audit
const FACET_AUDIT = bool('--facet-audit');
const MAX_TOL = bool('--max-tol');   // row.max_tol: the result's largest vertex and edge tolerance (mm)
const NO_FALLBACK = bool('--no-fallback');
const DEADLINE = Number(flag('--deadline', 'Infinity'));
const FACETED = bool('--faceted');
const [MESH_DIR, PATCH_DIR, STEM] = args;
if (!STEM) { console.error('usage: node test/knit.mjs <mesh dir> <patch dir> <stem>'); process.exit(2); }

const copy = (b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.length);
const positions = Float64Array.from(new Float32Array(copy(readFileSync(join(MESH_DIR, `${STEM}.pos`)))));
const labels = new Int32Array(copy(readFileSync(join(PATCH_DIR, `${STEM}.fid`))));
// <stem>.fits.json (test/studioExport.mjs --dump) or <stem>.json (surfaceScore --patches-out)
const fitsPath = existsSync(join(PATCH_DIR, `${STEM}.fits.json`)) ? join(PATCH_DIR, `${STEM}.fits.json`) : join(PATCH_DIR, `${STEM}.json`);
const fits = JSON.parse(readFileSync(fitsPath, 'utf8'));
const surfaces = new Map(fits.faces.filter((f) => f.params).map((f) => [f.id, [f.type, f.params]]));

const t0 = performance.now();
const oc = await loadOcct();
if (DUMP_KERNEL) {
  mkdirSync(DUMP_KERNEL, { recursive: true });
  let seq = 0;
  BUILD.DUMP = (oc, shell, mesh) => {
    const base = join(DUMP_KERNEL, `${STEM}.${String(seq).padStart(2, '0')}`);
    const solid = new oc.TopoDS_Solid();
    const b = new oc.BRep_Builder();
    b.MakeSolid(solid);
    b.Add(solid, shell);
    const tmp = `/tmp/dump-${seq}.brep`;
    oc.BRepTools.Write_3(shell, tmp, new oc.Message_ProgressRange_1());
    writeFileSync(`${base}.brep`, oc.FS.readFile(tmp));
    oc.FS.unlink(tmp);
    // opencascade.js's STEP writer crashes on a curved face's edge with no pcurve (it can't project one), so the
    // STEP copy gets them first; the .brep is the shell untouched
    try {
      const copy = new oc.BRepBuilderAPI_Copy_2(solid, true, false).Shape();
      const sae = new oc.ShapeAnalysis_Edge(), sfe = new oc.ShapeFix_Edge();
      for (const f of subShapes(oc, copy, oc.TopAbs_ShapeEnum.TopAbs_FACE)) {
        const face = oc.TopoDS.Face_1(f);
        for (const e of subShapes(oc, face, oc.TopAbs_ShapeEnum.TopAbs_EDGE)) {
          const edge = oc.TopoDS.Edge_1(e);
          if (!sae.HasPCurve_1(edge, face)) sfe.FixAddPCurve_1(edge, face, false, 1e-7);
        }
      }
      writeFileSync(`${base}.step`, writeStep(oc, copy));
    } catch (err) { console.error(`dump ${seq}: ${err}`); }
    writeFileSync(`${base}.json`, JSON.stringify({ stem: STEM, seq, mesh_volume: mesh.volume() }));
    seq++;
  };
}
const loadSecs = (performance.now() - t0) / 1000;
let row;
try {
  const options = { fallback: !NO_FALLBACK, deadline: DEADLINE, mode: FACETED ? 'faceted' : 'knit', split: !NO_SPLIT, axes: !NO_AXES, axesOptions: AXES_OPTS, tidy: !NO_TIDY, retryPlain: !NO_RETRY };
  let { solid, report } = knitSolid(oc, { positions, labels, surfaces }, options);
  row = { stem: STEM, ...report, load_secs: Math.round(loadSecs * 10) / 10 };
  if (REPAIR && !FACETED && report.tier !== 'analytic') {
    const r = repairGroups(oc, { positions, labels, surfaces }, { budget: BUDGET, split: !NO_SPLIT, axes: !NO_AXES, axesOptions: AXES_OPTS });
    ({ solid, report } = knitSolid(oc, { positions, labels: r.labels, surfaces: r.surfaces }, options));
    row = { stem: STEM, ...report, load_secs: row.load_secs, first: { tier: row.tier, secs: row.secs },
      repair: { ok: r.ok, moves: r.moves, problems: r.problems, builds: r.info.repair_builds, secs: r.info.secs } };
  }
  if (FACET_AUDIT && solid) row.audit = (await import('./facetAudit.mjs')).facetAudit(oc, solid, positions);
  if (MAX_TOL && solid) row.max_tol = (await import('./facetAudit.mjs')).maxTolerance(oc, solid);
  if (STEP && solid) writeFileSync(STEP, writeStep(oc, solid));
} catch (err) {
  row = { stem: STEM, tier: 'none', closed: false, valid: false, exception: String(err?.stack ?? err).slice(0, 300) };
}
if (OUT) writeFileSync(OUT, JSON.stringify(row));
console.log(JSON.stringify({ ...row, info: undefined }));
