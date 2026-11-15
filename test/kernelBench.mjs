/**
 * kernelBench.mjs — OCCT against solidcore on the same knitter builds: every shell test/knit.mjs --dump-kernel
 * wrote as it was handed to the kernel. Each kernel gets each build in its own process (with a timeout) and is
 * timed by stage; parsing the file is timed but left out of the kernel's total.
 *
 *   occt       opencascade.js (wasm, the build the studio runs) on <build>.brep, the knitter's own calls:
 *              ShapeFix_Shape, then score (free edges, solid, classifier, BRepCheck, volume), then STEP write
 *   sc         solidcore natively (sc-bench.exe) on <build>.step: step-brep build_body, check_brep Level2,
 *              brep_volume, STEP write; the healer only when that is not a checked solid
 *   sc-wasm    the same sc-bench built for wasm32-wasip1, under Node's WASI
 *   readback   solidcore's STEP output (sc-bench with SC_SAVE=<dir>, given as --saved <dir>) read back by OCCT
 *
 * Usage:  node test/kernelBench.mjs <dump dir> <out dir> --kernel occt|sc|sc-wasm [--sc <sc-bench.exe or .wasm>]
 *                                   [--workers 10] [--timeout 600] [--stems file] [--seq 0]
 */

import { spawn } from 'node:child_process';
import { readFileSync, writeFileSync, existsSync, mkdirSync, readdirSync } from 'node:fs';
import { join, dirname, basename } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const flag = (name, dflt) => { const i = args.indexOf(name); if (i < 0) return dflt; const v = args[i + 1]; args.splice(i, 2); return v; };

const ONE = flag('--one', null);
const READBACK = flag('--readback', null);
if (ONE) await occtOne(ONE);
else if (READBACK) await readback(READBACK, flag('--meta', null));
else await main();

async function main() {
  const KERNEL = flag('--kernel', 'occt');
  const SC = flag('--sc', null);
  const WORKERS = Number(flag('--workers', '10'));
  const TIMEOUT = Number(flag('--timeout', '600'));
  const STEMS = flag('--stems', null);
  const SAVED = flag('--saved', null);   // --kernel readback: the dir of solidcore's STEP files (sc-bench SC_SAVE)
  const SEQ = flag('--seq', null);
  const [DUMP, OUT] = args;
  if (!OUT || (KERNEL === 'sc' || KERNEL === 'sc-wasm') && !SC) {
    console.error('usage: node test/kernelBench.mjs <dump dir> <out dir> --kernel occt|sc|sc-wasm [--sc path]');
    process.exit(2);
  }
  mkdirSync(OUT, { recursive: true });
  let builds = readdirSync(DUMP).filter((f) => f.endsWith('.json')).map((f) => f.slice(0, -5)).sort();
  if (STEMS) {
    const only = new Set(readFileSync(STEMS, 'utf8').split(/\s+/).filter(Boolean));
    builds = builds.filter((b) => only.has(b.replace(/\.\d+$/, '')));
  }
  if (SEQ !== null) builds = builds.filter((b) => Number(b.split('.').pop()) === Number(SEQ));
  const t0 = Date.now();
  let done = 0;
  const run = (b) => new Promise((resolve) => {
    const out = join(OUT, `${b}.${KERNEL}.json`);
    if (existsSync(out)) { resolve(); return; }
    const file = KERNEL === 'readback' ? join(SAVED, `${b}.step`) : join(DUMP, `${b}.${KERNEL === 'occt' ? 'brep' : 'step'}`);
    const meta = JSON.parse(readFileSync(join(DUMP, `${b}.json`), 'utf8'));
    const head = { build: b, stem: meta.stem, seq: meta.seq, kernel: KERNEL, mesh_volume: meta.mesh_volume };
    if (!existsSync(file)) { writeFileSync(out, JSON.stringify({ ...head, error: 'no input file' })); resolve(); return; }
    const cmd = KERNEL === 'occt' ? [process.execPath, [fileURLToPath(import.meta.url), '--one', file]]
      : KERNEL === 'readback' ? [process.execPath, [fileURLToPath(import.meta.url), '--readback', file, '--meta', join(DUMP, `${b}.json`)]]
      : KERNEL === 'sc' ? [SC, [file, '--heal']]
        : [process.execPath, [join(HERE, 'scWasi.mjs'), SC, file, '--heal']];
    const started = Date.now();
    const p = spawn(cmd[0], cmd[1], { stdio: ['ignore', 'pipe', 'pipe'] });
    let stdout = '', stderr = '';
    p.stdout.on('data', (d) => { stdout += d; });
    p.stderr.on('data', (d) => { stderr += d; });
    const timer = setTimeout(() => p.kill(), TIMEOUT * 1000);
    p.on('close', (code) => {
      clearTimeout(timer);
      const runs = stdout.split('\n').filter((l) => l.startsWith('{')).map((l) => { try { return JSON.parse(l); } catch { return null; } }).filter(Boolean);
      const row = { ...head, wall_ms: Date.now() - started, runs };
      if (code !== 0) row.error = code === null ? 'timed out' : `exit ${code}: ${stderr.slice(-200)}`;
      writeFileSync(out, JSON.stringify(row));
      if (++done % 50 === 0) console.log(`  ${done}/${builds.length}, ${Math.round((Date.now() - t0) / 1000)} s`);
      resolve();
    });
  });
  const queue = builds.slice();
  await Promise.all(Array.from({ length: WORKERS }, async () => { while (queue.length) await run(queue.shift()); }));
  console.log(`${builds.length} builds, ${KERNEL}, ${Math.round((Date.now() - t0) / 1000)} s`);
}

/** One build through OCCT, as buildTopo and score run it; prints one JSON line. */
async function occtOne(file) {
  const { loadOcct, writeStep } = await import('../js/knit/index.js');
  const { score, subShapes } = await import('../js/knit/build.js');
  const oc = await loadOcct();
  const meta = JSON.parse(readFileSync(file.replace(/\.brep$/, '.json'), 'utf8'));
  const row = { run: 'occt' };
  const ms = (t) => Math.round((performance.now() - t) * 10) / 10;
  let t = performance.now();
  oc.FS.writeFile('/tmp/in.brep', readFileSync(file));
  const shell = new oc.TopoDS_Shape();
  oc.BRepTools.Read_2(shell, '/tmp/in.brep', new oc.BRep_Builder(), new oc.Message_ProgressRange_1());
  row.parse_ms = ms(t);
  row.file_faces = subShapes(oc, shell, oc.TopAbs_ShapeEnum.TopAbs_FACE).length;
  t = performance.now();
  const fix = new oc.ShapeFix_Shape_2(shell);
  fix.Perform(new oc.Message_ProgressRange_1());
  const result = fix.Shape();
  row.fix_ms = ms(t);
  row.faces = subShapes(oc, result, oc.TopAbs_ShapeEnum.TopAbs_FACE).length;
  t = performance.now();
  const sc = score(oc, result, { volume: () => meta.mesh_volume });
  row.score_ms = ms(t);
  Object.assign(row, { closed: sc.closed, valid: sc.valid, volume: sc.volume ?? null, vol_err: sc.vol_err,
    free_edges: sc.free_edges, nonmanifold_edges: sc.nonmanifold_edges, score_error: sc.error });
  if (sc.solid) {
    // the check and the volume on their own (score runs both, with the classifier and the edge map)
    t = performance.now();
    new oc.BRepCheck_Analyzer(sc.solid, true, false).IsValid_2();
    row.check_ms = ms(t);
    t = performance.now();
    oc.BRepGProp.VolumeProperties_1(sc.solid, new oc.GProp_GProps_1(), false, false, false);
    row.volume_ms = ms(t);
    t = performance.now();
    try {
      const text = writeStep(oc, sc.solid);
      row.export_bytes = text.length;
      // OCCT_SAVE=<dir>: the STEP, for --kernel readback (the same import judge solidcore's output gets)
      if (process.env.OCCT_SAVE) writeFileSync(join(process.env.OCCT_SAVE, basename(file).replace(/\.brep$/, '.step')), text);
    } catch (err) { row.export_error = String(err).slice(0, 120); }
    row.export_ms = ms(t);
  }
  process.stdout.write(`${JSON.stringify(row)}\n`);
}

/** A STEP file solidcore wrote, read back by OCCT (whose reader repairs on import, as CAD importers do) and
 * judged as score judges a build; prints one JSON line. */
async function readback(file, metaPath) {
  const { loadOcct } = await import('../js/knit/index.js');
  const { score, subShapes } = await import('../js/knit/build.js');
  const oc = await loadOcct();
  const meta = JSON.parse(readFileSync(metaPath, 'utf8'));
  const row = { run: 'readback' };
  oc.FS.writeFile('/tmp/in.step', readFileSync(file));
  const reader = new oc.STEPControl_Reader_1();
  if (reader.ReadFile('/tmp/in.step') !== oc.IFSelect_ReturnStatus.IFSelect_RetDone) {
    row.error = 'read failed';
  } else {
    reader.TransferRoots(new oc.Message_ProgressRange_1());
    const shape = reader.OneShape();
    row.faces = subShapes(oc, shape, oc.TopAbs_ShapeEnum.TopAbs_FACE).length;
    const sc = score(oc, shape, { volume: () => meta.mesh_volume });
    Object.assign(row, { closed: sc.closed, valid: sc.valid, volume: sc.volume ?? null, vol_err: sc.vol_err,
      free_edges: sc.free_edges, score_error: sc.error });
  }
  process.stdout.write(`${JSON.stringify(row)}
`);
}
