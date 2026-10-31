/**
 * reportReplay.mjs — a conversion report (the studio's Report button, js/studio/report.js), read face by face;
 * with the mesh in it, the reported export rebuilt.
 *
 * Usage:  node test/reportReplay.mjs <Untessellator-report.zip> [<out dir>]
 *
 * Prints the run (settings, auto-paint, export tier) and every group that went wrong somewhere: flagged by the
 * export check, rejected by the kernel, moved by Repair (in older reports, the export's repair loop), retyped,
 * absorbed or built as facets. Each row
 * says where the group is (centroid, share of the area), how it came to be (auto-paint seed with the model's
 * prediction and confidence and how its type was decided, or a stroke) and how well its surface fits, so a
 * failure can be traced to the step that caused it rather than guessed from a screenshot.
 *
 * With part.stl in the report it also writes test/knit.mjs's input to <out dir> (part.pos, part.fid,
 * part.fits.json; default: a folder named after the report) and builds it once, with no facet tiers: the
 * kernel's verdict on every face exactly as the studio painted it, which a report whose export timed out does
 * not carry. Those verdicts join the table. The full export (facet tiers) is test/knit.mjs on the same folder,
 * and the studio's Repair before it is test/knit.mjs --repair; on a part with hundreds of bad groups the repair
 * runs for most of an hour.
 */

import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { join, dirname, basename } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import { unzipSync, strFromU8 } from 'fflate';

const HERE = dirname(fileURLToPath(import.meta.url));
const [zipPath, outArg] = process.argv.slice(2);
if (!zipPath) {
  console.log('Usage: node test/reportReplay.mjs <Untessellator-report.zip> [<out dir>]');
  process.exit(1);
}
const files = unzipSync(readFileSync(zipPath));
const report = JSON.parse(strFromU8(files['report.json']));
// stltostep-report/1 is the same format, from before the rename
if (!['untessellator-report/1', 'stltostep-report/1'].includes(report.format)) {
  console.log(`warning: report format ${report.format}, this reads untessellator-report/1`);
}

const n = (x) => Number(x).toLocaleString('en-US');
const ex = report.export;
const faces = ex?.faces ?? {};
console.log(`report     ${report.created}  ${report.page}  ${report.browser}`);
console.log(`mesh       ${n(report.mesh.triangles)} triangles, fingerprint ${report.mesh.fingerprint?.slice(0, 16) ?? '-'}, ` +
  `mesh ${report.mesh.included ? 'included' : 'not included'}`);
console.log(`fill       model ${report.settings.model}, method ${report.settings.autoMethod}, refine ${report.settings.refine}, ` +
  `boundary wall ${report.settings.edgeWall}, group typer ${report.settings.groupTyper}`);
const ap = report.autoPaint;
console.log(`auto-paint ${ap ? `${ap.secs.toFixed(0)} s, ${n(ap.groups)} groups, ${ap.freeform} freeform, ${n(ap.unpainted)} unpainted` : 'not run'}`);
console.log(`check      ${report.check ? `${report.check.flagged} groups flagged${report.check.mesh ? `; mesh: ${report.check.mesh}` : ''}` : 'not run'}`);
console.log(`export     ${ex ? (ex.tier ? `tier ${ex.tier} (best ${ex.bestTier}), ${n(ex.facetTris ?? 0)} facet triangles, ` +
  `${ex.free_edges ?? 0} open edges, ${ex.secs} s` : `failed: ${ex.failed}`) : 'not run'}` +
  `${ex?.timedOut ? '; the analytic build was stopped, so no per-face verdicts: rebuild it below' : ''}`);

// Every group's story, keyed by id.
const groups = new Map(report.groups.map((g) => [g.id, { ...g, kernel: [], moves: [] }]));
const row = (id) => groups.get(id) ?? (groups.set(id, { id, kernel: [], moves: [], origin: null }), groups.get(id));
for (const [kind, ids] of Object.entries(faces.problems ?? {})) for (const id of ids) row(id).kernel.push(kind);
for (const m of faces.moves ?? []) {
  for (const id of m.groups ?? [m.group]) row(id).moves.push(`${m.move}${m.to ? `->${m.to}` : ''}${m.kept ? ' kept' : ''}`);
}
for (const [id, from, to] of faces.retyped ?? []) row(id).retyped = `${from}->${to}`;
for (const id of faces.absorbed ?? []) row(id).absorbed = true;
// the studio's Repair (reports since the export stopped repairing): every move it tried, by group
const gr = report.groupRepair;
if (gr) {
  console.log(`repair     ${gr.stopped ? 'stopped' : gr.ok ? 'closed solid' : 'not a solid yet'} after ${gr.secs} s, kept ${gr.kept} of ${gr.moves?.length ?? '?'} moves`);
  for (const m of gr.moves ?? []) {
    for (const id of m.groups ?? [m.group]) row(id).moves.push(`${m.move}${m.to ? ` ${m.from}->${m.to}` : ''}${m.kept ? ' kept' : ''}`);
  }
}
const allFacets = faces.demoted === 'all';
for (const id of allFacets ? [] : faces.demoted ?? []) row(id).facets = true;

// Where groups came from, and how auto-paint's typing went.
const tally = new Map();
let single = 0;
for (const g of report.groups) {
  const o = g.origin;
  const key = o ? `${o.by}${o.typedBy ? ` / typed by ${o.typedBy}` : ''}${o.edited ? ' / edited' : ''}` : 'unknown';
  tally.set(key, (tally.get(key) ?? 0) + 1);
  if (g.triangles === 1) single++;
}
console.log(`groups     ${n(report.groups.length)}, ${n(single)} of one triangle`);
for (const [k, c] of [...tally].sort((a, b) => b[1] - a[1])) console.log(`           ${String(c).padStart(6)}  ${k}`);
if (allFacets) console.log('           the export is the mesh itself: every group went out as facets');

// With the mesh: test/knit.mjs's input, and one build of it for the kernel's verdict on every face.
if (files['part.stl']) {
  const out = outArg ?? join(dirname(zipPath), basename(zipPath, '.zip'));
  mkdirSync(out, { recursive: true });
  const stl = files['part.stl'];
  const view = new DataView(stl.buffer, stl.byteOffset, stl.byteLength);
  const count = view.getUint32(80, true);
  const pos = new Float32Array(count * 9);
  for (let t = 0; t < count; t++) for (let k = 0; k < 9; k++) pos[t * 9 + k] = view.getFloat32(84 + t * 50 + 12 + k * 4, true);
  writeFileSync(join(out, 'part.pos'), Buffer.from(pos.buffer));
  writeFileSync(join(out, 'part.fid'), Buffer.from(files['part.fid']));
  writeFileSync(join(out, 'part.fits.json'), Buffer.from(files['part.fits.json']));
  if (files['view.png']) writeFileSync(join(out, 'view.png'), Buffer.from(files['view.png']));
  console.log(`\nBuilding ${out} once (test/knit.mjs, no facet tiers) for the kernel's verdict on every face…`);
  const rowPath = join(out, 'verdicts.json');
  spawnSync(process.execPath, [join(HERE, 'knit.mjs'), out, out, 'part', '--no-fallback', '--out', rowPath],
    { stdio: ['ignore', 'ignore', 'inherit'] });
  if (existsSync(rowPath)) {
    const r = JSON.parse(readFileSync(rowPath, 'utf8'));
    for (const [kind, ids] of Object.entries(r.faces?.problems ?? {})) for (const id of ids) row(id).verdict = [...(row(id).verdict ?? []), kind];
    const kinds = Object.entries(r.faces?.problems ?? {}).map(([k, v]) => `${k} ${v.length}`);
    console.log(`built      ${r.tier === 'analytic' ? 'a closed, valid solid with every face analytic' : `not a solid yet (${r.free_edges ?? 0} open edges)`}` +
      ` in ${r.secs} s; kernel rejected ${kinds.length ? kinds.join(', ') : 'no face'}`);
    console.log(`           the full export, with the facet tiers: node test/knit.mjs ${out} ${out} part (--repair: the studio's Repair first)`);
  } else console.log('built      the build wrote no result (see the error above)');
} else console.log('\nNo mesh in the report: no build, so only the verdicts the studio recorded.');

// The groups that went wrong somewhere, largest first.
const bad = [...groups.values()].filter((g) => g.flagged || g.kernel.length || g.verdict || g.moves.length || g.retyped || g.absorbed || g.facets)
  .sort((a, b) => (b.area ?? 0) - (a.area ?? 0));
console.log(`\n${bad.length} groups went wrong somewhere (largest first${bad.length > 60 ? ', first 60' : ''}):`);
const fmt = (x) => (x == null ? '-' : x.toPrecision(3));
for (const g of bad.slice(0, 60)) {
  const o = g.origin;
  const how = !o ? '' : o.by === 'auto-paint'
    ? `seed ${o.seed} model:${o.predicted ?? '-'}${o.confidence != null ? ` ${(o.confidence * 100).toFixed(0)}%` : ''} fill:${o.fill ?? '-'} typed:${o.typedBy ?? '-'}`
    : `${o.by} ${o.tool ?? ''} ${o.declared ?? ''}`.trim();
  const what = [
    g.flagged && `check: ${g.flagged.join('; ')}`,
    g.kernel.length && `kernel: ${g.kernel.join(',')}`,
    g.verdict && `kernel (one build): ${g.verdict.join(',')}`,
    g.moves.length && `repair: ${g.moves.join(', ')}`,
    g.retyped && `retyped ${g.retyped}`,
    g.absorbed && 'absorbed',
    g.facets && 'facets',
  ].filter(Boolean).join(' | ');
  console.log(`  #${String(g.id).padEnd(5)} ${String(g.type ?? 'untyped').padEnd(8)} ${String(g.triangles ?? '?').padStart(6)} tris ` +
    `${((g.area ?? 0) * 100).toFixed(2).padStart(6)}% at [${(g.centroid ?? []).map(fmt).join(', ')}] ` +
    `res ${fmt(g.fit?.residual)}  ${how}\n           ${what}`);
}
