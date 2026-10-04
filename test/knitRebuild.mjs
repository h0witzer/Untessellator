/**
 * knitRebuild.mjs — test/knit.mjs over a set of parts, one process each, several at once, resumable;
 * with --compare, side by side with ml/rebuild_patches.py's rows for the same patches.
 *
 * Usage:  node test/knitRebuild.mjs <mesh dir> <patch dir> <out dir> [--stems file] [--workers 8]
 *                                   [--timeout 900] [--compare rebuild.json] [-- knit.mjs flags]
 *
 * RECREATED = closed, BRepCheck valid, volume within 1% of the mesh's (the Python harness uses the
 * STEP volume where it knows one; --compare reads true_vol_err for that side when present).
 */

import { readFileSync, writeFileSync, existsSync, mkdirSync, readdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const all = process.argv.slice(2);
const dd = all.indexOf('--');
const extra = dd >= 0 ? all.slice(dd + 1) : [];
const args = dd >= 0 ? all.slice(0, dd) : all;
const flag = (name, dflt) => { const i = args.indexOf(name); if (i < 0) return dflt; const v = args[i + 1]; args.splice(i, 2); return v; };
const STEMS = flag('--stems', null);
const WORKERS = Number(flag('--workers', '8'));
const TIMEOUT = Number(flag('--timeout', '900'));
const COMPARE = flag('--compare', null);
const [MESH_DIR, PATCH_DIR, OUT_DIR] = args;
mkdirSync(OUT_DIR, { recursive: true });

let stems = readdirSync(PATCH_DIR).filter((f) => f.endsWith('.fid')).map((f) => f.slice(0, -4)).sort();
if (STEMS) {
  const only = new Set(readFileSync(STEMS, 'utf8').replace(/,/g, ' ').split(/\s+/).filter(Boolean));
  stems = stems.filter((s) => only.has(s));
}
const recreated = (r) => !!(r && r.closed && r.valid && r.vol_err !== null && r.vol_err !== undefined && r.vol_err < 0.01);

function run(stem) {
  const out = join(OUT_DIR, `${stem}.json`);
  if (existsSync(out)) return Promise.resolve();
  return new Promise((resolve) => {
    const p = spawn(process.execPath, [join(HERE, 'knit.mjs'), MESH_DIR, PATCH_DIR, stem, '--out', out, ...extra],
      { stdio: ['ignore', 'ignore', 'pipe'] });
    let err = '';
    p.stderr.on('data', (d) => { err += d; });
    const timer = setTimeout(() => p.kill(), TIMEOUT * 1000);
    p.on('close', (code) => {
      clearTimeout(timer);
      if (!existsSync(out)) writeFileSync(out, JSON.stringify({ stem, tier: 'none', closed: false, valid: false, error: code === null ? 'timed out' : `exit ${code}: ${err.slice(-200)}` }));
      resolve();
    });
  });
}

const queue = stems.slice();
let done = 0;
const t0 = Date.now();
await Promise.all(Array.from({ length: WORKERS }, async () => {
  while (queue.length) {
    await run(queue.shift());
    if (++done % 20 === 0) console.log(`  ${done}/${stems.length}, ${Math.round((Date.now() - t0) / 1000)} s`);
  }
}));
const rows = stems.map((s) => JSON.parse(readFileSync(join(OUT_DIR, `${s}.json`), 'utf8')));
writeFileSync(join(OUT_DIR, '_rows.json'), JSON.stringify(rows));
const tiers = {};
for (const r of rows) if (recreated(r)) tiers[r.tier] = (tiers[r.tier] ?? 0) + 1;
console.log(`\n${rows.length} parts: recreated ${rows.filter(recreated).length} (${Object.entries(tiers).map(([k, v]) => `${k} ${v}`).join(', ')}), ` +
  `errors ${rows.filter((r) => r.error || r.exception).length}, ${Math.round((Date.now() - t0) / 1000)} s`);

if (COMPARE) {
  const py = new Map(JSON.parse(readFileSync(COMPARE, 'utf8')).map((r) => [`${r.part}_${r.mesher}`, r]));
  const pyRec = (r) => {
    const e = r?.true_vol_err ?? r?.vol_err;
    return !!(r && r.closed && r.valid && e !== null && e !== undefined && e < 0.01);
  };
  const pyTier = (r) => (r?.info?.tier ?? (pyRec(r) ? 'analytic' : 'none'));
  const cells = {};
  const diffs = [];
  for (const r of rows) {
    const p = py.get(r.stem);
    if (!p) continue;
    const a = recreated(r) ? r.tier : 'none', b = pyRec(p) ? pyTier(p) : 'none';
    cells[`${b} -> ${a}`] = (cells[`${b} -> ${a}`] ?? 0) + 1;
    if (a !== b) diffs.push(`${r.stem}: python ${b}, js ${a}`);
  }
  console.log('\npython tier -> js tier (recreated parts by tier, "none" otherwise):');
  for (const [k, v] of Object.entries(cells).sort()) console.log(`  ${k.padEnd(24)} ${v}`);
  if (diffs.length) console.log('\ndiffering parts:\n  ' + diffs.join('\n  '));
}
