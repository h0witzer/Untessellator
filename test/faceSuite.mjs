// The face suite: every part of an eval set (.pos/.fid/.typ + manifest.json) through test/faceSuitePart.mjs,
// each in its own process with a time limit, a pool of them at once; results.jsonl and a summary.
// node test/faceSuite.mjs <mesh dir> <out dir> [--jobs 8] [--timeout 300] [--limit N]
import fs from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const args = process.argv.slice(2);
const flag = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args.splice(i, 2)[1] : d; };
const jobs = Number(flag('--jobs', 8)), timeout = Number(flag('--timeout', 300)), limit = Number(flag('--limit', 1e9));
const [meshDir, outDir] = args;
fs.mkdirSync(outDir, { recursive: true });
const manifest = JSON.parse(fs.readFileSync(path.join(meshDir, 'manifest.json'), 'utf8'));
const stems = manifest.parts.map((p) => Object.values(p.variants)[0].stem).slice(0, limit);
const partScript = path.join(path.dirname(fileURLToPath(import.meta.url)), 'faceSuitePart.mjs');
const resultsFile = path.join(outDir, 'results.jsonl');
const done = new Set(fs.existsSync(resultsFile) ? fs.readFileSync(resultsFile, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l).stem) : []);
const queue = stems.filter((s) => !done.has(s));
console.log(`${stems.length} parts, ${done.size} done before, ${queue.length} to run, ${jobs} at once`);

const runOne = (stem) => new Promise((resolve) => {
  const t0 = Date.now();
  const child = spawn(process.execPath, ['--max-old-space-size=6000', partScript, meshDir, stem, outDir], { stdio: ['ignore', 'pipe', 'pipe'] });
  let stdout = '', stderr = '';
  child.stdout.on('data', (d) => { stdout += d; });
  child.stderr.on('data', (d) => { stderr += d; });
  const timer = setTimeout(() => child.kill('SIGKILL'), timeout * 1000);
  child.on('close', (code) => {
    clearTimeout(timer);
    const line = stdout.trim().split('\n').filter((l) => l.startsWith('{')).pop();
    let row;
    try { row = JSON.parse(line); } catch { row = { stem, error: code === null ? `timed out after ${timeout} s` : `crashed (${code}): ${stderr.trim().split('\n').slice(-3).join(' | ').slice(0, 300)}` }; }
    row.wall = (Date.now() - t0) / 1000;
    fs.appendFileSync(resultsFile, JSON.stringify(row) + '\n');
    resolve(row);
  });
});
let next = 0, finished = 0;
await Promise.all(Array.from({ length: jobs }, async () => {
  while (next < queue.length) {
    const stem = queue[next++], r = await runOne(stem);
    finished++;
    const ok = r.occt?.closed && r.occt?.valid && r.occt?.faces === r.faces;
    console.log(`[${finished}/${queue.length}] ${stem}: ${r.error ?? (ok ? 'ok' : `occt ${JSON.stringify(r.occt)}`)} ${r.ms ? (r.ms / 1000).toFixed(1) + ' s' : ''}`);
  }
}));
console.log('done');
