// Summary of a face suite run (test/faceSuite.mjs results.jsonl): parts built, failures by reason, OCCT checks,
// time, and per face (freeform: Revolution / Extrusion / BSpline / Other; analytic: the rest) the share within
// 1e-3 and 1e-4 of the part's diagonal at the face's mesh vertices. --index writes dump/index.json for twosided.py.
// node test/faceSuiteSummary.mjs <out dir> [--index]
import fs from 'node:fs';
import path from 'node:path';

const [dir] = process.argv.slice(2);
const rows = fs.readFileSync(path.join(dir, 'results.jsonl'), 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l));
const by = new Map();
for (const r of rows) by.set(r.stem, r);                     // a rerun replaces an earlier row
const all = [...by.values()];
const q = (xs, f) => { const s = [...xs].sort((a, b) => a - b); return s.length ? s[Math.min(s.length - 1, Math.floor(f * s.length))] : NaN; };
// good: closed and valid in OCCT, with the faces written (one per true face, or a split face's pieces)
const built = all.filter((r) => !r.error), good = built.filter((r) => r.occt?.closed && r.occt?.valid && r.occt?.faces === r.stepFaces);
const split = good.filter((r) => r.splits?.length);
console.log(`parts ${all.length}: built ${built.length}, OCCT closed + valid + the faces written ${good.length} (${good.length - split.length} one face per true face, ${split.length} with faces split: ${split.map((r) => r.splits.reduce((s, x) => s + x.pieces - 1, 0)).reduce((a, b) => a + b, 0)} extra faces)`);
const reason = (e) => e.replace(/face \d+/g, 'face N').replace(/group \S+/g, 'group N').replace(/\d+(\.\d+)? s\b/g, 'N s').replace(/vertex \d+/g, 'vertex N').slice(0, 110);
const reasons = new Map();
for (const r of all.filter((x) => x.error)) reasons.set(reason(r.error), (reasons.get(reason(r.error)) ?? 0) + 1);
console.log('failures:');
for (const [k, n] of [...reasons].sort((a, b) => b[1] - a[1])) console.log(`  ${String(n).padStart(3)}  ${k}`);
const bad = built.filter((r) => !good.includes(r));
if (bad.length) {
  console.log('built but not good in OCCT:');
  for (const r of bad) console.log(`  ${r.stem}: ${JSON.stringify(r.occt)} (faces written ${r.stepFaces}, true faces ${r.faces})`);
}
if (split.length) {
  console.log('good with faces split:');
  for (const r of split) console.log(`  ${r.stem}: ${r.splits.map((x) => `face ${x.origin} in ${x.pieces}`).join(', ')}`);
}
const ms = built.map((r) => r.ms / 1000);
console.log(`time (built parts): median ${q(ms, 0.5).toFixed(1)} s, p90 ${q(ms, 0.9).toFixed(1)} s, max ${Math.max(...ms).toFixed(1)} s; triangles median ${q(built.map((r) => r.triangles), 0.5)}`);
const et = built.map((r) => r.edgeTol);
console.log(`edges on their faces (of diagonal): median ${q(et, 0.5).toExponential(1)}, p90 ${q(et, 0.9).toExponential(1)}, max ${Math.max(...et).toExponential(1)}`);
const FREE = new Set(['Revolution', 'Extrusion', 'BSpline', 'Other']);
for (const [label, pick] of [['freeform faces', (f) => FREE.has(f.type)], ['analytic faces', (f) => !FREE.has(f.type)]]) {
  const fs2 = built.flatMap((r) => r.faceErr.filter(pick));
  if (!fs2.length) continue;
  const n3 = fs2.filter((f) => f.err < 1e-3).length, n4 = fs2.filter((f) => f.err < 1e-4).length;
  console.log(`${label} (built parts): ${fs2.length}, within 1e-3 ${(100 * n3 / fs2.length).toFixed(1)} %, within 1e-4 ${(100 * n4 / fs2.length).toFixed(1)} %, worst ${Math.max(...fs2.map((f) => f.err)).toExponential(1)}`);
}
if (process.argv.includes('--index')) {
  const dump = path.join(dir, 'dump');
  const idx = good.filter((r) => fs.existsSync(path.join(dump, `${r.stem}.out.bin`))).map((r) => ({ name: r.stem, ms: r.ms, bodies: [{ quads: r.stepFaces, faces: r.stepFaces }] }));
  fs.writeFileSync(path.join(dump, 'index.json'), JSON.stringify(idx));
  console.log(`index.json: ${idx.length} parts for twosided.py`);
}
