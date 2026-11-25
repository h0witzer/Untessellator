/**
 * report.js — a conversion report: this run's numbers and a report file, sent to Untessellator's private inbox.
 *
 * The models ship in models/ and run in the browser, and the page promises that nothing is uploaded, so the
 * only way to learn how they do on real parts is for people to tell us. The Report button builds a report
 * file (a .zip) and, when the user presses Send, uploads it to the inbox (a Cloudflare Worker and an R2 bucket,
 * kept outside this repo), which opens an issue for it in a private repo that only the developer can read.
 * The user can add a note, a reply address and permission to keep the file as a test, or save the file
 * instead. The mesh's file name is left out.
 *
 * The summary separates the failure modes at a glance: which model and fill settings ran, how many groups
 * auto-paint made and of what type (hundreds of one-triangle cylinders is over-segmentation), and the
 * export's tier with what the kernel said about the faces. The file says it per face:
 *   report.json      first in the zip (the inbox checks for it): the summary, the user's note, and every
 *                    group: where it is (centroid, box, area), how it came to be (auto-paint seed,
 *                    the model's prediction and confidence, how its final type was decided, or a stroke),
 *                    its fit residual, what the export check flagged, and the export's verdict per face
 *                    (knit/index.js faceOutcomes: kernel problems, repair moves, retypes, facets)
 *   part.fid         the group of every triangle, as the knitter was given it (Int32)
 *   part.fits.json   every group's fitted surface, as the knitter was given it
 *   view.png         the view, when the user asks for it
 *   part.stl         the mesh as painted (repaired, in the file's coordinates), only when the user asks:
 *                    with it, test/reportReplay.mjs rebuilds the reported export exactly
 */

/** The inbox: a Cloudflare Worker, kept outside this public repo. */
const INBOX = 'https://reports.untessellator.com';
/** The inbox's size limit (its MAX_MB). */
export const INBOX_MAX_BYTES = 95 * 1024 * 1024;

/**
 * Where Send uploads. A copy of the page served from this machine may name a local inbox with
 * ?inbox=http://127.0.0.1:8787 (wrangler dev); the live site always uses its own, so a link can't redirect reports.
 */
export function inboxUrl() {
  const local = /^(localhost|127\.0\.0\.1)$/.test(location.hostname);
  const override = local ? new URLSearchParams(location.search).get('inbox') : null;
  return (override ?? INBOX).replace(/\/$/, '');
}

const n = (x) => Number(x).toLocaleString('en-US');
const pct = (x) => `${(100 * x).toFixed(x < 0.01 ? 1 : 0)}%`;

/** The browser, briefly: "Google Chrome 141 / Windows", or the start of the user agent. */
function browserName() {
  const data = navigator.userAgentData;
  if (data?.brands?.length) {
    const brand = data.brands.find((b) => !/not.?a.?brand|chromium/i.test(b.brand)) ?? data.brands[0];
    return `${brand.brand} ${brand.version}${data.platform ? ` / ${data.platform}` : ''}`;
  }
  return navigator.userAgent.slice(0, 120);
}

/**
 * The report's numbers as plain lines.
 *   triangles   the open mesh's triangle count (0: none open)
 *   repaired    the repair's changes, as the studio words them
 *   model       the seed model's label
 *   settings    the studio's settings object
 *   autoPaint   { secs, groups, freeform, unpainted, stopped } from the last auto-paint, or null
 *   groups      the face groups now, as { type, count } (type null: untyped, exported as a B-spline or facets)
 *   painted     triangles in any group
 *   check       the export check's last flags ({ groups: [[id, reasons]], mesh }), or null
 *   exported    the knitter's report from the last export, { failed } when it threw, or null
 *   groupRepair the last Repair of the groups ({ ok, kept, stopped, secs, moves }), or null
 *   since       edits to the groups since each of those ({ check, exported, repair }): a result made before the groups
 *               last changed is said to describe earlier groups
 */
export function conversionNumbers({ triangles, repaired, model, settings, autoPaint, groups, painted, check, exported,
  groupRepair = null, since = {} }) {
  const older = (k) => (since[k] > 0 ? ` [before the last ${n(since[k])} edit${since[k] === 1 ? '' : 's'}: it describes earlier groups]` : '');
  const lines = [`browser: ${browserName()}`];
  if (!triangles) return [...lines, 'mesh: none open'].join('\n');
  lines.push(`mesh: ${n(triangles)} triangles${repaired.length ? `; repaired: ${repaired.join(', ')}` : ''}`);
  const on = (x) => (x ? 'on' : 'off');
  lines.push(`fill: model ${model}, method ${settings.autoMethod}, refined fills ${on(settings.refine)}, ` +
    `boundary wall ${on(settings.edgeWall)}, group typer ${on(settings.groupTyper)}, curvature wall ${on(settings.autoTangent)}`);
  lines.push(autoPaint
    ? `auto-paint: ${autoPaint.stopped ? 'stopped' : 'done'} in ${autoPaint.secs.toFixed(0)} s, ${n(autoPaint.groups)} new groups` +
      `${autoPaint.freeform ? ` (${autoPaint.freeform} freeform)` : ''}, ${n(autoPaint.unpainted)} triangles left unpainted`
    : 'auto-paint: not run');
  const byType = new Map();
  let single = 0;
  for (const g of groups) {
    const type = g.type ?? 'untyped';
    byType.set(type, (byType.get(type) ?? 0) + 1);
    if (g.count === 1) single++;
  }
  const types = [...byType].sort((a, b) => b[1] - a[1]).map(([t, c]) => `${t} ${n(c)}`).join(', ');
  lines.push(`groups: ${n(groups.length)}${types ? ` (${types})` : ''}; ${n(single)} of one triangle; ` +
    `${n(painted)} of ${n(triangles)} triangles painted`);
  if (check) lines.push(`export check: ${check.groups.length ? `${n(check.groups.length)} groups flagged` : 'no groups flagged'}` +
    `${check.mesh ? `; mesh: ${check.mesh}` : ''}${older('check')}`);
  if (!exported) lines.push('export: not run');
  else if (exported.failed && !exported.tier) lines.push(`export: failed (${exported.failed})${older('exported')}`);
  else {
    const facets = exported.facetTris ? `, ${n(exported.facetTris)} facet triangles (${pct(exported.facetArea ?? 0)} of the area)` : '';
    const open = exported.free_edges ? `, ${n(exported.free_edges)} open edges` : '';
    const why = exported.timedOut ? ', timed out' : exported.failed ? `, build failed (${exported.failed})` : '';
    lines.push(`export: tier ${exported.tier}${exported.bestTier && exported.bestTier !== exported.tier ? ` (best ${exported.bestTier})` : ''}` +
      `${facets}${open}${why}, ${exported.secs} s${older('exported')}`);
    const f = exported.faces;
    if (exported.timedOut || exported.failed) {
      // the faceted rebuild ran in a fresh worker: the analytic build's verdicts went with the one it replaced
      lines.push('export faces: not recorded (the analytic build was stopped); the report file can rebuild it');
    } else if (f) {
      const kinds = Object.entries(f.problems ?? {}).map(([k, v]) => `${k} ${v.length}`);
      lines.push(`export faces: kernel rejected ${kinds.length ? kinds.join(', ') : 'none'}; ` +
        `facets ${f.demoted === 'all' ? 'all' : `${(f.demoted ?? []).length} groups`}` +
        `${f.splined?.length ? `; B-splines ${f.splined.length} groups (${pct(exported.splineArea ?? 0)} of the area)` : ''}`);
    }
  }
  if (groupRepair) {
    lines.push(`repair: ${groupRepair.stopped ? 'stopped' : groupRepair.ok ? 'closed solid' : 'not a solid yet'} after ` +
      `${groupRepair.secs} s, kept ${groupRepair.kept} of ${groupRepair.moves ? groupRepair.moves.length : '?'} moves${older('repair')}`);
  }
  return lines.join('\n');
}

/**
 * Where each group is and how big, from the knitter's input (positions in the file's coordinates, a
 * label per triangle): Map id -> { triangles, area (share of the part's), centroid, box: [min, max] }.
 * With the fits, these find a reported face again without the mesh.
 */
export function placeGroups(positions, labels) {
  const out = new Map();
  let total = 0;
  for (let t = 0; t < labels.length; t++) {
    const o = t * 9;
    const a = [positions[o], positions[o + 1], positions[o + 2]];
    const e1 = [positions[o + 3] - a[0], positions[o + 4] - a[1], positions[o + 5] - a[2]];
    const e2 = [positions[o + 6] - a[0], positions[o + 7] - a[1], positions[o + 8] - a[2]];
    const area = 0.5 * Math.hypot(e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]);
    total += area;
    let g = out.get(labels[t]);
    if (!g) out.set(labels[t], g = { triangles: 0, area: 0, sum: [0, 0, 0], min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] });
    g.triangles++;
    g.area += area;
    for (let v = 0; v < 3; v++) for (let k = 0; k < 3; k++) {
      const x = positions[o + v * 3 + k];
      g.sum[k] += (x * area) / 3;
      if (x < g.min[k]) g.min[k] = x;
      if (x > g.max[k]) g.max[k] = x;
    }
  }
  const r = (x) => Number(x.toPrecision(7));
  for (const [id, g] of out) {
    out.set(id, { triangles: g.triangles, area: r(g.area / Math.max(total, 1e-300)),
      centroid: g.sum.map((s) => r(s / Math.max(g.area, 1e-300))), box: [g.min.map(r), g.max.map(r)] });
  }
  return out;
}

/** A binary STL of a triangle soup (9 numbers per triangle), normals left zero as most readers recompute them. */
export function binaryStl(positions) {
  const count = positions.length / 9;
  const bytes = new Uint8Array(84 + 50 * count);
  const view = new DataView(bytes.buffer);
  bytes.set(new TextEncoder().encode('Untessellator conversion report mesh'), 0);
  view.setUint32(80, count, true);
  for (let t = 0; t < count; t++) {
    for (let k = 0; k < 9; k++) view.setFloat32(84 + t * 50 + 12 + k * 4, positions[t * 9 + k], true);
  }
  return bytes;
}

/** SHA-256 of the mesh as loaded, in hex: two reports of the same part share it. Null where the browser has no subtle crypto. */
export async function fingerprint(floats) {
  if (!globalThis.crypto?.subtle) return null;
  const digest = await crypto.subtle.digest('SHA-256', floats.buffer.slice(floats.byteOffset, floats.byteOffset + floats.byteLength));
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

/**
 * The report file's report.json, as an object. `groups` are the studio's ({ id, type, fit }), `placed` is
 * placeGroups' map, `origins` the studio's per-group origins (a group the Repair changed lists its retypes
 * under `repaired`), `check` the export check's last flags ({ groups: [[id, reasons]], mesh }), `groupRepair`
 * the last Repair of the groups, `summary` conversionNumbers' lines. When the user sends it, `message` holds
 * their note, reply address and whether the file may be kept to test future versions ({ note, contact, keep }).
 */
export function reportJson({ summary = '', page, settings, model, triangles, repair, fingerprint: print, includesMesh, autoPaint,
  groups, placed, origins, check, exported, groupRepair = null, since = {} }) {
  const flags = new Map(check?.groups ?? []);
  return {
    format: 'untessellator-report/1',
    created: new Date().toISOString(),
    page,
    browser: browserName(),
    summary,
    message: null,
    settings: { ...settings, model },
    mesh: { triangles, fingerprint: print, included: includesMesh, repair: repair ?? null },
    autoPaint: autoPaint ?? null,
    check: check ? { mesh: check.mesh, flagged: check.groups.length } : null,
    export: exported ?? null,
    groupRepair,
    // edits to the groups since the check, the export and the Repair: above zero, that result is of earlier groups
    editsSince: { check: since.check ?? 0, export: since.exported ?? 0, repair: since.repair ?? 0 },
    groups: groups.map((g) => ({
      id: g.id,
      type: g.type ?? null,
      ...(placed.get(g.id) ?? {}),
      fit: g.fit ? (g.fit.params ? { residual: g.fit.relResidual ?? null } : { failed: g.fit.reason }) : null,
      origin: origins.get(g.id) ?? null,
      flagged: since.check > 0 ? null : flags.get(g.id) ?? null,   // a check older than the groups flags none of them
    })),
  };
}

/**
 * Upload a report file to the inbox. `onProgress(share)` follows the upload (0..1); `signal` aborts it.
 * Resolves the report's reference ("20261121-K7Q2MX"); rejects with the inbox's reason, or a network error.
 * Everything the inbox needs is in the file's report.json. XMLHttpRequest, not fetch, for the upload's progress.
 */
export function sendToInbox(bytes, { onProgress, signal } = {}) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${inboxUrl()}/reports`);
    xhr.setRequestHeader('Content-Type', 'application/zip');
    xhr.upload.onprogress = (event) => { if (event.lengthComputable) onProgress?.(event.loaded / event.total); };
    xhr.onload = () => {
      let body = null;
      try { body = JSON.parse(xhr.responseText); } catch { /* not JSON: a proxy's page */ }
      if (xhr.status === 201 && body?.id) resolve(body.id);
      else reject(new Error(body?.error ?? `the inbox answered ${xhr.status}`));
    };
    xhr.onerror = () => reject(new Error('no connection to the inbox'));
    xhr.onabort = () => reject(Object.assign(new Error('stopped'), { name: 'AbortError' }));
    signal?.addEventListener('abort', () => xhr.abort(), { once: true });
    xhr.send(bytes);
  });
}
