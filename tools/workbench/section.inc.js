// ── workbench: color by, every group's fit, skip, Fix bad areas ──────────────

/**
 * Every view measures: each corner's distance off its group's surface, x the part diagonal (a decade level for
 * the heat map), each group's farthest triangle, and a verdict on the export check's bars. Past DOUBT.DEV (1e-3)
 * is 'off' (right 94-100% of the time on Onshape parts), past PREFLIGHT.DEV (1e-5) 'maybe' (about 65%), else 'on';
 * a group the export check or the kernel flags is 'off' too (a narrow arc, a seam gap, a rejected face), and a
 * skipped group is 'skip'. Color by: 'groups' each its own color, 'type' its surface type, 'fit' the verdict (the
 * reserved status steps), 'deviation' the renderer's heat ramp, per pixel.
 */
const VERDICTS = {
  off: { rgb: hexRgb('#d03b3b'), label: 'Likely wrong' },
  maybe: { rgb: hexRgb('#fab219'), label: 'Doubtful' },
  on: { rgb: hexRgb('#0ca30c'), label: 'On its surface' },
  none: { rgb: [0.42, 0.42, 0.42], label: 'No surface' },
  skip: { rgb: SKIP_RGB, label: 'Skipped' },
};
const NO_SURFACE = VERDICTS.none.rgb;
// Levels are clamped: an exact zero has no log, and an infinite one would interpolate to NaN on the GPU.
const LEVEL_RANGE = [-12, 3];

/**
 * Measure triangle t against its group's surface: each corner's level into mesh.levels (NO_LEVEL with no
 * surface), and the farthest corner's distance, x the diagonal, returned (NaN with no surface).
 */
function measureTriangle(t) {
  const { store, levels, model } = mesh;
  const fit = store.groups.get(store.owner[t])?.fit;
  if (!fit?.params) { levels.fill(NO_LEVEL, t * 3, t * 3 + 3); return NaN; }
  const diag = 2 * model.radius;
  const P = model.positions;
  let worst = 0;
  for (let k = 0; k < 3; k++) {
    const o = t * 9 + k * 3;
    // a cap the repair added: the export puts its corners on the surface (caps.js)
    let d = mesh.synthetic?.[t] ? 0 : Math.abs(distNormal(fit.type, fit.params, [P[o], P[o + 1], P[o + 2]])[0]) / diag;
    if (Number.isNaN(d)) d = Infinity;   // a NaN distance counts as far
    levels[t * 3 + k] = Math.min(LEVEL_RANGE[1], Math.max(LEVEL_RANGE[0], Math.log10(d)));
    if (d > worst) worst = d;
  }
  return worst;
}

function deviationBand(d) {
  if (Number.isNaN(d)) return 'none';
  return d <= PREFLIGHT.DEV ? 'on' : d <= DOUBT.DEV ? 'maybe' : 'off';
}

const isSkipGroup = (id) => mesh?.store.groups.get(id)?.type === 'skip';

/**
 * Recompute the verdict of groups `ids` (every group when null) from their triangles' distances and the flags;
 * returns the ids whose verdict changed.
 */
function updateVerdicts(ids = null) {
  const { store, deviation, verdicts, groupWorst, flags } = mesh;
  const worst = new Map();
  for (let t = 0; t < store.owner.length; t++) {
    const id = store.owner[t];
    if (id < 0 || (ids && !ids.has(id))) continue;
    worst.set(id, Math.max(worst.get(id) ?? -Infinity, deviation[t]));   // NaN (no surface) wins, as it should
  }
  if (!ids) { verdicts.clear(); groupWorst.clear(); }
  const changed = new Set();
  for (const id of ids ?? worst.keys()) {
    if (id < 0) continue;
    const w = worst.get(id);
    if (w === undefined) { if (verdicts.delete(id)) changed.add(id); groupWorst.delete(id); continue; }
    groupWorst.set(id, w);
    const verdict = isSkipGroup(id) ? 'skip' : Number.isNaN(w) ? 'none' : flags.has(id) ? 'off' : deviationBand(w);
    if (verdicts.get(id) !== verdict) { verdicts.set(id, verdict); changed.add(id); }
  }
  return changed;
}

/** C: the next color mode. */
function cycleColorBy() {
  setColorBy(COLOR_MODES[(COLOR_MODES.findIndex((m) => m.id === settings.deviation) + 1) % COLOR_MODES.length].id);
}
function renderColorBy() {
  document.getElementById('colorby').innerHTML = COLOR_MODES
    .map((m) => `<button type="button" data-color="${m.id}" aria-pressed="${settings.deviation === m.id}">${m.label}</button>`).join('');
}
/** Color the part by groups, type, fit or deviation. Turning to fit also runs the export check, for its flags. */
function setColorBy(mode) {
  const was = settings.deviation;
  settings.deviation = mode;
  renderColorBy();
  renderer.setHeat(mode === 'deviation');
  if (!mesh) return;
  recolor();
  drawLegend();
  showProbe(-1);
  refreshPanel();
  if (mode === 'fit' && was !== 'fit') runCheck();
}

/** The export check (knit/preflight.js) on the groups as they are: its flags make narrow arcs and seam gaps 'off'. */
async function runCheck() {
  if (!mesh || stroke || autoRun || !mesh.store.list().length) return;
  const current = mesh;
  say('Checking the groups…');
  await new Promise((resolve) => setTimeout(resolve, 30));   // let the message show: about 1 s per 55k triangles
  if (mesh !== current || stroke || autoRun) return;
  const doubts = exportDoubts(await knitPayload(current));
  if (mesh !== current) return;
  current.flags = doubts.groups;
  current.verdictFlags = current.flags;
  recolor();
  refreshPanel();
  const c = verdictCounts();
  say(`Checked: ${c.off} likely wrong, ${c.maybe} doubtful, ${c.on} on their surface${doubts.mesh ? ` · ${doubts.mesh}` : ''}`, c.off > 0);
}

/** The summary in the top bar: color by fit, list the groups that need a look, worst first. */
function showNeedsALook() {
  table.fits = new Set(['off', 'maybe']);
  table.by = 'none';
  table.sort = { key: 'fit', dir: 1 };
  setColorBy('fit');
}

/** How many groups have each verdict. */
function verdictCounts() {
  const counts = { off: 0, maybe: 0, on: 0, none: 0, skip: 0 };
  if (mesh) for (const v of mesh.verdicts.values()) counts[v]++;
  return counts;
}

/** Count the painted triangles in each band, and the groups holding off and maybe ones. */
function deviationCounts() {
  const { store, deviation, model } = mesh;
  const counts = { on: 0, maybe: 0, off: 0, none: 0, unpainted: 0, offGroups: new Set(), maybeGroups: new Set(), noneGroups: new Set() };
  for (let t = 0; t < model.count; t++) {
    const id = store.owner[t];
    if (id < 0) { counts.unpainted++; continue; }
    if (isSkipGroup(id)) continue;
    const band = deviationBand(deviation[t]);
    counts[band]++;
    if (band !== 'on') counts[`${band}Groups`].add(id);
  }
  return counts;
}

/** A length for the legend and the table: two figures, in exponent form when small. */
const shortLength = (v) => (!Number.isFinite(v) ? '—' : v < 1e-12 ? '≈0' : v >= 0.01 ? v.toPrecision(2) : v.toExponential(1));
const legendKey = (rgb, label, count) => `<span class="studio-legend__swatch" style="background:${css(rgb)}"></span>`
  + `<span>${label}</span><span class="studio-legend__value">${count.toLocaleString()}</span>`;

/** Fill the legend in for the color mode. */
function drawLegend() {
  if (!mesh) return;
  if (settings.deviation === 'deviation') { drawHeatLegend(); return; }
  const { store } = mesh;
  const unpainted = mesh.model.count - store.painted;
  const row = (mark, label, n) => `<div class="wb-keyrow">${mark}<span>${label}</span><span class="mono">${n.toLocaleString()}</span></div>`;
  const dot = (rgb) => `<span class="wb-swatch" style="background:${css(rgb)}"></span>`;
  const unpaintedRow = unpainted ? row(dot(UNPAINTED), 'Unpainted triangles', unpainted) : '';
  const groups = store.list();
  if (settings.deviation === 'type') {
    const c = new Map();
    for (const g of groups) c.set(g.type ?? 'freeform', (c.get(g.type ?? 'freeform') ?? 0) + 1);
    legendBody.innerHTML = `<span class="wb-ltitle">Surface type · groups</span>${SURFACE_KINDS.filter((t) => c.has(t))
      .map((t) => row(typeIcon(t, t === 'skip' ? SKIP_RGB : TYPE_RGB[t], 14), typeLabel(t), c.get(t))).join('')}${unpaintedRow}`;
  } else if (settings.deviation === 'fit') {
    const c = verdictCounts();
    legendBody.innerHTML = `<span class="wb-ltitle">Fit · will it come out right?</span>${['off', 'maybe', 'on', 'none', 'skip']
      .filter((v) => c[v] || v === 'off' || v === 'maybe' || v === 'on')
      .map((v) => row(v === 'skip' ? typeIcon('skip', SKIP_RGB, 14) : dot(VERDICTS[v].rgb), VERDICTS[v].label, c[v])).join('')}${unpaintedRow}`
      + '<span class="wb-lnote">The export check\'s bars. Hover a group for why.</span>';
  } else {
    legendBody.innerHTML = `<span class="wb-ltitle">Groups · ${groups.length}</span><span class="wb-lnote">Each face group in its own color, chosen to stand apart from the groups it borders.</span>${unpaintedRow}`;
  }
  legendDirty = false;
}

/** The heat view's legend: the ramp in model units with the export check's bars on it, and the counts. */
function drawHeatLegend() {
  const counts = deviationCounts();
  const diag = 2 * mesh.model.radius;
  const { from, stops } = HEAT;
  const top = from + stops.length - 1;
  const notes = { [from]: 'on it (float precision)', [Math.round(Math.log10(PREFLIGHT.DEV))]: 'export check: doubtful',
                  [Math.round(Math.log10(DOUBT.DEV))]: 'export check: wrong' };
  // hot at the top; stop k sits at the middle of its row
  const gradient = stops.map((hex, k) => `${hex} ${(((k + 0.5) / stops.length) * 100).toFixed(1)}%`).join(', ');
  const ticks = stops.map((_, i) => {
    const level = top - i;
    const prefix = level === top ? '≥ ' : level === from ? '≤ ' : '';
    return `<span class="studio-legend__tick" style="grid-row:${i + 1}">${prefix}${shortLength(10 ** level * diag)}</span>`
      + `<span class="studio-legend__note" style="grid-row:${i + 1}">${notes[level] ?? ''}</span>`;
  }).join('');
  const groups = (set) => `${set.size} group${set.size > 1 ? 's' : ''}`;
  legendBody.innerHTML = '<span class="wb-ltitle">Distance off the fitted surface</span>'
    + `<div class="studio-legend__scale" style="grid-template-rows:repeat(${stops.length}, 17px)">`
    + `<span class="studio-legend__bar" style="background:linear-gradient(to top, ${gradient})"></span>${ticks}</div>`
    + `<div class="studio-legend__keys">${legendKey(NO_SURFACE, 'No surface (freeform or failed fit)', counts.none)}${legendKey(UNPAINTED, 'Unpainted', counts.unpainted)}</div>`
    + `<div class="studio-legend__counts">Past the bar: ${counts.off.toLocaleString()} triangles${counts.off ? ` in ${groups(counts.offGroups)}` : ''}`
    + `<br>Doubtful: ${counts.maybe.toLocaleString()} triangles${counts.maybe ? ` in ${groups(counts.maybeGroups)}` : ''}</div>`;
  legendDirty = false;
}

/** Why group `id` has its verdict, in words. */
function verdictWhy(id) {
  const { store } = mesh;
  const group = store.groups.get(id), verdict = mesh.verdicts.get(id) ?? 'none';
  if (verdict === 'skip') return skipText(id);
  const why = mesh.flags.get(id);
  if (why?.length) return why.join('; ');
  if (verdict === 'none') {
    return group?.type ? `the ${typeLabel(group.type).toLowerCase()} fit failed${group.fit?.reason ? `: ${group.fit.reason}` : ''}`
      : 'freeform: the export fits a B-spline';
  }
  const off = shortLength((mesh.groupWorst.get(id) ?? NaN) * 2 * mesh.model.radius);
  return `${VERDICTS[verdict].label.toLowerCase()}: up to ${off} off its ${typeLabel(group.type).toLowerCase()}`;
}

/** The legend's reading for triangle t under the cursor (-1: none). */
function showProbe(t) {
  if (!mesh || t < 0) { legendProbe.textContent = 'Point at the part for a reading'; return; }
  const id = mesh.store.owner[t];
  if (id < 0) { legendProbe.textContent = 'Here: unpainted'; return; }
  const name = mesh.store.name(id);
  if (settings.deviation === 'deviation' && !isSkipGroup(id)) {
    const d = mesh.deviation[t];
    legendProbe.textContent = Number.isNaN(d) ? `Here: ${name} has no surface`
      : `Here: up to ${shortLength(d * 2 * mesh.model.radius)} off ${name} (${d.toExponential(0)} of the part)`;
    return;
  }
  legendProbe.textContent = `${name}: ${verdictWhy(id)}`;
}

/**
 * Where export sends skipped group `id` (knitPayload): the runs the painted groups surround join the group most
 * of their border touches; the rest, with nothing painted around them, are dropped.
 */
function skipFate(id) {
  const { store, model, synthetic } = mesh;
  const caps = Uint8Array.from(store.owner, (o, t) => (isSkipGroup(o) || synthetic?.[t] ? 1 : 0));
  const into = new Map();
  for (const [t, g] of capOwners(store.owner, caps, model.neighbours)) if (store.owner[t] === id) into.set(g, (into.get(g) ?? 0) + 1);
  const healed = [...into.values()].reduce((a, b) => a + b, 0);
  return { healed, dropped: store.count(id) - healed, into: [...into.keys()] };
}
function skipText(id) {
  const f = skipFate(id), names = f.into.map((g) => mesh.store.name(g)).join(', ');
  if (f.healed && f.dropped) return `skipped: ${f.healed} triangles heal over into ${names}; ${f.dropped} have nothing painted around them and are dropped`;
  if (f.healed) return `skipped: export heals it over with ${names}, its corners moved onto that surface`;
  return `skipped: nothing painted around it, so export drops its ${f.dropped} triangles`;
}

/** The edges band: group borders and creases, every triangle edge when asked (T), and skipped triangles' edges. */
function workbenchEdges(model, store) {
  const ends = bodyEdges(model, store.owner);
  const P = model.positions, extra = [];
  for (let t = 0; t < model.count; t++) {
    if (!settings.triangleEdges && !isSkipGroup(store.owner[t])) continue;
    for (let k = 0; k < 3; k++) {
      const nb = model.neighbours[t * 3 + k];
      if (nb >= 0 && nb < t && (settings.triangleEdges || isSkipGroup(store.owner[nb]))) continue;   // each edge once
      const a = t * 9 + k * 3, b = t * 9 + ((k + 1) % 3) * 3;
      extra.push(P[a], P[a + 1], P[a + 2], P[b], P[b + 1], P[b + 2]);
    }
  }
  if (!extra.length) return ends;
  const all = new Float32Array(ends.length + extra.length);
  all.set(ends);
  all.set(extra, ends.length);
  return all;
}
function toggleTriangleEdges() {
  settings.triangleEdges = !settings.triangleEdges;
  edgesDirty = true;
  requestRender();
  say(settings.triangleEdges ? 'Triangle edges on (T)' : 'Triangle edges off (T)');
}

// ── selection ──

/** The face-group table's state: sorting, filters, search, arrangement, and the shift-click anchor. */
const table = { sort: { key: 'name', dir: 1 }, types: new Set(), fits: new Set(), search: '', by: 'none', anchor: null };

/** The Select tool: the group under the cursor. */
function pickGroup(event) {
  if (!mesh) return;
  const point = devicePoint(event);
  const { centre } = renderer.pick(point.x, point.y, 0);
  const id = centre >= 0 ? mesh.store.owner[centre] : -1;
  selectGroup(id >= 0 ? id : null, event);
}
/** Pick a group: alone, added (Ctrl), or a run of the table's rows (Shift). */
function selectGroup(id, event = {}) {
  if (!mesh) return;
  if (id === null) { selection = new Set(); active = null; }
  else if (event.shiftKey && table.anchor !== null) {
    const rows = visibleRows().map((r) => r.id);
    const [a, b] = [rows.indexOf(table.anchor), rows.indexOf(id)].sort((x, y) => x - y);
    if (a >= 0) rows.slice(a, b + 1).forEach((g) => selection.add(g));
    active = id;
  } else if (event.ctrlKey || event.metaKey) {
    if (selection.has(id)) selection.delete(id); else selection.add(id);
    active = selection.has(id) ? id : [...selection].pop() ?? null;
  } else {
    const only = selection.size === 1 && selection.has(id);
    selection = new Set(only ? [] : [id]);
    active = only ? null : id;
  }
  if (id !== null) table.anchor = id;
  mesh.showFlags = false;
  edgesDirty = true;
  refreshPanel();
}
/** Frame groups `ids` in the view. */
function zoomTo(ids) {
  if (!mesh) return;
  const { model, store } = mesh;
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  const P = model.positions;
  for (let t = 0; t < model.count; t++) {
    if (!ids.has(store.owner[t])) continue;
    for (let k = 0; k < 9; k++) { const a = k % 3; lo[a] = Math.min(lo[a], P[t * 9 + k]); hi[a] = Math.max(hi[a], P[t * 9 + k]); }
  }
  if (lo[0] === Infinity) return;
  camera.center = lo.map((v, a) => (v + hi[a]) / 2);
  camera.distance = Math.max(Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) / 2, model.radius * 0.02) * 2.7;
  requestRender();
}

// ── Fix bad areas ──

/**
 * Fit-based fixes, found on demand and applied as one undo step. No retypes by fit residual: picked that way, a
 * retype went away from the true type 45 times in 152 on 300 Onshape parts, so retyping is the kernel-judged
 * Repair's job (js/knit, repairInView), which the card also offers.
 *   re-fill   a triangle off its group's surface goes to a neighbouring group whose fitted surface it lies on
 *             (the surface scorecard's triage, 2026-09-26: +0.8 to +2.0 found once, never down)
 *   merge     neighbouring groups of one type that one refitted surface keeps on its surface (PREFLIGHT.DEV, arcs
 *             of 10 degrees or more): fewer patches, no fit worse; a looser tolerance swallowed real faces
 *   skip      a small stray shell: a separate piece of the mesh under 1% of it (and 500 triangles)
 */
const fixes = { list: null, key: '', busy: false, open: false, unchecked: new Set() };
const fixKey = () => (mesh ? `${mesh.store.undoStack.length}|${mesh.store.redoStack.length}|${mesh.store.painted}|${mesh.flags.size}` : '');
const fixesFresh = () => fixes.list !== null && fixes.key === fixKey();
const AXIAL = new Set(['cylinder', 'cone', 'torus']);

/** The worst distance of triangles `list`'s corners off a surface (model units). */
function worstOff(type, params, list) {
  const P = mesh.model.positions;
  let worst = 0;
  for (const t of list) {
    for (let k = 0; k < 3; k++) {
      const o = t * 9 + k * 3;
      const d = Math.abs(distNormal(type, params, [P[o], P[o + 1], P[o + 2]])[0]);
      worst = Number.isNaN(d) ? Infinity : Math.max(worst, d);
    }
  }
  return worst;
}
/** The degrees of turn about an axial surface's axis that triangles `list` cover: 360 minus the widest gap. */
function arcDegrees(type, p, list) {
  const P = mesh.model.positions;
  const len = Math.hypot(...p.axis), ax = p.axis.map((v) => v / len);
  const o = type === 'cylinder' ? p.axisPoint : type === 'cone' ? p.apex : p.center;
  const seed = Math.abs(ax[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  const u = [ax[1] * seed[2] - ax[2] * seed[1], ax[2] * seed[0] - ax[0] * seed[2], ax[0] * seed[1] - ax[1] * seed[0]];
  const ul = Math.hypot(...u), e1 = u.map((v) => v / ul);
  const e2 = [ax[1] * e1[2] - ax[2] * e1[1], ax[2] * e1[0] - ax[0] * e1[2], ax[0] * e1[1] - ax[1] * e1[0]];
  const angles = [];
  for (const t of list) {
    for (let k = 0; k < 3; k++) {
      const i = t * 9 + k * 3, v = [P[i] - o[0], P[i + 1] - o[1], P[i + 2] - o[2]];
      angles.push(Math.atan2(v[0] * e2[0] + v[1] * e2[1] + v[2] * e2[2], v[0] * e1[0] + v[1] * e1[1] + v[2] * e1[2]));
    }
  }
  angles.sort((a, b) => a - b);
  let gap = angles[0] + 2 * Math.PI - angles[angles.length - 1];
  for (let i = 1; i < angles.length; i++) gap = Math.max(gap, angles[i] - angles[i - 1]);
  return 360 - (gap * 180) / Math.PI;
}

async function findFixes() {
  if (!mesh || fixes.busy) return;
  if (stroke || autoRun) { say('Finish or cancel (Esc) the stroke or auto-paint first', true); return; }
  const current = mesh, { store, model } = current;
  fixes.busy = true;
  fixes.open = true;
  refreshPanel();
  try {
    const { mesh: paintMesh } = await preparePaint();
    const facets = paintMesh.facets, diag = 2 * model.radius, soft = PREFLIGHT.DEV * diag, n = model.count;
    const owner = store.owner, P = model.positions, nb = model.neighbours;
    const yieldUi = () => new Promise((resolve) => setTimeout(resolve, 0));
    const lists = new Map();
    for (let t = 0; t < n; t++) if (owner[t] >= 0) (lists.get(owner[t]) ?? lists.set(owner[t], []).get(owner[t])).push(t);
    const out = [];

    // skip: small stray shells
    const comp = new Int32Array(n).fill(-1), sizes = [];
    for (let s = 0; s < n; s++) {
      if (comp[s] >= 0) continue;
      const c = sizes.length, stack = [s];
      comp[s] = c;
      let size = 0;
      while (stack.length) {
        const t = stack.pop();
        size++;
        for (let k = 0; k < 3; k++) { const m = nb[t * 3 + k]; if (m >= 0 && comp[m] < 0) { comp[m] = c; stack.push(m); } }
      }
      sizes.push(size);
    }
    let main = 0;
    sizes.forEach((size, c) => { if (size > sizes[main]) main = c; });
    const small = Math.min(500, Math.max(8, n * 0.01));
    const shells = new Map();
    for (let t = 0; t < n; t++) {
      const c = comp[t];
      if (c === main || sizes[c] > small || isSkipGroup(owner[t])) continue;
      (shells.get(c) ?? shells.set(c, []).get(c)).push(t);
    }
    for (const [c, tris] of shells) {
      if (tris.length !== sizes[c]) continue;
      out.push({ id: `shell-${tris[0]}`, kind: 'skip-shell', group: owner[tris[0]], tris,
        title: `Skip a stray shell of ${tris.length} triangle${tris.length > 1 ? 's' : ''}`,
        detail: `A separate piece, ${((100 * tris.length) / n).toFixed(2)}% of the mesh. Skipped, export drops it.` });
    }

    // re-fill: triangles off their group's surface, onto a neighbouring group whose surface they lie on
    const fitOf = (id) => { const g = store.groups.get(id); return g?.type && g.type !== 'skip' && g.fit?.params ? g.fit : null; };
    const onSurface = (fit, t) => {
      for (let k = 0; k < 3; k++) {
        const o = t * 9 + k * 3;
        if (!(Math.abs(distNormal(fit.type, fit.params, [P[o], P[o + 1], P[o + 2]])[0]) <= soft)) return false;
      }
      return true;
    };
    const moved = new Map();
    for (const [id, tris] of lists) {
      const verdict = current.verdicts.get(id);
      if ((verdict !== 'off' && verdict !== 'maybe') || !fitOf(id)) continue;
      let pending = tris.filter((t) => current.deviation[t] > PREFLIGHT.DEV), changed = true;
      while (changed && pending.length) {
        changed = false;
        pending = pending.filter((t) => {
          for (let k = 0; k < 3; k++) {
            const m = nb[t * 3 + k];
            if (m < 0) continue;
            const h = moved.get(m) ?? owner[m];
            if (h < 0 || h === id) continue;
            const fit = fitOf(h);
            if (fit && onSurface(fit, t)) { moved.set(t, h); changed = true; return false; }
          }
          return true;
        });
      }
    }
    const pairs = new Map();
    for (const [t, h] of moved) {
      const key = `${owner[t]}>${h}`;
      (pairs.get(key) ?? pairs.set(key, { from: owner[t], to: h, tris: [] }).get(key)).tris.push(t);
    }
    for (const m of pairs.values()) {
      const off = Math.max(...m.tris.map((t) => current.deviation[t])) * diag;
      out.push({ id: `refill-${m.from}-${m.to}`, kind: 'refill', group: m.from, ...m,
        title: `Re-fill ${m.tris.length} triangle${m.tris.length > 1 ? 's' : ''} of ${store.name(m.from)} into ${store.name(m.to)}`,
        detail: `They lie on ${store.name(m.to)}'s ${typeLabel(store.groups.get(m.to).type).toLowerCase()}, up to ${shortLength(off)} off ${store.name(m.from)}'s. Then refit both.` });
    }

    // merge: neighbouring groups of one type that one refitted surface keeps on its surface
    const near = new Map();
    for (let t = 0; t < n; t++) {
      const a = owner[t];
      if (a < 0) continue;
      for (let k = 0; k < 3; k++) {
        const m = nb[t * 3 + k];
        const b = m >= 0 ? owner[m] : -1;
        if (b >= 0 && b !== a) (near.get(a) ?? near.set(a, new Set()).get(a)).add(b);
      }
    }
    // groups on their surface (a narrow arc too: the merge must cover 10 degrees); re-fill sources never are
    const eligible = (id) => !!fitOf(id) && (current.groupWorst.get(id) ?? Infinity) <= PREFLIGHT.DEV;
    const used = new Set();
    let tried = 0;
    for (const id of [...lists.keys()].sort((a, b) => a - b)) {
      if (used.has(id) || !eligible(id)) continue;
      const type = store.groups.get(id).type, cluster = [id];
      let union = lists.get(id), worst = 0;
      for (let i = 0; i < cluster.length; i++) {
        for (const other of near.get(cluster[i]) ?? []) {
          if (cluster.includes(other) || used.has(other) || !eligible(other) || store.groups.get(other).type !== type) continue;
          const merged = union.concat(lists.get(other));
          const fit = fitRegionAs(type, fitTriangles(new Set(merged), current.synthetic), facets);
          if (++tried % 3 === 0) { say(`Finding fixes: trying merges (${tried})…`); await yieldUi(); if (mesh !== current) return; }
          if (!fit.params) continue;
          const w = worstOff(type, fit.params, merged);
          if (w > soft || (AXIAL.has(type) && arcDegrees(type, fit.params, merged) < DOUBT.ARC)) continue;
          cluster.push(other);
          union = merged;
          worst = w;
        }
      }
      if (cluster.length < 2) continue;
      cluster.forEach((g) => used.add(g));
      const into = cluster.reduce((best, g) => (lists.get(g).length > lists.get(best).length ? g : best));
      out.push({ id: `merge-${[...cluster].sort((a, b) => a - b).join('-')}`, kind: 'merge', group: into, groups: cluster, into,
        title: `Merge ${cluster.filter((g) => g !== into).map((g) => store.name(g)).join(', ')} into ${store.name(into)}`,
        detail: `One ${typeLabel(type).toLowerCase()} fits all ${cluster.length} to ${shortLength(worst)}: ${cluster.length} surface patches become 1, no fit worse.` });
    }

    if (mesh !== current) return;
    fixes.list = out.sort((a, b) => a.group - b.group);
    fixes.key = fixKey();
    fixes.unchecked.clear();
    say(out.length ? `Found ${out.length} fix${out.length > 1 ? 'es' : ''}` : 'No fit-based fixes found: try Repair with the kernel for groups that still fail');
  } catch (error) {
    console.error(error);
    say(`Finding fixes failed: ${error.message}`, true);
  } finally {
    fixes.busy = false;
    refreshPanel();
  }
}

/** Apply fixes `list` as one undo step; merges last, so a re-fill never lands in a group a merge has emptied. */
function applyFixes(list) {
  if (!mesh || !list.length) return;
  const order = [...list].sort((a, b) => (a.kind === 'merge') - (b.kind === 'merge'));
  editGroups(`Fix: ${list.length} change${list.length > 1 ? 's' : ''}`, (store) => {
    const touched = new Set();
    for (const p of order) {
      if (p.kind === 'skip-shell') {
        const g = store.create('skip');
        for (const t of p.tris) { if (store.owner[t] >= 0) touched.add(store.owner[t]); store.assign(t, g.id); }
        touched.add(g.id);
      } else if (p.kind === 'refill') {
        for (const t of p.tris) if (store.owner[t] === p.from) store.assign(t, p.to);
        touched.add(p.from);
        touched.add(p.to);
      } else if (p.kind === 'merge') {
        for (const g of p.groups) if (g !== p.into) for (const t of store.triangles(g)) store.assign(t, p.into);
        touched.add(p.into);
      }
    }
    return [...touched];
  });
  fixes.list = null;
  fixes.unchecked.clear();
  const kinds = { refill: 're-fills', merge: 'merges', 'skip-shell': 'stray shells skipped' };
  const tally = Object.entries(kinds).map(([k, l]) => [list.filter((p) => p.kind === k).length, l]).filter(([n]) => n);
  say(`Fixed, as one undo step: ${tally.map(([n, l]) => `${n} ${l}`).join(', ')}`);
}

renderColorBy();
