/** The groups as table rows: id, name, type, triangles, area, verdict and worst distance (model units). */
function groupRows() {
  const { store, model } = mesh;
  if (!mesh.triArea) {
    const P = model.positions, area = new Float32Array(model.count);
    for (let t = 0; t < model.count; t++) {
      const o = t * 9, ux = P[o + 3] - P[o], uy = P[o + 4] - P[o + 1], uz = P[o + 5] - P[o + 2];
      const vx = P[o + 6] - P[o], vy = P[o + 7] - P[o + 1], vz = P[o + 8] - P[o + 2];
      area[t] = Math.hypot(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx) / 2;
    }
    mesh.triArea = area;
  }
  const rows = new Map();
  for (const g of store.list()) rows.set(g.id, { id: g.id, group: g, name: store.name(g.id), type: g.type, count: 0, area: 0 });
  for (let t = 0; t < model.count; t++) { const r = rows.get(store.owner[t]); if (r) { r.count++; r.area += mesh.triArea[t]; } }
  const diag = 2 * model.radius;
  for (const r of rows.values()) { r.verdict = mesh.verdicts.get(r.id) ?? 'none'; r.worst = (mesh.groupWorst.get(r.id) ?? NaN) * diag; }
  return [...rows.values()];
}

const RANK = { off: 0, maybe: 1, none: 2, on: 3, skip: 4 };
function sortValue(r, key) {
  if (key === 'name') return r.id;
  if (key === 'type') return SURFACE_KINDS.indexOf(r.type ?? 'freeform');
  if (key === 'count') return r.count;
  if (key === 'area') return r.area;
  return RANK[r.verdict] * 1e12 - (Number.isFinite(r.worst) ? r.worst : 0);   // worst first
}
/** The table's rows as filtered, searched and sorted. */
function visibleRows() {
  if (!mesh) return [];
  const q = table.search.trim().toLowerCase();
  const rows = groupRows().filter((r) => (!table.types.size || table.types.has(r.type ?? 'freeform'))
    && (!table.fits.size || table.fits.has(r.verdict)) && (!q || r.name.toLowerCase().includes(q)));
  const { key, dir } = table.sort;
  rows.sort((a, b) => { const x = sortValue(a, key), y = sortValue(b, key); return ((x < y ? -1 : x > y ? 1 : 0) || a.id - b.id) * dir; });
  return rows;
}

const ROW_LIMIT = 400;
const esc = (text) => String(text).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const groupRgb = (r) => (r.type === 'skip' ? SKIP_RGB : settings.deviation === 'type' ? TYPE_RGB[r.type ?? 'freeform']
  : settings.deviation === 'fit' || settings.deviation === 'deviation' ? VERDICTS[r.verdict].rgb : r.group.colour);

/** The top bar, the status line and the face-group table with its fixes and inspector. */
function refreshPanel() {
  const hasMesh = !!mesh;
  const store = mesh?.store;
  toolbar.setDisabled('undo', store?.undoStack.length ? null : 'Nothing to undo');
  toolbar.setDisabled('redo', store?.redoStack.length ? null : 'Nothing to redo');
  toolbar.setDisabled('export', store?.list().length ? null : 'Paint at least one face group first');
  const slider = document.getElementById('brush-size');
  if (slider && Number(slider.value) !== settings.brushPx) { slider.value = settings.brushPx; document.getElementById('brush-px').textContent = `${settings.brushPx}px`; }
  const summary = document.getElementById('summary');
  if (!hasMesh) {
    shell.setStatus('');
    summary.hidden = true;
    panelEl.innerHTML = '<div class="wb-phead"><div class="wb-ptitle"><h2>Face groups</h2></div><p class="wb-lnote">Open or drop an STL, OBJ or 3MF file.</p></div>';
    return;
  }
  const pct = (100 * store.painted) / mesh.model.count;
  shell.setStatus(`${store.painted.toLocaleString()} of ${mesh.model.count.toLocaleString()} triangles painted (${pct.toFixed(pct < 10 ? 1 : 0)}%)`);
  const c = verdictCounts();
  summary.hidden = !store.list().length;
  summary.innerHTML = ['off', 'maybe', 'on', 'skip'].filter((v) => v !== 'skip' || c.skip)
    .map((v) => `<span><span class="wb-dot" style="background:${css(VERDICTS[v].rgb)}"></span>${c[v]} ${VERDICTS[v].label.toLowerCase()}</span>`).join('');

  const all = groupRows();
  const rows = visibleRows();
  const typeCounts = new Map();
  for (const r of all) typeCounts.set(r.type ?? 'freeform', (typeCounts.get(r.type ?? 'freeform') ?? 0) + 1);
  const picked = [...selection].filter((id) => store.count(id));
  const fitCell = (r) => (r.verdict === 'skip' ? '<span class="wb-skipped">skipped</span>'
    : r.verdict === 'none' ? `<span class="wb-dot" style="background:${css(NO_SURFACE)}"></span>none`
      : `<span class="wb-dot" style="background:${css(VERDICTS[r.verdict].rgb)}"></span>${shortLength(r.worst)}`);
  const row = (r) => `<tr class="wb-g${r.verdict === 'skip' ? ' wb-g--skip' : ''}" data-group="${r.id}" aria-selected="${selection.has(r.id)}" title="${esc(`${r.name}: ${verdictWhy(r.id)}`)}">
      <td><span class="wb-nm">${typeIcon(r.type, groupRgb(r))}${esc(r.name)}${mesh.flags.has(r.id) ? ' ⚠' : ''}</span></td>
      <td>${r.type ?? 'freeform'}</td><td class="wb-num">${r.count.toLocaleString()}</td><td class="wb-num wb-col-area">${shortLength(r.area)}</td><td class="wb-fit">${fitCell(r)}</td></tr>`;
  let body = '', shown = 0;
  const take = (list) => { const part = list.slice(0, Math.max(0, ROW_LIMIT - shown)); shown += part.length; return part.map(row).join(''); };
  if (table.by === 'none') body = take(rows);
  else {
    const keyOf = table.by === 'type' ? (r) => r.type ?? 'freeform' : (r) => r.verdict;
    for (const k of table.by === 'type' ? SURFACE_KINDS : ['off', 'maybe', 'none', 'on', 'skip']) {
      const part = rows.filter((r) => keyOf(r) === k);
      if (part.length) body += `<tr class="wb-sec"><td colspan="5">${table.by === 'type' ? typeLabel(k) : VERDICTS[k].label} · ${part.length}</td></tr>${take(part)}`;
    }
  }
  if (rows.length > shown) body += `<tr class="wb-unp"><td colspan="5">${(rows.length - shown).toLocaleString()} more: filter or search to see them</td></tr>`;
  const unpainted = mesh.model.count - store.painted;
  if (unpainted) body += `<tr class="wb-unp"><td>Unpainted</td><td></td><td class="wb-num">${unpainted.toLocaleString()}</td><td class="wb-col-area"></td><td></td></tr>`;
  const arrow = (k) => (table.sort.key === k ? (table.sort.dir > 0 ? ' ▲' : ' ▼') : '');
  const fitOptions = (current) => SURFACE_KINDS.map((t) => `<option value="${t}" ${current === t ? 'selected' : ''}>${typeLabel(t)}</option>`).join('');

  // the fixes card
  let card;
  const fresh = fixesFresh();
  if (fixes.busy) card = '<div class="wb-fixhead"><span class="wb-t">Fix bad areas</span><span class="wb-c">looking…</span></div>';
  else if (fresh && fixes.list.length) {
    const checked = fixes.list.filter((p) => !fixes.unchecked.has(p.id));
    card = `<button type="button" class="wb-fixhead" data-act="fix-toggle" aria-expanded="${fixes.open}"><span class="wb-t">Fix bad areas</span><span class="wb-c">${fixes.list.length} ready ${fixes.open ? '▾' : '▸'}</span></button>`
      + (fixes.open ? `<div class="wb-fixlist">${fixes.list.map((p) => `<label class="wb-fix"><input type="checkbox" id="fix-${p.id}" data-fix="${p.id}" ${fixes.unchecked.has(p.id) ? '' : 'checked'}><span>${esc(p.title)}<br><span class="wb-d">${esc(p.detail)}</span></span></label>`).join('')}
        <div class="wb-irow"><button type="button" class="wb-btn wb-primary wb-small" data-act="fix-apply" ${checked.length ? '' : 'disabled'}>Apply ${checked.length} as one step</button><button type="button" class="wb-btn wb-small" data-act="repair" title="Build the solid and let the kernel judge each change (retypes and merges), each its own undo step; Esc stops">Repair with the kernel</button></div></div>` : '');
  } else {
    card = `<div class="wb-fixhead"><span class="wb-t">Fix bad areas</span><span class="wb-c">${fresh ? 'nothing found' : ''}</span></div>`
      + `<div class="wb-fixlist"><div class="wb-irow"><button type="button" class="wb-btn wb-small" data-act="fix-find" title="Re-fill triangles off their surface into the neighbour they lie on, merge neighbours one surface fits, skip stray shells">Find fixes</button><button type="button" class="wb-btn wb-small" data-act="repair" title="Build the solid and let the kernel judge each change (retypes and merges), each its own undo step; Esc stops">Repair with the kernel</button></div></div>`;
  }

  // the inspector, or the bar for several groups
  let below = '';
  if (picked.length === 1) {
    const r = all.find((x) => x.id === picked[0]);
    const fit = r.group.fit;
    const p = fit?.params;
    const vec = (v) => v.map((x) => x.toFixed(3)).join(', ');
    const params = !p ? [] : [
      ...(p.normal ? [['Normal', vec(p.normal)]] : []),
      ...(p.axis ? [['Axis', vec(p.axis)]] : p.center ? [['Center', vec(p.center)]] : []),
      ...(p.radius != null ? [['Radius', shortLength(p.radius)]] : []),
      ...(p.halfAngle != null ? [['Half-angle', `${((p.halfAngle * 180) / Math.PI).toFixed(2)}°`]] : []),
      ...(p.majorRadius != null ? [['Radii', `${shortLength(p.majorRadius)}, ${shortLength(p.minorRadius)}`]] : []),
    ];
    below = `<div class="wb-inspector">
      <h3>${typeIcon(r.type, groupRgb(r), 18)}${esc(r.name)}</h3>
      <dl class="wb-kv"><dt>Fit as</dt><dd><select id="fit-as" aria-label="Fit ${esc(r.name)} as">${fitOptions(r.type ?? 'freeform')}</select></dd>
        ${params.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('')}
        <dt>Triangles</dt><dd>${r.count.toLocaleString()} · area ${shortLength(r.area)}</dd>
        ${Number.isFinite(r.worst) ? `<dt>Off the fit</dt><dd>up to ${shortLength(r.worst)}</dd>` : ''}</dl>
      <div class="wb-why"><span class="wb-dot" style="background:${css(VERDICTS[r.verdict].rgb)}"></span> ${esc(verdictWhy(r.id))}</div>
      <div class="wb-irow"><button type="button" class="wb-btn wb-small" data-act="zoom">Zoom to</button>${r.type === 'skip' ? '' : '<button type="button" class="wb-btn wb-small" data-act="refit">Refit</button>'}<button type="button" class="wb-btn wb-small" data-act="delete">Delete</button></div></div>`;
  } else if (picked.length > 1) {
    below = `<div class="wb-selbar"><span class="wb-n">${picked.length} selected</span>
      <button type="button" class="wb-btn wb-small" data-act="merge" title="Merge them into the largest">Merge</button>
      <select id="sel-fit-as" aria-label="Fit the selected groups as"><option value="">Fit as…</option>${fitOptions(null)}</select>
      <button type="button" class="wb-btn wb-small" data-act="zoom">Zoom to</button>
      <button type="button" class="wb-btn wb-small" data-act="delete">Delete</button><button type="button" class="wb-btn wb-small" data-act="clear">Clear</button></div>`;
  }

  const caret = document.activeElement?.id === 'search' ? document.activeElement.selectionStart : null;
  const scroll = panelEl.querySelector('.wb-tablewrap')?.scrollTop ?? 0;
  panelEl.innerHTML = `
    <div class="wb-phead">
      <div class="wb-ptitle"><h2>Face groups</h2><span class="wb-count">${all.length.toLocaleString()}</span><span class="wb-spacer"></span>
        <span class="wb-seg" role="group" aria-label="Arrange">${[['none', 'List'], ['type', 'By type'], ['fit', 'By fit']].map(([k, l]) => `<button type="button" data-by="${k}" aria-pressed="${table.by === k}">${l}</button>`).join('')}</span></div>
      <input class="wb-search" id="search" type="search" placeholder="Find a group by name" value="${esc(table.search)}" aria-label="Find a group by name">
      <div class="wb-filters">
        <div class="wb-frow"><span class="wb-glabel">Type</span>${SURFACE_KINDS.filter((t) => typeCounts.has(t)).map((t) => `<button type="button" class="wb-fchip" data-ftype="${t}" aria-pressed="${table.types.has(t)}">${typeIcon(t, t === 'skip' ? SKIP_RGB : TYPE_RGB[t], 14)}${typeLabel(t)} <b>${typeCounts.get(t)}</b></button>`).join('')}</div>
        <div class="wb-frow"><span class="wb-glabel">Fit</span>${['off', 'maybe', 'on', 'none', 'skip'].filter((v) => c[v]).map((v) => `<button type="button" class="wb-fchip" data-ffit="${v}" aria-pressed="${table.fits.has(v)}"><span class="wb-dot" style="background:${css(VERDICTS[v].rgb)}"></span>${VERDICTS[v].label} <b>${c[v]}</b></button>`).join('')}</div>
      </div>
    </div>
    <div class="wb-fixcard">${card}</div>
    ${picked.length > 1 ? below : ''}
    <div class="wb-tablewrap"><table>
      <thead><tr>${[['name', 'Group'], ['type', 'Type'], ['count', 'Tris', 'wb-num'], ['area', 'Area', 'wb-num wb-col-area'], ['fit', 'Fit · worst']].map(([k, l, cls]) => `<th class="${cls ?? ''}"><button type="button" data-sort="${k}">${l}${arrow(k)}</button></th>`).join('')}</tr></thead>
      <tbody>${body || '<tr><td colspan="5" class="wb-lnote">No groups yet: paint, or Auto-paint (P).</td></tr>'}</tbody></table></div>
    ${picked.length === 1 ? below : ''}`;
  const wrap = panelEl.querySelector('.wb-tablewrap');
  if (wrap) wrap.scrollTop = scroll;
  if (caret !== null) { const s = document.getElementById('search'); s.focus(); s.setSelectionRange(caret, caret); }
}

function setType(type) {
  settings.type = type;
  for (const t of FILL_TYPES) toolbar.setPressed(`type-${t}`, t === type);
  say(type === 'auto' ? 'Fill: the model picks the surface'
    : type === 'freeform' ? 'Freeform fill: grows to creases; the export fits a B-spline'
      : type === 'skip' ? 'Skip: paint what the import mangled; it is left out of fitting and the checks'
        : `Fill as ${typeLabel(type).toLowerCase()}`);
}

// the table's controls
panelEl.addEventListener('click', (event) => {
  if (!mesh) return;
  const el = event.target.closest('[data-act],[data-sort],[data-by],[data-ftype],[data-ffit],tr.wb-g');
  if (!el) return;
  const d = el.dataset;
  const picked = [...selection].filter((id) => mesh.store.count(id));
  if (d.sort) { table.sort = { key: d.sort, dir: table.sort.key === d.sort ? -table.sort.dir : 1 }; refreshPanel(); }
  else if (d.by) { table.by = d.by; refreshPanel(); }
  else if (d.ftype) { if (table.types.has(d.ftype)) table.types.delete(d.ftype); else table.types.add(d.ftype); refreshPanel(); }
  else if (d.ffit) { if (table.fits.has(d.ffit)) table.fits.delete(d.ffit); else table.fits.add(d.ffit); refreshPanel(); }
  else if (el.matches('tr.wb-g')) selectGroup(Number(d.group), event);
  else if (d.act === 'fix-find') findFixes();
  else if (d.act === 'fix-toggle') { fixes.open = !fixes.open; refreshPanel(); }
  else if (d.act === 'fix-apply') applyFixes(fixes.list.filter((p) => !fixes.unchecked.has(p.id)));
  else if (d.act === 'repair') repairInView();
  else if (d.act === 'zoom') zoomTo(new Set(picked));
  else if (d.act === 'refit') editGroups(`Refit ${mesh.store.name(picked[0])}`, () => picked);
  else if (d.act === 'delete') {
    editGroups(`Delete ${picked.length > 1 ? `${picked.length} groups` : mesh.store.name(picked[0])}`, (store) => {
      for (const id of picked) for (const t of store.triangles(id)) store.assign(t, -1);
    });
    selectGroup(null);
  } else if (d.act === 'merge') {
    const into = picked.reduce((best, id) => (mesh.store.count(id) > mesh.store.count(best) ? id : best));
    editGroups(`Merge ${picked.length} groups into ${mesh.store.name(into)}`, (store) => {
      for (const id of picked) if (id !== into) for (const t of store.triangles(id)) store.assign(t, into);
      return [into];
    });
    selection = new Set([into]);
    active = into;
    refreshPanel();
  } else if (d.act === 'clear') selectGroup(null);
});
panelEl.addEventListener('dblclick', (event) => {
  const tr = event.target.closest('tr.wb-g');
  if (!tr || !mesh) return;
  const id = Number(tr.dataset.group);
  selection = new Set([id]);
  active = id;
  edgesDirty = true;
  refreshPanel();
  zoomTo(new Set([id]));
});
panelEl.addEventListener('contextmenu', (event) => {
  const tr = event.target.closest('tr.wb-g');
  if (!tr || !mesh) return;
  event.preventDefault();
  groupMenu(Number(tr.dataset.group), event);
});
panelEl.addEventListener('change', (event) => {
  const t = event.target;
  if (!mesh) return;
  if (t.matches('[data-fix]')) { if (t.checked) fixes.unchecked.delete(t.dataset.fix); else fixes.unchecked.add(t.dataset.fix); refreshPanel(); }
  else if (t.id === 'fit-as' || (t.id === 'sel-fit-as' && t.value)) {
    const picked = [...selection].filter((id) => mesh.store.count(id));
    const type = t.value === 'freeform' ? null : t.value;
    editGroups(`Fit ${picked.length > 1 ? `${picked.length} groups` : mesh.store.name(picked[0])} as ${typeLabel(type).toLowerCase()}`, (store) => {
      for (const id of picked) store.groups.get(id).type = type;
      return picked;
    });
  }
});
panelEl.addEventListener('input', (event) => {
  if (event.target.id !== 'search') return;
  table.search = event.target.value;
  refreshPanel();
});

