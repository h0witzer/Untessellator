// The tool row: the tools, the surface to paint as (each its icon), the brush, auto-paint, the fill options and
// the report. `toolbar` gives the studio's code its calls (setPressed, setDisabled, get) by each control's data-tb.
const TOOL_ICONS = { select: 'select', fill: 'bucket', raw: 'brush', erase: 'eraseFill' };
const toolRow = document.getElementById('tools');
const typeTip = (type, i) => (type === 'auto' ? `The model picks the surface from the seed (${i + 1})`
  : type === 'freeform' ? `Freeform fill: grows to creases; exported as a B-spline (${i + 1})`
    : type === 'skip' ? `Skip what the import mangled: left out of fitting, auto-paint and the checks; export heals it over with the group around it, or drops a stray shell (${i + 1})`
      : `Fill as ${typeLabel(type).toLowerCase()}; painting into a group retypes it (${i + 1})`);
toolRow.innerHTML = `
  <div class="wb-tgroup">${[
    ['select', 'Select', 'Click a face to select its group (V)'],
    ['fill', 'Bucket', 'Fill: each triangle crossed fills its surface into the group (G)'],
    ['raw', 'Brush', 'Facet brush: just the triangles under the brush (B, or hold Shift)'],
    ['erase', 'Erase', 'Back to unpainted (E, or hold Alt)'],
  ].map(([id, label, tip]) => `<button type="button" class="wb-tool" data-tb="${id}" aria-pressed="${settings.tool === id}" title="${tip}">${ui.iconMarkup(TOOL_ICONS[id], { className: 'wb-ticon' })}${label}</button>`).join('')}</div>
  <span class="wb-vsep"></span>
  <div class="wb-tgroup"><span class="wb-glabel">Paint as</span>${FILL_TYPES.map((type, i) => `${type === 'skip' ? '<span class="wb-vsep wb-vsep--short"></span>' : ''}<button type="button" class="wb-ftype${type === 'skip' ? ' wb-ftype--skip' : ''}" data-tb="type-${type}" aria-pressed="${settings.type === type}" title="${typeTip(type, i)}">${typeIcon(type, type === 'auto' ? null : TYPE_RGB[type])}${type === 'auto' ? 'Auto' : typeLabel(type)}</button>`).join('')}</div>
  <span class="wb-vsep"></span>
  <label class="wb-size" for="brush-size">Brush <input type="range" id="brush-size" min="3" max="64" value="${settings.brushPx}"> <span class="mono" id="brush-px">${settings.brushPx}px</span></label>
  <span class="wb-vsep"></span>
  <button type="button" class="wb-tool" data-tb="autopaint" title="Auto-paint every unpainted face; your groups are left alone (P). Esc stops">${ui.iconMarkup('wand', { className: 'wb-ticon' })}Auto-paint</button>
  <button type="button" class="wb-tool" data-tb="fill-options" title="Fill method, model and options" aria-label="Fill options">${ui.iconMarkup('surface', { className: 'wb-ticon' })}</button>
  <button type="button" class="wb-tool" data-tb="report" title="Report how this conversion went: opens a GitHub issue with this run's numbers; nothing is sent until you submit it" aria-label="Report this conversion">${ui.iconMarkup('flag', { className: 'wb-ticon' })}</button>`;
toolRow.addEventListener('click', (event) => {
  const button = event.target.closest('[data-tb]');
  if (!button) return;
  const id = button.dataset.tb;
  if (['select', 'fill', 'raw', 'erase'].includes(id)) setTool(id);
  else if (id.startsWith('type-')) setType(id.slice(5));
  else if (id === 'autopaint') runAutoPaint();
  else if (id === 'fill-options') openFillMenu();
  else if (id === 'report') reportConversion();
});
document.getElementById('brush-size').addEventListener('input', (event) => {
  settings.brushPx = Number(event.target.value);
  document.getElementById('brush-px').textContent = `${settings.brushPx}px`;
  drawCursor();
});
document.getElementById('open').addEventListener('click', () => fileInput.click());
document.getElementById('undo').addEventListener('click', () => undo());
document.getElementById('redo').addEventListener('click', () => redo());
document.getElementById('export').addEventListener('click', () => exportStep());
document.getElementById('summary').addEventListener('click', () => showNeedsALook());
document.getElementById('colorby').addEventListener('click', (event) => {
  const button = event.target.closest('[data-color]');
  if (button) setColorBy(button.dataset.color);
});

const toolbar = {
  get: (id) => document.querySelector(`[data-tb="${id}"]`),
  setPressed(id, on) { this.get(id)?.setAttribute('aria-pressed', on ? 'true' : 'false'); },
  setDisabled(id, reason) {
    const button = this.get(id);
    if (!button) return;
    button.disabled = !!reason;
    if (reason) button.dataset.why = reason; else delete button.dataset.why;
  },
};
const panelEl = document.getElementById('panel');

