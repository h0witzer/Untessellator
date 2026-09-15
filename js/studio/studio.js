/**
 * studio.js — paint face groups onto a mesh, then export them as a STEP solid.
 *
 * Built on the Form Studio kit (vendor/form-studio-ui), which carries Onshape's frame,
 * camera, lighting and gestures as measured from the product. The mouse follows it:
 * RIGHT-drag orbits, the wheel zooms, a right click opens the menu for what is under the
 * cursor, Space clears the active group. Where a Part Studio's LEFT button selects, here it
 * paints:
 *
 *   drag            FILL: every unpainted triangle the cursor crosses fills its surface
 *                   (paintTool.paintAt: model proposal, then the analytic fitter) into the
 *                   stroke's group. A stroke that starts on unpainted mesh makes a new group;
 *                   one that starts on a group extends it. A fill never takes triangles from
 *                   another group: erase a spill first, or merge the groups. And a stroke
 *                   stays on its surface: a crossed triangle that can't be reached from
 *                   where the stroke landed without crossing a crease (CREASE_BARRIER for
 *                   the stroke's surface type) is skipped, so dragging over a bend paints
 *                   nothing beyond it.
 *   Shift-drag      FACETS: the triangles under the brush circle go into the group as they
 *                   are, for the pieces a fill leaves behind
 *   Alt-drag        ERASE: the triangles under the brush go back to unpainted
 *
 * A stroke is one undo step. Its fills run one at a time after the pointer moves on, so a
 * quick stroke queues several; Esc cancels the whole stroke.
 *
 * Nothing painted here is geometry. Export fits each group with the Gauss-map fitters and
 * builds the solid with the same B-rep builder as the main app.
 */

import * as ui from '../../vendor/form-studio-ui/index.js';
import { loadModelFile } from '../stlLoader.js';
import { fitRegionAs } from '../gaussMap.js';
import { CREASE_BARRIER, creaseFor } from '../regionGrow.js';
import { buildMeshModel, bodyEdges, outline } from './meshModel.js';
import { createRenderer } from './renderer.js';
import { GroupStore, typeLabel } from './groups.js';

const UNPAINTED = [0.74, 0.74, 0.74];
const FOV = ui.LIGHTING.fieldOfViewDegrees;

const METHODS = [
  { id: 'hyp', label: 'Model surface → fitter' },
  { id: 'model', label: 'Model region → fitter' },
  { id: 'veto', label: 'Model region + veto' },
  { id: 'hyp+veto', label: 'Model surface + veto' },
  { id: 'plain', label: 'No model' },
];
const TYPES = ['auto', 'plane', 'cylinder', 'cone', 'sphere', 'torus'];
// v6feat first: best of 13 models on the 80-part fill benchmark (studio type 52% clean over OCCT,
// NURBS and gmsh against 49% for the next best; 69/66/68% with simpler faces painted first,
// against s1's 60/51/58%). `features: 'v6'` makes paintTool build the extended inputs.
const MODELS = [
  { url: 'models/seedfill-v6feat.onnx', label: 'v6 (crease + seed fits)', features: 'v6' },
  { url: 'models/seedfill-s1nogmsh.onnx', label: 's1 (CAD exports)' },
  { url: 'models/seedfill-s3all.onnx', label: 's3 (all meshers)' },
];
const TYPE_ICONS = { plane: 'face', cylinder: 'cylinder', cone: 'cone', sphere: 'sphere', torus: 'torus' };

ui.registerIcons({
  folder: '<path d="M2 3.6h6.2l2 2H18v10.8H2zm1.6 4v7.2h12.8V7.6z" fill-rule="evenodd" />',
  bucket: '<path d="M8.4 1.8 16 9.4l-6.2 6.2-7.6-7.6 3-3-1.1-1.1 1.2-1.2 1.1 1.1zM4.5 8l5.3 5.3L13.7 9.4z" fill-rule="evenodd" /><path d="M16.6 12.2s1.9 2.3 1.9 3.6a1.9 1.9 0 0 1-3.8 0c0-1.3 1.9-3.6 1.9-3.6z" />',
  brush: '<path d="M17.8 2.2c.7.7-5.2 8.4-6.6 9.8L8 8.8c1.4-1.4 9.1-7.3 9.8-6.6zM6.9 10.1l3 3c-.3 2.8-2.5 5-7.9 4.7 2-1.5 1-6 4.9-7.7z" />',
  cone: '<path d="M10 1.6 17 15.2a7 2.8 0 0 1-14 0zm0 4.4-4.6 9.1a4.6 1.4 0 0 0 9.2 0z" fill-rule="evenodd" />',
  // A four-point spark: the model picks the surface.
  auto: '<path d="M8 1.5c.5 3.6 2.4 5.5 6 6-3.6.5-5.5 2.4-6 6-.5-3.6-2.4-5.5-6-6 3.6-.5 5.5-2.4 6-6zM15 11c.3 2 1.3 3 3.3 3.3-2 .3-3 1.3-3.3 3.3-.3-2-1.3-3-3.3-3.3 2-.3 3-1.3 3.3-3.3z" />',
});

const settings = {
  tool: 'fill',
  method: 'hyp',
  type: 'auto',
  model: MODELS[0].url,
  brushPx: 12,
  perspective: ui.LIGHTING.perspectiveByDefault,
  edges: true,
  smooth: false,
  occlusion: true,
};

// ── frame ─────────────────────────────────────────────────────────────────────

const shell = ui.createShell(document.getElementById('app'), { theme: 'dark' });
const canvas = document.createElement('canvas');
canvas.className = 'studio-canvas';
shell.stage.prepend(canvas);
const overlay = document.createElement('canvas');
overlay.className = 'studio-overlay';
canvas.after(overlay);
const empty = document.createElement('div');
empty.className = 'studio-empty';
empty.textContent = 'Open or drop an STL, OBJ or 3MF file';
shell.stage.append(empty);
const fileInput = document.createElement('input');
fileInput.type = 'file';
fileInput.accept = '.stl,.obj,.3mf';
fileInput.hidden = true;
document.body.append(fileInput);

const renderer = createRenderer(canvas);
const camera = new ui.OrbitCamera({ ...ui.STANDARD_VIEWS.Trimetric, distance: 5 });
const corner = ui.mountViewCube(shell.stage, {
  camera,
  onZoomFit: () => zoomFit(),
  perspective: { get: () => settings.perspective, set: (on) => { settings.perspective = on; requestRender(); } },
  edges: { get: () => settings.edges, set: (on) => { settings.edges = on; requestRender(); } },
  // A getter, so the menu shows the current state each time it opens.
  get extraMenu() {
    return [
      { id: 'smooth', label: 'Smooth shading', checked: settings.smooth, onSelect: () => { settings.smooth = !settings.smooth; renderer.setShading(settings.smooth); requestRender(); } },
      { id: 'occlusion', label: 'Ambient occlusion', checked: settings.occlusion, onSelect: () => { settings.occlusion = !settings.occlusion; requestRender(); } },
    ];
  },
});

const toolbar = ui.createToolbar(shell.toolbar, [
  { kind: 'tool', id: 'open', icon: 'folder', title: 'Open mesh (Ctrl+O)', onClick: () => fileInput.click() },
  { kind: 'separator' },
  { kind: 'toggle', id: 'fill', icon: 'bucket', title: 'Fill brush: each triangle crossed fills its surface into the group', pressed: true, onClick: () => setTool('fill') },
  { kind: 'toggle', id: 'raw', icon: 'brush', title: 'Facet brush (hold Shift): paints just the triangles under the brush', onClick: () => setTool('raw') },
  { kind: 'toggle', id: 'erase', icon: 'eraseFill', title: 'Eraser (hold Alt): back to unpainted', onClick: () => setTool('erase') },
  { kind: 'separator' },
  { kind: 'tool', id: 'undo', icon: 'undo', title: 'Undo (Ctrl+Z)', onClick: () => undo() },
  { kind: 'tool', id: 'redo', icon: 'redo', title: 'Redo (Ctrl+Y)', onClick: () => redo() },
  { kind: 'separator' },
  // The surface a fill paints with: one button each, since it changes stroke to stroke.
  ...TYPES.map((type, i) => ({
    kind: 'toggle',
    id: `type-${type}`,
    icon: type === 'auto' ? 'auto' : TYPE_ICONS[type],
    title: type === 'auto'
      ? `Model picks the surface from the seed (${i + 1})`
      : `Fill as ${typeLabel(type).toLowerCase()} (${i + 1})`,
    pressed: settings.type === type,
    onClick: () => setType(type),
  })),
  { kind: 'separator' },
  { kind: 'split', id: 'fill-options', icon: 'surface', title: 'Fill method and model', onClick: () => openFillMenu(), menu: () => fillMenu() },
  { kind: 'spacer' },
  { kind: 'labelled', id: 'export', icon: 'build', label: 'Export STEP', accent: true, onClick: () => exportStep() },
]);

const panel = ui.createPanel(shell.panel, { filter: null });
const section = panel.addSection({
  title: 'Face groups',
  buttons: [{ id: 'delete', icon: 'deleteFace', title: 'Delete the active group (Del)', onClick: () => active !== null && deleteGroup(active) }],
});

// ── state ─────────────────────────────────────────────────────────────────────

/** { model, geometry, name, store, colours } for the open mesh. */
let mesh = null;
let active = null;
let hoverKey = null;
let stroke = null;
let paintPromise = null;
let prepared = null;
let needsRender = true;
let edgesDirty = false;
let cursor = null;
const modifiers = { shift: false, alt: false };

window.studio = {
  get mesh() { return mesh; },
  get active() { return active; },
  get stroke() { return stroke; },
  get prepared() { return prepared; },
  settings,
  camera,
};

const requestRender = () => { needsRender = true; };
const say = (text, bad = false) => shell.say(text, bad);
const dpr = () => window.devicePixelRatio || 1;

// ── loading ───────────────────────────────────────────────────────────────────

async function openFile(file) {
  if (stroke) { say('Finish or cancel (Esc) the stroke before opening another mesh', true); return; }
  say(`Opening ${file.name}…`);
  try {
    const loaded = await loadModelFile(file);
    const raw = loaded.geometry ?? loaded;
    const geometry = raw.index ? raw.toNonIndexed() : raw;
    const model = buildMeshModel(geometry.attributes.position.array);
    const colours = new Float32Array(model.count * 9);
    for (let i = 0; i < colours.length; i += 3) colours.set(UNPAINTED, i);
    mesh = { model, geometry, name: file.name.replace(/\.[^.]+$/, ''), store: new GroupStore(model.count), colours };
    active = null;
    hoverKey = null;
    paintPromise = null;
    prepared = null;
    renderer.setMesh(model);
    renderer.setShading(settings.smooth);
    renderer.updateColours(colours, 0, model.count - 1);
    renderer.setBands('active', null);
    renderer.setBands('hover', null);
    edgesDirty = true;
    empty.hidden = true;
    zoomFit();
    refreshPanel();
    say(`${file.name}: ${model.count.toLocaleString()} triangles`);
    // Adjacency and curvature take a moment on a big mesh: start now so the first stroke doesn't wait.
    setTimeout(() => preparePaint().catch(() => {}), 50);
  } catch (error) {
    console.error(error);
    say(`Could not open ${file.name}: ${error.message}`, true);
  }
}

/** The paint tool and its per-mesh preparation, loaded on first need (it brings the ONNX runtime). */
function preparePaint() {
  if (!paintPromise) {
    const current = mesh;
    paintPromise = import('../paintTool.js').then((tool) => {
      const value = { tool, mesh: tool.preparePaintMesh(current.geometry) };
      if (mesh === current) prepared = value;
      return value;
    });
    paintPromise.catch((error) => {
      console.error(error);
      say(`The fill tool could not load: ${error.message}`, true);
      paintPromise = null;
    });
  }
  return paintPromise;
}

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) openFile(fileInput.files[0]);
  fileInput.value = '';
});
shell.stage.addEventListener('dragover', (event) => {
  event.preventDefault();
  shell.stage.classList.add('studio-dropping');
});
shell.stage.addEventListener('dragleave', () => shell.stage.classList.remove('studio-dropping'));
shell.stage.addEventListener('drop', (event) => {
  event.preventDefault();
  shell.stage.classList.remove('studio-dropping');
  if (event.dataTransfer.files[0]) openFile(event.dataTransfer.files[0]);
});

// ── camera ────────────────────────────────────────────────────────────────────

function zoomFit() {
  if (!mesh) return;
  const { centre, radius } = mesh.model;
  camera.center = [...centre];
  // An orthographic half height of about 1.1 radii leaves a margin round the body.
  camera.distance = radius * 2.7;
  requestRender();
}

const halfHeight = () => ui.orthographicHalfHeight(FOV, camera.distance);

function zoomAt(event) {
  const rect = canvas.getBoundingClientRect();
  const nx = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  const ny = 1 - ((event.clientY - rect.top) / rect.height) * 2;
  const aspect = rect.width / rect.height;
  const radius = mesh?.model.radius ?? 1;
  const before = halfHeight();
  const factor = event.deltaY > 0 ? 1.1 : 1 / 1.1;
  camera.distance = Math.min(radius * 200, Math.max(radius * 0.005, camera.distance * factor));
  // Keep the point under the cursor where it is.
  const shift = before - halfHeight();
  const { right, up } = camera;
  for (let a = 0; a < 3; a++) camera.center[a] += right[a] * nx * aspect * shift + up[a] * ny * shift;
  requestRender();
}

function pan(dx, dy) {
  const perPixel = (2 * halfHeight()) / canvas.clientHeight;
  const { right, up } = camera;
  for (let a = 0; a < 3; a++) camera.center[a] += -right[a] * dx * perPixel + up[a] * dy * perPixel;
  requestRender();
}

// ── painting ──────────────────────────────────────────────────────────────────

function setTool(tool) {
  settings.tool = tool;
  for (const id of ['fill', 'raw', 'erase']) toolbar.setPressed(id, id === tool);
  drawCursor();
}

const strokeMode = () => (modifiers.alt ? 'erase' : modifiers.shift ? 'raw' : settings.tool);

function devicePoint(event) {
  const rect = canvas.getBoundingClientRect();
  return { x: (event.clientX - rect.left) * dpr(), y: (event.clientY - rect.top) * dpr() };
}

function beginStroke(event) {
  if (!mesh) return;
  if (stroke) { say('Still filling the last stroke (Esc cancels it)', true); return; }
  const point = devicePoint(event);
  const mode = strokeMode();
  mesh.store.begin();
  stroke = {
    mode, gid: null, last: point, moved: 0, anchor: -1, zone: null, blocked: 0, lastMessage: null,
    started: renderer.pick(point.x, point.y, 0).centre,
    queue: [], queued: new Set(), pumping: false, released: false, cancelled: false, finishing: false,
    fills: 0, claimed: 0,
  };
  setHover(null);
  applyBrush(point);
}

/** The stroke's group, settled by the first triangle it lands on: that triangle's group, or a new one. */
function strokeGroup(t) {
  if (stroke.gid === null) {
    const { store } = mesh;
    const owner = store.owner[t];
    stroke.gid = owner >= 0 ? owner : store.create(settings.type === 'auto' ? null : settings.type).id;
    stroke.anchor = t;
    setActive(stroke.gid);
  }
  return stroke.gid;
}

function applyBrush(point) {
  if (stroke.cancelled) return;
  const { store } = mesh;
  if (stroke.mode === 'fill') {
    const { centre } = renderer.pick(point.x, point.y, 0);
    if (centre < 0) return;
    strokeGroup(centre);
    // Painted already, by this group or another (a fill never takes from another group).
    if (store.owner[centre] !== -1 || stroke.queued.has(centre)) return;
    stroke.queued.add(centre);
    stroke.queue.push(centre);
    pump();
    return;
  }
  const { centre, ids } = renderer.pick(point.x, point.y, settings.brushPx * dpr());
  if (!ids.size) return;
  const target = stroke.mode === 'erase' ? -1 : strokeGroup(centre >= 0 ? centre : ids.values().next().value);
  let changed = 0;
  for (const t of ids) if (store.assign(t, target)) changed++;
  if (changed) { flushChanged(); refreshPanel(); }
}

function moveStroke(point) {
  const from = stroke.last;
  const distance = Math.hypot(point.x - from.x, point.y - from.y);
  const step = stroke.mode === 'fill' ? 3 * dpr() : Math.max(2, (settings.brushPx * dpr()) / 2);
  const steps = Math.max(1, Math.ceil(distance / step));
  for (let i = 1; i <= steps; i++) {
    applyBrush({ x: from.x + ((point.x - from.x) * i) / steps, y: from.y + ((point.y - from.y) * i) / steps });
  }
  stroke.moved += distance / dpr();
  stroke.last = point;
}

async function pump() {
  const s = stroke;
  if (!s || s.pumping) return;
  s.pumping = true;
  try {
    const { tool, mesh: paintMesh } = await preparePaint();
    while (s.queue.length && !s.cancelled) {
      const seed = s.queue.shift();
      if (mesh.store.owner[seed] !== -1) continue;   // an earlier fill in this stroke took it
      const group = mesh.store.groups.get(s.gid);
      say(`Filling${s.queue.length ? ` (${s.queue.length} more queued)` : ''}…`);
      // A chosen surface button wins, and retypes a group the stroke extends. On the model
      // button a group keeps the type its first fill settled.
      const type = settings.type !== 'auto' ? settings.type : group.type ?? 'auto';
      if (!strokeZone(s, type, paintMesh.adjacency).has(seed)) { s.blocked++; continue; }
      const out = await tool.paintAt(seed, paintMesh, {
        method: settings.method, type, modelUrl: settings.model, taken: takenMask(s.gid),
        extended: MODELS.find((m) => m.url === settings.model)?.features === 'v6',
      });
      if (s.cancelled) break;
      if (settings.type !== 'auto') group.type = settings.type;
      else if (!group.type && out.fitter) group.type = out.fitter;
      let claimed = 0;
      for (const t of out.region) if (mesh.store.owner[t] === -1 && mesh.store.assign(t, s.gid)) claimed++;
      if (mesh.store.owner[seed] === -1 && mesh.store.assign(seed, s.gid)) claimed++;
      s.fills++;
      s.claimed += claimed;
      flushChanged();
      refreshPanel();
      s.lastMessage = fillMessage(out, claimed);
      say(s.lastMessage);
    }
  } catch (error) {
    console.error(error);
    s.queue.length = 0;
    say(`Fill failed: ${error.message}`, true);
  } finally {
    s.pumping = false;
  }
  if (s.released) finishStroke();
}

/**
 * Where a fill stroke may seed: the triangles reachable from where it landed (and from its
 * group's triangles) without crossing a crease for the stroke's surface, through unpainted
 * triangles or its own group's. Recomputed only when the surface type, and so the angle, changes.
 */
function strokeZone(s, type, adjacency) {
  const deg = type && type !== 'auto' ? creaseFor(CREASE_BARRIER, type) : 30;
  if (s.zone && s.zone.deg === deg) return s.zone.set;
  const { owner } = mesh.store;
  const set = new Set(s.anchor >= 0 ? [s.anchor] : []);
  for (let t = 0; t < owner.length; t++) if (owner[t] === s.gid) set.add(t);
  const stack = [...set];
  while (stack.length) {
    const t = stack.pop();
    for (const e of adjacency.get(t) || []) {
      const nb = e.neighbor;
      if (e.angle >= deg || set.has(nb) || (owner[nb] !== -1 && owner[nb] !== s.gid)) continue;
      set.add(nb);
      stack.push(nb);
    }
  }
  s.zone = { deg, set };
  return set;
}

/** Triangles painted into any group but `gid`: the model's patch and the fill walk around them. */
function takenMask(gid) {
  const { owner } = mesh.store;
  const mask = new Uint8Array(owner.length);
  for (let t = 0; t < owner.length; t++) if (owner[t] >= 0 && owner[t] !== gid) mask[t] = 1;
  return mask;
}

function fillMessage(out, claimed) {
  const parts = [`${typeLabel(out.fitter)} fill: +${claimed.toLocaleString()} triangles`];
  if (out.predicted) parts.push(`model said ${out.predicted}${out.confidence != null ? ` ${(out.confidence * 100).toFixed(0)}%` : ''}`);
  if (Number.isFinite(out.fit?.relResidual)) parts.push(`residual ${out.fit.relResidual.toPrecision(2)}`);
  parts.push(`${Math.round(out.ms)} ms`);
  if (out.notes.length) parts.push(out.notes[0]);
  return parts.join(' · ');
}

function releaseStroke() {
  stroke.released = true;
  if (!stroke.pumping) finishStroke();
}

function cancelStroke() {
  if (!stroke) return;
  stroke.cancelled = true;
  stroke.queue.length = 0;
  if (stroke.released && !stroke.pumping) finishStroke();
}

function finishStroke() {
  const s = stroke;
  if (!s || s.finishing) return;
  s.finishing = true;
  const { store } = mesh;
  if (s.cancelled) {
    store.cancel();
    stroke = null;
    flushChanged();
    if (active !== null && !store.count(active)) active = null;
    edgesDirty = true;
    refreshPanel();
    say('Stroke cancelled');
    return;
  }
  const touched = new Set([...store.edit.before.values()].filter((id) => id >= 0));
  if (s.gid !== null) touched.add(s.gid);
  for (const id of touched) refit(id);
  const name = s.gid !== null ? store.name(s.gid) : '';
  const label = s.mode === 'erase' ? 'Erase' : s.mode === 'raw' ? `Paint facets into ${name}` : `Fill ${name}`;
  if (s.blocked) say(`${s.lastMessage ? `${s.lastMessage} · ` : ''}stopped at a crease`);
  const changed = store.commit(label);
  stroke = null;
  // A click that painted nothing on empty space clears the active group, as it clears a selection.
  if (!changed && s.moved <= 4 && s.started < 0) setActive(null);
  if (active !== null && !store.count(active)) active = null;
  edgesDirty = true;
  refreshPanel();
}

/** Refit a group's whole region with its type, for the panel. Skipped until the fill tool has loaded. */
function refit(id) {
  const { store } = mesh;
  const group = store.groups.get(id);
  if (!group || !prepared) return;
  if (!group.type || !store.count(id)) { group.fit = null; return; }
  const result = fitRegionAs(group.type, store.triangles(id), prepared.mesh.facets);
  group.fit = result.params
    ? { params: result.params, relResidual: result.relResidual }
    : { reason: result.reason ?? 'the fit failed' };
}

/** Push changed triangles' colours to the GPU and mark the edges for a rebuild. */
function flushChanged() {
  if (!mesh) return;
  const { store, colours } = mesh;
  const changed = store.takeChanged();
  if (!changed.size) return;
  let first = Infinity;
  let last = -1;
  for (const t of changed) {
    const id = store.owner[t];
    const colour = id >= 0 ? store.groups.get(id)?.colour ?? UNPAINTED : UNPAINTED;
    colours.set(colour, t * 9);
    colours.set(colour, t * 9 + 3);
    colours.set(colour, t * 9 + 6);
    if (t < first) first = t;
    if (t > last) last = t;
  }
  renderer.updateColours(colours, first, last);
  edgesDirty = true;
  requestRender();
}

// ── groups ────────────────────────────────────────────────────────────────────

function setActive(id) {
  if (active === id) return;
  active = id;
  edgesDirty = true;
  refreshPanel();
}

/** A change to groups outside a stroke, as one undo step. */
function editGroups(label, change) {
  if (!mesh) return;
  if (stroke) { say('Finish or cancel (Esc) the stroke first', true); return; }
  const { store } = mesh;
  store.begin();
  const touched = change(store) ?? [];
  for (const id of touched) refit(id);
  store.commit(label);
  flushChanged();
  if (active !== null && !store.count(active)) active = null;
  edgesDirty = true;
  refreshPanel();
}

function deleteGroup(id) {
  const name = mesh.store.name(id);
  editGroups(`Delete ${name}`, (store) => {
    for (const t of store.triangles(id)) store.assign(t, -1);
  });
  say(`Deleted ${name}`);
}

function retype(id, type) {
  editGroups(`Fit ${mesh.store.name(id)} as ${typeLabel(type)}`, (store) => {
    store.groups.get(id).type = type;
    return [id];
  });
  const group = mesh.store.groups.get(id);
  if (group?.fit?.reason) say(`${typeLabel(type)} does not fit ${mesh.store.name(id)}: ${group.fit.reason}`, true);
}

function merge(from, into) {
  const name = mesh.store.name(from);
  editGroups(`Merge ${name} into ${mesh.store.name(into)}`, (store) => {
    for (const t of store.triangles(from)) store.assign(t, into);
    return [into];
  });
}

function groupMenu(id, event) {
  const { store } = mesh;
  const group = store.groups.get(id);
  if (!group) return;
  const fitAs = TYPES.filter((t) => t !== 'auto').map((type) => ({
    id: type, label: typeLabel(type), checked: group.type === type, onSelect: () => retype(id, type),
  }));
  fitAs.push({ id: 'freeform', label: 'Freeform (NURBS)', checked: !group.type, onSelect: () => retype(id, null) });
  const canMerge = active !== null && active !== id;
  ui.openMenu([
    { id: 'fit', label: 'Fit as', children: fitAs },
    { id: 'merge', label: canMerge ? `Merge into ${store.name(active)}` : 'Merge into the active group', disabled: !canMerge, onSelect: () => merge(id, active) },
    { separator: true },
    { id: 'delete', label: 'Delete group', onSelect: () => deleteGroup(id) },
  ], { at: { x: event.clientX, y: event.clientY }, compact: true });
}

function undo() {
  if (!mesh || stroke) return;
  const label = mesh.store.undo();
  if (!label) return;
  afterHistory();
  say(`Undid: ${label}`);
}

function redo() {
  if (!mesh || stroke) return;
  const label = mesh.store.redo();
  if (!label) return;
  afterHistory();
  say(`Redid: ${label}`);
}

function afterHistory() {
  // Group colours are fixed per id, but a restored group needs its triangles recoloured.
  flushChanged();
  if (active !== null && !mesh.store.count(active)) active = null;
  edgesDirty = true;
  refreshPanel();
}

// ── export ────────────────────────────────────────────────────────────────────

/** Fit every group with its painted type and build the STEP solid, as the main app's export does. */
async function exportStep() {
  if (!mesh) return;
  if (stroke) { say('Finish or cancel (Esc) the stroke first', true); return; }
  const { store, model, geometry, name } = mesh;
  const groups = store.list();
  if (!groups.length) { say('Paint at least one face group first', true); return; }
  toolbar.setDisabled('export', 'Exporting…');
  try {
    const [{ classifyGroupAsGauss }, builder] = await Promise.all([import('../surfaceFitter.js'), import('../brepBuilder.js')]);
    say(`Fitting ${groups.length} groups…`);
    const payload = groups.map((group) => {
      const part = { triangleIndices: [...store.triangles(group.id)] };
      part.surface = classifyGroupAsGauss(group.type ?? 'nurbs', part, geometry);
      return part;
    });
    const content = await builder.buildAndExportSTEP(payload, geometry, { schema: 'AP214' },
      (text, pct) => say(`${text} ${Math.round(pct)}%`));
    builder.downloadSTEP(content, `${name}.stp`);
    const unpainted = model.count - store.painted;
    const failed = payload.filter((p) => p.surface.degenerate).length;
    say([
      `Exported ${name}.stp from ${groups.length} groups`,
      failed ? `${failed} could not take their type and went out as freeform patches` : '',
      unpainted ? `${unpainted.toLocaleString()} unpainted triangles were left out` : '',
    ].filter(Boolean).join(' · '), failed > 0 || unpainted > 0);
  } catch (error) {
    console.error(error);
    say(`Export failed: ${error.message}`, true);
  } finally {
    refreshPanel();
  }
}

// ── panel, toolbar, status ────────────────────────────────────────────────────

const format = (v) => (Math.abs(v) >= 100 ? v.toFixed(1) : v.toPrecision(4));

function describeGroup(group) {
  const { store } = mesh;
  const head = `${store.name(group.id)}: ${store.count(group.id).toLocaleString()} triangles`;
  if (!group.type) return `${head} · exported as a freeform patch`;
  if (!group.fit) return head;
  if (group.fit.reason) return `${head} · ${typeLabel(group.type)} fit failed: ${group.fit.reason}`;
  const p = group.fit.params;
  const bits = [];
  if (p.majorRadius != null) bits.push(`R ${format(p.majorRadius)}`, `r ${format(p.minorRadius)}`);
  else if (p.radius != null) bits.push(`r ${format(p.radius)}`);
  if (p.halfAngle != null) bits.push(`half angle ${format((p.halfAngle * 180) / Math.PI)}°`);
  if (Number.isFinite(group.fit.relResidual)) bits.push(`residual ${group.fit.relResidual.toPrecision(2)}`);
  return [head, ...bits].join(' · ');
}

function swatch(type, colour) {
  const rgb = colour.map((v) => Math.round(v * 255)).join(',');
  return `<svg class="os-icon" viewBox="0 0 20 20" aria-hidden="true" style="color: rgb(${rgb})">${ui.iconInner(TYPE_ICONS[type] ?? 'surface')}</svg>`;
}

function refreshPanel() {
  const hasMesh = !!mesh;
  const store = mesh?.store;
  const groups = hasMesh ? store.list() : [];
  section.setTitle(`Face groups (${groups.length})`);
  const rows = groups.map((group) => ({
    id: group.id,
    icon: swatch(group.type, group.colour),
    label: store.name(group.id),
    value: store.count(group.id).toLocaleString(),
    selected: group.id === active,
    tooltip: describeGroup(group),
    onClick: () => setActive(group.id === active ? null : group.id),
    onContextMenu: (row, event) => groupMenu(group.id, event),
  }));
  if (hasMesh) {
    const unpainted = mesh.model.count - store.painted;
    if (unpainted) rows.push({ id: 'unpainted', icon: swatch('plane', UNPAINTED), label: 'Unpainted', value: unpainted.toLocaleString(), dim: true });
  }
  section.setRows(rows);
  section.setButtonDisabled('delete', active === null);
  toolbar.setDisabled('undo', store?.undoStack.length ? null : 'Nothing to undo');
  toolbar.setDisabled('redo', store?.redoStack.length ? null : 'Nothing to redo');
  toolbar.setDisabled('export', groups.length ? null : 'Paint at least one face group first');
  if (hasMesh) {
    const pct = (100 * store.painted) / mesh.model.count;
    shell.setStatus(`${store.painted.toLocaleString()} of ${mesh.model.count.toLocaleString()} triangles painted (${pct.toFixed(pct < 10 ? 1 : 0)}%)`);
  } else {
    shell.setStatus('');
  }
  shell.hint.textContent = `Shift: facets · Alt: erase · [ ]: brush ${settings.brushPx}px`;
}

function setType(type) {
  settings.type = type;
  for (const t of TYPES) toolbar.setPressed(`type-${t}`, t === type);
  say(type === 'auto' ? 'Fill: the model picks the surface' : `Fill as ${typeLabel(type).toLowerCase()}`);
}

function fillMenu() {
  const pick = (key, value) => () => {
    settings[key] = value;
    say(`Fill: ${METHODS.find((m) => m.id === settings.method).label}, ${MODELS.find((m) => m.url === settings.model).label}`);
  };
  return [
    { id: 'method', label: 'Fill method', children: METHODS.map((m) => ({ id: m.id, label: m.label, checked: settings.method === m.id, onSelect: pick('method', m.id) })) },
    { id: 'model', label: 'Model', children: MODELS.map((m) => ({ id: m.url, label: m.label, checked: settings.model === m.url, onSelect: pick('model', m.url) })) },
  ];
}

function openFillMenu() {
  const button = toolbar.get('fill-options');
  if (ui.isMenuOpen(button.parentElement)) return;
  ui.openMenu(fillMenu(), { anchor: button.parentElement, align: 'left' });
}

// ── hover and cursor ──────────────────────────────────────────────────────────

function setHover(key, triangle = -1) {
  if (key === hoverKey) return;
  hoverKey = key;
  let ends = null;
  if (key && mesh) {
    const { model, store } = mesh;
    if (key.startsWith('g')) {
      const id = Number(key.slice(1));
      ends = outline(model, store.triangles(id), (t) => store.owner[t] === id);
    } else {
      ends = outline(model, [triangle], (t) => t === triangle);
    }
  }
  renderer.setBands('hover', ends);
  requestRender();
}

function updateHover(point) {
  if (!mesh || strokeMode() !== 'fill') { setHover(null); return; }
  const { centre } = renderer.pick(point.x, point.y, 0);
  if (centre < 0) { setHover(null); return; }
  const id = mesh.store.owner[centre];
  if (id === active && id >= 0) setHover(null);
  else if (id >= 0) setHover(`g${id}`);
  else setHover(`t${centre}`, centre);
}

function drawCursor() {
  const context = overlay.getContext('2d');
  context.clearRect(0, 0, overlay.width, overlay.height);
  const mode = strokeMode();
  canvas.classList.toggle('studio-canvas--brush', mode !== 'fill');
  if (!cursor || !mesh || mode === 'fill') return;
  const ratio = dpr();
  context.beginPath();
  context.arc(cursor.x * ratio, cursor.y * ratio, settings.brushPx * ratio, 0, 2 * Math.PI);
  context.lineWidth = ratio;
  context.setLineDash(mode === 'erase' ? [4 * ratio, 3 * ratio] : []);
  context.strokeStyle = 'rgba(255, 255, 255, 0.85)';
  context.stroke();
}

// ── input ─────────────────────────────────────────────────────────────────────

let orbiting = null;
let panning = null;

const noteModifiers = (event) => {
  const changed = modifiers.shift !== event.shiftKey || modifiers.alt !== event.altKey;
  modifiers.shift = event.shiftKey;
  modifiers.alt = event.altKey;
  if (changed) { drawCursor(); if (strokeMode() !== 'fill') setHover(null); }
};

canvas.addEventListener('contextmenu', (event) => event.preventDefault());

canvas.addEventListener('pointerdown', (event) => {
  noteModifiers(event);
  canvas.setPointerCapture(event.pointerId);
  if (event.button === 2 && event.ctrlKey || event.button === 1) {
    event.preventDefault();
    panning = { x: event.clientX, y: event.clientY };
  } else if (event.button === 2) {
    orbiting = { x: event.clientX, y: event.clientY, moved: 0 };
  } else if (event.button === 0) {
    beginStroke(event);
  }
});

canvas.addEventListener('pointermove', (event) => {
  noteModifiers(event);
  const rect = canvas.getBoundingClientRect();
  cursor = { x: event.clientX - rect.left, y: event.clientY - rect.top };
  if (orbiting) {
    // Onshape's turntable, from the kit camera: the model follows the mouse.
    camera.drag(event.clientX - orbiting.x, event.clientY - orbiting.y, canvas.clientWidth, canvas.clientHeight);
    orbiting.moved += Math.abs(event.clientX - orbiting.x) + Math.abs(event.clientY - orbiting.y);
    orbiting.x = event.clientX;
    orbiting.y = event.clientY;
    requestRender();
  } else if (panning) {
    pan(event.clientX - panning.x, event.clientY - panning.y);
    panning.x = event.clientX;
    panning.y = event.clientY;
  } else if (stroke && !stroke.released) {
    moveStroke(devicePoint(event));
  } else {
    updateHover(devicePoint(event));
  }
  drawCursor();
});

canvas.addEventListener('pointerup', (event) => {
  if (orbiting) {
    const press = orbiting;
    orbiting = null;
    // A right button released where it went down is a click: the menu for what is there.
    if (press.moved <= 4 && mesh) {
      const point = devicePoint(event);
      const { centre } = renderer.pick(point.x, point.y, 0);
      if (centre >= 0 && mesh.store.owner[centre] >= 0) groupMenu(mesh.store.owner[centre], event);
    }
    return;
  }
  if (panning) { panning = null; return; }
  if (stroke && !stroke.released && event.button === 0) releaseStroke();
});

canvas.addEventListener('pointerleave', () => {
  cursor = null;
  if (!stroke) setHover(null);
  drawCursor();
});

canvas.addEventListener('wheel', (event) => {
  event.preventDefault();
  zoomAt(event);
}, { passive: false });

window.addEventListener('keydown', (event) => {
  if (event.target instanceof HTMLInputElement) return;
  noteModifiers(event);
  const mod = event.ctrlKey || event.metaKey;
  const key = event.key.toLowerCase();
  if (mod && key === 'z') { event.preventDefault(); if (event.shiftKey) redo(); else undo(); }
  else if (mod && key === 'y') { event.preventDefault(); redo(); }
  else if (mod && key === 'o') { event.preventDefault(); fileInput.click(); }
  else if (event.key === 'Escape') cancelStroke();
  else if (event.key === ' ') { event.preventDefault(); setActive(null); }
  else if ((event.key === 'Delete' || event.key === 'Backspace') && active !== null) deleteGroup(active);
  else if (event.key === '[' || event.key === ']') {
    settings.brushPx = Math.min(64, Math.max(3, Math.round(settings.brushPx * (event.key === ']' ? 1.25 : 0.8))));
    refreshPanel();
    drawCursor();
  } else if (key === 'f' && !mod) zoomFit();
  else if (!mod && event.key >= '1' && event.key <= String(TYPES.length)) setType(TYPES[Number(event.key) - 1]);
  else if (event.key === 'Alt') event.preventDefault();   // keep Alt from reaching a browser menu
});
window.addEventListener('keyup', (event) => {
  noteModifiers(event);
  if (event.key === 'Alt') event.preventDefault();
});
window.addEventListener('blur', () => { modifiers.shift = false; modifiers.alt = false; drawCursor(); });

// ── frame loop ────────────────────────────────────────────────────────────────

function frame(now) {
  camera.step(now);
  if (camera.isAnimating()) needsRender = true;
  const ratio = dpr();
  const width = Math.max(1, Math.floor(canvas.clientWidth * ratio));
  const height = Math.max(1, Math.floor(canvas.clientHeight * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
    overlay.width = width;
    overlay.height = height;
    needsRender = true;
    drawCursor();
  }
  if (edgesDirty && mesh) {
    const { model, store } = mesh;
    renderer.setBands('edges', bodyEdges(model, store.owner));
    renderer.setBands('active', active !== null && store.count(active)
      ? outline(model, store.triangles(active), (t) => store.owner[t] === active) : null);
    if (hoverKey?.startsWith('g')) { const key = hoverKey; hoverKey = null; setHover(key); }
    edgesDirty = false;
    needsRender = true;
  }
  if (needsRender) {
    needsRender = false;
    const view = camera.viewMatrix();
    const radius = mesh?.model.radius ?? 1;
    const centre = mesh?.model.centre ?? [0, 0, 0];
    // Depth range: the body's sphere, wherever panning has left it relative to the orbit centre.
    const reach = radius + Math.hypot(...camera.center.map((c, a) => c - centre[a]));
    const projection = settings.perspective
      ? ui.perspective(FOV, width / height, Math.max(camera.distance * 1e-3, camera.distance - 2 * reach), camera.distance + 2 * reach)
      : ui.orthographic(halfHeight(), width / height, camera.distance - 2 * reach, camera.distance + 2 * reach);
    renderer.draw({
      view, projection, width, height, ratio,
      occlusionPercent: settings.occlusion ? ui.LIGHTING.occlusion.percent : 0,
      occlusionRadius: radius * 0.06,
      edges: settings.edges,
    });
    corner.update();
  }
  requestAnimationFrame(frame);
}

refreshPanel();
requestAnimationFrame(frame);

const initial = new URLSearchParams(location.search).get('mesh');
if (initial) {
  fetch(initial)
    .then((response) => { if (!response.ok) throw new Error(`${response.status} ${response.statusText}`); return response.blob(); })
    .then((blob) => openFile(new File([blob], initial.split('/').pop())))
    .catch((error) => say(`Could not open ${initial}: ${error.message}`, true));
}
