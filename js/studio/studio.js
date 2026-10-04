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
import { creaseTableFor, creaseFor } from '../regionGrow.js';
import { coherentScores, seedOrder, autoPaint, AUTO_TANGENT } from '../autoPaint.js';
import { annotateEdges, edgeWall } from '../edgeWall.js';

/** The learned face-boundary scorer (ml/edge_train.py), scored once per mesh. */
const EDGE_MODEL = 'models/edge-boundary.onnx';
/**
 * REFINED FILLS (settings.refine, paintCore.refineFill): the second boundary model, trained on 2000
 * training-split ABC parts instead of 40 (runs/edge2), and the fill-refinement model inside a minimum
 * cut. Held out, clean single fills 78/65/69/48 -> 87/81/82/54% (OCCT / NURBS / gmsh / game-style) and
 * auto-paint area clean 75.3/75.3/76.0/65.8 -> 81.8/82.5/85.4/70.3%.
 * Since 2026-10-04 the refiner is runs/r3mp: six MeshConv layers, so it takes the patch graph. Clean
 * single fills with the cut 87/81/82/54 -> 89/83/86/55%; on the 300 Onshape exports surfaces found once
 * 84.8 -> 85.4% (docs/ml-experiment-scope.md, experiment 3).
 */
const EDGE_MODEL_V2 = 'models/edge-boundary-v2.onnx';
const REFINER = 'models/fill-refiner.onnx';
/** Plane fills keep the 2 degree crease instead: with the wall they measured worse, not better. */
const EDGE_WALL_TYPES = ['cylinder', 'cone', 'sphere', 'torus'];
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
// The toolbar's surface buttons: the types, then a freeform fill (no primitive: the export fits a B-spline).
const FILL_TYPES = [...TYPES, 'freeform'];
// A freeform fill floods the unpainted triangles to 30-degree creases. One click per true freeform face
// of the user's parts (scratchpad fffill.mjs): whole and clean on 20/20 Star faces at every resolution
// and 28/28 Cake Pan faces (coarse, medium); the coarse Ornament 8/14, 14/14 once its neighbours are
// painted; the Fencepost's twisted faces join tangentially, so a click takes three (one B-spline does).
// Not widened on "low-poly" meshes: the Star reads as one (step 50, its faces flat to 1 degree inside),
// and at 80 degrees a click flooded 10 of its 20 faces. A curvature-jump wall cut every face short
// (Star 29-84% covered).
const FREEFORM_CREASE = 30;
// v7gnn first: v6feat's recipe plus six rounds of message passing over the patch (ml/model.py
// MeshConv; the runner feeds it the patch graph). Validation IoU 0.821 -> 0.859. With refined fills,
// held-out auto-paint area clean 85.6/84.7/86.6/72.3% against 81.8/82.5/85.4/70.3% for v6feat
// (test/autoSeed.mjs --refine --crf, 40 odd ABC parts). `features: 'v6'` makes paintTool build the
// extended inputs; v6feat was the best of 13 models before it.
const MODELS = [
  { url: 'models/seedfill-v7gnn.onnx', label: 'v7 (message passing)', features: 'v6' },
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
  // A rippled sheet: a freeform (B-spline) fill.
  freeform: '<path d="M2 6.5c2.2-2.6 4.4-2.6 6.6 0s4.4 2.6 6.6 0L18 5.2v2.3c-2.2 2.6-4.4 2.6-6.6 0s-4.4-2.6-6.6 0L2 8.8z" /><path d="M2 12.2c2.2-2.6 4.4-2.6 6.6 0s4.4 2.6 6.6 0l2.8-1.3v2.3c-2.2 2.6-4.4 2.6-6.6 0s-4.4-2.6-6.6 0L2 14.5z" />',
  // A four-point spark: the model picks the surface.
  auto: '<path d="M8 1.5c.5 3.6 2.4 5.5 6 6-3.6.5-5.5 2.4-6 6-.5-3.6-2.4-5.5-6-6 3.6-.5 5.5-2.4 6-6zM15 11c.3 2 1.3 3 3.3 3.3-2 .3-3 1.3-3.3 3.3-.3-2-1.3-3-3.3-3.3 2-.3 3-1.3 3.3-3.3z" />',
  // A wand with sparks: auto-paint.
  wand: '<path d="M2.5 16.4 12.9 6l1.6 1.6L4.1 18z" /><path d="M14 4.9l1.3-1.3 1.6 1.6-1.3 1.3zM8.6 1.5l.6 1.5 1.5.6-1.5.6-.6 1.5-.6-1.5-1.5-.6 1.5-.6zM16.4 9.6l.5 1.2 1.2.5-1.2.5-.5 1.2-.5-1.2-1.2-.5 1.2-.5z" />',
});

const settings = {
  tool: 'fill',
  // 'model' (the model's region -> fitter), as auto-paint uses. Through the studio's own path with the
  // type chosen automatically (test/growEval.mjs studio:*, 40 held-out parts, v7 + refined fills) it
  // is clean on 83/76/77/51% of clicks against 76/69/76/48% for 'hyp' (the model's surface -> fitter).
  method: 'model',
  type: 'auto',
  model: MODELS[0].url,
  brushPx: 12,
  // The fill auto-paint uses per seed; 'model' is what test/autoSeed.mjs measured.
  autoMethod: 'model',
  // Auto-paint grows with the curvature-jump wall (autoPaint.AUTO_TANGENT). Off by default: it
  // recovers more area cleanly on ABC parts but splits faces into many more groups (the watch stand
  // goes from 66 to 277 new groups), each a merge for the user. Fill menu toggle.
  autoTangent: false,
  // Stop fills at the learned face boundary (js/edgeWall.js). Measured on held-out parts with the
  // true type declared: clean fills 62 -> 69% OCCT, 48 -> 58% gmsh, with spill 32 -> 17% and
  // 49 -> 34%. Fill menu toggle.
  edgeWall: true,
  // Refined fills: boundary model v2, then the refiner inside a minimum cut (paintCore.refineFill).
  // Confirmed through the studio's own path on 40 held-out ABC parts (v7, type chosen automatically):
  // clean single fills 75/63/65/46 -> 83/76/77/51% OCCT / NURBS / gmsh / game-style. Fill menu toggle.
  refine: true,
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
  ...FILL_TYPES.map((type, i) => ({
    kind: 'toggle',
    id: `type-${type}`,
    icon: type === 'auto' ? 'auto' : type === 'freeform' ? 'freeform' : TYPE_ICONS[type],
    title: type === 'auto'
      ? `Model picks the surface from the seed (${i + 1})`
      : type === 'freeform'
        ? `Freeform fill: grows to creases; exported as a B-spline (${i + 1})`
        : `Fill as ${typeLabel(type).toLowerCase()} (${i + 1})`,
    pressed: settings.type === type,
    onClick: () => setType(type),
  })),
  { kind: 'separator' },
  { kind: 'tool', id: 'autopaint', icon: 'wand', title: 'Auto-paint every unpainted face (P); Esc stops', onClick: () => runAutoPaint() },
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
/** The auto-paint in progress: { stop } — Esc sets stop. */
let autoRun = null;
let lastEdgeBuild = 0;
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
  get autoRun() { return autoRun; },
  settings,
  camera,
};

const requestRender = () => { needsRender = true; };
const say = (text, bad = false) => shell.say(text, bad);
const dpr = () => window.devicePixelRatio || 1;

// ── loading ───────────────────────────────────────────────────────────────────

async function openFile(file) {
  if (stroke || autoRun) { say('Finish or cancel (Esc) the stroke or auto-paint before opening another mesh', true); return; }
  say(`Opening ${file.name}…`);
  try {
    const loaded = await loadModelFile(file);
    const raw = loaded.geometry ?? loaded;
    const geometry = raw.index ? raw.toNonIndexed() : raw;
    const model = buildMeshModel(geometry.attributes.position.array);
    const colours = new Float32Array(model.count * 9);
    for (let i = 0; i < colours.length; i += 3) colours.set(UNPAINTED, i);
    mesh = { model, geometry, name: file.name.replace(/\.[^.]+$/, ''), store: new GroupStore(model.count), colours,
             offset: loaded.offset ?? [0, 0, 0] };   // the loader centred the mesh; the export moves it back
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
    // Seed scores for auto-paint too, so the button starts at once.
    const opened = mesh;
    setTimeout(() => preparePaint()
      .then(() => edgeScoresFor(opened))
      .then(() => coherenceFor(opened))
      .catch(() => {}), 50);
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
  if (autoRun) { say('Auto-painting (Esc stops it)', true); return; }
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

/** A group's stored type for a surface button: null (untyped: freeform at export) for the model and freeform. */
const groupTypeFor = (type) => (type === 'auto' || type === 'freeform' ? null : type);

/** The stroke's group, settled by the first triangle it lands on: that triangle's group, or a new one. */
function strokeGroup(t) {
  if (stroke.gid === null) {
    const { store } = mesh;
    const owner = store.owner[t];
    stroke.gid = owner >= 0 ? owner : store.create(groupTypeFor(settings.type)).id;
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
      // button a group keeps the type its first fill settled: an untyped group that already has
      // triangles (a freeform fill, facet brush or auto-paint's freeform join) grows freeform.
      const type = settings.type !== 'auto' ? settings.type
        : group.type ?? (mesh.store.count(s.gid) ? 'freeform' : 'auto');
      if (!strokeZone(s, type, paintMesh.adjacency).has(seed)) { s.blocked++; continue; }
      let out;
      if (type === 'freeform') out = freeformFill(seed, paintMesh);
      else {
        const refineUrl = await refineUrlFor(mesh);
        out = await tool.paintAt(seed, paintMesh, {
          method: settings.method, type, modelUrl: settings.model, taken: takenMask(s.gid),
          extended: MODELS.find((m) => m.url === settings.model)?.features === 'v6',
          grow: growOptions(), refineUrl,
        });
      }
      if (s.cancelled) break;
      if (settings.type !== 'auto') group.type = groupTypeFor(settings.type);
      else if (!group.type && out.fitter && type !== 'freeform') group.type = out.fitter;
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
  const deg = creaseOf(type, adjacency);
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

/** The crease a fill of `type` stops at (degrees). */
function creaseOf(type, adjacency) {
  if (!type || type === 'auto') return 30;
  if (type === 'freeform') return FREEFORM_CREASE;
  return creaseFor(creaseTableFor(adjacency), type);
}

/**
 * A freeform fill: the unpainted triangles reachable from `seed` without crossing a crease. No
 * primitive is fitted; the group exports as a B-spline (js/freeformFit.js).
 */
function freeformFill(seed, paintMesh) {
  const t0 = performance.now();
  const { adjacency } = paintMesh;
  const { owner } = mesh.store;
  const deg = creaseOf('freeform', adjacency);
  const region = new Set([seed]);
  const stack = [seed];
  while (stack.length) {
    const t = stack.pop();
    for (const e of adjacency.get(t) || []) {
      const nb = e.neighbor;
      if (region.has(nb) || owner[nb] !== -1 || e.angle >= deg) continue;
      region.add(nb);
      stack.push(nb);
    }
  }
  return { region, fitter: null, ms: performance.now() - t0, notes: [] };
}

// ── auto-paint ────────────────────────────────────────────────────────────────

/**
 * Score every edge of a loaded mesh with the boundary model, once, and build the wall growth uses.
 * A failure (no model file, no runtime) leaves `target.edgeWall` unset and fills behave as before.
 */
function edgeScoresFor(target) {
  // Scores live on the adjacency itself, so a switch of boundary model re-scores the mesh. Jobs run
  // one after another: the scores in place are always those of the last model asked for.
  const url = settings.refine ? EDGE_MODEL_V2 : EDGE_MODEL;
  if (!target.edgeScores || target.edgeUrl !== url) {
    target.edgeUrl = url;
    const previous = target.edgeScores ?? Promise.resolve();
    target.edgeScores = previous.catch(() => {}).then(() => preparePaint()).then(async ({ tool, mesh: paintMesh }) => {
      const { runner, meta } = await tool.edgeScorerFor(url);
      const { edges, ms } = await annotateEdges(paintMesh.facets, paintMesh.adjacency, runner, meta.columns);
      const logit = meta.thresholds['fpr0.02'].logit;
      target.edgeWall = edgeWall(1 / (1 + Math.exp(-logit)), EDGE_WALL_TYPES);
      console.log(`edge boundaries (${url}): ${edges} edges in ${ms.toFixed(0)} ms`);
      return target.edgeWall;
    });
    target.edgeScores.catch((error) => console.warn('no learned face boundaries:', error.message));
  }
  return target.edgeScores;
}

/**
 * The refiner for this fill, when refined fills are on and the mesh's edges carry the second boundary
 * model's scores (the refiner reads them). Waits for scoring that is still running.
 */
async function refineUrlFor(target) {
  if (!settings.refine) return null;
  try {
    await edgeScoresFor(target);
  } catch {
    return null;
  }
  return target.edgeUrl === EDGE_MODEL_V2 ? REFINER : null;
}

/** Growth options shared by strokes and auto-paint: the learned wall when it is on and ready. */
function growOptions() {
  return settings.edgeWall && mesh?.edgeWall ? { wall: mesh.edgeWall } : {};
}

/** Per-facet seed coherence for a loaded mesh (js/autoPaint.js), computed once and cached. */
function coherenceFor(target) {
  if (!target.coherence) {
    target.coherence = preparePaint().then(({ mesh: paintMesh }) => coherentScores(paintMesh.facets, paintMesh.adjacency));
  }
  return target.coherence;
}

/**
 * Fill every unpainted face, best seed first, as new groups. Existing groups are never touched, so
 * this works as a first pass or after the user has painted some faces. One undo step; Esc stops
 * and keeps what was painted.
 */
async function runAutoPaint() {
  if (!mesh) return;
  if (stroke || autoRun) { say('Finish or cancel (Esc) the stroke or auto-paint first', true); return; }
  const current = mesh;
  const { store } = current;
  autoRun = { stop: false };
  toolbar.setPressed('autopaint', true);
  setActive(null);
  setHover(null);
  let editOpen = false;
  try {
    say('Auto-paint: preparing…');
    const { tool, mesh: paintMesh } = await preparePaint();
    say('Auto-paint: scoring seeds…');
    const order = seedOrder(await coherenceFor(current), paintMesh.facets);
    if (autoRun.stop || mesh !== current) return;
    const extended = MODELS.find((m) => m.url === settings.model)?.features === 'v6';
    const refineUrl = await refineUrlFor(current);
    const created = [];
    const uiClock = { panel: 0, yield: performance.now() };
    store.begin();
    editOpen = true;
    const result = await autoPaint({
      facets: paintMesh.facets,
      order,
      owner: store.owner,
      fill: (seed, taken) => tool.paintAt(seed, paintMesh, {
        method: settings.autoMethod, type: 'auto', modelUrl: settings.model, taken, extended, reportFit: false,
        grow: { ...growOptions(), ...(settings.autoTangent ? { tangent: AUTO_TANGENT } : {}) }, refineUrl,
      }),
      onSegment: async (claim, out) => {
        const group = store.create(out.fitter ?? null);
        for (const t of claim) store.assign(t, group.id);
        created.push(group.id);
        flushChanged();
        // On a clock, not per group: the panel holds a row per group and a yield costs a timer
        // tick, and a low-poly mesh makes thousands of groups.
        const now = performance.now();
        if (now - uiClock.panel > 750) { refreshPanel(); uiClock.panel = now; }
        if (now - uiClock.yield > 40) {
          await new Promise((resolve) => setTimeout(resolve, 0));   // let a frame show the new groups
          uiClock.yield = performance.now();
        }
        return group.id;
      },
      // Every claim is kept: a flat rectangle is often exactly two triangles, and dropping groups
      // under three triangles threw away real faces (area clean 78 -> 69% on OCCT meshes).
      minKeep: 1,
      // Unpainted islands of up to 8 triangles left inside a face join the neighbouring new group.
      adjacency: paintMesh.adjacency,
      absorbUpTo: 8,
      onAbsorb: async (island, id) => {
        for (const t of island) store.assign(t, id);
        flushChanged();
      },
      onDiscard: async (claim, id) => {
        for (const t of claim) store.assign(t, -1);
        created.splice(created.indexOf(id), 1);
        flushChanged();
      },
      shouldStop: () => autoRun.stop || mesh !== current,
      onProgress: ({ segments, free }) => say(`Auto-paint: ${segments} groups · ${free.toLocaleString()} triangles left (Esc stops)`),
    });
    // Freeform faces back into one group each (js/freeformMerge.js): auto-paint knows only primitives and
    // carves a B-spline face into many. Only this run's groups, and only on a part made mostly of such
    // faces; the joined groups stay untyped, so the export fits them a B-spline. Part of the undo step.
    say('Auto-paint: joining freeform pieces…');
    const joined = Int32Array.from(store.owner);
    const { freeform } = tool.freeformMerge(joined, paintMesh.facets, paintMesh.adjacency, { only: new Set(created) });
    for (let t = 0; t < joined.length; t++) if (joined[t] !== store.owner[t]) store.assign(t, joined[t]);
    flushChanged();
    // One pass over the mesh for every new group's triangles, not one pass per group.
    const byGroup = new Map(created.map((id) => [id, new Set()]));
    for (let t = 0; t < store.owner.length; t++) byGroup.get(store.owner[t])?.add(t);
    for (const [id, tris] of byGroup) if (!tris.size) byGroup.delete(id);
    // Each finished group's export type (paintCore.typeFinalGroup): untyped groups get one, and a group
    // a simpler or truer primitive fits exactly takes it; freeform groups stay untyped.
    say('Auto-paint: typing groups…');
    for (const [id, tris] of byGroup) {
      const group = store.groups.get(id);
      if (freeform.has(id)) { group.type = null; refit(id, tris); continue; }
      const typed = tool.typeFinalGroup(tris, paintMesh.facets, group.type);
      if (typed.changed) group.type = typed.type;
      refit(id, tris);
      const now = performance.now();
      if (now - uiClock.yield > 40) {
        await new Promise((resolve) => setTimeout(resolve, 0));
        uiClock.yield = performance.now();
      }
    }
    store.commit(`Auto-paint ${result.segments} groups`);
    editOpen = false;
    const left = result.free > 0 ? ` · ${result.free.toLocaleString()} triangles left unpainted` : '';
    const ff = freeform.size ? ` (${freeform.size} freeform, joined from ${result.segments})` : '';
    say(`Auto-paint ${result.stopped ? 'stopped' : 'done'}: ${byGroup.size} new groups${ff}${left}`);
  } catch (error) {
    console.error(error);
    if (editOpen) { store.commit('Auto-paint (interrupted)'); editOpen = false; }
    say(`Auto-paint failed: ${error.message}`, true);
  } finally {
    autoRun = null;
    toolbar.setPressed('autopaint', false);
    edgesDirty = true;
    refreshPanel();
  }
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
function refit(id, tris = null) {
  const { store } = mesh;
  const group = store.groups.get(id);
  if (!group || !prepared) return;
  if (!group.type || !store.count(id)) { group.fit = null; return; }
  const result = fitRegionAs(group.type, tris ?? store.triangles(id), prepared.mesh.facets);
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
  if (stroke || autoRun) { say('Finish or cancel (Esc) the stroke or auto-paint first', true); return; }
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
  if (!mesh || stroke || autoRun) return;
  const label = mesh.store.undo();
  if (!label) return;
  afterHistory();
  say(`Undid: ${label}`);
}

function redo() {
  if (!mesh || stroke || autoRun) return;
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

/**
 * Build the STEP solid from the painted groups with the topo knitter (js/knit, the port of
 * ml/solid_rebuild.py): the B-rep's topology from the mesh, so neighbouring faces share their edges by
 * construction, each group's fitted surface, the kernel-judged repair loop, and flat facets only where
 * the analytic build cannot close. It replaces the sewing builder, which built every face on its own and
 * closed none of 31 fully analytic test parts. Runs in a worker: builds take seconds to minutes.
 */
async function exportStep() {
  if (!mesh) return;
  if (stroke || autoRun) { say('Finish or cancel (Esc) the stroke or auto-paint first', true); return; }
  const { store, model, geometry, name } = mesh;
  const groups = store.list();
  if (!groups.length) { say('Paint at least one face group first', true); return; }
  toolbar.setDisabled('export', 'Exporting…');
  try {
    let facets = prepared?.mesh?.facets;
    if (!facets) {
      const [{ buildFacets }, { attachCurvature }] = await Promise.all([import('../meshCore.js'), import('../curvature.js')]);
      const geo = geometry.index ? geometry.toNonIndexed() : geometry;
      const P = Float64Array.from(geo.attributes.position.array);
      facets = buildFacets(P);
      attachCurvature(P, facets);
    }
    say(`Fitting ${groups.length} groups…`);
    const surfaces = [];
    for (const group of groups) {
      if (!group.type) continue;                  // no analytic surface: the knitter makes it facets
      const fit = group.fit?.params ? group.fit : fitRegionAs(group.type, store.triangles(group.id), facets);
      if (fit.params) surfaces.push([group.id, [group.type, fit.params]]);
    }
    // The STEP goes where the file's mesh was: the loader centred it for the view, so the centre goes
    // back on the positions, and on every fitted surface (all are anchored by points, never directions).
    const [ox, oy, oz] = mesh.offset;
    const positions = Float64Array.from(facets.positions, (v, i) => v + (i % 3 === 0 ? ox : i % 3 === 1 ? oy : oz));
    const payload = { positions, labels: Int32Array.from(store.owner), surfaces: surfaces.map(([id, [type, p]]) => [id, [type, shifted(p, mesh.offset)]]) };
    let result;
    try {
      result = await knitInWorker({ ...payload, options: { deadline: EXPORT_DEADLINE_S } }, (text) => say(text), EXPORT_HARD_STOP_S);
    } catch (error) {
      if (error.name !== 'TimeoutError') throw error;
      // A single OCCT repair on badly fitting surfaces can run for most of an hour and cannot be
      // interrupted, so the worker is stopped and a fresh one exports the mesh itself.
      say(`The surfaces took over ${Math.round(EXPORT_HARD_STOP_S / 60)} minutes to join; exporting the mesh as a faceted solid…`, true);
      result = await knitInWorker({ ...payload, options: { mode: 'faceted' } }, (text) => say(text));
      result.report.timedOut = true;
    }
    const { step, report } = result;
    if (!step) throw new Error('no shape came out of the build');
    downloadStep(step, `${name}.stp`);
    say(exportSummary(`${name}.stp`, report, model.count - store.painted), report.tier !== 'analytic');
  } catch (error) {
    console.error(error);
    say(`Export failed: ${error.message}`, true);
  } finally {
    refreshPanel();
  }
}

/** Fitted surface params moved by `d`: every point-valued param (origin, axisPoint, apex, center). */
function shifted(p, d) {
  const out = { ...p };
  for (const k of ['origin', 'axisPoint', 'apex', 'center']) if (p[k]) out[k] = [p[k][0] + d[0], p[k][1] + d[1], p[k][2] + d[2]];
  return out;
}

// The knitter stops starting repair builds and facet tiers after EXPORT_DEADLINE_S; a build already
// running is stopped with its worker at EXPORT_HARD_STOP_S. Measured without them: 26 min (Fencepost,
// 54k triangles) and 49 min (Cake Pan, 23k) before a faceted result, both badly typed freeform parts.
const EXPORT_DEADLINE_S = 150;
const EXPORT_HARD_STOP_S = 240;

/** js/knit/worker.js, once per export (the kernel's memory goes with it); rejects with TimeoutError after `limitS`. */
function knitInWorker(payload, onStatus, limitS = Infinity) {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL('../knit/worker.js', import.meta.url), { type: 'module' });
    const timer = Number.isFinite(limitS) ? setTimeout(() => {
      worker.terminate();
      const error = new Error(`the solid build took longer than ${limitS} s`);
      error.name = 'TimeoutError';
      reject(error);
    }, limitS * 1000) : null;
    worker.onmessage = ({ data }) => {
      if (data.type === 'status') onStatus(data.text);
      else if (data.type === 'progress') {
        onStatus(data.freeform ? `Fitting freeform surfaces… ${data.freeform} of ${data.of}` : `Building the solid… attempt ${data.builds}`);
      }
      else {
        clearTimeout(timer);
        worker.terminate();
        if (data.type === 'done') resolve(data);
        else reject(new Error(data.message));
      }
    };
    worker.onerror = (event) => { clearTimeout(timer); worker.terminate(); reject(new Error(event.message || 'the export worker failed')); };
    worker.postMessage(payload);
  });
}

/** What the export holds, said plainly: every face analytic, or how much went out as facets, or why not a solid. */
function exportSummary(file, report, unpainted) {
  const pct = (x) => `${(100 * x).toFixed(x < 0.01 ? 1 : 0)}%`;
  const parts = [];
  if (report.tier === 'analytic') parts.push(`Exported ${file}: a closed solid, every face on its fitted surface`);
  else if (report.tier === 'hybrid' || report.tier === 'planar') {
    parts.push(`Exported ${file}: a closed solid; ${report.facetTris.toLocaleString()} triangles (${pct(report.facetArea)} of the area) ` +
      `went out as flat facets where ${report.tier === 'planar' ? 'the curved' : 'their'} surfaces would not join`);
  } else if (report.tier === 'faceted') {
    parts.push(report.timedOut
      ? `Exported ${file} as the mesh, a faceted solid: joining the fitted surfaces took too long`
      : `Exported ${file}: the surfaces would not join into a solid, so this is the mesh as a faceted solid`);
  } else {
    parts.push(`Exported ${file}, but it is NOT a closed solid` +
      (report.free_edges ? `: ${report.free_edges.toLocaleString()} open edges` : '') +
      (report.bestTier === 'faceted' ? ' - the mesh itself is open, so no surfaces can close it' : ''));
  }
  if (unpainted > 0) parts.push(`${unpainted.toLocaleString()} unpainted triangles went out as facets`);
  parts.push(`${report.secs} s`);
  return parts.join(' · ');
}

function downloadStep(text, filename) {
  const url = URL.createObjectURL(new Blob([text], { type: 'application/octet-stream' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
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
  shell.hint.textContent = `P: auto-paint · Shift: facets · Alt: erase · [ ]: brush ${settings.brushPx}px`;
}

function setType(type) {
  settings.type = type;
  for (const t of FILL_TYPES) toolbar.setPressed(`type-${t}`, t === type);
  say(type === 'auto' ? 'Fill: the model picks the surface'
    : type === 'freeform' ? 'Freeform fill: grows to creases; the export fits a B-spline'
      : `Fill as ${typeLabel(type).toLowerCase()}`);
}

function fillMenu() {
  const pick = (key, value) => () => {
    settings[key] = value;
    say(`Fill: ${METHODS.find((m) => m.id === settings.method).label}, ${MODELS.find((m) => m.url === settings.model).label}`);
  };
  return [
    { id: 'method', label: 'Fill method', children: METHODS.map((m) => ({ id: m.id, label: m.label, checked: settings.method === m.id, onSelect: pick('method', m.id) })) },
    { id: 'model', label: 'Model', children: MODELS.map((m) => ({ id: m.url, label: m.label, checked: settings.model === m.url, onSelect: pick('model', m.url) })) },
    { separator: true },
    {
      id: 'edgeWall', label: 'Stop fills at learned face borders', checked: settings.edgeWall,
      onSelect: () => {
        settings.edgeWall = !settings.edgeWall;
        say(settings.edgeWall
          ? 'Fills stop at the learned face borders'
          : 'Fills use the crease barrier only');
      },
    },
    {
      id: 'refine', label: 'Refined fills (second pass + minimum cut)', checked: settings.refine,
      onSelect: () => {
        settings.refine = !settings.refine;
        if (mesh) edgeScoresFor(mesh);
        say(settings.refine
          ? 'Fills get a second pass: boundary model v2, the refiner and a minimum cut'
          : 'Fills use the first pass only, with boundary model v1');
      },
    },
    {
      id: 'autoTangent', label: 'Auto-paint stops at curvature changes', checked: settings.autoTangent,
      onSelect: () => {
        settings.autoTangent = !settings.autoTangent;
        say(settings.autoTangent
          ? 'Auto-paint stops at curvature changes: cleaner faces, many more groups to merge'
          : 'Auto-paint grows across curvature changes');
      },
    },
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
  else if (event.key === 'Escape') { if (autoRun) autoRun.stop = true; else cancelStroke(); }
  else if (event.key === ' ') { event.preventDefault(); setActive(null); }
  else if ((event.key === 'Delete' || event.key === 'Backspace') && active !== null) deleteGroup(active);
  else if (event.key === '[' || event.key === ']') {
    settings.brushPx = Math.min(64, Math.max(3, Math.round(settings.brushPx * (event.key === ']' ? 1.25 : 0.8))));
    refreshPanel();
    drawCursor();
  } else if (key === 'f' && !mod) zoomFit();
  else if (key === 'p' && !mod) runAutoPaint();
  else if (!mod && event.key >= '1' && event.key <= String(FILL_TYPES.length)) setType(FILL_TYPES[Number(event.key) - 1]);
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
  // While auto-paint runs, edges are rebuilt at most twice a second rather than after every fill.
  if (edgesDirty && mesh && (!autoRun || now - lastEdgeBuild > 500)) {
    lastEdgeBuild = now;
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
