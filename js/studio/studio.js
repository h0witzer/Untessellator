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
import { repairMesh } from '../meshRepair.js';
import { preflight, PREFLIGHT } from '../knit/preflight.js';
import { distNormal } from '../knit/geometry.js';
import { capOwners, fitTriangles, projectCapVertices } from './caps.js';
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
/**
 * The group-level type model (settings.groupTyper, js/groupType.js): after auto-paint, each new group's
 * export type comes from a learned typer over the fit evidence classifyRegion computes, where its top
 * probability reaches the sidecar's threshold, and from paintCore.typeFinalGroup elsewhere.
 */
const GROUP_TYPER = 'models/group-typer.onnx';
/** Plane fills keep the 2 degree crease instead: with the wall they measured worse, not better. */
const EDGE_WALL_TYPES = ['cylinder', 'cone', 'sphere', 'torus'];
import { buildMeshModel, bodyEdges, outline } from './meshModel.js';
import { createRenderer, HEAT, NO_LEVEL } from './renderer.js';
import { GroupStore, typeLabel } from './groups.js';
import { conversionNumbers, reportUrl, reportJson, placeGroups, binaryStl, fingerprint } from './report.js';
import { zipSync, strToU8 } from 'fflate';

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
  // A flag on a pole: report how the conversion went.
  flag: '<path d="M4 2h1.6v16H4z" /><path d="M6.4 2.8h10l-2.6 3.9 2.6 3.9h-10z" />',
});

const settings = {
  tool: 'fill',
  // Repair a mesh as it opens (js/meshRepair.js): welds, duplicate triangles and walls between bodies,
  // zero-area needles, T-junction cracks, holes, winding. A clean mesh comes through unchanged. Measured on
  // 12 Onshape parts damaged each way (test/meshDamage.mjs): exports fully analytic 0-3 of 12 -> 11-12 of 12
  // for duplicates, flips, holes, cracks and unwelded seams. Fill menu toggle; applies to the next file opened.
  repair: true,
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
  // Type auto-painted groups with the group-level type model (GROUP_TYPER), typeFinalGroup elsewhere.
  groupTyper: true,
  perspective: ui.LIGHTING.perspectiveByDefault,
  edges: true,
  smooth: false,
  occlusion: true,
  // The deviation views (D steps through them; view cube menu): 'groups' colors each group by whether the
  // export check expects it to come out right, 'heat' maps where the mesh lies off each group's surface.
  // Off (null) by default, so painting shows group colors and measures nothing extra.
  deviation: null,
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
const legend = document.createElement('div');
legend.className = 'studio-legend';
legend.hidden = true;
const legendBody = document.createElement('div');
const legendProbe = document.createElement('div');
legendProbe.className = 'studio-legend__probe';
legend.append(legendBody, legendProbe);
shell.stage.append(legend);
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

// The view switch, by Export: group colors, the export check's verdict per group (which), or the heat map
// (where). Its own control, not the kit's: three labelled states, one lit. D steps through them.
const viewSwitch = document.createElement('div');
viewSwitch.className = 'studio-views';
viewSwitch.setAttribute('role', 'group');
viewSwitch.setAttribute('aria-label', 'View');
for (const [view, label, tip] of [
  ['off', 'Groups', 'Each group in its own color (D steps through the views)'],
  ['groups', 'Check', 'Each group colored by whether the export check expects it to come out right; hover one for why (D)'],
  ['heat', 'Heat map', 'Where the mesh lies off each group\'s fitted surface, on a log scale (D)'],
]) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'studio-views__button';
  button.dataset.view = view;
  button.textContent = label;
  button.setAttribute('data-tooltip', tip);
  button.setAttribute('aria-pressed', view === 'off' ? 'true' : 'false');
  button.addEventListener('click', () => setDeviationView(view === 'off' ? null : view));
  viewSwitch.append(button);
}

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
  { kind: 'element', id: 'view', element: viewSwitch },
  { kind: 'tool', id: 'report', icon: 'flag', title: 'Report how this conversion went: opens a GitHub issue with this run\'s numbers; nothing is sent until you submit it', onClick: () => reportConversion() },
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
let legendDirty = false;
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
    let geometry = raw.index ? raw.toNonIndexed() : raw;
    // The repaired soup replaces the loaded one: everything after (view, painting, export) indexes its
    // triangles. Caps the repair added are marked, so painting and fitting can leave them out (caps.js).
    let repair = null, synthetic = null;
    if (settings.repair) {
      say(`Opening ${file.name}: checking the mesh…`);
      await new Promise((resolve) => setTimeout(resolve, 0));   // let the message show: a big mesh takes seconds
      const r = repairMesh(geometry.attributes.position.array);
      repair = r.report;
      if (repairChanges(repair).length) {
        geometry = { index: null, attributes: { position: { array: Float32Array.from(r.positions) } } };
        if (repair.cap_triangles) synthetic = Uint8Array.from(r.src, (s) => (s < 0 ? 1 : 0));
      }
    }
    const model = buildMeshModel(geometry.attributes.position.array);
    const colours = new Float32Array(model.count * 9);
    for (let i = 0; i < colours.length; i += 3) colours.set(UNPAINTED, i);
    mesh = { model, geometry, name: file.name.replace(/\.[^.]+$/, ''), store: new GroupStore(model.count, model.neighbours), colours,
             offset: loaded.offset ?? [0, 0, 0],   // the loader centred the mesh; the export moves it back
             repair, synthetic, flags: new Map(), showFlags: false,
             // the deviation views: each triangle's farthest corner off its group's surface (x diagonal, NaN
             // with none), each corner's level for the renderer, the groups refit since last colored, and
             // per group its farthest triangle and its verdict ('on', 'maybe', 'off' or 'none')
             deviation: new Float32Array(model.count).fill(NaN), levels: new Float32Array(model.count * 3).fill(NO_LEVEL),
             refitted: new Set(), groupWorst: new Map(), verdicts: new Map(),
             // how each group came to be, for the conversion report (report.js): group id -> origin
             origins: new Map() };
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
    legendDirty = true;
    legend.hidden = !settings.deviation;
    empty.hidden = true;
    zoomFit();
    refreshPanel();
    const fixed = repair ? repairChanges(repair) : [];
    const left = repair ? repairLeft(repair) : null;
    say([`${file.name}: ${model.count.toLocaleString()} triangles`, ...(fixed.length ? [`repaired: ${fixed.join(', ')}`] : []),
      ...(left ? [left] : [])].join(' · '), !!left);
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

/** What repairMesh changed, as phrases; empty for a clean mesh. */
function repairChanges(r) {
  const n = (x) => x.toLocaleString();
  const out = [];
  if (r.dup_same) out.push(`${n(r.dup_same)} duplicate triangles removed`);
  if (r.dup_walls) out.push(`${n(r.dup_walls)} walls between touching bodies removed`);
  if (r.collapsed + r.needles) out.push(`${n(r.collapsed + r.needles)} zero-area triangles removed`);
  if (r.zipped) out.push(`${n(r.zipped)} cracks closed`);
  if (r.holes_capped) out.push(`${n(r.holes_capped)} hole${r.holes_capped > 1 ? 's' : ''} capped`);
  if (r.welded) out.push(`${n(r.welded)} unwelded corners joined`);
  if (r.flipped) out.push(`${n(r.flipped)} triangles turned outward`);
  return out;
}

/** What repairMesh could not fix and the export will meet, or null. */
function repairLeft(r) {
  const parts = [];
  if (r.nonmanifold_edges) parts.push(`${r.nonmanifold_edges.toLocaleString()} edges shared by 3 or more triangles (bodies touching)`);
  if (r.holes_left) parts.push(`${r.holes_left} hole${r.holes_left > 1 ? 's' : ''} it could not cap`);
  return parts.length ? `the mesh still has ${parts.join(' and ')}: the export may not be a closed solid` : null;
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
  if (autoRun) { say(`${autoRun.what ?? 'Auto-painting'} (Esc stops it)`, true); return; }
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
    if (owner < 0) mesh.origins.set(stroke.gid, { by: 'stroke', tool: stroke.mode, declared: settings.type, seed: t });
    else if (mesh.origins.has(owner)) mesh.origins.get(owner).edited = true;
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
  const started = performance.now();
  autoRun = { stop: false };
  toolbar.setPressed('autopaint', true);
  setActive(null);
  setHover(null);
  let editOpen = false;
  try {
    say('Auto-paint: preparing…');
    const { tool, mesh: paintMesh } = await preparePaint();
    say('Auto-paint: scoring seeds…');
    const caps = current.synthetic;
    const order = seedOrder(await coherenceFor(current), paintMesh.facets).filter((t) => !caps?.[t]);   // a cap seeds nothing
    if (autoRun.stop || mesh !== current) return;
    const extended = MODELS.find((m) => m.url === settings.model)?.features === 'v6';
    const refineUrl = await refineUrlFor(current);
    const created = [];
    const uiClock = { panel: 0, yield: performance.now() };
    store.begin();
    editOpen = true;
    let lastSeed = -1;                            // autoPaint calls onSegment right after its seed's fill
    const result = await autoPaint({
      facets: paintMesh.facets,
      order,
      owner: store.owner,
      fill: (seed, taken) => {
        lastSeed = seed;
        return tool.paintAt(seed, paintMesh, {
          method: settings.autoMethod, type: 'auto', modelUrl: settings.model, taken, extended, reportFit: false,
          grow: { ...growOptions(), ...(settings.autoTangent ? { tangent: AUTO_TANGENT } : {}) }, refineUrl,
        });
      },
      onSegment: async (claim, out) => {
        const group = store.create(out.fitter ?? null);
        for (const t of claim) store.assign(t, group.id);
        created.push(group.id);
        current.origins.set(group.id, { by: 'auto-paint', seed: lastSeed, predicted: out.predicted ?? null,
          confidence: out.confidence ?? null, fill: out.fitter ?? null, claimed: claim.size });
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
    // Caps the repair added join the group around them (caps.js), whatever the fills made of them.
    for (const [t, id] of capOwners(store.owner, caps, current.model.neighbours)) store.assign(t, id);
    flushChanged();
    // One pass over the mesh for every new group's triangles, not one pass per group. Typing and fitting see
    // the input's own triangles: a flat cap would make a curved group inexact.
    const byGroup = new Map(created.map((id) => [id, new Set()]));
    for (let t = 0; t < store.owner.length; t++) if (!caps?.[t]) byGroup.get(store.owner[t])?.add(t);
    for (const [id, tris] of byGroup) if (!tris.size) byGroup.delete(id);
    // Each finished group's export type (paintCore.typeFinalGroup): untyped groups get one, and a group
    // a simpler or truer primitive fits exactly takes it; freeform groups stay untyped.
    say('Auto-paint: typing groups…');
    // The learned typer decides where it is confident; it needs groups of 3 or more triangles. A model
    // that fails to load leaves every group to typeFinalGroup, as before.
    const learned = new Map();
    const typer = settings.groupTyper ? await tool.groupTyperFor(GROUP_TYPER).catch(() => null) : null;
    if (typer) {
      const rows = [];
      for (const [id, tris] of byGroup) {
        if (freeform.has(id)) continue;
        const f = tool.groupTypeFeatures(tris, paintMesh.facets, paintMesh.adjacency, store.groups.get(id).type);
        if (f) rows.push([id, f]);
        const now = performance.now();
        if (now - uiClock.yield > 40) {
          await new Promise((resolve) => setTimeout(resolve, 0));
          uiClock.yield = performance.now();
        }
      }
      if (rows.length) {
        const width = tool.N_GROUP_FEATURES, k = tool.GROUP_TYPES.length;
        const X = new Float32Array(rows.length * width);
        rows.forEach(([, f], i) => X.set(f.features, i * width));
        const probs = await typer.run(X, rows.length);
        rows.forEach(([id, f], i) => {
          const d = tool.decideGroupType(probs.subarray(i * k, i * k + k), f.rule, typer.threshold);
          learned.set(id, d.type);
          const origin = current.origins.get(id);
          if (origin) origin.typer = { by: d.by, p: Math.round(d.p * 1000) / 1000 };
        });
      }
    }
    for (const [id, tris] of byGroup) {
      const group = store.groups.get(id);
      const origin = current.origins.get(id);
      if (freeform.has(id)) {
        group.type = null;
        if (origin) origin.typedBy = 'freeform merge';
        refit(id, tris);
        continue;
      }
      if (learned.has(id)) {
        group.type = learned.get(id);
        if (origin) origin.typedBy = origin.typer?.by === 'model' ? 'group typer' : 'typer rule';
      } else {
        const typed = tool.typeFinalGroup(tris, paintMesh.facets, group.type);
        if (typed.changed) group.type = typed.type;
        if (origin) origin.typedBy = typed.changed ? 'exact rule' : 'fill';
      }
      refit(id, tris);
      const now = performance.now();
      if (now - uiClock.yield > 40) {
        await new Promise((resolve) => setTimeout(resolve, 0));
        uiClock.yield = performance.now();
      }
    }
    store.commit(`Auto-paint ${result.segments} groups`);
    editOpen = false;
    // counted now, not from the run: caps the repair added were placed after it
    let free = 0;
    for (let t = 0; t < store.owner.length; t++) if (store.owner[t] < 0) free++;
    const left = free > 0 ? ` · ${free.toLocaleString()} triangles left unpainted` : '';
    // for the conversion report (report.js)
    current.autoPaint = { secs: (performance.now() - started) / 1000, groups: byGroup.size, freeform: freeform.size,
                          unpainted: free, stopped: !!result.stopped };
    const ff = freeform.size ? ` (${freeform.size} freeform, joined from ${result.segments})` : '';
    say(`Auto-paint ${result.stopped ? 'stopped' : 'done'}: ${byGroup.size} new groups${ff}${left}`);
  } catch (error) {
    console.error(error);
    if (editOpen) { store.commit('Auto-paint (interrupted)'); editOpen = false; }
    say(`Auto-paint failed: ${error.message}`, true);
  } finally {
    autoRun = null;
    toolbar.setPressed('autopaint', false);
    flushChanged();   // groups the commit recolored apart from their neighbors
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
  flushChanged();   // a group the commit recolored apart from its neighbors
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
  mesh.flags.delete(id);   // a changed group is no longer the one the export check flagged
  if (!group || !prepared) return;
  // the deviation view recolors the group against its new surface (in the frame loop, once per frame)
  if (settings.deviation) mesh.refitted.add(id);
  if (!group.type || !store.count(id)) { group.fit = null; return; }
  const result = fitRegionAs(group.type, fitTriangles(tris ?? store.triangles(id), mesh.synthetic), prepared.mesh.facets);
  // the type goes with the params: a stroke can retype the group before its refit
  group.fit = result.params
    ? { type: group.type, params: result.params, relResidual: result.relResidual }
    : { reason: result.reason ?? 'the fit failed' };
}

/** Push changed triangles' colours to the GPU and mark the edges for a rebuild. */
function flushChanged() {
  if (!mesh) return;
  const changed = mesh.store.takeChanged();
  if (!changed.size) return;
  recolor(changed);
  edgesDirty = true;
}

/** Color triangles `tris` (every triangle when null) for the current view and push them to the GPU. */
function recolor(tris = null) {
  const { colours, model, store } = mesh;
  // The deviation views measure first: a group's verdict is its farthest triangle's, and a triangle that
  // changes it recolors the whole group in the group view.
  if (settings.deviation) {
    if (tris) for (const t of tris) mesh.deviation[t] = measureTriangle(t);
    else for (let t = 0; t < model.count; t++) mesh.deviation[t] = measureTriangle(t);
    const changed = updateVerdicts(tris ? new Set(Array.from(tris, (t) => store.owner[t])) : null);
    if (tris && changed.size && settings.deviation === 'groups') tris = new Set([...tris, ...trianglesOf(store.owner, changed)]);
  }
  let first = Infinity;
  let last = -1;
  const paint = (t) => {
    const color = triangleColor(t);
    colours.set(color, t * 9);
    colours.set(color, t * 9 + 3);
    colours.set(color, t * 9 + 6);
    if (t < first) first = t;
    if (t > last) last = t;
  };
  if (tris) for (const t of tris) paint(t);
  else for (let t = 0; t < model.count; t++) paint(t);
  renderer.updateColours(colours, first, last);
  if (settings.deviation === 'heat') renderer.updateLevels(mesh.levels, first, last);
  if (settings.deviation) legendDirty = true;
  requestRender();
}

/**
 * A triangle's color: its group's, or in the group view its group's verdict. In the heat view the renderer
 * colors measured corners by their levels instead, so only a triangle with no surface needs a color.
 */
function triangleColor(t) {
  const { store } = mesh;
  const id = store.owner[t];
  if (id < 0) return UNPAINTED;
  if (settings.deviation === 'groups') return VERDICTS[mesh.verdicts.get(id) ?? 'none'].rgb;
  if (settings.deviation === 'heat' && Number.isNaN(mesh.deviation[t])) return NO_SURFACE;
  return store.groups.get(id)?.colour ?? UNPAINTED;
}

// ── groups ────────────────────────────────────────────────────────────────────

function setActive(id) {
  if (active === id) return;
  active = id;
  if (id !== null && mesh) mesh.showFlags = false;   // selecting a group ends the flagged-groups outline
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
  // a restored group has its old surface back, so in the deviation view all of its triangles change
  if (settings.deviation) recolor();
  mesh.flags.clear();   // the export check's flags were for the groups as they were
  mesh.showFlags = false;
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
async function exportStep({ checked = false } = {}) {
  if (!mesh) return;
  if (stroke || autoRun) { say('Finish or cancel (Esc) the stroke or auto-paint first', true); return; }
  const current = mesh;
  const { store, model, name } = current;
  if (!store.list().length) { say('Paint at least one face group first', true); return; }
  toolbar.setDisabled('export', 'Exporting…');
  current.lastExport = null;                      // this attempt's result replaces any earlier one
  try {
    const payload = await knitPayload(current);
    // Before the build, the groups that will come out wrong (knit/preflight.js), so they can be fixed now
    // rather than after minutes of building.
    if (!checked) {
      say('Checking the groups…');
      await new Promise((resolve) => setTimeout(resolve, 0));
      const doubts = exportDoubts(payload);
      current.lastCheck = { groups: [...doubts.groups], mesh: doubts.mesh ?? null };   // for the conversion report
      if (doubts.groups.size || doubts.mesh) { offerDoubts(doubts); return; }
    }
    let result;
    try {
      result = await knitInWorker({ ...payload, options: { deadline: EXPORT_DEADLINE_S } }, (text) => say(text), EXPORT_HARD_STOP_S);
    } catch (error) {
      // A single OCCT repair on badly fitting surfaces can run for most of an hour and cannot be
      // interrupted, so the worker is stopped and a fresh one exports the mesh itself. A build that throws
      // can leave the kernel's memory corrupt (its fallback then fails too: "table index is out of bounds"),
      // so that too gets a fresh worker for the faceted solid.
      const timedOut = error.name === 'TimeoutError';
      say(timedOut
        ? `The surfaces took over ${Math.round(EXPORT_HARD_STOP_S / 60)} minutes to join; exporting the mesh as a faceted solid…`
        : `The solid build failed (${error.message}); exporting the mesh as a faceted solid…`, true);
      result = await knitInWorker({ ...payload, options: { mode: 'faceted' } }, (text) => say(text));
      if (timedOut) result.report.timedOut = true;
      else result.report.failed = error.message;
    }
    const { step, report } = result;
    current.lastExport = report;                  // for the conversion report (report.js)
    if (!step) throw new Error('no shape came out of the build');
    // Faces off their surfaces: the STEP waits while the user sees which groups failed and picks Repair or this
    // export as it is (a stopped or failed build went out as the mesh itself, with no verdicts to repair from)
    if (report.tier !== 'analytic' && !report.timedOut && !report.failed) { offerRepair(current, step, report); return; }
    downloadFile(step, `${name}.stp`);
    say(exportSummary(`${name}.stp`, report, model.count - store.painted), report.tier !== 'analytic');
  } catch (error) {
    console.error(error);
    current.lastExport ??= { failed: error.message };
    say(`Export failed: ${error.message}`, true);
  } finally {
    refreshPanel();
  }
}

/**
 * What the knitter is given for `current`: positions back where the file had them, a group per triangle, and
 * each typed group's fitted surface. The export builds from it, and the conversion report carries it, so a
 * reported run can be rebuilt exactly (test/reportReplay.mjs).
 */
async function knitPayload(current) {
  const { store, model, geometry, synthetic } = current;
  const groups = store.list();
  let facets = current === mesh ? prepared?.mesh?.facets : null;
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
    const fit = group.fit?.params ? group.fit : fitRegionAs(group.type, fitTriangles(store.triangles(group.id), synthetic), facets);
    if (fit.params) surfaces.push([group.id, [group.type, fit.params]]);
  }
  // The STEP goes where the file's mesh was: the loader centred it for the view, so the centre goes
  // back on the positions, and on every fitted surface (all are anchored by points, never directions).
  const [ox, oy, oz] = current.offset;
  const positions = Float64Array.from(facets.positions, (v, i) => v + (i % 3 === 0 ? ox : i % 3 === 1 ? oy : oz));
  const labels = Int32Array.from(store.owner);
  // Caps the repair added go to the group around them, and onto its surface (caps.js), however they were painted.
  for (const [t, id] of capOwners(labels, synthetic, model.neighbours)) labels[t] = id;
  const shiftedSurfaces = surfaces.map(([id, [type, p]]) => [id, [type, shifted(p, current.offset)]]);
  projectCapVertices(positions, labels, synthetic, new Map(shiftedSurfaces));
  return { positions, labels, surfaces: shiftedSurfaces };
}

/**
 * The Report button: what goes in the report file is the user's choice, since the picture and the mesh show
 * the part. Without a mesh open there is no file, only the issue.
 */
function reportConversion() {
  if (!mesh) { sendReport(); return; }
  ui.openMenu([
    { id: 'groups', label: 'Report the face groups and results', onSelect: () => sendReport() },
    { id: 'picture', label: '…and a picture of the view', onSelect: () => sendReport({ picture: true }) },
    { id: 'mesh', label: '…and the picture and the mesh (only if you may share it)', onSelect: () => sendReport({ picture: true, withMesh: true }) },
  ], { anchor: toolbar.get('report').parentElement, align: 'right' });
}

/**
 * Download the report file (report.js) and open the prefilled GitHub issue it goes into. Nothing is sent
 * from here: the user drags the file into the issue, checks it, and submits it or not.
 */
async function sendReport({ picture = false, withMesh = false } = {}) {
  if (stroke || autoRun) { say('Finish or cancel (Esc) the stroke or auto-paint first', true); return; }
  const current = mesh;
  const { store } = current ?? {};
  const modelLabel = MODELS.find((m) => m.url === settings.model)?.label ?? settings.model;
  const numbers = conversionNumbers({
    triangles: current?.model.count ?? 0,
    repaired: current?.repair ? repairChanges(current.repair) : [],
    model: modelLabel,
    settings,
    autoPaint: current?.autoPaint ?? null,
    groups: store ? store.list().map((g) => ({ type: g.type, count: store.count(g.id) })) : [],
    painted: store?.painted ?? 0,
    check: current?.lastCheck ?? null,
    exported: current?.lastExport ?? null,
    groupRepair: current?.lastRepair ?? null,
  });
  const issue =(extra = '') => reportUrl(numbers + extra, { tier: current?.lastExport?.tier, groups: store?.list().length ?? 0 });
  if (!current) {
    window.open(issue(), '_blank', 'noopener');
    say('Opened a conversion report on GitHub');
    return;
  }
  try {
    say('Building the report…');
    // The knitter's input exactly as an export would send it, so a reported run can be rebuilt.
    const payload = await knitPayload(current);
    const json = reportJson({
      page: location.href.split(/[?#]/)[0], settings, model: modelLabel, triangles: current.model.count,
      repair: current.repair, fingerprint: await fingerprint(current.geometry.attributes.position.array),
      includesMesh: withMesh, autoPaint: current.autoPaint, groups: store.list(),
      placed: placeGroups(payload.positions, payload.labels), origins: current.origins,
      check: current.lastCheck, exported: current.lastExport, groupRepair: current.lastRepair ?? null,
    });
    const files = {
      'report.json': strToU8(JSON.stringify(json, null, 1)),
      'part.fid': new Uint8Array(payload.labels.buffer),
      'part.fits.json': strToU8(JSON.stringify({ faces: payload.surfaces.map(([id, [type, params]]) => ({ id, type, params })) })),
    };
    if (picture) files['view.png'] = await captureView();
    if (withMesh) files['part.stl'] = binaryStl(payload.positions);
    const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const file = `Untessellator-report-${stamp}.zip`;
    downloadFile(zipSync(files, { level: 6 }), file);
    window.open(issue(`\nreport file: ${file}`), '_blank', 'noopener');
    say(`Downloaded ${file}: drag it into the GitHub issue that just opened, check the numbers, then submit it`);
  } catch (error) {
    console.error(error);
    say(`The report failed: ${error.message}`, true);
  }
}

/** The viewport as a PNG: resolved by the next frame, right after it draws (the drawing buffer is not kept). */
let viewCapture = null;
function captureView() {
  return new Promise((resolve) => { viewCapture = resolve; requestRender(); });
}

/**
 * The export check's bar (knit/preflight.js): only the near-certain flags. Measured against the true faces of
 * 275 Onshape parts and 45 fill_eval parts (test/exportRisk.mjs): a vertex more than 1e-3 x the part diagonal
 * off its group's surface, at the boundary or inside, or a curved group covering under 10 degrees, is a wrong
 * group 94-100% of the time on Onshape and alarmed at most 1 export whose groups were all right. The softer
 * bars (1e-5) catch more but are right only about 65% of the time.
 */
const DOUBT = { DEV: 1e-3, SEAM: 1e-3, ARC: 10 };

/** What the export check doubts: Map group -> reasons, and the mesh's own problem (or null). */
function exportDoubts({ positions, labels, surfaces }) {
  const pf = preflight(positions, labels, new Map(surfaces), DOUBT);
  const size = (x) => format(x * pf.diag);
  const groups = new Map();
  for (const [id, r] of pf.groups) {
    if (!mesh.store.groups.has(id)) continue;
    const type = typeLabel(r.type).toLowerCase();
    const why = [];
    if (r.flags.includes('off its surface')) why.push(`vertices inside it are up to ${size(r.dev)} off its ${type}`);
    if (r.flags.includes('seam gap')) why.push(`its border runs up to ${size(r.seam)} off its ${type}`);
    if (r.flags.includes('narrow arc')) why.push(`it covers only ${r.arc.toFixed(0)}° of its ${type}, which a cone, cylinder or torus fit about equally`);
    if (why.length) groups.set(id, why);
  }
  const m = pf.mesh;
  const holes = [];
  if (m.openEdges) holes.push(`${m.openEdges.toLocaleString()} open edges`);
  if (m.nonmanifoldEdges) holes.push(`${m.nonmanifoldEdges.toLocaleString()} edges shared by 3 or more triangles`);
  return { groups, mesh: holes.length ? `the mesh has ${holes.join(' and ')}, so the export will not be a closed solid` : null };
}

/** Say what the check found, mark the groups in the panel, and offer to show them or export anyway. */
function offerDoubts({ groups, mesh: meshDoubt }) {
  mesh.flags = groups;
  mesh.showFlags = false;
  const n = groups.size;
  const lines = [...groups].slice(0, 3).map(([id, why]) => `${mesh.store.name(id)}: ${why[0]}`);
  say([...(n ? [`${n} group${n > 1 ? 's' : ''} will likely come out wrong (⚠ in the list)`] : []), ...(meshDoubt ? [meshDoubt] : []),
    ...lines, ...(n > 3 ? [`and ${n - 3} more`] : [])].join(' · '), true);
  refreshPanel();
  const button = toolbar.get('export');
  ui.openMenu([
    ...(n ? [{ id: 'show', label: `Show the ${n} group${n > 1 ? 's' : ''}`, onSelect: () => showDoubts() }] : []),
    { id: 'export', label: 'Export anyway', onSelect: () => exportStep({ checked: true }) },
  ], { anchor: button.parentElement, align: 'right' });
}

/** The triangles whose group is in `ids`. */
function trianglesOf(owner, ids) {
  const out = [];
  for (let t = 0; t < owner.length; t++) if (ids.has(owner[t])) out.push(t);
  return out;
}

/** Outline every flagged group (the active group's outline style) until a group is selected. */
function showDoubts() {
  if (!mesh?.flags.size) return;
  setActive(null);
  mesh.showFlags = true;
  edgesDirty = true;
  requestRender();
  say(`Outlined: ${[...mesh.flags.keys()].map((id) => mesh.store.name(id)).join(', ')}. Hover a ⚠ row for why; fix, merge or retype them, then export again`);
}

// ── deviation view ────────────────────────────────────────────────────────────

/**
 * The deviation views (settings.deviation; the toolbar's view switch, D steps through them), measured
 * against the export check's bars: past DOUBT.DEV (1e-3 x the part diagonal) it calls a group wrong (94-100%
 * right on Onshape parts); past PREFLIGHT.DEV (1e-5), right about 65% of the time.
 *
 *   'groups'  WHICH groups: each group in one color, its verdict. Red where the export check flags it (off its
 *             surface, a seam gap, or a narrow arc, which no distance shows) or a triangle is past its bar,
 *             yellow past the soft bar, green on its surface. Hover a group for why.
 *   'heat'    WHERE in them: every corner gets its distance from its own group's surface as a decade level
 *             (log10); the renderer interpolates the level across the triangle and maps it through its HEAT
 *             ramp per pixel, so within a group the map is continuous, and a triangle of another face is a
 *             hot spot that cools to its edge on the group's surface.
 *
 * On a CAD tessellation every vertex lies on its face, so a right group sits at float precision (~1e-7, the
 * ramp's cold end) and a triangle of another face stands out by orders of magnitude. A wrong type that still
 * fits (a plane on a few strips of a cylinder) is on its surface: only the narrow-arc flag can catch it.
 */
const DEVIATION_VIEWS = [null, 'groups', 'heat'];
/** The group view's verdicts: the reserved status colors, always named beside them in the legend. */
const VERDICTS = {
  off: { rgb: hexRgb('#d03b3b'), label: 'Likely wrong' },
  maybe: { rgb: hexRgb('#fab219'), label: 'Doubtful' },
  on: { rgb: hexRgb('#0ca30c'), label: 'On its surface' },
  none: { rgb: [0.42, 0.42, 0.42], label: 'No surface (freeform or failed fit)' },
};
const NO_SURFACE = VERDICTS.none.rgb;

function hexRgb(hex) {
  return [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
}
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

/**
 * Recompute the verdict of groups `ids` (every group when null) from their triangles' distances and the export
 * check's flags; returns the ids whose verdict changed.
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
    const verdict = Number.isNaN(w) ? 'none' : flags.has(id) ? 'off' : deviationBand(w);
    if (verdicts.get(id) !== verdict) { verdicts.set(id, verdict); changed.add(id); }
  }
  return changed;
}

/** D: the next deviation view (groups, heat, off). */
function cycleDeviation() {
  setDeviationView(DEVIATION_VIEWS[(DEVIATION_VIEWS.indexOf(settings.deviation) + 1) % DEVIATION_VIEWS.length]);
}

/** Show the group verdicts, the heat map, or neither. Coming from neither, it also runs the export check. */
async function setDeviationView(view) {
  const was = settings.deviation;
  settings.deviation = view;
  for (const v of DEVIATION_VIEWS) viewSwitch.querySelector(`[data-view="${v ?? 'off'}"]`).setAttribute('aria-pressed', v === view ? 'true' : 'false');
  legend.hidden = !view || !mesh;
  renderer.setHeat(view === 'heat');
  if (!mesh) return;
  recolor();
  if (!view) { say('Group colors'); return; }
  drawLegend();
  showProbe(-1);
  say(deviationSummary(), hasBad());
  if (was || stroke || autoRun || !mesh.store.list().length) return;
  const current = mesh;
  await new Promise((resolve) => setTimeout(resolve, 50));   // let the view show: the check takes ~1 s per 55k triangles
  if (mesh !== current || !settings.deviation || stroke || autoRun) return;
  const doubts = exportDoubts(await knitPayload(current));
  if (mesh !== current || !settings.deviation || stroke || autoRun) return;
  current.flags = doubts.groups;
  recolor();                                       // a flagged group is red in the group view
  refreshPanel();
  say(deviationSummary(doubts.groups.size), hasBad());
}

const hasBad = () => [...mesh.verdicts.values()].includes('off');

/** How many groups have each verdict. */
function verdictCounts() {
  const counts = { off: 0, maybe: 0, on: 0, none: 0 };
  for (const v of mesh.verdicts.values()) counts[v]++;
  return counts;
}

/** Count the painted triangles in each band, and the groups holding off and maybe ones. */
function deviationCounts() {
  const { store, deviation, model } = mesh;
  const counts = { on: 0, maybe: 0, off: 0, none: 0, unpainted: 0, offGroups: new Set(), maybeGroups: new Set(), noneGroups: new Set() };
  for (let t = 0; t < model.count; t++) {
    const id = store.owner[t];
    if (id < 0) { counts.unpainted++; continue; }
    const band = deviationBand(deviation[t]);
    counts[band]++;
    if (band !== 'on') counts[`${band}Groups`].add(id);
  }
  return counts;
}

function deviationSummary(flagged = null) {
  const groups = (n) => `${n.toLocaleString()} group${n === 1 ? '' : 's'}`;
  const parts = [];
  if (settings.deviation === 'groups') {
    const v = verdictCounts();
    if (v.off) parts.push(`${groups(v.off)} likely wrong`);
    if (v.maybe) parts.push(`${groups(v.maybe)} doubtful`);
    if (!v.off && !v.maybe) parts.push(v.on ? 'every group with a surface is on it' : 'no group has a surface yet');
    if (v.none) parts.push(`${groups(v.none)} with no surface`);
    parts.push('Heat map shows where');
  } else {
    const c = deviationCounts();
    if (c.off) parts.push(`${c.off.toLocaleString()} triangles in ${groups(c.offGroups.size)} are past the export check's bar`);
    if (c.maybe) parts.push(`${c.maybe.toLocaleString()} triangles in ${groups(c.maybeGroups.size)} are doubtful`);
    if (!c.off && !c.maybe) parts.push(c.on ? 'every painted triangle with a surface is on it' : 'nothing painted has a surface yet');
    if (c.none) parts.push(`${groups(c.noneGroups.size)} have no surface`);
  }
  if (flagged !== null) parts.push(flagged ? `the export check flags ${groups(flagged)} (⚠ in the list)` : 'the export check flags no groups');
  return parts.join(' · ');
}

/** A length for the legend: two figures, in exponent form when small. */
const shortLength = (v) => (v >= 0.01 ? v.toPrecision(2) : v.toExponential(1));

const legendKey = (rgb, label, count) => `<span class="studio-legend__swatch" style="background:rgb(${rgb.map((v) => Math.round(v * 255)).join(',')})"></span>`
  + `<span>${label}</span><span class="studio-legend__value">${count.toLocaleString()}</span>`;

/** Fill the legend in for the view: the verdicts with group counts, or the heat ramp. */
function drawLegend() {
  if (settings.deviation === 'heat') { drawHeatLegend(); return; }
  const v = verdictCounts();
  legendBody.innerHTML = '<div class="studio-legend__title">Will each group come out right? (groups)</div>'
    + `<div class="studio-legend__keys">${Object.entries(VERDICTS).map(([k, b]) => legendKey(b.rgb, b.label, v[k])).join('')}`
    + `${legendKey(UNPAINTED, 'Unpainted (triangles)', mesh.model.count - mesh.store.painted)}</div>`
    + '<div class="studio-legend__counts">Red: the export check flags it, or it holds a triangle past its bar. Heat map: where.</div>';
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
  legendBody.innerHTML = '<div class="studio-legend__title">Distance off the fitted surface</div>'
    + `<div class="studio-legend__scale" style="grid-template-rows:repeat(${stops.length}, 17px)">`
    + `<span class="studio-legend__bar" style="background:linear-gradient(to top, ${gradient})"></span>${ticks}</div>`
    + `<div class="studio-legend__keys">${legendKey(NO_SURFACE, 'No surface (freeform or failed fit)', counts.none)}${legendKey(UNPAINTED, 'Unpainted', counts.unpainted)}</div>`
    + `<div class="studio-legend__counts">Past the bar: ${counts.off.toLocaleString()} triangles${counts.off ? ` in ${groups(counts.offGroups)}` : ''}`
    + `<br>Doubtful: ${counts.maybe.toLocaleString()} triangles${counts.maybe ? ` in ${groups(counts.maybeGroups)}` : ''}</div>`;
  legendDirty = false;
}

/** The legend's reading for triangle t under the cursor (-1: none): the group's verdict and why, or the distance there. */
function showProbe(t) {
  if (!mesh || t < 0) { legendProbe.textContent = `Point at a ${settings.deviation === 'groups' ? 'group' : 'face'} for a reading`; return; }
  const id = mesh.store.owner[t];
  if (id < 0) { legendProbe.textContent = 'Here: unpainted'; return; }
  const name = mesh.store.name(id);
  const diag = 2 * mesh.model.radius;
  if (settings.deviation === 'groups') {
    const verdict = mesh.verdicts.get(id) ?? 'none';
    const why = mesh.flags.get(id);
    const worst = mesh.groupWorst.get(id);
    legendProbe.textContent = why ? `${name}: ${why.join('; ')}`
      : verdict === 'none' ? `${name} has no surface`
        : `${name}: ${VERDICTS[verdict].label.toLowerCase()}, up to ${shortLength(worst * diag)} off`;
    return;
  }
  const d = mesh.deviation[t];
  legendProbe.textContent = Number.isNaN(d)
    ? `Here: ${name} has no surface`
    : `Here: up to ${shortLength(d * diag)} off ${name} (${d.toExponential(0)} of the part)`;
}

/** Fitted surface params moved by `d`: every point-valued param (origin, axisPoint, apex, center). */
function shifted(p, d) {
  const out = { ...p };
  for (const k of ['origin', 'axisPoint', 'apex', 'center']) if (p[k]) out[k] = [p[k][0] + d[0], p[k][1] + d[1], p[k][2] + d[2]];
  return out;
}

// The knitter starts no plain retry or facet tier after EXPORT_DEADLINE_S; a build already running is
// stopped with its worker at EXPORT_HARD_STOP_S. Measured without them: 26 min (Fencepost,
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
      : report.failed
        ? `Exported ${file} as the mesh, a faceted solid: the solid build failed (${report.failed})`
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

// ── repair ────────────────────────────────────────────────────────────────────

// The kernel's complaints about a group's face (knit/build.js patchProblems), as the ⚠ rows say them.
const KERNEL_PROBLEMS = {
  missing: 'the kernel could not build its face',
  invalid: 'its face came out invalid',
  orient: 'its face runs the wrong way against a neighbor',
  area: 'its face came out a quarter or more off the area of its triangles',
  open: 'it leaves a gap in the solid',
};

/** Map group -> reasons from the knitter's problems ({kind: [label]}), each label first through `idOf` (null: skip). */
function kernelFlags(problems, idOf, store) {
  const out = new Map();
  for (const [kind, labels] of Object.entries(problems ?? {})) {
    for (const label of labels) {
      const id = idOf(label);
      if (id == null || !store.groups.has(id)) continue;
      const why = out.get(id) ?? out.set(id, []).get(id);
      const text = KERNEL_PROBLEMS[kind] ?? kind;
      if (!why.includes(text)) why.push(text);
    }
  }
  return out;
}

/**
 * An export with faces off their surfaces: each group the kernel rejected, or the facet tier built as facets,
 * gets a ⚠ with why, and the user picks Repair (in view) or this export as it is. The STEP waits for that.
 */
function offerRepair(current, step, report) {
  const faces = report.faces ?? {};
  const pieces = new Map(faces.pieces ?? []);           // a freeform group's pieces are that group
  const idOf = (label) => pieces.get(label) ?? label;
  const flags = kernelFlags(faces.problems, idOf, current.store);
  if (Array.isArray(faces.demoted)) {
    for (const id of new Set(faces.demoted.map(idOf))) {
      if (!current.store.groups.has(id)) continue;
      (flags.get(id) ?? flags.set(id, []).get(id)).push('it went out as flat facets');
    }
  }
  current.flags = flags;
  current.showFlags = false;
  refreshPanel();
  const n = flags.size;
  const groups = `${n} group${n === 1 ? '' : 's'}`;
  const file = `${current.name}.stp`;
  const pct = (x) => `${(100 * x).toFixed(x < 0.01 ? 1 : 0)}%`;
  const what = report.tier === 'none' ? 'it is not a closed solid'
    : report.tier === 'faceted' ? 'it is the mesh as flat facets'
      : `${pct(report.facetArea)} of the area is flat facets`;
  say(`Not saved yet: ${what}${n ? `, from ${groups} the kernel rejected (⚠ in the list)` : ''}. Repair tries other types and merges on them in view; or save it as it is`, true);
  ui.openMenu([
    { id: 'repair', label: n ? `Repair the ${groups}` : 'Repair the groups', onSelect: () => repairInView() },
    ...(n ? [{ id: 'show', label: `Show the ${groups}`, onSelect: () => showDoubts() }] : []),
    { id: 'save', label: report.tier === 'none' ? 'Save it anyway (not a closed solid)' : 'Save it as it is', onSelect: () => {
      downloadFile(step, file);
      say(exportSummary(file, report, current.model.count - current.store.painted), true);
    } },
  ], { anchor: toolbar.get('export').parentElement, align: 'right' });
}

/**
 * Repair, in view: the repair loop (knit/index.js repairGroups) in a worker, on the groups as they stand. It
 * builds the solid as the export will, and while a group's face fails, tries that group as its next type or
 * merged into its neighbors, keeping a change only when the solid gets better. Each change it keeps lands here
 * at once as its own undo step, so the groups always show what the export will build; the group being tried
 * is outlined. Esc stops it (the worker with it), keeping the changes made so far.
 */
async function repairInView() {
  if (!mesh) return;
  if (stroke || autoRun) { say('Finish or cancel (Esc) the stroke or auto-paint first', true); return; }
  const current = mesh;
  let payload;
  try {
    payload = await knitPayload(current);
  } catch (error) {
    console.error(error);
    say(`Repair failed: ${error.message}`, true);
    return;
  }
  if (mesh !== current || stroke || autoRun) return;
  const known = new Set(payload.labels);         // the groups as sent; any other label is a freeform piece
  const made = new Map();                        // freeform pieces the repair made groups of: label -> id
  const idOf = (label) => (known.has(label) ? label : made.get(label) ?? null);
  const worker = new Worker(new URL('../knit/worker.js', import.meta.url), { type: 'module' });
  const started = performance.now();
  let kept = 0;
  let ended = false;
  const end = (text, bad, result = null) => {
    if (ended) return;
    ended = true;
    worker.terminate();
    autoRun = null;
    current.lastRepair = { ok: !!result?.ok, kept, stopped: !result, secs: Math.round((performance.now() - started) / 100) / 10,
                           moves: result?.moves ?? null };        // for the conversion report
    if (result) { current.flags = kernelFlags(result.problems, idOf, current.store); current.showFlags = false; }
    setActive(null);
    refreshPanel();
    say(text, bad);
  };
  const changes = () => `${kept} change${kept === 1 ? '' : 's'}${kept ? ' (each an undo step)' : ''}`;
  // auto-paint's guard: no strokes, edits or exports while it runs; Esc, which sets stop, ends it at once
  autoRun = {
    what: 'Repairing the groups',
    get stop() { return ended; },
    set stop(value) { if (value) end(`Repair stopped: kept ${changes()}`, false); },
  };
  toolbar.setDisabled('export', 'Repairing the groups (Esc stops it)');
  worker.onmessage = ({ data }) => {
    if (ended) return;
    if (mesh !== current) { end('Repair stopped: another mesh was opened', true); return; }
    if (data.type === 'status') say(`Repair: ${data.text}`);
    else if (data.type === 'progress') {
      if (data.freeform) say(`Repair: fitting freeform surfaces… ${data.freeform} of ${data.of}`);
      else if (data.trying) {
        const move = repairMoveText(current, data.trying, idOf);
        say(`Repair: trying ${data.trying.move === 'retype' ? move : `to ${move}`}… (${kept} kept so far)`);
        const id = data.trying.group !== undefined ? idOf(data.trying.group) : null;
        if (id !== null && current.store.groups.has(id)) setActive(id);
      }
    } else if (data.type === 'move') {
      applyRepairMove(current, data.move, idOf, made);
      kept++;
    } else if (data.type === 'done') {
      const r = data.repair;
      const left = kernelFlags(r.problems, idOf, current.store).size;
      end(r.ok
        ? `Repair: the groups now build a closed solid, every face on its surface · kept ${changes()} · Export to save it`
        : `Repair kept ${changes()}; ${left ? `${left} group${left === 1 ? '' : 's'} still fail (⚠ in the list): fix them by hand, or export and save with them as facets` : 'the solid still does not close'}`,
      !r.ok, r);
    } else end(`Repair failed: ${data.message}`, true);
  };
  worker.onerror = (event) => end(`Repair failed: ${event.message || 'the worker stopped'}`, true);
  say('Repair: building the solid to find what fails…');
  worker.postMessage({ ...payload, repair: true });
}

/** A repair move in words, as its undo step is named: "Cone 12 as a cylinder", "merge Plane 4 into its neighbors". */
function repairMoveText(current, m, idOf) {
  const name = (label) => { const id = idOf(label); return id !== null && current.store.groups.has(id) ? current.store.name(id) : 'a freeform piece'; };
  if (m.move === 'retype') return `${name(m.group)} as a ${typeLabel(m.to).toLowerCase()}`;
  if (m.move === 'absorb') return `merge ${name(m.group)} into its neighbors`;
  return `merge ${m.groups.length} sliver${m.groups.length === 1 ? '' : 's'} into their neighbors`;
}

/**
 * One change the repair kept, as its own undo step: triangles to their new groups (a freeform piece the
 * repair split off becomes a new, untyped group), and a retyped group's new type with the surface the repair
 * built (in the view's coordinates), not a refit, so the export builds what the repair did.
 */
function applyRepairMove(current, m, idOf, made) {
  const { store } = current;
  const label = `Repair: ${repairMoveText(current, m, idOf)}`;
  const touched = new Set();
  store.begin();
  const groupFor = (l) => {
    let id = idOf(l);
    if (id === null) {
      id = store.create(null).id;
      made.set(l, id);
      current.origins.set(id, { by: 'repair', split: true });
    }
    touched.add(id);
    return id;
  };
  for (let i = 0; i < m.changes.length; i += 2) {
    const was = store.owner[m.changes[i]];
    if (was >= 0) touched.add(was);
    store.assign(m.changes[i], groupFor(m.changes[i + 1]));
  }
  if (m.move === 'retype') {
    const id = groupFor(m.group);
    const group = store.groups.get(id);
    group.type = m.to;
    group.fit = { type: m.to, params: shifted(m.params, current.offset.map((v) => -v)) };
    const origin = current.origins.get(id) ?? current.origins.set(id, { by: 'repair' }).get(id);
    (origin.repaired ??= []).push({ from: m.from, to: m.to });
  }
  for (const id of touched) {
    current.flags.delete(id);                     // a changed group is no longer the one the kernel rejected
    if (settings.deviation) current.refitted.add(id);
  }
  store.commit(label);
  flushChanged();
  edgesDirty = true;
  refreshPanel();
}

function downloadFile(data, filename) {
  const url = URL.createObjectURL(new Blob([data], { type: 'application/octet-stream' }));
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
    // ⚠: the export check expects this group to come out wrong (exportDoubts); cleared when it changes
    ...(mesh.flags.has(group.id) ? { value: `⚠ ${store.count(group.id).toLocaleString()}` } : {}),
    tooltip: [describeGroup(group), ...(mesh.flags.get(group.id) ?? []).map((why) => `⚠ ${why}`)].join('\n'),
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
  shell.hint.textContent = `P: auto-paint · D: next view · Shift: facets · Alt: erase · [ ]: brush ${settings.brushPx}px`;
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
    { separator: true },
    {
      id: 'repair', label: 'Repair meshes as they open', checked: settings.repair,
      onSelect: () => {
        settings.repair = !settings.repair;
        say(settings.repair
          ? 'The next mesh opened is repaired: welds, duplicates, cracks, holes'
          : 'The next mesh opened is used as it is');
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
  const centre = mesh ? renderer.pick(point.x, point.y, 0).centre : -1;
  if (settings.deviation) showProbe(centre);
  if (!mesh || strokeMode() !== 'fill') { setHover(null); return; }
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
  else if (key === 'd' && !mod) cycleDeviation();
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
  // The deviation views: a new set of flags (the export check's, or a Repair's kernel rejections) changes
  // verdicts; groups refit since the last frame are recolored in one pass over the mesh.
  if (mesh && settings.deviation && mesh.verdictFlags !== mesh.flags) {
    mesh.verdictFlags = mesh.flags;
    recolor();
  }
  if (mesh?.refitted.size) {
    if (settings.deviation) recolor(trianglesOf(mesh.store.owner, mesh.refitted));
    mesh.refitted.clear();
  }
  if (legendDirty && mesh && settings.deviation && (!autoRun || now - lastEdgeBuild > 500)) drawLegend();
  // While auto-paint runs, edges are rebuilt at most twice a second rather than after every fill.
  if (edgesDirty && mesh && (!autoRun || now - lastEdgeBuild > 500)) {
    lastEdgeBuild = now;
    const { model, store } = mesh;
    renderer.setBands('edges', bodyEdges(model, store.owner));
    // The active group's outline; with none active, the groups the export check flagged when shown (showDoubts).
    const flagged = active === null && mesh.showFlags && mesh.flags.size ? mesh.flags : null;
    renderer.setBands('active', active !== null && store.count(active)
      ? outline(model, store.triangles(active), (t) => store.owner[t] === active)
      : flagged ? outline(model, trianglesOf(store.owner, flagged), (t) => flagged.has(store.owner[t])) : null);
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
    // captureView: the canvas is read now, in the task that drew it, before the browser presents and clears it
    if (viewCapture) {
      const done = viewCapture;
      viewCapture = null;
      canvas.toBlob((blob) => blob.arrayBuffer().then((buffer) => done(new Uint8Array(buffer))), 'image/png');
    }
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
