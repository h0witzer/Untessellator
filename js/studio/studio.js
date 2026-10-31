/**
 * studio.js — the Untessellator studio: paint face groups onto a mesh, see how well each one fits, fix what
 * doesn't, and export a STEP solid. index.html loads it.
 *
 * The page is the studio's own (index.html, workbench.css): a top bar; a tool row with Select, Bucket, Brush and
 * Erase, the surface to paint as (each with its icon, and Skip for what the import mangled), the brush and
 * Auto-paint; Color by over the viewport (groups, type, fit, deviation); and a face-group table with Fix bad areas
 * and an inspector. The viewport's pieces come from the Form Studio kit (vendor/form-studio-ui): the camera, view
 * cube, menus, tooltips, and Onshape's lighting and gestures as measured from the product. The mouse follows it:
 * RIGHT-drag orbits, the wheel zooms, a right click opens the menu for what is under the cursor, Space clears the
 * selection. Where a Part Studio's LEFT button selects, here it paints (with the Select tool, it selects):
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
import { typeRgb } from './palette.js';

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
const FILL_TYPES = [...TYPES, 'freeform', 'skip'];   // skip: mangled import geometry, set aside
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
const TYPE_ICONS = { plane: 'face', cylinder: 'cylinder', cone: 'cone', sphere: 'sphere', torus: 'torus', freeform: 'freeform', skip: 'skip', auto: 'auto' };

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
  // A circle struck through: skipped, set aside.
  skip: '<path d="M10 2a8 8 0 1 0 0 16 8 8 0 1 0 0-16zm0 2a6 6 0 1 1 0 12 6 6 0 1 1 0-12zM6.5 5.1l8.4 8.4-1.4 1.4-8.4-8.4z" fill-rule="evenodd" />',
  // An arrow: select a group.
  select: '<path d="M4 2l11 7-5 1.2L12.6 16l-2 .9-2.6-5.7L4 14z" />',
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
  // Color by (the viewport's switch; C steps through it): 'groups' each its own color, 'type' its surface type,
  // 'fit' the export check's verdict, 'deviation' the heat map. The workbench measures in every view (the table
  // shows each group's fit), so this is never off.
  deviation: 'groups',
  // Every triangle edge drawn over the shading (T), to see the tessellation.
  triangleEdges: false,
};

// ── frame ─────────────────────────────────────────────────────────────────────

// ── the page: colors, icons, and the shell the studio's code talks to ─────────
// index.html is the studio's own page: its top bar, tool row and face-group table around the kit's viewport
// and view cube. `shell` and `toolbar` keep the studio's calls on them working.

const hexRgb = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
const css = (rgb) => `rgb(${rgb.map((v) => Math.round(v * 255)).join(',')})`;
const SURFACE_KINDS = ['plane', 'cylinder', 'cone', 'sphere', 'torus', 'freeform', 'skip'];
// the types' colors (js/studio/palette.js): chosen to stay apart under color blindness, test/palette.mjs checks it
const TYPE_RGB = Object.fromEntries(SURFACE_KINDS.map((t) => [t, typeRgb(t)]));
// skipped triangles look the same in every view: the palette's dark gray, their triangle edges drawn over them
const SKIP_RGB = TYPE_RGB.skip;
/** A type's icon in a color: the panel's group swatch, the group drawn as its type. */
const typeIcon = (type, rgb, size = 16) => `<svg class="wb-ticon" viewBox="0 0 20 20" width="${size}" height="${size}" aria-hidden="true" style="color:${rgb ? css(rgb) : 'currentColor'}">${ui.iconInner(TYPE_ICONS[type ?? 'freeform'])}</svg>`;
const COLOR_MODES = [{ id: 'groups', label: 'Groups' }, { id: 'type', label: 'Type' }, { id: 'fit', label: 'Fit' }, { id: 'deviation', label: 'Deviation' }];

const statusLine = document.getElementById('status');
const shell = {
  stage: document.getElementById('view'),
  hint: document.getElementById('hint'),
  setStatus(text) { document.getElementById('painted').textContent = text; },
  say(text, bad = false) { statusLine.textContent = text; statusLine.classList.toggle('wb-status--bad', bad); },
};
ui.installTooltips(document.body);
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
legend.className = 'wb-legend';
legend.hidden = true;
const legendBody = document.createElement('div');
const legendProbe = document.createElement('div');
legendProbe.className = 'wb-probe';
legend.append(legendBody, legendProbe);
shell.stage.append(legend);

// The progress card, over the viewport while auto-paint, an export or Repair runs: what it is doing, a bar (filled
// to the share done where that is known, moving where it is not), the time so far, and Stop. One run at a time.
const busyEl = document.createElement('div');
busyEl.className = 'wb-busy';
busyEl.setAttribute('role', 'status');
busyEl.hidden = true;
busyEl.innerHTML = `<div class="wb-busy-head"><span class="wb-busy-title"></span><span class="wb-busy-time mono"></span>
  <button type="button" class="wb-btn wb-small" data-act="stop">Stop <kbd>Esc</kbd></button></div>
  <div class="wb-busy-bar"><span></span></div><div class="wb-busy-step"></div><div class="wb-busy-note"></div>`;
shell.stage.append(busyEl);
const busy = {
  onStop: null,
  started: 0,
  timer: 0,
  /** Show the card for a run called `title`; `onStop` is what Stop and Esc do (null: no Stop); `note` a line under the bar. */
  start(title, { onStop, note = '' }) {
    hideResult();
    this.onStop = onStop;
    busyEl.querySelector('[data-act="stop"]').hidden = !onStop;
    this.started = performance.now();
    busyEl.querySelector('.wb-busy-title').textContent = title;
    busyEl.querySelector('.wb-busy-note').textContent = note;
    this.step('', null);
    this.tick();
    clearInterval(this.timer);
    this.timer = setInterval(() => this.tick(), 1000);
    busyEl.hidden = false;
    shell.hint.hidden = true;
  },
  /** What the run is doing now, and the share of it done (0..1), or null when that is not known. */
  step(text, done = null) {
    busyEl.querySelector('.wb-busy-step').textContent = text;
    const bar = busyEl.querySelector('.wb-busy-bar');
    bar.classList.toggle('wb-busy-bar--moving', done === null);
    bar.firstElementChild.style.width = done === null ? '' : `${Math.round(100 * Math.min(1, Math.max(0, done)))}%`;
  },
  tick() {
    const s = Math.floor((performance.now() - this.started) / 1000);
    busyEl.querySelector('.wb-busy-time').textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
  },
  end() {
    clearInterval(this.timer);
    this.onStop = null;
    busyEl.hidden = true;
    shell.hint.hidden = !resultEl.hidden;   // a run that ended in a result card leaves the card up
  },
};
busyEl.addEventListener('click', (event) => { if (event.target.closest('[data-act="stop"]')) busy.onStop?.(); });
/** A run's step, said in both places: the status line and the progress card (which has the run's name above it). */
const progress = (text, done = null) => {
  say(text);
  const step = text.replace(/^(Auto-paint|Repair): /, '');
  busy.step(step[0].toUpperCase() + step.slice(1), done);
};

// The result card, in the progress card's place: what an export check, a build or Repair found, the groups it is
// about (each a link to the group), and what to do next, each step a button with a line on what it does.
const resultEl = document.createElement('div');
resultEl.className = 'wb-result';
resultEl.setAttribute('role', 'dialog');
resultEl.hidden = true;
shell.stage.append(resultEl);
let resultSteps = [];
/**
 * Show the result card: `title`, a `lead` sentence, `groups` ([id, why] pairs, worst first) and `steps`
 * ([{ label, detail, primary?, onSelect }]). `bad` marks it as a problem. A new run or a new mesh closes it.
 */
function showResult({ title, lead = '', groups = [], steps = [], bad = false }) {
  const SHOWN = 6;
  resultSteps = steps;
  resultEl.classList.toggle('wb-result--bad', bad);
  resultEl.setAttribute('aria-label', title);
  const row = ([id, why]) => {
    const g = mesh?.store.groups.get(id);
    if (!g) return '';
    return `<li><button type="button" data-group="${id}" title="Select and zoom to ${esc(mesh.store.name(id))}">${typeIcon(g.type, g.type === 'skip' ? SKIP_RGB : TYPE_RGB[g.type ?? 'freeform'], 14)}<b>${esc(mesh.store.name(id))}</b><span class="wb-rwhy">${esc(why)}</span></button></li>`;
  };
  resultEl.innerHTML = `<div class="wb-rhead"><span class="wb-rtitle">${esc(title)}</span>
      <button type="button" class="wb-iconbtn wb-rclose" data-act="close" aria-label="Close" title="Close (the ⚠ marks stay in the list)">✕</button></div>
    ${lead ? `<p class="wb-rlead">${esc(lead)}</p>` : ''}
    ${groups.length ? `<ul class="wb-rgroups">${groups.slice(0, SHOWN).map(row).join('')}</ul>${groups.length > SHOWN ? `<div class="wb-rmore">and ${groups.length - SHOWN} more, marked ⚠ in the list</div>` : ''}` : ''}
    ${steps.length ? `<div class="wb-rsteps">${steps.map((s, i) => `<div class="wb-rstep"><button type="button" class="wb-btn wb-small${s.primary ? ' wb-primary' : ''}" data-step="${i}">${esc(s.label)}</button><span>${esc(s.detail)}</span></div>`).join('')}</div>` : ''}`;
  resultEl.hidden = false;
  shell.hint.hidden = true;
}
/** Close the result card; a STEP preview lasts as long as its card, so the view goes back to the mesh. */
function hideResult() {
  resultEl.hidden = true;
  if (busyEl.hidden) shell.hint.hidden = false;
  leavePreview();
}

// The STEP preview: after an export builds, its solid (each face as the kernel tessellates it, colored by its surface
// type, its edges drawn) is drawn by a second renderer on a canvas over the mesh's, with the same camera, until it is
// saved or the card is closed. A shutter cuts the two: the input mesh left of it, colored by type too, the STEP right.
const stepCanvas = document.createElement('canvas');
stepCanvas.className = 'wb-stepcanvas';
stepCanvas.hidden = true;
overlay.before(stepCanvas);
/** The STEP's renderer: a WebGL context of its own, made at the first preview and kept. */
let stepRenderer = null;
const shutter = document.createElement('div');
shutter.className = 'wb-shutter';
shutter.hidden = true;
shutter.innerHTML = `<span class="wb-shutter-tag wb-shutter-tag--mesh">Mesh</span><span class="wb-shutter-tag wb-shutter-tag--step">STEP · not saved yet</span>
  <div class="wb-shutter-grip" title="Drag to compare the mesh with the STEP"></div>
  <div class="wb-shutter-handle" role="slider" tabindex="0" aria-label="Shutter between the mesh, left, and the STEP, right" aria-valuemin="0" aria-valuemax="100">◂ ▸</div>`;
shell.stage.append(shutter);
/** The shutter's place, as a share of the view's width from the left: the mesh shows left of it. */
let shutterAt = 0.5;
function setShutter(at) {
  shutterAt = Math.min(1, Math.max(0, at));
  const pct = `${(100 * shutterAt).toFixed(2)}%`;
  shutter.style.left = pct;
  stepCanvas.style.clipPath = `inset(0 0 0 ${pct})`;
  const handle = shutter.querySelector('.wb-shutter-handle');
  handle.setAttribute('aria-valuenow', String(Math.round(100 * shutterAt)));
  handle.setAttribute('aria-valuetext', `${Math.round(100 * shutterAt)}% mesh`);
  // a tag with no room beside the shutter goes, rather than run off the view
  const width = shell.stage.clientWidth;
  shutter.querySelector('.wb-shutter-tag--mesh').hidden = shutterAt * width < 70;
  shutter.querySelector('.wb-shutter-tag--step').hidden = (1 - shutterAt) * width < 170;
}
let shutterDrag = null;
shutter.addEventListener('pointerdown', (event) => {
  if (event.button !== 0 || !event.target.closest('.wb-shutter-grip, .wb-shutter-handle')) return;
  event.preventDefault();
  shutterDrag = event.pointerId;
  event.target.setPointerCapture(event.pointerId);
});
shutter.addEventListener('pointermove', (event) => {
  if (event.pointerId !== shutterDrag) return;
  const rect = shell.stage.getBoundingClientRect();
  setShutter((event.clientX - rect.left) / rect.width);
});
const endShutterDrag = (event) => { if (event.pointerId === shutterDrag) shutterDrag = null; };
shutter.addEventListener('pointerup', endShutterDrag);
shutter.addEventListener('wheel', (event) => { event.preventDefault(); zoomAt(event); }, { passive: false });   // zoom on, over the line
shutter.addEventListener('pointercancel', endShutterDrag);
shutter.addEventListener('keydown', (event) => {
  const step = event.shiftKey ? 0.01 : 0.05;
  const to = { ArrowLeft: shutterAt - step, ArrowRight: shutterAt + step, Home: 0, End: 1 }[event.key];
  if (to === undefined) return;
  event.preventDefault();
  event.stopPropagation();
  setShutter(to);
});
/** The preview on show: { current (the mesh it was built from), types (each face's) }, or null. */
let previewing = null;
resultEl.addEventListener('click', (event) => {
  const el = event.target.closest('[data-act],[data-step],[data-group]');
  if (!el) return;
  if (el.dataset.act === 'close') hideResult();
  else if (el.dataset.group) {
    const id = Number(el.dataset.group);
    if (!mesh?.store.groups.has(id)) return;
    if (!(selection.size === 1 && selection.has(id))) selectGroup(id);   // selectGroup toggles a sole selection off
    zoomTo(new Set([id]));
  } else {
    const step = resultSteps[Number(el.dataset.step)];
    hideResult();
    step?.onSelect();
  }
});
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
      { id: 'triangles', label: 'Triangle edges (T)', checked: settings.triangleEdges, onSelect: () => toggleTriangleEdges() },
      { id: 'smooth', label: 'Smooth shading', checked: settings.smooth, onSelect: () => { settings.smooth = !settings.smooth; renderer.setShading(settings.smooth); requestRender(); } },
      { id: 'occlusion', label: 'Ambient occlusion', checked: settings.occlusion, onSelect: () => { settings.occlusion = !settings.occlusion; requestRender(); } },
    ];
  },
});

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
  <button type="button" class="wb-tool" data-tb="fill-options" title="Fill method, model and options" aria-label="Fill options">${ui.iconMarkup('surface', { className: 'wb-ticon' })}</button>`;
toolRow.addEventListener('click', (event) => {
  const button = event.target.closest('[data-tb]');
  if (!button) return;
  const id = button.dataset.tb;
  if (['select', 'fill', 'raw', 'erase'].includes(id)) setTool(id);
  else if (id.startsWith('type-')) setType(id.slice(5));
  else if (id === 'autopaint') runAutoPaint();
  else if (id === 'fill-options') openFillMenu();
});
// the flag, next to Export STEP: report how this conversion went (report.js)
const reportButton = document.getElementById('report');
reportButton.innerHTML = ui.iconMarkup('flag', { className: 'wb-ticon' });
reportButton.addEventListener('click', () => reportConversion());
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

// ── state ─────────────────────────────────────────────────────────────────────

/** { model, geometry, name, store, colours } for the open mesh. */
let mesh = null;
let active = null;
/** The groups picked in the table or the viewport; `active` is the last one picked. */
let selection = new Set();
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
  get previewing() { return previewing; },
  settings,
  camera,
};

const requestRender = () => { needsRender = true; };
const say = (text, bad = false) => shell.say(text, bad);
const dpr = () => window.devicePixelRatio || 1;

// ── loading ───────────────────────────────────────────────────────────────────

async function openFile(file) {
  if (stroke || autoRun) { say('Finish or cancel (Esc) the stroke or auto-paint before opening another mesh', true); return; }
  if (busy.onStop) { say('Stop the export first (Esc), or let it finish', true); return; }
  busy.start(`Opening ${file.name}`, { onStop: null });
  progress(`Opening ${file.name}…`);
  try {
    const loaded = await loadModelFile(file);
    const raw = loaded.geometry ?? loaded;
    let geometry = raw.index ? raw.toNonIndexed() : raw;
    // The repaired soup replaces the loaded one: everything after (view, painting, export) indexes its
    // triangles. Caps the repair added are marked, so painting and fitting can leave them out (caps.js).
    let repair = null, synthetic = null;
    if (settings.repair) {
      progress(`Opening ${file.name}: checking the mesh…`);
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
    // The geometry kernel starts loading in a worker now, so the first export doesn't wait for it.
    warmKernel();
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
  } finally {
    busy.end();
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
  for (const id of ['select', 'fill', 'raw', 'erase']) toolbar.setPressed(id, id === tool);
  showHeldTool();
  drawCursor();
}

const strokeMode = () => (modifiers.alt ? 'erase' : modifiers.shift ? 'raw' : settings.tool);

/** While Alt holds the eraser or Shift the brush, that button lights up; the chosen tool keeps an outline meanwhile. */
function showHeldTool() {
  const held = modifiers.alt ? 'erase' : modifiers.shift ? 'raw' : null;
  for (const id of ['select', 'fill', 'raw', 'erase']) toolbar.get(id)?.toggleAttribute('data-held', id === held && id !== settings.tool);
  toolRow.classList.toggle('wb-tools--holding', held !== null && held !== settings.tool);
}

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
      if (type === 'freeform' || type === 'skip') out = freeformFill(seed, paintMesh);
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
  if (type === 'freeform' || type === 'skip') return FREEFORM_CREASE;   // a skip fill floods as a freeform one does
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
/** The share of auto-paint's progress bar given to painting; typing and fitting the new groups take the rest. */
const PAINT_SHARE = 0.8;

async function runAutoPaint() {
  if (!mesh) return;
  if (stroke || autoRun) { say('Finish or cancel (Esc) the stroke or auto-paint first', true); return; }
  const current = mesh;
  const { store } = current;
  const started = performance.now();
  const guard = autoRun = { stop: false };
  toolbar.setPressed('autopaint', true);
  setActive(null);
  setHover(null);
  busy.start('Auto-painting', {
    onStop: () => { guard.stop = true; },
    note: 'Your own groups are left alone. Stop keeps the groups found so far.',
  });
  let editOpen = false;
  try {
    progress('Auto-paint: preparing…');
    const { tool, mesh: paintMesh } = await preparePaint();
    progress('Auto-paint: scoring seeds…');
    const caps = current.synthetic;
    const order = seedOrder(await coherenceFor(current), paintMesh.facets).filter((t) => !caps?.[t]);   // a cap seeds nothing
    if (autoRun.stop || mesh !== current) return;
    const extended = MODELS.find((m) => m.url === settings.model)?.features === 'v6';
    const refineUrl = await refineUrlFor(current);
    const created = [];
    const uiClock = { panel: 0, yield: performance.now() };
    // The bar: painting is the first PAINT_SHARE of it, by triangles claimed (autoPaint counts the unpainted,
    // non-degenerate ones); typing the new groups the rest (on a 37,000-triangle part, 122 s and 28 s).
    let free0 = 0;
    for (let t = 0; t < store.owner.length; t++) if (store.owner[t] < 0 && !paintMesh.facets.degenerate[t]) free0++;
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
      onProgress: ({ segments, free }) => progress(`Auto-paint: ${segments} groups · ${free.toLocaleString()} triangles left`,
        PAINT_SHARE * (free0 ? 1 - free / free0 : 1)),
    });
    // Freeform faces back into one group each (js/freeformMerge.js): auto-paint knows only primitives and
    // carves a B-spline face into many. Only this run's groups, and only on a part made mostly of such
    // faces; the joined groups stay untyped, so the export fits them a B-spline. Part of the undo step.
    progress('Auto-paint: joining freeform pieces…', PAINT_SHARE);
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
    progress('Auto-paint: typing groups…', PAINT_SHARE);
    // The learned typer decides where it is confident; it needs groups of 3 or more triangles. A model
    // that fails to load leaves every group to typeFinalGroup, as before.
    const learned = new Map();
    const typer = settings.groupTyper ? await tool.groupTyperFor(GROUP_TYPER).catch(() => null) : null;
    // the bar's typing share: the typer's evidence for each group, then each group's type and fit
    const typingDone = (from, share, k) => PAINT_SHARE + (1 - PAINT_SHARE) * (from + share * k / Math.max(1, byGroup.size));
    if (typer) {
      const rows = [];
      let seen = 0;
      for (const [id, tris] of byGroup) {
        seen++;
        if (freeform.has(id)) continue;
        const f = tool.groupTypeFeatures(tris, paintMesh.facets, paintMesh.adjacency, store.groups.get(id).type);
        if (f) rows.push([id, f]);
        const now = performance.now();
        if (now - uiClock.yield > 40) {
          busy.step(`Weighing each group's type… ${seen} of ${byGroup.size}`, typingDone(0, 0.5, seen));
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
    let typed = 0;
    for (const [id, tris] of byGroup) {
      const group = store.groups.get(id);
      const origin = current.origins.get(id);
      typed++;
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
        busy.step(`Typing and fitting groups… ${typed} of ${byGroup.size}`, typer ? typingDone(0.5, 0.5, typed) : typingDone(0, 1, typed));
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
    busy.end();
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
  if (!group.type || group.type === 'skip' || !store.count(id)) { group.fit = null; return; }   // skipped: never fitted
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
    if (tris && changed.size && settings.deviation === 'fit') tris = new Set([...tris, ...trianglesOf(store.owner, changed)]);
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
  if (previewing) return;                         // the preview holds the renderer; leavePreview uploads it all
  renderer.updateColours(colours, first, last);
  if (settings.deviation === 'deviation') renderer.updateLevels(mesh.levels, first, last);
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
  const group = store.groups.get(id);
  if (group?.type === 'skip') return SKIP_RGB;
  if (settings.deviation === 'fit') return VERDICTS[mesh.verdicts.get(id) ?? 'none'].rgb;
  if (settings.deviation === 'type') return TYPE_RGB[group?.type ?? 'freeform'];
  if (settings.deviation === 'deviation' && Number.isNaN(mesh.deviation[t])) return NO_SURFACE;
  return group?.colour ?? UNPAINTED;
}

// ── groups ────────────────────────────────────────────────────────────────────

function setActive(id) {
  selection = new Set(id === null ? [] : [id]);
  if (active === id) { edgesDirty = true; refreshPanel(); return; }
  active = id;
  if (id !== null && mesh) mesh.showFlags = false;   // selecting a group ends the flagged-groups outline
  edgesDirty = true;
  refreshPanel();
}

/** A change to groups outside a stroke, as one undo step. */
function editGroups(label, change) {
  if (!mesh) return;
  if (stroke || autoRun) { say('Finish or cancel (Esc) the stroke or auto-paint first', true); return; }
  leavePreview();                                 // an edit is made on the groups, so they come back into view
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
  fitAs.push({ id: 'skip', label: 'Skip (set aside)', checked: group.type === 'skip', onSelect: () => retype(id, 'skip') });
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
  leavePreview();
  const label = mesh.store.undo();
  if (!label) return;
  afterHistory();
  say(`Undid: ${label}`);
}

function redo() {
  if (!mesh || stroke || autoRun) return;
  leavePreview();
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
  const { store, name } = current;
  if (!store.list().length) { say('Paint at least one face group first', true); return; }
  toolbar.setDisabled('export', 'Exporting…');
  current.lastExport = null;                      // this attempt's result replaces any earlier one
  // Stop (or Esc) ends the worker at once: a running kernel call cannot be interrupted any other way
  const run = { abort: null };
  busy.start('Exporting STEP', {
    onStop: () => run.abort?.(),
    note: `Most parts take seconds; past ${Math.round(EXPORT_HARD_STOP_S / 60)} minutes the mesh goes out as flat facets instead.`,
  });
  try {
    const payload = await knitPayload(current);
    // Before the build, the groups that will come out wrong (knit/preflight.js), so they can be fixed now
    // rather than after minutes of building.
    if (!checked) {
      progress('Checking the groups…');
      await new Promise((resolve) => setTimeout(resolve, 0));
      const doubts = exportDoubts(payload);
      current.lastCheck = { groups: [...doubts.groups], mesh: doubts.mesh ?? null };   // for the conversion report
      if (doubts.groups.size || doubts.mesh) { busy.end(); offerDoubts(doubts); return; }
    }
    let result;
    try {
      result = await knitInWorker({ ...payload, options: { deadline: EXPORT_DEADLINE_S } }, progress, EXPORT_HARD_STOP_S, run);
    } catch (error) {
      if (error.name === 'AbortError') throw error;
      // A single OCCT repair on badly fitting surfaces can run for most of an hour and cannot be
      // interrupted, so the worker is stopped and a fresh one exports the mesh itself. A build that throws
      // can leave the kernel's memory corrupt (its fallback then fails too: "table index is out of bounds"),
      // so that too gets a fresh worker for the faceted solid.
      const timedOut = error.name === 'TimeoutError';
      say(timedOut
        ? `The surfaces took over ${Math.round(EXPORT_HARD_STOP_S / 60)} minutes to join; exporting the mesh as a faceted solid…`
        : `The solid build failed (${failureText(error.message)}); exporting the mesh as a faceted solid…`, true);
      busy.step(timedOut ? 'The surfaces took too long to join: building the mesh as flat facets…' : 'The build failed: building the mesh as flat facets…');
      result = await knitInWorker({ ...payload, options: { mode: 'faceted' } }, progress, Infinity, run);
      if (timedOut) result.report.timedOut = true;
      else result.report.failed = error.message;
    }
    const { step, report } = result;
    current.lastExport = report;                  // for the conversion report (report.js)
    if (!step) throw new Error('no shape came out of the build');
    busy.end();
    // The solid on show, and the STEP waits for Save: fully analytic, it is ready; with faces off their surfaces,
    // the card says which groups failed and offers Repair or this export as it is; a stopped or failed build
    // went out as the mesh itself, with no verdicts to repair from. Without a preview, an analytic one saves now.
    const shown = enterPreview(current, result);
    if (report.timedOut || report.failed) offerAfterFacets(current, step, report);
    else if (report.tier !== 'analytic') offerRepair(current, step, report);
    else if (shown) offerSave(current, step, report);
    else saveStep(current, step, report, `${name}.stp`);
  } catch (error) {
    if (error.name === 'AbortError') { say('Export stopped: nothing was saved'); return; }
    console.error(error);
    current.lastExport ??= { failed: error.message };
    const why = failureText(error.message);
    say(`Export failed: ${why}`, true);
    showResult({
      title: 'Export failed', bad: true,
      lead: `${why[0].toUpperCase()}${why.slice(1)}. Nothing was saved. Exporting again usually gets past a one-off kernel failure; if it keeps failing, report it with the flag so the run can be replayed.`,
      steps: [
        { label: 'Export again', primary: true, detail: 'Builds the solid from the groups as they are now.', onSelect: () => exportStep({ checked: true }) },
        { label: 'Report it', detail: 'Downloads a report file and opens a prefilled GitHub issue; nothing is sent until you submit it.', onSelect: () => reportConversion() },
      ],
    });
  } finally {
    busy.end();
    refreshPanel();
  }
}

/**
 * A failure in words: OCCT's uncaught WebAssembly exceptions reach the worker as a bare number (an exception
 * pointer, such as "24"), which says nothing to the user. The report keeps the raw message.
 */
const failureText = (message) => (/^\d+$/.test(String(message ?? '').trim()) ? `the geometry kernel crashed, code ${message}` : String(message));

/** A build that ran out of time or failed went out as the mesh: why, what to try, or save it so. */
function offerAfterFacets(current, step, report) {
  const file = `${current.name}.stp`;
  say(`Not saved yet: ${report.timedOut ? 'the surfaces took too long to join' : 'the solid build failed'}, so only the mesh built`, true);
  showResult({
    title: 'Not saved yet: only the mesh built', bad: true,
    lead: report.timedOut
      ? `Joining the fitted surfaces took over ${Math.round(EXPORT_HARD_STOP_S / 60)} minutes, so the file would hold the mesh as flat facets. That usually means some groups sit badly on their surfaces.`
      : `The solid build failed (${failureText(report.failed)}), so the file would hold the mesh as flat facets.`,
    steps: [
      { label: 'Show likely-wrong groups', primary: true, detail: 'Colors the part by fit: red groups are off their surfaces. Repaint or retype them, then export again.', onSelect: () => showNeedsALook() },
      { label: 'Repair automatically', detail: 'Builds the solid and tries other types and merges where faces fail, keeping only what helps; each change is an undo step.', onSelect: () => repairInView() },
      { label: 'Save as flat facets', detail: 'Downloads the mesh as a faceted solid now.', onSelect: () => saveStep(current, step, report, file) },
    ],
  });
}

/** Download an export and say what it holds. */
function saveStep(current, step, report, file) {
  downloadFile(step, file);
  say(exportSummary(file, report, current.model.count - current.store.painted), report.tier !== 'analytic');
}

/** A fully analytic build, on show: save it, or go back to the groups. */
function offerSave(current, step, report) {
  const file = `${current.name}.stp`;
  const counts = new Map();
  for (const t of previewing?.types ?? []) counts.set(t, (counts.get(t) ?? 0) + 1);
  const kinds = SURFACE_KINDS.filter((t) => counts.has(t)).map((t) => `${counts.get(t)} ${typeLabel(t).toLowerCase()}${counts.get(t) === 1 || t === 'freeform' ? '' : 's'}`);
  say(`Ready to save: ${file}, a closed solid with every face on its fitted surface`);
  showResult({
    title: `Ready: ${file}`,
    lead: `A closed solid, every face on its fitted surface${previewing ? `: ${previewing.types.length} faces (${kinds.join(', ')})` : ''}. Turn it to check it, then save.`,
    steps: [
      { label: 'Save STEP', primary: true, detail: `Downloads ${file}.`, onSelect: () => saveStep(current, step, report, file) },
      { label: 'Back to editing', detail: 'Returns to the painted groups; nothing is saved.', onSelect: () => {} },
    ],
  });
}

/**
 * Show export `result`'s solid over the mesh, cut by the shutter; false when the worker sent no preview. The mesh
 * side is colored by type for the while, so a face reads the same color on both sides of the shutter.
 */
function enterPreview(current, result) {
  const p = result.preview;
  if (!p?.faces || mesh !== current) return false;
  try {
    stepRenderer ??= createRenderer(stepCanvas);
  } catch (error) {
    console.warn('no STEP preview:', error.message);
    return false;
  }
  const [ox, oy, oz] = current.offset;                // the export went back where the file had it; the view is centred
  const offset = [ox, oy, oz];
  const positions = Float32Array.from(p.positions, (v, i) => v - offset[i % 3]);
  const edges = Float32Array.from(p.edges, (v, i) => v - offset[i % 3]);
  const count = p.face.length;
  const { types } = p;
  const colours = new Float32Array(count * 9);
  for (let t = 0; t < count; t++) {
    const rgb = TYPE_RGB[types[p.face[t]]];
    for (let k = 0; k < 3; k++) colours.set(rgb, t * 9 + k * 3);
  }
  stepRenderer.setMesh({ count, positions, facetNormals: p.normals, smoothNormals: p.normals });
  stepRenderer.updateColours(colours, 0, count - 1);
  stepRenderer.setBands('edges', edges);
  // the mesh side, by type: its groups' surfaces in the same palette (recolor leaves the renderer alone meanwhile)
  const { store, model } = current;
  const byType = new Float32Array(model.count * 9);
  for (let t = 0; t < model.count; t++) {
    const g = store.groups.get(store.owner[t]);
    const rgb = !g ? UNPAINTED : g.type === 'skip' ? SKIP_RGB : TYPE_RGB[g.type ?? 'freeform'];
    for (let k = 0; k < 3; k++) byType.set(rgb, t * 9 + k * 3);
  }
  setHover(null);
  renderer.updateColours(byType, 0, model.count - 1);
  renderer.setHeat(false);
  renderer.setBands('active', null);
  previewing = { current, types };
  // the legend: the solid's faces by type
  const counts = new Map();
  for (const t of types) counts.set(t, (counts.get(t) ?? 0) + 1);
  legendBody.innerHTML = `<span class="wb-ltitle">STEP preview · ${types.length.toLocaleString()} faces</span>`
    + SURFACE_KINDS.filter((t) => counts.has(t)).map((t) => `<div class="wb-keyrow">${typeIcon(t, TYPE_RGB[t], 14)}<span>${typeLabel(t)}</span><span class="mono">${counts.get(t).toLocaleString()}</span></div>`).join('')
    + '<span class="wb-lnote">Drag the shutter: the mesh left of it, the STEP right. On the STEP, a dense web of edges is flat facets.</span>';
  legendProbe.textContent = '';
  legend.hidden = false;
  shell.stage.classList.add('wb-view--preview');
  stepCanvas.hidden = false;
  shutter.hidden = false;
  setShutter(0.5);
  cursor = null;
  drawCursor();
  requestRender();
  return true;
}

/** Back from the preview to the painted mesh, as it was. */
function leavePreview() {
  if (!previewing) return;
  const { current } = previewing;
  previewing = null;
  shell.stage.classList.remove('wb-view--preview');
  stepCanvas.hidden = true;
  shutter.hidden = true;
  shutterDrag = null;
  if (mesh === current) {
    const { model } = mesh;
    renderer.updateColours(mesh.colours, 0, model.count - 1);
    renderer.updateLevels(mesh.levels, 0, model.count - 1);
    renderer.setHeat(settings.deviation === 'deviation');
  }
  edgesDirty = true;
  legendDirty = true;
  requestRender();
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
  progress(`Fitting ${groups.length} groups…`);
  const surfaces = [];
  for (const group of groups) {
    if (!group.type || group.type === 'skip') continue;   // no surface: facets; skipped: below
    const fit = group.fit?.params ? group.fit : fitRegionAs(group.type, fitTriangles(store.triangles(group.id), synthetic), facets);
    if (fit.params) surfaces.push([group.id, [group.type, fit.params]]);
  }
  // The STEP goes where the file's mesh was: the loader centred it for the view, so the centre goes
  // back on the positions, and on every fitted surface (all are anchored by points, never directions).
  const [ox, oy, oz] = current.offset;
  const positions = Float64Array.from(facets.positions, (v, i) => v + (i % 3 === 0 ? ox : i % 3 === 1 ? oy : oz));
  let labels = Int32Array.from(store.owner);
  // Caps the repair added go to the group around them, and onto its surface (caps.js), however they were painted.
  // Skipped triangles go the same way: a run the painted groups surround joins the group around it, its corners
  // moved onto that surface; a run with nothing painted around it (a stray shell) is dropped.
  const skipped = Uint8Array.from(labels, (id) => (store.groups.get(id)?.type === 'skip' ? 1 : 0));
  const anySkip = skipped.includes(1);
  let caps = anySkip ? Uint8Array.from(skipped, (k, t) => (k || synthetic?.[t] ? 1 : 0)) : synthetic;
  for (const [t, id] of capOwners(labels, caps, model.neighbours)) labels[t] = id;
  // `source` maps a payload triangle back to the mesh's when stray ones were dropped (Repair's moves name triangles)
  let kept = positions, source = null;
  if (anySkip) {
    const keep = [];
    for (let t = 0; t < labels.length; t++) if (!(skipped[t] && store.groups.get(labels[t])?.type === 'skip')) keep.push(t);
    if (keep.length < labels.length) {
      kept = new Float64Array(keep.length * 9);
      keep.forEach((t, i) => kept.set(positions.subarray(t * 9, t * 9 + 9), i * 9));
      labels = Int32Array.from(keep, (t) => labels[t]);
      caps = Uint8Array.from(keep, (t) => caps[t]);
      source = Int32Array.from(keep);
    }
  }
  const shiftedSurfaces = surfaces.map(([id, [type, p]]) => [id, [type, shifted(p, current.offset)]]);
  projectCapVertices(kept, labels, caps, new Map(shiftedSurfaces));
  return { positions: kept, labels, surfaces: shiftedSurfaces, ...(source ? { source } : {}) };
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

/** Say what the check found, mark the groups in the panel, and offer fixes, a look at them, or export anyway. */
function offerDoubts({ groups, mesh: meshDoubt }) {
  mesh.flags = groups;
  mesh.showFlags = false;
  const n = groups.size;
  const them = `${n} group${n > 1 ? 's' : ''}`;
  say([...(n ? [`${them} will likely come out wrong (⚠ in the list)`] : []), ...(meshDoubt ? [meshDoubt] : [])].join(' · '), true);
  refreshPanel();
  showResult({
    title: n ? `Before exporting: ${them} will likely come out wrong` : 'Before exporting: the mesh is open',
    bad: true,
    lead: [
      ...(n ? [`Each sits badly on its surface, so its face would fail or come out as flat facets. Click one to zoom to it.`] : []),
      ...(meshDoubt ? [`${meshDoubt[0].toUpperCase()}${meshDoubt.slice(1)}.`] : []),
    ].join(' '),
    groups: [...groups].map(([id, why]) => [id, why[0]]),
    steps: [
      ...(n ? [
        { label: 'Find fixes', primary: true, detail: 'Looks for re-fills and merges that put these triangles on a surface; review and apply them under Fix bad areas.', onSelect: () => findFixes() },
        { label: 'Fix by hand', detail: 'Outlines them. A group spanning two faces: repaint it with the bucket. A narrow curve: set its type under Fit as, or merge it into its neighbor.', onSelect: () => showDoubts() },
      ] : []),
      { label: 'Export anyway', detail: n ? 'Builds the solid as it is: where these faces fail, you can still save with them as flat facets.' : 'Builds what it can: the file will be an open shell, not a solid.', onSelect: () => exportStep({ checked: true }) },
    ],
  });
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
  leavePreview();
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
  leavePreview();                                 // a group picked from a card or the table is shown on the mesh
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
  leavePreview();
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

/**
 * The next export's or Repair's worker, its kernel already loading (js/knit/worker.js loads it as it starts, about
 * 50 MB the first time). Each worker does one build, since a build that throws can leave the kernel's memory
 * corrupt; a new one warms when a mesh opens and once the last is gone, so one kernel is in memory at a time.
 */
let warmWorker = null;
const KNIT_WORKER = new URL('../knit/worker.js', import.meta.url);
function warmKernel() {
  if (warmWorker) return;
  const worker = new Worker(KNIT_WORKER, { type: 'module' });
  worker.onerror = () => { worker.terminate(); if (warmWorker === worker) warmWorker = null; };
  warmWorker = worker;
}
/** The warm worker for a build, or a cold one when none is warming. */
function takeWorker() {
  const worker = warmWorker ?? new Worker(KNIT_WORKER, { type: 'module' });
  warmWorker = null;
  worker.onerror = null;
  return worker;
}

/**
 * A build in a worker (takeWorker); rejects with TimeoutError after `limitS`. onStatus(text, done) gets each step,
 * with the share done (0..1) where it is known. `run.abort` is set to stop it: the worker goes and the promise
 * rejects with AbortError.
 */
function knitInWorker(payload, onStatus, limitS = Infinity, run = {}) {
  return new Promise((resolve, reject) => {
    const worker = takeWorker();
    let timer = null;
    const finish = () => { clearTimeout(timer); worker.terminate(); warmKernel(); };   // the next build's kernel loads now
    if (Number.isFinite(limitS)) {
      timer = setTimeout(() => {
        finish();
        const error = new Error(`the solid build took longer than ${limitS} s`);
        error.name = 'TimeoutError';
        reject(error);
      }, limitS * 1000);
    }
    run.abort = () => {
      finish();
      const error = new Error('stopped');
      error.name = 'AbortError';
      reject(error);
    };
    worker.onmessage = ({ data }) => {
      if (data.type === 'status') onStatus(data.text);
      else if (data.type === 'progress') {
        if (data.freeform) onStatus(`Fitting freeform surfaces… ${data.freeform} of ${data.of}`, data.freeform / data.of);
        else onStatus(`Building the solid… attempt ${data.builds}`);
      }
      else {
        finish();
        if (data.type === 'done') resolve(data);
        else reject(new Error(data.message));
      }
    };
    worker.onerror = (event) => { finish(); reject(new Error(event.message || 'the export worker failed')); };
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
        ? `Exported ${file} as the mesh, a faceted solid: the solid build failed (${failureText(report.failed)})`
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
  // not the knitter's: offerRepair's, for a group the facet tier built as facets (report.faces.demoted)
  facets: 'it went out as flat facets',
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
      (flags.get(id) ?? flags.set(id, []).get(id)).push(KERNEL_PROBLEMS.facets);
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
  say(`Not saved yet: ${what}${n ? `, from ${groups} the kernel rejected (⚠ in the list)` : ''}`, true);
  const lead = report.tier === 'none'
    ? `The faces would not close into a solid${report.free_edges ? ` (${report.free_edges.toLocaleString()} open edges)` : ''}.`
    : report.tier === 'faceted' ? 'The fitted surfaces would not join, so the file would hold the mesh as flat facets throughout.'
      : `The solid closes, but ${pct(report.facetArea)} of its area would go out as flat facets where faces would not join.`;
  showResult({
    title: n ? `Not saved yet: ${groups} would not build` : `Not saved yet: ${what}`,
    bad: true,
    lead: `${lead}${n ? ' The kernel rejected these groups; click one to zoom to it.' : ''}`,
    groups: [...flags].map(([id, why]) => [id, why.join('; ')]),
    steps: [
      { label: 'Repair automatically', primary: true, detail: 'Tries other surface types and merges on the failing groups, keeping only what makes the solid better; each change is an undo step. Then export again.', onSelect: () => repairInView() },
      ...(n ? [{ label: 'Fix by hand', detail: 'Outlines them: repaint a group that spans two faces, change its type under Fit as, or merge it into a neighbor. Then export again.', onSelect: () => showDoubts() }] : []),
      { label: report.tier === 'none' ? 'Save anyway' : 'Save as it is', detail: report.tier === 'none' ? 'Downloads it now; it will not be a closed solid.' : 'Downloads it now, with those areas as flat facets.', onSelect: () => saveStep(current, step, report, file) },
    ],
  });
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
  const worker = takeWorker();
  const started = performance.now();
  let kept = 0;
  let ended = false;
  const end = (text, bad, result = null) => {
    if (ended) return;
    ended = true;
    worker.terminate();
    warmKernel();                                 // the next build's kernel loads now
    autoRun = null;
    current.lastRepair = { ok: !!result?.ok, kept, stopped: !result, secs: Math.round((performance.now() - started) / 100) / 10,
                           moves: result?.moves ?? null };        // for the conversion report
    if (result) { current.flags = kernelFlags(result.problems, idOf, current.store); current.showFlags = false; }
    busy.end();
    setActive(null);
    refreshPanel();
    say(text, bad);
    if (result && mesh === current) offerAfterRepair(current, result.ok);
  };
  const changes = () => `${kept} change${kept === 1 ? '' : 's'}${kept ? ' (each an undo step)' : ''}`;
  // auto-paint's guard: no strokes, edits or exports while it runs; Esc, which sets stop, ends it at once
  const guard = autoRun = {
    what: 'Repairing the groups',
    get stop() { return ended; },
    set stop(value) { if (value) end(`Repair stopped: kept ${changes()}`, false); },
  };
  busy.start('Repairing the groups', {
    onStop: () => { guard.stop = true; },
    note: 'Each change it keeps lands now as its own undo step; Stop keeps the changes made so far.',
  });
  toolbar.setDisabled('export', 'Repairing the groups (Esc stops it)');
  worker.onmessage = ({ data }) => {
    if (ended) return;
    if (mesh !== current) { end('Repair stopped: another mesh was opened', true); return; }
    if (data.type === 'status') progress(`Repair: ${data.text}`);
    else if (data.type === 'progress') {
      if (data.freeform) progress(`Repair: fitting freeform surfaces… ${data.freeform} of ${data.of}`, data.freeform / data.of);
      else if (data.trying) {
        const move = repairMoveText(current, data.trying, idOf);
        progress(`Repair: trying ${data.trying.move === 'retype' ? move : `to ${move}`}… (${kept} kept so far)`);
        const id = data.trying.group !== undefined ? idOf(data.trying.group) : null;
        if (id !== null && current.store.groups.has(id)) setActive(id);
      }
    } else if (data.type === 'move') {
      const m = payload.source ? { ...data.move, changes: data.move.changes.map((v, i) => (i % 2 ? v : payload.source[v])) } : data.move;
      applyRepairMove(current, m, idOf, made);
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
  progress('Repair: building the solid to find what fails…');
  worker.postMessage({ ...payload, repair: true });
}

/** After a finished Repair: export now, or what still fails and how to fix it by hand. */
function offerAfterRepair(current, ok) {
  const left = [...current.flags].map(([id, why]) => [id, why.join('; ')]);
  const n = left.length;
  if (ok) {
    showResult({
      title: 'Repaired: every face now builds',
      lead: 'The groups build a closed solid with every face on its surface. Export to save it.',
      steps: [{ label: 'Export STEP', primary: true, detail: 'Builds the solid again from the groups as they are now and downloads it.', onSelect: () => exportStep({ checked: true }) }],
    });
    return;
  }
  showResult({
    title: n ? `Repair helped where it could: ${n} group${n === 1 ? '' : 's'} still fail` : 'Repair could not close the solid',
    bad: true,
    lead: n ? 'No other type or merge it tried made these faces build. Click one to zoom to it.' : 'No change it tried made the solid close.',
    groups: left,
    steps: [
      ...(n ? [{ label: 'Fix by hand', primary: true, detail: 'Outlines them: repaint a group that spans two faces, change its type under Fit as, or merge it into a neighbor.', onSelect: () => showDoubts() }] : []),
      { label: 'Export anyway', detail: 'Builds the solid as it is; where faces still fail, you can save with them as flat facets.', onSelect: () => exportStep({ checked: true }) },
    ],
  });
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
  for (const r of rows.values()) {
    r.verdict = mesh.verdicts.get(r.id) ?? 'none';
    r.worst = (mesh.groupWorst.get(r.id) ?? NaN) * diag;
    r.cause = flagCause(r.id);
  }
  return [...rows.values()];
}

/**
 * Why a flag, not the distance, makes group `id` likely wrong, in a word for the Fit column: 'facets' or
 * 'rejected' (the kernel: KERNEL_PROBLEMS), 'narrow arc' or 'seam gap' (the export check); null when its distance is
 * past the bar anyway, or it has no flag. Without it a rejected group reads red beside a worst of ≈0.
 */
function flagCause(id) {
  const why = mesh.flags.get(id);
  if (!why?.length || (mesh.groupWorst.get(id) ?? 0) > DOUBT.DEV) return null;
  if (why.includes(KERNEL_PROBLEMS.facets)) return 'facets';
  const kernel = new Set(Object.values(KERNEL_PROBLEMS));
  if (why.some((w) => kernel.has(w))) return 'rejected';
  if (why.some((w) => w.startsWith('it covers only'))) return 'narrow arc';
  if (why.some((w) => w.startsWith('its border runs'))) return 'seam gap';
  return 'flagged';
}

const RANK = { off: 0, maybe: 1, none: 2, on: 3, skip: 4 };
const order = (x, y) => (x < y ? -1 : x > y ? 1 : 0);
/**
 * How rows `a` and `b` compare on column `key`, ascending. Fit is worst first: by verdict, then by the farthest
 * distance, largest first (no distance last). Two keys, never one number: rank x 1e12 - worst lost every distance
 * under about 5e-4 to rounding, so the groups on their surfaces sorted by name.
 */
function compareRows(a, b, key) {
  if (key === 'fit') {
    const worst = (r) => (Number.isNaN(r.worst) ? -1 : r.worst);
    return RANK[a.verdict] - RANK[b.verdict] || order(worst(b), worst(a));
  }
  if (key === 'name') return a.id - b.id;
  if (key === 'type') return SURFACE_KINDS.indexOf(a.type ?? 'freeform') - SURFACE_KINDS.indexOf(b.type ?? 'freeform');
  return order(a[key], b[key]);   // count, area
}
/** The table's rows as filtered, searched and sorted. */
function visibleRows() {
  if (!mesh) return [];
  const q = table.search.trim().toLowerCase();
  const rows = groupRows().filter((r) => (!table.types.size || table.types.has(r.type ?? 'freeform'))
    && (!table.fits.size || table.fits.has(r.verdict)) && (!q || r.name.toLowerCase().includes(q)));
  const { key, dir } = table.sort;
  rows.sort((a, b) => (compareRows(a, b, key) || a.id - b.id) * dir);
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
      // red for a flag, not the distance: say which, rather than a worst of ≈0 beside the red
      : r.verdict === 'off' && r.cause ? `<span class="wb-dot" style="background:${css(VERDICTS.off.rgb)}"></span><span class="wb-cause">${r.cause}</span>`
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
  // the arrow points the way the numbers run: Fit's first order is worst first, so its numbers fall
  const arrow = (k) => (table.sort.key !== k ? '' : (k === 'fit' ? table.sort.dir < 0 : table.sort.dir > 0) ? ' ▲' : ' ▼');
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
  if (previewing) key = null;                     // the preview's triangles are not the mesh's
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
  if (previewing) return;
  const centre = mesh ? renderer.pick(point.x, point.y, 0).centre : -1;
  if (settings.deviation) showProbe(centre);
  if (!mesh || (strokeMode() !== 'fill' && strokeMode() !== 'select')) { setHover(null); return; }
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
  canvas.classList.toggle('studio-canvas--brush', !previewing && (mode === 'raw' || mode === 'erase'));   // no circle in a preview
  if (!cursor || !mesh || previewing || mode === 'fill' || mode === 'select') return;
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
  if (changed) { showHeldTool(); drawCursor(); if (strokeMode() !== 'fill') setHover(null); }
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
    if (previewing) say('This is the STEP preview: save it from the card, or go back to editing (Esc) to paint');
    else if (strokeMode() === 'select') pickGroup(event);
    else beginStroke(event);
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
    if (press.moved <= 4 && mesh && !previewing) {
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
  else if (event.key === 'Escape') {
    if (autoRun) autoRun.stop = true;
    else if (busy.onStop) busy.onStop();
    else if (previewing) hideResult();
    else cancelStroke();
  }
  else if (event.key === ' ') { event.preventDefault(); setActive(null); }
  else if ((event.key === 'Delete' || event.key === 'Backspace') && active !== null) deleteGroup(active);
  else if (event.key === '[' || event.key === ']') {
    settings.brushPx = Math.min(64, Math.max(3, Math.round(settings.brushPx * (event.key === ']' ? 1.25 : 0.8))));
    refreshPanel();
    drawCursor();
  } else if (key === 'f' && !mod) zoomFit();
  else if (key === 'c' && !mod) cycleColorBy();
  else if (key === 't' && !mod) toggleTriangleEdges();
  else if (key === 'v' && !mod) setTool('select');
  else if (key === 'g' && !mod) setTool('fill');
  else if (key === 'b' && !mod) setTool('raw');
  else if (key === 'e' && !mod) setTool('erase');
  else if (key === 'p' && !mod) runAutoPaint();
  else if (!mod && event.key >= '1' && event.key <= String(FILL_TYPES.length)) setType(FILL_TYPES[Number(event.key) - 1]);
  else if (event.key === 'Alt') event.preventDefault();   // keep Alt from reaching a browser menu
});
window.addEventListener('keyup', (event) => {
  noteModifiers(event);
  if (event.key === 'Alt') event.preventDefault();
});
window.addEventListener('blur', () => { modifiers.shift = false; modifiers.alt = false; showHeldTool(); drawCursor(); });

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
  if (mesh && !previewing && settings.deviation && mesh.verdictFlags !== mesh.flags) {
    mesh.verdictFlags = mesh.flags;
    recolor();
    if (!autoRun) refreshPanel();
  }
  if (mesh?.refitted.size && !previewing) {
    if (settings.deviation) recolor(trianglesOf(mesh.store.owner, mesh.refitted));
    mesh.refitted.clear();
    if (!autoRun) refreshPanel();   // auto-paint redraws the panel on its own clock
  }
  if (legendDirty && mesh && !previewing && settings.deviation && (!autoRun || now - lastEdgeBuild > 500)) drawLegend();
  // While auto-paint runs, edges are rebuilt at most twice a second rather than after every fill.
  if (edgesDirty && mesh && !previewing && (!autoRun || now - lastEdgeBuild > 500)) {
    lastEdgeBuild = now;
    const { model, store } = mesh;
    renderer.setBands('edges', workbenchEdges(model, store));
    // The picked groups' outline; with none, the groups the export check flagged when shown (showDoubts).
    const picked = new Set([...selection].filter((id) => store.count(id)));
    const flagged = !picked.size && mesh.showFlags && mesh.flags.size ? mesh.flags : null;
    const ring = picked.size ? picked : flagged;
    renderer.setBands('active', ring ? outline(model, trianglesOf(store.owner, ring), (t) => ring.has(store.owner[t])) : null);
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
    const look = {
      view, projection, width, height, ratio,
      occlusionPercent: settings.occlusion ? ui.LIGHTING.occlusion.percent : 0,
      occlusionRadius: radius * 0.06,
      edges: settings.edges,
    };
    renderer.draw(look);
    // the STEP preview, through the same camera; the shutter clips its canvas
    if (previewing) {
      if (stepCanvas.width !== width || stepCanvas.height !== height) { stepCanvas.width = width; stepCanvas.height = height; }
      stepRenderer.draw(look);
    }
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
