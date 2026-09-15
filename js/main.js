/**
 * main.js — STL → B-rep STEP converter entry point
 *
 * Workflow:
 *   1. Load STL / OBJ / 3MF  →  Three.js BufferGeometry for preview
 *   2. "Detect Groups"        →  faceGrouper.js (dihedral-angle segmentation)
 *   3. "Fit Surfaces"         →  surfaceFitter.js (plane / cylinder / cone / sphere)
 *   4. "Export STEP"          →  brepBuilder.js  (opencascade.js + STEPControl_Writer)
 */

import * as THREE from 'three';
import {
  initViewer, loadGeometry, setMeshMaterial, setWireframe,
  getCamera, getCurrentMesh, showFaceGroupColors, setViewerTheme,
  setGroupHighlight, setGroupHoverCallback,
  setBrepOverlay, setBrepOverlayVisible,
  setBrepFacesOverlay, setBrepFacesVisible,
  pickTriangle, setExclusionOverlay, setHoverPreview,
} from './viewer.js';
import { loadModelFile, computeBounds, getTriangleCount } from './stlLoader.js';
import { t, initLang, setLang, getLang, applyTranslations } from './i18n.js';
import { groupFaces } from './faceGrouper.js';
import { fitAllGroups, classifyGroupAs, classifyGroupAsGauss,
         fitAllGroupsGauss, fitAllGroupsLegacy } from './surfaceFitter.js';
import { initOC, buildAndExportSTEP, downloadSTEP, BUILD_VERSION,
         buildGroupAdjacencyMap, computeAnalyticalBoundaries } from './brepBuilder.js';
import { buildBrepOverlay, buildAnalyticalFacesMesh } from './brepVisualizer.js';

// ── State ─────────────────────────────────────────────────────────────────────

let currentGeometry = null;
let currentStlName  = 'model';
let currentGroups   = null;   // array of { triangleIndices, surface? }
let isBusy          = false;

// Reverse map: triangleIndex → groupIndex (rebuilt after each Detect Groups run)
let _triangleToGroup = null;
// The currently highlighted group index (-1 = none)
let _highlightedGroup = -1;

// Paint (experimental): prepared once per mesh on the first paint click
let paintMesh      = null;
let paintSelection = new Set();
let paintBusy      = false;

// ── DOM refs ─────────────────────────────────────────────────────────────────

const canvas        = document.getElementById('viewport');
const dropZone      = document.getElementById('drop-zone');
const dropHint      = document.getElementById('drop-hint');
const stlFileInput  = document.getElementById('stl-file-input');
const meshInfo      = document.getElementById('mesh-info');
const wireframeToggle    = document.getElementById('wireframe-toggle');
const brepOverlayToggle  = document.getElementById('brep-overlay-toggle');
const brepFacesToggle    = document.getElementById('brep-faces-toggle');

// Face detection panel
const creaseSlider  = document.getElementById('crease-angle');
const creaseVal     = document.getElementById('crease-angle-val');
const detectBtn     = document.getElementById('detect-btn');
const groupCount    = document.getElementById('group-count');

// Surface fitting panel
const fitBtn        = document.getElementById('fit-btn');
const classifierMode = document.getElementById('classifier-mode');
const classifierDiff = document.getElementById('classifier-diff');
const surfaceList   = document.getElementById('surface-list');

// Paint panel (experimental)
const paintToggle    = document.getElementById('paint-toggle');
const paintMethod    = document.getElementById('paint-method');
const paintType      = document.getElementById('paint-type');
const paintModel     = document.getElementById('paint-model');
const paintShowModel = document.getElementById('paint-show-model');
const paintClearBtn  = document.getElementById('paint-clear');
const paintStatus    = document.getElementById('paint-status');

// Export panel
const schemaSelect  = document.getElementById('step-schema');
const sewTolSlider  = document.getElementById('sew-tolerance');
const sewTolVal     = document.getElementById('sew-tolerance-val');
const exportBtn     = document.getElementById('export-btn');
const exportProgress = document.getElementById('export-progress');
const exportProgBar  = document.getElementById('export-progress-bar');
const exportProgLbl  = document.getElementById('export-progress-label');

// ── Init ──────────────────────────────────────────────────────────────────────

// Stamp the build version badge in the header
(function () {
  const badge = document.getElementById('build-badge');
  if (badge) badge.textContent = BUILD_VERSION;
})();

initViewer(canvas);

// Wire up the canvas hover callback: maps triangle index → group index → highlight
setGroupHoverCallback((triangleIdx) => {
  if (!_triangleToGroup || !currentGroups) return;
  const groupIdx = triangleIdx >= 0 ? _triangleToGroup[triangleIdx] : -1;
  if (groupIdx === _highlightedGroup) return;
  _highlightedGroup = groupIdx;
  // Update viewport overlay
  if (groupIdx >= 0 && currentGroups[groupIdx]) {
    setGroupHighlight(currentGroups[groupIdx].triangleIndices, currentGeometry);
  } else {
    setGroupHighlight(null, null);
  }
  // Sync the sidebar row highlight
  document.querySelectorAll('.surface-row').forEach((row, i) => {
    row.classList.toggle('viewport-hover', i === groupIdx);
  });
  // Scroll the highlighted row into view if it's offscreen
  if (groupIdx >= 0) {
    const rows = document.querySelectorAll('.surface-row');
    if (rows[groupIdx]) rows[groupIdx].scrollIntoView({ block: 'nearest' });
  }
});

// Apply saved theme
setViewerTheme(document.documentElement.getAttribute('data-theme') === 'light');

// Language
initLang();
(function () {
  const lang = getLang();
  document.querySelectorAll('.lang-btn').forEach(btn =>
    btn.classList.toggle('active', btn.dataset.langCode === lang));
})();

// Theme toggle
document.getElementById('theme-toggle').addEventListener('click', () => {
  const isLight = document.documentElement.getAttribute('data-theme') !== 'light';
  document.documentElement.setAttribute('data-theme', isLight ? 'light' : 'dark');
  localStorage.setItem('stlt-theme', isLight ? 'light' : 'dark');
  setViewerTheme(isLight);
});

// Language buttons
document.querySelectorAll('.lang-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const lang = btn.dataset.langCode;
    setLang(lang);
    document.querySelectorAll('.lang-btn').forEach(b =>
      b.classList.toggle('active', b.dataset.langCode === lang));
  });
});

wireEvents();

// ── Model loading ─────────────────────────────────────────────────────────────

async function handleModelFile(file) {
  if (isBusy) return;
  currentStlName = file.name.replace(/\.[^.]+$/, '');
  try {
    const { geometry } = await loadModelFile(file);
    currentGeometry = geometry;
    currentGroups = null;
    clearPaint();
    paintMesh = null;

    loadGeometry(geometry);
    dropHint.classList.add('loaded');   // hide hint via CSS (does not affect the file input)

    const triCount = getTriangleCount(geometry);
    geometry.computeBoundingBox();
    const box  = geometry.boundingBox;
    const size = new THREE.Vector3();
    box.getSize(size);
    meshInfo.textContent =
      `${triCount.toLocaleString()} triangles · ` +
      `${size.x.toFixed(1)} × ${size.y.toFixed(1)} × ${size.z.toFixed(1)} mm`;

    resetPanels();
    detectBtn.disabled = false;
    exportBtn.disabled = true;
  } catch (err) {
    alert(`Could not load file: ${err.message}`);
  }
}

function resetPanels() {
  groupCount.textContent = '';
  surfaceList.innerHTML  = '';
  exportProgress.classList.add('hidden');
  fitBtn.disabled = true;
  exportBtn.disabled = true;
  // Clear any lingering hover state
  setGroupHighlight(null, null);
  _triangleToGroup = null;
  _highlightedGroup = -1;
}

// Build a fast reverse-lookup array: triangleIndex → groupIndex.
// Called after groupFaces() so the canvas hover can identify the hovered group.
function _buildTriangleGroupMap() {
  if (!currentGeometry || !currentGroups) return;
  const triCount = currentGeometry.attributes.position.count / 3;
  _triangleToGroup = new Int32Array(triCount).fill(-1);
  currentGroups.forEach((g, gi) => {
    for (const t of g.triangleIndices) _triangleToGroup[t] = gi;
  });
}

// ── Face detection ────────────────────────────────────────────────────────────

async function handleDetect() {
  if (!currentGeometry || isBusy) return;
  setBusy(true, detectBtn);

  await yieldToUI();
  const creaseAngle = parseFloat(creaseSlider.value);

  try {
    currentGroups = groupFaces(currentGeometry, creaseAngle);
    _buildTriangleGroupMap();
    showFaceGroupColors(currentGeometry, currentGroups);
    groupCount.textContent = `${currentGroups.length} group${currentGroups.length !== 1 ? 's' : ''} detected`;
    fitBtn.disabled = false;
    exportBtn.disabled = true;
    surfaceList.innerHTML = '';
    // Clear any stale B-rep overlays from a previous Fit Surfaces run
    setBrepOverlay(null);
    setBrepFacesOverlay(null);
    if (brepOverlayToggle) brepOverlayToggle.checked = false;
  } catch (err) {
    console.error('Face detection failed:', err);
    alert(`Face detection failed: ${err.message}`);
  } finally {
    setBusy(false, detectBtn);
  }
}

// ── Surface fitting ───────────────────────────────────────────────────────────

async function handleFit() {
  if (!currentGroups || isBusy) return;
  setBusy(true, fitBtn);

  await yieldToUI();

  try {
    // Run BOTH classifiers and stash each result, so switching between them is
    // instant and every row can show what the other one said.
    fitAllGroupsLegacy(currentGroups, currentGeometry, '_legacy');
    fitAllGroupsGauss(currentGroups, currentGeometry, '_gauss');

    applyClassifier();
    exportBtn.disabled = false;
  } catch (err) {
    console.error('Surface fitting failed:', err);
    alert(`Surface fitting failed: ${err.message}`);
  } finally {
    setBusy(false, fitBtn);
  }
}

/** Which stashed classification is live, per the Classifier dropdown. */
function activeClassifierKey() {
  return (classifierMode?.value === 'legacy') ? '_legacy' : '_gauss';
}

/**
 * Point every group at the selected classifier's result, then re-render the
 * sidebar and rebuild the viewport overlays. Cheap — no refitting.
 */
function applyClassifier() {
  if (!currentGroups) return;
  const key = activeClassifierKey();
  for (const g of currentGroups) g.surface = g[key] ?? g.surface;

  renderSurfaceList();
  renderClassifierDiff();
  rebuildFitOverlays();
}

/** Summarise how many groups the two classifiers disagree about. */
function renderClassifierDiff() {
  if (!classifierDiff || !currentGroups) return;
  const changed = currentGroups.filter(
    g => g._legacy && g._gauss && g._legacy.type !== g._gauss.type).length;
  const active = activeClassifierKey() === '_gauss' ? 'Gauss map' : 'Legacy';

  if (!changed) {
    classifierDiff.textContent = `${active} — both classifiers agree on all ${currentGroups.length} groups.`;
    classifierDiff.classList.remove('diff-hot');
    return;
  }
  classifierDiff.textContent =
    `${active} — the two classifiers disagree on ${changed} of ${currentGroups.length} groups.`;
  classifierDiff.classList.add('diff-hot');
}

/** Rebuild the analytic wireframe + face overlays from the live classification. */
function rebuildFitOverlays() {
  // Compute analytical boundaries (pure JS — no OCCT needed)
  const { boundaries } = computeAnalyticalBoundaries(currentGroups, currentGeometry);

  // Wireframe shape indicator overlay (always built first)
  const overlay = buildBrepOverlay(currentGroups, currentGeometry, TYPE_COLORS);
  setBrepOverlay(overlay);
  if (brepOverlayToggle) {
    brepOverlayToggle.checked = true;
    setBrepOverlayVisible(true);
  }

  // Analytical face-mesh preview (semi-transparent solid shapes)
  const facesOverlay = buildAnalyticalFacesMesh(
    currentGroups, currentGeometry, boundaries, TYPE_COLORS,
  );
  setBrepFacesOverlay(facesOverlay);
  setBrepFacesVisible(true);
}

const TYPE_LABELS = { plane: 'Plane', cylinder: 'Cylinder', cone: 'Cone', sphere: 'Sphere', nurbs: 'NURBS' };
const TYPE_COLORS = { plane: '#4a9eff', cylinder: '#ff9a3c', cone: '#f97316', sphere: '#7dd67d', nurbs: '#c084fc' };

/** Human-readable defining parameter for a fitted surface, e.g. " 45.0°" or " r=4.00". */
function describeSurface(surface) {
  const p = surface?.params;
  if (!p) return '';
  if (surface.type === 'cone' && typeof p.halfAngle === 'number')
    return ` ${(p.halfAngle * 180 / Math.PI).toFixed(1)}°`;
  if ((surface.type === 'cylinder' || surface.type === 'sphere') && typeof p.radius === 'number')
    return ` r=${p.radius.toFixed(2)}`;
  return '';
}

/** Re-render just the summary chips at the top of the surface list. */
function updateChips() {
  const container = surfaceList?.querySelector('.surface-chips');
  if (!container) return;
  const tally = {};
  for (const g of currentGroups) {
    const type = g.surface?.type ?? 'unknown';
    tally[type] = (tally[type] || 0) + 1;
  }
  container.innerHTML = '';
  for (const [type, count] of Object.entries(tally)) {
    const chip = document.createElement('span');
    chip.className = 'surface-chip';
    chip.style.setProperty('--chip-color', TYPE_COLORS[type] ?? '#aaa');
    chip.textContent = `${count} × ${TYPE_LABELS[type] ?? type}`;
    container.appendChild(chip);
  }
}

function renderSurfaceList() {
  surfaceList.innerHTML = '';

  // Summary chips (rendered via shared helper so updateChips() can refresh them)
  const chips = document.createElement('div');
  chips.className = 'surface-chips';
  surfaceList.appendChild(chips);
  updateChips();

  // Per-group rows (cap at 200 to avoid DOM overload)
  const MAX_ROWS = 200;
  const shown = currentGroups.slice(0, MAX_ROWS);
  shown.forEach((g, i) => {
    const type  = g.surface?.type ?? 'unknown';
    const label = TYPE_LABELS[type] ?? type;
    const color = TYPE_COLORS[type] ?? '#aaa';
    const rms   = g.surface?.rms ?? 0;

    const row = document.createElement('div');
    row.className = 'surface-row';

    const dot = document.createElement('span');
    dot.className = 'surface-dot';
    dot.style.background = color;

    const name = document.createElement('span');
    name.className = 'surface-name';
    name.textContent = `Group ${i + 1} — ${label}${describeSurface(g.surface)}`;

    // When the two classifiers disagree about this group, say what the other
    // one called it — this is the whole point of the A/B.
    const other = activeClassifierKey() === '_gauss' ? g._legacy : g._gauss;
    if (other && g.surface && other.type !== g.surface.type) {
      const was = document.createElement('span');
      was.className = 'surface-rms';
      was.style.color = '#f97316';
      was.textContent = `was ${TYPE_LABELS[other.type] ?? other.type}`;
      name.appendChild(document.createTextNode('  '));
      name.appendChild(was);
    }

    const rmsEl = document.createElement('span');
    rmsEl.className = 'surface-rms';
    rmsEl.textContent = isFinite(rms) && rms > 0
      ? `RMS ${rms < 0.001 ? rms.toExponential(2) : rms.toFixed(4)}`
      : '';

    // Manual override select
    const sel = document.createElement('select');
    sel.className = 'surface-override';
    ['auto', 'plane', 'cylinder', 'cone', 'sphere', 'nurbs'].forEach(opt => {
      const o = document.createElement('option');
      o.value = opt; o.textContent = opt === 'auto' ? 'auto' : TYPE_LABELS[opt];
      if ((type === opt) || (opt === 'auto' && !g._override)) o.selected = true;
      sel.appendChild(o);
    });
    sel.addEventListener('change', () => {
      const val = sel.value;
      if (val === 'auto') {
        delete g._override;
        g.surface = g._autoSurface;
      } else {
        g._autoSurface = g._autoSurface ?? g.surface;
        g._override    = val;
        // Best-fit the requested primitive. This never fabricates parameters —
        // if the surface genuinely cannot be fitted it comes back flagged
        // degenerate with a reason, and we say so rather than drawing a shape
        // unrelated to the selected faces.
        g.surface = classifyGroupAsGauss(val, g, currentGeometry);
      }
      // Refresh the row's RMS display
      const newRms = g.surface?.rms ?? 0;
      rmsEl.textContent = isFinite(newRms) && newRms > 0
        ? `RMS ${newRms < 0.001 ? newRms.toExponential(2) : newRms.toFixed(4)}`
        : '';
      // Update the colour dot and label for this row
      const newColor = TYPE_COLORS[g.surface?.type] ?? '#aaa';
      dot.style.background = newColor;
      // A degenerate forced fit is stored as NURBS but still labelled with the
      // type the user asked for, so the row reflects their selection.
      const shownType = g.surface?.degenerate
        ? (g.surface.requestedType ?? g.surface.type)
        : g.surface?.type;
      name.textContent =
        `Group ${i + 1} — ${TYPE_LABELS[shownType] ?? shownType}` +
        describeSurface(g.surface);

      // A forced type that cannot be fitted is reported, not faked.
      row.classList.toggle('fit-degenerate', !!g.surface?.degenerate);
      if (g.surface?.degenerate) {
        rmsEl.textContent = 'cannot fit';
        rmsEl.style.color = '#f97316';
        row.title = g.surface.reason || 'This surface type cannot be fitted to these faces.';
      } else {
        rmsEl.style.color = '';
        row.title = '';
      }
      // Re-render the tally chips at the top of the surface list to reflect
      // the updated type counts across all groups.
      updateChips();
      // Rebuild both B-rep overlays to reflect the new surface type
      const overlay = buildBrepOverlay(currentGroups, currentGeometry, TYPE_COLORS);
      setBrepOverlay(overlay);
      if (brepOverlayToggle?.checked) setBrepOverlayVisible(true);
      // Rebuild face-mesh preview
      const { boundaries } = computeAnalyticalBoundaries(currentGroups, currentGeometry);
      const facesOverlay = buildAnalyticalFacesMesh(
        currentGroups, currentGeometry, boundaries, TYPE_COLORS,
      );
      setBrepFacesOverlay(facesOverlay);
      setBrepFacesVisible(true);
    });
    // Remember the auto classification
    g._autoSurface = g._autoSurface ?? g.surface;

    row.append(dot, name, rmsEl, sel);

    // Sidebar → viewport hover: highlight this group in the 3D viewer
    row.addEventListener('mouseenter', () => {
      setGroupHighlight(g.triangleIndices, currentGeometry);
      _highlightedGroup = i;
      row.classList.add('sidebar-hover');
    });
    row.addEventListener('mouseleave', () => {
      setGroupHighlight(null, null);
      _highlightedGroup = -1;
      row.classList.remove('sidebar-hover');
    });

    surfaceList.appendChild(row);
  });

  if (currentGroups.length > MAX_ROWS) {
    const more = document.createElement('p');
    more.className = 'surface-more';
    more.textContent = `… and ${currentGroups.length - MAX_ROWS} more groups`;
    surfaceList.appendChild(more);
  }
}

// ── Paint (experimental) ──────────────────────────────────────────────────────

/** Overlay geometry for a set of triangle indices of the loaded mesh. */
function trianglesToGeometry(geometry, triangles) {
  const pos = geometry.attributes.position;
  const out = new Float32Array(triangles.size * 9);
  let i = 0;
  for (const t of triangles) {
    for (let v = 0; v < 3; v++) {
      out[i++] = pos.getX(t * 3 + v);
      out[i++] = pos.getY(t * 3 + v);
      out[i++] = pos.getZ(t * 3 + v);
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(out, 3));
  geo.computeVertexNormals();
  return geo;
}

function clearPaint() {
  paintSelection = new Set();
  setExclusionOverlay(null);
  setHoverPreview(null);
  if (paintStatus) paintStatus.textContent = '';
}

/** One-line description of a fitted surface for the paint status. */
function describePaintFit(type, fit) {
  const p = fit?.params;
  if (!p) return fit?.reason ? `fit failed: ${fit.reason}` : '';
  const res = Number.isFinite(fit.relResidual) ? `, residual ${fit.relResidual.toExponential(1)}` : '';
  if (type === 'cylinder') return `r = ${p.radius.toFixed(3)}${res}`;
  if (type === 'sphere') return `r = ${p.radius.toFixed(3)}${res}`;
  if (type === 'cone') return `half-angle ${(p.halfAngle / DEG_RAD).toFixed(2)}°${res}`;
  if (type === 'torus') return `R = ${p.majorRadius.toFixed(3)}, r = ${p.minorRadius.toFixed(3)}${res}`;
  return res.replace(/^, /, '');
}
const DEG_RAD = Math.PI / 180;

async function handlePaintClick(e) {
  if (paintBusy || !currentGeometry) return;
  const tri = pickTriangle(e.clientX, e.clientY);
  if (tri < 0) return;
  paintBusy = true;
  try {
    const tool = await import('./paintTool.js');
    if (!paintMesh) {
      paintStatus.textContent = 'Preparing mesh…';
      await yieldToUI();
      paintMesh = tool.preparePaintMesh(currentGeometry);
    }
    paintStatus.textContent = 'Filling…';
    await yieldToUI();
    const r = await tool.paintAt(tri, paintMesh, {
      method: paintMethod.value, type: paintType.value, modelUrl: paintModel.value,
    });

    paintSelection = e.shiftKey ? new Set([...paintSelection, ...r.region]) : r.region;
    setExclusionOverlay(trianglesToGeometry(currentGeometry, paintSelection), 0xff6600, 0.8);
    setHoverPreview(paintShowModel.checked && r.modelRegion
      ? trianglesToGeometry(currentGeometry, r.modelRegion) : null);

    const lines = [];
    if (r.predicted) lines.push(`Model says: ${r.predicted} (${Math.round(100 * r.confidence)}%)`);
    lines.push(`Filled ${r.region.size.toLocaleString()} triangles as ${r.fitter ?? 'model region'}` +
      (e.shiftKey ? ` · selection ${paintSelection.size.toLocaleString()}` : ''));
    const fitText = r.fitter ? describePaintFit(r.fitter, r.fit) : '';
    if (fitText) lines.push(fitText);
    lines.push(`${r.calls} model call${r.calls === 1 ? '' : 's'} · ${Math.round(r.ms)} ms`);
    lines.push(...r.notes);
    paintStatus.textContent = lines.join('\n');
  } catch (err) {
    console.error('Paint failed:', err);
    paintStatus.textContent = `Paint failed: ${err.message}`;
  } finally {
    paintBusy = false;
  }
}

// ── STEP export ───────────────────────────────────────────────────────────────

async function handleExport() {
  if (!currentGroups || isBusy) return;
  const hasSurfaces = currentGroups.some(g => g.surface);
  if (!hasSurfaces) {
    alert('Run "Fit Surfaces" before exporting.');
    return;
  }

  setBusy(true, exportBtn);
  exportProgress.classList.remove('hidden');
  setProgress('', 0);

  try {
    const schema = schemaSelect.value;
    const sewTol = parseSewTol();

    const stepContent = await buildAndExportSTEP(
      currentGroups,
      currentGeometry,
      { schema, sewTol: sewTol || 1e-5 },
      (msg, pct) => setProgress(msg, pct),
    );

    downloadSTEP(stepContent, `${currentStlName}.stp`);
  } catch (err) {
    console.error('Export failed:', err);
    alert(`Export failed: ${err.message}`);
  } finally {
    setBusy(false, exportBtn);
    setTimeout(() => exportProgress.classList.add('hidden'), 2000);
  }
}

function parseSewTol() {
  const v = parseFloat(sewTolSlider.value);
  return isFinite(v) ? Math.pow(10, v) : 1e-4;
}

function setProgress(msg, pct) {
  exportProgBar.style.width = `${Math.max(0, Math.min(100, pct))}%`;
  exportProgLbl.textContent = msg;
}

// ── Event wiring ──────────────────────────────────────────────────────────────

function wireEvents() {
  // Model loading
  stlFileInput.addEventListener('change', e => {
    if (e.target.files[0]) handleModelFile(e.target.files[0]);
  });

  // Drag & drop
  dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault(); dropZone.classList.remove('drag-over');
    const file = [...e.dataTransfer.files].find(f => /\.(stl|obj|3mf)$/i.test(f.name));
    if (file) handleModelFile(file);
  });

  // Clicking the viewport (canvas) or the drop-zone background opens the file picker.
  // The canvas is positioned absolute on top of the drop-zone, so we listen on the
  // canvas directly.  We suppress the click during OrbitControls drags by checking
  // that the pointer hasn't moved more than a few pixels between pointerdown and click.
  let _pointerDownPos = null;
  canvas.addEventListener('pointerdown', e => {
    _pointerDownPos = { x: e.clientX, y: e.clientY };
  });
  canvas.addEventListener('click', e => {
    if (_pointerDownPos) {
      const dx = e.clientX - _pointerDownPos.x;
      const dy = e.clientY - _pointerDownPos.y;
      const genuine = dx * dx + dy * dy < 25;   // < 5px movement = a click, not an orbit drag
      if (genuine && !currentGeometry) stlFileInput.click();
      else if (genuine && paintToggle?.checked) handlePaintClick(e);
    }
    _pointerDownPos = null;
  });

  // Paint (experimental)
  paintClearBtn?.addEventListener('click', clearPaint);
  paintShowModel?.addEventListener('change', () => { if (!paintShowModel.checked) setHoverPreview(null); });

  // Crease angle slider
  creaseSlider.addEventListener('input', () => {
    creaseVal.value = creaseSlider.value;
  });
  creaseVal.addEventListener('change', () => {
    creaseSlider.value = Math.max(0, Math.min(90, parseFloat(creaseVal.value) || 30));
    creaseVal.value = creaseSlider.value;
  });

  // Action buttons
  detectBtn.addEventListener('click', handleDetect);
  fitBtn.addEventListener('click', handleFit);
  classifierMode?.addEventListener('change', applyClassifier);
  exportBtn.addEventListener('click', handleExport);

  // Sewing tolerance display
  sewTolSlider.addEventListener('input', () => {
    const exp = parseFloat(sewTolSlider.value);
    sewTolVal.textContent = `1e${exp}`;
  });

  // Wireframe
  wireframeToggle.addEventListener('change', () => setWireframe(wireframeToggle.checked));

  // B-rep wireframe overlay
  if (brepOverlayToggle) {
    brepOverlayToggle.addEventListener('change', () => setBrepOverlayVisible(brepOverlayToggle.checked));
  }

  // Analytical face-mesh preview toggle
  if (brepFacesToggle) {
    brepFacesToggle.addEventListener('change', () => setBrepFacesVisible(brepFacesToggle.checked));
  }

  // License overlay
  const licenseLink    = document.getElementById('license-link');
  const licenseOverlay = document.getElementById('license-overlay');
  const licenseClose   = document.getElementById('license-close');
  if (licenseLink) {
    licenseLink.addEventListener('click', () => licenseOverlay.classList.remove('hidden'));
    licenseClose.addEventListener('click', () => licenseOverlay.classList.add('hidden'));
    licenseOverlay.addEventListener('click', e => {
      if (e.target === licenseOverlay) licenseOverlay.classList.add('hidden');
    });
  }
}

// ── Utilities ─────────────────────────────────────────────────────────────────

function setBusy(busy, primaryBtn) {
  isBusy = busy;
  if (primaryBtn) primaryBtn.disabled = busy;
}

function yieldToUI() {
  return new Promise(resolve => setTimeout(resolve, 10));
}
