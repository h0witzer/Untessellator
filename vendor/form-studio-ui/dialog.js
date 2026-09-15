// SPDX-License-Identifier: LGPL-2.1-or-later
//
// A docked dialog, the shape the product gives its feature dialogs and its
// Graphics preferences.
//
// MEASURED 2026-09-11 by `tools/graphics-prefs-extract.mjs` and
// `tools/dialog-extract.mjs`, with Graphics preferences open in a live Part
// Studio:
//
//     box        300 wide, docked at the canvas's top-left; fill rgba(51,51,51,0.95);
//                shadow 0 5px 10px rgba(0,0,0,0.4); 12px / 21.6px text
//     header     29px, fill #333, 1px #666 below; title 500 16px; then two 28 x 28
//                buttons: a green (#009400) check that is 0.2 opaque until there is
//                something to commit, and a red cross
//     tabs       27px, flex; 500 12px, padding 2px 0; the active tab is #90cef1
//                over a 1px #3e94ff rule, the others white over 1px #d7d7d7
//     content    margin 4px 8px 0
//     section    a 25px toggle row (padding 4px 0) in #999 at 13.33px: a 13px
//                chevron, a 6px gap, the title; its rows indented 20px beside a
//                1px guide
//     slider row 42px (5px below): the label at 12px, the value right-aligned in
//                a 22px underlined field, a 2px track with a 12px thumb; a 14px
//                info glyph at the far right
//     colour row 26px: the label and a 96 x 26 swatch, radius 4, 1px #ccc
//     footer     20px, 8px below: "Reset to default" as a ghost button (1px #aaa,
//                padding 0 10px) and a 14px help glyph
//
// `createDialog` builds any dialog of that shape from a spec;
// `openGraphicsPreferences` is the product's own, wired to a `prefs` object the
// renderer reads (see render.js and viewport.js LIGHTING).

import { LIGHTING, PART_COLOR } from './viewport.js';

const CHEVRON = '<svg viewBox="0 0 20 20" width="13" height="13" aria-hidden="true"><path d="M3.5 7 10 13.5 16.5 7l-1.4-1.4L10 10.7 4.9 5.6z" fill="currentColor"/></svg>';
const CHECK = '<svg viewBox="0 0 24 24" width="24" height="24" aria-hidden="true"><path d="M4 12.5l1.8-1.8 4 4L18.2 6.3 20 8.1 9.8 18.3z" fill="currentColor"/></svg>';
const CROSS = '<svg viewBox="0 0 24 24" width="24" height="24" aria-hidden="true"><path d="M6.3 4.9 12 10.6l5.7-5.7 1.4 1.4L13.4 12l5.7 5.7-1.4 1.4L12 13.4l-5.7 5.7-1.4-1.4L10.6 12 4.9 6.3z" fill="currentColor"/></svg>';
const INFO = '<svg viewBox="0 0 14 14" width="14" height="14" aria-hidden="true"><circle cx="7" cy="7" r="6.2" fill="none" stroke="currentColor" stroke-width="1"/><rect x="6.3" y="6" width="1.4" height="4.5" fill="currentColor"/><rect x="6.3" y="3.5" width="1.4" height="1.4" fill="currentColor"/></svg>';
const HELP = '<svg viewBox="0 0 14 14" width="14" height="14" aria-hidden="true"><circle cx="7" cy="7" r="7" fill="currentColor"/><path d="M5 5.4c0-1.2.9-2 2.1-2s2 .8 2 1.8c0 .8-.5 1.2-1.1 1.6-.5.3-.7.6-.7 1.1v.4H6.2v-.5c0-.9.4-1.4 1-1.8.5-.3.7-.5.7-.9 0-.4-.3-.7-.8-.7s-.9.3-.9 1zm1.2 5.1h1.6v1.5H6.2z" fill="#333"/></svg>';

/**
 * Build a docked dialog.
 *
 * @param {HTMLElement} host   the stage; the dialog docks at its top-left
 * @param {{ title: string,
 *           tabs: Array<{ label: string, sections: Array<{ title: string, rows: Array<Row> }> }>,
 *           footer?: { reset?: { label?: string, onReset: () => void, enabled?: () => boolean }, help?: string },
 *           onCommit?: () => void, onCancel?: () => void, onClose?: () => void }} spec
 *   Row: { kind: 'slider', id, label, value, min, max, step, unit, decimals?, info?, onInput(value) }
 *      | { kind: 'color', id, label, value, info?, onInput(hex) }
 *      | { kind: 'check', id, label, value, info?, onInput(on) }
 * @returns {{ element: HTMLElement, close: () => void, refresh: () => void, set: (id: string, value) => void }}
 */
export function createDialog(host, spec) {
  const dialog = document.createElement('div');
  dialog.className = 'os-dialog';
  dialog.setAttribute('role', 'dialog');
  dialog.setAttribute('aria-label', spec.title);

  const header = document.createElement('div');
  header.className = 'os-dialog-header';
  const title = document.createElement('div');
  title.className = 'os-dialog-title';
  title.textContent = spec.title;
  const buttons = document.createElement('div');
  buttons.className = 'os-dialog-header-buttons';
  const commit = document.createElement('button');
  commit.type = 'button';
  commit.className = 'os-dialog-header-button os-dialog-header-button--commit';
  commit.innerHTML = CHECK;
  commit.setAttribute('aria-label', 'Accept');
  const cancel = document.createElement('button');
  cancel.type = 'button';
  cancel.className = 'os-dialog-header-button os-dialog-header-button--cancel';
  cancel.innerHTML = CROSS;
  cancel.setAttribute('aria-label', 'Close');
  buttons.append(commit, cancel);
  header.append(title, buttons);

  const tabs = document.createElement('nav');
  tabs.className = 'os-dialog-tabs';
  const panes = document.createElement('div');
  panes.className = 'os-dialog-content';
  const controls = new Map();
  const refreshers = [];

  spec.tabs.forEach((tab, index) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = `os-dialog-tab${index === 0 ? ' os-dialog-tab--active' : ''}`;
    button.textContent = tab.label;
    const pane = document.createElement('div');
    pane.className = 'os-dialog-pane';
    pane.hidden = index !== 0;
    button.addEventListener('click', () => {
      for (const other of tabs.children) other.classList.remove('os-dialog-tab--active');
      for (const otherPane of panes.children) otherPane.hidden = true;
      button.classList.add('os-dialog-tab--active');
      pane.hidden = false;
    });
    tabs.append(button);
    for (const section of tab.sections) {
      const block = document.createElement('section');
      block.className = 'os-dialog-section';
      const toggle = document.createElement('button');
      toggle.type = 'button';
      toggle.className = 'os-dialog-section-toggle';
      toggle.innerHTML = `<span class="os-dialog-chevron">${CHEVRON}</span><span>${section.title}</span>`;
      toggle.setAttribute('aria-expanded', 'true');
      const body = document.createElement('div');
      body.className = 'os-dialog-section-body';
      toggle.addEventListener('click', () => {
        const open = body.hidden;
        body.hidden = !open;
        toggle.setAttribute('aria-expanded', String(open));
        toggle.classList.toggle('os-dialog-section-toggle--collapsed', !open);
      });
      for (const row of section.rows) body.append(buildRow(row, controls, refreshers));
      block.append(toggle, body);
      pane.append(block);
    }
    panes.append(pane);
  });

  const footer = document.createElement('div');
  footer.className = 'os-dialog-footer';
  let resetButton = null;
  if (spec.footer?.reset) {
    resetButton = document.createElement('button');
    resetButton.type = 'button';
    resetButton.className = 'os-dialog-ghost-button';
    resetButton.textContent = spec.footer.reset.label ?? 'Reset to default';
    resetButton.addEventListener('click', () => {
      spec.footer.reset.onReset();
      api.refresh();
    });
    footer.append(resetButton);
  }
  if (spec.footer?.help) {
    const help = document.createElement('span');
    help.className = 'os-dialog-help';
    help.innerHTML = HELP;
    help.setAttribute('data-tooltip', spec.footer.help);
    footer.append(help);
  }

  dialog.append(header, tabs, panes, footer);
  host.append(dialog);

  const close = () => {
    if (!dialog.isConnected) return;
    dialog.remove();
    document.removeEventListener('keydown', onKey, true);
    spec.onClose?.();
  };
  const onKey = (event) => {
    if (event.key === 'Escape') {
      spec.onCancel?.();
      close();
    }
  };
  document.addEventListener('keydown', onKey, true);
  commit.addEventListener('click', () => {
    spec.onCommit?.();
    close();
  });
  cancel.addEventListener('click', () => {
    spec.onCancel?.();
    close();
  });
  // The dialog owns its pointer events: a press inside must not reach the stage.
  for (const type of ['pointerdown', 'pointerup', 'wheel', 'contextmenu']) dialog.addEventListener(type, (event) => event.stopPropagation());

  const api = {
    element: dialog,
    close,
    refresh() {
      for (const refresh of refreshers) refresh();
      if (resetButton && spec.footer.reset.enabled) resetButton.disabled = !spec.footer.reset.enabled();
    },
    set(id, value) {
      controls.get(id)?.set(value);
    },
  };
  api.refresh();
  return api;
}

function buildRow(row, controls, refreshers) {
  const container = document.createElement('div');
  container.className = `os-dialog-row os-dialog-row--${row.kind}`;
  const main = document.createElement('div');
  main.className = 'os-dialog-row-main';
  const info = document.createElement('span');
  info.className = 'os-dialog-info';
  info.innerHTML = INFO;
  if (row.info) info.setAttribute('data-tooltip', row.info);
  else info.style.visibility = 'hidden';

  if (row.kind === 'slider') {
    const decimals = row.decimals ?? (Number.isInteger(row.step) ? 0 : 1);
    const format = (v) => `${Number(v).toFixed(decimals).replace(/\.0+$/, '')}${row.unit ? ` ${row.unit}` : ''}`;
    const line = document.createElement('div');
    line.className = 'os-dialog-row-line';
    const label = document.createElement('span');
    label.className = 'os-dialog-label';
    label.textContent = row.label;
    const field = document.createElement('input');
    field.type = 'text';
    field.className = 'os-dialog-value';
    field.value = format(row.value);
    const slider = document.createElement('input');
    slider.type = 'range';
    slider.className = 'os-dialog-slider';
    slider.min = String(row.min);
    slider.max = String(row.max);
    slider.step = String(row.step ?? 1);
    slider.value = String(row.value);
    const apply = (raw, fromField) => {
      let value = Number.parseFloat(String(raw).replace(/[^0-9.\-]/g, ''));
      if (!Number.isFinite(value)) value = row.value;
      value = Math.max(row.min, Math.min(row.max, value));
      row.value = value;
      if (!fromField) field.value = format(value);
      slider.value = String(value);
      row.onInput?.(value);
    };
    slider.addEventListener('input', () => apply(slider.value, false));
    field.addEventListener('change', () => apply(field.value, false));
    field.addEventListener('keydown', (event) => { if (event.key === 'Enter') field.blur(); });
    line.append(label, field);
    main.append(line, slider);
    controls.set(row.id, { set: (value) => apply(value, false) });
    refreshers.push(() => { field.value = format(row.value); slider.value = String(row.value); });
  } else if (row.kind === 'color') {
    const line = document.createElement('div');
    line.className = 'os-dialog-row-line';
    const label = document.createElement('span');
    label.className = 'os-dialog-label';
    label.textContent = row.label;
    const swatch = document.createElement('input');
    swatch.type = 'color';
    swatch.className = 'os-dialog-swatch';
    swatch.value = row.value;
    swatch.addEventListener('input', () => { row.value = swatch.value; row.onInput?.(swatch.value); });
    line.append(label, swatch);
    main.append(line);
    controls.set(row.id, { set: (value) => { swatch.value = value; row.value = value; } });
    refreshers.push(() => { swatch.value = row.value; });
  } else if (row.kind === 'check') {
    const line = document.createElement('label');
    line.className = 'os-dialog-row-line os-dialog-check';
    const box = document.createElement('input');
    box.type = 'checkbox';
    box.checked = !!row.value;
    box.addEventListener('change', () => { row.value = box.checked; row.onInput?.(box.checked); });
    const label = document.createElement('span');
    label.className = 'os-dialog-label';
    label.textContent = row.label;
    line.append(box, label);
    main.append(line);
    controls.set(row.id, { set: (value) => { box.checked = !!value; row.value = !!value; } });
    refreshers.push(() => { box.checked = !!row.value; });
  }
  container.append(main, info);
  return container;
}

/** The defaults the product's dialog showed with "Reset to default" disabled. */
export const GRAPHICS_DEFAULTS = Object.freeze({
  ambientOcclusion: LIGHTING.occlusion.percent,
  fieldOfView: LIGHTING.fieldOfViewDegrees,
  lineWidth: 1,
  shininessPercent: LIGHTING.shininessPercent,
  specular: '#999999',
  /** Their menu shows "Perspective view" unchecked: orthographic is the default. */
  perspective: false,
  partColor: PART_COLOR.hex,
});

/**
 * The product's Graphics preferences, docked on `stage`, editing `prefs` in
 * place as the user drags. Rows are the ones this renderer honours: ambient
 * occlusion, field of view, shininess and the specular colour; the part colour
 * is this kit's addition, under Shaded bodies.
 *
 * @returns {{ element, close, refresh }}
 */
export function openGraphicsPreferences(stage, prefs, { onChange, onClose } = {}) {
  const changed = () => onChange?.(prefs);
  const differs = () => ['ambientOcclusion', 'fieldOfView', 'shininessPercent', 'specular', 'partColor']
    .some((key) => String(prefs[key]).toLowerCase() !== String(GRAPHICS_DEFAULTS[key]).toLowerCase());
  return createDialog(stage, {
    title: 'Graphics preferences',
    tabs: [{
      label: 'General',
      sections: [
        {
          title: 'Viewport',
          rows: [
            { kind: 'slider', id: 'ambientOcclusion', label: 'Ambient occlusion', value: prefs.ambientOcclusion, min: 0, max: 100, step: 0.5, unit: '%', decimals: 1,
              info: 'How much a screen-space occlusion pass darkens creases and contact. Theirs is 37.5 %.',
              onInput: (v) => { prefs.ambientOcclusion = v; changed(); } },
            { kind: 'slider', id: 'fieldOfView', label: 'Field of view', value: prefs.fieldOfView, min: 10, max: 120, step: 1, unit: 'deg',
              info: 'The vertical angle of the perspective view. Theirs is 45 deg.',
              onInput: (v) => { prefs.fieldOfView = v; changed(); } },
          ],
        },
        {
          title: 'Shaded bodies',
          rows: [
            { kind: 'slider', id: 'shininessPercent', label: 'Shininess', value: prefs.shininessPercent, min: 0, max: 100, step: 1, unit: '%',
              info: 'The specular exponent, as a percentage: 3 % is an exponent of 30, theirs.',
              onInput: (v) => { prefs.shininessPercent = v; changed(); } },
            { kind: 'color', id: 'specular', label: 'Specular color', value: prefs.specular,
              info: 'The colour of the highlight. Theirs is #999999.',
              onInput: (v) => { prefs.specular = v; changed(); } },
            { kind: 'color', id: 'partColor', label: 'Part color', value: prefs.partColor,
              info: 'The body’s appearance. The API reported #9dcfed for the probe part.',
              onInput: (v) => { prefs.partColor = v; changed(); } },
          ],
        },
      ],
    }],
    footer: {
      reset: {
        onReset: () => { Object.assign(prefs, GRAPHICS_DEFAULTS, { perspective: prefs.perspective }); changed(); },
        enabled: differs,
      },
      help: 'Every default here was read from the product’s own dialog and shader.',
    },
    onClose,
  });
}
