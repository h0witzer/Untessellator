// SPDX-License-Identifier: LGPL-2.1-or-later
//
// The left panel — a filter, section headers, list rows, a split to a second
// pane — as the product builds a Part Studio's and an Assembly's.
//
// MEASURED 2026-09-11 (tools/chrome-extract.mjs) and 2026-09-13
// (tools/panel-extract.mjs) on a live Part Studio and Assembly:
//
//     panel          200px wide, fill #333 — the SAME as the canvas (the 40px rail
//                    beside it is #2c2c2c); a 1px #666 splitter, then the canvas
//     filter row     37px: padding-top 7; a 30 x 30 funnel button in #90cef1,
//                    margin 0 3px; a 30px `os-field` (1px #888, radius 3, padding
//                    4px 22px 4px 5px, italic placeholder); the Assembly adds a
//                    30 x 30 list-view button in the same blue after the field
//     section header 30px row, padding-right 10: title at weight 500,
//                    13.33px / 30px, padding-left 4; then 30 x 30 icon buttons,
//                    padding 0 5px, which go to opacity 0.5 and 30% white when
//                    there is nothing to act on
//     list row       26px, flex, centred, padding-right 10; a 20 x 20 icon box
//                    margin-left 3 holding a 17 x 17 glyph; the name at
//                    13.33px / 20px, margin 0 0 1px 3px, ellipsised
//     group row      a 20px expander cell (a #999 chevron, 20px svg padded 4)
//                    then the name at x+3; its children indented 19px
//     dimmed         default geometry: icon and name at opacity 0.5; a part's
//                    icon at 0.8
//     hover          #0a4780         selected   rgb(19,78,150) #134e96
//     initials icon  a 17 x 15 chip, 1px #999 border, radius 2, 10px text
//     rollback bar   after the feature list: 6px tall, #b8b8b8, inset 7px
//     split          the panel is TWO panes: Features above, Parts below, with a
//                    1px #666 horizontal splitter between; the lower pane's
//                    heading ("Parts (1)") is itself a group row, not a header
//     handle         a 33 x 34 button at the panel's right edge, mid-height,
//                    1px #aaa on its top, right and bottom, that collapses the
//                    panel (shell.js builds it)
//
// A row's `value` (right-aligned secondary text) is this kit's addition for
// panels that show counts; the product's rows have none.

import { iconMarkup } from './icons.js';

/** The funnel their filter row starts with, and the list-view glyph the Assembly adds. */
const FUNNEL_SVG = '<svg class="os-icon" viewBox="0 0 20 20" aria-hidden="true"><path d="M2 3h16l-6 7v6l-4 2v-8z" fill="currentColor"/></svg>';
const CHEVRON_SVG = '<svg class="os-icon os-list-chevron" viewBox="0 0 20 20" aria-hidden="true"><path d="M3.5 7 10 13.5 16.5 7l-1.4-1.4L10 10.7 4.9 5.6z" fill="currentColor"/></svg>';

/**
 * Build a panel into `host`.
 *
 * @param {HTMLElement} host  gets `os-panel-body`
 * @param {{ filter?: { placeholder?: string, onInput?: (text: string) => void,
 *                      buttons?: Array<{ icon, title, onClick }> } | null,
 *           split?: boolean }} options
 *   `split` makes two panes with a horizontal splitter, the way a Part Studio
 *   has Features above Parts; `addSection(spec, { pane: 'bottom' })` puts a
 *   section below it.
 * @returns {{ element, filter: HTMLInputElement|null, footer,
 *             addSection: (spec, where?) => Section, setFooter: (...nodes) => void }}
 *
 * A Section has `setTitle(text)`, `setRows(rows)`, `setButtonDisabled(id, on)` and
 * `element`. Rows:
 *   { id, icon?, initials?, label, value?, dim?, iconOpacity?, selected?, tooltip?,
 *     onClick?, onContextMenu?, children?: rows[], expanded?: boolean }
 */
export function createPanel(host, options = {}) {
  host.classList.add('os-panel-body');
  let filter = null;
  if (options.filter !== null) {
    const row = document.createElement('div');
    row.className = 'os-panel-filter';
    const funnel = document.createElement('button');
    funnel.type = 'button';
    funnel.className = 'os-panel-filter-button';
    funnel.innerHTML = FUNNEL_SVG;
    funnel.setAttribute('aria-label', 'Filter');
    funnel.setAttribute('data-tooltip', 'Filter');
    const field = document.createElement('input');
    field.type = 'search';
    field.className = 'os-field os-field--search';
    field.placeholder = options.filter?.placeholder ?? 'Filter by name or type';
    field.addEventListener('input', () => options.filter?.onInput?.(field.value));
    row.append(funnel, field);
    for (const button of options.filter?.buttons ?? []) {
      const control = document.createElement('button');
      control.type = 'button';
      control.className = 'os-panel-filter-button';
      control.innerHTML = iconMarkup(button.icon, { className: 'os-icon' });
      if (button.title) control.setAttribute('data-tooltip', button.title);
      control.setAttribute('aria-label', button.title ?? '');
      if (button.onClick) control.addEventListener('click', (event) => button.onClick(event));
      row.append(control);
    }
    host.append(row);
    filter = field;
  }

  const top = document.createElement('div');
  top.className = 'os-panel-scroll os-panel-pane os-panel-pane--top';
  host.append(top);
  let bottom = null;
  if (options.split) {
    const splitter = document.createElement('div');
    splitter.className = 'os-panel-splitter';
    bottom = document.createElement('div');
    bottom.className = 'os-panel-scroll os-panel-pane os-panel-pane--bottom';
    host.append(splitter, bottom);
  }
  const footer = document.createElement('div');
  footer.className = 'os-panel-footer';
  footer.hidden = true;
  host.append(footer);

  const renderRows = (container, rows, depth) => {
    container.innerHTML = '';
    for (const row of rows) {
      const item = document.createElement('div');
      item.className = `os-list-item${row.dim ? ' os-list-item--dim' : ''}${row.children ? ' os-list-item--group' : ''}`;
      item.setAttribute('role', 'treeitem');
      if (row.id !== undefined) item.dataset.id = String(row.id);
      if (row.selected) item.setAttribute('aria-selected', 'true');
      if (row.tooltip) item.setAttribute('data-tooltip', row.tooltip);

      let children = null;
      if (row.children) {
        // A group: the 20px expander cell, no icon cell.
        const expander = document.createElement('span');
        expander.className = 'os-list-expander';
        expander.innerHTML = CHEVRON_SVG;
        item.append(expander);
        children = document.createElement('div');
        children.className = 'os-list-children';
        children.hidden = row.expanded === false;
        item.classList.toggle('os-list-item--collapsed', row.expanded === false);
        item.setAttribute('aria-expanded', String(row.expanded !== false));
        expander.addEventListener('click', (event) => {
          event.stopPropagation();
          const open = children.hidden;
          children.hidden = !open;
          item.classList.toggle('os-list-item--collapsed', !open);
          item.setAttribute('aria-expanded', String(open));
        });
      } else {
        const icon = document.createElement('span');
        icon.className = 'os-list-item-icon';
        if (row.initials) icon.innerHTML = `<span class="os-tool-initials">${row.initials}</span>`;
        else if (row.icon) icon.innerHTML = iconMarkup(row.icon, { className: 'os-icon' });
        if (row.iconOpacity !== undefined) icon.style.opacity = String(row.iconOpacity);
        item.append(icon);
      }
      const name = document.createElement('span');
      name.className = 'os-list-item-name';
      name.textContent = row.label ?? '';
      item.append(name);
      if (row.value !== undefined) {
        const value = document.createElement('span');
        value.className = 'os-list-item-value';
        value.textContent = String(row.value);
        item.append(value);
      }
      if (row.onClick) item.addEventListener('click', (event) => row.onClick(row, event));
      if (row.onContextMenu) item.addEventListener('contextmenu', (event) => { event.preventDefault(); row.onContextMenu(row, event); });
      container.append(item);
      if (children) {
        renderRows(children, row.children, depth + 1);
        container.append(children);
      }
    }
  };

  return {
    element: host,
    filter,
    footer,
    /** Show the footer (fields, a commit readout) below the panes. */
    setFooter(...nodes) {
      footer.innerHTML = '';
      footer.append(...nodes);
      footer.hidden = nodes.length === 0;
    },
    /**
     * A section. In the top pane it gets the Part Studio's header row (title
     * at 500 with icon buttons); in the bottom pane it is a group row, the way
     * "Parts (1)" is. `spec.rollback` adds the rollback bar after its rows.
     */
    addSection(spec, { pane = 'top' } = {}) {
      const target = pane === 'bottom' && bottom ? bottom : top;
      const section = document.createElement('section');
      section.className = `os-panel-section os-panel-section--${pane}`;
      if (spec.id) section.dataset.id = spec.id;
      const list = document.createElement('div');
      list.className = 'os-list';
      list.setAttribute('role', 'tree');
      const buttons = new Map();
      let title;
      if (pane === 'bottom' && bottom) {
        const header = document.createElement('div');
        header.className = 'os-list-item os-list-item--group os-panel-group-header';
        const expander = document.createElement('span');
        expander.className = 'os-list-expander';
        expander.innerHTML = CHEVRON_SVG;
        title = document.createElement('span');
        title.className = 'os-list-item-name';
        title.textContent = spec.title ?? '';
        header.append(expander, title);
        header.addEventListener('click', () => {
          list.hidden = !list.hidden;
          header.classList.toggle('os-list-item--collapsed', list.hidden);
        });
        section.append(header, list);
      } else {
        const header = document.createElement('div');
        header.className = 'os-panel-section-header';
        title = document.createElement('span');
        title.className = 'os-panel-section-title';
        title.textContent = spec.title ?? '';
        header.append(title);
        for (const button of spec.buttons ?? []) {
          const control = document.createElement('button');
          control.type = 'button';
          control.className = 'os-panel-header-button';
          control.innerHTML = iconMarkup(button.icon, { className: 'os-icon' });
          if (button.title) control.setAttribute('data-tooltip', button.title);
          control.setAttribute('aria-label', button.title ?? button.id ?? '');
          control.disabled = !!button.disabled;
          if (button.onClick) control.addEventListener('click', (event) => button.onClick(event));
          header.append(control);
          if (button.id) buttons.set(button.id, control);
        }
        section.append(header, list);
      }
      let rollback = null;
      if (spec.rollback) {
        rollback = document.createElement('div');
        rollback.className = 'os-rollback-bar';
        rollback.setAttribute('data-tooltip', 'Rollback bar');
        section.append(rollback);
      }
      target.append(section);
      return {
        element: section,
        rollback,
        setTitle(text) {
          title.textContent = text;
        },
        setRows(rows) {
          renderRows(list, rows, 0);
        },
        setButtonDisabled(id, on) {
          const control = buttons.get(id);
          if (control) control.disabled = !!on;
        },
      };
    },
  };
}
