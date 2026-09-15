// SPDX-License-Identifier: LGPL-2.1-or-later
//
// The feature toolbar, as the product builds it.
//
// MEASURED 2026-09-11 by `tools/chrome-extract.mjs` on a live Part Studio's
// toolbar, tool by tool and state by state:
//
//     bar            36px tall, padding 0 4px 0 0, the primary background, no
//                    border below it
//     tool           32 x 32, margin 1px, padding 6px, radius 3, no fill at rest;
//                    a 20 x 20 icon in the middle
//     hover          fill #666
//     pressed        fill #666 and an INSET shadow 0 1px 1px rgba(0,0,0,.247)
//     disabled       opacity 0.5, cursor not-allowed
//     split tool     a 50 x 32 group: the 32 tool with radius 3 0 0 3, then an
//                    18 x 32 caret button, padding 0 4px, radius 0 3 3 0,
//                    holding a 10 x 10 chevron
//     labelled       icon then text at 14px (Sketch is 88px wide); one per bar
//     separator      2px wide, 26px tall, margin 5px 0, fill #666 — not a 1px
//                    border-coloured rule
//     search         a 160 x 32 button at the right, fill #555, radius 3,
//                    padding 5px 10px, margin-left 10, holding key chips:
//                    11px text, 1px border #aaa with a 2px bottom, radius 2,
//                    min-width 17, padding 0 2px
//
// Labels exist in their DOM for every tool and are `display: none` for all but
// one, so the bar is icons with tooltips. This builds exactly that from a spec.

import { iconMarkup } from './icons.js';
import { openMenu, isMenuOpen } from './menu.js';

/** Their 10 x 10 chevron for a split tool's caret. */
export const CARET_SVG = '<svg viewBox="0 0 10 10" width="10" height="10" aria-hidden="true"><path d="M1.2 3.2 5 7l3.8-3.8-1.1-1.1L5 4.8 2.3 2.1z" fill="currentColor"/></svg>';

/**
 * Build a toolbar into `host`.
 *
 * @param {HTMLElement} host  gets the `os-toolbar` class
 * @param {Array<object>} spec  in order:
 *   { kind: 'tool', id, icon, title, shortcut?, onClick }        an icon button
 *   { kind: 'toggle', id, icon, title, pressed?, onClick }        an icon button that stays lit
 *   { kind: 'split', id, icon, title, onClick, menu: items }      a tool with a caret that opens a menu
 *   { kind: 'labelled', id, icon, label, title?, onClick, accent? } the one labelled action
 *   { kind: 'search', label?, keys?: string[], onClick }          the search button
 *   { kind: 'separator' } | { kind: 'spacer' } | { kind: 'element', element }
 * @returns {{ element: HTMLElement, get: (id: string) => HTMLElement|undefined,
 *             setPressed: (id: string, on: boolean) => void,
 *             setDisabled: (id: string, reason: string|null) => void,
 *             setTitle: (id: string, title: string) => void }}
 */
export function createToolbar(host, spec) {
  host.classList.add('os-toolbar');
  host.setAttribute('role', 'toolbar');
  const byId = new Map();
  const titles = new Map();

  const tooltipText = (entry) => (entry.shortcut ? `${entry.title} (${entry.shortcut})` : entry.title ?? '');

  const makeTool = (entry, extraClass = '') => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = `os-tool${extraClass}`;
    button.innerHTML = iconMarkup(entry.icon, { className: 'os-icon os-tool-icon' });
    if (entry.title) button.setAttribute('data-tooltip', tooltipText(entry));
    button.setAttribute('aria-label', entry.title ?? entry.label ?? entry.id ?? '');
    if (entry.onClick) button.addEventListener('click', (event) => entry.onClick(event, api));
    return button;
  };

  for (const entry of spec) {
    switch (entry.kind) {
      case 'separator': {
        const rule = document.createElement('div');
        rule.className = 'os-toolset-separator';
        host.append(rule);
        break;
      }
      case 'spacer': {
        const gap = document.createElement('div');
        gap.className = 'os-toolbar-spacer';
        host.append(gap);
        break;
      }
      case 'element': {
        host.append(entry.element);
        if (entry.id) byId.set(entry.id, entry.element);
        break;
      }
      case 'tool':
      case 'toggle': {
        const button = makeTool(entry);
        if (entry.kind === 'toggle') button.setAttribute('aria-pressed', entry.pressed ? 'true' : 'false');
        host.append(button);
        byId.set(entry.id, button);
        titles.set(entry.id, tooltipText(entry));
        break;
      }
      case 'split': {
        const group = document.createElement('div');
        group.className = 'os-toolgroup';
        const button = makeTool(entry);
        const caret = document.createElement('button');
        caret.type = 'button';
        caret.className = 'os-tool os-tool-caret';
        caret.innerHTML = CARET_SVG;
        caret.setAttribute('aria-label', `${entry.title ?? entry.id} options`);
        caret.setAttribute('aria-haspopup', 'menu');
        caret.addEventListener('click', (event) => {
          event.stopPropagation();
          if (isMenuOpen(group)) return;
          const items = typeof entry.menu === 'function' ? entry.menu() : entry.menu;
          openMenu(items, { anchor: group, align: 'left', onSelect: (item) => entry.onMenu?.(item, api) });
        });
        group.append(button, caret);
        host.append(group);
        byId.set(entry.id, button);
        byId.set(`${entry.id}.caret`, caret);
        titles.set(entry.id, tooltipText(entry));
        break;
      }
      case 'labelled': {
        const button = makeTool(entry, ` os-tool--labelled${entry.accent ? ' os-tool--accent' : ''}`);
        const label = document.createElement('span');
        label.className = 'os-tool-label';
        label.textContent = entry.label;
        button.append(label);
        if (!entry.title) button.removeAttribute('data-tooltip');
        host.append(button);
        byId.set(entry.id, button);
        titles.set(entry.id, tooltipText(entry));
        break;
      }
      case 'search': {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'os-search';
        const text = document.createElement('span');
        text.textContent = entry.label ?? 'Search tools…';
        button.append(text);
        if (entry.keys?.length) {
          const keys = document.createElement('span');
          keys.className = 'os-search-keys';
          for (const key of entry.keys) {
            const chip = document.createElement('kbd');
            chip.className = 'os-key';
            chip.textContent = key;
            keys.append(chip);
          }
          button.append(keys);
        }
        if (entry.onClick) button.addEventListener('click', (event) => entry.onClick(event, api));
        host.append(button);
        if (entry.id) byId.set(entry.id, button);
        break;
      }
      default:
        throw new Error(`toolbar: unknown entry kind ${entry.kind}`);
    }
  }

  // Every control the bar holds, addressable from the DOM. Without this the
  // only handle on a button is this closure, which leaves "the button exists
  // and is enabled" unanswerable from outside -- by an automated check, or by
  // any app embedding the kit that wants to point at one.
  for (const [id, element] of byId) element.setAttribute('data-tool-id', id);

  const api = {
    element: host,
    get: (id) => byId.get(id),
    setPressed(id, on) {
      const button = byId.get(id);
      if (button) button.setAttribute('aria-pressed', on ? 'true' : 'false');
    },
    /** Disable with a reason, which becomes the tooltip; null re-enables and restores the title. */
    setDisabled(id, reason) {
      const button = byId.get(id);
      if (!button) return;
      button.disabled = reason !== null && reason !== undefined && reason !== false;
      const text = button.disabled && typeof reason === 'string' ? reason : titles.get(id) ?? '';
      if (text) button.setAttribute('data-tooltip', text);
    },
    setTitle(id, title) {
      const button = byId.get(id);
      if (!button) return;
      titles.set(id, title);
      button.setAttribute('data-tooltip', title);
      button.setAttribute('aria-label', title);
    },
  };
  return api;
}
