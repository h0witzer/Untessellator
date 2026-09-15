// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Popup menus: a dropdown's list and a context menu, as the product draws them.
//
// MEASURED 2026-09-11 by `tools/chrome-extract.mjs`, with the camera dropdown's
// menu open under the view cube and a context menu open on empty canvas:
//
//   the dropdown menu (their `.view-cube`, 224 x 362)
//     fill          #333 (the primary background)
//     text          rgb(215,215,215) #d7d7d7, 12px / 18px
//     border        1px solid #d7d7d7, radius 4
//     shadow        0 5px 10px rgba(0, 0, 0, 0.4)
//     entry         22px tall, flex, aligned centre
//     icon column   27px, holding a check or a 16px glyph; a `<a>` padded 0 5px
//     label         after the column, no extra padding
//     hover         #0a4780, the same as a list row
//     disabled      text #999, no hover
//     separator     a 1px line, rgb(221,221,221), below an entry
//     submenu       a small right-pointing triangle at the right edge
//     placement     top edge on the button's bottom edge, right edges aligned
//
//   the context menu (their `.context-menu-list`, 121 wide on empty canvas,
//   250 wide on a face)
//     fill #333; border 1px solid #ccc, radius 4; the same shadow and font
//     entry         22px, text #fff; "list-no-icons": padding 2px 8px 2px 16px;
//                   "list-has-icons": padding 2px 8px 2px 35px with a 17px
//                   glyph at 10px from the left
//     hover         #0a4780; separators as above
//     submenu       an arrow at the right; the child opens beside the entry
//
// So the two are one component with a `compact` flag for the context menu's
// padding, and this is the second of the two places the chrome has a shadow.
//
// One menu is open at a time. Escape, a press outside, a wheel, or a resize
// closes it. Selecting an entry closes it first and then calls the entry's
// handler, so a handler that opens another menu is not closed by its own click.

let open = null;

function closeChild(entry) {
  if (entry.child) {
    closeChild(entry.child);
    entry.child.element.remove();
    entry.child = null;
  }
}
function closeOpen() {
  if (open) {
    const { element, onClose } = open;
    closeChild(open);
    open = null;
    element.remove();
    document.removeEventListener('pointerdown', onOutside, true);
    document.removeEventListener('keydown', onKey, true);
    window.removeEventListener('resize', closeOpen);
    window.removeEventListener('wheel', closeOpen, true);
    onClose?.();
  }
}
function insideChain(entry, target) {
  for (let node = entry; node; node = node.child) if (node.element.contains(target)) return true;
  return false;
}
function onOutside(event) {
  if (open && !insideChain(open, event.target) && !open.anchor?.contains(event.target)) closeOpen();
}
function onKey(event) {
  if (event.key === 'Escape') closeOpen();
}

/** Their submenu arrow: a small filled triangle pointing right. */
const ARROW_SVG = '<svg viewBox="0 0 10 10" width="10" height="10" aria-hidden="true"><path d="M3 1.5v7l4-3.5z" fill="currentColor"/></svg>';
/** Their check mark, drawn filled on the 20 grid. */
export const CHECK_SVG = '<svg viewBox="0 0 20 20" width="16" height="16" aria-hidden="true"><path d="M3 10.6l1.5-1.5 3.6 3.6L15.6 5.2 17.1 6.7 8.1 15.7z" fill="currentColor"/></svg>';

/**
 * Open a menu.
 *
 * @param {Array<{ id?: string, label?: string, icon?: string, checked?: boolean, disabled?: boolean,
 *                 submenu?: boolean, children?: Array<object>, separator?: boolean,
 *                 onSelect?: (item: object) => void }>} items
 *   `icon` is svg markup for the icon column; `checked` puts a check there
 *   instead; `children` opens a submenu beside the entry on hover.
 * @param {{ anchor?: HTMLElement, align?: 'right'|'left', at?: { x: number, y: number },
 *           compact?: boolean, icons?: boolean, minWidth?: number, onClose?: () => void,
 *           onSelect?: (item: object) => void }} options
 *   Anchor to a control (opens below it, right edges aligned by default), or
 *   place at a point for a context menu. `compact` is their "list-no-icons"
 *   context menu; `compact` with `icons` is their "list-has-icons" one.
 * @returns {{ element: HTMLElement, close: () => void }}
 */
export function openMenu(items, options = {}) {
  closeOpen();
  const entry = buildMenu(items, options);
  const { anchor = null, onClose } = options;
  open = { element: entry.element, anchor, onClose, child: null };
  entry.attachChildren(open);
  document.addEventListener('pointerdown', onOutside, true);
  document.addEventListener('keydown', onKey, true);
  window.addEventListener('resize', closeOpen);
  window.addEventListener('wheel', closeOpen, true);
  return { element: entry.element, close: closeOpen };
}

/** Build and place one menu level; submenus are built on hover by `attachChildren`. */
function buildMenu(items, options) {
  const { anchor = null, align = 'right', at = null, compact = false, icons = false, minWidth, onSelect } = options;
  const hasIcons = icons || (compact && items.some((item) => item.icon));
  const menu = document.createElement('div');
  menu.className = `os-menu${compact ? ' os-menu--compact' : ''}${compact && hasIcons ? ' os-menu--icons' : ''}`;
  menu.setAttribute('role', 'menu');
  if (minWidth) menu.style.minWidth = `${minWidth}px`;
  const submenuRows = [];

  for (const item of items) {
    if (item.separator) {
      const rule = document.createElement('div');
      rule.className = 'os-menu-separator';
      menu.append(rule);
      continue;
    }
    const row = document.createElement('div');
    row.className = `os-menu-item${item.disabled ? ' os-menu-item--disabled' : ''}`;
    row.setAttribute('role', 'menuitem');
    if (item.id) row.dataset.id = item.id;
    if (item.disabled) row.setAttribute('aria-disabled', 'true');
    if (!compact || hasIcons) {
      const column = document.createElement('span');
      column.className = 'os-menu-icon';
      column.innerHTML = item.checked ? CHECK_SVG : item.icon ?? '';
      row.append(column);
    }
    const label = document.createElement('span');
    label.className = 'os-menu-label';
    label.textContent = item.label ?? '';
    row.append(label);
    if (item.submenu || item.children) {
      const arrow = document.createElement('span');
      arrow.className = 'os-menu-arrow';
      arrow.innerHTML = ARROW_SVG;
      row.append(arrow);
    }
    row.addEventListener('pointerdown', (event) => {
      event.preventDefault();
      event.stopPropagation();
    });
    row.addEventListener('click', (event) => {
      event.stopPropagation();
      if (item.disabled || item.children) return;
      closeOpen();
      item.onSelect?.(item);
      onSelect?.(item);
    });
    if (item.children) submenuRows.push({ row, item });
    menu.append(row);
  }

  document.body.append(menu);
  // Position, then keep it on screen.
  const width = menu.offsetWidth;
  const height = menu.offsetHeight;
  let x;
  let y;
  if (anchor) {
    const rect = anchor.getBoundingClientRect();
    x = align === 'right' ? rect.right - width : rect.left;
    y = rect.bottom;
  } else if (at) {
    x = at.x;
    y = at.y + 2;
  } else {
    x = 0;
    y = 0;
  }
  x = Math.max(2, Math.min(window.innerWidth - width - 2, x));
  if (y + height > window.innerHeight - 2) y = Math.max(2, (anchor ? anchor.getBoundingClientRect().top : at?.y ?? 0) - height);
  menu.style.left = `${Math.round(x)}px`;
  menu.style.top = `${Math.round(y)}px`;

  return {
    element: menu,
    /** Wire the submenus: hover an entry with children and its menu opens beside it. */
    attachChildren(entry) {
      let timer = null;
      for (const { row, item } of submenuRows) {
        row.addEventListener('pointerenter', () => {
          clearTimeout(timer);
          timer = setTimeout(() => {
            closeChild(entry);
            const rect = row.getBoundingClientRect();
            const child = buildMenu(item.children, { ...options, anchor: null, at: null, compact, icons, minWidth: undefined });
            const cw = child.element.offsetWidth;
            let cx = rect.right - 2;
            if (cx + cw > window.innerWidth - 2) cx = rect.left - cw + 2;
            child.element.style.left = `${Math.round(cx)}px`;
            child.element.style.top = `${Math.round(Math.min(rect.top - 1, window.innerHeight - child.element.offsetHeight - 2))}px`;
            entry.child = { element: child.element, anchor: row, child: null };
            child.attachChildren(entry.child);
            row.classList.add('os-menu-item--open');
          }, 220);
        });
        row.addEventListener('pointerleave', (event) => {
          clearTimeout(timer);
          // leaving toward the child keeps it; anywhere else closes it
          const to = event.relatedTarget;
          if (entry.child && to && entry.child.element.contains(to)) return;
          timer = setTimeout(() => {
            if (entry.child && !entry.child.element.matches(':hover')) {
              closeChild(entry);
              row.classList.remove('os-menu-item--open');
            }
          }, 260);
        });
      }
      menu.addEventListener('pointerenter', () => {
        // back on the parent: a stale child closes unless its own row is hovered
        for (const { row } of submenuRows) if (!row.matches(':hover') && entry.child?.anchor === row) { closeChild(entry); row.classList.remove('os-menu-item--open'); }
      });
    },
  };
}

/** Close whatever menu is open. */
export function closeMenu() {
  closeOpen();
}

/** Is a menu open for this anchor? */
export function isMenuOpen(anchor) {
  return open !== null && (anchor === undefined || open.anchor === anchor);
}

/**
 * Wire a context menu to an element: right-click opens `items(event)` at the
 * pointer, in the compact style. Returns a function that unwires it.
 */
export function bindContextMenu(element, items, { icons = false } = {}) {
  const handler = (event) => {
    event.preventDefault();
    const list = typeof items === 'function' ? items(event) : items;
    if (!list || list.length === 0) return;
    openMenu(list, { at: { x: event.clientX, y: event.clientY }, compact: true, icons });
  };
  element.addEventListener('contextmenu', handler);
  return () => element.removeEventListener('contextmenu', handler);
}
