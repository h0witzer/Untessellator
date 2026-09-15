// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Tooltips, as the product shows them.
//
// MEASURED 2026-09-11 by `tools/chrome-extract.mjs`, hovering a feature-toolbar
// tool ("Revolve (shift+w)") in the live Part Studio:
//
//     fill          rgb(75, 87, 97)  #4b5761 — neither the panel nor the accent
//     text          #fff, 500 12px / 18px
//     padding       4px
//     max width     400px
//     shadow        0 1px 2px rgba(0, 0, 0, 0.6)     one of the two places the
//                                                    chrome carries a shadow
//     corners       square
//     placement     centred BELOW the control, its top on the control's
//                   bottom edge + 1px
//     appears       after a pause, well under a second; vanishes on leave,
//                   press or scroll
//
// One element serves the whole page. A control opts in with a `data-tooltip`
// attribute; `title` is deliberately not used, because the browser's own
// tooltip is the wrong shape and arrives on its own schedule.

const CLASS = 'os-tooltip';

let shared = null;

function element() {
  if (shared) return shared;
  shared = document.createElement('div');
  shared.className = CLASS;
  shared.setAttribute('role', 'tooltip');
  shared.hidden = true;
  document.body.append(shared);
  return shared;
}

/**
 * Show tooltips for every `[data-tooltip]` under `root`.
 *
 * @param {HTMLElement} root
 * @param {{ delay?: number }} options  delay before it appears, ms
 * @returns {{ dispose: () => void, hide: () => void }}
 */
export function installTooltips(root = document.body, { delay = 450 } = {}) {
  let timer = null;
  let current = null;
  const tip = element();

  const hide = () => {
    if (timer) clearTimeout(timer);
    timer = null;
    current = null;
    tip.hidden = true;
  };

  const show = (target) => {
    const text = target.getAttribute('data-tooltip');
    if (!text) return;
    tip.textContent = text;
    tip.hidden = false;
    const rect = target.getBoundingClientRect();
    const width = tip.offsetWidth;
    const height = tip.offsetHeight;
    let x = rect.left + rect.width / 2 - width / 2;
    let y = rect.bottom + 1;
    x = Math.max(2, Math.min(window.innerWidth - width - 2, x));
    if (y + height > window.innerHeight - 2) y = rect.top - height - 1;
    tip.style.left = `${Math.round(x)}px`;
    tip.style.top = `${Math.round(y)}px`;
  };

  const onOver = (event) => {
    const target = event.target.closest?.('[data-tooltip]');
    if (!target || target === current) return;
    hide();
    current = target;
    timer = setTimeout(() => {
      timer = null;
      if (current === target && target.isConnected) show(target);
    }, delay);
  };
  const onOut = (event) => {
    if (!current) return;
    const to = event.relatedTarget;
    if (to && current.contains(to)) return;
    hide();
  };

  root.addEventListener('pointerover', onOver);
  root.addEventListener('pointerout', onOut);
  root.addEventListener('pointerdown', hide, true);
  window.addEventListener('scroll', hide, true);
  window.addEventListener('blur', hide);

  return {
    hide,
    dispose() {
      hide();
      root.removeEventListener('pointerover', onOver);
      root.removeEventListener('pointerout', onOut);
      root.removeEventListener('pointerdown', hide, true);
      window.removeEventListener('scroll', hide, true);
      window.removeEventListener('blur', hide);
    },
  };
}
