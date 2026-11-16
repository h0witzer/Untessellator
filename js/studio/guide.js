/**
 * guide.js — the guide: Untessellator taught on a sample bracket, one small step at a time, over the part.
 *
 * Round 1 is the one-shot: Auto-paint, Color by Type, Export STEP. Round 2 loads the same bracket painted with four
 * mistakes (samples/SampleBracket.guide.json, written by test/guideBracket.mjs) and walks through their fixes: Find
 * fixes for the two it can mend, then by hand a group that covers two faces (Delete group, the bucket as Cylinder and
 * as Plane) and a wrong type that no check sees (Color by Type, Fit as Cylinder). Each fix was run in the page first
 * (f2b7aa2): the fixed paint exports to the Onshape STEP's 24 faces, the worst 5 nm off, as the one-shot does.
 *
 * Every step waits for the part to be in the state it asks for, judged against the bracket's answer key (each
 * triangle's true face), so it moves on once the user has done the thing, however they did it, and says what is off
 * when the part isn't right yet. The card sits top left over the view; the control a step needs pulses, and the
 * faces it means are outlined on the part. It reads the app through `app` (studio.js) and changes the part only by
 * opening its sample and loading its paint.
 */

const DATA_URL = 'samples/SampleBracket.guide.json';
const STL_URL = 'samples/SampleBracket.stl';
const STL_NAME = 'SampleBracket.stl';
const PAINT_LABEL = 'Load the guide\'s bad paint';
const TICK_MS = 250;
/** How long a step shows its check mark before the next one: long enough to see that it worked. */
const DONE_MS = 1100;
const ROUNDS = { 1: 'Round 1 · Auto-paint', 2: 'Round 2 · Fix by hand' };
const NAMES = { top: 'the top face', bottom: 'the bottom face', fillet: 'the fillet', hole: 'the large hole', cone: 'a countersink' };

const n = (v) => v.toLocaleString('en-US');
const cap = (s) => s[0].toUpperCase() + s.slice(1);
const q = (selector) => [...document.querySelectorAll(selector)];
const KEY = (k) => `<kbd>${k}</kbd>`;

export function createGuide(app) {
  /** The guide's file: the answer key, the faces' roles and the planted paint. */
  let data = null;
  /** Each true face's triangles. */
  let faceTris = [];
  /** The step on show; -1 while the guide is off. */
  let at = -1;
  /** A card in place of a step: { title, text, buttons } (asking first, or saying why the guide stopped). */
  let message = null;
  /** When the step's goal was first met (its check mark shows until DONE_MS later), or 0. */
  let doneSince = 0;
  /** The mesh the guide opened; any other (a new file, triangles deleted) ends it. */
  let sample = null;
  let planted = false;
  let timer = 0;
  let pulsed = [];
  let shown = '';

  const card = document.createElement('section');
  card.className = 'wb-guide';
  card.setAttribute('aria-label', 'Guide');
  card.hidden = true;
  app.stage.append(card);

  // ── the part, read against the answer key ──
  const store = () => app.mesh.store;
  const owner = () => app.mesh.store.owner;
  const role = (name) => data.roles[name];
  const tris = (name) => faceTris[role(name)];
  /** The group holding face f's triangles and nothing else, or -1. */
  function groupOf(f) {
    const list = faceTris[f], o = owner();
    const g = o[list[0]];
    if (g < 0 || list.some((t) => o[t] !== g)) return -1;
    return store().count(g) === list.length ? g : -1;
  }
  const typeOf = (g) => store().groups.get(g)?.type ?? null;
  /** Face f came back: one group, exactly its triangles, typed as the original's face. */
  const exact = (f) => { const g = groupOf(f); return g >= 0 && typeOf(g) === data.faces[f]; };
  const recovered = () => data.faces.reduce((k, _, f) => k + (exact(f) ? 1 : 0), 0);
  const unpainted = () => owner().reduce((k, o) => k + (o < 0 ? 1 : 0), 0);
  /** Some group holds triangles of both faces a and b. */
  function shared(a, b) {
    const o = owner(), groupsOfA = new Set();
    for (const t of faceTris[a]) if (o[t] >= 0) groupsOfA.add(o[t]);
    return faceTris[b].some((t) => groupsOfA.has(o[t]));
  }
  /** The groups as they are now built into a solid with every face on its surface. */
  const exported = () => { const m = app.mesh; return m.lastExport?.tier === 'analytic' && m.lastExportEdits === m.edits; };
  /** The export check ran on the groups as they are now and flagged some. */
  const checked = () => { const m = app.mesh; return !!m.lastCheck && m.lastCheck.edits === m.edits && m.flags.size > 0; };
  const paintThere = () => store().undoStack.some((u) => u.label === PAINT_LABEL);
  /** The two mistakes Find fixes is for are fixed: the spill back in the top face, the bottom face one group. */
  const firstFixesDone = () => exact(role('cone')) && groupOf(role('bottom')) >= 0;
  /** The flagged group holding the countersink (the spill's), else any flagged group, else null. */
  function flaggedCone() {
    const flags = app.mesh.flags, g = owner()[tris('cone')[0]];   // a cone triangle is never spill: its own group
    return flags.has(g) ? g : (flags.keys().next().value ?? null);
  }
  const countOff = () => app.mesh.flags.size;

  /**
   * In round 2, what has gone off the lesson's path, as { text, ring } or null: faces that were right in the bad paint
   * and aren't now (an edit, or Repair with the kernel, reached past the mistakes), or a face whose type the lesson
   * relies on changed. The steps can't finish from there, so the card says so instead of waiting.
   */
  function offTrack() {
    if (!planted || !paintThere()) return null;
    const roles = new Set(Object.values(data.roles));
    const changed = data.faces.map((_, f) => f).filter((f) => !roles.has(f) && !exact(f));
    // Repair with the kernel (next to Find fixes) retypes and merges failing groups, several undo steps at a time
    const stack = store().undoStack;
    let repairs = 0;
    while (repairs < stack.length && /^Repair/.test(stack[stack.length - 1 - repairs].label)) repairs++;
    const undo = `Undo (${KEY('Ctrl')}+${KEY('Z')})${repairs > 1 ? ` ${repairs} times` : ''}`;
    const back = repairs ? `${undo} to go back to before the repair, or start round 2 over.` : `${undo} to put ${changed.length === 1 ? 'it' : 'them'} back, or start round 2 over.`;
    const by = repairs ? 'Repair with the kernel changed' : '';
    if (changed.length) {
      return { text: repairs ? `${by} ${changed.length === 1 ? 'a face that was right' : `${n(changed.length)} faces that were right`} (outlined). ${back}`
        : `${changed.length === 1 ? 'A face that was right has' : `${n(changed.length)} faces that were right have`} changed (outlined). ${back}`,
        ring: changed.flatMap((f) => faceTris[f]) };
    }
    const o = owner();
    // a face's type: its main group's (the one with most of its triangles; spill puts a few elsewhere), null unpainted
    const typeAt = (name) => {
      const count = new Map();
      for (const t of tris(name)) count.set(o[t], (count.get(o[t]) ?? 0) + 1);
      const g = [...count].sort((a, b) => b[1] - a[1])[0][0];
      return g < 0 ? null : typeOf(g) ?? 'freeform';
    };
    const wrong = [['cone', ['cone']], ['bottom', ['plane']], ['top', ['plane']], ['hole', ['cone', 'cylinder']]]
      .find(([name, ok]) => typeAt(name) !== null && !ok.includes(typeAt(name)));
    if (!wrong) return null;
    return { text: `${repairs ? `${by} ${NAMES[wrong[0]]} to a ${typeAt(wrong[0])}.` : `${cap(NAMES[wrong[0]])} is now typed as ${typeAt(wrong[0])}.`} ${repairs ? back : `${undo} to put it back, or start round 2 over.`}`,
      ring: tris(wrong[0]) };
  }

  /** A step's checklist items met so far: each stays ticked once met, so undoing a filter doesn't untick it. */
  let ticked = new Set();
  /** What the step started from: the zoom count, so only a zoom made in the step ticks its item. */
  let entered = { zooms: 0 };
  function checklist(s) {
    const items = s.checklist?.() ?? [];
    items.forEach((item, k) => { if (!ticked.has(k) && item.done()) ticked.add(k); });
    return items.map((item, k) => ({ ...item, met: ticked.has(k) }));
  }

  /** What is off about face `name` after a bucket fill meant to paint it as `type`, or null. */
  function fillHint(name, type) {
    const list = tris(name), o = owner(), where = NAMES[name];
    const groups = new Set(list.map((t) => o[t]).filter((g) => g >= 0));
    if (!groups.size) return null;                               // not painted yet
    const undo = `Undo (${KEY('Ctrl')}+${KEY('Z')})`;
    const again = `${undo} and click closer to the middle of ${where}.`;
    if (groups.size > 1) return `${cap(where)} is split across ${groups.size} groups. Select them in the list (${KEY('Ctrl')}+click) and click <b>Merge</b>.`;
    const g = [...groups][0], got = typeOf(g) ?? 'freeform';
    const mine = list.filter((t) => o[t] === g).length;
    // a fill grows as far as its surface fits, so the wrong type is the likelier reason it stopped short or ran on
    if (got !== type && mine < list.length) return `That fill used ${cap(got)}, so it stopped early. ${undo}, set <b>Paint as</b> to <b>${cap(type)}</b>, and click again.`;
    if (store().count(g) > mine) return `That fill spread ${n(store().count(g) - mine)} triangles past ${where}. ${again}`;
    if (mine < list.length) return `${n(list.length - mine)} triangles of ${where} are still unpainted. ${again}`;
    if (got !== type) return `${cap(where)} is typed as ${got}. Right-click it and choose <b>Fit as</b> › <b>${cap(type)}</b>.`;
    return null;
  }
  const swatch = (type) => `<span class="wb-gsw" style="background:${app.typeColor(type)}"></span>`;
  const fillTargets = (type) => [...(app.tool !== 'fill' ? q('[data-tb="fill"]') : []), ...(app.type !== type ? q(`[data-tb="type-${type}"]`) : [])];

  // ── the steps ──
  // title, text (HTML; may read the part), and what the step waits for: `done` (moves on by itself), `next` (a button)
  // or `action` (a button that does it). `pause`: once done, it waits for Next, showing `doneText`, so what the step
  // just put on screen can be looked at. `checklist`: items that tick themselves off; all ticked is done. `target`
  // pulses controls; `ring` outlines triangles; `show` adds Show me; `leave` runs as the step is left.
  const STEPS = [
    {
      round: 1, title: 'The sample part', next: 'Start',
      text: () => `This bracket was exported from Onshape as an STL. It has ${n(data.triangles)} triangles and no CAD faces. Untessellator rebuilds the faces and exports a STEP file.
        <span class="wb-gaside">Right-drag to rotate. Scroll to zoom. ${KEY('F')} fits the part to the view.</span>`,
    },
    {
      round: 1, title: 'Auto-paint',
      text: () => `Click <b>Auto-paint</b> (${KEY('P')}). It sorts the triangles into faces and fits a surface to each one. This takes a few seconds.`,
      target: () => q('[data-tb="autopaint"]'),
      done: () => !app.autoRunning && store().list().length > 0 && unpainted() === 0,
      hint: () => (!app.autoRunning && store().list().length && unpainted() ? `${n(unpainted())} triangles are still unpainted. Click Auto-paint again to fill them.` : null),
    },
    {
      round: 1, title: 'Show surface types',
      text: () => {
        const k = recovered();
        return `Auto-paint found ${n(store().list().length)} faces${k === data.faces.length ? ', the same as the original part' : ''}. Each color is one face. Set <b>Color by</b> to <b>Type</b> to see each face's surface type.`;
      },
      target: () => q('[data-color="type"]'),
      done: () => app.colorBy === 'type',
    },
    {
      round: 1, title: 'Surface types', next: 'Next',
      text: () => `${swatch('plane')}Plane<br>${swatch('cylinder')}Cylinder: holes and rounded corners<br>${swatch('cone')}Cone: countersinks<br>${swatch('torus')}Torus: the rounded edges of the large hole
        <span class="wb-gaside">The list on the right shows each face's type and how closely it fits.</span>`,
    },
    {
      round: 1, title: 'Explore the faces', next: 'Next',
      text: () => 'Use the list to find faces on the part.',
      checklist: () => [
        { text: 'Click a type filter above the list, like <b>Cone</b>.', done: () => app.filters.types.length > 0,
          target: () => q('[data-ftype="cone"]') },
        { text: 'Click a face in the list to select it.', done: () => app.selection.length > 0,
          target: () => q('tr.wb-g').slice(0, 1) },
        { text: 'Click <b>Zoom to</b>, or double-click the row.', done: () => app.zooms.count > entered.zooms,
          target: () => q('.wb-inspector [data-act="zoom"]') },
        // the zoom first: the part starts in full view, which would tick this at once
        { text: `Press ${KEY('F')} to see the whole part again. <b>Zoom to fit</b> in the view cube's menu does the same.`, done: () => ticked.has(2) && app.viewFitted,
          target: () => q('.os-graphics-dropdown') },
      ],
      leave: () => app.clearFilters(),
    },
    {
      round: 1, title: 'Export',
      text: () => 'Click <b>Export STEP</b>. It checks the faces, builds the solid, and shows a preview.',
      target: () => q('#export'),
      done: exported,
    },
    {
      round: 1, title: 'Preview', next: 'Go to round 2',
      text: () => 'Every face is on its exact surface. Drag the white line to compare the mesh with the STEP. You don\'t need to save this file.',
    },
    {
      round: 2, title: 'Load a bad paint', next: 'Next', pause: true,
      text: () => 'Next is the same bracket with four painting mistakes. You\'ll fix two with Find fixes and two by hand.',
      doneText: () => 'The part now has four mistakes. Some are easy to spot and some aren\'t. Look it over, then click <b>Next</b> to run the check.',
      action: { label: 'Load it', run: () => loadBadPaint() },
      done: () => planted && paintThere(),
    },
    {
      round: 2, title: 'Run the check',
      text: () => 'Click <b>Export STEP</b>. The check runs before the build and flags faces that don\'t fit their surface.',
      target: () => q('#export'),
      // Export STEP's check or the summary's (Color by Fit) both flag; past the spill and the split, it has done its job
      done: () => checked() || app.mesh.flags.size > 0 || firstFixesDone(),
    },
    {
      round: 2, title: 'Inspect the flagged faces', next: 'Next', pause: true,
      text: () => 'Look at the flagged faces before you fix them.',
      doneText: () => 'The red slivers on the countersink are triangles from the top face that were painted into the wrong group. That\'s called spill. Turn the part to look, then click <b>Next</b> to fix it.',
      checklist: () => {
        const cone = flaggedCone();
        return [
          { text: `Click <b>${n(Math.max(1, countOff()))} likely wrong</b> at the top. The list shows only the faces that need a look.`, done: () => app.filters.fits.includes('off'),
            target: () => q('#summary') },
          { text: `Double-click <b>${cone === null ? 'a flagged face' : esc(store().name(cone))}</b> in the list to zoom to it.`, done: () => app.zooms.count > entered.zooms && app.zooms.ids.some((g) => app.mesh.flags.has(g)),
            target: () => q(cone === null ? 'tr.wb-g' : `tr.wb-g[data-group="${cone}"]`).slice(0, 1) },
          { text: 'Set <b>Color by</b> to <b>Deviation</b>. Red triangles are far from their face\'s surface.', done: () => app.colorBy === 'deviation',
            target: () => q('[data-color="deviation"]') },
        ];
      },
      leave: () => app.clearFilters(),
    },
    {
      round: 2, title: 'Find fixes',
      text: () => (app.mesh.flags.size ? `The check flagged ${n(app.mesh.flags.size)} face${app.mesh.flags.size === 1 ? '' : 's'}. ` : '') + 'Click <b>Find fixes</b> under Fix bad areas.',
      target: () => [...q('.wb-result [data-step]').filter((b) => b.textContent.trim() === 'Find fixes'), ...q('[data-act="fix-find"]')],
      // the fixes found, or the faces they'd fix already right, however that came about
      done: () => (app.fixes.fresh && app.fixes.count > 0) || firstFixesDone(),
      hint: () => (app.fixes.fresh && !app.fixes.count && !firstFixesDone() ? 'Find fixes found nothing to fix. Undo (Ctrl+Z) your last change, or start round 2 over.' : null),
    },
    {
      round: 2, title: 'Apply fixes',
      text: () => `Find fixes found ${n(app.fixes.count)} problems. ${n(data.spill.length)} triangles of the top face were painted into a countersink (outlined), and the bottom face was split into two groups. Click <b>Apply</b> under Fix bad areas.`,
      ring: () => data.spill,
      // the list to apply from; after a partial apply it is stale, and Find fixes makes it again
      target: () => q(!app.fixes.fresh ? '[data-act="fix-find"]' : app.fixes.open ? '[data-act="fix-apply"]' : '[data-act="fix-toggle"]'),
      done: firstFixesDone,
      hint: () => {
        const cone = exact(role('cone')), bottom = groupOf(role('bottom')) >= 0;
        if (cone === bottom) return null;
        return cone ? 'The bottom face is still split. Click <b>Find fixes</b> again and apply the merge.'
          : 'The countersink still has top-face triangles. Click <b>Find fixes</b> again and apply the re-fill.';
      },
    },
    {
      round: 2, title: 'Delete the group', show: true,
      text: () => 'The top face\'s group also covers the fillet (outlined). Find fixes can\'t separate them, so you\'ll repaint both. Right-click the top face and choose <b>Delete group</b>.',
      ring: () => [...tris('top'), ...tris('fillet')],
      done: () => !shared(role('top'), role('fillet')),
      hint: () => (tris('fillet').some((t) => owner()[t] < 0) ? 'Part of the fillet is still in the top face\'s group. Right-click the top face and choose <b>Delete group</b>.' : null),
    },
    {
      round: 2, title: 'Paint the fillet', show: true,
      text: () => `Select the <b>Bucket</b> (${KEY('G')}) and set <b>Paint as</b> to <b>Cylinder</b> (${KEY('3')}). Then click the fillet.`,
      ring: () => tris('fillet'),
      target: () => fillTargets('cylinder'),
      done: () => exact(role('fillet')),
      hint: () => fillHint('fillet', 'cylinder'),
    },
    {
      round: 2, title: 'Paint the top face',
      text: () => `Set <b>Paint as</b> to <b>Plane</b> (${KEY('2')}). Then click the top face.`,
      ring: () => tris('top'),
      target: () => fillTargets('plane'),
      done: () => exact(role('top')) && unpainted() === 0,
      hint: () => fillHint('top', 'plane') ?? (exact(role('top')) && unpainted() ? `${n(unpainted())} triangles are still unpainted. Click Auto-paint to fill them.` : null),
    },
    {
      round: 2, title: 'Check the types', show: true,
      text: () => `Some mistakes are subtle. They pass the check and only show up in the face types. Here the large hole is typed as a ${swatch('cone')}cone. A cone can fit a hole almost exactly, so nothing flags it.${app.colorBy !== 'type' ? ' Set <b>Color by</b> to <b>Type</b> to see it.' : ''} Right-click the hole and choose <b>Fit as</b> › <b>Cylinder</b>.`,
      ring: () => tris('hole'),
      target: () => (app.colorBy !== 'type' ? q('[data-color="type"]') : []),
      done: () => exact(role('hole')),
    },
    {
      round: 2, title: 'Export again',
      text: () => 'Click <b>Export STEP</b>.',
      target: () => q('#export'),
      done: exported,
    },
    {
      round: 2, final: true,
      title: () => (recovered() === data.faces.length ? `All ${data.faces.length} faces match` : `${recovered()} of ${data.faces.length} faces match`),
      text: () => {
        if (recovered() === data.faces.length) return 'Every face matches the original part. Open your own file to get started.';
        const off = Object.keys(NAMES).filter((k) => !exact(role(k))).map((k) => NAMES[k]);
        return `${off.length ? `${cap(off.join(', '))} ${off.length === 1 ? 'doesn\'t' : 'don\'t'} match the original yet.` : 'Some faces don\'t match the original yet.'} Start round 2 over to try again.`;
      },
    },
  ];
  const roundSteps = (r) => STEPS.filter((s) => s.round === r);

  // ── the card ──
  /**
   * The step's state now: what is off the lesson's path, its checklist, whether its goal is met (never while off the
   * path or mid-run), and what to pulse (the first open checklist item's control, else the step's) and outline.
   */
  function stateOf(s) {
    const off = s.round === 2 && !s.final ? offTrack() : null;
    const items = checklist(s);
    const goal = s.checklist ? items.every((item) => item.met) : !!s.done && s.done();
    const done = !off && !app.busy && goal;
    const open = items.find((item) => !item.met);
    const targets = done || app.busy ? [] : ((open ? open.target?.() : s.target?.()) ?? []);
    return { off, items, done, targets: targets.filter(Boolean), ring: off?.ring ?? s.ring?.() ?? null };
  }

  function render(st) {
    let html;
    if (message) {
      html = `<div class="wb-ghead"><span class="wb-ground">Guide</span><button type="button" class="wb-iconbtn wb-rclose" data-g="exit" aria-label="Close the guide" title="Close the guide">✕</button></div>
        <div class="wb-glive" aria-live="polite"><h3 class="wb-gtitle">${message.title}</h3><p class="wb-gtext">${message.text}</p></div>
        <div class="wb-grow"><span class="wb-spacer"></span>${message.buttons.map((b, i) => `<button type="button" class="wb-btn wb-small${b.primary ? ' wb-primary' : ''}" data-g="msg" data-i="${i}">${b.label}</button>`).join('')}</div>`;
    } else {
      const s = STEPS[at], { done } = st;
      const list = roundSteps(s.round);
      const i = list.indexOf(s);
      const title = typeof s.title === 'function' ? s.title() : s.title;
      // what's off the path first; a step's own hint waits out a stroke or a run, when the part is half done
      const hint = st.off?.text ?? (!done && !app.busy && s.hint ? s.hint() : null);
      const firstOpen = st.items.findIndex((item) => !item.met);
      const checks = st.items.length ? `<ol class="wb-gchecks">${st.items.map((item, k) => `<li class="${item.met ? 'wb-gcheck--met' : k === firstOpen ? 'wb-gcheck--now' : ''}"><span>${item.text}</span></li>`).join('')}</ol>` : '';
      const buttons = s.final
        ? '<button type="button" class="wb-btn wb-small" data-g="openown">Open a file…</button><button type="button" class="wb-btn wb-small wb-primary" data-g="exit">Close guide</button>'
        // a step with an action offers Next once it's done; a checklist's Next is a quiet skip until then
        : `${s.action && !done ? `<button type="button" class="wb-btn wb-small wb-primary" data-g="action">${s.action.label}</button>` : ''}${s.next && (!s.action || done) ? `<button type="button" class="wb-btn wb-small${s.checklist && !done ? '' : ' wb-primary'}" data-g="next">${s.next}</button>` : ''}`;
      const body = done && s.pause && s.doneText ? s.doneText() : s.text();
      // the bar counts a step done once its goal is met; a step that only reads is done on sight
      const filled = i + (done || s.final || (s.next && !s.checklist) ? 1 : 0);
      html = `<div class="wb-ghead"><span class="wb-ground">${ROUNDS[s.round]}</span><span class="wb-gcount">${i + 1} of ${list.length}</span>
          <button type="button" class="wb-iconbtn wb-rclose" data-g="exit" aria-label="Close the guide" title="Close the guide">✕</button></div>
        <div class="wb-gbar" aria-hidden="true"><span style="width:${(100 * filled) / list.length}%"></span></div>
        <div class="wb-glive" aria-live="polite"><h3 class="wb-gtitle${done ? ' wb-gtitle--done' : ''}">${title}</h3><p class="wb-gtext">${body}</p>${checks}</div>
        ${hint ? `<p class="wb-ghint">${hint}</p>` : ''}
        <div class="wb-grow">${s.show && s.ring ? '<button type="button" class="wb-btn wb-small" data-g="show" title="Frame the outlined faces">Show me</button>' : ''}<span class="wb-spacer"></span>${buttons}</div>
        ${s.round === 2 && at > STEPS.findIndex((x) => x.round === 2) ? '<button type="button" class="wb-glink" data-g="restart2">Start round 2 over</button>' : ''}`;
    }
    if (html !== shown) { card.innerHTML = html; shown = html; }
    card.hidden = false;
    app.stage.classList.add('wb-view--guide');
    app.stage.style.setProperty('--guide-h', `${card.offsetHeight}px`);
  }

  /** Pulse controls `els`, and only those. */
  function pulse(els) {
    for (const el of pulsed) if (!els.includes(el)) el.removeAttribute('data-guide');
    for (const el of els) el.setAttribute('data-guide', '');
    pulsed = els;
  }

  /** Outline triangles `tris` on the part, telling the app only when they change (each change rebuilds its edges). */
  let pointedAt = '';
  function point(tris) {
    const key = tris?.length ? `${tris.length}:${tris[0]}:${tris[tris.length - 1]}` : '';
    if (key === pointedAt) return;
    pointedAt = key;
    app.point(tris);
  }

  /** Show step `at` as the part is now; returns its state. */
  function show() {
    const st = stateOf(STEPS[at]);
    point(st.ring);
    render(st);
    pulse(st.targets);
    return st;
  }

  function go(i) {
    if (at >= 0 && at !== i) STEPS[at].leave?.();   // a step that had the user filter the list clears it
    at = i;
    message = null;
    doneSince = 0;
    ticked = new Set();
    entered = { zooms: app.zooms.count };
    show();
  }

  function tick() {
    if (at < 0 && !message) return;
    if (message) { render(null); pulse([]); return; }
    if (app.mesh !== sample) {
      showMessage('Guide stopped', 'A different mesh is open, or triangles were deleted. The guide only works on its sample part.',
        [{ label: 'Restart guide', primary: true, run: () => start() }, { label: 'Close', run: () => stop() }]);
      return;
    }
    const st = show();
    // a step that ends on something to look at (the bad paint, the spill) waits for Next; the rest move on by themselves
    if (st.done && !STEPS[at].pause) {
      doneSince ||= performance.now();
      if (performance.now() - doneSince >= DONE_MS && at + 1 < STEPS.length) go(at + 1);
    } else doneSince = 0;
  }

  /** A card in place of a step, with its own buttons ({ label, primary, run }). */
  function showMessage(title, text, buttons) {
    at = -1;
    message = { title, text, buttons };
    point(null);
    pulse([]);
    render(null);
    clearInterval(timer);
    timer = setInterval(tick, TICK_MS);
  }

  async function loadBadPaint() {
    if (app.busy) { app.say('Finish or stop what is running first (Esc)', true); return false; }
    const ok = await app.loadPaint(data.planted, PAINT_LABEL);
    if (ok) { planted = true; app.say('Loaded the bad paint'); }
    return ok;
  }

  card.addEventListener('click', async (event) => {
    const button = event.target.closest('[data-g]');
    if (!button) return;
    const what = button.dataset.g;
    if (what === 'exit') stop();
    else if (what === 'next') go(at + 1);
    else if (what === 'action') STEPS[at].action.run();
    else if (what === 'show') app.frame(STEPS[at].ring());
    else if (what === 'openown') { stop(); app.openFilePicker(); }
    else if (what === 'restart2') {
      if (await loadBadPaint()) go(STEPS.findIndex((s) => s.round === 2) + 1);
    } else if (what === 'msg') message?.buttons[Number(button.dataset.i)]?.run();
  });

  async function openSample() {
    if (app.busy) { app.say('Finish or stop what is running first (Esc)', true); return; }
    showMessage('Opening the sample', 'Loading the bracket…', []);
    const before = app.mesh;
    let opened = null;
    try { opened = await app.openSample(STL_URL, STL_NAME); } catch (error) { app.say(`The sample could not open: ${error.message}`, true); }
    if (!opened || opened === before || opened.model.count !== data.triangles || app.repaired) {
      showMessage('Couldn\'t open the sample', 'The sample bracket didn\'t load correctly. Try again. If it keeps failing, report it with the flag button.',
        [{ label: 'Try again', primary: true, run: () => openSample() }, { label: 'Close', run: () => stop() }]);
      return;
    }
    sample = opened;
    planted = false;
    address('/guide/');
    go(0);
  }

  async function start() {
    if (at >= 0 && app.mesh === sample) { card.classList.remove('wb-guide--flash'); void card.offsetWidth; card.classList.add('wb-guide--flash'); return; }
    if (!data) {
      try {
        const response = await fetch(DATA_URL);
        if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
        data = await response.json();
      } catch (error) {
        app.say(`The guide could not load: ${error.message}`, true);
        return;
      }
      faceTris = data.faces.map(() => []);
      data.key.forEach((f, t) => faceTris[f].push(t));
    }
    clearInterval(timer);
    timer = setInterval(tick, TICK_MS);
    const mine = app.mesh && app.mesh !== sample && app.mesh.store.painted > 0;
    if (mine) {
      showMessage('Replace your part?', `The guide opens a sample part in place of <b>${esc(app.mesh.name)}</b>. Your painting won't be kept. Export first if you need it.`,
        [{ label: 'Open the sample', primary: true, run: () => openSample() }, { label: 'Cancel', run: () => stop() }]);
      return;
    }
    openSample();
  }

  function stop() {
    clearInterval(timer);
    at = -1;
    message = null;
    shown = '';
    pulse([]);
    point(null);
    card.hidden = true;
    app.stage.classList.remove('wb-view--guide');
    address('/');
  }

  return { start, stop, get running() { return at >= 0 || !!message; } };
}

/**
 * The address bar says where the user is: untessellator.com/guide/ while the guide runs, the plain site once it ends
 * (other query parameters kept). /guide/ is guide/index.html, which hands a reload back to /?guide, so a reload
 * mid-guide starts it again and one after it closed doesn't. The page's own paths don't move: index.html's <base>.
 */
function address(path) {
  const url = new URL(location.href);
  url.searchParams.delete('guide');
  url.pathname = path;
  if (url.href !== location.href) history.replaceState(history.state, '', url);
}

const esc = (text) => String(text).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
