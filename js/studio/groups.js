/**
 * groups.js — the face groups a user paints, and their undo history.
 *
 * `owner[t]` is the id of the group triangle t belongs to, or -1 while it is unpainted.
 * A group's metadata (its surface type and latest fit) lives in `groups`; a group whose
 * last triangle is taken away is dropped when the edit that emptied it commits.
 *
 * Every change happens inside an edit: begin(), any number of assign() and meta changes,
 * commit(label). An edit that changed nothing is discarded, so an empty stroke leaves no
 * undo step. Undo restores ownership triangle by triangle and the group metadata as it
 * was when the edit began.
 *
 * Colors stay apart where groups meet. A group starts with groupColour(id); given the mesh's
 * triangle adjacency, each commit checks the groups the edit grew against the groups they
 * border, and one whose color reads too close to a neighbor's takes the free color farthest
 * from them all, in the same undo step. Otherwise a group keeps its color, through undo too.
 */

const TYPE_LABELS = {
  plane: 'Plane', cylinder: 'Cylinder', cone: 'Cone', sphere: 'Sphere', torus: 'Torus',
  skip: 'Skip',   // set aside: what the import mangled (studio.js); never fitted or typed
};

export function typeLabel(type) {
  return TYPE_LABELS[type] ?? 'Freeform';
}

/**
 * The group colors: clean ones, near the edge of what sRGB can show. Twelve OKLCH hues, each at the lightness where
 * it is most saturated (held to 0.64..0.86) and 0.88 of the gamut's chroma there; a second ring 15 degrees on, a
 * step lighter if warm, darker if cool; and deeper tones of the cool hues only, since a darkened yellow is olive and
 * a darkened orange brown, the mud the old golden-angle HSL colors (50% saturation, 62% lightness) were full of.
 * No pale tints, and no yellows or peach: the headlight puts a face at half light when it turns away, and those go
 * olive, tan or gray-brown there. Ambers are free to use: selection and hover are border lines (renderer.js
 * faceBorder bands), never fills. In order of hue. (2026-10-31)
 */
const GROUP_HEX = [
  '#f4356d', '#f45956', '#f4652d', '#f7a73b', '#a4ea44', '#8ffc94', '#4cf499', '#49e7ba', '#4ceed4', '#38b5a1', '#4be7e7', '#37b1bd', '#4ce9f9',
  '#3dbfe2', '#37adf3', '#247aad', '#277fe5', '#707ff2', '#432eec', '#8360f0', '#ab5ff2', '#7c24bd', '#be31db', '#ab2598', '#f53ad9', '#f55eb0',
];
const GROUP_RGB = GROUP_HEX.map((hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255));

/** A colour per group id: seven places round the hue order at a time (about 100 degrees), every color in turn. */
export function groupColour(id) {
  return [...GROUP_RGB[(Math.max(0, id - 1) * 7) % GROUP_RGB.length]];
}

// ── colors apart from their neighbors ────────────────────────────────────────

/**
 * How far apart two colors read: OKLab distance x100 (about 2 is barely visible), the least of normal vision / 1.6
 * and simulated protanopia, deuteranopia and tritanopia (Machado 2009): test/palette.mjs's measure. Bordering groups
 * closer than COLOUR_GAP read as one face, as a plane, cone and cylinder in three greens did (2026-10-31): the
 * old golden-angle colors of two ids were that close 45% of the time, so a color per id alone cannot keep them apart.
 */
const COLOUR_GAP = 9;
// A recolored group takes a color at least this far from its neighbors when one is free, with room to spare.
const CLEAR_GAP = 12;
const VISION = [
  [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
  [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],
  [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]],
  [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]],
];
// What a group's color must not be read as: unpainted triangles (studio.js UNPAINTED). The kit's selected and
// pre-highlight ambers are outlines, so a filled group can't be taken for them.
const AVOID = [[0.74, 0.74, 0.74]];

const clamp01 = (v) => Math.min(1, Math.max(0, v));
const toLinear = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
function oklab([r, g, b]) {
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s, 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s];
}
/** A color (sRGB, 0..1) as each vision sees it, in OKLab. */
function seen(rgb) {
  const c = rgb.map(toLinear);
  return VISION.map((m) => oklab(m.map((row) => clamp01(row[0] * c[0] + row[1] * c[1] + row[2] * c[2]))));
}
function gap(a, b) {
  let least = Infinity;
  for (let i = 0; i < a.length; i++) {
    const d = 100 * Math.hypot(a[i][0] - b[i][0], a[i][1] - b[i][1], a[i][2] - b[i][2]) / (i === 0 ? 1.6 : 1);
    if (d < least) least = d;
  }
  return least;
}
// The colors a group may move to: all of GROUP_HEX, so a recolored group is as clean as the rest
let spare = null;
function spareColours() {
  spare ??= GROUP_RGB.map((rgb) => ({ rgb, seen: seen(rgb) }));
  return spare;
}
const avoidSeen = AVOID.map(seen);

const cloneMeta = (groups) => new Map([...groups].map(([id, g]) => [id, { ...g, colour: [...g.colour] }]));

export class GroupStore {
  /** `neighbours` (optional): the mesh's triangle adjacency, three per triangle, -1 on an open edge. */
  constructor(triangleCount, neighbours = null) {
    this.owner = new Int32Array(triangleCount).fill(-1);
    this.groups = new Map();
    this.counts = new Map();
    this.nextId = 1;
    this.undoStack = [];
    this.redoStack = [];
    this.edit = null;
    this.neighbours = neighbours;
    /** Triangles to redraw since the last takeChanged(): their owner, or their group's color, changed. */
    this.changed = new Set();
  }

  get painted() {
    let n = 0;
    for (const c of this.counts.values()) n += c;
    return n;
  }

  count(id) {
    return this.counts.get(id) ?? 0;
  }

  /** Groups that hold at least one triangle, oldest first. */
  list() {
    return [...this.groups.values()].filter(g => this.count(g.id) > 0).sort((a, b) => a.id - b.id);
  }

  name(id) {
    const g = this.groups.get(id);
    return g ? `${typeLabel(g.type)} ${g.id}` : '';
  }

  triangles(id) {
    const out = new Set();
    const { owner } = this;
    for (let t = 0; t < owner.length; t++) if (owner[t] === id) out.add(t);
    return out;
  }

  begin() {
    if (this.edit) throw new Error('an edit is already open');
    this.edit = { before: new Map(), meta: cloneMeta(this.groups) };
  }

  create(type = null) {
    const id = this.nextId++;
    this.groups.set(id, { id, type, fit: null, colour: groupColour(id) });
    return this.groups.get(id);
  }

  assign(t, id) {
    const previous = this.owner[t];
    if (previous === id) return false;
    if (this.edit && !this.edit.before.has(t)) this.edit.before.set(t, previous);
    this.owner[t] = id;
    if (previous >= 0) this.counts.set(previous, this.count(previous) - 1);
    if (id >= 0) this.counts.set(id, this.count(id) + 1);
    this.changed.add(t);
    return true;
  }

  /** Close the open edit; returns true when it changed anything and became an undo step. */
  commit(label) {
    const edit = this.edit;
    if (!edit) return false;
    this.edit = null;
    for (const [id] of this.groups) if (this.count(id) === 0) this.groups.delete(id);
    // the groups this edit grew, apart in color from the groups they now border (part of this undo step)
    if (this.neighbours) this.separate(new Set([...edit.before.keys()].map((t) => this.owner[t]).filter((id) => id >= 0)));

    const tris = [];
    const before = [];
    const after = [];
    for (const [t, was] of edit.before) {
      if (this.owner[t] === was) continue;
      tris.push(t); before.push(was); after.push(this.owner[t]);
    }
    const metaAfter = cloneMeta(this.groups);
    const metaChanged = JSON.stringify([...edit.meta]) !== JSON.stringify([...metaAfter]);
    if (!tris.length && !metaChanged) return false;
    this.undoStack.push({
      label,
      tris: Int32Array.from(tris), before: Int32Array.from(before), after: Int32Array.from(after),
      metaBefore: edit.meta, metaAfter,
    });
    if (this.undoStack.length > 200) this.undoStack.shift();
    this.redoStack.length = 0;
    return true;
  }

  /** Abandon the open edit, putting everything back. */
  cancel() {
    const edit = this.edit;
    if (!edit) return;
    this.edit = null;
    for (const [t, was] of edit.before) this.owner[t] = was;
    for (const t of edit.before.keys()) this.changed.add(t);
    this.groups = edit.meta;
    this.recount();
  }

  undo() {
    return this.step(this.undoStack, this.redoStack, 'before', 'metaBefore');
  }

  redo() {
    return this.step(this.redoStack, this.undoStack, 'after', 'metaAfter');
  }

  step(from, to, side, metaSide) {
    if (this.edit || !from.length) return null;
    const entry = from.pop();
    const values = entry[side];
    for (let i = 0; i < entry.tris.length; i++) {
      this.owner[entry.tris[i]] = values[i];
      this.changed.add(entry.tris[i]);
    }
    const was = this.groups;
    this.groups = cloneMeta(entry[metaSide]);
    this.redrawRecolored(was);
    this.recount();
    to.push(entry);
    return entry.label;
  }

  /**
   * Each group in `ids` whose color reads closer than COLOUR_GAP to a group it borders, to unpainted gray or to the
   * highlight takes the spare color farthest from them all; its triangles are marked for a redraw. Lowest id first,
   * so a group checks against the colors its neighbors were just given.
   */
  separate(ids) {
    const { owner, neighbours } = this;
    const borders = new Map([...ids].filter((id) => this.groups.has(id)).map((id) => [id, new Set()]));
    if (!borders.size) return;
    for (let t = 0; t < owner.length; t++) {
      const near = borders.get(owner[t]);
      if (!near) continue;
      for (let k = 0; k < 3; k++) {
        const nb = neighbours[t * 3 + k];
        if (nb >= 0 && owner[nb] >= 0 && owner[nb] !== owner[t]) near.add(owner[nb]);
      }
    }
    const recolored = new Set();
    const pool = spareColours();
    for (const id of [...borders.keys()].sort((a, b) => a - b)) {
      const group = this.groups.get(id);
      const against = [...borders.get(id)].filter((n) => this.groups.has(n)).map((n) => seen(this.groups.get(n).colour)).concat(avoidSeen);
      const own = seen(group.colour);
      if (against.every((s) => gap(own, s) >= COLOUR_GAP)) continue;
      // each id starts the search at its own place in the pool and takes the first color clear of them all, so
      // recolored groups spread over the pool; the farthest alone is nearly always a deep tone
      let best = null;
      let bestGap = -1;
      for (let i = 0; i < pool.length; i++) {
        const c = pool[(i + id * 7) % pool.length];
        let least = Infinity;
        for (const s of against) least = Math.min(least, gap(c.seen, s));
        if (least >= CLEAR_GAP) { best = c; break; }
        if (least > bestGap) { bestGap = least; best = c; }
      }
      group.colour = [...best.rgb];
      recolored.add(id);
    }
    if (recolored.size) for (let t = 0; t < owner.length; t++) if (recolored.has(owner[t])) this.changed.add(t);
  }

  /** Mark for a redraw the triangles of every group whose color differs from its color in `was` (undo, redo). */
  redrawRecolored(was) {
    const ids = new Set();
    for (const [id, g] of this.groups) {
      const old = was.get(id);
      if (old && old.colour.some((v, i) => v !== g.colour[i])) ids.add(id);
    }
    if (ids.size) for (let t = 0; t < this.owner.length; t++) if (ids.has(this.owner[t])) this.changed.add(t);
  }

  recount() {
    this.counts.clear();
    for (const id of this.owner) if (id >= 0) this.counts.set(id, this.count(id) + 1);
  }

  takeChanged() {
    const out = this.changed;
    this.changed = new Set();
    return out;
  }
}
