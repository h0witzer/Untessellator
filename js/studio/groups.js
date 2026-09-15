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
 */

const TYPE_LABELS = {
  plane: 'Plane', cylinder: 'Cylinder', cone: 'Cone', sphere: 'Sphere', torus: 'Torus',
};

export function typeLabel(type) {
  return TYPE_LABELS[type] ?? 'Freeform';
}

/**
 * A colour per group id: golden-angle hues, stepping round the amber band the highlight
 * owns (Onshape marks geometry in amber, so a group must never read as selected).
 */
export function groupColour(id) {
  let hue = (id * 137.508 + 200) % 360;
  if (hue > 22 && hue < 62) hue = (hue + 45) % 360;
  const s = 0.5, l = 0.62;
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs(((hue / 60) % 2) - 1));
  const m = l - c / 2;
  const [r, g, b] = hue < 60 ? [c, x, 0] : hue < 120 ? [x, c, 0] : hue < 180 ? [0, c, x]
    : hue < 240 ? [0, x, c] : hue < 300 ? [x, 0, c] : [c, 0, x];
  return [r + m, g + m, b + m];
}

const cloneMeta = (groups) => new Map([...groups].map(([id, g]) => [id, { ...g, colour: [...g.colour] }]));

export class GroupStore {
  constructor(triangleCount) {
    this.owner = new Int32Array(triangleCount).fill(-1);
    this.groups = new Map();
    this.counts = new Map();
    this.nextId = 1;
    this.undoStack = [];
    this.redoStack = [];
    this.edit = null;
    /** Triangles whose owner changed since the last takeChanged(), for incremental redraws. */
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
    this.groups = cloneMeta(entry[metaSide]);
    this.recount();
    to.push(entry);
    return entry.label;
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
