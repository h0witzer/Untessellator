/**
 * paintClient.js — the page's side of js/paintWorker.js: fills, edge scoring, auto-paint's seed scores and the group
 * typer run there, so the viewport keeps drawing and taking input while they do.
 *
 * The worker holds one mesh at a time. Each call names the studio mesh it is for (the object studio.js keeps in
 * `mesh`), and a mesh the worker does not hold is sent first: an undo that swaps an older mesh back costs one load,
 * and its edges are scored again when next asked. Model URLs are made absolute here: the studio's are relative to
 * the page, and the worker would read them relative to js/.
 */

const absolute = (url) => (url ? new URL(url, document.baseURI).href : url);

export function createPaintClient() {
  let worker = null;
  let nextId = 1;
  let loaded = null;
  let nextKey = 1;
  const keys = new WeakMap();
  const pending = new Map();

  const start = () => {
    worker = new Worker(new URL('../paintWorker.js', import.meta.url), { type: 'module' });
    worker.onmessage = ({ data }) => {
      const call = pending.get(data.id);
      if (!call) return;
      if (data.progress !== undefined) { call.onProgress?.(data.progress); return; }
      pending.delete(data.id);
      if (data.error !== undefined) call.reject(new Error(data.error));
      else call.resolve(data.result);
    };
    // A worker that fails to load or throws outside a job: every call waiting on it fails, and the next starts anew.
    worker.onerror = (event) => {
      event.preventDefault?.();
      const error = new Error(event.message || 'the paint worker stopped');
      for (const call of pending.values()) call.reject(error);
      pending.clear();
      worker.terminate();
      worker = null;
      loaded = null;
    };
  };

  const post = (message, transfer = [], onProgress = null) => new Promise((resolve, reject) => {
    if (!worker) start();
    const id = nextId++;
    pending.set(id, { resolve, reject, onProgress });
    worker.postMessage({ ...message, id }, transfer);
  });

  const keyOf = (target) => {
    if (!keys.has(target)) keys.set(target, nextKey++);
    return keys.get(target);
  };

  /** The worker's key for `target`, loading it first unless the worker holds it already. */
  const ensure = (target) => {
    const key = keyOf(target);
    if (loaded !== target) {
      loaded = target;
      const geo = target.geometry.index ? target.geometry.toNonIndexed() : target.geometry;
      const positions = Float32Array.from(geo.attributes.position.array);   // a copy: the geometry keeps its own
      post({ op: 'load', key, positions }, [positions.buffer]).catch((error) => {
        console.error('paint worker: could not load the mesh', error);
        if (loaded === target) loaded = null;
      });
    }
    return key;
  };

  return {
    /** Score `target`'s edges with the boundary model at `url` (the worker keeps the last scores per mesh load). */
    scoreEdges: (target, url, wallTypes) => post({ op: 'scoreEdges', key: ensure(target), url: absolute(url), wallTypes }),

    /** autoPaint.coherentScores for `target`: a Float64Array, one per triangle. */
    coherence: (target) => post({ op: 'coherence', key: ensure(target) }),

    /**
     * paintTool.paintAt from `seed` on `target`, with `wall: true` for the learned wall (used when the edges are
     * scored). The result's region is an Int32Array.
     */
    fill: (target, seed, opts) => post({ op: 'fill', key: ensure(target), seed,
      opts: { ...opts, modelUrl: absolute(opts.modelUrl), refineUrl: absolute(opts.refineUrl) } }),

    /** The group typer's {probs, threshold} for `n` rows of `features`. */
    typer: (url, features, n) => post({ op: 'typer', url: absolute(url), features, n }),

    /** freeformMerge.js over `owner` (copied, not changed) for groups `only`: {owner: Int32Array, freeform: ids}. */
    freeformMerge: (target, owner, only) => post({ op: 'freeformMerge', key: ensure(target), owner: Int32Array.from(owner), only: [...only] }),

    /**
     * Auto-paint's new groups typed and fitted: `groups` [{id, tris: Int32Array, type, freeform}], the group typer at
     * `typerUrl` (null: none). Resolves to [{id, type, typedBy, typer, fit}]; onProgress({phase, done, of}).
     */
    typeGroups: (target, groups, typerUrl, onProgress) => post({ op: 'typeGroups', key: ensure(target), groups,
      typerUrl: absolute(typerUrl) }, [], onProgress),
  };
}
