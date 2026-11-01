/**
 * knit/ssi.js — solidcore's `ssi` (exact surface/surface intersection, rust/ssi-wasm) for the knitter. Two
 * fitted surfaces in, the curves they meet in out, every point with its parameters on both surfaces. ssi.wasm is
 * a plain module (no imports, no glue), so it loads the same way in Node and in the export worker.
 *
 *   const ssi = await loadSsi();
 *   const { curves, points } = ssi.intersect(specA, specB);       // throws SsiError on a refusal
 *
 * A surface spec is a flat array, its kind first (rust/ssi-wasm/src/lib.rs): plane 0, cylinder 1, cone 2,
 * sphere 3, torus 4, then the frame and the patch's domain. Curves: {closed, points: [{p, a: [u, v], b: [u, v]}]},
 * the parameters on ssi's own patches of the two surfaces.
 */

const KIND = { plane: 0, cylinder: 1, cone: 2, sphere: 3, torus: 4 };

/** A pair ssi will not build, with its code ('BOOLEAN_TANGENT', 'MARCHING_STALLED', ...). */
export class SsiError extends Error {
  constructor(message) {
    super(message);
    this.name = 'SsiError';
    this.code = message.split(':')[0];
  }
}

let loading = null, ready = null;

/** The module, instantiated once (loadOcct starts it beside the kernel). */
export function loadSsi() {
  loading ??= (async () => {
    const url = new URL('./ssi/ssi.wasm', import.meta.url);
    const bytes = url.protocol === 'file:'
      ? (await import('node:fs')).readFileSync(url)
      : new Uint8Array(await (await fetch(url)).arrayBuffer());
    const { instance } = await WebAssembly.instantiate(bytes, {});
    return (ready = wrap(instance.exports));
  })();
  return loading;
}

/** The module once loaded, else null: the knitter's builds are synchronous, so they take it only when it is there. */
export const ssiNow = () => ready;

function wrap(x) {
  const put = (spec) => {
    const p = x.ssi_alloc(spec.length);
    new Float64Array(x.memory.buffer, p, spec.length).set(spec);
    return p;
  };
  return {
    KIND,
    /** The curves and isolated points surfaces a and b meet in; feature > 0 resolves down to that length. */
    intersect(a, b, feature = 0) {
      const pa = put(a), pb = put(b);
      let status;
      try {
        status = x.ssi_intersect(pa, a.length, pb, b.length, feature);
      } finally {
        x.ssi_free(pa, a.length);
        x.ssi_free(pb, b.length);
      }
      if (status !== 0) {
        const bytes = new Uint8Array(x.memory.buffer, x.ssi_err_ptr(), x.ssi_err_len());
        throw new SsiError(new TextDecoder().decode(bytes));
      }
      const out = new Float64Array(x.memory.buffer, x.ssi_out_ptr(), x.ssi_out_len()).slice();
      let i = 0;
      const point = () => { const q = { p: [out[i], out[i + 1], out[i + 2]], a: [out[i + 3], out[i + 4]], b: [out[i + 5], out[i + 6]] }; i += 7; return q; };
      const curves = [];
      for (let c = out[i++]; c > 0; c--) {
        const closed = out[i++] === 1, n = out[i++], points = [];
        for (let k = 0; k < n; k++) points.push(point());
        curves.push({ closed, points });
      }
      const points = [];
      for (let k = out[i++]; k > 0; k--) points.push(point());
      return { curves, points };
    },
  };
}
