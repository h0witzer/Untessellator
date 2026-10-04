/**
 * knit/occt.js — OpenCASCADE for the knitter: opencascade.js 2.0 (OCCT 7.6-era, the "full" build), in
 * the browser from a CDN and in Node from node_modules, so the harnesses run what the studio runs.
 *
 * 1.1.4, the build the old exporter loads, binds neither GeomAPI_Interpolate nor
 * TopTools_IndexedDataMapOfShapeListOfShape, which the knitter needs; 2.0's full build has both
 * (TColgp_HArray1OfPnt still has no binding, so knit/geometry.js interpolates the B-splines itself).
 */

export const OCCT_VERSION = '2.0.0-beta.b5ff984';
const CDN = `https://unpkg.com/opencascade.js@${OCCT_VERSION}/dist/`;
const isNode = typeof process !== 'undefined' && !!process.versions?.node && typeof window === 'undefined'
  && typeof importScripts === 'undefined';

let loading = null;

/** The OCCT module, loaded once. */
export function loadOcct({ onStatus } = {}) {
  if (!loading) {
    loading = (async () => {
      if (isNode) {
        const { default: init } = await import('opencascade.js/dist/node.js');
        return init();
      }
      onStatus?.('Downloading the geometry kernel (about 50 MB, cached after the first time)…');
      const { default: factory } = await import(/* @vite-ignore */ `${CDN}opencascade.full.js`);
      onStatus?.('Starting the geometry kernel…');
      return new factory({ locateFile: (p) => (p.endsWith('.wasm') ? `${CDN}opencascade.full.wasm` : p) });
    })();
    loading.catch(() => { loading = null; });
  }
  return loading;
}

/** STEP AP214 text for a shape. */
export function writeStep(oc, shape) {
  const writer = new oc.STEPControl_Writer_1();
  const st = writer.Transfer(shape, oc.STEPControl_StepModelType.STEPControl_AsIs, true, new oc.Message_ProgressRange_1());
  if (st !== oc.IFSelect_ReturnStatus.IFSelect_RetDone) throw new Error('STEP transfer failed');
  const path = `/tmp/knit-${Date.now()}.step`;
  if (writer.Write(path) !== oc.IFSelect_ReturnStatus.IFSelect_RetDone) throw new Error('STEP write failed');
  const text = oc.FS.readFile(path, { encoding: 'utf8' });
  try { oc.FS.unlink(path); } catch { /* the file is in memory; nothing to clean */ }
  writer.delete();
  return text;
}
