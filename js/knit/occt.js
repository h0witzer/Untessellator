/**
 * knit/occt.js — OpenCASCADE for the knitter: opencascade.js 2.0 (OCCT 7.6-era, the "full" build), in
 * the browser from a CDN and in Node from node_modules, so the harnesses run what the studio runs.
 *
 * 1.1.4, the build the old exporter loads, binds neither GeomAPI_Interpolate nor
 * TopTools_IndexedDataMapOfShapeListOfShape, which the knitter needs; 2.0's full build has both
 * (TColgp_HArray1OfPnt still has no binding, so knit/geometry.js interpolates the B-splines itself).
 */

import { loadSsi } from './ssi.js';

export const OCCT_VERSION = '2.0.0-beta.b5ff984';
const CDN = `https://unpkg.com/opencascade.js@${OCCT_VERSION}/dist/`;
const isNode = typeof process !== 'undefined' && !!process.versions?.node && typeof window === 'undefined'
  && typeof importScripts === 'undefined';

let loading = null;

/** The OCCT module, loaded once. */
export function loadOcct({ onStatus } = {}) {
  if (!loading) {
    loading = (async () => {
      // ssi (exact edges, knit/ssi.js) loads beside the kernel; without it the knitter builds its edges as before
      await loadSsi().catch(() => null);
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

/**
 * STEP AP214 text for a shape. names: one per solid of `shape` (in its order): the file is then one product, `part`,
 * with each solid a MANIFOLD_SOLID_BREP of that name (XCAF sub-shape names, which CAD programs show as body names).
 * Without names, the plain writer, as before.
 */
export function writeStep(oc, shape, { names = null, part = 'Part' } = {}) {
  if (names) return writeNamed(oc, shape, names, part);
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

function writeNamed(oc, shape, names, part) {
  const solids = [];
  for (const ex = new oc.TopExp_Explorer_2(shape, oc.TopAbs_ShapeEnum.TopAbs_SOLID, oc.TopAbs_ShapeEnum.TopAbs_SHAPE); ex.More(); ex.Next()) solids.push(ex.Current());
  // the sub-shape names are written only when this is set before the document and the writer exist
  oc.STEPCAFControl_Controller.Init();
  oc.Interface_Static.SetIVal('write.stepcaf.subshapes.name', 1);
  const doc = new oc.Handle_TDocStd_Document_2(new oc.TDocStd_Document(new oc.TCollection_ExtendedString_1()));
  const tool = oc.XCAFDoc_DocumentTool.ShapeTool(doc.get().Main()).get();
  const top = tool.AddShape(shape, false, false);
  oc.TDataStd_Name.Set_1(top, new oc.TCollection_ExtendedString_2(part, true));
  solids.forEach((s, i) => {
    const label = tool.AddSubShape_1(top, s);
    if (!label.IsNull()) oc.TDataStd_Name.Set_1(label, new oc.TCollection_ExtendedString_2(names[i] ?? `Body ${i + 1}`, true));
  });
  const writer = new oc.STEPCAFControl_Writer_1();
  writer.SetNameMode(true);
  // multi null: one file (an empty string writes each product to a file of its own)
  if (!writer.Transfer_1(doc, oc.STEPControl_StepModelType.STEPControl_AsIs, null, new oc.Message_ProgressRange_1())) throw new Error('STEP transfer failed');
  const path = `/tmp/knit-${Date.now()}.step`;
  if (writer.Write(path) !== oc.IFSelect_ReturnStatus.IFSelect_RetDone) throw new Error('STEP write failed');
  const text = oc.FS.readFile(path, { encoding: 'utf8' });
  try { oc.FS.unlink(path); } catch { /* the file is in memory; nothing to clean */ }
  writer.delete();
  try { doc.get().Close(); } catch { /* not bound in every build */ }
  return text;
}
