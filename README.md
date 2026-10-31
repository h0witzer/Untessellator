# stlToStep

**Turn meshes back into real CAD.** stlToStep reads an STL, OBJ or 3MF file and writes a STEP
solid made of real faces:

- exact planes, cylinders, cones, spheres and tori
- B-splines only where nothing simpler fits

Open the result in any CAD program and you can select a hole, measure a diameter, or push a face.

It runs in your browser: nothing is uploaded, there's no per-conversion fee, and it's MIT.

Inspired by [CNCKitchen's stlTexturizer](https://github.com/CNCKitchen/stlTexturizer), which this
project began as a fork of.

## Why

Most parts on the internet were designed in CAD, then flattened into triangles to be printed or
shared. A typical "STL to STEP" converter wraps those triangles in a STEP file, and you get
thousands of flat faces you can't edit. The real job is recovering the surfaces the part was
designed with.

Commercial tools now sell that job by the credit. We think it should be a free capability that
works no matter what engineering software you use.

## Where it runs

There's one engine with several front ends:

| Front end | Status |
|---|---|
| **Standalone web app** (`studio.html`) | Works today, tied to no CAD system. It's static files plus libraries from public CDNs, so any static host should serve it. A tested deployment recipe is still wanted. |
| **Command line** (`test/studioExport.mjs`) | Works today. It runs the studio's own code and default settings headless. |
| **SolidCraft plugin** | Wanted. SolidCraft's WebAssembly plugin interface is still being designed ([tangency-labs/solidcraft#1221](https://github.com/tangency-labs/solidcraft/issues/1221)). Until it lands, SolidCraft opens our STEP files with its own reader. |
| **Onshape app** | In progress. The studio already looks and handles like Onshape (through the vendored Form Studio kit), but the integration with Onshape itself isn't built yet. |

## Quick start

```sh
git clone https://github.com/h0witzer/stlToStep && cd stlToStep
node server.js
```

Open <http://localhost:3000> in Chrome or Edge (`npm start` does the same as `node server.js`). The
page has to be served; opening `studio.html` directly from disk leaves it blank.

1. Drop an `.stl`, `.obj` or `.3mf` file onto the viewport, or open one from the toolbar.
2. Press **Auto-paint** (or <kbd>P</kbd>). Every face of the part gets a face group with a surface type
   and a fit. <kbd>Esc</kbd> stops it.
3. Fix anything it got wrong by hand:

   | Input | Action |
   |---|---|
   | Left-drag | Fill |
   | <kbd>Alt</kbd> + drag | Erase |
   | <kbd>Shift</kbd> + drag | Paint triangles directly |
   | Right-drag | Orbit |
   | Middle-drag, or <kbd>Ctrl</kbd> + right-drag | Pan |

4. Press **Export STEP**. The report says what you got: a fully analytic solid, or which regions
   fell back to flat facets and why.

The first export downloads the OpenCASCADE WebAssembly build. It's large, and the browser caches it
afterwards.

**Without a browser:**

```sh
npm install
node test/studioExport.mjs out/ part.stl     # writes out/part.step and out/part.json
```

## How it works

```
mesh ─► repair ─► paint face groups ─► fit + type each group ─► make relations exact ─► knit ─► STEP
```

1. **Repair.** Fix the defects that stop a solid from closing: holes, duplicate triangles, flipped
   triangles, unwelded vertices.
2. **Paint.** Split the mesh into one group per true face. This is the hard part. Small learned
   models help choose which triangles belong together, but they never decide a surface, and the
   pipeline is designed to work without them.
3. **Fit and type.** Give each group the simplest surface that holds every vertex: plane, cylinder,
   cone, sphere or torus. A B-spline is used only where none of those fits.
4. **Make relations exact.** Shared axes, tangency, equal radii and mirror symmetry become exact.
   Values are never rounded to nice numbers.
5. **Knit.** Take the topology from the mesh and the geometry from the fits. A geometry kernel
   checks the result. A repair loop fixes what it rejects, and regions that still can't be built
   fall back to facets, so the export still closes.

## What to expect

- **Best on meshes exported from CAD,** where every vertex lies on the true surface: your own
  exports, and most functional parts from Printables, MakerWorld or Thingiverse.
- **Scans, sculpts and decimated meshes** give mostly B-spline and faceted faces. Scan-to-CAD is a
  long-term goal.
- **Many exports still fall back to facets.** Dense exports from CAD systems usually come out fully
  analytic. Coarser meshes often don't, and a part can come out as a valid closed solid with every
  face a facet. The export report says which you got.
- **You get faces, not a feature history.** There are no sketches or timeline yet.
- **Large meshes are slow.** Parts with tens of thousands of triangles can take minutes to paint and
  export.

## How good is it?

We measure instead of claiming. The test sets are:

- held-out parts from the [ABC dataset](https://deep-geometry.github.io/abc-dataset/), each meshed
  four ways (OpenCASCADE, gmsh, NURBS sampling and a game-style mesher)
- 300 real Onshape exports

Each part is scored against its true CAD surfaces and rebuilt solid. Every experiment is written up,
dated and with its numbers, in [docs/ml-experiment-scope.md](docs/ml-experiment-scope.md). A public
head-to-head against other converters is next.

## Contributing

There's plenty to do: a SolidCraft plugin, a deployable standalone build, exact edges from
solidcore's intersection crate, faster fitters, feature recognition. Start with
[CONTRIBUTING.md](CONTRIBUTING.md). It covers setup, the code layout, the rules, how a change is
measured, and a ranked list of where help is needed.
Then open an issue saying what you're picking up.

## Dependencies

| Library | Version | Licence | Used for |
|---|---|---|---|
| [three.js](https://threejs.org/) | 0.170.0 | MIT | Mesh loaders |
| [fflate](https://github.com/101arrowz/fflate) | 0.8.2 | MIT | Reading 3MF archives |
| [onnxruntime-web](https://onnxruntime.ai/) | 1.22.0 | MIT | Running the optional models |
| [opencascade.js](https://github.com/donalffons/opencascade.js) | 2.0.0-beta | LGPL-2.1 | The geometry kernel behind the STEP export |
| Form Studio UI kit (`vendor/form-studio-ui/`) | vendored | LGPL-2.1-or-later | The studio's frame, toolbar, view cube and renderer |

The browser loads the libraries from jsDelivr and unpkg. The Node harnesses use `onnxruntime-node`
and `opencascade.js` from `npm install`.

## Licence

MIT; see [LICENSE](LICENSE). The file loader (`js/stlLoader.js`) still carries code from
stlTexturizer, so its copyright notice stays in LICENSE alongside ours. The two LGPL components
above keep their own licences.
