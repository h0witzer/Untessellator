# Contributing to stlToStep

STL is the worst way to hand someone a part. It has no faces, no edges and no dimensions, only
triangles. stlToStep turns the mesh back into the B-rep it came from. It recovers exact planes,
cylinders, cones, spheres and tori, uses B-splines only where nothing simpler fits, and knits them
into a closed STEP solid. It runs in the browser, nothing is uploaded, and it's MIT.

Commercial converters now sell this per conversion, on credits. We want it free, local and
*measurably* better. "Measurably" is where you come in. Every claim in this repo comes with a
number on held-out parts, and every change is judged by one.

This should work with whatever engineering software you use. Today it ships as a standalone web
app and a command line. A SolidCraft plugin and an Onshape app are the next front ends, all on the
same engine.

Help is welcome at any size, from a one-line fix to owning a whole area. This page covers how to
run the code, how it's laid out, the rules that aren't negotiable, how to measure a change, and
where help is needed most.

## If you also contribute to SolidCraft, read this first

SolidCraft's rule 2 forbids reading GPL, LGPL or AGPL code. This repo contains or loads LGPL code:

| What | Where | Licence |
|---|---|---|
| Form Studio UI kit | `vendor/form-studio-ui/` | LGPL-2.1-or-later |
| OpenCASCADE, through opencascade.js 2.0 | `node_modules/opencascade.js`; the browser loads it from unpkg | LGPL-2.1-only |
| OpenCASCADE, through OCP (`cadquery-ocp`) | the Python environment for `ml/` | LGPL-2.1 |

Everything else is our own MIT code. Some of it is written against OpenCASCADE's API:

- `js/knit/build.js` and `js/knit/occt.js` call OCCT directly.
- `js/knit/repair.js` drives them.
- `interpolateChain` in `js/knit/geometry.js` reproduces `GeomAPI_Interpolate`'s curves. It was
  built by measuring OCCT's output, not by reading its source.
- `js/brepBuilder.js`, the first version's sewing exporter, calls OCCT. Only `test/emitTruth.mjs`
  still uses it.
- Most kernel-facing files in `ml/` import OCP: `solid_rebuild.py`, `occ_tessellate.py`,
  `mesher_bake.py` and others.

Whether reading these files is allowed under rule 2 is for SolidCraft's maintainers to decide. Ask
them before you open these files.

There's plenty of work that touches none of it:

- segmentation: `js/paintCore.js`, `js/autoPaint.js`, `js/regionGrow.js`, `js/edgeWall.js`
- the models and their training in `ml/`
- the fitters: `js/gaussMap.js`, `js/surfaceFitter.js`, `js/freeformFit.js`, `js/nurbs.js`
- the knitter's kernel-free parts: `js/knit/mesh.js`, `js/knit/axes.js`, `js/knit/preflight.js`
- the scoring harnesses
- wasm bindings for solidcore's Rust crates (see "Where help is needed")

If you contribute something back to SolidCraft from here, send it as a method plus measurements,
with no code from `vendor/form-studio-ui`.

## Get it running

You need Node.js. It's tested on 24; older versions may work.

```sh
git clone https://github.com/h0witzer/stlToStep && cd stlToStep
npm install          # onnxruntime-node and opencascade.js, for the Node harnesses
npm start            # then open http://localhost:3000 in Chrome or Edge
```

In the studio, load a mesh, run **Auto-paint**, then **Export STEP**. The first export downloads the
OpenCASCADE wasm build. It's large, and the browser caches it afterwards.

To run the same thing without a browser, use the studio's code and default settings headless:

```sh
node test/studioExport.mjs out/ part.stl     # writes out/part.step and out/part.json (the report)
```

The Python side (`ml/`) needs Python 3.10+ and numpy. Training needs torch. Anything that reads STEP
needs `cadquery-ocp` (OCP), and the gmsh bake needs gmsh. [ml/README.md](ml/README.md) walks through
the stages.

## How it works

```
mesh ─► repair ─► paint groups ─► fit + type each group ─► make relations exact ─► knit ─► STEP
        meshRepair  paintCore,     gaussMap, freeformFit,    knit/axes.js            knit/
                    autoPaint      groupType                                         (repair loop,
                    (models help                                                     facet fallback)
                     choose triangles)
```

**Segmentation is the hard part.** Fitting a surface to the right triangles is nearly exact, so
almost every failure traces back to the wrong triangles. Models help choose them, and they're
optional.

**The knitter takes topology from the mesh and geometry from the fits.** Each boundary between two
groups becomes an edge on both fitted surfaces. OCCT then judges the result, a repair loop fixes
what it rejects, and fallback tiers (hybrid, planar, faceted) make sure a closed solid always comes
out. The report says which tier it reached.

| Path | What it is |
|---|---|
| `index.html`, `js/studio/` | The studio: paint face groups on a mesh, export a STEP solid. `js/studio/report.js` builds the Report button's report file and prefilled issue |
| `js/paintCore.js`, `js/paintTool.js` | One click becomes one face group. `paintCore` has no model runtime and runs the same in Node and in the browser, so the harnesses measure what ships |
| `js/autoPaint.js` | Paints a whole part, best seed first |
| `js/regionGrow.js`, `js/edgeWall.js`, `js/cutFill.js`, `js/mergeRegions.js`, `js/freeformMerge.js` | Growth, learned face boundaries, min-cut fills, crumb merging, freeform grouping |
| `js/gaussMap.js`, `js/surfaceFitter.js`, `js/freeformFit.js`, `js/nurbs.js` | Fitters: plane, cylinder, cone, sphere and torus, plus a B-spline where nothing analytic fits |
| `js/groupType.js`, `js/seedNet.js`, `js/*Features.js` | The optional models' inputs and decisions |
| `js/meshCore.js`, `js/meshRepair.js`, `js/stlLoader.js` | Load, weld, repair |
| `js/knit/` | The knitter, run in a worker. `axes.js` makes shared axes and tangency exact, and `preflight.js` predicts trouble before the kernel runs |
| `models/` | Shipped ONNX models, each with a `.json` sidecar |
| `ml/` | Python: ABC download, bakes, training, and the reference solid rebuild `solid_rebuild.py` |
| `test/` | Scoring harnesses and small fixtures. Each file's header says what it measures and how to run it |
| `docs/ml-experiment-scope.md` | The lab notebook: every experiment, dated, with its numbers. Read it before re-running an idea |
| `vendor/form-studio-ui/` | The LGPL UI kit, unmodified |

## Rules that aren't negotiable

1. **The core never depends on a model.**
   - A model may only propose which triangles belong together, plus a type hint.
   - Surfaces always come from the fitters, and topology from the knitter.
   - With `models/` deleted, the studio still has to paint and export.
2. **Recover the CAD surface, not the mesh.**
   - Fitters and typing are judged by how close they come to the true STEP surface. A lower mesh
     residual never justifies a regression against the truth.
   - Real relationships are made exact: shared axes, tangency, equal radii. Values are never rounded
     to nice numbers.
3. **Tolerances come from evidence.**
   - Use an empty gap measured in the data, or a published kernel tolerance: OCCT 1e-7 mm, ACIS
     1e-6 mm, Parasolid 1e-5 mm.
   - Never pick a tolerance that turns "close" into "exact". We're the writer, so every flaw in the
     file is ours, and different readers fail on it in different ways.
4. **Measure, then claim.**
   - A change in behaviour ships with before/after numbers from the gate that covers it, on held-out
     parts, plus the harness that produced them.
   - Grade against an independent truth (the true STEP, a second reader), never against the old
     code's output.
5. **Held-out stays held out.**
   - `ml/fetch_showcase.py` writes a hash list that the bake refuses to ingest.
   - The Onshape eval set excludes every model that appears in any bake.
   - Never train on an eval set, and never tune on one you report.
6. **Licences.**
   - Permissive dependencies are fine. Say in the PR which one you're adding and why.
   - Never copy code from:
     - next.BREP.io (its licence takes the copyright of modifications)
     - ReverseCAD (noncommercial)
     - DeFillet (AGPL)
     - JUniForm (GPL)
     - any other copyleft or noncommercial project
   - Learning from their ideas and their measured behaviour is fine.
7. **Patents.**
   - US 12,229,885 claims choosing intersection points by thresholding distances at sampled points.
     Find points only by root solves (Newton, sign change), and drop regions only by proof. Never add
     a "sample A, keep the points within ε of B" step.
   - US 11,907,617 matters if fillet corners (vertex blends) ever get their own patches.
   - When reporting a patent problem, give the number, the claim element and the route around it.
8. **Data never goes in git.**
   - ABC models stay their creators' copyright (Onshape Terms of Use 1.g.ii).
   - Game-asset rips are evaluation only (see [docs/game-assets-scope.md](docs/game-assets-scope.md)).
   - The STEP Tools stress set is All Rights Reserved.
   - `test/fixtures/` holds only small parts we may redistribute. Ask before adding one.
9. **Keep one behaviour.** Replace old behaviour rather than adding a flag to toggle it. Unreleased
   formats get no compatibility switches and no bridges.
10. **The engine is JavaScript, and it stays that way unless a measurement says otherwise.** It runs
    unchanged in the browser and in Node, with no build step.
    - Using an existing Rust crate through wasm is fine, such as solidcore's `ssi`, because that
      reuses proven code.
    - Porting our own working code to another language needs a measured reason, stated in the PR.
      One example would be a host where JavaScript is too slow.
11. **`vendor/form-studio-ui/` is never edited.** Where the studio needs different behaviour, build
    it in `js/studio/` from the kit's exports.

## Measuring a change

| Question | Harness |
|---|---|
| Is every true face found in one piece, on the right surface, with no spurious extras? | `test/surfaceScore.mjs`, on a segmentation from `test/autoSeed.mjs --dump-seg` |
| Does one click fill the right face? | `test/fillEval.mjs` |
| Do the painted groups become a closed, valid, fully analytic solid? | `test/knitRebuild.mjs` for many parts, `test/knit.mjs` for one |
| Are shared axes, tangency and equal radii exact? | `test/axisScore.mjs` |
| Does preflight flag the exports that go wrong? | `test/exportRisk.mjs` |
| Does it survive the damage real STLs carry? | `test/meshDamage.mjs`, then `test/studioExport.mjs` |

The data comes from [ABC](https://deep-geometry.github.io/abc-dataset/), a million Onshape models
with their true B-reps:

- `ml/fetch_abc.py` downloads and verifies chunks.
- `ml/stage1_bake.py` and `ml/mesher_bake.py` tessellate them.
- `ml/export_fill_eval.py`, `ml/export_abc_obj_eval.py` and `ml/export_fit_truth.py` write the eval
  sets and the true surfaces.

**Test on more than one mesher.** The same parts are meshed by OCCT, gmsh, a NURBS sampler and a
game-style mesher, because models learn a mesher's triangle pattern. A change has to hold on all of
them. The 300-part Onshape set (ABC's own Onshape exports) is the real-exporter gate.

**Make builder gates deterministic.** The repair loop has a time budget, so solids depend on machine
load. Compare builder changes with `--budget 100000` and a few workers.

**Real parts come in as conversion reports.** The studio's flag button opens an issue labelled
`conversion report`, with a report file attached by the user. `node test/reportReplay.mjs <report.zip>`
reads one. It prints every group that went wrong and the step it went wrong at: the model's seed and
prediction, the typing, the fit, the export check, or the kernel and repair loop. When the mesh is
included, it rebuilds the export locally. Turn each shareable mesh that fails into a fixture, with a
test that measures the fix.

**Known friction:** the scripts default to `D:/abc-data/...` paths, and the bakes take hours of CPU.
Ask Derek for the prepared eval sets rather than rebuilding them, and don't publish them. Making the
data root configurable is the first item below.

## Where help is needed most

Ranked, each with what "done" means.

1. **One data root and a one-step eval.** Replace the hard-coded `D:/abc-data` paths with one
   setting. Make a single command that runs the standard gates and prints the tables. Done: a new
   contributor gets the Onshape-set numbers from one command.
2. **A standalone web deployment.** The studio is live on GitHub Pages
   ([h0witzer.github.io/stlToStep](https://h0witzer.github.io/stlToStep/)), served straight from
   `main`. What's left:
   - a check in a headless browser, run on every push, that converts the fixtures
   - CDN libraries pinned, with an option to serve them ourselves so the app works offline
   - a neutral look for builds that have nothing to do with Onshape. The studio currently wears the
     Onshape-style Form Studio kit.
3. **A SolidCraft plugin.** SolidCraft has paused work on its own kernel, so this belongs in a plugin,
   not in their kernel crate. There are two stages:
   - *Now:* SolidCraft already opens STEP with its own reader (`file.insert_step`, the MCP `open`
     tool). Build a one-step "convert this mesh and insert it" over its MCP server or CLI.
   - *Once plugins land* ([solidcraft#1221](https://github.com/tangency-labs/solidcraft/issues/1221):
     sandboxed WebAssembly components with a WIT interface that change the model only through
     engine commands): ship it as a one-shot plugin. Two questions decide how, and both get
     measured before anyone ports code:
     - **OpenCASCADE.** SolidCraft's contributors can't work on code that carries it, and its full
       build is a 50 MB wasm file. In the studio's export path, only about 1,000 of our 16,000 lines
       of JavaScript call it (`js/knit/build.js`, `occt.js`, `repair.js`). Removing it doesn't
       depend on the language: a
       STEP can be written without a kernel (analytic faces trimmed by the mesh's own boundaries),
       and items 6 and 7 replace the parts that need exact geometry.
     - **Speed in the sandbox.** JavaScript can be compiled into a Wasm component (ComponentizeJS),
       but it then runs on an interpreter, not a JIT. Run the pipeline that way on the Onshape set
       first. If it's too slow, that's the measured reason to compile the hot paths (fitters and
       knitter) from another language.
   - Done: a mesh in SolidCraft becomes an editable solid in one command, measured on the Onshape
     set through SolidCraft's own reader.
4. **Publish a head-to-head on stl2step.ai's benchmark.** They published per-part results for 200 ABC
   chunk-0009 parts ([stl2step.ai/benchmark](https://stl2step.ai/benchmark), CSV under CC BY 4.0).
   - On 17 of 40 parts we checked, OCCT at a deflection of 0.005 × the bounding-box diagonal and
     0.3 rad gives exactly their triangle counts.
   - 25 of the 200 parts are in our training bakes, so score the other 175.
   - Done: their metric (one-to-one face match at triangle IoU ≥ 0.9) and ours, both published, with
     the solids.
5. **An engineering-ready gate for every STEP we write.** Build this first among the export items,
   because items 6 and 7 are judged by it. For each part:
   - closed and valid
   - read back by a second reader that shares no code with OCCT (solidcore's `step-import` +
     `step-brep`, or cadmpeg)
   - every edge on both of its faces within its written tolerance, measured between samples
   - pcurves agree with the 3D curves
   - analytic faces stay analytic
6. **Exact transversal edges from `ssi`.** solidcore's surface/surface intersection crate scores
   118/119 on the SSI corpus, against 82 for OCCT's best mode. It needs a `wasm-bindgen` wrapper, and
   then the knitter can take its edges from it. The new code is the binding, which suits anyone who
   knows solidcore. The knitter itself stays JavaScript.
7. **Write pcurves, seams and edge tolerances properly.** Use SolidCraft's portable B-rep schema 4 as
   the model. Split closed faces: the seam curves on four closed cones were the only thing that
   stopped cadmpeg from measuring our latest analytic export.
8. **Speed: closed-form fits first.** On exact CAD meshes, a closed-form cylinder fit is about 275×
   faster than the robust fitter, at a slightly larger deviation. Cone and torus fits dominate
   auto-paint time. Keep the robust fitter as the fallback.
9. **Repair broken meshes.** Use generalized winding numbers (Jacobson et al. 2013; Barill et al.
   2018) to decide inside and outside on open or self-intersecting meshes before knitting.
10. **Features and an agent interface.**
   - Recognise holes, extrudes and revolves as features, not only faces.
   - Add an MCP server, so agents can convert and check parts.
   - The commercial tools and several new open ones offer both; we offer neither.

Say what you're taking on in an issue before starting anything big, so two people don't build the
same thing.

## Pull requests

- Branch from `main` and open the PR against `main`.
- Put the numbers in the description: a before/after table from the gate that covers the change. If
  it touches segmentation, check every mesher and the Onshape set. If it touches the builder, use
  the deterministic knit gate.
- Commit the harness or the flag that reproduces your table.
- Name any new dependency and its licence.
- Commit no data, no meshes from third parties and no trained weights from data you can't share.
- Match the surrounding code:
  - plain ES modules, with no build step and no framework
  - each file opens with a header saying what it is and why, including the measurement that decided
    it
  - Python runs OCCT work one part per process (see `ml/isolated_pool.py`), because the kernel can
    take the process down
- Commit messages say what changed, in the imperative, for example "Ship the message-passing fill
  refiner (runs/r3mp)".

## Credits

stlToStep began as a fork of [CNCKitchen's stlTexturizer](https://github.com/CNCKitchen/stlTexturizer),
and its file loader still carries that code (see [LICENSE](LICENSE)).
