# Handoff: intersection, B-rep topology and naming work, for STLtoSTEP

Written 2026-10-30 for the agent picking up STLtoSTEP. It carries what was built and learned in
Derek's (h0witzer) recent work on tangency-labs/solidcore and tangency-labs/solidcraft, where it
overlaps this repo. Everything linked here is public. Nothing here comes from driving or reading
commercial CAD.

This doc describes the code at commit 02f4d58 (2026-09-15, branch `gauss-map-classifier`). Your
tree has work from 2026-10-05 or later that this doc's author never saw. Where section 1 or 6 says
what the code does today (`solid_rebuild.py`'s edges, `brepBuilder.js` sewing, the 29/31 and 23/31
counts), check it against your tree first and trust your tree.

## 1. What STLtoSTEP is, and why this applies

STLtoSTEP takes a mesh in, repairs it where needed, fits surfaces to regions, and writes the STEP
file. The goal is clean, engineering-ready CAD. We are the writer, so every flaw in the file is ours:
a reader downstream will not fix it, and different readers fail on it in different ways.

`ml/solid_rebuild.py` method `topo` takes topology from the mesh and geometry from the fitted
surfaces. Measured on 31 fully analytic ABC parts: 29/31 recreated with true surfaces, 23/31 with
fitted ones. The `sew` method that `js/brepBuilder.js` still uses closes 0/31.

How `topo` builds an edge today (`build_topo`, `project_onto`):

- Each mesh boundary vertex between two patches is pushed onto both surfaces by a min-norm
  Gauss-Newton step. This finds a point on the intersection near the vertex. It is a root solve,
  not a distance threshold (this matters, see section 5).
- A 3D curve is interpolated through those points (`GeomAPI_Interpolate`). Plane against plane gets
  an exact line; every other pair gets the interpolant.
- No pcurves are written. Between samples the curve lies near both surfaces, not on them, so every
  reader has to rebuild the trims and decide for itself how far off is acceptable.
- Corners are pushed onto every surface that meets there at once.

The solidcore work solved these pieces: exact intersection curves with both parameter curves,
tangency told apart from crossing by proof, a B-rep record with pcurves on every edge, and an
independent STEP reader. The SolidCraft work adds face and edge names that survive edits.

## 2. What exists

| Piece | Where | State | Numbers |
|---|---|---|---|
| `ssi`: surface/surface intersection, Rust, no dependencies, MIT OR Apache-2.0 | [solidcore `crates/ssi`](https://github.com/tangency-labs/solidcore/tree/main/crates/ssi) (PR #638) | merged 2026-10-25 | 118/119 on the ssi-corpus at 1e-6 |
| `solidcore-ssi`: solidcore surfaces → ssi patches, plus an edge builder | solidcore `crates/solidcore-ssi` (PR #946) | merged 2026-10-25; opt-in, not wired into booleans | 118/119 through the adapter |
| `step-import` (clean-room AP203/AP214 B-rep reader), `step-brep` (builds solidcore bodies from it), `step-export` (ISO 10303-21 writer) | solidcore `crates/step-*` (not our code) | merged | not measured by us |
| `nurbs-grid-mesh`: grid tessellator for dense NURBS faces, with a deviation check | solidcore `crates/nurbs-grid-mesh` (PRs #633, #1158) | merged | see below |
| Persistent naming proposal | [solidcraft #7990](https://github.com/tangency-labs/solidcraft/pull/7990), `docs/design/persistent-naming-proposal.md` | merged 2026-10-30 as the plan of record | — |
| Sheet metal RFC: the folded sheet's surfaces are the master, the flat pattern is derived | [solidcraft #6748](https://github.com/tangency-labs/solidcraft/issues/6748), branch `h0witzer/solidcraft:fix-6748-develop` | open, WIP | — |
| SSI corpus, 119 cases | [jyviko/ssi-corpus](https://github.com/jyviko/ssi-corpus) (MIT/Apache) | public | — |

How engines score on the corpus at 1e-6 (measured 2026-10-23):

| Engine | Pass (of 119) |
|---|---|
| ssi | 118 |
| OCCT 8.0.1 `BRepAlgoAPI_Section` | 82 |
| SMLib 1.0.1 | 82 |
| OCCT 8.0.1 `IntTools_FaceFace` | 69 |
| OCCT 8.0.1 `GeomInt_IntSS` | 68 |

- All three OCCT modes return nothing on 31 cases: tangent curves and isolated touches.
- Section collapses sphere loops of radius ≤ 1e-4 to a point.
- The one case ssi misses, `y23_fig1d_two_quads`, fails on every engine.

Timings must be interleaved per case in one pass. The same machine drifts 2-3x between runs.

nurbs-grid-mesh, measured:

- A degree-7, 318 × 281 test surface: worst error 0.25 mm with 0% of triangles over tolerance. SMLib
  on the same surface: 21.5% over tolerance, worst 4.65 mm.
- 55 STEP Tools AP203 files (13,347 faces): 99.7% mesh without error, 94.9% clean.
- Known gap: a trim loop that crosses a periodic seam is not supported yet.

SolidCraft `develop` (upstream) defines a portable B-rep, schema 4, that is a good model for what
STLtoSTEP should write:

- Pcurves on every edge: exact on planes, fitted cubics elsewhere.
- A seam edge gets two pcurves on its face.
- A canonical face/edge/vertex order, which persistent names rely on.

## 3. How ssi works

The entry point is `ssi::intersect_bodies(body_a: &[Patch], body_b: &[Patch], &Options) ->
Result<Intersection, Error>`.

- A `Patch` is a rational B-spline that may carry its exact analytic surface (`Patch::analytic`).
  `ssi::shapes::{plane, cylinder, cone, sphere, torus}` build these. They are the same five types
  STLtoSTEP fits.
- `Intersection` holds:
  - `curves`: each a `Curve { closed, points }`. Every point carries `p` and its `(patch, u, v)` on
    both surfaces, so pcurves come for free.
  - `points`: isolated touches.
  - `vertices`: where branches cross or meet.
- `Error::Tangent` means a contact the solver could not prove. It is never approximated.
- `Options` defaults: tolerance 1e-11, link 1e-7, resolution 1e-5.

The method, in order:

1. **Closed forms.** Plane against sphere, or against a degree-2 form that cuts it in an ellipse.
   The radius is computed as `√((R−|d|)(R+|d|))` with a compensated `d`.
2. **Field path.** One side's implicit equation is composed with the other patch's Bézier pieces,
   which gives a Bernstein polynomial `g` on that patch's own (u, v):
   - Cells where `g` has one sign are dropped by proof.
   - Monotone cells are swept as bracketed roots.
   - Critical points are proved one per cell (Hessian sign plus gradient winding). These give touch
     points and crossings.
   - Tangent curves are proved valleys of `g`.
   - A pole of the traced patch is divided out exactly.
3. **Exact implicit forms** for B-spline patches up to degree 10, from a composition-matrix null
   vector and proved by composition. With these, NURBS quadrics and tori take the field path.
4. **Fallback.** Subdivision with the Sederberg-Meyers loop test, then marching. Near tangencies,
   local Taylor fields with proved error bounds.

To try it from Python without writing bindings, use the corpus runner contract: `<runner>
<case.json> <mode> <tol> <out.json>`. The case JSON lists surfaces as `{"type": "plane" |
"cylinder" | "cone" | "sphere" | "torus" | "nurbs", ...}` (see any file in `ssi-corpus/cases/data`).
Emitting one case per adjacent face pair from `exportPatchFits` output is the zero-binding way in.

For the browser, solidcore builds for `wasm32`. A `wasm-bindgen` wrapper for ssi does not exist
yet.

## 4. Lessons that transfer

1. **Tangency is where engines fail, and fitted surfaces make it worse.** A fitted fillet and its
   neighbor are never exactly tangent, so their intersection is 0, 1 or 2 curves depending on noise.
   Do not intersect near-tangent fitted pairs:
   - Make tangency a constraint of the fit, then build the edge as the exact contact curve
     (cylinder on plane → line; torus on cylinder → circle).
   - Use SSI for transversal edges.
   - An `Error::Tangent` from ssi tells you which edges need the constrained route.
2. **Exact, not loose.** Recognize analytic surfaces exactly, and never choose a tolerance that
   turns "close" into "exact". If a definition needs a number:
   - Measure for an empty gap in the data.
   - Ground it in published kernel tolerances: OCCT 1e-7 mm, ACIS 1e-6, Parasolid 1e-5 mm,
     CATIA 1e-3 mm.
   - Ask Derek once, naming it as the definition question it is.

   For a writer this sets the bar. An edge written with a tolerance looser than the strictest
   reader's will open in one CAD system and not in another.
3. **Write the pcurves.** Many STEP files in the wild have none (`rear` and `moon_buggy_asm` in the
   STEP Tools AP203 set below), and every reader then rebuilds trims its own way. That is a common import failure, and it
   is one we can prevent. When building a face's loops, use the method that worked on all 776 faces
   of a hard mill-turn part:
   - Per face, fit a closed 2D loop to the edges' preimages, certified within tolerance by a
     Bernstein bound on `S(c(t)) − e(t)`.
   - Snap corners to shared 2D points so the loop closes exactly.
   - Link boundary uses into cycles by vertex id after dropping edges used twice (slits).
   - Close wrapped chains on periodic surfaces modulo the period.
   - Never snap a split point to a noise root. Doing so moved points by 1e-4 and recursed to
     depth 12.
4. **Seams and poles cause most bugs.**
   - Unwrap periodic coordinates while solving, then fold back into the domain.
   - Watch for a pole that sits on a seam.
   - Keep sphere poles away from face boundaries.
   - `solid_rebuild.py` already handles two of these traps: `seam_away`, and removing the
     natural-bound wires that `ShapeFix` adds.
5. **A big error is your own bug first.** A 41 mm "vertex gap" turned out to be a pairing bug, not
   a file defect.
6. **Check tools against an independent count, not against the old code.** Example: compare a
   classifier with a polygon parity count. Grading new code against old output only reproduces the
   old errors. For STEP output, that means checking with a second reader, not only OCCT.

## 5. Patents and licenses

- **US 12,229,885** (SSI; inventor Alyn Rockwood; Zoo; to about 2044) claims choosing intersection
  points by thresholding distances at sampled points. Find points by Newton or sign-change roots
  only, and drop regions only by proof. `project_onto` is a root solve and is fine. Do not add a
  "sample A, keep points within ε of B" step anywhere. Zoo's terms also forbid benchmarking their
  engine.
- **US 11,907,617** (n-sided ribbon blends; Rockwood; to about 2029) matters if fillet corners
  (vertex blends) ever get their own patches. Search Rockwood's name early in any surface or
  intersection idea.
- **How to report a patent.** Give the number, the claim element, and the route around it, with at
  most one line of caveat. The default response to a live patent is to find an unclaimed method
  that is better.
- **STLtoSTEP's license.** STLtoSTEP is MIT.
  - ssi, solidcore and nurbs-grid-mesh are MIT/Apache and can be vendored with notices.
  - The Form Studio UI kit in `vendor/form-studio-ui` is LGPL-2.1 and stays unmodified there.
  - ImplicitCAM is Derek's private AGPL project: use its lessons (section 4), never its code.
- **Contributing back to SolidCraft.** Its CONTRIBUTING rule 2 bars reading LGPL code. Anything
  offered there from here goes as method plus measurements, and contains no code from
  `vendor/form-studio-ui`.

## 6. Recommended next work, ranked

1. **An engineering-ready gate for every written file, automated.** Build it first: it is how the
   items below get judged. Per part:
   - Closed and valid by `BRepCheck`.
   - Read back by OCCT's STEP reader and by solidcore's `step-import` + `step-brep`, a second reader
     that shares no code with OCCT.
   - Every edge lies on both of its faces within the tolerance written for it, measured between
     samples, not at them.
   - Pcurves agree with their 3D curves.
   - Volume and face areas against the truth STEP where one exists (the ABC parts).
   - Analytic faces stay analytic: count B-spline faces where the fit was a plane, cylinder, cone,
     sphere or torus.

   Report today's `topo` output through the gate before changing anything.
2. **Exact transversal edges.** For every adjacent pair of analytic fits, get the intersection from
   ssi:
   - Pick the branch the mesh boundary chain follows.
   - Cut it at the corners: solve each corner on all of its surfaces, then on the curve.
   - Write the 3D curve plus both pcurves.

   Gate: the 29/31 and 23/31 counts must not drop. The edge deviation must improve under item 1.
3. **Tangency as a fit constraint** for fillet-like patches. The edge becomes the exact contact
   curve (lesson 1).
4. **Write edge tolerances and pcurves into the STEP**, following schema 4: a seam edge gets two
   pcurves, and faces/edges/vertices go out in canonical order.
5. **Browser export.** `js/brepBuilder.js` still sews. Port `topo`, then decide between ssi via
   wasm and OCCT intersectors in opencascade.js. OCCT misses the tangent cases (section 2).

Later:

- **Repair.** Generalized winding numbers decide inside and outside on broken meshes (holes,
  self-intersections, soups) without repairing first: Jacobson et al. 2013, and Barill et al. 2018
  for fast evaluation. This fits before the mesh pairing in `topo`, whose shell is open wherever the
  input mesh is open.
- **Persistent names** for Studio's painted groups once they have a history. Resolve a reference in
  this order: exact name → every piece of a split → a lineage guess from the same feature → lost.
  Never fall back to "nearest by geometry". Ordinals come from the tool's own construction order,
  never from kernel face order (#7990 changes 1, 4, 5).
- **Display and grading.** nurbs-grid-mesh for B-spline faces. ssi's `surface3::NurbsSurf::closest`
  gives a certified closest point for the true maximum distance from mesh to surface. It returns
  `None` after 20,000 heap pops, so check for `None`.

## 7. Working with Derek

- Lead with what to do next and show results: numbers, before/after tables, screenshots. Don't hand
  over multi-flag command recipes; offer a one-step launcher if one is needed.
- Measure before proposing anything upstream.
- A question of the form "does X meet rule Y" is answered from code and evidence, never put to
  Derek. Ask only about preferences and priorities.
- New kernel work goes in Rust; the browser gets it through wasm.
- Keep one behavior. Unreleased formats get no compatibility flags and no bridges.

## 8. Setting up on your machine

- **ssi and its runner.** Clone [tangency-labs/solidcore](https://github.com/tangency-labs/solidcore)
  and run `cargo build --release -p ssi`. The runner lands in `target/release/ssi_corpus_runner`.
  solidcore pins its own Rust toolchain.
- **Corpus.** Clone [jyviko/ssi-corpus](https://github.com/jyviko/ssi-corpus). To score:
  `SSI_ENGINE=ssi-rs SSI_RUNNER=<path to runner> python -I <ssi-corpus>/scoring/score.py fs 1e-6`.
  The scorer needs numpy; `-I` keeps Python from importing stray files from the working directory.
- **STEP stress set.** Fetch the 55 files from <https://www.steptools.com/docs/stpfiles/ap203/>.
  They are All Rights Reserved: keep them out of git.
- **Ask Derek for anything else.** These live only on Derek's other machine: the corpus score
  history, the standalone ssi repo, and SolidCraft naming change 4 (#5198, unpushed).
