# Game assets: a target market and a litmus test

Status: **measuring, routes scoped, little built.** Started 2026-09-13.

Two reasons to care about meshes that never touched a CAD package:

- **Market.** People want to turn game characters and props into cosplay builds, and
  keep asking how to open an FBX in parametric CAD and work with its surfaces. The tools
  they get pointed at today are Rhino (QuadRemesh → SubD → ToNURBS) and Fusion (quad
  mesh → T-spline). Both produce smooth NURBS and neither recovers analytic features.
- **Litmus.** A hand-modelled mesh is the far end of the mesher axis
  [`ml-experiment-scope.md`](ml-experiment-scope.md) opened. OCCT, NURBS-then-OCCT and
  gmsh are all algorithms, and the seeded-fill model already fails on the second one it
  meets (Cylinder 0.877 → 0.043 on gmsh). An artist is not an algorithm at all.

These meshes also break the solid-output assumption. They arrive as dozens of
overlapping shells, with open seams and duplicated faces. That changes what "success"
means here, and it is part of the test.

---

## Ground rules: licensing

The repository is **public and MIT-licensed**. Ripped game assets are copyrighted, and
extracting them usually breaks the game's EULA.

- **Rips are evaluation-only.** They live outside the repo, under
  `D:/abc-data/game/rips`. They are never trained on and never redistributed.
  `.gitignore` carries backstops (`test/fixtures/masterchief.stl`, `game-assets/`,
  `*.fbx`). A rule in a README would not survive `git add .`; an ignore rule does.
- **Anything trained on comes from explicitly permissive sources.** Checked 2026-09-13:

| Source | Licence | Formats | Notes |
|---|---|---|---|
| [Poly Haven](https://polyhaven.com/license) | CC0 | blend, fbx, gltf, usd | Licence page explicitly welcomes ML use. 521 models: 70 tools, 97 industrial, 31 electronics. Hand-modelled, quad-dominant. API: `api.polyhaven.com/assets?t=models`. |
| [Kenney](https://kenney.nl/support) | CC0 | fbx, glb, obj | Low-poly, pure triangles, so no quad test. The Blaster Kit is 40 models. |
| [Quaternius](https://quaternius.com/faq.html) | CC0 | blend, fbx | Has sci-fi guns, tanks and spaceships packs. Not yet fetched. |
| Blender Studio | mostly CC-BY, some CC-BY-ND | blend | **Excluded.** The production files need a subscription, and ND forbids derivatives. |

Downloaded so far: `D:/abc-data/game/cc0/kenney/blaster-kit` (40 models) and
`D:/abc-data/game/cc0/polyhaven` (5 hard-surface props, FBX, plus OBJ exports from
Blender that keep polygons).

---

## Measuring without ground truth: `test/meshReport.mjs`

`oracle.mjs` and `rescore.py` score against a STEP; these meshes have none.
`meshReport.mjs` measures what can still be measured:

- **Topology** on the pipeline's own weld: components, open edges, non-manifold edges,
  duplicate triangles.
- **Pattern**: triangle-area spread and crease groups per triangle.
- **Polygons** (OBJ only): the share of triangles, quads and n-gons.
- **Behaviour** from seeded fills: predicted types, fit residuals, and *bogus fits*,
  meaning fits whose own residual says the predicted surface is not there.

Model: v2ctrl with curvature columns zeroed, as it was trained (see Pitfall below). The
bogus threshold, relative residual > 1e-2, is provisional.

| mesh | origin | tris | shells | open / non-manifold edges | quads | top predicted types (60 seeds) | residual p50 | bogus fits |
|---|---|---|---|---|---|---|---|---|
| `masterchief.stl` | game rip | 56,190 | 49 | 756 / 0 | n/a (STL) | Cone 37, BSpline 9, Revolution 7 | 9.2e-2 | nearly all |
| Kenney Blaster Kit, 39 OBJs | CC0, low-poly | median 366 after dedupe | median 15 | 2,059 / 137 total | 0% | Plane 1,279, Cone 240 | planes ~1e-15 | 207; **417 plane fits failed** |
| `Drill_01` | Poly Haven | 2,926 | 22 | 362 / 0 | 87% | Cone 39, Plane 7 | 1.0e-1 | 38 of 50 |
| `pipe_wrench` | Poly Haven | 10,746 | 5 | 30 / 0 | 100% | Cone 45, Plane 6 | 3.0e-2 | 38 of 53 |
| `flathead_screwdriver` | Poly Haven | 2,300 | 3 | 52 / 0 | 90% | Cone 29, Plane 14, Revolution 9 | 2.3e-2 | 26 of 48 |
| `vintage_video_camera` | Poly Haven | 18,682 | 119 | 1,333 / 106 | 81% | Cone 23, Plane 13, Cylinder 11 | 2.3e-2 | 30 of 48 |
| `WetFloorSign_01` | Poly Haven | 228 | 1 | 0 / 0 | 68% | Plane 35, Cone 13, Extrusion 11 | 2.0e-4 | 18 of 39 |
| `3DBenchy.stl` | CAD reference | 225,706 | 1 | 29 / 0 | n/a | Revolution 15, BSpline 12, Torus 10 | 2.8e-2 | — |

### Findings

1. **Cone is what the type head says when it does not know.** On every non-CAD mesh
   except the flat-faced sign, the top prediction is Cone: Master Chief, all four curved
   Poly Haven props, and Suzanne (Cone 37) from the showcase set. CAD-origin Benchy gets
   Cone 7. The fits behind those predictions are mostly bogus. The residual already
   separates the cases, with no new model: this is the basis for route 3.
2. **Duplicates depend on the export format.** Kenney's OBJs repeat about half their
   triangles with the same winding. The GLBs of the same models have none. Blender
   silently drops repeats on import, so Blender cannot be used to check for them. Every
   repeated edge becomes four-owned, `buildAdjacency` finds no neighbours, and each
   triangle ends up as its own component. Measured by parsing the raw files.
3. **Low-poly means very few facets per surface.** After dedupe, Kenney's planes are
   exact (residual ~1e-15), yet 417 of 1,279 plane fits fail outright. Those faces are one
   or two triangles. A single triangle defines a plane exactly, so this is a fitter or
   region-size rule rejecting valid input, not missing information. Not yet diagnosed.
4. **Multi-shell is the normal case, and the builder drops shells.** 49, 15 (median),
   22 and 119 shells. `brepBuilder.js` handles a sewn compound by taking the *first*
   shell. Its comment says "largest", but no size is compared, so every other body is
   discarded without a warning.
5. **The artist's polygons carry information, and every loader throws them away.**
   Poly Haven props are 68–100% quads. Binary STL has no polygons. three's `OBJLoader`
   fan-triangulates and `FBXLoader` runs earcut, and neither keeps the original
   polygon boundaries.

### Pitfall: curvature must match training

v2ctrl is the control arm of the curvature A/B test, trained with columns 10-11 zeroed.
`rescore.py --zero-curvature` defaults on for that reason. `seedNet.buildFeatures` fills
those columns whenever `attachCurvature` has run, which every JS harness does because
`fitTorus` needs it.

The first `meshReport` run fed curvature to v2ctrl and read `masterchief.stl` as
**Plane 49 of 60**. Zeroed as trained, the same seeds read **Cone 37, Plane 1**. The
"plane attractor" conclusion drawn from the first run was an artefact and is withdrawn.

`meshReport` now zeroes the model's columns by default (`--curvature off`) while the
fitters keep real curvature. **`test/showcase.mjs` still has the mismatch**, so treat any
v2ctrl numbers from it with suspicion.

---

## Game-art degradation of ABC: `ml/game_bake.py`

Game meshes have no labels, but ABC does. `game_bake.py` changes what a game-art
pipeline changes and keeps every triangle's face id true. It writes the same bake format
as `mesher_bake.py`, so `train.py --extra-bake` and `rescore.py --bake/--intersect`
consume it unchanged.

| op | what it imitates | label-exact because |
|---|---|---|
| `lowpoly` | low-poly modelling: OCCT at angular deflection 0.6–1.0 rad | vertices stay on the true surface |
| `quadflip` | quads triangulated by the exporter: swap shared diagonals within a face | no vertex moves |
| `jitter` | hand-placed vertices: tangential noise, a fraction of the shortest edge | small; off by default |
| `unweld` | UV and mirror seams: pull one side's seam vertices back into its face, within the tangent plane | planar faces stay exactly planar (checked) |
| `shells` | overlapping parts: rigid translation of whole faces | a rigid move keeps the surface type |

**Invariants checked** on real parts (scratchpad `check_game_ops2.py`):
- 0 label mismatches across 6 parts and all densities.
- `quadflip` keeps every face's vertex set, keeps planar face area exactly, and reverses
  no normals. An earlier version, tested against the pair's mean normal only, folded
  9 of 5,024.
- `unweld` and `shells` open seams only *between* faces; no face's own closure changes.
- `unweld` (tangent-plane version), 25 parts, 1,837 seams, 35,941 open edges added: 0
  flipped triangles, and the worst plane face's off-plane distance is unchanged (2.4e-5 of
  the diagonal before and after, which is float32 noise). The first version failed this
  check; see the ablation below.
- `shells` moves faces rigidly (displacement std 2e-15).
- `jitter` leaves the weld census unchanged.
- Cost: `quadflip` takes 6 s on a 218k-triangle part.

Gaps are `max(rel × diag, 3e-4)`, because `meshCore` welds on an *absolute* 1e-4
quantum. A purely relative gap would round back onto its twin on small parts.

### Scored against native

Setup:
- 300 validation parts, each baked four ways.
- `rescore.py --intersect`, so every column covers identical parts.
- v3canon checkpoint, curvature zeroed as trained.
- medium density, about 4,950 seeds per column.
- Accuracy against canonical labels.

| | native | `lowpoly` only | topology only | all ops |
|---|---|---|---|---|
| triangles, all densities | — | 1.70 M | 10.04 M | 1.70 M |
| membership IoU | **0.807** | 0.622 | 0.744 | **0.639** |
| overall type accuracy | **0.842** | 0.705 | 0.706 | **0.543** |
| macro type accuracy | 0.697 | 0.599 | 0.583 | 0.447 |
| Plane | 0.974 | 0.959 | 0.978 | 0.951 |
| Cylinder | 0.873 | 0.752 | 0.787 | 0.541 |
| Cone | 0.878 | 0.822 | 0.757 | 0.664 |
| Sphere | 0.919 | 0.732 | 0.623 | 0.425 |
| Torus | 0.745 | 0.423 | 0.411 | 0.161 |
| Revolution | 0.239 | 0.133 | 0.239 | 0.067 |
| Extrusion | 0.700 | 0.580 | 0.599 | 0.424 |
| BSpline | 0.663 | 0.423 | 0.511 | 0.355 |

All columns use the tangent-plane `unweld`. The first version's numbers are kept only in
the ablation, as "v1".

"Topology only" means `quadflip,unweld,shells` applied to the native densities.

**The two kinds of damage hurt different things, and they compound.**

- **Low-poly geometry mostly costs membership** (0.807 → 0.622). Fewer facets per face
  means a 512-facet neighbourhood spans more faces. Planes barely notice (0.959).
- **The triangulation pattern costs curved types.** `quadflip` alone takes Sphere from
  0.919 to 0.644 and Torus from 0.745 to 0.487, without moving a vertex off its surface
  (ablation below). Open seams and overlapping shells barely matter.
- **All ops together land near gmsh** (macro 0.447 against gmsh's 0.423). Planes survive
  (0.951). Cylinders fare far better than on gmsh (0.541 against 0.043), but Sphere,
  Torus, Revolution and BSpline all collapse.
- **Cone is the fallback again.** Canonical Revolution seeds are predicted Cone 51% of
  the time, which matches the Cone-dominated predictions on the real game meshes above.
  The synthetic bake and the real assets fail the same way, which is some evidence the
  bake imitates the right things.

**Ablation: each topology op alone**, on the same parts and settings. "unweld v1" is the
first version, which offset seam vertices along the face normal. "v2" pulls them back
within the tangent plane.

| | native | `shells` | `unweld` v1 | `unweld` v2 | `quadflip` | all three (v1) |
|---|---|---|---|---|---|---|
| membership IoU | 0.807 | 0.833 | 0.813 | 0.825 | **0.708** | 0.733 |
| overall type accuracy | 0.842 | 0.830 | 0.768 | 0.823 | 0.749 | 0.643 |
| macro type accuracy | 0.697 | 0.691 | 0.647 | 0.685 | 0.615 | 0.540 |
| Plane | 0.974 | 0.974 | **0.786** | 0.963 | 0.974 | 0.783 |
| Cylinder | 0.873 | 0.861 | 0.834 | 0.837 | 0.845 | 0.780 |
| Sphere | 0.919 | 0.925 | 0.881 | 0.930 | **0.644** | 0.591 |
| Torus | 0.745 | 0.718 | 0.715 | 0.714 | **0.487** | 0.406 |

- **`quadflip` is the main damage to curved types and membership.** Sphere loses 0.28
  and Torus 0.26, and Plane does not move. It swaps diagonals and nothing else, with every
  vertex still on the true surface. This is the mesher-pattern failure in its purest form:
  the type head reads OCCT's triangulation of each surface type.
- **`shells` is nearly harmless.** It even raises membership, because a detached face is
  exactly bounded by its own shell. Overlapping plates are a problem for solid output,
  not for segmentation.
- **The Plane drop under `unweld` v1 was a defect in the degradation, not a finding.**
  - *Tested hypothesis, refuted.* The idea was that open seams truncate a plane seed's
    neighbourhood. Under v1, plane-seed patches were still a median of 512 facets with
    12% on the seed's face (native: 9%), and accuracy fell evenly across every patch-size
    and on-face bucket (0.978 → 0.815 in the dominant one).
  - *A re-weld on a coarse grid was not a usable control.* A grid coarse enough to close
    the gaps also collapsed real small triangles: 362 seeds were left without a patch,
    and Plane accuracy fell to 0.655.
  - *Diagnosis.* v1 moved plane vertices off their plane. OCCT meshes planes as long
    slivers, so lifting one corner by the gap tilts the whole triangle.
  - *Confirmed by v2.* Pulling the same seams back within the tangent plane gives Plane
    0.963 and overall 0.823. **Open seams by themselves barely affect the model**, and
    neither do overlapping shells. Among topology damage, the triangulation pattern is
    what matters.

---

## Routes, in dependency order

Each route states its first step and what would stop it.

1. **Held-out evaluation set.** Done in part: `meshReport.mjs`, CC0 samples, one rip.
   Next: hash-lock the set as `fetch_showcase.py` does, so no bake can ingest it, and add
   Quaternius. Rips stay in `D:/abc-data/game/rips` and are scored, never trained on.
2. **Game-art bake as training data.** Built. Next: score the three variants. If type
   accuracy collapses the way it did on gmsh, add the bake to the tessellation-diversity
   retrain alongside `bake_nurbs` and `bake_gmsh`. Stop if accuracy on the native split
   regresses beyond the noise between v2ctrl and v3canon.
3. **Freeform routing gate.** Deterministic first: when a region's best analytic fit
   exceeds a residual threshold, send it to the NURBS fallback instead of emitting the
   predicted primitive. Calibrate the threshold on ABC by comparing the residual
   distribution of genuine faces with that of `BSpline`/`Other` faces, not on game
   meshes. A learned gate comes later, trained with CC0 game art as negatives. Stop if
   no threshold separates genuine faces from freeform ones on ABC.
4. **Mesh hygiene pre-pass.** Remove exact and back-face duplicate triangles before
   adjacency, and report non-manifold edges. This touches `meshCore.js`, so `meshcore.py`
   and `test_meshcore.py` must change in the same commit, or training and inference
   drift apart.
5. **Multi-body output.** Fix the first-shell selection. Export every shell, as a solid
   where it closes and as a surface body where it does not. Validate by re-importing the
   STEP and counting bodies. `brepBuilder.js` has no headless test today, so this needs
   one first.
6. **Polygon-preserving input.** Keep a polygon id per triangle when an OBJ or FBX has
   quads. That needs our own face grouping; `FBXLoader` cannot provide it. glTF is
   triangles by specification, so it cannot help. Use the ids as a segmentation prior:
   triangles from one quad belong together, and edge loops mark where artists put
   surface boundaries. Measure on the Poly Haven props with `meshReport`, by claimed
   coverage and bogus rate with and without the prior.
7. **Hybrid analytic + SubD/NURBS output.** Analytic faces where the gate passes; for
   the rest, fit B-spline patches over quad regions (Catmull-Clark limit surface, or
   `GeomAPI_PointsToBSplineSurface` per patch). This is the direct answer to Rhino's
   ToNURBS and Fusion's T-splines. Blocked on routes 3, 5 and 6.
8. **Head-to-head.** Run the same CC0 props through Rhino QuadRemesh → ToNURBS and
   Fusion's quad mesh → T-spline, scored by deviation from the mesh, face count, and
   share of area recovered as analytic surfaces. CC0 only, so the results can be
   published.

---

## Sources

- Poly Haven licence: https://polyhaven.com/license
- Kenney licence: https://kenney.nl/support
- Quaternius FAQ: https://quaternius.com/faq.html
- Blender Studio remixing and licences: https://studio.blender.org/remixing/
- three.js OBJLoader (fan triangulation): https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/loaders/OBJLoader.js
- three.js FBXLoader (earcut triangulation): https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/loaders/FBXLoader.js
- Rhino QuadRemesh: https://www.rhino3d.com/features/quadremesh/
- DiffusionNet, discretization-agnostic learning: https://arxiv.org/abs/2012.00888
