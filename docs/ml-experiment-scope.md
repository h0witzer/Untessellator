# Scoping a learned segmentation front-end

Status: **research notes, nothing built.** Sources fetched 2026-09-10/11.

Target: a learned facet-segmentation proposer covering the *full* surface vocabulary —
including torus, surfaces of revolution, extrusions and B-splines — feeding the existing
analytic fitters. Numbers below are quoted from papers and dataset docs, not estimated;
anything inferred is marked as such.

## Why this is a segmentation problem, not a "mesh to STEP" problem

Two independent measurements agree.

**Ours** (`js/regionGrow.js` header, commit 1b6ff72): recoverable ceiling 62/77 faces;
crease segmentation 24; region growing 26. Every one of those 62 faces has >=10
supporting facets at both tessellation densities, so the gap is algorithmic, not
informational. Handed the right facets, our fitters recover a 0.5 mm cylinder radius
to six significant figures.

**Point2CAD** (CVPR 2024): with *ground-truth* segmentation their pipeline reaches
residual error 0.011 and point coverage 0.968. Swap in a learned backbone (HPNet) and
surface F-score drops to 0.742. Segmentation is the bottleneck, fitting is not.

## This does not make the core depend on a model

Point2CAD's five stages are (1) segmentation, (2) surface fitting, (3) clipping,
(4) pairwise intersection for edges, (5) edge intersection for corners — and **only
stage 1 is learned**. Their words: *"we split the reconstruction process into steps and
only use neural methods where necessary."* They report the method is backbone-agnostic,
swapping ParSeNet, HPNet and ground-truth segmentation interchangeably.

**Design rule:** the model may only emit a segmentation proposal
(`{facetIds, primitiveType}`) behind the same interface crease fill and `paintFrom`
already use in `regionGrow.js`. It never emits geometry, surface parameters, or
topology. If the model is absent, the tool works exactly as it does today.

---

# Part 1 — Dataset selection

## Verdict: ABC, and only ABC

ABC is the only corpus that labels faces by **surface type** across the full vocabulary.

### ABC surface taxonomy (from the paper's supplementary material)

> `type` — **Plane, Cylinder, Cone, Sphere, Torus, Revolution, Extrusion, BSpline, Other**

That is the complete target vocabulary, torus and revolves included. Per-patch, the
feature file gives:

| Field | Meaning |
|---|---|
| `type` | one of the nine above |
| `vert_indices` | mesh vertex indices in the patch (0-indexed) |
| `vert_parameters` | (u,v) parameters for those vertices |
| `face_indices` | **mesh face indices in the patch (0-indexed)** |
| type-specific params | full analytic parameters — see below |

`face_indices` is the line that matters: it is a direct **triangle → B-rep face**
mapping. That is precisely what `oracle.mjs` reconstructs by hand for one model, and it
ships for free with every model in the corpus.

The type-specific parameters are complete enough to supervise fitting too, not just
classification:

- **Torus** — `location`, `x_axis`, `z_axis`, max radius, min radius, with the stated
  parameterisation `p(u,v) = l + (rmax + rmin·cos v)(cos u·x + sin u·y) + r·sin v·z`
- **Revolution** — "a curve is rotated around" an axis; `location` + axis + profile curve
- **Extrusion** — "a curve is extruded"; `direction` + profile curve
- **BSpline** — `u_degree`, `v_degree`, `poles` (2D control point array), `u_knots`,
  `v_knots`, `weights` (for rational NURBS)

So a torus fitter can be supervised and validated directly, and B-spline patches come
with their real control net rather than an approximation.

### Scale and availability

- ~1,000,000 models with STEP, Parasolid, STL, meta.
- **~750,000 of those also have OBJ, features and statistics** — the feature file is the
  one carrying labels, so ~750k is the true usable corpus, not 1M.
- Distributed in chunks of 10,000 models per data type.
- Pre-made OBJ meshes at 512 / 1024 / 2048 vertices, uniform and curvature-adaptive.
- Licence: copyright stays with model creators, use under Onshape Terms of Use 1.g.ii.
  **Check this before anything ships**, particularly for a trained-weights release.

### Why the published benchmarks are not good enough — you were right

| Benchmark | Vocabulary | Verdict |
|---|---|---|
| ANSI (HPNet) | plane, sphere, cylinder, cone only | too limited, as you said |
| ABCParts (ParSeNet, HPNet) | sphere, plane, cone, cylinder, open/closed B-spline | **no torus** |

ParSeNet states its vocabulary explicitly: *"Each patch is labeled with a type, one of:
sphere, plane, cone, cylinder, open/closed B-spline patch."* The word "torus" appears
**zero times** in the HPNet paper and only twice in ParSeNet, neither time as a
supported type.

This has a consequence worth being clear-eyed about: **adding torus, Revolution and
Extrusion puts this past the published baselines, so the IoU tables are no longer an
apples-to-apples target.** They become a sanity floor on the 4-6 classes they share, not
a score to beat. The upside is that the fillet/round faces those papers drop are exactly
the faces `paint.mjs` recorded failing on, and the ones that dominate real printed parts.

## Fusion 360 Gallery: wrong labels, but keep it on the shelf

Its 8 per-face labels are **modeling operations, not surface types**: `ExtrudeSide`,
`ExtrudeEnd`, `CutSide`, `CutEnd`, `Fillet`, `Chamfer`, `RevolveSide`, `RevolveEnd`.
No geometric surface-type field per face. So it cannot replace ABC.

It is still worth knowing about for two reasons:

1. It ships `.seg` (per-triangle label) and **`.fidx` (per-triangle → B-rep face index)**
   alongside watertight, manifold OBJ meshes, plus 42,912 STEP files. Since
   `test/parseStep.mjs` already reads surface types out of STEP, surface-type labels
   could be *derived* here — `.fidx` gives triangle → face, STEP gives face → type.
2. `Fillet` and `Chamfer` as explicit classes are a direct attack on the tangent-junction
   problem. The `tangent-fillet` fixture fails today because a fillet has ~0 dihedral
   against its neighbours; a model that can say "this is a fillet face" solves that
   without a crease threshold at all.

Sizes: 35,680 models (30,459 train / 5,399 test), meshes near 2,500 edges, point clouds
at 2,048 samples.

## Data hygiene — budget real time for this

The ABC paper is candid about its own corpus:

> "there is a small percentage of imperfect models with broken boundaries,
> self-intersecting faces or edges, as well as duplicate vertices. In addition to that,
> there are many duplicate models, and especially models that are just made of single
> primitives such as a plane"

Required filters before training: deduplicate, drop single-primitive and trivially small
models, drop broken boundaries, and enforce a minimum patch count. Expect to discard a
meaningful fraction. The `statistics` YAML per model is there to make this cheap.

## Class balance — MEASURED

The ABC paper shows the patch-type distribution as a figure with no numbers, so Stage 0
measured it. First pass, 22,617 models / 1,582,809 patches (chunks 0–2):

| type | % of patches | % of models with ≥1 | verdict |
|---|---|---|---|
| Plane | 52.94% | 98.78% | |
| Cylinder | 29.77% | 85.86% | |
| BSpline | 5.44% | 17.73% | healthy |
| **Torus** | 3.86% | **22.46%** | abundant — better than feared |
| Extrusion | 3.38% | 6.61% | clustered: few models, many patches each |
| Cone | 2.97% | 22.31% | |
| Sphere | 1.29% | 10.43% | |
| **Revolution** | **0.24%** | **2.05%** | the binding constraint |
| Other | 0.10% | 1.18% | deferred, see below |

**Torus is in nearly a quarter of all models.** The class this direction exists to
cover, and the one ParSeNet and HPNet do not support at all, turns out to be cheap to
sample for. Revolution is the real problem at 106 usable models per chunk — roughly 19
chunks to reach a 2,000-model quota.

Tori in mechanical parts are usually **fillets**: numerous, but tiny in area and facet
count. Area-weighted or point-level metrics will hide failure on exactly the class that
motivated this. Report **per-class** face recovery, never the mean.

### `Other` is deliberately deferred

`Other` is usually a nontrivial fillet out of Onshape, but it is a catch-all and can be
almost anything, so each distinct element under it is a separate classification fight.
It needs ~43 chunks to reach quota against ~19 for Revolution — more than double the
corpus for the least useful class.

This is a deferral, not an exclusion. `Other` patches occur inside training models
regardless, so they can only be left untargeted, which means the model needs sane
**fallback** behaviour rather than a prediction. The converter already has one: the
NURBS fallback in `surfaceFitter.js` for when no primitive fits. Treat `Other` as
"route to fallback", not as a class, and revisit after something useful ships.
`stage0_report.py --no-quota` defaults to it.

### Two corrections Stage 0 forced

1. **A chunk yields ~7,539 feature files, not 10,000.** About 25% of model directories
   ship no `feat` file. Every "N chunks = 10N models" estimate was 25% optimistic.
2. **42.2% of parts share a structural fingerprint** (9,550 duplicates in 1,979 groups),
   so the usable pool was 12,972 of 22,617. ABC's duplicate problem is materially worse
   than the paper's "many duplicate models" implies — and that is before geometric
   dedup, which Stage 1 will catch more of.

Mitigations, unchanged: stratified sampling on rare types (as ABCParts was built by
selecting models with at least one B-spline), per-class loss weighting derived from the
**selected** subset rather than the corpus histogram, and per-class reporting.

## Rescuing the rare classes: what is actually label-preserving

Revolution is scarce. The obvious move is to augment the models we have. The trap is
that **most transforms silently destroy the type label**, and a corrupted label is worse
than a missing one.

| transform | effect on labels |
|---|---|
| rotation, translation, mirror, **uniform** scale | preserves every type. Free, but adds pose diversity only — not shape diversity |
| **non-uniform** scale, general | **destroys most types.** Cylinder → elliptic cylinder (no longer a `Cylinder`), Sphere → ellipsoid, Cone and Torus likewise |
| non-uniform scale **along a revolution's axis** | **preserves `Revolution`** — rotational symmetry about the axis survives; only the profile curve changes |
| affine scale along *or* across an extrusion direction | **preserves `Extrusion`** — the profile becomes a different curve; it is still a linear extrusion |

The lucky part: the two classes we are short on are the two with the most headroom.

The unlucky part, and why this cannot be applied naively: **a transform applies to the
whole model, not one patch.** A model containing a Revolution almost always also
contains planes, cylinders and fillets, so axial scaling to rescue Revolution corrupts
the majority classes in the same file. Rescuing 441 models by poisoning 18,000 is a bad
trade, so whole-model non-uniform scaling is ruled out.

### Tessellation-space augmentation is free — do it regardless

Because Stage 1 re-tessellates from STEP, the *same* surface at different densities and
deflection settings yields genuinely different meshes with identical labels. That
teaches the invariance that matters most here — surface type is independent of
triangulation — at zero labelling risk, and it is the axis the WatchStand measurement
says is real (Coarse 24 vs Medium 26). Build this first.

### The stronger move for Revolution: synthesise it

Augmentation multiplies examples, not diversity. `Revolution` is defined by a
*structural property* — a profile curve rotated about an axis — so 441 seed models in
new poses teaches the network that those 441 shapes are revolutions, not what makes
something a revolution. That is the classic small-sample overfit.

We know the definition exactly, so we can generate unlimited, diverse, perfectly
labelled examples: random profile (segments, arcs, splines), random axis, built as real
B-reps through OCCT and tessellated by the same Stage 1 path. Same for `Extrusion`.

Precedent: **CAD-Recode** (ICCV 2025) trained on **one million procedurally generated**
CAD models as CadQuery scripts, explicitly because repository-scraped datasets are
"restricted in scale" and do not "allow for control over the design features and
patterns included in training". That is precisely our situation — Revolution is rare in
ABC not because it is hard, but because Onshape users do not revolve much.

**Proposed split:** ABC for the naturally abundant classes, synthesis for `Revolution`
and `Extrusion`.

**How to test it before committing:** hold out the *real* ABC Revolution models as an
evaluation set and train on synthesised ones. If it transfers, synthesis is validated;
if it does not, the synthetic distribution is wrong and more of it will not help. Same
structure as the showcase holdout, and measurable early.

### A caveat on the few-shot analogy

Corridor's chroma-key result came from fine-tuning a large **pretrained image model** —
it already knew what edges, hair and people looked like, so they were teaching a narrow
mapping on top of a strong prior, and their synthetic cases varied actual content. Here
there is no pretrained prior for "surface of revolution"; the network learns geometry
from scratch. Few-shot works when a prior does the heavy lifting. Without one, diversity
has to come from the data — which is the argument for synthesis over pose augmentation.

### Roughness ladder

ParSeNet's published recipe is the starting point: sample points on the surface, add
noise uniform in **[-0.01, 0.01] along the normal**, perturb normals uniformly in
**[-3, 3] degrees**. Extend with the axes that matter for STL specifically and that no
scan-oriented paper covers: tessellation density, non-uniform density, decimation,
remeshing that destroys the original triangulation, a real STL export round trip, and
edge erosion (see Part 3). Training only on clean OCCT tessellation produces a model
that scores well and fails on a downloaded file.

---

# Part 2 — Training setup on the 3090

## The key architectural decision: go mesh-native, not point-cloud

Every published baseline samples a point cloud and then spends network capacity
*reconstructing* what a mesh already tells you exactly. This is the single biggest
advantage available and it is worth designing around rather than inheriting their code.

Evidence for how much normals alone are worth, from the papers: HPNet 91.3 → 94.2 seg
IoU on ANSI and 78.1 → 85.2 on ABCParts when normals are added to points; ParSeNet
71.32 → 81.20 on ABCParts. That is 3–10 IoU points for *estimated* normals. We have
exact per-facet normals, exact facet adjacency and exact dihedral angles, already
computed in `meshCore.js` and `faceGrouper.js`.

**Proposed representation:** nodes = triangles, edges = the dual graph (each triangle has
exactly 3 neighbours in a manifold mesh). Node features: centroid, unit normal, area,
plus per-edge dihedral angle. This is a plain graph net over a *static* adjacency.

Three consequences, all favourable:

1. **No KNN.** DGCNN's cost centre is the dynamic k-NN edge-feature tensor,
   `B × N × k × 2C`. At N=8192, k=20, C=64, fp32 that is ~84 MB per sample per layer
   (*my arithmetic, not quoted*). With a fixed 3-neighbour dual graph the same tensor is
   `B × N × 3 × 2C` ≈ 12.6 MB — roughly 7× cheaper.
2. **It exports to ONNX cleanly.** Static adjacency is a `gather`. Dynamic KNN exports
   but is slow in ONNX Runtime Web, which is the deployment target.
3. **Labels need no transfer.** ABC's `face_indices` are already per-triangle.

Reference model sizes from the ABC paper's own benchmark table: PointNet++ 1,402,180
parameters, DGCNN 1,724,100. These are *small* — ~7 MB at fp32, and the 3090's 24 GB is
nowhere near the binding constraint. Activation memory is.

## Pipeline stages

### Stage 0 — index the corpus (CPU, hours)

Download `feat` + `stat` chunks only (not STEP yet). Scan the YAML, build a manifest:
model id → patch-type histogram, patch count, quality flags. Output one Parquet/SQLite
file. **Deliverable: the measured class distribution**, which decides the sampling plan.

### Stage 1 — bake the dataset (CPU, the long pole)

This is where the real engineering time goes, not in training.

**Decision: re-tessellate from STEP rather than use the provided OBJ.** The pre-made
meshes are 512/1024/2048 vertices — far coarser than real parts (one WatchStand blob
alone is 4,512 triangles). The paper confirms the B-rep *"can be resampled at any desired
resolution."* Re-tessellating means tessellation density becomes a controlled
augmentation axis instead of a fixed property of the data, and OCCT's `BRepMesh` yields
the triangle → face mapping directly, so labels stay free. Cross-check against the feat
file's `face_indices` on a sample to validate the pipeline.

Use Stage 0's manifest to tessellate only the stratified subset — not 750k models.

Bake to a packed binary (memmap-able `.npy`), **never parse YAML in the training loop**.

Rough sizing (*my arithmetic*): at 16k facets/model, fp16 features + int32 adjacency +
labels ≈ 800 KB/model. 25k models × 3 density variants ≈ 30 GB on disk. Load by memmap
with a random crop to 8,192 facets per sample.

### Stage 2 — train (GPU)

| | |
|---|---|
| Framework | PyTorch, CUDA 12.x, **bf16 autocast** (Ampere supports it natively) + TF32 |
| Batch | 8–16 at 8,192 facets — measure, then raise until VRAM binds |
| Optimizer | Adam (both ParSeNet and HPNet use it) |
| Loss | per-facet embedding (triplet, per ParSeNet) + per-facet type CE, class-weighted |
| Clustering | mean-shift on embeddings, as both papers do — CPU post-process |
| Throughput | *estimate only*: ~10–20 min/epoch at 25k models; 100 epochs ≈ overnight to a weekend |

Neither ParSeNet nor HPNet publishes GPU type or wall-clock training time, so treat the
duration row as an extrapolation to be replaced by a measurement after 100 steps. For
calibration, the ABC paper reports one baseline taking *"3 days of training on the
highest resolution"* — on 2019 hardware, for a different task.

**Windows note:** PyTorch `DataLoader` multiprocessing is spawn-based and painful on
native Windows. Run training under **WSL2 with CUDA passthrough**, or set
`persistent_workers=True` and expect worker startup cost. The baked memmap format makes
this much less painful either way.

### Stage 3 — score

Primary metric is `oracle.mjs` **face recovery, reported per surface type**. Secondary is
point-level seg IoU, for literature comparison only.

**Do not conflate the two.** Published seg IoU is point-level mean IoU over matched
segments; ours is a face count ("did we recover this face at all"). A model at 90 seg IoU
can miss every small face — and small faces are exactly the WatchStand conical-face
failure `paint.mjs` recorded (0/4, flat across seedSize 12–80 × tolScale 0.5–1.5).

## Fitters that need to exist for the expanded vocabulary

Segmentation is worthless without somewhere to send the facets. Current coverage is
plane / cylinder / cone / sphere.

| Type | Work needed |
|---|---|
| Torus | New analytic fit: centre, axis, rmaj, rmin. Straightforward; ABC supervises it directly |
| Revolution | Segment → find axis → unwrap to (r, θ, z) → fit 2D profile curve → OCCT surface of revolution |
| Extrusion | Segment → find direction → project along it → fit 2D profile → OCCT linear extrusion |
| BSpline | Hardest. See Point2CAD's INR below before reaching for a large model |

Revolution and Extrusion are more tractable than they look: once segmented, both reduce
to *finding one axis/direction plus fitting a 2D curve*, which is a much smaller problem
than free-form surface fitting.

For B-splines, Point2CAD's implicit neural representation is worth copying before any
large generative model is considered: a single-hidden-layer autoencoder with a 2D (u,v)
bottleneck, mixed SiLU and sinusoidal activations, Adam for 1000 steps with 50 warm-up
and linear decay, fitted **per surface at inference time** in "a few seconds on a single
GPU." No dataset to build. It reports residual 0.002 / coverage 0.999 on freeform
patches, against ComplexGen's 0.021 / 0.938.

## Kill criterion

If the model does not beat region growing's 26/62 on a held-out model, stop. Do not tune,
do not scale up. Record the negative result in the module header the way `regionGrow.js`
records the RANSAC commit-order failure (14/77).

---

# Part 3 — the scan question

Honest read: **the mechanism transfers, the domain does not come for free.**

Nothing in the formulation is STL-specific. Facets with normals and adjacency describe a
scan mesh as readily as a printed one, and segment-then-fit-analytically is exactly what
commercial reverse-engineering tools do. Point2CAD's own framing is about "in-the-wild
objects." So the direction is real.

But fuzzing a clean STL does not simulate a scan. Four differences, roughly in order of
how much damage they do:

1. **Rounded edges.** Scanners erode sharp edges into small-radius blends. This is the
   killer, because sharp edges are the primary segmentation signal — it is the entire
   reason SED-Net adds a dedicated edge branch. Every crease-angle assumption in
   `faceGrouper.js` degrades here.
2. **Holes and occlusion.** Missing data in pockets, undercuts, internal features. A
   printed STL is watertight; a raw scan is not.
3. **Registration ghosting.** Multi-view alignment error produces doubled or smeared
   surfaces — a systematic artifact with no analogue in additive vertex noise.
4. **Non-uniform density and outliers**, plus scale/unit drift.

ParSeNet's published noise recipe (sample 10K points, perturb ±0.01 along the normal,
perturb normals ±3°) covers only item 4, and it is what everyone in this literature
calls "noise." Item 1 is the one to actually build an augmentation for: deliberate edge
erosion at controlled radius.

**Recommended posture:** keep the STL cleanup case as the beachhead — it is the real
motivation, the domain is clean, and success is measurable against `oracle.mjs` today.
But hold out a small **real scan** evaluation set from day one and score it every run
without ever training on it. It costs almost nothing and it is the only way to know when
the model crosses into scan-tolerant territory rather than guessing. Add edge-erosion to
the augmentation ladder early, since it also makes the STL case more robust to
low-quality triangulation — and ABC's own STL files are described as "high-resolution but
low-quality triangulation," which is a good proxy for internet slop.

On the competitive framing: Design X's moat is not primarily algorithmic — it is a guided
manual workflow, training, and support, at a price point that assumes a professional
metrology seat. An automatic, free, browser-based tool is not competing on the same axis.
The wedge is "no install, no licence, no operator," not "better surface fitting."

---

## Sources

- ABC dataset: https://deep-geometry.github.io/abc-dataset/ — paper https://arxiv.org/abs/1812.06216, supplementary https://cims.nyu.edu/gcl/papers/2019-ABC-Dataset-Supplemental.pdf (surface taxonomy and feature-file schema)
- Fusion 360 Gallery: https://github.com/AutodeskAILab/Fusion360GalleryDataset — segmentation docs https://github.com/AutodeskAILab/Fusion360GalleryDataset/blob/master/docs/segmentation.md
- ParSeNet (ECCV 2020): https://arxiv.org/abs/2003.12181 — code https://github.com/Hippogriff/parsenet-codebase
- HPNet (ICCV 2021): https://arxiv.org/abs/2105.10620 — code https://github.com/SimingYan/HPNet
- SED-Net (SIGGRAPH 2023): https://dl.acm.org/doi/10.1145/3588432.3591522 — joint surface+edge detection; full text paywalled, numbers not captured
- Point2CAD (CVPR 2024): https://arxiv.org/abs/2312.04962 — code https://github.com/prs-eth/point2cad

### Published baselines, for reference only

Point-level seg IoU on sampled point clouds. **Not comparable to our x/62 face count**,
and measured on vocabularies that exclude torus.

| Method | Input | ANSI seg IoU | ANSI type IoU | ABCParts seg IoU |
|---|---|---|---|---|
| Efficient RANSAC | p+n | 77.61 | — | 58.15 |
| SPFN | p+n | 88.05 | 93.13 | 73.41 |
| ParSeNet | p+n | 88.57 | 95.43 | 82.14 |
| HPNet | p | 91.3 | — | 78.1 |
| HPNet | p+n | 94.2 | ~98.9 | 85.2 |
