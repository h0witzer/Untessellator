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
measured it. **Final: chunks 0–19, 150,280 models, 9,490,332 patches.**

| type | % of patches | % of models with ≥1 | after stratified selection |
|---|---|---|---|
| Plane | 55.40% | 99.07% | 46.31% |
| Cylinder | 28.51% | 82.81% | 26.33% |
| Extrusion | 4.71% | 8.76% | **9.84%** |
| BSpline | 3.79% | 13.93% | **6.28%** |
| **Torus** | 3.24% | 19.05% | **5.22%** |
| Cone | 3.08% | 19.33% | 3.19% |
| Sphere | 1.01% | 8.52% | **2.23%** |
| **Revolution** | **0.19%** | **1.64%** | 0.41% |
| Other | 0.07% | 0.80% | 0.19% (not targeted) |

**Stratified selection more than doubles every rare class's share** — Extrusion 4.71 →
9.84%, Sphere 1.01 → 2.23%, Torus 3.24 → 5.22%, BSpline 3.79 → 6.28% — at a target of
25,000 parts drawn from a usable pool of 69,680.

Torus, the class this direction exists to cover and the one ParSeNet and HPNet do not
support at all, is comfortably abundant. **Revolution is the only genuine shortage**:
1,642 usable models, 12,413 patches in the selected set. Under the seeded-fill MVP that
is far less alarming than it looks, since each patch yields many (seed, neighbourhood)
samples rather than one — measure before building the synthesis pipeline.

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

1. **A chunk yields ~7,514 feature files, not 10,000.** About 25% of model directories
   ship no `feat` file. Every "N chunks = 10N models" estimate was 25% optimistic.
2. **53.2% of parts share a structural fingerprint** — 79,891 duplicates in 10,984
   groups, leaving 69,680 usable of 150,280. Note this *rose* with corpus size (42.2% at
   three chunks, 53.2% at twenty) because more chunks expose more cross-chunk twins, so
   expect it to climb further with more data. **Over half of ABC is structurally
   duplicate**, which is materially worse than the paper's "many duplicate models"
   implies — and that is before geometric dedup, which Stage 1 will catch more of.
   Effective corpus size is roughly half the nominal count; plan downloads accordingly.

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

## Can we hotrod a pretrained model?

Yes — and the obvious candidate is the wrong one to ship.

**HPNet publishes a pretrained checkpoint**, trained on ABCParts, DGCNN-derived
backbone. That is a zero-domain-gap warm start: same task, same dataset family. Swap
the 6-class type head for our 9 and fine-tune.

But licences decide this, and they invert the ranking:

| model | licence | pretrained on | shippable |
|---|---|---|---|
| **HPNet** checkpoint | **none** | ABCParts — exact match | research only |
| ParSeNet | **none** | ABCParts | research only |
| SED-Net | **none** | ABC | research only |
| CAD-Recode | **CC-BY-NC-4.0** | 1M synthetic CAD | non-commercial only; also Qwen2-1.5B, far too large for a browser |
| UV-Net | MIT | ABC / Fusion 360 | wrong direction — consumes B-rep, which we do not have at inference |
| Sonata | Apache-2.0 | indoor scenes (PTv3) | no — depends on `spconv`, which has no browser path |
| Point2CAD | Apache-2.0 | n/a (test-time optimisation) | yes, but there are no weights to inherit |
| **Point-MAE** | **MIT** | ShapeNet objects | **yes** |
| **Point-BERT** | **MIT** | ShapeNet objects | **yes** |

"No licence" means default copyright: the code can be read and the method reproduced,
but no rights to copy, modify or redistribute are granted — so derived weights cannot
be shipped in the converter. Worth a proper look before anything ships; the point here
is only that it is a real constraint and not a detail.

### The two-track split this implies

**Track A — measure, do not ship.** Warm-start from HPNet's checkpoint purely to answer
*does pretraining help this task at all?* Zero domain gap means it is the **upper
bound** on what any pretrained prior can buy here. If it barely beats from-scratch,
Point-MAE (bigger domain gap) will not do better either, and the whole question is
closed cheaply.

**Track B — ship.** If Track A shows real gains, pay for a clean-licensed version:
Point-MAE or Point-BERT (MIT), pretrained on ShapeNet. The domain gap is real — chairs
and lamps, not brackets — but masked-autoencoder features are largely local geometry,
so some transfer is plausible. No published Point-MAE → CAD-primitive-segmentation
numbers were found, so this is untested, which is exactly why Track A comes first.

### The tension with going mesh-native

Every pretrained model in that table is point-cloud shaped, while the architecture
recommendation above is a dual graph over triangles. Hotrodding means accepting point
cloud input.

It is a smaller loss than it looks: HPNet, ParSeNet and Point-MAE all take points **plus
normals**, and we can feed *exact* facet normals instead of estimated ones — keeping the
advantage the ablations price at 3–10 IoU while gaining the prior. What is given up is
exact adjacency. That is a measurable trade, not a matter of taste: run mesh-native
from-scratch against point-cloud warm-start and compare on `oracle.mjs` face recovery.

Whatever wins still has to reach ONNX Runtime Web — which keeps Sonata out regardless
of its licence, and means DGCNN's dynamic KNN needs an early latency measurement.

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

## The MVP: a seeded fill assistant, not a whole-mesh segmenter

**Decided.** The first model assists `paintFrom` — the user clicks a facet, the model
proposes which facets belong to the same surface and what type it is. Everything above
about whole-mesh segmentation is the long horizon; this is what gets built.

This is a much smaller problem than it first appeared, and the simplifications are
structural rather than cosmetic:

| whole-mesh segmentation | seeded fill |
|---|---|
| how many faces? decided by the model | the user points at one |
| mean-shift clustering post-process | **none** |
| variable-length output | fixed-size, one seed |
| whole mesh in memory | local neighbourhood, capped |
| batch job | per click, interactive |

Losing mean-shift matters more than it looks. It was a CPU post-process outside the
ONNX graph that would have needed a JS reimplementation, and it was the least
predictable part of the deployment story. Seeding deletes it.

**Shape of the thing**

- *Input*: k-ring neighbourhood of the seed facet, capped (512–1024 facets). Per facet:
  position relative to the seed centroid (scale-normalised), unit normal, dot of that
  normal with the seed's, area, ring index, dihedral across the edge to its parent.
  All of it already computed in `meshCore.js` / `faceGrouper.js`.
- *Output*: per-facet "same surface as seed?" plus the seed surface's type.
- *Budget*: well under 1M parameters, a few MB of ONNX, tens of milliseconds per click.
  Small next to the ~25 MB OCCT bundle already fetched from CDN.
- *Fallback*: absent or slow, `paintFrom` behaves exactly as it does today.

**Training data is the same bake.** Stage 1 is unchanged — the only difference is that
samples are drawn as (seed facet, local neighbourhood) pairs, so one model yields
hundreds of training examples instead of one. That makes the Revolution shortage much
less acute than the model-count table suggests: 913 Revolution models is far more than
913 examples.

**It is not a dead end.** Run it from many seeds and merge the proposals and you have
whole-mesh segmentation. The MVP is the primitive the full version would be built from,
so the work carries forward rather than being thrown away.

### The problem, measured

On baked ABC data, over 600 patches drawn from 216 models, a 512-facet adjacency BFS
around a random seed contains on average **37.4% facets that actually share the seed's
face** — so `pos_weight` for the binary head is about **1.67**, not the 1:5 an early
single-model probe suggested. (That probe sampled 12 seeds on one part and read 16.7%;
it was not representative, and the loss weighting should come from the corpus-wide
figure. Re-measure on the full bake before training.)

Either way the BFS over-reaches badly, which is the `paintFrom` failure stated
numerically — `regionGrow.js` records it qualitatively as "the seed patch crosses the
boundary and poisons the initial fit before growth begins". Cutting that superset down
is the model's entire job, and patch size is a hyperparameter worth sweeping rather than
fixing.

**The seeded framing does defuse the Revolution shortage, measurably.** Revolution is
0.19% of patches corpus-wide, but seeds landing on Revolution faces were 12.7% of that
sample — because seeds are drawn per facet, and the stratified selection deliberately
front-loads rare-type-rich models. Scarcity in *models* is not scarcity in *examples*.

Related, from the same bake: on a plane-and-cylinder ABC part, **26 cross-face edges stay
invisible to crease angle at every density in the ladder** — identical count at coarse,
medium and fine — confirming tangent junctions are structural geometry, not a
tessellation artifact that a finer export could fix. Meanwhile the usable crease window
moves with density (at 5° it falsely cuts 23.5% of within-face edges on the medium mesh;
at 15° it misses 2% of real boundaries), which is the threshold-tuning treadmill the
learned proposer is meant to get off.

## Result: locality decides which types are learnable

First real training run (v1: 24,790 models, 512-facet patches, 30k steps, rare-type seed
balancing). Membership: IoU 0.661 → **0.807**, precision **0.880**, recall **0.907**,
against an all-positive baseline of 0.309. Type macro-accuracy 0.503 → **0.745**.

But the per-class numbers split the vocabulary cleanly in two, and the split is not
about data volume:

| type | acc @ 16k | determined by |
|---|---|---|
| Plane | 0.96 | local differential geometry |
| Sphere | 0.96 | " |
| Cone | 0.81 | " |
| Cylinder | 0.75 | " |
| Torus | 0.72 | " |
| BSpline | 0.65 | free-form catch-all |
| Extrusion | 0.59 | direction is local, profile is not |
| **Revolution** | **0.05** | **rotational symmetry about an axis — global** |

Revolution never learns. Across the run it reads 0.00, 0.00, 0.09, 0.02, 0.01, 0.00,
0.11, 0.05 … 0.11, 0.09, while every other class climbs — and that is *with*
`--balance-seeds` oversampling it.

### Why — and it is not what it looked like

The first hypothesis was locality: that a 512-facet patch cannot see the rotational
symmetry that defines a revolve, so it would look like a generic curved patch, i.e. a
B-spline. **The confusion matrix refutes that**, and so does the coverage data.

Where Revolution seeds actually get predicted (n=50, v1 checkpoint):

| predicted | share |
|---|---|
| **Cylinder** | **56%** |
| Sphere | 16% |
| Revolution | 12% |
| Other | 8% |
| BSpline | 4% |

Not B-spline. **Cylinder and Sphere** — and that is geometrically correct. A surface of
revolution whose profile is a line parallel to the axis *is* a cylinder; one whose
profile is an arc centred on the axis *is* a sphere. The network is not failing to see
the surface; it is naming it by its geometry.

And the patch is not information-starved. Mean facets of a 512-patch lying on the seed's
own face:

    Sphere 211   Torus 294   Revolution 187   BSpline 174
    Cone 79      Plane 53    Extrusion 49     Cylinder 43

Revolution patches have **four times the coverage of Cylinder patches**. Starvation was
the wrong story.

### What this actually means

`SurfaceOfRevolution` is substantially a **provenance label, not a geometric one**. It
records that the modeller built the face with a revolve tool, and the CAD kernel stored
it that way — not that the resulting surface is distinguishable from a cylinder or
sphere. A classifier trained on geometry cannot recover an authoring decision that left
no geometric trace.

For this project that reframes the result entirely: **predicting `Cylinder` on a face
STEP calls `Revolution` is the better answer**, provided the cylinder fits within
tolerance. The converter wants the simplest analytic surface that fits, not the one the
original modeller happened to click. An analytic cylinder is more useful downstream than
a general surface of revolution.

Same pattern explains BSpline at 0.57 — its errors go to Cone 17%, Plane 9%, Sphere 6%,
i.e. spline patches that are locally conical or planar. Again arguably correct answers.

**Consequences:**

1. ~~**Do not build the Revolution synthesis pipeline.**~~ **Reversed — see "Canonical
   labels" below.** The provenance diagnosis was right, but the conclusion drawn from it
   was not: the fix is to *relabel* the faces that are primitives in disguise, after which
   the genuine revolves that remain are scarce and synthesis is worth doing after all.
2. **Label-match accuracy is the wrong metric for type.** Score by whether the predicted
   primitive *fits within tolerance* — which `oracle.mjs` already does, since it tests
   facets against real surfaces rather than comparing enum values.
3. Revolution and Extrusion fitters are still worth having for the cases where no
   simpler primitive fits; they just should not be forced when one does.

## Deployment: measured, not assumed

v1 exported to ONNX (opset 17, dynamic batch and facet count) and run under
onnxruntime, single-threaded CPU:

| patch size | ONNX vs PyTorch max abs diff | latency |
|---|---|---|
| 256 facets | 4.8e-06 | **1.57 ms** |
| 512 facets | 4.3e-06 | **3.04 ms** |
| 1024 facets | 4.3e-06 | **6.12 ms** |

**1.11 MB** on disk, against the ~25 MB OCCT WASM bundle the tool already fetches. The
numeric agreement is float32 round-off, so the exported graph is the trained model.

This vindicates the architecture constraint that drove the design. There is no KNN
anywhere — the k-ring BFS supplied locality when the patch was built — so every
operation is a matmul, LayerNorm, ReLU or masked max. WASM typically runs a few times
slower than native CPU, which still leaves 512-facet inference in the region of 6–15 ms
per click against a budget of "tens of milliseconds". Sparse convolution or dynamic-KNN
architectures would have had no browser path at all.

Still to verify: the same graph under **onnxruntime-web** specifically, in a real
browser, rather than desktop onnxruntime. Exporting is not the same as running fast in
WASM.

### Patch size is settled at 512

Matched ablation (batch 64, 24 seeds/mesh, identical but for patch width):

| step | 512 IoU | 2048 IoU | 512 Rev | 2048 Rev |
|---|---|---|---|---|
| 4000 | 0.693 | 0.623 | 0.00 | 0.00 |
| 8000 | 0.747 | 0.671 | 0.02 | 0.00 |

Larger patches are **worse** on membership and do nothing for Revolution. Both follow
from the diagnosis above: more surface cannot reveal a distinction that is not
geometric, and a wider patch lowers the positive rate while asking one fixed-capacity
max-pool to summarise more.

## Label noise: STEP translation splits periodic surfaces

The bake's cross-check found 43 of 24,800 models (0.17%) where OCCT's face-type
histogram disagrees with ABC's feature file. Characterised on a 4,000-model sample: 9
disagreements (0.23%), and OCCT consistently sees **more** faces, not different ones.

It is not a part-mismatch — feat and STEP part indices agree on 2,252 of 2,252 sampled
models. The signature points at STEP translation splitting periodic surfaces at seams:
on one model, Torus went 33 → 65 (nearly doubled — tori are doubly periodic), BSpline
10 → 62, Cylinder 185 → 223. ABC's feature file describes the original CAD B-rep; the
STEP is a translation of it.

Measured directly on baked meshes: cross-face edges that are tangent-continuous AND
share a surface type — the seam-split signature — are **1.0% of all cross-face edges**,
and **66% of those are Torus**.

Consequence: a logical cylinder split at its seam teaches the model that facets on the
far half are *not* the same surface, when for `paintFrom`'s purposes they plainly are.
That caps achievable membership IoU slightly. At 1% it is minor label noise rather than
a structural problem, so it is recorded and deprioritised — but if membership plateaus
short of expectations, merging adjacent same-type tangent faces before labelling is the
first thing to try.

## Validation plan: the whole corpus, not a showcase

The showcase set is a smoke test and must not be mistaken for validation. Fourteen
hand-picked models are a broader basis than one WatchStand, but they are still
hand-picked, and picked by us — the same selection bias, one order of magnitude up.

Proving the tool is useful means **reconstructing a substantial portion of ABC from mesh
input and scoring against the STEP each model came from**. Everything needed is already
on disk from Stage 1:

- 24,790 baked models with per-facet `face_id` and `face_type`, at three densities
- 200,000 STEP files, so ground-truth surfaces can be parsed for any of them
- `test/oracle.mjs` and `test/fitoracle.mjs` already score faces against exact surfaces

The benchmark is: for each model, run the full pipeline on the mesh alone, then report
faces recovered out of the oracle ceiling, per surface type, per tessellation density.
That yields a distribution rather than an anecdote, and it makes regressions visible —
today a change that helps WatchStand and breaks a thousand other parts looks like
progress.

Two things to build first, since the harnesses currently assume one fixture: truth-set
construction that refuses cases where the tolerance is large against the feature (as
`fitoracle.mjs` now does), and per-type reporting so a gain on planes cannot mask a loss
on cones.

### First showcase run — findings

- **Torus classification works.** 40/40 on an exact torus at medium and fine density,
  39-40/40 on the displaced "lumpy" variants. This is the class ParSeNet and HPNet do
  not support at all, and the one the vocabulary decision was made for.
- **There is no torus fitter.** `fitRegionAs` covers plane, cylinder, cone, sphere and
  a NURBS fallback. So the model identifies a torus correctly and the analytic side has
  nowhere to send it: 0 of 40 fits on the exact tori. Highest-value gap found.
- **Density sensitivity.** Seeded coverage collapses from 44.4% to 0.1% between coarse
  and fine tessellations of the same torus. A 36,864-facet single surface is denser
  per-face than anything in ABC, so this is probably out-of-distribution — but the
  density ladder was meant to train exactly this out, and did not.
- **3DBenchy fits best of the whole set**: mean relative residual 2.5e-2, with a type
  spread appropriate to a hull. Consistent with it being real CAD geometry.

## Kill criterion

The seeded framing gives a better one than 26/62, because `test/paint.mjs` already
measures it and the numbers are already recorded:

- **WatchStand conical faces: 0/4**, flat across seedSize 12–80 crossed with tolScale
  0.5–1.5. Structural, not a tuning problem: `neighbourhood()` is a plain adjacency BFS,
  so on a face small relative to its neighbours the seed patch crosses the boundary and
  poisons the initial fit before growth begins. **A learned seed proposer attacks
  exactly this.**
- **tangent-fillet over-reach ~25%**: the cylinder and top face bleed into the tangent
  blend, which near the junction genuinely lies within tolerance of both surfaces.

Targets: recover more than 0/4 on the conical faces, and cut the over-reach, without
regressing tangent-fillet's base plane (currently 100% purity and 100% recall) or its
cylinder radius (19.986 against a true 20).

If it does not beat the existing `paintFrom` on those measurements, stop. Do not tune,
do not scale up. Record the negative result in the module header the way
`regionGrow.js` records the RANSAC commit-order failure (14/77).

Whole-mesh 26/62 stays as the long-horizon number, not the MVP bar.

### Verdict: criterion NOT met — and the reason is not the model

`test/paintnet.mjs`, v1 checkpoint against the recorded baseline:

| WatchStand cone | baseline region | model region |
|---|---|---|
| 59.00° (30 facets) | 50.0% purity / 20.0% recall | **100% / 100%** |
| 45.00° (50 facets) | **0.0% / 0.0%** | **90.6% / 96.0%** |
| 9.85° (477 facets) | 100% / 1.9% | 100% / 14.7% |
| 9.85° (244 facets) | 100% / 8.2% | 100% / 22.1% |

The model does the job it was built for. On the 45° cone the BFS baseline captures
*nothing* correct; the model captures 96% of it at 90.6% purity. Type prediction was
Cone at 87% and 100% confidence on the two faces where a cone is well defined. Mean
2.7 ms per call including the k-ring BFS, feature build and inference.

**Cones recovered: 0/4 → 0/4** on the first run. Which turned out not to be a statement
about the model at all — see below.

But the diagnosis moved, and `test/fitoracle.mjs` was written to pin it down — handing
`fitRegionAs` the *exact* ground-truth facets:

    cone   9.85°, 477 facets  ->  22.34°      (err 12.48)
    cone   9.85°, 244 facets  ->  31.21°      (err 21.36)
    cone  59.00°,  30 facets  ->  DEGENERATE  ("no measurable taper")
    cone  45.00°,  50 facets  ->  86.00°      (err 41.00)
    cylinders .....................  4/6, both misses a 5% relative tolerance
                                     on sub-millimetre radii

**0 of 4 cones, from perfect input.** So `paint.mjs`'s 0/4 was never a segmentation
result on this fixture. The premise this entire experiment was scoped on — *"handed the
right facets, the fitters recover a 0.5 mm cylinder radius to six significant
figures"* — was measured on a **cylinder** and generalised. It does not hold for cones.

Two further observations from the same run:

- On tangent-fillet, model-seeded `paintFrom` is byte-identical to the baseline
  (1536 facets, 75% purity). The model's own region is far tighter — 205 facets at
  99.5% purity — but `growOn` expands it straight back out. **Over-reach is a property
  of the growth loop, not the seed**, so a better seed cannot fix it.
- Model-seeded sometimes scores *worse* than the model alone (59° cone: 30 facets → 1).
  Good region → bad cone fit → `growOn` rejects almost everything → collapse. Another
  consequence of the same defect.

**Revised ordering.** Fix `fitCone`, then re-run `fitoracle.mjs`, then re-evaluate the
model. Until a primitive passes the fit-from-truth oracle, no proposer can raise its
score — which is a much cheaper thing to know before optimising a network than after.

### fitCone fixed — criterion now MET

Two defects in `gaussMap.js`, both structural rather than tuning:

1. **Half-angle from an R-vs-Z regression.** That needs the patch to span axial extent.
   A narrow annular band has near-constant R, so the slope read ~0 and a perfectly
   conical 30-facet patch on its true axis was rejected as "no measurable taper". It now
   comes from the normals: `|n · axis| = sin(halfAngle)` holds on any patch, however
   narrow. This is the Gauss-map relation the module is named for, applied per facet.
2. **Axis refined on the positional residual.** That objective has a degenerate minimum —
   a wrong axis with a compensating half-angle fits a patch *better* than the true one
   (measured 8.4e-4 against 4.0e-3 on the 45° cone), so the search actively walked away
   from a correct Gauss-map seed. The axis is now chosen by the variance of `n · axis`,
   which is **exactly zero** on a cone's true axis.

The second fix matters methodologically too. A weighted blend of positional and normal
terms needed a trust constant of ~256 to reach 4/4 — tuned on four cones, i.e. fitted to
the fixture. Using the exact criterion instead scored **better** with no constant at all.

Results, from perfect input: **4/4** (9.66 / 8.49 / 58.96 / 44.99 against 9.85 / 9.85 /
59.00 / 45.00). And one click each:

| | cones recovered |
|---|---|
| baseline `paintFrom`, before the fix | 0/4 |
| baseline `paintFrom`, after the fix | 1/4 |
| **`paintFrom` seeded by the model** | **3/4** |

Per-cone error, baseline → model-seeded: 8.04° → **0.03°**, 61.67° → **0.12°**,
0.73° → **0.03°**, and 40.34° → 29.58° (the one remaining failure).

**The criterion is met.** The model earns its place — but note it was never the thing
that was broken, and the fitter fix alone accounts for the first quarter of the gain.

**Next weakness, measured:** on the 45° cone the proposed region is 90.6% pure and 96.0%
complete, yet fits to 15.42° where the exact facets give 44.99°. Five foreign facets out
of 53 wreck it. The next work is outlier robustness in the fit — not the region, and not
the model.

## Canonical labels: fix the noise instead of accepting it

The provenance finding above treated `Revolution` as unrecoverable. It is recoverable,
just not by the network: the label can be corrected at the source. `canonical.py` fits
every face's exact B-rep samples against Plane, Sphere, Cylinder, Cone and Torus and
relabels it as the first primitive whose residual is below 1e-6 of its extent. The label
becomes **the simplest exact surface**, which is what the converter wants anyway.

**Where the threshold comes from.** Residuals on stored-`Revolution` faces are bimodal —
a cluster at machine precision and a cluster far above — so the cut is unambiguous
there. `Extrusion` and `BSpline` residuals are continuous with no gap, so those are
relabelled only at the strict threshold. Fitter precision floors were fixed first (cone
all-axis search, torus Levenberg–Marquardt), since a fitter that misses exact primitives
would leave noise in place and call it signal.

**Validation split, 1,999 parts, zero face-order misalignments** (`relabel_sidecar.py`
checks every density). Largest moves: BSpline→Torus 1,168, Revolution→Cylinder 511,
BSpline→Plane 410, BSpline→Cylinder 220, Extrusion→Cylinder 202, Revolution→Torus 175,
Revolution→Cone 64. Roughly two thirds of `Revolution` faces are primitives.

### The existing model, re-scored — no retraining

`rescore.py`, v2ctrl checkpoint, identical 33,377 patches, per-class accuracy:

| class | stored labels | canonical labels |
|---|---|---|
| Plane | 0.964 | 0.964 |
| Cylinder | 0.881 | 0.876 |
| Torus | 0.833 | **0.743** |
| Revolution | 0.164 | **0.240** |
| BSpline | 0.651 | 0.664 |
| macro | 0.736 | 0.734 |

Of stored-Revolution seeds relabelled to a primitive, the model already predicted that
primitive 55.7% of the time — it was being marked wrong for right answers. Torus drops
because 1,168 faces stored as BSpline are tori, and the model was trained to call them
BSpline. Macro is flat, so **the gain is only available by retraining on canonical
labels**; `train.py --relabel` does that, and skips any real file without a sidecar
rather than mixing label schemes.

**Membership baseline for every later comparison.** Membership does not depend on type
labels, so it is the one number directly comparable between checkpoints trained on stored,
canonical or synthetic-mixed labels. v2ctrl on the same 33,377 patches, pooled IoU
**0.7975**; by canonical seed class Plane 0.834, Cylinder 0.768, Cone 0.783, Sphere 0.918,
Torus 0.866, **Revolution 0.624**, Extrusion 0.615, BSpline 0.625, Other 0.559.

### Retrained on canonical labels (v3canon): no change — and why

Same settings as v2ctrl (60k steps, curvature zeroed), labels from the sidecars. Scored
on the identical 33,377 patches:

| | v2ctrl | v3canon |
|---|---|---|
| membership IoU | 0.7975 | 0.7970 |
| macro type accuracy, canonical labels | 0.7339 | 0.7375 |
| Revolution (canonical) | 0.240 | 0.240 |
| Torus (canonical) | 0.743 | 0.740 |
| BSpline (canonical) | 0.664 | 0.687 |

Split by stored → canonical pair, the telling row is **BSpline → Torus (507 seeds): the
model predicts Torus 0.2% of the time before retraining and 0.4% after**, while calling
the same faces BSpline fell from 62% to 45% — so the corrected labels did reach training,
and the model still could not act on them. Native tori score 83%. These faces are exact
tori; if geometry decided the type, they would score like native ones.

The hypothesis this points to is that the type head reads OCCT's *triangulation pattern*
— a B-spline surface is meshed on its knot grid, an analytic torus on its own
parameterisation — rather than the surface. Tested directly below.

### Control: same geometry, two tessellations

150 validation parts with native torus faces. Each whole shape is converted with
`BRepBuilderAPI_NurbsConvert` — identical geometry, every face now stored as a B-spline —
and both versions are tessellated at the same density. Seeds on the same torus faces
(486 in each version). The conversion throws on 17 of 270 candidates and segfaults on 2,
so each part is meshed in an isolated worker process.

| checkpoint | mesh | predicted Torus | membership IoU |
|---|---|---|---|
| v2ctrl | native torus | **83.5%** | 0.866 |
| v2ctrl | NURBS-converted | **2.7%** (BSpline 88%) | 0.730 |
| v3canon | native torus | 84.2% | 0.863 |
| v3canon | NURBS-converted | 2.7% (BSpline 89%) | 0.728 |

Triangles per torus face are similar (413 native, 353 converted), so this is not a
density effect. **Identical geometry gets a different surface type, and a worse
segmentation, depending only on which mesher produced it.** Retraining on corrected labels
cannot fix that, which is why v3canon matched v2ctrl.

Which inputs carry the cue — one feature group zeroed at inference, same patches:

| zeroed | v2ctrl Torus %: native / NURBS | v3canon Torus %: native / NURBS |
|---|---|---|
| nothing | 83.5 / 2.7 | 84.2 / 2.7 |
| **log_area** | **87.4 / 42.0** | **89.9 / 48.4** |
| ring | 12.1 / 3.7 | 12.6 / 1.6 |
| log_area + ring | 1.4 / 0.2 | 1.2 / 2.1 |
| position + dist | 6.0 / 0.2 | 8.8 / 0.2 |
| normal + dot | 0.0 / 0.0 | 0.0 / 0.0 |

Zeroing `log_area` — each facet's area relative to the patch mean — lifts Torus on the
converted mesh from 2.7% to 42–48% while native stays near 88%. Relative triangle size
is where the mesher's pattern lives: a knot-grid mesh has a different area distribution
from an analytic surface's. Positions and normals carry the geometry and cannot be
removed without collapsing everything.

**Consequences.**

1. Type accuracy measured on OCCT meshes of ABC is partly accuracy at recognising OCCT's
   meshing of each stored type. A user's STL from any other exporter is out of
   distribution by construction, for type and for membership.
2. This is the strongest argument yet for the tessellation-diversity augmentation (other
   meshers, decimation, remeshing): not scan realism, but removing a shortcut the model
   demonstrably takes on clean CAD meshes.
3. `BRepBuilderAPI_NurbsConvert` is itself a free augmentation — same geometry, same
   canonical label, a different mesh pattern — available for every training part.
4. `log_area` should be dropped or heavily augmented. Zeroing it at inference costs
   membership (0.866 → 0.678) only because the model was trained to use it.

### Synthetic mix (v3synth): a small Revolution gain, a broader loss

Same settings as v3canon, plus the 4,000-part synthetic bake as 15% of training files.
Validation real-only, the identical 33,377 patches:

| | v3canon | v3synth |
|---|---|---|
| membership IoU | **0.797** | 0.776 |
| Revolution membership | 0.624 | **0.647** |
| Revolution type (canonical) | 0.240 | **0.272** |
| BSpline type | 0.687 | **0.759** |
| Cylinder type | 0.878 | **0.893** |
| Cone type | **0.900** | 0.821 |
| Extrusion type | **0.770** | 0.675 |
| Plane type | **0.967** | 0.948 |
| macro type accuracy | **0.738** | 0.729 |

Membership drops for Plane (0.836 → 0.781), Cylinder, Cone and Extrusion seeds. Synthetic
parts take training steps from real ones, and their planar and cylindrical faces are
meshed in a pattern real ABC parts do not share — which, given the mesher finding above,
the model would be expected to pick up. **Not adopted.** Revolution is better attacked
together with tessellation diversity than by adding more OCCT meshes of any kind.

### The whole validation split, re-meshed

`mesher_bake.py --variant nurbs` over the validation split, scored with v3canon (1,693
parts at first count; the comparison on parts common to every mesher follows):

| | native | NURBS-converted |
|---|---|---|
| membership IoU | 0.797 | 0.724 |
| macro type accuracy | 0.738 | 0.544 |
| Torus | 0.740 | **0.024** |
| Sphere | 0.926 | 0.599 |
| Cylinder | 0.878 | 0.637 |
| Cone | 0.900 | 0.712 |
| Plane | 0.967 | 0.971 |

Every curved analytic class degrades once the same surfaces are meshed on a knot grid;
planes, which mesh identically either way, do not move. The torus control was not a
special case.

### Three meshers, identical parts

`mesher_bake.py` over the validation split with two re-meshings, scored only on parts
present in all three (`rescore.py --intersect`): 1,801–1,802 parts, ~29,800 patches each.

- **nurbs:** 1,845 of 1,999 parts built. 25 genuine crashes and 129 OCCT exceptions.
- **gmsh:** 1,951 built. 25 gmsh exceptions and 22 timeouts. Its median triangle count
  is 1.10 / 1.51 / 1.40 × OCCT's at coarse / medium / fine (p90 3–4×). That sits within
  the spacing of OCCT's own density ladder, which the models train on.

| | native v2ctrl | native v3canon | native v3synth | nurbs v2ctrl | nurbs v3canon | nurbs v3synth | gmsh v2ctrl | gmsh v3canon | gmsh v3synth |
|---|---|---|---|---|---|---|---|---|---|
| membership IoU | 0.795 | 0.794 | 0.773 | 0.719 | 0.718 | 0.701 | **0.530** | **0.514** | **0.489** |
| macro type acc (canonical) | 0.748 | 0.748 | 0.740 | 0.528 | 0.546 | 0.534 | **0.412** | **0.423** | **0.409** |
| Plane | 0.963 | 0.965 | 0.945 | 0.962 | 0.964 | 0.939 | 0.837 | 0.827 | 0.844 |
| Cylinder | 0.875 | 0.877 | 0.890 | 0.614 | 0.622 | 0.633 | **0.036** | **0.043** | 0.072 |
| Cone | 0.909 | 0.906 | 0.818 | 0.714 | 0.721 | 0.566 | 0.623 | 0.600 | 0.425 |
| Sphere | 0.943 | 0.938 | 0.953 | 0.540 | 0.592 | 0.613 | 0.482 | 0.467 | 0.538 |
| Torus | 0.752 | 0.749 | 0.729 | 0.022 | 0.018 | 0.008 | 0.147 | 0.167 | 0.074 |
| Revolution | 0.238 | 0.207 | 0.254 | 0.105 | 0.179 | 0.241 | 0.133 | 0.169 | 0.216 |
| Extrusion | 0.780 | 0.781 | 0.686 | 0.606 | 0.607 | 0.495 | 0.584 | 0.586 | 0.480 |
| BSpline | 0.691 | 0.719 | 0.780 | 0.694 | 0.715 | 0.783 | 0.552 | 0.587 | 0.671 |

**The labels were checked before believing this.** Across 45,134 faces in 400 parts,
every gmsh face's area-weighted centroid lies within 1% of the diagonal of the native
mesh's face with the same id. Area agrees within 2% for 97.7% of cylinders and 96.3% of
tori. The larger area gaps on spheres (5.6% within 2%) and cones (58%) are chordal
approximation of doubly curved or apex-bound faces, not mislabeled faces, since their
centroids agree too.

**What the model sees instead** (v3canon, 300 parts, medium density):

| seed | native mesh predicted as | gmsh mesh predicted as |
|---|---|---|
| Cylinder | Cylinder 88%, Extrusion 5% | **Cone 36%, BSpline 32%, Extrusion 20%, Cylinder 4%** |
| Torus | Torus 79%, BSpline 8% | **BSpline 59%**, Torus 18%, Sphere 10% |
| Plane | Plane 98% | Plane 83%, Extrusion 9%, Cone 7% |

OCCT meshes a cylinder as long triangles running the length of its axis; gmsh fills it
with near-equilateral ones. **The model learned the strips, not the cylinder.**
On a mesher that is merely a different algorithm (not a scanner, not a decimator)
membership IoU falls from 0.79 to 0.51 and cylinders, the most common curved surface in
CAD, go from 88% to 4%. Neither corrected labels nor synthetic revolves changed this.

This is the most important measurement in the experiment so far: every earlier accuracy
figure is an accuracy on OCCT's meshes. The next step is training on several meshers,
with one held out as a test of generalisation to a mesher never seen.

### Real hand-made meshes: the baseline to beat

Game and prop meshes are the far end of the mesher axis. They have no STEP, so they are
scored without ground truth by `test/meshReport.mjs` (see `game-assets-scope.md`),
using 60 seeds per file with the model's curvature columns zeroed as trained. A **bogus
fit** is one whose own relative residual exceeds 1e-2: the predicted surface is not there.
Rips are evaluation-only; CC0 sources may be trained on.

| set | checkpoint | top predicted types | Cone share | bogus / fitted | failed / attempted | mean claimed |
|---|---|---|---|---|---|---|
| Kenney Blaster Kit, 39 low-poly OBJs (deduplicated) | v2ctrl | Plane 82%, Cone 15% | 15% | 35% | 661 / 2,283 | 61.5% |
| | v3canon | Plane 85%, Cone 13% | 13% | 34% | 617 / 2,304 | 62.7% |
| Poly Haven, 5 quad-modelled props | v2ctrl | Cone 50%, Plane 25% | 50% | 63% | 15 / 253 | 59.6% |
| | v3canon | Cone 49%, Plane 26% | 49% | 61% | 10 / 260 | 58.9% |
| Master Chief (rip, evaluation only) | v2ctrl | Cone 62%, BSpline 15% | 62% | 98% | 0 / 42 | 6.2% |
| | v3canon | Cone 57%, Revolution 13% | 57% | 100% | 0 / 43 | 6.2% |

The more hand-made the mesh, the more the type head falls back to Cone, and the fits
behind it do not hold. Corrected labels change nothing here, as expected of a model
trained only on OCCT meshes.

Queued next: `v4all` (native plus NURBS, gmsh and game-art copies of training parts, from
`game_bake.py`) and `v4nogmsh` (gmsh held out). Both will be scored on this table and on
the four labelled mesh sets.

### Training across meshers (v4all, v4nogmsh)

Both runs use v3canon's settings (60k steps, canonical labels, curvature zeroed), plus
re-meshed copies of TRAINING parts at 60% of training files. Validation stays native.

- **v4all:** 19,754 NURBS, 7,277 gmsh and 7,455 game-art copies.
- **v4nogmsh:** the same without gmsh, so gmsh is a mesher it never saw.

The copies come from `mesher_bake.py`, `game_bake.py` and the fixed `isolated_pool.py`.
NURBS conversion fails on 7.4% of training parts (1,453 OCCT exceptions and 232 crashes
that reproduce on retry). gmsh built 7,825 of 8,000.

**Labelled meshes**, same parts per column group:

| | native v3canon | native v4all | native v4nogmsh | nurbs v3canon | nurbs v4all | nurbs v4nogmsh | gmsh v3canon | gmsh v4all | gmsh v4nogmsh |
|---|---|---|---|---|---|---|---|---|---|
| membership IoU | 0.794 | 0.771 | 0.773 | 0.718 | 0.737 | 0.737 | 0.514 | **0.682** | 0.548 |
| macro type acc | 0.748 | 0.694 | 0.688 | 0.546 | 0.611 | 0.621 | 0.423 | **0.590** | 0.469 |
| Plane | 0.965 | 0.938 | 0.943 | 0.964 | 0.945 | 0.951 | 0.827 | 0.931 | 0.869 |
| Cylinder | 0.877 | 0.645 | 0.757 | 0.622 | 0.492 | 0.635 | 0.043 | **0.560** | **0.060** |
| Cone | 0.906 | 0.874 | 0.884 | 0.721 | 0.820 | 0.828 | 0.600 | 0.785 | 0.680 |
| Sphere | 0.938 | 0.934 | 0.934 | 0.592 | 0.872 | 0.897 | 0.467 | 0.875 | 0.671 |
| Torus | 0.749 | 0.623 | **0.775** | 0.018 | 0.155 | **0.502** | 0.167 | 0.306 | **0.730** |
| Revolution | 0.207 | 0.158 | 0.102 | 0.179 | 0.151 | 0.127 | 0.169 | 0.093 | 0.103 |
| Extrusion | 0.781 | 0.800 | 0.693 | 0.607 | 0.771 | 0.603 | 0.586 | 0.715 | 0.479 |
| BSpline | 0.719 | 0.675 | 0.541 | 0.715 | 0.694 | 0.584 | 0.587 | 0.545 | 0.336 |

| game-art (300 parts, `bake_game2`) | native v3canon | native v4all | native v4nogmsh | game v3canon | game v4all | game v4nogmsh |
|---|---|---|---|---|---|---|
| membership IoU | 0.807 | 0.788 | 0.786 | 0.639 | 0.670 | 0.666 |
| macro type acc | 0.697 | 0.648 | 0.643 | 0.447 | 0.475 | 0.481 |
| Sphere | 0.919 | 0.913 | 0.925 | 0.425 | 0.769 | 0.777 |
| Torus | 0.745 | 0.640 | 0.773 | 0.161 | 0.155 | 0.382 |

**Real hand-made meshes** (`meshReport`, 60 seeds per file):

| set | checkpoint | Cone share | bogus / fitted | failed / attempted | mean claimed |
|---|---|---|---|---|---|
| Kenney, 39 files | v3canon | 13% | 34% | 617 / 2,304 | 62.7% |
| | v4all | 17% | 37% | 213 / 2,074 | 73.5% |
| | v4nogmsh | 19% | 45% | 139 / 2,288 | 76.7% |
| Poly Haven, 5 props | v3canon | 49% | 61% | 10 / 260 | 58.9% |
| | v4all | 43% | 57% | 3 / 254 | 62.8% |
| | v4nogmsh | 50% | 66% | 7 / 271 | 60.1% |
| Master Chief (rip) | v3canon | 57% | 100% | 0 / 43 | 6.2% |
| | v4all | 65% | 98% | 0 / 44 | 5.8% |
| | v4nogmsh | 57% | 96% | 0 / 49 | 6.9% |

**Findings.**

1. **Diversity fixes the meshers it contains.** v4all on gmsh: membership IoU
   0.514 → 0.682, cylinders 0.043 → 0.560. The native-to-gmsh membership gap shrinks
   from 0.28 to 0.09. Spheres recover on every re-meshing (0.87–0.88 on NURBS and gmsh).
2. **It only partly generalises to a mesher it has not seen.** v4nogmsh on gmsh: IoU
   0.548 and **cylinders 0.060**, still collapsed. Spheres (0.671) and tori (0.730) do
   transfer. An isotropic, Delaunay-style mesh is its own family: the model stops reading
   the pattern only for families it was shown. Real exporters are more families still,
   so coverage has to be broad (decimation, isotropic remeshing and quad-dominant art are
   obvious next ones).
3. **Adding gmsh is a trade-off.** v4nogmsh beats v4all on native Cylinder (0.757 vs
   0.645) and on Torus everywhere, including native 0.775, above v3canon's 0.749. With
   gmsh in training, native cylinders are called Extrusion 17% and Cone 14%. Those are
   geometric neighbours, not pattern artefacts. The analytic fitters separate them by
   residual, which argues for fit-based arbitration after the type head.
4. **The cost on native meshes is modest:** about −0.02 membership IoU and −0.05 macro
   type accuracy for both, within the same 60k-step budget spread over 2.5× the files.
   BSpline (0.719 → 0.541 for v4nogmsh) and Revolution (0.207 → 0.10–0.16) lose most.
5. **Real hand-made meshes do not improve.** Failed fits fall and more of each mesh is
   claimed, but bogus fits hold or rise (Kenney 34% → 45% for v4nogmsh). The type head
   is not the lever for game art. The residual-gated freeform route in
   `game-assets-scope.md` is.

**Neither model replaces v3canon outright.** v4all is the better choice when meshes
come from unknown tools, and v3canon when they come from OCCT. The next training run
should add further mesher families before tuning anything else, and be judged on
membership and type across all of them.

### Removing the triangulation features (v5)

`log_area` (relative triangle area) and `ring` (BFS hop count) describe the mesh, not
the surface. The NURBS-torus test above already showed `log_area` carries the mesher
fingerprint. Four runs share v4nogmsh's settings (gmsh held out, NURBS + game-art copies,
60k steps, curvature zeroed) and today's data:

- `v5ctrl`: nothing extra zeroed
- `v5noarea`: `log_area` zeroed
- `v5noring`: `ring` zeroed
- `v5noboth`: both zeroed

The control exists because the NURBS pool grew after v4nogmsh trained (21,299 copies
against 19,754), so v4nogmsh is not the same data. `train.py --zero-features` stores the
columns in the checkpoint. `rescore.py` reads them back, and `export_onnx.py` zeroes
them inside the ONNX graph so the browser cannot feed them.

**Patch level** (`rescore.py`; native/NURBS/gmsh on 1,801–1,802 identical parts, game art
on 300):

| | native ctrl | native noboth | nurbs ctrl | nurbs noboth | **gmsh ctrl** | **gmsh noarea** | **gmsh noring** | **gmsh noboth** | game ctrl | game noboth |
|---|---|---|---|---|---|---|---|---|---|---|
| membership IoU | 0.774 | 0.743 | 0.735 | 0.727 | 0.545 | 0.579 | 0.569 | **0.620** | 0.670 | 0.635 |
| macro type acc | 0.684 | 0.651 | 0.614 | 0.611 | 0.470 | 0.498 | 0.486 | **0.542** | 0.480 | 0.467 |
| Plane | 0.935 | 0.944 | 0.941 | 0.954 | 0.806 | 0.921 | 0.818 | **0.918** | 0.918 | 0.967 |
| Cylinder | 0.729 | 0.624 | 0.632 | 0.556 | 0.066 | 0.061 | 0.127 | 0.131 | 0.582 | 0.548 |
| Sphere | 0.911 | 0.915 | 0.860 | 0.880 | 0.674 | 0.615 | 0.661 | **0.853** | 0.667 | 0.677 |
| Torus | 0.735 | 0.646 | 0.372 | 0.441 | 0.632 | 0.349 | 0.664 | **0.312** | 0.309 | 0.303 |
| BSpline | 0.592 | 0.543 | 0.618 | 0.572 | 0.394 | 0.665 | 0.360 | **0.645** | 0.365 | 0.362 |

**End to end** (`test/fillEval.mjs`, 80 validation parts, the same clicks as the
v3canon/v4 baseline). Share of automatic fills that are clean:

| | native | nurbs | gmsh | game |
|---|---|---|---|---|
| v3canon | 44% | 35% | 23% | 33% |
| v4all | 42% | 34% | 28% | 34% |
| v5ctrl | 44% | 36% | 24% | 34% |
| v5noarea | 40% | 34% | 24% | 32% |
| v5noring | 42% | 35% | 24% | 32% |
| v5noboth | 40% | 35% | 24% | 33% |

**Findings.**

1. **Removing both features generalises to an unseen mesher at the patch level.** On
   gmsh, membership IoU rises 0.545 → 0.620 and macro accuracy 0.470 → 0.542. That is
   about half the gap to v4all, which trained on gmsh (0.682). Planes, spheres,
   B-splines and extrusions gain most.
2. **Cylinders on isotropic meshes are still not fixed** (0.066 → 0.131). Tori get worse
   without `log_area` (0.632 → 0.312). The features were not the only route to the
   shortcut, and tori were partly relying on them legitimately.
3. **The cost on native meshes is real but modest:** IoU −0.031, macro −0.033,
   Cylinder −0.10. Game art, which all four runs trained on, gains nothing.
4. **On hand-made meshes** (`meshReport`), removing `log_area` moves the Cone fallback
   to B-spline on organic meshes (Master Chief 8% → 37%) and to Plane or Cylinder on
   low-poly art. Cone share falls by 10–18 points and bogus fits by a few points. `ring`
   alone does little.
5. **None of it reaches the user.** Clean fills stay flat or fall (native 44% → 40%,
   gmsh 24% everywhere). On gmsh about 60% of fills spill, for every model and even with
   the true type declared. **`paintFrom`'s growth step is the cap on fill quality**, so
   no further training run should be judged on fills until that spill is diagnosed.

### Surface hypothesis head (s1, s2, s3)

The spill diagnosis showed two things:
- `paintFrom`'s acceptance test admits facets of neighbouring analytic faces.
- The fitters fail on the small regions a click provides: 12-facet torus seeds come back
  degenerate 46% of the time.

This experiment gives the seeded-fill model a third output, **the analytic surface under the
seed**, used inside segmentation to guide growth. Final geometry still comes only from the
analytic fitters on the selected facets.

**Parameterisation** (`ml/surface_targets.py`), in the patch frame (seed at the origin, seed
normal +Z, lengths / scale):
- the axis, sign-free
- `asinh(kn)`, with kn = scale / (distance along the seed normal to the axis, centre or tube
  centre), signed by which side that point lies
- for tori, `asinh(ks)`, with ks = side × scale / R

Radii, centres and the cone apex follow from these plus the seed facet. A cone's half-angle
is |n · axis|.

**Labels:**
- `ml/surface_sidecar.py` reads exact parameters from STEP for every face: 24,990 models,
  3.01 M faces, 0 face-order mismatches, 9 minutes.
- Faces without an exact stored primitive carry no surface loss.
- `ml/test_surface_targets.py` round-trips targets → surface → distance of the seed face's
  own vertices. They land within 0.1–0.3% of patch scale, which is the chord offset of
  reconstructing through the seed centroid. The seed is facet 0 of every patch.

**Runs.** v5noboth's settings (60k steps, `log_area` and `ring` zeroed, NURBS + game-art
copies) plus `--surface-root`:
- `s1nogmsh`: curvature off, gmsh held out
- `s2nogmsh_curv`: curvature on
- `s3all`: curvature off, gmsh in training

Native validation IoU is 0.809 / 0.800 / 0.802.

**Parameters** (`ml/surface_eval.py`, 250 validation parts):

| | s1 native | s2 native | s3 native | s1 gmsh | s3 gmsh |
|---|---|---|---|---|---|
| axis error p50 (cylinder / cone / torus) | 2.7 / 5.2 / 4.9° | 2.7 / 6.0 / 5.9° | 2.2 / 5.6 / 5.7° | 5.8 / 7.2 / 5.2° | 2.9 / 6.7 / 5.9° |
| cylinder axis p90 | 10.6° | 11.4° | 8.3° | **74.7°** | 12.3° |
| kn sign, curved types | 95–100% | 95–100% | 96–100% | 91–100% | 94–100% |
| \|kn\| relative error p50 | 0.07–0.13 | 0.12–0.16 | 0.06–0.14 | 0.08–0.23 | 0.07–0.22 |

**As a strict in-patch gate** (paintFrom's tolD and 15°, true type, predicted parameters),
the hypothesis under-fills. Clean patches on native meshes, true surface vs predicted
parameters (s1):

| | true surface | predicted |
|---|---|---|
| Cylinder | 49% | 22% |
| Cone | 68% | 18% |
| Torus | 67% | 8% |

Recall p50 is 0.33–0.59. Locally the parameters are good, but a few degrees of axis error or
10% of curvature puts the far side of a 512-facet patch outside tolD.

**End to end** (`test/surfaceGateEval.mjs`, the fillEval clicks, true type declared). Clean
fills:
- `paintFrom`: today's fill from the model's region
- `+`: with the adaptive angle tolerance and membership veto
- `hyp → fit`: grow with the hypothesis surface, then hand that region to paintFrom

| | paintFrom | paintFrom+ | hyp → fit | hyp → fit+ |
|---|---|---|---|---|
| s1 native | 49% | 48% | **57%** | 56% |
| s3 native | 48% | 45% | **56%** | 54% |
| s1 gmsh | 29% | 31% | 31% | 32% |
| s3 gmsh | 28% | **35%** | 30% | 33% |
| s1 game | **33%** | 32% | 28% | 27% |
| s3 game | **32%** | **32%** | 28% | 28% |

With the model's own type instead ("auto"), native goes 40% → 46% (s1) and 38% → 42% (s3).
By true type on native, s1, paintFrom → hyp → fit:

| | paintFrom | hyp → fit |
|---|---|---|
| Cylinder | 41% | 47% |
| Cone | 53% | 65% |
| Sphere | 71% | 85% |
| Torus | 32% | 45% |
| Plane | 61% | 62% |

**Findings.**

1. **The predicted surface works as a START for the exact fitter, not as a boundary.** Handing
   a hypothesis-grown region to paintFrom gains 8 points on native meshes. The curved types
   gain most: tori +13, cones +12, spheres +14. Using the hypothesis as the boundary (`hyp
   only`) or as a filter on the fitted fill (`fit & hyp`) under-fills and scores below today.
   A looser 3× gate before the handoff changes nothing.
2. **gmsh is limited by the acceptance test, not the start.** Even s3, trained on gmsh, gains
   only 2 points from the hypothesis start, and spill stays above 50%. What helps there is
   the adaptive angle + veto (s3: 28% → 35%, planes 52% → 82%).
3. **The veto's earlier gain does not transfer across models.** With v4all's membership head
   it added about 10 points on native. With s1 and s3 it trades spill for incompleteness
   almost one for one (spill −11, incomplete +10). Its value depends on how the membership
   head is calibrated, so each model has to be measured, not assumed.
4. **Game art does not benefit.** Low-poly faces have too few facets, and the hypothesis
   start leaves 59–64% of fills incomplete.
5. **Curvature inputs do not help the surface head** (s2 ≤ s1). **Training on gmsh repeats
   v4all's trade:** gmsh cylinder axes fix (p90 75° → 12°), but native torus type accuracy
   falls (68% → 39%).

**Not built yet:** the hypothesis start in `js/seedNet.js` / `js/regionGrow.js` (it exists
only in the harness), and a boundary step that resolves the spill/incompleteness trade at
the acceptance test itself.

### Synthesis, and the statistic that has to match

With disguised primitives removed, genuine revolves are 60 seeds in the 2,000-part
validation sample. `synth_bake.py` generates them: B-spline, ellipse, hyperbola and
parabola profiles; full and partial revolves; booleans with boxes, holes, tilted second
revolves; each face audited by `canonical.py` (99.6% genuine). Synthetic parts join
**training only** (`train.py --synth`); validation stays real.

The check that matters is not realism by eye but the seed's view. For a Revolution seed,
the fraction of its 512-facet patch on the same face (`neighbourhood_stats.py`):

| | mean | p10 | p50 | p90 |
|---|---|---|---|---|
| real ABC, canonical (539 seeds, 90 parts) | 0.496 | 0.09 | 0.46 | 0.98 |
| synthetic, first cut | 0.840 | 0.71 | 0.85 | 1.00 |
| + plates, buried shafts, clusters | 0.801 | 0.64 | 0.82 | — |
| + curved neighbours (fillets, holes, bosses) | 0.711 | 0.35 | 0.76 | 0.96 |
| + scene weights fitted to real, and a `tiny` scene (measured, 359 parts) | 0.563 | 0.28 | 0.55 | 0.88 |
| real ABC, whole validation split (269 seeds) | 0.504 | 0.11 | 0.47 | 0.98 |

A model trained on the first cut could learn "Revolution seed ⇒ nearly everything
belongs", which real parts punish.

**Why adding plates did nothing.** The statistic is a *facet* fraction, and tessellation
deflection scales with the part, so a large plate is a handful of triangles while a
curved revolve gets hundreds. A small knob on a plate held **87.4% of the triangles and
3.6% of the area**. Only curved neighbours dilute a revolve's facet share, as real
parts' fillets and holes do. The remaining gap is the low tail: the smallest real
revolve faces carry about 12 triangles, so a `tiny` scene (a revolve a tenth the part's
size) was added and weighted up.

**It did not work, and neither did the next guess.** A tiny revolve still gets 100+
triangles, because angular deflection sets a minimum segment count around any circle
(tiny-scene mean 0.55, 1.1% of seeds under 0.2). Turned parts with a narrow curved band
moved the statistic the wrong way (mean 0.77): OCCT subdivides a curved profile along its
length and a straight one not at all, so revolve faces averaged 1,695 triangles against 96
per cylinder face. With many segments and gentle bands, every *other* class matched real
(Plane 0.08 vs 0.11, Cylinder 0.09 vs 0.10, Cone 0.17 vs 0.16, Torus 0.52 vs 0.47), but
Revolution stayed at 0.67.

**What the low tail actually is.** Characterising real Revolution seeds by rate: under 0.2
they sit on small faces (median 84 triangles) in complex parts (median 100 faces),
surrounded by Torus 30%, Plane 21%, Cylinder 19%. Fitting every real genuine-Revolution
face against the primitives explains why those faces are small:

| best primitive residual / extent | faces | triangles p50 | rate mean | nearest |
|---|---|---|---|---|
| 1e-6 – 1e-5 | 20 | 311 | 0.30 | Torus 20 |
| 1e-5 – 1e-4 | 15 | 84 | 0.14 | Torus 12 |
| 1e-4 – 1e-3 | 45 | 36 | 0.16 | Torus 35, Cylinder 10 |
| 1e-3 – 1e-2 | 66 | 263 | 0.36 | Torus 61 |
| > 1e-2 | 78 | 474 | 0.57 | Torus 62 |

**36% of real "genuine" revolves are within 1e-3 of a torus** — fillets approximated as a
revolved B-spline, tessellated like the fillet they approximate. They are the low tail;
clearly curved revolves (median residual 4.3e-2) are the high one. No weighting of
isolated-revolve scenes could have produced them. The generator now writes half of its
rounded lathe shoulders as a B-spline through a perturbed arc, 1e-4 to 1e-2 off.

That closed the gap. Turned parts with near-torus shoulders, 120 parts, Revolution seeds:
mean **0.494** (real 0.504), p10 0.18 (0.11), p50 0.49 (0.47), p90 0.80 (0.98), with Plane,
Cylinder, Cone and Torus also within a few hundredths of real. All 383 revolve faces are
genuine at 1e-6; built directly, their residuals are 33% 1e-5–1e-4, 43% 1e-4–1e-3, 16%
1e-3–1e-2 and 8% beyond, against real's heavier far tail (29% and 35%), which the
clearly curved scenes supply.

**The full training set.** Turned parts at ~60% of draws, the other scenes kept for
variety. Predicted from per-scene rates: mean 0.516. Measured on the 4,000-part bake
(3,993 built, 10,458 genuine revolve faces, 4.9 GB):

| Revolution seeds | mean | p10 | p50 | p90 |
|---|---|---|---|---|
| real ABC, validation split | 0.504 | 0.11 | 0.47 | 0.98 |
| synthetic training set | 0.524 | 0.21 | 0.50 | 0.83 |

Plane, Cylinder, Cone and Torus are within 0.03 of real on the mean. The tails stay
narrower than real at both ends; no scene weighting moved them (p10 0.21–0.24 and p90
0.82–0.86 at every lathe share tried), so closing them would need new scene types rather
than a new mix. Synthetic BSpline faces (0.49 vs 0.24) are fillet by-products, 438 seeds
against thousands of real ones, and not what the set is for.

This also raises a labelling question the 1e-6 threshold does not answer: a surface
within 1e-4 of a torus is, for a converter working to a tolerance, a torus. Whether such
faces should be labelled `Torus` depends on the export tolerance the tool promises, and is
a product decision rather than a measurement.

### The painting studio: growth, type, training and seeding (2026-09-14/15)

All numbers on `D:/abc-data/fill_eval` (80 validation parts, four meshers) unless stated.
"Clean" = surface purity and recall both at least 95%. Native / NURBS / gmsh / game in that order.

**Crease barrier** (`regionGrow.CREASE_BARRIER`, default in `paintFrom`). Growth never crosses an
edge whose dihedral is at or above plane 2°, cylinder/cone/sphere 30°, torus 45°; the seed
neighbourhood and any supplied seed region are cut to the crease-bounded piece too. Chosen by
`test/creaseStats.mjs` (planes have 0.0° within-face dihedrals at p99 on every mesher) and a
preset sweep. Declared fills, true type, no model: clean 19/12/15/15% → 56/48/44/31%. A tangent
junction has no crease, so this is a wall added to the surface test, never a replacement.

**Welding grid** (`meshCore.weldQuantum`, `meshcore.weld_quantum`). The fixed 1e4 grid merged the
vertices of parts modelled in metres (WatchStandMedium.3mf: 708 of 4,820 triangles lost a
neighbour, which looked exactly like degenerate triangles blocking fills). Now 1e6 / bounding-box
diagonal; identical for a 100 mm part.

**Type: simplest first, capped by the model** (`paintCore.paintWith`, type `auto`). The model
proposes the region; `classifyRegion` picks the simplest primitive within 1%; if that is more
complex than the model's own type and the model's type fits within 5%, the model's type wins.
With s1: true cylinders named Cylinder 58 → 76% native, 10 → 57% gmsh; clean fills 50/42/36/31
→ 52/46/42/29%.

**Sequential painting** (`fillEval --sequential`, oracle painting of simpler faces first, painted
facets excluded from growth and from the model's patch). Non-plane seeds with s1: clean
54/42/32/20% → 60/51/58/21%; gmsh tori 4 → 53%. Excluding painted facets degraded s1's own type
head on holed patches, which v6 fixes by augmentation (below).

**Feature screen** (`test/featureScreen.mjs`, `ml/feature_screen.py`, MLP probes, even parts train,
odd test). Crease bottleneck angle: membership AUC +0.04 native, +0.10 gmsh and game. Seed-local
primitive fits: type +11 to +17 points. Curvature, geodesic distance: nothing. Triangle shape
(control): helps only the mesher it trained on.

**Rerank of every export on the current pipeline.** Mean clean over native/NURBS/gmsh with the
studio type: v3canon, v4all, v4nogmsh, v5ctrl, v5noring, v2ctrl 49%; s1, s2, s3, v5noboth 47%;
v5noarea 46%. With the true type (region quality) 58% against 54–55% for every run with
`log_area` zeroed: removing the triangulation features cost region quality once growth stopped
being the cap.

**v6feat** (`train.py --features v6 --exclude-prob 0.5`, s3all's recipe with `log_area` kept).
Inputs: the 12 base columns plus 13 from `ml/seedfeatures.py` (mirror of `js/seedFeatures.js`,
`test/netparity.mjs` to ~1e-9): crease bottleneck, local crease, normal-spread eigenvalues,
closed-form plane/cylinder/cone/sphere fits on a crease-bounded 32-facet seed ring. Half the
training patches drop neighbouring faces as already painted. Best validation IoU 0.821 (step
57,500).

| | studio type | true type | own type | sequential (all/simpler) |
|---|---|---|---|---|
| v6feat | **52%** | **60%** | 48% | **69/66/68/21** |
| best earlier | 49% | 58% | 48% | s1: 60/51/58/21 |

Native tori with the studio type 54% against 37–40%; gmsh cones 61% against 45–51%. Game art is
unchanged. v6feat is the studio's default model.

**Seed order for automatic painting** (`test/autoSeed.mjs`, whole parts painted to completion with
v6feat). Area clean: random 75/73/53/63%, **coherent 78/77/73/64%**, largest facet 70/74/74/68%,
nearest crease 70/74/68/67%, farthest from crease 72/73/63/65%. "Coherent" seeds where one
closed-form primitive best explains the crease-bounded 24-facet neighbourhood
(`autoPaint.coherenceOf`); it is the only order that never lost to random. Largest-first loses on
OCCT meshes because their largest facets are slivers across a face; seeding at a crease puts the
first patch half off the face. Shipped as the studio's Auto-paint (`js/autoPaint.js`), which
fills only unpainted facets and so works mid-project.

Every claim must be kept, however small. A first version dropped fills under 3 facets (to avoid
one-triangle groups) and lost 9–11 points of area clean on OCCT and NURBS meshes (first 20 parts:
78.5 → 68.9%, 78.1 → 67.8%): CAD tessellators often make a flat rectangle of exactly two
triangles, so the rule threw away real and sometimes large faces. Neither forcing a rejected seed
into its fill nor absorbing small unpainted islands afterwards changes the result once every claim
is kept (78.5/78.1/76.3/72.5% on the first 20 parts; game art gains over the first harness from the
low-poly defaults below).

Shipped configuration on all 80 parts (`autopaint_final`): area clean 77.4/77.5/73.4/68.4%, faces
clean 66.0/60.2/57.7/58.4%, 100% of facets painted, 5.5/6.6/9.3/2.4 s per part in Node. Median
segments per true face 1.07/1.69/1.29/2.18 (the first harness: 1.08/1.67/1.36/2.82); a few parts
reach 186–352 segments per face, raising the means, because painting now continues until every
facet is claimed where the first harness stopped at a fill budget.

**Seed snap** (`test/snapEval.mjs`). Moving a click's seed to the most coherent facet within 3/6/12
median facet sizes loses: clean 60/53/44/29% → 57/50/40/29% at 3, worse further out, because
8–48% of snaps leave the clicked face. Coherence helps choose which face to start from, not where
on a face the user already chose. Restricting candidates to the model's proposed region and
moving only when the score at least halves is the best variant and still loses: 58/52/43/29%
against 60/53/44/29%, with 2–18% of snaps off the clicked face. Not shipped.

**Tangent (curvature-jump) wall** (`regionGrow.tangentWall`, off by default). `test/tangentStats.mjs`:
at a jump of 80% of the larger total curvature it flags 30–42% of smooth (< 30°) edges between
faces of different types with 1–7% false walls inside curved faces; inside planes it is unusable
(12–29% false walls: per-facet curvature is smeared near a plane's edges). Declared single fills,
curved types only: roughly neutral (66/57/50/31% against 66/57/49/32%; spill −3 to −4 points,
incompleteness up about as much); tori gain (native 56 → 60, NURBS 33 → 41), NURBS cones lose
(80 → 67). Inside automatic painting, where incompleteness is recoverable and spill is not, it
still loses, and the stronger the wall the more (all 80 parts, with the since-corrected rule that
dropped claims under 3 facets; area clean native/NURBS/gmsh/game): none 65.8/65.3/73.3/51.7%,
c80 64.7/62.6/72.6/50.4%, c65 63.4/59.5/70.8/49.3%, c50 60.8/55.9/65.4/45.9%, with more fills and
more segments per face each step. THAT RESULT WAS AN ARTEFACT of dropping small claims. With every
claim kept (first 20 parts) the strict wall wins: area clean none 78.5/78.1/76.3/72.5%, c80
82.2/82.9/76.2/75.1%, c65 80.8/79.6/72.8/73.5%; segments per face 1.32 → 1.64 (OCCT) with c80.
Once a stopped fill's leftovers become their own segments, splitting is cheap and stopping spill
pays. The cost is over-segmentation, which the area metric barely charges: in the studio the watch
stand auto-paints to 277 groups with the wall against 66 without (85 freeform, 66 cones), and
tangent-fillet's cylinder wall splits in two. So the wall is an Auto-paint option, off by default
(`settings.autoTangent`, fill menu). The next step is a merge pass after auto-paint that rejoins
neighbouring segments whose union still fits one surface, keeping the wall's gain without the
clutter. On all 80 parts the 20-part gain does not hold: area clean 77.7/76.0/72.7/68.3% with the
wall against 77.4/77.5/73.4/68.4% without (paired per part: +0.3 ± 1.3, −1.5 ± 1.9, −0.7 ± 0.6,
−0.1 ± 0.8 points), while median segments per face rise 1.06 → 1.38 on OCCT meshes. The wall stays
an option, off by default; a boundary signal that does not split true faces is still needed.

**Learned face boundary** (`js/edgeFeatures.js`, `js/edgeWall.js`, `ml/edge_screen.py`,
`ml/edge_train.py`). Every edge of every baked part carries an exact label, so the boundary can be
learned instead of tuned. `test/edgeScreen.mjs` dumps one row per manifold edge (1.1M rows, 451k
boundaries, 80 parts x 4 meshers, 40 s): dihedral statistics, curvature and its jump, normal-spread
eigenvalues, and CROSS FITS — each side's closed-form ring fit evaluated at the facet on the other
side. Triangle-size columns exist but are left out of the shipped input set: they add within a
mesher and collapse across one (native-trained, gmsh smooth recall 16% with them, 25% without),
the same shortcut `log_area` gave the type head.

The scorer (`D:/abc-data/runs/edge1`, 32 inputs, val AUC 0.980) is checked from JS on held-out
parts at the threshold that wrongly walls 2% of within-face edges: it walls 73/65/88/55% of SMOOTH
(< 30°) boundaries — which the crease barrier catches 0% of by construction — and 92-100% of sharp
ones, costing 2.1/3.4/0.8/6.6% of within-face edges. Scoring is once per mesh: 217 ms for 8,352
edges.

End to end (`fillEval --declared-only --parity odd`, true type, so spill is isolated), no wall →
wall at that threshold: clean 62 → 69% OCCT, 57 → 62% NURBS, 48 → 58% gmsh, 43 → 46% game, with
SPILL 32 → 17%, 38 → 25%, 49 → 34%, 33 → 17%. The gains are where the hand-built walls failed:
gmsh cylinders 37 → 58%, gmsh tori 4 → 32%, OCCT spheres 66 → 83%, OCCT tori 47 → 60%. A looser
threshold (5%) cuts spill further and loses more to incompleteness. Planes regress (94 → 90% OCCT,
94 → 80% NURBS): the model sometimes walls inside a large flat face, where the 2° plane crease
already does the job, so the wall may want to be skipped for plane fills.

**Low-poly defaults** (`regionGrow.creaseTableFor`, `lowPolyDefaults`). A mesh's step is the median
dihedral of its 1–60° edges: 7.3° OCCT and NURBS, 12.3° gmsh, 22.2° game art. From 18°, curved
types' barrier becomes 2 × step and growth accepts 1.5 × step of normal deviation and 0.75 facet
sizes of distance. Game art: clean 32 → 39%, incomplete 59 → 44%, recall p50 69 → 96%, spill
23 → 31%; cones 14 → 29%, spheres 2 → 11%, tori 2 → 10%. The barrier alone gave 33% and cost
cones. Meshes below the step are unchanged by construction.

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

#### The learned face boundary, measured and settled (2026-09-15)

The edge model (`models/edge-boundary.onnx`, trained by `ml/edge_train.py` from the 34-column
per-edge rows of `js/edgeFeatures.js`) scores every edge as within-face or between-face, and
`js/edgeWall.js` turns a threshold into a growth barrier. Thresholds are named by the share of
within-face edges they wrongly wall: `fpr1`, `fpr2`.

**Planes must be excluded.** At `fpr2` the wall helps every curved type and *hurts* planes, because
a plane's own 2° crease barrier is already tighter than anything the model contributes, so a
false wall can only cost recall. Declared-type fills, 252 held-out parts (`test/fillEval.mjs
--declared-only --parity odd`), clean % by true type:

| | native Plane | native Cyl | native Sphere | native Torus | nurbs Plane | gmsh Cyl | gmsh Torus |
|---|---|---|---|---|---|---|---|
| default | 94 | 43 | 66 | 47 | 94 | 37 | 4 |
| +edge=fpr2 | **90** | 51 | 83 | 60 | **80** | 58 | 32 |
| +edge=fpr2np | 94 | 51 | 83 | 60 | 94 | 58 | 32 |

`fpr2np` — the same wall, skipped for plane fills — keeps every curved gain and gives the planes
back. `fpr1` is worse than `fpr2` on curved types, so the looser threshold is the right one.

**Whole-part auto-paint agrees** (`test/autoSeed.mjs`, 20 parts, coherent order, area clean):

| | native | nurbs | gmsh | game |
|---|---|---|---|---|
| no wall | 78.5 | 78.1 | 76.3 | 72.4 |
| fpr2 | 78.8 | 77.1 | 79.4 | 64.8 |
| fpr2np | **81.3** | **80.5** | **79.5** | 71.1 |

The wall trades spill for splitting — native spill 20.1 → 14.6% of faces, split 7.7 → 13.1% — and
the trade is favourable because spill destroys a face while a split leaves two pieces of one, which
is exactly what the union-refit merge below is for. Game assets (eval-only rips, ~22° step angle)
lose slightly; they are the one mesh class the wall does not help.

**Settled:** `settings.edgeWall = true`, `EDGE_WALL_TYPES = ['cylinder','cone','sphere','torus']`,
threshold `fpr2`, for both stroke fills and auto-paint. Confirmed live in the browser: on
`tangent-fillet.stl` all 8352 edges score and 286 become walls, and auto-paint recovers exactly its
4 true faces (plane + 2 cylinders + torus); the watch stand paints 4820/4820 triangles into 66
groups in 11 s.

#### The merge pass: four rules tried, none shippable yet (2026-09-15)

Splitting is what remains once the wall has cut spill, so the pieces have to be rejoined. The
obvious rule — join touching regions that carry the same fitted type — would fix **under 1.2% of
faces**, because a split cylinder is routinely half cylinder and half cone. `js/mergeRegions.js`
instead **refits the union**: the fitters decide, as everywhere else. Every threshold tried for
that refit failed, and the failures are worth recording because they rhyme.

| rule | segments/face | faces clean (2 native parts) |
|---|---|---|
| no merge (baseline) | 2.15 | **52.2%** |
| union relResidual <= 1% of extent | 0.15 | 0.0% |
| union residual <= 1.5x worse piece + 1/4 facet | 0.33 | 15.3% |
| ...+ 1/1000 facet floor, impure pieces barred | 1.23 | 41.9% |

Each collapsed the part, because **relResidual divides by the region's own extent, so a bigger
union looks like a better fit**, and because an already-impure region's large residual is exactly
what licenses it to absorb more. Instrumenting one part said it outright: pieces holding their own
surfaces to 1e-4 were merged under an allowance of 1e-1, 500x looser than the geometry it judged.

`test/mergeScreen.mjs` then dumped the distribution instead of guessing again — every touching pair
of painted regions, both pieces' fits, their union, and whether the truth says they are one face.
Over **658 pairs on 5 native parts: 81 same-face, 370 genuinely different, 204 involving a region
already impure** (no right answer), and only 3 that are STEP's periodic split — a hypothesis worth
testing and, here, worth discarding.

No geometric test separates them, at a budget of 1% bad merges:

| test | keeps of good merges |
|---|---|
| union relResidual | 0% (different faces routinely fit one surface EXACTLY) |
| axis alignment | 0% |
| radius agreement | 0% |
| curvature jump across shared edges | 4% |
| small piece lies ON the big piece's surface | **17%** |

Comparing the two pieces' fitted parameters cannot work, because painting splits a face into one
large piece and slivers, and **a sliver's own fit is noise**: fragments of the SAME face disagreed
on axis by 80 degrees at the median and shared a fitted type only 10 times in 81. The asymmetric
test — the big piece proposes the surface, the small piece's facets are tested against it, exactly
`growAgainst`'s criterion — is the only one with real signal (small piece 0.61 tolD from the
surface when it is the same face, 17.9 when it is not), but its tails overlap: 10% of different-face
pairs sit within 1.31 tolD, because a sliver on a seam genuinely lies on both surfaces.

**Nothing ships on those rules.** But stratifying the same pairs by the SMALLER piece answers the
question they were asked to answer, and overturns the premise of the pass:

| smaller piece | pairs | same face | different | the test that works there |
|---|---|---|---|---|
| sliver, <= 2 tris | 227 | 41 | 151 | on-surface distance: keeps 87% at a 5% bad budget |
| small, 3-9 tris | 146 | 38 | 56 | union refit: keeps **89% at a 1% bad budget** |
| medium, 10-49 | 157 | **1** | 76 | — |
| large, 50+ | 128 | **1** | 87 | — |

**98% of all good merges involve a piece of 9 triangles or fewer**, and the medium and large strata
contain one apiece. There is no "one face painted as two halves" problem to solve: auto-paint does
not halve faces, it leaves crumbs around them. That is why every rule measured across all pairs
looked useless — each was being scored mostly on sliver pairs that no geometry can place, and on
substantial pairs that never need joining.

So the pass is not a merge between two faces. It is **fragment absorption**: for each region of at
most a few triangles, choose the neighbouring region whose surface best explains it. Framing it as
a choice among competing neighbours rather than a threshold test is what makes it safe — the
fragment goes to its best host, and the alternative hosts are the very pairs that made the
threshold statistics look bad.

The two strata need different evidence, and mixing them is worse than either (share of good merges
kept, at a given share of bad ones admitted):

| fragment | test | threshold | keeps |
|---|---|---|---|
| <= 2 tris | facets lie on host's surface | 1.06 tolD, normals <= 40 deg | 87% at 5% bad |
| <= 2 tris | *union refit* | any | **0%** — a sliver fits nothing |
| 3-9 tris | union refit, relResidual | 1.4e-4 | **89% at 1% bad** |
| 3-9 tris | *on-surface distance* | 0.47 tolD | 32% at 1% bad |
| <= 9 combined | on-surface distance | 1.06 tolD | 86% at 5% bad, 0% at 1% |

A sliver is too small to fit a surface, so it must be judged against someone else's; a 3-9 triangle
fragment carries enough shape that refitting host-plus-fragment is the sharper instrument. That is
what `js/mergeRegions.js` now implements.

**Across meshers** (4635 pairs, 10 parts x native/nurbs/gmsh: 527 same-face, 2746 different, 1358
involving an impure region) the framing holds and one threshold does not. Fragments of nine
triangles or fewer are **95% / 93% / 95%** of the good merges on native / nurbs / gmsh, so this is
not an artifact of one mesher. STEP's periodic split is again negligible — 4 pairs in 4635, after 3
in 658. `uRelTol` validates: 1.42e-4 keeps 79% of good fragment merges at a 5% bad budget. But
`distTol = 1.06 tolD` was fitted on five native parts and is too loose — on the full set the sliver
threshold falls to **0.337 tolD** (70% kept at 5% bad), and `onDist max <= 0.396` does slightly
better (71%). It is not safe to read a threshold off one mesher.

**Slivers-only absorption, 20 parts, at the loose 1.06 gate** (wall on, `--merge 2`), against the
same configuration without it:

| | faces clean | area clean | segments/face | spill |
|---|---|---|---|---|
| native | 72.3 -> 72.9 | 81.3 -> **79.3** | 1.81 -> 1.30 | 14.6 -> **16.4** |
| nurbs | 69.6 -> 70.8 | 80.5 -> **79.1** | 1.91 -> 1.23 | 16.3 -> **17.9** |
| gmsh | 69.4 -> 69.1 | 79.5 -> **79.2** | 1.77 -> 1.51 | 17.8 -> **18.6** |
| game | 64.2 -> 65.3 | 71.1 -> 71.9 | 2.84 -> 2.25 | 11.8 -> **15.6** |

Over-segmentation collapses and splitting falls with it, which is what the pass is for — but area
clean drops on three meshers of four and spill rises on all four, because a crumb admitted to the
wrong host turns a split face into an impure one. Area clean is the metric that matters for export,
so **this does not ship at 1.06**; `distTol` is now 0.337 and the pass must be re-measured before
any of it reaches the studio. (Per-part timings in these runs are contention-inflated — two
measurements shared the machine — so no runtime cost should be read off them.)

#### Verdict: absorption buys cheap faces and pays in expensive area (2026-09-15)

Re-measured at the cross-mesher gate, 20 parts, wall on, as *faces clean / area clean*:

| configuration | native | nurbs | gmsh | game |
|---|---|---|---|---|
| no merge | 72.3 / **81.3** | 69.6 / **80.5** | 69.4 / **79.5** | 64.2 / **71.1** |
| slivers only (<= 2), 0.337 | 73.2 / 79.4 | 70.9 / 79.1 | 69.5 / 79.5 | 64.6 / 68.8 |
| fragments (<= 9), 0.337 | **74.0** / 78.2 | **71.6** / 78.6 | 69.4 / 79.4 | **64.8** / 68.6 |

Tightening the gate worked as predicted — against the loose 1.06 run, spill fell on every mesher
and area clean recovered — but it moved area by ~0.2 points against a deficit of ~3, so no further
threshold tuning can close it. Every variant makes the same trade: over-segmentation nearly
resolves (segments/face 1.81 -> 1.27), splitting falls by a third, faces clean rises up to 2
points, and **area clean falls on all four meshers** while spill area climbs (native 11.6 ->
16.0%).

The divergence is the finding, not an anomaly. **Face count is unweighted**: a 3-triangle face
scores the same as a 2000-triangle one. Absorbing crumbs correctly fixes a great many tiny faces,
while a single crumb placed into a large host makes that whole large face impure. Optimising the
metric that counts faces therefore corrupts the metric that measures surface, and export quality is
surface.

**The pass stays out of the studio.** `js/mergeRegions.js` and `test/mergeScreen.mjs` remain as
measured, reproducible components, off by default (`--merge 0`); no automatic merge is wired into
the UI, and the studio's existing MANUAL per-group merge is untouched — joining groups by hand
still works exactly as before, it is only the automation of it that failed to earn its place. The
bar for any future attempt is to fix crumbs **without raising spill area above 11.6% on native**.
The most promising untried idea is to drop the competition entirely and absorb only crumbs that
touch exactly ONE region: those are unambiguous by construction, and the whole error budget here
comes from choosing wrongly among several candidate hosts.

One thing the current pass structurally cannot do: the 50+ stratum holds a handful of genuine
same-face pairs (5, 4 and 3 per mesher, 2-4% of good merges), and `onDist` separates them perfectly
there (good p50 0.02-0.97 tolD against bad 8-80). Absorption only offers small fragments to larger
hosts, so it never considers them. Two large regions of one face are rare, but when they happen
they are the cheapest merges on the board.

---

## Wider context and sharper discrimination: eight tests (2026-09-24)

The question was whether the models improve if they see many triangles beyond the next one, and if
the fill is made more discriminative. Eight suggestions were tested in order, each tuned on the EVEN
parts of `D:/abc-data/fill_eval` and then scored once on the 40 ODD parts. On even parts the edge
scores come from `runs/edge1odd`, trained on the odd half, so both halves carry held-out scores.
Single fills are scored by `test/growEval.mjs` with the TRUE type declared, so every difference is
the growth step's. It picks clicks exactly as fillEval does. Outputs are in `D:/abc-data/lab/`.
Native / NURBS / gmsh / game in that order. **The baseline is today's fill** (paintFrom plus the
learned wall at fpr2 on curved types): **70/65/60/47% clean** on the odd parts.

**0. The auto-paint wall gain was not inflated.** `autoSeed.mjs --parity` now exists. On the 40 odd
parts the wall gains area clean +3.3/+2.8/+3.9/−0.4 (73.1/73.7/72.2/64.6 against
69.8/70.9/68.3/65.0), the same as the mixed 20-part estimate. The odd half is harder than the
80-part average.

**1. Where fills still fail** (`growEval --leaks`).
- **Spilled facets, by how much of the crossed boundary the wall had covered:**
  - at least 80% walled (the fill went through gaps): 33/7/18/1%
  - 20-80% walled: 21/41/24/48%
  - under 20% walled (a boundary the model did not see): 42/46/56/42%

  Closing gaps can reach at most half of the spill. The rest needs better boundary evidence.
- **What stops incomplete fills inside their own face:** a false wall in about half of them on OCCT
  and NURBS meshes (53/49%), the fit test on gmsh (90%), and the crease barrier on game art (72%).
- **An oracle veto still gains +9/+7/+17/+1** on top of the wall and never breaks a click. It refuses
  a facet when its own B-rep face's true surface explains it twice as well as the fill's surface.
  gmsh spheres go 14 → 76%.
- With the true type declared, the seed model's region adds nothing over paintFrom's own 12-facet
  start (69/63/60/46).

**2. Min-cut growth and hysteresis: no general win.** The pieces:
- `js/maxflow.js`: Boykov-Kolmogorov max-flow, checked against Edmonds-Karp by
  `test/maxflowCheck.mjs`.
- `js/cutFill.js`: each grow/refit round is a minimum cut. Each facet earns (1 − its residual score)
  for being included, and each cut edge costs λ × length × (1 − p).
- `edgeWall.hysteresisWall`: weak edges become walls when they chain to strong ones.

Held out: the cut at λ=8 scores 70/65/60/48 and hysteresis 68/62/60/44. Both trade spill for
incompleteness about one for one. For the cut, spill goes 17→11, 25→11 and 34→12 while
incompleteness goes 14→20, 15→22 and 12→32. A minimum cut prefers short boundaries, so it trims tori
and spheres. It wins on cylinders (51→61, 54→62). Applied to cylinders only, the rule fixed on the
even parts gives 73/68/60/47 (+3/+3/0/0). With a plain length cost and no edge scores the cut loses
outright, so the learned scores are doing the work.

**3a. Boundary-path features for the seed model are real but small.** Two features were tested: the
lowest achievable maximum boundary probability along paths from the seed, and the lowest sum of
−log(1−p) (`test/boundaryScreen.mjs`, `ml/boundary_screen.py`). On crease-reachable facets they cut
v6feat's membership error by 14-21% (AUC +0.004 to +0.009 over its own logit). Step 1 shows the seed
region no longer limits declared fills, so this was not pursued.

**3b. Message passing makes a better seed model, and a smaller gain in fills.** `model.MeshConv`
(`train.py --gnn-layers 6`) adds six rounds of message passing between the per-facet encoder and the
pool. Each facet has at most three neighbours, so this is a fixed gather that exports to ONNX as a
Gather. The patch graph is `patches.patch_graph` / `seedNet.patchNeighbours`, checked by netparity.
`runs/v7gnn` repeats v6feat's recipe and data stream with 709k parameters instead of 409k.
- Validation IoU 0.821 → 0.859.
- fillEval `--ordered` on all 80 parts, v6feat → v7gnn:
  - model region alone, clean: 30/24/29/53 → 37/30/34/60
  - model's own type, accuracy: 75/66/67/54 → 84/70/71/64
  - model's own type, clean: 58/47/40/42 → 63/49/44/46
  - studio type (ordered+model), clean: 60/53/44/36 → 64/54/46/37
  - true type seeded by the region, clean: 69/61/50/43 → 71/62/51/45
- Auto-paint on the odd parts with edge2: area clean 75.3/75.3/76.0/65.8 → 76.3/77.0/77.2/66.8,
  faces clean +2.4/+2.0/+2.2/+0.8.

The model improves a lot and the fill a little, as step 1 predicts: growth and the fitters, not the
seed region, now decide most outcomes.

**4a. Weighting the loss toward smooth boundaries does not pay.** `train.py --hard-weight 4
--hard-rings 2 --boundary-seeds 0.3` weights the loss 4× on facets within 2 rings of a smooth (<30°)
boundary of the seed's face, and redraws 30% of the seeds from facets on such boundaries. The run is
`runs/v7hard`, with v6feat's recipe otherwise; validation IoU is 0.816 (v6feat 0.821).
- fillEval `--ordered` on all 80 parts, v6feat → v7hard:
  - model region alone, clean: 30/24/29/53 → 34/26/30/52
  - model's own type, type accuracy: flat
  - model's own type, clean: 58/47/40/42 → 57/46/41/42
  - studio type, clean: 60/53/44/36 → 61/53/45/36
- Auto-paint on the odd parts with edge2: area clean 75.3/75.3/76.0/65.8 → 76.1/75.3/76.7/66.9.

The region improves slightly and the fill does not. Message passing (3b) beat it on every measure.

**4b. Calibrating the membership veto fixes the veto, not the fill.** At τ=0.5 the veto wrongly
rejects 6.9% of the fill's own facets and catches only a third of the foreign ones. Setting τ by a
1% false-reject rate on the even parts (τ=0.22) gains +6/+6/+1/0 over τ=0.5. The veto path still
loses to the plain wall (71/59/59/37), because its adaptive angle and missing low-poly tolerances
cost completeness.

**5a. The edge model trained on 50× the parts gives the largest single gain.** It uses 2000
training-split parts × 4 meshers, 21M edges (`export_fill_eval.py --split train`), with thresholds
set on the even fill_eval parts (`edge_train.py --val-prefix --val-parity even`). The model is
`runs/edge2`.
- Validation AUC 0.980 → 0.990. Smooth-boundary recall at 2% false walls 77 → 86% (game 38 → 51).
- Held-out fills 78/65/69/48 (+8/0/+9/+1); paired clicks fixed/broken +24/−6, +11/−11, +28/−4, +6/−3.
- Spill 9/20/21/13 against 17/25/34/17. Native tori 60 → 74%, gmsh spheres 14 → 41%.
- Auto-paint area clean 75.3/75.3/76.0/65.8 (+2.2/+1.6/+3.8/+1.2); faces clean +6.6/+4.9/+7.1/+0.7.

**5b. A second-stage (stacked) edge model helps per edge but not per fill.** It also sees the first
stage's scores around each edge (`edgeFeatures.CONTEXT_COLUMNS`, out-of-fold stacking). On the same
1000 parts it raises smooth recall at 2% false walls 84 → 88%. Fills do not follow: 75/66/71/46 at
fpr2 and 77/70/69/47 at fpr1. Its errors line up along curves, and a chain of false walls splits a
face where scattered ones do not.

**6. Joint re-assignment (PEARL) is rejected.** `js/pearl.js` runs α-expansion over auto-paint's
segment surfaces, with the cut's pairwise term, then refits and repeats. On the even parts it lowers
area clean on CAD meshes and game art: with the strict rule and freeform segments locked,
77.8/79.5/87.0/65.5 against 82.6/83.2/85.5/72.0. Faces clean rises on gmsh. A per-facet audit
repeats the merge pass's finding: a facet within tolerance of two surfaces cannot be placed by fit.
Large OCCT slivers go to the wrong neighbour, and tori lose facets to spheres. It was not tested on
the odd parts.

**7. A refinement model is the best single-fill change.** It is `SeedFillNet` with 9 more inputs per
facet: whether the first fill holds it, its distance, angle and side relative to the fill's fitted
surface, the boundary-path features, and an optional negative click (`js/refineFeatures.js`,
`test/refineDump.mjs`, `ml/refine_train.py`, `runs/refine1`). It was trained on 26k fills of 1000
training-split parts, with edge scores from a model that never saw them; training takes 3 minutes.
The logit cut of 1 was chosen on the even parts. Held out:
- **79/69/65/49** (+9/+4/+5/+2). It almost never breaks a clean fill.
- With one simulated negative click where it still spills: 82/69/66/50. A click without a model
  (the simplest surface around the click claims what it explains better) gives 73/66/61/47.
- It stacks with the new edge model: **85/72/76/52**, and 87/72/78/53 with the click. Paired against
  edge2's wall: +18/−1, +16/−1, +17/−0, +11/−2.
- Regrowing after refining is worse than keeping the refined region. Retraining the refiner on
  edge2-quality scores changes nothing (84/71/75/51).
- Inside auto-paint (every fill refined before it is claimed): area clean +1.3/−0.8/+2.9/+3.1, faces
  clean +2.7/+1.1/+2.9/+4.6.

**8. A verifier over automatic candidates helps where they disagree.** `growEval --verifier-features`
gives every fill 15 region features computed without the truth: size, fit residual, how the outline
sits on the learned boundary and on creases, facets just outside that would also fit, and walls
inside. `ml/verifier_train.py` learns P(clean) on the even parts and, on the odd parts, picks per
click among 9 candidates: the refiner at three cuts, the wall at three thresholds, flood, the cut,
and refine+grow.
- Per-fill AUC 0.958.
- Clean **85/80/82/53**, against 85/72/76/52 for the refiner alone and 91/85/85/55 for a perfect
  chooser. Paired against the refiner: +8/−7, +20/−1, +17/−1, +4/−0.
- Mind the NaN: a region whose fit fails has no residual, and one NaN input made every score NaN, so
  the pick silently became the first method.

**Follow-ups (2026-09-25): the refiner loop, and the refiner inside a minimum cut.** Both run on
edge2 plus `runs/refine1`, tuned on the even parts and tested once on the odd parts
(`D:/abc-data/lab/s9_*`).
- **Refining repeatedly is flat.** `growEval iter:<n>` refits the surface to the refined region and
  runs the refiner again. Two passes score 85/71/77/53 against 85/72/76/52 for one pass, and three
  passes add nothing on the even parts. The refiner has converged after one pass.
- **The min-cut works once its per-facet term is learned.** This is Kalogerakis et al.'s full recipe,
  `growEval crf:a=2;l=4;r=1`. Inside the refiner's patch, the per-facet term is the refiner's logit
  (minus the tuned cut of 1, scaled by 1/4, clamped to ±1, weighted 2). Outside the patch it is the fit
  residual. Cut edges cost λ=4 × length × (1 − p). Held out it scores **87/81/82/54** against
  85/72/76/52 for the refiner alone. Paired against the refiner it fixes 9/25/17/8 clicks and breaks
  6/4/1/2. Against edge2's plain wall it fixes 22/40/33/17 and breaks 2/4/0/2. The biggest gains are
  NURBS spheres 62 → 86%, gmsh tori 53 → 66%, gmsh cones 71 → 82% and NURBS cylinders 64 → 75%.

  Step 2's cut failed because its per-facet term was the fit residual, which cannot separate a tangent
  neighbour that fits within tolerance. With a learned per-facet term, pricing the outline beats
  flooding. As one method it matches the nine-candidate verifier (85/80/82/53).
- **It carries over to auto-paint.** `autoSeed --refine runs/refine1/refine.onnx --crf` refines every
  fill and re-chooses it by the cut before it is claimed. On the odd parts with edge2 it scores area
  clean **81.8/82.5/85.4/70.3**. That compares with 76.6/74.5/78.9/68.9 for the refiner alone,
  75.3/75.3/76.0/65.8 for edge2's wall, and 73.1/73.7/72.2/64.6 for the shipped edge1 wall. Faces clean
  is 75.7/71.0/75.4/61.2 against 70.9/63.6/69.3/59.3 for the refiner. Split area falls (native
  15.8 → 11.4%), and it needs fewer fills (208 against 249 on native).

**Where this leaves single fills.** On the same held-out clicks, today's fill scores 70/65/60/47
clean. Edge2 plus the refiner plus the verifier scores 85/80/82/53. Edge2 plus the refiner inside the
minimum cut scores **87/81/82/54**.

**In the studio (2026-09-25).** `paintCore.refineFill` (the refiner, then the cut) runs behind
`paintWith`'s `refine` option, so the studio, `autoSeed` and `growEval studio:*` all run one piece of
code. The refactored auto-paint reproduces its earlier run exactly (81.8/82.5/85.4/70.3).
- **Browser.** `paintTool` feeds v7gnn its patch graph as int64 tensors; onnxruntime-web accepts them.
- **Settings.** The "Refined fills" setting switches on boundary model v2, the refiner and the cut for
  strokes and auto-paint.
- **Defaults.** Refined fills are on, the seed model is v7, and the stroke method is `model`, which
  beats `hyp` with or without refinement.

Confirmed through the studio's own path with the type chosen automatically, on the odd parts:
- **Single clicks:** today's defaults (v6feat, `hyp`, v1 wall) score 65/59/54/37. The new defaults
  score **83/76/77/51**.
- **Auto-paint area clean:** 73.1/73.7/72.2/64.6 → **85.6/84.7/86.6/72.3**. Faces clean is
  80.0/73.8/78.4/62.1.
- **v7gnn gains more inside the stack than alone:** +1.2 to +3.8 area clean over v6feat with the same
  stack.

**One real part disagrees about the boundary model.** `test/watchstandAuto.mjs` scores auto-paint on
the WatchStand meshes against `WatchStandTruth.step`: truth surfaces recovered out of 77, with a
ceiling of 62.

| configuration (WatchStandMedium) | groups | recovered |
|---|---|---|
| before (v6feat + v1 wall) | 63 | 34 |
| v1 wall + v7gnn + refiner | 68 | 34 |
| v2 wall + v7gnn + refiner (new default) | 117 | 35 |
| v2 wall alone (v6feat) | 139 | 33 |

- **Recovery is the same, but v2 nearly doubles the groups.** The refiner undoes part of it (139 →
  94 with v6feat). The coarse mesh behaves similarly (30 or 32 recovered either way; 62-75 groups).
- **On ABC, v2 does not over-segment and is better.** With v7gnn and refinement, v1 against v2 on
  the odd parts:
  - single clicks 79/70/68/48 against 83/76/77/51
  - auto-paint area clean 83.7/82.0/84.3/71.0 against 85.6/84.7/86.6/72.3
  - segments per face identical (5.93/10.38/1.71/3.31 against 6.08/10.43/1.71/3.33)
- **The difference is in smooth edges.** v2 walls 4.4% of the watch stand's smooth edges against
  2.7% for v1, and its stricter fpr1 threshold barely changes that.

**Real exports at scale: 300 Onshape tessellations (2026-09-25).**

*The set.* ABC also publishes the meshes Onshape itself exported (the `obj` format). Each feature
file lists the OBJ triangles of every surface (`face_indices`), so these are real-exporter meshes
with exact per-triangle labels.
- `ml/export_abc_obj_eval.py` writes 300 of them from chunk 0 to `D:/abc-data/onshape_eval`, in
  fillEval's format as variant `onshape`.
- It excludes any model that appears in any bake (every mesher, either split), in fill_eval or in
  the edge-training set, so no trained component has seen them.
- Every exported label set covered its mesh exactly once.
- The meshes are dense (median 30k triangles, capped at 60k) and mostly planes and cylinders:
  9,651 planes, 3,213 cylinders, 402 B-splines, 300 cones and 169 tori.

*Auto-paint* (`autoSeed --shard k/3`, outputs `D:/abc-data/lab/s12_onshape_*`):

| configuration | faces clean | area clean | groups |
|---|---|---|---|
| before (v6feat + v1 wall) | 84.9 | 92.5 | 35,828 |
| v1 wall + v7gnn + refiner | 87.7 | 93.9 | 33,858 |
| v2 wall + v7gnn + refiner (default) | 87.7 | 93.7 | 34,940 |

- **Both stacks beat today's defaults.** Area clean rises +1.4 (v1) and +1.2 (v2) points, paired SE
  0.2–0.3. They are better on 55 and 58 parts, and worse on 4 and 9. Both make fewer groups.
- **v1 and v2 tie on quality:** v2 − v1 = −0.2 area clean (SE 0.2). Faces clean is identical.
- **v2 makes 3% more groups.** The median per-part ratio is 1.00, but 6 of the 300 parts get at
  least 1.5× the groups.
  - On those parts every extra group sits on a cylinder or torus. Planes are never split.
  - Examples: 4 cylinders become 43 pieces; 70 cylinders become 403, of which 287 are slivers of
    3 facets or fewer.
- **The mechanism: v2 walls Onshape's cylinder strips.** Exact wall rates at fpr2 come from the
  labels (scratchpad `wallrate.mjs`).
  - Inside curved faces on Onshape meshes, v2 walls 1.81% of edges against 0.64% for v1 (smooth
    edges: 1.52 against 0.37%).
  - On our held-out meshers the ratio is only about 1.2× (native 2.63 against 2.11%).
  - v2 also finds more true tangent boundaries: 57 against 47% on Onshape, 89 against 74% on
    native.
- **By coarseness,** in triangles per true face: the coarsest third scores 94.5 (v1) against 94.0
  (v2) area clean, and the other two thirds tie.

*Real STEP and mesh pairs* (`test/realPairs.mjs`, 23 candidate pairs from Downloads). Few are
usable. Many are different parts that share a name. Others are rescaled copies, whose scale is no
unit conversion. A threaded test cube is 342 MB of XML. The loader now reads Bambu/Prusa multi-file
3MFs.
- The one informative pair is a SolidWorks 2025 export, `Base`, scored as truth surfaces recovered
  out of 158:

  | configuration | groups | recovered |
  |---|---|---|
  | v2 stack | 466 | 127 |
  | v1 stack | 494 | 121 |
  | before | 686 | 109 |

- The other usable pair ties.
- Caveat: the matcher pairs planes by normal alone, so extra groups can inflate recovery on parts
  with many parallel planes.

*Where this leaves the boundary model.*
- **On the meshes we generate,** v2 wins by about 2 points.
- **On real Onshape exports,** v1 and v2 tie, and v2's extra walls cut Onshape's cylinder strips.
  The WatchStand is a coarse Onshape export, so it shows this at its worst.
- **The fix to try is training data, not a choice between models.** ABC chunk 0 alone has 7,168
  labelled Onshape meshes, and only 300 of them are in this evaluation set. This repeats the type
  head's lesson: the model reads the mesher's pattern.

---

## Scoring what a solid needs: surfaces from confident cores (2026-09-26)

**The question.** A painted triangle can sit on the wrong face for two reasons. The fill made a
mistake, or the tessellation left a triangle that fits two surfaces equally well. If each group's
surface is fitted only on the triangles it is confident about, boundary slop may not matter. The topo
rebuild (`ml/solid_rebuild.py`) projects every boundary vertex onto both fitted surfaces, so a
ragged boundary between two correct surfaces lands on the true edge anyway.

**The scorecard** (`test/surfaceScore.mjs`, on `test/autoSeed.mjs --dump-seg` output). Truth comes
from `ml/export_fit_truth.py`, and for the Onshape set from `ml/export_abc_obj_eval.py --truth-only`,
whose surfaces lie on the OBJ vertices to about 1e-8 of the diagonal. Per true analytic face, with L
the group holding most of the face:
- **found:** L's surface has the true type and lies within 1e-3 × the part diagonal of every point
  of the true face (found4: 1e-4)
- **once:** found, and L holds at least 95% of the face
- **swallowed:** not found, and L belongs mostly to another face
- **spurious groups:** groups that are no face's L, i.e. extra faces in a solid
- **wrong-face paint,** split three ways:
  - harmless: every vertex within 1e-3 × diagonal of its group's own true surface
  - harmful
  - freeform: the group's own face is not analytic

End to end, `ml/rebuild_patches.py` runs the topo rebuild on the predicted groups of every fully
analytic part. A solid counts as **recreated** when it is closed and valid, with volume within 1%. The
reference is the STEP volume, or the mesh volume on the Onshape set.

Held out: the 40 odd fill_eval parts on all four meshers, and the 300 Onshape meshes. Outputs are in
`D:/abc-data/lab/s13/`. The numbers below are for the default stack unless marked.

**1. Confident cores do not help.** Paired change in found when each group is fitted on a core
instead of on every triangle:

| core | fill_eval | Onshape |
|---|---|---|
| facets at least 2 steps from another group | −0.2 | −0.0 |
| at least 3 steps | −0.6 (found4) | −0.0 |
| refiner membership ≥ 0.9 | −0.5 (found4) | −0.1 |
| no edge above the boundary model's threshold | −0.4 (found4) | −0.0 |
| refit without the farthest 20% | +0.1 (found4) | −0.0 |

- **The rebuilds agree:** Onshape 182 → 184 of 275 recreated, fill_eval 14 → 13 of 45.
- **Why:** a group of the right type that fits badly is only 0–6% of faces. On the true faces, the
  fitter recovers the surface 96–100% of the time. It copes with the spill the painting actually
  leaves.

**2. Spill is small, and on our meshers mostly real.**
- **fill_eval:** wrong-face paint is 1.4–2.6% of area: harmless 0.1–0.3, harmful 0.7–1.0, freeform
  0.3–1.6.
- **Onshape:** 1.2%: harmless 0.5, harmful 0.3, freeform 0.3.

**3. What the misses are.**
- **No surface at all.** The fill found no analytic type and left the group untyped. Typically this
  is a flat face meshed as two triangles, which `classifyRegion` refuses: it wants three.
- **Wrong type.** 18.5% of cylinders on fill_eval (as torus, cone, plane or sphere) and 21.6% on
  Onshape (as plane, cone or torus). 23% and 15% of cones. 41% of Onshape tori, mostly as cones.
- **Swallowed:** 3–9%.
- **Small faces.** Cylinders of at most 12 facets are found 46% (fill_eval) and 21% (Onshape) of the
  time, against 82% and 76% for larger ones.

**4. What helps.**

| groups | found, fill | once, fill | found, Onshape | once, Onshape | recreated, fill (45) | recreated, Onshape (275) |
|---|---|---|---|---|---|---|
| as painted | 66.7 | 60.8 | 93.1 | 89.5 | 9 | 175 |
| untyped groups typed | 81.4 | 75.1 | 93.4 | 89.7 | 14 | 182 |
| then triage and merge | 79.8 | 77.3 | 93.6 | 91.9 | 14 | 196 |
| true groups, fitted surfaces | 97.4 | 97.4 | 100 | 100 | 36 | 254 |

- **Typing untyped groups** runs `classifyRegion` on the group's own facets, or fits a plane through
  one or two coplanar facets.
  - fill_eval: +14.8 found, up on 105 of 160 part-meshes and down on none.
  - Onshape: only +0.3 found, yet +7 solids, because one group without a surface stops a solid
    closing.
- **Triage** moves a facet that is off its own group's core surface to an adjacent group whose core
  surface it is on. Once: +2.0 on fill_eval (82 up, 0 down) and +0.8 on Onshape (38 up, 0 down).
- **Merge** joins a group to an adjacent group whose core surface it lies on.
  - It removes spurious groups, but swallows faces: native swallowed goes 3.4 → 9.5%.
  - With triage it adds 14 solids on Onshape and none on fill_eval.
  - A tolerance of 2e-4 × diagonal beats 5e-4 and 1e-3.
- **Earlier rejections, rescored.** PEARL and the old merge pass were rejected by clean area. On this
  scorecard they add +1.9 to +2.6 and +0.2 to +0.4 to once.

**5. Typing is the biggest lever left.**
- **The bound:** typing each painted group as its majority face truly is (an oracle, not a method)
  would add +9.3 found on fill_eval (117 up, 0 down) and +3.6 on Onshape (94 up, 2 down).
- **Re-classifying every final group** with `classifyRegion` gets +1.1 on fill_eval (62 up, 33 down;
  worse on game meshes) and +1.4 on Onshape. With triage and merge on top, Onshape reaches once 93.3%
  (+3.7, 89 up, 12 down).
- **In rebuilds, re-classifying is the best step measured.** Recreated solids:

  | groups | fill_eval (45) | Onshape (275) |
  |---|---|---|
  | untyped groups typed | 14 | 182 |
  | every group re-classified | 16 | 198 |
  | re-classified, then triage and merge | 13 | **211** |
  | typed as their true faces (oracle) | 16 | 195 |

- **Truth typing does not beat typing by fit.** A group's surface has to fit the triangles the group
  actually holds, and the simplest type that fits them does that best. On Onshape, as-painted groups
  rebuild 64% of parts; re-classifying, triage and merge rebuild 77%; the ceiling is 92%.
- **Why the types are wrong:** a fill chooses its type on the model's first 512-facet patch, before
  the group has grown.

**6. The boundary models on this scorecard.**
- **fill_eval:** v2 − v1 = +1.3 found, +2.8 once.
- **Onshape:**
  - As painted: +0.0 found, −0.3 ± 0.2 once, and 3% more spurious groups.
  - After triage and merge: +0.2 once, and 196 against 191 solids recreated.

**What to take from it.**
- **Keep fitting every triangle.** A group's surface is right whenever its type is.
- **Ship typing of untyped groups.** It has no regressions anywhere, and adds 5 of 45 and 7 of 275
  rebuilt solids.
- **Then re-type every group on its final triangles at the end of auto-paint.** On real exports,
  with triage and merge, it takes rebuilt solids from 175 to 211 of 275. Game meshes need a guard:
  `classifyRegion` does worse there than the painted type.
- **Triage never makes things worse.** With merge, it buys solids on real exports at the cost of
  some swallowed faces.
- **Type selection on the final group is the next target.** A perfect typer is worth 9 points of
  found faces on our meshers.
- **Judge segmentation by this scorecard and by rebuilds.** Do not use clean alone: it misses
  surfaceless groups and spurious slivers, and it punishes harmless slop.

---

## Two lessons from next.BREP.io: exact final typing and a facet fallback (2026-10-02)

**Where they come from.** next.BREP.io's `stl2step` (Autodrop3d, Rust, no ML) was run on the same
meshes as ours (`D:/abc-data/lab/competitor_brepio`). It has a dihedral-growing segmentation and a
plane → cylinder → cone → sphere → torus fit cascade. Off OCCT meshes it loses clearly; once per part,
after triage/merge, ours vs theirs:

| mesher | ours | theirs |
|---|---|---|
| native | 81.9 | 70.9 |
| NURBS | 79.5 | 64.7 |
| gmsh | 82.2 | 60.6 |
| game | 65.6 | 52.5 |

It did two things better:
- **Onshape planes and cones.** It found 46 planes and 11 cones once that we did not. Some were
  whole groups of ours with the wrong painted type (a plane painted as a torus, fitting it to 2e-6);
  the rest were groups that merged several faces.
- **A solid for every part.** It returned a closed, valid solid for 36 of 45 fill_eval meshes,
  demoting regions it could not join exactly to facets. The knitter returned one for 15.

Outputs for this section are in `D:/abc-data/lab/s14_exact` and the competitor folder.

**1. Final-group typing at mesh precision** (`js/paintCore.js` `typeFinalGroup`; scorer variants
`exact` and `etm` = exact, then triage and merge).

**How it works.** `classifyRegion` on the finished group replaces the painted type only when its fit
holds every vertex within 1e-6 × the part diagonal, on at least 3× the type's degrees of freedom in
distinct vertices. Untyped groups are typed as before.
- **Why exact.** On pure groups (≥ 95% one true face) `classifyRegion` is right 98% of the time on
  Onshape and 92–95% on OCCT/NURBS/gmsh. A CAD tessellation's pure group has its vertices on its
  surface; an impure one fits no surface exactly.
- **Why the support guard.** Without it, low-poly groups are fitted exactly by any type with enough
  parameters. On game meshes 89 right cylinders of 7–10 vertices became tori, spheres and cones, and
  once fell 2.4 points.

**Once per part** (40 parts per mesher; up/down are parts against tm):

| mesher | tm | etm | parts up/down | rtm (every group re-classified), for reference |
|---|---|---|---|---|
| Onshape (40) | 94.5 | 95.5 | 5 / 0 | 95.8 |
| native | 81.9 | 82.8 | 8 / 1 | 82.9 |
| NURBS | 79.5 | 80.2 | 12 / 0 | 80.9 |
| gmsh | 82.2 | 84.3 | 19 / 0 | 85.5 |
| game | 65.6 | 65.5 | 5 / 4 | 63.0 |

- **All 300 Onshape meshes:** once 91.9 → 93.0, 52 parts up and 1 down.
- **Rebuilds** (repair loop): fully analytic solids Onshape 247 → 249 of 275 (3 gained, 1 lost; the lost
  part became a hybrid solid at 0.8% facets) and fill_eval 15 → 17 of 45.
- **Shipped:** the studio's auto-paint now types every finished group this way inside the same undo
  step. That also ships the typing of untyped groups that the scorecard section recommended.

**2. The facet fallback** (`ml/solid_rebuild.py --fallback`, `ml/rebuild_patches.py --fallback`).
When the knitter (with the repair loop) does not recreate a part, these are tried in order:
- **hybrid:** the patches it still has problems with, and every surfaceless patch, become facets (a
  planar face per triangle, straight shared edges), with up to 3 rounds adding newly failing patches.
- **planar:** only plane patches stay analytic.
- **faceted:** the mesh itself as a polyhedral solid, also used when the analytic build throws.

Each part reports its tier and its faceted share of area; a fully analytic solid is counted apart.
Recreated, by tier (same criterion as above):

| set | analytic | hybrid | planar | faceted | none |
|---|---|---|---|---|---|
| Onshape, tm-ring1 (275) | 247 | 6 (median 0.8% of area faceted) | 6 (median 29%) | 10 | 6 |
| Onshape, etm-ring1 (275) | 249 | 6 | 5 | 9 | 6 |
| fill_eval, etm-ring1 (45) | 17 | 13 (median 13%) | 2 | 0 | 6 |

- **Against the STEP volume.** On fill_eval a faceted or planar solid copies the mesh, so it passes only
  where the mesh is within 1% of the STEP volume. Against the mesh volume, as `stl2step` was judged, 39
  of 45 have a solid (next.BREP.io: 36). Onshape has no STEP volumes.
- **"none" is the input.** Every such part's mesh is open: 2 to 6,118 free edges even as pure facets,
  from holes or edges with 3+ triangles where bodies touch, not T-junctions or unwelded vertices.
  Closing them needs mesh repair (cap holes, resolve non-manifold edges), which is not built.
- **Cost:** the fallback adds a median of 2–6 s to a hybrid part and up to about 60 s to a faceted one.
- **Not in the browser yet:** the studio's STEP export is still the sewing builder; the knitter, its
  loop and this fallback are Python only. (Done the same day: next section.)

---

## The studio exports with the topo knitter (2026-10-02)

**What changed.** The studio's Export STEP no longer uses `js/brepBuilder.js`. That builder made every
face on its own, as a UV rectangle for curved faces, sewed them at 0.5% of the diagonal, and closed 0 of
31 fully analytic test parts. The export now runs `js/knit/`, a port of `ml/solid_rebuild.py`:
- topology from the mesh, so neighbouring faces share their edges by construction;
- the kernel-judged repair loop, with retype candidates from `gaussMap.fitRegionAs` in-process;
- the facet fallback.

It runs in a module Web Worker (`js/knit/worker.js`) on opencascade.js 2.0.0-beta.b5ff984, the full
build (OCCT 7.6 era, 50 MB from unpkg, cached). Node runs the same modules from a dev dependency
(`test/knit.mjs`, `test/knitRebuild.mjs`, `test/studioExport.mjs`).

The export message says what came out:
- every face analytic;
- the share of the area that went out as facets;
- the mesh as a faceted solid;
- or not a closed solid, and why.

The loader centres every mesh for the view; the export moves the positions and surfaces back, so the STEP
sits where the file's mesh did. Every export before this one was shifted.

**Two kernel differences, both found by holding the port against the Python measurements.**
- **`GeomAPI_Interpolate` cannot be called.** opencascade.js has no binding for its point-array
  argument, so `knit/geometry.js` builds the same curve. A first version with averaged knots made one
  valid torus face of 00032949_19 (gmsh) invalid, in the Python knitter as well when swapped in. OCCT's
  scheme, reverse-engineered and matched to 4e-15 on random chains:
  - open chains: a cubic with knots at the chord-length parameters; end tangents are the derivative of
    the cubic Lagrange polynomial through the first and last four points;
  - closed chains: periodic, C1 at the seam, with the tangent taken from the first four points.
- **OCCT 7.6's `ShapeFix_Shell`.** Its full `Perform` turned that part's valid shell unorientable, and its
  orientation pass is quadratic: 776 s on a 33,802-facet shell. The solid check therefore tries the
  shell as built first. Only an invalid shell of at most 3,000 faces gets the orientation fix, then
  `Perform`.

**Parity, on the same patches the Python harness rebuilt** (`D:/abc-data/lab/s15_knit`):

| set | recreated, Python | recreated, JS | fully analytic, Python | fully analytic, JS |
|---|---|---|---|---|
| Onshape, etm-ring1 (275) | 269 | 269 | 249 | **253** |
| fill_eval, etm-ring1 (45) | 39 by mesh volume | 39 | 17 | 17 |

- **Per part:** 248 Onshape parts are analytic in both. One part is analytic only in Python
  (00000890_001, planar tier in JS); 5 are analytic only in JS.
- **Time:** about 2× Python's on facet tiers. Onshape took 782 s on 8 workers.

**In the browser** (headless Chrome through `studio.html`, with puppeteer-core outside the repo):
- `Star.stl` from the user's Downloads: auto-paint 12 s, export 24 s, "a closed solid, every face on its
  fitted surface". The STEP is valid in OCCT 7.9 and 0.19% off the CAD volume, centred where the CAD is.
- The coarse `Star (2).stl`: a closed solid with 51% of the area faceted.

**The user's sample parts are freeform.** Their Onshape STEPs are almost all B-spline surfaces:
- Star: 20 of 20 faces
- Fencepost: 12 of 17
- Cake Pan: 28 of 30
- Ornament: 14 of 16

Auto-paint knows only primitives, so it carves each freeform face into many primitive patches: 458
groups for the ornament's 16 faces, 28 "tori" on the coarse fence post. Those are the bad surfaces the
user saw. The export now makes closed, valid solids of them, but with large facet shares:

| part | triangles | groups | result |
|---|---|---|---|
| Star | 3,330 | 27 | analytic |
| Star (1) | 1,688 | 45 | 94% facets |
| Star (2) | 1,190 | 35 | 51% facets |
| Fencepost (2) | 3,022 | 48 | 55% facets |
| Christmas Ornament 2 (2) | 15,900 | 458 | all facets |

**Known issues.**
- **Spikes at sharp tips.** Placing a corner on 3+ nearly coplanar surfaces is ill-conditioned: Star's
  analytic solid has thin fins reaching 100 mm past its tips, which validity and volume cannot see.
  Keeping such corners on the mesh vertex, with or without a vertex tolerance, made Star worse, so
  placement is left as measured. Python has the same behaviour.
- **Slow builds on badly typed coarse meshes.** OCCT's shape repair spends minutes on surfaces that do not
  fit their triangles: Fencepost (2)'s first build took 198 s in `ShapeFix_Shape` and 66 s in validity
  checks. A fit-deviation cutoff does not separate these, because correctly segmented freeform Onshape
  parts have deviations in the same range.
- **No freeform surfaces in the browser.** `ml/freeform_fit.py`'s B-splines are not ported, and auto-paint
  does not keep a freeform face as one group. For the user's parts, that is the next step.

## Freeform faces in the studio: one group, one B-spline (2026-10-03)

**What changed.** Three pieces, all in the studio's path:
1. **`js/freeformMerge.js`** runs at the end of auto-paint. It joins adjacent groups that no primitive fits
   exactly, and leaves the joined groups untyped (freeform).
2. **`js/freeformFit.js`** is a port of `ml/freeform_fit.py`. It fits one trimmed bicubic B-spline per
   untyped group:
   - a plane projection or LSCM map;
   - a ring or tube gets a periodic map and a surface periodic round it;
   - a group no single map covers is split by normal cones (`fitPieces`).
3. **`js/knit`** builds those as `Geom_BSplineSurface` faces. A ring's surface is periodic in u, and the
   repair adds its seam as it does a cylinder's.

**The fitter matches Python.**
- **Twisty true faces:** Star and Fencepost give identical errors and spans; Cake Pan (LSCM) is within
  hold-out noise.
- **Synthetic rings:** the same spans, with errors within hold-out noise.
- **Speed:** 2.5-4× faster than Python.

With true groups, the JS knitter makes every twisty part except the medium Ornament a closed, valid
solid with every face on a fitted surface. The coarse Ornament reached that only once the splitter
existed; it was a hybrid before. The medium Ornament splits into 49 pieces and still comes out faceted.

**The merge rule, and the gate that caught it.** A CAD tessellation's vertices lie on the true surface, so
a group on an analytic face fits its primitive exactly and a piece of a freeform face does not. But
auto-paint's groups on analytic faces are often impure, with a few triangles of a neighbour, or two faces
in one. Judged whole, 76% of their area was "inexact" on the 14 Onshape parts the merge acted on, and
they joined freeform pieces.

The grouping metric (`test/freeformSeg.mjs`) could not see this. The typing score (`test/surfaceScore.mjs`
fetm vs etm) could: analytic once on those 14 parts fell 89.7 → 63.4, and swallowed faces rose 1.6 →
24.3%. Three changes, now the defaults:
- **Robust exactness.** A group is also exact when its core (two boundary rings peeled off) fits a
  primitive exactly, or when one plane through a group triangle holds 30% of its vertices, not collinear.
  By area, groups on analytic faces judged exact went 23.6% → 95.3%; freeform pieces judged inexact went
  99.5 → 99.2%.
- **keepFlat.** A small group that holds a plane is never absorbed.
- **The part gate over judged groups only.** The coarse Fencepost's flat sides are a few large triangles
  each and held it at 0.35, so it was never merged.
- **Every inexact group is freeform, joined or not.** `joinedOnly`, a lone group keeping its primitive,
  had protected impure groups judged whole. With robust exactness it changes neither Onshape nor the ABC
  freeform set: 92.2 vs 92.3 and 79.8 vs 79.8 per part. It had typed the full Star's twisted faces as
  planes, because each is one pure group already, kept apart by creases.

Analytic faces found once on the right surface type, merge off → on:

| set | per part | face-weighted | parts merged | analytic faces lost |
|---|---|---|---|---|
| Onshape (300) | 93.0 → 92.2 | 84.8 → 84.7 | 7 | 10 |
| ABC freeform (150) | 81.5 → 79.8 | 76.1 → 76.0 | 33 | 11 |
| fill_eval, native | 82.8 → 82.7 | | | 2 |
| fill_eval, NURBS | 80.2 → 80.2 | | | 1 |
| fill_eval, gmsh | 84.3 → 84.3 | | | 0 |
| fill_eval, game | 65.5 → 65.2 | | | 8 |

Freeform faces kept as one group, by area (`test/freeformSeg.mjs`):

| set | merge off | merge on |
|---|---|---|
| user's twisty parts | 19% | 85% |
| fill native | 3% | 56% |
| ABC freeform set | 21% | 28% |

On the twisty parts, analytic faces match the unmerged run (77.8%).

**Solids, ABC freeform set (150 parts, auto groups, JS knitter, 300 s deadline):**

| groups | recreated | every face analytic | hybrid | faceted | errors |
|---|---|---|---|---|---|
| unmerged (etm) | 118 | 44 | 16 | 56 | 23 |
| merged, lone groups keep their primitive | 124 | 51 | 17 | 53 | 17 |
| merged, final defaults (every inexact group freeform) | 123 | **55** | 19 | 46 | 18 |

With true segments the Python knitter recreated 102 of 150 with freeform surfaces, so segmentation is
still most of the gap.

**The user's parts** (`test/studioExport.mjs`, the studio's own path; "before" is the 2026-10-02 export
table above; volume against the Onshape STEP):

| part | triangles | before | now |
|---|---|---|---|
| Star | 3,330 | analytic, 27 primitive groups, tip spikes | 20 B-spline + 2 small faces, valid, 0.11% |
| Star (1) | 1,688 | 94% facets | 20 B-spline + 7, valid, 0.66% |
| Star (2) | 1,190 | 51% facets | planar facets |
| Fencepost (1) | 11,606 | not measured | 6 B-spline + 11, valid, 0.25% |
| Fencepost (2) | 3,022 | 55% facets | 10 B-spline + 9, valid, 0.14% |
| Cake Pan (2) | 22,898 | not measured | all facets |
| Christmas Ornament 2 (2) | 15,900 | all facets | all facets |

The browser gives the same result: in headless Chrome through `studio.html`, Star auto-paints to "22 new
groups (20 freeform)" and exports a valid solid 0.107% off the CAD.

**A manual freeform fill.** The seventh surface button (key 7) fills without a primitive. It floods the
unpainted triangles to 30° creases and makes an untyped group, which the export fits as a B-spline. On the
user's parts, one click per true freeform face (seeded at the face's most interior triangle):

| part | faces filled whole by one click |
|---|---|
| Star, all three resolutions | 20/20, no spill |
| Cake Pan, coarse and medium | 28/28 |
| Ornament, coarse | 8/14; 14/14 once its neighbours are painted |
| Fencepost | one click takes 3 faces, joined tangentially, as the merge joins them |

- **Fixed crease.** The crease is not widened on "low-poly" meshes, as the primitive fills' is. The Star
  reads as low-poly (step angle 50°) because its faces are flat to 1° inside, so the step median is taken
  over its boundaries; at 80° one click flooded 10 of its 20 faces.
- **No curvature wall.** A curvature-jump wall cut every face short: 29-84% of each Star face covered.

In headless Chrome: three freeform clicks on Star, then auto-paint for the rest, exports a closed solid
with every face on its fitted surface.

**Still open (freeform).**
- **The Ornament.** Its narrow faces are curved bands two or three triangles across. Inside one face the
  median dihedral is 36° and the 95th percentile 82°, so 30° creases cut every band into lone pieces.
- **Cake Pan (2).** 66 groups knit with 13 invalid faces; with fewer groups (63) it had been 19% facets.
  The knit is fragile on groups this small.
- **Windows sleep.** The knitter's deadline is wall-clock: an export interrupted by sleep skips its facet
  tiers and comes out fully faceted.

**Tried and left off:**
- **The learned edge wall** builds walls inside twisting faces.
- **A union-fit test** cannot tell same-face from different-face pairs.
- **Island joins.**
- **A relative crease** had no effect.
- **A crease share.** At 0.8 it trims the Ornament's groups (medium 116 → 82) without changing any score.

## Shared axes in the builder (2026-10-03)

**Why.** Every patch's surface is fitted on its own. Two halves of one cylinder, a shaft's steps, or a
fillet and the cylinder it rounds come out with axes 0.0003° (clean CAD meshes) to 0.1° and more (coarser
meshes, impure groups) apart. Every edge between them is then a visible kink, and an end face a hair off
square. CAD has one axis there.

**What.** `js/knit/axes.js` `shareAxes` runs in `knitSolid` after the freeform fits and before the build
(on by default; `test/knit.mjs` and `test/studioExport.mjs` take `--no-axes`). It has four stages.

1. **Directions.** Cylinder, cone and torus axes and plane normals pool into one exact direction when
   within the pair's uncertainty:
   - the tolerance is 10× (σ₁ + σ₂), capped at 1°;
   - a plane's σ is its RMS over its extent, and an axis's is 100× that (a strip's axis is worse
     conditioned than its RMS);
   - clustering is greedy, most precise first, against the running mean, so it does not chain;
   - a direction within its tolerance of perpendicular to a more precise one is made exactly
     perpendicular to it (to two: their cross product).
2. **Axis lines.** Within a direction, axes whose lines pass within 1e-3 of the diagonal of each other
   are one line. Lines are compared at the point nearest each patch: a near-cylindrical cone's apex can sit
   thousands of units out. Sphere centres that close to a line move onto it.
3. **Refit about the fixed line, robustly.** The refit is the median radius, a trimmed profile line for
   cones, and a trimmed Gauss-Newton profile circle for tori and spheres. Cylinders on one line with radii
   within their precision become one radius, and planes of one direction with offsets within theirs become
   one plane. Everything uses each group's inlier vertices (5× the median distance to its own surface).
4. **The check.** A changed surface is kept only if its inlier RMS stays within 1.5× its own fit's plus
   1e-6 of the diagonal. A member of a shared line that fails as its own type may pass as another profile
   about any of the part's lines (cylinder, cone, torus, simplest first). Otherwise it keeps its own fit,
   and the shared values are recomputed without it.

**What did not work, in order:**
- **Means instead of medians in the refits.** A 3-triangle sliver across two faces kept its exact plane
  under the robust fit, but refitted through all its vertices it moved 3e-5 of the diagonal off its true
  face. A lower residual, the wrong surface.
- **Weighting members by vertex count.** Loose fits dragged the shared axis, so the exact members failed
  the check. Weighting by precision (vertices over squared RMS) fixed it.
- **A fixed 1° direction tolerance for planes.** It averaged real draft angles and small steps: 18 fill
  planes went from exact to 2e-5–3e-4 of the diagonal off their true faces.
- **Moving loose members onto any line their vertices fit 2× better.** It moved a fillet torus onto a
  cylinder 0.175 of the diagonal away. Removed.

**Scored against the true surfaces** (`test/axisScore.mjs`; adjacent group pairs whose true faces stand in
the relation exactly; "exact" means within 1e-9 rad and 1e-9 of the diagonal):

| relation | Onshape (300 parts): exact before → after | fill_eval, 4 meshers (120 parts) |
|---|---|---|
| coaxial | 0% → 87.8% (303 pairs) | 1.5% → 98.9% (710) |
| end face (axis along a plane normal) | 18.2% → 99.2% | 27.6% → 99.3% |
| along (axis across a plane normal) | 15.7% → 94.8% | 29.9% → 96.8% |
| perpendicular planes | 81.5% → 99.6% | 93.9% → 98.9% |
| parallel planes | 100% → 100% | 88.7% → 98.1% |

- **Recovery of each group's true face** (found4) is unchanged on both sets. No group got worse (2× and
  +1e-5 of the diagonal); 31 and 8 got better, mostly cones on cylinder strips retyped to cylinders.
- **False shares** are pairs made exact whose true faces are not in that relation. They are either
  rounding in the truth (planes 0.0001–0.0003° off square) or groups typed wrong already.
- **What is left on Onshape** is loose fits: axes 1-2° off, or lines 0.003-0.03 of the diagonal off.
  Moving those is what produced false shares.

**Solids** (`test/knitRebuild.mjs`, the same patches, `--no-axes` on the same code):

| set | recreated | every face analytic | tier changes | knit time |
|---|---|---|---|---|
| Onshape 275 | 269 → 269 | 254 → 256 | 4 better, 2 worse | −19% |
| fill_eval 45 | 39 → 39 | 17 → 18 | 3 better, 1 worse | −10% |

- **The worse fill part**, 00021038_1 (gmsh), is a revolved part. All 23 surfaces sit on one axis and fit
  their vertices as before. One loose cone meets its neighbour at a grazing angle; its face grew 40%
  instead of 24%, and the volume missed the 1% bar. That is the knitter's near-tangency fragility, not a
  wrong surface.
- **Kinks in exported STEPs** (axial faces within 1° and 1e-3 of the diagonal), from the user's CAD files
  tessellated and run through `test/studioExport.mjs`:
  - shaft: 45 kinked pairs → 45 exact;
  - knob: 7 → 7 exact, and its 20 plane pairs off square → 0;
  - volume vs the CAD unchanged (shaft 0.40%, knob 0.001%).
- **Cost.** The axis step takes 20-100 ms per part, 1.3 s on a 311-group pulley. That pulley's single
  build took 202 s against 79 s without (it exports as facets either way).
- **Python.** `ml/solid_rebuild.py` does not have it yet.

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
