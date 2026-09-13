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
