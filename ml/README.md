# ml/ — learned segmentation experiment

Scoping and rationale: [`docs/ml-experiment-scope.md`](../docs/ml-experiment-scope.md).

Nothing here is wired into the browser tool, and by design nothing ever will be in the
critical path — a trained model may only propose a segmentation
(`{facetIds, primitiveType}`) behind the same interface crease fill and `paintFrom`
already use in [`js/regionGrow.js`](../js/regionGrow.js). If the model is absent, the
converter works exactly as it does today.

Stage 0 is standard library only. No `pip install`, no virtualenv, nothing to set up —
it has to run before any ML toolchain exists. Python 3.10+ (tested on 3.10.11).

---

## Stage 0 — index the corpus

**The question it answers:** what is the actual class distribution of surface types in
ABC, and which models are worth the cost of tessellating in Stage 1?

This matters because the ABC paper publishes the patch-type distribution only as a
figure with no numbers, and because the expected shape — planes and cylinders
dominating, `Torus`/`Revolution`/`Extrusion`/`BSpline` rare — is exactly the setup
where a model scores well on mean IoU while never predicting the classes the
experiment exists to cover. Tori in mechanical parts are usually fillets: numerous,
but tiny in facet count.

### 1. Get the feature files

Stage 0 needs **only the `feat` chunks** — not STEP, not OBJ, not Parasolid. That is
the entire reason it comes first: it is the cheapest possible way to decide what to
download and tessellate later.

```bash
python fetch_abc.py --format feat --chunks 0-2 --out D:/abc-data/archives
```

`fetch_abc.py` reads ABC's per-format manifests and verifies every archive against
the published `size.yml` (`--verify md5` for the stronger check). Downloads resume,
so interrupting is safe. `--plan-only` prints the budget without downloading.

**Measured sizes**, per chunk of 10,000 models, compressed:

| format | per chunk | all 100 | needed? |
|---|---|---|---|
| `stl2` | 6.59 GB | 659 GB | no |
| `obj`  | 6.36 GB | 636 GB | no |
| `feat` | 3.62 GB | 362 GB | **Stage 0** |
| `para` | 1.97 GB | 198 GB | no |
| `step` | 1.07 GB | 107 GB | **Stage 1** |
| `stat`, `meta` | <2 MB | 0.3 GB | no |

Three chunks of `feat` + `step` is 16.4 GB of download for 30,000 models — enough to
fill the default 25k Stage 1 target. Feature files decompress roughly 3.7×, so budget
~40 GB extracted for three chunks. The histogram stabilises quickly and the index
resumes rather than rescanning, so start small and add chunks if a rare class comes
up short.

Note that the two largest formats are the two we deliberately do not want: we
re-tessellate from STEP so mesh density is a controlled variable rather than a fixed
property of the download.

Extract with 7-Zip:

```bash
7z x D:/abc-data/archives/abc_0000_feat_v00.7z -oD:/abc-data/feat
```

> **Licensing:** copyright stays with the model creators; use is governed by Onshape
> Terms of Use 1.g.ii. Worth reading before any trained weights are published.

### 2. Index

```bash
python stage0_index.py --root ./abc/feat --db abc_index.sqlite
```

Useful flags: `--jobs N` (defaults to cores−1), `--limit N` for a trial run,
`--force` to rescan. Re-running skips files already indexed, so interrupting is safe.

### 3. Report

```bash
python stage0_report.py --db abc_index.sqlite \
    --target 25000 --min-per-type 2000 \
    --emit-manifest stage1_manifest.jsonl
```

Prints the measured distribution, corpus hygiene, rare-class exposure, and a
stratified selection; `--emit-manifest` writes the chosen models as JSON Lines for
Stage 1 to consume.

### 4. Tests

```bash
python test_stage0.py
```

---

## The showcase set — held out, never trained on

```bash
python fetch_showcase.py --out D:/abc-data/showcase
```

Twelve assets that answer a question the ABC validation split cannot: what does a
model trained entirely on parametric CAD do when handed geometry that never touched a
parametric modeller — and is the output good enough to import into one? They are also
the models people recognise on sight.

They split into three categories because they test different things:

**A — CAD-origin, distributed mesh-only.** The actual product use case: modelled
parametrically, thrown onto the internet as an STL with the STEP left behind.
`3DBenchy.stl` (225,706 triangles, verified 48.0 mm tall against spec, so it is in real
millimetres and fitted radii can be checked against design intent). Its CAD source
went **public domain (CC0) in February 2025**, so unlike most of the internet's STL
slop this one can be *scored*, not just eyeballed.

**B — parametric but not CAD, with known ground truth.**

| asset | ground truth |
|---|---|
| `teapot_bezier{0,1,2}.stl` | 3,488 / 19,480 / 145,620 triangles off the **same 28 bicubic Bézier patches** (verified: 28 `surface(` records, 448 control points, 16 per patch) |
| `donut_clean_{coarse,medium,fine}.stl` | exactly **one torus**, R=1.0 r=0.4 — max implicit-torus residual 6.2e-08, i.e. float32 exact |
| `donut_lumpy_*.stl` | same torus displaced along the normal by a fixed sum of sinusoids, residual 5.9e-02 (~15% of r) |

The teapot ladder is a tessellation-density experiment on *fixed* parametric ground
truth, which is exactly what the scope doc wants and which we get for free. The clean
donut is the single cleanest test of whether `Torus` classification works at all — one
of the classes ParSeNet and HPNet do not support. The lumpy variant asks whether that
classification survives organic displacement or collapses to spline.

**C — never parametric.** `Suzanne.stl`, a hand-authored subdivision cage, and coarse
(~1k triangles) so it also probes thin per-face facet support. There is no "right
answer" in analytic terms here — and note that a subdivision surface is genuinely well
approximated by NURBS, so "it emitted spline patches everywhere" is a *correct*
result, not a failure. Judge it on whether patches follow the form rather than
shattering into noise, and on whether the STEP opens in a CAD kernel at all.

### Training exclusion is structural

`fetch_showcase.py` writes `holdout_sha256.txt`. **Stage 1 must refuse to ingest any
file whose SHA-256 appears there.** A note in a README is a convention; a hash check is
a guarantee. Upstream hashes are pinned too — a mismatch means the file changed and
earlier eval runs are no longer comparable, which is precisely when you want to be
told rather than silently drift.

### Licences

All clear for evaluation; two carry conditions worth knowing before anything is
published or demonstrated.

| asset | licence | note |
|---|---|---|
| 3DBenchy | **CC0 1.0** | Public domain since Feb 2025, superseding the original CC BY-ND. Derivatives and redistribution fine. |
| Utah teapot | free, commercial use permitted | Newell/Blinn via University of Utah |
| donuts | CC0 | generated by the script |
| Suzanne | **GPL-3.0-or-later** | Willem-Paul van Overbruggen. Fine for evaluation; check before redistributing a derived STEP. |

## Validated against real ABC data

ABC publishes one sample model on its site. The scanner was cross-checked against a
full PyYAML parse (`CSafeLoader`) of
`00000050_80d90bfdd2e74e709956122a_features_000.yml`, 1.66 MB for a single model:

| | PyYAML | line scanner |
|---|---|---|
| time | 751 ms | **24 ms** (31× faster) |
| curves | `BSpline 7, Circle 47, Line 52` | identical |
| surfaces | `Cylinder 23, Plane 20` | identical |
| sharp curves | 77 | 77 |

Exact match on every count. That one file also demonstrates the hazard the design
exists to avoid: it contains **7 B-spline curves and zero B-spline surfaces**, so a
naive global `type:` count would report 7 spline surfaces that are not there — an
error in the worst possible direction, inflating one of the rare classes the
experiment is built to measure.

Extrapolated to the ~750k models with feature files, the parser choice is the
difference between roughly 5 hours and 6 days of single-threaded scanning.

## How the scanner works, and why

**It does not parse YAML.** ABC feature files carry full control nets, knot vectors and
per-vertex `(u,v)` parameters, so they run to megabytes each — the sample model above
is 1.66 MB on its own. A conforming parse materialises all of it to count nine
strings. `stage0_index.py` reads each file once, line by line, in constant memory,
looking only at `type:` and `sharp:` lines.

**Section roles are detected, not hardcoded.** The ABC supplementary documents both
taxonomies but not the YAML key names, and the two vocabularies overlap:

```
surfaces: Plane Cylinder Cone Sphere Torus Revolution Extrusion BSpline Other
curves:   Line Circle Ellipse BSpline Other
```

`BSpline` and `Other` are valid in **both**. Counting `type:` globally would silently
merge B-spline surfaces into B-spline curves — and B-spline surfaces are one of the
classes that motivated this whole direction. So the scanner tracks which top-level
section each `type:` sits in and infers each section's role from the unambiguous types
it contains. The detected names are written to `scan_meta` so the inference is
auditable rather than assumed, and the report prints them.

**Entry level is the effective column, not the dash.** ABC emits keys in
**alphabetical** order, so the dash usually carries `location:` and `type:` arrives
undashed a few lines later:

```yaml
curves:
- location: [0.0, -3.175, 0.0]
  radius: 17.5145949
  sharp: true
  type: Circle          # indent 2, no dash
```

An entry whose other keys all sort after `"type"` puts it on the dash instead. Both
are the same entry level, so counts are bucketed by the column the key starts at
(counting the dash as indentation) and reduced to the shallowest column per section.
A nested `type:` — a `Revolution` patch's profile curve, say — is strictly deeper and
drops out. An earlier version preferred dashed lines outright, which on real data
would have discarded every undashed entry in a section the moment one entry happened
to sort `type` first.

**Parts are keyed separately from models.** One ABC model can contain several solids,
each with its own `..._features_001.yml`, `..._002.yml`. The primary key is
`(model_id, part)`; keying on the 8-digit id alone would silently drop all but one.

Each of those four is a way the histogram could come out *silently wrong* rather than
loudly broken, so each has a test in `test_stage0.py`.

---

## Before trusting the numbers

1. **Check the detected section names** in the report header. If they are not
   `curves -> curves` and `surfaces -> surfaces`, the corpus differs from what the
   paper describes and everything downstream needs re-reading.
2. **Check `unknown_types` is empty.** Any type name outside the documented taxonomy
   means format drift, and the report says so loudly.
3. **Check `ambiguous_sections` is near zero.** It means files whose sections could not
   be resolved even with the learned role map.
4. **Sanity-check one file by hand** against the histogram before committing to a
   25k-model tessellation run.

## Known limits

- **`struct_hash` is structural, not geometric.** It fingerprints patch and curve
  histograms only, so it catches exact structural twins and under-counts near
  duplicates. Real geometric dedup belongs in Stage 1, where the meshes exist.
- **Hygiene filters are coarse.** `single_primitive`, `no_surfaces` and patch-count
  thresholds catch what the ABC paper admits to ("broken boundaries, self-intersecting
  faces or edges, duplicate vertices... models that are just made of single primitives
  such as a plane"), but a feature file cannot reveal a self-intersecting mesh. That
  check needs the geometry.
- **The `stat` YAML files are not read.** Their schema was not confirmed from the
  paper, and guessing field names would produce confidently wrong quality flags. The
  feature file already supplies patch count, curve count and sharp-curve count.

## Next

Stage 1 reads `stage1_manifest.jsonl`, pulls the matching STEP files, and tessellates
them with OCCT at several densities — capturing the triangle → face mapping from
`BRepMesh` and cross-checking it against the feature file's `face_indices`. Note that
**OBJ is 1-indexed while the YAML is 0-indexed**; the ABC supplementary calls this out
explicitly and it is an easy off-by-one to ship.
