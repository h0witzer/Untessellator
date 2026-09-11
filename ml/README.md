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

Chunks and per-format manifest files are at <https://deep-geometry.github.io/abc-dataset/>.
Each chunk holds 10,000 models; roughly 750k of ABC's 1M models have feature files
(the format was added after the initial release). Download a few chunks with the
manifest and `curl`/`wget`, then extract — they ship as `.7z`:

```bash
7z x abc_0000_feat_v00.7z -o./abc/feat
```

Start with two or three chunks. The histogram stabilises quickly and you can always
index more later — the index resumes rather than rescanning.

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

## How the scanner works, and why

**It does not parse YAML.** ABC feature files carry full control nets, knot vectors and
per-vertex `(u,v)` parameters, so they run to megabytes each. A conforming parse
materialises all of it to count nine strings. `stage0_index.py` reads each file once,
line by line, in constant memory, looking only at `type:` and `sharp:` lines.

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

**Nested `type:` fields are excluded.** A `Revolution` patch has a profile curve with
its own `type:`. Counts are bucketed by (has-dash, indent) and reduced to the
shallowest dashed level per section, so a child object cannot be mistaken for a patch.

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
