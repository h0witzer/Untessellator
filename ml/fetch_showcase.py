#!/usr/bin/env python3
"""
fetch_showcase.py - build the held-out showcase set.

These meshes are NEVER trained on. They exist to answer a question the ABC validation
split cannot: what does a model trained entirely on parametric CAD do when handed
geometry that never touched a parametric modeller - and is the result good enough to
import into one?

They also happen to be the models people recognise on sight, which matters if this
ever gets demonstrated to anyone.


THREE CATEGORIES, BECAUSE THEY TEST DIFFERENT THINGS

  A. CAD-origin, distributed as mesh only.
     The actual product use case: something was modelled parametrically, then thrown
     onto the internet as an STL with the STEP left behind. Success means recovering
     analytic faces. 3DBenchy is the ideal specimen - and since its CAD source went
     public domain in 2025, it can be SCORED, not just eyeballed.

  B. Parametric, but not CAD.
     Known mathematical ground truth from a non-CAD lineage. The Utah teapot is 28
     bicubic Bezier patches (verified: 28 `surface(` records, 448 `pt(` control
     points, 16 per patch); a correct model should segment it into roughly 28 spline
     patches. The generated tori are exactly one Torus of known major/minor radius -
     the cleanest available test of whether Torus classification works at all, and a
     class ParSeNet and HPNet do not support. They are PROBES, not donuts; nobody is
     meant to recognise them.

  C. Never parametric.
     The Blender Guru donut - the actual one, subdivision cage plus displacement plus
     a particle system scattering several hundred sprinkles - and Suzanne, a
     hand-authored subdivision cage. There is no "right answer" in analytic terms.
     The interesting question is not accuracy but whether the segmentation is
     COHERENT - patches that follow the form rather than shattering into noise - and
     whether the resulting B-rep will open in a CAD kernel at all.

Category C is the stress test, but note that a good result there is not necessarily
an analytic one: a subdivision surface is genuinely well approximated by NURBS, so
"it emitted spline patches everywhere" is a correct answer, not a failure. Judge it
on patch coherence and whether the STEP imports cleanly.


TRAINING EXCLUSION IS STRUCTURAL, NOT A CONVENTION

Every asset's SHA-256 is written to `holdout_sha256.txt`. Stage 1 must refuse to
ingest any file whose hash appears there. A note in a README is not a guarantee; a
hash check is.

Usage:
    python fetch_showcase.py --out D:/abc-data/showcase
    python fetch_showcase.py --out D:/abc-data/showcase --verify
"""

from __future__ import annotations

import argparse
import hashlib
import math
import struct
import subprocess
import sys
from pathlib import Path

# ----------------------------------------------------------------------------------
# Remote assets. Hashes are pinned from the versions retrieved 2026-09-11; a mismatch
# means upstream changed the file and the eval set is no longer comparable to earlier
# runs, which is exactly when you want to be told.
# ----------------------------------------------------------------------------------

REMOTE = [
    {
        "name": "3DBenchy.stl",
        "category": "A",
        "url": "https://raw.githubusercontent.com/CreativeTools/3DBenchy/master/Single-part/3DBenchy.stl",
        "sha256": "6ab57f1c3f8e86bc3cbd302c6fa6270acf06277c6335454e922419c25d42e97e",
        "bytes": 11285384,
        "license": "CC0 1.0 Universal (public domain)",
        "attribution": "Creative Tools / NTI Group",
        "note": "Released to the public domain Feb 2025, superseding the original "
                "CC BY-ND. Derivative works and redistribution are permitted.",
        "tests": "CAD-origin geometry distributed mesh-only: the product use case. "
                 "Hull is swept/lofted, chimney is a true cylinder, windows are "
                 "circles - a mix of analytic and freeform in one part.",
    },
    {
        "name": "Suzanne.stl",
        "category": "C",
        "url": "https://upload.wikimedia.org/wikipedia/commons/e/e3/Suzanne.stl",
        "sha256": "bac920b489a6840588c2e14de8bf3301881b81b20e8d76dfc313859864e546c3",
        "bytes": 177369,
        "license": "GPL-3.0-or-later",
        "attribution": "Willem-Paul van Overbruggen (Blender Foundation mascot)",
        "note": "GPL. Fine for evaluation and demonstration; check the licence before "
                "redistributing any derived STEP.",
        "tests": "Hand-authored subdivision cage. Coarse (~1k triangles), so it also "
                 "probes behaviour when facet support per face is thin.",
    },
    {
        "name": "teapot_bezier",
        "category": "B",
        "url": "https://users.cs.utah.edu/~dejohnso/models/teapot_bezier",
        "sha256": "22d59967e342b8e6f2b4e78e8db50a44f5aa370661f069b478ae8d0d76d9395e",
        "bytes": 24103,
        "license": "Freely available, commercial use permitted",
        "attribution": "Martin Newell / Jim Blinn; compilation by Thomas V Thompson II, University of Utah",
        "note": "GROUND TRUTH, not a mesh: 28 bicubic Bezier patches, 448 control points.",
        "tests": "The reference the .tris ladder below is scored against.",
    },
]

# The teapot ships pre-triangulated at three densities from the SAME Bezier source,
# which is a ready-made tessellation-density ladder on fixed ground truth - the exact
# experiment docs/ml-experiment-scope.md wants, without having to build it.
TEAPOT_TRIS = [
    ("teapot_bezier0.tris", 3488, "a4d9f07cadf458f262d102b43ff02ce0a0f11ac145abd676b7de138d18f3615c", 295069),
    ("teapot_bezier1.tris", 19480, "d739a27546f5f6a2f728e950e0c7e231297e61f3bbb9c6bc869ea35edfa8b57e", 1649410),
    ("teapot_bezier2.tris", 145620, "3666201587474c53fa60eb77460ce93318977d395d579db03c16990598959768", 12334759),
]
TEAPOT_BASE = "https://users.cs.utah.edu/~dejohnso/models"

# The REAL Blender Guru donut - subdivision cage, displaced, with a particle system
# scattering six sprinkle meshes over the icing. Needs Blender to convert, since the
# sprinkles only exist as depsgraph instances (see blend_to_stl.py).
BLEND_DONUT = {
    "name": "4_5.blend",
    "url": "https://raw.githubusercontent.com/adcimon/blender-donut/master/4_5.blend",
    "sha256": "785b4bba88343adf53bccdb29e90d66f9fcd41cbdbdba158a3b4ea58c2c7f975",
    "license": "MIT",
    "attribution": "Adrian Ciborro Montes (adcimon/blender-donut), (c) 2021",
    "note": "Final state of the Blender Guru beginner tutorial. Converted locally by "
            "blend_to_stl.py; the .blend itself is not an eval asset.",
}

# Exact analytic tori. These are NOT donuts and are not meant to look like one - they
# are the ground-truth probe for Torus classification, which is a class ParSeNet and
# HPNet do not support at all. Values are arbitrary but FIXED; the fitted radii are
# compared against them.
TORUS_R = 1.0      # major radius
TORUS_r = 0.4      # minor radius
TORUS_LODS = [("coarse", 48, 24), ("medium", 96, 48), ("fine", 192, 96)]


# ----------------------------------------------------------------------------------
# Mesh writing
# ----------------------------------------------------------------------------------

def write_binary_stl(path: Path, tris: list[tuple]) -> None:
    """tris: [((nx,ny,nz), (x,y,z), (x,y,z), (x,y,z)), ...]"""
    with path.open("wb") as fh:
        fh.write(b"generated by fetch_showcase.py - held-out eval asset".ljust(80, b"\0"))
        fh.write(struct.pack("<I", len(tris)))
        for n, a, b, c in tris:
            fh.write(struct.pack("<12fH", *n, *a, *b, *c, 0))


def torus_mesh(R: float, r: float, nu: int, nv: int, lumpy: float = 0.0) -> list[tuple]:
    """Triangulated torus. `lumpy` displaces along the normal by a fixed sum of
    sinusoids - deterministic, so the lumpy donut is reproducible rather than a
    different mesh on every run."""
    def point(i: int, j: int):
        u = 2.0 * math.pi * i / nu
        v = 2.0 * math.pi * j / nv
        cu, su = math.cos(u), math.sin(u)
        cv, sv = math.cos(v), math.sin(v)
        nx, ny, nz = cv * cu, cv * su, sv
        rr = r
        if lumpy:
            rr += lumpy * (math.sin(3 * u) * math.cos(5 * v)
                           + 0.5 * math.sin(7 * u + 1.3) * math.cos(3 * v + 0.7)
                           + 0.25 * math.sin(11 * u + 2.1) * math.cos(7 * v + 1.9))
        return ((R + rr * cv) * cu, (R + rr * cv) * su, rr * sv), (nx, ny, nz)

    grid = [[point(i, j) for j in range(nv + 1)] for i in range(nu + 1)]
    tris = []
    for i in range(nu):
        for j in range(nv):
            p00, n00 = grid[i][j]
            p10, _ = grid[i + 1][j]
            p11, _ = grid[i + 1][j + 1]
            p01, _ = grid[i][j + 1]
            tris.append((n00, p00, p10, p11))
            tris.append((n00, p00, p11, p01))
    return tris


def tris_to_stl(src: Path, dst: Path) -> int:
    """Convert Utah's .tris format (count, then blank-separated vertex triples)."""
    nums: list[float] = []
    with src.open("r", encoding="utf-8", errors="replace") as fh:
        first = fh.readline().strip()
        declared = int(first) if first.isdigit() else 0
        for line in fh:
            parts = line.split()
            if len(parts) == 3:
                nums.extend(float(p) for p in parts)
    tris = []
    for k in range(0, len(nums) - 8, 9):
        a = (nums[k], nums[k + 1], nums[k + 2])
        b = (nums[k + 3], nums[k + 4], nums[k + 5])
        c = (nums[k + 6], nums[k + 7], nums[k + 8])
        ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
        nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
        m = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
        tris.append(((nx / m, ny / m, nz / m), a, b, c))
    if declared and len(tris) != declared:
        print(f"  note: {src.name} declared {declared} triangles, parsed {len(tris)}")
    write_binary_stl(dst, tris)
    return len(tris)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, dest: Path) -> bool:
    rc = subprocess.run(["curl", "-sL", "--retry", "3", "-o", str(dest), url]).returncode
    return rc == 0 and dest.exists() and dest.stat().st_size > 0


_BLENDER_OVERRIDE: str | None = None


def find_blender() -> str | None:
    """Locate a Blender executable. Optional: only the donut conversion needs it."""
    if _BLENDER_OVERRIDE:
        return _BLENDER_OVERRIDE
    from shutil import which
    found = which("blender")
    if found:
        return found
    roots = [
        Path("C:/Program Files/Blender Foundation"),
        Path("C:/Program Files (x86)/Steam/steamapps/common/Blender"),
        Path("/Applications/Blender.app/Contents/MacOS"),
        Path("/usr/share/blender"),
    ]
    candidates: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for exe in ("blender.exe", "Blender", "blender"):
            candidates.extend(root.glob(f"*/{exe}"))
            direct = root / exe
            if direct.exists():
                candidates.append(direct)
    # Newest install wins, so a machine with several versions uses the current one.
    candidates.sort(key=lambda p: str(p), reverse=True)
    return str(candidates[0]) if candidates else None


# ----------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("showcase"))
    ap.add_argument("--verify", action="store_true", help="re-hash existing files and stop on drift")
    ap.add_argument("--blender", help="path to the Blender executable (only the donut needs it)")
    args = ap.parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if args.blender:
        global _BLENDER_OVERRIDE
        _BLENDER_OVERRIDE = args.blender

    records: list[dict] = []
    failures = 0

    print(f"showcase set -> {out.resolve()}\n")

    # --- remote -------------------------------------------------------------------
    for item in REMOTE:
        dest = out / item["name"]
        if not dest.exists():
            print(f"fetching {item['name']} ...")
            if not fetch(item["url"], dest):
                print(f"  FAILED: {item['url']}", file=sys.stderr)
                failures += 1
                continue
        digest = sha256_of(dest)
        ok = digest == item["sha256"]
        status = "ok" if ok else "HASH MISMATCH"
        if not ok:
            print(f"  {item['name']}: {status}")
            print(f"    expected {item['sha256']}")
            print(f"    got      {digest}")
            print("    upstream changed - earlier eval runs are no longer comparable")
            failures += 1
        else:
            print(f"  {item['name']:24} {dest.stat().st_size:>10,} bytes  {status}")
        records.append({**item, "file": item["name"], "sha256_actual": digest})

    for name, count, sha, nbytes in TEAPOT_TRIS:
        dest = out / name
        if not dest.exists():
            print(f"fetching {name} ...")
            if not fetch(f"{TEAPOT_BASE}/{name}", dest):
                print(f"  FAILED: {name}", file=sys.stderr)
                failures += 1
                continue
        digest = sha256_of(dest)
        ok = digest == sha
        print(f"  {name:24} {dest.stat().st_size:>10,} bytes  {'ok' if ok else 'HASH MISMATCH'}")
        if not ok:
            failures += 1
        stl = out / f"{Path(name).stem}.stl"
        n = tris_to_stl(dest, stl)
        print(f"    -> {stl.name} ({n:,} triangles)")
        records.append({
            "name": stl.name, "category": "B", "file": stl.name,
            "license": "Freely available, commercial use permitted",
            "attribution": "Martin Newell / Jim Blinn, University of Utah",
            "note": f"Tessellation of the same 28 Bezier patches at {count:,} triangles.",
            "tests": "Density ladder on fixed parametric ground truth: does recovered "
                     "patch count stay near 28 as density changes?",
            "sha256_actual": sha256_of(stl),
        })

    # --- the real Blender donut ---------------------------------------------------
    print("\nBlender Guru donut ...")
    blend = out / BLEND_DONUT["name"]
    if not blend.exists():
        print(f"  fetching {BLEND_DONUT['name']} (16 MB) ...")
        fetch(BLEND_DONUT["url"], blend)
    if blend.exists():
        digest = sha256_of(blend)
        if digest != BLEND_DONUT["sha256"]:
            print(f"  HASH MISMATCH on {blend.name} - upstream changed", file=sys.stderr)
            failures += 1
        blender = find_blender()
        if not blender:
            print("  Blender not found - skipping donut conversion.")
            print("  Install Blender, or pass --blender <path to blender executable>.")
            print("  Everything else in the showcase set is unaffected.")
        else:
            full = out / "blender_donut.stl"
            plain = out / "blender_donut_plain.stl"
            print(f"  converting with {blender} ...")
            rc = subprocess.run(
                [blender, "-b", str(blend), "-P", str(Path(__file__).with_name("blend_to_stl.py")),
                 "--", str(full), str(plain)],
                capture_output=True, text=True,
            )
            for line in rc.stdout.splitlines():
                if line.startswith(("WROTE", "ERROR")):
                    print("   ", line)
            if full.exists() and plain.exists():
                records.append({
                    "name": full.name, "category": "C", "file": full.name,
                    "license": BLEND_DONUT["license"], "attribution": BLEND_DONUT["attribution"],
                    "note": "Donut + icing + sprinkles, scaled to ~90 mm across. Sprinkles "
                            "are particle instances realised at export; a naive export "
                            "silently produces a bald donut.",
                    "tests": "The recognisable one. Subdivision cage plus displacement plus "
                             "several hundred tiny disconnected sprinkle solids - tests "
                             "whether segmentation stays coherent when most connected "
                             "components are noise-sized.",
                    "sha256_actual": sha256_of(full),
                })
                records.append({
                    "name": plain.name, "category": "C", "file": plain.name,
                    "license": BLEND_DONUT["license"], "attribution": BLEND_DONUT["attribution"],
                    "note": "Donut + icing only, no sprinkles.",
                    "tests": "Same shape without the sprinkle confetti, so a poor result on "
                             "the full donut can be attributed to the sprinkles or not.",
                    "sha256_actual": sha256_of(plain),
                })
            else:
                print("  conversion produced no output", file=sys.stderr)
                failures += 1

    # --- generated analytic tori ---------------------------------------------------
    print(f"\ngenerating exact tori (ground truth R={TORUS_R} r={TORUS_r}) ...")
    print("  note: these are probes, not donuts - see MANIFEST.md")
    for label, nu, nv in TORUS_LODS:
        for lumpy, tag in ((0.0, "exact"), (0.035, "lumpy")):
            name = f"torus_{tag}_{label}.stl"
            dest = out / name
            tris = torus_mesh(TORUS_R, TORUS_r, nu, nv, lumpy)
            write_binary_stl(dest, tris)
            print(f"  {name:24} {len(tris):>10,} triangles")
            records.append({
                "name": name, "category": "B", "file": name,
                "license": "CC0 (generated by this script)",
                "attribution": "generated - fetch_showcase.py",
                "note": (f"Exact torus, R={TORUS_R}, r={TORUS_r}, {nu}x{nv} segments."
                         if not lumpy else
                         f"Same torus displaced along the normal by a fixed sum of "
                         f"sinusoids (amplitude {lumpy})."),
                "tests": ("Ground truth is exactly one Torus. Does the model predict "
                          "Torus, and does the fitted radius match?" if not lumpy else
                          "Does Torus classification survive displacement, or fall back "
                          "to spline?"),
                "sha256_actual": sha256_of(dest),
            })

    # --- holdout guard ------------------------------------------------------------
    guard = out / "holdout_sha256.txt"
    with guard.open("w", encoding="utf-8") as fh:
        fh.write("# Files that must NEVER enter training data.\n")
        fh.write("# Stage 1 must refuse to ingest any file whose SHA-256 appears here.\n")
        for r in sorted(records, key=lambda r: r["file"]):
            fh.write(f"{r['sha256_actual']}  {r['file']}\n")
    print(f"\nwrote {guard.name} ({len(records)} hashes)")

    # --- manifest -----------------------------------------------------------------
    man = out / "MANIFEST.md"
    cat_titles = {
        "A": "Category A - CAD-origin, distributed mesh-only (the product use case)",
        "B": "Category B - parametric but not CAD (known ground truth)",
        "C": "Category C - never parametric (stress test)",
    }
    with man.open("w", encoding="utf-8") as fh:
        fh.write("# Showcase set - held out, never trained on\n\n")
        fh.write("Generated by `ml/fetch_showcase.py`. Every hash here is also in\n")
        fh.write("`holdout_sha256.txt`, which Stage 1 must check against.\n")
        for cat in ("A", "B", "C"):
            rows = [r for r in records if r["category"] == cat]
            if not rows:
                continue
            fh.write(f"\n## {cat_titles[cat]}\n\n")
            for r in rows:
                fh.write(f"### `{r['file']}`\n\n")
                fh.write(f"- **Tests:** {r['tests']}\n")
                fh.write(f"- **Licence:** {r['license']}\n")
                fh.write(f"- **Attribution:** {r['attribution']}\n")
                if r.get("note"):
                    fh.write(f"- **Note:** {r['note']}\n")
                fh.write(f"- **SHA-256:** `{r['sha256_actual']}`\n\n")
    print(f"wrote {man.name}")

    if failures:
        print(f"\n{failures} problem(s) - see above", file=sys.stderr)
        return 1
    print(f"\n{len(records)} assets ready. None of these may appear in training data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
