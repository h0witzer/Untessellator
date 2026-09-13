#!/usr/bin/env python3
"""
stage1_bake.py - tessellate the selected models and bake labelled training samples.

Reads Stage 0's manifest, finds each model's STEP, meshes it at every density in the
ladder, and writes one .npz per model holding raw triangles plus per-triangle labels.

WHAT IS STORED, AND WHY SO LITTLE

    positions  (ntri, 9) float32  raw triangle vertices, STL-style, NOT welded
    face_id    (ntri,)   int32    which B-rep face each triangle came from
    face_type  (ntri,)   uint8    that face's surface type, index into TYPE_NAMES

Centroids, normals, areas and adjacency are NOT stored. They are derived at training
time by `meshcore.py`, which is a checked port of `js/meshCore.js` - see
`test_meshcore.py`, which runs both implementations over the same meshes and compares
facets and full adjacency. Deriving rather than storing means there is exactly one
definition of a facet and one weld tolerance in the system, so training data cannot
drift from what the converter computes at inference. It also roughly halves the bake.

HOLDOUT ENFORCEMENT IS A HASH CHECK, NOT A CONVENTION

If a showcase holdout file is present, every STEP is hashed and skipped if it appears
in it. The showcase set is meant to stay unseen; a note in a README would not survive
someone pointing this at the wrong directory.

Usage:
    python stage1_bake.py --manifest D:/abc-data/stage1_manifest.jsonl \
        --step-root D:/abc-data/step --out D:/abc-data/bake --jobs 8
    python stage1_bake.py ... --limit 200 --verify      # cross-check against feat files

Needs the OCCT venv: D:/abc-data/venv/Scripts/python.exe
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from occ_tessellate import (  # noqa: E402
    DENSITY_LADDER, TYPE_NAMES, bbox_diagonal, face_type_histogram, read_step, tessellate,
)

MIN_TRIS = 32          # below this a mesh cannot support a useful neighbourhood
MAX_TRIS = 400_000     # guards against pathological models eating a worker


def load_holdout(path: Path | None) -> set[str]:
    if not path or not path.exists():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.add(line.split()[0].lower())
    return out


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def find_step(step_root: Path, model_id: str) -> Path | None:
    """ABC lays out one directory per model id, with the part index in the filename."""
    d = step_root / model_id
    if not d.is_dir():
        return None
    hits = sorted(d.glob("*.step")) or sorted(d.glob("*.stp"))
    return hits[0] if hits else None


def bake_one(task: tuple) -> dict:
    model_id, part, step_path, out_path, do_verify, feat_surfaces, holdout = task
    rec = {"model_id": model_id, "part": part, "status": "ok", "densities": {},
           "note": "", "mismatch": None}
    try:
        sp = Path(step_path)
        if holdout and sha256_of(sp) in holdout:
            rec["status"] = "holdout"
            return rec

        shape = read_step(str(sp))
        if shape is None:
            rec["status"] = "unreadable"
            return rec
        diag = bbox_diagonal(shape)
        if diag <= 0:
            rec["status"] = "degenerate_bbox"
            return rec

        if do_verify and feat_surfaces:
            occ = face_type_histogram(shape)
            want = {k: v for k, v in feat_surfaces.items() if v}
            got = {k: v for k, v in occ.items() if v}
            if got != want:
                rec["mismatch"] = {"feat": want, "occ": got}

        arrays: dict[str, np.ndarray] = {}
        for name, lin, ang in DENSITY_LADDER:
            t = tessellate(shape, lin, ang, diag)
            n = len(t)
            if n < MIN_TRIS or n > MAX_TRIS:
                rec["densities"][name] = n
                continue
            arrays[f"{name}_positions"] = t.positions
            arrays[f"{name}_face_id"] = t.face_id
            arrays[f"{name}_face_type"] = t.face_type
            rec["densities"][name] = n
            rec["n_faces"] = t.n_faces

        if not arrays:
            rec["status"] = "no_usable_density"
            return rec

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, model_id=model_id, part=part, diag=np.float64(diag), **arrays)
    except Exception as exc:                      # one bad model must not kill the run
        rec["status"] = "error"
        rec["note"] = f"{type(exc).__name__}: {exc}"[:200]
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--step-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--holdout", type=Path, default=Path("D:/abc-data/showcase/holdout_sha256.txt"))
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verify", action="store_true",
                    help="cross-check OCCT surface types against the feat histogram")
    ap.add_argument("--force", action="store_true", help="re-bake models already present")
    args = ap.parse_args(argv)

    holdout = load_holdout(args.holdout)
    print(f"holdout hashes loaded: {len(holdout)}"
          + ("" if holdout else "  (WARNING: none - showcase files will NOT be excluded)"))

    entries = []
    with args.manifest.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    print(f"manifest: {len(entries):,} entries")

    tasks, missing, skipped = [], 0, 0
    for e in entries:
        mid = e["model_id"]
        out_path = args.out / mid[:5] / f"{mid}_{e['part']}.npz"
        if out_path.exists() and not args.force:
            skipped += 1
            continue
        sp = find_step(args.step_root, mid)
        if sp is None:
            missing += 1
            continue
        tasks.append((mid, e["part"], str(sp), str(out_path), args.verify,
                      e.get("surfaces", {}), holdout))
        if args.limit and len(tasks) >= args.limit:
            break

    print(f"to bake: {len(tasks):,}   already done: {skipped:,}   "
          f"no STEP on disk: {missing:,}")
    if not tasks:
        print("nothing to do")
        return 0

    t0 = time.time()
    stats: dict[str, int] = {}
    tri_totals: dict[str, int] = {}
    mismatches = 0
    done = 0

    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for rec in pool.map(bake_one, tasks, chunksize=4):
            done += 1
            stats[rec["status"]] = stats.get(rec["status"], 0) + 1
            for k, v in rec.get("densities", {}).items():
                tri_totals[k] = tri_totals.get(k, 0) + v
            if rec.get("mismatch"):
                mismatches += 1
                if mismatches <= 5:
                    print(f"  TYPE MISMATCH {rec['model_id']}: "
                          f"feat={rec['mismatch']['feat']} occ={rec['mismatch']['occ']}")
            if rec["status"] in ("error",) and stats[rec["status"]] <= 5:
                print(f"  error {rec['model_id']}: {rec['note']}")
            if done % 250 == 0:
                rate = done / max(1e-6, time.time() - t0)
                eta = (len(tasks) - done) / max(rate, 1e-6) / 60
                print(f"  {done:,}/{len(tasks):,}  {rate:.1f}/s  eta {eta:.0f} min", flush=True)

    elapsed = time.time() - t0
    print(f"\nbaked {done:,} in {elapsed / 60:.1f} min ({done / max(elapsed, 1e-6):.1f}/s)")
    print("status: " + ", ".join(f"{k}={v:,}" for k, v in sorted(stats.items())))
    print("triangles: " + ", ".join(f"{k}={v:,}" for k, v in tri_totals.items()))
    if args.verify:
        ok = done - mismatches
        print(f"type cross-check vs feat files: {ok:,} agree, {mismatches:,} differ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
