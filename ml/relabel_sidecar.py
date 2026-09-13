#!/usr/bin/env python3
"""
relabel_sidecar.py - per-face canonical surface types for baked models, without re-baking.

The bake stores face_type per triangle from the B-rep's stored surface type. canonical.py
decides what each face geometrically IS. Face ids in the bake are the order
TopExp_Explorer visits faces in occ_tessellate.tessellate, which is deterministic for a
given STEP, so a small sidecar mapping face id -> canonical type is enough: no
re-tessellation, and the 30 GB bake is untouched.

That order assumption is load-bearing, so every model is CHECKED rather than trusted:
the stored type this script computes for face j must equal the bake's face_type on every
triangle whose face_id is j. A model that fails the check is reported and skipped.

Output, one file per model:  <out>/<prefix>/<model>_<part>.npz
    stored     uint8[n_faces]   TYPE_INDEX of the stored surface type
    canonical  uint8[n_faces]   TYPE_INDEX of the simplest primitive it exactly is

Usage:
    python relabel_sidecar.py --files val --bake D:/abc-data/bake --out D:/abc-data/relabel
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from occ_tessellate import TYPE_INDEX, TYPE_NAMES, face_type_name, read_step  # noqa: E402
from stage1_bake import find_step  # noqa: E402

TOL = 1e-6


def val_split(bake: Path, seed: int = 0, val_frac: float = 0.08):
    """Reproduce train.py's split exactly: sorted glob, numpy shuffle, first 8%."""
    files = sorted(bake.glob("*/*.npz"))
    np.random.default_rng(seed).shuffle(files)
    nval = max(1, int(len(files) * val_frac))
    return files[:nval], files[nval:]


def one(task):
    npz_path, step_root, out_root = task
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from canonical import canonical_type

    rec = {"file": npz_path, "status": "ok", "changed": Counter(), "n_faces": 0}
    try:
        d = np.load(npz_path)
        mid, part = str(d["model_id"]), str(d["part"])
        sp = find_step(Path(step_root), mid)
        if sp is None:
            rec["status"] = "no_step"
            return rec
        shape = read_step(str(sp))
        if shape is None:
            rec["status"] = "unreadable"
            return rec
        stored, canon = [], []
        ex = TopExp_Explorer(shape, TopAbs_FACE)
        while ex.More():
            f = TopoDS.Face_s(ex.Current())
            ex.Next()
            st = face_type_name(f)
            c = canonical_type(f, TOL, st)[0]
            stored.append(TYPE_INDEX[st])
            canon.append(TYPE_INDEX[c])
            if c != st:
                rec["changed"][(st, c)] += 1
        stored = np.array(stored, np.uint8)
        canon = np.array(canon, np.uint8)
        rec["n_faces"] = len(stored)

        # Alignment check against the bake, on every density present.
        for dens in ("coarse", "medium", "fine"):
            key = f"{dens}_face_id"
            if key not in d.files:
                continue
            fid = d[key]
            fty = d[f"{dens}_face_type"]
            if fid.size and (fid.max() >= len(stored) or np.any(stored[fid] != fty)):
                rec["status"] = "misaligned"
                return rec

        out = Path(out_root) / mid[:5] / f"{mid}_{part}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, stored=stored, canonical=canon)
    except Exception as exc:
        rec["status"] = f"error:{type(exc).__name__}"
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bake", type=Path, default=Path("D:/abc-data/bake"))
    ap.add_argument("--step-root", type=Path, default=Path("D:/abc-data/step"))
    ap.add_argument("--out", type=Path, default=Path("D:/abc-data/relabel"))
    ap.add_argument("--files", choices=["val", "train", "all"], default="val")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    val, train = val_split(args.bake)
    files = {"val": val, "train": train, "all": val + train}[args.files]
    if args.limit:
        files = files[: args.limit]
    print(f"sidecars for {len(files):,} {args.files} models, tol {TOL}")

    t0 = time.time()
    status, changed = Counter(), Counter()
    faces = 0
    tasks = [(str(f), str(args.step_root), str(args.out)) for f in files]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for i, rec in enumerate(pool.map(one, tasks, chunksize=4), 1):
            status[rec["status"]] += 1
            if rec["status"] == "ok":
                faces += rec["n_faces"]
                changed.update(rec["changed"])
            if i % 250 == 0:
                print(f"  {i:,}/{len(files):,}  {i / (time.time() - t0):.1f}/s", flush=True)

    print(f"\ndone in {(time.time() - t0) / 60:.1f} min")
    print("status: " + ", ".join(f"{k} {v:,}" for k, v in status.most_common()))
    print(f"faces examined: {faces:,}")
    print("relabels (stored -> canonical):")
    for (a, b), n in changed.most_common():
        print(f"  {a:>10} -> {b:<10} {n:>7,}")


if __name__ == "__main__":
    raise SystemExit(main())
