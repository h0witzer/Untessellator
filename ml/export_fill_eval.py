#!/usr/bin/env python3
"""
export_fill_eval.py - write validation parts, meshed several ways, for test/fillEval.mjs.

fillEval.mjs has to run the real JS path (seedNet.proposeRegion -> regionGrow.paintFrom),
and Node cannot read .npz. So each part is written as flat files per mesher:

    <stem>.pos   float32, 9 per triangle (raw, unwelded, as the bake stores it)
    <stem>.fid   int32 B-rep face id per triangle
    <stem>.typ   uint8 canonical surface type per triangle (relabel sidecar)

plus manifest.json. The FIRST variant is the reference fillEval.mjs picks seed faces on;
face ids are shared across bakes, so every mesher is clicked on the same faces.

Parts come only from the main bake's validation split (the same split train.py uses), and
only parts present in every variant are kept. Re-meshed training copies are drawn from the
training split, so no exported part reached training under any mesher.

Usage:
    python export_fill_eval.py --out D:/abc-data/fill_eval --n 80
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from relabel_sidecar import val_split  # noqa: E402

DEFAULT_VARIANTS = ("native=D:/abc-data/bake,nurbs=D:/abc-data/bake_nurbs,"
                    "gmsh=D:/abc-data/bake_gmsh,game=D:/abc-data/bake_game2")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=80, help="parts to export")
    ap.add_argument("--variants", default=DEFAULT_VARIANTS,
                    help="name=bake_dir pairs, comma separated; the first is the reference")
    ap.add_argument("--bake", type=Path, default=Path("D:/abc-data/bake"),
                    help="main bake whose validation split defines the parts")
    ap.add_argument("--relabel", type=Path, default=Path("D:/abc-data/relabel"))
    ap.add_argument("--density", default="medium")
    ap.add_argument("--max-tris", type=int, default=150_000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--split", choices=("val", "train"), default="val",
                    help="val (the default) for evaluation sets; train for TRAINING data of models "
                         "other than the seed model, e.g. the edge model (test/edgeScreen.mjs), "
                         "which then has every validation part held out")
    args = ap.parse_args(argv)

    variants = dict(pair.split("=", 1) for pair in args.variants.split(","))
    variants = {k: Path(v) for k, v in variants.items()}
    args.out.mkdir(parents=True, exist_ok=True)

    val, train = val_split(args.bake)
    pool = val if args.split == "val" else train
    cands = [f for f in pool
             if all((root / f.relative_to(args.bake)).exists() for root in variants.values())
             and (args.relabel / f.parent.name / f.name).exists()]
    rng = np.random.default_rng(args.seed)
    pick = [cands[i] for i in sorted(rng.choice(len(cands), size=min(args.n, len(cands)), replace=False))]

    key = f"{args.density}_positions"
    manifest = []
    for f in pick:
        rel = f.relative_to(args.bake)
        canon = np.load(args.relabel / f.parent.name / f.name)["canonical"]
        entry = {"part": f.stem, "variants": {}}
        for name, root in variants.items():
            d = np.load(root / rel)
            if key not in d.files or len(d[key]) > args.max_tris:
                continue
            fid = d[f"{args.density}_face_id"].astype(np.int32)
            stem = f"{f.stem}_{name}"
            d[key].astype(np.float32).tofile(args.out / f"{stem}.pos")
            fid.tofile(args.out / f"{stem}.fid")
            canon[fid].astype(np.uint8).tofile(args.out / f"{stem}.typ")
            entry["variants"][name] = {"stem": stem, "tris": int(len(fid))}
        if entry["variants"]:
            manifest.append(entry)

    (args.out / "manifest.json").write_text(json.dumps(
        {"variants": list(variants), "density": args.density, "parts": manifest}, indent=1))
    per = sum(len(e["variants"]) for e in manifest) / max(len(manifest), 1)
    print(f"{len(cands):,} {args.split} parts present in all of {', '.join(variants)}; "
          f"exported {len(manifest)} ({per:.2f} variants each) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
