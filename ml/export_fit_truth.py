#!/usr/bin/env python3
"""
export_fit_truth.py - true analytic surface per B-rep face, for test/fitTruth.mjs.

The fill_eval export (export_fill_eval.py) already has each part meshed several ways as
flat .pos/.fid/.typ files. This adds, per part, the surface every face id really is, so
the fitters can be scored against exact parameters rather than against each other:

    <out>/<part>.srf   float64, 12 per face:  stored kind, canonical kind, params(10)

params use surface_targets.py's layout for the STORED kind. They are only meaningful when
stored == canonical; faces whose label was corrected (a BSpline that is really a torus)
carry no primitive parameters in the STEP and are skipped by the harness.

Usage:
    python export_fit_truth.py --fill-eval D:/abc-data/fill_eval --out D:/abc-data/fit_truth
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

NAMES = ("Plane", "Cylinder", "Cone", "Sphere", "Torus", "Revolution", "Extrusion", "BSpline", "Other")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fill-eval", type=Path, default=Path("D:/abc-data/fill_eval"))
    ap.add_argument("--surface", type=Path, default=Path("D:/abc-data/surface"))
    ap.add_argument("--relabel", type=Path, default=Path("D:/abc-data/relabel"))
    ap.add_argument("--out", type=Path, default=Path("D:/abc-data/fit_truth"))
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((args.fill_eval / "manifest.json").read_text())
    usable, relabelled = Counter(), Counter()
    written = 0
    for entry in manifest["parts"]:
        part = entry["part"]
        s = args.surface / part[:5] / f"{part}.npz"
        r = args.relabel / part[:5] / f"{part}.npz"
        if not (s.exists() and r.exists()):
            continue
        sd, rd = np.load(s), np.load(r)
        kind, canon = sd["kind"], rd["canonical"]
        if len(kind) != len(canon) or np.any(kind != rd["stored"]):
            print(f"  {part}: sidecars disagree, skipped")
            continue
        rows = np.concatenate([kind[:, None].astype(float), canon[:, None].astype(float), sd["params"]], 1)
        rows.astype(np.float64).tofile(args.out / f"{part}.srf")
        written += 1
        for k, c in zip(kind, canon):
            if c <= 4:
                (usable if k == c else relabelled)[NAMES[c]] += 1
    print(f"wrote {written} parts -> {args.out}")
    print("faces with exact params:   " + ", ".join(f"{k} {v}" for k, v in usable.most_common()))
    print("relabelled, no params:     " + ", ".join(f"{k} {v}" for k, v in relabelled.most_common()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
