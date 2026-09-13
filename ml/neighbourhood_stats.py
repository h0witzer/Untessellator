#!/usr/bin/env python3
"""
neighbourhood_stats.py - does a synthetic bake look like real parts from a seed's view?

A model trained on seeded patches learns from what a patch looks like around its seed.
If synthetic data differs systematically - most visibly in how much of a patch shares the
seed's face - the model can learn a shortcut that holds on synthetic parts and fails on
real ones. This compares the two, per canonical class, on identical sampling.

The first synthetic set failed this check for Revolution seeds (mean 0.84, p10 0.71,
against real 0.52 and 0.08), because every part was dominated by its revolve. Run this
after any generator change, before baking at scale.

Real data uses train.py's validation split with relabel sidecars, so its labels are
canonical like the synthetic bake's.

Usage:
    python neighbourhood_stats.py --synth D:/abc-data/synth_smoke
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from meshcore import adjacency_csr, build_adjacency, build_facets  # noqa: E402
from occ_tessellate import TYPE_NAMES  # noqa: E402
from patches import extract_patch  # noqa: E402
from relabel_sidecar import val_split  # noqa: E402


def measure(files, sidecar_root=None, cap=600, per=3, density="medium", seed=5):
    rng = np.random.default_rng(seed)
    pos, used = defaultdict(list), 0
    for f in files:
        if used >= cap:
            break
        try:
            d = np.load(f)
            key = f"{density}_positions"
            if key not in d.files:
                continue
            fid = d[f"{density}_face_id"]
            fty = d[f"{density}_face_type"]
            if sidecar_root is not None:
                mid, part = str(d["model_id"]), str(d["part"])
                sc = Path(sidecar_root) / mid[:5] / f"{mid}_{part}.npz"
                if not sc.exists():
                    continue
                canon = np.load(sc)["canonical"][fid]
            else:
                canon = fty
            P = d[key]
            if P.shape[0] < 32:
                continue
        except Exception:
            continue          # tolerate a sidecar still being written
        fa = build_facets(P)
        pairs, _ = build_adjacency(P, fa)
        indptr, indices = adjacency_csr(pairs, fa["count"])
        used += 1
        for t in np.unique(canon):
            cand = np.flatnonzero((canon == t) & ~fa["degenerate"])
            for s in rng.choice(cand, size=min(per, cand.size), replace=False):
                p = extract_patch(fa, indptr, indices, fid, canon, int(s), 512, rng=None, spin=False)
                if p is not None:
                    pos[int(t)].append(float(p["same"].mean()))
    return pos, used


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synth", type=Path, required=True)
    ap.add_argument("--bake", type=Path, default=Path("D:/abc-data/bake"))
    ap.add_argument("--relabel", type=Path, default=Path("D:/abc-data/relabel"))
    ap.add_argument("--real-cap", type=int, default=600)
    ap.add_argument("--synth-cap", type=int, default=100000)
    args = ap.parse_args(argv)

    val, _ = val_split(args.bake)
    real, ru = measure(val, args.relabel, cap=args.real_cap)
    syn, su = measure(sorted(args.synth.glob("*/*.npz")), None, cap=args.synth_cap)

    print(f"real (canonical labels): {ru} models    synthetic: {su} models\n")
    head = f"{'n':>6} {'mean':>6} {'p10':>5} {'p50':>5} {'p90':>5}"
    print(f"  {'class':<11} {head}   {head}")
    print(f"  {'':<11} {'--- real ---':^30}   {'--- synthetic ---':^30}")

    def cols(v):
        if not v:
            return f"{'-':>6} {'-':>6} {'-':>5} {'-':>5} {'-':>5}"
        a = np.array(v)
        return (f"{len(a):>6} {a.mean():6.3f} {np.percentile(a, 10):5.2f} "
                f"{np.percentile(a, 50):5.2f} {np.percentile(a, 90):5.2f}")

    for k, name in enumerate(TYPE_NAMES):
        if real[k] or syn[k]:
            print(f"  {name:<11} {cols(real[k])}   {cols(syn[k])}")


if __name__ == "__main__":
    raise SystemExit(main())
