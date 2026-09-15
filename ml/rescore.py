#!/usr/bin/env python3
"""
rescore.py - score an existing checkpoint against STORED and CANONICAL labels.

The question it answers: of the type head's apparent errors, how many were the model
being wrong, and how many were the label being wrong? A revolved line predicted as
Cylinder is scored an error against the stored label and correct against the canonical
one. Running both on identical patches separates label noise from model ability before
any retrain is spent.

Uses train.py's exact validation split, so no patch here was seen in training.

Seeds are drawn per stored surface type (up to --per-type per mesh), because uniform
seeding would leave rare classes like Revolution with a handful of samples. Per-class
accuracy is conditional on the class, so this does not bias it.

Usage:
    python rescore.py --ckpt D:/abc-data/runs/v2ctrl/best.pt
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from meshcore import adjacency_csr, build_adjacency, build_facets  # noqa: E402
from model import SeedFillNet  # noqa: E402
from occ_tessellate import TYPE_NAMES  # noqa: E402
from patches import FEATURE_NAMES, N_FEATURES, N_TYPES, collate, extract_patch  # noqa: E402
from relabel_sidecar import val_split  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, default=Path("D:/abc-data/runs/v2ctrl/best.pt"))
    ap.add_argument("--bake", type=Path, default=Path("D:/abc-data/bake"))
    ap.add_argument("--split-bake", type=Path, default=Path("D:/abc-data/bake"),
                    help="bake whose split defines the validation parts; --bake may be a "
                         "re-meshed variant holding the same parts (mesher_bake.py)")
    ap.add_argument("--intersect", type=Path, nargs="*", default=[],
                    help="only parts present in every one of these bakes, so native and "
                         "re-meshed variants are scored on identical parts")
    ap.add_argument("--relabel", type=Path, default=Path("D:/abc-data/relabel"))
    ap.add_argument("--density", default="medium")
    ap.add_argument("--per-type", type=int, default=4)
    ap.add_argument("--max-facets", type=int, default=512)
    ap.add_argument("--zero-curvature", action="store_true", default=True,
                    help="columns 10-11 at zero, as the control arm was trained")
    ap.add_argument("--zero-features", default=None,
                    help="comma list of FEATURE_NAMES to zero. Default: the columns the "
                         "checkpoint was trained with at zero (train.py --zero-features)")
    args = ap.parse_args(argv)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    net = SeedFillNet(in_features=N_FEATURES, n_types=N_TYPES)
    net.load_state_dict(ck["model"])
    net.to(dev).eval()
    if args.zero_features is not None:
        zero_cols = {FEATURE_NAMES.index(n) for n in args.zero_features.split(",") if n}
    else:
        # checkpoints from before --zero-features carry no zero_cols; curvature is added below
        zero_cols = set(ck["args"].get("zero_cols", ()))
    if args.zero_curvature:
        zero_cols |= {FEATURE_NAMES.index("k1s"), FEATURE_NAMES.index("k2s")}
    zero_cols = sorted(zero_cols)

    val, _ = val_split(args.split_bake)
    if args.bake != args.split_bake:
        # A variant bake holds only some parts, so its own sorted glob would give a
        # different split. Take the main split and look for the same files in the variant.
        val = [p for p in (args.bake / f.relative_to(args.split_bake) for f in val) if p.exists()]
    if args.intersect:
        val = [p for p in val
               if all((other / p.relative_to(args.bake)).exists() for other in args.intersect)]
    rng = np.random.default_rng(123)
    samples, stored_lab, canon_lab = [], [], []
    used = missing = 0
    for f in val:
        d = np.load(f)
        key = f"{args.density}_positions"
        if key not in d.files:
            continue
        mid, part = str(d["model_id"]), str(d["part"])
        side = args.relabel / mid[:5] / f"{mid}_{part}.npz"
        if not side.exists():
            missing += 1
            continue
        s = np.load(side)
        canon = s["canonical"]
        pos = d[key]
        fid = d[f"{args.density}_face_id"]
        fty = d[f"{args.density}_face_type"]
        if pos.shape[0] < 32:
            continue
        facets = build_facets(pos)
        pairs, _ = build_adjacency(pos, facets)
        indptr, indices = adjacency_csr(pairs, facets["count"])
        used += 1
        for t in np.unique(fty):
            cand = np.flatnonzero((fty == t) & ~facets["degenerate"])
            if cand.size == 0:
                continue
            for seed in rng.choice(cand, size=min(args.per_type, cand.size), replace=False):
                p = extract_patch(facets, indptr, indices, fid, fty, int(seed),
                                  args.max_facets, rng=None, spin=False, curv=None)
                if p is None:
                    continue
                samples.append(p)
                stored_lab.append(int(fty[seed]))
                canon_lab.append(int(canon[fid[seed]]))

    print(f"checkpoint {args.ckpt}  (step {ck.get('step')})  "
          f"zeroed: {', '.join(FEATURE_NAMES[i] for i in zero_cols) or 'none'}")
    print(f"validation models used {used:,}   without sidecar {missing:,}   patches {len(samples):,}\n")

    preds, tps, fps, fns = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(samples), 256):
            x, y, m, _ = collate(samples[i:i + 256], args.max_facets)
            if zero_cols:
                x[:, :, zero_cols] = 0.0
            sl, tl = net(torch.from_numpy(x).to(dev), torch.from_numpy(m).to(dev))
            preds.append(tl.float().argmax(1).cpu().numpy())
            member = (sl.float().sigmoid().cpu().numpy() > 0.5) & (m > 0)
            truth = (y > 0.5) & (m > 0)
            tps.append((member & truth).sum(1))
            fps.append((member & ~truth).sum(1))
            fns.append((~member & truth).sum(1))
    pred = np.concatenate(preds)
    TP, FP, FN = (np.concatenate(a).astype(np.float64) for a in (tps, fps, fns))
    S = np.array(stored_lab)
    C = np.array(canon_lab)

    print(f"  {'type':<11} {'stored n':>8} {'acc':>7}   {'canon n':>8} {'acc':>7}")
    macro_s, macro_c = [], []
    for k, name in enumerate(TYPE_NAMES):
        ns, nc = int((S == k).sum()), int((C == k).sum())
        a_s = float((pred[S == k] == k).mean()) if ns else None
        a_c = float((pred[C == k] == k).mean()) if nc else None
        if a_s is not None:
            macro_s.append(a_s)
        if a_c is not None:
            macro_c.append(a_c)
        fs = f"{a_s:7.3f}" if a_s is not None else "      -"
        fc = f"{a_c:7.3f}" if a_c is not None else "      -"
        print(f"  {name:<11} {ns:>8} {fs}   {nc:>8} {fc}")
    print(f"\n  macro accuracy   stored {np.mean(macro_s):.4f}   canonical {np.mean(macro_c):.4f}")
    print(f"  overall accuracy stored {np.mean(pred == S):.4f}   canonical {np.mean(pred == C):.4f}")
    print(f"  seed labels changed by relabeling: {int((S != C).sum()):,} of {len(S):,}")

    rev = TYPE_NAMES.index("Revolution")
    for label, L in (("stored", S), ("canonical", C)):
        sel = L == rev
        if sel.any():
            cnt = Counter(TYPE_NAMES[p] for p in pred[sel])
            parts = ", ".join(f"{k} {100 * v / sel.sum():.0f}%" for k, v in cnt.most_common())
            print(f"\n  seeds labelled Revolution ({label}, n={int(sel.sum())}) predicted as: {parts}")
    moved = (S == rev) & (C != rev)
    if moved.any():
        agree = float((pred[moved] == C[moved]).mean())
        print(f"  of the stored-Revolution seeds relabeled to a primitive (n={int(moved.sum())}), "
              f"the model already predicted that primitive {100 * agree:.1f}% of the time")

    # Membership does not depend on type labels, so this is the one number directly
    # comparable across checkpoints trained on stored, canonical or synthetic-mixed labels.
    # Pooled IoU (sums over patches) per canonical seed class, matching train.py's metric.
    iou = lambda sel: TP[sel].sum() / max(TP[sel].sum() + FP[sel].sum() + FN[sel].sum(), 1e-9)
    print(f"\n  membership IoU (pooled)  all seeds {iou(np.ones_like(C, bool)):.4f}")
    print("  by canonical seed class: " + ", ".join(
        f"{name} {iou(C == k):.3f}" for k, name in enumerate(TYPE_NAMES) if (C == k).any()))


if __name__ == "__main__":
    raise SystemExit(main())
