#!/usr/bin/env python3
"""
edge_screen.py - can a face BOUNDARY be recognised from an edge's neighbourhood?

Reads the dump from test/edgeScreen.mjs and trains small probe networks on groups of features,
the way ml/feature_screen.py chose the v6 model inputs. Parts are split in half (even index trains,
odd tests) and every mesher of the test parts is scored, so a signal that only works on the mesher
it trained on shows up.

The headline is not AUC but the operating point growth needs:

    FPR 2%   the threshold where only 2% of WITHIN-FACE edges are called boundaries
    recall   the share of true boundary edges caught there
    smooth   the share of boundary edges under 30 degrees caught there - the ones the crease
             barrier cannot see at all, which is where fills spill

The crease rule itself (dihedral >= 30) is printed as the baseline in the same terms.

Usage:
    python edge_screen.py D:/abc-data/feature_screen/edges
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch import nn


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def at_fpr(scores: np.ndarray, labels: np.ndarray, smooth: np.ndarray, fpr: float = 0.02):
    """Threshold set on within-face edges; returns (boundary recall, smooth boundary recall)."""
    neg = scores[labels == 0]
    if neg.size == 0:
        return float("nan"), float("nan")
    thr = np.quantile(neg, 1 - fpr)
    hit = scores >= thr
    pos = labels == 1
    rec = float(hit[pos].mean()) if pos.any() else float("nan")
    sm = pos & smooth
    return rec, (float(hit[sm].mean()) if sm.any() else float("nan"))


def mlp(k: int) -> nn.Module:
    return nn.Sequential(nn.Linear(k, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))


def train_probe(X, y, train_mask, test_masks, dev, inits=3, epochs=4):
    mu = X[train_mask].mean(0)
    sd = X[train_mask].std(0) + 1e-6
    xtr = torch.tensor((X[train_mask] - mu) / sd, device=dev)
    ytr = torch.tensor(y[train_mask], dtype=torch.float32, device=dev)
    pos = float(ytr.mean())
    out = {name: [] for name in test_masks}
    for init in range(inits):
        torch.manual_seed(init)
        net = mlp(X.shape[1]).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=2e-3)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / max(pos, 1e-6), device=dev))
        n = len(xtr)
        for _ in range(epochs):
            perm = torch.randperm(n, device=dev)
            for s in range(0, n, 8192):
                idx = perm[s:s + 8192]
                opt.zero_grad()
                loss_fn(net(xtr[idx]).squeeze(1), ytr[idx]).backward()
                opt.step()
        with torch.no_grad():
            for name, m in test_masks.items():
                xt = torch.tensor((X[m] - mu) / sd, device=dev)
                out[name].append(net(xt).squeeze(1).cpu().numpy())
    return {k: np.mean(v, axis=0) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--fpr", type=float, default=0.02)
    args = ap.parse_args()
    hdr = json.load(open(args.prefix + ".json"))
    X = np.fromfile(args.prefix + ".feat", np.float32).reshape(-1, hdr["ncol"])
    M = np.fromfile(args.prefix + ".meta", np.int32).reshape(-1, 5)
    part, mesher, boundary, same_type, dihedral10 = M.T
    y = boundary.astype(np.float32)
    smooth = dihedral10 < 300          # under 30 degrees: invisible to the crease barrier
    cols = {c: i for i, c in enumerate(hdr["columns"])}
    groups = {g: [cols[c] for c in names] for g, names in hdr["groups"].items()}
    meshers = hdr["meshers"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{len(X):,} edges ({int(y.sum()):,} boundaries, interior sampled at "
          f"{hdr['interiorRate']}), device {dev}")

    base = groups["dihedral"]
    configs = {"dihedral only": base}
    for g in groups:
        if g != "dihedral":
            configs[f"+{g}"] = base + groups[g]
    configs["all"] = sorted({i for g in groups.values() for i in g})
    # Triangle-size features describe the TRIANGULATION, not the surface: the type head's failure
    # mode (docs/ml-experiment-scope.md, "Removing the triangulation features"). These configs show
    # what survives without them, and whether that transfers to a mesher it never trained on.
    configs["all but size"] = sorted({i for g, cols_ in groups.items() if g != "size" for i in cols_})
    configs["dihedral+curv+fit"] = sorted(set(groups["dihedral"] + groups["curv"] + groups["fit"]))
    configs["curv+fit only"] = sorted(set(groups["curv"] + groups["fit"]))

    train_part = part % 2 == 0
    trainings = {"native": [meshers.index("native")],
                 "all meshers": list(range(len(meshers)))}

    for tname, train_meshers in trainings.items():
        tr = train_part & np.isin(mesher, train_meshers)
        tests = {m: (~train_part) & (mesher == i) for i, m in enumerate(meshers)}
        print(f"\n== trained on {tname} ({int(tr.sum()):,} edges); scored on held-out parts")
        # The crease rule, in the same terms: its false-positive rate is what it costs inside faces.
        for m, mask in tests.items():
            crease = dihedral10[mask] >= 300
            lab = y[mask] == 1
            fp = float(crease[~lab].mean())
            print(f"   crease >= 30 on {m:<7} walls {fp:5.1%} of within-face edges, catches "
                  f"{float(crease[lab].mean()):5.1%} of boundaries, {0.0:5.1%} of smooth ones")
        print(f"  {'inputs':<16}" + "".join(f"{m:>26}" for m in meshers))
        print(f"  {'':<16}" + "".join(f"{'AUC  recall  smooth':>26}" for _ in meshers))
        for cname, idx in configs.items():
            scores = train_probe(X[:, idx], y, tr, tests, dev)
            cells = []
            for m, mask in tests.items():
                s = scores[m]
                a = auc(s, y[mask])
                rec, sm = at_fpr(s, y[mask], smooth[mask], args.fpr)
                cells.append(f"{a:>8.3f}{rec:>8.1%}{sm:>8.1%}  ")
            print(f"  {cname:<16}" + "".join(cells), flush=True)


if __name__ == "__main__":
    main()
