#!/usr/bin/env python3
"""
boundary_screen.py - does the seed model miss what the learned face boundary knows?

Reads test/boundaryScreen.mjs's dump and trains per-facet membership probes on the even parts,
scored on the odd parts of every mesher. The question is not whether edge-path features are
informative on their own (they are built from a boundary model), but whether they add anything
to the trained v6 model's own membership logit. So the decisive comparison is
`logit` against `logit + edge`: a probe that can only rescale the model's answer, against one
that can also correct it with the path features.

AUC is reported over all patch facets and over the HARD ones: facets reachable from the seed
without crossing a 30-degree crease (v6 `bottleneck` < 30/90), where the crease barrier cannot
help and spill actually happens.

Usage:
    python boundary_screen.py D:/abc-data/lab/s3_boundary
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch import nn

from feature_screen import auc, mlp


def fit_probe(X, y, train, dev, inits=3, epochs=4):
    """Train `inits` probes on the training rows; return mean scores for every row."""
    mu = X[train].mean(0)
    sd = X[train].std(0) + 1e-6
    xa = torch.tensor((X - mu) / sd, dtype=torch.float32, device=dev)
    xtr = xa[torch.tensor(train, device=dev)]
    ytr = torch.tensor(y[train], dtype=torch.float32, device=dev)
    pos = float(ytr.mean())
    out = np.zeros(len(X))
    for init in range(inits):
        torch.manual_seed(init)
        net = mlp(X.shape[1], 1).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=2e-3)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / max(pos, 1e-6), device=dev))
        n = len(xtr)
        for _ in range(epochs):
            perm = torch.randperm(n, device=dev)
            for s in range(0, n, 8192):
                idx = perm[s:s + 8192]
                opt.zero_grad()
                loss = loss_fn(net(xtr[idx]).squeeze(1), ytr[idx])
                loss.backward()
                opt.step()
        with torch.no_grad():
            out += torch.cat([net(xa[s:s + 65536]).squeeze(1) for s in range(0, len(xa), 65536)]).cpu().numpy()
    return out / inits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    args = ap.parse_args()
    hdr = json.load(open(args.prefix + ".json"))
    X = np.fromfile(args.prefix + ".feat", np.float32).reshape(-1, hdr["ncol"])
    M = np.fromfile(args.prefix + ".meta", np.int32).reshape(-1, 5)
    seed, mesher, part, stype, same = M.T
    cols = {c: i for i, c in enumerate(hdr["columns"])}
    g = {k: [cols[c] for c in v] for k, v in hdr["groups"].items()}
    meshers = hdr["meshers"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    hard = X[:, cols["bottleneck"]] < 30 / 90
    train = part % 2 == 0
    print(f"{len(X):,} rows, {hdr['seeds']} seeds; train {train.sum():,} rows (even parts, all meshers); "
          f"hard (crease-reachable) {hard.mean():.0%} of rows; device {dev}")

    configs = {
        "v6 inputs": g["v6"],
        "v6 + edge": g["v6"] + g["edge"],
        "logit": g["logit"],
        "logit + edge": g["logit"] + g["edge"],
        "logit + v6 + edge": g["logit"] + g["v6"] + g["edge"],
        "edge only": g["edge"],
    }
    # the model itself, no probe
    raw = X[:, cols["v6logit"]]
    print(f"\n  {'inputs':<20}" + "".join(f"{m + ' all':>13}{m + ' hard':>13}" for m in meshers))
    line = []
    for i, m in enumerate(meshers):
        sel = (~train) & (mesher == i)
        line += [f"{auc(raw[sel], same[sel]):>13.3f}", f"{auc(raw[sel & hard], same[sel & hard]):>13.3f}"]
    print(f"  {'v6feat as is':<20}" + "".join(line), flush=True)
    for name, idx in configs.items():
        s = fit_probe(X[:, idx], same, train, dev)
        line = []
        for i, m in enumerate(meshers):
            sel = (~train) & (mesher == i)
            line += [f"{auc(s[sel], same[sel]):>13.3f}", f"{auc(s[sel & hard], same[sel & hard]):>13.3f}"]
        print(f"  {name:<20}" + "".join(line), flush=True)


if __name__ == "__main__":
    main()
