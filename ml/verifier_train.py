#!/usr/bin/env python3
"""
verifier_train.py - can a model pick the best of several candidate fills for one click?

    python verifier_train.py D:/abc-data/lab/s8_even.json D:/abc-data/lab/s8_odd.json

Both files are test/growEval.mjs --verifier-features outputs over the same methods: the first on
the even parts (trains), the second on the odd parts (tests). Every fill carries region features
computed without the truth (growEval.regionFeatures: size, fit residual, where its outline sits on
the learned boundary and on creases, facets just outside that would also fit, walls inside). A
small network learns P(clean) per (fill, method); for each test click the fill it rates highest is
chosen. Reported per mesher against the default method (the first in the file), the best single
method, and the oracle best-of-N.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np
import torch
from torch import nn


def clean(r):
    return r["surfacePurity"] >= 0.95 and r["recall"] >= 0.95


def table(path):
    rows = json.load(open(path))["rows"]
    methods = []
    for r in rows:
        if r["method"] not in methods:
            methods.append(r["method"])
    return rows, methods


def features(rows, methods):
    X = np.array([r["vf"] + [1.0 if r["method"] == m else 0.0 for m in methods] for r in rows], np.float32)
    # a region whose fit failed has no residual: give it a fixed out-of-range value, not NaN, which
    # would poison the standardisation and make every score NaN (argmax then picks the first method)
    X = np.clip(np.nan_to_num(X, nan=-20.0, posinf=20.0, neginf=-20.0), -20.0, 20.0)
    y = np.array([1.0 if clean(r) else 0.0 for r in rows], np.float32)
    return X, y


def main():
    train_path, test_path = sys.argv[1], sys.argv[2]
    tr, methods = table(train_path)
    te, methods_te = table(test_path)
    assert methods == methods_te, "train and test must run the same methods"
    Xtr, ytr = features(tr, methods)
    Xte, _ = features(te, methods)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    xt = torch.tensor((Xtr - mu) / sd, device=dev)
    yt = torch.tensor(ytr, device=dev)
    scores = np.zeros(len(te))
    for seed in range(5):
        torch.manual_seed(seed)
        net = nn.Sequential(nn.Linear(Xtr.shape[1], 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1)).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-3)
        for _ in range(600):
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(net(xt).squeeze(1), yt)
            loss.backward()
            opt.step()
        with torch.no_grad():
            scores += net(torch.tensor((Xte - mu) / sd, device=dev)).squeeze(1).cpu().numpy()
    by = defaultdict(list)
    for r, s in zip(te, scores):
        by[(r["mesher"], r["part"], r["face"])].append((s, r))
    print(f"methods: {', '.join(methods)}; train {len(tr)} fills, test {len(te)} fills")
    for mesher in ["native", "nurbs", "gmsh", "game"]:
        clicks = [v for k, v in by.items() if k[0] == mesher]
        if not clicks:
            continue
        n = len(clicks)
        default = sum(clean(next(r for _, r in c if r["method"] == methods[0])) for c in clicks) / n
        per = {m: sum(clean(next(r for _, r in c if r["method"] == m)) for c in clicks) / n for m in methods}
        best_single = max(per.items(), key=lambda kv: kv[1])
        chosen = sum(clean(max(c, key=lambda sr: (sr[0], sr[1]["method"] == methods[0]))[1]) for c in clicks) / n
        oracle = sum(any(clean(r) for _, r in c) for c in clicks) / n
        print(f"  {mesher:<7} n {n:>4}  default ({methods[0]}) {default:.0%}   best single ({best_single[0]}) "
              f"{best_single[1]:.0%}   VERIFIER {chosen:.0%}   oracle best-of-{len(methods)} {oracle:.0%}")


if __name__ == "__main__":
    main()
