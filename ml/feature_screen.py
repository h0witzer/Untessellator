#!/usr/bin/env python3
"""
feature_screen.py - which candidate inputs are worth a training run?

Reads the dump from test/featureScreen.mjs and trains small probe networks with the model's
current inputs (base) and with each candidate group added. Two probes:

  membership   per facet: is it on the seed's face?  scored by AUC
  type         per seed, from pooled patch features (mean and std over the patch, mean
               within 0.5 patch radii): Plane / Cylinder / Cone / Sphere / Torus accuracy

Parts are split in half (even index trains, odd tests), and every mesher of the test parts is
scored, so a group that helps only on the mesher it trained on shows up as not transferring.
Two training sets: native (OCCT) only, like s1; and native + nurbs + gmsh, like s3.

A probe is not the model: it has no attention over the patch. Use it to rank groups for a
training run, not to predict the run's numbers.

Usage:
    python feature_screen.py D:/abc-data/feature_screen/screen
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch import nn

TYPES = ("Plane", "Cylinder", "Cone", "Sphere", "Torus")


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def mlp(k: int, out: int) -> nn.Module:
    return nn.Sequential(nn.Linear(k, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, out))


def standardise(train: np.ndarray, *others: np.ndarray):
    mu = train.mean(0)
    sd = train.std(0) + 1e-6
    return [(a - mu) / sd for a in (train, *others)]


def membership(X, y, train_mask, test_masks, dev, inits=3, epochs=3):
    xtr, *xte = standardise(X[train_mask], *(X[m] for m in test_masks.values()))
    ytr = torch.tensor(y[train_mask], dtype=torch.float32, device=dev)
    xtr_t = torch.tensor(xtr, device=dev)
    pos = float(ytr.mean())
    out = {name: [] for name in test_masks}
    for init in range(inits):
        torch.manual_seed(init)
        net = mlp(X.shape[1], 1).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=2e-3)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / max(pos, 1e-6), device=dev))
        n = len(xtr_t)
        for _ in range(epochs):
            perm = torch.randperm(n, device=dev)
            for s in range(0, n, 8192):
                idx = perm[s:s + 8192]
                opt.zero_grad()
                loss = loss_fn(net(xtr_t[idx]).squeeze(1), ytr[idx])
                loss.backward()
                opt.step()
        with torch.no_grad():
            for (name, m), xt in zip(test_masks.items(), xte):
                score = net(torch.tensor(xt, device=dev)).squeeze(1).cpu().numpy()
                out[name].append(auc(score, y[m]))
    return {k: float(np.mean(v)) for k, v in out.items()}


def pooled(X, seed, dist):
    """Per seed: mean and std over the patch, and mean over facets within 0.5 patch radii."""
    s = seed - seed.min()
    S = s.max() + 1
    cnt = np.bincount(s, minlength=S)[:, None]
    tot = np.zeros((S, X.shape[1]))
    sq = np.zeros((S, X.shape[1]))
    np.add.at(tot, s, X)
    np.add.at(sq, s, X.astype(np.float64) ** 2)
    mean = tot / cnt
    std = np.sqrt(np.maximum(sq / cnt - mean ** 2, 0))
    near = dist < 0.5
    ltot = np.zeros((S, X.shape[1]))
    np.add.at(ltot, s[near], X[near])
    lcnt = np.maximum(np.bincount(s[near], minlength=S), 1)[:, None]
    return np.concatenate([mean, std, ltot / lcnt], 1).astype(np.float32)


def type_probe(P, labels, train_mask, test_masks, dev, inits=3, epochs=400):
    xtr, *xte = standardise(P[train_mask], *(P[m] for m in test_masks.values()))
    xtr_t = torch.tensor(xtr, device=dev)
    ytr = torch.tensor(labels[train_mask], device=dev)
    out = {name: [] for name in test_masks}
    for init in range(inits):
        torch.manual_seed(init)
        net = mlp(P.shape[1], len(TYPES)).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-3)
        for _ in range(epochs):
            opt.zero_grad()
            loss = nn.functional.cross_entropy(net(xtr_t), ytr)
            loss.backward()
            opt.step()
        with torch.no_grad():
            for (name, m), xt in zip(test_masks.items(), xte):
                pred = net(torch.tensor(xt, device=dev)).argmax(1).cpu().numpy()
                out[name].append(float((pred == labels[m]).mean()))
    return {k: float(np.mean(v)) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    args = ap.parse_args()
    hdr = json.load(open(args.prefix + ".json"))
    X = np.fromfile(args.prefix + ".feat", np.float32).reshape(-1, hdr["ncol"])
    M = np.fromfile(args.prefix + ".meta", np.int32).reshape(-1, 5)
    seed, mesher, part, stype, same = M.T
    cols = {c: i for i, c in enumerate(hdr["columns"])}
    groups = {g: [cols[c] for c in names] for g, names in hdr["groups"].items()}
    meshers = hdr["meshers"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{len(X)} rows, {hdr['seeds']} seeds, device {dev}")

    base = groups["base"]
    configs = {"base": base}
    for g in groups:
        if g != "base":
            configs[f"+{g}"] = base + groups[g]
    configs["+all but shape"] = base + [i for g in groups if g not in ("base", "shape") for i in groups[g]]

    train_part = part % 2 == 0
    trainings = {"native": [meshers.index("native")],
                 "native+nurbs+gmsh": [meshers.index(m) for m in ("native", "nurbs", "gmsh")]}

    # Seed-level table for the type probe.
    first = np.unique(seed, return_index=True)[1]
    s_mesher, s_part, s_type = mesher[first], part[first], stype[first]
    type_names = hdr["typeNames"]
    keep_types = np.array([type_names[t] in TYPES for t in s_type])
    s_label = np.array([TYPES.index(type_names[t]) if type_names[t] in TYPES else -1 for t in s_type])
    dist = X[:, cols["dist"]]

    for tname, train_meshers in trainings.items():
        tr_rows = train_part & np.isin(mesher, train_meshers)
        test_rows = {m: (~train_part) & (mesher == i) for i, m in enumerate(meshers)}
        tr_seeds = keep_types & (s_part % 2 == 0) & np.isin(s_mesher, train_meshers)
        test_seeds = {m: keep_types & (s_part % 2 == 1) & (s_mesher == i) for i, m in enumerate(meshers)}
        print(f"\n== trained on {tname}; scored on the held-out parts "
              f"({tr_rows.sum()} training facets, {tr_seeds.sum()} training seeds; "
              f"test seeds per mesher {[int(v.sum()) for v in test_seeds.values()]})")
        print(f"  {'inputs':<18}" + "".join(f"{'AUC ' + m:>13}" for m in meshers)
              + "  |" + "".join(f"{'type ' + m:>13}" for m in meshers))
        base_row = None
        for cname, idx in configs.items():
            mem = membership(X[:, idx], same, tr_rows, test_rows, dev)
            P = pooled(X[:, idx], seed, dist)
            typ = type_probe(P, s_label, tr_seeds, test_seeds, dev)
            if base_row is None:
                base_row = (mem, typ)
                cells = [f"{mem[m]:>13.3f}" for m in meshers] + ["  |"] + [f"{typ[m]:>13.0%}" for m in meshers]
            else:
                cells = [f"{mem[m]:>7.3f} {mem[m] - base_row[0][m]:+.3f}" for m in meshers] + ["  |"] + \
                        [f"{typ[m]:>7.0%} {100 * (typ[m] - base_row[1][m]):+4.0f}" for m in meshers]
            print(f"  {cname:<18}" + "".join(cells), flush=True)


if __name__ == "__main__":
    main()
