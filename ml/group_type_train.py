#!/usr/bin/env python3
"""
group_type_train.py - train the group-level type model on test/groupTypeDump.mjs rows.

    python group_type_train.py --train D:/abc-data/lab/s20/gt_typer \\
        --fill D:/abc-data/lab/s20/gt_fill --extra onshape=D:/abc-data/lab/s20/gt_onshape \\
        --out D:/abc-data/runs/gtype1

WHAT IT LEARNS. A finished painted group's export type (js/groupType.js GROUP_TYPES: plane, cylinder,
cone, sphere, torus, freeform) from the fit evidence classifyRegion already computes, the group's shape,
and today's decisions (the painted type, the classifier's pick, typeFinalGroup's pick). Labels are the
canonical type of the true face holding most of the group's area; rows are weighted by that face's share
of the group (purity), and classes by inverse square-root frequency.

SPLITS. --train: groups painted on parts no shipped model saw (export_fill_eval.py --exclude fill_eval).
--fill: fill_eval's groups, split by manifest index like test/autoSeed.mjs --parity: EVEN parts
validate (early stopping, the override threshold), ODD parts test. --extra: more test sets (Onshape).

THE DECISION. The model overrides today's rule only where its top probability reaches a threshold,
chosen on the even parts to maximise per-face accuracy (groups holding most of an analytic face,
counted per face: what 'surfaces found' depends on). Reports, per set and mesher, the accuracy
of the painted type, the rule, the classifier, the model alone and the combined decision, area-weighted
(each group by its share of its part's area) and per group, plus the groups the combination fixes and
breaks against the rule.

The export bakes the input normalisation in: input `features` [n, 58], output `probs` [n, 6].
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

CLASSES = ["plane", "cylinder", "cone", "sphere", "torus", "freeform"]
FILL_MANIFEST = "D:/abc-data/fill_eval/manifest.json"


def load(prefixes):
    X, R = [], []
    cols = None
    for p in prefixes:
        h = json.load(open(p + ".json"))
        cols = h["columns"]
        x = np.fromfile(p + ".feat", np.float32).reshape(-1, len(cols))
        rows = [json.loads(l) for l in open(p + ".rows.jsonl")]
        assert len(rows) == len(x), p
        X.append(x)
        R += rows
    return np.concatenate(X), R, cols


def as_class(t):
    return CLASSES.index(t) if t in CLASSES else CLASSES.index("freeform")


class GroupTyper(nn.Module):
    def __init__(self, mean, std, width=128, n_out=6, dropout=0.1):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
        k = len(mean)
        self.net = nn.Sequential(nn.Linear(k, width), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(width, width), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(width, n_out))

    def logits(self, x):
        return self.net((x - self.mean) / self.std)

    def forward(self, x):
        return torch.softmax(self.logits(x), dim=-1)


def face_rows(rows):
    """Groups holding most of a true ANALYTIC face (coverage >= 0.5): the segment surfaceScore scores
    that face by. Accuracy over them, counted per face, is what 'surfaces found' depends on."""
    return np.array([r["label"] != "freeform" and r["coverage"] >= 0.5 for r in rows])


def decide(probs, rule_cls, thr):
    top = probs.argmax(1)
    return np.where(probs.max(1) >= thr, top, rule_cls)


def report(name, rows, probs, thr, by="mesher"):
    y = np.array([as_class(r["label"]) for r in rows])
    w = np.array([r["areaShare"] for r in rows])
    rule = np.array([as_class(r["rule"]) for r in rows])
    painted = np.array([as_class(r["painted"]) for r in rows])
    classified = np.array([as_class(r["classified"]) for r in rows])
    model = probs.argmax(1)
    comb = decide(probs, rule, thr)
    groups = sorted({r[by] for r in rows})
    fr = face_rows(rows)
    out = {}
    print(f"\n{name}  ({len(rows)} groups, {int(fr.sum())} holding most of an analytic face; "
          f"accuracy per FACE / area-weighted over all groups; combined at {thr:.2f})")
    print(f"  {'':10s} {'painted':>14s} {'rule':>14s} {'classifier':>14s} {'model':>14s} {'combined':>14s}   faces fixed/broken vs rule")
    for g in groups + ["all"]:
        sel = np.array([g == "all" or r[by] == g for r in rows])
        sf = sel & fr
        acc = lambda p: (100 * (p[sf] == y[sf]).mean() if sf.any() else float("nan"),
                         100 * (w[sel] * (p[sel] == y[sel])).sum() / max(w[sel].sum(), 1e-12))
        fixed = int(((comb == y) & (rule != y) & sf).sum())
        broken = int(((comb != y) & (rule == y) & sf).sum())
        cells = [acc(p) for p in (painted, rule, classified, model, comb)]
        print(f"  {g:10s} " + " ".join(f"{a:6.1f}/{b:6.1f}" for a, b in cells) + f"   +{fixed}/-{broken}")
        out[g] = {k: c for k, c in zip(("painted", "rule", "classifier", "model", "combined"), cells)}
        out[g]["fixed"], out[g]["broken"] = fixed, broken
    # per true class, combined vs rule recall: over face-holding groups for primitives, every group for freeform
    line = []
    for k, c in enumerate(CLASSES):
        s = (y == k) & (fr if c != "freeform" else np.ones_like(fr))
        if s.sum():
            line.append(f"{c} {100 * (rule[s] == k).mean():.0f}->{100 * (comb[s] == k).mean():.0f} (n={int(s.sum())})")
    print("  recall rule->combined: " + ", ".join(line))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, help="comma-separated dump prefixes")
    ap.add_argument("--fill", required=True, help="comma-separated fill_eval dump prefixes (even validates, odd tests)")
    ap.add_argument("--extra", nargs="*", default=[], help="name=prefix[,prefix] extra test sets")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    Xtr, Rtr, cols = load(args.train.split(","))
    Xf, Rf, _ = load(args.fill.split(","))
    index = {e["part"]: i for i, e in enumerate(json.load(open(FILL_MANIFEST))["parts"])}
    even = np.array([index[r["part"]] % 2 == 0 for r in Rf])
    Xva, Rva = Xf[even], [r for r, e in zip(Rf, even) if e]
    Xte, Rte = Xf[~even], [r for r, e in zip(Rf, even) if not e]
    extras = {}
    for spec in args.extra:
        name, pre = spec.split("=", 1)
        extras[name] = load(pre.split(","))[:2]

    ytr = np.array([as_class(r["label"]) for r in Rtr])
    wtr = np.array([r["purity"] for r in Rtr], np.float32)
    counts = np.bincount(ytr, minlength=len(CLASSES)).astype(np.float64)
    cw = 1.0 / np.sqrt(np.maximum(counts, 1))
    cw = cw / (cw * counts).sum() * counts.sum()
    print(f"train {len(Rtr)} groups ({', '.join(f'{c} {int(n)}' for c, n in zip(CLASSES, counts))}); "
          f"validate {len(Rva)} (fill_eval even); test {len(Rte)} (fill_eval odd)"
          + "".join(f"; {k} {len(v[1])}" for k, v in extras.items()))

    mean, std = Xtr.mean(0), Xtr.std(0) + 1e-6
    net = GroupTyper(mean, std, args.width)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    xt, yt, wt = torch.from_numpy(Xtr), torch.from_numpy(ytr), torch.from_numpy(wtr)
    cwt = torch.tensor(cw, dtype=torch.float32)
    xv = torch.from_numpy(Xva)
    yv = np.array([as_class(r["label"]) for r in Rva])
    fv = face_rows(Rva)
    best, best_state = -1.0, None
    for ep in range(1, args.epochs + 1):
        net.train()
        perm = torch.randperm(len(xt))
        for s in range(0, len(xt), 1024):
            i = perm[s:s + 1024]
            loss = (nn.functional.cross_entropy(net.logits(xt[i]), yt[i], weight=cwt, reduction="none") * wt[i]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            pv = net(xv).numpy()
        acc = (pv.argmax(1)[fv] == yv[fv]).mean()
        if acc > best:
            best, best_state = acc, {k: v.clone() for k, v in net.state_dict().items()}
        if ep % 10 == 0:
            print(f"  epoch {ep:3d}  loss {float(loss):.4f}  val per-face model accuracy {100 * acc:.2f}%")
    net.load_state_dict(best_state)
    net.eval()

    # the override threshold, on the even parts
    with torch.no_grad():
        pv = net(xv).numpy()
    rule_v = np.array([as_class(r["rule"]) for r in Rva])
    best_thr, best_acc = 1.01, (rule_v[fv] == yv[fv]).mean()
    for thr in np.arange(0.30, 1.0001, 0.02):
        acc = (decide(pv, rule_v, thr)[fv] == yv[fv]).mean()
        if acc > best_acc + 1e-9:
            best_thr, best_acc = float(thr), acc
    print(f"\nthreshold chosen on the even parts: {best_thr:.2f} (per-face accuracy {100 * best_acc:.2f}%, rule alone {100 * (rule_v[fv] == yv[fv]).mean():.2f}%)")

    results = {"threshold": best_thr}
    with torch.no_grad():
        results["val"] = report("fill_eval EVEN (validation)", Rva, net(xv).numpy(), best_thr)
        results["test"] = report("fill_eval ODD (test)", Rte, net(torch.from_numpy(Xte)).numpy(), best_thr)
        for name, (X, R) in extras.items():
            results[name] = report(name, R, net(torch.from_numpy(X)).numpy(), best_thr)

    torch.save({"model": net.state_dict(), "columns": cols, "classes": CLASSES, "threshold": best_thr,
                "args": vars(args) | {"out": str(args.out)}}, args.out / "gtype.pt")
    onnx_path = args.out / "group-typer.onnx"
    torch.onnx.export(net, torch.zeros(1, len(cols)), str(onnx_path), input_names=["features"],
                      output_names=["probs"], dynamic_axes={"features": {0: "groups"}, "probs": {0: "groups"}},
                      opset_version=17)
    (args.out / "group-typer.json").write_text(json.dumps(
        {"columns": cols, "classes": CLASSES, "threshold": best_thr, "results": results}, indent=1))
    print(f"\n-> {onnx_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
