#!/usr/bin/env python3
"""
edge_train.py - train the face-BOUNDARY scorer on dumped mesh edges and export it for the browser.

    python edge_train.py D:/abc-data/feature_screen/edges --out D:/abc-data/runs/edge1

Input is test/edgeScreen.mjs's dump: one row per manifold edge of 80 ABC parts meshed four ways,
labelled from the B-rep faces. Parts are split by index (even train, odd validate), so no part is in
both, and every mesher is scored separately.

WHAT IT IS FOR. Growth stops at a crease today, which cannot see a tangent junction: a fillet
running into a plane has no dihedral to notice, and that is where fills spill (with the true type
declared, 23% of clicks spill on OCCT meshes and 43% on gmsh). The probe screen (ml/edge_screen.py)
showed the boundary is learnable from the edge's neighbourhood: at a threshold that wrongly walls 2%
of within-face edges it catches 84% of smooth boundaries on OCCT meshes against 0% for the crease
rule.

INPUT SET. `--columns` selects feature groups from the dump's header. The default leaves out `size`
(triangle areas): it adds a little within a mesher and collapses across meshers, the same shortcut
`log_area` gave the type head.

The export bakes the input normalisation into the graph, so js/edgeFeatures.js can feed raw rows.
A sidecar names the columns in order, and the operating thresholds measured on validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


class EdgeNet(nn.Module):
    """Normalisation, then an MLP: raw feature rows in, one boundary logit out."""

    def __init__(self, mean: np.ndarray, std: np.ndarray, width: int = 96):
        super().__init__()
        k = len(mean)
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Linear(k, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net((features - self.mean) / self.std).squeeze(-1)


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def operating(scores: np.ndarray, labels: np.ndarray, smooth: np.ndarray, fpr: float):
    """Threshold from within-face edges; boundary recall and smooth-boundary recall there."""
    neg = scores[labels == 0]
    thr = float(np.quantile(neg, 1 - fpr))
    hit = scores >= thr
    pos = labels == 1
    sm = pos & smooth
    return thr, float(hit[pos].mean()), (float(hit[sm].mean()) if sm.any() else float("nan"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--columns", default="dihedral,curv,spread,fit",
                    help="feature groups from the dump header; 'size' is excluded by default")
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--fpr", type=float, default=0.02, help="within-face false-wall rate to report")
    ap.add_argument("--val-prefix", default=None,
                    help="a second dump (e.g. the fill_eval parts) to validate and set thresholds on; "
                         "then EVERY row of the main dump trains")
    ap.add_argument("--val-parity", choices=("all", "even", "odd"), default="all",
                    help="with --val-prefix, validate (and set thresholds) on these parts of it only, "
                         "keeping the rest untouched for a final test")
    ap.add_argument("--train-parity", choices=("even", "odd"), default="even",
                    help="which half of the parts trains; the other half validates and sets the "
                         "thresholds. A second model trained on the odd half gives the even parts "
                         "held-out scores too (test/growEval.mjs --edge-wall-even)")
    args = ap.parse_args()

    hdr = json.load(open(args.prefix + ".json"))
    groups = hdr["groups"]
    cols = {c: i for i, c in enumerate(hdr["columns"])}
    wanted = [g.strip() for g in args.columns.split(",") if g.strip()]
    unknown = [g for g in wanted if g not in groups]
    if unknown:
        ap.error(f"unknown groups {unknown}; choose from {', '.join(groups)}")
    names = [c for g in wanted for c in groups[g]]
    idx = [cols[c] for c in names]

    def load_cols(prefix, ncol, rows=None):
        """The chosen columns of a dump, read in chunks through a memmap so a multi-GB dump never
        sits in memory twice. `rows` (a boolean mask) keeps a subset."""
        mm = np.memmap(prefix + ".feat", np.float32, mode="r").reshape(-1, ncol)
        keep = np.arange(len(mm)) if rows is None else np.nonzero(rows)[0]
        out = np.empty((len(keep), len(idx)), np.float32)
        for s in range(0, len(keep), 1_000_000):
            out[s:s + 1_000_000] = mm[keep[s:s + 1_000_000]][:, idx]
        return out

    M = np.fromfile(args.prefix + ".meta", np.int32).reshape(-1, 5)
    if args.val_prefix:
        vh = json.load(open(args.val_prefix + ".json"))
        if vh["columns"] != hdr["columns"] or vh["meshers"] != hdr["meshers"]:
            ap.error("--val-prefix dump has different columns or meshers")
        MV = np.fromfile(args.val_prefix + ".meta", np.int32).reshape(-1, 5)
        vrows = np.ones(len(MV), bool) if args.val_parity == "all" else             (MV[:, 0] % 2 == (0 if args.val_parity == "even" else 1))
        xtr_np, xva_np = load_cols(args.prefix, hdr["ncol"]), load_cols(args.val_prefix, vh["ncol"], vrows)
        MT, MVa = M, MV[vrows]
    else:
        train = M[:, 0] % 2 == (0 if args.train_parity == "even" else 1)
        xtr_np, xva_np = load_cols(args.prefix, hdr["ncol"], train), load_cols(args.prefix, hdr["ncol"], ~train)
        MT, MVa = M[train], M[~train]
    ytr_np = MT[:, 2].astype(np.float32)
    va_mesher, va_y, va_smooth = MVa[:, 1], MVa[:, 2].astype(np.float32), MVa[:, 4] < 300
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(xtr_np) + len(xva_np):,} edges, {len(idx)} features ({', '.join(wanted)}); "
          f"train {len(xtr_np):,} / validate {len(xva_np):,}; device {dev}")

    s1 = np.zeros(len(idx)); s2 = np.zeros(len(idx))
    for s in range(0, len(xtr_np), 1_000_000):
        c = xtr_np[s:s + 1_000_000].astype(np.float64)
        s1 += c.sum(0); s2 += (c * c).sum(0)
    mean = (s1 / len(xtr_np)).astype(np.float32)
    std = (np.sqrt(np.maximum(s2 / len(xtr_np) - (s1 / len(xtr_np)) ** 2, 0)) + 1e-6).astype(np.float32)
    net = EdgeNet(mean, std, args.width).to(dev)
    pos_rate = float(ytr_np.mean())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos_rate) / max(pos_rate, 1e-6), device=dev))
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)

    xtr = torch.from_numpy(xtr_np).to(dev)
    ytr = torch.from_numpy(ytr_np).to(dev)
    xva = torch.from_numpy(xva_np).to(dev)
    X = xva_np   # for the export's input width
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * max(1, len(xtr) // args.batch + 1), pct_start=0.1)

    best = -1.0
    for epoch in range(1, args.epochs + 1):
        net.train()
        perm = torch.randperm(len(xtr), device=dev)
        total = 0.0
        for s in range(0, len(xtr), args.batch):
            i = perm[s:s + args.batch]
            opt.zero_grad()
            loss = loss_fn(net(xtr[i]), ytr[i])
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss) * len(i)
        net.eval()
        with torch.no_grad():
            scores = torch.cat([net(xva[s:s + 65536]) for s in range(0, len(xva), 65536)]).cpu().numpy()
        overall = auc(scores, va_y)
        line = []
        for i, m in enumerate(hdr["meshers"]):
            sel = va_mesher == i
            _thr, rec, sm = operating(scores[sel], va_y[sel], va_smooth[sel], args.fpr)
            line.append(f"{m} {auc(scores[sel], va_y[sel]):.3f}/{rec:.0%}/{sm:.0%}")
        print(f"  epoch {epoch:>2}  loss {total / len(xtr):.4f}  val AUC {overall:.4f}   "
              + "  ".join(line), flush=True)
        if overall > best:
            best = overall
            torch.save({"model": net.state_dict(), "columns": names, "args": vars(args),
                        "mean": mean, "std": std, "auc": overall}, args.out / "edge.pt")

    # Thresholds from the best model, per mesher and pooled, for the fill to use.
    ck = torch.load(args.out / "edge.pt", map_location=dev, weights_only=False)
    net.load_state_dict(ck["model"])
    net.eval()
    with torch.no_grad():
        scores = torch.cat([net(xva[s:s + 65536]) for s in range(0, len(xva), 65536)]).cpu().numpy()
    thresholds = {}
    for fpr in (0.01, 0.02, 0.05):
        thr, rec, sm = operating(scores, va_y, va_smooth, fpr)
        thresholds[f"fpr{fpr}"] = {"logit": thr, "recall": rec, "smooth_recall": sm}
        print(f"  pooled at {fpr:.0%} false walls: logit {thr:+.3f}, boundaries {rec:.1%}, smooth {sm:.1%}")

    onnx_path = args.out / "edge.onnx"
    torch.onnx.export(net.cpu().eval(), torch.zeros(1, X.shape[1]), str(onnx_path),
                      input_names=["features"], output_names=["logit"],
                      dynamic_axes={"features": {0: "edges"}, "logit": {0: "edges"}}, opset_version=17)
    (args.out / "edge.json").write_text(json.dumps({
        "columns": names, "groups": wanted, "thresholds": thresholds, "auc": best,
        "n_features": X.shape[1],
    }, indent=1))
    print(f"\nbest val AUC {best:.4f} -> {onnx_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
