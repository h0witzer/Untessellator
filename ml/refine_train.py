#!/usr/bin/env python3
"""
refine_train.py - train the fill-REFINEMENT model on test/refineDump.mjs rows.

    python refine_train.py --train D:/abc-data/lab/refine_train_0,... --val D:/abc-data/lab/refine_val
                           --out D:/abc-data/runs/refine1 [--gnn-layers 6]

The seed model predicts "which facets share this click's face" from geometry alone. This one gets a
first attempt as well: the fill paintFrom made, the surface fitted to it, every facet's distance and
angle to that surface, the learned boundary along paths from the click, and optionally a NEGATIVE
CLICK on a spill (js/refineFeatures.js). It is the seed model's architecture (model.SeedFillNet)
with a wider input; its type head is unused.

Training rows must come from parts that no model in the loop has seen: the dump's fills use an edge
model, whose scores are inputs here. Validation is reported for plain patches and for patches with
a click separately, and in terms of the fill: how much of the fill's error (spilled facets in the
patch, missed ones) the refined region repairs.

The export is model.export_onnx's graph (outputs same_logit, type_logit), with a sidecar naming the
input columns.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import SeedFillNet, count_parameters, export_onnx  # noqa: E402

MAX_FACETS = 512


def load(prefixes: list[str], eprob: bool = False):
    """Concatenated dumps. With `eprob`, EA carries a second edge attribute per neighbour, the learned
    boundary probability (<prefix>.eprob, written by refineDump.mjs since 2026-10-03)."""
    X, M, NB, EA, patches = [], [], [], [], []
    base = 0
    for p in prefixes:
        hdr = json.load(open(p + ".json"))
        ncol = hdr["ncol"]
        x = np.fromfile(p + ".feat", np.float32).reshape(-1, ncol)
        m = np.fromfile(p + ".meta", np.int32).reshape(-1, 7)
        nb = np.fromfile(p + ".nbr", np.int32).reshape(-1, 3)
        ea = np.fromfile(p + ".eattr", np.float32).reshape(-1, 3)
        if eprob:
            ea = np.stack([ea, np.fromfile(p + ".eprob", np.float32).reshape(-1, 3)], axis=-1)
        X.append(x); M.append(m); NB.append(nb); EA.append(ea)
        off = base
        for rec in hdr["patches"]:
            patches.append({**rec, "start": off})
            off += rec["n"]
        base += len(x)
        cols = hdr["refineColumns"]
    return (np.concatenate(X), np.concatenate(M), np.concatenate(NB), np.concatenate(EA), patches, ncol, cols)


class Batches:
    def __init__(self, data, idx, batch, shuffle, rng):
        self.X, self.M, self.NB, self.EA, self.patches = data
        self.idx, self.batch, self.shuffle, self.rng = idx, batch, shuffle, rng

    def __iter__(self):
        order = self.rng.permutation(self.idx) if self.shuffle else self.idx
        for s in range(0, len(order), self.batch):
            chunk = order[s:s + self.batch]
            b = len(chunk)
            x = np.zeros((b, MAX_FACETS, self.X.shape[1]), np.float32)
            y = np.zeros((b, MAX_FACETS), np.float32)
            reg = np.zeros((b, MAX_FACETS), np.float32)
            m = np.zeros((b, MAX_FACETS), np.float32)
            nb = np.full((b, MAX_FACETS, 3), -1, np.int64)
            ea = np.zeros((b, MAX_FACETS) + self.EA.shape[1:], np.float32)
            click = np.zeros(b, bool)
            for i, pi in enumerate(chunk):
                rec = self.patches[pi]
                a, n = rec["start"], min(rec["n"], MAX_FACETS)
                x[i, :n] = self.X[a:a + n]
                y[i, :n] = self.M[a:a + n, 4]
                reg[i, :n] = self.M[a:a + n, 6]
                m[i, :n] = 1
                g = self.NB[a:a + n]
                nb[i, :n] = np.where(g < n, g, -1)
                ea[i, :n] = self.EA[a:a + n]
                click[i] = rec["click"] >= 0
            yield (torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m), torch.from_numpy(nb),
                   torch.from_numpy(ea), torch.from_numpy(reg), torch.from_numpy(click))


@torch.no_grad()
def evaluate(net, loader, device, gnn):
    net.eval()
    acc = {k: np.zeros(4) for k in ("plain", "click")}   # tp fp fn, patches
    fix = {k: np.zeros(4) for k in ("plain", "click")}   # spilled before, spilled after, missed before, missed after
    for x, y, m, nb, ea, reg, click in loader:
        x, y, m = x.to(device), y.to(device), m.to(device)
        extra = (nb.to(device), ea.to(device)) if gnn else ()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            sl = net(x, m, *extra)[0]
        pred = (sl.float().sigmoid() > 0.5).float() * m
        reg = reg.to(device) * m
        for key, sel in (("plain", ~click), ("click", click)):
            if not sel.any():
                continue
            s = sel.to(device)
            p, t, r, mm = pred[s], y[s] * m[s], reg[s], m[s]
            acc[key] += [float((p * t).sum()), float((p * (1 - t) * mm).sum()), float(((1 - p) * t).sum()), float(s.sum())]
            fix[key] += [float((r * (1 - t) * mm).sum()), float((p * (1 - t) * mm).sum()),
                         float(((1 - r) * t).sum()), float(((1 - p) * t).sum())]
    net.train()
    out = {}
    for key in acc:
        tp, fp, fn, n = acc[key]
        if n == 0:
            continue
        sb, sa, mb, ma = fix[key]
        out[key] = {"iou": tp / max(tp + fp + fn, 1e-9), "patches": int(n),
                    "spilled_before": sb, "spilled_after": sa, "missed_before": mb, "missed_after": ma}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, help="comma-separated dump prefixes")
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--gnn-layers", type=int, default=0)
    ap.add_argument("--agg", choices=("max", "gate", "attn"), default="max",
                    help="how each MeshConv layer combines a facet's three messages (model.MeshConv)")
    ap.add_argument("--edge-dim", type=int, choices=(1, 2), default=1,
                    help="2 adds the learned boundary probability (.eprob) to the dihedral as an edge attribute")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    *tr, trp, ncol, cols = load(args.train.split(","), args.edge_dim == 2)
    *va, vap, _, _ = load(args.val.split(","), args.edge_dim == 2)
    train_data = (*tr, trp)
    val_data = (*va, vap)
    pos = float(tr[1][:, 4].mean())
    pw = torch.tensor((1 - pos) / max(pos, 1e-6), device=device)
    n_click = sum(1 for p in trp if p["click"] >= 0)
    print(f"train {len(trp):,} patches ({n_click:,} with a click), {len(tr[0]):,} rows; val {len(vap):,} patches; "
          f"{ncol} inputs; positive rate {pos:.3f}; device {device}")

    net = SeedFillNet(in_features=ncol, n_types=2, surface=False, gnn_layers=args.gnn_layers,
                      edge_dim=args.edge_dim, agg=args.agg).to(device)
    print(f"parameters {count_parameters(net):,}; message passing {args.gnn_layers} layers, "
          f"aggregation {args.agg}, edge attributes {args.edge_dim}")
    steps = args.epochs * ((len(trp) + args.batch - 1) // args.batch)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    train_loader = Batches(train_data, np.arange(len(trp)), args.batch, True, rng)
    val_loader = Batches(val_data, np.arange(len(vap)), 128, False, rng)
    best = -1.0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tot = 0.0
        k = 0
        for x, y, m, nb, ea, _reg, _click in train_loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            extra = (nb.to(device), ea.to(device)) if args.gnn_layers else ()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                sl = net(x, m, *extra)[0]
                loss = F.binary_cross_entropy_with_logits(sl.float(), y, weight=m, pos_weight=pw,
                                                          reduction="sum") / m.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss)
            k += 1
        ev = evaluate(net, val_loader, device, args.gnn_layers)
        line = []
        for key, r in ev.items():
            line.append(f"{key}: IoU {r['iou']:.4f}, spilled facets {r['spilled_before']:.0f} -> {r['spilled_after']:.0f}, "
                        f"missed {r['missed_before']:.0f} -> {r['missed_after']:.0f} ({r['patches']} patches)")
        print(f"epoch {epoch:>2}  loss {tot / max(k, 1):.4f}  {time.time() - t0:.0f} s   " + "   ".join(line), flush=True)
        score = ev["plain"]["iou"]
        if score > best:
            best = score
            torch.save({"model": net.state_dict(), "epoch": epoch, "args": vars(args), "ncol": ncol,
                        "columns": cols}, args.out / "refine.pt")
    ck = torch.load(args.out / "refine.pt", map_location="cpu", weights_only=False)
    net = SeedFillNet(in_features=ncol, n_types=2, surface=False, gnn_layers=args.gnn_layers,
                      edge_dim=args.edge_dim, agg=args.agg)
    net.load_state_dict(ck["model"])
    export_onnx(net, str(args.out / "refine.onnx"), in_features=ncol)
    (args.out / "refine.json").write_text(json.dumps({"refine": True, "n_features": ncol, "refineColumns": cols,
                                                      "gnn_layers": args.gnn_layers, "agg": args.agg,
                                                      "edge_dim": args.edge_dim, "epoch": ck["epoch"]}))
    print(f"best plain IoU {best:.4f} (epoch {ck['epoch']}) -> {args.out / 'refine.onnx'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
