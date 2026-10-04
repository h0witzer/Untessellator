#!/usr/bin/env python3
"""
patch_eval.py - the quick held-out gate for patch models: where membership and type go wrong.

    python patch_eval.py --dump D:/abc-data/lab/refine_val \\
        --models v7gnn=D:/abc-data/runs/v7gnn/best.pt refine1=D:/abc-data/runs/refine1/refine.pt \\
        [--ablate-dihedral] [--out report.json]

The dump is test/refineDump.mjs output on parts no model trained on: each 512-facet patch around a
click, meshed four ways (OCCT, NURBS->OCCT, gmsh, game-style), with the seed model's 25 inputs, the
refiner's 9 fill-state columns, the patch graph, and the truth. Seed models read the first 25
columns; refiners read all 34. It runs in minutes, so every training arm is scored here before any
fill is grown (test/growEval.mjs remains the gate that counts).

Per model, per mesher:
  IoU, precision, recall    pooled over facets inside the patch; seed logit > 0, refiner >= 1
  clean patches             share with >= 95% precision and >= 95% recall inside the patch
  error by border distance  error rate by ring distance to the nearest facet of another face
                            (0 = touching a border, 6 = six or more rings away), and the share of
                            all errors within 2 rings. Whole-region mistakes sit far from borders.
  type                      per-type accuracy and confusion for the 5 analytic types present
  messages                  per MeshConv layer, the share of message pre-activation energy from
                            own state / neighbour difference / edge attributes, and the update
                            ratio |relu(aggregate)| / |h|
Baselines from the dump itself: the first fill (in_region) and the fit test connected to the seed.

Findings from the first run (2026-10-03) are in docs/ml-experiment-scope.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import build_from_args  # noqa: E402

MESHERS = ["OCCT", "NURBS", "gmsh", "game"]
TYPES = ['Plane', 'Cylinder', 'Cone', 'Sphere', 'Torus', 'Revolution', 'Extrusion', 'BSpline', 'Other']
DMAX = 6


def load_dump(prefix: str):
    hdr = json.load(open(prefix + ".json"))
    nc = hdr["ncol"]
    X = np.fromfile(prefix + ".feat", np.float32).reshape(-1, nc)
    M = np.fromfile(prefix + ".meta", np.int32).reshape(-1, 7)
    NB = np.fromfile(prefix + ".nbr", np.int32).reshape(-1, 3)
    EA = np.fromfile(prefix + ".eattr", np.float32).reshape(-1, 3)
    pe = Path(prefix + ".eprob")
    EP = np.fromfile(pe, np.float32).reshape(-1, 3) if pe.exists() else None
    patches, off = [], 0
    for r in hdr["patches"]:
        patches.append({**r, "start": off})
        off += r["n"]
    cols = hdr["refineColumns"]
    return hdr, X, M, NB, EA, EP, patches, cols


def load_model(path: str, dev):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    args = dict(ck["args"] if isinstance(ck["args"], dict) else vars(ck["args"]))
    refiner = "ncol" in ck
    if refiner:
        args["n_features"] = ck["ncol"]
    net = build_from_args(args, 2 if refiner else len(ck.get("type_names", TYPES)))
    net.load_state_dict(ck["model"])
    net.emit_aux = False
    return net.to(dev).eval(), refiner, args


def dist_to_border(y, nb, n):
    d = np.full(n, 99)
    q = deque()
    for i in range(n):
        for j in nb[i]:
            if 0 <= j < n and y[j] != y[i]:
                d[i] = 0
                q.append(i)
                break
    while q:
        i = q.popleft()
        for j in nb[i]:
            if 0 <= j < n and d[j] > d[i] + 1:
                d[j] = d[i] + 1
                q.append(j)
    return np.minimum(d, DMAX)


def connected(mask, nb, n):
    out = np.zeros(n, bool)
    mask = mask.copy()
    mask[0] = True
    out[0] = True
    q = deque([0])
    while q:
        i = q.popleft()
        for j in nb[i]:
            if 0 <= j < n and mask[j] and not out[j]:
                out[j] = True
                q.append(j)
    return out


class Tally:
    def __init__(self):
        self.c = np.zeros((4, 3))
        self.clean = np.zeros(4)
        self.npatch = np.zeros(4)
        self.err_d = np.zeros(DMAX + 1)
        self.cnt_d = np.zeros(DMAX + 1)
        self.conf = np.zeros((9, 9), np.int64)

    def add(self, mi, p, y, d):
        tp, fp, fn = (p & y).sum(), (p & ~y).sum(), (~p & y).sum()
        self.c[mi] += [tp, fp, fn]
        self.npatch[mi] += 1
        self.clean[mi] += (tp >= .95 * max(tp + fp, 1)) and (tp >= .95 * max(tp + fn, 1))
        e = p != y
        np.add.at(self.err_d, d[e], 1)
        np.add.at(self.cnt_d, d, 1)

    def report(self):
        tp, fp, fn = self.c.T
        tot = self.c.sum(0)
        r = {"iou": (tp / np.maximum(tp + fp + fn, 1)).round(4).tolist(),
             "precision": (tp / np.maximum(tp + fp, 1)).round(4).tolist(),
             "recall": (tp / np.maximum(tp + fn, 1)).round(4).tolist(),
             "iou_pooled": round(float(tot[0] / max(tot.sum(), 1)), 4),
             "clean": (self.clean / np.maximum(self.npatch, 1)).round(4).tolist(),
             "err_by_ring": (self.err_d / np.maximum(self.cnt_d, 1)).round(4).tolist(),
             "err_share_within_2": round(float(self.err_d[:3].sum() / max(self.err_d.sum(), 1)), 4)}
        if self.conf.sum():
            rows = self.conf[:5]
            r["type_acc"] = {TYPES[i]: round(float(rows[i, i] / max(rows[i].sum(), 1)), 4) for i in range(5)}
            r["type_acc_all"] = round(float(np.trace(self.conf) / max(self.conf.sum(), 1)), 4)
            r["confusion"] = self.conf[:5].tolist()
        return r


@torch.no_grad()
def message_composition(net, x, m, nb, ea):
    """Per MeshConv layer: energy shares of the message's three input blocks, update ratio."""
    out = []
    mk = m.unsqueeze(-1)
    h = net.enc1(x) * mk
    emask = ((nb >= 0) & (m.unsqueeze(-1) > 0)).float()
    safe = torch.where(nb >= 0, nb, torch.zeros_like(nb))
    for conv in net.convs:
        agg, z = conv.aggregate(h, safe, emask, ea)
        w = h.shape[-1]
        W = conv.msg[0].weight
        terms = [z[..., :w] @ W[:, :w].T, z[..., w:2 * w] @ W[:, w:2 * w].T, z[..., 2 * w:] @ W[:, 2 * w:].T]
        en = [float(((t ** 2).sum(-1) * emask).sum()) for t in terms]
        upd = torch.relu(agg)
        out.append(en + [float((upd.norm(dim=-1) * m).sum()), float((h.norm(dim=-1) * m).sum())])
        h = conv.norm(h + upd) * mk
    return np.array(out) if out else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", default="D:/abc-data/lab/refine_val")
    ap.add_argument("--models", nargs="+", required=True, help="name=checkpoint.pt ...")
    ap.add_argument("--refiner-threshold", type=float, default=1.0)
    ap.add_argument("--ablate-dihedral", action="store_true", help="also score seed models with eattr = 0")
    ap.add_argument("--baselines", action="store_true", help="also score the first fill and the fit test")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hdr, X, M, NB, EA, EP, patches, cols = load_dump(args.dump)
    nc = hdr["ncol"]
    C = {c: 25 + i for i, c in enumerate(cols)}

    models = []
    for spec in args.models:
        name, path = spec.split("=", 1)
        net, refiner, margs = load_model(path, dev)
        models.append((name, net, refiner))
        if args.ablate_dihedral and not refiner and net.gnn_layers:
            models.append((name + " (dihedral=0)", net, "nodih"))
    tallies = {name: Tally() for name, _, _ in models}
    comps = {}
    if args.baselines:
        for b in ("first fill", "fit test, connected"):
            tallies[b] = Tally()

    for s in range(0, len(patches), 64):
        ch = patches[s:s + 64]
        b = len(ch)
        x = np.zeros((b, 512, nc), np.float32)
        m = np.zeros((b, 512), np.float32)
        nb = np.full((b, 512, 3), -1, np.int64)
        ea = np.zeros((b, 512, 3), np.float32)
        ep = np.zeros((b, 512, 3), np.float32)
        for i, r in enumerate(ch):
            a, n = r["start"], min(r["n"], 512)
            x[i, :n] = X[a:a + n]
            m[i, :n] = 1
            g = NB[a:a + n]
            nb[i, :n] = np.where(g < n, g, -1)
            ea[i, :n] = EA[a:a + n]
            if EP is not None:
                ep[i, :n] = EP[a:a + n]
        xt, mt, nbt, eat = (torch.from_numpy(t).to(dev) for t in (x, m, nb, ea))
        ept = torch.from_numpy(ep).to(dev)
        xt[..., 10:12] = 0
        preds, types = {}, {}
        for name, net, kind in models:
            feats = xt if kind is True else xt[..., :net.enc1[0].in_features]
            e = eat
            if net.gnn_layers and net.convs[0].edge_dim == 2:
                e = torch.stack([eat, ept], dim=-1)
            if kind == "nodih":
                e = torch.zeros_like(e)
            extra = (nbt, e) if net.gnn_layers else ()
            outs = net(feats, mt, *extra)
            thr = args.refiner_threshold if kind is True else 0.0
            preds[name] = (outs[0] >= thr if kind is True else outs[0] > thr).cpu().numpy()
            if kind is not True:
                types[name] = outs[1].argmax(1).cpu().numpy()
            if net.gnn_layers and kind != "nodih":
                c = message_composition(net, feats, mt, nbt, e)
                comps[name] = c if name not in comps else comps[name] + c
        for i, r in enumerate(ch):
            a, n = r["start"], min(r["n"], 512)
            y = M[a:a + n, 4].astype(bool)
            g = nb[i, :n]
            d = dist_to_border(y, g, n)
            mi = r["mesher"]
            for name in preds:
                tallies[name].add(mi, preds[name][i, :n], y, d)
                if name in types:
                    tallies[name].conf[r["type"], types[name][i]] += 1
            if args.baselines:
                f = x[i, :n]
                tallies["first fill"].add(mi, f[:, C["in_region"]] > .5, y, d)
                fit = (f[:, C["dist"]] <= 1) & (f[:, C["angle"]] <= 1)
                tallies["fit test, connected"].add(mi, connected(fit, g, n), y, d)

    report = {"dump": args.dump, "patches": len(patches),
              "per_mesher": dict(zip(MESHERS, np.bincount([p["mesher"] for p in patches], minlength=4).tolist())),
              "models": {}}
    print(f"{len(patches)} held-out patches  ({', '.join(f'{k} {v}' for k, v in report['per_mesher'].items())})\n")
    print(f"{'model':28s} {'IoU OCCT/NURBS/gmsh/game':28s} {'pooled':>7s} {'clean patches':26s} "
          f"{'err@border':>10s} {'err@6+':>7s} {'<=2 rings':>9s} {'type':>6s}")
    for name, t in tallies.items():
        r = t.report()
        if name in comps:
            c = comps[name]
            en = c[:, :3]
            r["messages"] = [{"own": round(float(e[0] / e.sum()), 4), "neighbour_diff": round(float(e[1] / e.sum()), 4),
                              "edge": round(float(e[2] / e.sum()), 4), "update_ratio": round(float(u / hh), 4)}
                             for e, u, hh in zip(en, c[:, 3], c[:, 4])]
        report["models"][name] = r
        print(f"{name:28s} {'/'.join(f'{v:.3f}' for v in r['iou']):28s} {r['iou_pooled']:7.3f} "
              f"{'/'.join(f'{v:.2f}' for v in r['clean']):26s} {r['err_by_ring'][0]:10.3f} "
              f"{r['err_by_ring'][-1]:7.3f} {r['err_share_within_2']:9.2f} "
              f"{r.get('type_acc_all', float('nan')):6.3f}")
    for name, r in report["models"].items():
        if "type_acc" in r:
            print(f"\n{name} type accuracy: " + ", ".join(f"{k} {v:.2f}" for k, v in r["type_acc"].items()))
        if "messages" in r:
            print(f"{name} messages (own / neighbour diff / edge, update): "
                  + "  ".join(f"L{i + 1} {mm['own']:.2f}/{mm['neighbour_diff']:.2f}/{mm['edge']:.3f}, {mm['update_ratio']:.2f}"
                              for i, mm in enumerate(r["messages"])))
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))
        print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
