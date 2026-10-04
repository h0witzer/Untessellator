#!/usr/bin/env python3
"""
test_model_variants.py - every SeedFillNet option builds, trains, and exports to the same numbers.

    python test_model_variants.py

For each variant (aggregation max / gate / attn, edge attributes 1 or 2, seed concat, seed
cross-attention, the training-only per-facet surface head): a forward and backward pass on a
ragged batch with padding and missing neighbours, a check that padding cannot move any real
facet's output, and an ONNX export compared with PyTorch under onnxruntime. The training-only
head must be absent from the export. Exit status is non-zero on any failure.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import SeedFillNet, count_parameters, export_onnx  # noqa: E402

VARIANTS = {
    "pointnet": dict(),
    "v7 (max)": dict(gnn_layers=2, surface=True),
    "gate": dict(gnn_layers=2, agg="gate"),
    "attn": dict(gnn_layers=2, agg="attn"),
    "gate, 2 edge attrs": dict(gnn_layers=2, agg="gate", edge_dim=2),
    "seed concat": dict(gnn_layers=2, seed_concat=True),
    "seed attention": dict(gnn_layers=2, seed_attn=4),
    "facet surface": dict(gnn_layers=2, surface=True, facet_surface=True),
}


def batch(n_feat: int, edge_dim: int, rng):
    b, n = 3, 40
    x = torch.from_numpy(rng.normal(size=(b, n, n_feat)).astype(np.float32))
    m = torch.ones(b, n)
    m[1, 25:] = 0
    m[2, 10:] = 0
    nbr = torch.from_numpy(rng.integers(-1, n, size=(b, n, 3)).astype(np.int64))
    nbr = torch.where(nbr >= m.sum(1, keepdim=True).long().unsqueeze(-1), torch.full_like(nbr, -1), nbr)
    shape = (b, n, 3) if edge_dim == 1 else (b, n, 3, edge_dim)
    ea = torch.from_numpy(rng.random(size=shape).astype(np.float32))
    return x, m, nbr, ea


def main() -> int:
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    try:
        import onnxruntime as ort
    except ImportError:
        ort = None
        print("onnxruntime not installed: export comparison skipped")
    fails = 0
    for name, kw in VARIANTS.items():
        net = SeedFillNet(in_features=25, n_types=9, width=64, **kw)
        ed = kw.get("edge_dim", 1)
        x, m, nbr, ea = batch(25, ed, rng)
        extra = (nbr, ea) if kw.get("gnn_layers") else ()
        net.emit_aux = bool(kw.get("facet_surface"))
        outs = net(x, m, *extra)
        loss = sum(o.float().pow(2).mean() for o in outs)
        loss.backward()
        expect = 2 + bool(kw.get("surface")) + bool(kw.get("facet_surface"))
        ok = len(outs) == expect and all(torch.isfinite(o).all() for o in outs)
        # padding must not reach real facets: scramble the padded rows and compare
        net.eval()
        net.emit_aux = False
        with torch.no_grad():
            a = net(x, m, *extra)
            x2 = x.clone()
            x2[m == 0] = 1e3
            b = net(x2, m, *extra)
        leak = max(float(((a[0] - b[0]) * m).abs().max()), float((a[1] - b[1]).abs().max()))
        ok &= leak < 1e-4
        diff = float("nan")
        if ort is not None:
            with tempfile.TemporaryDirectory() as d:
                path = str(Path(d) / "m.onnx")
                export_onnx(net, path, n_facets=40, in_features=25)
                sess = ort.InferenceSession(path)
                feeds = {"features": x.numpy(), "mask": m.numpy()}
                if extra:
                    feeds.update(nbr=nbr.numpy(), eattr=ea.numpy())
                onnx_out = sess.run(None, feeds)
                names = [o.name for o in sess.get_outputs()]
                ok &= len(onnx_out) == 2 + bool(kw.get("surface"))      # never the training-only head
                diff = max(float(np.abs(o - t.numpy()).max()) for o, t in zip(onnx_out, a))
                ok &= diff < 1e-4
        fails += not ok
        print(f"  {name:20s} params {count_parameters(net):>8,}  outputs {len(outs)}  padding leak {leak:.1e}  "
              f"onnx diff {diff:.1e}  {'ok' if ok else 'FAIL'}")
    print("PASS" if not fails else f"FAIL ({fails})")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
