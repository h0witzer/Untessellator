#!/usr/bin/env python3
"""
export_onnx.py - export a train.py checkpoint for js/seedNet.js, with its zeroed inputs baked in.

    python export_onnx.py D:/abc-data/runs/v5noarea/best.pt            # -> seedfill.onnx beside it
    python export_onnx.py D:/abc-data/runs/v5noarea/best.pt out.onnx

WHY THE ZEROING GOES INSIDE THE GRAPH

A run trained with --no-curvature or --zero-features saw those columns at exactly zero.
seedNet.buildFeatures fills every column it can, so a browser caller feeds such a model
inputs it never saw unless it remembers to zero them. That is not hypothetical:
test/meshReport.mjs's first run did it with curvature and read masterchief.stl as Plane
49/60 instead of Cone 37/60. Here the columns are multiplied by zero as the graph's first
operation, so the exported model cannot be fed them.

The export is checked against PyTorch on a random patch (with the same columns zeroed on
the PyTorch side) and the exit status is non-zero if they disagree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import json

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import SeedFillNet, export_onnx  # noqa: E402
from patches import FEATURE_NAMES, N_FEATURES, N_TYPES  # noqa: E402


def zeroed_columns(ckpt_args: dict) -> list[int]:
    """Columns a run held at zero, from its stored arguments."""
    zero = set(ckpt_args.get("zero_cols", ()))
    if ckpt_args.get("no_curvature"):
        zero |= {FEATURE_NAMES.index("k1s"), FEATURE_NAMES.index("k2s")}
    return sorted(zero)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__)
        return 2
    ckpt = Path(argv[0])
    out = Path(argv[1]) if len(argv) > 1 else ckpt.with_name("seedfill.onnx")

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    zero = zeroed_columns(ck["args"])
    # v6 runs (train.py --features v6) take the extended inputs; the browser must build them
    # with buildFeatures(..., extended = true), which the sidecar JSON tells it.
    n_features = int(ck["args"].get("n_features", N_FEATURES))
    net = SeedFillNet(in_features=n_features, n_types=N_TYPES,
                      surface=ck["args"].get("surface_root") is not None)
    net.load_state_dict(ck["model"])
    net.eval()
    export_onnx(net, str(out), in_features=n_features, zero_cols=zero)
    out.with_suffix(".json").write_text(json.dumps({
        "features": ck["args"].get("features", "base"), "n_features": n_features,
        "zero_cols": zero, "step": ck.get("step"),
    }))

    import onnxruntime as ort
    x = np.random.default_rng(0).normal(size=(1, 300, n_features)).astype(np.float32)
    m = np.ones((1, 300), np.float32)
    outs_onnx = ort.InferenceSession(str(out)).run(None, {"features": x, "mask": m})
    xz = x.copy()
    xz[:, :, zero] = 0.0
    with torch.no_grad():
        outs_t = net(torch.from_numpy(xz), torch.from_numpy(m))
    diff = max(float(np.abs(a - b.numpy()).max()) for a, b in zip(outs_onnx, outs_t))
    print(f"{out}: step {ck.get('step')}, zeroed {[FEATURE_NAMES[i] for i in zero] or 'none'}, "
          f"max |onnx - torch| {diff:.2e}")
    return 0 if diff < 1e-4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
