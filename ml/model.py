#!/usr/bin/env python3
"""
model.py - the seeded-fill proposer.

Given a seed facet's neighbourhood, predict (a) which facets share the seed's B-rep
face and (b) what surface type that face is. Feeds the existing analytic fitters; it
never emits geometry.

ARCHITECTURE: PointNet-style, and deliberately so

A shared per-facet MLP, one masked max-pool to a patch descriptor, then a per-facet
head that sees local features concatenated with that descriptor. Roughly 200k
parameters, about 0.8 MB in fp32.

The reason for this shape rather than something stronger is DEPLOYMENT. The converter
is client-side WASM with no server, so whatever trains here has to reach ONNX Runtime
Web. That rules out sparse convolution outright, and makes DGCNN's dynamic k-NN
expensive. Here there is no KNN at all: the k-ring BFS already supplied locality when
the patch was built, so the network only needs to reason over a fixed-size set. Every
operation is a matmul, a LayerNorm, a ReLU or a masked max - all of which export
cleanly and run fast in the browser.

MASKING IS LOAD-BEARING

Patches are ragged and padded to a fixed width. Padding is excluded from the max-pool
by setting it to a large negative value before reducing, and excluded from the loss by
the same mask. Get this wrong and the global descriptor is contaminated by padding,
which is a silent accuracy leak that looks like underfitting.

TWO HEADS, DIFFERENT SHAPES

The membership head is per-facet and binary; the type head is per-patch and 9-way.
They share the encoder because deciding "is this facet on my surface" and "what surface
am I on" are the same geometric question asked two ways.
"""

from __future__ import annotations

import torch
import torch.nn as nn

NEG = -1e4   # finite, so it survives bf16/fp16 autocast; -inf produces NaNs


def mlp(sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class SeedFillNet(nn.Module):
    def __init__(self, in_features: int = 10, n_types: int = 9, width: int = 256):
        super().__init__()
        w = width
        self.enc1 = mlp([in_features, w // 4, w // 2])     # local features
        self.enc2 = mlp([w // 2, w, w])                    # features that get pooled
        self.seg = mlp([w // 2 + w, w, w // 2, 1])         # per-facet membership
        self.cls = mlp([w, w // 2, n_types])               # per-patch surface type

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """x: (B, N, F)   mask: (B, N) with 1 for real facets, 0 for padding."""
        m = mask.unsqueeze(-1)
        local = self.enc1(x) * m                      # (B, N, w/2)
        pooled_in = self.enc2(local)                  # (B, N, w)
        pooled_in = pooled_in.masked_fill(m == 0, NEG)
        glob = pooled_in.max(dim=1).values            # (B, w)

        n = local.shape[1]
        joined = torch.cat([local, glob.unsqueeze(1).expand(-1, n, -1)], dim=-1)
        same_logit = self.seg(joined).squeeze(-1)     # (B, N)
        type_logit = self.cls(glob)                   # (B, n_types)
        return same_logit, type_logit


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def export_onnx(model: nn.Module, path: str, n_facets: int = 512,
                in_features: int = 10) -> None:
    """Export with a dynamic batch and facet count.

    Facet count is dynamic because real patches are ragged and we would rather not pad
    every browser-side call to the training width. Verify the exported graph actually
    runs in onnxruntime-web before trusting the latency budget - a model that exports
    is not automatically a model that runs fast in WASM.
    """
    model.eval()
    x = torch.randn(1, n_facets, in_features)
    m = torch.ones(1, n_facets)
    torch.onnx.export(
        model, (x, m), path,
        input_names=["features", "mask"],
        output_names=["same_logit", "type_logit"],
        dynamic_axes={"features": {0: "batch", 1: "facets"},
                      "mask": {0: "batch", 1: "facets"},
                      "same_logit": {0: "batch", 1: "facets"},
                      "type_logit": {0: "batch"}},
        opset_version=17,
    )


if __name__ == "__main__":
    net = SeedFillNet()
    print(f"parameters: {count_parameters(net):,}  (~{count_parameters(net) * 4 / 1e6:.2f} MB fp32)")
    x = torch.randn(4, 512, 10)
    m = torch.ones(4, 512)
    m[0, 300:] = 0
    a, b = net(x, m)
    print("same_logit", tuple(a.shape), "type_logit", tuple(b.shape))
