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


class MeshConv(nn.Module):
    """One round of message passing over the patch's facet adjacency.

    A facet in a manifold mesh has at most three neighbours, so the patch's graph is a fixed
    (B, N, 3) index tensor: a gather, an MLP on (own, neighbour - own, dihedral), a masked max
    over the three, and a residual. No k-NN anywhere, so it exports as a plain Gather and runs in
    the browser like the rest. Each layer lets a facet see one ring further, and unlike the
    patch-wide max-pool it can carry PATH information: whether the way back to the seed crosses
    something that looks like a boundary.
    """

    def __init__(self, width: int):
        super().__init__()
        self.msg = mlp([2 * width + 1, width, width])
        self.norm = nn.LayerNorm(width)

    def forward(self, h: torch.Tensor, nbr: torch.Tensor, emask: torch.Tensor, eattr: torch.Tensor):
        b, n, w = h.shape
        idx = nbr.reshape(b, n * 3, 1).expand(b, n * 3, w)
        hj = torch.gather(h, 1, idx).reshape(b, n, 3, w)
        hi = h.unsqueeze(2).expand(b, n, 3, w)
        m = self.msg(torch.cat([hi, hj - hi, eattr.unsqueeze(-1)], dim=-1))
        m = m.masked_fill(emask.unsqueeze(-1) == 0, NEG)
        agg = m.max(dim=2).values
        agg = agg.masked_fill(emask.sum(-1, keepdim=True) == 0, 0.0)
        return self.norm(h + torch.relu(agg))


def mlp(sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class SeedFillNet(nn.Module):
    def __init__(self, in_features: int = 10, n_types: int = 9, width: int = 256,
                 surface: bool = False, n_surface: int = 5, gnn_layers: int = 0):
        super().__init__()
        w = width
        self.enc1 = mlp([in_features, w // 4, w // 2])     # local features
        # Optional message passing over the facet adjacency (MeshConv), between the per-facet
        # encoder and the pool. 0 keeps the original PointNet and loads old checkpoints unchanged.
        self.gnn_layers = gnn_layers
        self.convs = nn.ModuleList([MeshConv(w // 2) for _ in range(gnn_layers)])
        self.enc2 = mlp([w // 2, w, w])                    # features that get pooled
        self.seg = mlp([w // 2 + w, w, w // 2, 1])         # per-facet membership
        self.cls = mlp([w, w // 2, n_types])               # per-patch surface type
        # Optional surface hypothesis under the seed (surface_targets.py). It reads the seed
        # facet's own local embedding as well as the pooled descriptor, because the targets
        # are defined AT the seed. Off by default, so existing checkpoints load unchanged.
        self.surface = surface
        if surface:
            self.surf = mlp([w // 2 + w, w, w // 2, n_surface])

    def forward(self, x: torch.Tensor, mask: torch.Tensor, nbr: torch.Tensor | None = None,
                eattr: torch.Tensor | None = None):
        """x: (B, N, F)   mask: (B, N) with 1 for real facets, 0 for padding.
        nbr: (B, N, 3) int64 patch-local neighbour indices, -1 where there is none; eattr: (B, N, 3)
        the dihedral of each of those edges / 90. Only models with gnn_layers read them."""
        m = mask.unsqueeze(-1)
        local = self.enc1(x) * m                      # (B, N, w/2)
        if self.gnn_layers:
            emask = ((nbr >= 0) & (mask.unsqueeze(-1) > 0)).to(local.dtype)
            safe = torch.where(nbr >= 0, nbr, torch.zeros_like(nbr))
            for conv in self.convs:
                local = conv(local, safe, emask, eattr) * m
        pooled_in = self.enc2(local)                  # (B, N, w)
        pooled_in = pooled_in.masked_fill(m == 0, NEG)
        glob = pooled_in.max(dim=1).values            # (B, w)

        n = local.shape[1]
        joined = torch.cat([local, glob.unsqueeze(1).expand(-1, n, -1)], dim=-1)
        same_logit = self.seg(joined).squeeze(-1)     # (B, N)
        type_logit = self.cls(glob)                   # (B, n_types)
        if self.surface:
            # the seed is facet 0 of every patch: k_ring starts from it
            surface = self.surf(torch.cat([local[:, 0], glob], dim=-1))
            return same_logit, type_logit, surface
        return same_logit, type_logit


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ZeroInputs(nn.Module):
    """Hold chosen feature columns at zero, as the first operation of the graph."""

    def __init__(self, inner: nn.Module, in_features: int, zero_cols):
        super().__init__()
        self.inner = inner
        keep = torch.ones(in_features)
        keep[list(zero_cols)] = 0.0
        self.register_buffer("keep", keep)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, *rest):
        return self.inner(x * self.keep, mask, *rest)


def export_onnx(model: nn.Module, path: str, n_facets: int = 512,
                in_features: int = 10, zero_cols=()) -> None:
    """Export with a dynamic batch and facet count.

    Facet count is dynamic because real patches are ragged and we would rather not pad
    every browser-side call to the training width. Verify the exported graph actually
    runs in onnxruntime-web before trusting the latency budget - a model that exports
    is not automatically a model that runs fast in WASM.

    `zero_cols` are columns the model was trained with at zero. They are zeroed inside
    the exported graph, so a browser caller cannot feed them by mistake.
    """
    model.eval()
    has_surface = getattr(model, "surface", False)
    gnn = getattr(model, "gnn_layers", 0)
    if zero_cols:
        model = ZeroInputs(model, in_features, zero_cols).eval()
    x = torch.randn(1, n_facets, in_features)
    m = torch.ones(1, n_facets)
    names = ["same_logit", "type_logit"] + (["surface"] if has_surface else [])
    axes = {"features": {0: "batch", 1: "facets"},
            "mask": {0: "batch", 1: "facets"},
            "same_logit": {0: "batch", 1: "facets"},
            "type_logit": {0: "batch"}}
    if has_surface:
        axes["surface"] = {0: "batch"}
    inputs, input_names = (x, m), ["features", "mask"]
    if gnn:
        # a chain, so every gather index is valid; the real graph comes from the caller
        nbr = torch.full((1, n_facets, 3), -1, dtype=torch.int64)
        nbr[0, 1:, 0] = torch.arange(n_facets - 1)
        nbr[0, :-1, 1] = torch.arange(1, n_facets)
        inputs += (nbr, torch.rand(1, n_facets, 3))
        input_names += ["nbr", "eattr"]
        axes["nbr"] = {0: "batch", 1: "facets"}
        axes["eattr"] = {0: "batch", 1: "facets"}
    torch.onnx.export(
        model, inputs, path,
        input_names=input_names,
        output_names=names,
        dynamic_axes=axes,
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
