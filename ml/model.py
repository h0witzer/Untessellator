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

import math

import torch
import torch.nn as nn

NEG =-1e4   # finite, so it survives bf16/fp16 autocast; -inf produces NaNs


class MeshConv(nn.Module):
    """One round of message passing over the patch's facet adjacency.

    A facet in a manifold mesh has at most three neighbours, so the patch's graph is a fixed
    (B, N, 3) index tensor: a gather, an MLP on (own, neighbour - own, dihedral), a masked max
    over the three, and a residual. No k-NN anywhere, so it exports as a plain Gather and runs in
    the browser like the rest. Each layer lets a facet see one ring further, and unlike the
    patch-wide max-pool it can carry PATH information: whether the way back to the seed crosses
    something that looks like a boundary.
    """

    AGG = ("max", "gate", "attn")
    HEADS = 4

    def __init__(self, width: int, edge_dim: int = 1, agg: str = "max"):
        """`edge_dim` edge attributes per neighbour (1 = the dihedral, as v7gnn was trained).
        `agg` chooses how the three messages combine:
          max   elementwise max (the original layer; parameter shapes unchanged)
          gate  per-channel sigmoid gates from the same input, then a gate-weighted mean
                (GatedGCN style): a neighbour across a border can be shut out channel by channel
          attn  GAT-style softmax over the neighbours, HEADS heads, each weighting its slice
        """
        super().__init__()
        if agg not in self.AGG:
            raise ValueError(f"agg must be one of {self.AGG}")
        self.edge_dim, self.agg = edge_dim, agg
        din = 2 * width + edge_dim
        self.msg = mlp([din, width, width])
        self.norm = nn.LayerNorm(width)
        if agg == "gate":
            self.gate = nn.Linear(din, width)
        elif agg == "attn":
            self.score = mlp([din, width // 2, self.HEADS])

    def forward(self, h: torch.Tensor, nbr: torch.Tensor, emask: torch.Tensor, eattr: torch.Tensor):
        agg, _ = self.aggregate(h, nbr, emask, eattr)
        return self.norm(h + torch.relu(agg))

    def aggregate(self, h: torch.Tensor, nbr: torch.Tensor, emask: torch.Tensor, eattr: torch.Tensor):
        """The combined message per facet, before the residual, and the message input z =
        [h_i, h_j - h_i, edge attributes] (ml/patch_eval.py reads both)."""
        b, n, w = h.shape
        idx = nbr.reshape(b, n * 3, 1).expand(b, n * 3, w)
        hj = torch.gather(h, 1, idx).reshape(b, n, 3, w)
        hi = h.unsqueeze(2).expand(b, n, 3, w)
        e = eattr.unsqueeze(-1) if eattr.dim() == 3 else eattr          # (B, N, 3, E)
        z = torch.cat([hi, hj - hi, e], dim=-1)
        m = self.msg(z)
        em = emask.unsqueeze(-1)
        if self.agg == "gate":
            g = torch.sigmoid(self.gate(z)) * em
            agg = (g * m).sum(2) / (g.sum(2) + 1e-6)
        elif self.agg == "attn":
            s = self.score(z).masked_fill(em == 0, NEG)                  # (B, N, 3, H)
            a = torch.softmax(s, dim=2) * em
            agg = (a.unsqueeze(-1) * m.reshape(b, n, 3, self.HEADS, w // self.HEADS)).sum(2).reshape(b, n, w)
        else:
            agg = m.masked_fill(em == 0, NEG).max(dim=2).values
        return agg.masked_fill(emask.sum(-1, keepdim=True) == 0, 0.0), z


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
                 surface: bool = False, n_surface: int = 5, gnn_layers: int = 0,
                 edge_dim: int = 1, agg: str = "max", seed_concat: bool = False,
                 seed_attn: int = 0, facet_surface: bool = False, contrastive: int = 0):
        super().__init__()
        w = width
        self.enc1 = mlp([in_features, w // 4, w // 2])     # local features
        # Optional message passing over the facet adjacency (MeshConv), between the per-facet
        # encoder and the pool. 0 keeps the original PointNet and loads old checkpoints unchanged.
        self.gnn_layers = gnn_layers
        self.convs = nn.ModuleList([MeshConv(w // 2, edge_dim, agg) for _ in range(gnn_layers)])
        self.enc2 = mlp([w // 2, w, w])                    # features that get pooled
        # SEED COMPARISON (2026-10-03). The membership head otherwise sees each facet and the pooled
        # patch, never the seed's own embedding; it knows the seed only through seed-relative inputs.
        #   seed_concat  append h_seed and h_i - h_seed to every facet's head input
        #   seed_attn    H heads of cross-attention with the seed as the query: a seed-focused
        #                context vector, and each facet's H raw scores, appended to the head input
        self.seed_concat, self.seed_attn = seed_concat, seed_attn
        seg_in = w // 2 + w
        if seed_concat:
            seg_in += w
        if seed_attn:
            d = w // 2
            self.q, self.k, self.v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
            seg_in += d + seed_attn
        self.seg = mlp([seg_in, w, w // 2, 1])             # per-facet membership
        self.cls = mlp([w, w // 2, n_types])               # per-patch surface type
        # Optional surface hypothesis under the seed (surface_targets.py). It reads the seed
        # facet's own local embedding as well as the pooled descriptor, because the targets
        # are defined AT the seed. Off by default, so existing checkpoints load unchanged.
        self.surface = surface
        if surface:
            self.surf = mlp([w // 2 + w, w, w // 2, n_surface])
        # TRAINING-ONLY SURFACE SPACE (2026-10-03). A shallow head on every facet's own embedding
        # regresses ITS face's surface (axis, asinh kn, asinh ks) and type. It shapes the encoder
        # during training and is never exported: forward returns it only when emit_aux is set.
        self.facet_surface = facet_surface
        self.emit_aux = False
        if facet_surface:
            self.fsurf = mlp([w // 2, w // 2, n_surface + n_types])
        # TRAINING-ONLY CROSS-MESHER SPACE (experiment 4). A projection of each facet's embedding in
        # which the same face meshed by another mesher lands together (train.py --pair-bakes). Never
        # exported; keep_local makes forward() keep the per-facet embeddings for the loss.
        self.contrastive = contrastive
        self.keep_local = False
        self.last_local = None
        if contrastive:
            self.cproj = mlp([w // 2, w // 2, contrastive])

    def embed(self, x: torch.Tensor, mask: torch.Tensor, nbr: torch.Tensor | None = None,
              eattr: torch.Tensor | None = None) -> torch.Tensor:
        """The per-facet embeddings the heads read (after enc1 and the MeshConv layers)."""
        m = mask.unsqueeze(-1)
        local = self.enc1(x) * m
        if self.gnn_layers:
            emask = ((nbr >= 0) & (mask.unsqueeze(-1) > 0)).to(local.dtype)
            safe = torch.where(nbr >= 0, nbr, torch.zeros_like(nbr))
            for conv in self.convs:
                local = conv(local, safe, emask, eattr) * m
        return local

    def forward(self, x: torch.Tensor, mask: torch.Tensor, nbr: torch.Tensor | None = None,
                eattr: torch.Tensor | None = None):
        """x: (B, N, F)   mask: (B, N) with 1 for real facets, 0 for padding.
        nbr: (B, N, 3) int64 patch-local neighbour indices, -1 where there is none; eattr: (B, N, 3)
        the dihedral of each of those edges / 90, or (B, N, 3, E) with edge_dim > 1. Only models
        with gnn_layers read them."""
        m = mask.unsqueeze(-1)
        local = self.embed(x, mask, nbr, eattr)       # (B, N, w/2)
        if self.keep_local:
            self.last_local = local
        pooled_in = self.enc2(local)                  # (B, N, w)
        pooled_in = pooled_in.masked_fill(m == 0, NEG)
        glob = pooled_in.max(dim=1).values            # (B, w)

        n = local.shape[1]
        parts = [local, glob.unsqueeze(1).expand(-1, n, -1)]
        # the seed is facet 0 of every patch: k_ring starts from it
        seed = local[:, :1]
        if self.seed_concat:
            parts += [seed.expand(-1, n, -1), (local - seed) * m]
        if self.seed_attn:
            hh = self.seed_attn
            b, _, d = local.shape
            q = self.q(local[:, 0]).reshape(b, 1, hh, d // hh)
            k = self.k(local).reshape(b, n, hh, d // hh)
            v = self.v(local).reshape(b, n, hh, d // hh)
            sc = (k * q).sum(-1) / math.sqrt(d // hh)                  # (B, N, H)
            att = torch.softmax(sc.masked_fill(m == 0, NEG), dim=1)
            ctx = (att.unsqueeze(-1) * v).sum(1).reshape(b, 1, d)
            parts += [ctx.expand(-1, n, -1), sc * m]
        same_logit = self.seg(torch.cat(parts, dim=-1)).squeeze(-1)    # (B, N)
        type_logit = self.cls(glob)                   # (B, n_types)
        out = (same_logit, type_logit)
        if self.surface:
            out += (self.surf(torch.cat([local[:, 0], glob], dim=-1)),)
        if self.facet_surface and self.emit_aux:
            out += (self.fsurf(local),)
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_from_args(args: dict, n_types: int = 9) -> SeedFillNet:
    """The network a train.py / refine_train.py run trained, from its stored arguments. Every
    option defaults to off, so checkpoints written before an option existed load unchanged."""
    a = args if isinstance(args, dict) else vars(args)
    return SeedFillNet(in_features=int(a.get("n_features", a.get("ncol", 10))), n_types=n_types,
                       surface=a.get("surface_root") is not None,
                       gnn_layers=int(a.get("gnn_layers", 0) or 0),
                       edge_dim=int(a.get("edge_dim", 1) or 1), agg=a.get("agg", "max") or "max",
                       seed_concat=bool(a.get("seed_concat", False)),
                       seed_attn=int(a.get("seed_attn", 0) or 0),
                       facet_surface=bool(a.get("facet_surface", False)),
                       contrastive=int(a.get("contrastive_dim", 0) or 0) if a.get("pair_bakes") else 0)


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
    model.emit_aux = False          # training-only heads never reach the browser
    has_surface = getattr(model, "surface", False)
    gnn = getattr(model, "gnn_layers", 0)
    edge_dim = model.convs[0].edge_dim if gnn else 1
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
        inputs += (nbr, torch.rand(1, n_facets, 3) if edge_dim == 1 else torch.rand(1, n_facets, 3, edge_dim))
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
