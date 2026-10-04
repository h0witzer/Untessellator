#!/usr/bin/env python3
"""
train.py - train the seeded-fill proposer on baked ABC patches.

    python train.py --bake D:/abc-data/bake --out D:/abc-data/runs/v1

SPLIT IS BY MODEL, NOT BY PATCH

Every mesh yields dozens of overlapping patches, so splitting patches at random would
put near-identical neighbourhoods in both train and validation and report a score that
means nothing. Files are partitioned by model id before any patch is drawn.

WHAT IS REPORTED, AND WHY NOT ACCURACY

Membership is class-imbalanced (measured ~37% positive), so accuracy is close to
useless - predicting "everything belongs" already scores 37%. The headline number is
IoU between predicted and true membership, with precision and recall alongside, because
the two failure modes are not equivalent: under-reach leaves the user to click again,
while over-reach silently poisons the surface fit downstream, which is exactly the
`paintFrom` bug this model exists to fix. Type accuracy is reported PER CLASS, since a
model that never predicts Revolution would still post a fine average.

LOSS WEIGHTING IS MEASURED, NOT GUESSED

`--pos-weight auto` samples real patches and sets the binary weight from the observed
positive rate. Type weighting is inverse-frequency over the same sample. Both are
printed so the numbers behind a run are recoverable from its log.
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
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import build_from_args, count_parameters  # noqa: E402
from occ_tessellate import TYPE_NAMES  # noqa: E402
from patches import (BakedPatches, FEATURE_NAMES, FEATURE_NAMES_V6, N_TYPES, collate,  # noqa: E402
                     collate_facet_surface, collate_graph, collate_pairs, collate_surface,
                     collate_weights)


class PatchStream(IterableDataset):
    def __init__(self, files: list[Path], max_facets: int, seeds_per_mesh: int,
                 densities: tuple[str, ...], type_weights=None, seed: int = 0,
                 relabel_root: Path | None = None, surface_root: Path | None = None,
                 extended: bool = False, exclude_prob: float = 0.0, gnn: bool = False,
                 hard_rings: int = 0, boundary_seed_prob: float = 0.0, facet_surface: bool = False,
                 pair_roots=None, pair_prob: float = 0.0, native_root=None):
        self.facet_surface = facet_surface
        self.pair_roots, self.pair_prob, self.native_root = pair_roots, pair_prob, native_root
        self.gnn = gnn
        self.hard_rings = hard_rings
        self.boundary_seed_prob = boundary_seed_prob
        self.extended = extended
        self.exclude_prob = exclude_prob
        self.relabel_root = relabel_root
        self.surface_root = surface_root
        self.files = files
        self.max_facets = max_facets
        self.seeds_per_mesh = seeds_per_mesh
        self.densities = densities
        self.type_weights = type_weights
        self.seed = seed

    def __iter__(self):
        info = get_worker_info()
        wid, nw = (0, 1) if info is None else (info.id, info.num_workers)
        mine = self.files[wid::nw]                     # shard files across workers
        ds = BakedPatches(root=Path("."), densities=self.densities,
                          max_facets=self.max_facets, seeds_per_mesh=self.seeds_per_mesh,
                          type_weights=self.type_weights, seed=self.seed + wid * 977,
                          files=mine, relabel_root=self.relabel_root,
                          surface_root=self.surface_root, extended=self.extended,
                          exclude_prob=self.exclude_prob, gnn=self.gnn,
                          hard_rings=getattr(self, "hard_rings", 0),
                          boundary_seed_prob=getattr(self, "boundary_seed_prob", 0.0),
                          facet_surface=getattr(self, "facet_surface", False),
                          pair_roots=getattr(self, "pair_roots", None),
                          pair_prob=getattr(self, "pair_prob", 0.0),
                          native_root=getattr(self, "native_root", None))
        for s in ds:
            yield s


class Collate:
    """Module-level and picklable on purpose.

    Windows DataLoader workers are spawned, not forked, so everything handed to them is
    pickled. A closure over `args` - the obvious way to write this - fails with
    "Can't pickle local object", and the traceback points at multiprocessing rather
    than at the collate function, so it is worth not rediscovering.
    """

    def __init__(self, max_facets: int, zero_cols: tuple[int, ...] = (), surface: bool = False,
                 gnn: bool = False, hard_weight: float = 1.0, facet_surface: bool = False,
                 pairs: bool = False):
        self.facet_surface = facet_surface
        self.pairs = pairs
        self.max_facets = max_facets
        self.zero_cols = list(zero_cols)
        self.surface = surface
        self.gnn = gnn
        self.hard_weight = hard_weight

    def __call__(self, batch):
        x, y, m, t = collate(batch, self.max_facets)
        if self.zero_cols:
            x[:, :, self.zero_cols] = 0.0
        out = (torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m), torch.from_numpy(t))
        if self.surface:
            s, k = collate_surface(batch)
            out += (torch.from_numpy(s), torch.from_numpy(k))
        # Layout: x y m t [surface target, mask] [facet target, mask, type] [loss weights] [nbr eattr]
        # - the graph stays last; batch_index() names the positions.
        if getattr(self, "facet_surface", False):
            out += tuple(torch.from_numpy(a) for a in collate_facet_surface(batch, self.max_facets))
        if getattr(self, "pairs", False):
            out += tuple(torch.from_numpy(a) for a in collate_pairs(batch, self.max_facets, self.zero_cols))
        if getattr(self, "hard_weight", 1.0) != 1.0:
            out += (torch.from_numpy(collate_weights(batch, self.max_facets, self.hard_weight)),)
        if self.gnn:
            nbr, ea = collate_graph(batch, self.max_facets)
            out += (torch.from_numpy(nbr), torch.from_numpy(ea))
        return out


def batch_index(args) -> dict:
    """Positions of the optional tensors in a collated TRAINING batch (Collate's layout)."""
    i, at = 4, {}
    if args.surface_root is not None:
        at["surface"], at["surface_mask"] = i, i + 1
        i += 2
    if getattr(args, "facet_surface", False):
        at["fsurf"], at["fsurf_mask"], at["ftype"] = i, i + 1, i + 2
        i += 3
    if getattr(args, "pair_bakes", None):
        at["pairs"] = i                      # 7 tensors: index, x, mask, nbr, eattr, main labels, partner labels
        i += 7
    if args.hard_weight != 1.0:
        at["weight"] = i
    return at


def contrastive_loss(net, batch, at, local, extra_graph, tau: float):
    """Multi-positive InfoNCE between each paired main patch and its partner from another mesher:
    a facet's positives are the partner's facets on the same face. Both directions; facets with no
    positive are skipped. Returns 0 when the batch carries no pairs."""
    j = at["pairs"]
    idx, xp, mp, nbp, eap, fa, fb = batch[j:j + 7]
    if idx.numel() == 0:
        return local.new_zeros(())
    lp = net.embed(xp, mp, nbp, eap) if extra_graph else net.embed(xp, mp)
    za = F.normalize(net.cproj(local[idx]).float(), dim=-1)
    zb = F.normalize(net.cproj(lp).float(), dim=-1)
    s = torch.einsum("pad,pbd->pab", za, zb) / tau
    va, vb = fa >= 0, fb >= 0
    pos = (fa.unsqueeze(2) == fb.unsqueeze(1)) & va.unsqueeze(2) & vb.unsqueeze(1)

    def side(sim, pos, valid_q, valid_k):
        sim = sim.masked_fill(~valid_k.unsqueeze(1), -1e4)
        lse_all = torch.logsumexp(sim, dim=-1)
        lse_pos = torch.logsumexp(sim.masked_fill(~pos, -1e4), dim=-1)
        use = valid_q & pos.any(-1)
        return ((lse_all - lse_pos) * use).sum() / use.sum().clamp(min=1)

    return 0.5 * (side(s, pos, va, vb) + side(s.transpose(1, 2), pos.transpose(1, 2), vb, va))


def facet_surface_loss(pred, tgt, msk, ftype, m, cls_w):
    """The training-only per-facet surface space: each facet's own face's axis (sign-free),
    asinh kn, asinh ks, and type."""
    pred = pred.float()
    ax = F.normalize(pred[..., :3], dim=-1)
    ma, mk, ms = msk[..., 0] * m, msk[..., 1] * m, msk[..., 2] * m
    la = ((1 - (ax * tgt[..., :3]).sum(-1) ** 2) * ma).sum() / ma.sum().clamp(min=1)
    lk = (F.smooth_l1_loss(pred[..., 3], tgt[..., 3], reduction="none") * mk).sum() / mk.sum().clamp(min=1)
    ls = (F.smooth_l1_loss(pred[..., 4], tgt[..., 4], reduction="none") * ms).sum() / ms.sum().clamp(min=1)
    nt = pred.shape[-1] - 5
    lt = F.cross_entropy(pred[..., 5:].reshape(-1, nt), ftype.reshape(-1), weight=cls_w, ignore_index=-100)
    return la + lk + ls + 0.5 * lt


def surface_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    """Axis (sign-free: 1 - cos^2), asinh(kn) and asinh(ks), each averaged over the patches
    that carry that target."""
    pred = pred.float()
    ax = F.normalize(pred[:, :3], dim=1)
    la = ((1 - (ax * target[:, :3]).sum(1) ** 2) * mask[:, 0]).sum() / mask[:, 0].sum().clamp(min=1)
    lk = (F.smooth_l1_loss(pred[:, 3], target[:, 3], reduction="none") * mask[:, 1]).sum() \
        / mask[:, 1].sum().clamp(min=1)
    ls = (F.smooth_l1_loss(pred[:, 4], target[:, 4], reduction="none") * mask[:, 2]).sum() \
        / mask[:, 2].sum().clamp(min=1)
    return la + lk + ls


def make_loader(files, args, type_weights, shuffle_seed, exclude_prob=0.0, training=False):
    # Validation keeps the plain seed distribution but still marks the band, so the log can report
    # how the membership head does exactly where fills go wrong.
    hard = args.hard_weight != 1.0
    stream = PatchStream(files, args.max_facets, args.seeds_per_mesh,
                         tuple(args.densities.split(",")), type_weights, shuffle_seed,
                         args.relabel, args.surface_root, args.features == "v6", exclude_prob,
                         args.gnn_layers > 0, args.hard_rings if hard else 0,
                         args.boundary_seeds if training else 0.0,
                         facet_surface=args.facet_surface and training,
                         pair_roots=args.pair_bakes if training else None,
                         pair_prob=args.pair_prob if training else 0.0, native_root=args.bake)
    return DataLoader(stream, batch_size=args.batch,
                      collate_fn=Collate(args.max_facets, args.zero_cols, args.surface_root is not None,
                                         args.gnn_layers > 0, args.hard_weight,
                                         facet_surface=args.facet_surface and training,
                                         pairs=bool(args.pair_bakes) and training),
                      num_workers=args.workers, persistent_workers=args.workers > 0,
                      prefetch_factor=4 if args.workers > 0 else None)


def measure_stats(files, args, n: int = 800):
    """Observe the real positive rate and type frequencies before training."""
    ds = BakedPatches(root=Path("."), densities=tuple(args.densities.split(",")),
                      max_facets=args.max_facets, seeds_per_mesh=8, seed=12345,
                      files=files[: max(1, min(len(files), 400))], relabel_root=args.relabel)
    pos = tot = 0
    types = np.zeros(N_TYPES, np.int64)
    k = 0
    for s in ds:
        pos += float(s["same"].sum())
        tot += s["n"]
        types[s["type"]] += 1
        k += 1
        if k >= n:
            break
    rate = pos / max(tot, 1)
    return rate, types, k


def graph_inputs(batch, model):
    """The (nbr, eattr) pair a message-passing model takes: always the last two batch tensors."""
    return tuple(batch[-2:]) if getattr(model, "gnn_layers", 0) else ()


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int, weight_index: int | None = None):
    model.eval()
    model.emit_aux = False
    tp = fp = fn_ = 0.0
    btp = bfp = bfn = 0.0          # the same, inside the smooth-boundary band only
    type_hit = np.zeros(N_TYPES, np.int64)
    type_tot = np.zeros(N_TYPES, np.int64)
    axis_err, kn_sign, kn_err = [], [], []
    seen = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x, y, m, t = (b.to(device) for b in batch[:4])
        extra = tuple(b.to(device) for b in graph_inputs(batch, model))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outs = model(x, m, *extra)
        sl, tl = outs[0], outs[1]
        if len(outs) == 3:
            sp = outs[2].float().cpu()
            st, sk = batch[4], batch[5]
            a = F.normalize(sp[:, :3], dim=1)
            sel = sk[:, 0] > 0
            if sel.any():
                cos = (a[sel] * st[sel, :3]).sum(1).abs().clamp(max=1.0)
                axis_err.extend(torch.rad2deg(torch.acos(cos)).tolist())
            sel = sk[:, 1] > 0
            if sel.any():
                kn_err.extend((sp[sel, 3] - st[sel, 3]).abs().tolist())
                curved = sel & (st[:, 3].abs() > 1e-6)
                kn_sign.extend((torch.sign(sp[curved, 3]) == torch.sign(st[curved, 3])).float().tolist())
        pred = ((sl.float().sigmoid() > 0.5).float()) * m
        truth = y * m
        tp += float((pred * truth).sum())
        fp += float((pred * (1 - truth) * m).sum())
        fn_ += float(((1 - pred) * truth).sum())
        if weight_index is not None:
            band = (batch[weight_index].to(device) > 1.0).float() * m
            btp += float((pred * truth * band).sum())
            bfp += float((pred * (1 - truth) * band).sum())
            bfn += float(((1 - pred) * truth * band).sum())
        pt = tl.float().argmax(1)
        for c in range(N_TYPES):
            sel = t == c
            if sel.any():
                type_tot[c] += int(sel.sum())
                type_hit[c] += int((pt[sel] == c).sum())
        seen += 1
    iou = tp / max(tp + fp + fn_, 1e-9)
    prec = tp / max(tp + fp, 1e-9)
    rec = tp / max(tp + fn_, 1e-9)
    model.train()
    med = lambda v: float(np.median(v)) if v else None
    band = None
    if weight_index is not None:
        band = {"iou": btp / max(btp + bfp + bfn, 1e-9), "precision": btp / max(btp + bfp, 1e-9),
                "recall": btp / max(btp + bfn, 1e-9)}
    return {"iou": iou, "precision": prec, "recall": rec, "band": band,
            "type_hit": type_hit, "type_tot": type_tot, "batches": seen,
            "axis_err_p50": med(axis_err), "kn_asinh_err_p50": med(kn_err),
            "kn_sign_acc": float(np.mean(kn_sign)) if kn_sign else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bake", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--densities", default="coarse,medium,fine")
    ap.add_argument("--max-facets", type=int, default=512)
    ap.add_argument("--seeds-per-mesh", type=int, default=24)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-batches", type=int, default=60)
    ap.add_argument("--val-frac", type=float, default=0.08)
    ap.add_argument("--pos-weight", default="auto")
    ap.add_argument("--type-loss-weight", type=float, default=0.5)
    ap.add_argument("--balance-seeds", action="store_true",
                    help="bias seed sampling towards rare surface types")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--relabel", type=Path, default=None,
                    help="relabel sidecar root: train and validate on canonical surface types")
    ap.add_argument("--synth", type=Path, default=None,
                    help="synthetic bake to mix into TRAINING only; validation stays real")
    ap.add_argument("--synth-frac", type=float, default=0.15,
                    help="fraction of training files drawn from --synth")
    ap.add_argument("--extra-bake", type=Path, nargs="*", default=[],
                    help="re-meshed bakes of the same parts (mesher_bake.py); their copies of "
                         "TRAINING-split parts are added to training, never validation")
    ap.add_argument("--extra-frac", type=float, default=0.5,
                    help="fraction of training files drawn from --extra-bake, all dirs together")
    ap.add_argument("--no-curvature", action="store_true",
                    help="zero the two curvature columns. The control arm: any gain from "
                         "curvature has to be measured against an otherwise identical run, "
                         "not against a previous run with different hyperparameters.")
    ap.add_argument("--surface-root", type=Path, default=None,
                    help="surface_sidecar.py output: adds the surface-hypothesis head and its loss")
    ap.add_argument("--surface-loss-weight", type=float, default=1.0)
    ap.add_argument("--features", choices=("base", "v6"), default="base",
                    help="base: the 12 inputs of patches.FEATURE_NAMES; v6: plus the 13 extended "
                         "columns of seedfeatures.py (checked against js/seedFeatures.js by "
                         "test/netparity.mjs)")
    ap.add_argument("--exclude-prob", type=float, default=0.0,
                    help="training patches drawn with some neighbouring faces already painted "
                         "(removed from the patch); validation always uses full patches")
    ap.add_argument("--hard-weight", type=float, default=1.0,
                    help="loss weight for facets within --hard-rings of a smooth (<30 deg) boundary "
                         "of the seed's face, against 1 elsewhere")
    ap.add_argument("--hard-rings", type=int, default=2)
    ap.add_argument("--boundary-seeds", type=float, default=0.0,
                    help="share of training seeds redrawn from facets on a smooth face boundary")
    ap.add_argument("--gnn-layers", type=int, default=0,
                    help="message-passing layers over the patch's facet adjacency (model.MeshConv) "
                         "between the per-facet encoder and the pool; 0 is the PointNet")
    ap.add_argument("--edge-dim", type=int, default=1, help=argparse.SUPPRESS)
    ap.add_argument("--agg", choices=("max", "gate", "attn"), default="max",
                    help="how MeshConv combines a facet's three messages (model.MeshConv)")
    ap.add_argument("--seed-concat", action="store_true",
                    help="give the membership head the seed's embedding and h_i - h_seed")
    ap.add_argument("--seed-attn", type=int, default=0,
                    help="heads of seed-query cross-attention feeding the membership head; 0 = off")
    ap.add_argument("--facet-surface", action="store_true",
                    help="training-only head: every facet regresses its own face's surface and type "
                         "(needs --surface-root); never exported")
    ap.add_argument("--facet-loss-weight", type=float, default=0.5)
    ap.add_argument("--pair-bakes", type=Path, nargs="*", default=[],
                    help="re-meshed bakes to pair native patches with (experiment 4): a training-only "
                         "contrastive space where the same face from two meshers lands together")
    ap.add_argument("--pair-prob", type=float, default=0.5)
    ap.add_argument("--contrastive-dim", type=int, default=64)
    ap.add_argument("--contrastive-weight", type=float, default=0.1)
    ap.add_argument("--contrastive-tau", type=float, default=0.1)
    ap.add_argument("--ema", type=float, default=0.0,
                    help="also keep an exponential moving average of the weights with this decay, "
                         "evaluated and saved separately (best_ema.pt); the trained weights are untouched")
    ap.add_argument("--zero-features", default="",
                    help="comma list of patches.FEATURE_NAMES to hold at zero, e.g. "
                         "log_area,ring: inputs that describe the triangulation rather than "
                         "the surface. The columns are stored in the checkpoint as zero_cols; "
                         "rescore.py and export_onnx.py zero the same ones.")
    args = ap.parse_args(argv)
    zero = [n for n in args.zero_features.split(",") if n]
    if args.no_curvature:
        zero += ["k1s", "k2s"]
    names = FEATURE_NAMES_V6 if args.features == "v6" else FEATURE_NAMES
    unknown = sorted(set(zero) - set(names))
    if unknown:
        ap.error(f"unknown features {unknown}; choose from {', '.join(names)}")
    args.zero_cols = tuple(sorted({names.index(n) for n in zero}))
    args.n_features = len(names)

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    files = sorted(args.bake.glob("*/*.npz"))
    if not files:
        print(f"no baked files under {args.bake}", file=sys.stderr)
        return 2
    rng = np.random.default_rng(args.seed)
    rng.shuffle(files)
    nval = max(1, int(len(files) * args.val_frac))
    val_files, train_files = files[:nval], files[nval:]

    # Synthetic parts join TRAINING only. Validation stays real ABC with canonical labels,
    # because the question is always how the model does on real parts; a validation score
    # inflated by synthetic parts it was built to handle would answer nothing.
    n_synth = 0
    if args.synth is not None and args.synth_frac > 0:
        synth = sorted(args.synth.glob("*/*.npz"))
        rng.shuffle(synth)
        want = int(len(train_files) * args.synth_frac / max(1e-9, 1 - args.synth_frac))
        synth = synth[:want]
        n_synth = len(synth)
        train_files = list(train_files) + list(synth)
        rng.shuffle(train_files)

    # Re-meshed copies of the same TRAINING parts (NURBS-converted, gmsh, ...). The model
    # measurably reads OCCT's triangulation pattern, so showing it one part meshed several
    # ways is the point. Only parts from this run's own training split are taken, so no
    # validation part reaches training under another mesher; relabel sidecars apply as usual.
    n_extra = {}
    if args.extra_bake and args.extra_frac > 0:
        real_train = [f for f in train_files if args.bake in f.parents]
        pool = []
        for extra in args.extra_bake:
            got = [extra / f.relative_to(args.bake) for f in real_train]
            got = [g for g in got if g.exists()]
            n_extra[str(extra)] = len(got)
            pool.extend(got)
        rng.shuffle(pool)
        want = int(len(train_files) * args.extra_frac / max(1e-9, 1 - args.extra_frac))
        pool = pool[:want]
        n_extra = {k: sum(1 for g in pool if str(g).startswith(k)) for k in n_extra}
        train_files = list(train_files) + pool
        rng.shuffle(train_files)
    print(f"device      {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"baked files {len(files):,}  ->  train {len(train_files):,}  val {len(val_files):,}"
          "   (split by model, never by patch)")
    print(f"labels      {'canonical (relabel sidecars)' if args.relabel else 'stored B-rep types'}")
    if n_extra:
        print("re-meshed   " + ", ".join(f"{Path(k).name} {v:,}" for k, v in n_extra.items())
              + f" training copies ({100 * sum(n_extra.values()) / len(train_files):.1f}% of "
              "training files); validation is native only")
    if n_synth:
        print(f"synthetic   {n_synth:,} parts mixed into training "
              f"({100 * n_synth / len(train_files):.1f}% of training files); validation is real-only")

    rate, types, nsamp = measure_stats(train_files, args)
    pw = (1 - rate) / max(rate, 1e-6) if args.pos_weight == "auto" else float(args.pos_weight)
    print(f"measured over {nsamp} patches: positive rate {rate:.3f} -> pos_weight {pw:.2f}")
    seen_types = {TYPE_NAMES[i]: int(c) for i, c in enumerate(types) if c}
    print(f"seed types  {seen_types}")

    tw = None
    if args.balance_seeds:
        inv = 1.0 / np.maximum(types, 1)
        tw = (inv / inv.sum()).astype(np.float64)
        print(f"seed sampling biased to rare types: "
              f"{ {TYPE_NAMES[i]: round(float(v), 4) for i, v in enumerate(tw)} }")
    cls_w = torch.tensor((1.0 / np.maximum(types, 1)) / (1.0 / np.maximum(types, 1)).sum()
                         * N_TYPES, dtype=torch.float32, device=device)

    if args.facet_surface and args.surface_root is None:
        print("--facet-surface needs --surface-root", file=sys.stderr)
        return 2
    model = build_from_args(vars(args), N_TYPES).to(device)
    model.emit_aux = args.facet_surface
    model.keep_local = bool(args.pair_bakes)
    print(f"pairs       {'off' if not args.pair_bakes else f'p={args.pair_prob} with ' + ', '.join(Path(b).name for b in args.pair_bakes) + f' (dim {args.contrastive_dim}, weight {args.contrastive_weight}, tau {args.contrastive_tau})'}")
    print(f"variants    agg {args.agg}, seed concat {args.seed_concat}, seed attention heads "
          f"{args.seed_attn}, per-facet surface head {args.facet_surface} "
          f"(weight {args.facet_loss_weight}), EMA {args.ema or 'off'}")
    ema = ema_params = model_params = None
    if args.ema > 0:
        import copy
        ema = copy.deepcopy(model).eval()
        for p in ema.parameters():
            p.requires_grad_(False)
        ema_params, model_params = list(ema.parameters()), list(model.parameters())
    at = batch_index(args)
    print(f"graph       {args.gnn_layers} message-passing layers" if args.gnn_layers else "graph       none (PointNet)")
    print(f"hard band   loss weight {args.hard_weight} within {args.hard_rings} rings of a smooth boundary; "
          f"boundary seeds {args.boundary_seeds:.0%}")
    print(f"surface     {'head on, sidecars ' + str(args.surface_root) if args.surface_root else 'off'}")
    print(f"curvature   {'DISABLED (control arm)' if args.no_curvature else 'enabled'}")
    print(f"inputs      {args.features} ({args.n_features} columns); painted-neighbour "
          f"augmentation p={args.exclude_prob}")
    print(f"zeroed      {', '.join(names[i] for i in args.zero_cols) or 'none'}")
    print(f"parameters  {count_parameters(model):,} "
          f"(~{count_parameters(model) * 4 / 1e6:.2f} MB fp32)")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=args.steps, pct_start=0.05)
    pos_w = torch.tensor(pw, device=device)

    train_loader = make_loader(train_files, args, tw, args.seed, args.exclude_prob, training=True)
    val_loader = make_loader(val_files, args, None, args.seed + 1)

    best = best_ema = -1.0
    run_con = 0.0
    step = 0
    t0 = time.time()
    run_loss = run_seg = run_cls = 0.0
    log_path = args.out / "train_log.jsonl"
    logf = log_path.open("a", encoding="utf-8")
    print(f"\ntraining for {args.steps:,} steps, batch {args.batch}, "
          f"{args.max_facets} facets/patch\n")

    done = False
    while not done:
        for batch in train_loader:
            batch = [b.to(device, non_blocking=True) for b in batch]
            x, y, m, t = batch[:4]
            wm = m
            if args.hard_weight != 1.0:
                wm = batch[at["weight"]]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                outs = model(x, m, *graph_inputs(batch, model))
                sl, tl = outs[0], outs[1]
                seg = F.binary_cross_entropy_with_logits(
                    sl.float(), y, weight=wm, pos_weight=pos_w, reduction="sum") / wm.sum().clamp(min=1)
                cls = F.cross_entropy(tl.float(), t, weight=cls_w)
                loss = seg + args.type_loss_weight * cls
                surf = torch.zeros((), device=device)
                if args.surface_root is not None:
                    surf = surface_loss(outs[2], batch[at["surface"]], batch[at["surface_mask"]])
                    loss = loss + args.surface_loss_weight * surf
                if args.facet_surface:
                    fl = facet_surface_loss(outs[-1], batch[at["fsurf"]], batch[at["fsurf_mask"]],
                                            batch[at["ftype"]], m, cls_w)
                    loss = loss + args.facet_loss_weight * fl
                if args.pair_bakes:
                    cl = contrastive_loss(model, batch, at, model.last_local, args.gnn_layers > 0,
                                          args.contrastive_tau)
                    loss = loss + args.contrastive_weight * cl
                    run_con = (run_con if step % 100 else 0.0) + float(cl)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if ema is not None:
                with torch.no_grad():
                    torch._foreach_mul_(ema_params, args.ema)
                    torch._foreach_add_(ema_params, model_params, alpha=1 - args.ema)

            run_loss += float(loss); run_seg += float(seg); run_cls += float(cls)
            run_surf = (run_surf if step % 100 else 0.0) + float(surf)
            step += 1

            if step % 100 == 0:
                rate_s = step / (time.time() - t0)
                print(f"  step {step:>6,}/{args.steps:,}  loss {run_loss / 100:.4f} "
                      f"(seg {run_seg / 100:.4f} cls {run_cls / 100:.4f} surf {run_surf / 100:.4f}"
                      + (f" pair {run_con / 100:.4f}" if args.pair_bakes else "") + ")  "
                      f"lr {sched.get_last_lr()[0]:.2e}  {rate_s:.1f} it/s", flush=True)
                run_loss = run_seg = run_cls = 0.0

            if step % args.eval_every == 0 or step == args.steps:
                vw = (6 if args.surface_root is not None else 4) if args.hard_weight != 1.0 else None
                ev = evaluate(model, val_loader, device, args.eval_batches, vw)
                model.emit_aux = args.facet_surface
                ev_ema = evaluate(ema, val_loader, device, args.eval_batches, vw) if ema is not None else None
                per = {TYPE_NAMES[i]: (round(float(ev["type_hit"][i] / ev["type_tot"][i]), 3)
                                       if ev["type_tot"][i] else None)
                       for i in range(N_TYPES)}
                macro = np.mean([v for v in per.values() if v is not None])
                print(f"\n  [eval @ {step:,}] membership IoU {ev['iou']:.4f}  "
                      f"precision {ev['precision']:.4f}  recall {ev['recall']:.4f}"
                      + (f"   EMA IoU {ev_ema['iou']:.4f}" if ev_ema else ""))
                print(f"              type macro-acc {macro:.3f}  {per}", flush=True)
                if ev["band"]:
                    print(f"              smooth-boundary band: IoU {ev['band']['iou']:.4f}  "
                          f"precision {ev['band']['precision']:.4f}  recall {ev['band']['recall']:.4f}", flush=True)
                if ev["axis_err_p50"] is not None or ev["kn_sign_acc"] is not None:
                    print(f"              surface: axis error p50 {ev['axis_err_p50']} deg  "
                          f"asinh(kn) error p50 {ev['kn_asinh_err_p50']}  kn sign {ev['kn_sign_acc']}",
                          flush=True)
                print("", flush=True)
                logf.write(json.dumps({"step": step, "iou": ev["iou"],
                                       "precision": ev["precision"], "recall": ev["recall"],
                                       "type_macro": float(macro), "per_type": per,
                                       "axis_err_p50": ev["axis_err_p50"],
                                       "kn_asinh_err_p50": ev["kn_asinh_err_p50"],
                                       "kn_sign_acc": ev["kn_sign_acc"], "band": ev["band"],
                                       "ema_iou": ev_ema["iou"] if ev_ema else None}) + "\n")
                logf.flush()
                if ev["iou"] > best:
                    best = ev["iou"]
                    torch.save({"model": model.state_dict(), "step": step,
                                "iou": best, "args": vars(args),
                                "pos_weight": pw, "type_names": list(TYPE_NAMES)},
                               args.out / "best.pt")
                if ev_ema and ev_ema["iou"] > best_ema:
                    best_ema = ev_ema["iou"]
                    torch.save({"model": ema.state_dict(), "step": step, "iou": best_ema, "args": vars(args),
                                "pos_weight": pw, "type_names": list(TYPE_NAMES), "ema": args.ema},
                               args.out / "best_ema.pt")

            if step >= args.steps:
                done = True
                break

    logf.close()
    print(f"\nbest membership IoU {best:.4f}  ->  {args.out / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
