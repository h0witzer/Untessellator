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
from model import SeedFillNet, count_parameters  # noqa: E402
from occ_tessellate import TYPE_NAMES  # noqa: E402
from patches import BakedPatches, N_FEATURES, N_TYPES, collate  # noqa: E402


class PatchStream(IterableDataset):
    def __init__(self, files: list[Path], max_facets: int, seeds_per_mesh: int,
                 densities: tuple[str, ...], type_weights=None, seed: int = 0,
                 relabel_root: Path | None = None):
        self.relabel_root = relabel_root
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
                          files=mine, relabel_root=self.relabel_root)
        for s in ds:
            yield s


CURV_COLS = (10, 11)


class Collate:
    """Module-level and picklable on purpose.

    Windows DataLoader workers are spawned, not forked, so everything handed to them is
    pickled. A closure over `args` - the obvious way to write this - fails with
    "Can't pickle local object", and the traceback points at multiprocessing rather
    than at the collate function, so it is worth not rediscovering.
    """

    def __init__(self, max_facets: int, drop_curvature: bool = False):
        self.max_facets = max_facets
        self.drop_curvature = drop_curvature

    def __call__(self, batch):
        x, y, m, t = collate(batch, self.max_facets)
        if self.drop_curvature:
            x[:, :, CURV_COLS[0]:CURV_COLS[1] + 1] = 0.0
        return (torch.from_numpy(x), torch.from_numpy(y),
                torch.from_numpy(m), torch.from_numpy(t))


def make_loader(files, args, type_weights, shuffle_seed):
    stream = PatchStream(files, args.max_facets, args.seeds_per_mesh,
                         tuple(args.densities.split(",")), type_weights, shuffle_seed,
                         args.relabel)
    return DataLoader(stream, batch_size=args.batch, collate_fn=Collate(args.max_facets, args.no_curvature),
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


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int):
    model.eval()
    tp = fp = fn_ = 0.0
    type_hit = np.zeros(N_TYPES, np.int64)
    type_tot = np.zeros(N_TYPES, np.int64)
    seen = 0
    for i, (x, y, m, t) in enumerate(loader):
        if i >= max_batches:
            break
        x, y, m, t = x.to(device), y.to(device), m.to(device), t.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            sl, tl = model(x, m)
        pred = ((sl.float().sigmoid() > 0.5).float()) * m
        truth = y * m
        tp += float((pred * truth).sum())
        fp += float((pred * (1 - truth) * m).sum())
        fn_ += float(((1 - pred) * truth).sum())
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
    return {"iou": iou, "precision": prec, "recall": rec,
            "type_hit": type_hit, "type_tot": type_tot, "batches": seen}


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
    args = ap.parse_args(argv)

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

    model = SeedFillNet(in_features=N_FEATURES, n_types=N_TYPES).to(device)
    print(f"curvature   {'DISABLED (control arm)' if args.no_curvature else 'enabled'}")
    print(f"parameters  {count_parameters(model):,} "
          f"(~{count_parameters(model) * 4 / 1e6:.2f} MB fp32)")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=args.steps, pct_start=0.05)
    pos_w = torch.tensor(pw, device=device)

    train_loader = make_loader(train_files, args, tw, args.seed)
    val_loader = make_loader(val_files, args, None, args.seed + 1)

    best = -1.0
    step = 0
    t0 = time.time()
    run_loss = run_seg = run_cls = 0.0
    log_path = args.out / "train_log.jsonl"
    logf = log_path.open("a", encoding="utf-8")
    print(f"\ntraining for {args.steps:,} steps, batch {args.batch}, "
          f"{args.max_facets} facets/patch\n")

    done = False
    while not done:
        for x, y, m, t in train_loader:
            x, y, m, t = (x.to(device, non_blocking=True), y.to(device, non_blocking=True),
                          m.to(device, non_blocking=True), t.to(device, non_blocking=True))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                sl, tl = model(x, m)
                seg = F.binary_cross_entropy_with_logits(
                    sl.float(), y, weight=m, pos_weight=pos_w, reduction="sum") / m.sum().clamp(min=1)
                cls = F.cross_entropy(tl.float(), t, weight=cls_w)
                loss = seg + args.type_loss_weight * cls

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            run_loss += float(loss); run_seg += float(seg); run_cls += float(cls)
            step += 1

            if step % 100 == 0:
                rate_s = step / (time.time() - t0)
                print(f"  step {step:>6,}/{args.steps:,}  loss {run_loss / 100:.4f} "
                      f"(seg {run_seg / 100:.4f} cls {run_cls / 100:.4f})  "
                      f"lr {sched.get_last_lr()[0]:.2e}  {rate_s:.1f} it/s", flush=True)
                run_loss = run_seg = run_cls = 0.0

            if step % args.eval_every == 0 or step == args.steps:
                ev = evaluate(model, val_loader, device, args.eval_batches)
                per = {TYPE_NAMES[i]: (round(float(ev["type_hit"][i] / ev["type_tot"][i]), 3)
                                       if ev["type_tot"][i] else None)
                       for i in range(N_TYPES)}
                macro = np.mean([v for v in per.values() if v is not None])
                print(f"\n  [eval @ {step:,}] membership IoU {ev['iou']:.4f}  "
                      f"precision {ev['precision']:.4f}  recall {ev['recall']:.4f}")
                print(f"              type macro-acc {macro:.3f}  {per}\n", flush=True)
                logf.write(json.dumps({"step": step, "iou": ev["iou"],
                                       "precision": ev["precision"], "recall": ev["recall"],
                                       "type_macro": float(macro), "per_type": per}) + "\n")
                logf.flush()
                if ev["iou"] > best:
                    best = ev["iou"]
                    torch.save({"model": model.state_dict(), "step": step,
                                "iou": best, "args": vars(args),
                                "pos_weight": pw, "type_names": list(TYPE_NAMES)},
                               args.out / "best.pt")

            if step >= args.steps:
                done = True
                break

    logf.close()
    print(f"\nbest membership IoU {best:.4f}  ->  {args.out / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
