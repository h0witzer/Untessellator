#!/usr/bin/env python3
"""
surface_eval.py - how good is the predicted surface hypothesis, and would it stop spill?

    python surface_eval.py --ckpt D:/abc-data/runs/s1nogmsh/best.pt

Validation parts only (train.py's split), meshed several ways, identical parts in every
column. For each patch whose seed face has an exact analytic surface:

  parameters  axis error (sign-free), sign of kn, |kn| relative error; for tori also ks
  gating      each patch facet is accepted by paintFrom's own test - |distance| <= tolD
              (0.5 x median sqrt(area) of the mesh) and normal within 15 deg - against:
                oracle      the true STEP surface
                given type  the TRUE type with the PREDICTED parameters
                predicted   the predicted type with the predicted parameters (what would ship)
              scored against same-face labels. A patch is clean when precision and recall are
              both >= 95%.

The gate only sees the 512-facet patch, so this measures whether the hypothesis separates the
seed's face from its neighbours locally, not a whole fill.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import surface_targets as st  # noqa: E402
from export_onnx import zeroed_columns  # noqa: E402
from model import SeedFillNet  # noqa: E402
from occ_tessellate import TYPE_NAMES  # noqa: E402
from patches import N_FEATURES, N_TYPES, collate, derive, extract_patch  # noqa: E402
from relabel_sidecar import val_split  # noqa: E402

KIND_OF_NAME = {"Plane": st.PLANE, "Cylinder": st.CYL, "Cone": st.CONE,
                "Sphere": st.SPHERE, "Torus": st.TORUS}
COS15 = math.cos(math.radians(15))
DEFAULT_VARIANTS = ("native=D:/abc-data/bake,nurbs=D:/abc-data/bake_nurbs,"
                    "gmsh=D:/abc-data/bake_gmsh,game=D:/abc-data/bake_game2")


def gate(surface, d, n, tol):
    if surface is None:
        return None
    dist, sn = st.distance(surface[0], surface[1], d)
    return (np.abs(dist) <= tol) & (np.abs((sn * n).sum(1)) >= COS15)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--bake", type=Path, default=Path("D:/abc-data/bake"))
    ap.add_argument("--variants", default=DEFAULT_VARIANTS)
    ap.add_argument("--surface-root", type=Path, default=Path("D:/abc-data/surface"))
    ap.add_argument("--relabel", type=Path, default=Path("D:/abc-data/relabel"))
    ap.add_argument("--density", default="medium")
    ap.add_argument("--parts", type=int, default=250)
    ap.add_argument("--per-kind", type=int, default=4)
    args = ap.parse_args(argv)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    net = SeedFillNet(in_features=N_FEATURES, n_types=N_TYPES, surface=True)
    net.load_state_dict(ck["model"])
    net.to(dev).eval()
    zero = zeroed_columns(ck["args"])

    variants = {k: Path(v) for k, v in (pair.split("=", 1) for pair in args.variants.split(","))}
    val, _ = val_split(args.bake)
    files = [f for f in val
             if all((root / f.relative_to(args.bake)).exists() for root in variants.values())
             and (args.surface_root / f.relative_to(args.bake)).exists()][: args.parts]
    print(f"checkpoint {args.ckpt} (step {ck.get('step')}), zeroed {zero}, "
          f"{len(files)} validation parts in all of {', '.join(variants)}")

    rows = defaultdict(list)
    key = f"{args.density}_positions"
    for f in files:
        rel = f.relative_to(args.bake)
        sd = np.load(args.surface_root / rel)
        surface = (sd["kind"], sd["params"])
        relabel = args.relabel / rel
        canon = np.load(relabel)["canonical"] if relabel.exists() else None
        rng = np.random.default_rng(abs(hash(f.stem)) % (2 ** 32))
        for vname, root in variants.items():
            d = np.load(root / rel)
            if key not in d.files or len(d[key]) < 32:
                continue
            pos = d[key]
            fid = d[f"{args.density}_face_id"]
            ftype = canon[fid] if canon is not None else d[f"{args.density}_face_type"]
            facets, indptr, indices, curv = derive(pos)
            tol_d = 0.5 * float(np.median(np.sqrt(np.maximum(facets["areas"], 0))))
            kinds_here = surface[0][fid]
            samples = []
            for kind in st.ANALYTIC:
                cand = np.flatnonzero((kinds_here == kind) & ~facets["degenerate"])
                if cand.size == 0:
                    continue
                for seed in rng.choice(cand, size=min(args.per_kind, cand.size), replace=False):
                    p = extract_patch(facets, indptr, indices, fid, ftype, int(seed), 512,
                                      rng=None, spin=False, curv=curv, surface=surface)
                    if p is not None:
                        samples.append(p)
            if not samples:
                continue
            x, _, m, _ = collate(samples, 512)
            if zero:
                x[:, :, zero] = 0.0
            with torch.no_grad():
                _, tl, sp = net(torch.from_numpy(x).to(dev), torch.from_numpy(m).to(dev))
            pred_type = tl.float().argmax(1).cpu().numpy()
            sp = sp.float().cpu().numpy()

            for p, pt, pr in zip(samples, pred_type, sp):
                sf = int(fid[p["seed"]])
                kind = int(surface[0][sf])
                tgt, msk = p["surface"], p["surface_mask"]
                rec = {"type_ok": float(int(pt) == int(ftype[p["seed"]]))}
                if msk[0]:
                    a = pr[:3] / (np.linalg.norm(pr[:3]) + 1e-12)
                    rec["axis"] = math.degrees(math.acos(min(1.0, abs(float(a @ tgt[:3])))))
                if msk[1]:
                    tk, pk = math.sinh(float(tgt[3])), math.sinh(float(pr[3]))
                    if kind == st.PLANE:
                        rec["flat"] = float(abs(pk) < 0.05)
                    else:
                        rec["kn_sign"] = float(np.sign(pk) == np.sign(tk))
                        rec["kn_rel"] = abs(abs(pk) - abs(tk)) / max(abs(tk), 1e-9)
                if msk[2]:
                    tk, pk = math.sinh(float(tgt[4])), math.sinh(float(pr[4]))
                    rec["ks_sign"] = float(np.sign(pk) == np.sign(tk))
                    rec["ks_rel"] = abs(abs(pk) - abs(tk)) / max(abs(tk), 1e-9)

                feats = p["features"].astype(np.float64)
                dfr, nfr = feats[:, 0:3], feats[:, 3:6]
                truth = p["same"] > 0.5
                # ABC splits one geometric surface into several faces (coplanar splits,
                # periodic seams). As in test/fillEval.mjs, "surface" precision also accepts
                # facets of the seed face's canonical type; strict counts only the face itself.
                same_type = ftype[p["ids"]] == ftype[p["seed"]]
                tol = tol_d / p["scale"]
                pred_kind = KIND_OF_NAME.get(TYPE_NAMES[int(pt)])
                hyps = {
                    "oracle": st.to_frame(kind, surface[1][sf], p["q"], p["R"], p["scale"]),
                    "given type": st.reconstruct(kind, pr[0:3], pr[3], pr[4]),
                    "predicted": st.reconstruct(pred_kind, pr[0:3], pr[3], pr[4]),
                }
                for name, hyp in hyps.items():
                    acc = gate(hyp, dfr, nfr, tol)
                    if acc is None:
                        rec[f"{name} clean"] = 0.0
                        rec[f"{name} surface clean"] = 0.0
                        rec[f"{name} recall"] = 0.0
                        rec[f"{name} precision"] = 0.0
                        continue
                    tp = float((acc & truth).sum())
                    prec = tp / max(float(acc.sum()), 1.0)
                    sprec = float((acc & (truth | same_type)).sum()) / max(float(acc.sum()), 1.0)
                    recall = tp / max(float(truth.sum()), 1.0)
                    rec[f"{name} clean"] = float(prec >= 0.95 and recall >= 0.95)
                    rec[f"{name} surface clean"] = float(sprec >= 0.95 and recall >= 0.95)
                    rec[f"{name} precision"] = sprec
                    rec[f"{name} recall"] = recall
                rows[(vname, TYPE_NAMES[kind])].append(rec)

    def agg(rs, k, fn):
        v = [r[k] for r in rs if k in r]
        return fn(v) if v else float("nan")

    p50 = lambda v: float(np.median(v))
    p90 = lambda v: float(np.quantile(v, 0.9))
    mean = lambda v: float(np.mean(v))
    for vname in variants:
        print(f"\n{vname}")
        print(f"  {'type':<9} {'n':>5} {'type ok':>8} {'axis p50':>9} {'axis p90':>9} {'kn sign':>8} "
              f"{'|kn| err':>9} {'ks err':>7} | clean: {'oracle':>6} {'given':>6} {'pred':>6} "
              f"| surface-clean: {'oracle':>6} {'pred':>6} | precision / recall pred")
        for kind in ("Plane", "Cylinder", "Cone", "Sphere", "Torus"):
            rs = rows.get((vname, kind), [])
            if not rs:
                continue
            sign = agg(rs, "kn_sign", mean) if kind != "Plane" else agg(rs, "flat", mean)
            print(f"  {kind:<9} {len(rs):>5} {agg(rs, 'type_ok', mean):>8.0%} "
                  f"{agg(rs, 'axis', p50):>9.2f} {agg(rs, 'axis', p90):>9.1f} {sign:>8.0%} "
                  f"{agg(rs, 'kn_rel', p50):>9.3f} {agg(rs, 'ks_rel', p50):>7.3f} | "
                  f"       {agg(rs, 'oracle clean', mean):>6.0%} {agg(rs, 'given type clean', mean):>6.0%} "
                  f"{agg(rs, 'predicted clean', mean):>6.0%} | "
                  f"               {agg(rs, 'oracle surface clean', mean):>6.0%} "
                  f"{agg(rs, 'predicted surface clean', mean):>6.0%} | "
                  f"{agg(rs, 'predicted precision', p50):.2f} / {agg(rs, 'predicted recall', p50):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
