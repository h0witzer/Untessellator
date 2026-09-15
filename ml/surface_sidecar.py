#!/usr/bin/env python3
"""
surface_sidecar.py - exact analytic surface parameters per face, for the surface-hypothesis head.

Face ids in every bake (native, NURBS, gmsh, game art) index the STEP's TopExp_Explorer faces,
so one sidecar per model serves all of them. The order is CHECKED, as relabel_sidecar.py does:
the stored type computed here for face j must equal the bake's face_type on every triangle
with face_id j, or the model is skipped.

Output, one file per model:  <out>/<prefix>/<model>_<part>.npz
    kind    uint8[n_faces]        TYPE_INDEX of the stored surface type
    params  float64[n_faces, 10]  layout in surface_targets.py; zeros when not analytic

Usage:
    python surface_sidecar.py --files all --out D:/abc-data/surface
    python surface_sidecar.py --files val --limit 50 --out D:/abc-data/surface
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from occ_tessellate import TYPE_INDEX, face_type_name, read_step  # noqa: E402
from relabel_sidecar import val_split  # noqa: E402
from stage1_bake import find_step  # noqa: E402


def _xyz(p):
    return (p.X(), p.Y(), p.Z())


def face_params(face) -> np.ndarray:
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_SurfaceType as GA

    out = np.zeros(10)
    ad = BRepAdaptor_Surface(face)
    st = ad.GetType()
    if st == GA.GeomAbs_Plane:
        s = ad.Plane()
        out[0:3], out[3:6] = _xyz(s.Location()), _xyz(s.Axis().Direction())
    elif st == GA.GeomAbs_Cylinder:
        s = ad.Cylinder()
        out[0:3], out[3:6], out[6] = _xyz(s.Location()), _xyz(s.Axis().Direction()), s.Radius()
    elif st == GA.GeomAbs_Cone:
        s = ad.Cone()
        out[0:3], out[3:6], out[6] = _xyz(s.Apex()), _xyz(s.Axis().Direction()), abs(s.SemiAngle())
    elif st == GA.GeomAbs_Sphere:
        s = ad.Sphere()
        out[0:3], out[6] = _xyz(s.Location()), s.Radius()
    elif st == GA.GeomAbs_Torus:
        s = ad.Torus()
        out[0:3], out[3:6] = _xyz(s.Location()), _xyz(s.Axis().Direction())
        out[6], out[7] = s.MajorRadius(), s.MinorRadius()
    return out


def one(task):
    npz_path, step_root, out_root = task
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    rec = {"file": npz_path, "status": "ok", "n_faces": 0}
    try:
        d = np.load(npz_path)
        mid, part = str(d["model_id"]), str(d["part"])
        out = Path(out_root) / mid[:5] / f"{mid}_{part}.npz"
        if out.exists():
            rec["status"] = "exists"
            return rec
        sp = find_step(Path(step_root), mid)
        if sp is None:
            rec["status"] = "no_step"
            return rec
        shape = read_step(str(sp))
        if shape is None:
            rec["status"] = "unreadable"
            return rec
        kinds, params = [], []
        ex = TopExp_Explorer(shape, TopAbs_FACE)
        while ex.More():
            f = TopoDS.Face_s(ex.Current())
            ex.Next()
            kinds.append(TYPE_INDEX[face_type_name(f)])
            try:
                params.append(face_params(f))
            except Exception:
                params.append(np.zeros(10))
        kinds = np.array(kinds, np.uint8)
        params = np.array(params, np.float64).reshape(-1, 10)
        rec["n_faces"] = len(kinds)

        for dens in ("coarse", "medium", "fine"):
            key = f"{dens}_face_id"
            if key not in d.files:
                continue
            fid = d[key]
            if fid.size and (fid.max() >= len(kinds) or np.any(kinds[fid] != d[f"{dens}_face_type"])):
                rec["status"] = "misaligned"
                return rec

        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, kind=kinds, params=params)
    except Exception as exc:
        rec["status"] = f"error:{type(exc).__name__}"
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bake", type=Path, default=Path("D:/abc-data/bake"))
    ap.add_argument("--step-root", type=Path, default=Path("D:/abc-data/step"))
    ap.add_argument("--out", type=Path, default=Path("D:/abc-data/surface"))
    ap.add_argument("--files", choices=["val", "train", "all"], default="all")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    val, train = val_split(args.bake)
    files = {"val": val, "train": train, "all": val + train}[args.files]
    if args.limit:
        files = files[: args.limit]
    print(f"surface sidecars for {len(files):,} {args.files} models -> {args.out}", flush=True)

    t0 = time.time()
    status, faces = Counter(), 0
    tasks = [(str(f), str(args.step_root), str(args.out)) for f in files]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for i, rec in enumerate(pool.map(one, tasks, chunksize=4), 1):
            status[rec["status"]] += 1
            faces += rec["n_faces"]
            if i % 500 == 0:
                print(f"  {i:,}/{len(files):,}  {i / (time.time() - t0):.1f}/s", flush=True)
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min")
    print("status: " + ", ".join(f"{k} {v:,}" for k, v in status.most_common()))
    print(f"faces: {faces:,}")


if __name__ == "__main__":
    raise SystemExit(main())
