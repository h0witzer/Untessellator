#!/usr/bin/env python3
"""
test_surface_targets.py - do the surface targets describe the surface they were computed from?

Before any GPU time is spent, check the whole chain on real baked parts: sidecar parameters ->
targets in a (spun) patch frame -> reconstruct() -> distance of the seed face's own VERTICES to
the reconstructed surface. Vertices lie on the true surface, so a correct round trip leaves only
the offset from reconstructing through the seed CENTROID (a chord, not the surface) plus the
seed facet's normal error. A sign, frame or formula mistake shows up as distances of order 1.

Also asserts that the seed is facet 0 of every patch, which the surface head relies on.

Usage:
    python test_surface_targets.py --surface D:/abc-data/surface --parts 40
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import surface_targets as st  # noqa: E402
from occ_tessellate import TYPE_NAMES  # noqa: E402
from patches import derive, extract_patch  # noqa: E402
from relabel_sidecar import val_split  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bake", type=Path, default=Path("D:/abc-data/bake"))
    ap.add_argument("--surface", type=Path, default=Path("D:/abc-data/surface"))
    ap.add_argument("--parts", type=int, default=40)
    ap.add_argument("--per-kind", type=int, default=6)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(0)
    val, _ = val_split(args.bake)
    res = defaultdict(list)
    true_res = defaultdict(list)
    failures = defaultdict(int)
    bad_seed_order = used = 0
    for f in val:
        sc = args.surface / f.relative_to(args.bake)
        if not sc.exists():
            continue
        used += 1
        if used > args.parts:
            break
        sd = np.load(sc)
        surface = (sd["kind"], sd["params"])
        d = np.load(f)
        pos = d["medium_positions"]
        fid = d["medium_face_id"]
        fty = d["medium_face_type"]
        facets, indptr, indices, curv = derive(pos)
        V = pos.reshape(-1, 3, 3).astype(np.float64)
        for kind in st.ANALYTIC:
            cand = np.flatnonzero((surface[0][fid] == kind) & ~facets["degenerate"])
            if cand.size == 0:
                continue
            for seed in rng.choice(cand, size=min(args.per_kind, cand.size), replace=False):
                p = extract_patch(facets, indptr, indices, fid, fty, int(seed), 512,
                                  rng=rng, spin=True, curv=curv, surface=surface)
                if p is None:
                    continue
                if int(p["ids"][0]) != int(seed):
                    bad_seed_order += 1
                tgt, msk = p["surface"], p["surface_mask"]
                if not msk[1]:
                    failures[TYPE_NAMES[kind] + ": no kn target"] += 1
                    continue
                rec = st.reconstruct(kind, tgt[0:3], tgt[3], tgt[4])
                if rec is None:
                    failures[TYPE_NAMES[kind] + ": reconstruct returned None"] += 1
                    continue
                face = int(fid[seed])
                verts = np.unique(V[fid == face].reshape(-1, 3), axis=0)
                X = (verts - p["q"]) @ p["R"].T / p["scale"]
                dist, _ = st.distance(rec[0], rec[1], X)
                res[TYPE_NAMES[kind]].append(float(np.median(np.abs(dist))))
                tr = st.to_frame(kind, surface[1][face], p["q"], p["R"], p["scale"])
                tdist, _ = st.distance(tr[0], tr[1], X)
                true_res[TYPE_NAMES[kind]].append(float(np.median(np.abs(tdist))))

    q = lambda a, p: float(np.quantile(a, p)) if a else float("nan")
    print(f"{min(used, args.parts)} parts; seed not facet 0 in {bad_seed_order} patches")
    print("median |vertex distance| to the surface, in patch-scale units (per seed, then p50 / p90):")
    ok = bad_seed_order == 0
    for kind in ("Plane", "Cylinder", "Cone", "Sphere", "Torus"):
        r, t = res.get(kind, []), true_res.get(kind, [])
        if not r:
            continue
        print(f"  {kind:<9} n {len(r):4d}   reconstructed p50 {q(r, .5):.2e} p90 {q(r, .9):.2e}"
              f"   true surface p50 {q(t, .5):.2e} p90 {q(t, .9):.2e}")
        if q(r, .5) > 0.02:
            ok = False
    for k, v in failures.items():
        print(f"  {k}: {v}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
