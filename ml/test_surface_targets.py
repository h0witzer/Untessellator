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


def check_facet_targets(trials: int = 400, seed: int = 0) -> int:
    """facet_targets() must equal targets() facet by facet, for every kind, on random geometry."""
    rng = np.random.default_rng(seed)
    worst, bad = 0.0, 0
    for _ in range(trials):
        kinds = rng.choice([st.PLANE, st.CYL, st.CONE, st.SPHERE, st.TORUS], size=4)
        params = rng.normal(size=(4, 10))
        params[:, 6] = rng.uniform(0.2, 3.0, size=4)
        params[:, 7] = rng.uniform(0.05, 0.5, size=4)
        params[kinds == st.CONE, 6] = rng.uniform(0.1, 1.4, size=int((kinds == st.CONE).sum()))
        n = 40
        fid = rng.integers(0, 5, size=n)            # face 4 has no sidecar row: masked out
        cen = rng.normal(size=(n, 3)) * 2
        nrm = rng.normal(size=(n, 3))
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
        th = rng.uniform(0, 2 * np.pi)
        R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
        scale = float(rng.uniform(0.5, 5))
        tgt, msk = st.facet_targets(kinds, params, fid, cen, nrm, R, scale)
        for i in range(n):
            if fid[i] >= len(kinds):
                ref_t, ref_m = np.zeros(5), np.zeros(3)
            else:
                ax, ha, kn, hk, ks, hs = st.targets(int(kinds[fid[i]]), params[fid[i]], cen[i], nrm[i], R, scale)
                ref_t, ref_m = np.r_[ax, kn, ks], np.array([ha, hk, hs])
            if not np.array_equal(ref_m, msk[i]):
                bad += 1
                continue
            d = float(np.abs((ref_t - tgt[i]) * np.r_[ref_m[0], ref_m[0], ref_m[0], ref_m[1], ref_m[2]]).max())
            worst = max(worst, d)
    ok = bad == 0 and worst < 1e-4
    print(f"facet_targets vs targets: {trials} patches, mask mismatches {bad}, worst value error {worst:.2e}"
          f" -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--facets" in sys.argv:
        raise SystemExit(check_facet_targets())
    raise SystemExit(main())
