#!/usr/bin/env python3
"""
freeform_eval.py - how well freeform_fit.fit_group recovers ABC's true non-analytic surfaces, face
by face, with TRUE segments (the fitter isolated from segmentation and typing).

    python freeform_eval.py --mesh D:/abc-data/freeform_eval --truth D:/abc-data/freeform_truth \\
        --out ff_eval.json [--workers 8] [--limit N] [--set LAMBDA=1e-6 TOL=1e-5 ...] [--fits DIR | --fits '']

Tangency (G1 to analytic neighbours, solid_rebuild.tangent_normals) uses the neighbours' surfaces as
fitted by test/exportPatchFits.mjs on the true groups (--fits); --fits '' fits without it.

Per freeform face (stored type Revolution / Extrusion / BSpline / Other):

  dev     max distance from the fitted surface of dense samples of the TRUE surface over the face
          (7 barycentric points per triangle, projected onto the true surface), x the part diagonal
          - surfaceScore.mjs's "found" measure: found3 dev < 1e-3, found4 dev < 1e-4
  floor   max distance of the face's mesh vertices from the true surface: ~1e-6 normally, but up to
          ~5e-4 on some faces (brute-force confirmed: ABC's B-spline is an approximation of the
          fillet Onshape tessellated), so found4 is out of reach where floor > 1e-4
  vert    the fit's own max vertex error (what the fitter sees)

Faces without an exact definition (Other; the ellipse sweeps, whose ABC records are broken) are
fitted and reported with vert only. "arc" marks B-splines with an exact circular direction (rational
degree 2 with 3 or 5 poles): fillets.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abc_freeform as af  # noqa: E402
import freeform_fit as ff  # noqa: E402
import nurbs  # noqa: E402

BARY = np.array([[1 / 3, 1 / 3, 1 / 3], [.5, .5, 0], [0, .5, .5], [.5, 0, .5],
                 [2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]])
MAX_SAMPLES = 3000
FLOOR_SAMPLES = 5000             # vertices the truth floor is measured on (a real export's face can hold 300k)


def kind_of(d):
    k = d.get("type")
    if d.get("stored_type") not in (None, "BSpline"):
        return f"{d['stored_type']}-converted"          # export_real_pairs.py: a STEP kind as a B-spline
    if k == "BSpline":
        P = np.asarray(d["poles"])
        rat = d.get("u_rational") or d.get("v_rational")
        arc = rat and ((d["v_degree"] == 2 and P.shape[1] in (3, 5)) or (d["u_degree"] == 2 and P.shape[0] in (3, 5)))
        return "BSpline-arc" if arc else "BSpline"
    if k in ("Extrusion", "Revolution"):
        return f"{k}-{(d.get('curve') or {}).get('type', '?')}"
    return k


def group(mesh, g):
    tris = mesh.T[mesh.F == g]
    vid, inv = np.unique(tris, return_inverse=True)
    return mesh.V[vid], inv.reshape(-1, 3)


def run_part(job):
    stem, part, mesh_dir, truth_dir, fits_dir, overrides = job
    for k, v in overrides.items():
        setattr(ff, k, v)
    import solid_rebuild as sr                      # Mesh: dedup + consistent winding
    pos = np.fromfile(Path(mesh_dir) / f"{stem}.pos", np.float32)
    fid = np.fromfile(Path(mesh_dir) / f"{stem}.fid", np.int32)
    defs = json.loads((Path(truth_dir) / f"{part}.ffs.json").read_text())
    mesh = sr.Mesh(pos, fid)
    # the analytic neighbours' fitted surfaces (test/exportPatchFits.mjs on the true groups), for
    # tangency where they meet a freeform face smoothly
    surfaces = {}
    if fits_dir and (Path(fits_dir) / f"{stem}.json").exists():
        fits = json.loads((Path(fits_dir) / f"{stem}.json").read_text())
        surfaces = {f["id"]: (f["type"], f["params"]) for f in fits["faces"] if f["params"]}
    rows = []
    rng = np.random.default_rng(0)
    for g_str, d in defs.items():
        g = int(g_str)
        if not (mesh.F == g).any():
            continue
        V, T = group(mesh, g)
        row = {"part": part, "stem": stem, "face": g, "kind": kind_of(d), "tris": int(len(T))}
        normals = None
        if surfaces:
            tn = sr.tangent_normals(mesh, g, surfaces)
            if tn:
                vid = np.unique(mesh.T[mesh.F == g])
                keys = np.array(sorted(tn), np.int64)
                normals = (np.searchsorted(vid, keys), np.array([tn[k] for k in keys]))
        t0 = time.time()
        try:
            pieces, failed = ff.fit_pieces(V, T, scale=mesh.diag, normals=normals)
        except Exception as exc:                     # a fitter crash is a failed face, not a lost part
            pieces, failed = [], [(np.arange(len(T)), {"fail": f"exception {type(exc).__name__}: {exc}"[:160]})]
        row["secs"] = round(time.time() - t0, 3)
        row["pieces"] = len(pieces)
        if pieces:
            info = max((p[2] for p in pieces), key=lambda i: i.get("max_err", 0))
            row.update({k: v for k, v in info.items()
                        if k in ("param", "spans", "rounds", "max_err", "rms_err", "tangent_vertices",
                                 "tangent_agree", "tangent_used")})
            row["depth"] = max(p[2].get("depth", 0) for p in pieces)
            if len(pieces) > 1:
                row["param"] = "split"
        if failed:
            row["fail"] = failed[0][1].get("fail", "?")
            row["unfitted_tris"] = int(sum(len(t) for t, _ in failed))
        truth = None
        try:
            truth = af.truth_surface(d, V)
        except Exception as exc:
            row["truth_error"] = repr(exc)[:120]
        if truth is not None:
            Vs = V if len(V) <= FLOOR_SAMPLES else V[rng.choice(len(V), FLOOR_SAMPLES, replace=False)]
            _, _, Q = truth.closest(Vs)
            row["floor"] = float(np.linalg.norm(Q - Vs, axis=1).max() / mesh.diag)
        if pieces and not failed and truth is not None:
            owner = np.empty(len(T), np.int64)
            for k, (tri, _, _) in enumerate(pieces):
                owner[tri] = k
            sel = np.arange(len(T))
            if len(T) * len(BARY) > MAX_SAMPLES:
                sel = rng.choice(len(T), MAX_SAMPLES // len(BARY), replace=False)
            pts = np.einsum("kb,tbc->tkc", BARY, V[T[sel]]).reshape(-1, 3)
            who = np.repeat(owner[sel], len(BARY))
            _, _, on_truth = truth.closest(pts)
            dev = np.zeros(len(pts))
            for k, (_, params, _) in enumerate(pieces):
                m = who == k
                if m.any():
                    _, _, on_fit = nurbs.Surface.from_params(params).closest(on_truth[m])
                    dev[m] = np.linalg.norm(on_fit - on_truth[m], axis=1) / mesh.diag
            row["dev"] = float(dev.max())
            row["dev_rms"] = float(np.sqrt((dev ** 2).mean()))
        rows.append(row)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", type=Path, default=Path("D:/abc-data/freeform_eval"))
    ap.add_argument("--truth", type=Path, default=Path("D:/abc-data/freeform_truth"))
    ap.add_argument("--fits", default="D:/abc-data/lab/freeform/patches/truth-all",
                    help="fitted surfaces of the true groups (<stem>.json) for tangency; '' for none")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--variants", default="", help="comma list of mesh variants to grade (default all)")
    ap.add_argument("--set", nargs="*", default=[], help="freeform_fit constants, NAME=value")
    args = ap.parse_args(argv)
    overrides = {}
    for s in args.set:
        k, v = s.split("=")
        overrides[k] = type(getattr(ff, k))(float(v)) if not isinstance(getattr(ff, k), int) else int(float(v))
    manifest = json.loads((args.mesh / "manifest.json").read_text())
    jobs = []
    for e in manifest["parts"]:
        if not (args.truth / f"{e['part']}.ffs.json").exists():
            continue
        for vname, v in e["variants"].items():
            if args.variants and vname not in args.variants.split(","):
                continue
            jobs.append((v["stem"], e["part"], str(args.mesh), str(args.truth), args.fits, overrides))
        if args.limit and len(jobs) >= args.limit:
            break
    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        for i, rs in enumerate(ex.map(run_part, jobs)):
            rows += rs
            if i % 25 == 24:
                print(f"  {i + 1}/{len(jobs)} parts, {time.time() - t0:.0f}s", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows))
    summarize(rows, overrides)
    return 0


def summarize(rows, overrides=None):
    split = [r for r in rows if r.get("pieces", 1) > 1]
    print(f"\n{len(rows)} freeform faces  {overrides or ''}  split into pieces: {len(split)}"
          f" (pieces p50 {np.median([r['pieces'] for r in split]) if split else 0:.0f})")
    print(f"  {'kind':22} {'faces':>5} {'fitted':>6} {'graded':>6} {'found3':>7} {'found4':>7} {'f4 reach':>8}"
          f" {'dev p50':>8} {'dev p90':>8} {'secs p50':>8}  fails")
    by = defaultdict(list)
    for r in rows:
        by[r["kind"]].append(r)
    by["ALL"] = rows
    for k in sorted(by, key=lambda k: (k == "ALL", -len(by[k]))):
        rs = by[k]
        fitted = [r for r in rs if "fail" not in r]
        graded = [r for r in rs if "dev" in r]
        f3 = sum(r["dev"] < 1e-3 for r in graded)
        f4 = sum(r["dev"] < 1e-4 for r in graded)
        reach = sum(r.get("floor", 0) <= 1e-4 for r in graded)
        devs = np.array([r["dev"] for r in graded]) if graded else np.array([np.nan])
        secs = np.array([r["secs"] for r in rs])
        fails = defaultdict(int)
        for r in rs:
            if "fail" in r:
                fails[r["fail"].split(":")[0]] += 1
        pc = lambda a, b: f"{100 * a / b:6.1f}%" if b else "     - "
        print(f"  {k:22} {len(rs):5} {len(fitted):6} {len(graded):6} {pc(f3, len(graded))} {pc(f4, len(graded))}"
              f" {pc(reach, len(graded))} {np.median(devs):8.1e} {np.quantile(devs, .9):8.1e} {np.median(secs):8.2f}  {dict(fails)}")


if __name__ == "__main__":
    raise SystemExit(main())
